"""Tests for the global fact-inventory extraction (Phase 2)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.storage import LocalCourseArtifactStorage
from app.syllabus_facts import (
    FACT_SCOPES,
    IMPORTANCE_VALUE,
    batch_section_groups,
    build_fact_inventory,
    build_heuristic_fallback_inventory,
    build_section_groups,
    calibrate_importance,
    compute_usefulness_score,
    detect_assignment_series,
    merge_facts,
    normalize_fact,
    salvage_late_work_policy_facts,
    semantic_merge_facts,
    statement_entailment_violation,
    verify_evidence_quote,
)

LATE_POLICY_QUOTE = "One 48-hour extension is allowed for Bot Projects 1 through 7."
ATTENDANCE_QUOTE = (
    "Students must notify the instructor at least one hour before class if absent."
)
PANTRY_QUOTE = "The Husky Pantry is an optional food resource for students."


async def _orthogonal_embed(texts, *, model=None):
    """Deterministic embeddings where every text is orthogonal (never merges)."""
    count = len(texts)
    embeddings = [
        [1.0 if row == col else 0.0 for col in range(count)] for row in range(count)
    ]
    return {"embeddings": embeddings, "model": model or "test-embed"}


async def _raising_embed(texts, *, model=None):
    """Embedding stub that fails, exercising semantic-merge fail-open behavior."""
    raise HTTPException(status_code=503, detail="Ollama embeddings unavailable.")


def _chunk(chunk_id: str, section_title: str, text: str, order: int = 1) -> dict:
    return {
        "chunkId": chunk_id,
        "sectionTitle": section_title,
        "text": text,
        "order": order,
    }


def _index(course_id: str, chunks: list[dict]) -> dict:
    return {
        "courseId": course_id,
        "embeddingModel": "nomic-embed-text",
        "chunkCount": len(chunks),
        "chunks": chunks,
    }


def _policy_chunks() -> list[dict]:
    return [
        _chunk(
            "chunk-001",
            "Late Work Policy",
            f"Late Work Policy\n{LATE_POLICY_QUOTE} Requests must be made before the deadline.",
            order=1,
        ),
        _chunk(
            "chunk-002",
            "Attendance",
            f"Attendance\n{ATTENDANCE_QUOTE}",
            order=2,
        ),
        _chunk(
            "chunk-003",
            "Campus Resources",
            f"Campus Resources\n{PANTRY_QUOTE}",
            order=3,
        ),
    ]


def _llm_facts_payload() -> str:
    return json.dumps(
        {
            "facts": [
                {
                    "statement": "One 48-hour extension is allowed for Bot Projects 1 through 7.",
                    "importance": "high",
                    "studentAskLikelihood": 0.9,
                    "complexity": 2,
                    "sourceChunkIds": ["chunk-001"],
                    "evidenceQuote": LATE_POLICY_QUOTE,
                    "kind": "late_work",
                    "scope": "course_wide",
                },
                {
                    "statement": "The Husky Pantry is an optional food resource.",
                    "importance": "low",
                    "studentAskLikelihood": 0.2,
                    "complexity": 1,
                    "sourceChunkIds": ["chunk-003"],
                    "evidenceQuote": PANTRY_QUOTE,
                    "kind": "resource",
                    "scope": "resource",
                },
            ]
        }
    )


class LateWorkPolicySalvageTests(unittest.IsolatedAsyncioTestCase):
    def _policy_chunks(self) -> list[dict]:
        return [
            _chunk(
                "pol-001",
                "Policies",
                (
                    "Late Policy. Assignments in the Project series are due Fridays at "
                    "11:59 p.m. Among these projects, you may choose one project for "
                    "which you want to use one 48-hour extension per quarter, no "
                    "questions asked."
                ),
                order=1,
            ),
            _chunk(
                "pol-002",
                "Policies",
                (
                    "No extension is possible for the Demo and Feedback assignments as "
                    "those involve coordinating the entire class community. No "
                    "extension is possible for the Reflection assignment because grades "
                    "must be submitted on time."
                ),
                order=2,
            ),
            _chunk(
                "pol-003",
                "Contact",
                (
                    "If I do not respond to your message in 48 hours, please nudge me "
                    "via Discord. This is not an assignment extension policy."
                ),
                order=3,
            ),
        ]

    def test_salvages_allowed_extension_with_limit(self) -> None:
        facts = salvage_late_work_policy_facts(
            self._policy_chunks(), existing_facts=[]
        )
        extension_facts = [
            f for f in facts if "48-hour extension" in f["statement"].lower()
        ]
        self.assertEqual(len(extension_facts), 1)
        fact = extension_facts[0]
        self.assertEqual(fact["kind"], "late_work")
        self.assertEqual(fact["scope"], "course_wide")
        self.assertEqual(fact["importance"], "high")
        self.assertEqual(fact["sourceChunkIds"], ["pol-001"])
        self.assertIn("48-hour extension", fact["evidenceQuote"])

    def test_salvages_no_extension_exclusions(self) -> None:
        facts = salvage_late_work_policy_facts(
            self._policy_chunks(), existing_facts=[]
        )
        exclusion_facts = [
            f for f in facts if "no extension is possible" in f["statement"].lower()
        ]
        self.assertGreaterEqual(len(exclusion_facts), 2)
        sources = {tuple(f["sourceChunkIds"]) for f in exclusion_facts}
        self.assertEqual(sources, {("pol-002",)})

    def test_general_rule_and_exception_both_recovered(self) -> None:
        facts = salvage_late_work_policy_facts(
            self._policy_chunks(), existing_facts=[]
        )
        statements = " ".join(f["statement"].lower() for f in facts)
        self.assertIn("48-hour extension", statements)
        self.assertIn("no extension is possible", statements)

    def test_response_time_nudge_is_not_salvaged_as_extension(self) -> None:
        facts = salvage_late_work_policy_facts(
            self._policy_chunks(), existing_facts=[]
        )
        for fact in facts:
            blob = f"{fact['statement']} {fact['evidenceQuote']}".lower()
            self.assertNotIn("nudge me", blob)
            self.assertNotIn("respond to your message", blob)

    def test_does_not_duplicate_when_llm_already_extracted(self) -> None:
        chunks = self._policy_chunks()
        existing = [
            {
                "statement": (
                    "Students may use one 48-hour extension per quarter for one "
                    "project in the series."
                ),
                "importance": "high",
                "importanceScore": IMPORTANCE_VALUE["high"],
                "studentAskLikelihood": 0.9,
                "complexity": 2,
                "sourceChunkIds": ["pol-001"],
                "evidenceQuote": (
                    "you may choose one project for which you want to use one "
                    "48-hour extension per quarter, no questions asked."
                ),
                "kind": "late_work",
                "scope": "course_wide",
            }
        ]
        facts = salvage_late_work_policy_facts(chunks, existing_facts=existing)
        extension_salvaged = [
            f for f in facts if "48-hour extension" in f["statement"].lower()
        ]
        self.assertEqual(extension_salvaged, [])

    def test_evidence_verification_still_strict(self) -> None:
        chunks = [
            _chunk(
                "pol-bad",
                "Policies",
                "Office hours are available by appointment.",
                order=1,
            )
        ]
        # Fabricated late-work sentence that is not in the chunk must not appear.
        facts = salvage_late_work_policy_facts(chunks, existing_facts=[])
        self.assertEqual(facts, [])

    async def test_inventory_recovers_policy_even_if_llm_omits_it(self) -> None:
        chunks = self._policy_chunks()
        # LLM returns only a deadline fact, omitting the late-work policies.
        payload = json.dumps(
            {
                "facts": [
                    {
                        "statement": (
                            "Assignments in the Project series are due Fridays at "
                            "11:59 p.m."
                        ),
                        "importance": "medium",
                        "studentAskLikelihood": 0.7,
                        "complexity": 1,
                        "sourceChunkIds": ["pol-001"],
                        "evidenceQuote": (
                            "Assignments in the Project series are due Fridays at "
                            "11:59 p.m."
                        ),
                        "kind": "deadline",
                        "scope": "course_wide",
                    }
                ]
            }
        )
        completion_fn = AsyncMock(
            return_value={"answer": payload, "model": "qwen3:4b"}
        )
        inventory = await build_fact_inventory(
            raw_chunks=chunks,
            completion_fn=completion_fn,
            embed_fn=_orthogonal_embed,
        )
        self.assertFalse(inventory["fallbackUsed"])
        statements = [f["statement"].lower() for f in inventory["facts"]]
        joined = " ".join(statements)
        self.assertTrue(any("48-hour extension" in s for s in statements))
        self.assertIn("no extension is possible", joined)
        extension_facts = [
            f for f in inventory["facts"] if "48-hour extension" in f["statement"].lower()
        ]
        self.assertEqual(extension_facts[0]["importance"], "high")
        self.assertEqual(extension_facts[0]["kind"], "late_work")
        # No duplicate extension facts from salvage + merge.
        self.assertEqual(len(extension_facts), 1)


class BatchingTests(unittest.TestCase):
    def test_oversized_single_section_is_split(self) -> None:
        # Wiki-exported syllabi often share one sectionTitle across all chunks.
        chunks = [
            _chunk(
                f"chunk-{index:03d}",
                "Software Engineering (Fall 2025)",
                ("Policy text paragraph. " * 40) + f"Chunk {index}.",
                order=index,
            )
            for index in range(1, 12)
        ]
        groups = build_section_groups(chunks)
        self.assertEqual(len(groups), 1)
        batches = batch_section_groups(groups, char_budget=1200)
        self.assertGreater(len(batches), 1)
        for batch in batches:
            batch_chars = sum(
                len(chunk["text"]) for group in batch for chunk in group["chunks"]
            )
            # A single oversized chunk may still exceed the budget alone; packs of
            # multiple chunks must stay at or under the budget.
            chunk_count = sum(len(group["chunks"]) for group in batch)
            if chunk_count > 1:
                self.assertLessEqual(batch_chars, 1200)


class EvidenceVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunk_lookup = {
            chunk["chunkId"]: chunk["text"] for chunk in _policy_chunks()
        }

    def test_verifies_quote_in_listed_chunk(self) -> None:
        verified = verify_evidence_quote(
            LATE_POLICY_QUOTE, ["chunk-001"], self.chunk_lookup
        )
        self.assertIsNotNone(verified)
        assert verified is not None
        quote, chunk_ids = verified
        self.assertEqual(chunk_ids, ["chunk-001"])
        self.assertIn("48-hour", quote)

    def test_repairs_source_when_quote_in_other_chunk(self) -> None:
        verified = verify_evidence_quote(
            ATTENDANCE_QUOTE, ["chunk-999"], self.chunk_lookup
        )
        self.assertIsNotNone(verified)
        assert verified is not None
        _, chunk_ids = verified
        self.assertEqual(chunk_ids, ["chunk-002"])

    def test_whitespace_normalized_match(self) -> None:
        noisy = "One 48-hour   extension is allowed\nfor Bot Projects 1 through 7."
        verified = verify_evidence_quote(noisy, ["chunk-001"], self.chunk_lookup)
        self.assertIsNotNone(verified)

    def test_rejects_unverifiable_quote(self) -> None:
        verified = verify_evidence_quote(
            "You may get unlimited extensions on all work.",
            ["chunk-001"],
            self.chunk_lookup,
        )
        self.assertIsNone(verified)

    def test_rejects_too_short_quote(self) -> None:
        self.assertIsNone(verify_evidence_quote("late", ["chunk-001"], self.chunk_lookup))


class NormalizeFactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunk_lookup = {
            chunk["chunkId"]: chunk["text"] for chunk in _policy_chunks()
        }

    def test_normalizes_required_fields_and_scope(self) -> None:
        fact = normalize_fact(
            {
                "statement": "One 48-hour extension is allowed for Bot Projects 1 through 7.",
                "importance": "HIGH",
                "studentAskLikelihood": 90,
                "complexity": "medium",
                "sourceChunkIds": ["chunk-001"],
                "evidenceQuote": LATE_POLICY_QUOTE,
                "kind": "Late Work",
                "scope": "course-wide",
            },
            chunk_lookup=self.chunk_lookup,
        )
        self.assertIsNotNone(fact)
        assert fact is not None
        self.assertEqual(fact["importance"], "high")
        self.assertEqual(fact["scope"], "course_wide")
        self.assertEqual(fact["kind"], "late_work")
        self.assertEqual(fact["complexity"], 2)
        self.assertAlmostEqual(fact["studentAskLikelihood"], 0.9)
        self.assertIn(fact["scope"], FACT_SCOPES)
        self.assertEqual(fact["sourceChunkIds"], ["chunk-001"])

    def test_unknown_scope_falls_back_to_other(self) -> None:
        fact = normalize_fact(
            {
                "statement": ATTENDANCE_QUOTE,
                "importance": "medium",
                "studentAskLikelihood": 0.5,
                "complexity": 1,
                "sourceChunkIds": ["chunk-002"],
                "evidenceQuote": ATTENDANCE_QUOTE,
                "kind": "attendance",
                "scope": "totally_made_up",
            },
            chunk_lookup=self.chunk_lookup,
        )
        assert fact is not None
        self.assertEqual(fact["scope"], "other")

    def test_invalid_evidence_drops_fact(self) -> None:
        fact = normalize_fact(
            {
                "statement": "Every assignment can be turned in a week late for free.",
                "importance": "high",
                "studentAskLikelihood": 0.9,
                "complexity": 1,
                "sourceChunkIds": ["chunk-001"],
                "evidenceQuote": "Every assignment can be turned in a week late for free.",
                "kind": "late_work",
                "scope": "course_wide",
            },
            chunk_lookup=self.chunk_lookup,
        )
        self.assertIsNone(fact)

    def test_short_statement_drops_fact(self) -> None:
        fact = normalize_fact(
            {
                "statement": "Late.",
                "importance": "high",
                "studentAskLikelihood": 0.9,
                "complexity": 1,
                "sourceChunkIds": ["chunk-001"],
                "evidenceQuote": LATE_POLICY_QUOTE,
                "kind": "late_work",
                "scope": "course_wide",
            },
            chunk_lookup=self.chunk_lookup,
        )
        self.assertIsNone(fact)


class MergeFactsTests(unittest.TestCase):
    def _fact(
        self,
        statement: str,
        *,
        importance: str = "medium",
        ask: float = 0.5,
        complexity: int = 1,
        source_ids: list[str] | None = None,
        evidence: str | None = None,
        kind: str = "policy",
        scope: str = "course_wide",
    ) -> dict:
        from app.syllabus_facts import IMPORTANCE_VALUE

        return {
            "statement": statement,
            "importance": importance,
            "importanceScore": IMPORTANCE_VALUE[importance],
            "studentAskLikelihood": ask,
            "complexity": complexity,
            "sourceChunkIds": source_ids or ["chunk-001"],
            "evidenceQuote": evidence or statement,
            "kind": kind,
            "scope": scope,
        }

    def test_exact_duplicate_statements_merge(self) -> None:
        merged = merge_facts(
            [
                self._fact("Office hours are optional.", importance="medium"),
                self._fact(
                    "office hours are optional",
                    importance="high",
                    source_ids=["chunk-002"],
                ),
            ]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["importance"], "high")
        self.assertEqual(sorted(merged[0]["sourceChunkIds"]), ["chunk-001", "chunk-002"])

    def test_contained_statement_with_source_overlap_merges(self) -> None:
        merged = merge_facts(
            [
                self._fact(
                    "Late work loses 10 percent per day up to three days.",
                    source_ids=["chunk-001"],
                ),
                self._fact("Late work loses 10 percent per day.", source_ids=["chunk-001"]),
            ]
        )
        self.assertEqual(len(merged), 1)

    def test_distinct_statements_are_not_merged(self) -> None:
        merged = merge_facts(
            [
                self._fact(ATTENDANCE_QUOTE, source_ids=["chunk-002"]),
                self._fact(LATE_POLICY_QUOTE, source_ids=["chunk-001"]),
            ]
        )
        self.assertEqual(len(merged), 2)


class EntailmentTests(unittest.TestCase):
    def test_obligation_requires_grounding_in_evidence(self) -> None:
        # A list of AI tools does not mean students MUST use them.
        reason = statement_entailment_violation(
            "Students must use AI tools for coding assistance.",
            "AI tools: Chat GPT, Microsoft CoPilot, and Gemini",
        )
        self.assertEqual(reason, "obligation_not_in_evidence")

    def test_availability_does_not_become_requirement(self) -> None:
        reason = statement_entailment_violation(
            "Students must attend office hours.",
            "Office hours appointments are available by request.",
        )
        self.assertEqual(reason, "obligation_not_in_evidence")

    def test_grounded_obligation_is_allowed(self) -> None:
        reason = statement_entailment_violation(
            "Grade-related discussion must happen via Canvas.",
            "all grade-related discussion must happen via Canvas",
        )
        self.assertIsNone(reason)

    def test_conditional_nudge_is_not_a_response_guarantee(self) -> None:
        reason = statement_entailment_violation(
            "The instructor responds to messages within 48 hours.",
            "If I do not respond to your message in 48 hours, please nudge me via a Discord DM.",
        )
        self.assertEqual(reason, "guarantee_from_conditional")

    def test_negated_guarantee_from_nudge_evidence_is_allowed(self) -> None:
        reason = statement_entailment_violation(
            "The instructor does not guarantee a response within 48 hours.",
            "If I do not respond to your message in 48 hours, please nudge me via a Discord DM.",
        )
        self.assertIsNone(reason)

    def test_optional_must_be_grounded(self) -> None:
        reason = statement_entailment_violation(
            "Office hours are optional.",
            "You can meet with me via an office hours appointment.",
        )
        self.assertEqual(reason, "optional_not_in_evidence")

    def test_grounded_optional_is_allowed(self) -> None:
        reason = statement_entailment_violation(
            "Attending office hours is optional.",
            "Attending open lab periods and office hours is optional.",
        )
        self.assertIsNone(reason)

    def test_normalize_fact_drops_overstated_statement(self) -> None:
        chunk_lookup = {
            "chunk-ai": "The course lists AI tools: Chat GPT, Microsoft CoPilot, and Gemini.",
        }
        fact = normalize_fact(
            {
                "statement": "Students must use AI tools for coding assistance.",
                "importance": "high",
                "studentAskLikelihood": 0.6,
                "complexity": 1,
                "sourceChunkIds": ["chunk-ai"],
                "evidenceQuote": "AI tools: Chat GPT, Microsoft CoPilot, and Gemini",
                "kind": "tools",
                "scope": "course_wide",
            },
            chunk_lookup=chunk_lookup,
        )
        self.assertIsNone(fact)


class ImportanceCalibrationTests(unittest.TestCase):
    def _fact(self, *, importance: str, kind: str, scope: str, statement: str) -> dict:
        return {
            "importance": importance,
            "kind": kind,
            "scope": scope,
            "statement": statement,
            "evidenceQuote": statement,
        }

    def test_resource_downranked_to_low(self) -> None:
        fact = self._fact(
            importance="high",
            kind="resource",
            scope="resource",
            statement="There is a Spotify playlist for the class.",
        )
        self.assertEqual(calibrate_importance(fact), "low")

    def test_routine_tool_capped_at_medium(self) -> None:
        fact = self._fact(
            importance="high",
            kind="tools",
            scope="course_wide",
            statement="The course uses Panopto to host lecture videos.",
        )
        self.assertEqual(calibrate_importance(fact), "medium")

    def test_policy_with_consequence_stays_high(self) -> None:
        fact = self._fact(
            importance="high",
            kind="grading",
            scope="course_wide",
            statement="Not filing the absence form will impact your grade.",
        )
        self.assertEqual(calibrate_importance(fact), "high")

    def test_assignment_deadline_capped_at_medium(self) -> None:
        fact = self._fact(
            importance="high",
            kind="deadline",
            scope="assignment_specific",
            statement="Bot Project Task #3 is due Friday, October 17.",
        )
        self.assertEqual(calibrate_importance(fact), "medium")


class AssignmentSeriesTests(unittest.TestCase):
    def test_detects_bot_project_task_series(self) -> None:
        series = detect_assignment_series("Bot Project Task #3 is due Friday.")
        self.assertIsNotNone(series)
        assert series is not None
        self.assertEqual(series["seriesKey"], "bot_project_task")
        self.assertEqual(series["seriesOrdinal"], 3)

    def test_non_series_statement_returns_none(self) -> None:
        self.assertIsNone(
            detect_assignment_series("Office hours are held by appointment.")
        )


class SemanticMergeTests(unittest.IsolatedAsyncioTestCase):
    def _fact(self, statement: str, *, importance: str = "medium", source: str = "c1") -> dict:
        return {
            "statement": statement,
            "importance": importance,
            "importanceScore": IMPORTANCE_VALUE[importance],
            "studentAskLikelihood": 0.5,
            "complexity": 1,
            "sourceChunkIds": [source],
            "evidenceQuote": statement,
            "kind": "tools",
            "scope": "course_wide",
        }

    async def test_semantic_near_duplicates_collapse(self) -> None:
        vectors = {
            "Discord is used for course chat.": [1.0, 0.0],
            "Discord is the course chat platform.": [0.98, 0.05],
            "The final exam is on December 9.": [0.0, 1.0],
        }

        async def embed(texts, *, model=None):
            return {"embeddings": [vectors[text] for text in texts], "model": "x"}

        facts = [
            self._fact("Discord is used for course chat.", source="c1"),
            self._fact(
                "Discord is the course chat platform.", importance="high", source="c2"
            ),
            self._fact("The final exam is on December 9.", source="c3"),
        ]
        merged = await semantic_merge_facts(facts, embed_fn=embed)
        self.assertEqual(len(merged), 2)
        discord = next(f for f in merged if "Discord" in f["statement"])
        # Highest-importance duplicate is kept and sources are unioned.
        self.assertEqual(discord["importance"], "high")
        self.assertEqual(sorted(discord["sourceChunkIds"]), ["c1", "c2"])

    async def test_semantic_merge_fails_open_on_embedding_error(self) -> None:
        facts = [
            self._fact("Discord is used for course chat."),
            self._fact("GitHub is used for assignments."),
        ]
        merged = await semantic_merge_facts(facts, embed_fn=_raising_embed)
        self.assertEqual(len(merged), 2)


class UsefulnessScoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_high_value_policy_outranks_minor_resource(self) -> None:
        completion_fn = AsyncMock(
            return_value={"answer": _llm_facts_payload(), "model": "qwen3:4b"}
        )
        inventory = await build_fact_inventory(
            raw_chunks=_policy_chunks(),
            completion_fn=completion_fn,
            embed_fn=_orthogonal_embed,
        )

        self.assertFalse(inventory["fallbackUsed"])
        self.assertEqual(inventory["factCount"], 2)

        facts_by_kind = {fact["kind"]: fact for fact in inventory["facts"]}
        policy = facts_by_kind["late_work"]
        resource = facts_by_kind["resource"]
        self.assertGreater(policy["usefulnessScore"], resource["usefulnessScore"])
        # High-value policy must be ranked first.
        self.assertEqual(inventory["facts"][0]["kind"], "late_work")

    def test_resource_scope_downranks_score(self) -> None:
        from app.syllabus_facts import IMPORTANCE_VALUE

        base = {
            "importance": "medium",
            "importanceScore": IMPORTANCE_VALUE["medium"],
            "studentAskLikelihood": 0.6,
            "complexity": 1,
            "statement": "Some neutral statement about the course.",
            "evidenceQuote": "Some neutral statement about the course.",
        }
        course_wide = compute_usefulness_score({**base, "scope": "course_wide"})
        resource = compute_usefulness_score({**base, "scope": "resource"})
        self.assertGreater(course_wide, resource)


def _series_chunks() -> list[dict]:
    return [
        _chunk(
            "task-1",
            "Assignments",
            "Assignments\nBot Project Task #1 is due Friday, October 3, 11:59 pm.",
            order=1,
        ),
        _chunk(
            "task-2",
            "Assignments",
            "Assignments\nBot Project Task #2 is due Friday, October 10, 11:59 pm.",
            order=2,
        ),
    ]


def _series_payload() -> str:
    return json.dumps(
        {
            "facts": [
                {
                    "statement": "Bot Project Task #1 is due Friday, October 3.",
                    "importance": "high",
                    "studentAskLikelihood": 0.8,
                    "complexity": 1,
                    "sourceChunkIds": ["task-1"],
                    "evidenceQuote": "Bot Project Task #1 is due Friday, October 3, 11:59 pm.",
                    "kind": "deadline",
                    "scope": "assignment_specific",
                },
                {
                    "statement": "Bot Project Task #2 is due Friday, October 10.",
                    "importance": "high",
                    "studentAskLikelihood": 0.8,
                    "complexity": 1,
                    "sourceChunkIds": ["task-2"],
                    "evidenceQuote": "Bot Project Task #2 is due Friday, October 10, 11:59 pm.",
                    "kind": "deadline",
                    "scope": "assignment_specific",
                },
            ]
        }
    )


class InventoryRepresentationTests(unittest.IsolatedAsyncioTestCase):
    async def test_importance_shows_variation(self) -> None:
        completion_fn = AsyncMock(
            return_value={"answer": _llm_facts_payload(), "model": "qwen3:4b"}
        )
        inventory = await build_fact_inventory(
            raw_chunks=_policy_chunks(),
            completion_fn=completion_fn,
            embed_fn=_orthogonal_embed,
        )
        importances = {fact["importance"] for fact in inventory["facts"]}
        self.assertGreater(len(importances), 1)

    async def test_assignment_series_metadata_present(self) -> None:
        completion_fn = AsyncMock(
            return_value={"answer": _series_payload(), "model": "qwen3:4b"}
        )
        inventory = await build_fact_inventory(
            raw_chunks=_series_chunks(),
            completion_fn=completion_fn,
            embed_fn=_orthogonal_embed,
        )
        self.assertEqual(inventory["factCount"], 2)
        for fact in inventory["facts"]:
            self.assertEqual(fact["seriesKey"], "bot_project_task")
            self.assertEqual(fact["assignmentGroup"], "Bot Project Task")
            self.assertIn(fact["seriesOrdinal"], {1, 2})
            # Repeated assignment-series deadlines should not all be top priority.
            self.assertEqual(fact["importance"], "medium")
        self.assertEqual(inventory["countsBySeries"].get("bot_project_task"), 2)

    async def test_reports_duplicates_removed(self) -> None:
        chunks = [
            _chunk("d1", "Tools", "Tools\nDiscord is used for course chat.", 1),
            _chunk("d2", "Tools", "Tools\nDiscord is the course chat platform.", 2),
            _chunk("d3", "Exams", "Exams\nThe final exam is on December 9.", 3),
        ]
        payload = json.dumps(
            {
                "facts": [
                    {
                        "statement": "Discord is used for course chat.",
                        "importance": "medium",
                        "studentAskLikelihood": 0.6,
                        "complexity": 1,
                        "sourceChunkIds": ["d1"],
                        "evidenceQuote": "Discord is used for course chat.",
                        "kind": "tools",
                        "scope": "course_wide",
                    },
                    {
                        "statement": "Discord is the course chat platform.",
                        "importance": "medium",
                        "studentAskLikelihood": 0.6,
                        "complexity": 1,
                        "sourceChunkIds": ["d2"],
                        "evidenceQuote": "Discord is the course chat platform.",
                        "kind": "tools",
                        "scope": "course_wide",
                    },
                    {
                        "statement": "The final exam is on December 9.",
                        "importance": "medium",
                        "studentAskLikelihood": 0.6,
                        "complexity": 1,
                        "sourceChunkIds": ["d3"],
                        "evidenceQuote": "The final exam is on December 9.",
                        "kind": "exam",
                        "scope": "schedule",
                    },
                ]
            }
        )
        vectors = {
            "Discord is used for course chat.": [1.0, 0.0],
            "Discord is the course chat platform.": [0.98, 0.05],
            "The final exam is on December 9.": [0.0, 1.0],
        }

        async def embed(texts, *, model=None):
            return {"embeddings": [vectors[text] for text in texts], "model": "x"}

        completion_fn = AsyncMock(return_value={"answer": payload, "model": "qwen3:4b"})
        inventory = await build_fact_inventory(
            raw_chunks=chunks,
            completion_fn=completion_fn,
            embed_fn=embed,
        )
        self.assertEqual(inventory["factCount"], 2)
        self.assertEqual(inventory["duplicatesRemoved"], 1)


class FallbackInventoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_llm_output_uses_fallback(self) -> None:
        completion_fn = AsyncMock(
            return_value={"answer": json.dumps({"facts": []}), "model": "qwen3:4b"}
        )
        inventory = await build_fact_inventory(
            raw_chunks=_policy_chunks(),
            completion_fn=completion_fn,
            embed_fn=_orthogonal_embed,
        )
        self.assertTrue(inventory["fallbackUsed"])
        self.assertGreater(inventory["factCount"], 0)
        for fact in inventory["facts"]:
            self.assertIn(fact["scope"], FACT_SCOPES)

    async def test_llm_failure_uses_fallback(self) -> None:
        completion_fn = AsyncMock(
            side_effect=HTTPException(status_code=503, detail="Ollama is unavailable.")
        )
        inventory = await build_fact_inventory(
            raw_chunks=_policy_chunks(),
            completion_fn=completion_fn,
            embed_fn=_orthogonal_embed,
        )
        self.assertTrue(inventory["fallbackUsed"])
        self.assertGreater(inventory["factCount"], 0)

    def test_fallback_evidence_is_verifiable(self) -> None:
        facts = build_heuristic_fallback_inventory(_policy_chunks())
        self.assertGreater(len(facts), 0)
        lookup = {chunk["chunkId"]: chunk["text"] for chunk in _policy_chunks()}
        for fact in facts:
            verified = verify_evidence_quote(
                fact["evidenceQuote"], fact["sourceChunkIds"], lookup
            )
            self.assertIsNotNone(verified)

    async def test_no_chunks_returns_empty_inventory(self) -> None:
        inventory = await build_fact_inventory(
            raw_chunks=[], completion_fn=AsyncMock(), embed_fn=_orthogonal_embed
        )
        self.assertEqual(inventory["factCount"], 0)
        self.assertEqual(inventory["facts"], [])


class FactInventoryEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        root = Path(self._temp_dir.name)
        self.storage = LocalCourseArtifactStorage(
            root_dir=root / "course_data",
            index_dir=root / "indexes",
        )
        self._storage_patch = patch(
            "app.main.get_course_artifact_storage",
            return_value=self.storage,
        )
        self._storage_patch.start()
        # Keep the endpoint hermetic: semantic merge uses a deterministic stub.
        self._embed_patch = patch(
            "app.syllabus_facts.embed_ollama_texts",
            new=_orthogonal_embed,
        )
        self._embed_patch.start()
        self.client = TestClient(app)
        self.course_id = "css-360-summer-2026-demo"
        self.url = f"/api/courses/{self.course_id}/facts/inventory"
        self.storage.save_index(self.course_id, _index(self.course_id, _policy_chunks()))

    def tearDown(self) -> None:
        self._embed_patch.stop()
        self._storage_patch.stop()
        self._temp_dir.cleanup()

    def test_endpoint_returns_inventory_without_generating_seeds(self) -> None:
        completion_fn = AsyncMock(
            return_value={"answer": _llm_facts_payload(), "model": "qwen3:4b"}
        )
        seed_spy = MagicMock()
        with patch(
            "app.syllabus_facts.generate_starter_ollama_completion",
            new=completion_fn,
        ), patch(
            "app.main.generate_starter_seeds_for_course",
            new=seed_spy,
        ):
            response = self.client.post(self.url)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["courseId"], self.course_id)
        self.assertFalse(body["fallbackUsed"])
        self.assertEqual(body["factCount"], 2)
        self.assertEqual(len(body["facts"]), 2)
        self.assertNotIn("seeds", body)
        seed_spy.assert_not_called()

        fact = body["facts"][0]
        for key in (
            "factId",
            "statement",
            "importance",
            "importanceScore",
            "studentAskLikelihood",
            "complexity",
            "usefulnessScore",
            "sourceChunkIds",
            "evidenceQuote",
            "kind",
            "scope",
            "seriesKey",
            "assignmentGroup",
            "seriesOrdinal",
        ):
            self.assertIn(key, fact)
        self.assertIn("countsByScope", body)
        self.assertIn("countsByKind", body)
        self.assertIn("countsBySeries", body)
        self.assertIn("duplicatesRemoved", body)

    def test_endpoint_missing_course_index_returns_404(self) -> None:
        response = self.client.post(
            "/api/courses/css-999-missing-course/facts/inventory"
        )
        self.assertEqual(response.status_code, 404)

    def test_endpoint_falls_back_when_llm_unavailable(self) -> None:
        completion_fn = AsyncMock(
            side_effect=HTTPException(status_code=503, detail="Ollama is unavailable.")
        )
        with patch(
            "app.syllabus_facts.generate_starter_ollama_completion",
            new=completion_fn,
        ):
            response = self.client.post(self.url)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["fallbackUsed"])
        self.assertGreater(body["factCount"], 0)


if __name__ == "__main__":
    unittest.main()
