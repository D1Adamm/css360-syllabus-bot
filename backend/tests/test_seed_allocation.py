"""Tests for deterministic fact-inventory slot allocation (Phase 3)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.seed_allocation import (
    MAX_SLOTS_PER_FACT,
    SCENARIO_LIKE_STYLES,
    allocate_slots,
    compute_allocation_caps,
    compute_desired_slots,
    compute_ranking_score,
    fact_supports_scenario_like_style,
    suggest_question_styles,
)
from app.storage import LocalCourseArtifactStorage


def _fact(
    fact_id: str,
    *,
    statement: str = "A syllabus fact.",
    importance: str = "medium",
    importance_score: float = 0.6,
    ask: float = 0.5,
    complexity: int = 1,
    usefulness: float = 0.6,
    kind: str = "policy",
    scope: str = "course_wide",
    source_chunk_ids: list[str] | None = None,
    series_key: str | None = None,
    assignment_group: str | None = None,
) -> dict:
    return {
        "factId": fact_id,
        "statement": statement,
        "importance": importance,
        "importanceScore": importance_score,
        "studentAskLikelihood": ask,
        "complexity": complexity,
        "usefulnessScore": usefulness,
        "sourceChunkIds": source_chunk_ids or [f"chunk-{fact_id}"],
        "evidenceQuote": statement,
        "kind": kind,
        "scope": scope,
        "seriesKey": series_key,
        "assignmentGroup": assignment_group,
        "seriesOrdinal": None,
    }


class DesiredSlotsTests(unittest.TestCase):
    def test_contact_high_usefulness_low_complexity_gets_one(self) -> None:
        fact = _fact(
            "fact-contact",
            statement="Instructor email is instructor@uw.edu.",
            importance="high",
            importance_score=0.9,
            ask=0.85,
            complexity=1,
            usefulness=0.85,
            kind="contact",
            scope="course_wide",
        )
        result = compute_desired_slots(fact)
        self.assertEqual(result["desiredSlots"], 1)
        self.assertIn("simple_contact_or_lookup_stays_at_one", result["reasons"])

    def test_complex_late_work_gets_multiple_slots(self) -> None:
        fact = _fact(
            "fact-late",
            statement=(
                "One 48-hour extension is allowed for projects 1-7 except Demo "
                "and Reflection which have no extensions."
            ),
            importance="high",
            importance_score=0.9,
            ask=0.9,
            complexity=3,
            usefulness=0.86,
            kind="late_work",
            scope="course_wide",
        )
        result = compute_desired_slots(fact)
        self.assertGreaterEqual(result["desiredSlots"], 2)
        self.assertLessEqual(result["desiredSlots"], MAX_SLOTS_PER_FACT)
        self.assertTrue(
            any(
                reason
                in {
                    "high_complexity_multiple_conditions",
                    "policy_with_conditions_or_exceptions",
                    "moderate_complexity_with_high_ask",
                }
                for reason in result["reasons"]
            )
        )

    def test_minor_resource_gets_zero_desired(self) -> None:
        fact = _fact(
            "fact-pantry",
            statement="The campus pantry is an optional food resource.",
            importance="low",
            importance_score=0.3,
            ask=0.2,
            complexity=1,
            usefulness=0.25,
            kind="resource",
            scope="resource",
        )
        result = compute_desired_slots(fact)
        self.assertEqual(result["desiredSlots"], 0)
        self.assertIn("minor_resource_deprioritized", result["reasons"])


class AllocationBehaviorTests(unittest.TestCase):
    def test_total_allocated_slots_at_most_target(self) -> None:
        facts = [
            _fact(
                f"fact-{index:02d}",
                importance="high",
                importance_score=0.9,
                ask=0.8,
                complexity=2,
                usefulness=0.8,
                kind="policy",
                scope="course_wide",
                source_chunk_ids=[f"chunk-{index}"],
            )
            for index in range(1, 40)
        ]
        result = allocate_slots(facts, target_count=50)
        self.assertLessEqual(result["summary"]["allocatedSlots"], 50)
        self.assertEqual(result["summary"]["targetCount"], 50)

    def test_allocation_is_deterministic(self) -> None:
        facts = [
            _fact(
                "fact-01",
                importance="high",
                importance_score=0.9,
                ask=0.9,
                complexity=3,
                usefulness=0.86,
                kind="late_work",
            ),
            _fact(
                "fact-02",
                importance="high",
                importance_score=0.9,
                ask=0.8,
                complexity=1,
                usefulness=0.82,
                kind="contact",
            ),
            _fact(
                "fact-03",
                importance="low",
                importance_score=0.3,
                ask=0.2,
                complexity=1,
                usefulness=0.2,
                kind="resource",
                scope="resource",
            ),
            _fact(
                "fact-04",
                importance="medium",
                importance_score=0.6,
                ask=0.6,
                complexity=1,
                usefulness=0.55,
                kind="deadline",
                scope="assignment_specific",
                series_key="project",
            ),
        ]
        first = allocate_slots(facts, target_count=10)
        second = allocate_slots(facts, target_count=10)
        self.assertEqual(first, second)

    def test_minor_resource_gets_zero_when_better_facts_exist(self) -> None:
        facts = [
            _fact(
                "fact-late",
                importance="high",
                importance_score=0.9,
                ask=0.9,
                complexity=3,
                usefulness=0.86,
                kind="late_work",
            ),
            _fact(
                "fact-attend",
                importance="high",
                importance_score=0.9,
                ask=0.85,
                complexity=2,
                usefulness=0.84,
                kind="attendance",
            ),
            _fact(
                "fact-grade",
                importance="high",
                importance_score=0.9,
                ask=0.8,
                complexity=2,
                usefulness=0.82,
                kind="grading",
            ),
            _fact(
                "fact-pantry",
                importance="low",
                importance_score=0.3,
                ask=0.2,
                complexity=1,
                usefulness=0.2,
                kind="resource",
                scope="resource",
            ),
        ]
        result = allocate_slots(facts, target_count=8)
        by_id = {item["factId"]: item for item in result["allocations"]}
        self.assertEqual(by_id["fact-pantry"]["slotCount"], 0)
        self.assertGreaterEqual(by_id["fact-late"]["slotCount"], 2)

    def test_contact_gets_exactly_one_slot(self) -> None:
        facts = [
            _fact(
                "fact-contact",
                importance="high",
                importance_score=0.9,
                ask=0.9,
                complexity=1,
                usefulness=0.85,
                kind="contact",
            ),
            _fact(
                "fact-late",
                importance="high",
                importance_score=0.9,
                ask=0.9,
                complexity=3,
                usefulness=0.86,
                kind="late_work",
            ),
        ]
        result = allocate_slots(facts, target_count=10)
        by_id = {item["factId"]: item for item in result["allocations"]}
        self.assertEqual(by_id["fact-contact"]["slotCount"], 1)

    def test_assignment_heavy_inventory_preserves_course_wide(self) -> None:
        facts = [
            _fact(
                "fact-cw-late",
                importance="high",
                importance_score=0.9,
                ask=0.9,
                complexity=3,
                usefulness=0.86,
                kind="late_work",
                scope="course_wide",
                source_chunk_ids=["chunk-cw-1"],
            ),
            _fact(
                "fact-cw-attend",
                importance="high",
                importance_score=0.9,
                ask=0.85,
                complexity=2,
                usefulness=0.84,
                kind="attendance",
                scope="course_wide",
                source_chunk_ids=["chunk-cw-2"],
            ),
            _fact(
                "fact-cw-grade",
                importance="high",
                importance_score=0.9,
                ask=0.8,
                complexity=2,
                usefulness=0.82,
                kind="grading",
                scope="course_wide",
                source_chunk_ids=["chunk-cw-3"],
            ),
        ]
        for index in range(1, 25):
            facts.append(
                _fact(
                    f"fact-as-{index:02d}",
                    importance="medium",
                    importance_score=0.6,
                    ask=0.55,
                    complexity=1,
                    usefulness=0.55,
                    kind="deadline",
                    scope="assignment_specific",
                    source_chunk_ids=[f"chunk-as-{index}"],
                    series_key="project",
                )
            )

        result = allocate_slots(facts, target_count=20)
        summary = result["summary"]
        self.assertGreater(summary["byScope"].get("course_wide", 0), 0)
        # Soft reservation should keep course_wide from being starved.
        self.assertGreaterEqual(
            summary["courseWideAllocated"],
            min(summary["courseWideReserve"], 3),
        )
        assignment_slots = summary["byScope"].get("assignment_specific", 0)
        caps = compute_allocation_caps(20)
        self.assertLessEqual(
            assignment_slots, caps["perScope"]["assignment_specific"]
        )

    def test_repeated_assignment_series_deadlines_are_capped(self) -> None:
        facts = [
            _fact(
                f"fact-deadline-{index:02d}",
                importance="medium",
                importance_score=0.6,
                ask=0.55,
                complexity=1,
                usefulness=0.55,
                kind="deadline",
                scope="assignment_specific",
                source_chunk_ids=[f"chunk-{index}"],
                series_key="bot_project",
            )
            for index in range(1, 20)
        ]
        # Add one strong course-wide fact so the series does not fill everything.
        facts.append(
            _fact(
                "fact-cw",
                importance="high",
                importance_score=0.9,
                ask=0.9,
                complexity=2,
                usefulness=0.85,
                kind="late_work",
                scope="course_wide",
                source_chunk_ids=["chunk-cw"],
            )
        )
        result = allocate_slots(facts, target_count=30)
        series_slots = result["summary"]["bySeries"].get("bot_project", 0)
        caps = compute_allocation_caps(30)
        self.assertLessEqual(series_slots, caps["perSeries"])

    def test_one_source_chunk_cannot_dominate(self) -> None:
        shared = "chunk-shared"
        facts = [
            _fact(
                f"fact-{index:02d}",
                importance="high",
                importance_score=0.9,
                ask=0.85,
                complexity=2,
                usefulness=0.84,
                kind="policy",
                scope="course_wide",
                source_chunk_ids=[shared],
            )
            for index in range(1, 15)
        ]
        # Competing facts from other chunks.
        for index in range(15, 25):
            facts.append(
                _fact(
                    f"fact-{index:02d}",
                    importance="high",
                    importance_score=0.85,
                    ask=0.8,
                    complexity=2,
                    usefulness=0.8,
                    kind="attendance",
                    scope="course_wide",
                    source_chunk_ids=[f"chunk-other-{index}"],
                )
            )

        result = allocate_slots(facts, target_count=40)
        caps = compute_allocation_caps(40)
        shared_slots = 0
        for row in result["ranking"]:
            if shared in row["sourceChunkIds"]:
                shared_slots += row["slotCount"]
        self.assertLessEqual(shared_slots, caps["perSourceChunk"])

    def test_kind_and_scope_caps_apply(self) -> None:
        facts = [
            _fact(
                f"fact-deadline-{index:02d}",
                importance="medium",
                importance_score=0.65,
                ask=0.6,
                complexity=1,
                usefulness=0.6,
                kind="deadline",
                scope="schedule",
                source_chunk_ids=[f"chunk-d-{index}"],
            )
            for index in range(1, 30)
        ]
        facts.append(
            _fact(
                "fact-policy",
                importance="high",
                importance_score=0.9,
                ask=0.9,
                complexity=2,
                usefulness=0.85,
                kind="late_work",
                scope="course_wide",
                source_chunk_ids=["chunk-policy"],
            )
        )
        result = allocate_slots(facts, target_count=40)
        caps = compute_allocation_caps(40)
        deadline_slots = result["summary"]["byKind"].get("deadline", 0)
        schedule_slots = result["summary"]["byScope"].get("schedule", 0)
        self.assertLessEqual(deadline_slots, caps["deadlineKind"])
        self.assertLessEqual(schedule_slots, caps["perScope"]["schedule"])

    def test_multi_slot_reasons_exposed(self) -> None:
        facts = [
            _fact(
                "fact-late",
                importance="high",
                importance_score=0.9,
                ask=0.9,
                complexity=3,
                usefulness=0.86,
                kind="late_work",
            )
        ]
        result = allocate_slots(facts, target_count=5)
        alloc = result["allocations"][0]
        self.assertGreaterEqual(alloc["slotCount"], 2)
        self.assertIn(
            "additional_slot_justified_by_complexity_or_scenarios",
            alloc["reasons"],
        )
        self.assertTrue(len(alloc["suggestedStyles"]) >= 1)

    def test_ranking_metadata_explains_zero_one_and_multi(self) -> None:
        facts = [
            _fact(
                "fact-multi",
                importance="high",
                importance_score=0.9,
                ask=0.9,
                complexity=3,
                usefulness=0.86,
                kind="late_work",
            ),
            _fact(
                "fact-one",
                importance="high",
                importance_score=0.9,
                ask=0.85,
                complexity=1,
                usefulness=0.85,
                kind="contact",
            ),
            _fact(
                "fact-zero",
                importance="low",
                importance_score=0.3,
                ask=0.2,
                complexity=1,
                usefulness=0.2,
                kind="resource",
                scope="resource",
            ),
        ]
        result = allocate_slots(facts, target_count=10)
        by_id = {item["factId"]: item for item in result["ranking"]}
        self.assertGreaterEqual(by_id["fact-multi"]["slotCount"], 2)
        self.assertEqual(by_id["fact-one"]["slotCount"], 1)
        self.assertEqual(by_id["fact-zero"]["slotCount"], 0)
        self.assertGreater(by_id["fact-multi"]["rankingScore"], 0)
        skipped_ids = {item["factId"] for item in result["summary"]["skippedFacts"]}
        self.assertIn("fact-zero", skipped_ids)


class RankingScoreTests(unittest.TestCase):
    def test_course_wide_policy_outranks_minor_resource(self) -> None:
        policy = _fact(
            "a",
            importance="high",
            importance_score=0.9,
            ask=0.9,
            complexity=2,
            usefulness=0.86,
            kind="late_work",
            scope="course_wide",
        )
        resource = _fact(
            "b",
            importance="low",
            importance_score=0.3,
            ask=0.2,
            complexity=1,
            usefulness=0.25,
            kind="resource",
            scope="resource",
        )
        self.assertGreater(compute_ranking_score(policy), compute_ranking_score(resource))


async def _orthogonal_embed(texts, *, model=None):
    count = len(texts)
    embeddings = [
        [1.0 if row == col else 0.0 for col in range(count)] for row in range(count)
    ]
    return {"embeddings": embeddings, "model": model or "test-embed"}


class FactAllocationEndpointTests(unittest.TestCase):
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
        self._embed_patch = patch(
            "app.syllabus_facts.embed_ollama_texts",
            new=_orthogonal_embed,
        )
        self._embed_patch.start()
        self.client = TestClient(app)
        self.course_id = "css-360-summer-2026-alloc"
        self.url = f"/api/courses/{self.course_id}/facts/allocation"
        self.storage.save_index(
            self.course_id,
            {
                "courseId": self.course_id,
                "embeddingModel": "nomic-embed-text",
                "chunkCount": 2,
                "chunks": [
                    {
                        "chunkId": "chunk-001",
                        "sectionTitle": "Late Work Policy",
                        "text": (
                            "Late Work Policy\nOne 48-hour extension is allowed "
                            "for Bot Projects 1 through 7."
                        ),
                        "order": 1,
                    },
                    {
                        "chunkId": "chunk-002",
                        "sectionTitle": "Campus Resources",
                        "text": (
                            "Campus Resources\nThe Husky Pantry is an optional "
                            "food resource for students."
                        ),
                        "order": 2,
                    },
                ],
            },
        )

    def tearDown(self) -> None:
        self._embed_patch.stop()
        self._storage_patch.stop()
        self._temp_dir.cleanup()

    def test_endpoint_returns_allocation_without_generating_seeds(self) -> None:
        llm_payload = json.dumps(
            {
                "facts": [
                    {
                        "statement": (
                            "One 48-hour extension is allowed for Bot Projects "
                            "1 through 7."
                        ),
                        "importance": "high",
                        "studentAskLikelihood": 0.9,
                        "complexity": 2,
                        "sourceChunkIds": ["chunk-001"],
                        "evidenceQuote": (
                            "One 48-hour extension is allowed for Bot Projects "
                            "1 through 7."
                        ),
                        "kind": "late_work",
                        "scope": "course_wide",
                    },
                    {
                        "statement": "The Husky Pantry is an optional food resource.",
                        "importance": "low",
                        "studentAskLikelihood": 0.2,
                        "complexity": 1,
                        "sourceChunkIds": ["chunk-002"],
                        "evidenceQuote": (
                            "The Husky Pantry is an optional food resource for "
                            "students."
                        ),
                        "kind": "resource",
                        "scope": "resource",
                    },
                ]
            }
        )
        completion_fn = AsyncMock(
            return_value={"answer": llm_payload, "model": "qwen3:4b"}
        )
        seed_spy = MagicMock()
        with patch(
            "app.syllabus_facts.generate_starter_ollama_completion",
            new=completion_fn,
        ), patch(
            "app.main.generate_starter_seeds_for_course",
            new=seed_spy,
        ):
            response = self.client.post(
                self.url,
                json={"targetCount": 10},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["courseId"], self.course_id)
        self.assertIn("allocations", body)
        self.assertIn("summary", body)
        self.assertIn("ranking", body)
        self.assertIn("facts", body)
        self.assertNotIn("seeds", body)
        self.assertLessEqual(body["summary"]["allocatedSlots"], 10)
        self.assertEqual(body["summary"]["targetCount"], 10)
        seed_spy.assert_not_called()

        for key in (
            "targetCount",
            "allocatedSlots",
            "byScope",
            "byKind",
            "bySeries",
            "skippedFacts",
            "cappedFacts",
        ):
            self.assertIn(key, body["summary"])

        if body["allocations"]:
            alloc = body["allocations"][0]
            for key in (
                "factId",
                "slotCount",
                "desiredSlots",
                "rankingScore",
                "suggestedStyles",
                "reasons",
            ):
                self.assertIn(key, alloc)

    def test_endpoint_missing_course_returns_404(self) -> None:
        response = self.client.post(
            "/api/courses/css-999-missing-course/facts/allocation",
            json={"targetCount": 50},
        )
        self.assertEqual(response.status_code, 404)


class SingleSlotScenarioStyleTests(unittest.TestCase):
    """The CSS 350 regression: scenario styles unreachable at one slot per fact.

    Breadth-first allocation gives every fact exactly one slot whenever facts
    are plentiful. Scenario-like styles are never the first entry in any branch,
    so truncating to one slot dropped all of them — and the run's scenario
    minimum became unreachable before the model was even asked.
    """

    ALL_KINDS = (
        "contact",
        "office_hours",
        "deadline",
        "late_work",
        "attendance",
        "policy",
        "accommodation",
        "grading",
        "requirement",
        "submission",
        "communication",
        "other",
    )

    @staticmethod
    def _kind_fact(kind: str, complexity: int) -> dict:
        return {"kind": kind, "complexity": complexity}

    def test_scenario_style_survives_truncation_to_one_slot(self) -> None:
        styles = suggest_question_styles(
            self._kind_fact("late_work", 2), 1, prefer_scenario_like=True
        )

        self.assertEqual(styles, ["scenario"])

    def test_default_behaviour_is_unchanged(self) -> None:
        """Without the flag, allocation keeps producing exactly what it did."""
        for kind in self.ALL_KINDS:
            for complexity in (1, 2, 3, 4):
                with self.subTest(kind=kind, complexity=complexity):
                    styles = suggest_question_styles(
                        self._kind_fact(kind, complexity), 1
                    )
                    self.assertEqual(len(styles), 1)
                    self.assertNotIn(styles[0], SCENARIO_LIKE_STYLES)

    def test_preference_never_invents_a_style_the_fact_did_not_earn(self) -> None:
        """Simple lookups stay direct even when a run is starved of scenarios."""
        for kind in ("contact", "office_hours"):
            for complexity in (1, 2, 3, 4):
                with self.subTest(kind=kind, complexity=complexity):
                    styles = suggest_question_styles(
                        self._kind_fact(kind, complexity), 1, prefer_scenario_like=True
                    )
                    self.assertEqual(styles, ["factual"])

        simple_deadline = suggest_question_styles(
            self._kind_fact("deadline", 1), 1, prefer_scenario_like=True
        )
        self.assertEqual(simple_deadline, ["factual"])

    def test_communication_facts_stay_procedural(self) -> None:
        styles = suggest_question_styles(
            self._kind_fact("communication", 3), 1, prefer_scenario_like=True
        )
        self.assertEqual(styles, ["procedural"])

    def test_some_kind_reaches_a_scenario_style_at_one_slot(self) -> None:
        """The invariant whose absence let the regression ship.

        It is not enough that individual kinds behave; at one slot per fact,
        *something* has to be able to produce a scenario-like style, or the
        minimum is arithmetically unreachable for every possible syllabus.
        """
        reachable = [
            (kind, complexity)
            for kind in self.ALL_KINDS
            for complexity in (1, 2, 3, 4)
            if any(
                style in SCENARIO_LIKE_STYLES
                for style in suggest_question_styles(
                    self._kind_fact(kind, complexity), 1, prefer_scenario_like=True
                )
            )
        ]

        self.assertTrue(reachable)

    def test_eligibility_helper_matches_what_the_styles_do(self) -> None:
        self.assertTrue(
            fact_supports_scenario_like_style(self._kind_fact("late_work", 2), 1)
        )
        self.assertTrue(
            fact_supports_scenario_like_style(self._kind_fact("policy", 3), 1)
        )
        self.assertFalse(
            fact_supports_scenario_like_style(self._kind_fact("contact", 1), 1)
        )
        self.assertFalse(
            fact_supports_scenario_like_style(self._kind_fact("deadline", 1), 1)
        )
        self.assertFalse(
            fact_supports_scenario_like_style(self._kind_fact("communication", 4), 1)
        )

    def test_multi_slot_facts_keep_every_style_they_earned(self) -> None:
        """Promotion reorders; it must not drop the fact's other styles."""
        styles = suggest_question_styles(
            self._kind_fact("late_work", 3), 3, prefer_scenario_like=True
        )

        self.assertEqual(sorted(styles), ["exception", "policy", "scenario"])
        self.assertIn(styles[0], SCENARIO_LIKE_STYLES)

    def test_allocation_still_stores_unpromoted_styles(self) -> None:
        """Allocation has no deficit knowledge, so its output is untouched."""
        facts = [
            _fact(
                f"fact-{index:02d}",
                kind="late_work",
                complexity=2,
                source_chunk_ids=[f"chunk-{index:03d}"],
            )
            for index in range(1, 6)
        ]
        allocation = allocate_slots(facts=facts, target_count=5)

        for entry in allocation["allocations"]:
            if entry["slotCount"] == 1:
                self.assertNotIn(entry["suggestedStyles"][0], SCENARIO_LIKE_STYLES)


if __name__ == "__main__":
    unittest.main()
