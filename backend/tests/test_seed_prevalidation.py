"""Tests for Phase 5 deterministic pre-validation and prompt grounding rules."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.seed_generation import FACT_SEED_GENERATION_PROMPT_MARKER, _build_fact_seed_prompt
from app.seed_prevalidation import (
    detect_dropped_condition,
    detect_modal_escalation,
    detect_response_time_guarantee,
    prevalidate_candidate,
)
from app.syllabus_facts import statement_entailment_violation
from app.storage import LocalCourseArtifactStorage


def _fact(**overrides) -> dict:
    base = {
        "factId": "fact-01",
        "statement": "Students may use one 48-hour extension per quarter.",
        "importance": "high",
        "importanceScore": 0.9,
        "studentAskLikelihood": 0.9,
        "complexity": 2,
        "usefulnessScore": 0.86,
        "sourceChunkIds": ["chunk-001"],
        "evidenceQuote": (
            "Among these projects, you may choose one project for which you want "
            "to use one 48-hour extension per quarter, no questions asked."
        ),
        "kind": "late_work",
        "scope": "course_wide",
    }
    base.update(overrides)
    return base


class ModalEscalationTests(unittest.TestCase):
    def test_may_to_must_is_rejected(self) -> None:
        reason = detect_modal_escalation(
            answer="Students must use a 48-hour extension.",
            evidence="Students may use one 48-hour extension per quarter.",
        )
        self.assertEqual(reason, "modal_escalation")

    def test_recommendation_to_requirement_is_rejected(self) -> None:
        reason = detect_modal_escalation(
            answer="Using the library proxy is required.",
            evidence="Using the library proxy is recommended for off-campus access.",
        )
        self.assertEqual(reason, "recommendation_as_requirement")

    def test_important_to_must_is_rejected(self) -> None:
        """Real failure: soft 'important to' must not become 'must'."""
        evidence = (
            "It is important to tell me if you are not coming to class at least "
            "one hour before class begins."
        )
        answer = (
            "Yes, you must notify the instructor at least one hour before class "
            "if absent."
        )
        reason = detect_modal_escalation(answer=answer, evidence=evidence)
        self.assertEqual(reason, "recommendation_as_requirement")

        result = prevalidate_candidate(
            candidate={
                "question": "Do I need to notify the instructor if I miss class?",
                "answer": answer,
            },
            fact={
                "statement": evidence,
                "evidenceQuote": evidence,
                "sourceChunkIds": ["chunk-001"],
            },
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["category"], "modal_escalation")
        # Also caught by entailment once "important" is not obligation grounding.
        self.assertEqual(
            statement_entailment_violation(answer, evidence),
            "obligation_not_in_evidence",
        )

    def test_welcome_to_must_is_rejected(self) -> None:
        reason = detect_modal_escalation(
            answer="You must contact DRS to request accommodations.",
            evidence=(
                "If you have not yet established services through DRS you are "
                "welcome to contact DRS."
            ),
        )
        self.assertIn(reason, {"recommendation_as_requirement", "modal_escalation"})

    def test_optional_drs_can_wording_remains_optional(self) -> None:
        evidence = (
            "If you have not yet established services through DRS you are "
            "welcome to contact DRS."
        )
        answer = "You can contact DRS to request accommodations."
        self.assertIsNone(detect_modal_escalation(answer=answer, evidence=evidence))
        result = prevalidate_candidate(
            candidate={
                "question": "Can I contact DRS about accommodations?",
                "answer": answer,
            },
            fact={
                "statement": evidence,
                "evidenceQuote": evidence,
                "sourceChunkIds": ["chunk-001"],
            },
        )
        self.assertIsNone(result)

    def test_grounded_must_still_passes(self) -> None:
        evidence = (
            "Students must notify the instructor at least one hour before class "
            "if absent."
        )
        answer = (
            "Yes. Students must notify the instructor at least one hour before "
            "class if absent."
        )
        self.assertIsNone(detect_modal_escalation(answer=answer, evidence=evidence))
        result = prevalidate_candidate(
            candidate={
                "question": "Do I need to notify before missing class?",
                "answer": answer,
            },
            fact={
                "statement": evidence,
                "evidenceQuote": evidence,
                "sourceChunkIds": ["chunk-001"],
            },
        )
        self.assertIsNone(result)

    def test_faithful_may_wording_passes(self) -> None:
        reason = detect_modal_escalation(
            answer="Students may use one 48-hour extension per quarter.",
            evidence="Students may use one 48-hour extension per quarter.",
        )
        self.assertIsNone(reason)


class DroppedConditionTests(unittest.TestCase):
    def test_general_discussion_fallback_becoming_universal_is_rejected(self) -> None:
        evidence = (
            "If you don't see an obvious place to ask your question, go ahead "
            "and ask it in the #general-discussion channel."
        )
        question = "Where should I ask questions about assignments in Discord?"
        answer = "In the #general-discussion channel"

        reason = detect_dropped_condition(
            question=question,
            answer=answer,
            evidence=evidence,
        )
        self.assertEqual(reason, "dropped_condition")

        result = prevalidate_candidate(
            candidate={"question": question, "answer": answer},
            fact={
                "statement": evidence,
                "evidenceQuote": evidence,
                "sourceChunkIds": ["chunk-001"],
            },
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["category"], "qualifier_mismatch")
        self.assertEqual(result["reason"], "dropped_condition")

    def test_answer_preserving_condition_passes(self) -> None:
        evidence = (
            "If you don't see an obvious place to ask your question, go ahead "
            "and ask it in the #general-discussion channel."
        )
        question = (
            "Where should I ask if I don't see an obvious Discord channel for "
            "my question?"
        )
        answer = (
            "If you don't see an obvious place to ask, use the "
            "#general-discussion channel."
        )
        self.assertIsNone(
            detect_dropped_condition(
                question=question,
                answer=answer,
                evidence=evidence,
            )
        )
        result = prevalidate_candidate(
            candidate={"question": question, "answer": answer},
            fact={
                "statement": evidence,
                "evidenceQuote": evidence,
                "sourceChunkIds": ["chunk-001"],
            },
        )
        self.assertIsNone(result)


class ResponseTimeGuaranteeTests(unittest.TestCase):
    def test_nudge_is_not_a_guarantee(self) -> None:
        evidence = (
            "If I do not respond within 24 hours, please nudge me. "
            "A delayed reply is usually unintentional."
        )
        reason = detect_response_time_guarantee(
            answer="The instructor guarantees a response within 24 hours.",
            evidence=evidence,
        )
        self.assertEqual(reason, "response_time_guarantee")


class PrevalidateCandidateTests(unittest.TestCase):
    def test_modal_escalation_category(self) -> None:
        result = prevalidate_candidate(
            candidate={
                "question": "Can I get an extension?",
                "answer": "You must request a 48-hour extension.",
            },
            fact=_fact(),
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["category"], "modal_escalation")

    def test_missing_one_extension_limit(self) -> None:
        result = prevalidate_candidate(
            candidate={
                "question": "How many extensions can I use?",
                "answer": "You may use extensions on projects whenever you need them.",
            },
            fact=_fact(),
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["category"], "qualifier_mismatch")

    def test_faithful_candidate_passes_prevalidation(self) -> None:
        result = prevalidate_candidate(
            candidate={
                "question": "How many 48-hour extensions do I get?",
                "answer": (
                    "You may choose one project for a single 48-hour extension "
                    "per quarter."
                ),
            },
            fact=_fact(),
        )
        self.assertIsNone(result)


class FactPromptTests(unittest.TestCase):
    def test_prompt_includes_anti_escalation_rules(self) -> None:
        prompt = _build_fact_seed_prompt(
            fact=_fact(),
            chunk_texts=["Evidence chunk text about extensions."],
            count=1,
        )
        self.assertIn(FACT_SEED_GENERATION_PROMPT_MARKER, prompt)
        self.assertIn('may" / "can"', prompt)
        self.assertIn("must", prompt.lower())
        self.assertIn("guarantee", prompt.lower())
        self.assertIn("exactly ONE", prompt)


class PrevalidationSkipsLlmTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        root = Path(self._temp.name)
        self.storage = LocalCourseArtifactStorage(
            root_dir=root / "course_data",
            index_dir=root / "indexes",
        )
        self.course_id = "css-360-phase5-demo"
        chunks = [
            {
                "chunkId": "chunk-001",
                "sectionTitle": "Late Policy",
                "text": (
                    "Late Policy. Among these projects, you may choose one project "
                    "for which you want to use one 48-hour extension per quarter."
                )
                * 2,
                "order": 1,
            },
            {
                "chunkId": "chunk-002",
                "sectionTitle": "Attendance",
                "text": (
                    "Attendance. Students must notify the instructor at least one "
                    "hour before class if absent."
                )
                * 2,
                "order": 2,
            },
        ]
        self.storage.save_index(
            self.course_id,
            {"courseId": self.course_id, "chunkCount": 2, "chunks": chunks},
        )
        self.facts = [
            _fact(
                factId="fact-01",
                sourceChunkIds=["chunk-001"],
            ),
            _fact(
                factId="fact-02",
                statement="Students must notify the instructor before class if absent.",
                evidenceQuote=(
                    "Students must notify the instructor at least one hour before "
                    "class if absent."
                ),
                kind="attendance",
                sourceChunkIds=["chunk-002"],
                complexity=1,
            ),
        ]

    def tearDown(self) -> None:
        self._temp.cleanup()

    async def test_escalated_candidate_does_not_call_llm_validation(self) -> None:
        from app.seed_generation import (
            SEED_GENERATION_MODEL,
            VALIDATION_PROMPT_MARKER,
            generate_starter_seeds_for_course,
        )

        inventory = {
            "model": SEED_GENERATION_MODEL,
            "facts": self.facts,
            "factCount": 2,
            "droppedCount": 0,
            "duplicatesRemoved": 0,
            "fallbackUsed": False,
            "cached": True,
            "countsByScope": {},
            "countsByKind": {},
            "countsBySeries": {},
        }

        calls = {"validation": 0, "generation": 0}

        async def _fake(prompt: str, **kwargs):
            if VALIDATION_PROMPT_MARKER in prompt:
                calls["validation"] += 1
                return {
                    "answer": json.dumps(
                        {
                            "grounded": 0.9,
                            "correct": 0.9,
                            "clear": 0.85,
                            "useful": 0.85,
                            "naturalStudentWording": 0.85,
                            "categoryCorrect": 0.8,
                            "notTrivialOrTemporary": 0.8,
                            "unsupportedClaims": [],
                            "reason": "ok",
                        }
                    ),
                    "model": SEED_GENERATION_MODEL,
                }
            calls["generation"] += 1
            if calls["generation"] == 1:
                # Escalated answer — should be rejected pre-validation.
                payload = {
                    "seeds": [
                        {
                            "question": "Do I have to use an extension?",
                            "answer": "You must use a 48-hour extension on a project.",
                            "category": "late work",
                            "questionType": "direct",
                        }
                    ]
                }
            else:
                payload = {
                    "seeds": [
                        {
                            "question": "Do I need to notify before missing class?",
                            "answer": (
                                "Yes. Students must notify the instructor at least "
                                "one hour before class if absent."
                            ),
                            "category": "attendance",
                            "questionType": "direct",
                        }
                    ]
                }
            return {"answer": json.dumps(payload), "model": SEED_GENERATION_MODEL}

        with (
            patch(
                "app.seed_generation.load_or_build_fact_inventory",
                new=AsyncMock(return_value=inventory),
            ),
            patch(
                "app.seed_generation.generate_starter_ollama_completion",
                new=AsyncMock(side_effect=_fake),
            ),
            patch(
                "app.seed_generation.embed_ollama_texts",
                new=AsyncMock(
                    return_value={"embeddings": [[1.0, 0.0], [0.0, 1.0]], "model": "t"}
                ),
            ),
        ):
            result = await generate_starter_seeds_for_course(
                course_id=self.course_id,
                target_count=1,
                storage=self.storage,
            )

        self.assertGreaterEqual(
            result["progress"]["candidatesRejectedPreValidation"], 1
        )
        self.assertGreaterEqual(
            result["progress"]["candidatesRejectedModalEscalation"], 1
        )
        # Only the faithful second candidate should reach LLM validation.
        self.assertEqual(calls["validation"], 1)
        self.assertEqual(result["progress"]["validationCalls"], 1)
        self.assertEqual(result["progress"]["finalCount"], 1)
        self.assertTrue(result["progress"]["factInventoryCached"])
        self.assertEqual(result["progress"]["factExtractionCalls"], 0)

    async def test_valid_candidate_still_requires_llm_validation(self) -> None:
        from app.seed_generation import (
            SEED_GENERATION_MODEL,
            VALIDATION_PROMPT_MARKER,
            generate_starter_seeds_for_course,
        )

        inventory = {
            "model": SEED_GENERATION_MODEL,
            "facts": [self.facts[1]],
            "factCount": 1,
            "droppedCount": 0,
            "duplicatesRemoved": 0,
            "fallbackUsed": False,
            "cached": True,
            "countsByScope": {},
            "countsByKind": {},
            "countsBySeries": {},
        }

        async def _fake(prompt: str, **kwargs):
            if VALIDATION_PROMPT_MARKER in prompt:
                return {
                    "answer": json.dumps(
                        {
                            "grounded": 0.9,
                            "correct": 0.9,
                            "clear": 0.85,
                            "useful": 0.85,
                            "naturalStudentWording": 0.85,
                            "categoryCorrect": 0.8,
                            "notTrivialOrTemporary": 0.8,
                            "unsupportedClaims": [],
                            "reason": "ok",
                        }
                    ),
                    "model": SEED_GENERATION_MODEL,
                }
            return {
                "answer": json.dumps(
                    {
                        "seeds": [
                            {
                                "question": "Do I need to notify before missing class?",
                                "answer": (
                                    "Yes. Students must notify the instructor at "
                                    "least one hour before class if absent."
                                ),
                                "category": "attendance",
                                "questionType": "direct",
                            }
                        ]
                    }
                ),
                "model": SEED_GENERATION_MODEL,
            }

        with (
            patch(
                "app.seed_generation.load_or_build_fact_inventory",
                new=AsyncMock(return_value=inventory),
            ),
            patch(
                "app.seed_generation.generate_starter_ollama_completion",
                new=AsyncMock(side_effect=_fake),
            ),
            patch(
                "app.seed_generation.embed_ollama_texts",
                new=AsyncMock(
                    return_value={"embeddings": [[1.0]], "model": "t"}
                ),
            ),
        ):
            result = await generate_starter_seeds_for_course(
                course_id=self.course_id,
                target_count=1,
                storage=self.storage,
            )

        self.assertEqual(result["progress"]["validationCalls"], 1)
        self.assertEqual(result["progress"]["candidatesRejectedPreValidation"], 0)
        self.assertEqual(result["progress"]["finalCount"], 1)
        self.assertIsNotNone(result["seeds"][0].get("validation"))


if __name__ == "__main__":
    unittest.main()
