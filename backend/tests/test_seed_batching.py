"""Phase 6 batching helpers (unused by live starter path) and sequential-path guard."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from app.seed_batching import (
    build_batch_fact_seed_prompt,
    build_batch_validation_prompt,
    parse_batch_seed_payload,
    parse_batch_validation_payload,
)
from app.seed_generation import (
    SEED_GENERATION_MODEL,
    VALIDATION_PROMPT_MARKER,
    _generate_candidates_for_facts_batch,
    _validate_candidates_batch,
    generate_starter_seeds_for_course,
)
from app.seed_prevalidation import prevalidate_candidate
from app.storage import LocalCourseArtifactStorage


def _fact(fact_id: str, statement: str, chunk_id: str = "chunk-001") -> dict:
    return {
        "factId": fact_id,
        "statement": statement,
        "importance": "high",
        "importanceScore": 0.9,
        "studentAskLikelihood": 0.9,
        "complexity": 1,
        "usefulnessScore": 0.85,
        "sourceChunkIds": [chunk_id],
        "evidenceQuote": statement,
        "kind": "policy",
        "scope": "course_wide",
        "seriesKey": None,
    }


def _passing_components(**overrides) -> dict:
    base = {
        "grounded": 0.9,
        "correct": 0.91,
        "clear": 0.82,
        "useful": 0.84,
        "naturalStudentWording": 0.83,
        "categoryCorrect": 0.88,
        "notTrivialOrTemporary": 0.8,
        "unsupportedClaims": [],
        "reason": "Grounded and useful.",
    }
    base.update(overrides)
    return base


class ParseBatchSeedPayloadTests(unittest.TestCase):
    def test_maps_by_fact_id(self) -> None:
        raw = json.dumps(
            {
                "seeds": [
                    {
                        "factId": "fact-b",
                        "question": "Question B?",
                        "answer": "Answer B with enough detail.",
                    },
                    {
                        "factId": "fact-a",
                        "question": "Question A?",
                        "answer": "Answer A with enough detail.",
                    },
                ]
            }
        )
        parsed = parse_batch_seed_payload(
            raw, expected_fact_ids=["fact-a", "fact-b"]
        )
        self.assertEqual([item["factId"] for item in parsed], ["fact-a", "fact-b"])
        self.assertEqual(parsed[0]["question"], "Question A?")
        self.assertEqual(parsed[1]["question"], "Question B?")

    def test_positional_fallback_and_partial_recovery(self) -> None:
        raw = (
            '{"seeds":['
            '{"question":"Q1?","answer":"Answer one is long enough."},'
            '{"question":"Q2?","answer":"Answer two is long enough."},'
            '{"question":"broken"'  # malformed trailing item
        )
        parsed = parse_batch_seed_payload(
            raw, expected_fact_ids=["fact-1", "fact-2", "fact-3"]
        )
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["factId"], "fact-1")
        self.assertEqual(parsed[1]["factId"], "fact-2")

    def test_malformed_item_does_not_drop_valid_siblings(self) -> None:
        raw = json.dumps(
            {
                "seeds": [
                    {
                        "factId": "fact-1",
                        "question": "Good question one?",
                        "answer": "Good answer one with detail.",
                    },
                    "not-an-object",
                    {
                        "factId": "fact-3",
                        "question": "Good question three?",
                        "answer": "Good answer three with detail.",
                    },
                ]
            }
        )
        parsed = parse_batch_seed_payload(
            raw, expected_fact_ids=["fact-1", "fact-2", "fact-3"]
        )
        self.assertEqual(
            [(item["factId"], item["question"]) for item in parsed],
            [
                ("fact-1", "Good question one?"),
                ("fact-3", "Good question three?"),
            ],
        )


class ParseBatchValidationPayloadTests(unittest.TestCase):
    def test_maps_results_by_candidate_id(self) -> None:
        raw = json.dumps(
            {
                "results": [
                    {"candidateId": "c1", **_passing_components(reason="second")},
                    {"candidateId": "c0", **_passing_components(reason="first")},
                ]
            }
        )
        mapped = parse_batch_validation_payload(
            raw, expected_candidate_ids=["c0", "c1"]
        )
        assert mapped is not None
        self.assertEqual(mapped["c0"]["reason"], "first")
        self.assertEqual(mapped["c1"]["reason"], "second")

    def test_one_malformed_item_does_not_corrupt_valid_items(self) -> None:
        raw = json.dumps(
            {
                "results": [
                    {"candidateId": "c0", **_passing_components()},
                    {"candidateId": "c1", "grounded": "bad", "reason": "x"},
                    {"candidateId": "c2", **_passing_components(reason="ok-c2")},
                ]
            }
        )
        mapped = parse_batch_validation_payload(
            raw, expected_candidate_ids=["c0", "c1", "c2"]
        )
        assert mapped is not None
        self.assertIsNotNone(mapped["c0"])
        self.assertIsNone(mapped["c1"])
        self.assertEqual(mapped["c2"]["reason"], "ok-c2")

    def test_single_rubric_shape_only_for_one_candidate(self) -> None:
        raw = json.dumps(_passing_components())
        self.assertIsNone(
            parse_batch_validation_payload(raw, expected_candidate_ids=["c0", "c1"])
        )
        mapped = parse_batch_validation_payload(
            raw, expected_candidate_ids=["c0"]
        )
        assert mapped is not None
        self.assertIsNotNone(mapped["c0"])


class BatchGenerateValidateUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_batch_preserves_fact_mapping(self) -> None:
        items = [
            {
                "fact": _fact("fact-a", "Late work may be submitted within 24 hours."),
                "chunk_texts": ["Late work may be submitted within 24 hours."],
                "suggested_styles": ["factual"],
            },
            {
                "fact": _fact("fact-b", "Teams must submit weekly status updates."),
                "chunk_texts": ["Teams must submit weekly status updates."],
                "suggested_styles": ["factual"],
            },
        ]

        async def _fake(prompt: str, **kwargs: object) -> dict[str, str]:
            self.assertIn("fact-a", prompt)
            self.assertIn("fact-b", prompt)
            return {
                "answer": json.dumps(
                    {
                        "seeds": [
                            {
                                "factId": "fact-b",
                                "question": "Do teams submit weekly updates?",
                                "answer": "Yes, teams must submit weekly status updates.",
                                "category": "Projects",
                                "questionType": "direct",
                            },
                            {
                                "factId": "fact-a",
                                "question": "Can I submit late work?",
                                "answer": "Late work may be submitted within 24 hours.",
                                "category": "Late work",
                                "questionType": "direct",
                            },
                        ]
                    }
                ),
                "model": SEED_GENERATION_MODEL,
            }

        results, _, metrics = await _generate_candidates_for_facts_batch(
            items=items,
            completion_fn=_fake,
        )
        self.assertEqual(metrics["generation_calls"], 1)
        self.assertEqual(metrics["generation_batch_calls"], 1)
        self.assertEqual(results["fact-a"][0]["factId"], "fact-a")
        self.assertEqual(results["fact-b"][0]["factId"], "fact-b")
        self.assertEqual(
            results["fact-a"][0]["evidenceQuote"],
            "Late work may be submitted within 24 hours.",
        )
        self.assertEqual(results["fact-a"][0]["sourceChunkIds"], ["chunk-001"])

    async def test_generate_batch_split_on_malformed(self) -> None:
        items = [
            {
                "fact": _fact("fact-a", "Policy A text for grounding."),
                "chunk_texts": ["Policy A text for grounding."],
                "suggested_styles": ["factual"],
            },
            {
                "fact": _fact("fact-b", "Policy B text for grounding."),
                "chunk_texts": ["Policy B text for grounding."],
                "suggested_styles": ["factual"],
            },
        ]
        calls = {"n": 0}

        async def _fake(prompt: str, **kwargs: object) -> dict[str, str]:
            calls["n"] += 1
            if "fact-a" in prompt and "fact-b" in prompt:
                return {"answer": "not-json", "model": SEED_GENERATION_MODEL}
            fact_id = "fact-a" if "fact-a" in prompt else "fact-b"
            return {
                "answer": json.dumps(
                    {
                        "seeds": [
                            {
                                "factId": fact_id,
                                "question": f"Question for {fact_id}?",
                                "answer": f"Answer for {fact_id} with detail.",
                                "category": "General",
                                "questionType": "direct",
                            }
                        ]
                    }
                ),
                "model": SEED_GENERATION_MODEL,
            }

        results, _, metrics = await _generate_candidates_for_facts_batch(
            items=items,
            completion_fn=_fake,
        )
        self.assertGreaterEqual(metrics["generation_calls"], 3)
        self.assertEqual(len(results["fact-a"]), 1)
        self.assertEqual(len(results["fact-b"]), 1)
        self.assertEqual(calls["n"], 3)

    async def test_validate_batch_maps_and_splits(self) -> None:
        candidates = [
            {
                "candidateId": "c0",
                "question": "Q0?",
                "answer": "A0",
                "topic_name": "General",
                "question_type": "direct",
                "chunk_text": "source",
            },
            {
                "candidateId": "c1",
                "question": "Q1?",
                "answer": "A1",
                "topic_name": "General",
                "question_type": "direct",
                "chunk_text": "source",
            },
        ]
        calls = {"n": 0}

        async def _fake(prompt: str, **kwargs: object) -> dict[str, str]:
            calls["n"] += 1
            if "c0" in prompt and "c1" in prompt:
                return {"answer": "{bad", "model": SEED_GENERATION_MODEL}
            candidate_id = "c0" if "### Candidate id: c0" in prompt or (
                "Question:\nQ0?" in prompt
            ) else "c1"
            payload = _passing_components(reason=f"ok-{candidate_id}")
            return {"answer": json.dumps(payload), "model": SEED_GENERATION_MODEL}

        mapped, metrics = await _validate_candidates_batch(
            candidates=candidates,
            completion_fn=_fake,
        )
        self.assertGreaterEqual(metrics["validation_calls"], 3)
        self.assertIsNotNone(mapped["c0"])
        self.assertIsNotNone(mapped["c1"])
        self.assertEqual(mapped["c0"]["reason"], "ok-c0")
        self.assertEqual(mapped["c1"]["reason"], "ok-c1")


class Phase6StarterIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.storage = LocalCourseArtifactStorage(Path(self._temp_dir.name))
        self.course_id = "css-360-batch-test"
        chunks = []
        facts = []
        for index in range(1, 8):
            chunk_id = f"chunk-{index:03d}"
            text = (
                f"Course policy {index}: students may request clarification on "
                f"assignment {index} within one week. " + ("x" * 40)
            )
            chunks.append(
                {
                    "chunkId": chunk_id,
                    "id": chunk_id,
                    "text": text,
                    "order": index,
                    "sectionPath": ["Policies", f"Item {index}"],
                }
            )
            facts.append(
                _fact(
                    f"fact-{index:02d}",
                    f"Students may request clarification on assignment {index} "
                    "within one week.",
                    chunk_id=chunk_id,
                )
            )
        self.storage.save_index(
            self.course_id,
            {"courseId": self.course_id, "chunks": chunks},
        )
        self.facts = facts

        async def _fake_inventory(**kwargs):
            return {
                "courseId": self.course_id,
                "facts": self.facts,
                "factCount": len(self.facts),
                "cached": True,
                "model": SEED_GENERATION_MODEL,
                "stats": {},
            }

        self._inventory_patch = patch(
            "app.seed_generation.load_or_build_fact_inventory",
            new=AsyncMock(side_effect=_fake_inventory),
        )
        self._inventory_patch.start()
        self._embed_patch = patch(
            "app.seed_generation.embed_ollama_texts",
            new=AsyncMock(side_effect=self._orthogonal_embed),
        )
        self._embed_patch.start()

    @staticmethod
    async def _orthogonal_embed(texts, *, model=None):
        dim = max(16, len(texts) + 8)
        embeddings: list[list[float]] = []
        for text in texts:
            vector = [0.0] * dim
            bucket = sum(ord(ch) for ch in str(text)) % dim
            vector[bucket] = 1.0
            vector[(bucket + len(str(text)) + 1) % dim] = 0.5
            embeddings.append(vector)
        return {"embeddings": embeddings, "model": model or "test-embed"}

    async def asyncTearDown(self) -> None:
        self._inventory_patch.stop()
        self._embed_patch.stop()
        self._temp_dir.cleanup()

    async def test_live_starter_uses_sequential_not_batching(self) -> None:
        calls = {"gen": 0, "val": 0}

        async def _fake(prompt: str, **kwargs: object) -> dict[str, str]:
            if VALIDATION_PROMPT_MARKER in prompt:
                calls["val"] += 1
                # Live path must use single-candidate validation prompts.
                self.assertLessEqual(prompt.count("### Candidate id:"), 0)
                return {
                    "answer": json.dumps(_passing_components()),
                    "model": SEED_GENERATION_MODEL,
                }
            calls["gen"] += 1
            # Live path must use single-fact generation prompts.
            self.assertNotIn("multiple syllabus facts", prompt.lower())
            import re

            fact_ids = re.findall(r"Fact id:\s*(\S+)", prompt)
            if not fact_ids:
                fact_ids = ["fact-01"]
            seeds = [
                {
                    "factId": fact_id,
                    "question": f"What is the clarification window for {fact_id}?",
                    "answer": (
                        "Students may request clarification within one week."
                    ),
                    "category": "Policies",
                    "questionType": "direct",
                    "sourceChunkIds": ["chunk-001"],
                }
                for fact_id in fact_ids
            ]
            return {
                "answer": json.dumps({"seeds": seeds}),
                "model": SEED_GENERATION_MODEL,
            }

        with patch(
            "app.seed_generation.generate_starter_ollama_completion",
            new=AsyncMock(side_effect=_fake),
        ):
            result = await generate_starter_seeds_for_course(
                course_id=self.course_id,
                target_count=5,
                storage=self.storage,
            )

        progress = result["progress"]
        self.assertEqual(progress["finalCount"], 5)
        self.assertEqual(progress["factInventoryCached"], True)
        self.assertEqual(progress["factExtractionCalls"], 0)
        self.assertEqual(progress["generationBatchCalls"], 0)
        self.assertEqual(progress["validationBatchCalls"], 0)
        self.assertEqual(progress["maxGenerationBatchSize"], 0)
        self.assertEqual(progress["maxValidationBatchSize"], 0)
        self.assertEqual(progress["candidatesAccepted"], 5)
        self.assertEqual(calls["gen"], progress["generationCalls"])
        self.assertEqual(calls["val"], progress["validationCalls"])
        self.assertGreaterEqual(progress["validationCalls"], 1)
        self.assertGreaterEqual(progress["candidatesValidated"], 5)
        for seed in result["seeds"]:
            self.assertTrue(seed.get("factId"))
            self.assertTrue(seed.get("evidenceQuote"))
            self.assertTrue(seed.get("validation"))

    async def test_prevalidation_still_blocks_before_llm(self) -> None:
        rejection = prevalidate_candidate(
            candidate={
                "question": "Must I use the extension?",
                "answer": "Students must use one 48-hour extension.",
                "category": "Late work",
                "questionType": "direct",
                "sourceChunkIds": ["chunk-001"],
            },
            fact=_fact(
                "fact-01",
                "Students may use one 48-hour extension per quarter.",
            ),
            source_text="Students may use one 48-hour extension per quarter.",
        )
        self.assertIsNotNone(rejection)

    def test_batch_prompts_include_markers(self) -> None:
        gen_prompt = build_batch_fact_seed_prompt(
            items=[
                {
                    "fact": _fact("fact-a", "Statement A"),
                    "chunk_texts": ["Chunk A"],
                },
                {
                    "fact": _fact("fact-b", "Statement B"),
                    "chunk_texts": ["Chunk B"],
                },
            ]
        )
        self.assertIn("fact-a", gen_prompt)
        self.assertIn("fact-b", gen_prompt)
        val_prompt = build_batch_validation_prompt(
            candidates=[
                {
                    "candidateId": "c0",
                    "question": "Q?",
                    "answer": "A",
                    "topic_name": "T",
                    "question_type": "direct",
                    "chunk_text": "S",
                }
            ]
        )
        self.assertIn(VALIDATION_PROMPT_MARKER, val_prompt)


if __name__ == "__main__":
    unittest.main()
