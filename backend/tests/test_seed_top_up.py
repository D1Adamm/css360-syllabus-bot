"""Tests for safe dataset top-up against existing Firebase seeds."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.firebase_seeds import (
    compute_top_up_gap,
    summarize_existing_seed_examples,
)
from app.seed_dedupe import normalize_question_for_dedupe
from app.seed_generation import (
    SEED_GENERATION_MODEL,
    VALIDATION_PROMPT_MARKER,
    generate_starter_seeds_for_course,
)
from app.storage import LocalCourseArtifactStorage


def _passing_validation() -> str:
    return json.dumps(
        {
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
    )


def _fact(fact_id: str, statement: str, chunk_id: str) -> dict:
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


class TopUpGapHelperTests(unittest.TestCase):
    def test_compute_gap(self) -> None:
        self.assertEqual(
            compute_top_up_gap(existing_count=33, target_count=50),
            {
                "existingCount": 33,
                "targetCount": 50,
                "missingCount": 17,
                "alreadyComplete": False,
            },
        )
        self.assertTrue(
            compute_top_up_gap(existing_count=50, target_count=50)["alreadyComplete"]
        )
        self.assertEqual(
            compute_top_up_gap(existing_count=60, target_count=50)["missingCount"],
            0,
        )

    def test_summarize_existing_seeds(self) -> None:
        summary = summarize_existing_seed_examples(
            {
                "a": {
                    "question": "Can I submit late?",
                    "factId": "fact-01",
                    "normalizedQuestionKey": "custom-key",
                },
                "b": {"instruction": "What is grading?", "factId": "fact-02"},
                "c": "bad",
            }
        )
        self.assertEqual(summary["existingCount"], 2)
        self.assertIn("fact-01", summary["factIds"])
        self.assertIn("fact-02", summary["factIds"])
        self.assertIn("custom-key", summary["seenQuestionKeys"])
        self.assertIn(
            normalize_question_for_dedupe("What is grading?"),
            summary["seenQuestionKeys"],
        )


class TopUpGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        root = Path(self._temp.name)
        self.storage = LocalCourseArtifactStorage(
            root_dir=root / "course_data",
            index_dir=root / "indexes",
        )
        self.course_id = "css-360-winter-2026-a7rp"
        chunks = []
        facts = []
        for index in range(1, 12):
            chunk_id = f"chunk-{index:03d}"
            text = (
                f"Course policy {index}: students may request clarification on "
                f"topic {index} within one week. " + ("detail " * 20)
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
                    f"Students may request clarification on topic {index} "
                    "within one week.",
                    chunk_id,
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

        self._inventory = patch(
            "app.seed_generation.load_or_build_fact_inventory",
            new=AsyncMock(side_effect=_fake_inventory),
        )
        self._inventory.start()
        self._embed = patch(
            "app.seed_generation.embed_ollama_texts",
            new=AsyncMock(side_effect=self._orthogonal_embed),
        )
        self._embed.start()

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
        self._inventory.stop()
        self._embed.stop()
        self._temp.cleanup()

    async def test_top_up_noop_when_already_at_target(self) -> None:
        existing = {
            f"seed-{i}": {
                "question": f"Existing question {i}?",
                "answer": f"Existing answer {i} with enough detail.",
                "factId": f"fact-{i:02d}",
            }
            for i in range(1, 51)
        }
        with patch(
            "app.seed_generation.fetch_course_seed_examples",
            new=AsyncMock(return_value=existing),
        ) as mock_fetch:
            with patch(
                "app.seed_generation.generate_starter_ollama_completion",
                new=AsyncMock(),
            ) as mock_ollama:
                result = await generate_starter_seeds_for_course(
                    course_id=self.course_id,
                    target_count=50,
                    save=False,
                    top_up=True,
                    storage=self.storage,
                )

        mock_fetch.assert_awaited()
        mock_ollama.assert_not_called()
        progress = result["progress"]
        self.assertTrue(progress["topUp"])
        self.assertEqual(progress["existingCount"], 50)
        self.assertEqual(progress["missingCount"], 0)
        self.assertEqual(progress["finalCount"], 0)
        self.assertEqual(progress["totalCount"], 50)
        self.assertEqual(progress["status"], "ready")
        self.assertEqual(progress["ollamaCalls"], 0)

    async def test_top_up_generates_only_missing_and_dedupes(self) -> None:
        existing = {
            "seed-1": {
                "question": "Existing locked question one?",
                "answer": "Existing answer one.",
                "factId": "fact-01",
            },
            "seed-2": {
                "question": "Existing locked question two?",
                "answer": "Existing answer two.",
                "factId": "fact-02",
            },
            "seed-3": {
                "question": "Existing locked question three?",
                "answer": "Existing answer three.",
                "factId": "fact-03",
            },
        }
        call_state = {"n": 0}

        async def _fake(prompt: str, **kwargs: object) -> dict[str, str]:
            if VALIDATION_PROMPT_MARKER in prompt:
                return {"answer": _passing_validation(), "model": SEED_GENERATION_MODEL}
            call_state["n"] += 1
            # First generation tries a duplicate of an existing question.
            if call_state["n"] == 1:
                question = "Existing locked question one?"
            else:
                question = f"Brand new top-up question {call_state['n']}?"
            return {
                "answer": json.dumps(
                    {
                        "seeds": [
                            {
                                "question": question,
                                "answer": (
                                    "Students may request clarification within "
                                    "one week for this policy."
                                ),
                                "category": "Policies",
                                "questionType": "direct",
                            }
                        ]
                    }
                ),
                "model": SEED_GENERATION_MODEL,
            }

        saved_records: list[dict] = []

        async def _fake_persist(**kwargs):
            seeds = kwargs["seeds"]
            saved_records.extend(seeds)
            return {
                "generatedCount": len(seeds),
                "savedCount": len(seeds),
                "alreadyExistingCount": 0,
                "failedToSaveCount": 0,
            }

        with (
            patch(
                "app.seed_generation.fetch_course_seed_examples",
                new=AsyncMock(return_value=existing),
            ),
            patch(
                "app.seed_generation.generate_starter_ollama_completion",
                new=AsyncMock(side_effect=_fake),
            ),
            patch(
                "app.seed_generation.persist_accepted_seeds",
                new=AsyncMock(side_effect=_fake_persist),
            ),
        ):
            result = await generate_starter_seeds_for_course(
                course_id=self.course_id,
                target_count=5,
                save=True,
                top_up=True,
                storage=self.storage,
            )

        progress = result["progress"]
        self.assertTrue(progress["topUp"])
        self.assertEqual(progress["existingCount"], 3)
        self.assertEqual(progress["missingCount"], 2)
        self.assertEqual(progress["finalCount"], 2)
        self.assertEqual(progress["savedCount"], 2)
        self.assertEqual(progress["totalCount"], 5)
        self.assertGreaterEqual(progress["duplicatesRemoved"], 1)
        self.assertEqual(len(saved_records), 2)
        for seed in saved_records:
            self.assertNotEqual(
                normalize_question_for_dedupe(seed["question"]),
                normalize_question_for_dedupe("Existing locked question one?"),
            )


if __name__ == "__main__":
    unittest.main()
