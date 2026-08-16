"""Fact extraction must say when it loses part of a syllabus.

The CSS 350 regression: a 140-chunk syllabus that had previously yielded 80
facts produced 8, and nothing in the run said why. Batches whose response was
truncated mid-JSON were discarded and reported as if they had simply contained
nothing, so a half-failed extraction and a thin syllabus looked identical — and
every count downstream is a function of the fact count, so the run reported
9 examples out of 50 with no way to tell which had happened.

These tests pin the distinction. None of them call Ollama.
"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from app.syllabus_facts import (
    BATCH_OUTCOME_CALL_FAILED,
    BATCH_OUTCOME_EMPTY,
    BATCH_OUTCOME_OK,
    BATCH_OUTCOME_PARSE_FAILED,
    build_fact_inventory,
)


async def _orthogonal_embed(texts, *, model=None):
    """Distinct vectors per text, so semantic merge never collapses fixtures."""
    dim = 16
    vectors = []
    for text in texts:
        vector = [0.0] * dim
        bucket = sum(ord(char) for char in str(text)) % dim
        vector[bucket] = 1.0
        vector[(bucket + len(str(text)) + 1) % dim] = 0.5
        vectors.append(vector)
    return {"embeddings": vectors, "model": model or "fake-embed"}


def _css350_chunks(count: int = 40) -> list[dict[str, object]]:
    """A CSS 350-shaped index: many chunks, one section title.

    The single shared `sectionTitle` is what makes this course batch the way it
    does — `build_section_groups` cannot split on section, so packing falls to
    the character budget alone.
    """
    return [
        {
            "chunkId": f"c{index:03d}",
            "order": index,
            "sectionTitle": "CSS 350 Syllabus",
            "text": (
                f"Policy {index}: assignment {index} is due on day {index} of the "
                f"quarter, and late submissions for assignment {index} are "
                f"accepted within 24 hours with a 10 percent penalty. "
            )
            * 12,
        }
        for index in range(count)
    ]


def _facts_payload(chunk_id: str, *, count: int = 2) -> str:
    text = (
        f"Policy {chunk_id.lstrip('c').lstrip('0') or '0'}: assignment "
        f"{chunk_id.lstrip('c').lstrip('0') or '0'} is due on day "
        f"{chunk_id.lstrip('c').lstrip('0') or '0'} of the quarter"
    )
    return json.dumps(
        {
            "facts": [
                {
                    "statement": f"{text} (fact {index}).",
                    "importance": "high",
                    "studentAskLikelihood": 0.8,
                    "complexity": 2,
                    "sourceChunkIds": [chunk_id],
                    "evidenceQuote": text,
                    "kind": "deadline",
                    "scope": "assignment_specific",
                }
                for index in range(count)
            ]
        }
    )


def _first_chunk_id(prompt: str, chunks: list[dict[str, object]]) -> str:
    """Which chunk this batch's prompt is about, for building a valid answer."""
    for chunk in chunks:
        if str(chunk["chunkId"]) in prompt:
            return str(chunk["chunkId"])
    return str(chunks[0]["chunkId"])


#: A response cut off mid-object, exactly as an exhausted num_predict leaves it.
TRUNCATED_ANSWER = (
    '{"facts": [{"statement": "Assignment 1 is due on day 1 of the quarter.", '
    '"importance": "high", "studentAskLikelihood": 0.8, "complexity": 2, '
    '"sourceChunkIds": ["c000"], "evidenceQuote": "Policy 1: assignment 1 is du'
)


class BatchOutcomeTests(unittest.IsolatedAsyncioTestCase):
    """The four ways a batch can end are counted separately."""

    async def test_healthy_extraction_records_no_failures(self) -> None:
        chunks = _css350_chunks(12)

        async def completion(prompt, **kwargs):
            return {"answer": _facts_payload(_first_chunk_id(prompt, chunks)), "model": "m"}

        inventory = await build_fact_inventory(
            raw_chunks=chunks,
            completion_fn=AsyncMock(side_effect=completion),
            embed_fn=_orthogonal_embed,
        )

        extraction = inventory["extraction"]
        self.assertGreater(extraction["batchCount"], 0)
        self.assertEqual(extraction["batchCount"], extraction["batchesOk"])
        self.assertEqual(extraction["batchesParseFailed"], 0)
        self.assertEqual(extraction["batchesCallFailed"], 0)
        self.assertGreater(inventory["factCount"], 0)
        self.assertFalse(inventory["fallbackUsed"])

    async def test_truncated_json_is_counted_not_swallowed(self) -> None:
        """The CSS 350 signature: some batches parse, most are cut off.

        `fallbackUsed` stays False — the heuristic fallback only fires when
        every batch fails — so before this change the run had no signal at all
        that most of the syllabus had been dropped.
        """
        chunks = _css350_chunks(40)
        calls = {"n": 0}

        async def completion(prompt, **kwargs):
            calls["n"] += 1
            if calls["n"] % 4 != 1:
                return {"answer": TRUNCATED_ANSWER, "model": "m"}
            return {"answer": _facts_payload(_first_chunk_id(prompt, chunks)), "model": "m"}

        inventory = await build_fact_inventory(
            raw_chunks=chunks,
            completion_fn=AsyncMock(side_effect=completion),
            embed_fn=_orthogonal_embed,
        )

        extraction = inventory["extraction"]
        self.assertGreater(extraction["batchesParseFailed"], 0)
        self.assertEqual(extraction["batchesCallFailed"], 0)
        self.assertGreater(extraction["batchesOk"], 0)
        # Partial failure, not total — which is why the fallback never fired.
        self.assertFalse(inventory["fallbackUsed"])
        self.assertEqual(
            extraction["batchCount"],
            extraction["batchesOk"]
            + extraction["batchesEmpty"]
            + extraction["batchesParseFailed"]
            + extraction["batchesCallFailed"],
        )

    async def test_call_failures_are_counted_apart_from_parse_failures(self) -> None:
        """A timeout and a truncated answer need different fixes."""
        chunks = _css350_chunks(12)
        completion = AsyncMock(
            side_effect=HTTPException(status_code=503, detail="Ollama timed out.")
        )

        inventory = await build_fact_inventory(
            raw_chunks=chunks,
            completion_fn=completion,
            embed_fn=_orthogonal_embed,
        )

        extraction = inventory["extraction"]
        self.assertGreater(extraction["batchesCallFailed"], 0)
        self.assertEqual(extraction["batchesParseFailed"], 0)
        # Total failure still reaches the heuristic fallback, unchanged.
        self.assertTrue(inventory["fallbackUsed"])

    async def test_an_empty_answer_is_not_a_failure(self) -> None:
        """A model that read the batch and found nothing has not failed."""
        chunks = _css350_chunks(8)
        completion = AsyncMock(
            return_value={"answer": json.dumps({"facts": []}), "model": "m"}
        )

        inventory = await build_fact_inventory(
            raw_chunks=chunks,
            completion_fn=completion,
            embed_fn=_orthogonal_embed,
        )

        extraction = inventory["extraction"]
        self.assertGreater(extraction["batchesEmpty"], 0)
        self.assertEqual(extraction["batchesParseFailed"], 0)
        self.assertEqual(extraction["batchesCallFailed"], 0)

    async def test_outcome_vocabulary_is_distinct(self) -> None:
        self.assertEqual(
            len(
                {
                    BATCH_OUTCOME_OK,
                    BATCH_OUTCOME_EMPTY,
                    BATCH_OUTCOME_PARSE_FAILED,
                    BATCH_OUTCOME_CALL_FAILED,
                }
            ),
            4,
        )


class InventoryOutputBudgetTests(unittest.IsolatedAsyncioTestCase):
    """The output cap that was truncating extraction."""

    async def test_extraction_asks_for_more_output_than_it_used_to(self) -> None:
        from app.ollama import get_starter_inventory_num_predict

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STARTER_INVENTORY_NUM_PREDICT", None)
            # 1024 could not hold a batch's worth of facts with verbatim quotes.
            self.assertGreaterEqual(get_starter_inventory_num_predict(), 3072)

    async def test_num_predict_stays_env_overridable(self) -> None:
        from app.ollama import get_starter_inventory_num_predict

        with patch.dict(os.environ, {"STARTER_INVENTORY_NUM_PREDICT": "512"}):
            self.assertEqual(get_starter_inventory_num_predict(), 512)


class InventoryModelTests(unittest.IsolatedAsyncioTestCase):
    """Extraction may run on a different model from generation."""

    async def test_defaults_to_the_generation_model(self) -> None:
        from app.ollama import SEED_GENERATION_MODEL, get_starter_inventory_model

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STARTER_INVENTORY_MODEL", None)
            self.assertEqual(get_starter_inventory_model(), SEED_GENERATION_MODEL)

    async def test_extraction_uses_the_configured_inventory_model(self) -> None:
        chunks = _css350_chunks(6)

        async def completion(prompt, **kwargs):
            return {"answer": _facts_payload(_first_chunk_id(prompt, chunks)), "model": "m"}

        completion_fn = AsyncMock(side_effect=completion)

        with patch.dict(os.environ, {"STARTER_INVENTORY_MODEL": "qwen3:4b"}):
            inventory = await build_fact_inventory(
                raw_chunks=chunks,
                completion_fn=completion_fn,
                embed_fn=_orthogonal_embed,
            )

        self.assertEqual(inventory["model"], "qwen3:4b")
        for call in completion_fn.await_args_list:
            self.assertEqual(call.kwargs["model"], "qwen3:4b")

    async def test_an_explicit_model_argument_still_wins(self) -> None:
        chunks = _css350_chunks(4)

        async def completion(prompt, **kwargs):
            return {"answer": _facts_payload(_first_chunk_id(prompt, chunks)), "model": "m"}

        with patch.dict(os.environ, {"STARTER_INVENTORY_MODEL": "qwen3:4b"}):
            inventory = await build_fact_inventory(
                raw_chunks=chunks,
                completion_fn=AsyncMock(side_effect=completion),
                embed_fn=_orthogonal_embed,
                model="explicit-model",
            )

        self.assertEqual(inventory["model"], "explicit-model")


class InventoryCacheVersionTests(unittest.IsolatedAsyncioTestCase):
    """An inventory built before the fix must not survive it."""

    async def test_a_previous_version_cache_is_rebuilt(self) -> None:
        import tempfile
        from pathlib import Path

        from app.fact_inventory_cache import (
            FACT_INVENTORY_CACHE_VERSION,
            compute_index_fingerprint,
            load_or_build_fact_inventory,
        )
        from app.storage import LocalCourseArtifactStorage

        # The bump is the point: without it a course whose cache holds a
        # degraded 8-fact inventory keeps it forever.
        self.assertGreaterEqual(FACT_INVENTORY_CACHE_VERSION, 2)

        chunks = _css350_chunks(6)
        course_id = "css-350-winter-2026-drlb"

        with tempfile.TemporaryDirectory() as tmp:
            storage = LocalCourseArtifactStorage(Path(tmp))
            storage.save_fact_inventory(
                course_id,
                {
                    "cacheVersion": 1,
                    "indexFingerprint": compute_index_fingerprint(chunks),
                    "inventory": {
                        "model": "qwen3:4b",
                        "facts": [],
                        "factCount": 0,
                        "droppedCount": 0,
                        "duplicatesRemoved": 0,
                        "fallbackUsed": False,
                    },
                },
            )

            async def completion(prompt, **kwargs):
                return {
                    "answer": _facts_payload(_first_chunk_id(prompt, chunks)),
                    "model": "m",
                }

            inventory = await load_or_build_fact_inventory(
                course_id=course_id,
                raw_chunks=chunks,
                storage=storage,
                completion_fn=AsyncMock(side_effect=completion),
                embed_fn=_orthogonal_embed,
            )

        self.assertFalse(inventory["cached"])
        self.assertGreater(inventory["factCount"], 0)
        self.assertIn("extraction", inventory)


if __name__ == "__main__":
    unittest.main()
