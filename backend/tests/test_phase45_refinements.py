"""Tests for Phase 4.5: fact-inventory cache, breadth-first alloc, backfill."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.fact_inventory_cache import (
    compute_index_fingerprint,
    load_or_build_fact_inventory,
)
from app.seed_allocation import allocate_slots
from app.storage import LocalCourseArtifactStorage


def _chunk(chunk_id: str, text: str, order: int = 1) -> dict:
    return {
        "chunkId": chunk_id,
        "sectionTitle": chunk_id,
        "text": text,
        "order": order,
    }


def _fact(
    fact_id: str,
    *,
    kind: str = "policy",
    scope: str = "course_wide",
    complexity: int = 1,
    usefulness: float = 0.85,
    ask: float = 0.85,
    importance: str = "high",
    importance_score: float = 0.9,
    chunk: str = "chunk-001",
) -> dict:
    return {
        "factId": fact_id,
        "statement": f"Statement for {fact_id}",
        "importance": importance,
        "importanceScore": importance_score,
        "studentAskLikelihood": ask,
        "complexity": complexity,
        "usefulnessScore": usefulness,
        "sourceChunkIds": [chunk],
        "evidenceQuote": f"Evidence for {fact_id}",
        "kind": kind,
        "scope": scope,
        "seriesKey": None,
        "assignmentGroup": None,
        "seriesOrdinal": None,
    }


class IndexFingerprintTests(unittest.TestCase):
    def test_fingerprint_changes_when_chunk_text_changes(self) -> None:
        a = [_chunk("c1", "Late work policy A" * 10)]
        b = [_chunk("c1", "Late work policy B" * 10)]
        self.assertNotEqual(
            compute_index_fingerprint(a),
            compute_index_fingerprint(b),
        )

    def test_fingerprint_stable_for_same_chunks(self) -> None:
        chunks = [_chunk("c1", "Same text" * 20, order=2)]
        self.assertEqual(
            compute_index_fingerprint(chunks),
            compute_index_fingerprint(chunks),
        )


class FactInventoryCacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        root = Path(self._temp.name)
        self.storage = LocalCourseArtifactStorage(
            root_dir=root / "course_data",
            index_dir=root / "indexes",
        )
        self.course_id = "css-360-cache-demo"
        self.chunks = [
            _chunk("chunk-001", "Attendance policy requires notice. " * 8),
            _chunk("chunk-002", "Grading uses a fixed scale. " * 8, order=2),
        ]
        self.storage.save_index(
            self.course_id,
            {
                "courseId": self.course_id,
                "chunkCount": 2,
                "chunks": self.chunks,
            },
        )

    def tearDown(self) -> None:
        self._temp.cleanup()

    async def test_cache_reuse_skips_extraction(self) -> None:
        build_calls = {"n": 0}

        async def _build(**kwargs):
            build_calls["n"] += 1
            return {
                "model": "qwen3:4b",
                "facts": [_fact("fact-01")],
                "factCount": 1,
                "droppedCount": 0,
                "duplicatesRemoved": 0,
                "fallbackUsed": False,
                "countsByScope": {},
                "countsByKind": {},
                "countsBySeries": {},
            }

        with patch(
            "app.fact_inventory_cache.build_fact_inventory",
            new=AsyncMock(side_effect=_build),
        ):
            first = await load_or_build_fact_inventory(
                course_id=self.course_id,
                raw_chunks=self.chunks,
                storage=self.storage,
            )
            second = await load_or_build_fact_inventory(
                course_id=self.course_id,
                raw_chunks=self.chunks,
                storage=self.storage,
            )

        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(build_calls["n"], 1)
        self.assertEqual(second["factCount"], 1)
        self.assertTrue(self.storage.fact_inventory_path(self.course_id).is_file())

    async def test_force_refresh_rebuilds(self) -> None:
        build_calls = {"n": 0}

        async def _build(**kwargs):
            build_calls["n"] += 1
            return {
                "model": "qwen3:4b",
                "facts": [_fact(f"fact-{build_calls['n']:02d}")],
                "factCount": 1,
                "droppedCount": 0,
                "duplicatesRemoved": 0,
                "fallbackUsed": False,
                "countsByScope": {},
                "countsByKind": {},
                "countsBySeries": {},
            }

        with patch(
            "app.fact_inventory_cache.build_fact_inventory",
            new=AsyncMock(side_effect=_build),
        ):
            await load_or_build_fact_inventory(
                course_id=self.course_id,
                raw_chunks=self.chunks,
                storage=self.storage,
            )
            refreshed = await load_or_build_fact_inventory(
                course_id=self.course_id,
                raw_chunks=self.chunks,
                storage=self.storage,
                force_refresh=True,
            )

        self.assertEqual(build_calls["n"], 2)
        self.assertFalse(refreshed["cached"])
        self.assertEqual(refreshed["facts"][0]["factId"], "fact-02")

    async def test_index_change_invalidates_cache(self) -> None:
        build_calls = {"n": 0}

        async def _build(**kwargs):
            build_calls["n"] += 1
            return {
                "model": "qwen3:4b",
                "facts": [_fact(f"fact-{build_calls['n']:02d}")],
                "factCount": 1,
                "droppedCount": 0,
                "duplicatesRemoved": 0,
                "fallbackUsed": False,
                "countsByScope": {},
                "countsByKind": {},
                "countsBySeries": {},
            }

        with patch(
            "app.fact_inventory_cache.build_fact_inventory",
            new=AsyncMock(side_effect=_build),
        ):
            await load_or_build_fact_inventory(
                course_id=self.course_id,
                raw_chunks=self.chunks,
                storage=self.storage,
            )
            # Replacing the syllabus index must drop the facts cache.
            new_chunks = [
                _chunk("chunk-001", "REPLACED syllabus text about extensions. " * 8)
            ]
            self.storage.save_index(
                self.course_id,
                {
                    "courseId": self.course_id,
                    "chunkCount": 1,
                    "chunks": new_chunks,
                },
            )
            self.assertFalse(
                self.storage.fact_inventory_path(self.course_id).is_file()
            )
            rebuilt = await load_or_build_fact_inventory(
                course_id=self.course_id,
                raw_chunks=new_chunks,
                storage=self.storage,
            )

        self.assertEqual(build_calls["n"], 2)
        self.assertFalse(rebuilt["cached"])

    async def test_no_cross_course_contamination(self) -> None:
        other_id = "css-361-other-course"
        other_chunks = [_chunk("chunk-x", "Other course syllabus content. " * 8)]
        self.storage.save_index(
            other_id,
            {"courseId": other_id, "chunkCount": 1, "chunks": other_chunks},
        )

        async def _build_for(**kwargs):
            # Distinguish by first chunk text.
            chunks = kwargs.get("raw_chunks") or []
            label = "A" if "Attendance" in str(chunks[0].get("text")) else "B"
            return {
                "model": "qwen3:4b",
                "facts": [_fact(f"fact-{label}")],
                "factCount": 1,
                "droppedCount": 0,
                "duplicatesRemoved": 0,
                "fallbackUsed": False,
                "countsByScope": {},
                "countsByKind": {},
                "countsBySeries": {},
            }

        with patch(
            "app.fact_inventory_cache.build_fact_inventory",
            new=AsyncMock(side_effect=_build_for),
        ):
            first = await load_or_build_fact_inventory(
                course_id=self.course_id,
                raw_chunks=self.chunks,
                storage=self.storage,
            )
            second = await load_or_build_fact_inventory(
                course_id=other_id,
                raw_chunks=other_chunks,
                storage=self.storage,
            )

        self.assertEqual(first["facts"][0]["factId"], "fact-A")
        self.assertEqual(second["facts"][0]["factId"], "fact-B")
        self.assertNotEqual(
            self.storage.fact_inventory_path(self.course_id),
            self.storage.fact_inventory_path(other_id),
        )


class BreadthFirstAllocationTests(unittest.TestCase):
    def test_small_target_prefers_distinct_facts(self) -> None:
        kinds = [
            "late_work",
            "attendance",
            "grading",
            "contact",
            "communication",
            "accommodation",
        ]
        facts = [
            _fact(
                f"fact-{index:02d}",
                kind=kinds[index - 1],
                complexity=3 if index == 1 else 1,
                usefulness=0.9 - (index * 0.01),
                chunk=f"chunk-{index:03d}",
            )
            for index in range(1, 7)
        ]
        result = allocate_slots(facts, target_count=5)
        with_slots = [
            item for item in result["allocations"] if item["slotCount"] > 0
        ]
        self.assertEqual(sum(item["slotCount"] for item in with_slots), 5)
        # Breadth first: five distinct facts, not multi-slot on the complex one.
        self.assertEqual(len(with_slots), 5)
        self.assertTrue(all(item["slotCount"] == 1 for item in with_slots))

    def test_larger_target_allows_multi_slot_after_breadth(self) -> None:
        kinds = [
            "late_work",
            "attendance",
            "grading",
            "contact",
            "communication",
            "requirement",
            "policy",
            "office_hours",
            "accommodation",
            "exam",
        ]
        facts = [
            _fact(
                f"fact-{index:02d}",
                kind=kinds[(index - 1) % len(kinds)],
                complexity=3 if index == 1 else 1,
                usefulness=0.9 - (index * 0.005),
                chunk=f"chunk-{index:03d}",
            )
            for index in range(1, 12)
        ]
        # More slots than distinct facts → breadth fills first, then depth.
        result = allocate_slots(facts, target_count=20)
        by_id = {item["factId"]: item for item in result["allocations"]}
        self.assertGreaterEqual(by_id["fact-01"]["slotCount"], 2)
        self.assertLessEqual(result["summary"]["allocatedSlots"], 20)
        distinct = sum(1 for item in result["allocations"] if item["slotCount"] > 0)
        self.assertGreaterEqual(distinct, 8)


class BackfillGenerationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        root = Path(self._temp.name)
        self.storage = LocalCourseArtifactStorage(
            root_dir=root / "course_data",
            index_dir=root / "indexes",
        )
        self.course_id = "css-360-backfill-demo"
        chunks = [
            _chunk(
                f"chunk-{index:03d}",
                f"Section {index} syllabus policy details. " * 10,
                order=index,
            )
            for index in range(1, 8)
        ]
        self.storage.save_index(
            self.course_id,
            {"courseId": self.course_id, "chunkCount": len(chunks), "chunks": chunks},
        )
        self.facts = [
            _fact(
                f"fact-{index:02d}",
                kind=[
                    "late_work",
                    "attendance",
                    "grading",
                    "contact",
                    "communication",
                    "requirement",
                    "policy",
                ][index - 1],
                chunk=f"chunk-{index:03d}",
            )
            for index in range(1, 8)
        ]

    def tearDown(self) -> None:
        self._temp.cleanup()

    async def test_rejected_candidate_backfilled_from_next_fact(self) -> None:
        from app.seed_generation import (
            SEED_GENERATION_MODEL,
            VALIDATION_PROMPT_MARKER,
            generate_starter_seeds_for_course,
        )

        inventory = {
            "model": SEED_GENERATION_MODEL,
            "facts": self.facts,
            "factCount": len(self.facts),
            "droppedCount": 0,
            "duplicatesRemoved": 0,
            "fallbackUsed": False,
            "countsByScope": {},
            "countsByKind": {},
            "countsBySeries": {},
            "cached": True,
        }

        call_state = {"n": 0}

        async def _fake(prompt: str, **kwargs):
            if VALIDATION_PROMPT_MARKER in prompt:
                # Accept all validated candidates.
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
            call_state["n"] += 1
            n = call_state["n"]
            if n == 1:
                # Invalid short answer → programmatic rejection → backfill.
                payload = {
                    "seeds": [
                        {
                            "question": "Bad?",
                            "answer": "No",
                            "category": "late work",
                            "questionType": "direct",
                        }
                    ]
                }
            else:
                payload = {
                    "seeds": [
                        {
                            "question": f"Useful student question {n}?",
                            "answer": f"Grounded answer number {n} with detail.",
                            "category": "policy",
                            "questionType": "direct",
                        }
                    ]
                }
            return {"answer": json.dumps(payload), "model": SEED_GENERATION_MODEL}

        async def _embed(texts, *, model=None):
            dim = 16
            embeddings = []
            for text in texts:
                vector = [0.0] * dim
                bucket = sum(ord(ch) for ch in str(text)) % dim
                vector[bucket] = 1.0
                embeddings.append(vector)
            return {"embeddings": embeddings, "model": model or "test"}

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
                new=_embed,
            ),
        ):
            result = await generate_starter_seeds_for_course(
                course_id=self.course_id,
                target_count=3,
                storage=self.storage,
            )

        self.assertEqual(result["progress"]["finalCount"], 3)
        self.assertLessEqual(result["progress"]["finalCount"], 3)
        self.assertGreaterEqual(result["progress"]["candidatesRejectedInvalid"], 1)
        self.assertTrue(result["progress"]["factInventoryCached"])
        self.assertEqual(result["progress"]["factExtractionCalls"], 0)
        fact_ids = {seed["factId"] for seed in result["seeds"]}
        self.assertGreaterEqual(len(fact_ids), 2)

    async def test_cached_inventory_reports_zero_extraction_calls(self) -> None:
        from app.seed_generation import (
            SEED_GENERATION_MODEL,
            VALIDATION_PROMPT_MARKER,
            generate_starter_seeds_for_course,
        )

        inventory = {
            "model": SEED_GENERATION_MODEL,
            "facts": self.facts[:5],
            "factCount": 5,
            "droppedCount": 0,
            "duplicatesRemoved": 0,
            "fallbackUsed": False,
            "countsByScope": {},
            "countsByKind": {},
            "countsBySeries": {},
            "cached": True,
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
                                "question": "A solid student question here?",
                                "answer": "A grounded syllabus answer with detail.",
                                "category": "policy",
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
                    return_value={"embeddings": [[1.0, 0.0], [0.0, 1.0]], "model": "t"}
                ),
            ),
        ):
            result = await generate_starter_seeds_for_course(
                course_id=self.course_id,
                target_count=2,
                storage=self.storage,
            )

        self.assertTrue(result["progress"]["factInventoryCached"])
        self.assertEqual(result["progress"]["factExtractionCalls"], 0)
        self.assertLessEqual(result["progress"]["finalCount"], 2)


if __name__ == "__main__":
    unittest.main()
