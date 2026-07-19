"""Tests for starter Ollama timeout resilience."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from app.ollama import (
    generate_starter_ollama_completion,
    is_ollama_timeout_error,
)
from app.starter_jobs import (
    clear_active_starter_jobs_for_tests,
    run_auto_starter_seed_generation,
)
from app.storage import LocalCourseArtifactStorage
from app.syllabus_plan import plan_syllabus_topics


def _timeout_error() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail="Ollama request timed out. Ensure Ollama is running and responsive.",
    )


def _chunk(chunk_id: str, section_title: str, text: str) -> dict[str, str]:
    return {
        "chunkId": chunk_id,
        "sectionTitle": section_title,
        "text": text,
    }


class StarterTimeoutHelperTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_then_successful_retry(self) -> None:
        with (
            patch(
                "app.ollama.get_starter_ollama_retry_delay_seconds",
                return_value=0.0,
            ),
            patch(
                "app.ollama.get_starter_ollama_timeout_seconds",
                return_value=300.0,
            ),
            patch(
                "app.ollama.generate_ollama_completion",
                new=AsyncMock(
                    side_effect=[
                        _timeout_error(),
                        {"answer": '{"ok": true}', "model": "qwen3:4b"},
                    ]
                ),
            ) as mock_generate,
        ):
            result = await generate_starter_ollama_completion(
                "prompt",
                model="qwen3:4b",
                response_format="json",
                think=False,
            )

        self.assertEqual(result["answer"], '{"ok": true}')
        self.assertEqual(mock_generate.await_count, 2)
        self.assertEqual(mock_generate.await_args.kwargs["timeout"], 300.0)

    async def test_non_timeout_errors_are_not_retried(self) -> None:
        with (
            patch(
                "app.ollama.get_starter_ollama_retry_delay_seconds",
                return_value=0.0,
            ),
            patch(
                "app.ollama.generate_ollama_completion",
                new=AsyncMock(
                    side_effect=HTTPException(
                        status_code=502,
                        detail="Ollama returned malformed JSON for seed generation.",
                    )
                ),
            ) as mock_generate,
        ):
            with self.assertRaises(HTTPException) as ctx:
                await generate_starter_ollama_completion("prompt", model="qwen3:4b")

        self.assertEqual(ctx.exception.status_code, 502)
        self.assertEqual(mock_generate.await_count, 1)
        self.assertFalse(is_ollama_timeout_error(ctx.exception))


class PlannerTimeoutFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_planner_timeout_uses_section_title_fallback(self) -> None:
        raw_chunks = [
            _chunk("chunk-001", "Late Policy", "Late work may be submitted within 24 hours." * 4),
            _chunk("chunk-002", "Office Hours", "Office hours are Tuesdays at 2pm." * 4),
            _chunk("chunk-003", "Projects", "Team projects require weekly status updates." * 4),
            _chunk("chunk-004", "Grading", "Grades are based on assignments and exams." * 4),
        ]

        async def _always_timeout(prompt: str, **kwargs: object) -> dict[str, str]:
            raise _timeout_error()

        plan = await plan_syllabus_topics(
            raw_chunks=raw_chunks,
            target_count=6,
            batch_size=15,
            completion_fn=_always_timeout,
        )

        names = sorted(topic["name"] for topic in plan["topics"])
        self.assertEqual(
            names,
            ["Grading", "Late Policy", "Office Hours", "Projects"],
        )

    async def test_planner_timeout_splits_batch_then_recovers(self) -> None:
        raw_chunks = [
            _chunk(f"chunk-{index:03d}", f"Section {index}", f"Policy text {index}. " * 20)
            for index in range(1, 5)
        ]
        calls = {"n": 0}

        async def _timeout_then_split_success(prompt: str, **kwargs: object) -> dict[str, str]:
            calls["n"] += 1
            # First full-batch attempt fails (after outer retry would already have run
            # if using generate_starter_ollama_completion). Custom fn simulates that.
            if calls["n"] == 1:
                raise _timeout_error()
            return {
                "answer": json.dumps(
                    {
                        "topics": [
                            {
                                "name": f"Recovered topic {calls['n']}",
                                "importance": "medium",
                                "sourceChunkIds": ["chunk-001"],
                                "suggestedExampleCount": 2,
                                "summary": "Recovered after split",
                                "scheduleHeavy": False,
                            }
                        ]
                    }
                ),
                "model": "qwen3:4b",
            }

        plan = await plan_syllabus_topics(
            raw_chunks=raw_chunks,
            target_count=4,
            batch_size=15,
            completion_fn=_timeout_then_split_success,
        )

        self.assertGreaterEqual(len(plan["topics"]), 1)
        self.assertGreaterEqual(calls["n"], 3)


class GenerationTimeoutTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        root = Path(self._temp_dir.name)
        self.storage = LocalCourseArtifactStorage(
            root_dir=root / "course_data",
            index_dir=root / "indexes",
        )
        self.course_id = "css-360-timeout-course"
        chunks = [
            {
                "chunkId": f"chunk-{index:03d}",
                "sectionTitle": f"Section {index}",
                "text": f"Section {index} syllabus details " * 20,
                "order": index,
                "embedding": [1.0, 0.0, 0.0],
            }
            for index in range(1, 5)
        ]
        self.storage.save_index(
            self.course_id,
            {
                "courseId": self.course_id,
                "embeddingModel": "nomic-embed-text",
                "chunkCount": len(chunks),
                "chunks": chunks,
            },
        )

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    async def test_repeated_generation_timeout_raises_concise_failure(self) -> None:
        from app.seed_generation import generate_starter_seeds_for_course

        facts = [
            {
                "factId": "fact-01",
                "statement": "Late work may be submitted within 24 hours for half credit.",
                "importance": "high",
                "importanceScore": 0.9,
                "studentAskLikelihood": 0.9,
                "complexity": 1,
                "usefulnessScore": 0.86,
                "sourceChunkIds": ["chunk-001"],
                "evidenceQuote": "Late work may be submitted within 24 hours for half credit.",
                "kind": "late_work",
                "scope": "course_wide",
                "seriesKey": None,
                "assignmentGroup": None,
                "seriesOrdinal": None,
            }
        ]

        with (
            patch(
                "app.seed_generation.load_or_build_fact_inventory",
                new=AsyncMock(
                    return_value={
                        "model": "qwen3:4b",
                        "facts": facts,
                        "factCount": 1,
                        "droppedCount": 0,
                        "duplicatesRemoved": 0,
                        "fallbackUsed": False,
                        "countsByScope": {},
                        "countsByKind": {},
                        "countsBySeries": {},
                    }
                ),
            ),
            patch(
                "app.seed_generation.generate_starter_ollama_completion",
                new=AsyncMock(side_effect=_timeout_error()),
            ),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await generate_starter_seeds_for_course(
                    course_id=self.course_id,
                    target_count=3,
                    storage=self.storage,
                )

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("timed out after retries", str(ctx.exception.detail))
        self.assertIn("RAG index remain available", str(ctx.exception.detail))


class StarterJobTimeoutFailureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        clear_active_starter_jobs_for_tests()
        self._auto_env = patch.dict(
            os.environ,
            {"AUTO_STARTER_SEED_GENERATION": "true"},
            clear=False,
        )
        self._auto_env.start()

    def tearDown(self) -> None:
        self._auto_env.stop()
        clear_active_starter_jobs_for_tests()

    async def test_job_status_becomes_failed_instead_of_staying_generating(self) -> None:
        from app.starter_jobs import _active_starter_jobs

        _active_starter_jobs.add("css-360-timeout-job")
        patches: list[dict] = []

        async def _capture_patch(course_id: str, updates: dict) -> bool:
            patches.append(updates)
            return True

        with (
            patch(
                "app.starter_jobs.get_starter_auto_generate_count",
                return_value=3,
            ),
            patch(
                "app.starter_jobs.best_effort_patch_starter_seed_generation",
                new=AsyncMock(side_effect=_capture_patch),
            ),
            patch(
                "app.starter_jobs.generate_starter_seeds_for_course",
                new=AsyncMock(
                    side_effect=HTTPException(
                        status_code=503,
                        detail=(
                            "Starter seed generation timed out after retries. "
                            "The syllabus and RAG index remain available."
                        ),
                    )
                ),
            ),
        ):
            await run_auto_starter_seed_generation("css-360-timeout-job")

        self.assertEqual(patches[0]["status"], "generating")
        self.assertEqual(patches[-1]["status"], "failed")
        self.assertIn("timed out after retries", patches[-1]["error"])
        self.assertNotIn("css-360-timeout-job", _active_starter_jobs)
