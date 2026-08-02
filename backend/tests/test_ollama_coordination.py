"""Tests for Ollama generation locking and global starter-job protection."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.ollama import generate_base_model_response, generate_ollama_completion
from app.ollama_coordination import (
    end_starter_job,
    get_ollama_generation_lock,
    get_starter_job_status,
    ollama_generation_slot,
    starter_job_slot,
    try_begin_starter_job,
)
from app.starter_jobs import (
    clear_active_starter_jobs_for_tests,
    try_queue_starter_seed_generation,
)


class CoordinationResetMixin:
    def setUp(self) -> None:
        clear_active_starter_jobs_for_tests()

    def tearDown(self) -> None:
        clear_active_starter_jobs_for_tests()


class StarterJobSlotTests(CoordinationResetMixin, unittest.IsolatedAsyncioTestCase):
    async def test_two_manual_jobs_cannot_run_concurrently(self) -> None:
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        second_blocked = asyncio.Event()

        async def _slow_generate(**kwargs):
            first_started.set()
            await release_first.wait()
            return {
                "courseId": kwargs["course_id"],
                "model": "qwen3:4b",
                "targetCount": kwargs["target_count"],
                "seeds": [],
                "progress": {
                    "eligibleChunks": 0,
                    "selectedChunks": 0,
                    "chunksProcessed": 0,
                    "chunksSkipped": 0,
                    "planningCalls": 0,
                    "mergeCalls": 0,
                    "factExtractionCalls": 0,
                    "factInventoryCached": True,
                    "factCount": 0,
                    "allocatedFactCount": 0,
                    "allocatedSlots": 0,
                    "backfillAttempts": 0,
                    "backfillAccepted": 0,
                    "generationCalls": 0,
                    "generationBatchCalls": 0,
                    "maxGenerationBatchSize": 0,
                    "validationCalls": 0,
                    "validationBatchCalls": 0,
                    "maxValidationBatchSize": 0,
                    "ollamaCalls": 0,
                    "embeddingCalls": 0,
                    "candidatesGenerated": 0,
                    "candidatesValidated": 0,
                    "candidatesAccepted": 0,
                    "candidatesRejected": 0,
                    "duplicatesRemoved": 0,
                    "semanticDuplicatesRemoved": 0,
                    "candidatesRejectedInvalid": 0,
                    "candidatesRejectedValidation": 0,
                    "candidatesRejectedUnsupportedClaims": 0,
                    "candidatesRejectedBalancing": 0,
                    "candidatesRejectedPreValidation": 0,
                    "candidatesRejectedQualifierMismatch": 0,
                    "candidatesRejectedModalEscalation": 0,
                    "scheduleCount": 0,
                    "scenarioOrClarificationMinimum": 1,
                    "timeoutFailures": 0,
                    "finalCount": 0,
                    "savedCount": 0,
                    "elapsedMs": 1,
                    "status": "failed",
                    "topUp": False,
                    "existingCount": 0,
                    "missingCount": 0,
                    "totalCount": 0,
                },
            }

        client = TestClient(app)

        async def _run_first() -> None:
            async with starter_job_slot("css-360-first-aaaa", "manual"):
                first_started.set()
                await release_first.wait()

        async def _run_second() -> None:
            await first_started.wait()
            try:
                async with starter_job_slot("css-360-second-bbbb", "manual"):
                    pass
            except HTTPException as exc:
                second_blocked.set()
                self.assertEqual(exc.status_code, 409)
                self.assertEqual(exc.detail["code"], "generation_in_progress")
                self.assertEqual(exc.detail["courseId"], "css-360-first-aaaa")
                raise

        first_task = asyncio.create_task(_run_first())
        await first_started.wait()
        status = get_starter_job_status()
        self.assertTrue(status["active"])
        self.assertEqual(status["courseId"], "css-360-first-aaaa")
        self.assertEqual(status["operation"], "manual")
        self.assertIsNotNone(status["startedAt"])

        with self.assertRaises(HTTPException):
            await _run_second()
        self.assertTrue(second_blocked.is_set())

        # HTTP endpoint path for a second manual request while first owns the slot.
        with patch(
            "app.main.generate_starter_seeds_for_course",
            new=AsyncMock(side_effect=_slow_generate),
        ):
            # Slot still held by first_task.
            response = client.post(
                "/api/courses/css-360-second-bbbb/seeds/generate-starter",
                json={"targetCount": 3, "save": False},
            )
        self.assertEqual(response.status_code, 409)
        body = response.json()["detail"]
        self.assertEqual(body["code"], "generation_in_progress")
        self.assertEqual(body["courseId"], "css-360-first-aaaa")
        self.assertIn("Wait", body["message"])

        release_first.set()
        await first_task
        self.assertFalse(get_starter_job_status()["active"])

    async def test_lock_releases_after_success(self) -> None:
        async with starter_job_slot("css-360-success-aaaa", "manual"):
            self.assertTrue(get_starter_job_status()["active"])
        self.assertFalse(get_starter_job_status()["active"])

        # A second job can start after release.
        began = await try_begin_starter_job("css-360-success-bbbb", "manual")
        self.assertTrue(began)
        await end_starter_job(course_id="css-360-success-bbbb")

    async def test_lock_releases_after_exception(self) -> None:
        with self.assertRaises(RuntimeError):
            async with starter_job_slot("css-360-error-aaaa", "top_up"):
                self.assertTrue(get_starter_job_status()["active"])
                raise RuntimeError("boom")

        self.assertFalse(get_starter_job_status()["active"])
        began = await try_begin_starter_job("css-360-error-bbbb", "manual")
        self.assertTrue(began)
        await end_starter_job(course_id="css-360-error-bbbb")

    async def test_auto_and_manual_cannot_run_concurrently(self) -> None:
        with (
            patch(
                "app.starter_jobs.get_starter_auto_generate_count",
                return_value=3,
            ),
            patch(
                "app.starter_jobs._durable_status_is_active",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.starter_jobs.best_effort_patch_starter_seed_generation",
                new=AsyncMock(return_value=True),
            ),
            patch.dict(
                "os.environ",
                {"AUTO_STARTER_SEED_GENERATION": "true"},
                clear=False,
            ),
        ):
            queued = await try_queue_starter_seed_generation("css-360-auto-aaaa")
        self.assertTrue(queued["queued"])
        self.assertTrue(get_starter_job_status()["active"])
        self.assertEqual(get_starter_job_status()["operation"], "automatic")

        client = TestClient(app)
        with patch(
            "app.main.generate_starter_seeds_for_course",
            new=AsyncMock(),
        ) as mock_generate:
            response = client.post(
                "/api/courses/css-360-manual-bbbb/seeds/generate-starter",
                json={"targetCount": 3, "save": False},
            )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["code"], "generation_in_progress")
        self.assertEqual(response.json()["detail"]["courseId"], "css-360-auto-aaaa")
        mock_generate.assert_not_awaited()

    async def test_top_up_cannot_run_during_another_job(self) -> None:
        began = await try_begin_starter_job("css-360-owner-aaaa", "manual")
        self.assertTrue(began)

        client = TestClient(app)
        with patch(
            "app.main.generate_starter_seeds_for_course",
            new=AsyncMock(),
        ) as mock_generate:
            response = client.post(
                "/api/courses/css-360-topup-bbbb/seeds/top-up",
                json={"targetCount": 3, "save": False},
            )
        self.assertEqual(response.status_code, 409)
        detail = response.json()["detail"]
        self.assertEqual(detail["code"], "generation_in_progress")
        self.assertEqual(detail["courseId"], "css-360-owner-aaaa")
        mock_generate.assert_not_awaited()

        await end_starter_job(course_id="css-360-owner-aaaa")


class StarterStatusEndpointTests(
    CoordinationResetMixin,
    unittest.IsolatedAsyncioTestCase,
):
    async def test_status_inactive_and_active(self) -> None:
        client = TestClient(app)
        inactive = client.get("/api/starter-generation/status")
        self.assertEqual(inactive.status_code, 200)
        self.assertEqual(
            inactive.json(),
            {
                "active": False,
                "courseId": None,
                "operation": None,
                "startedAt": None,
            },
        )

        await try_begin_starter_job("css-360-status-aaaa", "top_up")
        active = client.get("/api/starter-generation/status")
        self.assertEqual(active.status_code, 200)
        body = active.json()
        self.assertTrue(body["active"])
        self.assertEqual(body["courseId"], "css-360-status-aaaa")
        self.assertEqual(body["operation"], "top_up")
        self.assertIsInstance(body["startedAt"], str)


class OllamaGenerationLockTests(CoordinationResetMixin, unittest.IsolatedAsyncioTestCase):
    async def test_base_waits_while_starter_owns_ollama_lock(self) -> None:
        order: list[str] = []
        starter_holding = asyncio.Event()
        release_starter = asyncio.Event()

        async def _starter_hold() -> None:
            async with ollama_generation_slot():
                order.append("starter_acquired")
                starter_holding.set()
                await release_starter.wait()
                order.append("starter_released")

        async def _base_after_hold() -> None:
            await starter_holding.wait()
            order.append("base_waiting")
            async with ollama_generation_slot():
                order.append("base_acquired")

        starter_task = asyncio.create_task(_starter_hold())
        base_task = asyncio.create_task(_base_after_hold())
        await starter_holding.wait()
        await asyncio.sleep(0.01)
        self.assertEqual(order, ["starter_acquired", "base_waiting"])
        self.assertTrue(get_ollama_generation_lock().locked())

        release_starter.set()
        await asyncio.gather(starter_task, base_task)
        self.assertEqual(
            order,
            [
                "starter_acquired",
                "base_waiting",
                "starter_released",
                "base_acquired",
            ],
        )

    async def test_rag_waits_while_starter_owns_ollama_lock(self) -> None:
        order: list[str] = []
        starter_holding = asyncio.Event()
        release_starter = asyncio.Event()

        async def _blocked_completion(prompt: str, **kwargs):
            # Simulate generate_ollama_completion acquiring the shared slot.
            async with ollama_generation_slot():
                order.append("starter_call")
                starter_holding.set()
                await release_starter.wait()
                return {"answer": "seed", "model": "qwen3:4b"}

        async def _rag_call() -> None:
            await starter_holding.wait()
            order.append("rag_waiting")
            async with ollama_generation_slot():
                order.append("rag_call")

        starter_task = asyncio.create_task(_blocked_completion("starter"))
        rag_task = asyncio.create_task(_rag_call())
        await starter_holding.wait()
        await asyncio.sleep(0.01)
        self.assertEqual(order, ["starter_call", "rag_waiting"])

        release_starter.set()
        await asyncio.gather(starter_task, rag_task)
        self.assertEqual(order, ["starter_call", "rag_waiting", "rag_call"])

    async def test_generate_ollama_completion_uses_shared_lock(self) -> None:
        holding = asyncio.Event()
        release = asyncio.Event()
        order: list[str] = []

        async def _hold_lock() -> None:
            async with ollama_generation_slot():
                order.append("holder")
                holding.set()
                await release.wait()

        mock_response = AsyncMock()
        # Not used until lock is free; still provide a valid path.
        http_response = type("R", (), {"status_code": 200, "text": ""})()
        http_response.json = lambda: {"response": "ok", "model": "llama3.2:3b"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=http_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        holder = asyncio.create_task(_hold_lock())
        await holding.wait()

        async def _call_generate() -> None:
            order.append("generate_waiting")
            with patch("app.ollama.httpx.AsyncClient", return_value=mock_client):
                await generate_ollama_completion("hello")
            order.append("generate_done")

        gen_task = asyncio.create_task(_call_generate())
        await asyncio.sleep(0.01)
        self.assertEqual(order, ["holder", "generate_waiting"])
        self.assertEqual(mock_client.post.await_count, 0)

        release.set()
        await asyncio.gather(holder, gen_task)
        self.assertEqual(order, ["holder", "generate_waiting", "generate_done"])
        self.assertEqual(mock_client.post.await_count, 1)

    async def test_base_model_still_works_sequentially(self) -> None:
        with patch(
            "app.ollama.generate_ollama_completion",
            new=AsyncMock(
                return_value={
                    "answer": "No syllabus was provided.",
                    "model": "llama3.2:3b",
                }
            ),
        ) as mock_generate:
            result = await generate_base_model_response("When is the exam?")
        self.assertEqual(result["response_type"], "base")
        mock_generate.assert_awaited_once()


class EmbeddingsRemainUnlockedTests(CoordinationResetMixin, unittest.IsolatedAsyncioTestCase):
    async def test_embed_does_not_require_generation_lock(self) -> None:
        from app.ollama import embed_ollama_texts

        holding = asyncio.Event()
        release = asyncio.Event()

        async def _hold() -> None:
            async with ollama_generation_slot():
                holding.set()
                await release.wait()

        mock_response = type("R", (), {"status_code": 200, "text": ""})()
        mock_response.json = lambda: {
            "embeddings": [[0.1, 0.2]],
            "model": "nomic-embed-text",
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        holder = asyncio.create_task(_hold())
        await holding.wait()

        with patch("app.ollama.httpx.AsyncClient", return_value=mock_client):
            # Must complete even while generation lock is held.
            result = await asyncio.wait_for(
                embed_ollama_texts(["hello"]),
                timeout=0.5,
            )

        self.assertEqual(result["embeddings"], [[0.1, 0.2]])
        release.set()
        await holder


if __name__ == "__main__":
    unittest.main()
