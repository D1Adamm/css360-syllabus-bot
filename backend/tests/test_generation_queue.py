"""The shared generation queue: bounded concurrency, bounded waiting, honest refusals."""

from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import generation_queue as gq
from app.generation_queue import (
    BUSY_CODE,
    ClientDisconnected,
    GenerationQueue,
    clean_timings,
    get_generation_queue,
    ollama_timings,
    reset_generation_queue_for_tests,
)


def _queue(**overrides) -> GenerationQueue:
    settings = {"max_concurrency": 1, "max_waiting": 10, "queue_timeout": 5.0}
    settings.update(overrides)
    return GenerationQueue(**settings)


class _Request:
    """Just enough of a Starlette request for `bind_request`."""

    def __init__(self, comparison_id: str | None = None, gone: bool = False) -> None:
        self.headers = {gq.COMPARISON_HEADER: comparison_id} if comparison_id else {}
        self.gone = gone

    async def is_disconnected(self) -> bool:
        return self.gone


async def _hold(queue: GenerationQueue, label: str, order: list[str], release: asyncio.Event, *, comparison: str | None = None, priority: str = "interactive") -> None:
    gq.bind_request(_Request(comparison))
    async with queue.slot(label, priority=priority):
        order.append(label)
        await release.wait()


class ConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_generation_at_a_time_by_default(self) -> None:
        queue = _queue()
        order: list[str] = []
        release = asyncio.Event()
        first = asyncio.create_task(_hold(queue, "a", order, release))
        await asyncio.sleep(0)
        second = asyncio.create_task(_hold(queue, "b", order, release))
        await asyncio.sleep(0.01)
        self.assertEqual(order, ["a"])
        self.assertEqual(queue.snapshot()["active"], 1)
        self.assertEqual(queue.snapshot()["waiting"], 1)
        release.set()
        await asyncio.gather(first, second)
        self.assertEqual(order, ["a", "b"])
        self.assertEqual(queue.snapshot()["active"], 0)

    async def test_configured_concurrency_runs_that_many(self) -> None:
        queue = _queue(max_concurrency=2)
        order: list[str] = []
        release = asyncio.Event()
        tasks = [asyncio.create_task(_hold(queue, name, order, release)) for name in "abc"]
        await asyncio.sleep(0.01)
        self.assertEqual(sorted(order), ["a", "b"])
        release.set()
        await asyncio.gather(*tasks)
        self.assertEqual(queue.snapshot()["active"], 0)

    async def test_interactive_requests_go_before_waiting_background_work(self) -> None:
        queue = _queue()
        order: list[str] = []
        release = asyncio.Event()
        holder = asyncio.create_task(_hold(queue, "holder", order, release))
        await asyncio.sleep(0)
        starter = asyncio.create_task(_hold(queue, "starter", order, release, priority="background"))
        await asyncio.sleep(0)
        student = asyncio.create_task(_hold(queue, "student", order, release))
        await asyncio.sleep(0.01)
        release.set()
        await asyncio.gather(holder, starter, student)
        self.assertEqual(order, ["holder", "student", "starter"])

    async def test_a_comparison_that_began_first_finishes_first(self) -> None:
        """Student A's later request (RAG after Base) goes ahead of student B's."""
        queue = _queue()
        order: list[str] = []
        release = asyncio.Event()
        a_base = asyncio.create_task(_hold(queue, "a-base", order, release, comparison="run-a"))
        await asyncio.sleep(0)
        b_base = asyncio.create_task(_hold(queue, "b-base", order, release, comparison="run-b"))
        await asyncio.sleep(0)
        a_rag = asyncio.create_task(_hold(queue, "a-rag", order, release, comparison="run-a"))
        await asyncio.sleep(0.01)
        release.set()
        await asyncio.gather(a_base, b_base, a_rag)
        self.assertEqual(order, ["a-base", "a-rag", "b-base"])

    async def test_a_comparison_keeps_its_place_even_if_its_first_request_ran_at_once(self) -> None:
        queue = _queue()
        order: list[str] = []
        release_a = asyncio.Event()
        release_rest = asyncio.Event()
        a_base = asyncio.create_task(_hold(queue, "a-base", order, release_a, comparison="run-a"))
        await asyncio.sleep(0)
        b_base = asyncio.create_task(_hold(queue, "b-base", order, release_rest, comparison="run-b"))
        c_base = asyncio.create_task(_hold(queue, "c-base", order, release_rest, comparison="run-c"))
        await asyncio.sleep(0)
        # A's RAG arrives after B and C queued, but A began first.
        a_rag = asyncio.create_task(_hold(queue, "a-rag", order, release_rest, comparison="run-a"))
        await asyncio.sleep(0.01)
        release_a.set()
        await a_base
        release_rest.set()
        await asyncio.gather(b_base, c_base, a_rag)
        self.assertEqual(order, ["a-base", "a-rag", "b-base", "c-base"])


class RefusalTests(unittest.IsolatedAsyncioTestCase):
    async def _assert_busy(self, call, reason: str) -> HTTPException:
        with self.assertRaises(HTTPException) as caught:
            await call
        exc = caught.exception
        self.assertEqual(exc.status_code, 503)
        self.assertEqual(exc.detail["code"], BUSY_CODE)
        self.assertEqual(exc.detail["reason"], reason)
        self.assertIn("try again", exc.detail["message"])
        self.assertEqual(exc.headers["Retry-After"], str(exc.detail["retryAfterSeconds"]))
        return exc

    async def test_full_queue_refuses_at_once(self) -> None:
        queue = _queue(max_waiting=1)
        release = asyncio.Event()
        order: list[str] = []
        holder = asyncio.create_task(_hold(queue, "holder", order, release))
        await asyncio.sleep(0)
        waiter = asyncio.create_task(_hold(queue, "waiter", order, release))
        await asyncio.sleep(0)
        await self._assert_busy(_hold(queue, "third", order, release), "queue_full")
        self.assertEqual(queue.counters["rejectedFull"], 1)
        release.set()
        await asyncio.gather(holder, waiter)
        self.assertNotIn("third", order)

    async def test_wait_longer_than_the_limit_is_refused_without_running(self) -> None:
        queue = _queue(queue_timeout=0.05)
        release = asyncio.Event()
        order: list[str] = []
        holder = asyncio.create_task(_hold(queue, "holder", order, release))
        await asyncio.sleep(0)
        await self._assert_busy(_hold(queue, "late", order, release), "queue_timeout")
        self.assertEqual(queue.snapshot()["waiting"], 0)
        release.set()
        await holder
        self.assertEqual(order, ["holder"])
        self.assertEqual(queue.snapshot()["active"], 0)

    async def test_expected_wait_beyond_the_limit_is_refused_up_front(self) -> None:
        queue = _queue(queue_timeout=10.0)
        queue._record_service(20.0)  # one generation takes 20 s; one is running
        release = asyncio.Event()
        order: list[str] = []
        holder = asyncio.create_task(_hold(queue, "holder", order, release))
        await asyncio.sleep(0)
        await self._assert_busy(_hold(queue, "next", order, release), "estimated_wait")
        self.assertEqual(queue.counters["rejectedEstimate"], 1)
        release.set()
        await holder

    async def test_a_borderline_estimate_waits_instead_of_being_refused(self) -> None:
        queue = _queue(queue_timeout=10.0)
        queue._record_service(12.0)  # just over the limit: the estimate may be stale
        release = asyncio.Event()
        order: list[str] = []
        holder = asyncio.create_task(_hold(queue, "holder", order, release))
        await asyncio.sleep(0)
        waiter = asyncio.create_task(_hold(queue, "next", order, release))
        await asyncio.sleep(0.01)
        self.assertEqual(queue.counters["rejectedEstimate"], 0)
        release.set()
        await asyncio.gather(holder, waiter)
        self.assertEqual(order, ["holder", "next"])

    async def test_background_work_is_never_refused(self) -> None:
        queue = _queue(max_waiting=1, queue_timeout=0.01)
        release = asyncio.Event()
        order: list[str] = []
        holder = asyncio.create_task(_hold(queue, "holder", order, release))
        await asyncio.sleep(0)
        starter = asyncio.create_task(_hold(queue, "starter", order, release, priority="background"))
        await asyncio.sleep(0.05)
        self.assertFalse(starter.done())
        release.set()
        await asyncio.gather(holder, starter)
        self.assertEqual(order, ["holder", "starter"])


class AbandonmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_waiter_whose_client_left_does_not_run(self) -> None:
        queue = _queue()
        release = asyncio.Event()
        order: list[str] = []
        holder = asyncio.create_task(_hold(queue, "holder", order, release))
        await asyncio.sleep(0)

        async def gone() -> None:
            request = _Request(gone=True)
            gq.bind_request(request)
            async with queue.slot("gone"):
                order.append("gone")

        with patch.object(gq, "DISCONNECT_POLL_SECONDS", 0.01):
            with self.assertRaises(ClientDisconnected):
                await gone()
        self.assertEqual(queue.counters["clientGone"], 1)
        release.set()
        await holder
        self.assertEqual(order, ["holder"])
        self.assertEqual(queue.snapshot()["active"], 0)

    async def test_a_cancelled_waiter_gives_back_its_place(self) -> None:
        queue = _queue()
        release = asyncio.Event()
        order: list[str] = []
        holder = asyncio.create_task(_hold(queue, "holder", order, release))
        await asyncio.sleep(0)
        doomed = asyncio.create_task(_hold(queue, "doomed", order, release))
        await asyncio.sleep(0)
        doomed.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await doomed
        after = asyncio.create_task(_hold(queue, "after", order, release))
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(holder, after)
        self.assertEqual(order, ["holder", "after"])
        self.assertEqual(queue.snapshot()["active"], 0)

    async def test_an_exception_inside_the_slot_releases_it(self) -> None:
        queue = _queue()
        with self.assertRaises(RuntimeError):
            async with queue.slot("boom"):
                raise RuntimeError("generation failed")
        self.assertEqual(queue.snapshot()["active"], 0)
        async with queue.slot("next"):
            self.assertEqual(queue.snapshot()["active"], 1)


class TimingTests(unittest.TestCase):
    def test_ollama_durations_become_milliseconds(self) -> None:
        timings = ollama_timings({
            "load_duration": 3_500_000_000,
            "prompt_eval_duration": 11_000_000_000,
            "eval_duration": 800_000_000,
            "total_duration": 15_400_000_000,
            "prompt_eval_count": 2085,
            "eval_count": 27,
        })
        self.assertEqual(timings, {
            "load_ms": 3500, "prompt_eval_ms": 11000, "eval_ms": 800,
            "ollama_total_ms": 15400, "prompt_tokens": 2085, "output_tokens": 27,
        })

    def test_reported_timings_are_restricted_to_known_numbers(self) -> None:
        self.assertEqual(
            clean_timings({"load_ms": 12, "prompt_tokens": 40, "hostname": "node", "eval_ms": True, "eval_ms_x": 1}),
            {"load_ms": 12, "prompt_tokens": 40},
        )
        self.assertEqual(clean_timings("nope"), {})


class RouteTests(unittest.TestCase):
    """The browser sees a structured 503 with Retry-After, not a hang."""

    def setUp(self) -> None:
        reset_generation_queue_for_tests()

    def tearDown(self) -> None:
        reset_generation_queue_for_tests()

    def test_busy_reaches_the_client_as_a_structured_503(self) -> None:
        from app.auth.dependencies import current_principal
        from app.auth.principal import Principal, StaffUser
        from app.main import app

        admin = Principal(user=StaffUser(user_id="u", email="a@uw.edu", display_name="A", role="admin"))
        app.dependency_overrides[current_principal] = lambda: admin
        self.addCleanup(app.dependency_overrides.pop, current_principal, None)

        with patch(
            "app.main.generate_base_model_response",
            AsyncMock(side_effect=gq.busy_exception("queue_full", 42)),
        ):
            response = TestClient(app).post(
                "/api/base-model/generate",
                json={"courseId": "css-360-winter-2026-a7rp", "question": "When is the exam?"},
                headers={"X-Requested-With": "SyllabusModelLab"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["retry-after"], "42")
        self.assertEqual(response.json()["detail"]["code"], BUSY_CODE)

    def test_generate_routes_bind_the_comparison_id(self) -> None:
        from app.auth.dependencies import current_principal
        from app.auth.principal import Principal, StaffUser
        from app.main import app

        admin = Principal(user=StaffUser(user_id="u", email="a@uw.edu", display_name="A", role="admin"))
        app.dependency_overrides[current_principal] = lambda: admin
        self.addCleanup(app.dependency_overrides.pop, current_principal, None)
        seen: list[str | None] = []

        async def fake_base(question: str) -> dict[str, str]:
            seen.append(gq._comparison_id.get())
            return {"answer": "a", "model": "m", "response_type": "base"}

        with patch("app.main.generate_base_model_response", side_effect=fake_base):
            TestClient(app).post(
                "/api/base-model/generate",
                json={"courseId": "css-360-winter-2026-a7rp", "question": "When is the exam?"},
                headers={"X-Requested-With": "SyllabusModelLab", "X-Comparison-Id": "run-123-abc"},
            )
        self.assertEqual(seen, ["run-123-abc"])


class FineTunedClientQueueTests(unittest.IsolatedAsyncioTestCase):
    """The fine-tuned conditions take the same turns as Base and RAG."""

    def setUp(self) -> None:
        reset_generation_queue_for_tests(_queue())

    def tearDown(self) -> None:
        reset_generation_queue_for_tests()

    async def test_fine_tuned_call_waits_for_a_running_base_generation(self) -> None:
        from app.finetuned_client import generate_finetuned_response

        order: list[str] = []
        release = asyncio.Event()
        holder = asyncio.create_task(_hold(get_generation_queue(), "base", order, release))
        await asyncio.sleep(0)

        async def post(url, json):
            order.append("fineTuned")
            response = type("R", (), {"status_code": 200, "text": ""})()
            response.json = lambda: {
                "answer": "ok", "model": "css360e-v1:latest", "adapterLoaded": True,
                "courseId": "c-1", "modelVersion": "v1", "generationSeconds": 1.0,
                "timings": {"load_ms": 0, "prompt_tokens": 10},
            }
            return response

        client = AsyncMock()
        client.post = AsyncMock(side_effect=post)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        with patch.dict("os.environ", {"FINETUNED_SERVICE_URL": "http://127.0.0.1:9001"}), \
                patch("app.finetuned_client.httpx.AsyncClient", return_value=client):
            task = asyncio.create_task(
                generate_finetuned_response("q", course_id="c-1", model_version="v1")
            )
            await asyncio.sleep(0.01)
            self.assertEqual(order, ["base"])
            release.set()
            result = await task
        await holder
        self.assertEqual(order, ["base", "fineTuned"])
        self.assertEqual(result["answer"], "ok")


class BaseChatKeepAliveTests(unittest.IsolatedAsyncioTestCase):
    """Base and RAG keep their model resident between students."""

    def setUp(self) -> None:
        reset_generation_queue_for_tests(_queue())

    def tearDown(self) -> None:
        reset_generation_queue_for_tests()

    async def _sent_payload(self, env: dict[str, str]) -> dict:
        from app.ollama import generate_ollama_chat

        sent: list[dict] = []

        async def post(url, json):
            sent.append(json)
            response = type("R", (), {"status_code": 200, "text": ""})()
            response.json = lambda: {"message": {"content": "ok"}, "load_duration": 0}
            return response

        client = AsyncMock()
        client.post = AsyncMock(side_effect=post)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        with patch.dict("os.environ", env), patch("app.ollama.httpx.AsyncClient", return_value=client):
            with self.assertLogs("app.generation_queue", level="INFO") as logs:
                await generate_ollama_chat("prompt", options={"num_ctx": 4096}, condition="rag")
        self.assertIn("condition=rag", logs.output[0])
        self.assertIn("outcome=ok", logs.output[0])
        return sent[0]

    async def test_keep_alive_defaults_to_thirty_minutes(self) -> None:
        with patch.dict("os.environ"):
            os.environ.pop("BASE_MODEL_KEEP_ALIVE", None)
            payload = await self._sent_payload({})
        self.assertEqual(payload["keep_alive"], "30m")
        self.assertEqual(payload["options"], {"num_ctx": 4096})

    async def test_empty_keep_alive_sends_none(self) -> None:
        payload = await self._sent_payload({"BASE_MODEL_KEEP_ALIVE": ""})
        self.assertNotIn("keep_alive", payload)


if __name__ == "__main__":
    unittest.main()
