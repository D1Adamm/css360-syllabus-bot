"""The classroom answer cache's rules: what is matched, served, stored, shared.

No database and no model: `answer_cache_fake.FakeAnswerCache` stands in for
PostgreSQL and for the identity lookups, and the pipelines are fakes that
count how often they run.
"""

from __future__ import annotations

import asyncio
import os
import unittest
from typing import Any
from unittest.mock import patch

from fastapi import HTTPException

from app import answer_cache, answer_flights
from app.answer_cache import normalize_question, question_key, serve, serve_detailed
from app.generation_mode import GenerationMode
from app.generation_queue import (
    background_generation,
    bind_request,
    generation_slot,
    get_generation_queue,
    log_generation,
    note_generation_detail,
)
from app.schemas import BaseModelGenerateResponse, FineTunedGenerateResponse

from tests.answer_cache_fake import FakeAnswerCache

COURSE = "css-360-winter-2026-a7rp"
OTHER = "css-350-spring-2026-n3h9"
QUESTION = "When is Homework 2 due?"


class FakePipelines:
    """Counts runs; answers through the real generation queue like a real pipeline."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, int]] = []
        self.gate: asyncio.Event | None = None
        self.fail_with: BaseException | None = None
        self.answer = "It is due Friday."
        self.ft_model = "css360-v2:latest"
        self.ft_version = "v2"
        self.ft_engine = "vm"

    async def _generate(self, label: str, course_id: str, question: str, top_k: int) -> None:
        self.calls.append((label, course_id, question, top_k))
        note_generation_detail(promptSha256="p" * 64)
        async with generation_slot(label) as slot:
            if self.gate is not None:
                await self.gate.wait()
            if self.fail_with is not None:
                raise self.fail_with
            log_generation(slot, model="llama3.2:3b", outcome="ok", elapsed_seconds=0.01, timings={"output_tokens": 7}, engine="vm")

    async def base(self, course_id: str, question: str, top_k: int) -> BaseModelGenerateResponse:
        await self._generate("base", course_id, question, top_k)
        return BaseModelGenerateResponse(answer=self.answer, model="llama3.2:3b", responseType="base", courseId=course_id)

    async def fine_tuned(self, course_id: str, question: str, top_k: int) -> FineTunedGenerateResponse:
        await self._generate("fineTuned", course_id, question, top_k)
        return FineTunedGenerateResponse(
            answer=self.answer, model=self.ft_model, responseType="fineTuned", courseId=course_id,
            modelVersion=self.ft_version, adapterLoaded=True, generationSeconds=1.0, engine=self.ft_engine,
        )

    def install(self, testcase: unittest.TestCase) -> "FakePipelines":
        registered = {
            "base": answer_cache.RegisteredPipeline(self.base, BaseModelGenerateResponse),
            "fineTuned": answer_cache.RegisteredPipeline(self.fine_tuned, FineTunedGenerateResponse),
        }
        patcher = patch.dict(answer_cache._pipelines, registered, clear=True)
        patcher.start()
        testcase.addCleanup(patcher.stop)
        return self


class _Request:
    def __init__(self, comparison_id: str | None = None) -> None:
        self.headers = {"x-comparison-id": comparison_id} if comparison_id else {}
        self.gone = False

    async def is_disconnected(self) -> bool:
        return self.gone


class NormalisationTests(unittest.TestCase):
    def test_case_spacing_quotes_and_trailing_punctuation_do_not_matter(self) -> None:
        same = [
            "When is Homework 2 due?",
            "  when is   homework 2 due ",
            "WHEN IS HOMEWORK 2 DUE?!",
            "When is Homework 2 due.",
        ]
        self.assertEqual({normalize_question(q) for q in same}, {"when is homework 2 due"})
        self.assertEqual(normalize_question("What’s the “late” policy?"), "what's the \"late\" policy")

    def test_questions_that_differ_in_substance_never_match(self) -> None:
        pairs = [
            ("When is Homework 2 due?", "When is Homework 3 due?"),
            ("Is C allowed?", "Is C++ allowed?"),
            ("What is the late policy?", "What is the late work policy?"),
            ("When is the exam?", "When is the exam review?"),
            ("Section A time?", "Section B time?"),
        ]
        for left, right in pairs:
            with self.subTest(left=left):
                self.assertNotEqual(question_key(left), question_key(right))


class CacheTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.cache = FakeAnswerCache().install(self)
        self.pipes = FakePipelines().install(self)
        self.cache.select(COURSE, QUESTION)

    async def until(self, condition, timeout: float = 2.0) -> None:
        """Wait for a state instead of sleeping a guessed interval: the cache's
        lookups run in a thread pool, so when a request reaches the queue varies."""
        deadline = asyncio.get_running_loop().time() + timeout
        while not condition():
            if asyncio.get_running_loop().time() > deadline:
                self.fail("condition was not reached in time")
            await asyncio.sleep(0.002)

    @staticmethod
    def waiting_requests() -> int:
        """How many interactive requests are attached to a shared generation."""
        return sum(len(flight.checks) for flight in answer_flights._flights.values())


class DisabledTests(unittest.IsolatedAsyncioTestCase):
    async def test_with_the_flag_off_the_pipeline_runs_and_nothing_else_happens(self) -> None:
        cache = FakeAnswerCache().install(self, enabled=False)
        pipes = FakePipelines().install(self)
        cache.select(COURSE, QUESTION)
        first = await serve("base", COURSE, QUESTION)
        second = await serve("base", COURSE, QUESTION)
        self.assertEqual(len(pipes.calls), 2)
        self.assertFalse(first.cached)
        self.assertIsNone(second.cache_entry_id)
        self.assertEqual((cache.reads, cache.writes, cache.identity_calls), (0, 0, []))

    async def test_the_flag_is_off_unless_explicitly_turned_on(self) -> None:
        for value in ("", "0", "false", "no", "off", "maybe"):
            with self.subTest(value=value), patch.dict(os.environ, {answer_cache.ENABLED_ENV: value}):
                self.assertFalse(answer_cache.cache_enabled())
        with patch.dict(os.environ, {answer_cache.ENABLED_ENV: " TRUE "}):
            self.assertTrue(answer_cache.cache_enabled())


class HitAndMissTests(CacheTestCase):
    async def test_a_selected_question_is_generated_once_then_served_from_the_store(self) -> None:
        first = await serve_detailed("base", COURSE, QUESTION)
        second = await serve_detailed("base", COURSE, "  when is homework 2 due ")
        self.assertEqual((first.outcome, second.outcome), ("stored", "hit"))
        self.assertEqual(len(self.pipes.calls), 1)
        self.assertFalse(first.response.cached)
        self.assertTrue(second.response.cached)
        self.assertEqual(second.response.answer, "It is due Friday.")
        self.assertEqual(second.response.cache_entry_id, first.response.cache_entry_id)
        self.assertIsNotNone(second.response.generated_at)

    async def test_a_hit_never_touches_the_generation_queue(self) -> None:
        await serve("base", COURSE, QUESTION)
        started = get_generation_queue().counters["started"]
        await serve("base", COURSE, QUESTION)
        self.assertEqual(get_generation_queue().counters["started"], started)

    async def test_an_unselected_question_is_generated_every_time_and_never_stored(self) -> None:
        for _ in range(2):
            served = await serve_detailed("base", COURSE, "When is Homework 3 due?")
            self.assertEqual(served.outcome, "generated")
            self.assertIsNone(served.response.cache_entry_id)
        self.assertEqual(len(self.pipes.calls), 2)
        self.assertEqual(self.cache.stored(), [])
        self.assertEqual(self.cache.reads, 0)

    async def test_deselecting_a_question_stops_its_answer_being_served(self) -> None:
        await serve("base", COURSE, QUESTION)
        self.cache.select(COURSE)
        served = await serve_detailed("base", COURSE, QUESTION)
        self.assertEqual(served.outcome, "generated")
        self.assertFalse(served.response.cached)
        self.assertEqual(len(self.pipes.calls), 2)

    async def test_each_approach_is_stored_and_served_on_its_own(self) -> None:
        self.cache.set_identity("fineTuned", COURSE, {"model": {"tag": "css360-v2:latest", "digest": "d", "courseVersion": "v2"}})
        await serve("base", COURSE, QUESTION)
        # Base is stored; Fine-Tuned is not, and generates normally.
        fine_tuned = await serve_detailed("fineTuned", COURSE, QUESTION)
        base = await serve_detailed("base", COURSE, QUESTION)
        self.assertEqual((base.outcome, fine_tuned.outcome), ("hit", "stored"))
        self.assertEqual({row["condition"] for row in self.cache.stored()}, {"base", "fineTuned"})

    async def test_one_courses_answer_is_never_served_to_another(self) -> None:
        self.cache.select(OTHER, QUESTION)
        self.pipes.answer = "Course A answer."
        await serve("base", COURSE, QUESTION)
        self.pipes.answer = "Course B answer."
        other = await serve_detailed("base", OTHER, QUESTION)
        self.assertEqual(other.outcome, "stored")
        self.assertEqual(other.response.answer, "Course B answer.")
        again = await serve("base", COURSE, QUESTION)
        self.assertEqual(again.answer, "Course A answer.")

    async def test_a_changed_identity_is_a_miss_and_the_old_row_is_kept(self) -> None:
        await serve("base", COURSE, QUESTION)
        self.cache.set_identity("base", COURSE, {"model": {"tag": "llama3.2:3b", "digest": "d2"}})
        served = await serve_detailed("base", COURSE, QUESTION)
        self.assertEqual(served.outcome, "stored")
        self.assertEqual(len(self.pipes.calls), 2)
        self.assertEqual(len(self.cache.stored(COURSE)), 2)

    async def test_a_different_top_k_is_a_different_answer(self) -> None:
        await serve("base", COURSE, QUESTION, top_k=4)
        served = await serve_detailed("base", COURSE, QUESTION, top_k=6)
        self.assertEqual(served.outcome, "stored")

    async def test_the_stored_row_records_what_produced_it_and_no_person(self) -> None:
        await serve("base", COURSE, QUESTION)
        (row,) = self.cache.stored()
        self.assertEqual(row["source"], "live")
        self.assertEqual(row["questionText"], QUESTION)
        self.assertEqual(row["identity"]["model"], {"tag": "llama3.2:3b", "digest": "d1"})
        self.assertEqual(row["provenance"]["model"], "llama3.2:3b")
        self.assertEqual(row["provenance"]["engine"], "vm")
        self.assertEqual(row["provenance"]["timings"], {"output_tokens": 7})
        self.assertEqual(row["provenance"]["promptSha256"], "p" * 64)
        for key in answer_cache.CACHE_FIELDS:
            self.assertNotIn(key, row["payload"])
        text = repr(row).lower()
        for word in ("participant", "userid", "session", "email"):
            self.assertNotIn(word, text)


class NeverStoredTests(CacheTestCase):
    async def test_a_failure_is_never_stored_and_reaches_the_caller(self) -> None:
        self.pipes.fail_with = HTTPException(status_code=502, detail="Ollama failed.")
        with self.assertRaises(HTTPException) as raised:
            await serve("base", COURSE, QUESTION)
        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(self.cache.stored(), [])
        # The next request really retries.
        self.pipes.fail_with = None
        self.assertEqual((await serve_detailed("base", COURSE, QUESTION)).outcome, "stored")
        self.assertEqual(len(self.pipes.calls), 2)

    async def test_an_empty_answer_is_never_stored(self) -> None:
        self.pipes.answer = "   "
        served = await serve_detailed("base", COURSE, QUESTION)
        self.assertEqual(served.outcome, "generated")
        self.assertEqual(self.cache.stored(), [])

    async def test_an_identity_that_moved_during_generation_is_not_stored(self) -> None:
        original = self.cache._build_identity

        async def moving(condition: str, course_id: str, *, top_k: int, fresh: bool = False):
            if fresh:
                self.cache.set_identity("base", COURSE, {"model": {"tag": "llama3.2:3b", "digest": "rebuilt"}})
            return await original(condition, course_id, top_k=top_k, fresh=fresh)

        with patch("app.answer_cache.build_identity", new=moving):
            served = await serve_detailed("base", COURSE, QUESTION)
        self.assertEqual(served.outcome, "generated")
        self.assertEqual(served.response.answer, "It is due Friday.")
        self.assertEqual(self.cache.stored(), [])

    async def test_the_identity_is_read_fresh_before_storing(self) -> None:
        await serve("base", COURSE, QUESTION)
        self.assertEqual([fresh for *_rest, fresh in self.cache.identity_calls], [False, True])

    async def test_an_unknown_identity_reads_and_writes_nothing(self) -> None:
        self.cache.set_identity("base", COURSE, "base_model_unavailable")
        for _ in range(2):
            self.assertEqual((await serve_detailed("base", COURSE, QUESTION)).outcome, "generated")
        self.assertEqual((self.cache.reads, self.cache.writes), (0, 0))
        self.assertEqual(len(self.pipes.calls), 2)

    async def test_a_fine_tuned_answer_from_another_version_or_model_is_not_stored(self) -> None:
        self.cache.set_identity("fineTuned", COURSE, {"model": {"tag": "css360-v2:latest", "digest": "d", "courseVersion": "v2"}})
        self.pipes.ft_version = "v3"
        self.assertEqual((await serve_detailed("fineTuned", COURSE, QUESTION)).outcome, "generated")
        self.pipes.ft_version, self.pipes.ft_model = "v2", "css360-v1:latest"
        self.assertEqual((await serve_detailed("fineTuned", COURSE, QUESTION)).outcome, "generated")
        self.assertEqual(self.cache.stored(), [])
        # The tag is compared the way Ollama names models: `:latest` is implied.
        self.pipes.ft_model = "css360-v2"
        self.assertEqual((await serve_detailed("fineTuned", COURSE, QUESTION)).outcome, "stored")

    async def test_gpu_mode_bypasses_the_cache_entirely(self) -> None:
        await serve("base", COURSE, QUESTION)
        gpu = GenerationMode(kind="gpu", gpu_url="http://127.0.0.1:9100")
        reads, writes = self.cache.reads, self.cache.writes
        with patch("app.answer_cache.current_mode", return_value=gpu):
            served = await serve_detailed("base", COURSE, QUESTION)
        self.assertEqual(served.outcome, "generated")
        self.assertFalse(served.response.cached)
        self.assertEqual((self.cache.reads, self.cache.writes), (reads, writes))
        self.assertEqual(len(self.pipes.calls), 2)

    async def test_an_answer_the_gpu_produced_is_not_stored_even_if_the_mode_reads_vm(self) -> None:
        self.cache.set_identity("fineTuned", COURSE, {"model": {"tag": "css360-v2:latest", "digest": "d", "courseVersion": "v2"}})
        self.pipes.ft_engine = "gpu"
        self.assertEqual((await serve_detailed("fineTuned", COURSE, QUESTION)).outcome, "generated")
        self.assertEqual(self.cache.stored(), [])


class FailOpenTests(CacheTestCase):
    async def test_a_failing_read_falls_through_to_live_generation(self) -> None:
        await serve("base", COURSE, QUESTION)
        self.cache.fail_reads = True
        served = await serve_detailed("base", COURSE, QUESTION)
        self.assertEqual(served.response.answer, "It is due Friday.")
        self.assertFalse(served.response.cached)
        self.assertEqual(len(self.pipes.calls), 2)

    async def test_a_failing_write_still_returns_the_answer(self) -> None:
        self.cache.fail_writes = True
        served = await serve_detailed("base", COURSE, QUESTION)
        self.assertEqual(served.outcome, "generated")
        self.assertEqual(served.response.answer, "It is due Friday.")

    async def test_a_failing_selection_read_means_nothing_is_cached(self) -> None:
        self.cache.fail_selection = True
        answer_cache.invalidate_selection()
        self.assertEqual((await serve_detailed("base", COURSE, QUESTION)).outcome, "generated")
        self.assertEqual(self.cache.stored(), [])

    async def test_a_failing_identity_lookup_means_nothing_is_cached(self) -> None:
        async def broken(*_args: Any, **_kwargs: Any):
            raise RuntimeError("ollama unreachable")

        with patch("app.answer_cache.build_identity", new=broken):
            self.assertEqual((await serve_detailed("base", COURSE, QUESTION)).outcome, "generated")
        self.assertEqual(self.cache.stored(), [])

    async def test_a_stored_row_that_no_longer_fits_the_response_is_a_miss(self) -> None:
        await serve("base", COURSE, QUESTION)
        (row,) = self.cache.stored()
        row["payload"] = {"unexpected": True}
        served = await serve_detailed("base", COURSE, QUESTION)
        self.assertFalse(served.response.cached)
        self.assertEqual(len(self.pipes.calls), 2)


class SharingTests(CacheTestCase):
    """Identical concurrent requests share one generation (Phase 2)."""

    async def _concurrently(self, count: int, question: str = "An unselected question?", **kwargs: Any):
        self.pipes.gate = asyncio.Event()
        tasks = [asyncio.create_task(serve("base", kwargs.get("course", COURSE), question)) for _ in range(count)]
        await self.until(lambda: self.waiting_requests() == count and get_generation_queue().active == 1)
        return tasks

    async def test_twenty_identical_requests_cost_one_generation(self) -> None:
        tasks = await self._concurrently(20)
        queue = get_generation_queue()
        # One running, nobody waiting: the other nineteen never entered the queue.
        self.assertEqual((queue.active, queue.snapshot()["waiting"]), (1, 0))
        self.pipes.gate.set()
        answers = await asyncio.gather(*tasks)
        self.assertEqual(len(self.pipes.calls), 1)
        self.assertEqual({a.answer for a in answers}, {"It is due Friday."})
        self.assertEqual(answer_flights.active_flight_count(), 0)
        # Unselected, so shared in memory only.
        self.assertEqual(self.cache.stored(), [])

    async def test_sharing_follows_the_normalised_question(self) -> None:
        self.pipes.gate = asyncio.Event()
        tasks = [
            asyncio.create_task(serve("base", COURSE, "When is the exam?")),
            asyncio.create_task(serve("base", COURSE, "  when is the EXAM ")),
            asyncio.create_task(serve("base", COURSE, "When is the exam review?")),
        ]
        await self.until(lambda: self.waiting_requests() == 3)
        self.pipes.gate.set()
        await asyncio.gather(*tasks)
        self.assertEqual(len(self.pipes.calls), 2)

    async def test_different_courses_approaches_and_configurations_never_share(self) -> None:
        self.cache.set_identity("fineTuned", COURSE, {"model": {"tag": "css360-v2:latest", "digest": "d", "courseVersion": "v2"}})
        self.pipes.gate = asyncio.Event()
        question = "An unselected question?"
        tasks = [
            asyncio.create_task(serve("base", COURSE, question)),
            asyncio.create_task(serve("base", OTHER, question)),
            asyncio.create_task(serve("fineTuned", COURSE, question)),
            asyncio.create_task(serve("base", COURSE, question, top_k=6)),
        ]
        await self.until(lambda: self.waiting_requests() == 4)
        self.assertEqual(answer_flights.active_flight_count(), 4)
        self.pipes.gate.set()
        await asyncio.gather(*tasks)
        self.assertEqual(len(self.pipes.calls), 4)

    async def test_a_shared_selected_question_is_stored_once(self) -> None:
        tasks = await self._concurrently(5, QUESTION)
        self.pipes.gate.set()
        answers = await asyncio.gather(*tasks)
        self.assertEqual(len(self.pipes.calls), 1)
        self.assertEqual(len(self.cache.stored()), 1)
        self.assertEqual(len({a.cache_entry_id for a in answers}), 1)

    async def test_a_failure_reaches_everyone_and_is_not_remembered(self) -> None:
        self.pipes.fail_with = HTTPException(status_code=502, detail="Ollama failed.")
        tasks = await self._concurrently(3)
        self.pipes.gate.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual([getattr(r, "status_code", None) for r in results], [502, 502, 502])
        self.assertEqual(len(self.pipes.calls), 1)
        self.pipes.fail_with = None
        self.pipes.gate = None
        self.assertEqual((await serve("base", COURSE, "An unselected question?")).answer, "It is due Friday.")
        self.assertEqual(len(self.pipes.calls), 2)

    async def test_cancelling_one_request_does_not_cancel_the_generation(self) -> None:
        tasks = await self._concurrently(3)
        tasks[0].cancel()
        await asyncio.sleep(0)
        self.pipes.gate.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self.assertIsInstance(results[0], asyncio.CancelledError)
        self.assertEqual([r.answer for r in results[1:]], ["It is due Friday."] * 2)
        self.assertEqual(len(self.pipes.calls), 1)

    async def test_a_waiting_generation_survives_one_student_leaving_but_not_all(self) -> None:
        question = "An unselected question?"
        blocker_gate = asyncio.Event()

        async def blocker() -> None:
            async with generation_slot("blocker"):
                await blocker_gate.wait()

        holding = asyncio.create_task(blocker())
        await self.until(lambda: get_generation_queue().active == 1)

        requests = [_Request(), _Request()]

        async def ask(request: _Request):
            bind_request(request)
            return await serve("base", COURSE, question)

        with patch("app.generation_queue.DISCONNECT_POLL_SECONDS", 0.01):
            tasks = [asyncio.create_task(ask(r)) for r in requests]
            queue = get_generation_queue()
            await self.until(lambda: self.waiting_requests() == 2 and queue.snapshot()["waiting"] == 1)
            self.assertEqual(queue.snapshot()["waiting"], 1)

            requests[0].gone = True
            await asyncio.sleep(0.05)
            # One of two has gone: the shared request keeps its place.
            self.assertEqual(queue.snapshot()["waiting"], 1)
            self.assertEqual(queue.counters["clientGone"], 0)

            requests[1].gone = True
            await self.until(lambda: queue.counters["clientGone"] == 1)
            self.assertEqual(queue.snapshot()["waiting"], 0)
            self.assertEqual(queue.counters["clientGone"], 1)

        blocker_gate.set()
        await holding
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual([getattr(r, "status_code", None) for r in results], [499, 499])
        self.assertEqual(self.pipes.calls, [("base", COURSE, question, 4)])
        self.assertEqual(get_generation_queue().counters["started"], 1)

    async def test_the_queue_limits_still_apply_to_the_shared_request(self) -> None:
        queue = get_generation_queue()
        queue.max_waiting = 1
        blocker_gate = asyncio.Event()

        async def blocker(label: str) -> None:
            async with generation_slot(label):
                await blocker_gate.wait()

        holding = [asyncio.create_task(blocker("a")), asyncio.create_task(blocker("b"))]
        await asyncio.sleep(0.01)
        results = await asyncio.gather(
            *(serve("base", COURSE, "An unselected question?") for _ in range(3)), return_exceptions=True
        )
        # Sharing is not a way round the limit: all three are told "busy" and
        # no model is reached. (A refusal is instant, so a request arriving
        # after it is refused in its own right rather than sharing it.)
        self.assertGreaterEqual(queue.counters["rejectedFull"], 1)
        for result in results:
            self.assertEqual(result.detail["code"], "generation_busy")
        # Only the two blockers ever take a turn: one running, one waiting.
        self.assertEqual(queue.counters["started"], 1)
        blocker_gate.set()
        await asyncio.gather(*holding)
        self.assertEqual(queue.counters["started"], 2)

    async def test_a_student_does_not_wait_on_background_work_that_has_not_started(self) -> None:
        question = "An unselected question?"
        blocker_gate = asyncio.Event()

        async def blocker() -> None:
            async with generation_slot("blocker"):
                await blocker_gate.wait()

        async def prepare():
            with background_generation():
                return await serve("base", COURSE, question)

        holding = asyncio.create_task(blocker())
        queue = get_generation_queue()
        await self.until(lambda: queue.active == 1)
        background = asyncio.create_task(prepare())
        await self.until(lambda: queue.snapshot()["backgroundWaiting"] == 1)
        student = asyncio.create_task(serve("base", COURSE, question))
        await self.until(lambda: queue.snapshot()["waiting"] == 1)
        # Both wait in the queue: the student's own request, ahead of preparation.
        self.assertEqual((queue.snapshot()["waiting"], queue.snapshot()["backgroundWaiting"]), (1, 1))
        blocker_gate.set()
        await asyncio.gather(holding, student, background)
        self.assertEqual(len(self.pipes.calls), 2)

    async def test_a_student_joins_background_work_that_is_already_running(self) -> None:
        question = "An unselected question?"
        self.pipes.gate = asyncio.Event()

        async def prepare():
            with background_generation():
                return await serve("base", COURSE, question)

        background = asyncio.create_task(prepare())
        await self.until(lambda: get_generation_queue().active == 1)
        student = asyncio.create_task(serve("base", COURSE, question))
        await self.until(lambda: self.waiting_requests() == 1)
        self.pipes.gate.set()
        answers = await asyncio.gather(background, student)
        self.assertEqual(len(self.pipes.calls), 1)
        self.assertEqual({a.answer for a in answers}, {"It is due Friday."})

    async def test_preparation_waits_for_a_students_generation_instead_of_repeating_it(self) -> None:
        self.pipes.gate = asyncio.Event()
        student = asyncio.create_task(serve("base", COURSE, QUESTION))
        await self.until(lambda: get_generation_queue().active == 1)

        async def prepare():
            with background_generation():
                return await serve_detailed("base", COURSE, QUESTION, source="prepared")

        background = asyncio.create_task(prepare())
        await asyncio.sleep(0.01)
        self.pipes.gate.set()
        await asyncio.gather(student, background)
        self.assertEqual(len(self.pipes.calls), 1)
        (row,) = self.cache.stored()
        self.assertEqual(row["source"], "live")


class ActivityTests(CacheTestCase):
    async def test_a_cache_hit_counts_as_student_activity(self) -> None:
        await serve("base", COURSE, QUESTION)
        queue = get_generation_queue()
        queue._last_interactive = None
        await serve("base", COURSE, QUESTION)
        self.assertIsNotNone(queue.seconds_since_interactive_activity())

    async def test_preparation_is_not_student_activity(self) -> None:
        queue = get_generation_queue()
        with background_generation():
            await serve("base", COURSE, QUESTION, )
        self.assertIsNone(queue.seconds_since_interactive_activity())


if __name__ == "__main__":
    unittest.main()
