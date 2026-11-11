"""Classroom preparation: one generation at a time, stop, resume, partial readiness.

No database and no model. `FakeAnswerCache` plays PostgreSQL and the identity
lookups; the four pipelines are fakes that take a real turn in the real
generation queue, so ordering and priority are the queue's own.
"""

from __future__ import annotations

import asyncio
import os
import unittest
from typing import Any
from unittest.mock import patch

from fastapi import HTTPException

from app import answer_cache, classroom_preparation as prep
from app.answer_cache import serve_detailed
from app.answer_identity import (
    REASON_FINETUNED_SERVICE,
    REASON_NO_COURSE_MODEL,
    REASON_NO_INDEX,
)
from app.error_classification import (
    FINETUNED_UNAVAILABLE,
    OLLAMA_TIMEOUT,
    SYLLABUS_NOT_INDEXED,
    ClassifiedHTTPException,
)
from app.generation_mode import GenerationMode
from app.generation_queue import GenerationQueue, generation_slot, reset_generation_queue_for_tests
from app.schemas import (
    BaseModelGenerateResponse,
    FineTunedGenerateResponse,
    FineTunedRagGenerateResponse,
    RagGenerateResponse,
)

from tests.answer_cache_fake import FakeAnswerCache

COURSE = "css-360-winter-2026-a7rp"
OTHER = "css-350-spring-2026-n3h9"
FT_IDENTITY = {"model": {"tag": "css360-v2:latest", "digest": "d", "courseVersion": "v2"}}
QUESTIONS = ["When is the final exam?", "What is the late policy?", "How is the grade calculated?"]


class FourPipelines:
    """The four approaches as fakes that queue like the real ones and count runs."""

    def __init__(self) -> None:
        self.started: list[tuple[str, str]] = []
        self.finished: list[tuple[str, str]] = []
        self.running = 0
        self.max_running = 0
        self.gate: asyncio.Event | None = None
        #: condition -> exception to raise, or a callable(question) -> exception | None
        self.failures: dict[str, Any] = {}
        self.hang: set[str] = set()

    async def _generate(self, condition: str, question: str) -> None:
        self.started.append((condition, question))
        async with generation_slot(condition):
            self.running += 1
            self.max_running = max(self.max_running, self.running)
            try:
                if self.gate is not None:
                    await self.gate.wait()
                if condition in self.hang:
                    await asyncio.Event().wait()
                failure = self.failures.get(condition)
                if callable(failure):
                    failure = failure(question)
                if failure is not None:
                    raise failure
                await asyncio.sleep(0)
            finally:
                self.running -= 1
        self.finished.append((condition, question))

    async def base(self, course_id: str, question: str, top_k: int):
        await self._generate("base", question)
        return BaseModelGenerateResponse(answer=f"base: {question}", model="llama3.2:3b", responseType="base", courseId=course_id)

    async def rag(self, course_id: str, question: str, top_k: int):
        await self._generate("rag", question)
        return RagGenerateResponse(
            courseId=course_id, answer=f"rag: {question}", model="llama3.2:3b",
            sources=[{"chunkId": "c1", "sectionTitle": "Exams", "text": "The final is in week 11.", "score": 0.9}],
            retrievedChunks=[{"chunkId": "c1", "section": "Exams", "text": "The final is in week 11.", "score": 0.9}],
        )

    async def fine_tuned(self, course_id: str, question: str, top_k: int):
        await self._generate("fineTuned", question)
        return FineTunedGenerateResponse(
            answer=f"ft: {question}", model="css360-v2:latest", responseType="fineTuned", courseId=course_id,
            modelVersion="v2", adapterLoaded=True, generationSeconds=1.0, engine="vm",
        )

    async def fine_tuned_rag(self, course_id: str, question: str, top_k: int):
        await self._generate("fineTunedRag", question)
        return FineTunedRagGenerateResponse(
            courseId=course_id, answer=f"ft+rag: {question}", model="css360-v2:latest",
            sources=[{"chunkId": "c1", "sectionTitle": "Exams", "text": "The final is in week 11.", "score": 0.9}],
            retrievedChunks=[{"chunkId": "c1", "section": "Exams", "text": "The final is in week 11.", "score": 0.9}],
            modelVersion="v2", adapterLoaded=True, generationSeconds=1.0, engine="vm",
        )

    def install(self, testcase: unittest.TestCase) -> "FourPipelines":
        registered = {
            "base": answer_cache.RegisteredPipeline(self.base, BaseModelGenerateResponse),
            "rag": answer_cache.RegisteredPipeline(self.rag, RagGenerateResponse),
            "fineTuned": answer_cache.RegisteredPipeline(self.fine_tuned, FineTunedGenerateResponse),
            "fineTunedRag": answer_cache.RegisteredPipeline(self.fine_tuned_rag, FineTunedRagGenerateResponse),
        }
        patcher = patch.dict(answer_cache._pipelines, registered, clear=True)
        patcher.start()
        testcase.addCleanup(patcher.stop)
        return self


class PreparationTestCase(unittest.IsolatedAsyncioTestCase):
    quiet = "0"

    def setUp(self) -> None:
        self.cache = FakeAnswerCache().install(self)
        self.pipes = FourPipelines().install(self)
        for course in (COURSE, OTHER):
            self.cache.set_identity("fineTuned", course, FT_IDENTITY)
            self.cache.set_identity("fineTunedRag", course, FT_IDENTITY)
        self.cache.select(COURSE, *QUESTIONS)
        env = patch.dict(os.environ, {
            "CLASSROOM_PREP_QUIET_SECONDS": self.quiet,
            "CLASSROOM_PREP_STOP_GRACE_SECONDS": "30",
            "CLASSROOM_PREP_BATCH_SIZE": "1",
        })
        env.start()
        self.addCleanup(env.stop)
        poll = patch("app.classroom_preparation.POLL_SECONDS", 0.005)
        poll.start()
        self.addCleanup(poll.stop)

    async def asyncTearDown(self) -> None:
        run = prep.active_run()
        if run is not None and run.task is not None:
            run.task.cancel()
            await asyncio.gather(run.task, return_exceptions=True)

    async def finish(self, course_id: str = COURSE) -> dict[str, Any]:
        run = prep.active_run()
        if run is not None and run.task is not None:
            await asyncio.wait_for(asyncio.gather(run.task, return_exceptions=True), timeout=5)
        return await prep.course_status(course_id)

    async def until(self, condition, timeout: float = 2.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while not condition():
            if asyncio.get_running_loop().time() > deadline:
                self.fail("condition was not reached in time")
            await asyncio.sleep(0.002)


class FullRunTests(PreparationTestCase):
    async def test_every_selected_question_gets_all_four_real_answers_one_at_a_time(self) -> None:
        await prep.start(COURSE)
        status = await self.finish()
        self.assertEqual(status["state"], "ready")
        self.assertEqual(
            status["counts"],
            {"selectedQuestions": 3, "expectedAnswers": 12, "readyAnswers": 12, "missingAnswers": 0,
             "outdatedAnswers": 0, "unavailableAnswers": 0, "fullyPreparedQuestions": 3, "preparableAnswers": 0},
        )
        self.assertEqual(len(self.pipes.finished), 12)
        self.assertEqual(self.pipes.max_running, 1)
        rows = self.cache.stored(COURSE)
        self.assertEqual({row["source"] for row in rows}, {"prepared"})
        # The pipelines' own answers, never a training answer.
        self.assertEqual(
            {row["payload"]["answer"] for row in rows if row["condition"] == "rag"},
            {f"rag: {q}" for q in QUESTIONS},
        )
        self.assertFalse(status["canStart"])
        self.assertEqual(status["startBlockedReason"], "nothing_to_prepare")
        self.assertEqual(status["lastRun"]["state"], "completed")
        self.assertEqual(status["lastRun"]["generated"], 12)

    async def test_the_default_order_is_question_by_question(self) -> None:
        await prep.start(COURSE)
        await self.finish()
        self.assertEqual(
            self.pipes.finished[:5],
            [("base", QUESTIONS[0]), ("rag", QUESTIONS[0]), ("fineTuned", QUESTIONS[0]),
             ("fineTunedRag", QUESTIONS[0]), ("base", QUESTIONS[1])],
        )

    async def test_a_batch_groups_each_approach_within_it(self) -> None:
        with patch.dict(os.environ, {"CLASSROOM_PREP_BATCH_SIZE": "2"}):
            await prep.start(COURSE)
            await self.finish()
        self.assertEqual(
            [condition for condition, _q in self.pipes.finished[:8]],
            ["rag", "rag", "fineTunedRag", "fineTunedRag", "base", "base", "fineTuned", "fineTuned"],
        )
        # The last, short batch is complete too.
        self.assertEqual(len(self.pipes.finished), 12)

    async def test_each_answer_is_usable_by_students_as_soon_as_it_is_stored(self) -> None:
        self.pipes.gate = asyncio.Event()
        await prep.start(COURSE)
        await self.until(lambda: self.pipes.running == 1)
        self.pipes.gate.set()
        await self.until(lambda: len(self.cache.stored()) >= 1)
        self.pipes.gate = asyncio.Event()
        # The run is still going; the first answer is already a hit.
        served = await serve_detailed("base", COURSE, QUESTIONS[0])
        self.assertEqual(served.outcome, "hit")
        self.assertTrue(served.response.cached)
        self.pipes.gate.set()
        await self.finish()

    async def test_starting_twice_is_one_run(self) -> None:
        self.pipes.gate = asyncio.Event()
        first = await prep.start(COURSE)
        second = await prep.start(COURSE)
        self.assertIs(first, second)
        self.pipes.gate.set()
        await self.finish()
        self.assertEqual(len(self.pipes.finished), 12)


class CountsTests(PreparationTestCase):
    async def test_fully_prepared_questions_are_counted_not_inferred(self) -> None:
        self.cache.select(COURSE, QUESTIONS[0], QUESTIONS[1])
        for condition in ("base", "rag", "fineTuned", "fineTunedRag"):
            await serve_detailed(condition, COURSE, QUESTIONS[0])
        for condition in ("base", "rag"):
            await serve_detailed(condition, COURSE, QUESTIONS[1])
        status = await prep.course_status(COURSE)
        counts = status["counts"]
        self.assertEqual((counts["readyAnswers"], counts["expectedAnswers"]), (6, 8))
        self.assertEqual(counts["fullyPreparedQuestions"], 1)
        self.assertEqual(status["state"], "partially_ready")
        by_question = {q["question"]: q["answers"] for q in status["questions"]}
        self.assertEqual(by_question[QUESTIONS[1]], {"base": "ready", "rag": "ready", "fineTuned": "missing", "fineTunedRag": "missing"})
        self.assertTrue(status["canStart"])
        self.assertIn("1 of 2 questions fully ready", status["message"])

    async def test_a_course_with_no_fine_tuned_model_expects_two_answers_per_question(self) -> None:
        for condition in ("fineTuned", "fineTunedRag"):
            self.cache.set_identity(condition, COURSE, REASON_NO_COURSE_MODEL)
        await prep.start(COURSE)
        status = await self.finish()
        self.assertEqual(status["state"], "ready")
        self.assertEqual((status["counts"]["expectedAnswers"], status["counts"]["readyAnswers"]), (6, 6))
        self.assertEqual({c for c, _q in self.pipes.finished}, {"base", "rag"})
        self.assertFalse(status["conditions"]["fineTuned"]["expected"])
        self.assertTrue(any("no active fine-tuned model" in notice for notice in status["notices"]))
        self.assertEqual(status["questions"][0]["answers"]["fineTuned"], "not_expected")

    async def test_a_course_with_no_index_does_not_expect_the_retrieval_answers(self) -> None:
        for condition in ("rag", "fineTunedRag"):
            self.cache.set_identity(condition, COURSE, REASON_NO_INDEX)
        status = await prep.course_status(COURSE)
        self.assertEqual(status["counts"]["expectedAnswers"], 6)
        self.assertTrue(any("not indexed" in notice for notice in status["notices"]))

    async def test_a_model_that_is_down_is_expected_but_unavailable(self) -> None:
        for condition in ("fineTuned", "fineTunedRag"):
            self.cache.set_identity(condition, COURSE, REASON_FINETUNED_SERVICE)
        await prep.start(COURSE)
        status = await self.finish()
        counts = status["counts"]
        self.assertEqual((counts["expectedAnswers"], counts["readyAnswers"], counts["unavailableAnswers"]), (12, 6, 6))
        self.assertEqual(status["state"], "partially_ready")
        self.assertEqual({c for c, _q in self.pipes.started}, {"base", "rag"})
        self.assertTrue(any("fine-tuned model is unavailable" in notice for notice in status["notices"]))
        # Nothing that can be prepared is left, so there is no button to press.
        self.assertFalse(status["canStart"])
        self.assertEqual(status["questions"][0]["answers"]["fineTuned"], "unavailable")

    async def test_answers_go_out_of_date_when_what_produced_them_changes(self) -> None:
        await prep.start(COURSE)
        await self.finish()
        for condition in ("rag", "fineTunedRag"):
            fields = FT_IDENTITY if condition == "fineTunedRag" else self.cache.default_fields
            self.cache.set_identity(condition, COURSE, {**fields, "index": "replaced-syllabus"})
        status = await prep.course_status(COURSE)
        self.assertEqual(status["state"], "partially_ready")
        self.assertEqual((status["counts"]["readyAnswers"], status["counts"]["outdatedAnswers"]), (6, 6))
        self.assertTrue(any("out of date" in notice for notice in status["notices"]))
        self.assertEqual(status["questions"][0]["answers"]["rag"], "outdated")
        # Resuming regenerates only the outdated ones; the old rows are kept.
        done = len(self.pipes.finished)
        await prep.start(COURSE)
        status = await self.finish()
        self.assertEqual(status["state"], "ready")
        self.assertEqual(len(self.pipes.finished) - done, 6)
        self.assertEqual(len(self.cache.stored(COURSE)), 18)

    async def test_nothing_selected_is_not_started(self) -> None:
        self.cache.set_selection(COURSE, [])
        status = await prep.course_status(COURSE)
        self.assertEqual((status["state"], status["startBlockedReason"]), ("not_started", "nothing_selected"))
        self.assertEqual([q["selected"] for q in status["questions"]], [False, False, False])
        self.assertIsNone(status["questions"][0]["answers"])


class EligibilityTests(PreparationTestCase):
    async def test_which_questions_are_listed_and_prepared(self) -> None:
        cache = self.cache
        cache.seeds[COURSE], cache.selection[COURSE] = [], []
        curated = cache.add_question(COURSE, "Curated, never reviewed?", origin="prototype", review_status=None, selected=True)
        drafted = cache.add_question(COURSE, "AI draft, unreviewed?", origin="ai_generated", review_status="generated", selected=True)
        rejected = cache.add_question(COURSE, "Rejected?", origin="ai_generated", review_status="rejected", selected=True)
        cache.add_question(COURSE, "Rejected and unselected?", origin="prototype", review_status="rejected")
        cache.add_question(COURSE, "A student's unapproved question?", origin="user", participant_id="p1")
        approved = cache.add_question(COURSE, "A student's approved question?", origin="user", participant_id="p1", review_status="approved")
        cache.add_question(COURSE, "What is the weather today?", origin="authored")

        status = await prep.course_status(COURSE)
        listed = {q["seedId"]: q for q in status["questions"]}
        # The unapproved student question and the unselected rejected one are not listed at all.
        self.assertEqual(len(listed), 5)
        self.assertNotIn("A student's unapproved question?", {q["question"] for q in status["questions"]})
        self.assertEqual(listed[curated]["sourceLabel"], "Staff-curated")
        self.assertEqual(listed[drafted]["sourceLabel"], "AI-generated")
        self.assertEqual(listed[approved]["sourceLabel"], "Student contribution")
        self.assertFalse(listed[approved]["selected"])
        # A selected question that became ineligible stays visible, flagged, and is skipped.
        self.assertEqual((listed[rejected]["eligible"], listed[rejected]["ineligibleReason"]), (False, "rejected"))
        self.assertIsNone(listed[rejected]["answers"])
        self.assertEqual(status["counts"]["selectedQuestions"], 2)
        self.assertTrue(any("can no longer be prepared" in notice for notice in status["notices"]))

        await prep.start(COURSE)
        await self.finish()
        self.assertEqual({q for _c, q in self.pipes.finished}, {"Curated, never reviewed?", "AI draft, unreviewed?"})

    async def test_a_staff_origin_on_a_participants_row_is_still_a_students(self) -> None:
        self.cache.seeds[COURSE], self.cache.selection[COURSE] = [], []
        self.cache.add_question(COURSE, "Mislabelled?", origin="prototype", participant_id="p1")
        status = await prep.course_status(COURSE)
        self.assertEqual(status["questions"], [])

    async def test_two_selected_questions_with_the_same_wording_are_generated_once(self) -> None:
        self.cache.select(COURSE, "When is the exam?", "when is the EXAM")
        await prep.start(COURSE)
        status = await self.finish()
        self.assertEqual(len(self.pipes.finished), 4)
        self.assertEqual((status["counts"]["readyAnswers"], status["counts"]["fullyPreparedQuestions"]), (8, 2))


class StopAndResumeTests(PreparationTestCase):
    async def test_stop_keeps_what_is_done_and_resume_does_only_the_rest(self) -> None:
        self.pipes.gate = asyncio.Event()
        await prep.start(COURSE)
        self.pipes.gate.set()
        await self.until(lambda: len(self.pipes.finished) >= 5)
        self.assertTrue(await prep.request_stop(COURSE))
        status = await self.finish()
        done = len(self.pipes.finished)
        self.assertEqual(status["state"], "stopped")
        self.assertLess(done, 12)
        self.assertEqual(status["counts"]["readyAnswers"], done)
        self.assertEqual(len(self.cache.stored(COURSE)), done)
        self.assertIn("completed answers are saved", status["message"])
        self.assertTrue(status["canStart"])
        self.assertEqual(status["lastRun"]["state"], "stopped")

        # Every stored answer is usable now, including on a half-prepared question.
        ready = [(q["question"], c) for q in status["questions"] for c, s in q["answers"].items() if s == "ready"]
        for question, condition in ready:
            self.assertEqual((await serve_detailed(condition, COURSE, question)).outcome, "hit")
        self.assertEqual(len(self.pipes.finished), done)

        await prep.start(COURSE)
        status = await self.finish()
        self.assertEqual(status["state"], "ready")
        self.assertEqual(len(self.pipes.finished), 12)
        self.assertEqual(len(self.cache.stored(COURSE)), 12)

    async def test_stop_lets_the_running_generation_finish_and_saves_it(self) -> None:
        self.pipes.gate = asyncio.Event()
        await prep.start(COURSE)
        await self.until(lambda: self.pipes.running == 1)
        await prep.request_stop(COURSE)
        status = await prep.course_status(COURSE)
        self.assertEqual((status["state"], status["stopping"], status["canStart"]), ("stopping", True, False))
        self.assertIn("Stopping after the current answer", status["message"])
        self.assertEqual(self.cache.runs[COURSE]["state"], "stopping")
        await asyncio.sleep(0.03)
        # Still generating: a stop does not cut a running generation short.
        self.assertEqual(self.pipes.running, 1)
        self.pipes.gate.set()
        status = await self.finish()
        self.assertEqual(status["state"], "stopped")
        self.assertEqual(self.pipes.finished, [("base", QUESTIONS[0])])
        self.assertEqual(status["counts"]["readyAnswers"], 1)

    async def test_stop_while_waiting_for_a_turn_drops_the_item_at_once(self) -> None:
        release = asyncio.Event()

        async def student() -> None:
            async with generation_slot("base"):
                await release.wait()

        holding = asyncio.create_task(student())
        await asyncio.sleep(0.01)
        await prep.start(COURSE)
        await self.until(lambda: len(self.pipes.started) == 1)
        await prep.request_stop(COURSE)
        status = await self.finish()
        self.assertEqual(status["state"], "stopped")
        self.assertEqual(self.pipes.finished, [])
        self.assertEqual(self.pipes.max_running, 0)
        release.set()
        await holding

    async def test_a_stalled_generation_cannot_leave_the_run_stopping_forever(self) -> None:
        self.pipes.hang.add("base")
        with patch.dict(os.environ, {"CLASSROOM_PREP_STOP_GRACE_SECONDS": "0.05"}):
            await prep.start(COURSE)
            await self.until(lambda: self.pipes.running == 1)
            await prep.request_stop(COURSE)
            status = await self.finish()
        self.assertEqual(status["state"], "stopped")
        self.assertFalse(status["running"])
        self.assertEqual(self.cache.runs[COURSE]["reason"], "stop_forced")
        self.assertIn("still being generated", status["message"])
        self.assertIsNone(prep.active_run())
        self.assertEqual(self.pipes.running, 0)
        self.assertTrue(status["canStart"])

    async def test_a_worker_stuck_outside_a_generation_is_cancelled_after_the_grace_period(self) -> None:
        async def never(*_args: Any, **_kwargs: Any):
            await asyncio.Event().wait()

        real_plan = prep.build_plan
        calls = {"n": 0}

        async def plan_then_hang(course_id: str):
            calls["n"] += 1
            if calls["n"] == 1:
                return await real_plan(course_id)
            return await never()

        with patch.dict(os.environ, {"CLASSROOM_PREP_STOP_GRACE_SECONDS": "0.02"}):
            with patch("app.classroom_preparation.build_plan", new=plan_then_hang):
                await prep.start(COURSE)
                await self.until(lambda: calls["n"] >= 2)
                await prep.request_stop(COURSE)
                await asyncio.sleep(0.08)
                self.assertIsNone(prep.active_run())
            status = await prep.course_status(COURSE)
        self.assertEqual((status["state"], status["running"]), ("stopped", False))
        self.assertEqual(self.cache.runs[COURSE]["reason"], "stop_forced")

    async def test_stopping_a_course_that_is_not_running_does_nothing(self) -> None:
        self.assertFalse(await prep.request_stop(COURSE))
        await prep.start(COURSE)
        self.assertFalse(await prep.request_stop(OTHER))
        await self.finish()

    async def test_a_question_deselected_during_the_run_is_not_prepared(self) -> None:
        self.pipes.gate = asyncio.Event()
        await prep.start(COURSE)
        await self.until(lambda: self.pipes.running == 1)
        self.cache.set_selection(COURSE, ["s1"])
        self.pipes.gate.set()
        status = await self.finish()
        self.assertEqual({q for _c, q in self.pipes.finished}, {QUESTIONS[0]})
        self.assertEqual((status["state"], status["counts"]["expectedAnswers"]), ("ready", 4))


class RestartTests(PreparationTestCase):
    async def test_a_run_from_a_previous_process_is_reported_as_interrupted(self) -> None:
        self.pipes.gate = asyncio.Event()
        await prep.start(COURSE)
        self.pipes.gate.set()
        await self.until(lambda: len(self.cache.stored(COURSE)) >= 3)
        # The backend dies: the task is gone, the row still says running.
        run = prep.active_run()
        self.cache.runs[COURSE] = {**self.cache.runs[COURSE], "state": "running", "bootId": "an-earlier-process"}
        with patch("app.classroom_preparation._finish_sync"):
            run.task.cancel()
            await asyncio.gather(run.task, return_exceptions=True)
        prep.reset_for_tests()
        kept = len(self.cache.stored(COURSE))
        self.assertGreaterEqual(kept, 3)

        status = await prep.course_status(COURSE)
        self.assertEqual((status["state"], status["running"]), ("stopped", False))
        self.assertEqual(status["lastRun"]["state"], "interrupted")
        self.assertIn("interrupted when the server restarted", status["message"])
        self.assertEqual(status["counts"]["readyAnswers"], kept)
        self.assertTrue(status["canStart"])

        # Manual resume: only what is missing is generated.
        before = len(self.pipes.finished)
        await prep.start(COURSE)
        status = await self.finish()
        self.assertEqual(status["state"], "ready")
        self.assertEqual(len(self.pipes.finished) - before, 12 - kept)

    async def test_backend_shutdown_ends_the_run_and_records_it(self) -> None:
        self.pipes.gate = asyncio.Event()
        await prep.start(COURSE)
        await self.until(lambda: self.pipes.running == 1)
        await prep.shutdown()
        self.assertEqual(self.cache.runs[COURSE]["state"], "stopped")
        self.assertEqual(self.cache.runs[COURSE]["reason"], "cancelled")
        self.assertIsNone(prep.active_run())

    async def test_a_status_row_that_cannot_be_written_does_not_lose_answers(self) -> None:
        def broken(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("database down")

        with patch("app.classroom_preparation._write_finish", new=broken), patch(
            "app.classroom_preparation._write_progress", new=broken
        ):
            await prep.start(COURSE)
            status = await self.finish()
        self.assertEqual(status["counts"]["readyAnswers"], 12)
        self.assertEqual(status["state"], "ready")


class StudentsFirstTests(PreparationTestCase):
    quiet = "120"

    def setUp(self) -> None:
        super().setUp()
        self.now = [1000.0]
        reset_generation_queue_for_tests(
            GenerationQueue(max_concurrency=1, max_waiting=160, queue_timeout=30.0, clock=lambda: self.now[0])
        )

    async def student(self, label: str = "base") -> None:
        async with generation_slot(label):
            await asyncio.sleep(0)

    async def test_preparation_waits_for_the_quiet_period_after_student_activity(self) -> None:
        await self.student()
        await prep.start(COURSE)
        await asyncio.sleep(0.03)
        status = await prep.course_status(COURSE)
        self.assertEqual((status["state"], status["waiting"]), ("preparing", "quiet"))
        self.assertIn("Waiting for students to finish", status["message"])
        self.assertEqual(self.pipes.started, [])

        self.now[0] += 119
        await asyncio.sleep(0.03)
        self.assertEqual(self.pipes.started, [])

        self.now[0] += 2
        await self.until(lambda: len(self.pipes.finished) == 12)
        self.assertEqual((await self.finish())["state"], "ready")

    async def test_with_no_student_since_start_preparation_begins_at_once(self) -> None:
        await prep.start(COURSE)
        await self.until(lambda: len(self.pipes.finished) == 12)

    async def test_a_student_request_goes_ahead_and_restarts_the_quiet_period(self) -> None:
        self.pipes.gate = asyncio.Event()
        await prep.start(COURSE)
        await self.until(lambda: self.pipes.running == 1)
        # A student arrives while preparation is generating its first answer.
        order: list[str] = []

        async def student() -> None:
            async with generation_slot("rag"):
                order.append("student")

        asking = asyncio.create_task(student())
        await asyncio.sleep(0.01)
        self.pipes.gate.set()
        await asking
        await asyncio.sleep(0.03)
        # The first answer was saved; nothing else starts until the site is quiet again.
        self.assertEqual(self.pipes.finished, [("base", QUESTIONS[0])])
        self.assertEqual(order, ["student"])
        self.assertEqual((await prep.course_status(COURSE))["waiting"], "quiet")
        self.now[0] += 121
        await self.until(lambda: len(self.pipes.finished) == 12)

    async def test_a_stored_answer_served_to_a_student_also_holds_preparation_off(self) -> None:
        await prep.start(COURSE)
        await self.until(lambda: len(self.pipes.finished) == 12)
        await self.finish()
        self.cache.select(OTHER, "When is the midterm?")
        await serve_detailed("base", COURSE, QUESTIONS[0])  # a hit: no generation, but a student is here
        await prep.start(OTHER)
        await asyncio.sleep(0.03)
        self.assertEqual((await prep.course_status(OTHER))["waiting"], "quiet")
        self.assertEqual(len(self.pipes.started), 12)

    async def test_stop_during_the_quiet_wait_is_immediate(self) -> None:
        await self.student()
        await prep.start(COURSE)
        await asyncio.sleep(0.02)
        await prep.request_stop(COURSE)
        status = await self.finish()
        self.assertEqual(status["state"], "stopped")
        self.assertEqual(self.pipes.started, [])

    async def test_preparation_waits_while_a_starter_job_runs(self) -> None:
        with patch("app.classroom_preparation.is_starter_job_active", return_value=True):
            await prep.start(COURSE)
            await asyncio.sleep(0.03)
            status = await prep.course_status(COURSE)
            self.assertEqual(status["waiting"], "starter")
            self.assertEqual(self.pipes.started, [])
        await self.until(lambda: len(self.pipes.finished) == 12)


class FailureTests(PreparationTestCase):
    async def test_a_fine_tuned_outage_skips_those_approaches_and_keeps_going(self) -> None:
        self.pipes.failures["fineTuned"] = ClassifiedHTTPException(503, FINETUNED_UNAVAILABLE, "down")
        await prep.start(COURSE)
        status = await self.finish()
        # One attempt, then both fine-tuned approaches are left alone for this run.
        self.assertEqual([c for c, _q in self.pipes.started].count("fineTuned"), 1)
        self.assertNotIn("fineTunedRag", {c for c, _q in self.pipes.started})
        self.assertEqual({c for c, _q in self.pipes.finished}, {"base", "rag"})
        self.assertEqual(status["counts"]["readyAnswers"], 6)
        self.assertEqual(status["state"], "partially_ready")
        self.assertEqual(self.cache.runs[COURSE]["state"], "completed")
        self.assertTrue(any("fine-tuned model did not respond" in notice for notice in status["notices"]))
        self.assertEqual([row for row in self.cache.stored() if row["condition"].startswith("fineTuned")], [])
        self.assertTrue(status["canStart"])  # retry is the same button

    async def test_three_failures_in_a_row_end_the_run_as_failed(self) -> None:
        self.pipes.failures["base"] = ClassifiedHTTPException(503, OLLAMA_TIMEOUT, "timed out")
        self.pipes.failures["rag"] = ClassifiedHTTPException(503, OLLAMA_TIMEOUT, "timed out")
        self.cache.set_identity("fineTuned", COURSE, REASON_NO_COURSE_MODEL)
        self.cache.set_identity("fineTunedRag", COURSE, REASON_NO_COURSE_MODEL)
        await prep.start(COURSE)
        status = await self.finish()
        self.assertEqual(status["state"], "failed")
        self.assertEqual(len(self.pipes.started), 3)
        self.assertIn("The base model did not respond.", status["message"])
        self.assertIn("completed answers are saved", status["message"])
        self.assertEqual(self.cache.stored(), [])
        self.assertTrue(status["canStart"])

    async def test_one_failure_between_successes_does_not_end_the_run(self) -> None:
        self.pipes.failures["rag"] = lambda question: (
            HTTPException(status_code=502, detail="bad") if question == QUESTIONS[0] else None
        )
        await prep.start(COURSE)
        status = await self.finish()
        self.assertEqual(status["counts"]["readyAnswers"], 11)
        self.assertEqual(self.cache.runs[COURSE]["state"], "completed")
        self.assertEqual(status["lastRun"]["failed"], 1)

    async def test_a_question_with_no_matching_syllabus_text_does_not_count_towards_stopping(self) -> None:
        missing = ClassifiedHTTPException(404, SYLLABUS_NOT_INDEXED, "nothing retrieved")
        self.pipes.failures["rag"] = missing
        self.pipes.failures["fineTunedRag"] = missing
        await prep.start(COURSE)
        status = await self.finish()
        self.assertEqual(self.cache.runs[COURSE]["state"], "completed")
        self.assertEqual(status["counts"]["readyAnswers"], 6)

    async def test_a_failed_answer_is_not_retried_within_the_same_run(self) -> None:
        self.pipes.failures["base"] = lambda question: HTTPException(status_code=502, detail="bad")
        self.cache.select(COURSE, QUESTIONS[0])
        await prep.start(COURSE)
        await self.finish()
        self.assertEqual([c for c, _q in self.pipes.started].count("base"), 1)

    async def test_an_answer_that_cannot_be_saved_is_counted_as_a_failure(self) -> None:
        self.cache.fail_writes = True
        self.cache.select(COURSE, QUESTIONS[0])
        await prep.start(COURSE)
        status = await self.finish()
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["counts"]["readyAnswers"], 0)
        self.assertIn("not saved", status["message"])

    async def test_a_database_failure_mid_run_ends_it_cleanly(self) -> None:
        self.pipes.gate = asyncio.Event()
        await prep.start(COURSE)
        await self.until(lambda: self.pipes.running == 1)
        self.cache.fail_plan = True
        self.pipes.gate.set()
        run = prep.active_run()
        await asyncio.wait_for(asyncio.gather(run.task, return_exceptions=True), timeout=5)
        self.assertIsNone(prep.active_run())
        self.cache.fail_plan = False
        status = await prep.course_status(COURSE)
        self.assertEqual(status["state"], "failed")
        self.assertEqual(self.cache.runs[COURSE]["reason"], "internal_error")
        self.assertEqual(status["counts"]["readyAnswers"], 1)

    async def test_switching_to_gpu_mode_stops_the_run(self) -> None:
        self.pipes.gate = asyncio.Event()
        await prep.start(COURSE)
        await self.until(lambda: self.pipes.running == 1)
        gpu = GenerationMode(kind="gpu", gpu_url="http://127.0.0.1:9100")
        with patch("app.classroom_preparation.current_mode", return_value=gpu):
            self.pipes.gate.set()
            run = prep.active_run()
            await asyncio.wait_for(asyncio.gather(run.task, return_exceptions=True), timeout=5)
        self.assertEqual(self.cache.runs[COURSE]["reason"], "gpu_mode")
        self.assertEqual(len(self.pipes.finished), 1)


class RefusalTests(PreparationTestCase):
    async def _refused(self, course_id: str = COURSE) -> str:
        with self.assertRaises(prep.PreparationRefused) as raised:
            await prep.start(course_id)
        return raised.exception.code

    async def test_refused_when_switched_off(self) -> None:
        with patch.dict(os.environ, {answer_cache.ENABLED_ENV: ""}):
            self.assertEqual(await self._refused(), "disabled")
            status = await prep.course_status(COURSE)
        self.assertEqual((status["enabled"], status["canStart"], status["startBlockedReason"]), (False, False, "disabled"))
        self.assertTrue(any("switched off" in notice for notice in status["notices"]))

    async def test_refused_in_gpu_mode(self) -> None:
        gpu = GenerationMode(kind="gpu", gpu_url="http://127.0.0.1:9100")
        with patch("app.classroom_preparation.current_mode", return_value=gpu):
            self.assertEqual(await self._refused(), "gpu_mode")
            status = await prep.course_status(COURSE)
        self.assertEqual(status["startBlockedReason"], "gpu_mode")

    async def test_only_one_course_is_prepared_at_a_time(self) -> None:
        self.cache.select(OTHER, "When is the midterm?")
        self.pipes.gate = asyncio.Event()
        await prep.start(COURSE)
        self.assertEqual(await self._refused(OTHER), "another_course")
        status = await prep.course_status(OTHER)
        self.assertEqual((status["running"], status["startBlockedReason"]), (False, "another_course"))
        self.assertTrue(any("Another course" in notice for notice in status["notices"]))
        self.pipes.gate.set()
        await self.finish()
        # Free again once the first course has finished.
        await prep.start(OTHER)
        self.assertEqual((await self.finish(OTHER))["state"], "ready")

    async def test_refused_with_nothing_selected(self) -> None:
        self.cache.set_selection(COURSE, [])
        self.assertEqual(await self._refused(), "nothing_selected")
        self.assertIsNone(prep.active_run())


if __name__ == "__main__":
    unittest.main()
