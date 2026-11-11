"""Classroom preparation: generate the selected questions' answers before class.

Staff select some of a course's existing questions (`classroom_questions`).
Preparation sends each one through the four real classroom pipelines, one
generation at a time, and every successful answer is stored at once through
`app.answer_cache`, where students' requests find it. Nothing here copies a
training answer and nothing here talks to a model directly.

How a run works
---------------
One run at a time on the whole server, as a task in the backend process:

1. Before every generation the work list is recomputed from the database: the
   selected questions, the approaches the course supports, and which answers
   already exist under the current fingerprints. An answer that exists is
   skipped; a question deselected or edited meanwhile is picked up. Starting
   and resuming are therefore the same thing.
2. The next generation starts only when no student request has been active or
   waiting for `CLASSROOM_PREP_QUIET_SECONDS` (default 120), and no starter
   job is running.
3. The generation takes a *background* turn in the existing generation queue
   (`background_generation`), so every student request goes first.
4. A stored answer is usable by students immediately; nothing waits for the
   question's other approaches or for the run to finish.

Stopping
--------
`request_stop` is graceful. No further generation is scheduled. A generation
that has not reached a model (waiting for a quiet period or for its turn) is
dropped at once. One that is already generating is left to finish and its
answer is kept; every model call has a 120 s timeout, so that wait is bounded.
A run that still has not ended `CLASSROOM_PREP_STOP_GRACE_SECONDS` (default
180) after the request is cancelled outright, so "stopping" can never be
permanent. That last resort drops the backend's connection only: Ollama may
keep computing that one answer for a while. Preparation starts nothing else
after it, so the only overlap possible is with a student's request.

Restarts
--------
The run lives in this process. Stored answers survive a restart; at most the
one answer being generated is lost. Nothing resumes on its own. The status row
(`classroom_preparation`) carries the id of the process that started the run,
so a row still marked running by a process that no longer exists is reported
as interrupted, never as running.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from app import (
    answer_cache,
    answer_flights,
    db_answer_cache,
    db_classroom_preparation,
    db_classroom_questions,
)
from app.answer_identity import (
    FINE_TUNED_CONDITIONS,
    GROUNDED_CONDITIONS,
    REASON_BASE_MODEL,
    REASON_EMBEDDING_MODEL,
    REASON_FINETUNED_MODEL,
    REASON_FINETUNED_SERVICE,
    REASON_NO_COURSE_MODEL,
    REASON_NO_INDEX,
    build_identity,
)
from app.db import db_connection
from app.error_classification import (
    FINETUNED_BAD_RESPONSE,
    FINETUNED_NOT_CONFIGURED,
    FINETUNED_REJECTED,
    FINETUNED_TIMEOUT,
    FINETUNED_UNAVAILABLE,
    FINETUNED_UPSTREAM_ERROR,
    MODEL_NOT_ACTIVATED,
    MODEL_VERSION_NOT_SERVABLE,
    OLLAMA_BAD_RESPONSE,
    OLLAMA_TIMEOUT,
    OLLAMA_UNAVAILABLE,
    OLLAMA_UPSTREAM_ERROR,
    SYLLABUS_NOT_INDEXED,
)
from app.generation_mode import current_mode
from app.generation_queue import background_generation, get_generation_queue, is_busy_exception
from app.ollama_coordination import is_starter_job_active
from app.retrieval_diversity import DEFAULT_TOP_K

logger = logging.getLogger(__name__)

#: Names this backend process. See "Restarts" above.
BOOT_ID = uuid.uuid4().hex

CONDITIONS = db_answer_cache.CONDITIONS
CONDITION_LABELS = {
    "base": "Base",
    "rag": "RAG",
    "fineTuned": "Fine-Tuned",
    "fineTunedRag": "Fine-Tuned + RAG",
}
#: Within a batch of more than one question: retrieval approaches first, then
#: the direct ones, so each Ollama server sees prompts of one kind in a row.
BATCHED_CONDITION_ORDER = ("rag", "fineTunedRag", "base", "fineTuned")

DEFAULT_QUIET_SECONDS = 120.0
DEFAULT_STOP_GRACE_SECONDS = 180.0
#: Question by question unless a measurement on the VM justifies batching
#: (`scripts/answer_repeatability_check.py`).
DEFAULT_BATCH_SIZE = 1
#: How often a waiting run looks again (quiet period, stop request).
POLL_SECONDS = 1.0
MAX_CONSECUTIVE_FAILURES = 3

#: An approach the course cannot have at all, as opposed to one that is
#: temporarily unavailable. Not expected, so not counted as missing.
NOT_EXPECTED_REASONS = frozenset({REASON_NO_COURSE_MODEL, REASON_NO_INDEX})

READY = "ready"
MISSING = "missing"
OUTDATED = "outdated"
UNAVAILABLE = "unavailable"
NOT_EXPECTED = "not_expected"

FINETUNED_FAILURE_CODES = frozenset(
    {
        FINETUNED_BAD_RESPONSE,
        FINETUNED_NOT_CONFIGURED,
        FINETUNED_REJECTED,
        FINETUNED_TIMEOUT,
        FINETUNED_UNAVAILABLE,
        FINETUNED_UPSTREAM_ERROR,
        MODEL_NOT_ACTIVATED,
        MODEL_VERSION_NOT_SERVABLE,
    }
)
BASE_FAILURE_CODES = frozenset(
    {OLLAMA_BAD_RESPONSE, OLLAMA_TIMEOUT, OLLAMA_UNAVAILABLE, OLLAMA_UPSTREAM_ERROR}
)


def _float_env(name: str, default: float) -> float:
    try:
        value = float((os.getenv(name) or "").strip() or default)
    except ValueError:
        return default
    return value if value >= 0 else default


def quiet_seconds() -> float:
    return _float_env("CLASSROOM_PREP_QUIET_SECONDS", DEFAULT_QUIET_SECONDS)


def stop_grace_seconds() -> float:
    return _float_env("CLASSROOM_PREP_STOP_GRACE_SECONDS", DEFAULT_STOP_GRACE_SECONDS)


def batch_size() -> int:
    try:
        value = int((os.getenv("CLASSROOM_PREP_BATCH_SIZE") or "").strip() or DEFAULT_BATCH_SIZE)
    except ValueError:
        return DEFAULT_BATCH_SIZE
    return value if value >= 1 else DEFAULT_BATCH_SIZE


class PreparationRefused(Exception):
    """Preparation cannot start; `message` says why in plain words."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# --------------------------------------------------------------------------- #
# The plan: what is selected, what is expected, what already exists
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ConditionPlan:
    expected: bool
    available: bool
    reason: str | None = None
    fingerprint: str | None = None


@dataclass
class Plan:
    course_id: str
    #: Every listable question (eligible, or selected but no longer eligible).
    listing: list[dict[str, Any]]
    #: The selected, eligible questions, in the order they are prepared.
    questions: list[dict[str, Any]]
    conditions: dict[str, ConditionPlan]
    #: (seed id, condition) -> READY | MISSING | OUTDATED | UNAVAILABLE | NOT_EXPECTED
    answers: dict[tuple[str, str], str]
    last_run: dict[str, Any] | None

    def expected_conditions(self) -> list[str]:
        return [c for c in CONDITIONS if self.conditions[c].expected]

    def counts(self) -> dict[str, int]:
        expected = self.expected_conditions()
        states = [self.answers[(q["seedId"], c)] for q in self.questions for c in expected]
        fully = sum(
            1
            for q in self.questions
            if expected and all(self.answers[(q["seedId"], c)] == READY for c in expected)
        )
        ready = states.count(READY)
        return {
            "selectedQuestions": len(self.questions),
            "expectedAnswers": len(states),
            "readyAnswers": ready,
            "missingAnswers": len(states) - ready,
            "outdatedAnswers": states.count(OUTDATED),
            "unavailableAnswers": states.count(UNAVAILABLE),
            "fullyPreparedQuestions": fully,
        }


def _read_plan_rows(course_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    with db_connection() as connection:
        listing = db_classroom_questions.list_course_questions(connection, course_id)
        selected = db_classroom_questions.list_selected_questions(connection, course_id)
        last_run = db_classroom_preparation.get_run(connection, course_id)
    return listing, selected, last_run


def _read_entry_keys(course_id: str, keys: list[str]) -> list[dict[str, str]]:
    with db_connection() as connection:
        return db_answer_cache.list_entry_keys(connection, course_id, keys)


async def build_plan(course_id: str) -> Plan:
    """What preparing this course means right now. Raises if the database cannot say."""
    # The database first: without it there is nothing to plan, and no reason
    # to ask Ollama anything.
    listing, selected, last_run = await run_in_threadpool(_read_plan_rows, course_id)
    questions = [{**item, "key": answer_cache.question_key(item["question"])} for item in selected]
    rows = await run_in_threadpool(_read_entry_keys, course_id, [q["key"] for q in questions])

    conditions: dict[str, ConditionPlan] = {}
    for condition in CONDITIONS:
        try:
            result = await build_identity(condition, course_id, top_k=DEFAULT_TOP_K)
            identity, reason = result.identity, result.reason
        except Exception:  # noqa: BLE001 - an unknown identity is an unavailable approach
            logger.warning("classroom preparation: identity lookup failed for %s/%s", course_id, condition, exc_info=True)
            identity, reason = None, "identity_unavailable"
        if identity is not None:
            conditions[condition] = ConditionPlan(True, True, None, identity.fingerprint)
        else:
            conditions[condition] = ConditionPlan(reason not in NOT_EXPECTED_REASONS, False, reason)

    existing: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        existing.setdefault((row["condition"], row["questionKey"]), set()).add(row["fingerprint"])

    answers: dict[tuple[str, str], str] = {}
    for question in questions:
        for condition in CONDITIONS:
            plan = conditions[condition]
            prints = existing.get((condition, question["key"]), set())
            if not plan.expected:
                state = NOT_EXPECTED
            elif plan.fingerprint is not None and plan.fingerprint in prints:
                state = READY
            elif not plan.available:
                state = UNAVAILABLE
            elif prints:
                state = OUTDATED
            else:
                state = MISSING
            answers[(question["seedId"], condition)] = state
    return Plan(course_id, listing, questions, conditions, answers, last_run)


def ordered_items(plan: Plan, size: int) -> list[tuple[dict[str, Any], str]]:
    """Every (question, condition) in the order it would be prepared.

    A batch size of 1 is question by question, in the card order. A larger
    one takes that many questions at a time and groups each approach within
    the batch.
    """
    order = CONDITIONS if size <= 1 else BATCHED_CONDITION_ORDER
    items: list[tuple[dict[str, Any], str]] = []
    step = max(1, size)
    for start in range(0, len(plan.questions), step):
        batch = plan.questions[start : start + step]
        if size <= 1:
            items.extend((question, condition) for question in batch for condition in order)
        else:
            items.extend((question, condition) for condition in order for question in batch)
    return items


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


@dataclass
class Run:
    course_id: str
    started_by: str | None
    started_at: datetime
    task: "asyncio.Task[None] | None" = None
    stop_requested_at: float | None = None
    #: {"seedId", "condition"} while one answer is being generated.
    current: dict[str, str] | None = None
    #: "quiet" or "starter" while the run is holding off.
    waiting: str | None = None
    generated: int = 0
    failed: int = 0
    #: (question key, condition) already tried in this run; never retried in it.
    attempted: set[tuple[str, str]] = field(default_factory=set)
    #: condition -> plain reason, for approaches given up on for this run.
    skipped_conditions: dict[str, str] = field(default_factory=dict)
    last_problem: str | None = None
    finished: bool = False

    @property
    def stopping(self) -> bool:
        return self.stop_requested_at is not None


_active: Run | None = None


def active_run() -> Run | None:
    _reap()
    return _active


def reset_for_tests() -> None:
    global _active
    _active = None


def _reap() -> None:
    """Forget a finished run; cancel one that has outlived its stop request."""
    global _active
    run = _active
    if run is None:
        return
    if run.task is None or run.task.done():
        _active = None
        return
    if run.stop_requested_at is not None and (
        time.monotonic() - run.stop_requested_at > stop_grace_seconds() + POLL_SECONDS * 5
    ):
        # The worker's own safety net should have fired by now. It did not, so
        # the worker is stuck somewhere other than a generation.
        logger.error("classroom preparation: run for %s did not stop; cancelling it", run.course_id)
        answer_flights.cancel_background(include_started=True)
        run.task.cancel()
        _finish_sync(run, "stopped", "stop_forced", FORCED_STOP_MESSAGE)
        _active = None


FORCED_STOP_MESSAGE = (
    "Preparation was stopped while an answer was still being generated, so that one "
    "answer was not saved. Your completed answers are saved."
)


def _write_finish(run: Run, state: str, reason: str | None, message: str | None) -> None:
    with db_connection() as connection:
        db_classroom_preparation.finish_run(
            connection,
            run.course_id,
            boot_id=BOOT_ID,
            state=state,
            reason=reason,
            message=message,
            generated=run.generated,
            failed=run.failed,
        )


def _finish_sync(run: Run, state: str, reason: str | None, message: str | None) -> None:
    if run.finished:
        return
    run.finished = True
    run.current = None
    run.waiting = None
    try:
        _write_finish(run, state, reason, message)
    except Exception:  # noqa: BLE001 - the answers are what matter; the row is a courtesy
        logger.warning("classroom preparation: could not record how the run for %s ended", run.course_id, exc_info=True)
    logger.info(
        "classroom preparation: course=%s ended state=%s reason=%s generated=%d failed=%d",
        run.course_id, state, reason or "-", run.generated, run.failed,
    )


async def _finish(run: Run, state: str, reason: str | None = None, message: str | None = None) -> None:
    await run_in_threadpool(_finish_sync, run, state, reason, message)


def _write_progress(run: Run) -> None:
    with db_connection() as connection:
        db_classroom_preparation.record_progress(
            connection, run.course_id, boot_id=BOOT_ID, generated=run.generated, failed=run.failed
        )


async def _record_progress(run: Run) -> None:
    try:
        await run_in_threadpool(_write_progress, run)
    except Exception:  # noqa: BLE001 - best effort
        logger.warning("classroom preparation: could not record progress for %s", run.course_id, exc_info=True)


def _classify_failure(exc: BaseException) -> tuple[str, str]:
    """`(scope, plain message)`: scope is fineTuned, base, question or other."""
    code = getattr(exc, "code", None)
    if code in FINETUNED_FAILURE_CODES:
        return "fineTuned", "The course's fine-tuned model did not respond, so those answers could not be prepared."
    if code == SYLLABUS_NOT_INDEXED:
        return "question", "No matching syllabus text was found for a question, so its RAG answers could not be prepared."
    if code in BASE_FAILURE_CODES:
        return "base", "The base model did not respond."
    if is_busy_exception(exc):
        return "other", "The server was too busy to prepare an answer."
    if isinstance(exc, HTTPException):
        return "other", "An answer could not be generated."
    return "other", "An unexpected problem stopped an answer from being prepared."


def _next_item(plan: Plan, run: Run) -> tuple[dict[str, Any], str] | None:
    for question, condition in ordered_items(plan, batch_size()):
        if plan.answers[(question["seedId"], condition)] not in (MISSING, OUTDATED):
            continue
        if condition in run.skipped_conditions or (question["key"], condition) in run.attempted:
            continue
        return question, condition
    return None


async def _wait_until_quiet(run: Run) -> bool:
    """Hold off while students are using the site. False if a stop was requested."""
    while True:
        if run.stopping:
            return False
        if is_starter_job_active():
            run.waiting = "starter"
        else:
            idle = get_generation_queue().seconds_since_interactive_activity()
            if idle is None or idle >= quiet_seconds():
                run.waiting = None
                return True
            run.waiting = "quiet"
        await asyncio.sleep(POLL_SECONDS)


async def _prepare_one(course_id: str, condition: str, question: str) -> answer_cache.Served:
    with background_generation():
        return await answer_cache.serve_detailed(
            condition, course_id, question, top_k=DEFAULT_TOP_K, source="prepared"
        )


async def _generate_item(run: Run, question: dict[str, Any], condition: str) -> Any:
    """Prepare one answer; returns a `Served`, or the exception it ended with."""
    child = asyncio.create_task(_prepare_one(run.course_id, condition, question["question"]))
    try:
        while not child.done():
            await asyncio.wait({child}, timeout=POLL_SECONDS)
            if child.done() or run.stop_requested_at is None:
                continue
            if not answer_flights.background_generation_started():
                # Not at a model yet: drop it now rather than wait for a turn.
                answer_flights.cancel_background()
                child.cancel()
            elif time.monotonic() - run.stop_requested_at > stop_grace_seconds():
                # Generating for longer than every timeout allows: last resort.
                logger.error("classroom preparation: forcing a stalled generation to stop (%s)", condition)
                answer_flights.cancel_background(include_started=True)
                child.cancel()
    except asyncio.CancelledError:
        answer_flights.cancel_background(include_started=True)
        child.cancel()
        raise
    return (await asyncio.gather(child, return_exceptions=True))[0]


async def _work(run: Run) -> None:
    consecutive_failures = 0
    try:
        while True:
            if run.stopping:
                await _finish(run, "stopped", "stopped_by_staff")
                return
            if not answer_cache.cache_enabled():
                await _finish(run, "failed", "disabled", "Classroom answers were switched off on this server.")
                return
            if current_mode().is_gpu:
                await _finish(
                    run, "stopped", "gpu_mode",
                    "The site switched to the GPU server, so preparation stopped. Your completed answers are saved.",
                )
                return

            plan = await build_plan(run.course_id)
            item = _next_item(plan, run)
            if item is None:
                await _finish(run, "completed", None, run.last_problem)
                return
            if not await _wait_until_quiet(run):
                continue

            question, condition = item
            run.current = {"seedId": question["seedId"], "condition": condition}
            run.attempted.add((question["key"], condition))
            result = await _generate_item(run, question, condition)
            run.current = None

            if isinstance(result, asyncio.CancelledError):
                # Dropped by a stop request: before it reached a model (the
                # loop then ends as an ordinary stop), or by the last resort.
                if (
                    run.stop_requested_at is not None
                    and time.monotonic() - run.stop_requested_at > stop_grace_seconds()
                ):
                    await _finish(run, "stopped", "stop_forced", FORCED_STOP_MESSAGE)
                    return
                continue
            if isinstance(result, BaseException):
                scope, message = _classify_failure(result)
                run.failed += 1
                run.last_problem = message
                logger.warning(
                    "classroom preparation: course=%s condition=%s failed (%s)",
                    run.course_id, condition, getattr(result, "code", None) or result.__class__.__name__,
                )
                if scope == "fineTuned":
                    for name in FINE_TUNED_CONDITIONS:
                        run.skipped_conditions[name] = message
                if scope != "question":
                    consecutive_failures += 1
            elif result.outcome == "generated":
                # Answered but not stored: the model or syllabus changed while
                # it was generating, or the question was deselected meanwhile.
                run.failed += 1
                run.last_problem = (
                    "An answer was generated but not saved, because the syllabus, a model "
                    "or the selection changed while it was being prepared."
                )
                consecutive_failures += 1
            else:
                if result.outcome == "stored":
                    run.generated += 1
                consecutive_failures = 0

            await _record_progress(run)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                await _finish(run, "failed", "repeated_failures", run.last_problem)
                return
    except asyncio.CancelledError:
        _finish_sync(run, "stopped", "cancelled", "Preparation was interrupted. Your completed answers are saved.")
        raise
    except Exception:  # noqa: BLE001 - a run must always end in a recorded state
        logger.exception("classroom preparation: run for %s failed", run.course_id)
        await _finish(
            run, "failed", "internal_error",
            "Preparation stopped because of an unexpected problem. Your completed answers are saved.",
        )


def _begin(course_id: str, started_by: str | None) -> None:
    with db_connection() as connection:
        db_classroom_preparation.begin_run(connection, course_id, boot_id=BOOT_ID, started_by=started_by)


async def start(course_id: str, *, started_by: str | None = None) -> Run:
    """Start (or resume) preparing a course. Idempotent while it is already running."""
    global _active
    _reap()
    if _active is not None and _active.course_id == course_id:
        return _active
    if not answer_cache.cache_enabled():
        raise PreparationRefused("disabled", "Classroom answers are switched off on this server.")
    if current_mode().is_gpu:
        raise PreparationRefused(
            "gpu_mode",
            "The site is using the GPU server right now. Answers are prepared only in normal (VM) mode.",
        )
    if _active is not None:
        raise PreparationRefused(
            "another_course",
            "Another course's answers are being prepared right now. Only one course is prepared at a time.",
        )
    plan = await build_plan(course_id)
    if not plan.questions:
        raise PreparationRefused("nothing_selected", "Select at least one question to prepare.")
    # A second request may have started a run while this one was reading.
    if _active is not None:
        if _active.course_id == course_id:
            return _active
        raise PreparationRefused(
            "another_course",
            "Another course's answers are being prepared right now. Only one course is prepared at a time.",
        )
    run = Run(course_id=course_id, started_by=started_by, started_at=datetime.now(timezone.utc))
    _active = run
    try:
        await run_in_threadpool(_begin, course_id, started_by)
    except Exception:
        if _active is run:
            _active = None
        raise
    run.task = asyncio.create_task(_work(run))
    logger.info("classroom preparation: course=%s started", course_id)
    return run


def _write_stopping(course_id: str) -> None:
    with db_connection() as connection:
        db_classroom_preparation.mark_stopping(connection, course_id, boot_id=BOOT_ID)


async def request_stop(course_id: str) -> bool:
    """Ask the course's run to stop gracefully. False when it is not running."""
    run = active_run()
    if run is None or run.course_id != course_id:
        return False
    if run.stop_requested_at is None:
        run.stop_requested_at = time.monotonic()
        if not answer_flights.background_generation_started():
            answer_flights.cancel_background()
        try:
            await run_in_threadpool(_write_stopping, course_id)
        except Exception:  # noqa: BLE001 - best effort
            logger.warning("classroom preparation: could not record the stop request for %s", course_id, exc_info=True)
        logger.info("classroom preparation: course=%s stop requested", course_id)
    return True


async def shutdown() -> None:
    """Backend shutdown: end the run so its row does not claim to be running."""
    run = _active
    if run is None or run.task is None or run.task.done():
        return
    run.task.cancel()
    try:
        await run.task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001 - shutdown must not fail
        pass


# --------------------------------------------------------------------------- #
# Status, in plain words
# --------------------------------------------------------------------------- #

CONDITION_NOTICES = {
    REASON_NO_COURSE_MODEL: "This course has no active fine-tuned model, so only Base and RAG answers are prepared.",
    REASON_NO_INDEX: "This course's syllabus is not indexed yet, so RAG answers cannot be prepared.",
    REASON_FINETUNED_SERVICE: "The course's fine-tuned model is unavailable right now, so those answers cannot be prepared.",
    REASON_FINETUNED_MODEL: "The course's fine-tuned model is unavailable right now, so those answers cannot be prepared.",
    REASON_BASE_MODEL: "The base model is unavailable right now, so Base and RAG answers cannot be prepared.",
    REASON_EMBEDDING_MODEL: "Syllabus search is unavailable right now, so RAG answers cannot be prepared.",
}


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _state(plan: Plan, counts: dict[str, int], run: Run | None) -> tuple[str, bool]:
    """`(state, interrupted)` for the course."""
    if run is not None:
        return ("stopping" if run.stopping else "preparing"), False
    if counts["selectedQuestions"] == 0:
        return "not_started", False
    if counts["expectedAnswers"] > 0 and counts["missingAnswers"] == 0:
        return "ready", False
    last = plan.last_run
    if last is not None:
        if last["state"] in db_classroom_preparation.ACTIVE_STATES:
            # Marked running, but no run exists in this process: it belonged
            # to a backend that has since restarted.
            return "stopped", True
        if last["state"] in ("failed", "stopped"):
            return last["state"], False
    return ("partially_ready" if counts["readyAnswers"] > 0 else "not_started"), False


def _headline(state: str, counts: dict[str, int], run: Run | None, plan: Plan, interrupted: bool) -> str:
    selected, full = counts["selectedQuestions"], counts["fullyPreparedQuestions"]
    if state == "preparing":
        if run is not None and run.waiting == "quiet":
            return "Waiting for students to finish. Preparation continues on its own when the site is quiet."
        if run is not None and run.waiting == "starter":
            return "Waiting for example generation to finish. Preparation continues on its own afterwards."
        if run is not None and run.current is not None:
            return f"Preparing classroom answers. Currently generating {CONDITION_LABELS[run.current['condition']]}."
        return "Preparing classroom answers."
    if state == "stopping":
        return "Stopping after the current answer. Your completed answers are saved."
    if state == "ready":
        return f"All {_plural(selected, 'question')} are ready for class." if selected != 1 else "The selected question is ready for class."
    if state == "stopped":
        if interrupted:
            return "Preparation was interrupted when the server restarted. Your completed answers are saved and students can use them."
        last_message = (plan.last_run or {}).get("message")
        return last_message or "Preparation stopped. Your completed answers are saved and students can use them now."
    if state == "failed":
        problem = (plan.last_run or {}).get("message") or "An answer could not be prepared."
        return f"Preparation stopped because of a problem. {problem} Your completed answers are saved."
    if state == "partially_ready":
        return (
            f"{full} of {_plural(selected, 'question')} fully ready. "
            "Students can already use every prepared answer."
        )
    if selected == 0:
        return "Choose the questions you want ready for class, then prepare their answers."
    return f"{_plural(selected, 'question')} selected. No answers are prepared yet."


async def course_status(course_id: str) -> dict[str, Any]:
    """Everything the preparation page shows for one course."""
    plan = await build_plan(course_id)
    counts = plan.counts()
    run = active_run()
    other_course_running = run is not None and run.course_id != course_id
    own_run = run if run is not None and run.course_id == course_id else None
    state, interrupted = _state(plan, counts, own_run)
    enabled = answer_cache.cache_enabled()
    on_gpu = current_mode().is_gpu

    preparable = sum(
        1
        for question in plan.questions
        for condition in CONDITIONS
        if plan.answers[(question["seedId"], condition)] in (MISSING, OUTDATED)
    )
    blocked: str | None = None
    if own_run is not None:
        blocked = "running"
    elif not enabled:
        blocked = "disabled"
    elif on_gpu:
        blocked = "gpu_mode"
    elif other_course_running:
        blocked = "another_course"
    elif counts["selectedQuestions"] == 0:
        blocked = "nothing_selected"
    elif preparable == 0:
        blocked = "nothing_to_prepare"

    notices: list[str] = []
    if not enabled:
        notices.append("Classroom answers are switched off on this server, so prepared answers are not used yet.")
    if on_gpu:
        notices.append("The site is using the GPU server right now. Answers are prepared and used only in normal (VM) mode.")
    if other_course_running:
        notices.append("Another course's answers are being prepared right now. Only one course is prepared at a time.")
    for reason in dict.fromkeys(plan.conditions[c].reason for c in CONDITIONS):
        if reason in CONDITION_NOTICES and CONDITION_NOTICES[reason] not in notices:
            notices.append(CONDITION_NOTICES[reason])
    if own_run is not None:
        for message in dict.fromkeys(own_run.skipped_conditions.values()):
            if message not in notices:
                notices.append(message)
    if counts["outdatedAnswers"] > 0:
        notices.append(
            f"{_plural(counts['outdatedAnswers'], 'answer')} out of date because the syllabus or a model changed. "
            "Prepare the remaining answers to refresh them."
        )
    ineligible = sum(1 for item in plan.listing if item["selected"] and not item["eligible"])
    if ineligible:
        notices.append(
            f"{_plural(ineligible, 'selected question')} can no longer be prepared (rejected, or a student's "
            "question that is not approved) and will be skipped."
        )
    if state == "partially_ready" and plan.last_run and plan.last_run.get("message"):
        notices.append(plan.last_run["message"])

    selected_ids = {question["seedId"] for question in plan.questions}
    questions = []
    for item in plan.listing:
        answers = (
            {condition: plan.answers[(item["seedId"], condition)] for condition in CONDITIONS}
            if item["seedId"] in selected_ids
            else None
        )
        questions.append({**item, "answers": answers})

    last = plan.last_run
    return {
        "courseId": course_id,
        "enabled": enabled,
        "state": state,
        "running": own_run is not None,
        "stopping": own_run is not None and own_run.stopping,
        "waiting": own_run.waiting if own_run is not None else None,
        "current": dict(own_run.current) if own_run is not None and own_run.current else None,
        "canStart": blocked is None,
        "startBlockedReason": blocked,
        "counts": {**counts, "preparableAnswers": preparable},
        "conditions": {
            condition: {
                "label": CONDITION_LABELS[condition],
                "expected": plan.conditions[condition].expected,
                "available": plan.conditions[condition].available,
                "reason": plan.conditions[condition].reason,
            }
            for condition in CONDITIONS
        },
        "message": _headline(state, counts, own_run, plan, interrupted),
        "notices": notices,
        "lastRun": (
            {
                "state": "interrupted" if interrupted else last["state"],
                "startedAt": last["startedAt"],
                "finishedAt": last["finishedAt"],
                "reason": last["reason"],
                "generated": own_run.generated if own_run is not None else last["generated"],
                "failed": own_run.failed if own_run is not None else last["failed"],
            }
            if last is not None
            else None
        ),
        "questions": questions,
    }
