"""Process-local coordination for Ollama generation and starter seed jobs.

Two complementary guards:

1. ``ollama_generation_slot`` — a turn in the process-wide generation queue
   (``app.generation_queue``), which bounds every generation this backend
   asks for: Base, RAG, the fine-tuned service's calls, and starter
   completions, which queue behind classroom requests. Embeddings are not
   covered.
2. Global starter-job slot — only one starter seed job (automatic, manual, or
   top-up) may run on this CPU VM at a time.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Literal

from fastapi import HTTPException

from app.generation_queue import (
    Priority,
    SlotInfo,
    generation_slot,
    reset_generation_queue_for_tests,
)

StarterOperation = Literal["automatic", "manual", "top_up"]

_starter_state_lock = asyncio.Lock()
_starter_job_active = False
_starter_job_course_id: str | None = None
_starter_job_operation: StarterOperation | None = None
_starter_job_started_at: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@asynccontextmanager
async def ollama_generation_slot(
    label: str = "starter",
    *,
    priority: Priority = "background",
) -> AsyncIterator[SlotInfo]:
    """A turn in the shared generation queue for one local Ollama generation.

    Defaults are the starter pipeline's: background work that waits behind
    every classroom request. Classroom calls pass their condition and
    `priority="interactive"`.
    """
    async with generation_slot(label, priority=priority) as info:
        yield info


def starter_generation_in_progress_detail(
    *,
    course_id: str | None = None,
) -> dict[str, Any]:
    active_course = course_id if course_id is not None else _starter_job_course_id
    detail: dict[str, Any] = {
        "code": "generation_in_progress",
        "message": (
            "Starter seed generation is already running. "
            "Wait for it to finish before starting another job."
        ),
    }
    if active_course:
        detail["courseId"] = active_course
    return detail


def raise_starter_generation_in_progress(
    *,
    course_id: str | None = None,
) -> None:
    raise HTTPException(
        status_code=409,
        detail=starter_generation_in_progress_detail(course_id=course_id),
    )


async def try_begin_starter_job(
    course_id: str,
    operation: StarterOperation,
) -> bool:
    """Claim the global starter-job slot. Returns False if another job is active."""
    global _starter_job_active, _starter_job_course_id
    global _starter_job_operation, _starter_job_started_at

    async with _starter_state_lock:
        if _starter_job_active:
            return False
        _starter_job_active = True
        _starter_job_course_id = course_id
        _starter_job_operation = operation
        _starter_job_started_at = _utc_now()
        return True


async def end_starter_job(*, course_id: str | None = None) -> None:
    """Release the global starter-job slot.

    When ``course_id`` is provided, only release if that course still owns the slot
    (avoids clearing a newer job after races / cancelled tasks).
    """
    global _starter_job_active, _starter_job_course_id
    global _starter_job_operation, _starter_job_started_at

    async with _starter_state_lock:
        if course_id is not None and _starter_job_course_id != course_id:
            return
        _starter_job_active = False
        _starter_job_course_id = None
        _starter_job_operation = None
        _starter_job_started_at = None


def get_starter_job_status() -> dict[str, Any]:
    """Read-only snapshot for the status endpoint (no secrets / prompts)."""
    return {
        "active": _starter_job_active,
        "courseId": _starter_job_course_id,
        "operation": _starter_job_operation,
        "startedAt": _starter_job_started_at,
    }


def is_starter_job_active() -> bool:
    return _starter_job_active


def get_active_starter_job_course_id() -> str | None:
    return _starter_job_course_id


@asynccontextmanager
async def starter_job_slot(
    course_id: str,
    operation: StarterOperation,
) -> AsyncIterator[None]:
    """Acquire the global starter slot or raise HTTP 409; always release in finally."""
    acquired = await try_begin_starter_job(course_id, operation)
    if not acquired:
        raise_starter_generation_in_progress(
            course_id=get_active_starter_job_course_id(),
        )
    try:
        yield
    finally:
        await end_starter_job(course_id=course_id)


def reset_ollama_coordination_for_tests() -> None:
    """Reset process-local coordination state (tests only)."""
    global _starter_job_active, _starter_job_course_id
    global _starter_job_operation, _starter_job_started_at

    _starter_job_active = False
    _starter_job_course_id = None
    _starter_job_operation = None
    _starter_job_started_at = None

    # Recreate the lock and the queue so tests never leave them held across
    # cases. Safe only when no tasks are awaiting them (test teardown).
    global _starter_state_lock
    _starter_state_lock = asyncio.Lock()
    reset_generation_queue_for_tests()
