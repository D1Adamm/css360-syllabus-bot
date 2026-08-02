"""Background starter-seed generation jobs after syllabus indexing."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from app.course_id import assert_valid_course_id
from app.firebase_metadata import (
    best_effort_patch_starter_seed_generation,
    read_starter_seed_generation_status,
)
from app.ollama_coordination import (
    end_starter_job,
    get_active_starter_job_course_id,
    is_starter_job_active,
    reset_ollama_coordination_for_tests,
    try_begin_starter_job,
)
from app.seed_generation import generate_starter_seeds_for_course

logger = logging.getLogger(__name__)

DEFAULT_STARTER_AUTO_GENERATE_COUNT = 50
ACTIVE_STARTER_STATUSES = frozenset({"queued", "generating"})

# Process-local guard: course ids with a scheduled or running auto job.
_active_starter_jobs: set[str] = set()


def _env_flag_enabled(name: str, *, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def is_auto_starter_seed_generation_enabled() -> bool:
    """Whether syllabus upload should auto-queue starter seed generation.

    Controlled by AUTO_STARTER_SEED_GENERATION (default true for backward
    compatibility). Set AUTO_STARTER_SEED_GENERATION=false to index syllabi
    without starting a long background seed job. Manual
    POST /seeds/generate-starter still works.
    """
    return _env_flag_enabled("AUTO_STARTER_SEED_GENERATION", default=True)


def get_starter_auto_generate_count() -> int:
    raw = os.getenv("STARTER_AUTO_GENERATE_COUNT")
    if raw is None:
        return DEFAULT_STARTER_AUTO_GENERATE_COUNT
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_STARTER_AUTO_GENERATE_COUNT
    return max(0, min(value, 50))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_course_starter_job_active(course_id: str) -> bool:
    return course_id in _active_starter_jobs


def clear_active_starter_jobs_for_tests() -> None:
    """Reset process-local job set and global coordination (tests only)."""
    _active_starter_jobs.clear()
    reset_ollama_coordination_for_tests()


async def _durable_status_is_active(course_id: str) -> bool:
    status = await read_starter_seed_generation_status(course_id)
    return status in ACTIVE_STARTER_STATUSES


def _resolve_terminal_status(
    *,
    target_count: int,
    final_count: int,
    saved_count: int,
) -> str:
    if target_count <= 0:
        return "ready"
    if final_count >= target_count and saved_count >= target_count:
        return "ready"
    if final_count > 0 or saved_count > 0:
        return "partial"
    return "failed"


async def try_queue_starter_seed_generation(course_id: str) -> dict[str, Any]:
    """Mark a course for auto starter generation if no job is already active.

    Returns a small status dict for the upload response. Does not run generation.
    """
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError:
        return {"queued": False, "status": "not_started", "reason": "invalid_course_id"}

    auto_enabled = is_auto_starter_seed_generation_enabled()
    target_count = get_starter_auto_generate_count()
    if not auto_enabled or target_count <= 0:
        return {
            "queued": False,
            "status": "not_started",
            "reason": "auto_generate_disabled",
            "targetCount": 0 if not auto_enabled else target_count,
        }

    if safe_course_id in _active_starter_jobs:
        return {
            "queued": False,
            "status": "generating",
            "reason": "job_already_active",
            "targetCount": target_count,
        }

    if is_starter_job_active():
        return {
            "queued": False,
            "status": "generating",
            "reason": "global_job_active",
            "targetCount": target_count,
            "activeCourseId": get_active_starter_job_course_id(),
        }

    if await _durable_status_is_active(safe_course_id):
        return {
            "queued": False,
            "status": "queued",
            "reason": "durable_status_active",
            "targetCount": target_count,
        }

    began = await try_begin_starter_job(safe_course_id, "automatic")
    if not began:
        return {
            "queued": False,
            "status": "generating",
            "reason": "global_job_active",
            "targetCount": target_count,
            "activeCourseId": get_active_starter_job_course_id(),
        }

    _active_starter_jobs.add(safe_course_id)
    queued_payload = {
        "status": "queued",
        "targetCount": target_count,
        "finalCount": 0,
        "savedCount": 0,
        "failedToSaveCount": 0,
        "error": None,
        "startedAt": None,
        "completedAt": None,
    }
    await best_effort_patch_starter_seed_generation(safe_course_id, queued_payload)

    return {
        "queued": True,
        "status": "queued",
        "targetCount": target_count,
    }


async def run_auto_starter_seed_generation(course_id: str) -> None:
    """Background worker: process a previously queued starter-seed job.

    Expected entry status is ``queued``. Does not no-op solely because status is
    queued — that is the normal state for the job about to run.
    """
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError:
        return

    target_count = get_starter_auto_generate_count()
    if not is_auto_starter_seed_generation_enabled() or target_count <= 0:
        _active_starter_jobs.discard(safe_course_id)
        await end_starter_job(course_id=safe_course_id)
        return

    # Normally claimed in try_queue; claim here if a caller invoked the worker directly.
    if get_active_starter_job_course_id() != safe_course_id:
        began = await try_begin_starter_job(safe_course_id, "automatic")
        if not began:
            logger.warning(
                "Skipping automatic starter generation for %s; another job is active",
                safe_course_id,
            )
            _active_starter_jobs.discard(safe_course_id)
            return

    started_at = _utc_now()
    await best_effort_patch_starter_seed_generation(
        safe_course_id,
        {
            "status": "generating",
            "targetCount": target_count,
            "startedAt": started_at,
            "error": None,
        },
    )

    try:
        result = await generate_starter_seeds_for_course(
            course_id=safe_course_id,
            target_count=target_count,
            save=True,
        )
        progress = result.get("progress") or {}
        persistence = result.get("persistence") or {}
        final_count = int(progress.get("finalCount", 0))
        saved_count = int(persistence.get("savedCount", 0))
        failed_to_save = int(persistence.get("failedToSaveCount", 0))
        terminal = _resolve_terminal_status(
            target_count=target_count,
            final_count=final_count,
            saved_count=saved_count,
        )
        await best_effort_patch_starter_seed_generation(
            safe_course_id,
            {
                "status": terminal,
                "targetCount": target_count,
                "finalCount": final_count,
                "savedCount": saved_count,
                "failedToSaveCount": failed_to_save,
                "error": None,
                "startedAt": started_at,
                "completedAt": _utc_now(),
            },
        )
    except Exception as exc:  # noqa: BLE001 - background job must not crash the server
        logger.exception(
            "Automatic starter seed generation failed for course %s",
            safe_course_id,
        )
        if isinstance(exc, HTTPException):
            error_text = str(exc.detail)
        else:
            error_text = str(exc)
        await best_effort_patch_starter_seed_generation(
            safe_course_id,
            {
                "status": "failed",
                "targetCount": target_count,
                "finalCount": 0,
                "savedCount": 0,
                "failedToSaveCount": 0,
                "error": error_text[:500],
                "startedAt": started_at,
                "completedAt": _utc_now(),
            },
        )
    finally:
        _active_starter_jobs.discard(safe_course_id)
        await end_starter_job(course_id=safe_course_id)
