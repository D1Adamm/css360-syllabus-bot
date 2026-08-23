"""The stored starter-seed generation record, in PostgreSQL.

`starter_seed_generation` is one row per course describing the automatic
starter run: what it was aiming for, what the course ended up holding, when it
started and finished, and why it stopped short if it did. The professor-facing
course page reads it, so it has to describe the course rather than any single
run.

That distinction is the reason `reconcile_starter_seed_generation` exists and
is worth keeping through the migration. The record used to be written from one
run's own tally, and a later top-up added seeds by a different route without
touching it — which is how a course holding fifty seeds went on reporting
"partial, 9 of 50" indefinitely. Reconciliation derives the record from the
stored seed count instead, so any route that adds seeds leaves it true.

Everything here is best-effort in the same way it was before: a status write
that fails must not fail the request whose seeds are already saved. The failure
is logged rather than swallowed silently, and the caller is told by a `None`
return that nothing was written, so it can fall back to its own numbers.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from app.course_id import assert_valid_course_id
from app.db import db_connection, translate_db_errors
from app.db_courses import (
    course_exists,
    get_starter_seed_generation,
    upsert_starter_seed_generation,
)
from app.seed_persistence import count_course_seed_examples

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def read_starter_seed_generation(course_id: str) -> dict[str, Any] | None:
    """The stored record, or None when there is none or it could not be read."""
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError:
        return None

    try:
        with translate_db_errors("reading starter seed generation state"):
            with db_connection() as connection:
                return get_starter_seed_generation(connection, safe_course_id)
    except (HTTPException, ValueError):
        return None


async def read_starter_seed_generation_status(course_id: str) -> str | None:
    """The current status, or None when it is missing or unreadable."""
    block = await read_starter_seed_generation(course_id)
    if not block:
        return None
    status = block.get("status")
    return status if isinstance(status, str) else None


async def best_effort_patch_starter_seed_generation(
    course_id: str,
    updates: dict[str, Any],
) -> bool:
    """Merge starter-status fields without failing the caller.

    Returns whether the write landed. A course row that does not exist yet is
    not an error worth failing a generation run over — the foreign key would
    refuse the write anyway, and the run's seeds are the durable part.
    """
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError:
        return False

    try:
        with translate_db_errors("updating starter seed generation state"):
            with db_connection() as connection:
                if not course_exists(connection, safe_course_id):
                    logger.warning(
                        "Not recording starter status for unknown course %s",
                        safe_course_id,
                    )
                    return False
                upsert_starter_seed_generation(connection, safe_course_id, updates)
        return True
    except (HTTPException, ValueError):
        logger.warning(
            "Could not record starter seed generation status for %s",
            safe_course_id,
            exc_info=True,
        )
        return False


def resolve_reconciled_starter_status(
    *,
    target_count: int,
    actual_count: int,
) -> str:
    """Terminal status implied by what the course actually holds.

    Same vocabulary the job has always written, judged against the stored seed
    count rather than one run's tally:

      - at or above target -> ready
      - some seeds, below target -> partial (the UI already reads this as a
        usable course; see `starterSeedGeneration.ts`)
      - nothing stored -> failed
    """
    if target_count <= 0:
        return "ready"
    if actual_count >= target_count:
        return "ready"
    if actual_count > 0:
        return "partial"
    return "failed"


async def resolve_target_count(course_id: str, fallback: int) -> int:
    """Prefer the target already recorded for the course.

    A top-up asked for 50 against a record that already says 50; re-deriving the
    target from whatever the caller happened to pass would let one odd request
    rewrite what the course was aiming for.
    """
    block = await read_starter_seed_generation(course_id)
    if block:
        stored = block.get("targetCount")
        try:
            stored_int = int(stored)
        except (TypeError, ValueError):
            stored_int = 0
        if stored_int > 0:
            return stored_int
    return max(0, int(fallback))


async def reconcile_starter_seed_generation(
    course_id: str,
    *,
    target_count: int,
    started_at: str | None = None,
    completed_at: str | None = None,
    failed_to_save_count: int | None = None,
    error: str | None = None,
    force_status: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Rewrite the record from the course's real seed count.

    Returns the patch that was applied, or None when the count could not be
    read — in which case nothing is written and the caller keeps its own
    numbers.

    `force_status` preserves failure semantics: a run that raised still records
    `failed` and its error, but its counts are reconciled rather than reported
    as zero, because "this failed" and "this course has no seeds" are different
    facts and the record was previously asserting both.
    """
    actual_count = await count_course_seed_examples(course_id)
    if actual_count is None:
        return None

    effective_target = await resolve_target_count(course_id, target_count)
    status = force_status or resolve_reconciled_starter_status(
        target_count=effective_target,
        actual_count=actual_count,
    )

    patch: dict[str, Any] = {
        "status": status,
        "targetCount": effective_target,
        # Both describe the course as it now stands. `savedCount` is what the
        # professor-facing UI shows as the example count, and `finalCount` is
        # its fallback, so a course with 50 stored seeds must report 50 in both
        # or the two disagree about the same course.
        "finalCount": actual_count,
        "savedCount": actual_count,
        # The completion this record now describes is the reconciliation that
        # produced it, not the original run's.
        "completedAt": completed_at or _utc_now(),
    }

    if started_at is not None:
        patch["startedAt"] = started_at
    if failed_to_save_count is not None:
        patch["failedToSaveCount"] = int(failed_to_save_count)
    patch["error"] = error
    if extra:
        patch.update(extra)

    await best_effort_patch_starter_seed_generation(course_id, patch)
    return patch
