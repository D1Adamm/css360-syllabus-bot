"""Firebase helpers for course metadata updates (starter seed job status)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import HTTPException

from app.course_id import assert_valid_course_id
from app.firebase_seeds import (
    FIREBASE_REQUEST_TIMEOUT_SECONDS,
    FirebaseConfigurationError,
    _request_url,
    fetch_course_seed_examples,
)


def course_metadata_path(course_id: str) -> str:
    safe_course_id = assert_valid_course_id(course_id)
    return f"courses/{safe_course_id}/metadata"


def starter_seed_generation_path(course_id: str) -> str:
    return f"{course_metadata_path(course_id)}/starterSeedGeneration"


async def fetch_course_metadata(course_id: str) -> dict[str, Any] | None:
    assert_valid_course_id(course_id)
    url = _request_url(course_metadata_path(course_id))

    try:
        async with httpx.AsyncClient(timeout=FIREBASE_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.get(url)
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Firebase is unavailable while reading course metadata.",
        ) from exc

    if response.status_code == 401:
        raise HTTPException(
            status_code=503,
            detail=(
                "Firebase rejected the metadata request (401). Provide "
                "FIREBASE_AUTH_TOKEN if database rules require authentication."
            ),
        )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=503,
            detail=f"Firebase metadata request failed with HTTP {response.status_code}.",
        )

    payload = response.json()
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=503,
            detail="Firebase returned an unexpected metadata payload shape.",
        )
    return payload


async def patch_starter_seed_generation(
    course_id: str,
    updates: dict[str, Any],
) -> None:
    """Patch courses/{courseId}/metadata/starterSeedGeneration (merge)."""
    assert_valid_course_id(course_id)
    url = _request_url(starter_seed_generation_path(course_id))

    try:
        async with httpx.AsyncClient(timeout=FIREBASE_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.patch(url, json=updates)
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Firebase is unavailable while updating starter seed status.",
        ) from exc

    if response.status_code == 401:
        raise HTTPException(
            status_code=503,
            detail=(
                "Firebase rejected the starter-seed status update (401). Provide "
                "FIREBASE_AUTH_TOKEN if database rules require authentication."
            ),
        )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=503,
            detail=(
                "Firebase starter-seed status update failed with "
                f"HTTP {response.status_code}."
            ),
        )


async def read_starter_seed_generation_status(course_id: str) -> str | None:
    """Return the current starterSeedGeneration.status, or None if missing."""
    try:
        metadata = await fetch_course_metadata(course_id)
    except (FirebaseConfigurationError, HTTPException):
        return None

    if not metadata:
        return None
    block = metadata.get("starterSeedGeneration")
    if not isinstance(block, dict):
        return None
    status = block.get("status")
    return status if isinstance(status, str) else None


async def best_effort_patch_starter_seed_generation(
    course_id: str,
    updates: dict[str, Any],
) -> bool:
    """Write starter status without failing the caller when Firebase is unavailable."""
    try:
        await patch_starter_seed_generation(course_id, updates)
        return True
    except (FirebaseConfigurationError, HTTPException, ValueError):
        return False


# --------------------------------------------------------------------------- #
# Reconciling starter status against what is actually stored
#
# The record is written by whichever run produced it, and until now only the
# automatic post-upload job wrote one at all. A later top-up adds seeds through
# a different route and left the record describing the first run forever — which
# is how a course holding 50 seeds kept reporting "partial, 9 of 50".
#
# The fix is to stop deriving the record from one run's own output and derive it
# from the course instead: after a run that persisted anything, count what is
# actually stored under seedExamples and write that.
# --------------------------------------------------------------------------- #


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


async def count_course_seed_examples(course_id: str) -> int | None:
    """How many seeds the course actually holds, or None if Firebase can't say.

    None is not zero, and the difference matters: a failed read must leave the
    caller free to keep whatever counts it already had rather than overwrite a
    true record with a false zero.
    """
    try:
        payload = await fetch_course_seed_examples(course_id)
    except (FirebaseConfigurationError, HTTPException, ValueError):
        return None

    if not payload:
        return 0
    return sum(1 for record in payload.values() if isinstance(record, dict))


async def read_starter_seed_generation(course_id: str) -> dict[str, Any] | None:
    """The stored starterSeedGeneration block, or None when absent."""
    try:
        metadata = await fetch_course_metadata(course_id)
    except (FirebaseConfigurationError, HTTPException, ValueError):
        return None

    if not metadata:
        return None
    block = metadata.get("starterSeedGeneration")
    return block if isinstance(block, dict) else None


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
