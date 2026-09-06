"""Per-course fact-inventory cache shared by inspection and starter generation.

Cache files live beside the course index:
  data/indexes/{courseId}.facts.json

Invalidation:
- ``save_index`` / ``remove_index`` delete the cache (syllabus replacement)
- payload stores an ``indexFingerprint`` over chunk id+text+order; mismatch
  forces rebuild even if the file remains

Builds are tracked per course while they run. A rebuild is minutes to hours of
CPU-bound model calls, so two callers asking for the same course must share one
build rather than each starting their own — and a caller that cannot wait that
long (the admin page) can start a build, return, and ask again later.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from app.storage import CourseArtifactStorage
from app.syllabus_facts import build_fact_inventory

logger = logging.getLogger(__name__)

# 2: extraction reports per-batch outcomes, and the output budget that was
# truncating extraction was raised. Inventories built before that are not just
# missing the new field — they may be missing most of their facts, and a course
# whose cache predates the fix would keep the degraded inventory forever. The
# bump forces one rebuild per course.
FACT_INVENTORY_CACHE_VERSION = 2

#: Builds currently running, by course id. Process-local, like the Ollama lock.
_inflight_builds: dict[str, asyncio.Task[dict[str, Any]]] = {}
_inflight_started_at: dict[str, str] = {}
#: Why the most recent build for a course failed, until someone reads it.
_last_build_errors: dict[str, str] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_index_fingerprint(raw_chunks: list[Any]) -> str:
    """Stable fingerprint of syllabus chunk evidence used for invalidation."""
    parts: list[str] = []
    for chunk in raw_chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunkId") or chunk.get("id") or "").strip()
        text = str(chunk.get("text") or "")
        order = chunk.get("order", "")
        parts.append(f"{chunk_id}\n{order}\n{text}")
    digest = hashlib.sha256("\n--\n".join(parts).encode("utf-8")).hexdigest()
    return digest


def _inventory_payload_valid(
    payload: dict[str, Any],
    *,
    fingerprint: str,
) -> bool:
    if int(payload.get("cacheVersion") or 0) != FACT_INVENTORY_CACHE_VERSION:
        return False
    if str(payload.get("indexFingerprint") or "") != fingerprint:
        return False
    inventory = payload.get("inventory")
    if not isinstance(inventory, dict):
        return False
    facts = inventory.get("facts")
    return isinstance(facts, list)


def _cached_inventory(
    course_id: str,
    storage: CourseArtifactStorage,
    fingerprint: str,
) -> dict[str, Any] | None:
    cached_payload = storage.load_fact_inventory(course_id)
    if cached_payload is None or not _inventory_payload_valid(
        cached_payload, fingerprint=fingerprint
    ):
        return None
    inventory = dict(cached_payload["inventory"])
    inventory["cached"] = True
    inventory["indexFingerprint"] = fingerprint
    return inventory


def peek_fact_inventory(
    *,
    course_id: str,
    raw_chunks: list[Any],
    storage: CourseArtifactStorage,
) -> dict[str, Any] | None:
    """The cached inventory for these chunks, or None. Never builds anything."""
    return _cached_inventory(course_id, storage, compute_index_fingerprint(raw_chunks))


async def _build_and_store(
    *,
    course_id: str,
    raw_chunks: list[Any],
    storage: CourseArtifactStorage,
    fingerprint: str,
    completion_fn,
    embed_fn,
    build_kwargs: dict[str, Any],
) -> dict[str, Any]:
    inventory = await build_fact_inventory(
        raw_chunks=raw_chunks,
        completion_fn=completion_fn,
        embed_fn=embed_fn,
        **build_kwargs,
    )
    # Persist only the inspectable inventory body (not call wrappers).
    persistable = {
        key: value
        for key, value in inventory.items()
        if key not in {"cached", "indexFingerprint"}
    }
    storage.save_fact_inventory(
        course_id,
        {
            "cacheVersion": FACT_INVENTORY_CACHE_VERSION,
            "indexFingerprint": fingerprint,
            "inventory": persistable,
        },
    )
    result = dict(persistable)
    result["cached"] = False
    result["indexFingerprint"] = fingerprint
    return result


def _describe_failure(exc: BaseException) -> str:
    if isinstance(exc, HTTPException):
        return str(exc.detail)
    text = str(exc).strip() or exc.__class__.__name__
    return text[:300]


def start_fact_inventory_build(
    *,
    course_id: str,
    raw_chunks: list[Any],
    storage: CourseArtifactStorage,
    completion_fn=None,
    embed_fn=None,
    **build_kwargs: Any,
) -> asyncio.Task[dict[str, Any]]:
    """The build running for this course, starting one if there is none.

    Returns without waiting. Whoever needs the result awaits the task; whoever
    cannot wait asks `fact_inventory_build_status` later. The task's outcome is
    always collected by the done callback, so an unawaited failure is recorded
    for the next caller instead of being dropped by the event loop.
    """
    existing = _inflight_builds.get(course_id)
    if existing is not None and not existing.done():
        return existing

    fingerprint = compute_index_fingerprint(raw_chunks)
    task = asyncio.get_running_loop().create_task(
        _build_and_store(
            course_id=course_id,
            raw_chunks=raw_chunks,
            storage=storage,
            fingerprint=fingerprint,
            completion_fn=completion_fn,
            embed_fn=embed_fn,
            build_kwargs=build_kwargs,
        )
    )
    _inflight_builds[course_id] = task
    _inflight_started_at[course_id] = _utc_now()
    _last_build_errors.pop(course_id, None)

    def _finished(done: asyncio.Task[dict[str, Any]]) -> None:
        if _inflight_builds.get(course_id) is done:
            _inflight_builds.pop(course_id, None)
            _inflight_started_at.pop(course_id, None)
        if done.cancelled():
            _last_build_errors[course_id] = "The build was cancelled before it finished."
            return
        exc = done.exception()
        if exc is not None:
            _last_build_errors[course_id] = _describe_failure(exc)
            logger.warning(
                "Fact inventory build failed for course %s: %s", course_id, exc
            )

    task.add_done_callback(_finished)
    return task


def fact_inventory_build_status(course_id: str) -> dict[str, Any]:
    """Whether a build is running for this course, and how the last one ended."""
    task = _inflight_builds.get(course_id)
    building = task is not None and not task.done()
    return {
        "building": building,
        "startedAt": _inflight_started_at.get(course_id) if building else None,
        "lastError": _last_build_errors.get(course_id),
    }


def take_fact_inventory_build_error(course_id: str) -> str | None:
    """Return and clear the recorded failure, so a retry starts clean."""
    return _last_build_errors.pop(course_id, None)


def reset_fact_inventory_builds_for_tests() -> None:
    """Forget every tracked build and failure (tests only)."""
    _inflight_builds.clear()
    _inflight_started_at.clear()
    _last_build_errors.clear()


async def load_or_build_fact_inventory(
    *,
    course_id: str,
    raw_chunks: list[Any],
    storage: CourseArtifactStorage,
    force_refresh: bool = False,
    completion_fn=None,
    embed_fn=None,
    **build_kwargs: Any,
) -> dict[str, Any]:
    """Return fact inventory, reusing a valid per-course cache when possible.

    Returns a dict with:
    - all standard inventory fields (facts, factCount, ...)
    - ``cached``: True when served from disk without extraction
    - ``indexFingerprint``: fingerprint used for this inventory

    A build already running for the course is joined rather than duplicated:
    two admins clicking Inspect, or an inspection during starter generation,
    cost one pass over the syllabus, not two competing for the same CPU.
    """
    fingerprint = compute_index_fingerprint(raw_chunks)

    if not force_refresh:
        cached = _cached_inventory(course_id, storage, fingerprint)
        if cached is not None:
            return cached

    task = start_fact_inventory_build(
        course_id=course_id,
        raw_chunks=raw_chunks,
        storage=storage,
        completion_fn=completion_fn,
        embed_fn=embed_fn,
        **build_kwargs,
    )
    # Each awaiter gets its own copy; a shared task result mutated by one
    # caller must not change under another.
    return dict(await task)
