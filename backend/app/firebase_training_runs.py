"""Enqueue a training run into the Firebase queue Tillicum claims from.

This is the backend half of the write the browser used to make. It exists so
the browser can stop writing Firebase without moving the queue out from under
the cluster runner.

The queue contract is `scripts/lib/training_queue.py`, and it is not negotiable
here: runs live at `courses/{courseId}/trainingRuns/{runId}`, the run id is the
key and is deliberately not repeated inside the record, and the runner requires
`courseId`, `state`, `mode`, and `enqueuedAt` on every record it parses.

Concurrency is preserved the same way the runner does it. The browser used a
Realtime Database transaction over the whole `trainingRuns` node, because the
property being guarded belongs to the set — no two runs outstanding for one
course at once — not to any single run. Over REST the equivalent is a
conditional write:

    GET  .../trainingRuns.json  with `X-Firebase-ETag: true`
    PUT  .../trainingRuns.json  with `if-match: <that etag>`

The PUT lands only if nothing changed in between, so two admins clicking at the
same moment cannot both enqueue: the loser gets 412 and is retried once against
the new state, where it sees the winner's active run and is refused properly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import HTTPException

from app.course_id import assert_valid_course_id
from app.db_training_runs import (
    MODES,
    TERMINAL_RUN_STATES,
    generate_training_run_id,
)
from app.firebase_seeds import (
    FIREBASE_REQUEST_TIMEOUT_SECONDS,
    _request_url,
)

# One retry only. A second 412 means genuine contention, and at that point the
# honest answer is that something else is queueing for this course.
MAX_CAS_ATTEMPTS = 2


class ActiveTrainingRunError(Exception):
    """This course already has a run that is not finished."""


def course_training_runs_path(course_id: str) -> str:
    safe_course_id = assert_valid_course_id(course_id)
    return f"courses/{safe_course_id}/trainingRuns"


def _has_active_run(payload: Any) -> bool:
    """Whether any stored run is still outstanding.

    Mirrors `findActiveTrainingRun` and the runner's own view: anything that is
    not `succeeded` or `failed` is work in flight.
    """
    if not isinstance(payload, dict):
        return False

    for raw in payload.values():
        if not isinstance(raw, dict):
            continue
        state = raw.get("state")
        if isinstance(state, str) and state not in TERMINAL_RUN_STATES:
            return True
    return False


def build_run_record(
    *,
    course_id: str,
    mode: str,
    dataset_ref: str,
    approved_example_count: int,
    train_examples: int,
    validation_examples: int,
    enqueued_at: str,
) -> dict[str, Any]:
    """The stored record, exactly as `training_queue.parse_run` expects it.

    No `runId` field: the key is the id, and a second copy could disagree with
    it after a manual edit. No `jobId`: only the cluster can produce one, and a
    placeholder would make an unsubmitted run look submitted.
    """
    return {
        "courseId": course_id,
        "mode": mode,
        "state": "queued",
        "enqueuedAt": enqueued_at,
        "updatedAt": enqueued_at,
        "datasetRef": dataset_ref,
        "approvedExampleCount": max(0, int(approved_example_count)),
        "trainExamples": max(0, int(train_examples)),
        "validationExamples": max(0, int(validation_examples)),
        # No runner has taken it yet. The runner increments this as it claims.
        "attempt": 0,
    }


async def _get_runs_with_etag(client: httpx.AsyncClient, url: str) -> tuple[Any, str]:
    response = await client.get(url, headers={"X-Firebase-ETag": "true"})
    if response.status_code >= 400:
        raise HTTPException(
            status_code=503,
            detail=f"Firebase training-run read failed with HTTP {response.status_code}.",
        )
    etag = response.headers.get("ETag", "")
    payload = response.json()
    return payload, etag


async def enqueue_training_run(
    *,
    course_id: str,
    mode: str,
    dataset_ref: str,
    approved_example_count: int = 0,
    train_examples: int = 0,
    validation_examples: int = 0,
) -> dict[str, Any]:
    """Write one `queued` run, refusing while this course has an active one.

    Returns `{"runId": ..., "record": {...}}` on success. Raises
    ActiveTrainingRunError when the course is already busy, and HTTPException
    when Firebase itself could not be reached — the caller must not mirror
    anything to PostgreSQL in either case.
    """
    safe_course_id = assert_valid_course_id(course_id)
    if mode not in MODES:
        raise ValueError(f"Unknown training mode: {mode!r}. Expected one of {MODES}.")

    url = _request_url(course_training_runs_path(safe_course_id))

    try:
        async with httpx.AsyncClient(
            timeout=FIREBASE_REQUEST_TIMEOUT_SECONDS
        ) as client:
            for attempt in range(MAX_CAS_ATTEMPTS):
                current, etag = await _get_runs_with_etag(client, url)

                if _has_active_run(current):
                    raise ActiveTrainingRunError(
                        f'Course "{safe_course_id}" already has an active training run.'
                    )

                now = datetime.now(timezone.utc)
                run_id = generate_training_run_id(now)
                record = build_run_record(
                    course_id=safe_course_id,
                    mode=mode,
                    dataset_ref=dataset_ref,
                    approved_example_count=approved_example_count,
                    train_examples=train_examples,
                    validation_examples=validation_examples,
                    enqueued_at=now.isoformat(),
                )

                existing = current if isinstance(current, dict) else {}
                next_value = {**existing, run_id: record}

                response = await client.put(
                    url,
                    json=next_value,
                    headers={"if-match": etag} if etag else None,
                )

                if response.status_code == 412:
                    # Someone else wrote first. Re-read and decide again rather
                    # than forcing this run over whatever they queued.
                    if attempt + 1 < MAX_CAS_ATTEMPTS:
                        continue
                    raise ActiveTrainingRunError(
                        "Another training run was queued for this course at the "
                        "same moment. Reload and try again."
                    )

                if response.status_code >= 400:
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "Firebase training-run write failed with HTTP "
                            f"{response.status_code}."
                        ),
                    )

                return {"runId": run_id, "record": record}
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=503,
            detail="Firebase timed out while queueing a training run.",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail="Firebase is unavailable while queueing a training run.",
        ) from exc

    raise HTTPException(  # pragma: no cover - loop always returns or raises
        status_code=503,
        detail="Could not queue a training run.",
    )
