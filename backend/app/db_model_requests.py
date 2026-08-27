"""PostgreSQL repository for course model requests.

Matches `CourseModelRequest` exactly, including the nested `preparation` and
`training` blocks, which stay whole in JSONB. They are admin-only detail whose
shape is still moving; flattening them would mean a schema migration every time
a field is added, and neither is ever queried by field.

The one-active-request-per-course rule from `createCourseModelRequest` is kept.
The guarantee comes from a conditional INSERT that only fires when no active
row exists, so two concurrent requests cannot both see an empty slot.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from app.course_id import assert_valid_course_id
from app.db_mapping import (
    as_int,
    bind_jsonb,
    build_patch,
    optional_string,
    put_optional,
    to_iso,
)

REQUEST_COLUMNS = """
    course_id, status, requested_at, updated_at, approved_example_count,
    failure_message, preparation, preparation_error, training, launch_error,
    current_run_id
"""

REQUEST_STATUSES = ("requested", "preparing", "training", "ready", "failed")

# Only these block a second request. `ready` and `failed` are terminal — a
# failed request must not lock a course out forever.
ACTIVE_STATUSES = ("requested", "preparing", "training")

REQUEST_PATCH_COLUMNS = {
    "status": "status",
    "updatedAt": "updated_at",
    "approvedExampleCount": "approved_example_count",
    "failureMessage": "failure_message",
    "preparation": "preparation",
    "preparationError": "preparation_error",
    "training": "training",
    "launchError": "launch_error",
    "currentRunId": "current_run_id",
}

JSONB_COLUMNS = frozenset({"preparation", "training"})


class ActiveModelRequestError(Exception):
    """Raised when a course already has an outstanding request."""


def _bind(parameters: dict[str, Any]) -> dict[str, Any]:
    return bind_jsonb(parameters, JSONB_COLUMNS)


def map_model_request(row: Mapping[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "courseId": row["course_id"],
        "status": row["status"],
        "requestedAt": to_iso(row.get("requested_at")),
        # `parseCourseModelRequest` falls back to requestedAt when updatedAt is
        # missing; the column is NOT NULL, so the fallback is belt and braces.
        "updatedAt": to_iso(row.get("updated_at")) or to_iso(row.get("requested_at")),
        "approvedExampleCount": as_int(row.get("approved_example_count")),
    }

    put_optional(record, "failureMessage", optional_string(row.get("failure_message")))
    put_optional(
        record, "preparationError", optional_string(row.get("preparation_error"))
    )
    put_optional(record, "launchError", optional_string(row.get("launch_error")))
    put_optional(record, "currentRunId", optional_string(row.get("current_run_id")))

    preparation = row.get("preparation")
    if isinstance(preparation, dict):
        record["preparation"] = preparation
    training = row.get("training")
    if isinstance(training, dict):
        record["training"] = training

    return record


def get_model_request(conn: Any, course_id: str) -> dict[str, Any] | None:
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            f"SELECT {REQUEST_COLUMNS} FROM model_requests WHERE course_id = %s",
            (safe_course_id,),
        )
        row = cursor.fetchone()
    return map_model_request(row) if row else None


def is_active(request: Mapping[str, Any] | None) -> bool:
    return request is not None and request.get("status") in ACTIVE_STATUSES


def lock_model_request(conn: Any, course_id: str) -> dict[str, Any] | None:
    """Read one request under a row lock, or None when there is none.

    `SELECT ... FOR UPDATE` inside the caller's transaction. Everything a
    retry does — retiring the outstanding run, queueing its replacement,
    repointing `current_run_id` — is a read-then-write across two tables, and
    the request row is the one thing both tables hang off. Taking it first
    means a second retry for the same course waits here and, when it proceeds,
    reads the pointer the first one already moved rather than the stale one it
    started from.

    Deliberately not folded into `get_model_request`: an unqualified read must
    stay lock-free, because every listing route does one.
    """
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            f"SELECT {REQUEST_COLUMNS} FROM model_requests "
            "WHERE course_id = %s FOR UPDATE",
            (safe_course_id,),
        )
        row = cursor.fetchone()
    return map_model_request(row) if row else None


def create_model_request(
    conn: Any,
    course_id: str,
    approved_example_count: int,
) -> dict[str, Any]:
    """Open a request, refusing while one is still outstanding.

    The INSERT ... WHERE NOT EXISTS is the whole guard: it re-checks for an
    active row at write time under the row lock, so a double-submitted form
    cannot produce two requests. A read-then-write would let both callers see
    the same empty slot.
    """
    safe_course_id = assert_valid_course_id(course_id)
    now = datetime.now(timezone.utc).isoformat()
    parameters = {
        "course_id": safe_course_id,
        "status": "requested",
        "requested_at": now,
        "updated_at": now,
        "approved_example_count": max(0, as_int(approved_example_count)),
        "active_statuses": list(ACTIVE_STATUSES),
    }

    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO model_requests (
                course_id, status, requested_at, updated_at, approved_example_count
            )
            SELECT %(course_id)s, %(status)s, %(requested_at)s, %(updated_at)s,
                   %(approved_example_count)s
            WHERE NOT EXISTS (
                SELECT 1 FROM model_requests
                WHERE course_id = %(course_id)s
                  AND status = ANY(%(active_statuses)s)
            )
            ON CONFLICT (course_id) DO UPDATE SET
                status = EXCLUDED.status,
                requested_at = EXCLUDED.requested_at,
                updated_at = EXCLUDED.updated_at,
                approved_example_count = EXCLUDED.approved_example_count,
                failure_message = NULL,
                preparation = NULL,
                preparation_error = NULL,
                training = NULL,
                launch_error = NULL,
                current_run_id = NULL
            """,
            parameters,
        )
        if cursor.rowcount == 0:
            raise ActiveModelRequestError(
                f'Course "{safe_course_id}" already has an outstanding model request.'
            )

    created = get_model_request(conn, safe_course_id)
    if created is None:  # pragma: no cover - defensive
        raise ActiveModelRequestError(
            f'The model request for "{safe_course_id}" could not be read back.'
        )
    return created


def update_model_request_for_run(
    conn: Any,
    course_id: str,
    run_id: str,
    patch: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Merge onto the request only while `run_id` still owns it.

    The ownership guard for every callback a cluster worker makes. A run that
    an admin retried is no longer this course's current run, and a late report
    from it — a job that finished after its run was retired — must not be able
    to move `status`, `training`, `failure_message` or `current_run_id` out
    from under the replacement.

    Written as one conditional UPDATE rather than a read followed by a write.
    Callers take the row lock first, so the read would be correct too; the
    condition is here as well because it costs one clause and it means the
    guarantee does not depend on every future caller remembering to lock.

    `current_run_id IS NULL` counts as ownable. A run enqueued through the
    operational route before anything pointed at it is in exactly that state,
    and the first callback for it is the write that claims it.

    Returns None when the guard rejected the write, and None when there is no
    request at all. Callers distinguish the two by having already read it.
    """
    safe_course_id = assert_valid_course_id(course_id)

    assignments = build_patch(patch, REQUEST_PATCH_COLUMNS)
    if not assignments:
        return get_model_request(conn, safe_course_id)

    assignments.setdefault("updated_at", datetime.now(timezone.utc).isoformat())

    sets = ", ".join(f"{column} = %({column})s" for column in assignments)
    parameters = {
        **assignments,
        "course_id": safe_course_id,
        "expected_run_id": run_id,
    }
    with conn.cursor() as cursor:
        cursor.execute(
            f"UPDATE model_requests SET {sets} "
            "WHERE course_id = %(course_id)s "
            "AND (current_run_id IS NULL OR current_run_id = %(expected_run_id)s)",
            _bind(parameters),
        )
        if cursor.rowcount == 0:
            return None

    return get_model_request(conn, safe_course_id)


def update_model_request(
    conn: Any,
    course_id: str,
    patch: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Merge fields onto an existing request. None when there is none.

    A merge, matching `updateCourseModelRequest`: preparation records extra
    fields onto a request that already exists and must not clobber
    `requestedAt` or the professor's original approved count.
    """
    safe_course_id = assert_valid_course_id(course_id)
    if get_model_request(conn, safe_course_id) is None:
        return None

    assignments = build_patch(patch, REQUEST_PATCH_COLUMNS)
    if not assignments:
        return get_model_request(conn, safe_course_id)

    # Any patch is a state change; the timestamp travels with it unless the
    # caller set one explicitly.
    assignments.setdefault("updated_at", datetime.now(timezone.utc).isoformat())

    sets = ", ".join(f"{column} = %({column})s" for column in assignments)
    parameters = {**assignments, "course_id": safe_course_id}
    with conn.cursor() as cursor:
        cursor.execute(
            f"UPDATE model_requests SET {sets} WHERE course_id = %(course_id)s",
            _bind(parameters),
        )

    return get_model_request(conn, safe_course_id)
