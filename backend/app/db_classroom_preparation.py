"""PostgreSQL repository for how a course's last preparation run ended.

One row per course (`classroom_preparation`), overwritten by the next run.
It exists so the preparation page can say something true after the backend
restarts: the run itself lives in the backend process, and when that process
is gone the row is what remains.

Progress is never stored here. How many answers are ready is counted from
`answer_cache`, so a stale row can misdescribe how a run ended but never what
is actually prepared.

`boot_id` names the process that owns the run. A write that ends a run only
applies to a row the same process started, so a late write from an old run
cannot overwrite a newer one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from app.course_id import assert_valid_course_id
from app.db_mapping import as_int, optional_string, to_iso

ACTIVE_STATES = ("running", "stopping")
FINAL_STATES = ("completed", "stopped", "failed")

COLUMNS = """
    course_id, state, boot_id, started_at, started_by, stop_requested_at,
    finished_at, reason, message, generated_count, failed_count
"""


def map_run(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "courseId": row["course_id"],
        "state": row["state"],
        "bootId": row["boot_id"],
        "startedAt": to_iso(row["started_at"]),
        "stopRequestedAt": to_iso(row.get("stop_requested_at")),
        "finishedAt": to_iso(row.get("finished_at")),
        "reason": optional_string(row.get("reason")),
        "message": optional_string(row.get("message")),
        "generated": as_int(row.get("generated_count")),
        "failed": as_int(row.get("failed_count")),
    }


def get_run(conn: Any, course_id: str) -> dict[str, Any] | None:
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            f"SELECT {COLUMNS} FROM classroom_preparation WHERE course_id = %s",
            (safe_course_id,),
        )
        row = cursor.fetchone()
    return map_run(row) if row else None


def begin_run(
    conn: Any,
    course_id: str,
    *,
    boot_id: str,
    started_by: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record that this process started a run, replacing the previous run's row."""
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO classroom_preparation
                (course_id, state, boot_id, started_at, started_by,
                 stop_requested_at, finished_at, reason, message,
                 generated_count, failed_count)
            VALUES (%s, 'running', %s, %s, %s, NULL, NULL, NULL, NULL, 0, 0)
            ON CONFLICT (course_id) DO UPDATE SET
                state = 'running',
                boot_id = EXCLUDED.boot_id,
                started_at = EXCLUDED.started_at,
                started_by = EXCLUDED.started_by,
                stop_requested_at = NULL,
                finished_at = NULL,
                reason = NULL,
                message = NULL,
                generated_count = 0,
                failed_count = 0
            RETURNING {COLUMNS}
            """,
            (safe_course_id, boot_id, now or datetime.now(timezone.utc), started_by),
        )
        row = cursor.fetchone()
    return map_run(row)


def mark_stopping(
    conn: Any, course_id: str, *, boot_id: str, now: datetime | None = None
) -> bool:
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE classroom_preparation
            SET state = 'stopping', stop_requested_at = %s
            WHERE course_id = %s AND boot_id = %s AND state = 'running'
            """,
            (now or datetime.now(timezone.utc), safe_course_id, boot_id),
        )
        return cursor.rowcount > 0


def record_progress(
    conn: Any, course_id: str, *, boot_id: str, generated: int, failed: int
) -> bool:
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE classroom_preparation
            SET generated_count = %s, failed_count = %s
            WHERE course_id = %s AND boot_id = %s
            """,
            (int(generated), int(failed), safe_course_id, boot_id),
        )
        return cursor.rowcount > 0


def finish_run(
    conn: Any,
    course_id: str,
    *,
    boot_id: str,
    state: str,
    reason: str | None,
    message: str | None,
    generated: int,
    failed: int,
    now: datetime | None = None,
) -> bool:
    """End the run this process started. A row another process owns is left alone."""
    if state not in FINAL_STATES:
        raise ValueError(f"A run ends as one of {FINAL_STATES}, not {state!r}.")
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE classroom_preparation
            SET state = %s, finished_at = %s, reason = %s, message = %s,
                generated_count = %s, failed_count = %s
            WHERE course_id = %s AND boot_id = %s
            """,
            (
                state,
                now or datetime.now(timezone.utc),
                reason,
                message,
                int(generated),
                int(failed),
                safe_course_id,
                boot_id,
            ),
        )
        return cursor.rowcount > 0
