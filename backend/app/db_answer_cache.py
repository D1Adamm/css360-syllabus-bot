"""PostgreSQL repository for the classroom answer cache.

One row is one real answer from one approach to one staff-selected question
under one configuration (`fingerprint`). See `app.answer_cache` for when a row
may be written or read, and `db/migrations/004_classroom_answer_cache.sql`
for the table.

Rows are never updated or deleted here. A changed configuration is a different
fingerprint, so an old row stops being found; it stays so that a rating which
recorded its `entry_id` can still be traced to the answer the student saw.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from app.course_id import assert_valid_course_id
from app.db_mapping import to_iso

CONDITIONS = ("base", "rag", "fineTuned", "fineTunedRag")
SOURCES = ("prepared", "live")

ENTRY_COLUMNS = """
    entry_id, course_id, condition, question_key, fingerprint, question_text,
    payload, identity, provenance, source, generated_at
"""


def map_entry(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "entryId": row["entry_id"],
        "courseId": row["course_id"],
        "condition": row["condition"],
        "questionKey": row["question_key"],
        "fingerprint": row["fingerprint"],
        "questionText": row["question_text"],
        "payload": row["payload"] if isinstance(row["payload"], dict) else {},
        "identity": row["identity"] if isinstance(row["identity"], dict) else {},
        "provenance": row["provenance"] if isinstance(row["provenance"], dict) else {},
        "source": row["source"],
        "generatedAt": to_iso(row["generated_at"]),
    }


def get_entry(
    conn: Any, course_id: str, condition: str, question_key: str, fingerprint: str
) -> dict[str, Any] | None:
    """The stored answer for exactly this course, approach, question and configuration."""
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {ENTRY_COLUMNS} FROM answer_cache
            WHERE course_id = %s AND condition = %s
              AND question_key = %s AND fingerprint = %s
            """,
            (safe_course_id, condition, question_key, fingerprint),
        )
        row = cursor.fetchone()
    return map_entry(row) if row else None


def insert_entry(
    conn: Any,
    *,
    course_id: str,
    condition: str,
    question_key: str,
    fingerprint: str,
    question_text: str,
    payload: Mapping[str, Any],
    identity: Mapping[str, Any],
    provenance: Mapping[str, Any],
    source: str,
    generated_at: datetime | None = None,
) -> dict[str, Any] | None:
    """Store one answer. Returns the new row, or None when one already existed.

    The first answer stored for a configuration stays: a second writer (a
    student and the preparation run finishing the same question together) is a
    no-op rather than a replacement, so nobody is ever shown two different
    stored answers for one configuration.
    """
    from psycopg.types.json import Json

    safe_course_id = assert_valid_course_id(course_id)
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}.")
    if source not in SOURCES:
        raise ValueError(f"Unknown source {source!r}.")
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO answer_cache ({ENTRY_COLUMNS})
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (course_id, condition, question_key, fingerprint) DO NOTHING
            RETURNING {ENTRY_COLUMNS}
            """,
            (
                uuid.uuid4().hex,
                safe_course_id,
                condition,
                question_key,
                fingerprint,
                question_text,
                Json(dict(payload)),
                Json(dict(identity)),
                Json(dict(provenance)),
                source,
                generated_at or datetime.now(timezone.utc),
            ),
        )
        row = cursor.fetchone()
    return map_entry(row) if row else None


def list_entry_keys(
    conn: Any, course_id: str, question_keys: Iterable[str]
) -> list[dict[str, str]]:
    """Which answers exist for these questions, under any configuration.

    Only the keys, not the answers: the preparation status counts ready,
    missing and outdated answers from this without reading any text.
    """
    safe_course_id = assert_valid_course_id(course_id)
    keys = sorted(set(question_keys))
    if not keys:
        return []
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT condition, question_key, fingerprint FROM answer_cache
            WHERE course_id = %s AND question_key = ANY(%s)
            """,
            (safe_course_id, keys),
        )
        rows = cursor.fetchall()
    return [
        {
            "condition": row["condition"],
            "questionKey": row["question_key"],
            "fingerprint": row["fingerprint"],
        }
        for row in rows
    ]
