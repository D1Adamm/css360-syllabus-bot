"""PostgreSQL repository for anonymous participants.

A participant is a random identifier bound to one course. It is created when
a classroom code is redeemed and is the only thing research data ever
references. There is deliberately nothing here to update except
`last_seen_at`: the row has no name, no contact, and no attribute a student
could be asked to correct.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

from app.course_id import assert_valid_course_id
from app.db_mapping import put_optional, to_iso

PARTICIPANT_COLUMNS = "participant_id, course_id, invitation_id, created_at, last_seen_at"


def new_participant_id() -> str:
    return str(uuid.uuid4())


def map_participant(row: Mapping[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "participantId": str(row["participant_id"]),
        "courseId": row["course_id"],
        "createdAt": to_iso(row.get("created_at")),
    }
    invitation_id = row.get("invitation_id")
    put_optional(
        record, "invitationId", str(invitation_id) if invitation_id is not None else None
    )
    put_optional(record, "lastSeenAt", to_iso(row.get("last_seen_at")))
    return record


def create_participant(
    conn: Any,
    *,
    course_id: str,
    invitation_id: str | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    safe_course_id = assert_valid_course_id(course_id)
    moment = (now or datetime.now(timezone.utc)).isoformat()
    participant_id = new_participant_id()
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO participants (
                participant_id, course_id, invitation_id, created_at, last_seen_at
            ) VALUES (%s, %s, %s, %s, %s)
            """,
            (participant_id, safe_course_id, invitation_id, moment, moment),
        )
    created = get_participant(conn, participant_id)
    if created is None:  # pragma: no cover - defensive
        raise RuntimeError("The participant could not be read back after insert.")
    return created


def get_participant(conn: Any, participant_id: str) -> dict[str, Any] | None:
    with conn.cursor() as cursor:
        cursor.execute(
            f"SELECT {PARTICIPANT_COLUMNS} FROM participants WHERE participant_id = %s",
            (participant_id,),
        )
        row = cursor.fetchone()
    return map_participant(row) if row else None


def touch_participant(conn: Any, participant_id: str, now: datetime | None = None) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            "UPDATE participants SET last_seen_at = %s WHERE participant_id = %s",
            ((now or datetime.now(timezone.utc)).isoformat(), participant_id),
        )


def count_participants(conn: Any, course_id: str) -> int:
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) AS total FROM participants WHERE course_id = %s",
            (safe_course_id,),
        )
        row = cursor.fetchone()
    return int(row["total"]) if row else 0


def count_participants_for_invitation(conn: Any, invitation_id: str) -> int:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) AS total FROM participants WHERE invitation_id = %s",
            (invitation_id,),
        )
        row = cursor.fetchone()
    return int(row["total"]) if row else 0
