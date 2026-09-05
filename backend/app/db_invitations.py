"""PostgreSQL repository for invitations.

Four kinds in one table (see the migration for why): a reusable classroom
`student` code bound to a course, and single-use `professor`, `admin` and
`reset` tokens. What they share is the lifecycle — created by someone,
possibly expiring, possibly revoked, used some number of times up to a limit —
and the one operation that matters for security:

`consume` is a single conditional UPDATE. It increments the use count only if
the row is still redeemable at that instant, and returns the row only if it
did. Two concurrent redemptions of a single-use invitation therefore cannot
both succeed: the second UPDATE matches no row and the caller sees None. There
is no read-then-write for a race to slip between.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from app.course_id import assert_valid_course_id
from app.db_mapping import optional_string, put_optional, to_iso

STUDENT = "student"
PROFESSOR = "professor"
ADMIN = "admin"
RESET = "reset"
KINDS = (STUDENT, PROFESSOR, ADMIN, RESET)
TOKEN_KINDS = (PROFESSOR, ADMIN, RESET)

#: Lifecycle states derived from the row. `used` covers a single-use
#: invitation that has been accepted.
ACTIVE = "active"
REVOKED = "revoked"
EXPIRED = "expired"
USED = "used"

INVITATION_COLUMNS = """
    invitation_id, kind, code, course_id, target_user_id, label, created_by,
    created_at, expires_at, max_uses, use_count, revoked_at, revoked_by,
    accepted_at, accepted_by_user_id
"""


class CodeCollisionError(Exception):
    """The generated classroom code is already taken; generate another."""


def new_invitation_id() -> str:
    return str(uuid.uuid4())


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(timezone.utc)


def _aware(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _uuid_text(value: Any) -> str | None:
    return str(value) if value is not None else None


def status_of(row: Mapping[str, Any], now: datetime) -> str:
    """Where an invitation is in its lifecycle, from the raw row."""
    if row.get("revoked_at") is not None:
        return REVOKED
    expires_at = _aware(row.get("expires_at"))
    if expires_at is not None and expires_at <= now:
        return EXPIRED
    max_uses = row.get("max_uses")
    if max_uses is not None and int(row.get("use_count") or 0) >= int(max_uses):
        return USED
    return ACTIVE


def is_redeemable(row: Mapping[str, Any], now: datetime) -> bool:
    return status_of(row, now) == ACTIVE


def map_invitation(row: Mapping[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """One row as the API record. The token hash is never part of it."""
    moment = _now(now)
    record: dict[str, Any] = {
        "invitationId": str(row["invitation_id"]),
        "kind": row["kind"],
        "status": status_of(row, moment),
        "createdAt": to_iso(row.get("created_at")),
        "useCount": int(row.get("use_count") or 0),
    }
    put_optional(record, "code", optional_string(row.get("code")))
    put_optional(record, "courseId", optional_string(row.get("course_id")))
    put_optional(record, "targetUserId", _uuid_text(row.get("target_user_id")))
    put_optional(record, "label", optional_string(row.get("label")))
    put_optional(record, "createdBy", _uuid_text(row.get("created_by")))
    put_optional(record, "expiresAt", to_iso(row.get("expires_at")))
    if row.get("max_uses") is not None:
        record["maxUses"] = int(row["max_uses"])
    put_optional(record, "revokedAt", to_iso(row.get("revoked_at")))
    put_optional(record, "revokedBy", _uuid_text(row.get("revoked_by")))
    put_optional(record, "acceptedAt", to_iso(row.get("accepted_at")))
    put_optional(record, "acceptedByUserId", _uuid_text(row.get("accepted_by_user_id")))
    # Present when the query joined `users` or `courses` for display.
    put_optional(record, "createdByName", optional_string(row.get("created_by_name")))
    put_optional(record, "courseName", optional_string(row.get("course_name")))
    return record


def create_student_invitation(
    conn: Any,
    *,
    course_id: str,
    code: str,
    created_by: str | None,
    label: str | None = None,
    expires_at: datetime | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Insert one classroom code. Raises CodeCollisionError if the code exists.

    The unique index on `code` is the guard; `ON CONFLICT DO NOTHING` plus a
    rowcount check lets the caller draw another code rather than fail.
    """
    safe_course_id = assert_valid_course_id(course_id)
    invitation_id = new_invitation_id()
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO invitations (
                invitation_id, kind, code, course_id, label, created_by,
                created_at, expires_at, max_uses
            ) VALUES (%s, 'student', %s, %s, %s, %s, %s, %s, NULL)
            ON CONFLICT (code) WHERE code IS NOT NULL DO NOTHING
            """,
            (
                invitation_id,
                code,
                safe_course_id,
                optional_string(label),
                created_by,
                _now(now).isoformat(),
                expires_at.isoformat() if expires_at else None,
            ),
        )
        if cursor.rowcount == 0:
            raise CodeCollisionError(f"Classroom code {code} is already in use.")
    created = get_invitation(conn, invitation_id, now=now)
    if created is None:  # pragma: no cover - defensive
        raise RuntimeError("The invitation could not be read back after insert.")
    return created


def create_token_invitation(
    conn: Any,
    *,
    kind: str,
    token_hash: str,
    created_by: str | None,
    expires_at: datetime,
    now: datetime | None = None,
    course_ids: Iterable[str] = (),
    target_user_id: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Insert one single-use privileged invitation and its course grants."""
    if kind not in TOKEN_KINDS:
        raise ValueError(f"{kind!r} is not a token invitation kind.")
    if kind == RESET and not target_user_id:
        raise ValueError("A reset invitation needs a target user.")
    grants = [assert_valid_course_id(course_id) for course_id in course_ids]
    if kind != PROFESSOR and grants:
        raise ValueError("Only professor invitations assign courses.")

    invitation_id = new_invitation_id()
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO invitations (
                invitation_id, kind, token_hash, target_user_id, label,
                created_by, created_at, expires_at, max_uses
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 1)
            """,
            (
                invitation_id,
                kind,
                token_hash,
                target_user_id,
                optional_string(label),
                created_by,
                _now(now).isoformat(),
                expires_at.isoformat(),
            ),
        )
        for course_id in grants:
            cursor.execute(
                "INSERT INTO invitation_course_grants (invitation_id, course_id) "
                "VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (invitation_id, course_id),
            )
    created = get_invitation(conn, invitation_id, now=now)
    if created is None:  # pragma: no cover - defensive
        raise RuntimeError("The invitation could not be read back after insert.")
    if grants:
        created["courseIds"] = grants
    return created


def get_invitation(
    conn: Any, invitation_id: str, *, now: datetime | None = None
) -> dict[str, Any] | None:
    with conn.cursor() as cursor:
        cursor.execute(
            f"SELECT {INVITATION_COLUMNS} FROM invitations WHERE invitation_id = %s",
            (invitation_id,),
        )
        row = cursor.fetchone()
    return map_invitation(row, now) if row else None


def find_by_code(conn: Any, code: str, *, now: datetime | None = None) -> dict[str, Any] | None:
    """The student invitation behind a classroom code, in any state.

    Joined to the course so the join page can name it. Whether it is
    redeemable is the caller's question, answered by `status`.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT i.invitation_id, i.kind, i.code, i.course_id, i.target_user_id,
                   i.label, i.created_by, i.created_at, i.expires_at, i.max_uses,
                   i.use_count, i.revoked_at, i.revoked_by, i.accepted_at,
                   i.accepted_by_user_id, c.name AS course_name
            FROM invitations i
            JOIN courses c ON c.course_id = i.course_id
            WHERE i.code = %s AND i.kind = 'student'
            """,
            (code,),
        )
        row = cursor.fetchone()
    return map_invitation(row, now) if row else None


def find_by_token_hash(
    conn: Any, token_hash: str, *, now: datetime | None = None
) -> dict[str, Any] | None:
    with conn.cursor() as cursor:
        cursor.execute(
            f"SELECT {INVITATION_COLUMNS} FROM invitations WHERE token_hash = %s",
            (token_hash,),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    record = map_invitation(row, now)
    if record["kind"] == PROFESSOR:
        record["courseIds"] = list_grants(conn, record["invitationId"])
    return record


def consume(
    conn: Any,
    invitation_id: str,
    *,
    now: datetime | None = None,
    accepted_by_user_id: str | None = None,
) -> dict[str, Any] | None:
    """Redeem once, atomically. None means it was not redeemable at that instant.

    The WHERE clause is the whole guard: not revoked, not expired, uses left.
    A single-use invitation accepted twice at the same moment yields one row
    for one caller and no row for the other.
    """
    moment = _now(now).isoformat()
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            UPDATE invitations
            SET use_count = use_count + 1,
                accepted_at = CASE
                    WHEN %(accepted_by)s IS NULL THEN accepted_at ELSE %(now)s
                END,
                accepted_by_user_id = COALESCE(%(accepted_by)s, accepted_by_user_id)
            WHERE invitation_id = %(invitation_id)s
              AND revoked_at IS NULL
              AND (expires_at IS NULL OR expires_at > %(now)s)
              AND (max_uses IS NULL OR use_count < max_uses)
            RETURNING {INVITATION_COLUMNS}
            """,
            {
                "invitation_id": invitation_id,
                "now": moment,
                "accepted_by": accepted_by_user_id,
            },
        )
        row = cursor.fetchone()
    return map_invitation(row, now) if row else None


def revoke(
    conn: Any,
    invitation_id: str,
    *,
    revoked_by: str | None,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Revoke once. Already-revoked rows are left as they were."""
    with conn.cursor() as cursor:
        cursor.execute(
            "UPDATE invitations SET revoked_at = %s, revoked_by = %s "
            "WHERE invitation_id = %s AND revoked_at IS NULL",
            (_now(now).isoformat(), revoked_by, invitation_id),
        )
    return get_invitation(conn, invitation_id, now=now)


def list_student_invitations(
    conn: Any, course_id: str, *, now: datetime | None = None
) -> list[dict[str, Any]]:
    """Every classroom code ever issued for one course, newest first."""
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT i.invitation_id, i.kind, i.code, i.course_id, i.target_user_id,
                   i.label, i.created_by, i.created_at, i.expires_at, i.max_uses,
                   i.use_count, i.revoked_at, i.revoked_by, i.accepted_at,
                   i.accepted_by_user_id, u.display_name AS created_by_name
            FROM invitations i
            LEFT JOIN users u ON u.user_id = i.created_by
            WHERE i.course_id = %s AND i.kind = 'student'
            ORDER BY i.created_at DESC
            """,
            (safe_course_id,),
        )
        rows = cursor.fetchall()
    return [map_invitation(row, now) for row in rows]


def list_invitations(
    conn: Any,
    *,
    kinds: Iterable[str] = TOKEN_KINDS,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Privileged invitations for the admin page, newest first, with grants."""
    wanted = [kind for kind in kinds if kind in KINDS]
    if not wanted:
        return []
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT i.invitation_id, i.kind, i.code, i.course_id, i.target_user_id,
                   i.label, i.created_by, i.created_at, i.expires_at, i.max_uses,
                   i.use_count, i.revoked_at, i.revoked_by, i.accepted_at,
                   i.accepted_by_user_id, u.display_name AS created_by_name
            FROM invitations i
            LEFT JOIN users u ON u.user_id = i.created_by
            WHERE i.kind = ANY(%s)
            ORDER BY i.created_at DESC
            """,
            (wanted,),
        )
        rows = cursor.fetchall()
    records = [map_invitation(row, now) for row in rows]
    for record in records:
        if record["kind"] == PROFESSOR:
            record["courseIds"] = list_grants(conn, record["invitationId"])
    return records


def list_grants(conn: Any, invitation_id: str) -> list[str]:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT course_id FROM invitation_course_grants "
            "WHERE invitation_id = %s ORDER BY course_id ASC",
            (invitation_id,),
        )
        rows = cursor.fetchall()
    return [row["course_id"] for row in rows]


def count_active_student_invitations(conn: Any, course_id: str) -> int:
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT count(*) AS total FROM invitations
            WHERE course_id = %s AND kind = 'student' AND revoked_at IS NULL
              AND (expires_at IS NULL OR expires_at > %s)
            """,
            (safe_course_id, _now(None).isoformat()),
        )
        row = cursor.fetchone()
    return int(row["total"]) if row else 0


def revoke_active_student_invitations(
    conn: Any,
    course_id: str,
    *,
    revoked_by: str | None,
    now: datetime | None = None,
) -> int:
    """Retire every live code for a course, as the first half of a replacement."""
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE invitations SET revoked_at = %s, revoked_by = %s
            WHERE course_id = %s AND kind = 'student' AND revoked_at IS NULL
            """,
            (_now(now).isoformat(), revoked_by, safe_course_id),
        )
        return max(0, cursor.rowcount)
