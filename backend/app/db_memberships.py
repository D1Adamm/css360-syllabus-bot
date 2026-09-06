"""PostgreSQL repository for course memberships.

Which professors may act on which courses. This table, not `users.role`, is
what course-scoped authorization reads: a professor with no row for a course
is refused that course exactly as an anonymous visitor would be.
Administrators have no rows here and need none.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from app.course_id import assert_valid_course_id
from app.db_mapping import put_optional, to_iso

MEMBERSHIP_COLUMNS = "course_id, user_id, membership_role, granted_by, granted_at"

INSTRUCTOR = "instructor"


def map_membership(row: Mapping[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "courseId": row["course_id"],
        "userId": str(row["user_id"]),
        "membershipRole": row.get("membership_role") or INSTRUCTOR,
        "grantedAt": to_iso(row.get("granted_at")),
    }
    granted_by = row.get("granted_by")
    put_optional(record, "grantedBy", str(granted_by) if granted_by is not None else None)
    # Present when the query joined `users`.
    put_optional(record, "email", row.get("email"))
    put_optional(record, "displayName", row.get("display_name"))
    return record


def list_course_ids_for_user(conn: Any, user_id: str) -> list[str]:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT course_id FROM course_memberships WHERE user_id = %s "
            "ORDER BY course_id ASC",
            (user_id,),
        )
        rows = cursor.fetchall()
    return [row["course_id"] for row in rows]


def list_memberships(conn: Any) -> list[dict[str, Any]]:
    with conn.cursor() as cursor:
        cursor.execute(
            f"SELECT {MEMBERSHIP_COLUMNS} FROM course_memberships "
            "ORDER BY course_id ASC, granted_at ASC"
        )
        rows = cursor.fetchall()
    return [map_membership(row) for row in rows]


def list_memberships_for_course(conn: Any, course_id: str) -> list[dict[str, Any]]:
    """Instructors of one course, with the display fields an admin page shows."""
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT m.course_id, m.user_id, m.membership_role, m.granted_by,
                   m.granted_at, u.email, u.display_name
            FROM course_memberships m
            JOIN users u ON u.user_id = m.user_id
            WHERE m.course_id = %s
            ORDER BY m.granted_at ASC
            """,
            (safe_course_id,),
        )
        rows = cursor.fetchall()
    return [map_membership(row) for row in rows]


def add_membership(
    conn: Any,
    *,
    course_id: str,
    user_id: str,
    granted_by: str | None,
    now: datetime | None = None,
) -> bool:
    """Grant instructor membership. Returns False when it already existed."""
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO course_memberships (
                course_id, user_id, membership_role, granted_by, granted_at
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (course_id, user_id) DO NOTHING
            """,
            (
                safe_course_id,
                user_id,
                INSTRUCTOR,
                granted_by,
                (now or datetime.now(timezone.utc)).isoformat(),
            ),
        )
        return cursor.rowcount > 0


def remove_membership(conn: Any, *, course_id: str, user_id: str) -> bool:
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            "DELETE FROM course_memberships WHERE course_id = %s AND user_id = %s",
            (safe_course_id, user_id),
        )
        return cursor.rowcount > 0
