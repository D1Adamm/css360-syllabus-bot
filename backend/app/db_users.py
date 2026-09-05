"""PostgreSQL repository for professor and administrator accounts.

The only people with accounts. Students never appear here; they are
`participants`, and the two are kept apart on purpose — a participant row
cannot be joined to a name because there is no column to join it on.

`password_hash` leaves this module through exactly one function,
`find_credentials`, which the login route calls and nothing else does. Every
other read maps a row to the public record and drops the hash.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

from app.db_mapping import optional_string, put_optional, to_iso

ROLES = ("admin", "professor")

USER_COLUMNS = """
    user_id, email, display_name, role, created_at, disabled_at,
    last_login_at, created_via_invitation_id
"""


class UserAlreadyExistsError(Exception):
    """Raised when an email address is already an account."""


def new_user_id() -> str:
    return str(uuid.uuid4())


def normalize_email(email: object) -> str:
    return str(email or "").strip().lower()


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(timezone.utc)


def map_user(row: Mapping[str, Any]) -> dict[str, Any]:
    """One `users` row as the API's user record. Never includes the hash."""
    record: dict[str, Any] = {
        "userId": str(row["user_id"]),
        "email": row["email"],
        "displayName": row["display_name"],
        "role": row["role"],
        "createdAt": to_iso(row.get("created_at")),
        "disabled": row.get("disabled_at") is not None,
    }
    put_optional(record, "disabledAt", to_iso(row.get("disabled_at")))
    put_optional(record, "lastLoginAt", to_iso(row.get("last_login_at")))
    invitation_id = row.get("created_via_invitation_id")
    put_optional(
        record,
        "createdViaInvitationId",
        str(invitation_id) if invitation_id is not None else None,
    )
    return record


def create_user(
    conn: Any,
    *,
    email: str,
    display_name: str,
    role: str,
    password_hash: str,
    invitation_id: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Insert one account. Raises UserAlreadyExistsError on a duplicate email.

    `ON CONFLICT DO NOTHING` against the case-insensitive unique index plus a
    rowcount check, rather than a prior SELECT: two concurrent invitation
    acceptances for the same address cannot both see an empty table.
    """
    if role not in ROLES:
        raise ValueError(f"Unknown role {role!r}.")
    normalized_email = normalize_email(email)
    if not normalized_email or "@" not in normalized_email:
        raise ValueError("A valid email address is required.")
    cleaned_name = optional_string(display_name)
    if not cleaned_name:
        raise ValueError("A display name is required.")

    parameters = {
        "user_id": new_user_id(),
        "email": normalized_email,
        "display_name": cleaned_name,
        "role": role,
        "password_hash": password_hash,
        "created_at": _now(now).isoformat(),
        "created_via_invitation_id": invitation_id,
    }
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO users (
                user_id, email, display_name, role, password_hash, created_at,
                created_via_invitation_id
            ) VALUES (
                %(user_id)s, %(email)s, %(display_name)s, %(role)s,
                %(password_hash)s, %(created_at)s, %(created_via_invitation_id)s
            )
            ON CONFLICT (lower(email)) DO NOTHING
            """,
            parameters,
        )
        if cursor.rowcount == 0:
            raise UserAlreadyExistsError(
                "An account with that email address already exists."
            )

    created = get_user(conn, parameters["user_id"])
    if created is None:  # pragma: no cover - defensive
        raise UserAlreadyExistsError("The account could not be read back after insert.")
    return created


def get_user(conn: Any, user_id: str) -> dict[str, Any] | None:
    with conn.cursor() as cursor:
        cursor.execute(
            f"SELECT {USER_COLUMNS} FROM users WHERE user_id = %s", (user_id,)
        )
        row = cursor.fetchone()
    return map_user(row) if row else None


def find_credentials(conn: Any, email: str) -> dict[str, Any] | None:
    """The one read that returns a password hash. For the login route only."""
    normalized_email = normalize_email(email)
    if not normalized_email:
        return None
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT user_id, role, password_hash, disabled_at FROM users "
            "WHERE lower(email) = %s",
            (normalized_email,),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    return {
        "userId": str(row["user_id"]),
        "role": row["role"],
        "passwordHash": row["password_hash"],
        "disabled": row.get("disabled_at") is not None,
    }


def list_users(conn: Any) -> list[dict[str, Any]]:
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT {USER_COLUMNS} FROM users ORDER BY created_at ASC, email ASC")
        rows = cursor.fetchall()
    return [map_user(row) for row in rows]


def set_password_hash(conn: Any, user_id: str, password_hash: str) -> bool:
    with conn.cursor() as cursor:
        cursor.execute(
            "UPDATE users SET password_hash = %s WHERE user_id = %s",
            (password_hash, user_id),
        )
        return cursor.rowcount > 0


def set_disabled(
    conn: Any, user_id: str, *, disabled: bool, now: datetime | None = None
) -> dict[str, Any] | None:
    with conn.cursor() as cursor:
        cursor.execute(
            "UPDATE users SET disabled_at = %s WHERE user_id = %s",
            (_now(now).isoformat() if disabled else None, user_id),
        )
        if cursor.rowcount == 0:
            return None
    return get_user(conn, user_id)


def set_role(conn: Any, user_id: str, role: str) -> dict[str, Any] | None:
    if role not in ROLES:
        raise ValueError(f"Unknown role {role!r}.")
    with conn.cursor() as cursor:
        cursor.execute(
            "UPDATE users SET role = %s WHERE user_id = %s", (role, user_id)
        )
        if cursor.rowcount == 0:
            return None
    return get_user(conn, user_id)


def record_login(conn: Any, user_id: str, now: datetime | None = None) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            "UPDATE users SET last_login_at = %s WHERE user_id = %s",
            (_now(now).isoformat(), user_id),
        )


def count_active_admins(conn: Any) -> int:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) AS total FROM users "
            "WHERE role = 'admin' AND disabled_at IS NULL"
        )
        row = cursor.fetchone()
    return int(row["total"]) if row else 0
