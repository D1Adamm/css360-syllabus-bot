"""PostgreSQL repository for sessions, staff and participant alike.

A session row is what a cookie points at. The cookie holds a random token;
the row holds its SHA-256, the principal it belongs to, and the three
timestamps that decide whether it is still good: an absolute deadline, the
last time it was used, and whether it was revoked. Because the row is the
authority, "sign out everywhere", disabling an account, and changing a
password can all end sessions immediately.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from app.db_mapping import to_iso

USER = "user"
PARTICIPANT = "participant"


def new_session_id() -> str:
    return str(uuid.uuid4())


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(timezone.utc)


def create_session(
    conn: Any,
    *,
    kind: str,
    token_hash: str,
    lifetime: timedelta,
    now: datetime | None = None,
    user_id: str | None = None,
    participant_id: str | None = None,
) -> dict[str, Any]:
    if kind == USER and (user_id is None or participant_id is not None):
        raise ValueError("A user session needs a user and no participant.")
    if kind == PARTICIPANT and (participant_id is None or user_id is not None):
        raise ValueError("A participant session needs a participant and no user.")
    if kind not in (USER, PARTICIPANT):
        raise ValueError(f"Unknown session kind {kind!r}.")

    moment = _now(now)
    expires_at = moment + lifetime
    session_id = new_session_id()
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO auth_sessions (
                session_id, token_hash, principal_kind, user_id, participant_id,
                created_at, expires_at, last_seen_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                session_id,
                token_hash,
                kind,
                user_id,
                participant_id,
                moment.isoformat(),
                expires_at.isoformat(),
                moment.isoformat(),
            ),
        )
    return {"sessionId": session_id, "expiresAt": to_iso(expires_at)}


def find_staff_session(conn: Any, token_hash: str) -> dict[str, Any] | None:
    """The session row joined to its account, for the principal resolver.

    Returns the raw columns — the resolver applies the expiry, idle and
    disabled rules itself so that they live in one place and can be tested
    without a database.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT s.session_id, s.expires_at, s.last_seen_at, s.revoked_at,
                   u.user_id, u.email, u.display_name, u.role, u.disabled_at
            FROM auth_sessions s
            JOIN users u ON u.user_id = s.user_id
            WHERE s.token_hash = %s AND s.principal_kind = 'user'
            """,
            (token_hash,),
        )
        return cursor.fetchone()


def find_participant_session(conn: Any, token_hash: str) -> dict[str, Any] | None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT s.session_id, s.expires_at, s.last_seen_at, s.revoked_at,
                   p.participant_id, p.course_id
            FROM auth_sessions s
            JOIN participants p ON p.participant_id = s.participant_id
            WHERE s.token_hash = %s AND s.principal_kind = 'participant'
            """,
            (token_hash,),
        )
        return cursor.fetchone()


def touch_session(conn: Any, session_id: str, now: datetime | None = None) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            "UPDATE auth_sessions SET last_seen_at = %s WHERE session_id = %s",
            (_now(now).isoformat(), session_id),
        )


def revoke_session(conn: Any, session_id: str, now: datetime | None = None) -> bool:
    with conn.cursor() as cursor:
        cursor.execute(
            "UPDATE auth_sessions SET revoked_at = %s "
            "WHERE session_id = %s AND revoked_at IS NULL",
            (_now(now).isoformat(), session_id),
        )
        return cursor.rowcount > 0


def revoke_sessions_for_user(
    conn: Any,
    user_id: str,
    now: datetime | None = None,
    *,
    keep_session_id: str | None = None,
) -> int:
    """End every session of one account, optionally sparing the current one."""
    with conn.cursor() as cursor:
        if keep_session_id is None:
            cursor.execute(
                "UPDATE auth_sessions SET revoked_at = %s "
                "WHERE user_id = %s AND revoked_at IS NULL",
                (_now(now).isoformat(), user_id),
            )
        else:
            cursor.execute(
                "UPDATE auth_sessions SET revoked_at = %s "
                "WHERE user_id = %s AND revoked_at IS NULL AND session_id <> %s",
                (_now(now).isoformat(), user_id, keep_session_id),
            )
        return max(0, cursor.rowcount)


def revoke_sessions_for_participant(
    conn: Any, participant_id: str, now: datetime | None = None
) -> int:
    with conn.cursor() as cursor:
        cursor.execute(
            "UPDATE auth_sessions SET revoked_at = %s "
            "WHERE participant_id = %s AND revoked_at IS NULL",
            (_now(now).isoformat(), participant_id),
        )
        return max(0, cursor.rowcount)


def delete_stale_sessions(
    conn: Any,
    now: datetime | None = None,
    *,
    grace: timedelta = timedelta(days=30),
) -> int:
    """Housekeeping: drop rows long past any use. Called opportunistically."""
    cutoff = (_now(now) - grace).isoformat()
    with conn.cursor() as cursor:
        cursor.execute(
            "DELETE FROM auth_sessions "
            "WHERE expires_at < %s OR (revoked_at IS NOT NULL AND revoked_at < %s)",
            (cutoff, cutoff),
        )
        return max(0, cursor.rowcount)
