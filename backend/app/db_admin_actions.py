"""PostgreSQL repository for the privileged-action audit trail.

Every invitation created or revoked, every role or membership change, every
account disabled, every bulk deletion of research data leaves a row here with
who did it and to what. The rows are append-only; nothing in the application
updates or deletes them.

What never goes in: a password, a token, a session secret, a classroom code.
`record_action` strips any detail key that looks like one, so a caller cannot
log a credential by accident.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from app.db_mapping import bind_jsonb, put_optional, to_iso

ACTION_COLUMNS = """
    action_id, actor_user_id, actor_role, action, target_kind, target_id,
    course_id, detail, created_at
"""

#: Detail keys containing any of these are dropped before the row is written.
SENSITIVE_KEY_FRAGMENTS = ("token", "password", "secret", "hash", "cookie", "code")

JSONB_COLUMNS = frozenset({"detail"})


def scrub_detail(detail: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Drop anything that looks like a credential, at any depth."""
    if not detail:
        return None
    cleaned: dict[str, Any] = {}
    for key, value in detail.items():
        lowered = str(key).lower()
        if any(fragment in lowered for fragment in SENSITIVE_KEY_FRAGMENTS):
            continue
        if isinstance(value, Mapping):
            nested = scrub_detail(value)
            if nested:
                cleaned[str(key)] = nested
            continue
        cleaned[str(key)] = value
    return cleaned or None


def map_action(row: Mapping[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "actionId": int(row["action_id"]),
        "action": row["action"],
        "createdAt": to_iso(row.get("created_at")),
    }
    actor = row.get("actor_user_id")
    put_optional(record, "actorUserId", str(actor) if actor is not None else None)
    put_optional(record, "actorRole", row.get("actor_role"))
    put_optional(record, "actorName", row.get("actor_name"))
    put_optional(record, "actorEmail", row.get("actor_email"))
    put_optional(record, "targetKind", row.get("target_kind"))
    put_optional(record, "targetId", row.get("target_id"))
    put_optional(record, "courseId", row.get("course_id"))
    if row.get("detail"):
        record["detail"] = row["detail"]
    return record


def record_action(
    conn: Any,
    *,
    actor_user_id: str | None,
    actor_role: str | None,
    action: str,
    target_kind: str | None = None,
    target_id: str | None = None,
    course_id: str | None = None,
    detail: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> None:
    if not action or not action.strip():
        raise ValueError("An audit action needs a name.")
    parameters = bind_jsonb(
        {
            "actor_user_id": actor_user_id,
            "actor_role": actor_role,
            "action": action.strip(),
            "target_kind": target_kind,
            "target_id": target_id,
            "course_id": course_id,
            "detail": scrub_detail(detail),
            "created_at": (now or datetime.now(timezone.utc)).isoformat(),
        },
        JSONB_COLUMNS,
    )
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO admin_actions (
                actor_user_id, actor_role, action, target_kind, target_id,
                course_id, detail, created_at
            ) VALUES (
                %(actor_user_id)s, %(actor_role)s, %(action)s, %(target_kind)s,
                %(target_id)s, %(course_id)s, %(detail)s, %(created_at)s
            )
            """,
            parameters,
        )


SEED_REVIEW_ACTION = "seed.review"


def record_seed_review(
    conn: Any,
    *,
    actor_user_id: str | None,
    actor_role: str | None,
    course_id: str,
    seed_id: str,
    review_status: str,
    text_edited: bool,
    notes_changed: bool = False,
) -> None:
    """One administrator's review decision on one example.

    Professors reviewing their own courses are doing the job the course
    membership exists for, and are not audited. An administrator reviewing is
    acting for a course's instructors from outside the membership table, which
    is exactly the kind of privileged, cross-course action this trail is for.
    The decision and whether the text changed are recorded; the question and
    answer themselves stay in `seed_examples`, where the edit history already
    keeps the original wording.
    """
    record_action(
        conn,
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        action=SEED_REVIEW_ACTION,
        target_kind="seed",
        target_id=seed_id,
        course_id=course_id,
        detail={
            "reviewStatus": review_status,
            "textEdited": bool(text_edited),
            "notesChanged": bool(notes_changed),
        },
    )


def list_actions(
    conn: Any, *, limit: int = 100, before_id: int | None = None
) -> list[dict[str, Any]]:
    """Newest first, paged by action id."""
    bounded = max(1, min(int(limit), 500))
    with conn.cursor() as cursor:
        if before_id is None:
            cursor.execute(
                """
                SELECT a.action_id, a.actor_user_id, a.actor_role, a.action,
                       a.target_kind, a.target_id, a.course_id, a.detail,
                       a.created_at, u.display_name AS actor_name,
                       u.email AS actor_email
                FROM admin_actions a
                LEFT JOIN users u ON u.user_id = a.actor_user_id
                ORDER BY a.action_id DESC
                LIMIT %s
                """,
                (bounded,),
            )
        else:
            cursor.execute(
                """
                SELECT a.action_id, a.actor_user_id, a.actor_role, a.action,
                       a.target_kind, a.target_id, a.course_id, a.detail,
                       a.created_at, u.display_name AS actor_name,
                       u.email AS actor_email
                FROM admin_actions a
                LEFT JOIN users u ON u.user_id = a.actor_user_id
                WHERE a.action_id < %s
                ORDER BY a.action_id DESC
                LIMIT %s
                """,
                (int(before_id), bounded),
            )
        rows = cursor.fetchall()
    return [map_action(row) for row in rows]
