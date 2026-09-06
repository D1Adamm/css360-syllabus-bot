"""Classroom codes for one course: `/api/courses/{course_id}/student-invites`.

What a professor sees on the Invite page and an administrator sees on the
course page: the active code, its metadata, and the actions to create,
replace and revoke. Every route is course-staff only, and the course comes
from the path, so a professor can only ever manage codes for a course they
hold a membership in — the same guard every other course-scoped route uses.

The code itself is returned here, on every read, because it has to be shown
again. Nothing else secret exists for a student invitation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app import db_admin_actions, db_courses, db_invitations, db_participants
from app.auth.codes import generate_code
from app.auth.dependencies import require_course_staff
from app.auth.principal import Principal
from app.db import db_connection, translate_db_errors

router = APIRouter(prefix="/api/courses/{course_id}/student-invites", tags=["student-invites"])

#: Draws before giving up on a code collision. With ~887 million codes and a
#: handful in use, the second draw is already astronomically unlikely.
CODE_DRAW_ATTEMPTS = 5
MAX_LABEL_LENGTH = 120


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run(action: str, work: Callable[[Any], Any]) -> Any:
    with translate_db_errors(action):
        with db_connection() as connection:
            return work(connection)


class InviteModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class StudentInviteCreateRequest(InviteModel):
    label: str | None = Field(default=None, max_length=MAX_LABEL_LENGTH)
    #: ISO 8601. Omit for a code that lasts until it is revoked.
    expires_at: str | None = Field(default=None, alias="expiresAt")
    #: Revoke every other live code for the course in the same transaction —
    #: "replace the code" rather than "add another".
    replace_existing: bool = Field(default=False, alias="replaceExisting")


class StudentInviteRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    invitation_id: str = Field(alias="invitationId")
    course_id: str = Field(alias="courseId")
    code: str
    status: str
    created_at: str = Field(alias="createdAt")
    use_count: int = Field(default=0, alias="useCount")
    participant_count: int = Field(default=0, alias="participantCount")
    label: str | None = None
    expires_at: str | None = Field(default=None, alias="expiresAt")
    revoked_at: str | None = Field(default=None, alias="revokedAt")
    created_by_name: str | None = Field(default=None, alias="createdByName")


class StudentInviteListResponse(InviteModel):
    course_id: str = Field(alias="courseId")
    count: int
    invites: list[StudentInviteRecord]


def parse_expiry(value: str | None, now: datetime) -> datetime | None:
    if value is None or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail="expiresAt must be an ISO 8601 timestamp."
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if parsed <= now:
        raise HTTPException(status_code=422, detail="expiresAt must be in the future.")
    return parsed


def _with_participant_count(conn: Any, record: dict[str, Any]) -> dict[str, Any]:
    return {
        **record,
        "participantCount": db_participants.count_participants_for_invitation(
            conn, record["invitationId"]
        ),
    }


@router.get("", response_model=StudentInviteListResponse)
def list_student_invites(
    course_id: str, principal: Principal = Depends(require_course_staff)
) -> StudentInviteListResponse:
    """Every code ever issued for the course, newest first, live ones included."""
    now = _utc_now()

    def work(conn: Any) -> list[dict[str, Any]]:
        return [
            _with_participant_count(conn, record)
            for record in db_invitations.list_student_invitations(conn, course_id, now=now)
        ]

    invites = _run("listing classroom codes", work)
    return StudentInviteListResponse(
        courseId=course_id,
        count=len(invites),
        invites=[StudentInviteRecord(**record) for record in invites],
    )


@router.post("", response_model=StudentInviteRecord, status_code=201)
def create_student_invite(
    course_id: str,
    body: StudentInviteCreateRequest,
    principal: Principal = Depends(require_course_staff),
) -> StudentInviteRecord:
    """Issue a classroom code. With `replaceExisting`, retire the current ones first."""
    now = _utc_now()
    expires_at = parse_expiry(body.expires_at, now)
    actor = principal.user
    assert actor is not None

    def work(conn: Any) -> dict[str, Any]:
        if not db_courses.course_exists(conn, course_id):
            raise HTTPException(status_code=404, detail=f'Course "{course_id}" was not found.')
        replaced = 0
        if body.replace_existing:
            replaced = db_invitations.revoke_active_student_invitations(
                conn, course_id, revoked_by=actor.user_id, now=now
            )
        created: dict[str, Any] | None = None
        for _ in range(CODE_DRAW_ATTEMPTS):
            try:
                created = db_invitations.create_student_invitation(
                    conn,
                    course_id=course_id,
                    code=generate_code(),
                    created_by=actor.user_id,
                    label=body.label,
                    expires_at=expires_at,
                    now=now,
                )
                break
            except db_invitations.CodeCollisionError:
                continue
        if created is None:  # pragma: no cover - astronomically unlikely
            raise HTTPException(
                status_code=503, detail="Could not allocate a classroom code. Try again."
            )
        db_admin_actions.record_action(
            conn,
            actor_user_id=actor.user_id,
            actor_role=actor.role,
            action="invitation.create",
            target_kind="invitation",
            target_id=created["invitationId"],
            course_id=course_id,
            detail={
                "kind": "student",
                "label": body.label,
                "expiresAt": created.get("expiresAt"),
                "replaced": replaced,
            },
            now=now,
        )
        return {**created, "createdByName": actor.display_name, "participantCount": 0}

    return StudentInviteRecord(**_run("creating a classroom code", work))


@router.post("/{invitation_id}/revoke", response_model=StudentInviteRecord)
def revoke_student_invite(
    course_id: str,
    invitation_id: str,
    principal: Principal = Depends(require_course_staff),
) -> StudentInviteRecord:
    """Retire a code. Participants who already joined keep their access."""
    now = _utc_now()
    actor = principal.user
    assert actor is not None

    def work(conn: Any) -> dict[str, Any]:
        existing = db_invitations.get_invitation(conn, invitation_id, now=now)
        # A code belonging to another course is "not found" here, not "forbidden":
        # the path names a course, and this id is not one of its codes.
        if (
            existing is None
            or existing["kind"] != db_invitations.STUDENT
            or existing.get("courseId") != course_id
        ):
            raise HTTPException(status_code=404, detail="That classroom code was not found.")
        revoked = db_invitations.revoke(
            conn, invitation_id, revoked_by=actor.user_id, now=now
        )
        assert revoked is not None
        if existing["status"] != db_invitations.REVOKED:
            db_admin_actions.record_action(
                conn,
                actor_user_id=actor.user_id,
                actor_role=actor.role,
                action="invitation.revoke",
                target_kind="invitation",
                target_id=invitation_id,
                course_id=course_id,
                detail={"kind": "student"},
                now=now,
            )
        return _with_participant_count(conn, revoked)

    return StudentInviteRecord(**_run("revoking a classroom code", work))
