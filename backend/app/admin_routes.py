"""Administration: `/api/admin`.

People and their access. Every route here takes `require_admin`, so a
professor session is refused with 403 before any handler runs — there is no
code path in the application by which a professor creates a professor or
administrator invitation, changes a role, or edits a membership.

    GET    /api/admin/users                              accounts with their courses
    PATCH  /api/admin/users/{user_id}                    disable/enable, change role
    PUT    /api/admin/users/{user_id}/courses/{course_id}   grant instructor membership
    DELETE /api/admin/users/{user_id}/courses/{course_id}   remove it
    POST   /api/admin/users/{user_id}/reset-invite        one-time password reset link
    POST   /api/admin/invitations                         professor or admin invitation
    GET    /api/admin/invitations                         those, newest first
    POST   /api/admin/invitations/{invitation_id}/revoke
    GET    /api/admin/audit                               admin_actions, newest first

Every mutation writes to `admin_actions`. Invitation tokens are returned
exactly once, in the creating response, and are stored only as a hash.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from app import (
    db_admin_actions,
    db_courses,
    db_invitations,
    db_memberships,
    db_sessions,
    db_users,
)
from app.auth.bootstrap import INVITE_PATH_PREFIX
from app.auth.dependencies import require_admin
from app.auth.principal import Principal
from app.auth.settings import (
    admin_invitation_lifetime,
    professor_invitation_lifetime,
    reset_invitation_lifetime,
)
from app.auth.tokens import generate_token, hash_token
from app.course_id import assert_valid_course_id
from app.db import db_connection, translate_db_errors

router = APIRouter(prefix="/api/admin", tags=["admin"])

MAX_PROFESSOR_INVITE = timedelta(days=30)
MAX_ADMIN_INVITE = timedelta(hours=72)
MAX_LABEL_LENGTH = 120


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run(action: str, work: Callable[[Any], Any]) -> Any:
    with translate_db_errors(action):
        with db_connection() as connection:
            return work(connection)


def _safe_course_id(course_id: str) -> str:
    try:
        return assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _audit(conn: Any, actor: Principal, action: str, **fields: Any) -> None:
    user = actor.user
    assert user is not None
    db_admin_actions.record_action(
        conn,
        actor_user_id=user.user_id,
        actor_role=user.role,
        action=action,
        **fields,
    )


# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #


class AdminModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class UserRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    user_id: str = Field(alias="userId")
    email: str
    display_name: str = Field(alias="displayName")
    role: str
    disabled: bool = False
    created_at: str | None = Field(default=None, alias="createdAt")
    last_login_at: str | None = Field(default=None, alias="lastLoginAt")
    course_ids: list[str] = Field(default_factory=list, alias="courseIds")


class UserListResponse(AdminModel):
    count: int
    users: list[UserRecord]


class UserUpdateRequest(AdminModel):
    disabled: bool | None = None
    role: str | None = None


class MembershipResponse(AdminModel):
    user_id: str = Field(alias="userId")
    course_id: str = Field(alias="courseId")
    member: bool


class InvitationCreateRequest(AdminModel):
    kind: str
    course_ids: list[str] = Field(default_factory=list, alias="courseIds")
    expires_in_hours: float | None = Field(default=None, alias="expiresInHours", gt=0)
    label: str | None = Field(default=None, max_length=MAX_LABEL_LENGTH)


class InvitationRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    invitation_id: str = Field(alias="invitationId")
    kind: str
    status: str
    created_at: str = Field(alias="createdAt")
    expires_at: str | None = Field(default=None, alias="expiresAt")
    label: str | None = None
    course_ids: list[str] = Field(default_factory=list, alias="courseIds")
    created_by_name: str | None = Field(default=None, alias="createdByName")
    accepted_at: str | None = Field(default=None, alias="acceptedAt")
    revoked_at: str | None = Field(default=None, alias="revokedAt")
    target_user_id: str | None = Field(default=None, alias="targetUserId")


class CreatedInvitationResponse(InvitationRecord):
    """The one response that carries the token. It is never returned again."""

    token: str
    path: str


class InvitationListResponse(AdminModel):
    count: int
    invitations: list[InvitationRecord]


class AuditListResponse(AdminModel):
    count: int
    actions: list[dict[str, Any]]


# --------------------------------------------------------------------------- #
# Users and memberships
# --------------------------------------------------------------------------- #


@router.get("/users", response_model=UserListResponse)
def list_users(principal: Principal = Depends(require_admin)) -> UserListResponse:
    def work(conn: Any) -> list[dict[str, Any]]:
        users = db_users.list_users(conn)
        by_user: dict[str, list[str]] = {}
        for membership in db_memberships.list_memberships(conn):
            by_user.setdefault(membership["userId"], []).append(membership["courseId"])
        return [{**user, "courseIds": sorted(by_user.get(user["userId"], []))} for user in users]

    users = _run("listing accounts", work)
    return UserListResponse(count=len(users), users=[UserRecord(**user) for user in users])


def _guard_last_admin(conn: Any, target: dict[str, Any], *, losing_admin: bool) -> None:
    """Refuse a change that would leave the system without an administrator."""
    if losing_admin and target["role"] == "admin" and not target["disabled"]:
        if db_users.count_active_admins(conn) <= 1:
            raise HTTPException(
                status_code=409,
                detail="That would leave no active administrator. Add another first.",
            )


@router.patch("/users/{user_id}", response_model=UserRecord)
def update_user(
    user_id: str,
    body: UserUpdateRequest,
    principal: Principal = Depends(require_admin),
) -> UserRecord:
    """Disable or re-enable an account, or change its global role."""
    if body.disabled is None and body.role is None:
        raise HTTPException(status_code=422, detail="Nothing to change.")
    if body.role is not None and body.role not in db_users.ROLES:
        raise HTTPException(status_code=422, detail="role must be admin or professor.")
    actor = principal.user
    assert actor is not None
    if user_id == actor.user_id and (body.disabled or (body.role and body.role != "admin")):
        raise HTTPException(
            status_code=409, detail="You cannot disable or demote your own account."
        )
    now = _utc_now()

    def work(conn: Any) -> dict[str, Any]:
        target = db_users.get_user(conn, user_id)
        if target is None:
            raise HTTPException(status_code=404, detail="That account was not found.")
        updated = target
        if body.disabled is not None and body.disabled != target["disabled"]:
            _guard_last_admin(conn, target, losing_admin=body.disabled)
            updated = db_users.set_disabled(conn, user_id, disabled=body.disabled, now=now) or target
            if body.disabled:
                db_sessions.revoke_sessions_for_user(conn, user_id, now)
            _audit(
                conn,
                principal,
                "user.disable" if body.disabled else "user.enable",
                target_kind="user",
                target_id=user_id,
                now=now,
            )
        if body.role is not None and body.role != target["role"]:
            _guard_last_admin(conn, target, losing_admin=body.role != "admin")
            updated = db_users.set_role(conn, user_id, body.role) or updated
            _audit(
                conn,
                principal,
                "user.role_change",
                target_kind="user",
                target_id=user_id,
                detail={"from": target["role"], "to": body.role},
                now=now,
            )
        course_ids = db_memberships.list_course_ids_for_user(conn, user_id)
        return {**updated, "courseIds": course_ids}

    return UserRecord(**_run("updating an account", work))


@router.put("/users/{user_id}/courses/{course_id}", response_model=MembershipResponse)
def add_membership(
    user_id: str, course_id: str, principal: Principal = Depends(require_admin)
) -> MembershipResponse:
    safe_course_id = _safe_course_id(course_id)
    now = _utc_now()
    actor = principal.user
    assert actor is not None

    def work(conn: Any) -> bool:
        if db_users.get_user(conn, user_id) is None:
            raise HTTPException(status_code=404, detail="That account was not found.")
        if not db_courses.course_exists(conn, safe_course_id):
            raise HTTPException(
                status_code=404, detail=f'Course "{safe_course_id}" was not found.'
            )
        added = db_memberships.add_membership(
            conn, course_id=safe_course_id, user_id=user_id, granted_by=actor.user_id, now=now
        )
        if added:
            _audit(
                conn,
                principal,
                "membership.add",
                target_kind="user",
                target_id=user_id,
                course_id=safe_course_id,
                now=now,
            )
        return True

    member = _run("granting course access", work)
    return MembershipResponse(userId=user_id, courseId=safe_course_id, member=member)


@router.delete("/users/{user_id}/courses/{course_id}", response_model=MembershipResponse)
def remove_membership(
    user_id: str, course_id: str, principal: Principal = Depends(require_admin)
) -> MembershipResponse:
    safe_course_id = _safe_course_id(course_id)
    now = _utc_now()

    def work(conn: Any) -> bool:
        removed = db_memberships.remove_membership(
            conn, course_id=safe_course_id, user_id=user_id
        )
        if removed:
            _audit(
                conn,
                principal,
                "membership.remove",
                target_kind="user",
                target_id=user_id,
                course_id=safe_course_id,
                now=now,
            )
        return removed

    _run("removing course access", work)
    return MembershipResponse(userId=user_id, courseId=safe_course_id, member=False)


# --------------------------------------------------------------------------- #
# Invitations
# --------------------------------------------------------------------------- #


def _lifetime_for(kind: str, requested_hours: float | None) -> timedelta:
    if kind == db_invitations.PROFESSOR:
        default, ceiling = professor_invitation_lifetime(), MAX_PROFESSOR_INVITE
    else:
        default, ceiling = admin_invitation_lifetime(), MAX_ADMIN_INVITE
    if requested_hours is None:
        return min(default, ceiling)
    requested = timedelta(hours=requested_hours)
    if requested > ceiling:
        raise HTTPException(
            status_code=422,
            detail=f"A {kind} invitation may last at most {int(ceiling.total_seconds() // 3600)} hours.",
        )
    return requested


@router.post("/invitations", response_model=CreatedInvitationResponse, status_code=201)
def create_invitation(
    body: InvitationCreateRequest, principal: Principal = Depends(require_admin)
) -> CreatedInvitationResponse:
    """Mint a single-use professor or administrator invitation.

    The kind is taken from the body but the *permission* to mint either kind
    comes from `require_admin`; there is no route a professor can reach that
    accepts `kind`.
    """
    if body.kind not in (db_invitations.PROFESSOR, db_invitations.ADMIN):
        raise HTTPException(status_code=422, detail="kind must be professor or admin.")
    course_ids = [_safe_course_id(course_id) for course_id in body.course_ids]
    if body.kind == db_invitations.ADMIN and course_ids:
        raise HTTPException(
            status_code=422, detail="An admin invitation does not take courses."
        )
    lifetime = _lifetime_for(body.kind, body.expires_in_hours)
    now = _utc_now()
    actor = principal.user
    assert actor is not None

    def work(conn: Any) -> dict[str, Any]:
        for course_id in course_ids:
            if not db_courses.course_exists(conn, course_id):
                raise HTTPException(
                    status_code=404, detail=f'Course "{course_id}" was not found.'
                )
        token = generate_token()
        created = db_invitations.create_token_invitation(
            conn,
            kind=body.kind,
            token_hash=hash_token(token),
            created_by=actor.user_id,
            expires_at=now + lifetime,
            now=now,
            course_ids=course_ids,
            label=body.label,
        )
        _audit(
            conn,
            principal,
            "invitation.create",
            target_kind="invitation",
            target_id=created["invitationId"],
            detail={"kind": body.kind, "courseIds": course_ids, "label": body.label},
            now=now,
        )
        return {
            **created,
            "courseIds": course_ids,
            "createdByName": actor.display_name,
            "token": token,
            "path": INVITE_PATH_PREFIX + token,
        }

    return CreatedInvitationResponse(**_run("creating an invitation", work))


@router.get("/invitations", response_model=InvitationListResponse)
def list_invitations(
    principal: Principal = Depends(require_admin),
    status: str | None = Query(default=None),
) -> InvitationListResponse:
    now = _utc_now()
    records = _run(
        "listing invitations",
        lambda conn: db_invitations.list_invitations(conn, now=now),
    )
    if status:
        records = [record for record in records if record["status"] == status]
    return InvitationListResponse(
        count=len(records), invitations=[InvitationRecord(**record) for record in records]
    )


@router.post("/invitations/{invitation_id}/revoke", response_model=InvitationRecord)
def revoke_invitation(
    invitation_id: str, principal: Principal = Depends(require_admin)
) -> InvitationRecord:
    now = _utc_now()

    def work(conn: Any) -> dict[str, Any]:
        existing = db_invitations.get_invitation(conn, invitation_id, now=now)
        if existing is None or existing["kind"] not in db_invitations.TOKEN_KINDS:
            raise HTTPException(status_code=404, detail="That invitation was not found.")
        revoked = db_invitations.revoke(
            conn, invitation_id, revoked_by=principal.user.user_id, now=now  # type: ignore[union-attr]
        )
        assert revoked is not None
        if existing["status"] != db_invitations.REVOKED:
            _audit(
                conn,
                principal,
                "invitation.revoke",
                target_kind="invitation",
                target_id=invitation_id,
                detail={"kind": existing["kind"]},
                now=now,
            )
        if revoked["kind"] == db_invitations.PROFESSOR:
            revoked["courseIds"] = db_invitations.list_grants(conn, invitation_id)
        return revoked

    return InvitationRecord(**_run("revoking an invitation", work))


@router.post(
    "/users/{user_id}/reset-invite", response_model=CreatedInvitationResponse, status_code=201
)
def create_reset_invitation(
    user_id: str, principal: Principal = Depends(require_admin)
) -> CreatedInvitationResponse:
    """A one-time link that lets an existing account set a new password.

    The administrator hands the link to the person out of band; there is no
    mail path. Accepting it ends every session the account had.
    """
    now = _utc_now()
    actor = principal.user
    assert actor is not None

    def work(conn: Any) -> dict[str, Any]:
        target = db_users.get_user(conn, user_id)
        if target is None:
            raise HTTPException(status_code=404, detail="That account was not found.")
        if target["disabled"]:
            raise HTTPException(status_code=409, detail="That account is disabled.")
        token = generate_token()
        created = db_invitations.create_token_invitation(
            conn,
            kind=db_invitations.RESET,
            token_hash=hash_token(token),
            created_by=actor.user_id,
            expires_at=now + reset_invitation_lifetime(),
            now=now,
            target_user_id=user_id,
            label=f"reset for {target['email']}",
        )
        _audit(
            conn,
            principal,
            "user.reset_invite",
            target_kind="user",
            target_id=user_id,
            detail={"invitationId": created["invitationId"]},
            now=now,
        )
        return {
            **created,
            "createdByName": actor.display_name,
            "token": token,
            "path": INVITE_PATH_PREFIX + token,
        }

    return CreatedInvitationResponse(**_run("creating a reset link", work))


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #


@router.get("/audit", response_model=AuditListResponse)
def list_audit(
    principal: Principal = Depends(require_admin),
    limit: int = Query(default=100, ge=1, le=500),
    before: int | None = Query(default=None, ge=1),
) -> AuditListResponse:
    actions = _run(
        "reading the audit trail",
        lambda conn: db_admin_actions.list_actions(conn, limit=limit, before_id=before),
    )
    return AuditListResponse(count=len(actions), actions=actions)
