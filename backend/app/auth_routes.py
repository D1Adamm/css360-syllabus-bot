"""Establishing and ending sessions: `/api/auth`.

The only routes that create a principal. Everything else in the application
takes one that already exists.

    POST /api/auth/login                       professor/admin sign-in
    POST /api/auth/logout                      end every session in this browser
    GET  /api/auth/session                     who am I, for the frontend
    POST /api/auth/join                        redeem a classroom code
    GET  /api/auth/invitations/{token}         preview a privileged invitation
    POST /api/auth/invitations/{token}/accept  create the account, or reset a password
    POST /api/auth/password                    change my own password

Errors are deliberately uninformative where a guess could be refined by them:
a wrong password and an unknown email produce the same 401, and an invalid,
expired, revoked or exhausted invitation produces the same 404. Every
credential-bearing route is throttled on failures.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from app import (
    db_admin_actions,
    db_courses,
    db_invitations,
    db_memberships,
    db_participants,
    db_sessions,
    db_users,
)
from app.auth import rate_limit
from app.auth.codes import normalize_code
from app.auth.cookies import clear_session_cookie, set_session_cookie
from app.auth.dependencies import current_principal, require_csrf, require_user
from app.auth.passwords import (
    hash_password,
    needs_rehash,
    password_problem,
    verify_password,
)
from app.auth.principal import Principal
from app.auth.settings import (
    PARTICIPANT_COOKIE_NAME,
    STAFF_COOKIE_NAME,
    participant_session_lifetime,
    staff_session_lifetime,
)
from app.auth.tokens import generate_token, hash_token, is_plausible_token
from app.db import db_connection, translate_db_errors

router = APIRouter(prefix="/api/auth", tags=["auth"])

LOGIN_FAILED_DETAIL = "The email address or password is not correct."
CODE_INVALID_DETAIL = (
    "That code isn't valid right now. Check it with your instructor and try again."
)
INVITATION_INVALID_DETAIL = (
    "This invitation link is not valid. Ask the administrator who sent it for a new one."
)
TOO_MANY_ATTEMPTS_DETAIL = "Too many attempts. Wait a little and try again."
EMAIL_TAKEN_DETAIL = (
    "An account with that email address already exists. Sign in instead, or ask "
    "an administrator for a password reset link."
)

#: Verified against when an email is unknown, so that path costs the same time
#: as a real verification and timing cannot say whether an account exists.
_DUMMY_HASH: str | None = None


def _dummy_hash() -> str:
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password(generate_token())
    return _DUMMY_HASH


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run(action: str, work: Callable[[Any], Any]) -> Any:
    """One connection, one transaction, driver errors mapped to 503."""
    with translate_db_errors(action):
        with db_connection() as connection:
            return work(connection)


def _throttle(limiter: rate_limit.RateLimiter, key: str) -> None:
    retry_after = limiter.retry_after(key)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail=TOO_MANY_ATTEMPTS_DETAIL,
            headers={"Retry-After": str(int(retry_after) + 1)},
        )


# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #


class AuthModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class LoginRequest(AuthModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=1024)


class UserSession(AuthModel):
    user_id: str = Field(alias="userId")
    email: str
    display_name: str = Field(alias="displayName")
    role: str
    course_ids: list[str] = Field(default_factory=list, alias="courseIds")


class ParticipantSession(AuthModel):
    """What a student's browser is told about itself: the course, nothing else."""

    course_id: str = Field(alias="courseId")


class SessionResponse(AuthModel):
    user: UserSession | None = None
    participant: ParticipantSession | None = None


class JoinRequest(AuthModel):
    code: str = Field(min_length=1, max_length=200)


class JoinResponse(AuthModel):
    course_id: str = Field(alias="courseId")
    course_name: str | None = Field(default=None, alias="courseName")
    already_joined: bool = Field(default=False, alias="alreadyJoined")


class InvitationCourse(AuthModel):
    course_id: str = Field(alias="courseId")
    name: str | None = None


class InvitationPreview(AuthModel):
    kind: str
    label: str | None = None
    expires_at: str | None = Field(default=None, alias="expiresAt")
    courses: list[InvitationCourse] = Field(default_factory=list)
    target_email: str | None = Field(default=None, alias="targetEmail")


class AcceptInvitationRequest(AuthModel):
    email: str | None = Field(default=None, max_length=320)
    display_name: str | None = Field(default=None, alias="displayName", max_length=200)
    password: str = Field(min_length=1, max_length=1024)


class PasswordChangeRequest(AuthModel):
    current_password: str = Field(alias="currentPassword", min_length=1, max_length=1024)
    new_password: str = Field(alias="newPassword", min_length=1, max_length=1024)


def session_response(principal: Principal) -> SessionResponse:
    user = None
    if principal.user is not None:
        user = UserSession(
            userId=principal.user.user_id,
            email=principal.user.email,
            displayName=principal.user.display_name,
            role=principal.user.role,
            courseIds=sorted(principal.user.course_ids),
        )
    participant = None
    if principal.participant is not None:
        participant = ParticipantSession(courseId=principal.participant.course_id)
    return SessionResponse(user=user, participant=participant)


def _staff_principal(conn: Any, user_id: str, session_id: str) -> Principal:
    """Build the principal a freshly created staff session represents."""
    from app.auth.principal import ADMIN, StaffUser

    record = db_users.get_user(conn, user_id)
    if record is None:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="The account could not be read back.")
    course_ids = (
        frozenset()
        if record["role"] == ADMIN
        else frozenset(db_memberships.list_course_ids_for_user(conn, user_id))
    )
    return Principal(
        user=StaffUser(
            user_id=user_id,
            email=record["email"],
            display_name=record["displayName"],
            role=record["role"],
            course_ids=course_ids,
            session_id=session_id,
        )
    )


# --------------------------------------------------------------------------- #
# Sign in / sign out
# --------------------------------------------------------------------------- #


@router.post("/login", status_code=204, dependencies=[Depends(require_csrf)])
def login(body: LoginRequest, request: Request, response: Response) -> Response:
    email = db_users.normalize_email(body.email)
    address = rate_limit.client_address(request)
    _throttle(rate_limit.LOGIN_PER_CLIENT, address)
    _throttle(rate_limit.LOGIN_PER_ACCOUNT, email)
    now = _utc_now()

    def work(conn: Any) -> str | None:
        credentials = db_users.find_credentials(conn, email)
        if credentials is None:
            verify_password(body.password, _dummy_hash())
            return None
        if credentials["disabled"]:
            verify_password(body.password, _dummy_hash())
            return None
        if not verify_password(body.password, credentials["passwordHash"]):
            return None

        user_id = credentials["userId"]
        if needs_rehash(credentials["passwordHash"]):
            db_users.set_password_hash(conn, user_id, hash_password(body.password))
        token = generate_token()
        db_sessions.create_session(
            conn,
            kind=db_sessions.USER,
            token_hash=hash_token(token),
            lifetime=staff_session_lifetime(),
            now=now,
            user_id=user_id,
        )
        db_users.record_login(conn, user_id, now)
        db_sessions.delete_stale_sessions(conn, now)
        return token

    token = _run("signing in", work)
    if token is None:
        rate_limit.LOGIN_PER_CLIENT.record_failure(address)
        rate_limit.LOGIN_PER_ACCOUNT.record_failure(email)
        raise HTTPException(status_code=401, detail=LOGIN_FAILED_DETAIL)

    rate_limit.LOGIN_PER_ACCOUNT.reset(email)
    result = Response(status_code=204)
    set_session_cookie(result, STAFF_COOKIE_NAME, token, staff_session_lifetime())
    return result


@router.post("/logout", status_code=204)
def logout(principal: Principal = Depends(current_principal)) -> Response:
    """End every session this browser holds and clear both cookies.

    Sign out means nothing is left behind on this machine — the staff session
    and, if the professor had joined their own course, the participant one.
    """
    now = _utc_now()
    sessions = [
        session_id
        for session_id in (
            principal.user.session_id if principal.user else None,
            principal.participant.session_id if principal.participant else None,
        )
        if session_id
    ]
    if sessions:

        def work(conn: Any) -> None:
            for session_id in sessions:
                db_sessions.revoke_session(conn, session_id, now)

        _run("signing out", work)

    result = Response(status_code=204)
    clear_session_cookie(result, STAFF_COOKIE_NAME)
    clear_session_cookie(result, PARTICIPANT_COOKIE_NAME)
    return result


@router.get("/session", response_model=SessionResponse)
def read_session(principal: Principal = Depends(current_principal)) -> SessionResponse:
    """Who this browser is. `{user: null, participant: null}` is a normal answer."""
    return session_response(principal)


# --------------------------------------------------------------------------- #
# Students: classroom codes
# --------------------------------------------------------------------------- #


@router.post("/join", response_model=JoinResponse, dependencies=[Depends(require_csrf)])
def join_course(
    body: JoinRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
) -> Response:
    """Redeem a classroom code: a new anonymous participant, a new session.

    Re-entering the code for a course this browser already joined keeps the
    existing participant rather than minting another, so a class that
    re-clicks the link every week does not become a class of hundreds.
    """
    address = rate_limit.client_address(request)
    _throttle(rate_limit.JOIN_CODE_PER_CLIENT, address)
    _throttle(rate_limit.JOIN_CODE_GLOBAL, rate_limit.GLOBAL_KEY)

    def failed() -> HTTPException:
        rate_limit.JOIN_CODE_PER_CLIENT.record_failure(address)
        rate_limit.JOIN_CODE_GLOBAL.record_failure(rate_limit.GLOBAL_KEY)
        return HTTPException(status_code=404, detail=CODE_INVALID_DETAIL)

    code = normalize_code(body.code)
    if code is None:
        raise failed()
    now = _utc_now()

    def work(conn: Any) -> dict[str, Any] | None:
        invitation = db_invitations.find_by_code(conn, code, now=now)
        if invitation is None or invitation["status"] != db_invitations.ACTIVE:
            return None
        course_id = invitation["courseId"]
        course_name = invitation.get("courseName")

        if principal.participant_for(course_id) is not None:
            db_participants.touch_participant(
                conn, principal.participant.participant_id, now  # type: ignore[union-attr]
            )
            return {"courseId": course_id, "courseName": course_name, "token": None}

        if db_invitations.consume(conn, invitation["invitationId"], now=now) is None:
            return None
        participant = db_participants.create_participant(
            conn, course_id=course_id, invitation_id=invitation["invitationId"], now=now
        )
        token = generate_token()
        db_sessions.create_session(
            conn,
            kind=db_sessions.PARTICIPANT,
            token_hash=hash_token(token),
            lifetime=participant_session_lifetime(),
            now=now,
            participant_id=participant["participantId"],
        )
        # A replaced participant session (the browser joined another course)
        # is retired so it cannot be presented again.
        if principal.participant is not None and principal.participant.session_id:
            db_sessions.revoke_session(conn, principal.participant.session_id, now)
        return {"courseId": course_id, "courseName": course_name, "token": token}

    result = _run("joining a course", work)
    if result is None:
        raise failed()

    rate_limit.JOIN_CODE_PER_CLIENT.reset(address)
    payload = JoinResponse(
        courseId=result["courseId"],
        courseName=result.get("courseName"),
        alreadyJoined=result["token"] is None,
    )
    response = Response(
        content=payload.model_dump_json(by_alias=True),
        media_type="application/json",
    )
    if result["token"] is not None:
        set_session_cookie(
            response, PARTICIPANT_COOKIE_NAME, result["token"], participant_session_lifetime()
        )
    return response


# --------------------------------------------------------------------------- #
# Staff: invitations
# --------------------------------------------------------------------------- #


def _redeemable_invitation(conn: Any, token: str, now: datetime) -> dict[str, Any] | None:
    invitation = db_invitations.find_by_token_hash(conn, hash_token(token), now=now)
    if invitation is None or invitation["status"] != db_invitations.ACTIVE:
        return None
    if invitation["kind"] not in db_invitations.TOKEN_KINDS:
        return None
    return invitation


def _course_names(conn: Any, course_ids: list[str]) -> list[InvitationCourse]:
    courses: list[InvitationCourse] = []
    for course_id in course_ids:
        record = db_courses.get_course(conn, course_id)
        name = record["metadata"]["name"] if record else None
        courses.append(InvitationCourse(courseId=course_id, name=name))
    return courses


@router.get("/invitations/{token}", response_model=InvitationPreview)
def preview_invitation(token: str, request: Request) -> InvitationPreview:
    """What accepting this link would do, without consuming it."""
    address = rate_limit.client_address(request)
    _throttle(rate_limit.INVITE_PER_CLIENT, address)
    if not is_plausible_token(token):
        rate_limit.INVITE_PER_CLIENT.record_failure(address)
        raise HTTPException(status_code=404, detail=INVITATION_INVALID_DETAIL)
    now = _utc_now()

    def work(conn: Any) -> InvitationPreview | None:
        invitation = _redeemable_invitation(conn, token, now)
        if invitation is None:
            return None
        preview = InvitationPreview(
            kind=invitation["kind"],
            label=invitation.get("label"),
            expiresAt=invitation.get("expiresAt"),
            courses=_course_names(conn, invitation.get("courseIds", [])),
        )
        if invitation["kind"] == db_invitations.RESET:
            target = db_users.get_user(conn, invitation["targetUserId"])
            preview.target_email = target["email"] if target else None
        return preview

    preview = _run("reading an invitation", work)
    if preview is None:
        rate_limit.INVITE_PER_CLIENT.record_failure(address)
        raise HTTPException(status_code=404, detail=INVITATION_INVALID_DETAIL)
    return preview


@router.post(
    "/invitations/{token}/accept",
    response_model=SessionResponse,
    dependencies=[Depends(require_csrf)],
)
def accept_invitation(
    token: str, body: AcceptInvitationRequest, request: Request
) -> Response:
    """Turn a privileged invitation into an account (or a new password) and a session.

    Everything happens in one transaction: the account, its memberships, the
    conditional consume, and the audit row. If the consume finds the invitation
    already used — two people racing the same link — the transaction rolls
    back and nothing was created.
    """
    address = rate_limit.client_address(request)
    _throttle(rate_limit.INVITE_PER_CLIENT, address)
    if not is_plausible_token(token):
        rate_limit.INVITE_PER_CLIENT.record_failure(address)
        raise HTTPException(status_code=404, detail=INVITATION_INVALID_DETAIL)

    problem = password_problem(body.password)
    if problem:
        raise HTTPException(status_code=422, detail=problem)
    now = _utc_now()

    class _Refused(Exception):
        def __init__(self, status_code: int, detail: str) -> None:
            self.status_code = status_code
            self.detail = detail

    def work(conn: Any) -> tuple[str, Principal]:
        invitation = _redeemable_invitation(conn, token, now)
        if invitation is None:
            raise _Refused(404, INVITATION_INVALID_DETAIL)
        kind = invitation["kind"]
        invitation_id = invitation["invitationId"]

        if kind == db_invitations.RESET:
            user_id = invitation["targetUserId"]
            if not db_users.set_password_hash(conn, user_id, hash_password(body.password)):
                raise _Refused(404, INVITATION_INVALID_DETAIL)
            db_sessions.revoke_sessions_for_user(conn, user_id, now)
            role = (db_users.get_user(conn, user_id) or {}).get("role")
            action = "invitation.accept_reset"
        else:
            email = db_users.normalize_email(body.email)
            display_name = (body.display_name or "").strip()
            if not email or "@" not in email:
                raise _Refused(422, "A valid email address is required.")
            if not display_name:
                raise _Refused(422, "A display name is required.")
            try:
                user = db_users.create_user(
                    conn,
                    email=email,
                    display_name=display_name,
                    role=kind,
                    password_hash=hash_password(body.password),
                    invitation_id=invitation_id,
                    now=now,
                )
            except db_users.UserAlreadyExistsError:
                raise _Refused(409, EMAIL_TAKEN_DETAIL)
            user_id = user["userId"]
            role = kind
            for course_id in invitation.get("courseIds", []):
                db_memberships.add_membership(
                    conn,
                    course_id=course_id,
                    user_id=user_id,
                    granted_by=invitation.get("createdBy"),
                    now=now,
                )
            action = "invitation.accept"

        if db_invitations.consume(conn, invitation_id, now=now, accepted_by_user_id=user_id) is None:
            raise _Refused(404, INVITATION_INVALID_DETAIL)

        db_admin_actions.record_action(
            conn,
            actor_user_id=user_id,
            actor_role=role,
            action=action,
            target_kind="invitation",
            target_id=invitation_id,
            detail={"kind": kind, "courseIds": invitation.get("courseIds", [])},
            now=now,
        )
        session_token = generate_token()
        session = db_sessions.create_session(
            conn,
            kind=db_sessions.USER,
            token_hash=hash_token(session_token),
            lifetime=staff_session_lifetime(),
            now=now,
            user_id=user_id,
        )
        db_users.record_login(conn, user_id, now)
        return session_token, _staff_principal(conn, user_id, session["sessionId"])

    try:
        session_token, principal = _run("accepting an invitation", work)
    except _Refused as refused:
        if refused.status_code == 404:
            rate_limit.INVITE_PER_CLIENT.record_failure(address)
        raise HTTPException(status_code=refused.status_code, detail=refused.detail)

    response = Response(
        content=session_response(principal).model_dump_json(by_alias=True),
        media_type="application/json",
    )
    set_session_cookie(response, STAFF_COOKIE_NAME, session_token, staff_session_lifetime())
    return response


# --------------------------------------------------------------------------- #
# Staff: own password
# --------------------------------------------------------------------------- #


@router.post("/password", status_code=204)
def change_password(
    body: PasswordChangeRequest, principal: Principal = Depends(require_user)
) -> Response:
    problem = password_problem(body.new_password)
    if problem:
        raise HTTPException(status_code=422, detail=problem)
    user = principal.user
    assert user is not None
    now = _utc_now()

    def work(conn: Any) -> bool:
        credentials = db_users.find_credentials(conn, user.email)
        if credentials is None or not verify_password(
            body.current_password, credentials["passwordHash"]
        ):
            return False
        db_users.set_password_hash(conn, user.user_id, hash_password(body.new_password))
        # Every other session ends; this one continues.
        db_sessions.revoke_sessions_for_user(
            conn, user.user_id, now, keep_session_id=user.session_id
        )
        return True

    if not _run("changing a password", work):
        raise HTTPException(status_code=400, detail="The current password is not correct.")
    return Response(status_code=204)
