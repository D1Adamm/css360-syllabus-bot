"""Request-scoped identity and the guards every protected route declares.

`current_principal` is the one place cookies are read. It turns the two
session cookies into a `Principal` by looking the tokens up in
`auth_sessions`, applying the expiry, idle and disabled rules, and — for a
state-changing request that arrived with a session — checking the CSRF
header. Everything downstream receives a `Principal` and never a cookie.

The guards are what routes name in their signatures:

    require_user           any signed-in professor or administrator
    require_admin          an administrator
    require_course_staff   an administrator, or a professor with a membership
                           in the `course_id` path parameter
    require_course_access  course staff, or a participant who joined that course
    require_participant    a participant who joined that course

Each returns the Principal so the handler can attribute what it writes.
Refusals are 401 for "nobody is signed in" and 403 for "signed in, not
allowed"; the course id is validated first so a malformed id is a 400 for
everyone, exactly as the handlers already report it.

Tests replace `current_principal` through `app.dependency_overrides`; the
guards, which build on it, are then exercised with any principal a test
chooses. The authorization matrix test removes the override and drives the
real thing.
"""

from __future__ import annotations

import hmac
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

from fastapi import Depends, HTTPException, Request

from app import db_memberships, db_participants, db_sessions
from app.auth.principal import ANONYMOUS, ADMIN, Participant, Principal, StaffUser
from app.auth.settings import (
    CSRF_HEADER_NAME,
    CSRF_HEADER_VALUE,
    PARTICIPANT_COOKIE_NAME,
    STAFF_COOKIE_NAME,
    allowed_origins,
    staff_session_idle_timeout,
)
from app.auth.tokens import hash_token, is_plausible_token
from app.course_id import assert_valid_course_id
from app.db import db_connection, translate_db_errors

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

#: `last_seen_at` is written at most this often per session, so a busy page
#: does not turn every read into a write.
TOUCH_INTERVAL = timedelta(minutes=5)

SIGN_IN_DETAIL = "Sign in to continue."
JOIN_DETAIL = "Join this course with your class code to continue."
FORBIDDEN_DETAIL = "You do not have access to this."
ADMIN_ONLY_DETAIL = "Only an administrator can do this."
CSRF_DETAIL = "This request must come from the application."


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Resolving the principal
# --------------------------------------------------------------------------- #


def _load_staff(conn: Any, token: str, now: datetime) -> StaffUser | None:
    if not is_plausible_token(token):
        return None
    row = db_sessions.find_staff_session(conn, hash_token(token))
    if row is None or row.get("revoked_at") is not None:
        return None
    if row.get("disabled_at") is not None:
        return None

    expires_at = _aware(row.get("expires_at"))
    last_seen_at = _aware(row.get("last_seen_at"))
    if expires_at is None or last_seen_at is None or expires_at <= now:
        return None

    session_id = str(row["session_id"])
    if now - last_seen_at > staff_session_idle_timeout():
        # Idle too long. Retire the row so the same token cannot come back
        # later within its absolute lifetime.
        db_sessions.revoke_session(conn, session_id, now)
        return None

    user_id = str(row["user_id"])
    role = row["role"]
    course_ids = (
        frozenset()
        if role == ADMIN
        else frozenset(db_memberships.list_course_ids_for_user(conn, user_id))
    )
    if now - last_seen_at >= TOUCH_INTERVAL:
        db_sessions.touch_session(conn, session_id, now)

    return StaffUser(
        user_id=user_id,
        email=row["email"],
        display_name=row["display_name"],
        role=role,
        course_ids=course_ids,
        session_id=session_id,
    )


def _load_participant(conn: Any, token: str, now: datetime) -> Participant | None:
    if not is_plausible_token(token):
        return None
    row = db_sessions.find_participant_session(conn, hash_token(token))
    if row is None or row.get("revoked_at") is not None:
        return None

    expires_at = _aware(row.get("expires_at"))
    last_seen_at = _aware(row.get("last_seen_at"))
    if expires_at is None or expires_at <= now:
        return None

    session_id = str(row["session_id"])
    participant_id = str(row["participant_id"])
    if last_seen_at is None or now - last_seen_at >= TOUCH_INTERVAL:
        db_sessions.touch_session(conn, session_id, now)
        db_participants.touch_participant(conn, participant_id, now)

    return Participant(
        participant_id=participant_id,
        course_id=row["course_id"],
        session_id=session_id,
    )


def resolve_principal(request: Request) -> Principal:
    """Who the cookies say is asking. Anonymous when they say nobody."""
    staff_token = request.cookies.get(STAFF_COOKIE_NAME)
    participant_token = request.cookies.get(PARTICIPANT_COOKIE_NAME)
    if not staff_token and not participant_token:
        return ANONYMOUS

    now = _utc_now()
    with translate_db_errors("checking the session"):
        with db_connection() as conn:
            user = _load_staff(conn, staff_token, now) if staff_token else None
            participant = (
                _load_participant(conn, participant_token, now)
                if participant_token
                else None
            )
    return Principal(user=user, participant=participant)


# --------------------------------------------------------------------------- #
# Cross-site request forgery
# --------------------------------------------------------------------------- #


def _origin_allowed(origin: str, request: Request) -> bool:
    normalized = origin.strip().rstrip("/")
    if not normalized or normalized.lower() == "null":
        return False
    if normalized in allowed_origins():
        return True
    # Same-origin: the browser's Origin names the host it is talking to. This
    # is what a same-origin deployment behind a proxy that forwards Host sees.
    origin_host = urlparse(normalized).netloc.lower()
    request_host = (request.headers.get("host") or "").strip().lower()
    return bool(origin_host) and origin_host == request_host


def enforce_csrf(request: Request) -> None:
    """Refuse a state-changing request that a cross-site page could have sent.

    Two independent checks. The custom header cannot be set by a cross-site
    form or a simple cross-site fetch. The Origin header, when the browser
    sends one, must name this site.
    """
    presented = request.headers.get(CSRF_HEADER_NAME, "")
    if not hmac.compare_digest(presented, CSRF_HEADER_VALUE):
        raise HTTPException(status_code=403, detail=CSRF_DETAIL)
    origin = request.headers.get("origin")
    if origin and not _origin_allowed(origin, request):
        raise HTTPException(status_code=403, detail=CSRF_DETAIL)


def require_csrf(request: Request) -> None:
    """For public state-changing routes that establish a session: login, join, accept."""
    if request.method.upper() not in SAFE_METHODS:
        enforce_csrf(request)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def current_principal(request: Request) -> Principal:
    """The request's identity, with CSRF enforced for cookie-authenticated writes."""
    principal = resolve_principal(request)
    if request.method.upper() not in SAFE_METHODS and not principal.is_anonymous:
        enforce_csrf(request)
    return principal


def _safe_course_id(course_id: str) -> str:
    try:
        return assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def require_user(principal: Principal = Depends(current_principal)) -> Principal:
    if not principal.is_staff:
        raise HTTPException(status_code=401, detail=SIGN_IN_DETAIL)
    return principal


def require_admin(principal: Principal = Depends(current_principal)) -> Principal:
    if not principal.is_staff:
        raise HTTPException(status_code=401, detail=SIGN_IN_DETAIL)
    if not principal.is_admin:
        raise HTTPException(status_code=403, detail=ADMIN_ONLY_DETAIL)
    return principal


def require_course_staff(
    course_id: str, principal: Principal = Depends(current_principal)
) -> Principal:
    safe_course_id = _safe_course_id(course_id)
    if not principal.is_staff:
        raise HTTPException(status_code=401, detail=SIGN_IN_DETAIL)
    if not principal.can_staff_course(safe_course_id):
        raise HTTPException(status_code=403, detail=FORBIDDEN_DETAIL)
    return principal


def require_course_access(
    course_id: str, principal: Principal = Depends(current_principal)
) -> Principal:
    safe_course_id = _safe_course_id(course_id)
    authorize_course_access(principal, safe_course_id)
    return principal


def require_participant(
    course_id: str, principal: Principal = Depends(current_principal)
) -> Principal:
    safe_course_id = _safe_course_id(course_id)
    if principal.participant_for(safe_course_id) is None:
        if principal.is_anonymous:
            raise HTTPException(status_code=401, detail=JOIN_DETAIL)
        raise HTTPException(status_code=403, detail=JOIN_DETAIL)
    return principal


def authorize_course_access(principal: Principal, course_id: str) -> None:
    """The course-access rule, callable from a handler that reads the course
    from a request body rather than the path."""
    if principal.is_anonymous:
        raise HTTPException(status_code=401, detail=JOIN_DETAIL)
    if not principal.can_access_course(course_id):
        raise HTTPException(status_code=403, detail=FORBIDDEN_DETAIL)


def authorize_course_staff(principal: Principal, course_id: str) -> None:
    if not principal.is_staff:
        raise HTTPException(status_code=401, detail=SIGN_IN_DETAIL)
    if not principal.can_staff_course(course_id):
        raise HTTPException(status_code=403, detail=FORBIDDEN_DETAIL)
