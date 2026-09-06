"""Setting and clearing the two session cookies.

Attributes, and why:

  HttpOnly   the token is never readable from JavaScript, so a script injected
             into a page cannot lift a session;
  Secure     the browser sends it over HTTPS only. On by default; a developer
             on plain HTTP turns it off with AUTH_COOKIE_SECURE=false;
  SameSite   Lax: sent on same-site requests and top-level navigations, never
             on a cross-site POST, which is the first half of CSRF protection;
  Path=/api  only API requests carry it.

`Max-Age` matches the session's absolute lifetime. Expiry is still decided by
the row in `auth_sessions`; the cookie's own deadline only keeps the browser
from presenting a token the server would refuse anyway.
"""

from __future__ import annotations

from datetime import timedelta

from fastapi import Response

from app.auth.settings import COOKIE_PATH, cookie_secure


def set_session_cookie(response: Response, name: str, token: str, lifetime: timedelta) -> None:
    response.set_cookie(
        key=name,
        value=token,
        max_age=int(lifetime.total_seconds()),
        path=COOKIE_PATH,
        secure=cookie_secure(),
        httponly=True,
        samesite="lax",
    )


def clear_session_cookie(response: Response, name: str) -> None:
    response.delete_cookie(
        key=name,
        path=COOKIE_PATH,
        secure=cookie_secure(),
        httponly=True,
        samesite="lax",
    )
