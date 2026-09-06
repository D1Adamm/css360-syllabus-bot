"""Authentication configuration, read from the environment.

Every value has a safe default, and none of them is a secret — the secrets
are the tokens the tables store hashed. The one value a deployment should
think about is `AUTH_COOKIE_SECURE`, and the safe answer there is the default:
cookies are marked `Secure` unless a developer working over plain HTTP says
otherwise explicitly. A production host behind Nginx and HTTPS needs to set
nothing.
"""

from __future__ import annotations

import os
from datetime import timedelta

#: The professor/admin session cookie and the anonymous student session cookie.
#: Two cookies, deliberately: a professor can open their own classroom code in
#: the same browser and walk the student flow without losing the staff session.
STAFF_COOKIE_NAME = "sml_staff"
PARTICIPANT_COOKIE_NAME = "sml_participant"

#: Cookies are only ever needed on API requests. Scoping them here keeps them
#: off every static asset request and off any other path the origin serves.
COOKIE_PATH = "/api"

#: Cross-site request forgery: a custom header a cross-site form cannot set.
#: The browser client sends it on every state-changing request; the backend
#: refuses cookie-authenticated mutations without it.
CSRF_HEADER_NAME = "x-requested-with"
CSRF_HEADER_VALUE = "SyllabusModelLab"

FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def _env_flag(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in FALSE_VALUES


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def cookie_secure() -> bool:
    """Whether session cookies carry the `Secure` attribute. Default on."""
    return _env_flag("AUTH_COOKIE_SECURE", True)


def staff_session_lifetime() -> timedelta:
    """Absolute lifetime of a professor/admin session."""
    return timedelta(days=_env_int("AUTH_STAFF_SESSION_MAX_DAYS", 7))


def staff_session_idle_timeout() -> timedelta:
    """A staff session unused for this long is over, whatever its deadline."""
    return timedelta(hours=_env_int("AUTH_STAFF_SESSION_IDLE_HOURS", 24))


def participant_session_lifetime() -> timedelta:
    """Absolute lifetime of a student session. No idle timeout: a quarter."""
    return timedelta(days=_env_int("AUTH_PARTICIPANT_SESSION_DAYS", 180))


def professor_invitation_lifetime() -> timedelta:
    return timedelta(days=_env_int("AUTH_PROFESSOR_INVITE_DAYS", 7))


def admin_invitation_lifetime() -> timedelta:
    return timedelta(hours=_env_int("AUTH_ADMIN_INVITE_HOURS", 24))


def reset_invitation_lifetime() -> timedelta:
    return timedelta(hours=_env_int("AUTH_RESET_INVITE_HOURS", 24))


def public_origin() -> str | None:
    """The site origin, e.g. https://aiswe.uwb.edu, when configured.

    Optional. Used to build the bootstrap invitation URL from the command line
    and accepted as a CSRF origin; the browser builds its own links from
    `window.location`, so nothing in the running application depends on it.
    """
    value = (os.getenv("APP_PUBLIC_ORIGIN") or "").strip().rstrip("/")
    return value or None


def allowed_origins() -> frozenset[str]:
    """Origins a state-changing browser request may come from.

    The same list `CORSMiddleware` is configured with, plus the public origin.
    On the VM that list is the site origin itself; locally it is the Vite dev
    server. An `Origin` header naming anything else is refused.
    """
    raw = os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    origins = {item.strip().rstrip("/") for item in raw.split(",") if item.strip()}
    configured = public_origin()
    if configured:
        origins.add(configured)
    return frozenset(origins)
