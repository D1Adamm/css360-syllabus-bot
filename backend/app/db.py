"""PostgreSQL connection configuration for the Firebase migration target.

None of the pre-existing Firebase routes read from here. Live reads and writes
still go to Firebase Realtime Database; this module is the one place that knows
how to reach PostgreSQL, used by the migration importer and by the parallel
`/api/db` routes that exist so the frontend can be cut over later.

Credentials come from DATABASE_URL only — never from a literal in the tree, and
never in an error message or log line: a psycopg connection error can carry the
DSN, so failures are reported with our own text rather than the driver's.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

from fastapi import HTTPException

from app.config import load_backend_env

if TYPE_CHECKING:  # pragma: no cover - typing only
    from psycopg import Connection

DATABASE_URL_ENV_VAR = "DATABASE_URL"


class DatabaseConfigurationError(Exception):
    """Raised when PostgreSQL access is requested but not configured."""


def get_database_url() -> str:
    """Return DATABASE_URL, loading backend/.env first if it is not set yet.

    Raises DatabaseConfigurationError when unset or blank so callers fail with
    a message that says what to set rather than a driver-level connection error.
    """
    database_url = os.getenv(DATABASE_URL_ENV_VAR, "").strip()
    if not database_url:
        # A script run outside uvicorn may not have imported anything that
        # loaded backend/.env yet.
        load_backend_env()
        database_url = os.getenv(DATABASE_URL_ENV_VAR, "").strip()

    if not database_url:
        raise DatabaseConfigurationError(
            "PostgreSQL is not configured. Set DATABASE_URL in the backend "
            "environment, e.g. "
            "postgresql://user:password@localhost:5432/syllabus_bot"
        )
    return database_url


def connect(**kwargs: Any) -> "Connection":
    """Open a synchronous psycopg connection using DATABASE_URL.

    psycopg is imported lazily: the live FastAPI app must keep starting on
    hosts where the driver is not installed.
    """
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - depends on install state
        raise DatabaseConfigurationError(
            "psycopg is not installed. Run `pip install -r requirements.txt` "
            "in the backend environment."
        ) from exc

    return psycopg.connect(get_database_url(), **kwargs)


@contextmanager
def db_connection(**kwargs: Any) -> Iterator["Connection"]:
    """Open a connection whose rows arrive as dicts, as one transaction.

    `with psycopg.connect(...)` commits on a clean exit and rolls back if the
    block raises, so a route that writes several tables either lands all of it
    or none. Repository functions take the connection rather than opening their
    own, which is what lets one route span several of them — and what lets the
    tests drive them without a server.
    """
    from psycopg.rows import dict_row

    kwargs.setdefault("row_factory", dict_row)
    with connect(**kwargs) as connection:
        yield connection


def _driver_error_types() -> tuple[type[BaseException], ...]:
    try:
        import psycopg
    except ImportError:  # pragma: no cover - depends on install state
        return ()
    return (psycopg.Error,)


@contextmanager
def translate_db_errors(action: str) -> Iterator[None]:
    """Turn driver and configuration failures into clean 503s.

    Deliberately drops the driver's own message. psycopg puts the connection
    string — password included — into some connection errors, and an API
    response is the last place that should appear. `action` names what was
    being attempted so the response is still useful.
    """
    try:
        yield
    except HTTPException:
        raise
    except DatabaseConfigurationError as exc:
        # Safe to forward: this message is ours and names no credentials.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except _driver_error_types() as exc:
        raise HTTPException(
            status_code=503,
            detail=f"PostgreSQL is unavailable while {action}.",
        ) from exc
