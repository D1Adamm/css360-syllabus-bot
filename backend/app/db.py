"""PostgreSQL connection configuration for the Firebase migration target.

Deliberately not wired into any FastAPI route. Live reads and writes still go
to Firebase Realtime Database; this module exists so the migration importer
(and later, migrated code paths) have one place that knows how to reach
PostgreSQL.

Credentials come from DATABASE_URL only — never from a literal in the tree.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

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
