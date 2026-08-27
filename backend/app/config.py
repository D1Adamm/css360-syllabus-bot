"""Backend configuration bootstrap.

Loads backend/.env once so uvicorn and tests can rely on process environment
variables without exporting them manually.

Why loading is refused under test
---------------------------------
`backend/.env` on the application VM holds real production credentials —
DATABASE_URL among them. Every module here imports this one, and this one used
to load that file unconditionally, so a pytest process was fully configured for
production the moment it imported anything. Removing DATABASE_URL from the
environment before running the suite did not help: `app.db.get_database_url`
calls back into `load_backend_env()` precisely when the variable is missing, so
`env -u DATABASE_URL pytest` re-read the file and reconnected to the live
database. A route test that stubbed one write but not the status write after it
then edited a real course's row.

So test mode is a barrier rather than a convention: under it this module loads
no env file at all, and `app.db` refuses `DATABASE_URL` outright and accepts
only `TEST_DATABASE_URL`, which nothing in a deployment sets. A developer who
forgets `env -u` gets the same protection as one who remembers it, because the
barrier is decided by how the process was started rather than by what it was
asked to unset.

Test mode is on when `APP_ENV` names it, and — because a forgotten variable is
exactly the failure this exists to stop — also when the process is running under
pytest at all. An explicit non-test `APP_ENV` wins over the pytest check, which
is what lets a deliberate integration suite run against a real (test) database.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = BACKEND_ROOT / ".env"

APP_ENV_VAR = "APP_ENV"

#: `APP_ENV` values that mean "this process must not touch production".
TEST_APP_ENVS = frozenset({"test", "testing"})

_env_loaded_from: Path | None = None


def app_env() -> str:
    """The configured environment name, lowercased. Empty when unset."""
    return (os.getenv(APP_ENV_VAR) or "").strip().lower()


def is_test_mode() -> bool:
    """Whether this process is barred from loading production configuration.

    An explicit `APP_ENV` is authoritative in both directions: `test` turns the
    barrier on, and any other non-empty value turns it off — that is how a
    deliberate integration run against a real database opts out.

    With `APP_ENV` unset, the presence of pytest decides. That is the case the
    incident came from: nobody had set anything, and the suite was configured
    for production purely because the file was there.
    """
    configured = app_env()
    if configured:
        return configured in TEST_APP_ENVS
    return "pytest" in sys.modules


def load_backend_env(
    env_file: Path | str | None = None,
    *,
    override: bool = False,
    allow_in_test: bool = False,
) -> bool:
    """Load environment variables from a .env file.

    Defaults to backend/.env. Returns True if the file was found and loaded.
    Existing process environment values win unless override=True.

    Returns False without reading anything in test mode. `allow_in_test` exists
    for tests of this function itself, which pass their own temporary file —
    never for production paths, which are not in test mode to begin with.
    """
    global _env_loaded_from

    if is_test_mode() and not allow_in_test:
        return False

    path = Path(env_file) if env_file is not None else DEFAULT_ENV_FILE
    loaded = load_dotenv(path, override=override)
    if loaded:
        _env_loaded_from = path
    return loaded


# Load backend/.env on import so os.getenv works for subsequent app modules.
load_backend_env()
