"""Suite-wide guarantee that tests never touch a real external service.

Why this exists
---------------
`backend/.env` holds real credentials, and pytest loads it exactly like the
running backend does. Anything a test forgets to stub is therefore fully
configured while the suite runs, and reaches the real thing — silently, because
several status writes are deliberately best-effort and swallow their own
failures.

That is not hypothetical. A route test posting `save: true` to
`/seeds/generate-starter` stubbed seed persistence but not the starter-status
reconciliation that runs after it, and every full-suite run rewrote a live
course's record.

The guards are at the boundaries, not at any helper. Stubbing helpers only ever
covers the paths someone remembered; these cover the ones nobody did, including
paths added later.

Three layers, deliberately
--------------------------
1. `block_external_http` fails any HTTP request to a non-local host. Loud, and
   it names the URL, so a future leak is a failing test rather than a write to
   production. This is what would catch the training worker's queue client if
   it were ever exercised unstubbed.
2. `postgres_unconfigured_by_default` removes DATABASE_URL and TEST_DATABASE_URL,
   so `app.db.connect` raises `DatabaseConfigurationError` before psycopg opens
   a socket. Nothing in the suite talks to a real database; repository behaviour
   is tested against the recording fake in `test_db_repositories.py`, and route
   behaviour by patching `db_connection`.

   This fixture is now the second line of defence rather than the only one.
   `app.config.is_test_mode` stops backend/.env being read at all, and
   `app.db.get_database_url` ignores DATABASE_URL under test — which is what
   closes the hole this fixture used to leave open: it removed the variable,
   and `get_database_url` reloaded backend/.env precisely because it was
   missing, reconnecting the suite to production. See `test_test_isolation.py`.
3. `training_worker_unconfigured_by_default` removes TRAINING_WORKER_TOKEN, so
   the queue router refuses with 503 unless a test sets one. A test that means
   to exercise an authenticated queue call says so.

Firebase used to need a layer of its own here. It no longer does: there is no
Firebase code left in the runtime tree to configure, and no test mocks it. The
one module that still parses a Firebase export — `app/firebase_snapshot.py` —
reads a JSON file and opens no connection at all.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pytest

# Set before anything imports app.config, which decides at import time whether
# it may read backend/.env. conftest is imported ahead of every test module, so
# this is the earliest point in the process that can state the intent.
#
# app.config also treats "pytest is imported" as test mode on its own, so the
# barrier holds even if this line is removed or a test process reaches app.config
# by some other path. This is the explicit half of the same statement.
os.environ.setdefault("APP_ENV", "test")

# Hosts a test may legitimately reach: the in-process ASGI transport, and the
# local services (Ollama, the fine-tuned inference tunnel) that tests point at
# loopback. Nothing here leaves the machine.
ALLOWED_HOSTS = frozenset(
    {
        "testserver",
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
    }
)

DATABASE_ENV_VARS = ("DATABASE_URL", "TEST_DATABASE_URL")
TRAINING_WORKER_ENV_VARS = ("TRAINING_WORKER_TOKEN",)


class ExternalRequestBlocked(AssertionError):
    """Raised when a test attempts a request to a non-local host."""


def _assert_local(request: Any) -> None:
    host = request.url.host
    if host and host not in ALLOWED_HOSTS:
        raise ExternalRequestBlocked(
            f"A test tried to reach {request.method} {request.url}.\n"
            "Backend tests must not make external requests. Stub the boundary "
            "the code under test actually uses — for storage that is "
            "app.db.db_connection (or the repository function above it), and "
            "remember that a route may write status after the work it does."
        )


@pytest.fixture(autouse=True)
def block_external_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any non-local HTTP request, sync or async.

    Patching `send` rather than the transport keeps `TestClient` working: its
    requests carry the `testserver` host and are allowed through to the ASGI
    app as usual.
    """
    real_async_send = httpx.AsyncClient.send
    real_sync_send = httpx.Client.send

    async def guarded_async_send(self: httpx.AsyncClient, request: Any, **kwargs: Any):
        _assert_local(request)
        return await real_async_send(self, request, **kwargs)

    def guarded_sync_send(self: httpx.Client, request: Any, **kwargs: Any):
        _assert_local(request)
        return real_sync_send(self, request, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", guarded_async_send)
    monkeypatch.setattr(httpx.Client, "send", guarded_sync_send)


@pytest.fixture(autouse=True)
def postgres_unconfigured_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test as though PostgreSQL were not configured.

    Production code is unchanged: an unset DATABASE_URL is a state `app.db`
    already handles, and handles the same way in production — a 503 that names
    what to set. A test that wants storage behaviour patches `db_connection`
    with a fake, which is both faster and honest about what it is asserting.

    TEST_DATABASE_URL goes too. It is the one variable that can legitimately
    open a connection from a test process, so a suite that is not deliberately
    an integration suite should not inherit it from the developer's shell.
    """
    for name in DATABASE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def training_worker_unconfigured_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test as though the training worker token were unset.

    The queue router then refuses with 503, which is the deployed behaviour for
    an unconfigured backend. A test exercising an authenticated queue call sets
    the variable itself.
    """
    for name in TRAINING_WORKER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
