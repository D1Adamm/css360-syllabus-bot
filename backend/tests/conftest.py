"""Suite-wide guarantee that tests never touch a real external service.

Why this exists
---------------
`backend/.env` holds a real FIREBASE_DATABASE_URL, and pytest loads it like the
running backend does. Every Firebase helper is therefore fully configured while
tests run, so any code path a test forgets to stub reaches the live database —
silently, because the starter-status writes are deliberately best-effort and
swallow their own failures.

That is not hypothetical. A route test posting `save: true` to
`/seeds/generate-starter` stubbed seed persistence but not the starter-status
reconciliation that runs after it, and every full-suite run recreated
`courses/css-360-summer-2026-demo/metadata` in production.

The guard is at the network boundary — httpx's `send`, the single point every
request passes through — rather than at any helper. Stubbing helpers only ever
covers the paths someone remembered; this covers the ones nobody did, including
paths added later.

Two layers, deliberately
------------------------
1. `block_external_http` fails any request to a non-local host. Loud, and it
   names the URL, so a future leak is a failing test rather than a write to
   production.
2. `firebase_unconfigured_by_default` removes the Firebase credentials from the
   environment, so helpers raise `FirebaseConfigurationError` before building a
   request at all. Tests that want configured behavior set the variable
   themselves, which several already do.

Layer 2 alone would be enough today. Layer 1 is what keeps it true: it fails
even if a test sets a real URL, and it catches anything that is not Firebase.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

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

FIREBASE_ENV_VARS = ("FIREBASE_DATABASE_URL", "FIREBASE_AUTH_TOKEN")


class ExternalRequestBlocked(AssertionError):
    """Raised when a test attempts a request to a non-local host."""


def _assert_local(request: Any) -> None:
    host = request.url.host
    if host and host not in ALLOWED_HOSTS:
        raise ExternalRequestBlocked(
            f"A test tried to reach {request.method} {request.url}.\n"
            "Backend tests must not make external requests. Stub the boundary "
            "the code under test actually uses — for Firebase that is "
            "app.firebase_seeds.fetch_course_seed_examples / "
            "save_course_seed_example, or "
            "app.firebase_metadata.best_effort_patch_starter_seed_generation — "
            "and remember that a route may write status after the work it does."
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
def firebase_unconfigured_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test as though Firebase were not configured.

    Production code is unchanged: an unset FIREBASE_DATABASE_URL is a state the
    helpers already handle, and handle the same way in production. A test that
    wants configured behavior sets the variable itself.
    """
    for name in FIREBASE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
