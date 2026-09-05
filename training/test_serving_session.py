"""Where `serving_session.py` gets its backend URL and worker token.

`./scripts/start_finetuned_tunnel.sh --from-backend` runs `show` on the UWB VM,
and there the token lives in `backend/.env` — the file the backend service
itself loads. `show` used to read only the repository's `.env.local` / `.env`,
so a normal checkout failed with "Missing TRAINING_WORKER_TOKEN" until an
operator sourced `backend/.env` by hand. These pin down the loading order, that
nothing is ever printed, and that the two ways `show` can fail stay
distinguishable to the shell script.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "lib"))


def _load_serving_session():
    path = REPO_ROOT / "training" / "serving_session.py"
    spec = importlib.util.spec_from_file_location("serving_session", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


serving_session = _load_serving_session()

CONFIG_VARS = ("TRAINING_API_BASE_URL", "VITE_API_BASE_URL", "TRAINING_WORKER_TOKEN")
TOKEN = "not-a-real-token-6f1c"
URL = "https://aiswe.example.test"


class _EnvIsolation(unittest.TestCase):
    """Start every test with none of the configuration variables set."""

    def setUp(self) -> None:
        saved = {name: os.environ.get(name) for name in CONFIG_VARS}
        for name in CONFIG_VARS:
            os.environ.pop(name, None)

        def restore() -> None:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

        self.addCleanup(restore)


class ConfigurationLoadingTests(_EnvIsolation):
    def setUp(self) -> None:
        super().setUp()
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "backend").mkdir()

    def write_backend_env(self, *, token: str = TOKEN, url: str = URL) -> Path:
        path = self.root / "backend" / ".env"
        path.write_text(
            "DATABASE_URL=postgresql://ignored@localhost/ignored\n"
            f"TRAINING_API_BASE_URL={url}\n"
            f"TRAINING_WORKER_TOKEN={token}\n",
            encoding="utf-8",
        )
        os.chmod(path, 0o600)
        return path

    def test_backend_env_supplies_the_url_and_the_token(self) -> None:
        """The VM case: nothing exported, nothing at the repo root."""
        self.write_backend_env()

        serving_session.load_configuration(self.root)

        self.assertEqual(os.environ["TRAINING_WORKER_TOKEN"], TOKEN)
        self.assertEqual(os.environ["TRAINING_API_BASE_URL"], URL)

    def test_an_exported_variable_wins_over_backend_env(self) -> None:
        self.write_backend_env(token="file-token")
        os.environ["TRAINING_WORKER_TOKEN"] = "exported-token"

        serving_session.load_configuration(self.root)

        self.assertEqual(os.environ["TRAINING_WORKER_TOKEN"], "exported-token")

    def test_the_repository_env_local_wins_over_backend_env(self) -> None:
        """Tillicum's file keeps its precedence; backend/.env is a fallback."""
        self.write_backend_env(token="backend-token")
        local = self.root / ".env.local"
        local.write_text("TRAINING_WORKER_TOKEN=cluster-token\n", encoding="utf-8")
        os.chmod(local, 0o600)

        serving_session.load_configuration(self.root)

        self.assertEqual(os.environ["TRAINING_WORKER_TOKEN"], "cluster-token")

    def test_a_missing_backend_env_is_not_an_error(self) -> None:
        """On Tillicum there is no backend/.env; loading must simply skip it."""
        serving_session.load_configuration(self.root)

        self.assertNotIn("TRAINING_WORKER_TOKEN", os.environ)

    def test_loading_never_prints_the_token(self) -> None:
        # Even an over-readable file — the one case that does produce output
        # for the cluster copy — must not echo what it holds.
        path = self.write_backend_env()
        os.chmod(path, 0o644)
        out, err = io.StringIO(), io.StringIO()

        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            serving_session.load_configuration(self.root)

        self.assertNotIn(TOKEN, out.getvalue())
        self.assertNotIn(TOKEN, err.getvalue())


class _FakeQueue:
    def __init__(
        self,
        session: Optional[dict[str, Any]] = None,
        *,
        error: Optional[Exception] = None,
    ) -> None:
        self._session = session
        self._error = error

    def current_serving_session(self) -> Optional[dict[str, Any]]:
        if self._error is not None:
            raise self._error
        return self._session


class ShowExitStatusTests(_EnvIsolation):
    """The tunnel script reads the exit status to say what went wrong.

    1 is "asked the backend, no session recorded"; 2 is "could not ask" —
    configuration missing or the request failing. Before the split, both were
    reported as "No serving session is recorded. Start one on Tillicum first",
    which sent an operator to the wrong machine for a missing token.
    """

    def setUp(self) -> None:
        super().setUp()
        # Keep the test away from the real repository's env files.
        loader = patch.object(serving_session, "load_configuration")
        loader.start()
        self.addCleanup(loader.stop)

    def run_show(self, **queue_patch: Any) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.ExitStack() as stack:
            if queue_patch:
                stack.enter_context(
                    patch.object(serving_session, "build_queue", **queue_patch)
                )
            stack.enter_context(contextlib.redirect_stdout(out))
            stack.enter_context(contextlib.redirect_stderr(err))
            status = serving_session.main(["show", "--json"])
        return status, out.getvalue(), err.getvalue()

    def test_missing_configuration_exits_2_and_names_what_to_set(self) -> None:
        status, _, err = self.run_show()

        self.assertEqual(status, serving_session.EXIT_CANNOT_REACH_BACKEND)
        self.assertIn("TRAINING_API_BASE_URL", err)

    def test_a_missing_token_alone_exits_2_and_names_the_token(self) -> None:
        os.environ["TRAINING_API_BASE_URL"] = URL

        status, _, err = self.run_show()

        self.assertEqual(status, serving_session.EXIT_CANNOT_REACH_BACKEND)
        self.assertIn("TRAINING_WORKER_TOKEN", err)

    def test_an_unreachable_backend_exits_2(self) -> None:
        failure = serving_session.TrainingQueueError("HTTP 401 from the backend")

        status, _, err = self.run_show(return_value=_FakeQueue(error=failure))

        self.assertEqual(status, serving_session.EXIT_CANNOT_REACH_BACKEND)
        self.assertIn("Could not reach the application", err)

    def test_no_recorded_session_exits_1(self) -> None:
        status, out, _ = self.run_show(return_value=_FakeQueue(None))

        self.assertEqual(status, 1)
        self.assertIn("No fine-tuned serving session", out)

    def test_a_recorded_session_is_printed_as_json(self) -> None:
        session = {
            "sessionId": "serve-264787",
            "jobId": "264787",
            "node": "g014",
            "port": 8001,
            "state": "ready",
            "expiresAt": "2026-09-04T20:00:00Z",
        }

        status, out, _ = self.run_show(return_value=_FakeQueue(session))

        self.assertEqual(status, 0)
        self.assertEqual(json.loads(out)["node"], "g014")


if __name__ == "__main__":
    unittest.main()
