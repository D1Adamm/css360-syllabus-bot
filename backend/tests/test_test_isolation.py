"""Regression tests for the production-configuration barrier.

The incident these exist for: on the application VM, `env -u DATABASE_URL pytest
backend/tests -q` connected to the live database anyway. `app.db.get_database_url`
loaded `backend/.env` *because* the variable was missing, and a route test using
a real course id then rewrote that course's `starter_seed_generation` row.

Unsetting the variable is therefore not the protection — it was the trigger.
These tests assert the two properties that are:

  1. backend/.env is not read while the process is in test mode, whatever it
     contains and however the process was invoked.
  2. `DATABASE_URL` cannot supply a DSN in test mode even when it is set. Only
     `TEST_DATABASE_URL` can, and nothing in a deployment sets that.

Both are asserted against a temporary env file rather than the real one, so the
suite never has to read production credentials to prove it will not read them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from app import config, db

BACKEND_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _clear_app_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start from no explicit APP_ENV so the pytest fallback is exercised too."""
    monkeypatch.delenv(config.APP_ENV_VAR, raising=False)


def _write_env_file(directory: Path, body: str) -> Path:
    path = directory / ".env"
    path.write_text(body, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Test-mode detection
# --------------------------------------------------------------------------- #


def test_running_under_pytest_is_test_mode_without_any_configuration() -> None:
    """No APP_ENV at all still counts as test mode.

    This is the case the incident happened in. Nobody had configured anything.
    """
    assert config.is_test_mode() is True


def test_app_env_test_is_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.APP_ENV_VAR, "test")
    assert config.is_test_mode() is True


def test_app_env_testing_is_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.APP_ENV_VAR, "Testing")
    assert config.is_test_mode() is True


def test_explicit_non_test_app_env_disables_the_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deliberate integration run can opt out, and only deliberately."""
    monkeypatch.setenv(config.APP_ENV_VAR, "production")
    assert config.is_test_mode() is False


# --------------------------------------------------------------------------- #
# backend/.env must not be read
# --------------------------------------------------------------------------- #


def test_load_backend_env_reads_nothing_in_test_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = _write_env_file(
        tmp_path, "DATABASE_URL=postgresql://prod@db.example/prod\n"
    )
    monkeypatch.delenv("DATABASE_URL", raising=False)

    assert config.load_backend_env(env_file) is False
    assert "DATABASE_URL" not in __import__("os").environ


def test_load_backend_env_reads_the_file_outside_test_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production path is unchanged — the barrier is not a global disable."""
    env_file = _write_env_file(tmp_path, "SOME_BACKEND_SETTING=configured\n")
    monkeypatch.setenv(config.APP_ENV_VAR, "production")
    monkeypatch.delenv("SOME_BACKEND_SETTING", raising=False)

    assert config.load_backend_env(env_file) is True

    import os

    assert os.environ["SOME_BACKEND_SETTING"] == "configured"


def test_the_real_backend_env_file_is_refused_by_default() -> None:
    """The no-argument call — the one every app module makes — reads nothing.

    Asserted against the real default path deliberately: this is the exact call
    `app.config` makes at import and `app.db` used to make on a missing
    DATABASE_URL. It returns False whether or not backend/.env exists on this
    machine, and on the VM, where it does exist and holds the production DSN,
    False is the whole protection.
    """
    assert config.load_backend_env() is False


# --------------------------------------------------------------------------- #
# get_database_url fails closed
# --------------------------------------------------------------------------- #


def test_database_url_is_ignored_in_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even an explicitly exported production DSN does not reach psycopg."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://prod@db.example/prod")
    monkeypatch.delenv(db.TEST_DATABASE_URL_ENV_VAR, raising=False)

    with pytest.raises(db.DatabaseConfigurationError) as excinfo:
        db.get_database_url()

    assert "TEST_DATABASE_URL" in str(excinfo.value)
    assert "db.example" not in str(excinfo.value)


def test_missing_database_url_does_not_reload_the_env_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact `env -u DATABASE_URL pytest` shape, asserted directly.

    `get_database_url` must not call back into `load_backend_env` at all under
    test — not even a call that would return False — because the reason the
    original code called it was that the variable was missing, which is the
    normal state of every test in this suite.
    """
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv(db.TEST_DATABASE_URL_ENV_VAR, raising=False)

    calls: list[object] = []
    monkeypatch.setattr(
        db, "load_backend_env", lambda *args, **kwargs: calls.append(args) or False
    )

    with pytest.raises(db.DatabaseConfigurationError):
        db.get_database_url()

    assert calls == []


def test_test_database_url_is_the_one_accepted_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deliberately configured throwaway database still works."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://prod@db.example/prod")
    monkeypatch.setenv(
        db.TEST_DATABASE_URL_ENV_VAR, "postgresql://tester@localhost:5432/scratch"
    )

    assert db.get_database_url() == "postgresql://tester@localhost:5432/scratch"


def test_production_still_reads_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """The deployed backend is unaffected by any of this."""
    monkeypatch.setenv(config.APP_ENV_VAR, "production")
    monkeypatch.setenv("DATABASE_URL", "postgresql://app@localhost:5432/syllabus_bot")

    assert db.get_database_url() == "postgresql://app@localhost:5432/syllabus_bot"


def test_connect_refuses_before_importing_psycopg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No socket is opened: the refusal happens before the driver is used."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://prod@db.example/prod")
    monkeypatch.delenv(db.TEST_DATABASE_URL_ENV_VAR, raising=False)

    with pytest.raises(db.DatabaseConfigurationError):
        db.connect()


# --------------------------------------------------------------------------- #
# The pytest fallback cannot reach production
#
# `is_test_mode()` falls back to "is pytest imported?" when APP_ENV is unset,
# which is the case the incident happened in — nobody had configured anything.
# That fallback is only safe if a real server process never has pytest in
# `sys.modules`, and that is a property of the dependency graph rather than
# something this code controls. So it is asserted, in a subprocess, because
# inside a pytest run the answer is trivially yes.
# --------------------------------------------------------------------------- #


def _run_in_clean_interpreter(source: str) -> str:
    """Run a snippet in a fresh interpreter with no APP_ENV and no pytest."""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=str(BACKEND_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "HOME": str(BACKEND_ROOT)},
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def test_importing_the_app_does_not_pull_in_pytest() -> None:
    """The fallback is invisible to uvicorn, and this is why.

    If a dependency ever started importing pytest at module scope, the fallback
    would silently switch a production backend into test mode and it would stop
    reading its own DATABASE_URL. This fails loudly if that ever happens.
    """
    output = _run_in_clean_interpreter(
        "import sys; import app.main; import uvicorn; "
        "print('pytest' in sys.modules)"
    )

    assert output == "False"


def test_a_server_shaped_process_is_not_in_test_mode() -> None:
    output = _run_in_clean_interpreter(
        "import app.main; from app.config import is_test_mode; print(is_test_mode())"
    )

    assert output == "False"


def test_a_server_shaped_process_still_reads_database_url() -> None:
    """The production path is genuinely unchanged, asserted end to end."""
    output = _run_in_clean_interpreter(
        "import os; os.environ['DATABASE_URL'] = 'postgresql://app@localhost/x'; "
        "import app.main; from app.db import get_database_url; print(get_database_url())"
    )

    assert output == "postgresql://app@localhost/x"
