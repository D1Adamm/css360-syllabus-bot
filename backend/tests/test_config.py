"""Tests for backend .env configuration loading.

These exercise the *production* behaviour of `load_backend_env` and
`get_database_url`, so each one declares `APP_ENV=production` in the environment
it patches in. Without that declaration the process's own test-mode barrier
applies — no env file is read and DATABASE_URL is ignored — which is asserted
separately in `test_test_isolation.py`.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import load_backend_env
from app.db import DatabaseConfigurationError, get_database_url


class BackendEnvLoadingTests(unittest.TestCase):
    def test_load_backend_env_sets_database_url_from_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "DATABASE_URL=postgresql://example@localhost:5432/example\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"APP_ENV": "production"}, clear=True):
                self.assertNotIn("DATABASE_URL", os.environ)
                loaded = load_backend_env(env_path, override=True)

                self.assertTrue(loaded)
                self.assertEqual(
                    os.environ.get("DATABASE_URL"),
                    "postgresql://example@localhost:5432/example",
                )
                self.assertEqual(
                    get_database_url(),
                    "postgresql://example@localhost:5432/example",
                )

    def test_load_backend_env_returns_false_when_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.env"
            with patch.dict(os.environ, {"APP_ENV": "production"}, clear=True):
                loaded = load_backend_env(missing, override=True)
                self.assertFalse(loaded)
                self.assertNotIn("DATABASE_URL", os.environ)


class DatabaseConfigurationTests(unittest.TestCase):
    """An unset DATABASE_URL must fail with our message, not the driver's.

    A psycopg connection error can carry the DSN, password included. The point
    of raising here is that nothing gets as far as building one.
    """

    def test_missing_database_url_names_what_to_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Point .env loading at an empty directory so the real backend/.env
            # cannot quietly satisfy the lookup.
            with patch.dict(
                os.environ, {"APP_ENV": "production"}, clear=True
            ), patch(
                "app.db.load_backend_env",
                side_effect=lambda: load_backend_env(Path(temp_dir) / ".env"),
            ):
                with self.assertRaises(DatabaseConfigurationError) as caught:
                    get_database_url()

        message = str(caught.exception)
        self.assertIn("DATABASE_URL", message)
        self.assertIn("PostgreSQL is not configured", message)


if __name__ == "__main__":
    unittest.main()
