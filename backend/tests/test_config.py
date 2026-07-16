"""Tests for backend .env configuration loading."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import load_backend_env
from app.firebase_seeds import get_firebase_database_url


class BackendEnvLoadingTests(unittest.TestCase):
    def test_load_backend_env_sets_firebase_database_url_from_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "FIREBASE_DATABASE_URL=https://example-test-default-rtdb.firebaseio.com\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                self.assertNotIn("FIREBASE_DATABASE_URL", os.environ)
                loaded = load_backend_env(env_path, override=True)

                self.assertTrue(loaded)
                self.assertEqual(
                    os.environ.get("FIREBASE_DATABASE_URL"),
                    "https://example-test-default-rtdb.firebaseio.com",
                )
                self.assertEqual(
                    get_firebase_database_url(),
                    "https://example-test-default-rtdb.firebaseio.com",
                )

    def test_load_backend_env_returns_false_when_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.env"
            with patch.dict(os.environ, {}, clear=True):
                loaded = load_backend_env(missing, override=True)
                self.assertFalse(loaded)
                self.assertNotIn("FIREBASE_DATABASE_URL", os.environ)
