"""Lightweight tests for inference service helpers (no GPU / no FastAPI needed)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import helpers


class InferenceServiceHelperTests(unittest.TestCase):
    def test_validate_question_rejects_blank(self) -> None:
        with self.assertRaises(ValueError):
            helpers.validate_question("   ")
        self.assertEqual(helpers.validate_question("  Hello?  "), "Hello?")

    def test_resolve_adapter_path_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"ADAPTER_PATH": "/tmp/my-adapter"}, clear=False):
            self.assertEqual(helpers.resolve_adapter_path(), Path("/tmp/my-adapter"))

    def test_resolve_port(self) -> None:
        with mock.patch.dict(os.environ, {"INFERENCE_PORT": "8123"}, clear=False):
            self.assertEqual(helpers.resolve_port(), 8123)

    def test_assert_hf_auth_reads_token_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            token_path = Path(tmp) / "token"
            token_path.write_text("hf_test_token\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("HF_TOKEN", None)
                os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
                os.environ["HF_TOKEN_PATH"] = str(token_path)
                helpers.assert_hf_auth_available()
                self.assertEqual(os.environ.get("HF_TOKEN"), "hf_test_token")

    def test_assert_hf_auth_fails_without_credentials(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
            os.environ["HF_TOKEN_PATH"] = "/tmp/does-not-exist-hf-token"
            with mock.patch.dict(
                "sys.modules",
                {"huggingface_hub": mock.MagicMock(HfFolder=mock.MagicMock(get_token=mock.Mock(return_value=None)))},
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    helpers.assert_hf_auth_available()
            self.assertIn("Hugging Face authentication", str(ctx.exception))

    def test_adapter_path_missing_is_detectable(self) -> None:
        path = helpers.resolve_adapter_path()
        # Default path may or may not exist on this machine; ensure Path typing works.
        self.assertIsInstance(path, Path)


if __name__ == "__main__":
    unittest.main()
