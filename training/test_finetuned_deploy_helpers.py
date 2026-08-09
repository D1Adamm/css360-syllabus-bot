"""Unit tests for fine-tuned deploy helper parsing / .env updates."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


def _load_helpers():
    path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "lib"
        / "finetuned_deploy_helpers.py"
    )
    spec = importlib.util.spec_from_file_location("finetuned_deploy_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = _load_helpers()


class ParseSbatchJobIdTests(unittest.TestCase):
    def test_standard_output(self) -> None:
        self.assertEqual(
            helpers.parse_sbatch_job_id("Submitted batch job 216829\n"),
            "216829",
        )

    def test_ignores_noise_lines(self) -> None:
        text = "sbatch: info: ...\nSubmitted batch job 42\n"
        self.assertEqual(helpers.parse_sbatch_job_id(text), "42")

    def test_missing_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            helpers.parse_sbatch_job_id("nope\n")


class ParseSqueueTests(unittest.TestCase):
    def test_running_line(self) -> None:
        parsed = helpers.parse_squeue_job_line("216829 R g014 00:12:03 00:47:57")
        assert parsed is not None
        self.assertEqual(parsed["job_id"], "216829")
        self.assertEqual(parsed["state"], "R")
        self.assertEqual(parsed["node"], "g014")
        self.assertEqual(parsed["elapsed"], "00:12:03")
        self.assertEqual(parsed["time_left"], "00:47:57")

    def test_pending_without_node(self) -> None:
        parsed = helpers.parse_squeue_job_line("216829 PD (null) 0:00 1:00:00")
        assert parsed is not None
        self.assertEqual(parsed["state"], "PD")
        self.assertEqual(parsed["node"], "")

    def test_active_and_running_helpers(self) -> None:
        self.assertTrue(helpers.is_active_slurm_state("PD"))
        self.assertTrue(helpers.is_active_slurm_state("RUNNING"))
        self.assertTrue(helpers.is_running_slurm_state("R"))
        self.assertFalse(helpers.is_running_slurm_state("PD"))


class HostnameValidationTests(unittest.TestCase):
    def test_accepts_simple_node(self) -> None:
        self.assertEqual(helpers.validate_compute_hostname("g001"), "g001")
        self.assertEqual(helpers.validate_compute_hostname("n3148"), "n3148")

    def test_rejects_injection(self) -> None:
        for bad in ("g001;rm -rf /", "g001$(reboot)", "g001`id`", "-evil", ""):
            with self.assertRaises(ValueError):
                helpers.validate_compute_hostname(bad)


class WaitForNodeStdoutTests(unittest.TestCase):
    def test_accepts_single_hostname(self) -> None:
        self.assertEqual(helpers.parse_wait_for_node_stdout("g014\n"), "g014")

    def test_pending_status_messages_contaminate_stdout(self) -> None:
        contaminated = (
            "State: PD (waiting for allocation...)\n"
            "State: RUNNING (node not reported yet...)\n"
            "g014\n"
        )
        with self.assertRaises(ValueError) as ctx:
            helpers.parse_wait_for_node_stdout(contaminated)
        self.assertIn("exactly one hostname line", str(ctx.exception))
        self.assertIn("State: PD", str(ctx.exception))

    def test_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            helpers.parse_wait_for_node_stdout("\n")


class HealthReadyTests(unittest.TestCase):
    def test_ready(self) -> None:
        self.assertTrue(
            helpers.health_payload_is_ready(
                {"status": "ok", "adapterLoaded": True, "cudaAvailable": True}
            )
        )

    def test_not_ready(self) -> None:
        self.assertFalse(
            helpers.health_payload_is_ready({"status": "ok", "adapterLoaded": False})
        )
        self.assertFalse(helpers.health_payload_is_ready({"status": "starting"}))

    def test_ready_snake_case(self) -> None:
        self.assertTrue(
            helpers.health_payload_is_ready(
                {"status": "ok", "adapter_loaded": True}
            )
        )


class UpdateEnvKeyTests(unittest.TestCase):
    def test_updates_existing_key(self) -> None:
        original = "FOO=1\nFINETUNED_SERVICE_URL=http://old\nBAR=2\n"
        updated = helpers.update_env_key(
            original, "FINETUNED_SERVICE_URL", "http://127.0.0.1:9001"
        )
        self.assertIn("FINETUNED_SERVICE_URL=http://127.0.0.1:9001\n", updated)
        self.assertIn("FOO=1\n", updated)
        self.assertIn("BAR=2\n", updated)
        self.assertNotIn("http://old", updated)

    def test_appends_missing_key(self) -> None:
        original = "FOO=1\n"
        updated = helpers.update_env_key(
            original, "FINETUNED_SERVICE_URL", "http://127.0.0.1:9001"
        )
        self.assertTrue(updated.startswith("FOO=1\n"))
        self.assertTrue(updated.endswith("FINETUNED_SERVICE_URL=http://127.0.0.1:9001\n"))

    def test_preserves_comments(self) -> None:
        original = "# keep me\nFOO=1\n"
        updated = helpers.update_env_key(original, "FINETUNED_SERVICE_URL", "x")
        self.assertTrue(updated.startswith("# keep me\n"))

    def test_update_env_file_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "backend.env"
            path.write_text("A=1\n", encoding="utf-8")
            changed = helpers.update_env_file(path, "FINETUNED_SERVICE_URL", "http://x")
            self.assertTrue(changed)
            text = path.read_text(encoding="utf-8")
            self.assertIn("A=1\n", text)
            self.assertIn("FINETUNED_SERVICE_URL=http://x\n", text)


if __name__ == "__main__":
    unittest.main()
