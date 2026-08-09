"""Unit tests for QLoRA training automation helpers (no GPU / no Slurm)."""

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
        / "qlora_training_helpers.py"
    )
    spec = importlib.util.spec_from_file_location("qlora_training_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = _load_helpers()


class CourseIdTests(unittest.TestCase):
    def test_valid(self) -> None:
        self.assertEqual(
            helpers.validate_course_id("css-360-winter-2026-a7rp"),
            "css-360-winter-2026-a7rp",
        )

    def test_invalid(self) -> None:
        for bad in ("", "../etc", "CSS-360", "a/b", "x;rm", "-bad", "bad-"):
            with self.assertRaises(ValueError):
                helpers.validate_course_id(bad)


class JsonlValidationTests(unittest.TestCase):
    def test_counts_and_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jsonl"
            val = root / "validation.jsonl"
            manifest = root / "manifest.json"
            train.write_text(
                '{"instruction":"Q1","response":"A1"}\n'
                '{"instruction":"Q2","response":"A2"}\n',
                encoding="utf-8",
            )
            val.write_text(
                '{"instruction":"Q3","response":"A3"}\n',
                encoding="utf-8",
            )
            manifest.write_text('{"courseId":"css-360-winter-2026-a7rp"}\n', encoding="utf-8")
            counts = helpers.validate_course_export_dir(root)
            self.assertEqual(counts["train_count"], 2)
            self.assertEqual(counts["validation_count"], 1)

    def test_rejects_bad_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text('{"instruction":"Q","response":""}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                helpers.validate_instruction_response_jsonl(path)


class VersionedOutputTests(unittest.TestCase):
    def test_versioned_path(self) -> None:
        path = helpers.versioned_training_output_dir(
            user="madamk",
            course_id="css-360-winter-2026-a7rp",
            run_id="20260809T234500Z",
            mode="full",
        )
        self.assertEqual(
            path,
            "/gpfs/projects/simswe/madamk/training_outputs/qlora-runs/"
            "css-360-winter-2026-a7rp/20260809T234500Z-full",
        )
        self.assertNotIn("css-360-qlora/adapter", path)

    def test_live_adapter_detection(self) -> None:
        live = helpers.live_adapter_dir(user="madamk")
        self.assertTrue(helpers.is_live_adapter_path(live, user="madamk"))
        versioned = helpers.versioned_training_output_dir(
            user="madamk",
            course_id="css-360-winter-2026-a7rp",
            run_id="20260809T234500Z",
            mode="smoke",
        )
        self.assertFalse(helpers.is_live_adapter_path(f"{versioned}/adapter", user="madamk"))


class AdapterValidationTests(unittest.TestCase):
    def test_accepts_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run" / "adapter"
            root.mkdir(parents=True)
            (root / "adapter_config.json").write_text("{}", encoding="utf-8")
            (root / "adapter_model.safetensors").write_bytes(b"fake")
            self.assertEqual(helpers.validate_adapter_source(root), root.resolve())

    def test_rejects_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "adapter"
            root.mkdir()
            (root / "adapter_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                helpers.validate_adapter_source(root)

    def test_rejects_live_as_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Simulate a directory whose resolved path ends with the live suffix
            # by validating the helper's path string check independently.
            live = helpers.live_adapter_dir(user="someone")
            self.assertTrue(helpers.is_live_adapter_path(live))


class SqueueParseTests(unittest.TestCase):
    def test_parse(self) -> None:
        parsed = helpers.parse_squeue_training_line(
            "216900 css360-qlora-train R g014 00:10:00 07:50:00"
        )
        assert parsed is not None
        self.assertEqual(parsed["job_id"], "216900")
        self.assertEqual(parsed["name"], "css360-qlora-train")
        self.assertEqual(parsed["node"], "g014")
        self.assertTrue(helpers.is_active_slurm_state(parsed["state"]))


class BackupPathTests(unittest.TestCase):
    def test_backup_dir(self) -> None:
        path = helpers.backup_destination_dir(user="madamk", stamp="20260809T234500Z")
        self.assertEqual(
            path,
            "/gpfs/projects/simswe/madamk/training_outputs/adapter-backups/20260809T234500Z",
        )


class RequireTrainingOutputDirTests(unittest.TestCase):
    def test_missing_raises_with_helper_guidance(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            helpers.require_training_output_dir(None, user="madamk")
        msg = str(ctx.exception)
        self.assertIn("TRAINING_OUTPUT_DIR is required", msg)
        self.assertIn("start_qlora_training.sh", msg)

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            helpers.require_training_output_dir("   ", user="madamk")

    def test_refuses_live_parent_and_adapter(self) -> None:
        live = helpers.live_adapter_dir(user="madamk")
        parent = str(Path(live).parent)
        for bad in (live, parent, f"{parent}/"):
            with self.assertRaises(ValueError) as ctx:
                helpers.require_training_output_dir(bad, user="madamk")
            self.assertIn("live inference adapter", str(ctx.exception))

    def test_accepts_versioned_run_dir(self) -> None:
        path = helpers.versioned_training_output_dir(
            user="madamk",
            course_id="css-360-winter-2026-a7rp",
            run_id="20260809T234500Z",
            mode="full",
        )
        self.assertEqual(
            helpers.require_training_output_dir(path, user="madamk"),
            path,
        )


class TrainSlurmSafetyTests(unittest.TestCase):
    """Prove train/smoke Slurm scripts fail closed without TRAINING_OUTPUT_DIR."""

    def test_train_slurm_requires_helper_and_has_no_live_default(self) -> None:
        text = (
            Path(__file__).resolve().parent / "train.slurm"
        ).read_text(encoding="utf-8")
        self.assertIn("require-training-output-dir", text)
        self.assertNotIn(
            'OUT_DIR="${TRAINING_OUTPUT_DIR:-/gpfs/projects/simswe/${USER}/training_outputs/css-360-qlora}"',
            text,
        )
        self.assertIn("TRAINING_OUTPUT_DIR is required", text)

    def test_smoke_slurm_requires_helper(self) -> None:
        text = (
            Path(__file__).resolve().parent / "smoke.slurm"
        ).read_text(encoding="utf-8")
        self.assertIn("require-training-output-dir", text)
        self.assertNotIn(
            'OUT_DIR="${TRAINING_OUTPUT_DIR:-/gpfs/projects/simswe/${USER}/training_outputs/css-360-qlora-smoke}"',
            text,
        )

    def test_cli_missing_env_fails_like_train_slurm(self) -> None:
        import os
        import subprocess
        import sys

        helper = (
            Path(__file__).resolve().parent.parent
            / "scripts"
            / "lib"
            / "qlora_training_helpers.py"
        )
        env = os.environ.copy()
        env.pop("TRAINING_OUTPUT_DIR", None)
        proc = subprocess.run(
            [sys.executable, str(helper), "require-training-output-dir", "--user", "madamk"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("TRAINING_OUTPUT_DIR is required", proc.stderr)


if __name__ == "__main__":
    unittest.main()
