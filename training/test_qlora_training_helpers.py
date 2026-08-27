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


class SlurmJobNameIsolationTests(unittest.TestCase):
    """Per-course Slurm job names: CSS360 must not collide with CSS490/CSS350."""

    COURSE_360 = "css-360-winter-2026-a7rp"
    COURSE_490 = "css-490-spring-2026-x9kq"
    COURSE_350 = "css-350-autumn-2026-m2np"

    def test_name_includes_course_and_mode(self) -> None:
        smoke = helpers.slurm_training_job_name(course_id=self.COURSE_360, mode="smoke")
        full = helpers.slurm_training_job_name(course_id=self.COURSE_360, mode="full")
        self.assertEqual(smoke, f"qlora-smoke-{self.COURSE_360}")
        self.assertEqual(full, f"qlora-train-{self.COURSE_360}")
        self.assertNotEqual(smoke, full)

    def test_simultaneous_css360_and_css490_names_differ(self) -> None:
        n360 = helpers.slurm_training_job_name(course_id=self.COURSE_360, mode="full")
        n490 = helpers.slurm_training_job_name(course_id=self.COURSE_490, mode="full")
        n350 = helpers.slurm_training_job_name(course_id=self.COURSE_350, mode="full")
        self.assertNotEqual(n360, n490)
        self.assertNotEqual(n360, n350)
        self.assertNotEqual(n490, n350)
        self.assertTrue(helpers.is_qlora_training_job_name(n360))
        self.assertTrue(helpers.is_qlora_training_job_name(n490))

    def test_active_selection_isolates_simultaneous_courses(self) -> None:
        name_360 = helpers.slurm_training_job_name(course_id=self.COURSE_360, mode="full")
        name_490 = helpers.slurm_training_job_name(course_id=self.COURSE_490, mode="full")
        lines = [
            f"1001 {name_360} R g014 00:10:00 07:50:00",
            f"1002 {name_490} PD (null) 0:00 8:00:00",
        ]
        hit_360 = helpers.select_active_training_job_line(
            lines, course_id=self.COURSE_360, mode="full"
        )
        hit_490 = helpers.select_active_training_job_line(
            lines, course_id=self.COURSE_490, mode="full"
        )
        miss_350 = helpers.select_active_training_job_line(
            lines, course_id=self.COURSE_350, mode="full"
        )
        assert hit_360 is not None
        assert hit_490 is not None
        self.assertEqual(hit_360["job_id"], "1001")
        self.assertEqual(hit_490["job_id"], "1002")
        self.assertIsNone(miss_350)

    def test_css360_active_job_does_not_block_css490(self) -> None:
        name_360 = helpers.slurm_training_job_name(course_id=self.COURSE_360, mode="smoke")
        lines = [f"2001 {name_360} R g001 00:01:00 00:44:00"]
        self.assertIsNotNone(
            helpers.select_active_training_job_line(
                lines, course_id=self.COURSE_360, mode="smoke"
            )
        )
        self.assertIsNone(
            helpers.select_active_training_job_line(
                lines, course_id=self.COURSE_490, mode="smoke"
            )
        )

    def test_legacy_name_only_matches_same_course_via_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta_dir = Path(tmp)
            (meta_dir / "qlora-job-3001.env").write_text(
                f"COURSE_ID={self.COURSE_360}\nMODE=full\nJOB_NAME=css360-qlora-train\n",
                encoding="utf-8",
            )
            lines = ["3001 css360-qlora-train R g014 00:05:00 07:55:00"]
            self.assertEqual(
                helpers.select_active_training_job_line(
                    lines,
                    course_id=self.COURSE_360,
                    mode="full",
                    meta_dir=meta_dir,
                )["job_id"],
                "3001",
            )
            # CSS490 must not treat the legacy CSS360 job as its own.
            self.assertIsNone(
                helpers.select_active_training_job_line(
                    lines,
                    course_id=self.COURSE_490,
                    mode="full",
                    meta_dir=meta_dir,
                )
            )

    def test_legacy_name_without_meta_does_not_block_other_courses(self) -> None:
        lines = ["3002 css360-qlora-train R g014 00:05:00 07:55:00"]
        self.assertIsNone(
            helpers.select_active_training_job_line(
                lines, course_id=self.COURSE_490, mode="full"
            )
        )

    def test_status_recognition_covers_new_and_legacy_names(self) -> None:
        self.assertTrue(helpers.is_qlora_training_job_name("css360-qlora-smoke"))
        self.assertTrue(helpers.is_qlora_training_job_name("css360-qlora-train"))
        self.assertTrue(
            helpers.is_qlora_training_job_name(
                helpers.slurm_training_job_name(course_id=self.COURSE_490, mode="smoke")
            )
        )
        self.assertFalse(helpers.is_qlora_training_job_name("css360-ft-infer"))
        self.assertFalse(helpers.is_qlora_training_job_name("qlora-train-BAD/COURSE"))

    def test_log_prefix_for_new_and_legacy(self) -> None:
        self.assertEqual(helpers.log_prefix_for_slurm_job_name("css360-qlora-smoke"), "smoke")
        self.assertEqual(helpers.log_prefix_for_slurm_job_name("css360-qlora-train"), "train")
        self.assertEqual(
            helpers.log_prefix_for_slurm_job_name(
                helpers.slurm_training_job_name(course_id=self.COURSE_360, mode="smoke")
            ),
            "smoke",
        )
        self.assertEqual(
            helpers.log_prefix_for_slurm_job_name(
                helpers.slurm_training_job_name(course_id=self.COURSE_490, mode="full")
            ),
            "train",
        )

    def test_start_script_overrides_job_name_with_sbatch_j(self) -> None:
        text = (
            Path(__file__).resolve().parent / "start_qlora_training.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('sbatch -J "${JOB_NAME}"', text)
        self.assertIn("slurm-job-name", text)
        self.assertIn("select-active-training-job", text)
        self.assertNotIn('JOB_NAME="css360-qlora-smoke"', text)
        self.assertNotIn('JOB_NAME="css360-qlora-train"', text)

    def test_status_script_scans_recognized_names_not_hardcoded_pair(self) -> None:
        text = (
            Path(__file__).resolve().parent / "status_qlora_training.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("is-qlora-training-job-name", text)
        self.assertNotIn(
            "for JOB_NAME in css360-qlora-smoke css360-qlora-train",
            text,
        )


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


# --------------------------------------------------------------------------- #
# Slurm wall-time policy
#
# The CSS 350 full run asked for 8 hours and used 48 seconds: 0.01 GPU hours
# against an 8-hour reservation. That is not only untidy — an 8-hour request
# queues behind everything the scheduler can fit in front of it, so the habit
# costs wall-clock latency on every run of every size.
# --------------------------------------------------------------------------- #


class SlurmWalltimeTests(unittest.TestCase):
    def test_the_real_css350_run_now_asks_for_an_hour_not_eight(self) -> None:
        """37 train examples, 3 epochs, accumulation 8 — 15 optimizer steps."""
        plan = helpers.estimate_training_walltime_seconds(
            mode="full", train_examples=37
        )

        self.assertEqual(plan["optimizerSteps"], 15)
        self.assertEqual(plan["walltime"], "01:00:00")
        self.assertEqual(plan["clamped"], "floor")

    def test_a_tiny_dataset_still_gets_the_floor(self) -> None:
        """The floor covers what does not scale with data.

        A cold Hugging Face download of a 3B model, a slow shared filesystem and
        a busy node are all fixed costs, and an estimate that only counted
        optimizer steps would produce a request a first run could not finish in.
        """
        plan = helpers.estimate_training_walltime_seconds(
            mode="full", train_examples=1
        )

        self.assertEqual(plan["walltime"], "01:00:00")

    def test_a_larger_course_gets_proportionally_more(self) -> None:
        plan = helpers.estimate_training_walltime_seconds(
            mode="full", train_examples=2000
        )

        self.assertEqual(plan["optimizerSteps"], 750)
        self.assertEqual(plan["walltime"], "04:40:00")
        self.assertIsNone(plan["clamped"])

    def test_the_estimate_grows_monotonically_with_the_dataset(self) -> None:
        sizes = [37, 200, 800, 2000, 4000]
        seconds = [
            helpers.estimate_training_walltime_seconds(
                mode="full", train_examples=size
            )["seconds"]
            for size in sizes
        ]

        self.assertEqual(seconds, sorted(seconds))

    def test_a_very_large_course_is_capped_and_says_so(self) -> None:
        """A QOS has a maximum, and a request above it is rejected rather than queued.

        `clamped: ceiling` is the signal an operator acts on — it is the one case
        where the policy may genuinely be wrong for a course.
        """
        plan = helpers.estimate_training_walltime_seconds(
            mode="full", train_examples=5000
        )

        self.assertEqual(plan["walltime"], "08:00:00")
        self.assertEqual(plan["clamped"], "ceiling")
        self.assertGreater(plan["estimatedSeconds"], plan["seconds"])

    def test_the_ceiling_can_be_raised_deliberately(self) -> None:
        plan = helpers.estimate_training_walltime_seconds(
            mode="full",
            train_examples=5000,
            ceiling_seconds=16 * 3600,
        )

        self.assertEqual(plan["walltime"], "10:55:00")
        self.assertIsNone(plan["clamped"])

    def test_a_smoke_run_keeps_its_fixed_debug_walltime(self) -> None:
        """Smoke runs execute three optimizer steps; their cost is overhead only."""
        plan = helpers.estimate_training_walltime_seconds(
            mode="smoke", train_examples=5000
        )

        self.assertEqual(plan["walltime"], "00:45:00")
        self.assertIsNone(plan["optimizerSteps"])

    def test_a_walltime_is_rounded_up_to_the_next_minute(self) -> None:
        """A request is a limit: a job needing 61s that asks for 1m is killed."""
        self.assertEqual(helpers.format_slurm_walltime(61), "00:02:00")
        self.assertEqual(helpers.format_slurm_walltime(3600), "01:00:00")
        self.assertEqual(helpers.format_slurm_walltime(1), "00:01:00")

    def test_walltime_parsing_accepts_the_shapes_an_operator_types(self) -> None:
        self.assertEqual(helpers.parse_slurm_walltime("02:00:00"), 7200)
        self.assertEqual(helpers.parse_slurm_walltime("16:00"), 57600)
        self.assertEqual(helpers.parse_slurm_walltime("00:45:00"), 2700)

    def test_an_unparseable_walltime_is_refused_rather_than_passed_through(self) -> None:
        """A value this cannot bound-check would defeat the ceiling."""
        for bad in ("2 hours", "1-00:00:00", "99:99:99", "", "8h"):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    helpers.parse_slurm_walltime(bad)


# --------------------------------------------------------------------------- #
# Per-course serving paths
# --------------------------------------------------------------------------- #


class CourseServingPathTests(unittest.TestCase):
    def test_each_course_gets_its_own_versioned_directory(self) -> None:
        """The property the old single adapter path could not have.

        `training_outputs/css-360-qlora/adapter` had no course in it, so
        publishing CSS 360 replaced whatever CSS 350 was being served with — and
        nothing about a request could detect it, because a request carried no
        course either.
        """
        css350 = helpers.course_version_adapter_dir(
            user="madamk", course_id="css-350-spring-2026-n3h9", version="v1"
        )
        css360 = helpers.course_version_adapter_dir(
            user="madamk", course_id="css-360-winter-2026-a7rp", version="v1"
        )

        self.assertNotEqual(css350, css360)
        self.assertTrue(css350.endswith("/serving/css-350-spring-2026-n3h9/v1/adapter"))
        self.assertTrue(css360.endswith("/serving/css-360-winter-2026-a7rp/v1/adapter"))

    def test_versions_of_one_course_do_not_collide(self) -> None:
        first = helpers.course_version_adapter_dir(
            user="madamk", course_id="css-350-spring-2026-n3h9", version="v1"
        )
        second = helpers.course_version_adapter_dir(
            user="madamk", course_id="css-350-spring-2026-n3h9", version="v2"
        )

        self.assertNotEqual(first, second)

    def test_an_invalid_course_id_never_becomes_a_path(self) -> None:
        for bad in ("../etc", "CSS-350", "css_350", ""):
            with self.subTest(course_id=bad):
                with self.assertRaises(ValueError):
                    helpers.course_serving_dir(user="madamk", course_id=bad)

    def test_an_invalid_version_never_becomes_a_path(self) -> None:
        for bad in ("latest", "1", "v", "../v1", ""):
            with self.subTest(version=bad):
                with self.assertRaises(ValueError):
                    helpers.validate_model_version(bad)


class RelativeOutputRefTests(unittest.TestCase):
    def test_the_machine_specific_prefix_is_stripped(self) -> None:
        """A stored reference outlives the account and the cluster home it came from."""
        ref = helpers.relative_training_output_ref(
            "/gpfs/projects/simswe/madamk/training_outputs/"
            "qlora-runs/css-350-spring-2026-n3h9/20260827T064701Z-full"
        )

        self.assertEqual(
            ref, "qlora-runs/css-350-spring-2026-n3h9/20260827T064701Z-full"
        )
        self.assertNotIn("madamk", ref)
        self.assertFalse(ref.startswith("/"))

    def test_a_trailing_slash_is_ignored(self) -> None:
        self.assertEqual(
            helpers.relative_training_output_ref(
                "/gpfs/x/training_outputs/qlora-runs/a/b/"
            ),
            "qlora-runs/a/b",
        )

    def test_a_path_with_no_marker_is_returned_relative_rather_than_dropped(self) -> None:
        """Recording an unusual reference beats recording nothing."""
        self.assertEqual(
            helpers.relative_training_output_ref("/somewhere/else/adapter"),
            "somewhere/else/adapter",
        )
