"""Unit tests for QLoRA training helpers (no GPU / no model download)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from train_qlora import (
    TrainingDataError,
    average_seconds_per_step,
    build_runtime_report,
    choose_conservative_step_average,
    effective_batch_size,
    estimate_gpu_hours,
    estimate_optimizer_steps,
    estimate_conservative_total_seconds,
    estimate_training_only_seconds,
    load_instruction_response_jsonl,
    parse_args,
    resolve_smoke_limits,
    write_json,
)


class TrainQloraHelperTests(unittest.TestCase):
    def test_missing_file_rejected(self) -> None:
        with self.assertRaises(TrainingDataError) as ctx:
            load_instruction_response_jsonl("/tmp/does-not-exist-css360-train.jsonl")
        self.assertIn("does not exist", str(ctx.exception))

    def test_load_valid_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"instruction": "Q1?", "response": "A1."}),
                        json.dumps({"instruction": "Q2?", "response": "A2."}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rows = load_instruction_response_jsonl(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["instruction"], "Q1?")

    def test_empty_training_dataset_rejected_by_smoke_limits(self) -> None:
        with self.assertRaises(TrainingDataError):
            resolve_smoke_limits(
                smoke_test=True,
                train_records=[],
                validation_records=[],
            )

    def test_malformed_jsonl_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text('{"instruction": "Q?", "response": "A."}\n{not-json\n', encoding="utf-8")
            with self.assertRaises(TrainingDataError) as ctx:
                load_instruction_response_jsonl(path)
            self.assertIn("Malformed JSONL", str(ctx.exception))

    def test_blank_instruction_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blank_i.jsonl"
            path.write_text(
                json.dumps({"instruction": "  ", "response": "Answer."}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(TrainingDataError) as ctx:
                load_instruction_response_jsonl(path)
            self.assertIn("Blank instruction", str(ctx.exception))

    def test_blank_response_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blank_r.jsonl"
            path.write_text(
                json.dumps({"instruction": "Question?", "response": ""}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(TrainingDataError) as ctx:
                load_instruction_response_jsonl(path)
            self.assertIn("Blank response", str(ctx.exception))

    def test_smoke_argument_resolution(self) -> None:
        args = parse_args(["--smoke-test", "--epochs", "3"])
        self.assertTrue(args.smoke_test)
        self.assertEqual(args.epochs, 3.0)

        train = [{"instruction": f"Q{i}?", "response": f"A{i}."} for i in range(10)]
        val = [{"instruction": f"V{i}?", "response": f"B{i}."} for i in range(5)]
        smoke_train, smoke_val, max_steps = resolve_smoke_limits(
            smoke_test=True,
            train_records=train,
            validation_records=val,
        )
        self.assertEqual(len(smoke_train), 4)
        self.assertEqual(len(smoke_val), 2)
        self.assertEqual(max_steps, 3)

        full_train, full_val, full_steps = resolve_smoke_limits(
            smoke_test=False,
            train_records=train,
            validation_records=val,
        )
        self.assertEqual(len(full_train), 10)
        self.assertEqual(len(full_val), 5)
        self.assertIsNone(full_steps)

    def test_effective_batch_size(self) -> None:
        self.assertEqual(
            effective_batch_size(
                per_device_batch_size=1,
                gradient_accumulation_steps=8,
                gpu_count=1,
            ),
            8,
        )

    def test_optimizer_step_calculation_for_48_examples(self) -> None:
        # 48 examples, batch 8, 3 epochs => 6 steps/epoch * 3 = 18
        steps = estimate_optimizer_steps(
            train_example_count=48,
            epochs=3,
            per_device_batch_size=1,
            gradient_accumulation_steps=8,
            gpu_count=1,
        )
        self.assertEqual(steps, 18)

    def test_runtime_estimate_calculation(self) -> None:
        avg = average_seconds_per_step([10.0, 4.0, 6.0], exclude_first=False)
        self.assertAlmostEqual(avg or 0.0, 20.0 / 3.0)
        skip = average_seconds_per_step([10.0, 4.0, 6.0], exclude_first=True)
        self.assertAlmostEqual(skip or 0.0, 5.0)
        conservative = choose_conservative_step_average(avg, skip)
        self.assertAlmostEqual(conservative or 0.0, 20.0 / 3.0)

        training_only = estimate_training_only_seconds(
            average_seconds_per_step_value=conservative,
            estimated_optimizer_steps=18,
        )
        self.assertIsNotNone(training_only)
        assert training_only is not None
        self.assertGreater(training_only, 0)

        total = estimate_conservative_total_seconds(
            estimated_training_only_seconds=training_only,
            model_load_seconds=30.0,
            evaluation_seconds=5.0,
            epochs=3,
            average_seconds_per_step_value=conservative,
        )
        self.assertIsNotNone(total)
        assert total is not None
        self.assertGreater(total, training_only)

    def test_gpu_hour_calculation(self) -> None:
        hours = estimate_gpu_hours(elapsed_seconds=7200, gpu_count=1)
        self.assertAlmostEqual(hours or 0.0, 2.0)
        hours2 = estimate_gpu_hours(elapsed_seconds=3600, gpu_count=2)
        self.assertAlmostEqual(hours2 or 0.0, 2.0)

    def test_zero_completed_step_handling(self) -> None:
        self.assertIsNone(average_seconds_per_step([]))
        self.assertIsNone(
            estimate_training_only_seconds(
                average_seconds_per_step_value=None,
                estimated_optimizer_steps=18,
            )
        )
        self.assertIsNone(
            estimate_gpu_hours(elapsed_seconds=None, gpu_count=1)
        )

    def test_runtime_report_serialization(self) -> None:
        report = build_runtime_report(
            mode="smoke",
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            gpu_count=1,
            train_example_count=4,
            validation_example_count=2,
            epochs=3,
            effective_batch=8,
            estimated_optimizer_steps=18,
            completed_steps=3,
            model_load_seconds=12.5,
            training_seconds=9.0,
            evaluation_seconds=1.5,
            total_elapsed_seconds=25.0,
            average_seconds_per_step_value=3.0,
            average_seconds_per_step_excluding_first=2.5,
            estimated_training_only_seconds=54.0,
            estimated_conservative_total_seconds=80.0,
            estimated_gpu_hours=80.0 / 3600.0,
            actual_gpu_hours=None,
            git_commit_sha="abc123",
            slurm_job_id="999",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime-report.json"
            write_json(path, report)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["mode"], "smoke")
            self.assertEqual(loaded["completedSteps"], 3)
            self.assertEqual(loaded["estimatedOptimizerSteps"], 18)
            self.assertEqual(loaded["slurmJobId"], "999")
            self.assertIn("timestamp", loaded)


if __name__ == "__main__":
    unittest.main()
