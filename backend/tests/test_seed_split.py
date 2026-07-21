"""Tests for deterministic approved-export train/validation splitting."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.seed_export import write_json, write_jsonl
from app.seed_split import (
    DEFAULT_SPLIT_SEED,
    TrainingSplitError,
    approved_export_status,
    compute_validation_size,
    prepare_training_split,
    split_records,
)


def _record(n: int) -> dict[str, str]:
    return {"instruction": f"Question {n}?", "response": f"Answer {n}."}


class TrainingSplitTests(unittest.TestCase):
    def test_validation_size_for_54_is_six(self) -> None:
        self.assertEqual(compute_validation_size(54), 6)
        self.assertEqual(compute_validation_size(2), 1)

    def test_split_is_deterministic_for_seed_360(self) -> None:
        records = [_record(i) for i in range(54)]
        train_a, val_a = split_records(records, split_seed=DEFAULT_SPLIT_SEED)
        train_b, val_b = split_records(records, split_seed=DEFAULT_SPLIT_SEED)
        self.assertEqual(len(train_a), 48)
        self.assertEqual(len(val_a), 6)
        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertEqual(len(train_a) + len(val_a), 54)

    def test_prepare_training_split_writes_files_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            course_id = "css-360-winter-2026-a7rp"
            out_dir = root / "data" / "exports" / course_id
            out_dir.mkdir(parents=True)
            records = [_record(i) for i in range(54)]
            write_jsonl(out_dir / "approved-finetune.jsonl", records)
            write_json(
                out_dir / "approved-export-summary.json",
                {"exportedAt": "2026-07-21T17:07:16.069789+00:00"},
            )

            first = prepare_training_split(course_id, export_root=root, created_at="fixed-time")
            second = prepare_training_split(course_id, export_root=root, created_at="fixed-time")

            self.assertEqual(first["trainExamples"], 48)
            self.assertEqual(first["validationExamples"], 6)
            self.assertEqual(first["totalExamples"], 54)
            self.assertEqual(first["splitSeed"], 360)

            train_path = Path(first["files"]["trainJsonl"])
            val_path = Path(first["files"]["validationJsonl"])
            manifest_path = Path(first["files"]["manifestJson"])
            self.assertTrue(train_path.is_file())
            self.assertTrue(val_path.is_file())
            self.assertTrue(manifest_path.is_file())

            train_text = train_path.read_text(encoding="utf-8")
            val_text = val_path.read_text(encoding="utf-8")
            self.assertEqual(train_text, Path(second["files"]["trainJsonl"]).read_text(encoding="utf-8"))
            self.assertEqual(val_text, Path(second["files"]["validationJsonl"]).read_text(encoding="utf-8"))

            train_lines = [line for line in train_text.splitlines() if line.strip()]
            val_lines = [line for line in val_text.splitlines() if line.strip()]
            self.assertEqual(len(train_lines), 48)
            self.assertEqual(len(val_lines), 6)
            for line in train_lines + val_lines:
                row = json.loads(line)
                self.assertEqual(set(row.keys()), {"instruction", "response"})
                self.assertTrue(str(row["instruction"]).strip())
                self.assertTrue(str(row["response"]).strip())

            manifest = first["manifest"]
            self.assertEqual(manifest["courseId"], course_id)
            self.assertEqual(manifest["splitSeed"], 360)
            self.assertEqual(manifest["totalExamples"], 54)
            self.assertEqual(manifest["trainExamples"], 48)
            self.assertEqual(manifest["validationExamples"], 6)
            self.assertEqual(
                manifest["sourceExportTimestamp"],
                "2026-07-21T17:07:16.069789+00:00",
            )
            self.assertIn("datasetVersion", manifest)
            self.assertIn("sourceFile", manifest)
            self.assertIn("createdAt", manifest)
            self.assertIn("trainFile", manifest)
            self.assertIn("validationFile", manifest)

    def test_missing_source_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TrainingSplitError) as ctx:
                prepare_training_split("missing-course", export_root=Path(tmp))
            self.assertIn("does not exist", str(ctx.exception))

    def test_invalid_source_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            course_id = "css-360-winter-2026-a7rp"
            out_dir = root / "data" / "exports" / course_id
            out_dir.mkdir(parents=True)
            (out_dir / "approved-finetune.jsonl").write_text(
                '{"instruction": "Q?", "response": "A."}\n{not-json\n',
                encoding="utf-8",
            )
            with self.assertRaises(TrainingSplitError) as ctx:
                prepare_training_split(course_id, export_root=root)
            self.assertIn("malformed JSON", str(ctx.exception))

    def test_fewer_than_two_examples_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            course_id = "css-360-winter-2026-a7rp"
            out_dir = root / "data" / "exports" / course_id
            out_dir.mkdir(parents=True)
            write_jsonl(out_dir / "approved-finetune.jsonl", [_record(1)])
            with self.assertRaises(TrainingSplitError) as ctx:
                prepare_training_split(course_id, export_root=root)
            self.assertIn("at least 2", str(ctx.exception))

    def test_approved_export_status_reports_existence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            course_id = "css-360-winter-2026-a7rp"
            missing = approved_export_status(course_id, export_root=root)
            self.assertFalse(missing["exists"])
            self.assertEqual(missing["exampleCount"], 0)

            out_dir = root / "data" / "exports" / course_id
            out_dir.mkdir(parents=True)
            write_jsonl(out_dir / "approved-finetune.jsonl", [_record(1), _record(2)])
            present = approved_export_status(course_id, export_root=root)
            self.assertTrue(present["exists"])
            self.assertEqual(present["exampleCount"], 2)


if __name__ == "__main__":
    unittest.main()
