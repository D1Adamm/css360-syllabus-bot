"""Tests for Phase 8 approved-only export and per-course isolation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.seed_export import (
    FinetuneJsonlValidationError,
    export_approved_seeds,
    finetune_record,
    validate_finetune_jsonl,
    write_generation_snapshot,
)


class SeedExportTests(unittest.TestCase):
    def test_approved_only_jsonl_and_metadata(self) -> None:
        seeds = [
            {
                "id": "a",
                "question": "Approved Q?",
                "answer": "Approved A with detail.",
                "reviewStatus": "approved",
                "factId": "fact-1",
                "evidenceQuote": "quote",
                "sourceChunkIds": ["c1"],
                "origin": "ai_generated",
                "validation": {"score": 0.9},
            },
            {
                "id": "b",
                "question": "Generated Q?",
                "answer": "Generated A with detail.",
                "reviewStatus": "generated",
                "factId": "fact-2",
                "origin": "ai_generated",
            },
            {
                "id": "c",
                "question": "Rejected Q?",
                "answer": "Rejected A with detail.",
                "reviewStatus": "rejected",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = export_approved_seeds(
                course_id="css-360-winter-2026-a7rp",
                seeds=seeds,
                export_root=root,
            )
            self.assertEqual(summary["approvedCount"], 1)
            self.assertEqual(summary["exportedCount"], 1)
            self.assertEqual(summary["validatedCount"], 1)
            self.assertTrue(summary["validationPassed"])
            self.assertEqual(summary["skippedCount"], 2)
            self.assertEqual(summary["courseId"], "css-360-winter-2026-a7rp")
            # The store is named by nothing in the summary any more: seeds come
            # from `seed_examples` scoped by courseId, and the old
            # `firebasePath` key described a node that no longer exists.
            self.assertNotIn("firebasePath", summary)

            finetune_path = Path(summary["files"]["finetuneJsonl"])
            self.assertEqual(summary["exportPath"], str(finetune_path))
            metadata_path = Path(summary["files"]["metadataJson"])
            lines = [
                line
                for line in finetune_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(lines), 1)
            row = json.loads(lines[0])
            self.assertEqual(row, {"instruction": "Approved Q?", "response": "Approved A with detail."})
            self.assertEqual(set(row.keys()), {"instruction", "response"})

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(len(metadata), 1)
            self.assertEqual(metadata[0]["factId"], "fact-1")
            self.assertEqual(metadata[0]["reviewStatus"], "approved")
            self.assertEqual(metadata[0]["evidenceQuote"], "quote")

    def test_valid_export_passes_jsonl_validation(self) -> None:
        seeds = [
            {
                "id": "a",
                "question": "Approved Q?",
                "answer": "Approved A with detail.",
                "reviewStatus": "approved",
            },
            {
                "id": "b",
                "question": "Also approved?",
                "answer": "Also approved answer.",
                "reviewStatus": "approved",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            summary = export_approved_seeds(
                course_id="css-360-winter-2026-a7rp",
                seeds=seeds,
                export_root=Path(tmp),
            )
            self.assertTrue(summary["validationPassed"])
            self.assertEqual(summary["exportedCount"], 2)
            self.assertEqual(summary["validatedCount"], 2)
            validated = validate_finetune_jsonl(
                Path(summary["exportPath"]),
                expected_count=2,
            )
            self.assertEqual(validated, 2)

    def test_malformed_jsonl_record_fails_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approved-finetune.jsonl"
            path.write_text(
                '{"instruction": "Q?", "response": "A."}\n{not-json\n',
                encoding="utf-8",
            )
            with self.assertRaises(FinetuneJsonlValidationError) as ctx:
                validate_finetune_jsonl(path, expected_count=2)
            self.assertEqual(ctx.exception.line_number, 2)
            self.assertEqual(ctx.exception.reason, "malformed JSON")
            self.assertIn("line 2", str(ctx.exception))

    def test_blank_instruction_fails_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approved-finetune.jsonl"
            path.write_text(
                json.dumps({"instruction": "   ", "response": "Answer here."}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(FinetuneJsonlValidationError) as ctx:
                validate_finetune_jsonl(path, expected_count=1)
            self.assertEqual(ctx.exception.line_number, 1)
            self.assertEqual(ctx.exception.reason, "blank instruction")
            self.assertIn("blank instruction", str(ctx.exception))

    def test_blank_response_fails_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approved-finetune.jsonl"
            path.write_text(
                json.dumps({"instruction": "Question?", "response": ""}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(FinetuneJsonlValidationError) as ctx:
                validate_finetune_jsonl(path, expected_count=1)
            self.assertEqual(ctx.exception.line_number, 1)
            self.assertEqual(ctx.exception.reason, "blank response")
            self.assertIn("blank response", str(ctx.exception))

    def test_exported_count_mismatch_fails_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approved-finetune.jsonl"
            path.write_text(
                json.dumps({"instruction": "Only one?", "response": "Only one answer."})
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(FinetuneJsonlValidationError) as ctx:
                validate_finetune_jsonl(path, expected_count=2)
            self.assertEqual(ctx.exception.reason, "exported count mismatch")
            self.assertIn("expected 2 records but found 1", str(ctx.exception))

    def test_finetune_record_uses_dual_name_fields(self) -> None:
        row = finetune_record(
            {"instruction": "I?", "response": "R.", "question": "Q?", "answer": "A."}
        )
        self.assertEqual(row["instruction"], "I?")
        self.assertEqual(row["response"], "R.")

    def test_generation_snapshot_writes_course_scoped_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = write_generation_snapshot(
                course_id="css-360-winter-2026-a7rp",
                seeds=[
                    {
                        "question": "Q?",
                        "answer": "A detailed enough.",
                        "reviewStatus": "generated",
                        "factId": "f1",
                    }
                ],
                progress={"finalCount": 1},
                export_root=root,
            )
            self.assertTrue(path.is_file())
            latest = root / "data" / "exports" / "css-360-winter-2026-a7rp" / "generated-snapshot-latest.json"
            self.assertTrue(latest.is_file())
            payload = json.loads(latest.read_text(encoding="utf-8"))
            self.assertNotIn("firebasePath", payload)
            self.assertEqual(payload["courseId"], "css-360-winter-2026-a7rp")
            self.assertEqual(payload["seedCount"], 1)


if __name__ == "__main__":
    unittest.main()
