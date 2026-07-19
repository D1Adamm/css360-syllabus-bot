"""Tests for Phase 8 approved-only export and Firebase path isolation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.firebase_seeds import course_seed_examples_path
from app.seed_export import (
    export_approved_seeds,
    finetune_record,
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
            self.assertEqual(summary["skippedCount"], 2)
            self.assertEqual(
                summary["firebasePath"],
                "courses/css-360-winter-2026-a7rp/seedExamples",
            )
            # No competing root seedExamples path.
            self.assertNotEqual(summary["firebasePath"], "seedExamples")
            self.assertEqual(
                course_seed_examples_path("css-360-winter-2026-a7rp"),
                summary["firebasePath"],
            )

            finetune_path = Path(summary["files"]["finetuneJsonl"])
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
            self.assertEqual(payload["firebasePath"], "courses/css-360-winter-2026-a7rp/seedExamples")
            self.assertEqual(payload["seedCount"], 1)


if __name__ == "__main__":
    unittest.main()
