"""Unit tests for inference comparison helpers (no GPU / no model download)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from compare_inference import (
    ComparisonError,
    build_comparison_summary,
    find_exact_training_overlap,
    find_near_duplicate_warning,
    load_heldout_questions,
    validate_heldout_against_references,
    write_json,
    write_jsonl,
)


class CompareInferenceHelperTests(unittest.TestCase):
    def test_load_heldout_questions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "heldout.json"
            questions = [f"Question {i} about CSS 360 topic {i}?" for i in range(1, 9)]
            path.write_text(json.dumps({"questions": questions}), encoding="utf-8")
            loaded = load_heldout_questions(path)
            self.assertEqual(len(loaded), 8)
            self.assertEqual(loaded[0], questions[0])

    def test_exact_overlap_detection(self) -> None:
        refs = ["What should I do if I need to miss class?"]
        self.assertTrue(
            find_exact_training_overlap(
                "what should i do if i need to miss class?",
                refs,
            )
        )
        self.assertFalse(
            find_exact_training_overlap(
                "If I arrive late, do I still need to document it?",
                refs,
            )
        )

    def test_near_duplicate_warning_logic(self) -> None:
        refs = ["What should I do if I need to miss class?"]
        warning = find_near_duplicate_warning(
            "What should I do if I need to miss class today?",
            refs,
            threshold=0.6,
        )
        self.assertIsNotNone(warning)
        self.assertIn("Near-duplicate", warning or "")

        no_warning = find_near_duplicate_warning(
            "Where is the campus bookstore open on Sundays?",
            refs,
            threshold=0.72,
        )
        self.assertIsNone(no_warning)

    def test_validate_rejects_exact_overlap(self) -> None:
        refs = ["Can I use AI tools like Copilot on assignments in this course?"]
        with self.assertRaises(ComparisonError):
            validate_heldout_against_references(
                ["Can I use AI tools like Copilot on assignments in this course?"],
                refs,
            )

    def test_result_serialization(self) -> None:
        results = [
            {
                "question": "Sample held-out question?",
                "baseResponse": "Base answer.",
                "fineTunedResponse": "Fine-tuned answer.",
                "baseGenerationSeconds": 1.25,
                "fineTunedGenerationSeconds": 1.5,
                "exactTrainingOverlap": False,
                "nearDuplicateWarning": None,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "comparison_results.json", results)
            write_jsonl(root / "comparison_results.jsonl", results)
            loaded = json.loads((root / "comparison_results.json").read_text(encoding="utf-8"))
            line = (root / "comparison_results.jsonl").read_text(encoding="utf-8").strip()
            self.assertEqual(loaded[0]["question"], "Sample held-out question?")
            self.assertEqual(json.loads(line)["fineTunedResponse"], "Fine-tuned answer.")

    def test_summary_creation(self) -> None:
        summary = build_comparison_summary(
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            adapter_path="/gpfs/projects/simswe/user/training_outputs/css-360-qlora/adapter",
            question_count=9,
            total_base_generation_seconds=12.0,
            total_finetuned_generation_seconds=15.5,
            git_commit_sha="abc123",
            slurm_job_id="42",
        )
        self.assertEqual(summary["questionCount"], 9)
        self.assertEqual(summary["totalBaseGenerationSeconds"], 12.0)
        self.assertEqual(summary["totalFineTunedGenerationSeconds"], 15.5)
        self.assertEqual(summary["slurmJobId"], "42")
        self.assertIn("timestamp", summary)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "comparison_summary.json"
            write_json(path, summary)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["modelId"], "meta-llama/Llama-3.2-3B-Instruct")


if __name__ == "__main__":
    unittest.main()
