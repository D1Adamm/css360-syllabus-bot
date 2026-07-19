"""Tests for Phase 8 dataset quality inspection."""

from __future__ import annotations

import unittest

from app.seed_dataset_quality import inspect_seed_dataset


class SeedDatasetQualityTests(unittest.TestCase):
    def test_flags_duplicates_and_coverage(self) -> None:
        seeds = [
            {
                "id": "1",
                "question": "What is the late work policy?",
                "answer": "Late work may be submitted within 24 hours.",
                "evidenceQuote": "Late work may be submitted within 24 hours.",
                "factId": "f1",
                "origin": "ai_generated",
                "reviewStatus": "generated",
                "validation": {
                    "score": 0.9,
                    "unsupportedClaims": [],
                    "reason": "ok",
                    "components": {},
                },
            },
            {
                "id": "2",
                "question": "What is the late work policy??",
                "answer": "Students must submit late work within 24 hours.",
                "evidenceQuote": "Late work may be submitted within 24 hours.",
                "factId": "f2",
                "origin": "ai_generated",
                "reviewStatus": "generated",
                "validation": {
                    "score": 0.7,
                    "unsupportedClaims": ["Invented must"],
                    "reason": "weak",
                    "components": {},
                },
            },
            {
                "id": "3",
                "question": "Where are office hours?",
                "answer": "Office hours are Tuesdays at 2pm for help.",
                "evidenceQuote": "Office hours are Tuesdays at 2pm.",
                "factId": "f3",
                "origin": "ai_generated",
                "reviewStatus": "approved",
                "category": "contact",
            },
        ]
        report = inspect_seed_dataset(seeds)
        self.assertEqual(report["seedCount"], 3)
        self.assertEqual(report["approvedCount"], 1)
        self.assertIn("near_duplicate_question", report["issueCounts"])
        self.assertIn("low_validation_score", report["issueCounts"])
        self.assertIn("unsupported_claims", report["issueCounts"])
        self.assertIn("missing_qualifiers_or_modal_escalation", report["issueCounts"])
        self.assertIn("late_work_extensions", report["coverage"]["presentThemes"])
        self.assertIn("instructor_contact_help", report["coverage"]["presentThemes"])


if __name__ == "__main__":
    unittest.main()
