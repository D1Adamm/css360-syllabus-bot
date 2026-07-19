"""Tests for Phase 8 seed review status and edit provenance."""

from __future__ import annotations

import unittest

from app.seed_review import (
    apply_seed_review,
    is_approved_for_export,
    resolve_review_status,
)


class SeedReviewTests(unittest.TestCase):
    def test_resolve_prefers_review_status(self) -> None:
        self.assertEqual(
            resolve_review_status({"reviewStatus": "approved", "status": "generated"}),
            "approved",
        )
        self.assertEqual(resolve_review_status({"status": "rejected"}), "rejected")
        self.assertEqual(resolve_review_status({}), "generated")

    def test_edit_preserves_grounding_and_originals(self) -> None:
        record = {
            "id": "seed-1",
            "question": "May I submit late?",
            "answer": "Late work may be submitted within 24 hours.",
            "instruction": "May I submit late?",
            "response": "Late work may be submitted within 24 hours.",
            "factId": "fact-02",
            "evidenceQuote": "Late work may be submitted within 24 hours.",
            "sourceChunkIds": ["chunk-001"],
            "origin": "ai_generated",
            "status": "generated",
            "reviewStatus": "generated",
            "validation": {"score": 0.9, "reason": "ok", "unsupportedClaims": [], "components": {}},
        }
        updated = apply_seed_review(
            record,
            review_status="edited",
            question="Can I turn work in late?",
            answer="You may submit late work within 24 hours for half credit.",
            review_notes="Clearer student wording",
        )
        self.assertEqual(updated["reviewStatus"], "edited")
        self.assertEqual(updated["status"], "edited")
        self.assertEqual(updated["originalQuestion"], "May I submit late?")
        self.assertEqual(
            updated["originalAnswer"],
            "Late work may be submitted within 24 hours.",
        )
        self.assertEqual(updated["question"], "Can I turn work in late?")
        self.assertEqual(updated["instruction"], "Can I turn work in late?")
        self.assertEqual(updated["factId"], "fact-02")
        self.assertEqual(
            updated["evidenceQuote"],
            "Late work may be submitted within 24 hours.",
        )
        self.assertEqual(updated["sourceChunkIds"], ["chunk-001"])
        self.assertEqual(updated["origin"], "ai_generated")
        self.assertIsNotNone(updated["validation"])
        self.assertEqual(updated["reviewNotes"], "Clearer student wording")

    def test_approve_without_edit_does_not_invent_originals(self) -> None:
        record = {
            "question": "Q?",
            "answer": "A sufficiently detailed answer.",
            "reviewStatus": "generated",
            "factId": "fact-1",
        }
        updated = apply_seed_review(record, review_status="approved")
        self.assertEqual(updated["reviewStatus"], "approved")
        self.assertNotIn("originalQuestion", updated)
        self.assertTrue(is_approved_for_export(updated))
        self.assertFalse(is_approved_for_export(record))

    def test_text_change_with_generated_status_becomes_edited(self) -> None:
        record = {"question": "Old?", "answer": "Old answer here."}
        updated = apply_seed_review(
            record,
            review_status="generated",
            question="New?",
            answer="New answer here.",
        )
        self.assertEqual(updated["reviewStatus"], "edited")

    def test_invalid_status_raises(self) -> None:
        with self.assertRaises(ValueError):
            apply_seed_review({"question": "Q?", "answer": "A."}, review_status="accepted")


if __name__ == "__main__":
    unittest.main()
