"""Tests for Firebase seed persistence helpers."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from app.firebase_seeds import (
    FirebaseConfigurationError,
    build_firebase_seed_record,
    collect_normalized_question_keys,
    derive_source_section,
    get_firebase_database_url,
    persist_accepted_seeds,
)
from app.seed_dedupe import normalize_question_for_dedupe


class FirebaseSeedHelperTests(unittest.TestCase):
    def test_build_firebase_seed_record_includes_dual_names(self) -> None:
        record = build_firebase_seed_record(
            {
                "question": "When are office hours?",
                "answer": "Office hours are on Tuesdays at 2pm.",
                "category": "office hours",
                "sourceChunkIds": ["chunk-001"],
                "validation": {
                    "grounded": True,
                    "correct": True,
                    "clear": True,
                    "useful": True,
                    "score": 0.95,
                    "reason": "Supported by chunk.",
                },
            },
            source_section="Office Hours",
            created_at="2026-07-16T12:00:00+00:00",
        )

        self.assertEqual(record["question"], "When are office hours?")
        self.assertEqual(record["instruction"], "When are office hours?")
        self.assertEqual(record["answer"], "Office hours are on Tuesdays at 2pm.")
        self.assertEqual(record["response"], "Office hours are on Tuesdays at 2pm.")
        self.assertEqual(record["sourceChunkIds"], ["chunk-001"])
        self.assertEqual(record["sourceSection"], "Office Hours")
        self.assertEqual(record["difficulty"], "Medium")
        self.assertTrue(record["directlyAnswered"])
        self.assertEqual(record["origin"], "ai_generated")
        self.assertEqual(record["status"], "generated")
        self.assertEqual(record["createdAt"], "2026-07-16T12:00:00+00:00")
        self.assertEqual(
            record["normalizedQuestionKey"],
            normalize_question_for_dedupe("When are office hours?"),
        )
        self.assertIsNone(record["factId"])
        self.assertIsNone(record["evidenceQuote"])

    def test_build_firebase_seed_record_preserves_fact_metadata(self) -> None:
        record = build_firebase_seed_record(
            {
                "question": "Can I use an extension?",
                "answer": "One 48-hour extension is allowed.",
                "category": "late work",
                "sourceChunkIds": ["chunk-056"],
                "factId": "fact-02",
                "evidenceQuote": "one 48-hour extension per quarter",
            },
            source_section="Late Policy",
            created_at="2026-07-16T12:00:00+00:00",
        )
        self.assertEqual(record["factId"], "fact-02")
        self.assertEqual(record["evidenceQuote"], "one 48-hour extension per quarter")
        self.assertEqual(record["instruction"], record["question"])
        self.assertEqual(record["response"], record["answer"])

    def test_collect_normalized_question_keys_reads_question_and_instruction(self) -> None:
        keys = collect_normalized_question_keys(
            {
                "seed-1": {
                    "instruction": "Can I submit late?",
                    "normalizedQuestionKey": "custom-key",
                },
                "seed-2": {
                    "question": "What is the grading policy?",
                },
            }
        )

        self.assertIn("custom-key", keys)
        self.assertIn(normalize_question_for_dedupe("Can I submit late?"), keys)
        self.assertIn(normalize_question_for_dedupe("What is the grading policy?"), keys)

    def test_derive_source_section_prefers_section_title(self) -> None:
        section = derive_source_section(
            ["chunk-001", "chunk-002"],
            {"chunk-001": "Late Policy", "chunk-002": "Late Policy"},
        )
        self.assertEqual(section, "Late Policy")

    def test_get_firebase_database_url_requires_configuration(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(FirebaseConfigurationError):
                get_firebase_database_url()


class FirebaseSeedPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_persist_accepted_seeds_saves_new_records(self) -> None:
        seeds = [
            {
                "question": "When are office hours?",
                "answer": "Office hours are on Tuesdays at 2pm.",
                "category": "office hours",
                "sourceChunkIds": ["chunk-001"],
                "validation": {"score": 0.95},
            }
        ]

        with (
            patch(
                "app.firebase_seeds.fetch_course_seed_examples",
                new=AsyncMock(return_value={}),
            ),
            patch(
                "app.firebase_seeds.save_course_seed_example",
                new=AsyncMock(return_value="push-abc"),
            ) as mock_save,
        ):
            result = await persist_accepted_seeds(
                course_id="css-360-summer-2026-demo",
                seeds=seeds,
                chunk_sections={"chunk-001": "Office Hours"},
            )

        self.assertEqual(result["generatedCount"], 1)
        self.assertEqual(result["savedCount"], 1)
        self.assertEqual(result["alreadyExistingCount"], 0)
        self.assertEqual(result["failedToSaveCount"], 0)
        mock_save.assert_awaited_once()
        saved_record = mock_save.await_args.args[1]
        self.assertEqual(saved_record["sourceSection"], "Office Hours")
        self.assertEqual(saved_record["instruction"], "When are office hours?")

    async def test_persist_accepted_seeds_skips_existing_questions(self) -> None:
        seeds = [
            {
                "question": "When are office hours?",
                "answer": "Office hours are on Tuesdays at 2pm.",
                "category": "office hours",
                "sourceChunkIds": ["chunk-001"],
                "validation": {"score": 0.95},
            }
        ]
        existing_key = normalize_question_for_dedupe("When are office hours?")

        with (
            patch(
                "app.firebase_seeds.fetch_course_seed_examples",
                new=AsyncMock(
                    return_value={
                        "seed-existing": {
                            "instruction": "When are office hours?",
                            "normalizedQuestionKey": existing_key,
                        }
                    }
                ),
            ),
            patch(
                "app.firebase_seeds.save_course_seed_example",
                new=AsyncMock(),
            ) as mock_save,
        ):
            result = await persist_accepted_seeds(
                course_id="css-360-summer-2026-demo",
                seeds=seeds,
                chunk_sections={"chunk-001": "Office Hours"},
            )

        self.assertEqual(result["savedCount"], 0)
        self.assertEqual(result["alreadyExistingCount"], 1)
        mock_save.assert_not_called()

    async def test_persist_accepted_seeds_counts_partial_failures(self) -> None:
        seeds = [
            {
                "question": "Question one for saving?",
                "answer": "Answer one with enough detail.",
                "category": "general",
                "sourceChunkIds": ["chunk-001"],
                "validation": {"score": 0.95},
            },
            {
                "question": "Question two for saving?",
                "answer": "Answer two with enough detail.",
                "category": "general",
                "sourceChunkIds": ["chunk-002"],
                "validation": {"score": 0.95},
            },
        ]

        async def _save_side_effect(course_id: str, record: dict) -> str:
            if record["question"].startswith("Question one"):
                return "push-1"
            raise HTTPException(status_code=503, detail="Firebase write failed.")

        with (
            patch(
                "app.firebase_seeds.fetch_course_seed_examples",
                new=AsyncMock(return_value={}),
            ),
            patch(
                "app.firebase_seeds.save_course_seed_example",
                new=AsyncMock(side_effect=_save_side_effect),
            ),
        ):
            result = await persist_accepted_seeds(
                course_id="css-360-summer-2026-demo",
                seeds=seeds,
                chunk_sections={
                    "chunk-001": "Section 1",
                    "chunk-002": "Section 2",
                },
            )

        self.assertEqual(result["savedCount"], 1)
        self.assertEqual(result["failedToSaveCount"], 1)

    async def test_persist_enforces_course_isolation_on_save_path(self) -> None:
        with (
            patch(
                "app.firebase_seeds.fetch_course_seed_examples",
                new=AsyncMock(return_value={}),
            ) as mock_fetch,
            patch(
                "app.firebase_seeds.save_course_seed_example",
                new=AsyncMock(return_value="push-1"),
            ) as mock_save,
        ):
            await persist_accepted_seeds(
                course_id="css-430-summer-2026-ibce",
                seeds=[
                    {
                        "question": "Course-specific question here?",
                        "answer": "Course-specific answer with detail.",
                        "category": "general",
                        "sourceChunkIds": ["chunk-001"],
                        "validation": {"score": 0.9},
                    }
                ],
                chunk_sections={"chunk-001": "Syllabus"},
            )

        mock_fetch.assert_awaited_once_with("css-430-summer-2026-ibce")
        self.assertEqual(mock_save.await_args.args[0], "css-430-summer-2026-ibce")
