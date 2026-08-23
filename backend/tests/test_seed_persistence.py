"""Tests for PostgreSQL seed persistence and the record helpers around it.

The storage tests drive `persist_accepted_seeds` with the repository functions
stubbed. That is the right seam here: the SQL those functions emit is covered
statement-by-statement in `test_db_repositories.py`, and what this file is
about is the behaviour layered on top — which seeds are skipped as duplicates,
which are counted as failures, that one bad record does not take the batch down
with it, and that a course only ever reads and writes its own seeds.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

from fastapi import HTTPException

from app.seed_dedupe import normalize_question_for_dedupe
from app.seed_persistence import (
    build_seed_record,
    collect_normalized_question_keys,
    compute_top_up_gap,
    derive_source_section,
    persist_accepted_seeds,
    summarize_existing_seed_examples,
)


class FakeConnection:
    """Enough of a psycopg connection for the persistence layer.

    `transaction()` is the part that matters: `persist_accepted_seeds` opens one
    per seed so a single unwritable record cannot abort the others. The fake
    records each block and whether it ended in an exception, so a test can
    assert the savepoint really wrapped the failing insert.
    """

    def __init__(self) -> None:
        self.transactions: list[str] = []

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.transactions.append("begin")
        try:
            yield
        except Exception:
            self.transactions.append("rollback")
            raise
        self.transactions.append("commit")


@contextmanager
def _fake_connection(connection: FakeConnection) -> Iterator[FakeConnection]:
    yield connection


class SeedRecordHelperTests(unittest.TestCase):
    def test_build_seed_record_includes_dual_names(self) -> None:
        record = build_seed_record(
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
        self.assertEqual(record["reviewStatus"], "generated")
        self.assertEqual(record["createdAt"], "2026-07-16T12:00:00+00:00")
        self.assertEqual(
            record["normalizedQuestionKey"],
            normalize_question_for_dedupe("When are office hours?"),
        )
        self.assertIsNone(record["factId"])
        self.assertIsNone(record["evidenceQuote"])

    def test_build_seed_record_preserves_fact_metadata(self) -> None:
        record = build_seed_record(
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
                "seed-2": {"question": "What is the grading policy?"},
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

    def test_summarize_existing_seed_examples_counts_and_collects(self) -> None:
        summary = summarize_existing_seed_examples(
            {
                "seed-1": {"question": "Can I submit late?", "factId": "fact-1"},
                "seed-2": {"instruction": "What is the grading policy?"},
            }
        )
        self.assertEqual(summary["existingCount"], 2)
        self.assertEqual(summary["factIds"], {"fact-1"})
        self.assertIn(
            normalize_question_for_dedupe("Can I submit late?"),
            summary["seenQuestionKeys"],
        )

    def test_compute_top_up_gap(self) -> None:
        self.assertEqual(
            compute_top_up_gap(existing_count=9, target_count=50)["missingCount"], 41
        )
        self.assertTrue(
            compute_top_up_gap(existing_count=50, target_count=50)["alreadyComplete"]
        )


class SeedPersistenceTests(unittest.IsolatedAsyncioTestCase):
    COURSE = "css-360-summer-2026-demo"

    def _patches(
        self,
        *,
        connection: FakeConnection,
        existing: list[dict[str, Any]] | None = None,
        create_side_effect: Any = None,
        course_exists: bool = True,
    ) -> Any:
        return (
            patch(
                "app.seed_persistence.db_connection",
                lambda **kwargs: _fake_connection(connection),
            ),
            patch(
                "app.seed_persistence.course_exists",
                return_value=course_exists,
            ),
            patch(
                "app.seed_persistence.list_seeds",
                return_value=list(existing or []),
            ),
            patch(
                "app.seed_persistence.create_seed",
                side_effect=create_side_effect
                or (lambda conn, course_id, record: {**record, "id": "seed-new"}),
            ),
        )

    async def test_saves_new_records_with_derived_section(self) -> None:
        connection = FakeConnection()
        created: list[tuple[str, dict[str, Any]]] = []

        def _create(conn: Any, course_id: str, record: dict[str, Any]) -> dict[str, Any]:
            created.append((course_id, record))
            return {**record, "id": "seed-new"}

        patches = self._patches(connection=connection, create_side_effect=_create)
        with patches[0], patches[1], patches[2], patches[3]:
            result = await persist_accepted_seeds(
                course_id=self.COURSE,
                seeds=[
                    {
                        "question": "When are office hours?",
                        "answer": "Office hours are on Tuesdays at 2pm.",
                        "category": "office hours",
                        "sourceChunkIds": ["chunk-001"],
                        "validation": {"score": 0.95},
                    }
                ],
                chunk_sections={"chunk-001": "Office Hours"},
            )

        self.assertEqual(result["generatedCount"], 1)
        self.assertEqual(result["savedCount"], 1)
        self.assertEqual(result["alreadyExistingCount"], 0)
        self.assertEqual(result["failedToSaveCount"], 0)

        self.assertEqual(len(created), 1)
        course_id, record = created[0]
        self.assertEqual(course_id, self.COURSE)
        self.assertEqual(record["sourceSection"], "Office Hours")
        self.assertEqual(record["instruction"], "When are office hours?")
        # One savepoint per seed, committed.
        self.assertEqual(connection.transactions, ["begin", "commit"])

    async def test_skips_questions_the_course_already_has(self) -> None:
        connection = FakeConnection()
        existing_key = normalize_question_for_dedupe("When are office hours?")
        created: list[dict[str, Any]] = []

        def _create(conn: Any, course_id: str, record: dict[str, Any]) -> dict[str, Any]:
            created.append(record)
            return {**record, "id": "seed-new"}

        patches = self._patches(
            connection=connection,
            existing=[
                {
                    "id": "seed-existing",
                    "instruction": "When are office hours?",
                    "normalizedQuestionKey": existing_key,
                }
            ],
            create_side_effect=_create,
        )
        with patches[0], patches[1], patches[2], patches[3]:
            result = await persist_accepted_seeds(
                course_id=self.COURSE,
                seeds=[
                    {
                        "question": "When are office hours?",
                        "answer": "Office hours are on Tuesdays at 2pm.",
                        "category": "office hours",
                        "sourceChunkIds": ["chunk-001"],
                        "validation": {"score": 0.95},
                    }
                ],
                chunk_sections={"chunk-001": "Office Hours"},
            )

        self.assertEqual(result["savedCount"], 0)
        self.assertEqual(result["alreadyExistingCount"], 1)
        self.assertEqual(created, [])

    async def test_one_unwritable_seed_does_not_lose_the_others(self) -> None:
        """The savepoint's whole purpose, asserted directly."""
        connection = FakeConnection()

        def _create(conn: Any, course_id: str, record: dict[str, Any]) -> dict[str, Any]:
            if record["question"].startswith("Question two"):
                raise ValueError("that record is not storable")
            return {**record, "id": "seed-1"}

        patches = self._patches(connection=connection, create_side_effect=_create)
        with patches[0], patches[1], patches[2], patches[3]:
            result = await persist_accepted_seeds(
                course_id=self.COURSE,
                seeds=[
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
                ],
                chunk_sections={"chunk-001": "Section 1", "chunk-002": "Section 2"},
            )

        self.assertEqual(result["savedCount"], 1)
        self.assertEqual(result["failedToSaveCount"], 1)
        # The failing seed rolled its own savepoint back; the good one committed.
        self.assertEqual(
            connection.transactions, ["begin", "commit", "begin", "rollback"]
        )

    async def test_reads_and_writes_only_the_named_course(self) -> None:
        connection = FakeConnection()
        listed: list[str] = []
        created_courses: list[str] = []

        def _list(conn: Any, course_id: str) -> list[dict[str, Any]]:
            listed.append(course_id)
            return []

        def _create(conn: Any, course_id: str, record: dict[str, Any]) -> dict[str, Any]:
            created_courses.append(course_id)
            return {**record, "id": "seed-1"}

        with (
            patch(
                "app.seed_persistence.db_connection",
                lambda **kwargs: _fake_connection(connection),
            ),
            patch("app.seed_persistence.course_exists", return_value=True),
            patch("app.seed_persistence.list_seeds", side_effect=_list),
            patch("app.seed_persistence.create_seed", side_effect=_create),
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

        self.assertEqual(listed, ["css-430-summer-2026-ibce"])
        self.assertEqual(created_courses, ["css-430-summer-2026-ibce"])

    async def test_refuses_to_save_seeds_for_an_unknown_course(self) -> None:
        """A 404, not a silent zero.

        The foreign key would refuse these rows anyway; failing here says so in
        a message an operator can act on rather than surfacing a driver error.
        """
        connection = FakeConnection()
        patches = self._patches(connection=connection, course_exists=False)

        with patches[0], patches[1], patches[2], patches[3]:
            with self.assertRaises(HTTPException) as caught:
                await persist_accepted_seeds(
                    course_id="css-360-winter-2026-a7rp",
                    seeds=[
                        {
                            "question": "A question for a missing course?",
                            "answer": "An answer with enough detail.",
                            "category": "general",
                            "sourceChunkIds": [],
                            "validation": {"score": 0.9},
                        }
                    ],
                    chunk_sections={},
                )

        self.assertEqual(caught.exception.status_code, 404)

    async def test_no_seeds_touches_no_storage(self) -> None:
        with patch("app.seed_persistence.db_connection") as connect:
            result = await persist_accepted_seeds(
                course_id=self.COURSE, seeds=[], chunk_sections={}
            )
        connect.assert_not_called()
        self.assertEqual(result["generatedCount"], 0)
        self.assertEqual(result["savedCount"], 0)


if __name__ == "__main__":
    unittest.main()
