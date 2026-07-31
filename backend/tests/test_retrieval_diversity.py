"""Unit tests for diverse course RAG chunk selection."""

from __future__ import annotations

import unittest

from app.retrieval_diversity import (
    CANDIDATE_POOL_SIZE,
    DEFAULT_TOP_K,
    MAX_CHUNK_CONTEXT_CHARS,
    MAX_TOTAL_CONTEXT_CHARS,
    apply_context_budget,
    diversity_section_key,
    select_diverse_course_chunks,
    token_jaccard,
    truncate_chunk_text,
)


def _chunk(
    chunk_id: str,
    section: str,
    text: str,
    score: float,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "section": section,
        "text": text,
        "score": score,
    }


class RetrievalDiversityTests(unittest.TestCase):
    def test_multi_part_query_prefers_distinct_sections(self) -> None:
        ranked = [
            _chunk(
                "grade-1",
                "Grade Questions",
                "Grade questions must be discussed privately with the instructor.",
                0.95,
            ),
            _chunk(
                "grade-2",
                "Grade Questions",
                "Do not post grade concerns publicly in Discord channels.",
                0.94,
            ),
            _chunk(
                "contact-1",
                "Contact",
                "Email the instructor for private grade discussion and clarification.",
                0.9,
            ),
            _chunk(
                "late-1",
                "Late Policy",
                "Late work receives a daily penalty unless an extension is approved.",
                0.88,
            ),
            _chunk(
                "makeup-1",
                "Impact of Missing Class",
                "There is no makeup for missed in-class activities.",
                0.86,
            ),
            _chunk(
                "ext-1",
                "Extensions",
                "Extensions require advance notice and instructor approval.",
                0.84,
            ),
            _chunk(
                "office-1",
                "Office Hours",
                "Office hours are available for communication channel questions.",
                0.7,
            ),
        ]

        selected = select_diverse_course_chunks(ranked, top_k=5, candidate_pool_size=10)
        sections = [chunk["section"] for chunk in selected]

        self.assertEqual(len(selected), 5)
        self.assertEqual(len(set(sections)), 5)
        self.assertIn("Grade Questions", sections)
        self.assertIn("Late Policy", sections)
        self.assertIn("Impact of Missing Class", sections)
        self.assertIn("Extensions", sections)

    def test_near_duplicate_chunks_are_reduced(self) -> None:
        shared = (
            "Software Engineering (Fall 2025)\n"
            "Late submissions lose ten percent per day and may not be accepted "
            "after three days without an approved extension request from the instructor."
        )
        near_dup = (
            "Software Engineering (Fall 2025)\n"
            "Late submissions lose ten percent per day and may not be accepted "
            "after three days without an approved extension request from faculty."
        )
        distinct = (
            "Software Engineering (Fall 2025)\n"
            "Makeup exams are only offered for documented emergencies and must be "
            "requested before the original exam date whenever possible."
        )
        ranked = [
            _chunk("dup-1", "Software Engineering (Fall 2025)", shared, 0.97),
            _chunk("dup-2", "Software Engineering (Fall 2025)", near_dup, 0.96),
            _chunk("dup-3", "Software Engineering (Fall 2025)", shared + " Extra note.", 0.95),
            _chunk("other-1", "Software Engineering (Fall 2025)", distinct, 0.9),
            _chunk(
                "other-2",
                "Software Engineering (Fall 2025)",
                "Grade disputes must be raised privately within one week of the grade release.",
                0.88,
            ),
        ]

        selected = select_diverse_course_chunks(ranked, top_k=4, candidate_pool_size=10)
        selected_ids = [chunk["chunk_id"] for chunk in selected]

        self.assertLessEqual(len(selected), 4)
        self.assertIn("dup-1", selected_ids)
        self.assertNotIn("dup-2", selected_ids)
        self.assertIn("other-1", selected_ids)
        self.assertIn("other-2", selected_ids)
        self.assertGreaterEqual(token_jaccard(shared, near_dup), 0.82)

    def test_single_topic_query_keeps_top_relevant_section(self) -> None:
        ranked = [
            _chunk(
                "late-1",
                "Late Policy",
                "Late project tasks lose half credit after 24 hours.",
                0.99,
            ),
            _chunk(
                "late-2",
                "Late Policy",
                "Late project tasks may use one extension token per quarter.",
                0.91,
            ),
            _chunk(
                "office-1",
                "Office Hours",
                "Office hours are Tuesdays at 2pm for general questions.",
                0.7,
            ),
            _chunk(
                "ai-1",
                "Use of AI Tools",
                "AI tools are allowed for brainstorming with citation.",
                0.4,
            ),
        ]

        selected = select_diverse_course_chunks(ranked, top_k=3, candidate_pool_size=10)

        self.assertEqual(selected[0]["chunk_id"], "late-1")
        self.assertEqual(selected[0]["section"], "Late Policy")
        # First pass keeps one Late Policy chunk; later fill can add another
        # non-duplicate Late Policy chunk or the next best section.
        self.assertTrue(any(chunk["section"] == "Office Hours" for chunk in selected))

    def test_diversity_key_uses_inline_heading_for_generic_titles(self) -> None:
        chunk = _chunk(
            "c1",
            "Software Engineering (Fall 2025)",
            "Late Policy\nLate work loses ten percent each day.",
            0.9,
        )
        self.assertEqual(diversity_section_key(chunk), "Late Policy")

    def test_default_top_k_is_four_with_pool_of_ten(self) -> None:
        self.assertEqual(DEFAULT_TOP_K, 4)
        self.assertEqual(CANDIDATE_POOL_SIZE, 10)

    def test_default_selection_returns_at_most_four_chunks(self) -> None:
        topics = [
            "Late work loses credit each day past the deadline.",
            "Makeup exams require documentation before the exam date.",
            "Grade disputes must be raised privately within one week.",
            "Office hours happen Tuesday afternoons in the lab space.",
            "Discord channels handle general logistics announcements.",
            "Attendance surveys open one hour before every session.",
            "Group projects need a signed team charter document.",
            "Reading responses are due each Sunday evening online.",
            "Academic integrity violations go to the conduct office.",
            "Extensions require advance notice from the student.",
        ]
        ranked = [
            _chunk(f"c{index}", f"Section {index}", text, 0.9 - index / 100)
            for index, text in enumerate(topics)
        ]

        selected = select_diverse_course_chunks(ranked)

        self.assertEqual(len(selected), 4)


class ContextBudgetTests(unittest.TestCase):
    def test_truncates_at_paragraph_boundary_when_practical(self) -> None:
        first_paragraph = "Late work loses ten percent of credit per day."
        text = f"{first_paragraph}\n\n" + ("Additional trailing policy detail. " * 20)

        truncated = truncate_chunk_text(text, 80)

        self.assertEqual(truncated, first_paragraph)

    def test_truncates_at_sentence_boundary_when_no_paragraph_break(self) -> None:
        first_sentence = "Late work loses credit after the posted deadline."
        text = f"{first_sentence} " + ("Additional policy detail follows here. " * 20)

        truncated = truncate_chunk_text(text, 80)

        self.assertEqual(truncated, first_sentence)
        self.assertTrue(truncated.endswith("."))
        self.assertLessEqual(len(truncated), 80)

    def test_falls_back_to_word_boundary_without_usable_punctuation(self) -> None:
        text = "policy detail " * 30

        truncated = truncate_chunk_text(text, 50)

        self.assertLessEqual(len(truncated), 50)
        self.assertFalse(truncated.endswith(" "))
        self.assertTrue(truncated.startswith("policy detail"))

    def test_short_text_is_unchanged(self) -> None:
        self.assertEqual(truncate_chunk_text("Short policy.", 500), "Short policy.")

    def test_per_chunk_and_total_limits_are_enforced(self) -> None:
        long_text = "Policy sentence about deadlines and penalties. " * 200
        chunks = [_chunk(f"c{index}", f"Section {index}", long_text, 0.9) for index in range(4)]

        budgeted = apply_context_budget(chunks)

        total = sum(len(chunk["text"]) for chunk in budgeted)
        self.assertLessEqual(total, MAX_TOTAL_CONTEXT_CHARS)
        for chunk in budgeted:
            self.assertLessEqual(len(chunk["text"]), MAX_CHUNK_CONTEXT_CHARS)

    def test_budget_preserves_order_and_source_metadata(self) -> None:
        long_text = "Policy sentence about deadlines and penalties. " * 100
        chunks = [
            _chunk("top-1", "Late Policy", long_text, 0.99),
            _chunk("top-2", "Makeup Policy", long_text, 0.95),
            _chunk("top-3", "Contact", long_text, 0.9),
        ]

        budgeted = apply_context_budget(chunks)

        self.assertEqual(budgeted[0]["chunk_id"], "top-1")
        self.assertEqual(budgeted[0]["section"], "Late Policy")
        self.assertEqual(budgeted[0]["score"], 0.99)
        ids = [chunk["chunk_id"] for chunk in budgeted]
        self.assertEqual(ids, sorted(ids, key=lambda item: ["top-1", "top-2", "top-3"].index(item)))

    def test_budget_does_not_mutate_input_chunks(self) -> None:
        long_text = "Policy sentence about deadlines and penalties. " * 100
        chunks = [_chunk("c1", "Late Policy", long_text, 0.9)]

        apply_context_budget(chunks)

        self.assertEqual(chunks[0]["text"], long_text)


if __name__ == "__main__":
    unittest.main()
