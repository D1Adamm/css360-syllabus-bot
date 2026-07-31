"""Tests for deterministic multi-facet question parsing and candidate merging."""

from __future__ import annotations

import unittest

from app.retrieval_diversity import (
    MULTI_FACET_TOP_K,
    select_coverage_aware_chunks,
)
from app.retrieval_facets import (
    MAX_FACETS,
    assign_missing_facet_matches,
    extract_question_facets,
    is_multi_facet_question,
    merge_scored_candidates,
)

MULTI_PART_QUESTION = (
    "Based only on the syllabus, explain how grade discussions should happen, "
    "which communication channels I should use including Discord and email, "
    "what happens with extensions and late work, and whether missed activities "
    "or exams can be made up."
)


class FacetExtractionTests(unittest.TestCase):
    def test_multi_part_question_produces_multiple_facets(self) -> None:
        facets = extract_question_facets(MULTI_PART_QUESTION)

        self.assertGreaterEqual(len(facets), 3)
        joined = " ".join(facets).lower()
        self.assertTrue(any("grade" in facet.lower() for facet in facets))
        self.assertTrue(
            any("discord" in facet.lower() or "email" in facet.lower() or "communication" in facet.lower() for facet in facets)
        )
        self.assertTrue(
            any("extension" in facet.lower() or "late" in facet.lower() for facet in facets)
        )
        self.assertIn("grade", joined)

    def test_facet_count_is_capped(self) -> None:
        long_question = (
            "Tell me about attendance, grading, late work, extensions, makeup exams, "
            "office hours, Discord rules, email turnaround, AI tools, and textbook policy."
        )
        facets = extract_question_facets(long_question)

        self.assertLessEqual(len(facets), MAX_FACETS)
        self.assertEqual(len(facets), MAX_FACETS)

    def test_ordinary_single_topic_question_keeps_simple_path(self) -> None:
        self.assertEqual(extract_question_facets("What is the late policy?"), [])
        self.assertFalse(is_multi_facet_question("What is the late policy?"))
        self.assertFalse(is_multi_facet_question("How should I contact the instructor?"))

    def test_ignores_short_or_generic_fragments(self) -> None:
        facets = extract_question_facets(
            "Based only on the syllabus, please explain the late policy and extensions."
        )
        lowered = [facet.lower() for facet in facets]
        self.assertTrue(all("syllabus" not in facet or "late" in facet or "extension" in facet for facet in lowered))
        self.assertFalse(any(facet in {"the syllabus", "please", "explain"} for facet in lowered))


class CandidateMergeTests(unittest.TestCase):
    def test_merged_candidates_are_deduplicated_by_chunk_id(self) -> None:
        full_pool = [
            {
                "chunk_id": "grade-1",
                "section": "Grade Questions",
                "text": "Discuss grades privately.",
                "score": 0.95,
                "matched_facets": [],
            },
            {
                "chunk_id": "late-1",
                "section": "Late Policy",
                "text": "Late work loses credit.",
                "score": 0.7,
                "matched_facets": [],
            },
        ]
        facet_pool = [
            {
                "chunk_id": "late-1",
                "section": "Late Policy",
                "text": "Late work loses credit.",
                "score": 0.93,
                "matched_facets": ["late work"],
            },
            {
                "chunk_id": "ext-1",
                "section": "Extensions",
                "text": "Extensions need approval.",
                "score": 0.91,
                "matched_facets": ["extensions"],
            },
        ]

        merged = merge_scored_candidates(full_pool, facet_pool)
        by_id = {chunk["chunk_id"]: chunk for chunk in merged}

        self.assertEqual(len(merged), 3)
        self.assertEqual(by_id["late-1"]["score"], 0.93)
        self.assertEqual(by_id["late-1"]["matched_facets"], ["late work"])
        self.assertEqual(by_id["ext-1"]["matched_facets"], ["extensions"])


class CoverageAwareSelectionTests(unittest.TestCase):
    def test_final_selection_covers_multiple_facets(self) -> None:
        facets = ["grade discussions", "late work", "extensions", "missed activities"]
        ranked = [
            {
                "chunk_id": "grade-1",
                "section": "Grade Questions",
                "text": "Grade discussions must be private.",
                "score": 0.99,
                "matched_facets": ["grade discussions"],
            },
            {
                "chunk_id": "grade-2",
                "section": "Grade Questions",
                "text": "Do not post grades in public channels.",
                "score": 0.98,
                "matched_facets": ["grade discussions"],
            },
            {
                "chunk_id": "grade-3",
                "section": "Grade Questions",
                "text": "Wait one week before disputing a grade.",
                "score": 0.97,
                "matched_facets": ["grade discussions"],
            },
            {
                "chunk_id": "late-1",
                "section": "Late Policy",
                "text": "Late work loses ten percent per day.",
                "score": 0.8,
                "matched_facets": ["late work"],
            },
            {
                "chunk_id": "ext-1",
                "section": "Extensions",
                "text": "Extensions require advance notice.",
                "score": 0.79,
                "matched_facets": ["extensions"],
            },
            {
                "chunk_id": "makeup-1",
                "section": "Makeup Policy",
                "text": "Missed activities generally cannot be made up.",
                "score": 0.78,
                "matched_facets": ["missed activities"],
            },
        ]

        selected = select_coverage_aware_chunks(
            ranked,
            facets=facets,
            top_k=MULTI_FACET_TOP_K,
            candidate_pool_size=10,
        )
        selected_ids = {chunk["chunk_id"] for chunk in selected}
        covered = {
            facet
            for chunk in selected
            for facet in chunk.get("matched_facets") or []
        }

        self.assertLessEqual(len(selected), MULTI_FACET_TOP_K)
        self.assertIn("grade-1", selected_ids)
        self.assertIn("late-1", selected_ids)
        self.assertIn("ext-1", selected_ids)
        self.assertTrue({"late work", "extensions"} <= covered)

    def test_assign_missing_facet_matches_uses_lexical_overlap(self) -> None:
        chunks = [
            {
                "chunk_id": "late-1",
                "section": "Late Policy",
                "text": "Late submissions lose credit after the deadline.",
                "score": 0.9,
                "matched_facets": [],
            }
        ]
        assign_missing_facet_matches(chunks, ["late work", "extensions"])
        self.assertIn("late work", chunks[0]["matched_facets"])
        self.assertNotIn("extensions", chunks[0]["matched_facets"])


if __name__ == "__main__":
    unittest.main()
