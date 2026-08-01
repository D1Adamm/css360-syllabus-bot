"""Lightweight RAG quality pass: prompt fidelity + relevance floor."""

from __future__ import annotations

import unittest

from app.rag import build_rag_prompt
from app.retrieval_diversity import (
    DEFAULT_TOP_K,
    MAX_CHUNK_CONTEXT_CHARS,
    MAX_FINAL_TOP_K,
    MAX_TOTAL_CONTEXT_CHARS,
    MULTI_FACET_TOP_K,
    MULTI_FACET_TOTAL_CONTEXT_CHARS,
    RELATIVE_SCORE_FLOOR,
    apply_relevance_floor,
    resolve_final_top_k,
    resolve_total_context_limit,
)


def _chunk(
    chunk_id: str,
    section: str,
    text: str,
    score: float,
    matched_facets: list[str] | None = None,
) -> dict:
    payload = {
        "chunk_id": chunk_id,
        "section": section,
        "text": text,
        "score": score,
    }
    if matched_facets is not None:
        payload["matched_facets"] = matched_facets
    return payload


class RagPromptQualityTests(unittest.TestCase):
    def test_ordered_lists_must_remain_in_source_order(self) -> None:
        prompt = build_rag_prompt(
            "How should I contact the instructor?",
            [
                {
                    "section": "Contact",
                    "text": (
                        "Prefer, in this order:\n"
                        "1 public messages via Discord\n"
                        "2 private messages via Discord\n"
                        "3 private messages via Canvas"
                    ),
                }
            ],
        )
        self.assertIn(
            "Preserve numbered and ordered lists in the exact order shown",
            prompt,
        )
        self.assertIn("Do not reverse, reorder, or renumber", prompt)

    def test_exact_numeric_limits_and_exceptions_must_be_preserved(self) -> None:
        prompt = build_rag_prompt(
            "What is the extension policy?",
            [
                {
                    "section": "Late work",
                    "text": (
                        "You may use one 48-hour extension per quarter, no questions asked. "
                        "No extension is possible for the Demo assignment."
                    ),
                }
            ],
        )
        self.assertIn("Preserve exact numbers", prompt)
        self.assertIn("quantities, limits, durations", prompt)
        self.assertIn("no questions asked", prompt)
        self.assertIn("one 48-hour extension", prompt)
        self.assertIn("No extension is possible for the Demo assignment.", prompt)

    def test_unsupported_conditions_must_not_be_added(self) -> None:
        prompt = build_rag_prompt(
            "Can I get an extension?",
            [{"section": "Late work", "text": "one 48-hour extension, no questions asked."}],
        )
        self.assertIn(
            "Never add qualifications, eligibility requirements, reasons, exceptions",
            prompt,
        )
        self.assertIn("claim-by-claim check against the excerpts", prompt)

    def test_policy_scope_must_not_be_generalized(self) -> None:
        prompt = build_rag_prompt(
            "What happens if I miss standup?",
            [
                {
                    "section": "Missed activities",
                    "text": "Missing standup affects participation credit for standup only.",
                }
            ],
        )
        self.assertIn(
            "Never broaden a policy about one assignment, activity, session, or situation",
            prompt,
        )
        self.assertIn("all assignments, exams, attendance, missed work", prompt)

    def test_multi_facet_prompt_lists_facets_and_requires_each_part(self) -> None:
        facets = [
            "preferred contact order",
            "extension limits and exclusions",
            "missed activity makeup rules",
        ]
        prompt = build_rag_prompt(
            "Explain contact order, extensions, and missed activity makeup.",
            [
                {"section": "Contact", "text": "Prefer public Discord before private Discord."},
                {"section": "Late work", "text": "One 48-hour extension, no questions asked."},
            ],
            facets=facets,
        )
        self.assertIn("Requested parts of the question", prompt)
        for facet in facets:
            self.assertIn(facet, prompt)
        self.assertIn("Address every requested part exactly once", prompt)
        self.assertIn("retrieved excerpts do not specify", prompt)
        self.assertIn("one concise subsection or labeled paragraph per requested part", prompt)
        self.assertIn("Omit unrelated details", prompt)

    def test_single_topic_prompt_omits_facet_block(self) -> None:
        prompt = build_rag_prompt(
            "What is the late policy?",
            [{"section": "Late work", "text": "Late work loses 10% per day."}],
            facets=[],
        )
        self.assertNotIn("Requested parts of the question", prompt)
        self.assertNotIn("subsection or labeled paragraph per requested part", prompt)
        self.assertIn("Student question:\nWhat is the late policy?", prompt)

    def test_prompt_is_course_generic(self) -> None:
        prompt = build_rag_prompt(
            "What is the attendance policy?",
            [{"section": "Attendance", "text": "Submit the absence form before class."}],
        )
        self.assertNotIn("CSS360", prompt)
        self.assertNotIn("CSS 360", prompt)
        self.assertIn("course syllabus", prompt)


class RelevanceFloorTests(unittest.TestCase):
    def test_unrelated_low_relevance_chunk_filtered_from_multi_facet(self) -> None:
        question = (
            "Explain grade discussion rules, late extensions, and missed class makeup."
        )
        facets = ["grade discussion", "late extensions", "missed class makeup"]
        selected = [
            _chunk(
                "grade-1",
                "Grade Questions",
                "Book a private timeslot within 1 week for grade discussion.",
                0.92,
                matched_facets=["grade discussion"],
            ),
            _chunk(
                "late-1",
                "Late Policy",
                "One 48-hour extension per quarter, no questions asked.",
                0.90,
                matched_facets=["late extensions"],
            ),
            _chunk(
                "miss-1",
                "Impact of Missing Class",
                "There is no direct makeup for missed in-class activities.",
                0.88,
                matched_facets=["missed class makeup"],
            ),
            _chunk(
                "ai-1",
                "Use of AI Tools",
                "Students may use AI tools when assignment instructions allow it.",
                0.41,
                matched_facets=[],
            ),
        ]
        kept = apply_relevance_floor(
            selected,
            question=question,
            facets=facets,
            multi_facet=True,
        )
        kept_ids = [chunk["chunk_id"] for chunk in kept]
        self.assertIn("grade-1", kept_ids)
        self.assertIn("late-1", kept_ids)
        self.assertIn("miss-1", kept_ids)
        self.assertNotIn("ai-1", kept_ids)
        self.assertLessEqual(len(kept), MAX_FINAL_TOP_K)

        for chunk in kept:
            diag = chunk["retrieval_diagnostics"]
            self.assertIn(diag["retentionReason"], {"top_score", "facet_match", "lexical_overlap"})
            self.assertIn("fullQueryScore", diag)
            self.assertIn("lexicalOverlap", diag)
            self.assertIn("diversityKey", diag)
            self.assertIn("coverage_contribution", chunk)

    def test_semantic_neighbor_with_low_literal_overlap_is_kept(self) -> None:
        question = "What happens if coursework arrives after the due date?"
        selected = [
            _chunk(
                "late-semantic",
                "Submission timing",
                "Work turned in past the deadline receives a daily penalty unless approved.",
                0.91,
            ),
            _chunk(
                "noise",
                "Playlist",
                "Optional music recommendations for the quarter.",
                0.30,
            ),
        ]
        kept = apply_relevance_floor(
            selected,
            question=question,
            facets=[],
            multi_facet=False,
        )
        kept_ids = [chunk["chunk_id"] for chunk in kept]
        self.assertIn("late-semantic", kept_ids)
        self.assertNotIn("noise", kept_ids)
        self.assertEqual(
            kept[0]["retrieval_diagnostics"]["retentionReason"],
            "top_score",
        )

    def test_high_score_low_overlap_neighbor_not_removed(self) -> None:
        """Cosine-strong paraphrase should survive even without shared content tokens."""
        question = "extension allowance for project deliverables"
        selected = [
            _chunk(
                "anchor",
                "Late Policy",
                "extension allowance for project deliverables is limited to one use",
                0.95,
                matched_facets=["extension allowance"],
            ),
            _chunk(
                "paraphrase",
                "Deferral guidance",
                "A single forty-eight hour grace window may be requested privately.",
                0.86,  # above 0.55 * 0.95
                matched_facets=[],
            ),
        ]
        kept = apply_relevance_floor(
            selected,
            question=question,
            facets=["extension allowance"],
            multi_facet=True,
        )
        self.assertEqual({chunk["chunk_id"] for chunk in kept}, {"anchor", "paraphrase"})
        paraphrase = next(chunk for chunk in kept if chunk["chunk_id"] == "paraphrase")
        self.assertEqual(
            paraphrase["retrieval_diagnostics"]["retentionReason"],
            "relative_score",
        )

    def test_ordinary_single_topic_keeps_near_top_candidates(self) -> None:
        question = "What is the late policy?"
        selected = [
            _chunk(
                "late-1",
                "Late Policy",
                "Late submissions lose ten percent per day.",
                0.93,
            ),
            _chunk(
                "late-2",
                "Late Policy details",
                "Weekend due dates still count toward the late submission penalty.",
                0.84,
            ),
            _chunk(
                "late-3",
                "Deadlines",
                "Assignments are due at 11:59 p.m. unless noted otherwise.",
                0.79,
            ),
            _chunk(
                "late-4",
                "Extensions",
                "Ask before the deadline if you need an extension.",
                0.74,
            ),
        ]
        kept = apply_relevance_floor(
            selected,
            question=question,
            facets=[],
            multi_facet=False,
        )
        self.assertEqual(len(kept), 4)
        self.assertEqual([chunk["chunk_id"] for chunk in kept], [c["chunk_id"] for c in selected])

    def test_topk_and_context_budget_constants_unchanged(self) -> None:
        self.assertEqual(DEFAULT_TOP_K, 4)
        self.assertEqual(MULTI_FACET_TOP_K, 5)
        self.assertEqual(MAX_FINAL_TOP_K, 5)
        self.assertEqual(MAX_CHUNK_CONTEXT_CHARS, 900)
        self.assertEqual(MAX_TOTAL_CONTEXT_CHARS, 3000)
        self.assertEqual(MULTI_FACET_TOTAL_CONTEXT_CHARS, 4200)
        self.assertEqual(resolve_final_top_k(4, multi_facet=False), 4)
        self.assertEqual(resolve_final_top_k(4, multi_facet=True), 5)
        self.assertEqual(resolve_total_context_limit(multi_facet=False), 3000)
        self.assertEqual(resolve_total_context_limit(multi_facet=True), 4200)
        self.assertEqual(RELATIVE_SCORE_FLOOR, 0.55)


if __name__ == "__main__":
    unittest.main()
