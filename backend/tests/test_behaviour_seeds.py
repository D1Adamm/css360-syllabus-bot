"""Authored behaviour examples become seeds, validated before any write."""

from __future__ import annotations

import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path

from app.behaviour_seeds import (
    BehaviourSeedError,
    behaviour_seed,
    load_behaviour_records,
    normalized_question,
)

AUTHORED = Path(__file__).resolve().parents[2] / "training" / "behaviour_seeds" / "css-360-winter-2026-a7rp.jsonl"

ABSTAIN = {
    "question": "How many students are in each project group?",
    "questionType": "unanswerable",
    "response": "The syllabus does not say how many students are in a group.",
}
CORRECT = {
    "question": "What time on Saturday is the Reflection due?",
    "questionType": "false_premise",
    "response": "The Reflection is not due on a Saturday: the syllabus sets it for Sunday, December 14.",
}


class BehaviourSeedTests(unittest.TestCase):
    def test_an_abstention_becomes_a_pending_authored_seed(self) -> None:
        seed = behaviour_seed(ABSTAIN, now="2026-09-11T00:00:00+00:00")
        self.assertEqual(seed["instruction"], ABSTAIN["question"])
        self.assertEqual(seed["response"], ABSTAIN["response"])
        self.assertEqual(seed["questionType"], "unanswerable")
        self.assertEqual(seed["origin"], "authored")
        self.assertEqual(seed["reviewStatus"], "generated")
        self.assertEqual(seed["status"], "generated")
        self.assertFalse(seed["directlyAnswered"])
        self.assertEqual(seed["category"], "behaviour")
        self.assertNotIn("reviewedAt", seed)

    def test_approve_marks_it_approved_with_a_note(self) -> None:
        seed = behaviour_seed(CORRECT, approve=True, now="2026-09-11T00:00:00+00:00")
        self.assertEqual(seed["reviewStatus"], "approved")
        self.assertEqual(seed["reviewedAt"], "2026-09-11T00:00:00+00:00")
        self.assertIn("false-premise correction", seed["reviewNotes"])
        self.assertTrue(seed["directlyAnswered"])

    def test_invalid_records_are_refused_with_a_reason(self) -> None:
        cases = {
            "questionType": {**ABSTAIN, "questionType": "direct"},
            "response": {**ABSTAIN, "response": "  "},
            "question": {**ABSTAIN, "question": ""},
            "syllabus": {**ABSTAIN, "response": "I do not know."},
        }
        for name, record in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(BehaviourSeedError):
                    behaviour_seed(record)

    def test_the_authored_css360_file_is_valid_and_balanced(self) -> None:
        records = load_behaviour_records(AUTHORED)
        counts = {kind: sum(1 for r in records if r["questionType"] == kind) for kind in ("unanswerable", "false_premise")}
        self.assertEqual(counts, {"unanswerable": 20, "false_premise": 14})
        self.assertEqual([r["line"] for r in records], list(range(1, 35)))
        self.assertEqual(len({normalized_question(r["question"]) for r in records}), 34)
        for record in records:
            if record["questionType"] == "unanswerable":
                self.assertTrue(record.get("absentTerms"), record["question"])

    def test_every_abstention_asks_for_something_the_syllabus_does_not_state(self) -> None:
        """`absentTerms` are phrases that would answer the question; none may
        occur in the syllabus. Needs the local syllabus text, which is not in
        the repository, so this is skipped where it is absent."""
        syllabus_path = Path(__file__).resolve().parents[1] / "course_data" / "css-360-winter-2026-a7rp" / "syllabus.txt"
        if not syllabus_path.is_file():
            self.skipTest("local CSS 360 syllabus text is not present")
        syllabus = " ".join(syllabus_path.read_text(encoding="utf-8").lower().split())
        for record in load_behaviour_records(AUTHORED):
            if record["questionType"] != "unanswerable":
                continue
            for term in record["absentTerms"]:
                pattern = r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])"
                self.assertIsNone(
                    re.search(pattern, syllabus),
                    f"{record['question']!r}: the syllabus contains {term!r}",
                )

    def test_no_authored_question_overlaps_a_held_out_question(self) -> None:
        """The two held-out sets stay held out: same measures and thresholds as
        evaluation/check_overlap.py, every authored question against every
        held-out question for the course."""
        evaluation = Path(__file__).resolve().parents[2] / "evaluation"
        spec = importlib.util.spec_from_file_location("check_overlap", evaluation / "check_overlap.py")
        check_overlap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(check_overlap)

        held_out: list[tuple[str, str]] = []
        bank = json.loads((evaluation / "held_out_questions.json").read_text(encoding="utf-8"))
        for course in bank["courses"]:
            if course["courseId"] == "css-360-winter-2026-a7rp":
                held_out += [(q["id"], q["question"]) for q in course["questions"]]
        benchmark = evaluation / "model_version_benchmark" / "questions.json"
        if benchmark.is_file():
            held_out += [(q["id"], q["question"]) for q in json.loads(benchmark.read_text())["questions"]]
        self.assertGreaterEqual(len(held_out), 20)

        for record in load_behaviour_records(AUTHORED):
            scores, label, other = check_overlap.nearest(record["question"], held_out)
            self.assertEqual(
                check_overlap.verdict_for(scores), "clearly held out",
                f"{record['question']!r} vs held-out {label} {other!r}: {scores}",
            )

    def test_a_bad_line_fails_the_whole_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            path.write_text(json.dumps(ABSTAIN) + "\nnot json\n", encoding="utf-8")
            with self.assertRaises(BehaviourSeedError):
                load_behaviour_records(path)
            path.write_text("", encoding="utf-8")
            with self.assertRaises(BehaviourSeedError):
                load_behaviour_records(path)

    def test_normalised_questions_ignore_case_spacing_and_punctuation(self) -> None:
        self.assertEqual(
            normalized_question("  How many   students are in each group?"),
            normalized_question("how many students are in each group"),
        )


if __name__ == "__main__":
    unittest.main()
