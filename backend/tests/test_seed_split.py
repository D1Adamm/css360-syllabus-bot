"""The mixed-format train/validation split.

Every approved seed becomes a bare record (the question as the user turn) and,
when the production retriever's excerpts support its answer, a grounded record
(the shared grounded prompt as the user turn). Authored behaviour seeds become
grounded records only. The split is stratified by kind, deterministic for a
seed, and the manifest says what was built from what.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from app.grounded_generation import build_grounded_prompt, prompt_template_fingerprint
from app.retrieval_facets import extract_question_facets
from app.seed_export import write_json, write_jsonl
from app.seed_split import (
    COMPOSITION_TARGETS,
    abstention_citation_analysis,
    support_analysis,
    DEFAULT_SPLIT_SEED,
    TrainingSplitError,
    approved_export_status,
    build_mixed_records,
    composition,
    composition_percentages,
    compute_validation_size,
    prepare_training_split,
    record_kind,
    split_mixed_records,
    split_records,
    unsupported_numbers,
)
from app.storage import LocalCourseArtifactStorage

COURSE = "css-360-winter-2026-a7rp"

LATE = {"chunk_id": "chunk-067", "section": "Late Policy", "text": "one 48-hour extension per quarter, due Fridays at 11:59 p.m.", "score": 0.9}
GRADING = {"chunk_id": "chunk-058", "section": "Grading", "text": "Sprint 1: 10% Sprint 2: 15% Sprint 3: 15%", "score": 0.8}


def _record(n: int, **extra: Any) -> dict[str, Any]:
    # A figure-free answer whose wording the fake retriever's Late Policy
    # excerpt shares, so the seed is rendered grounded and attributed to it.
    return {"instruction": f"Question {n}?", "response": "An extension is available once per quarter.", **extra}


async def fake_retrieve(*, course_id: str, question: str, top_k: int = 4, storage=None):
    """The production retriever's shape, keyed on the question's subject."""
    if "extension" in question.lower():
        return "nomic-embed-text", [LATE, GRADING]
    if "sprint" in question.lower():
        return "nomic-embed-text", [GRADING]
    if "nothing" in question.lower():
        return "nomic-embed-text", []
    return "nomic-embed-text", [LATE]


def _run(coro):
    return asyncio.run(coro)


class ValidationSizeAndKindTests(unittest.TestCase):
    def test_validation_size_for_54_is_six(self) -> None:
        self.assertEqual(compute_validation_size(54), 6)
        self.assertEqual(compute_validation_size(2), 1)

    def test_split_records_is_deterministic_for_seed_360(self) -> None:
        records = [_record(i) for i in range(54)]
        train_a, val_a = split_records(records, split_seed=DEFAULT_SPLIT_SEED)
        train_b, val_b = split_records(records, split_seed=DEFAULT_SPLIT_SEED)
        self.assertEqual((len(train_a), len(val_a)), (48, 6))
        self.assertEqual((train_a, val_a), (train_b, val_b))

    def test_question_type_decides_the_kind(self) -> None:
        self.assertEqual(record_kind({}), "answerable")
        self.assertEqual(record_kind({"questionType": "direct"}), "answerable")
        self.assertEqual(record_kind({"questionType": "unanswerable"}), "abstain")
        self.assertEqual(record_kind({"questionType": "False_Premise"}), "false_premise")

    def test_unsupported_numbers_are_the_answers_numbers_the_excerpts_lack(self) -> None:
        context = "one 48-hour extension per quarter, due Fridays at 11:59 p.m. Reflection: 800-1200 words."
        self.assertEqual(unsupported_numbers("Use your 48-hour extension by 11:59 p.m.", context), [])
        self.assertEqual(unsupported_numbers("The essay is 800-1200 words.", context), [])
        self.assertEqual(unsupported_numbers("Sprint 3 is 15%.", context), ["3", "15%"])
        self.assertEqual(unsupported_numbers("No numbers here.", context), [])
        # Whole numbers only: 1 is not inside 11:59 or 1200, but 48 is in 48-hour.
        self.assertEqual(unsupported_numbers("Sprint 1 is due.", context), ["1"])
        self.assertEqual(unsupported_numbers("A 48 hour window.", context), [])
        self.assertEqual(unsupported_numbers("It is 15 percent.", "Sprint 3: 15%"), [])

    def test_support_analysis_attributes_by_figures_then_by_wording(self) -> None:
        dishonesty = {"chunk_id": "c-dish", "section": "Academic Dishonesty", "text": "The first incident of plagiarism will result in the student's receiving a zero on the plagiarized assignment."}
        ai = {"chunk_id": "c-ai", "section": "Use of AI Tools", "text": "students are permitted to use AI-based tools on some assignments"}
        feedback = {"chunk_id": "c-feedback", "section": "Bot Feedback", "text": "Bot Feedback Task: give feedback to 1 other group Due Monday, December 1, 11:59 p.m."}
        premade = {"chunk_id": "c-premade", "section": "Pre-made Demos", "text": "Record a pre-made demo of your bot (2-5 minutes). Due Monday, November 24th, 11:59 p.m."}

        # No figures: the excerpt sharing most of the answer's wording.
        no_figures = support_analysis(
            "The first incident of plagiarism results in a zero on the plagiarized assignment.", [ai, dishonesty]
        )
        self.assertEqual(no_figures["supportChunkIds"], ["c-dish"])
        self.assertTrue(no_figures["supported"])
        # Shared figures: only the excerpt holding every figure the answer states.
        figures = support_analysis(
            "Your bot demo feedback is due by 11:59 p.m. on December 1 (Monday).", [premade, feedback]
        )
        self.assertEqual(figures["supportChunkIds"], ["c-feedback"])
        self.assertEqual(figures["missingNumbers"], [])
        # Nothing in common: not supported, and says so.
        unrelated = support_analysis("Use Docker on Windows.", [LATE, GRADING])
        self.assertEqual(unrelated["supportChunkIds"], [])
        self.assertFalse(unrelated["supported"])
        # A missing figure is unsupported even when the wording matches.
        wrong_figure = support_analysis("The first of the 3 incidents is a zero on the assignment.", [dishonesty])
        self.assertEqual(wrong_figure["missingNumbers"], ["3"])
        self.assertFalse(wrong_figure["supported"])


class MixedRecordsTests(unittest.TestCase):
    def test_every_answerable_seed_is_bare_and_grounded_and_behaviour_seeds_grounded_only(self) -> None:
        records = [
            {"instruction": "Can I get an extension?", "response": "One 48-hour extension per quarter.", "seedId": "s1", "questionType": "direct"},
            {"instruction": "How many students per group?", "response": "The syllabus does not say.", "seedId": "s2", "questionType": "unanswerable"},
            {"instruction": "Grade questions go by email, right?", "response": "No: the syllabus routes it through Canvas, with one 48-hour extension per quarter.", "seedId": "s3", "questionType": "false_premise"},
        ]
        mixed, skipped = _run(build_mixed_records(COURSE, records, retrieve=fake_retrieve))

        self.assertEqual(skipped, [])
        self.assertEqual(
            [(r["format"], r["kind"], r["seedId"]) for r in mixed],
            [
                ("bare", "answerable", "s1"),
                ("grounded", "answerable", "s1"),
                ("grounded", "abstain", "s2"),
                ("grounded", "false_premise", "s3"),
            ],
        )
        bare, grounded = mixed[0], mixed[1]
        self.assertEqual(bare["instruction"], "Can I get an extension?")
        self.assertEqual(bare["response"], "One 48-hour extension per quarter.")
        # The grounded user turn is exactly the production prompt over the
        # production retriever's chunks, with the same facets.
        self.assertEqual(
            grounded["instruction"],
            build_grounded_prompt(
                "Can I get an extension?", [LATE, GRADING], extract_question_facets("Can I get an extension?")
            ),
        )
        self.assertEqual(grounded["question"], "Can I get an extension?")
        self.assertEqual([c["chunkId"] for c in grounded["context"]], ["chunk-067", "chunk-058"])
        self.assertEqual(grounded["supportChunkIds"], ["chunk-067"])
        self.assertEqual(mixed[2]["supportChunkIds"], [])
        self.assertEqual(mixed[3]["supportChunkIds"], ["chunk-067"])

    def test_an_answer_the_excerpts_do_not_support_is_not_rendered_grounded(self) -> None:
        records = [
            {"instruction": "How much is sprint three?", "response": "Sprint 3 is 30% of the grade.", "seedId": "s9"},
        ]
        mixed, skipped = _run(build_mixed_records(COURSE, records, retrieve=fake_retrieve))
        self.assertEqual([(r["format"], r["kind"]) for r in mixed], [("bare", "answerable")])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["seedId"], "s9")
        self.assertEqual(skipped[0]["missing"], ["30%"])

    def test_a_behaviour_seed_citing_an_unretrieved_figure_is_skipped_too(self) -> None:
        records = [
            {"instruction": "What time on Saturday is the Reflection due?", "response": "Not Saturday: Sunday, December 14 at 11:59 p.m.", "seedId": "fp", "questionType": "false_premise"},
            {"instruction": "How many quizzes are there?", "response": "The syllabus does not say; it mentions one 48-hour extension only.", "seedId": "ab", "questionType": "unanswerable"},
        ]
        mixed, skipped = _run(build_mixed_records(COURSE, records, retrieve=fake_retrieve))
        self.assertEqual([r["kind"] for r in mixed], ["abstain"])
        self.assertEqual([(s["seedId"], s["missing"]) for s in skipped], [("fp", ["14"])])

    def test_an_abstention_that_cites_unretrieved_details_is_skipped(self) -> None:
        group = {"chunk_id": "c-group", "section": "Notes on Group Work", "text": "Group process is key. I will be collecting confidential feedback on how groups are functioning, and I reserve the right to rearrange group membership."}
        plain = abstention_citation_analysis("How many students are in a group?", "The syllabus does not say how many students are in a group.", [LATE])
        self.assertTrue(plain["supported"])
        self.assertIsNone(plain["coverage"])
        cited = "The syllabus does not say how many students are in a group. It says the instructor collects confidential feedback on how groups are functioning and may rearrange group membership."
        self.assertTrue(abstention_citation_analysis("How many students are in a group?", cited, [group])["supported"])
        unsupported = abstention_citation_analysis("How many students are in a group?", cited, [LATE, GRADING])
        self.assertFalse(unsupported["supported"])
        self.assertIn("confidential", unsupported["missing"])

        records = [{"instruction": "How many students are in a group?", "response": cited, "seedId": "ab", "questionType": "unanswerable"}]
        mixed, skipped = _run(build_mixed_records(COURSE, records, retrieve=fake_retrieve))
        self.assertEqual(mixed, [])
        self.assertEqual(skipped[0]["reason"], "abstention cites details the retrieved excerpts do not contain")

    def test_a_question_that_retrieves_nothing_is_reported_not_rendered(self) -> None:
        records = [{"instruction": "Tell me nothing?", "response": "Nothing.", "seedId": "s0"}]
        mixed, skipped = _run(build_mixed_records(COURSE, records, retrieve=fake_retrieve))
        self.assertEqual([(r["format"]) for r in mixed], ["bare"])
        self.assertEqual(skipped[0]["reason"], "no excerpts retrieved")

    def test_the_support_check_works_without_seed_ids_and_chunk_ids(self) -> None:
        records = [{"instruction": "Can I get an extension?", "response": "Yes, once per quarter."}]
        mixed, skipped = _run(build_mixed_records(COURSE, records, retrieve=fake_retrieve))
        self.assertEqual(skipped, [])
        self.assertEqual(mixed[1]["seedId"], "record-1")
        # No figures, so attribution is by wording: the Late Policy excerpt.
        self.assertEqual(mixed[1]["supportChunkIds"], ["chunk-067"])

    def test_an_answer_or_correction_no_excerpt_supports_is_not_rendered_grounded(self) -> None:
        records = [
            {"instruction": "Can I get an extension?", "response": "Use Docker on Windows.", "seedId": "a"},
            {"instruction": "Can I get an extension by email?", "response": "Not so: bring your own laptop.", "seedId": "f", "questionType": "false_premise"},
            {"instruction": "Can I get an extension on weekends?", "response": "The syllabus does not say.", "seedId": "u", "questionType": "unanswerable"},
        ]
        mixed, skipped = _run(build_mixed_records(COURSE, records, retrieve=fake_retrieve))
        self.assertEqual([(r["format"], r["kind"]) for r in mixed], [("bare", "answerable"), ("grounded", "abstain")])
        self.assertEqual(
            [(s["seedId"], s["reason"]) for s in skipped],
            [("a", "no retrieved excerpt supports the response"), ("f", "no retrieved excerpt supports the response")],
        )
        self.assertIn("bestContainment", skipped[0])


class StratifiedSplitTests(unittest.TestCase):
    def _mixed(self, answerable: int = 20, abstain: int = 6, false_premise: int = 4) -> list[dict[str, Any]]:
        records = []
        for i in range(answerable):
            records.append({"format": "bare", "kind": "answerable", "instruction": f"Q{i}?", "response": "A."})
            records.append({"format": "grounded", "kind": "answerable", "instruction": f"P{i}", "response": "A."})
        for i in range(abstain):
            records.append({"format": "grounded", "kind": "abstain", "instruction": f"U{i}", "response": "The syllabus does not say."})
        for i in range(false_premise):
            records.append({"format": "grounded", "kind": "false_premise", "instruction": f"F{i}", "response": "Not so."})
        return records

    def test_validation_holds_every_kind_and_the_split_is_deterministic(self) -> None:
        records = self._mixed()
        train_a, val_a = split_mixed_records(records)
        train_b, val_b = split_mixed_records(records)
        self.assertEqual((train_a, val_a), (train_b, val_b))
        self.assertEqual(len(train_a) + len(val_a), len(records))
        self.assertEqual(
            composition(val_a),
            {"bare/answerable": 2, "grounded/answerable": 2, "grounded/abstain": 1, "grounded/false_premise": 1},
        )
        self.assertEqual(composition(train_a)["grounded/abstain"], 5)
        # Interleaved, not grouped: the first few training rows are not all one kind.
        self.assertGreater(len({r["kind"] for r in train_a[:6]}), 1)

    def test_a_lone_example_of_a_kind_trains_rather_than_validates(self) -> None:
        records = self._mixed(answerable=3, abstain=1, false_premise=0)
        train, validation = split_mixed_records(records)
        self.assertEqual(composition(validation)["grounded/abstain"], 0)
        self.assertEqual(composition(train)["grounded/abstain"], 1)

    def test_percentages_are_reported_against_the_targets(self) -> None:
        counts = composition(self._mixed())
        self.assertEqual(counts, {"bare/answerable": 20, "grounded/answerable": 20, "grounded/abstain": 6, "grounded/false_premise": 4})
        self.assertEqual(composition_percentages(counts), {"bare/answerable": 0.4, "grounded/answerable": 0.4, "grounded/abstain": 0.12, "grounded/false_premise": 0.08})
        self.assertEqual(sum(COMPOSITION_TARGETS.values()), 1.0)


class PrepareTrainingSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.out_dir = self.root / "data" / "exports" / COURSE
        self.out_dir.mkdir(parents=True)
        self.storage = LocalCourseArtifactStorage(root_dir=self.root / "course_data", index_dir=self.root / "indexes")
        self.storage.save_index(COURSE, {"courseId": COURSE, "chunks": [{"chunkId": "chunk-067", "sectionTitle": "Late Policy", "text": "x", "embedding": [1.0]}]})

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _export(self, records: list[dict[str, Any]]) -> None:
        write_jsonl(self.out_dir / "approved-finetune.jsonl", records)
        write_json(self.out_dir / "approved-export-summary.json", {"exportedAt": "2026-07-21T17:07:16.069789+00:00"})

    def _prepare(self, **kwargs):
        return _run(
            prepare_training_split(
                COURSE, export_root=self.root, created_at="fixed-time", storage=self.storage,
                retrieve=fake_retrieve, **kwargs,
            )
        )

    def test_writes_mixed_files_and_a_self_describing_manifest(self) -> None:
        records = [
            {"instruction": f"Can I get an extension {i}?", "response": "One 48-hour extension.", "seedId": f"a{i}", "questionType": "direct"}
            for i in range(10)
        ] + [
            {"instruction": f"Unknown thing {i}?", "response": "The syllabus does not say.", "seedId": f"u{i}", "questionType": "unanswerable"}
            for i in range(4)
        ] + [
            {"instruction": f"Wrong premise {i}?", "response": "The syllabus says otherwise: one 48-hour extension per quarter.", "seedId": f"f{i}", "questionType": "false_premise"}
            for i in range(2)
        ]
        self._export(records)

        first = self._prepare()
        second = self._prepare()

        self.assertEqual(first["sourceExamples"], 16)
        self.assertEqual(first["totalExamples"], 26)
        self.assertEqual(first["trainExamples"] + first["validationExamples"], 26)
        self.assertEqual(first["composition"]["total"], {"bare/answerable": 10, "grounded/answerable": 10, "grounded/abstain": 4, "grounded/false_premise": 2})
        self.assertEqual(first["groundedSkipped"], [])

        train_path = Path(first["files"]["trainJsonl"])
        val_path = Path(first["files"]["validationJsonl"])
        self.assertEqual(train_path.read_text(), Path(second["files"]["trainJsonl"]).read_text())
        self.assertEqual(val_path.read_text(), Path(second["files"]["validationJsonl"]).read_text())

        rows = [json.loads(line) for line in (train_path.read_text() + val_path.read_text()).splitlines() if line.strip()]
        self.assertEqual(len(rows), 26)
        for row in rows:
            self.assertTrue(row["instruction"].strip() and row["response"].strip())
            self.assertIn(row["format"], ("bare", "grounded"))
            if row["format"] == "grounded":
                self.assertIn("Syllabus excerpts:", row["instruction"])
                self.assertEqual(row["instruction"], build_grounded_prompt(row["question"], [{"section": c["section"], "text": c["text"]} for c in row["context"]], extract_question_facets(row["question"])))
            else:
                self.assertNotIn("Syllabus excerpts:", row["instruction"])

        manifest = first["manifest"]
        self.assertEqual(manifest["format"], "mixed-v4")
        self.assertEqual(manifest["datasetVersion"], f"{COURSE}-mixed-split-seed360-n26")
        self.assertEqual(manifest["sourceExportTimestamp"], "2026-07-21T17:07:16.069789+00:00")
        self.assertEqual(manifest["composition"]["validation"]["grounded/abstain"], 1)
        self.assertEqual(manifest["compositionTargets"], COMPOSITION_TARGETS)
        self.assertEqual(manifest["promptTemplate"], {"name": "grounded-v1", "sha256": prompt_template_fingerprint()})
        self.assertEqual(manifest["retrieval"]["topK"], 4)
        self.assertEqual(manifest["retrieval"]["indexChunkCount"], 1)
        self.assertEqual(len(manifest["retrieval"]["indexSha256"]), 64)
        self.assertEqual(set(manifest["checksums"]), {"train.jsonl", "validation.jsonl"})
        for key in ("sourceFile", "createdAt", "trainFile", "validationFile", "splitSeed"):
            self.assertIn(key, manifest)

    def test_an_older_export_without_question_types_still_splits(self) -> None:
        self._export([_record(i) for i in range(12)])
        summary = self._prepare()
        self.assertEqual(summary["composition"]["total"]["bare/answerable"], 12)
        self.assertEqual(summary["composition"]["total"]["grounded/abstain"], 0)
        self.assertEqual(summary["totalExamples"], 24)

    def test_skipped_grounded_seeds_are_listed_in_the_manifest(self) -> None:
        self._export([
            _record(1),
            {"instruction": "How much is sprint three?", "response": "Sprint 3 is 30%.", "seedId": "bad"},
        ])
        summary = self._prepare()
        self.assertEqual([s["seedId"] for s in summary["manifest"]["groundedSkipped"]], ["bad"])
        self.assertEqual(summary["composition"]["total"], {"bare/answerable": 2, "grounded/answerable": 1, "grounded/abstain": 0, "grounded/false_premise": 0})

    def test_missing_source_file_raises(self) -> None:
        with self.assertRaises(TrainingSplitError):
            self._prepare()

    def test_invalid_source_file_raises(self) -> None:
        (self.out_dir / "approved-finetune.jsonl").write_text('{"instruction": "Q?"}\n', encoding="utf-8")
        with self.assertRaises(TrainingSplitError):
            self._prepare()

    def test_fewer_than_two_examples_raises(self) -> None:
        self._export([_record(1)])
        with self.assertRaises(TrainingSplitError):
            self._prepare()

    def test_a_retrieval_failure_is_a_split_error(self) -> None:
        from fastapi import HTTPException

        async def failing(**kwargs):
            raise HTTPException(status_code=404, detail="No syllabus index found")

        self._export([_record(1), _record(2)])
        with self.assertRaises(TrainingSplitError) as ctx:
            _run(prepare_training_split(COURSE, export_root=self.root, storage=self.storage, retrieve=failing))
        self.assertIn("No syllabus index found", str(ctx.exception))

    def test_approved_export_status_reports_existence(self) -> None:
        self.assertFalse(approved_export_status(COURSE, export_root=self.root)["exists"])
        self._export([_record(1), _record(2)])
        status = approved_export_status(COURSE, export_root=self.root)
        self.assertTrue(status["exists"])
        self.assertEqual(status["exampleCount"], 2)


if __name__ == "__main__":
    unittest.main()
