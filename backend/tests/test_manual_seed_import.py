"""The staff seed import: validate everything first, write through create_seed, never twice.

The database helpers are replaced by an in-memory store keyed by course, so
these tests see exactly which rows the importer would write and to where.
"""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from app import manual_seed_import as importer
from app.manual_seed_import import (
    ImportValidationError,
    manual_seed_id,
    parse_seed_file,
    run_import,
)

COURSE = "css360d-fall-2026-q0ne"
OTHER = "css-350-fall-2026-abcd"
BACKEND = Path(__file__).resolve().parent.parent

SEEDS = [
    {
        "question": "When are assignments due?",
        "answer": "Unless otherwise noted, assignments are due at 11:59 PM Pacific on the listed due date.",
        "category": "Assignments",
        "sourceSection": "Assignments",
    },
    {"question": "Where does the class meet?", "answer": "UW2-005."},
    {"question": "Is there a final exam?", "answer": "No. The course ends with a team project."},
]


class FakeStore:
    """Seeds per course, and the calls the importer makes."""

    def __init__(self) -> None:
        self.courses = {COURSE, OTHER}
        self.seeds: dict[str, dict[str, dict[str, Any]]] = {COURSE: {}, OTHER: {}}
        self.fail_on: set[str] = set()
        self.committed = False

    # the connection
    @contextmanager
    def connection(self, **kwargs: Any) -> Iterator["FakeStore"]:
        yield self

    @contextmanager
    def transaction(self) -> Iterator[None]:
        yield

    def rollback(self) -> None:
        pass

    # the repositories
    def course_exists(self, conn: Any, course_id: str) -> bool:
        return course_id in self.courses

    def list_seeds(self, conn: Any, course_id: str) -> list[dict[str, Any]]:
        return list(self.seeds[course_id].values())

    def create_seed(
        self, conn: Any, course_id: str, seed: dict[str, Any], *, seed_id: str | None = None
    ) -> dict[str, Any]:
        if seed["question"] in self.fail_on:
            raise RuntimeError("simulated insert failure")
        if seed_id in self.seeds[course_id]:
            raise RuntimeError("duplicate key value violates unique constraint")
        record = {**seed, "id": seed_id, "instruction": seed["question"]}
        self.seeds[course_id][seed_id] = record
        return record

    def count_seeds_by_review_status(self, conn: Any, course_id: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for seed in self.seeds[course_id].values():
            status = seed.get("reviewStatus") or "generated"
            counts[status] = counts.get(status, 0) + 1
        return counts


class ImporterTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FakeStore()
        for target, replacement in (
            ("app.manual_seed_import.db_connection", self.store.connection),
            ("app.db_courses.course_exists", self.store.course_exists),
            ("app.db_seeds.list_seeds", self.store.list_seeds),
            ("app.db_seeds.create_seed", self.store.create_seed),
            ("app.db_seeds.count_seeds_by_review_status", self.store.count_seeds_by_review_status),
        ):
            patcher = patch(target, new=replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        self.lines: list[str] = []

    def write(self, payload: Any, name: str = "seeds.json") -> Path:
        path = self.dir / name
        path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
        return path

    def run_import(self, payload: Any = SEEDS, **kwargs: Any) -> int:
        kwargs.setdefault("confirm", lambda plan: True)
        return run_import(COURSE, self.write(payload), out=self.lines.append, **kwargs)

    def output(self) -> str:
        return "\n".join(self.lines)

    def stored(self, course_id: str = COURSE) -> list[dict[str, Any]]:
        return list(self.store.seeds[course_id].values())


class ValidationTests(unittest.TestCase):
    def problems(self, payload: Any) -> list[str]:
        text = payload if isinstance(payload, str) else json.dumps(payload)
        with self.assertRaises(ImportValidationError) as caught:
            parse_seed_file(text)
        return caught.exception.problems

    def test_the_file_must_be_json(self) -> None:
        self.assertIn("not valid JSON", self.problems("[{question: 1}")[0])

    def test_the_file_must_be_a_non_empty_list_of_objects(self) -> None:
        self.assertIn("JSON list", self.problems({"question": "q", "answer": "a"})[0])
        self.assertIn("no seeds", self.problems([])[0])
        self.assertIn("must be an object", self.problems(["a question"])[0])

    def test_question_and_answer_are_required_and_non_empty(self) -> None:
        problems = self.problems(
            [{"question": "Q?"}, {"answer": "A."}, {"question": "   ", "answer": "A."}, {"question": "Q2?", "answer": ""}]
        )
        self.assertEqual(len(problems), 4)
        self.assertIn("Seed 1: 'answer' is missing or empty.", problems)
        self.assertIn("Seed 2: 'question' is missing or empty.", problems)

    def test_unknown_fields_are_refused_so_typos_are_caught(self) -> None:
        problems = self.problems([{"question": "Q?", "anwser": "A."}])
        self.assertTrue(any("unknown field(s) anwser" in p for p in problems))

    def test_optional_fields_must_be_text(self) -> None:
        problems = self.problems([{"question": "Q?", "answer": "A.", "category": 3}])
        self.assertIn("Seed 1: 'category' must be text.", problems)

    def test_duplicate_questions_inside_the_file_are_refused(self) -> None:
        problems = self.problems(
            [
                {"question": "When are assignments due?", "answer": "A."},
                {"question": "  when are ASSIGNMENTS due  ", "answer": "B."},
            ]
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("Seed 2: duplicates the question in seed 1", problems[0])

    def test_every_problem_is_reported_at_once(self) -> None:
        problems = self.problems([{"question": "Q?"}, {"answer": "A."}, {"question": "Q?", "answer": "A.", "x": 1}])
        self.assertEqual(len(problems), 3)

    def test_a_valid_file_parses(self) -> None:
        seeds = parse_seed_file(json.dumps(SEEDS))
        self.assertEqual([s.question for s in seeds], [s["question"] for s in SEEDS])
        self.assertEqual(seeds[0].category, "Assignments")
        self.assertIsNone(seeds[1].category)


class RunTests(ImporterTestCase):
    def test_an_invalid_file_writes_nothing(self) -> None:
        self.assertEqual(self.run_import([{"question": "Q?"}]), 1)
        self.assertEqual(self.stored(), [])
        self.assertIn("the import was not run", self.output())

    def test_a_course_that_does_not_exist_writes_nothing(self) -> None:
        code = run_import("css-999-fall-2026-none", self.write(SEEDS), out=self.lines.append, confirm=lambda plan: True)
        self.assertEqual(code, 1)
        self.assertIn('Course "css-999-fall-2026-none" does not exist.', self.output())

    def test_a_malformed_course_id_writes_nothing(self) -> None:
        self.assertEqual(run_import("Bad Id", self.write(SEEDS), out=self.lines.append), 1)

    def test_a_dry_run_reports_and_writes_nothing(self) -> None:
        self.assertEqual(self.run_import(dry_run=True, confirm=lambda plan: self.fail("asked")), 0)
        self.assertEqual(self.stored(), [])
        out = self.output()
        for label, value in (
            ("Course:", COURSE),
            ("Seeds in file:", "3"),
            ("Existing course seeds:", "0"),
            ("New seeds to insert:", "3"),
            ("Duplicates skipped:", "0"),
            ("Review status to assign:", "generated"),
        ):
            self.assertRegex(out, rf"{label}\s+{value}")
        self.assertIn("Dry run: nothing was written.", out)

    def test_the_default_import_is_staff_authored_and_pending_review(self) -> None:
        self.assertEqual(self.run_import(), 0)
        stored = self.stored()
        self.assertEqual(len(stored), 3)
        for seed in stored:
            self.assertEqual(seed["origin"], "prototype")
            self.assertEqual(seed["reviewStatus"], "generated")
            self.assertNotIn("reviewedAt", seed)
            self.assertIn("Staff-authored", seed["notes"])
            self.assertTrue(seed["id"].startswith("manual-"))
        self.assertRegex(self.output(), r"inserted:\s+3")
        self.assertRegex(self.output(), r"approved course seeds:\s+0")

    def test_approved_imports_are_approved_with_a_review_record(self) -> None:
        self.assertEqual(self.run_import(approved=True), 0)
        for seed in self.stored():
            self.assertEqual(seed["reviewStatus"], "approved")
            self.assertEqual(seed["status"], "approved")
            self.assertTrue(seed["reviewedAt"])
            self.assertIn("authored", seed["reviewNotes"])
        by_question = {seed["question"]: seed for seed in self.stored()}
        first = by_question["When are assignments due?"]
        self.assertEqual(first["category"], "Assignments")
        self.assertEqual(first["sourceSection"], "Assignments")
        self.assertEqual(by_question["Where does the class meet?"]["category"], "general")
        out = self.output()
        self.assertRegex(out, r"Review status to assign:\s+approved")
        self.assertRegex(out, r"total course seeds:\s+3")
        self.assertRegex(out, r"approved course seeds:\s+3")

    def test_running_the_same_file_twice_inserts_nothing_the_second_time(self) -> None:
        self.assertEqual(self.run_import(approved=True), 0)
        self.lines.clear()
        self.assertEqual(self.run_import(approved=True), 0)
        self.assertEqual(len(self.stored()), 3)
        out = self.output()
        self.assertRegex(out, r"New seeds to insert:\s+0")
        self.assertRegex(out, r"Duplicates skipped:\s+3")
        self.assertIn("Nothing to insert", out)

    def test_a_question_already_stored_by_anyone_is_skipped(self) -> None:
        # A student contribution with the same question, differently punctuated.
        self.store.seeds[COURSE]["student-1"] = {
            "id": "student-1",
            "question": "when are assignments due",
            "instruction": "when are assignments due",
            "origin": "user",
        }
        self.assertEqual(self.run_import(), 0)
        questions = sorted(seed["question"] for seed in self.stored())
        self.assertEqual(
            questions, ["Is there a final exam?", "Where does the class meet?", "when are assignments due"]
        )
        self.assertRegex(self.output(), r"Duplicates skipped:\s+1")
        self.assertIn("skip (already stored): When are assignments due?", self.output())

    def test_a_seed_id_already_taken_is_never_written_twice(self) -> None:
        # The question was edited in review after an earlier import; its stable id remains.
        from app.seed_dedupe import normalize_question_for_dedupe

        seed_id = manual_seed_id(COURSE, normalize_question_for_dedupe("Is there a final exam?"))
        self.store.seeds[COURSE][seed_id] = {"id": seed_id, "question": "Is there a final exam at all?"}
        self.assertEqual(self.run_import(), 0)
        self.assertRegex(self.output(), r"New seeds to insert:\s+2")

    def test_declining_confirmation_writes_nothing(self) -> None:
        self.assertEqual(self.run_import(confirm=lambda plan: False), 1)
        self.assertEqual(self.stored(), [])
        self.assertIn("Cancelled", self.output())

    def test_one_failed_insert_does_not_stop_the_rest(self) -> None:
        self.store.fail_on = {"Where does the class meet?"}
        self.assertEqual(self.run_import(), 2)
        self.assertEqual(len(self.stored()), 2)
        out = self.output()
        self.assertRegex(out, r"inserted:\s+2")
        self.assertRegex(out, r"failed:\s+1")
        self.assertIn("failed: Where does the class meet? — simulated insert failure", out)

    def test_no_other_course_is_touched(self) -> None:
        self.store.seeds[OTHER]["x"] = {"id": "x", "question": "Something else?"}
        self.assertEqual(self.run_import(approved=True), 0)
        self.assertEqual(list(self.store.seeds[OTHER]), ["x"])
        self.assertEqual(len(self.stored()), 3)

    def test_seed_ids_are_stable_per_course_and_question(self) -> None:
        self.assertEqual(manual_seed_id(COURSE, "a b"), manual_seed_id(COURSE, "a b"))
        self.assertNotEqual(manual_seed_id(COURSE, "a b"), manual_seed_id(OTHER, "a b"))


class CommandLineTests(ImporterTestCase):
    def load_cli(self) -> Any:
        path = BACKEND.parent / "scripts" / "import_manual_seeds.py"
        spec = importlib.util.spec_from_file_location("import_manual_seeds", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_the_command_line_runs_a_dry_run(self) -> None:
        cli = self.load_cli()
        path = self.write(SEEDS)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(["--course", COURSE, "--file", str(path), "--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(self.stored(), [])
        self.assertIn("Dry run: nothing was written.", buffer.getvalue())

    def test_yes_skips_the_prompt_and_approved_is_passed_through(self) -> None:
        cli = self.load_cli()
        path = self.write(SEEDS)
        with patch("builtins.input", side_effect=AssertionError("prompted")), redirect_stdout(io.StringIO()):
            code = cli.main(["--course", COURSE, "--file", str(path), "--approved", "--yes"])
        self.assertEqual(code, 0)
        self.assertTrue(all(seed["reviewStatus"] == "approved" for seed in self.stored()))

    def test_without_yes_the_prompt_decides(self) -> None:
        cli = self.load_cli()
        path = self.write(SEEDS)
        with patch("builtins.input", return_value="n"), redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["--course", COURSE, "--file", str(path)]), 1)
        self.assertEqual(self.stored(), [])


class NothingGeneratesTests(unittest.TestCase):
    def test_the_importer_loads_no_generation_training_or_syllabus_code(self) -> None:
        probe = (
            "import sys, runpy\n"
            "import app.manual_seed_import\n"
            "loaded = sorted(m for m in sys.modules if m in {\n"
            "  'app.main', 'app.ollama', 'app.ollama_coordination', 'app.starter_jobs',\n"
            "  'app.seed_generation', 'app.training_launch', 'app.storage', 'app.course_index',\n"
            "  'app.rag', 'app.finetuned_client', 'app.student_visibility', 'httpx'})\n"
            "print(loaded)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=BACKEND,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(BACKEND), "APP_ENV": "test", "PATH": "/usr/bin:/bin"},
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "[]", result.stderr)


if __name__ == "__main__":
    unittest.main()
