"""Tests for scripts/answer_repeatability_check.py.

A fake transport plays the backend; nothing touches the network or a model.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


repeat = _load("scripts/answer_repeatability_check.py", "answer_repeatability_check")
checks = repeat.checks

SEEDS = [
    {"id": "s3", "origin": "prototype", "question": "When is the  final exam?", "reviewStatus": "generated"},
    {"id": "s1", "origin": "ai_generated", "instruction": "What is the late policy?", "status": "approved"},
    {"id": "s2", "origin": "prototype", "question": "when is the final exam?"},
    {"id": "s4", "origin": "prototype", "question": "Rejected one?", "reviewStatus": "rejected"},
    {"id": "s5", "origin": "user", "question": "A student's question?", "reviewStatus": "approved"},
    {"id": "s6", "origin": "prototype", "question": "Contributed by a participant?", "participantId": "p-1"},
    {"id": "s7", "origin": "authored", "question": "What is the capital of France?"},
    {"id": "s8", "origin": "prototype", "question": "   "},
]


def _result(status: int, body: Any = None, *, seconds: float = 1.0, headers: dict[str, str] | None = None):
    return checks.HttpResult(status, body, seconds, headers=headers or {})


class FakeBackend:
    """Answers the routes the script uses and records every generate request."""

    def __init__(
        self,
        *,
        answer: Any = None,
        mode: str = "vm",
        queue: list[dict[str, Any] | None] | None = None,
        seeds: list[dict[str, Any]] | None = None,
        seconds: Any = None,
    ) -> None:
        self.answer = answer or (lambda mode, question, n: _result(200, {"answer": f"Answer to {question} by {mode}."}))
        self.mode = mode
        self.queue = list(queue or [])
        self.seeds = SEEDS if seeds is None else seeds
        self.seconds = seconds
        self.generated: list[tuple[str, str]] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.logged_out = False
        self.paths: list[str] = []

    def request(self, method: str, url: str, *, body=None, headers=None, timeout: float = 30.0):
        path = url.split("//", 1)[1].split("/", 1)[1]
        self.paths.append(f"{method} /{path}")
        if url.endswith("/api/auth/login"):
            return _result(204, None, headers={"set-cookie": "sml_staff=TOKEN; HttpOnly; Path=/api"})
        if url.endswith("/api/auth/logout"):
            self.logged_out = True
            return _result(204)
        if url.endswith("/api/admin/generation-mode"):
            return _result(200, {"mode": self.mode})
        if url.endswith("/api/admin/generation-queue"):
            status = self.queue.pop(0) if self.queue else {"active": 0, "waiting": 0}
            return _result(200, status) if status is not None else _result(500)
        if url.endswith("/seeds"):
            return _result(200, {"seeds": self.seeds})
        mode = next(m for m, route in checks.MODE_ROUTES.items() if url.endswith(route))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.generated.append((mode, body["question"]))
        try:
            result = self.answer(mode, body["question"], len(self.generated))
            if self.seconds is not None:
                result.seconds = self.seconds(mode, len(self.generated))
            return result
        finally:
            self.in_flight -= 1


def _run(argv: list[str], backend: FakeBackend, *, password: str | None = "pw") -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    env = {checks.PASSWORD_ENV: password} if password else {}
    with redirect_stdout(out), redirect_stderr(err), unittest.mock.patch.dict("os.environ", env):
        code = repeat.main(argv, transport=backend, sleep=lambda _seconds: None)
    return code, out.getvalue(), err.getvalue()


ADMIN = ["--admin-email", "admin@example.edu", "--course-id", "c-1", "--yes"]


class PlanTests(unittest.TestCase):
    def test_passes_parse_and_reject_unknown_orders(self) -> None:
        self.assertEqual(repeat.parse_passes("question, batched"), ["question", "batched"])
        for bad in ("", "random", ",".join(["question"] * (repeat.MAX_PASSES + 1))):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                repeat.parse_passes(bad)

    def test_both_orders_cover_every_question_and_condition_exactly_once(self) -> None:
        expected = sorted((q, c) for q in range(7) for c in repeat.CONDITIONS)
        for order in repeat.ORDERS:
            with self.subTest(order=order):
                self.assertEqual(sorted(repeat.build_pass(order, 7, 3)), expected)

    def test_question_order_finishes_one_question_before_the_next(self) -> None:
        steps = repeat.build_pass("question", 2, 5)
        self.assertEqual(steps[:4], [(0, c) for c in repeat.CONDITIONS])

    def test_batched_order_groups_each_approach_within_a_batch(self) -> None:
        steps = repeat.build_pass("batched", 3, 2)
        self.assertEqual(
            steps[:8],
            [(0, "rag"), (1, "rag"), (0, "fineTunedRag"), (1, "fineTunedRag"),
             (0, "base"), (1, "base"), (0, "fineTuned"), (1, "fineTuned")],
        )
        # The last, short batch is complete too.
        self.assertEqual(steps[8:], [(2, c) for c in repeat.BATCHED_CONDITION_ORDER])


class QuestionSelectionTests(unittest.TestCase):
    def test_only_staff_selectable_questions_are_eligible(self) -> None:
        eligible = repeat.eligible_questions(SEEDS)
        # Sorted by id; the duplicate wording, the rejected one, the student
        # ones, the behaviour example and the blank one are all left out.
        self.assertEqual([item["ref"] for item in eligible], ["s1", "s3"])
        self.assertEqual(eligible[1]["question"], "When is the final exam?")

    def test_pick_is_spread_and_stable(self) -> None:
        eligible = [{"ref": f"s{n:02d}", "question": f"Q{n}?"} for n in range(10)]
        picked = repeat.pick_questions(eligible, 5)
        self.assertEqual([item["ref"] for item in picked], ["s00", "s02", "s04", "s06", "s08"])
        self.assertEqual(repeat.pick_questions(eligible, 5), picked)
        self.assertEqual(repeat.pick_questions(eligible, 50), eligible)

    def test_questions_file_skips_comments_blanks_and_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.txt"
            path.write_text("# header\nWhen is the exam?\n\nwhen is  the exam?\nHow is it graded?\n", encoding="utf-8")
            questions = repeat.read_questions_file(path)
        self.assertEqual([q["question"] for q in questions], ["When is the exam?", "How is it graded?"])


class SampleTests(unittest.TestCase):
    def test_hash_ignores_whitespace_runs_but_not_wording(self) -> None:
        self.assertEqual(repeat.answer_hash("Due  Friday.\n"), repeat.answer_hash("Due Friday."))
        self.assertNotEqual(repeat.answer_hash("Due Friday."), repeat.answer_hash("Due Monday."))

    def test_outcomes(self) -> None:
        busy = {"detail": {"code": "generation_busy", "message": "Busy", "retryAfterSeconds": 30}}
        cases = [
            (_result(200, {"answer": "Yes."}), "ok"),
            (_result(200, {"answer": "  "}), "empty"),
            (_result(503, busy), "busy"),
            (_result(503, {"detail": "unavailable"}), "http_503"),
            (checks.HttpResult(0, None, 1.0, error="timeout", timed_out=True), "timeout"),
            (checks.HttpResult(0, None, 1.0, error="refused"), "network"),
        ]
        for result, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(repeat.classify(result), expected)

    def test_a_sample_keeps_no_answer_or_question_text(self) -> None:
        result = _result(200, {"answer": "The exam is on Friday.", "model": "llama3.2:3b", "engine": "vm"})
        sample = repeat.sample_of(
            result, pass_index=0, order="question", question={"ref": "s1", "question": "When is the exam?"}, condition="base"
        )
        text = json.dumps(sample)
        self.assertNotIn("Friday", text)
        self.assertNotIn("When is the exam", text)
        self.assertEqual(sample["question"], "s1")
        self.assertEqual(sample["answerChars"], len("The exam is on Friday."))
        self.assertEqual(sample["engine"], "vm")


class IdleGuardTests(unittest.TestCase):
    def test_waits_while_others_generate_then_returns(self) -> None:
        statuses = [{"active": 1, "waiting": 2}, {"active": 0, "waiting": 1}, {"active": 0, "waiting": 0}]
        slept: list[float] = []
        repeat.wait_until_idle(lambda: statuses.pop(0), max_wait=60, sleep=slept.append, clock=lambda: 0.0, log=lambda _m: None)
        self.assertEqual(len(slept), 2)
        self.assertEqual(statuses, [])

    def test_gives_up_when_the_queue_stays_busy(self) -> None:
        now = [0.0]

        def sleep(seconds: float) -> None:
            now[0] += seconds

        with self.assertRaises(repeat.Abort):
            repeat.wait_until_idle(
                lambda: {"active": 1, "waiting": 0}, max_wait=12, sleep=sleep, clock=lambda: now[0], log=lambda _m: None
            )

    def test_an_unreadable_queue_stops_the_run(self) -> None:
        with self.assertRaises(repeat.Abort):
            repeat.wait_until_idle(lambda: None, max_wait=60, sleep=lambda _s: None, log=lambda _m: None)


class SummaryTests(unittest.TestCase):
    @staticmethod
    def _sample(pass_index: int, order: str, ref: str, condition: str, digest: str | None, seconds: float = 1.0):
        return {
            "pass": pass_index, "order": order, "question": ref, "condition": condition,
            "outcome": "ok" if digest else "busy", "seconds": seconds, "answerSha256": digest,
        }

    def test_repeatability_counts_questions_whose_answer_never_changed(self) -> None:
        s = self._sample
        samples = [
            s(0, "question", "q1", "base", "a"), s(1, "batched", "q1", "base", "a"),
            s(0, "question", "q2", "base", "a"), s(1, "batched", "q2", "base", "b"),
            # Answered once only: nothing to compare.
            s(0, "question", "q3", "base", "a"), s(1, "batched", "q3", "base", None),
        ]
        base = repeat.summarize_repeatability(samples)["base"]
        self.assertEqual((base["questionsCompared"], base["identical"], base["rate"]), (2, 1, 0.5))
        self.assertEqual(base["varyingQuestions"], ["q2"])
        self.assertEqual(base["mostDistinctAnswers"], 2)
        self.assertIsNone(repeat.summarize_repeatability(samples)["rag"]["rate"])

    def test_orders_are_timed_on_the_same_answers_only(self) -> None:
        s = self._sample
        samples = [
            s(0, "question", "q1", "rag", "a", 10.0), s(1, "batched", "q1", "rag", "a", 8.0),
            # Failed in the batched pass, so its 30 s is not counted for either order.
            s(0, "question", "q2", "rag", "a", 30.0), s(1, "batched", "q2", "rag", None, 1.0),
        ]
        timing = repeat.summarize_orders(samples, ["question", "batched"])
        self.assertEqual(timing["comparison"]["pairsCompared"], 1)
        self.assertEqual(timing["orders"]["question"]["meanSeconds"], 10.0)
        self.assertEqual(timing["orders"]["batched"]["meanSeconds"], 8.0)
        self.assertEqual(timing["comparison"]["batchedSavingPercent"], 20.0)

    def test_a_single_order_has_no_comparison(self) -> None:
        samples = [self._sample(0, "question", "q1", "rag", "a", 10.0)]
        timing = repeat.summarize_orders(samples, ["question"])
        self.assertNotIn("batched", timing["orders"])
        self.assertNotIn("batchedSavingPercent", timing["comparison"])


class RunTests(unittest.TestCase):
    def test_full_run_is_sequential_read_only_and_keeps_no_text(self) -> None:
        backend = FakeBackend(seconds=lambda mode, n: 2.0)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "report.json"
            code, out, _err = _run(ADMIN + ["--questions", "2", "--passes", "question,batched", "--out", str(out_path)], backend)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            written = out_path.read_text(encoding="utf-8")
        self.assertEqual(code, 0)
        # 4 warm-up requests, then 2 questions x 4 approaches x 2 passes.
        self.assertEqual(len(backend.generated), 4 + 16)
        self.assertEqual(backend.max_in_flight, 1)
        self.assertEqual({q for _m, q in backend.generated[:4]}, {checks.PROBE_QUESTION})
        self.assertEqual(report["questions"], ["s1", "s3"])
        self.assertEqual(report["questionsAvailable"], 2)
        self.assertIsNone(report["aborted"])
        for condition in repeat.CONDITIONS:
            self.assertEqual(report["repeatability"][condition]["rate"], 1.0)
        self.assertEqual(report["orderTiming"]["comparison"]["pairsCompared"], 8)
        for text in (out, written):
            self.assertNotIn("Answer to", text)
            self.assertNotIn("final exam", text)
        self.assertIn("| base | 2 | 2 | 100% |", out)
        # Only reads, the four generate routes, and the session's own login/logout.
        allowed = {"POST /api/auth/login", "POST /api/auth/logout", "GET /api/admin/generation-mode",
                   "GET /api/admin/generation-queue", "GET /api/db/courses/c-1/seeds"}
        allowed |= {f"POST {route}" for route in checks.MODE_ROUTES.values()}
        self.assertLessEqual(set(backend.paths), allowed)
        self.assertTrue(backend.logged_out)

    def test_a_changing_answer_is_reported_not_hidden(self) -> None:
        def answer(mode: str, question: str, n: int):
            text = f"Answer {n}" if mode == "fineTuned" else "Same answer"
            return _result(200, {"answer": text})

        backend = FakeBackend(answer=answer)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "report.json"
            code, out, _err = _run(ADMIN + ["--questions", "2", "--passes", "question,question", "--no-warmup", "--out", str(out_path)], backend)
            report = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(report["repeatability"]["fineTuned"]["rate"], 0.0)
        self.assertEqual(report["repeatability"]["base"]["rate"], 1.0)
        self.assertIn("| fineTuned | 2 | 0 | 0% |", out)

    def test_refuses_gpu_mode_before_generating_anything(self) -> None:
        backend = FakeBackend(mode="gpu")
        code, _out, err = _run(ADMIN, backend)
        self.assertEqual(code, 1)
        self.assertIn("GPU mode", err)
        self.assertEqual(backend.generated, [])
        self.assertTrue(backend.logged_out)

    def test_stops_when_an_answer_comes_from_the_gpu_mid_run(self) -> None:
        def answer(mode: str, question: str, n: int):
            return _result(200, {"answer": "x", "engine": "gpu" if n == 3 else "vm"})

        backend = FakeBackend(answer=answer)
        code, _out, err = _run(ADMIN + ["--no-warmup"], backend)
        self.assertEqual(code, 1)
        self.assertIn("Tillicum GPU", err)
        self.assertEqual(len(backend.generated), 3)

    def test_a_base_answer_from_the_gpu_is_recognised_by_its_model_name(self) -> None:
        def answer(mode: str, question: str, n: int):
            return _result(200, {"answer": "x", "model": "llama3.2:3b (Tillicum GPU)" if n == 2 else "llama3.2:3b"})

        backend = FakeBackend(answer=answer)
        code, _out, err = _run(ADMIN + ["--no-warmup"], backend)
        self.assertEqual(code, 1)
        self.assertIn("Tillicum GPU", err)
        self.assertEqual(len(backend.generated), 2)

    def test_stops_after_three_failures_in_a_row_and_keeps_partial_samples(self) -> None:
        busy = {"detail": {"code": "generation_busy", "message": "Busy", "retryAfterSeconds": 30}}

        def answer(mode: str, question: str, n: int):
            return _result(200, {"answer": "x"}) if n <= 2 else _result(503, busy)

        backend = FakeBackend(answer=answer)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "report.json"
            code, out, err = _run(ADMIN + ["--no-warmup", "--out", str(out_path)], backend)
            report = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(code, 1)
        self.assertIn("in a row failed", err)
        self.assertEqual(len(backend.generated), 2 + repeat.MAX_CONSECUTIVE_FAILURES)
        self.assertEqual(len(report["samples"]), 5)
        self.assertIn("in a row failed", report["aborted"])
        self.assertNotIn("Same answer every time?", out)

    def test_a_failed_warm_up_measures_nothing(self) -> None:
        backend = FakeBackend(answer=lambda mode, question, n: _result(503, {"detail": "unavailable"}))
        code, _out, err = _run(ADMIN, backend)
        self.assertEqual(code, 1)
        self.assertIn("warm-up", err)
        self.assertEqual(len(backend.generated), 1)

    def test_waits_for_other_users_between_requests(self) -> None:
        backend = FakeBackend(queue=[{"active": 1, "waiting": 3}, {"active": 0, "waiting": 0}])
        code, out, _err = _run(ADMIN + ["--questions", "1", "--passes", "question", "--no-warmup"], backend)
        self.assertEqual(code, 0)
        self.assertIn("waiting for the queue to be idle", out)
        self.assertEqual(len(backend.generated), 4)

    def test_an_unreadable_queue_stops_before_the_first_request(self) -> None:
        backend = FakeBackend(queue=[None])
        code, _out, err = _run(ADMIN + ["--no-warmup"], backend)
        self.assertEqual(code, 1)
        self.assertIn("could not be read", err)
        self.assertEqual(backend.generated, [])

    def test_no_eligible_questions_is_an_error(self) -> None:
        backend = FakeBackend(seeds=[SEEDS[4], SEEDS[5]])
        code, _out, err = _run(ADMIN, backend)
        self.assertEqual(code, 1)
        self.assertIn("No questions", err)
        self.assertEqual(backend.generated, [])

    def test_usage_errors_return_2_without_any_request(self) -> None:
        for argv in (
            ["--course-id", "c-1", "--no-auth", "--yes"],
            ADMIN + ["--passes", "sideways"],
            ADMIN + ["--questions", "0"],
            ADMIN + ["--questions", str(repeat.MAX_QUESTIONS + 1)],
            ADMIN + ["--batch-size", "0"],
        ):
            with self.subTest(argv=argv):
                backend = FakeBackend()
                code, _out, _err = _run(argv, backend)
                self.assertEqual(code, 2)
                self.assertEqual(backend.paths, [])

    def test_refuses_to_run_unattended_without_yes(self) -> None:
        backend = FakeBackend()
        stdin = sys.stdin
        sys.stdin = io.StringIO("")
        try:
            code, _out, _err = _run(["--admin-email", "admin@example.edu", "--course-id", "c-1"], backend)
        finally:
            sys.stdin = stdin
        self.assertEqual(code, 2)
        self.assertEqual(backend.generated, [])
        self.assertTrue(backend.logged_out)

    def test_no_auth_run_reads_a_questions_file_and_skips_admin_routes(self) -> None:
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.txt"
            path.write_text("When is the exam?\n", encoding="utf-8")
            code, _out, _err = _run(
                ["--no-auth", "--course-id", "c-1", "--questions-file", str(path), "--passes", "question", "--no-warmup", "--yes"],
                backend,
                password=None,
            )
        self.assertEqual(code, 0)
        self.assertEqual(len(backend.generated), 4)
        self.assertFalse(any("/api/admin/" in path or "/api/auth/" in path for path in backend.paths))


if __name__ == "__main__":
    unittest.main()
