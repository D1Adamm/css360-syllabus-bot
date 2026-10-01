"""Tests for scripts/classroom_load_test.py and scripts/warm_classroom_models.py.

A fake transport plays the backend; nothing touches the network or a model.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import threading
import unittest
from contextlib import redirect_stdout
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


load_test = _load("scripts/classroom_load_test.py", "classroom_load_test")
warm = _load("scripts/warm_classroom_models.py", "warm_classroom_models")
checks = sys.modules["finetuned_production_checks"]


def _result(status: int, body: Any = None, *, seconds: float = 1.0, timed_out: bool = False):
    return checks.HttpResult(status, body, seconds, error="timeout" if timed_out else None, timed_out=timed_out)


class FakeBackend:
    """Answers the four generate routes; records the order requests started in."""

    def __init__(self, outcomes: dict[str, Any] | None = None) -> None:
        self.outcomes = outcomes or {}
        self.started: list[str] = []
        self.comparison_ids: list[str | None] = []
        self.lock = threading.Lock()

    def request(self, method: str, url: str, *, body=None, headers=None, timeout: float = 30.0):
        if url.endswith("/api/ps"):
            return _result(200, {"models": [{"name": "llama3.2:3b"}]})
        if url.endswith("/api/admin/generation-queue"):
            return _result(200, {"active": 1, "waiting": 3})
        mode = next(m for m, path in checks.MODE_ROUTES.items() if url.endswith(path))
        with self.lock:
            self.started.append(mode)
            self.comparison_ids.append((headers or {}).get("X-Comparison-Id"))
        outcome = self.outcomes.get(mode)
        if outcome is not None:
            return outcome
        return _result(200, {"answer": "An answer.", "courseId": body["courseId"]})


class PlanTests(unittest.TestCase):
    def test_levels_parse_dedupe_and_cap(self) -> None:
        self.assertEqual(load_test.parse_levels("1, 5,5,10"), [1, 5, 10])
        for bad in ("", "0", "x", str(load_test.MAX_STUDENTS + 1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                load_test.parse_levels(bad)


class StudentScheduleTests(unittest.TestCase):
    def test_one_student_makes_the_compare_pages_four_requests_with_rag_after_base(self) -> None:
        backend = FakeBackend()
        session = checks.BackendSession("http://127.0.0.1:8001", backend)
        samples = load_test.run_student(
            lambda mode, q, cid: checks.generate(session, mode, "c-1", question=q, comparison_id=cid),
            student=0,
            question="When is the exam?",
            clock=lambda: 0.0,
        )
        self.assertEqual(sorted(s["condition"] for s in samples), sorted(load_test.CONDITIONS))
        self.assertLess(backend.started.index("base"), backend.started.index("rag"))
        self.assertTrue(all(s["ok"] for s in samples))
        # Like the Compare page: one X-Comparison-Id on all four requests.
        self.assertEqual(len(backend.comparison_ids), 4)
        self.assertEqual(len(set(backend.comparison_ids)), 1)
        self.assertIsNotNone(backend.comparison_ids[0])

    def test_outcomes_are_classified(self) -> None:
        busy = {"detail": {"code": "generation_busy", "message": "busy"}}
        self.assertEqual(load_test.classify(_result(200, {"answer": "x"})), "ok")
        self.assertEqual(load_test.classify(_result(200, {"answer": " "})), "empty")
        self.assertEqual(load_test.classify(_result(503, busy)), "busy")
        self.assertEqual(load_test.classify(_result(503, {"detail": "down"})), "http_503")
        self.assertEqual(load_test.classify(_result(0, None, timed_out=True)), "timeout")
        self.assertEqual(load_test.classify(_result(0, None)), "network")


class QuestionTests(unittest.TestCase):
    def test_default_questions_repeat_and_unique_ones_never_do(self) -> None:
        shared = [load_test.question_for(i, level=30, unique=False) for i in range(30)]
        self.assertLess(len(set(shared)), 30)
        unique = [load_test.question_for(i, level=n, unique=True) for n in (10, 30) for i in range(n)]
        self.assertEqual(len(set(unique)), len(unique))
        self.assertTrue(all(q.endswith("?") for q in unique))


class SummaryTests(unittest.TestCase):
    def test_level_summary_counts_complete_comparisons_and_outcomes(self) -> None:
        def row(student, condition, outcome, start, end):
            return {
                "student": student, "condition": condition, "outcome": outcome,
                "ok": outcome == "ok", "timedOut": outcome == "timeout",
                "seconds": end - start, "start": start, "end": end,
            }

        samples = [row(0, c, "ok", 0.0, 10.0) for c in load_test.CONDITIONS]
        samples += [row(1, c, "ok", 0.0, 20.0) for c in ("base", "fineTuned", "fineTunedRag")]
        samples.append(row(1, "rag", "busy", 20.0, 20.5))
        summary = load_test.summarize_level(samples, students=2)
        self.assertEqual(summary["requests"], 8)
        self.assertEqual(summary["okRequests"], 7)
        self.assertEqual(summary["completeComparisons"], 1)
        self.assertEqual(summary["outcomes"], {"ok": 7, "busy": 1})
        self.assertEqual(summary["wallSeconds"], 20.5)
        self.assertEqual(summary["perCondition"]["rag"]["outcomes"], {"ok": 1, "busy": 1})

    def test_model_loads_count_new_arrivals_not_the_starting_state(self) -> None:
        timeline = [["base"], ["base"], None, ["ft"], ["base"], ["base", "embed"]]
        self.assertEqual(load_test.count_loads(timeline), 3)
        self.assertEqual(load_test.count_loads([None, ["a"], ["a"]]), 0)


class MainTests(unittest.TestCase):
    def test_runs_levels_and_prints_a_table_without_answer_text(self) -> None:
        backend = FakeBackend({"rag": _result(503, {"detail": {"code": "generation_busy"}})})
        out = io.StringIO()
        with redirect_stdout(out):
            code = load_test.main(
                ["--no-auth", "--course-id", "c-1", "--levels", "1,2", "--cooldown", "0",
                 "--no-system", "--ollama-url", "http://127.0.0.1:11434", "--yes"],
                transport=backend,
            )
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("| 1 | 3/4 | 0 |", text)
        self.assertIn("| 2 | 6/8 | 0 |", text)
        self.assertNotIn("An answer.", text)
        # Three students in total, one id each, shared by that student's four requests.
        self.assertEqual(len(set(backend.comparison_ids)), 3)

    def test_refuses_to_run_unattended_without_yes(self) -> None:
        with redirect_stdout(io.StringIO()):
            stdin = sys.stdin
            sys.stdin = io.StringIO("")
            try:
                code = load_test.main(["--no-auth", "--course-id", "c-1", "--levels", "1"], transport=FakeBackend())
            finally:
                sys.stdin = stdin
        self.assertEqual(code, 2)


class HarnessGuardTests(unittest.TestCase):
    def test_refuses_when_a_database_is_configured_anywhere(self) -> None:
        import tempfile

        harness = _load("scripts/classroom_load_harness.py", "classroom_load_harness")
        self.assertIsNotNone(harness.deployment_refusal({"DATABASE_URL": "postgresql://x"}, None))
        self.assertIsNone(harness.deployment_refusal({"DATABASE_URL": "  "}, None))
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            for text, refused in (
                ('DATABASE_URL="postgresql://prod"\n', True),
                ("export DATABASE_URL=postgresql://prod\n", True),
                ("DATABASE_URL=\nOLLAMA_MODEL=llama3.2:3b\n", False),
                ("# DATABASE_URL=postgresql://commented\n", False),
            ):
                with self.subTest(text=text):
                    env.write_text(text, encoding="utf-8")
                    self.assertEqual(harness.deployment_refusal({}, env) is not None, refused)

    def test_main_returns_2_before_importing_the_app(self) -> None:
        harness = _load("scripts/classroom_load_harness.py", "classroom_load_harness")
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"DATABASE_URL": "postgresql://prod"}), redirect_stdout(io.StringIO()):
            import contextlib

            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(harness.main(["--course-id", "c-1"]), 2)


class WarmTests(unittest.TestCase):
    def test_warm_uses_the_classroom_context_size(self) -> None:
        # Read from backend/app/grounded_generation.py, which the backend suite
        # already pins equal to the fine-tuned service's DEFAULT_NUM_CTX.
        self.assertEqual(warm.classroom_num_ctx(), 4096)

    def test_warm_request_loads_without_generating(self) -> None:
        sent: list[dict] = []

        def fake_post(url, body, *, timeout):
            sent.append({"url": url, "body": body})
            return 200, {"done": True}, 0.5

        original = warm.post
        warm.post = fake_post
        try:
            line = warm.warm_chat_model("http://127.0.0.1:11435", "css360e-v1:latest", num_ctx=4096, keep_alive="4h", timeout=5)
        finally:
            warm.post = original
        self.assertTrue(line.startswith("ok"))
        self.assertEqual(sent[0]["url"], "http://127.0.0.1:11435/api/chat")
        self.assertEqual(sent[0]["body"], {
            "model": "css360e-v1:latest", "messages": [], "keep_alive": "4h", "options": {"num_ctx": 4096},
        })


class WarmVerifyTests(unittest.TestCase):
    def test_unreachable_server_is_a_fail_line_not_a_traceback(self) -> None:
        line = warm.warm_chat_model("http://127.0.0.1:9", "m:latest", num_ctx=4096, keep_alive="30m", timeout=2)
        self.assertTrue(line.startswith("FAIL"), line)

    def test_residency_check_wants_the_model_on_that_server_with_the_classroom_context(self) -> None:
        original = warm.resident
        try:
            warm.resident = lambda url: [{"name": "llama3.2:3b", "context_length": 4096, "expires_at": "t"}]
            self.assertTrue(warm.check_resident("u", "llama3.2:3b", num_ctx=4096).startswith("PASS"))
            self.assertTrue(warm.check_resident("u", "css360e-v1", num_ctx=4096).startswith("FAIL"))
            warm.resident = lambda url: [{"name": "css360e-v1:latest", "context_length": 2048}]
            self.assertIn("context 2048", warm.check_resident("u", "css360e-v1:latest", num_ctx=4096))
            warm.resident = lambda url: None
            self.assertTrue(warm.check_resident("u", "x", num_ctx=None).startswith("FAIL"))
        finally:
            warm.resident = original


if __name__ == "__main__":
    unittest.main()
