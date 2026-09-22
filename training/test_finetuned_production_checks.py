"""Tests for the production verification script and the latency probe.

Every HTTP call goes to a fake transport that plays the backend, the local
service and Ollama; host facts (ss, systemctl, the env file) are canned. No
network, no systemd, no credentials beyond a made-up pair that never leaves
the test process.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: dataclasses under `from __future__ import
    # annotations` resolve their types through sys.modules[cls.__module__].
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


checks = _load("scripts/lib/finetuned_production_checks.py", "finetuned_production_checks")
verify = _load("scripts/verify_finetuned_production.py", "verify_finetuned_production")
probe = _load("scripts/finetuned_latency_probe.py", "finetuned_latency_probe")

CSS350 = "css-350-spring-2026-n3h9"
CSS360 = "css-360-winter-2026-a7rp"
MAPPING = f"FINETUNED_OLLAMA_MODELS={CSS350}@v1=css350-ft-v1:latest,{CSS360}@v2=css360-ft-v2:latest\nFINETUNED_KEEP_ALIVE=30m\n"
ANSWER = "The course meets on Mondays and Wednesdays at 10:00."


class FakeTransport:
    """The three loopback servers, answered from a table the test controls."""

    def __init__(self, *, mapping: dict[str, str] | None = None, backend_version: dict[str, str] | None = None, ollama_models: list[str] | None = None, login_ok: bool = True, fail_modes: set[str] = frozenset(), timeout_modes: set[str] = frozenset()) -> None:
        self.mapping = mapping or {CSS350: "v1", CSS360: "v2"}
        self.backend_version = backend_version or dict(self.mapping)
        self.ollama_models = ollama_models if ollama_models is not None else ["llama3.2:3b", "css350-ft-v1:latest", "css360-ft-v2:latest"]
        self.login_ok = login_ok
        self.fail_modes = set(fail_modes)
        self.timeout_modes = set(timeout_modes)
        self.calls: list[tuple[str, str, dict[str, str], Any]] = []
        self.logged_out = False

    def request(self, method: str, url: str, *, body: Any = None, headers: dict[str, str] | None = None, timeout: float = 30.0) -> checks.HttpResult:
        headers = headers or {}
        self.calls.append((method, url, headers, body))
        path = url.split("//", 1)[1].split("/", 1)[1] if "//" in url else url
        path = "/" + path
        if ":11434" in url:
            return checks.HttpResult(200, {"models": [{"name": n, "digest": "d" * 64} for n in self.ollama_models]}, 0.01)
        if ":9001" in url:
            return self._service(path, body)
        return self._backend(method, path, headers, body)

    def _service(self, path: str, body: Any) -> checks.HttpResult:
        if path == "/health":
            courses = [{"courseId": c, "versions": [v], "currentVersion": v} for c, v in sorted(self.mapping.items()) if f"{c.split('-')[0]}{c.split('-')[1]}-ft-{v}:latest" in self.ollama_models]
            return checks.HttpResult(200, {"status": "ok", "adapterLoaded": bool(courses), "courses": courses, "models": []}, 0.01)
        if path == "/generate":
            course, version = body["courseId"], body.get("modelVersion")
            if self.mapping.get(course) != version:
                return checks.HttpResult(409, {"detail": "not mapped"}, 0.01)
            return checks.HttpResult(200, {"answer": ANSWER, "model": "x", "courseId": course, "modelVersion": version, "adapterLoaded": True, "generationSeconds": 3.0}, 3.2)
        return checks.HttpResult(404, None, 0.01)

    def _backend(self, method: str, path: str, headers: dict[str, str], body: Any) -> checks.HttpResult:
        if path == "/api/health":
            return checks.HttpResult(200, {"status": "ok", "service": "syllabus-model-lab-backend"}, 0.01)
        if path == "/api/auth/login":
            if headers.get(checks.CSRF_HEADER_NAME) != checks.CSRF_HEADER_VALUE:
                return checks.HttpResult(403, {"detail": "csrf"}, 0.01)
            if not self.login_ok:
                return checks.HttpResult(401, {"detail": "no"}, 0.01)
            return checks.HttpResult(204, None, 0.02, headers={"set-cookie": "sml_staff=SECRET-TOKEN; HttpOnly; Secure; Path=/api; SameSite=Lax"})
        signed_in = "sml_staff=SECRET-TOKEN" in headers.get("Cookie", "")
        if path == "/api/auth/logout":
            self.logged_out = True
            return checks.HttpResult(204, None, 0.01)
        if not signed_in:
            return checks.HttpResult(401, {"detail": "sign in"}, 0.01)
        if path == "/api/fine-tuned/health":
            return checks.HttpResult(200, {"status": "ok", "adapterLoaded": True, "courses": [{"courseId": c} for c in self.mapping]}, 0.01)
        if method == "POST" and headers.get(checks.CSRF_HEADER_NAME) != checks.CSRF_HEADER_VALUE:
            return checks.HttpResult(403, {"detail": "csrf"}, 0.01)
        mode = {v: k for k, v in checks.MODE_ROUTES.items()}.get(path)
        if mode is None:
            return checks.HttpResult(404, None, 0.01)
        if mode in self.timeout_modes:
            return checks.HttpResult(0, None, 180.0, error="timeout", timed_out=True)
        if mode in self.fail_modes:
            return checks.HttpResult(503, {"detail": "service down"}, 0.5)
        course = body["courseId"]
        payload: dict[str, Any] = {"answer": ANSWER, "model": "m", "courseId": course, "responseType": mode}
        if mode.startswith("fineTuned"):
            payload.update({"modelVersion": self.backend_version.get(course), "adapterLoaded": True, "generationSeconds": 4.0})
        return checks.HttpResult(200, payload, 4.5 if mode.startswith("fineTuned") else 2.0)


class FakeFacts:
    def __init__(self, *, ss: str | None = 'LISTEN 0 2048 127.0.0.1:9001 0.0.0.0:* users:(("python",pid=777,fd=6))\n', enabled: str = "enabled", active: str = "active", env: str | None = MAPPING) -> None:
        self.ss, self.enabled, self.active, self.env = ss, enabled, active, env

    def listeners_text(self) -> str | None:
        return self.ss

    def unit_property(self, unit: str, verb: str) -> str | None:
        return self.enabled if verb == "is-enabled" else self.active

    def env_text(self, path: Path) -> str | None:
        return self.env


def _verify(transport: FakeTransport, facts: FakeFacts, *argv: str, credentials: tuple[str, str] | None = ("admin@uw.edu", "pw")) -> tuple[list[checks.Check], str]:
    args = verify.build_parser().parse_args(list(argv))
    results = verify.run(args, facts=facts, transport=transport, credentials=credentials)
    out = io.StringIO()
    with redirect_stdout(out):
        checks.print_checks(results)
    return results, out.getvalue()


def _by_status(results: list[checks.Check], status: str) -> list[str]:
    return [name for s, name, _ in results if s == status]


class SessionTests(unittest.TestCase):
    def test_login_keeps_the_secure_cookie_and_sends_it_with_the_csrf_header(self) -> None:
        transport = FakeTransport()
        session = checks.BackendSession("http://127.0.0.1:8001", transport)
        self.assertTrue(session.login("admin@uw.edu", "pw").ok)
        self.assertTrue(session.signed_in)
        result = checks.generate(session, "fineTuned", CSS350)
        self.assertTrue(result.ok)
        _, _, headers, _ = transport.calls[-1]
        self.assertEqual(headers["Cookie"], "sml_staff=SECRET-TOKEN")
        self.assertEqual(headers[checks.CSRF_HEADER_NAME], checks.CSRF_HEADER_VALUE)
        session.logout()
        self.assertTrue(transport.logged_out)
        self.assertFalse(session.signed_in)

    def test_a_failed_login_leaves_the_session_anonymous(self) -> None:
        session = checks.BackendSession("http://127.0.0.1:8001", FakeTransport(login_ok=False))
        self.assertFalse(session.login("admin@uw.edu", "bad").ok)
        self.assertFalse(session.signed_in)

    def test_credentials_come_from_the_environment_never_from_argv(self) -> None:
        self.assertIsNone(checks.resolve_admin_credentials(None, env={}, prompt=False))
        self.assertIsNone(checks.resolve_admin_credentials("a@uw.edu", env={}, prompt=False))
        self.assertEqual(checks.resolve_admin_credentials("a@uw.edu", env={checks.PASSWORD_ENV: "s"}, prompt=False), ("a@uw.edu", "s"))
        self.assertEqual(checks.resolve_admin_credentials(None, env={checks.EMAIL_ENV: "b@uw.edu", checks.PASSWORD_ENV: "s"}, prompt=False), ("b@uw.edu", "s"))
        for action in verify.build_parser()._actions + probe.build_parser()._actions:
            self.assertNotIn("password", (action.dest or "").lower())


class JudgeTests(unittest.TestCase):
    def _result(self, **body: Any) -> checks.HttpResult:
        return checks.HttpResult(200, body, 2.0)

    def test_a_good_fine_tuned_answer_passes_and_reports_only_length_and_version(self) -> None:
        ok, detail = checks.judge_generation(self._result(answer=ANSWER, courseId=CSS350, modelVersion="v1", adapterLoaded=True), course_id=CSS350, mode="fineTuned", allowed_versions=["v1"])
        self.assertTrue(ok)
        self.assertIn("v1", detail)
        self.assertNotIn("Mondays", detail)

    def test_wrong_course_wrong_version_or_no_adapter_fail(self) -> None:
        base = dict(answer=ANSWER, courseId=CSS350, modelVersion="v1", adapterLoaded=True)
        self.assertFalse(checks.judge_generation(self._result(**{**base, "courseId": CSS360}), course_id=CSS350, mode="fineTuned")[0])
        self.assertFalse(checks.judge_generation(self._result(**base), course_id=CSS350, mode="fineTuned", allowed_versions=["v2"])[0])
        self.assertFalse(checks.judge_generation(self._result(**{**base, "adapterLoaded": False}), course_id=CSS350, mode="fineTuned")[0])
        self.assertFalse(checks.judge_generation(self._result(**{**base, "answer": " "}), course_id=CSS350, mode="fineTuned")[0])
        self.assertTrue(checks.judge_generation(self._result(answer="x", courseId=CSS350), course_id=CSS350, mode="rag")[0])

    def test_errors_and_timeouts_fail_with_the_status(self) -> None:
        ok, detail = checks.judge_generation(checks.HttpResult(503, {"detail": "down"}, 0.1), course_id=CSS350, mode="base")
        self.assertFalse(ok)
        self.assertIn("503", detail)
        ok, detail = checks.judge_generation(checks.HttpResult(0, None, 180.0, error="timeout", timed_out=True), course_id=CSS350, mode="base")
        self.assertFalse(ok)
        self.assertIn("timed out", detail)

    def test_direct_generation_requires_the_exact_version(self) -> None:
        good = self._result(answer=ANSWER, courseId=CSS360, modelVersion="v2", adapterLoaded=True, model="css360-ft-v2:latest")
        self.assertTrue(checks.judge_direct_generation(good, course_id=CSS360, version="v2")[0])
        self.assertFalse(checks.judge_direct_generation(good, course_id=CSS360, version="v3")[0])


class LatencyMathTests(unittest.TestCase):
    def test_percentiles_are_nearest_rank(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        self.assertEqual(checks.percentile(values, 0.5), 5.0)
        self.assertEqual(checks.percentile(values, 0.95), 10.0)
        self.assertEqual(checks.percentile([3.0], 0.95), 3.0)
        self.assertIsNone(checks.percentile([], 0.5))

    def test_summary_counts_failures_and_timeouts_and_excludes_them_from_quantiles(self) -> None:
        samples = [
            {"ok": True, "seconds": 2.0, "timedOut": False},
            {"ok": True, "seconds": 4.0, "timedOut": False},
            {"ok": False, "seconds": 180.0, "timedOut": True},
            {"ok": False, "seconds": 0.5, "timedOut": False},
        ]
        summary = checks.summarize_latencies(samples)
        self.assertEqual((summary["requests"], summary["ok"], summary["failed"], summary["timeouts"]), (4, 2, 2, 1))
        self.assertEqual((summary["median"], summary["p95"], summary["max"], summary["min"]), (3.0, 4.0, 4.0, 2.0))
        empty = checks.summarize_latencies([])
        self.assertIsNone(empty["median"])


class VerifyScriptTests(unittest.TestCase):
    def test_a_healthy_deployment_passes_every_check(self) -> None:
        transport = FakeTransport()
        results, out = _verify(transport, FakeFacts())
        self.assertEqual(_by_status(results, "FAIL"), [])
        self.assertEqual(_by_status(results, "SKIP"), [])
        names = [name for _, name, _ in results]
        for expected in (
            "aiswe-finetuned is-enabled", "aiswe-finetuned is-active", "port 9001 owned by the local service", "ollama reachable",
            "ollama has css350-ft-v1:latest", f"{CSS350} is mapped", "service /health status=ok", "service /health adapterLoaded=true",
            f"service reports {CSS350} v1 servable", f"direct /generate {CSS350} v1", f"direct /generate {CSS360} v2",
            "backend /api/health", "backend /api/fine-tuned/health",
            f"backend fineTuned {CSS350}", f"backend fineTunedRag {CSS350}", f"backend base {CSS350}", f"backend rag {CSS350}",
            f"backend fineTuned {CSS360}", f"backend fineTunedRag {CSS360}",
        ):
            self.assertIn(expected, names)
        self.assertTrue(transport.logged_out)
        self.assertNotIn(ANSWER, out)
        self.assertNotIn("SECRET-TOKEN", out)
        self.assertIn("summary:", out)

    def test_without_credentials_backend_generation_is_skip_not_pass(self) -> None:
        results, _ = _verify(FakeTransport(), FakeFacts(), credentials=None)
        self.assertEqual(_by_status(results, "FAIL"), [])
        skipped = _by_status(results, "SKIP")
        self.assertIn("backend /api/fine-tuned/health", skipped)
        self.assertIn(f"backend fineTuned {CSS350}", skipped)
        # The unauthenticated checks still ran.
        self.assertIn(f"direct /generate {CSS360} v2", _by_status(results, "PASS"))

    def test_a_tunnel_on_the_port_fails(self) -> None:
        results, _ = _verify(FakeTransport(), FakeFacts(ss='LISTEN 0 128 127.0.0.1:9001 0.0.0.0:* users:(("ssh",pid=4242,fd=5))\n'))
        self.assertTrue(any("SSH forward" in name for name in _by_status(results, "FAIL")))

    def test_a_disabled_or_inactive_unit_fails(self) -> None:
        results, _ = _verify(FakeTransport(), FakeFacts(enabled="disabled", active="inactive"))
        self.assertIn("aiswe-finetuned is-enabled", _by_status(results, "FAIL"))
        self.assertIn("aiswe-finetuned is-active", _by_status(results, "FAIL"))

    def test_a_missing_ollama_model_and_an_unmapped_course_fail(self) -> None:
        transport = FakeTransport(ollama_models=["llama3.2:3b", "css360-ft-v2:latest"])
        results, _ = _verify(transport, FakeFacts())
        failed = _by_status(results, "FAIL")
        self.assertIn("ollama has css350-ft-v1:latest", failed)
        self.assertIn(f"service reports {CSS350} v1 servable", failed)
        results, _ = _verify(FakeTransport(), FakeFacts(env=f"FINETUNED_OLLAMA_MODELS={CSS360}@v2=css360-ft-v2\n"))
        self.assertIn(f"{CSS350} is mapped", _by_status(results, "FAIL"))

    def test_a_backend_version_that_this_host_does_not_map_fails(self) -> None:
        # The registry publishes v2 for CSS 350 but the VM only maps v1.
        transport = FakeTransport(backend_version={CSS350: "v2", CSS360: "v2"})
        results, _ = _verify(transport, FakeFacts())
        failed = {name: detail for s, name, detail in results if s == "FAIL"}
        self.assertIn(f"backend fineTuned {CSS350}", failed)
        self.assertIn("not mapped", failed[f"backend fineTuned {CSS350}"])

    def test_a_failed_sign_in_fails_and_skips_generation(self) -> None:
        results, _ = _verify(FakeTransport(login_ok=False), FakeFacts())
        self.assertIn("administrator sign-in", _by_status(results, "FAIL"))
        self.assertIn(f"backend rag {CSS360}", _by_status(results, "SKIP"))

    def test_skip_generate_asks_no_model_anything(self) -> None:
        transport = FakeTransport()
        results, _ = _verify(transport, FakeFacts(), "--skip-generate")
        self.assertEqual(_by_status(results, "FAIL"), [])
        self.assertFalse(any(url.endswith("/generate") for _, url, _, _ in transport.calls))

    def test_main_exit_code_follows_the_checks(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            code = verify.main(["--admin-email", ""], facts=FakeFacts(), transport=FakeTransport())
        self.assertEqual(code, 0)
        with redirect_stdout(io.StringIO()):
            code = verify.main([], facts=FakeFacts(active="failed"), transport=FakeTransport())
        self.assertEqual(code, 1)


class LatencyProbeTests(unittest.TestCase):
    def test_concurrency_rules(self) -> None:
        self.assertEqual(probe.parse_concurrency("1,2", allow_four=False), [1, 2])
        self.assertEqual(probe.parse_concurrency("1, 2, 4", allow_four=True), [1, 2, 4])
        with self.assertRaises(ValueError):
            probe.parse_concurrency("4", allow_four=False)
        with self.assertRaises(ValueError):
            probe.parse_concurrency("8", allow_four=True)
        with self.assertRaises(ValueError):
            probe.parse_concurrency("0", allow_four=False)

    def test_the_default_plan_is_small(self) -> None:
        plan = probe.build_plan(courses=[CSS350, CSS360], modes=list(probe.DEFAULT_MODES), concurrency=[1, 2], requests=2)
        self.assertEqual(len(plan["cells"]), 4)
        self.assertEqual(plan["totalRequests"], 4 * (1 + 2 * 2))
        with self.assertRaises(ValueError):
            probe.build_plan(courses=[CSS350], modes=["fineTuned"], concurrency=[1], requests=50)
        with self.assertRaises(ValueError):
            probe.build_plan(courses=[CSS350], modes=["nope"], concurrency=[1], requests=1)

    def test_run_probe_orchestrates_first_then_each_level_and_totals_add_up(self) -> None:
        plan = probe.build_plan(courses=[CSS350], modes=["fineTuned", "fineTunedRag"], concurrency=[1, 2], requests=3)
        calls: list[tuple[str, str]] = []
        clock = iter(range(1, 100))

        def send(course: str, mode: str) -> dict[str, Any]:
            calls.append((course, mode))
            n = next(clock)
            if mode == "fineTunedRag" and n % 5 == 0:
                return {"ok": False, "seconds": 180.0, "timedOut": True, "status": 0, "detail": "timed out"}
            return {"ok": True, "seconds": float(n), "timedOut": False, "status": 200, "detail": ""}

        lines: list[str] = []
        report = probe.run_probe(plan, send, log=lines.append)
        self.assertEqual(len(calls), plan["totalRequests"])
        self.assertEqual(calls[:7], [(CSS350, "fineTuned")] * 7)
        cell = report["cells"][0]
        self.assertTrue(cell["first"]["ok"])
        self.assertEqual(cell["byConcurrency"]["1"]["requests"], 3)
        self.assertEqual(cell["byConcurrency"]["2"]["ok"], 3)
        totals = report["totals"]
        self.assertEqual(totals["requests"], 14)
        self.assertEqual(totals["ok"] + totals["failed"], 14)
        self.assertGreaterEqual(totals["timeouts"], 1)
        self.assertEqual(totals["timeouts"], sum(1 for c in report["cells"] if c["first"]["timedOut"]) + sum(s["timeouts"] for c in report["cells"] for s in c["byConcurrency"].values()))
        table = probe.render_table(report)
        self.assertIn("median", table)
        self.assertIn("total: 14 requests", table)
        self.assertNotIn(ANSWER, table)
        self.assertNotIn("answer", json.dumps(report).lower().replace("answers are", ""))

    def test_main_runs_against_the_fake_backend_and_writes_only_timings(self) -> None:
        transport = FakeTransport()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "latency.json"
            out = io.StringIO()
            with redirect_stdout(out), unittest.mock.patch.dict("os.environ", {checks.PASSWORD_ENV: "pw"}):
                code = probe.main(["--admin-email", "admin@uw.edu", "--requests", "1", "--out", str(out_path)], transport=transport)
            self.assertEqual(code, 0, out.getvalue())
            report = json.loads(out_path.read_text())
            self.assertEqual(report["totals"]["requests"], 4 * (1 + 1 * 2))
            self.assertEqual(report["totals"]["failed"], 0)
            self.assertNotIn(ANSWER, out_path.read_text())
            self.assertNotIn("SECRET-TOKEN", out.getvalue())
            self.assertTrue(transport.logged_out)
            self.assertNotIn(4, probe.parse_concurrency(probe.build_parser().get_default("concurrency"), allow_four=True))

    def test_main_refuses_without_credentials_and_on_a_failed_sign_in(self) -> None:
        with redirect_stdout(io.StringIO()), unittest.mock.patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop(checks.PASSWORD_ENV, None)
            os.environ.pop(checks.EMAIL_ENV, None)
            self.assertEqual(probe.main([], transport=FakeTransport()), 2)
        with redirect_stdout(io.StringIO()), unittest.mock.patch.dict("os.environ", {checks.PASSWORD_ENV: "bad"}):
            self.assertEqual(probe.main(["--admin-email", "a@uw.edu"], transport=FakeTransport(login_ok=False)), 1)

    def test_a_failing_mode_makes_the_exit_code_nonzero(self) -> None:
        with redirect_stdout(io.StringIO()), unittest.mock.patch.dict("os.environ", {checks.PASSWORD_ENV: "pw"}):
            code = probe.main(["--admin-email", "a@uw.edu", "--requests", "1", "--concurrency", "1"], transport=FakeTransport(fail_modes={"fineTunedRag"}))
        self.assertEqual(code, 1)


import unittest.mock  # noqa: E402  (used by the probe tests above)

if __name__ == "__main__":
    unittest.main()
