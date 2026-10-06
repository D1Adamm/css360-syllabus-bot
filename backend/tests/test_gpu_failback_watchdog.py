"""scripts/gpu_failback_watchdog.py: failing classroom GPU mode back to the VM, and only when it should.

Everything here runs against a temporary mode file, state directory and
finetuned.env, with a fake transport for the GPU service and a fake (or
intercepted) VM warm-up. Nothing reads or writes the real
`~/.config/aiswe/generation-mode.json`, the real tunnel, Ollama or systemd.
"""

from __future__ import annotations

import configparser
import datetime as dt
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from app import generation_mode as gm

# The switch script exactly as its own tests load it, so both test files (and
# the watchdog, which looks it up in sys.modules) share one module object.
from test_classroom_gpu_mode_script import COURSE, FakeWorld, script

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("gpu_failback_watchdog", REPO_ROOT / "scripts" / "gpu_failback_watchdog.py")
watchdog = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = watchdog
_spec.loader.exec_module(watchdog)
helpers = watchdog.helpers
checks = script.checks

GPU = "http://127.0.0.1:9101"
NOW = dt.datetime(2026, 10, 1, 18, 0, 0, tzinfo=dt.timezone.utc)
WRAPPER = REPO_ROOT / "scripts" / "gpu_failback_watchdog.sh"
UNITS = REPO_ROOT / "scripts" / "systemd"


def z(moment: dt.datetime) -> str:
    """As training/serving_session.py records `expiresAt`."""
    return moment.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def healthy(**overrides) -> dict:
    body = {"status": "ok", "adapterLoaded": True, "cudaAvailable": True, "targets": ["course", "base"],
            "decoding": script.service_helpers.decoding_summary(256), "secondsRemaining": 7200, "gpuName": "H100",
            "courses": [{"courseId": COURSE, "versions": ["v1"]}], "served": {"base": 4, "course": 4}}
    body.update(overrides)
    return body


class Clock:
    def __init__(self, now: dt.datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> dt.datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += dt.timedelta(seconds=seconds)


class FakeGpu:
    """The GPU service behind the tunnel: one scripted answer per /health call.

    "up" is a healthy answer, "down" is no answer at all (the tunnel's port
    refuses), a dict is that /health body. When the script runs out, `default`.
    """

    def __init__(self, *answers, default="up") -> None:
        self.answers = list(answers)
        self.default = default
        self.calls: list[tuple[str, str]] = []
        self.on_request = None

    def request(self, method, url, *, body=None, headers=None, timeout=30.0):
        self.calls.append((method, url))
        if self.on_request is not None:
            self.on_request()
        answer = self.answers.pop(0) if self.answers else self.default
        if answer == "down":
            return checks.HttpResult(0, None, 0.01, error="[Errno 111] Connection refused")
        return checks.HttpResult(200, healthy() if answer == "up" else answer, 0.01)


class FakeWarm:
    """The VM warm-up. Records what it was asked, and the mode in effect when it began."""

    def __init__(self, ok: bool = True, lines=None, raises: BaseException | None = None) -> None:
        self.ok = ok
        self.lines = lines if lines is not None else (["PASS llama3.2:3b @ http://127.0.0.1:11434: resident"] if ok else
                                                      ["FAIL llama3.2:3b @ http://127.0.0.1:11434: HTTP 0 no answer"])
        self.raises = raises
        self.calls: list[tuple[str, str]] = []
        self.mode_when_called: list[str] = []

    def __call__(self, cgm, settings, course_id, version):
        self.calls.append((course_id, version))
        self.mode_when_called.append(gm.parse_mode(gm.mode_file_path().read_text()).kind)
        if self.raises is not None:
            raise self.raises
        return self.ok, self.lines


class WatchdogCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.mode_file = root / "config" / "generation-mode.json"
        self.mode_file.parent.mkdir()
        self.state_dir = root / "state"
        self.finetuned_env = root / "finetuned.env"
        self.finetuned_env.write_text(
            f"FINETUNED_OLLAMA_MODELS={COURSE}@v1=css360e-v1:latest\nOLLAMA_BASE_URL=http://127.0.0.1:11435\n")
        self.env = patch.dict(os.environ, {gm.MODE_FILE_ENV: str(self.mode_file),
                                           helpers.STATE_DIR_ENV: str(self.state_dir)})
        self.env.start()
        os.environ.pop("JOURNAL_STREAM", None)
        os.environ.pop("OLLAMA_MODEL", None)
        for name in ("AISWE_GPU_FAILBACK_FAILURES", "AISWE_GPU_FAILBACK_EXPIRY_MARGIN",
                     "AISWE_GPU_FAILBACK_HEALTH_TIMEOUT", "AISWE_GPU_FAILBACK_KEEP_ALIVE"):
            os.environ.pop(name, None)
        # The switch script's own paths: its tunnel socket and the model mapping.
        self.patches = [patch.object(script, "STATE_ROOT", root / "switch-state"),
                        patch.object(script, "FINETUNED_ENV", self.finetuned_env)]
        for p in self.patches:
            p.start()
        gm.reset_for_tests()
        self.clock = Clock()

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()
        self.env.stop()
        self.tmp.cleanup()
        gm.reset_for_tests()

    # -- helpers ----------------------------------------------------------- #

    def gpu_content(self, *, job="296331", node="g014", expires: dt.datetime | str | None = None,
                    since: dt.datetime | None = None, course=COURSE, version="v1") -> dict:
        """What `start` writes."""
        if isinstance(expires, dt.datetime):
            expires = z(expires)
        return {"mode": "gpu", "gpuUrl": GPU, "since": (since or self.clock.now - dt.timedelta(minutes=30)).isoformat(timespec="seconds"),
                "node": node, "jobId": job, "expiresAt": expires or "", "courseId": course, "modelVersion": version,
                "switchedBy": "madamk"}

    def write_gpu(self, **kwargs) -> str:
        self.mode_file.write_text(json.dumps(self.gpu_content(**kwargs), indent=2) + "\n")
        return self.mode_file.read_text()

    def check(self, gpu=None, warm=None, argv=()) -> tuple[int, str]:
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(out):
            code = watchdog.main(list(argv), transport=gpu if gpu is not None else FakeGpu(),
                                 clock=self.clock, warm=warm if warm is not None else FakeWarm())
        return code, out.getvalue()

    def mode(self):
        return gm.parse_mode(self.mode_file.read_text())

    def state(self) -> dict:
        return helpers.read_state()

    def log(self) -> list[dict]:
        return helpers.read_log(limit=100)


class VmModeTests(WatchdogCase):
    def test_already_in_vm_mode_is_a_no_op(self) -> None:
        self.mode_file.write_text(json.dumps({"mode": "vm", "since": "2026-10-01T17:00:00+00:00", "switchedBy": "madamk"}))
        before = (self.mode_file.read_text(), self.mode_file.stat().st_mtime_ns)
        gpu, warm = FakeGpu(), FakeWarm()
        for _ in range(3):
            code, out = self.check(gpu, warm)
            self.assertEqual(code, 0, out)
            self.clock.advance(15)
        self.assertEqual((self.mode_file.read_text(), self.mode_file.stat().st_mtime_ns), before)
        self.assertEqual(gpu.calls, [])  # the GPU is not even asked
        self.assertEqual(warm.calls, [])
        self.assertEqual(self.log(), [])
        self.assertEqual(self.state()["lastCheck"]["outcome"], "vm_mode")

    def test_no_mode_file_is_vm_mode(self) -> None:
        gpu, warm = FakeGpu(), FakeWarm()
        code, out = self.check(gpu, warm)
        self.assertEqual(code, 0, out)
        self.assertFalse(self.mode_file.exists())  # and it is not created
        self.assertEqual((gpu.calls, warm.calls, self.log()), ([], [], []))

    def test_a_mode_file_the_backend_ignores_is_left_alone(self) -> None:
        # Invalid means VM to the backend (generation_mode.py), so there is nothing to fail back from.
        for text in ("{oops", json.dumps({"mode": "gpu", "gpuUrl": "http://10.0.0.5:9101"})):
            with self.subTest(text=text):
                self.mode_file.write_text(text)
                gpu = FakeGpu("down")
                code, out = self.check(gpu)
                self.assertEqual(code, 0, out)
                self.assertEqual(self.mode_file.read_text(), text)
                self.assertEqual(gpu.calls, [])


class HealthyGpuTests(WatchdogCase):
    def test_gpu_mode_with_a_healthy_gpu_is_a_no_op(self) -> None:
        text = self.write_gpu(expires=self.clock.now + dt.timedelta(hours=2))
        mtime = self.mode_file.stat().st_mtime_ns
        gpu, warm = FakeGpu(), FakeWarm()
        for _ in range(5):
            code, out = self.check(gpu, warm)
            self.assertEqual(code, 0, out)
            self.clock.advance(15)
        self.assertEqual((self.mode_file.read_text(), self.mode_file.stat().st_mtime_ns), (text, mtime))
        self.assertTrue(self.mode().is_gpu)
        self.assertEqual(gpu.calls, [("GET", GPU + "/health")] * 5)
        self.assertEqual((warm.calls, self.log()), ([], []))
        self.assertEqual(self.state()["lastCheck"]["outcome"], "healthy")
        self.assertIsNone(self.state()["streak"])

    def test_an_expiry_that_cannot_be_read_is_not_an_expiry(self) -> None:
        self.write_gpu(expires="at the end of class")
        code, out = self.check(FakeGpu())
        self.assertEqual(code, 0, out)
        self.assertTrue(self.mode().is_gpu)


class ExpiryTests(WatchdogCase):
    def test_a_passed_expiresAt_switches_to_the_vm_at_once(self) -> None:
        self.write_gpu(expires=self.clock.now - dt.timedelta(minutes=1))
        self.assertTrue(gm.current_mode().is_gpu)  # what the backend sees before
        gpu, warm = FakeGpu(), FakeWarm()
        code, out = self.check(gpu, warm)
        self.assertEqual(code, 0, out)
        self.assertFalse(self.mode().is_gpu)
        self.assertFalse(gm.current_mode().is_gpu)  # and after: the next request generates on the VM
        self.assertEqual(gpu.calls, [])  # no health check is needed to know the allocation is over
        self.assertIn("AUTOMATIC FAILBACK", out)

        # The mode file says who switched, and keeps the job for `stop`'s reminder.
        written = json.loads(self.mode_file.read_text())
        self.assertEqual((written["mode"], written["switchedBy"]), ("vm", "gpu-failback-watchdog"))
        self.assertEqual(oct(self.mode_file.stat().st_mode & 0o777), "0o600")

        # The log: when, why, which job, node, course and version, and the warm-up.
        failback, warmup = self.log()
        self.assertEqual(failback["event"], "failback")
        self.assertEqual(failback["at"], "2026-10-01T18:00:00+00:00")
        self.assertEqual(failback["reason"], "expired")
        self.assertEqual((failback["jobId"], failback["node"]), ("296331", "g014"))
        self.assertEqual((failback["courseId"], failback["modelVersion"]), (COURSE, "v1"))
        self.assertEqual((warmup["event"], warmup["ok"], warmup["jobId"]), ("failback_warmup", True, "296331"))
        self.assertEqual(self.state()["lastFailback"]["vmWarmup"], "ok")
        self.assertEqual(warm.calls, [(COURSE, "v1")])

    def test_it_switches_just_before_the_job_is_killed_not_after(self) -> None:
        # 20 s left, 30 s margin: the next check would come after the job is gone.
        self.write_gpu(expires=self.clock.now + dt.timedelta(seconds=20))
        code, out = self.check(FakeGpu())
        self.assertEqual(code, 0, out)
        self.assertFalse(self.mode().is_gpu)
        self.assertEqual(self.log()[0]["reason"], "expired")

    def test_outside_the_margin_nothing_happens(self) -> None:
        self.write_gpu(expires=self.clock.now + dt.timedelta(seconds=45))
        self.check(FakeGpu())
        self.assertTrue(self.mode().is_gpu)
        # With no margin, only a passed expiresAt counts.
        self.write_gpu(expires=self.clock.now + dt.timedelta(seconds=20))
        self.check(FakeGpu({**healthy(), "secondsRemaining": 20}), argv=["--expiry-margin", "0"])
        self.assertTrue(self.mode().is_gpu)
        self.assertEqual(self.log(), [])

    def test_the_gpu_service_reporting_no_time_left_is_an_expiry(self) -> None:
        # Started with --node: the mode file has no expiresAt, the service knows its own deadline.
        self.write_gpu(expires=None)
        code, out = self.check(FakeGpu(healthy(secondsRemaining=12)))
        self.assertEqual(code, 0, out)
        self.assertFalse(self.mode().is_gpu)
        self.assertEqual(self.log()[0]["reason"], "expired")

    def test_expiry_formats(self) -> None:
        for value in ("2026-10-01T17:59:00Z", "2026-10-01T17:59:00+00:00", "2026-10-01T10:59:00-07:00",
                      "2026-10-01T17:59:00", NOW.timestamp() - 60):
            with self.subTest(value=value):
                self.assertLess(helpers.parse_time(value), NOW)
        for value in ("", None, "soon", True, {}):
            with self.subTest(value=value):
                self.assertIsNone(helpers.parse_time(value))


class HealthCheckTests(WatchdogCase):
    def test_one_failed_check_does_not_fail_back(self) -> None:
        text = self.write_gpu(expires=self.clock.now + dt.timedelta(hours=2))
        warm = FakeWarm()
        # A blip, a recovery, two more blips, a recovery: never three in a row.
        gpu = FakeGpu("down", "up", "down", "down", "up")
        outcomes = []
        for _ in range(5):
            code, out = self.check(gpu, warm)
            self.assertEqual(code, 0, out)
            streak = self.state()["streak"]
            outcomes.append((self.state()["lastCheck"]["outcome"], streak["failures"] if streak else 0))
            self.clock.advance(15)
        self.assertEqual(outcomes, [("check_failed", 1), ("healthy", 0), ("check_failed", 1), ("check_failed", 2), ("healthy", 0)])
        self.assertEqual(self.mode_file.read_text(), text)
        self.assertEqual((warm.calls, self.log()), ([], []))

    def test_three_failed_checks_in_a_row_fail_back(self) -> None:
        self.write_gpu(expires=self.clock.now + dt.timedelta(hours=2))
        gpu, warm = FakeGpu(default="down"), FakeWarm()
        for expected in ("1/3", "2/3"):
            code, out = self.check(gpu, warm)
            self.assertEqual(code, 0, out)
            self.assertIn(f"FAILED {expected}", out)
            self.assertTrue(self.mode().is_gpu)
            self.clock.advance(15)
        code, out = self.check(gpu, warm)
        self.assertEqual(code, 0, out)
        self.assertFalse(self.mode().is_gpu)
        failback = self.log()[0]
        self.assertEqual((failback["reason"], failback["consecutiveFailures"]), ("gpu_unreachable", 3))
        self.assertEqual((failback["jobId"], failback["node"], failback["courseId"], failback["modelVersion"]),
                         ("296331", "g014", COURSE, "v1"))
        self.assertIn("Connection refused", failback["detail"])
        self.assertEqual(warm.calls, [(COURSE, "v1")])

    def test_a_gpu_that_answers_but_cannot_serve_is_unhealthy(self) -> None:
        cases = {
            "still loading": healthy(status="loading", adapterLoaded=False),
            "another build's decoding": healthy(decoding={**script.service_helpers.decoding_summary(256), "num_ctx": 2048}),
            "the course is no longer published": healthy(courses=[]),
            "not a health answer": {"detail": "Not Found"},
        }
        for name, body in cases.items():
            with self.subTest(name):
                self.write_gpu(job=name)
                gpu = FakeGpu(default=body)
                for _ in range(3):
                    code, out = self.check(gpu)
                    self.clock.advance(15)
                self.assertFalse(self.mode().is_gpu, out)
                self.assertEqual(self.log()[-2]["reason"], "gpu_unhealthy")

    def test_failures_long_apart_are_not_consecutive(self) -> None:
        self.write_gpu()
        gpu = FakeGpu(default="down")
        self.check(gpu)
        self.clock.advance(15)
        self.check(gpu)
        self.clock.advance(3600)  # the timer was stopped for an hour
        code, out = self.check(gpu)
        self.assertIn("FAILED 1/3", out)
        self.assertTrue(self.mode().is_gpu)

    def test_the_threshold_is_configurable(self) -> None:
        self.write_gpu()
        with patch.dict(os.environ, {"AISWE_GPU_FAILBACK_FAILURES": "2"}):
            self.check(FakeGpu("down"))
            self.assertTrue(self.mode().is_gpu)
            self.clock.advance(15)
            self.check(FakeGpu("down"))
        self.assertFalse(self.mode().is_gpu)

    def test_dry_run_changes_nothing(self) -> None:
        text = self.write_gpu(expires=self.clock.now - dt.timedelta(hours=1))
        warm = FakeWarm()
        code, out = self.check(FakeGpu("down"), warm, argv=["--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("DRY RUN: would fail back", out)
        self.assertEqual(self.mode_file.read_text(), text)
        self.assertEqual((warm.calls, self.log(), self.state()), ([], [], {}))


class StaleSessionTests(WatchdogCase):
    """The incident: the Slurm job expired, the mode file still named the old node and job."""

    def test_a_stale_gpu_mode_file_after_the_job_expired_switches_to_the_vm(self) -> None:
        self.write_gpu(job="296331", node="g014", since=self.clock.now - dt.timedelta(hours=3),
                       expires=self.clock.now - dt.timedelta(hours=1))
        gpu = FakeGpu(default="down")
        code, out = self.check(gpu)
        self.assertEqual(code, 0, out)
        self.assertFalse(self.mode().is_gpu)
        failback = self.log()[0]
        self.assertEqual((failback["reason"], failback["jobId"], failback["node"]), ("expired", "296331", "g014"))

    def test_the_same_without_a_recorded_expiry_takes_three_checks(self) -> None:
        self.write_gpu(job="296331", node="g014", expires=None)
        gpu = FakeGpu(default="down")
        for _ in range(2):
            self.check(gpu)
            self.assertTrue(self.mode().is_gpu)
            self.clock.advance(15)
        self.check(gpu)
        self.assertFalse(self.mode().is_gpu)
        self.assertEqual(self.log()[0]["reason"], "gpu_unreachable")


class FakeOllama:
    """`urllib.request.urlopen` for scripts/warm_classroom_models.py: both VM Ollama servers."""

    def __init__(self, case: WatchdogCase, *, down: bool = False) -> None:
        self.case = case
        self.down = down
        self.requests: list[tuple[str, str]] = []
        self.modes: list[str] = []
        self.loaded: dict[str, list[str]] = {}

    def __call__(self, request, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        self.modes.append(self.case.mode().kind)
        if self.down:
            raise urllib.error.URLError("[Errno 111] Connection refused")
        origin = url.split("/api/")[0]
        if url.endswith("/api/ps"):
            body = {"models": [{"name": name, "context_length": 4096, "expires_at": "later", "size": 2 << 30}
                               for name in self.loaded.get(origin, [])]}
        else:
            model = json.loads(request.data)["model"]
            self.requests.append((url, model))
            self.loaded.setdefault(origin, []).append(model if ":" in model else model + ":latest")
            body = {"done": True}

        class Response(io.BytesIO):
            status = 200

            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

        return Response(json.dumps(body).encode())


class WarmUpTests(WatchdogCase):
    def expire(self) -> None:
        self.write_gpu(expires=self.clock.now - dt.timedelta(minutes=5))

    def run_real_warm_up(self, ollama: FakeOllama) -> tuple[int, str]:
        out = io.StringIO()
        with patch("urllib.request.urlopen", ollama), redirect_stdout(out), redirect_stderr(out):
            code = watchdog.main([], transport=FakeGpu(), clock=self.clock)
        return code, out.getvalue()

    def test_the_mode_is_switched_before_any_model_is_warmed(self) -> None:
        self.expire()
        warm = FakeWarm()
        self.check(FakeGpu(), warm)
        self.assertEqual(warm.mode_when_called, ["vm"])

    def test_it_warms_the_models_stop_warms_after_the_switch(self) -> None:
        # The real warm_classroom_models.py, with Ollama intercepted.
        self.expire()
        ollama = FakeOllama(self)
        code, out = self.run_real_warm_up(ollama)
        self.assertEqual(code, 0, out)
        self.assertEqual(ollama.requests, [
            ("http://127.0.0.1:11434/api/chat", "llama3.2:3b"),         # base
            ("http://127.0.0.1:11434/api/embed", "nomic-embed-text"),   # embedding
            ("http://127.0.0.1:11435/api/chat", "css360e-v1:latest"),   # the course's fine-tuned model
        ])
        self.assertEqual(set(ollama.modes), {"vm"})  # every Ollama call came after the switch
        self.assertEqual(self.state()["lastFailback"]["vmWarmup"], "ok")

    def test_a_failed_warm_up_leaves_the_mode_vm_and_is_logged(self) -> None:
        self.expire()
        code, out = self.run_real_warm_up(FakeOllama(self, down=True))
        self.assertEqual(code, 1)
        self.assertFalse(self.mode().is_gpu)
        self.assertIn("VM warm-up after the failback FAILED (the mode stays VM)", out)
        failback, warmup = self.log()
        self.assertEqual((failback["event"], failback["reason"]), ("failback", "expired"))
        self.assertEqual((warmup["event"], warmup["ok"]), ("failback_warmup", False))
        self.assertTrue(any(line.startswith("FAIL llama3.2:3b") for line in warmup["detail"]), warmup)
        self.assertEqual(self.state()["lastFailback"]["vmWarmup"], "failed")

        # Rerunning is harmless: VM mode, no second failback, no second warm-up.
        warm = FakeWarm()
        code, out = self.check(FakeGpu("down"), warm)
        self.assertEqual((code, warm.calls, len(self.log())), (0, [], 2))
        self.assertFalse(self.mode().is_gpu)

    def test_a_course_with_no_vm_mapping_still_warms_the_base_models(self) -> None:
        self.finetuned_env.write_text("FINETUNED_OLLAMA_MODELS=\n")
        self.expire()
        ollama = FakeOllama(self)
        code, out = self.run_real_warm_up(ollama)
        self.assertEqual(code, 1)
        self.assertEqual([model for _, model in ollama.requests], ["llama3.2:3b", "nomic-embed-text"])
        self.assertIn("is not mapped", self.log()[1]["detail"][0])
        self.assertFalse(self.mode().is_gpu)

    def test_a_crashing_warm_up_is_logged_not_raised(self) -> None:
        self.expire()
        with patch.object(script, "warm_vm_models", side_effect=RuntimeError("boom")):
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(out):
                code = watchdog.main([], transport=FakeGpu(), clock=self.clock)
        self.assertEqual(code, 1)
        self.assertFalse(self.mode().is_gpu)
        self.assertIn("boom", self.log()[1]["detail"][0])


class ManualStartRaceTests(WatchdogCase):
    def test_a_session_started_during_the_check_is_not_overwritten(self) -> None:
        # Two failed checks against the dead session A.
        self.write_gpu(job="111", node="g001")
        gpu = FakeGpu(default="down")
        for _ in range(2):
            self.check(gpu)
            self.clock.advance(15)
        # While the third check waits for the dead GPU, a human runs stop + start: session B.
        session_b = self.gpu_content(job="222", node="g002", expires=self.clock.now + dt.timedelta(hours=3))
        gpu.on_request = lambda: script.write_mode_file(session_b)
        warm = FakeWarm()
        code, out = self.check(gpu, warm)
        self.assertEqual(code, 0, out)
        self.assertTrue(self.mode().is_gpu)
        self.assertEqual(self.mode().details["jobId"], "222")
        self.assertEqual((warm.calls, self.log()), ([], []))
        self.assertEqual(self.state()["lastCheck"]["outcome"], "superseded")

        # And A's failures are not held against B: B starts from zero.
        gpu.on_request = None
        self.clock.advance(15)
        code, out = self.check(gpu, warm)
        self.assertIn("FAILED 1/3", out)
        self.assertEqual(self.mode().details["jobId"], "222")

    def test_failures_against_the_old_session_do_not_count_against_a_new_one(self) -> None:
        self.write_gpu(job="111", node="g001")
        gpu = FakeGpu(default="down")
        for _ in range(2):
            self.check(gpu)
            self.clock.advance(5)
        # Between two checks, a human runs stop + start: session B, which is slow to answer once.
        self.write_gpu(job="222", node="g002", expires=self.clock.now + dt.timedelta(hours=3))
        code, out = self.check(gpu)
        self.assertIn("FAILED 1/3 (job 222 on g002)", out)
        self.assertTrue(self.mode().is_gpu)
        self.assertEqual(self.log(), [])

    def test_a_start_writing_at_the_same_moment_wins(self) -> None:
        # `start` holds the mode lock and is about to write B; the watchdog has
        # already decided A is expired. It must wait, look again, and stand down.
        self.write_gpu(job="111", node="g001", expires=self.clock.now - dt.timedelta(hours=1))
        session_b = self.gpu_content(job="222", node="g002", expires=self.clock.now + dt.timedelta(hours=3))
        holding, release = threading.Event(), threading.Event()

        def manual_start() -> None:
            with helpers.mode_lock(self.mode_file, timeout=5):
                holding.set()
                release.wait(5)
                script._replace_mode_file(session_b)

        thread = threading.Thread(target=manual_start)
        thread.start()
        self.assertTrue(holding.wait(5))
        threading.Timer(0.3, release.set).start()
        warm = FakeWarm()
        began = time.monotonic()
        code, out = self.check(FakeGpu(), warm)  # blocks on the lock until B is written
        self.assertGreaterEqual(time.monotonic() - began, 0.25)  # it did wait for the writer
        thread.join(5)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.mode().details["jobId"], "222")
        self.assertTrue(self.mode().is_gpu)
        self.assertEqual((warm.calls, self.log()), ([], []))

    def test_a_lock_held_too_long_defers_the_failback_to_the_next_run(self) -> None:
        text = self.write_gpu(expires=self.clock.now - dt.timedelta(hours=1))
        with helpers.mode_lock(self.mode_file, timeout=1):
            code, out = self.check(FakeGpu(), argv=["--lock-timeout", "0.2"])
        self.assertEqual(code, 0, out)
        self.assertEqual(self.mode_file.read_text(), text)
        self.assertEqual(self.state()["lastCheck"]["outcome"], "deferred")
        self.clock.advance(15)
        self.check(FakeGpu())
        self.assertFalse(self.mode().is_gpu)

    def test_many_interleavings_never_lose_the_new_session(self) -> None:
        for attempt in range(25):
            self.write_gpu(job="111", node="g001", expires=self.clock.now - dt.timedelta(hours=1))
            session_b = self.gpu_content(job=str(1000 + attempt), node="g002",
                                         expires=self.clock.now + dt.timedelta(hours=3))
            barrier = threading.Barrier(2)
            results: list = []

            def run_watchdog() -> None:
                barrier.wait(5)
                results.append(self.check_quiet())

            def manual_start() -> None:
                barrier.wait(5)
                script.write_mode_file(session_b)

            threads = [threading.Thread(target=run_watchdog), threading.Thread(target=manual_start)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(10)
            self.assertEqual(results, [0])
            self.assertTrue(self.mode().is_gpu, f"attempt {attempt}: the new session was overwritten")
            self.assertEqual(self.mode().details["jobId"], str(1000 + attempt))

    def check_quiet(self) -> int:
        # redirect_stdout is process-wide, so the threaded test does not capture.
        with patch.object(watchdog, "emit"):
            return watchdog.main([], transport=FakeGpu(), clock=self.clock, warm=FakeWarm())

    def test_start_after_a_failback_activates_a_new_session_the_watchdog_leaves_alone(self) -> None:
        self.write_gpu(job="111", node="g001", expires=self.clock.now - dt.timedelta(hours=1))
        self.check(FakeGpu())
        self.assertFalse(self.mode().is_gpu)

        world = FakeWorld()
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(out):
            code = script.main(["start", "--course", COURSE, "--version", "v1", "--node", "g014", "--job-id", "123",
                                "--no-tunnel", "--no-auth"], transport=world)
        self.assertEqual(code, 0, out.getvalue())
        started = self.mode_file.read_text()
        self.assertEqual(self.mode().details["jobId"], "123")

        for _ in range(3):
            self.clock.advance(15)
            code, _ = self.check(world)
            self.assertEqual(code, 0)
        self.assertEqual(self.mode_file.read_text(), started)
        self.assertEqual(len(self.log()), 2)  # only the first session's failback and warm-up


class ManualWriteLockTests(WatchdogCase):
    """`start` and `stop` write under the same lock; a stuck lock never blocks the way back to the VM."""

    def test_a_switch_to_gpu_is_refused_while_the_lock_is_held(self) -> None:
        self.mode_file.write_text(json.dumps({"mode": "vm"}))
        with helpers.mode_lock(self.mode_file, timeout=1), patch.object(script, "MODE_LOCK_SECONDS", 0.2):
            with self.assertRaises(script.Abort):
                script.write_mode_file(self.gpu_content())
        self.assertFalse(self.mode().is_gpu)

    def test_a_switch_to_the_vm_is_never_refused(self) -> None:
        self.write_gpu()
        out = io.StringIO()
        with helpers.mode_lock(self.mode_file, timeout=1), patch.object(script, "MODE_LOCK_SECONDS", 0.2), redirect_stdout(out):
            script.write_mode_file({"mode": "vm", "since": script.now_iso(), "switchedBy": "madamk"})
        self.assertFalse(self.mode().is_gpu)
        self.assertIn("switching to the VM regardless", out.getvalue())

    def test_the_lock_is_released_when_its_holder_dies(self) -> None:
        # A process killed while holding the lock leaves nothing behind to clean up.
        holder = subprocess.Popen(
            [sys.executable, "-c",
             "import fcntl, os, sys, time; fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600); "
             "fcntl.flock(fd, fcntl.LOCK_EX); print('held', flush=True); time.sleep(60)",
             str(helpers.lock_path(self.mode_file))], stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(holder.stdout.readline().strip(), "held")
            with self.assertRaises(helpers.LockBusy):
                with helpers.mode_lock(self.mode_file, timeout=0.2):
                    pass
            holder.kill()
            holder.wait(5)
            with helpers.mode_lock(self.mode_file, timeout=2) as held:
                self.assertTrue(held)
        finally:
            holder.kill()
            holder.stdout.close()


class World(FakeWorld):
    """FakeWorld plus the VM fine-tuned service, which `stop` asks for its health."""

    def request(self, method, url, *, body=None, headers=None, timeout=30.0):
        if url == "http://127.0.0.1:9001/health":
            return self._r(200, {"status": "ok"})
        return super().request(method, url, body=body, headers=headers, timeout=timeout)


class ManualStopRaceTests(WatchdogCase):
    STOP = ["stop", "--course", COURSE, "--version", "v1", "--no-auth"]

    def stop(self) -> tuple[int, str]:
        out = io.StringIO()
        with patch.object(script, "warm_vm_models", return_value=True), redirect_stdout(out), redirect_stderr(out):
            code = script.main(self.STOP, transport=World())
        return code, out.getvalue()

    def test_a_stop_during_the_check_wins_and_nothing_is_logged(self) -> None:
        self.write_gpu()
        gpu = FakeGpu(default="down")
        for _ in range(2):
            self.check(gpu)
            self.clock.advance(15)
        stopped = {"mode": "vm", "since": "2026-10-01T18:00:31+00:00", "switchedBy": "madamk"}
        gpu.on_request = lambda: script.write_mode_file(stopped)
        warm = FakeWarm()
        code, out = self.check(gpu, warm)
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads(self.mode_file.read_text()), stopped)  # the human's file, untouched
        self.assertEqual((warm.calls, self.log()), ([], []))

    def test_stop_after_a_failback_is_the_normal_stop_and_still_names_the_job(self) -> None:
        self.write_gpu(job="296331", expires=self.clock.now - dt.timedelta(minutes=1))
        self.check(FakeGpu())
        code, out = self.stop()
        self.assertEqual(code, 0, out)
        self.assertIn("VM MODE ACTIVE: all four conditions are generated on this VM, and verified.", out)
        # The watchdog never cancels a job; `stop` still knows which one to remind about.
        self.assertIn("REMINDER: Tillicum job 296331 is still running", out)
        self.assertFalse(self.mode().is_gpu)

        # The watchdog after the stop: nothing to do, nothing new logged.
        events = len(self.log())
        warm = FakeWarm()
        self.clock.advance(15)
        code, _ = self.check(FakeGpu("down"), warm)
        self.assertEqual((code, warm.calls, len(self.log())), (0, [], events))

    def test_stop_and_the_watchdog_at_the_same_moment_end_in_vm_mode(self) -> None:
        for attempt in range(25):
            self.write_gpu(expires=self.clock.now - dt.timedelta(hours=1))
            barrier = threading.Barrier(2)
            errors: list = []

            def run_watchdog() -> None:
                try:
                    barrier.wait(5)
                    with patch.object(watchdog, "emit"):
                        watchdog.main([], transport=FakeGpu(), clock=self.clock, warm=FakeWarm())
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

            def manual_stop() -> None:
                try:
                    barrier.wait(5)
                    script.write_mode_file({"mode": "vm", "since": script.now_iso(), "switchedBy": "madamk"})
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=run_watchdog), threading.Thread(target=manual_stop)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(10)
            self.assertEqual(errors, [])
            self.assertFalse(self.mode().is_gpu, f"attempt {attempt}")


class RestartSafetyTests(WatchdogCase):
    def test_rerunning_after_a_failback_is_harmless(self) -> None:
        self.write_gpu(expires=self.clock.now - dt.timedelta(minutes=1))
        warm = FakeWarm()
        self.check(FakeGpu(), warm)
        text = self.mode_file.read_text()
        for _ in range(4):
            self.clock.advance(15)
            code, _ = self.check(FakeGpu("down"), warm)
            self.assertEqual(code, 0)
        self.assertEqual(self.mode_file.read_text(), text)
        self.assertEqual((len(warm.calls), len(self.log())), (1, 2))

    def test_a_run_killed_during_the_warm_up_is_finished_by_the_next(self) -> None:
        # systemctl stop / a reboot while the models load: the process dies after the switch.
        self.write_gpu(expires=self.clock.now - dt.timedelta(minutes=1))
        with self.assertRaises(KeyboardInterrupt):
            self.check(FakeGpu(), FakeWarm(raises=KeyboardInterrupt()))
        self.assertFalse(self.mode().is_gpu)  # the switch had already happened
        self.assertEqual([r["event"] for r in self.log()], ["failback"])  # and was already logged
        self.assertEqual(self.state()["lastFailback"]["vmWarmup"], "pending")

        self.clock.advance(15)
        warm = FakeWarm()
        code, out = self.check(FakeGpu(), warm)
        self.assertEqual(code, 0, out)
        self.assertEqual(warm.calls, [(COURSE, "v1")])
        self.assertEqual(self.state()["lastFailback"]["vmWarmup"], "ok")
        self.assertEqual([r["event"] for r in self.log()], ["failback", "failback_warmup"])

        self.clock.advance(15)
        self.check(FakeGpu(), warm)
        self.assertEqual(len(warm.calls), 1)  # once

    def test_an_interrupted_warm_up_is_not_retried_for_ever(self) -> None:
        self.write_gpu(expires=self.clock.now - dt.timedelta(minutes=1))
        for _ in range(watchdog.MAX_WARMUP_ATTEMPTS):
            with self.assertRaises(KeyboardInterrupt):
                self.check(FakeGpu(), FakeWarm(raises=KeyboardInterrupt()))
            self.clock.advance(15)
        warm = FakeWarm()
        code, out = self.check(FakeGpu(), warm)
        self.assertEqual((code, warm.calls), (0, []))
        self.assertEqual(self.state()["lastFailback"]["vmWarmup"], "failed")
        self.assertEqual((self.log()[-1]["event"], self.log()[-1]["ok"]), ("failback_warmup", False))
        self.assertFalse(self.mode().is_gpu)

    def test_a_torn_state_file_is_treated_as_no_state(self) -> None:
        self.write_gpu()
        self.state_dir.mkdir(parents=True)
        (self.state_dir / helpers.STATE_FILE).write_text('{"streak": {"failures": 2, "finger')
        code, out = self.check(FakeGpu("down"))
        self.assertEqual(code, 0, out)
        self.assertIn("FAILED 1/3", out)
        self.assertTrue(self.mode().is_gpu)

    def test_two_runs_at_once_do_not_both_act(self) -> None:
        self.write_gpu(expires=self.clock.now - dt.timedelta(minutes=1))
        with helpers.run_lock():
            warm = FakeWarm()
            code, out = self.check(FakeGpu(), warm)
        self.assertEqual(code, 0)
        self.assertIn("another watchdog run is in progress", out)
        self.assertTrue(self.mode().is_gpu)
        self.assertEqual(warm.calls, [])


class NoCostNoCredentialsTests(WatchdogCase):
    def test_a_failback_runs_no_command_and_needs_no_sign_in(self) -> None:
        # No ssh, no scancel, no sbatch, no tunnel, no admin session: any of them would raise here.
        forbidden = AssertionError("the watchdog must not do this")
        self.write_gpu(expires=self.clock.now - dt.timedelta(minutes=1))
        with patch("subprocess.run", side_effect=forbidden), patch("subprocess.Popen", side_effect=forbidden), \
                patch.object(script, "sign_in", side_effect=forbidden), \
                patch.object(script, "cancel_job", side_effect=forbidden), \
                patch.object(script, "open_tunnel", side_effect=forbidden), \
                patch.object(script, "close_tunnel", side_effect=forbidden), \
                patch.object(script, "serving_session", side_effect=forbidden):
            code, out = self.check(FakeGpu())
            self.assertEqual(code, 0, out)
            self.assertFalse(self.mode().is_gpu)
            self.write_gpu()
            for _ in range(3):
                self.clock.advance(15)
                code, out = self.check(FakeGpu(default="down"))
                self.assertEqual(code, 0, out)
        self.assertFalse(self.mode().is_gpu)

    def test_the_watchdog_can_only_ever_write_vm_mode(self) -> None:
        source = (REPO_ROOT / "scripts" / "gpu_failback_watchdog.py").read_text()
        for word in ("scancel", "sbatch", "srun", "subprocess", "getpass", "open_tunnel", "start_finetuned_service"):
            self.assertNotIn(word, source)
        self.assertEqual(source.count("replace_mode_file_if_unchanged("), 1)
        self.assertNotIn("write_mode_file(", source)


class StatusTests(WatchdogCase):
    def status(self, timer) -> tuple[int, str]:
        out = io.StringIO()
        report = script.watchdog_report

        def report_at_the_test_clock(mode, *, timer):
            return report(mode, timer=timer, now=self.clock.now)

        with patch.object(script, "watchdog_timer_state", return_value=timer), \
                patch.object(script, "tunnel_alive", return_value=False), \
                patch.object(script, "watchdog_report", report_at_the_test_clock), \
                redirect_stdout(out), redirect_stderr(out):
            code = script.main(["status"], transport=World())
        return code, out.getvalue()

    RUNNING = {"enabled": "enabled", "active": "active"}

    def test_status_shows_the_watchdog_its_last_check_and_its_last_failback(self) -> None:
        self.write_gpu(job="296331", node="g014", expires=self.clock.now - dt.timedelta(minutes=1))
        self.check(FakeGpu())
        self.clock.advance(15)
        self.check(FakeGpu())
        self.clock.advance(12)
        code, out = self.status(self.RUNNING)
        self.assertEqual(code, 0, out)
        self.assertIn("MODE: VM (normal)", out)
        self.assertIn("switchedBy: gpu-failback-watchdog", out)
        self.assertIn("AUTOMATIC FAILBACK at 2026-10-01T18:00:00+00:00: expired (job 296331 on g014)", out)
        self.assertIn("Automatic failback watchdog (aiswe-gpu-failback.timer):", out)
        self.assertIn("timer          RUNNING (enabled, active)", out)
        self.assertIn("last check     2026-10-01T18:00:15+00:00 (12s ago): VM mode, nothing to do", out)
        self.assertIn(f"last failback  2026-10-01T18:00:00+00:00: expired (job 296331 on g014, {COURSE}@v1); VM warm-up ok", out)
        self.assertNotIn("NOTE:", out)

    def test_status_shows_a_failed_warm_up(self) -> None:
        self.write_gpu(expires=self.clock.now - dt.timedelta(minutes=1))
        self.check(FakeGpu(), FakeWarm(ok=False))
        code, out = self.status(self.RUNNING)
        self.assertIn("VM WARM-UP FAILED", out)
        self.assertIn("FAIL llama3.2:3b", out)

    def test_status_in_gpu_mode_shows_the_check_count(self) -> None:
        self.write_gpu(expires=self.clock.now + dt.timedelta(hours=2))
        self.check(FakeGpu("down"))
        code, out = self.status(self.RUNNING)
        self.assertIn("GPU check FAILED 1/3 (job 296331 on g014)", out)
        self.assertIn("last failback  none", out)

    def test_status_says_when_gpu_mode_has_no_watchdog(self) -> None:
        self.write_gpu(expires=self.clock.now + dt.timedelta(hours=2))
        for timer, expected in (({"enabled": "not-found", "active": "inactive"}, "NOT INSTALLED"),
                                ({"enabled": "disabled", "active": "inactive"}, "NOT RUNNING (disabled, inactive)"),
                                (None, "systemctl is not available")):
            with self.subTest(timer=timer):
                code, out = self.status(timer)
                self.assertEqual(code, 0, out)  # a note, not a new failure of `status`
                self.assertIn(expected, out)
                self.assertIn("NOTE: GPU mode is on without the failback watchdog", out)
                self.assertIn("last check     never", out)

    def test_status_says_when_a_running_watchdog_has_stopped_checking(self) -> None:
        self.check(FakeGpu())
        self.clock.advance(600)
        code, out = self.status(self.RUNNING)
        self.assertIn("NOTE: the watchdog timer is active but its last check was 10 min ago", out)

    def test_the_timer_state_is_read_from_systemctl(self) -> None:
        def fake_run(argv, **kwargs):
            self.assertEqual(argv[:2] + argv[3:], ["systemctl", "--user", "aiswe-gpu-failback.timer"])
            answer = {"is-enabled": ("enabled\n", 0), "is-active": ("inactive\n", 3)}[argv[2]]
            return subprocess.CompletedProcess(argv, answer[1], stdout=answer[0], stderr="")

        with patch.object(script.shutil, "which", return_value="/usr/bin/systemctl"):
            self.assertEqual(script.watchdog_timer_state(fake_run), {"enabled": "enabled", "active": "inactive"})
            missing = lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, stdout="", stderr="No such file")  # noqa: E731
            self.assertEqual(script.watchdog_timer_state(missing)["enabled"], "not-found")
        with patch.object(script.shutil, "which", return_value=None):
            self.assertIsNone(script.watchdog_timer_state(fake_run))


def unit(name: str) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None, strict=False, delimiters=("=",), comment_prefixes=("#",))
    parser.optionxform = str
    parser.read(UNITS / name)
    return parser


class SystemdUnitTests(unittest.TestCase):
    """The unit files and their installer. systemctl is a recording fake: no real systemd here."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.config = root / "config"
        self.calls = root / "systemctl.log"
        bin_dir = root / "bin"
        bin_dir.mkdir()
        (bin_dir / "systemctl").write_text(
            '#!/usr/bin/env bash\necho "$*" >> "$FAKE_SYSTEMCTL_LOG"\n'
            'case "$*" in *is-enabled*) echo enabled;; *is-active*) echo active;; esac\n')
        (bin_dir / "loginctl").write_text("#!/usr/bin/env bash\necho yes\n")
        for tool in bin_dir.iterdir():
            tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
        self.env = {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "HOME": str(root / "home"), "USER": "tester",
            "XDG_CONFIG_HOME": str(self.config), "FAKE_SYSTEMCTL_LOG": str(self.calls),
            "AISWE_GPU_FAILBACK_STATE_DIR": str(root / "state"),
            gm.MODE_FILE_ENV: str(root / "generation-mode.json"),
        }
        self.mode_file = root / "generation-mode.json"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_wrapper(self, *argv) -> subprocess.CompletedProcess:
        return subprocess.run([str(WRAPPER), *argv], env=self.env, capture_output=True, text=True, timeout=120)

    def systemctl_calls(self) -> list[str]:
        return self.calls.read_text().splitlines() if self.calls.exists() else []

    def test_the_service_is_one_check_and_cannot_respawn_or_be_enabled_alone(self) -> None:
        service = unit("aiswe-gpu-failback.service")
        self.assertEqual(service["Service"]["Type"], "oneshot")
        self.assertEqual(service["Service"]["ExecStart"], "@REPO_ROOT@/scripts/gpu_failback_watchdog.sh run")
        self.assertNotIn("Restart", service["Service"])
        self.assertNotIn("Install", service)  # only the timer starts it
        # Long enough for three model loads at the watchdog's per-model timeout.
        self.assertGreaterEqual(int(service["Service"]["TimeoutStartSec"]), 3 * watchdog.DEFAULT_WARM_TIMEOUT)
        self.assertTrue(service["Service"]["EnvironmentFile"].startswith("-"))  # optional

    def test_the_timer_fires_every_15_seconds(self) -> None:
        timer = unit("aiswe-gpu-failback.timer")
        self.assertEqual(timer["Timer"]["OnCalendar"], "*:*:0/15")
        self.assertEqual(timer["Timer"]["AccuracySec"], "1s")  # the default (1 min) would defeat it
        self.assertEqual(timer["Timer"]["Unit"], helpers.SERVICE_UNIT)
        self.assertEqual(timer["Install"]["WantedBy"], "timers.target")
        self.assertTrue((UNITS / helpers.TIMER_UNIT).is_file() and (UNITS / helpers.SERVICE_UNIT).is_file())

    def test_install_renders_enables_and_starts_and_is_repeatable(self) -> None:
        first = self.run_wrapper("install")
        self.assertEqual(first.returncode, 0, first.stderr)
        unit_dir = self.config / "systemd" / "user"
        service = (unit_dir / "aiswe-gpu-failback.service").read_text()
        self.assertNotIn("@REPO_ROOT@", service)
        exec_start = next(line for line in service.splitlines() if line.startswith("ExecStart="))
        command, argument = exec_start[len("ExecStart="):].rsplit(" ", 1)
        self.assertEqual((Path(command).resolve(), argument), (WRAPPER.resolve(), "run"))
        self.assertTrue(os.access(command, os.X_OK))
        self.assertTrue((unit_dir / "aiswe-gpu-failback.timer").is_file())
        self.assertEqual(self.systemctl_calls(), ["--user daemon-reload", "--user enable aiswe-gpu-failback.timer",
                                                  "--user restart aiswe-gpu-failback.timer"])
        self.assertIn("written", first.stdout)

        # Again (an update, a re-install): same files, timer restarted cleanly.
        second = self.run_wrapper("install")
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(second.stdout.count("unchanged"), 2)
        self.assertEqual(self.systemctl_calls()[3:], ["--user daemon-reload", "--user enable aiswe-gpu-failback.timer",
                                                      "--user restart aiswe-gpu-failback.timer"])

    def test_install_check_writes_and_enables_nothing(self) -> None:
        result = self.run_wrapper("install", "--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.config / "systemd").exists())
        self.assertEqual(self.systemctl_calls(), [])

    def test_uninstall_stops_disables_and_removes_and_is_repeatable(self) -> None:
        self.run_wrapper("install")
        self.calls.unlink()
        for _ in range(2):
            result = self.run_wrapper("uninstall")
            self.assertEqual(result.returncode, 0, result.stderr)
        unit_dir = self.config / "systemd" / "user"
        self.assertEqual(list(unit_dir.iterdir()), [])
        self.assertEqual(self.systemctl_calls()[:4], [
            "--user disable --now aiswe-gpu-failback.timer", "--user stop aiswe-gpu-failback.service",
            "--user daemon-reload", "--user reset-failed aiswe-gpu-failback.service"])

    def test_the_command_the_unit_runs_works_as_a_process(self) -> None:
        # Exactly what ExecStart runs, in a clean environment: VM mode, so a no-op.
        result = self.run_wrapper("run")
        self.assertEqual((result.returncode, result.stdout.strip()), (0, "gpu-failback: VM mode, nothing to do"), result.stderr)
        self.assertFalse(self.mode_file.exists())

        # GPU mode with a passed expiry, as a dry run: it would fail back, and writes nothing.
        text = json.dumps({"mode": "gpu", "gpuUrl": "http://127.0.0.1:9199", "jobId": "296331", "node": "g014",
                           "expiresAt": "2026-01-01T00:00:00Z", "courseId": COURSE, "modelVersion": "v1"})
        self.mode_file.write_text(text)
        result = self.run_wrapper("run", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DRY RUN: would fail back to the VM now (expired", result.stderr)
        self.assertEqual(self.mode_file.read_text(), text)

    def test_under_systemd_only_warnings_reach_the_journal(self) -> None:
        # LogLevelMax=notice drops the routine line (info); a failback is a warning.
        self.assertEqual(unit("aiswe-gpu-failback.service")["Service"]["LogLevelMax"], "notice")
        self.env["JOURNAL_STREAM"] = "8:12345"
        self.assertTrue(self.run_wrapper("run").stdout.startswith("<6>gpu-failback: VM mode"))
        self.mode_file.write_text(json.dumps({"mode": "gpu", "gpuUrl": "http://127.0.0.1:9199",
                                              "expiresAt": "2026-01-01T00:00:00Z"}))
        self.assertIn("<4>gpu-failback: DRY RUN", self.run_wrapper("run", "--dry-run").stderr)


if __name__ == "__main__":
    unittest.main()
