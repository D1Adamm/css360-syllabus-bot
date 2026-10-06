"""scripts/classroom_gpu_mode.py: switching only after every check, and switching back on failure."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from app import generation_mode as gm

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("classroom_gpu_mode", REPO_ROOT / "scripts" / "classroom_gpu_mode.py")
script = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = script
_spec.loader.exec_module(script)
checks = script.checks

COURSE = "css360e-autumn-2026-c08m"
GPU = "http://127.0.0.1:9101"


def decoding(cap: int) -> dict:
    return script.service_helpers.decoding_summary(cap)


class FakeWorld:
    """The backend, the GPU service and the VM's Ollama, as one fake transport."""

    def __init__(self, *, ft_on_gpu: bool = True, gpu_versions=("v1",), gpu_prompt_tokens: int = 31) -> None:
        self.ft_on_gpu = ft_on_gpu
        self.gpu_versions = list(gpu_versions)
        self.gpu_prompt_tokens = gpu_prompt_tokens
        self.served = {"base": 0, "course": 0}

    def _r(self, status, body):
        return checks.HttpResult(status, body, 0.2)

    def request(self, method, url, *, body=None, headers=None, timeout=30.0):
        mode = gm.current_mode()
        if url.endswith("/api/health"):
            return self._r(200, {"status": "ok"})
        if url.endswith("/api/admin/generation-mode"):
            return self._r(200, gm.describe(mode, ollama_url="o", ollama_model="llama3.2:3b", finetuned_url="f"))
        if url == GPU + "/health":
            return self._r(200, {"status": "ok", "adapterLoaded": True, "cudaAvailable": True, "targets": ["course", "base"],
                                 "decoding": decoding(256), "secondsRemaining": 7200, "gpuName": "H100",
                                 "courses": [{"courseId": COURSE, "versions": self.gpu_versions}], "served": dict(self.served)})
        if url == GPU + "/generate":
            target = body.get("target", "course")
            self.served[target] += 1
            return self._r(200, {"answer": "ok", "target": target, "courseId": body.get("courseId"),
                                 "modelVersion": body.get("modelVersion"), "generationSeconds": 0.5,
                                 "decoding": decoding(body["maxNewTokens"]),
                                 "timings": {"prompt_tokens": self.gpu_prompt_tokens, "output_tokens": 40}})
        if url.endswith("/api/chat"):  # the VM's Ollama, for the token-count comparison
            return self._r(200, {"prompt_eval_count": 31})
        for condition, route in checks.MODE_ROUTES.items():
            if url.endswith(route):
                on_gpu = mode.is_gpu and (condition in ("base", "rag") or self.ft_on_gpu)
                if on_gpu:
                    self.served["base" if condition in ("base", "rag") else "course"] += 1
                model = (f"m + {COURSE}@v1 (Tillicum GPU)" if condition.startswith("fine") else "m (Tillicum GPU)") if on_gpu else "llama3.2:3b"
                return self._r(200, {"answer": "An answer.", "courseId": COURSE, "model": model, "modelVersion": "v1"})
        return self._r(404, None)


class SwitchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.mode_file = Path(self.tmp.name) / "generation-mode.json"
        # The failback watchdog's state, which `status` reads, is kept out of the real one too.
        self.env = patch.dict(os.environ, {gm.MODE_FILE_ENV: str(self.mode_file),
                                           script.failback.STATE_DIR_ENV: str(Path(self.tmp.name) / "watchdog")})
        self.env.start()
        gm.reset_for_tests()

    def tearDown(self) -> None:
        self.env.stop()
        self.tmp.cleanup()
        gm.reset_for_tests()

    def run_script(self, argv, world) -> tuple[int, str]:
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(out):
            code = script.main(argv, transport=world)
        return code, out.getvalue()

    START = ["start", "--course", COURSE, "--version", "v1", "--node", "g014", "--job-id", "123", "--no-tunnel", "--no-auth"]

    def test_start_switches_only_after_every_check_and_proves_the_gpu(self) -> None:
        world = FakeWorld()
        code, out = self.run_script(self.START, world)
        self.assertEqual(code, 0, out)
        mode = gm.parse_mode(self.mode_file.read_text())
        self.assertTrue(mode.is_gpu)
        self.assertEqual(mode.gpu_url, GPU)
        self.assertEqual(mode.details["jobId"], "123")
        self.assertIn("all four conditions answered on the GPU", out)
        self.assertEqual(oct(self.mode_file.stat().st_mode & 0o777), "0o600")

    def test_a_version_not_published_on_the_gpu_changes_nothing(self) -> None:
        code, out = self.run_script(self.START, FakeWorld(gpu_versions=("v2",)))
        self.assertEqual(code, 1)
        self.assertFalse(self.mode_file.exists())
        self.assertIn("not published on Tillicum", out)

    def test_a_prompt_format_mismatch_changes_nothing(self) -> None:
        code, out = self.run_script(self.START, FakeWorld(gpu_prompt_tokens=37))
        self.assertEqual(code, 1)
        self.assertFalse(self.mode_file.exists())
        self.assertIn("prompt format mismatch", out)

    def test_a_failure_after_the_switch_puts_the_vm_back(self) -> None:
        # The backend's fine-tuned answers do not come from the GPU: start must undo itself.
        code, out = self.run_script(self.START, FakeWorld(ft_on_gpu=False))
        self.assertEqual(code, 1)
        self.assertFalse(gm.parse_mode(self.mode_file.read_text()).is_gpu)
        self.assertIn("restored the previous mode (VM)", out)

    def test_start_refuses_when_already_in_gpu_mode(self) -> None:
        self.mode_file.write_text(json.dumps({"mode": "gpu", "gpuUrl": GPU}))
        code, out = self.run_script(self.START, FakeWorld())
        self.assertEqual(code, 1)
        self.assertIn("already in GPU mode", out)

    def test_status_names_the_engine_for_each_condition(self) -> None:
        self.mode_file.write_text(json.dumps({"mode": "gpu", "gpuUrl": GPU, "jobId": "123"}))
        with patch.object(script, "tunnel_alive", return_value=True):
            code, out = self.run_script(["status"], FakeWorld())
        self.assertEqual(code, 0, out)
        self.assertIn("MODE: CLASSROOM GPU", out)
        self.assertEqual(out.count("Tillicum GPU via"), 4)

    def test_status_warns_when_gpu_mode_has_no_gpu(self) -> None:
        self.mode_file.write_text(json.dumps({"mode": "gpu", "gpuUrl": "http://127.0.0.1:9199"}))
        code, out = self.run_script(["status", "--gpu-port", "9199"], FakeWorld())
        self.assertEqual(code, 1)
        self.assertIn("GPU SERVICE IS UNREACHABLE", out)

    def test_an_invalid_mode_file_is_reported_as_vm(self) -> None:
        self.mode_file.write_text("{oops")
        code, out = self.run_script(["status"], FakeWorld())
        self.assertIn("MODE: VM (the file is invalid", out)


if __name__ == "__main__":
    unittest.main()
