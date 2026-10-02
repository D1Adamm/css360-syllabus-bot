"""Classroom GPU mode: choosing the engine, and holding GPU answers to production settings."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import generation_mode as gm
from app.generation_queue import GenerationQueue, get_generation_queue, reset_generation_queue_for_tests
from app.gpu_generation import check_decoding, expected_decoding
from app.grounded_generation import classroom_options, grounded_options

GPU_URL = "http://127.0.0.1:9101"


def gpu_decoding(num_predict: int) -> dict:
    """What an up-to-date GPU service reports (training/inference_service/helpers.decoding_summary)."""
    return {
        "num_predict": num_predict, "temperature": 0, "repeat_penalty": 1.05, "repeat_last_n": 4096,
        "seed": 360, "num_ctx": 4096, "promptFormat": "ollama-llama3.2-chat",
        "stop": ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
    }


class ModeFileMixin:
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "generation-mode.json"
        self.env = patch.dict(os.environ, {gm.MODE_FILE_ENV: str(self.path)})
        self.env.start()
        gm.reset_for_tests()
        reset_generation_queue_for_tests(GenerationQueue(max_concurrency=1, max_waiting=10, queue_timeout=5))

    def tearDown(self) -> None:
        self.env.stop()
        self.tmp.cleanup()
        gm.reset_for_tests()
        reset_generation_queue_for_tests()

    def write(self, data) -> None:
        text = data if isinstance(data, str) else json.dumps(data)
        self.path.write_text(text, encoding="utf-8")
        # A rewrite within the same mtime tick must still be seen.
        stat = self.path.stat()
        os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))


class ModeSelectionTests(ModeFileMixin, unittest.TestCase):
    def test_no_file_means_vm(self) -> None:
        self.assertFalse(gm.current_mode().is_gpu)

    def test_gpu_mode_and_back(self) -> None:
        self.write({"mode": "gpu", "gpuUrl": GPU_URL + "/", "node": "g014", "jobId": "123"})
        mode = gm.current_mode()
        self.assertTrue(mode.is_gpu)
        self.assertEqual(mode.gpu_url, GPU_URL)
        self.assertEqual(mode.details, {"node": "g014", "jobId": "123"})
        self.write({"mode": "vm"})
        self.assertFalse(gm.current_mode().is_gpu)

    def test_anything_unclear_generates_on_the_vm_and_says_why(self) -> None:
        for bad in ("{not json", json.dumps({"mode": "gpu"}), json.dumps({"mode": "fast"}),
                    json.dumps({"mode": "gpu", "gpuUrl": "http://10.0.0.5:9101"}),
                    json.dumps({"mode": "gpu", "gpuUrl": "https://127.0.0.1:9101"}),
                    json.dumps({"mode": "gpu", "gpuUrl": "http://127.0.0.1:9101/generate"})):
            with self.subTest(bad=bad):
                self.write(bad)
                with self.assertLogs("app.generation_mode", level="ERROR"):
                    mode = gm.current_mode()
                self.assertFalse(mode.is_gpu)
                self.assertTrue(mode.error)

    def test_a_mode_change_restarts_the_queue_wait_estimate(self) -> None:
        import app.main  # noqa: F401 - registers the listener

        get_generation_queue()._record_service(10.0)
        self.write({"mode": "gpu", "gpuUrl": GPU_URL})
        gm.current_mode()
        self.assertIsNone(get_generation_queue()._service_ewma)

    def test_tests_never_read_the_default_file(self) -> None:
        with patch.dict(os.environ, {gm.MODE_FILE_ENV: ""}):
            self.assertNotEqual(gm.mode_file_path(), gm.DEFAULT_MODE_FILE)


class DecodingCheckTests(unittest.TestCase):
    def test_production_settings_pass(self) -> None:
        check_decoding(gpu_decoding(128), expected_decoding(classroom_options()))
        check_decoding(gpu_decoding(256), expected_decoding(grounded_options()))
        check_decoding(gpu_decoding(128), expected_decoding(max_new_tokens=128))

    def test_any_difference_is_refused(self) -> None:
        for key, value in (("num_predict", 256), ("repeat_penalty", 1.1), ("num_ctx", 8192),
                           ("temperature", 0.7), ("promptFormat", "hf-chat-template")):
            with self.subTest(key=key):
                reported = {**gpu_decoding(128), key: value}
                with self.assertRaises(HTTPException) as caught:
                    check_decoding(reported, expected_decoding(classroom_options()))
                self.assertEqual(caught.exception.status_code, 502)

    def test_an_old_service_that_reports_nothing_is_refused(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            check_decoding(None, expected_decoding(classroom_options()))
        self.assertIn("older build", caught.exception.detail)


def _client_returning(payload: dict, seen: list):
    async def post(url, json):
        seen.append((url, json))
        response = type("R", (), {"status_code": 200, "text": ""})()
        response.json = lambda: payload
        return response

    client = AsyncMock()
    client.post = AsyncMock(side_effect=post)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


class RoutingTests(ModeFileMixin, unittest.IsolatedAsyncioTestCase):
    async def test_base_and_rag_go_to_the_gpu_with_the_same_prompt_and_cap(self) -> None:
        from app.ollama import generate_ollama_chat

        self.write({"mode": "gpu", "gpuUrl": GPU_URL})
        seen: list = []
        reply = {"answer": "GPU answer.", "model": "meta-llama/Llama-3.2-3B-Instruct", "target": "base",
                 "decoding": gpu_decoding(128), "timings": {"prompt_tokens": 900, "output_tokens": 40}}
        with patch("app.gpu_generation.httpx.AsyncClient", return_value=_client_returning(reply, seen)):
            with self.assertLogs("app.generation_queue", level="INFO") as logs:
                result = await generate_ollama_chat("THE PROMPT", options=classroom_options(), condition="rag")
        self.assertEqual(seen, [(GPU_URL + "/generate", {"question": "THE PROMPT", "target": "base", "maxNewTokens": 128})])
        self.assertEqual(result["answer"], "GPU answer.")
        self.assertIn("Tillicum GPU", result["model"])
        self.assertIn("engine=gpu", logs.output[0])
        self.assertIn("prompt_tokens=900", logs.output[0])

    async def test_vm_mode_still_asks_the_vm_ollama(self) -> None:
        from app.ollama import OLLAMA_BASE_URL, generate_ollama_chat

        seen: list = []
        with patch("app.ollama.httpx.AsyncClient", return_value=_client_returning({"message": {"content": "VM."}}, seen)):
            result = await generate_ollama_chat("P", options=classroom_options(), condition="base")
        self.assertEqual(seen[0][0], f"{OLLAMA_BASE_URL}/api/chat")
        self.assertEqual(result["answer"], "VM.")

    async def test_a_gpu_answer_with_the_wrong_cap_is_not_shown(self) -> None:
        from app.ollama import generate_ollama_chat

        self.write({"mode": "gpu", "gpuUrl": GPU_URL})
        reply = {"answer": "Long.", "model": "m", "target": "base", "decoding": gpu_decoding(256), "timings": {}}
        with patch("app.gpu_generation.httpx.AsyncClient", return_value=_client_returning(reply, [])):
            with self.assertRaises(HTTPException) as caught:
                await generate_ollama_chat("P", options=classroom_options(), condition="base")
        self.assertEqual(caught.exception.status_code, 502)

    async def test_fine_tuned_goes_to_the_gpu_with_the_course_and_version(self) -> None:
        from app.finetuned_client import generate_finetuned_response

        self.write({"mode": "gpu", "gpuUrl": GPU_URL})
        seen: list = []
        reply = {"answer": "FT.", "model": "meta-llama/Llama-3.2-3B-Instruct", "adapterLoaded": True,
                 "courseId": "css360e-autumn-2026-c08m", "modelVersion": "v1", "generationSeconds": 1.2,
                 "target": "course", "decoding": gpu_decoding(128), "timings": {"output_tokens": 50}}
        with patch.dict(os.environ, {"FINETUNED_SERVICE_URL": "http://127.0.0.1:9001"}), \
                patch("app.finetuned_client.httpx.AsyncClient", return_value=_client_returning(reply, seen)):
            result = await generate_finetuned_response(
                "Q?", course_id="css360e-autumn-2026-c08m", model_version="v1", max_new_tokens=128, condition="fineTuned"
            )
        self.assertEqual(seen[0][0], GPU_URL + "/generate")
        self.assertEqual(seen[0][1], {"question": "Q?", "courseId": "css360e-autumn-2026-c08m", "modelVersion": "v1", "maxNewTokens": 128})
        self.assertEqual(result["model_version"], "v1")
        self.assertIn("css360e-autumn-2026-c08m@v1 (Tillicum GPU)", result["model"])

    async def test_fine_tuned_from_an_old_gpu_build_is_refused(self) -> None:
        from app.finetuned_client import generate_finetuned_response

        self.write({"mode": "gpu", "gpuUrl": GPU_URL})
        reply = {"answer": "Rambling.", "model": "m", "adapterLoaded": True, "courseId": "c-1",
                 "modelVersion": "v1", "generationSeconds": 9.0}
        with patch.dict(os.environ, {"FINETUNED_SERVICE_URL": "http://127.0.0.1:9001"}), \
                patch("app.finetuned_client.httpx.AsyncClient", return_value=_client_returning(reply, [])):
            with self.assertRaises(HTTPException) as caught:
                await generate_finetuned_response("Q?", course_id="c-1", model_version="v1", max_new_tokens=128)
        self.assertEqual(caught.exception.status_code, 502)

    async def test_fine_tuned_in_vm_mode_uses_the_vm_service_unchecked(self) -> None:
        from app.finetuned_client import generate_finetuned_response

        seen: list = []
        reply = {"answer": "VM FT.", "model": "css360e-v1:latest", "adapterLoaded": True, "courseId": "c-1",
                 "modelVersion": "v1", "generationSeconds": 2.0}
        with patch.dict(os.environ, {"FINETUNED_SERVICE_URL": "http://127.0.0.1:9001"}), \
                patch("app.finetuned_client.httpx.AsyncClient", return_value=_client_returning(reply, seen)):
            result = await generate_finetuned_response("Q?", course_id="c-1", model_version="v1", max_new_tokens=128)
        self.assertEqual(seen[0][0], "http://127.0.0.1:9001/generate")
        self.assertEqual(result["model"], "css360e-v1:latest")


class GpuServiceContractTests(unittest.TestCase):
    """The GPU service's own settings are production's, from its source, not a copy."""

    def test_the_gpu_service_reports_exactly_what_the_backend_expects(self) -> None:
        import importlib.util
        import sys

        service_dir = Path(__file__).resolve().parents[2] / "training" / "inference_service"
        spec = importlib.util.spec_from_file_location("tillicum_service_helpers", service_dir / "helpers.py")
        helpers = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = helpers
        spec.loader.exec_module(helpers)

        for cap, options in ((128, classroom_options()), (256, grounded_options())):
            with self.subTest(cap=cap):
                check_decoding(helpers.decoding_summary(cap), expected_decoding(options))
        # The prompt the GPU renders is the one the VM's /api/chat renders.
        self.assertEqual(
            helpers.render_production_prompt("Q?"),
            "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n\n<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\nQ?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        )


class StatusRouteTests(ModeFileMixin, unittest.TestCase):
    def test_admin_status_reports_mode_and_routes(self) -> None:
        from app.auth.dependencies import current_principal
        from app.auth.principal import Principal, StaffUser
        from app.main import app

        admin = Principal(user=StaffUser(user_id="u", email="a@uw.edu", display_name="A", role="admin"))
        app.dependency_overrides[current_principal] = lambda: admin
        self.addCleanup(app.dependency_overrides.pop, current_principal, None)
        self.write({"mode": "gpu", "gpuUrl": GPU_URL, "node": "g014"})
        body = TestClient(app).get("/api/admin/generation-mode").json()
        self.assertEqual(body["mode"], "gpu")
        self.assertEqual(set(body["routes"]), {"base", "rag", "fineTuned", "fineTunedRag"})
        self.assertTrue(all("Tillicum GPU" in route for route in body["routes"].values()))
        self.assertEqual(body["details"], {"node": "g014"})


if __name__ == "__main__":
    unittest.main()
