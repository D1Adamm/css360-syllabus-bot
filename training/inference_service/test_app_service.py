"""The Tillicum GPU service's classroom contract: production prompt, cap, stops, base target.

The engine itself needs CUDA; here a fake stands in for it, and the pure
helpers are tested directly. The engine's generation path was also run on CPU
with the real Llama 3 tokenizer and a tiny random model (prompt token counts
equal Ollama's: 31, 43 and 68 for the three probe prompts).
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, Optional
from unittest import mock

from fastapi.testclient import TestClient

import app as service
import helpers

COURSE = "css360e-autumn-2026-c08m"


class PromptAndSettingsTests(unittest.TestCase):
    def test_prompt_is_ollamas_llama32_rendering_without_bos(self) -> None:
        self.assertEqual(
            helpers.render_production_prompt("When is the final exam?"),
            "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n\n<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\nWhen is the final exam?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
        self.assertNotIn("Today Date", helpers.render_production_prompt("x"))

    def test_exactly_one_bos(self) -> None:
        self.assertEqual(helpers.with_single_bos([5, 6], 1), [1, 5, 6])
        self.assertEqual(helpers.with_single_bos([1, 5, 6], 1), [1, 5, 6])
        self.assertEqual(helpers.with_single_bos([1, 1, 5], 1), [1, 5])

    def test_output_cap(self) -> None:
        self.assertEqual(helpers.resolve_max_new_tokens(None), 256)
        self.assertEqual(helpers.resolve_max_new_tokens(128), 128)
        for bad in (0, 257):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                helpers.resolve_max_new_tokens(bad)

    def test_decoding_summary_is_the_vm_options(self) -> None:
        self.assertEqual(
            helpers.decoding_summary(128),
            {"num_predict": 128, "temperature": 0, "repeat_penalty": 1.05, "repeat_last_n": 4096,
             "seed": 360, "num_ctx": 4096, "promptFormat": "ollama-llama3.2-chat",
             "stop": ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]},
        )


class FakeEngine:
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    base_loaded = True
    model = object()

    def __init__(self) -> None:
        self.calls = []

    def generate(self, question: str, *, target: str, course_id: Optional[str], model_version: Optional[str],
                 max_new_tokens: Optional[int]) -> Dict[str, Any]:
        self.calls.append((question, target, course_id, model_version, max_new_tokens))
        cap = helpers.resolve_max_new_tokens(max_new_tokens)
        return {"answer": " An answer. ", "target": target, "courseId": course_id,
                "modelVersion": "v1" if target == "course" else None, "generationSeconds": 0.4,
                "decoding": helpers.decoding_summary(cap),
                "timings": {"prompt_tokens": 31, "output_tokens": 12, "eval_ms": 400, "ollama_total_ms": 400}}


class GenerateRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = FakeEngine()
        patcher = mock.patch.object(service, "ENGINE", self.engine)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(service.app)  # no lifespan: no model load

    def test_course_target_honours_the_classroom_cap_and_echoes_settings(self) -> None:
        body = self.client.post("/generate", json={"question": "Q?", "courseId": COURSE, "modelVersion": "v1",
                                                   "maxNewTokens": 128}).json()
        self.assertEqual(self.engine.calls, [("Q?", "course", COURSE, "v1", 128)])
        self.assertEqual(body["decoding"]["num_predict"], 128)
        self.assertEqual(body["target"], "course")
        self.assertEqual(body["courseId"], COURSE)
        self.assertEqual(body["timings"]["output_tokens"], 12)

    def test_base_target_needs_no_course(self) -> None:
        body = self.client.post("/generate", json={"question": "P", "target": "base", "maxNewTokens": 128}).json()
        self.assertEqual(self.engine.calls, [("P", "base", None, None, 128)])
        self.assertEqual(body["target"], "base")
        self.assertIsNone(body["modelVersion"])

    def test_course_target_without_a_course_is_422(self) -> None:
        response = self.client.post("/generate", json={"question": "Q?"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.engine.calls, [])

    def test_cap_above_the_production_maximum_is_422(self) -> None:
        response = self.client.post("/generate", json={"question": "Q?", "courseId": COURSE, "maxNewTokens": 4096})
        self.assertEqual(response.status_code, 422)

    def test_unknown_target_is_422(self) -> None:
        response = self.client.post("/generate", json={"question": "Q?", "target": "merged"})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
