"""The backend's benchmark client against the benchmark service's real ASGI app.

Same design as `test_finetuned_local_service_contract.py`: one patch of
`httpx.AsyncClient` dispatches a request to the service's port into its
`TestClient` and a request to Ollama's port into a fake. The call under test
is `generate_benchmark_answer` -> `benchmark_service.py` -> (fake) Ollama,
with no socket anywhere. What is pinned is the seam: the options the service
reports are the backend's grounded options, the prompt reaches Ollama as one
user turn byte for byte, and the echoes the backend checks are the ones the
service sends.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from fastapi.testclient import TestClient

from app.grounded_generation import grounded_options
from app.research_benchmark_client import (
    BenchmarkConditionError,
    generate_benchmark_answer,
    prompt_sha256,
)

SERVICE_DIR = Path(__file__).resolve().parents[2] / "training" / "inference_service"


def _load_service():
    if str(SERVICE_DIR) not in sys.path:
        sys.path.append(str(SERVICE_DIR))
    spec = importlib.util.spec_from_file_location(
        "local_benchmark_service", SERVICE_DIR / "benchmark_service.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


service = _load_service()

BASE_TAG = "llama3.2:3b"
V2_TAG = "css360-ft-v2:latest"
#: Full manifest digests as `/api/tags` reports them; their first twelve
#: characters are what the lineage record holds.
BASE_DIGEST = "a80c4f17acd5" + "1" * 52
V2_DIGEST = "dea74c57f25a" + "2" * 52
MAPPING = f"base={BASE_TAG},v2=css360-ft-v2"
PROMPT = "You are answering a student's question about their course.\n\nStudent question:\nWhen?\n\nAnswer:"


class _Response:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> Any:
        return self._payload


class _FakeOllama:
    def __init__(self) -> None:
        self.chat_requests: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> "_FakeOllama":
        return self

    async def __aenter__(self) -> "_FakeOllama":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def get(self, url: str, **kwargs: Any) -> _Response:
        return _Response(200, {"models": [
            {"name": V2_TAG, "digest": V2_DIGEST},
            {"name": BASE_TAG, "digest": BASE_DIGEST},
        ]})

    async def post(self, url: str, *, json: Any = None, **kwargs: Any) -> _Response:
        self.chat_requests.append(json)
        return _Response(200, {
            "model": json["model"],
            "message": {"role": "assistant", "content": "Tuesdays at 2pm."},
            "done": True,
            "done_reason": "stop",
            "total_duration": 4_000_000,
            "load_duration": 1_000_000,
            "prompt_eval_count": 120,
            "eval_count": 8,
            "eval_duration": 2_000_000,
        })


class _Network:
    def __init__(self, wrapper: TestClient, ollama: _FakeOllama) -> None:
        self.wrapper = wrapper
        self.ollama = ollama

    def __call__(self, *args: Any, **kwargs: Any) -> "_Network":
        return self

    async def __aenter__(self) -> "_Network":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    @staticmethod
    def _is_wrapper(url: str) -> bool:
        return ":9002/" in url

    async def get(self, url: str, **kwargs: Any) -> Any:
        if self._is_wrapper(url):
            return self.wrapper.get(url.split(":9002", 1)[1])
        return await self.ollama.get(url, **kwargs)

    async def post(self, url: str, *, json: Any = None, **kwargs: Any) -> Any:
        if self._is_wrapper(url):
            return self.wrapper.post(url.split(":9002", 1)[1], json=json)
        return await self.ollama.post(url, json=json, **kwargs)


class BenchmarkServiceContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._env = mock.patch.dict(os.environ, {
            service.MODEL_MAP_ENV: MAPPING,
            "CSS360_BENCHMARK_SERVICE_URL": "http://127.0.0.1:9002",
            "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
            "BENCHMARK_INFERENCE_PORT": "",
            "CSS360_BENCHMARK_KEEP_ALIVE": "",
            # Set for the production service; the benchmark service must ignore it.
            "FINETUNED_NUM_CTX": "8192",
        })
        self._env.start()
        self.addCleanup(self._env.stop)
        self.wrapper = TestClient(service.app)
        self.ollama = _FakeOllama()
        self._network = mock.patch.object(service.httpx, "AsyncClient", _Network(self.wrapper, self.ollama))
        self._network.start()
        self.addCleanup(self._network.stop)

    async def test_end_to_end_for_an_adapter_alias(self) -> None:
        result = await generate_benchmark_answer("v2", PROMPT, timeout=30.0)

        self.assertEqual(result["alias"], "v2")
        self.assertEqual(result["model"], V2_TAG)
        self.assertEqual(result["modelDigest"], V2_DIGEST)
        self.assertEqual(result["answer"], "Tuesdays at 2pm.")
        self.assertEqual(result["promptSha256"], prompt_sha256(PROMPT))
        self.assertEqual(result["options"], grounded_options())
        self.assertEqual(result["ollama"], {
            "totalDurationNs": 4_000_000, "loadDurationNs": 1_000_000, "promptEvalCount": 120,
            "evalCount": 8, "evalDurationNs": 2_000_000, "doneReason": "stop",
        })
        self.assertIsInstance(result["generationSeconds"], float)

        (sent,) = self.ollama.chat_requests
        self.assertEqual(sent["model"], V2_TAG)
        self.assertEqual(sent["messages"], [{"role": "user", "content": PROMPT}])
        self.assertIs(sent["stream"], False)
        self.assertEqual(sent["options"], grounded_options())
        self.assertEqual(sent["options"]["num_ctx"], 4096)
        self.assertNotIn("keep_alive", sent)

    async def test_the_control_alias_is_the_base_model(self) -> None:
        result = await generate_benchmark_answer("base", PROMPT, timeout=30.0)
        self.assertEqual(result["model"], BASE_TAG)
        self.assertEqual(result["modelDigest"], BASE_DIGEST)
        self.assertEqual(self.ollama.chat_requests[0]["model"], BASE_TAG)

    async def test_an_unmapped_alias_is_alias_not_mapped_and_ollama_is_never_asked(self) -> None:
        with self.assertRaises(BenchmarkConditionError) as caught:
            await generate_benchmark_answer("v4_vm", PROMPT, timeout=30.0)
        self.assertEqual(caught.exception.code, "alias_not_mapped")
        self.assertEqual(self.ollama.chat_requests, [])

    def test_the_service_health_lists_every_alias(self) -> None:
        body = self.wrapper.get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["engine"], "ollama")
        self.assertEqual([row["alias"] for row in body["aliases"]], list(service.ALIASES))
        by_alias = {row["alias"]: row for row in body["aliases"]}
        self.assertEqual(by_alias["v2"], {"alias": "v2", "ollamaModel": V2_TAG, "mapped": True,
                                          "available": True, "digest": V2_DIGEST})
        self.assertEqual(by_alias["v4_vm"], {"alias": "v4_vm", "ollamaModel": None, "mapped": False,
                                             "available": False, "digest": None})
        self.assertEqual(body["servable"], ["base", "v2"])


if __name__ == "__main__":
    unittest.main()
