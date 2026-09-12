"""The local Ollama-backed inference service satisfies the backend's contract.

`training/inference_service/ollama_service.py` answers `/health` and
`/generate` on the UWB VM so the fine-tuned paths no longer need a Tillicum
session. The backend client was not changed for it, and this is the test that
says it did not need to be: the wrapper's actual responses go through
`finetuned_client`'s own validation, and the whole client call is driven
end to end with the wrapper standing in for the network.

The wrapper's own behaviour — mapping, isolation, Ollama failure codes — is
tested beside it in `training/inference_service/test_ollama_service.py`. What
is pinned here is only the seam between the two.
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

from app.finetuned_client import (
    _validate_generate_payload,
    check_finetuned_service_health,
    generate_finetuned_response,
    public_service_health,
)
from app.schemas import FineTunedHealthResponse

SERVICE_DIR = Path(__file__).resolve().parents[2] / "training" / "inference_service"

COURSE = "css-360-winter-2026-a7rp"
OTHER_COURSE = "css-350-spring-2026-n3h9"
MODEL = "css360-ft-v2:latest"
MAPPING = f"{COURSE}@v2={MODEL}"


def _load_service():
    # Appended, not prepended: that directory also holds `app.py`, and the
    # backend's `app` package must keep winning. `helpers` has no backend
    # namesake, so it resolves from the appended path.
    if str(SERVICE_DIR) not in sys.path:
        sys.path.append(str(SERVICE_DIR))
    spec = importlib.util.spec_from_file_location(
        "local_ollama_service", SERVICE_DIR / "ollama_service.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: the module uses `from __future__ import
    # annotations`, and pydantic resolves those strings through
    # `sys.modules[cls.__module__]`. An unregistered module leaves every
    # request model "not fully defined".
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


service = _load_service()


class _Response:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> Any:
        return self._payload


class _FakeOllama:
    """`httpx.AsyncClient` as seen from inside the wrapper."""

    def __init__(self, answer: str = "Late work loses 10% per day.") -> None:
        self.answer = answer

    def __call__(self, *args: Any, **kwargs: Any) -> "_FakeOllama":
        return self

    async def __aenter__(self) -> "_FakeOllama":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def get(self, url: str, **kwargs: Any) -> _Response:
        return _Response(200, {"models": [{"name": MODEL}, {"name": "llama3.2:3b"}]})

    async def post(self, url: str, *, json: Any = None, **kwargs: Any) -> _Response:
        return _Response(
            200,
            {
                "model": json["model"],
                "message": {"role": "assistant", "content": self.answer},
                "done": True,
            },
        )


class _Network:
    """`httpx.AsyncClient` for both sides at once.

    The backend client and the wrapper share the one `httpx` module, so one
    patch of `httpx.AsyncClient` is seen by both. This fake dispatches on the
    URL: a request to the wrapper's port goes into its ASGI app through the
    `TestClient`, and a request to Ollama's port is answered by `_FakeOllama`.
    The call under test is therefore `generate_finetuned_response` -> wrapper
    -> (fake) Ollama, with no real socket anywhere.
    """

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
        return ":9001/" in url

    async def get(self, url: str, **kwargs: Any) -> Any:
        if self._is_wrapper(url):
            return self.wrapper.get(url.split(":9001", 1)[1])
        return await self.ollama.get(url, **kwargs)

    async def post(self, url: str, *, json: Any = None, **kwargs: Any) -> Any:
        if self._is_wrapper(url):
            return self.wrapper.post(url.split(":9001", 1)[1], json=json)
        return await self.ollama.post(url, json=json, **kwargs)


class LocalServiceContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._env = mock.patch.dict(
            os.environ,
            {
                service.MODEL_MAP_ENV: MAPPING,
                "INFERENCE_PORT": "",
                "FINETUNED_SERVICE_URL": "http://127.0.0.1:9001",
                "FINETUNED_SERVICE_TIMEOUT_SECONDS": "",
            },
        )
        self._env.start()
        self.addCleanup(self._env.stop)
        self.wrapper = TestClient(service.app)
        # One patch, seen by the backend client and the wrapper alike.
        self._network = mock.patch.object(
            service.httpx, "AsyncClient", _Network(self.wrapper, _FakeOllama())
        )
        self._network.start()
        self.addCleanup(self._network.stop)

    def test_a_generate_response_passes_the_backend_validator(self) -> None:
        raw = self.wrapper.post(
            "/generate",
            json={"courseId": COURSE, "modelVersion": "v2", "question": "Late policy?"},
        )
        self.assertEqual(raw.status_code, 200)

        validated = _validate_generate_payload(raw.json(), expected_course_id=COURSE)

        self.assertEqual(validated["course_id"], COURSE)
        self.assertEqual(validated["model_version"], "v2")
        self.assertTrue(validated["adapter_loaded"])
        self.assertEqual(validated["model"], MODEL)
        self.assertIsInstance(validated["generation_seconds"], float)

    def test_the_backend_refuses_the_wrapper_if_it_answered_for_another_course(self) -> None:
        """The wrapper echoes the course it served; the backend compares. Both
        halves of the isolation check, exercised against the real payload."""
        raw = self.wrapper.post(
            "/generate",
            json={"courseId": COURSE, "modelVersion": "v2", "question": "Late policy?"},
        ).json()

        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as ctx:
            _validate_generate_payload(raw, expected_course_id=OTHER_COURSE)
        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn(COURSE, ctx.exception.detail)

    async def test_the_backend_client_end_to_end_through_the_wrapper(self) -> None:
        result = await generate_finetuned_response(
            "What is the late policy?", course_id=COURSE, model_version="v2"
        )

        self.assertEqual(result["answer"], "Late work loses 10% per day.")
        self.assertEqual(result["course_id"], COURSE)
        self.assertEqual(result["model_version"], "v2")
        self.assertEqual(result["model"], MODEL)
        self.assertTrue(result["adapter_loaded"])
        self.assertEqual(result["response_type"], "fineTuned")

    async def test_an_unmapped_course_reaches_the_backend_as_a_409(self) -> None:
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as ctx:
            await generate_finetuned_response(
                "What is the late policy?", course_id=OTHER_COURSE, model_version="v1"
            )
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn(OTHER_COURSE, ctx.exception.detail)

    async def test_health_is_readable_by_the_backend_and_safe_for_the_browser(self) -> None:
        probe = await check_finetuned_service_health()

        self.assertEqual(probe["status"], "ok")
        self.assertTrue(probe["adapterLoaded"])
        self.assertEqual(
            probe["courses"],
            [{"courseId": COURSE, "versions": ["v2"], "currentVersion": "v2"}],
        )
        self.assertIsNone(probe["secondsRemaining"])

        public = public_service_health(probe)
        self.assertNotIn("hostname", public)
        self.assertNotIn("port", public)
        self.assertNotIn("serviceUrl", public)
        # The wrapper's operator-only fields never reach the response model.
        self.assertNotIn("ollamaUrl", public)
        self.assertNotIn("models", public)
        parsed = FineTunedHealthResponse(**public)
        self.assertEqual(parsed.courses[0]["courseId"], COURSE)


V3_MODEL = "css360-cpu-v3-test:latest"
BOTH_MAPPED = f"{COURSE}@v2={MODEL},{COURSE}@v3={V3_MODEL}"


class _FakeOllamaWithBoth(_FakeOllama):
    """Ollama holding both course models, as the VM does while v3 is under test."""

    async def get(self, url: str, **kwargs: Any) -> _Response:
        return _Response(
            200,
            {"models": [{"name": MODEL}, {"name": V3_MODEL}, {"name": "llama3.2:3b"}]},
        )


class ExplicitVersionContractTests(unittest.IsolatedAsyncioTestCase):
    """Two versions mapped for one course: the model-testing situation.

    The wrapper's `/health` calls the highest servable version `currentVersion`
    — its own notion, which the backend never reads to resolve anything. What
    the backend sends is honoured exactly: an explicit v2 is answered by the
    v2 model although v3 is higher, an explicit v3 by the v3 model, and a
    version this host does not have is a refusal that names it, not an answer
    from the nearest one. The version the wrapper echoes is the one the
    backend checks.
    """

    def setUp(self) -> None:
        self._env = mock.patch.dict(
            os.environ,
            {
                service.MODEL_MAP_ENV: BOTH_MAPPED,
                "INFERENCE_PORT": "",
                "FINETUNED_SERVICE_URL": "http://127.0.0.1:9001",
                "FINETUNED_SERVICE_TIMEOUT_SECONDS": "",
            },
        )
        self._env.start()
        self.addCleanup(self._env.stop)
        self.wrapper = TestClient(service.app)
        self._network = mock.patch.object(
            service.httpx, "AsyncClient", _Network(self.wrapper, _FakeOllamaWithBoth())
        )
        self._network.start()
        self.addCleanup(self._network.stop)

    async def test_health_calls_the_highest_servable_version_current_and_the_backend_ignores_it(self) -> None:
        probe = await check_finetuned_service_health()
        self.assertEqual(
            probe["courses"],
            [{"courseId": COURSE, "versions": ["v2", "v3"], "currentVersion": "v3"}],
        )

    async def test_an_explicit_v2_is_answered_by_v2_although_v3_is_higher(self) -> None:
        result = await generate_finetuned_response(
            "What is the late policy?", course_id=COURSE, model_version="v2"
        )
        self.assertEqual(result["model_version"], "v2")
        self.assertEqual(result["model"], MODEL)

    async def test_an_explicit_v3_is_answered_by_v3(self) -> None:
        result = await generate_finetuned_response(
            "What is the late policy?", course_id=COURSE, model_version="v3"
        )
        self.assertEqual(result["model_version"], "v3")
        self.assertEqual(result["model"], V3_MODEL)

    async def test_a_version_this_host_does_not_have_is_refused_not_substituted(self) -> None:
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as ctx:
            await generate_finetuned_response(
                "What is the late policy?", course_id=COURSE, model_version="v4"
            )
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn(COURSE, ctx.exception.detail)
        self.assertIn('version "v4"', ctx.exception.detail)

    def test_the_wrapper_echoes_the_version_it_served_and_the_backend_compares(self) -> None:
        from fastapi import HTTPException

        raw = self.wrapper.post(
            "/generate",
            json={"courseId": COURSE, "modelVersion": "v3", "question": "Late policy?"},
        ).json()
        self.assertEqual(raw["modelVersion"], "v3")
        self.assertEqual(raw["model"], V3_MODEL)

        accepted = _validate_generate_payload(
            raw, expected_course_id=COURSE, expected_model_version="v3"
        )
        self.assertEqual(accepted["model_version"], "v3")
        with self.assertRaises(HTTPException) as ctx:
            _validate_generate_payload(raw, expected_course_id=COURSE, expected_model_version="v2")
        self.assertEqual(ctx.exception.status_code, 502)


class SharedDecodingTests(unittest.TestCase):
    def test_the_wrapper_decodes_with_the_backends_grounded_options(self) -> None:
        """The base model (backend, `generate_ollama_chat`) and the course
        adapter (this wrapper) must be decoded identically for the grounded
        comparison to be about weights. The wrapper builds its options from
        its own constants; this pins them to the backend's."""
        from app.grounded_generation import GROUNDED_OPTIONS

        with mock.patch.dict(os.environ, {"FINETUNED_NUM_CTX": ""}):
            self.assertEqual(service.build_generation_options(), dict(GROUNDED_OPTIONS))


class _CapturingClient:
    """`httpx.AsyncClient` that records the one payload the backend posts."""

    def __init__(self, sink: dict) -> None:
        self.sink = sink

    def __call__(self, *args: Any, **kwargs: Any) -> "_CapturingClient":
        return self

    async def __aenter__(self) -> "_CapturingClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def post(self, url: str, *, json: Any = None, **kwargs: Any) -> _Response:
        self.sink["url"] = url
        self.sink["json"] = json
        return _Response(200, {"model": json["model"], "message": {"role": "assistant", "content": "ok"}})


class RequestsDifferOnlyInModelTests(unittest.IsolatedAsyncioTestCase):
    """The base model and the course adapter are asked in the same words.

    The backend posts the grounded prompt to its Ollama for the base target;
    the wrapper posts the same prompt to the same Ollama for the adapter. The
    two JSON bodies must be identical except for `model` (and the wrapper's
    optional `keep_alive`, which affects residency, not output).
    """

    async def test_the_base_request_and_the_adapter_request_are_the_same_call(self) -> None:
        from app import ollama as backend_ollama
        from app.grounded_rag import BASE_TARGET, generate_from_prompt

        prompt = "PROMPT TEXT"
        captured: dict = {}
        with mock.patch.object(backend_ollama.httpx, "AsyncClient", _CapturingClient(captured)):
            await generate_from_prompt(prompt, course_id=COURSE, target=BASE_TARGET)
        with mock.patch.dict(os.environ, {"FINETUNED_KEEP_ALIVE": "", "FINETUNED_NUM_CTX": ""}):
            adapter_payload = service.build_chat_request(prompt, ollama_model=MODEL)

        base_payload = captured["json"]
        self.assertTrue(captured["url"].endswith("/api/chat"))
        self.assertEqual(base_payload["model"], backend_ollama.OLLAMA_MODEL)
        self.assertEqual(adapter_payload["model"], MODEL)
        strip = lambda payload: {k: v for k, v in payload.items() if k not in ("model", "keep_alive")}
        self.assertEqual(strip(base_payload), strip(adapter_payload))
        self.assertEqual(base_payload["messages"], [{"role": "user", "content": prompt}])
        self.assertEqual(base_payload["options"]["num_ctx"], 4096)
        self.assertEqual(base_payload["options"]["num_predict"], 256)
        self.assertEqual(base_payload["options"]["temperature"], 0)


if __name__ == "__main__":
    unittest.main()
