"""The benchmark-service client: fixed codes, fixed messages, checked echoes.

Every failure the service can produce is a `BenchmarkConditionError` with a
stable code, so the route can record it per condition; every message is a
fixed string, so nothing the service or its Ollama wrote reaches a caller.
The alias and the prompt hash the service echoes are checked against what
was sent, because an answer from the wrong alias or the wrong prompt bytes
looks exactly like the right one.
"""

from __future__ import annotations

import json
import os
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from app.grounded_generation import grounded_options
from app.research_benchmark_client import (
    BenchmarkConditionError,
    _project_ollama_timings,
    generate_benchmark_answer,
    prompt_sha256,
)

SERVICE_URL = "http://127.0.0.1:9002"
PROMPT = "You are answering a student's question...\n\nStudent question:\nWhen?\n\nAnswer:"

#: What the service, or its Ollama, can put in a failure body.
POISONED_BODY = (
    '{"detail":{"code":"ollama_error","message":"Ollama at http://127.0.0.1:11434 returned '
    'HTTP 500 for css360-v4-test:latest: model blob /home/testuser/.ollama/models/blobs '
    'not readable; see /home/testuser/model_artifacts/css360-v4/Modelfile"}}'
)
LEAKS = ("127.0.0.1", "11434", "/home/testuser", ".ollama", "model_artifacts", "Modelfile",
         "blobs", "not readable")


def ok_payload(alias: str = "v2", prompt: str = PROMPT, **overrides: Any) -> dict[str, Any]:
    payload = {
        "alias": alias,
        "model": "css360-ft-v2:latest",
        "modelDigest": "dea74c57f25a" + "0" * 52,
        "answer": "Tuesdays at 2pm.",
        "promptSha256": prompt_sha256(prompt),
        "options": grounded_options(),
        "generationSeconds": 1.25,
        "ollama": {"totalDurationNs": 5, "evalCount": 9, "doneReason": "stop", "extra": 1},
    }
    payload.update(overrides)
    return payload


def _response(status_code: int, payload: Any = None, text: str | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    if payload is None:
        response.json.side_effect = ValueError("not json")
        response.text = text if text is not None else POISONED_BODY
    else:
        response.json.return_value = payload
        response.text = text if text is not None else json.dumps(payload)
    return response


def _client(response: MagicMock) -> MagicMock:
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


class ClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        env = patch.dict(os.environ, {"CSS360_BENCHMARK_SERVICE_URL": SERVICE_URL + "/"})
        env.start()
        self.addCleanup(env.stop)

    def _patch(self, response: MagicMock | None = None, *, error: BaseException | None = None) -> MagicMock:
        client = _client(response or _response(200, ok_payload()))
        if error is not None:
            client.post = AsyncMock(side_effect=error)
        factory = patch("app.research_benchmark_client.httpx.AsyncClient", return_value=client)
        mock = factory.start()
        self.addCleanup(factory.stop)
        self.factory = mock
        return client

    async def _expect(self, code: str, response: MagicMock | None = None, *,
                      error: BaseException | None = None, alias: str = "v2") -> BenchmarkConditionError:
        self._patch(response, error=error)
        with self.assertRaises(BenchmarkConditionError) as caught:
            await generate_benchmark_answer(alias, PROMPT, timeout=30.0)
        self.assertEqual(caught.exception.code, code)
        for leak in LEAKS:
            self.assertNotIn(leak, caught.exception.message)
        return caught.exception

    async def test_not_configured_is_a_condition_error_and_no_request(self) -> None:
        with patch.dict(os.environ, {"CSS360_BENCHMARK_SERVICE_URL": ""}):
            client = self._patch()
            with self.assertRaises(BenchmarkConditionError) as caught:
                await generate_benchmark_answer("v2", PROMPT, timeout=30.0)
        self.assertEqual(caught.exception.code, "service_not_configured")
        client.post.assert_not_awaited()

    async def test_a_good_answer_is_returned_with_the_exact_request_shape(self) -> None:
        client = self._patch()
        result = await generate_benchmark_answer("v2", PROMPT, timeout=42.0)

        client.post.assert_awaited_once_with(f"{SERVICE_URL}/generate", json={"alias": "v2", "prompt": PROMPT})
        self.assertEqual(self.factory.call_args.kwargs["timeout"], 42.0)
        self.assertEqual(result, {
            "alias": "v2",
            "model": "css360-ft-v2:latest",
            "modelDigest": "dea74c57f25a" + "0" * 52,
            "answer": "Tuesdays at 2pm.",
            "promptSha256": prompt_sha256(PROMPT),
            "options": grounded_options(),
            "generationSeconds": 1.25,
            "ollama": {"totalDurationNs": 5, "evalCount": 9, "doneReason": "stop"},
        })

    async def test_an_empty_answer_is_passed_through_for_the_route_to_classify(self) -> None:
        """The client keeps the service's timing intact; the route turns an
        answer with no text into a failed, unscored generation."""
        self._patch(_response(200, ok_payload(answer="")))
        result = await generate_benchmark_answer("v2", PROMPT, timeout=30.0)
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["generationSeconds"], 1.25)
        self.assertEqual(result["ollama"]["doneReason"], "stop")

    async def test_a_missing_or_malformed_digest_is_recorded_as_unknown(self) -> None:
        """The digest is a record, not a gate: an answer without one is kept."""
        for broken in ({"modelDigest": None}, {"modelDigest": ""}, {"modelDigest": 42}, {"modelDigest": "  "}):
            with self.subTest(broken=broken):
                self._patch(_response(200, ok_payload(**broken)))
                result = await generate_benchmark_answer("v2", PROMPT, timeout=30.0)
                self.assertIsNone(result["modelDigest"])
        payload = ok_payload()
        del payload["modelDigest"]
        self._patch(_response(200, payload))
        result = await generate_benchmark_answer("v2", PROMPT, timeout=30.0)
        self.assertIsNone(result["modelDigest"])

    async def test_timeouts_and_connection_failures(self) -> None:
        await self._expect("service_timeout", error=httpx.ReadTimeout("slow"))
        await self._expect("service_unavailable", error=httpx.ConnectError("refused"))

    async def test_server_errors_are_logged_in_full_and_summarised_in_public(self) -> None:
        with self.assertLogs("app.research_benchmark_client", level="WARNING") as logs:
            exc = await self._expect("service_error", _response(503))
        self.assertIn("HTTP 503", exc.message)
        logged = "\n".join(logs.output)
        self.assertIn(POISONED_BODY, logged)
        self.assertIn("v2", logged)

    async def test_an_unmapped_alias_names_the_alias(self) -> None:
        exc = await self._expect("alias_not_mapped", _response(409), alias="v4_vm")
        self.assertIn('"v4_vm"', exc.message)

    async def test_other_rejections(self) -> None:
        for status in (400, 401, 404, 422):
            with self.subTest(status=status):
                exc = await self._expect("service_rejected", _response(status))
                self.assertIn(f"HTTP {status}", exc.message)

    async def test_malformed_bodies(self) -> None:
        await self._expect("malformed_response", _response(200))  # invalid JSON
        await self._expect("malformed_response", _response(200, ["not", "an", "object"]))
        for broken in (
            {"answer": 7},
            {"model": ""},
            {"model": None},
            {"options": "greedy"},
            {"generationSeconds": "fast"},
            {"generationSeconds": True},
        ):
            with self.subTest(broken=broken):
                await self._expect("malformed_response", _response(200, ok_payload(**broken)))

    async def test_a_different_alias_is_refused(self) -> None:
        await self._expect("alias_mismatch", _response(200, ok_payload(alias="v3")))

    async def test_a_different_prompt_hash_is_refused(self) -> None:
        await self._expect("prompt_integrity", _response(200, ok_payload(promptSha256="0" * 64)))
        await self._expect("prompt_integrity", _response(200, ok_payload(promptSha256=None)))

    async def test_an_upper_case_echo_of_the_same_hash_is_accepted(self) -> None:
        self._patch(_response(200, ok_payload(promptSha256=prompt_sha256(PROMPT).upper())))
        result = await generate_benchmark_answer("v2", PROMPT, timeout=30.0)
        self.assertEqual(result["promptSha256"], prompt_sha256(PROMPT))

    def test_timing_projection_keeps_only_known_numbers(self) -> None:
        self.assertEqual(
            _project_ollama_timings({"evalCount": 3, "loadDurationNs": True, "doneReason": " length ",
                                     "path": "/home/testuser", "evalDurationNs": 1.5, "totalDurationNs": 10}),
            {"evalCount": 3, "totalDurationNs": 10, "doneReason": "length"},
        )
        self.assertEqual(_project_ollama_timings("nope"), {})
        self.assertEqual(_project_ollama_timings(None), {})


if __name__ == "__main__":
    unittest.main()
