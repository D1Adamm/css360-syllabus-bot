"""The benchmark-only inference service, against a fake Ollama.

What is pinned: the alias allowlist and the alias -> tag mapping (server
configuration, refused when malformed or when it names an alias that does
not exist); one decoding recipe for every alias, pinned to the shared
grounded options whatever the environment says; the prompt forwarded byte
for byte as one user turn and hashed as sent; the failure codes; and the
loopback binding on its own port. Nothing here reaches a real Ollama.
"""

from __future__ import annotations

import hashlib
import os
import sys
import unittest
from typing import Any, Dict, Optional
from unittest import mock

import httpx
from fastapi.testclient import TestClient

import benchmark_service
from test_ollama_service import FakeOllama

BASE_TAG = "llama3.2:3b"
V2_TAG = "css360-ft-v2:latest"
V4_VM_TAG = "css360-v4-test:latest"
FULL_MAP = (
    f"base={BASE_TAG},v2=css360-ft-v2,v3=css360-cpu-v3-test:latest,"
    f"v4_vm={V4_VM_TAG},v4_tillicum=css360-v4-tillicum-test:latest"
)
PROMPT = "You are answering a student's question.\n\nStudent question:\nWhen?\n\nAnswer:"
ANSWER = "Tuesdays at 2pm."

EXPECTED_OPTIONS = {
    "num_predict": 256,
    "temperature": 0,
    "repeat_penalty": 1.05,
    "repeat_last_n": 4096,
    "seed": 360,
    "num_ctx": 4096,
}


def _env(mapping: Optional[str], **extra: str) -> "mock._patch_dict":
    values = {
        benchmark_service.MODEL_MAP_ENV: mapping or "",
        "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
        benchmark_service.PORT_ENV: "",
        benchmark_service.KEEP_ALIVE_ENV: "",
        benchmark_service.TIMEOUT_ENV: "",
        "FINETUNED_NUM_CTX": "",
        "FINETUNED_KEEP_ALIVE": "",
    }
    values.update(extra)
    return mock.patch.dict(os.environ, values, clear=False)


def _client() -> TestClient:
    return TestClient(benchmark_service.app)


#: Full manifest digests, as `/api/tags` reports them.
DIGESTS = {
    BASE_TAG: "a80c4f17acd5" + "1" * 52,
    V2_TAG: "dea74c57f25a" + "2" * 52,
    V4_VM_TAG: "b8763e820e93" + "4" * 52,
}


class DigestFakeOllama(FakeOllama):
    """The production fake, with digests on `/api/tags` as the real server has."""

    async def get(self, url: str, **kwargs: Any):
        self.requests.append({"method": "GET", "url": url})
        if self.tags_error is not None:
            raise self.tags_error
        return type(
            "R", (), {"status_code": 200, "text": "",
                      "json": lambda self_, payload={"models": [
                          {"name": name, "digest": DIGESTS.get(name, "f" * 64)} for name in self.models
                      ]}: payload}
        )()


class AliasMapTests(unittest.TestCase):
    def test_every_alias_maps_and_untagged_names_become_latest(self) -> None:
        parsed = benchmark_service.parse_alias_map(FULL_MAP)
        self.assertEqual(parsed, {
            "base": BASE_TAG,
            "v2": V2_TAG,
            "v3": "css360-cpu-v3-test:latest",
            "v4_vm": V4_VM_TAG,
            "v4_tillicum": "css360-v4-tillicum-test:latest",
        })

    def test_separators_and_emptiness(self) -> None:
        self.assertEqual(benchmark_service.parse_alias_map(f"base={BASE_TAG}\n v2={V2_TAG}, "),
                         {"base": BASE_TAG, "v2": V2_TAG})
        self.assertEqual(benchmark_service.parse_alias_map(None), {})
        self.assertEqual(benchmark_service.parse_alias_map(" , "), {})

    def test_an_alias_outside_the_allowlist_fails_the_whole_map(self) -> None:
        for bad in ("v1=css360-ft-v1-test", "v5=css360-v4-test", "v4=css360-v4-test",
                    "css-360-winter-2026-a7rp@v2=css360-ft-v2", "V2=css360-ft-v2", "latest=x"):
            with self.subTest(entry=bad):
                with self.assertRaises(benchmark_service.BenchmarkConfigError):
                    benchmark_service.parse_alias_map(f"base={BASE_TAG},{bad}")

    def test_malformed_entries_fail_the_whole_map(self) -> None:
        for bad in ("v2", "v2=", "=css360-ft-v2", "v2=a b"):
            with self.subTest(entry=bad):
                with self.assertRaises(benchmark_service.BenchmarkConfigError):
                    benchmark_service.parse_alias_map(bad)

    def test_one_alias_two_tags_is_refused_and_the_same_tag_twice_is_fine(self) -> None:
        with self.assertRaises(benchmark_service.BenchmarkConfigError):
            benchmark_service.parse_alias_map(f"v2={V2_TAG},v2=css360-other")
        self.assertEqual(benchmark_service.parse_alias_map(f"v2=css360-ft-v2,v2={V2_TAG}"), {"v2": V2_TAG})

    def test_resolution_never_borrows_another_aliass_tag(self) -> None:
        mapping = {"base": BASE_TAG, "v2": V2_TAG}
        self.assertEqual(benchmark_service.resolve_alias("v2", mapping=mapping), V2_TAG)
        with self.assertRaises(LookupError) as caught:
            benchmark_service.resolve_alias("v4_vm", mapping=mapping)
        self.assertIn("v4_vm", str(caught.exception))
        self.assertIn(benchmark_service.MODEL_MAP_ENV, str(caught.exception))
        self.assertNotIn(V2_TAG, str(caught.exception))
        with self.assertRaises(benchmark_service.BenchmarkConfigError):
            benchmark_service.resolve_alias("v1", mapping=mapping)
        with self.assertRaises(LookupError):
            benchmark_service.resolve_alias("v2", mapping={})


class DecodingTests(unittest.TestCase):
    def test_the_options_are_the_shared_grounded_recipe(self) -> None:
        with _env(FULL_MAP):
            self.assertEqual(benchmark_service.benchmark_options(), EXPECTED_OPTIONS)

    def test_the_production_context_override_is_ignored(self) -> None:
        with _env(FULL_MAP, FINETUNED_NUM_CTX="8192"):
            options = benchmark_service.benchmark_options()
        self.assertEqual(options["num_ctx"], 4096)
        self.assertEqual(options["repeat_last_n"], 4096)

    def test_the_prompt_is_one_user_turn_and_nothing_else(self) -> None:
        with _env(FULL_MAP):
            payload = benchmark_service.build_chat_request(PROMPT, ollama_model=V2_TAG)
        self.assertEqual(payload, {
            "model": V2_TAG,
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": False,
            "options": EXPECTED_OPTIONS,
        })

    def test_keep_alive_is_the_benchmarks_own_variable(self) -> None:
        with _env(FULL_MAP, FINETUNED_KEEP_ALIVE="30m"):
            self.assertNotIn("keep_alive", benchmark_service.build_chat_request(PROMPT, ollama_model=V2_TAG))
        with _env(FULL_MAP, CSS360_BENCHMARK_KEEP_ALIVE="15m"):
            self.assertEqual(benchmark_service.build_chat_request(PROMPT, ollama_model=V2_TAG)["keep_alive"], "15m")

    def test_digests_are_read_from_the_tags_listing(self) -> None:
        payload = {"models": [
            {"name": "css360-ft-v2", "digest": "ab" * 32},
            {"name": BASE_TAG, "digest": "cd" * 32},
            {"name": "broken", "digest": 7},
            {"name": "", "digest": "ef" * 32},
            {"model": "css360-v4-test:latest", "digest": "  " + "12" * 32 + " "},
            "not a dict",
        ]}
        self.assertEqual(benchmark_service.model_digests(payload), {
            V2_TAG: "ab" * 32,
            BASE_TAG: "cd" * 32,
            V4_VM_TAG: "12" * 32,
        })
        self.assertEqual(benchmark_service.model_digests({"models": "x"}), {})
        self.assertEqual(benchmark_service.model_digests(None), {})

    def test_timings_are_copied_by_name(self) -> None:
        self.assertEqual(
            benchmark_service.extract_ollama_timings({
                "total_duration": 10, "load_duration": True, "eval_count": 3, "eval_duration": 1.5,
                "done_reason": " stop ", "model": "x", "message": {"content": "y"},
            }),
            {"totalDurationNs": 10, "evalCount": 3, "doneReason": "stop"},
        )
        self.assertEqual(benchmark_service.extract_ollama_timings("nope"), {})


class HealthTests(unittest.TestCase):
    def test_health_lists_every_alias_mapped_or_not(self) -> None:
        fake = FakeOllama(models=[V2_TAG, BASE_TAG])
        with _env(f"base={BASE_TAG},v2={V2_TAG},v4_vm={V4_VM_TAG}"), mock.patch.object(
            benchmark_service.httpx, "AsyncClient", fake
        ):
            body = _client().get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["engine"], "ollama")
        self.assertEqual([row["alias"] for row in body["aliases"]], list(benchmark_service.ALIASES))
        by_alias = {row["alias"]: row for row in body["aliases"]}
        self.assertEqual(by_alias["v2"], {"alias": "v2", "ollamaModel": V2_TAG, "mapped": True,
                                          "available": True, "digest": None})
        self.assertEqual(by_alias["v4_vm"], {"alias": "v4_vm", "ollamaModel": V4_VM_TAG, "mapped": True,
                                             "available": False, "digest": None})
        self.assertEqual(by_alias["v3"], {"alias": "v3", "ollamaModel": None, "mapped": False,
                                          "available": False, "digest": None})
        self.assertEqual(body["servable"], ["base", "v2"])
        self.assertIsNone(body["detail"])
        for key in ("hostname", "port", "ollamaUrl"):
            self.assertNotIn(key, body)

    def test_health_reports_the_digest_ollama_holds_for_each_available_tag(self) -> None:
        fake = DigestFakeOllama(models=[V2_TAG, BASE_TAG])
        with _env(f"base={BASE_TAG},v2={V2_TAG},v4_vm={V4_VM_TAG}"), mock.patch.object(
            benchmark_service.httpx, "AsyncClient", fake
        ):
            body = _client().get("/health").json()
        by_alias = {row["alias"]: row for row in body["aliases"]}
        self.assertEqual(by_alias["v2"]["digest"], DIGESTS[V2_TAG])
        self.assertEqual(by_alias["base"]["digest"], DIGESTS[BASE_TAG])
        self.assertIsNone(by_alias["v4_vm"]["digest"])  # mapped, not present

    def test_health_when_ollama_is_down(self) -> None:
        fake = FakeOllama(tags_error=httpx.ConnectError("connection refused"))
        with _env(FULL_MAP), mock.patch.object(benchmark_service.httpx, "AsyncClient", fake):
            body = _client().get("/health").json()
        self.assertEqual(body["status"], "unavailable")
        self.assertEqual(body["servable"], [])
        self.assertTrue(all(row["available"] is False for row in body["aliases"]))

    def test_startup_refuses_a_malformed_mapping(self) -> None:
        with _env("v9=css360-v9"):
            with self.assertRaises(benchmark_service.BenchmarkConfigError):
                with TestClient(benchmark_service.app):
                    pass


class GenerateTests(unittest.TestCase):
    def _generate(self, fake: FakeOllama, body: Dict[str, Any], mapping: str = FULL_MAP, **extra: str):
        with _env(mapping, **extra), mock.patch.object(benchmark_service.httpx, "AsyncClient", fake):
            return _client().post("/generate", json=body)

    def test_a_good_generation(self) -> None:
        fake = FakeOllama(chat_payload={
            "model": V4_VM_TAG,
            "message": {"role": "assistant", "content": f"  {ANSWER}\n"},
            "done": True, "done_reason": "stop",
            "total_duration": 9, "load_duration": 2, "prompt_eval_count": 50,
            "prompt_eval_duration": 3, "eval_count": 7, "eval_duration": 4,
        })
        response = self._generate(fake, {"alias": "v4_vm", "prompt": PROMPT})

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["alias"], "v4_vm")
        self.assertEqual(body["model"], V4_VM_TAG)
        self.assertEqual(body["answer"], ANSWER)
        self.assertEqual(body["promptSha256"], hashlib.sha256(PROMPT.encode("utf-8")).hexdigest())
        self.assertEqual(body["options"], EXPECTED_OPTIONS)
        self.assertIsInstance(body["generationSeconds"], float)
        self.assertEqual(body["ollama"], {
            "totalDurationNs": 9, "loadDurationNs": 2, "promptEvalCount": 50,
            "promptEvalDurationNs": 3, "evalCount": 7, "evalDurationNs": 4, "doneReason": "stop",
        })
        (sent,) = fake.chat_requests
        self.assertEqual(sent["url"], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(sent["json"]["model"], V4_VM_TAG)
        self.assertEqual(sent["json"]["messages"], [{"role": "user", "content": PROMPT}])
        self.assertEqual(sent["json"]["options"], EXPECTED_OPTIONS)

    def test_the_served_digest_is_looked_up_per_generation(self) -> None:
        fake = DigestFakeOllama(models=[V2_TAG, BASE_TAG, V4_VM_TAG])
        body = self._generate(fake, {"alias": "v4_vm", "prompt": PROMPT}).json()
        self.assertEqual(body["modelDigest"], DIGESTS[V4_VM_TAG])
        # One tags lookup and one chat call, in that order, for the one generation.
        self.assertEqual([r["method"] for r in fake.requests], ["GET", "POST"])
        self.assertTrue(fake.requests[0]["url"].endswith("/api/tags"))

    def test_a_failed_digest_lookup_does_not_fail_the_generation(self) -> None:
        """The answer is the point; the digest is its label. Without the label
        the answer is still returned, with `modelDigest` null."""
        fake = FakeOllama(answer=ANSWER, tags_error=httpx.ConnectError("tags refused"))
        response = self._generate(fake, {"alias": "v2", "prompt": PROMPT})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(response.json()["modelDigest"])
        self.assertEqual(response.json()["answer"], ANSWER)
        # The chat call still happened; only the tags lookup failed.
        self.assertEqual(len(fake.chat_requests), 1)

    def test_a_tag_absent_from_the_listing_has_no_digest(self) -> None:
        fake = DigestFakeOllama(models=[BASE_TAG])
        body = self._generate(fake, {"alias": "v2", "prompt": PROMPT}).json()
        self.assertIsNone(body["modelDigest"])

    def test_the_prompt_travels_verbatim_including_its_whitespace(self) -> None:
        raw = "  leading, trailing and\n\ninternal   whitespace kept \n"
        fake = FakeOllama()
        body = self._generate(fake, {"alias": "v2", "prompt": raw}).json()
        self.assertEqual(fake.chat_requests[0]["json"]["messages"][0]["content"], raw)
        self.assertEqual(body["promptSha256"], hashlib.sha256(raw.encode("utf-8")).hexdigest())

    def test_each_alias_is_answered_by_its_own_tag(self) -> None:
        fake = FakeOllama()
        for alias, tag in (("base", BASE_TAG), ("v2", V2_TAG), ("v4_vm", V4_VM_TAG)):
            with self.subTest(alias=alias):
                body = self._generate(fake, {"alias": alias, "prompt": PROMPT}).json()
                self.assertEqual(body["alias"], alias)
                self.assertEqual(body["model"], tag)
        self.assertEqual([r["json"]["model"] for r in fake.chat_requests], [BASE_TAG, V2_TAG, V4_VM_TAG])

    def test_an_unknown_alias_is_422_and_ollama_is_never_asked(self) -> None:
        fake = FakeOllama()
        for bad in ("v1", "v5", "V2", "css360-v4-test:latest", "", "base "):
            with self.subTest(alias=bad):
                self.assertEqual(self._generate(fake, {"alias": bad, "prompt": PROMPT}).status_code, 422)
        self.assertEqual(fake.chat_requests, [])

    def test_a_request_may_name_nothing_but_an_alias_and_a_prompt(self) -> None:
        fake = FakeOllama()
        for key, value in (("model", V2_TAG), ("tag", V2_TAG), ("courseId", "css-360-winter-2026-a7rp"),
                           ("modelVersion", "v2"), ("options", {"temperature": 1}), ("num_ctx", 8192),
                           ("url", "http://127.0.0.1:11434"), ("keep_alive", "1h")):
            with self.subTest(field=key):
                response = self._generate(fake, {"alias": "v2", "prompt": PROMPT, key: value})
                self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(fake.chat_requests, [])

    def test_a_blank_or_oversized_prompt_is_422(self) -> None:
        fake = FakeOllama()
        self.assertEqual(self._generate(fake, {"alias": "v2", "prompt": "   "}).status_code, 422)
        self.assertEqual(self._generate(fake, {"alias": "v2", "prompt": ""}).status_code, 422)
        too_long = "q" * (benchmark_service.MAX_PROMPT_CHARS + 1)
        self.assertEqual(self._generate(fake, {"alias": "v2", "prompt": too_long}).status_code, 422)
        self.assertEqual(fake.chat_requests, [])

    def test_an_unmapped_alias_is_409(self) -> None:
        fake = FakeOllama()
        response = self._generate(fake, {"alias": "v4_tillicum", "prompt": PROMPT}, mapping=f"base={BASE_TAG}")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["code"], "alias_not_mapped")
        self.assertIn("v4_tillicum", response.json()["detail"]["message"])
        self.assertEqual(fake.chat_requests, [])

    def test_with_no_mapping_every_alias_is_refused(self) -> None:
        fake = FakeOllama()
        for alias in benchmark_service.ALIASES:
            with self.subTest(alias=alias):
                self.assertEqual(self._generate(fake, {"alias": alias, "prompt": PROMPT}, mapping="").status_code, 409)
        self.assertEqual(fake.chat_requests, [])

    def test_ollama_failure_codes(self) -> None:
        cases = [
            (FakeOllama(chat_status=404, chat_text="model not found"), 409, "model_missing"),
            (FakeOllama(chat_status=500, chat_text="runner terminated"), 503, "ollama_error"),
            (FakeOllama(chat_error=httpx.ReadTimeout("slow")), 503, "ollama_timeout"),
            (FakeOllama(chat_error=httpx.ConnectError("refused")), 503, "ollama_unavailable"),
            (FakeOllama(chat_status=400, chat_text="invalid options"), 502, "ollama_rejected"),
            (FakeOllama(chat_payload={"model": V2_TAG, "done": True}), 502, "malformed_ollama_response"),
        ]
        for fake, status, code in cases:
            with self.subTest(code=code):
                response = self._generate(fake, {"alias": "v2", "prompt": PROMPT})
                self.assertEqual(response.status_code, status, response.text)
                self.assertEqual(response.json()["detail"]["code"], code)

    def test_the_timeout_is_the_benchmarks_own_variable(self) -> None:
        with _env(FULL_MAP, CSS360_BENCHMARK_OLLAMA_TIMEOUT_SECONDS="7"):
            self.assertEqual(benchmark_service.resolve_timeout_seconds(), 7.0)
        with _env(FULL_MAP, CSS360_BENCHMARK_OLLAMA_TIMEOUT_SECONDS="0"):
            self.assertEqual(benchmark_service.resolve_timeout_seconds(), 1.0)
        with _env(FULL_MAP, CSS360_BENCHMARK_OLLAMA_TIMEOUT_SECONDS="soon"):
            self.assertEqual(benchmark_service.resolve_timeout_seconds(), 120.0)
        with _env(FULL_MAP, FINETUNED_OLLAMA_TIMEOUT_SECONDS="7"):
            self.assertEqual(benchmark_service.resolve_timeout_seconds(), 120.0)


class BindingTests(unittest.TestCase):
    def test_loopback_on_its_own_port(self) -> None:
        self.assertEqual(benchmark_service.BIND_HOST, "127.0.0.1")
        with _env(FULL_MAP):
            self.assertEqual(benchmark_service.resolve_port(), 9002)
        with _env(FULL_MAP, BENCHMARK_INFERENCE_PORT="9102"):
            self.assertEqual(benchmark_service.resolve_port(), 9102)
        with _env(FULL_MAP, INFERENCE_PORT="9001"):
            # The production service's port variable is not this service's.
            self.assertEqual(benchmark_service.resolve_port(), 9002)
        with _env(FULL_MAP, BENCHMARK_INFERENCE_PORT="port"):
            with self.assertRaises(RuntimeError):
                benchmark_service.resolve_port()

    def test_main_binds_loopback_even_if_a_host_override_is_in_the_environment(self) -> None:
        fake_uvicorn = mock.MagicMock()
        with _env(FULL_MAP, INFERENCE_HOST="0.0.0.0", BENCHMARK_INFERENCE_HOST="0.0.0.0"), mock.patch.dict(
            sys.modules, {"uvicorn": fake_uvicorn}
        ):
            benchmark_service.main()
        _, kwargs = fake_uvicorn.run.call_args
        self.assertEqual(kwargs["host"], "127.0.0.1")
        self.assertEqual(kwargs["port"], 9002)


if __name__ == "__main__":
    unittest.main()
