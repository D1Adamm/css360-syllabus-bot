"""The CSS 360 controlled benchmark route: protected, fixed, isolated, private.

`POST /api/research/css360/benchmark/pair` and `/standalone` exist for a
benchmark runner and for nobody else. What is held here, section by section:

1. off by default, and off means 404 — the same 404 as an unmounted path —
   for anyone, with any credential, until the flag, a long enough token and
   the service URL are all configured;
2. the credential is a bearer token compared in constant time; no session
   cookie is one, and no database is opened to decide;
3. the limits: body size before the body is read, question length, condition
   count and allowlist, unknown fields refused, a failure limiter on bad
   tokens, a request limiter, a concurrency slot that refuses, a bounded
   per-condition timeout;
4. the pair route retrieves once for the fixed course at the fixed depth, and
   every condition receives the same prompt bytes, the same ordered chunks and
   the same decoding options; the hashes on the response are of exactly those;
5. every answer carries the lineage projection of the artifact that produced
   it; a served tag other than the recorded one makes the condition invalid
   and unscored, and an answer with no text makes the generation a failed,
   unscored one, both keeping their timing and their error record;
6. one condition's failure is that condition's error entry, timed, and the
   request still answers 200;
7. the standalone route retrieves nothing and sends the bare question;
8. no response — success or refusal — names a path, a port, a host, the
   token, an environment variable or anything about the database;
9. the administrator's model-testing routes are untouched;
10. every answer records the Ollama digest that served it, every condition an
    outcome, and the response a summary that counts every attempt.

The benchmark service is stubbed at the seam the route calls it through
(`generate_benchmark_answer`); retrieval at `retrieve_and_prompt`. The lineage
record is the real `evaluation/model_lineage.json`.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import json
import os
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from app import research_benchmark_client, research_benchmark_lineage
from app import research_benchmark_routes as routes
from app.auth.dependencies import current_principal
from app.auth.principal import Principal, StaffUser
from app.grounded_generation import (
    GROUNDED_NUM_CTX,
    GROUNDED_NUM_PREDICT,
    PROMPT_TEMPLATE_NAME,
    build_grounded_prompt,
    grounded_options,
    prompt_template_fingerprint,
)
from app.main import app
from app.research_benchmark_client import BenchmarkConditionError
from app.research_benchmark_lineage import (
    LINEAGE_SUMMARY_FIELDS,
    LineageUnavailable,
    lineage_for_alias,
    load_lineage,
)
from app.retrieval_diversity import DEFAULT_TOP_K
from app.schemas import (
    ModelTestingGenerateRequest,
    ModelTestingGenerateResponse,
    ModelTestingPairRequest,
    ModelTestingPairResponse,
)
from route_classification import CLASSIFICATION, REQUIRE_ADMIN, RESEARCH_TOKEN

#: The real `current_principal` runs — and is never consulted, which is the point.
pytestmark = pytest.mark.auth

COURSE = "css-360-winter-2026-a7rp"
PAIR = "/api/research/css360/benchmark/pair"
STANDALONE = "/api/research/css360/benchmark/standalone"
TOKEN = "research-benchmark-token-" + "k7" * 24
SERVICE_URL = "http://127.0.0.1:9002"
ENABLED_ENV = {
    "CSS360_BENCHMARK_ENABLED": "true",
    "CSS360_BENCHMARK_TOKEN": TOKEN,
    "CSS360_BENCHMARK_SERVICE_URL": SERVICE_URL,
}
AUTH = {"Authorization": f"Bearer {TOKEN}"}
QUESTION = "When are office hours?"

LINEAGE = load_lineage()
EXPECTED_TAGS = {
    "base": "llama3.2:3b",
    "v2": "css360-ft-v2:latest",
    "v3": "css360-cpu-v3-test:latest",
    "v4_vm": "css360-v4-test:latest",
    "v4_tillicum": "css360-v4-tillicum-test:latest",
}
#: The record's twelve-character digests; the fake service reports them as the
#: prefix of a full manifest digest, as Ollama's API would.
EXPECTED_DIGESTS = {alias: lineage_for_alias(alias, LINEAGE)["ollamaDigest"] for alias in EXPECTED_TAGS}


def full_digest(alias: str) -> str:
    return EXPECTED_DIGESTS[alias] + "0" * (64 - len(EXPECTED_DIGESTS[alias]))

CHUNKS: list[dict[str, Any]] = [
    {"chunkId": "c1", "section": "Office Hours",
     "text": "Office hours are Tuesdays at 2pm in the CSS lab.", "score": 0.91},
    {"chunkId": "c2", "section": "Late Work",
     "text": "Late work loses 10% per day for up to three days.", "score": 0.77},
]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepared(question: str = QUESTION) -> dict[str, Any]:
    """What `retrieve_and_prompt` returns, with the real grounded template."""
    prompt_chunks = [{"section": c["section"], "text": c["text"]} for c in CHUNKS]
    return {
        "courseId": COURSE,
        "question": question,
        "facets": [],
        "chunks": prompt_chunks,
        "prompt": build_grounded_prompt(question, prompt_chunks, []),
        "sources": [
            {"chunkId": c["chunkId"], "sectionTitle": c["section"], "text": c["text"],
             "score": c["score"]}
            for c in CHUNKS
        ],
        "retrievedChunks": [dict(c) for c in CHUNKS],
    }


class FakeService:
    """Stands in for `generate_benchmark_answer`. Records; answers from a script."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.failures: dict[str, BaseException] = {}
        self.tags: dict[str, str] = {}
        self.options: dict[str, dict[str, Any]] = {}
        self.delays: dict[str, float] = {}
        self.answers: dict[str, str] = {}
        self.ollama: dict[str, dict[str, Any]] = {}
        self.digests: dict[str, str | None] = {}
        self.block_alias: str | None = None
        self.entered = threading.Event()
        self.release = threading.Event()

    async def __call__(self, alias: str, prompt: str, *, timeout: float) -> dict[str, Any]:
        self.calls.append({"alias": alias, "prompt": prompt, "timeout": timeout})
        if alias == self.block_alias:
            self.entered.set()
            while not self.release.is_set():
                await asyncio.sleep(0.01)
        delay = self.delays.get(alias)
        if delay:
            await asyncio.sleep(delay)
        failure = self.failures.get(alias)
        if failure is not None:
            raise failure
        return {
            "alias": alias,
            "model": self.tags.get(alias, EXPECTED_TAGS[alias]),
            "modelDigest": self.digests.get(alias, full_digest(alias)),
            "answer": self.answers.get(alias, f"Answer from {alias}."),
            "promptSha256": sha(prompt),
            "options": self.options.get(alias, grounded_options()),
            "generationSeconds": 0.5,
            "ollama": self.ollama.get(
                alias, {"evalCount": 12, "loadDurationNs": 1000, "doneReason": "stop"}
            ),
        }

    @property
    def aliases(self) -> list[str]:
        return [call["alias"] for call in self.calls]

    @property
    def prompts(self) -> list[str]:
        return [call["prompt"] for call in self.calls]


def admin() -> Principal:
    return Principal(
        user=StaffUser(user_id="u-admin", email="admin@uw.edu", display_name="Admin",
                       role="admin", session_id="s")
    )


class BenchmarkRouteTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app, base_url="https://testserver")
        self.addCleanup(app.dependency_overrides.clear)
        self.service = FakeService()
        self.patch("app.research_benchmark_routes.generate_benchmark_answer", new=self.service)
        self.retrieve = self.patch(
            "app.research_benchmark_routes.retrieve_and_prompt",
            new=AsyncMock(side_effect=lambda course_id, question, top_k: prepared(question)),
        )

    def patch(self, target: str, **kwargs: Any) -> Any:
        patcher = patch(target, **kwargs)
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock

    def enable(self, **overrides: str) -> None:
        patcher = patch.dict(os.environ, {**ENABLED_ENV, **overrides})
        patcher.start()
        self.addCleanup(patcher.stop)

    def post(self, path: str, body: dict[str, Any], headers: dict[str, str] | None = None, **kw: Any):
        return self.client.post(path, json=body, headers=AUTH if headers is None else headers, **kw)

    def assert_nothing_generated(self) -> None:
        self.assertEqual(self.service.calls, [])
        self.retrieve.assert_not_awaited()


# --------------------------------------------------------------------------- #
# 1. Off by default, and off is 404
# --------------------------------------------------------------------------- #


class DisabledTests(BenchmarkRouteTestCase):
    def test_off_by_default_is_404_for_both_routes_even_with_a_token(self) -> None:
        for path in (PAIR, STANDALONE):
            with self.subTest(path=path):
                response = self.post(path, {"question": QUESTION})
                self.assertEqual(response.status_code, 404)
                self.assertEqual(response.json(), {"detail": "Not Found"})
        # Whatever the method: an unmounted path answers 404 to a GET too.
        self.assertEqual(self.client.get(PAIR, headers=AUTH).status_code, 404)
        self.assert_nothing_generated()

    def test_every_missing_piece_of_configuration_is_404(self) -> None:
        cases = {
            "flag only": {"CSS360_BENCHMARK_ENABLED": "true"},
            "flag and a short token": {"CSS360_BENCHMARK_ENABLED": "true",
                                       "CSS360_BENCHMARK_TOKEN": "short",
                                       "CSS360_BENCHMARK_SERVICE_URL": SERVICE_URL},
            "flag and token, no url": {"CSS360_BENCHMARK_ENABLED": "true",
                                       "CSS360_BENCHMARK_TOKEN": TOKEN},
            "token and url, flag off": {"CSS360_BENCHMARK_ENABLED": "0",
                                        "CSS360_BENCHMARK_TOKEN": TOKEN,
                                        "CSS360_BENCHMARK_SERVICE_URL": SERVICE_URL},
            "token and url, no flag": {"CSS360_BENCHMARK_TOKEN": TOKEN,
                                       "CSS360_BENCHMARK_SERVICE_URL": SERVICE_URL},
            "flag spelled oddly": {"CSS360_BENCHMARK_ENABLED": "enabled",
                                   "CSS360_BENCHMARK_TOKEN": TOKEN,
                                   "CSS360_BENCHMARK_SERVICE_URL": SERVICE_URL},
        }
        for label, env in cases.items():
            with self.subTest(case=label), patch.dict(os.environ, env):
                self.assertFalse(routes.benchmark_settings().enabled)
                response = self.post(PAIR, {"question": QUESTION})
                self.assertEqual(response.status_code, 404)
                self.assertEqual(response.json(), {"detail": "Not Found"})
        self.assert_nothing_generated()

    def test_the_guard_itself_answers_404_when_off(self) -> None:
        """Defence in depth: the guard refuses without the middleware in front."""
        request = Request({"type": "http", "method": "POST", "path": PAIR, "headers": [],
                           "query_string": b"", "client": ("testclient", 1)})
        with self.assertRaises(HTTPException) as caught:
            routes.require_css360_benchmark_access(request)
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.detail, "Not Found")

    def test_the_routes_are_classified_and_have_no_root_alias(self) -> None:
        self.assertEqual(CLASSIFICATION[("POST", PAIR)], RESEARCH_TOKEN)
        self.assertEqual(CLASSIFICATION[("POST", STANDALONE)], RESEARCH_TOKEN)
        mounted = {getattr(route, "path", None) for route in app.routes}
        self.assertNotIn("/research/css360/benchmark/pair", mounted)
        self.assertNotIn("/research/css360/benchmark/standalone", mounted)


# --------------------------------------------------------------------------- #
# 2. The credential
# --------------------------------------------------------------------------- #


class BearerAuthTests(BenchmarkRouteTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.enable()

    def test_no_credential_is_401_with_a_challenge(self) -> None:
        response = self.post(PAIR, {"question": QUESTION}, headers={})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers.get("www-authenticate"), "Bearer")
        self.assert_nothing_generated()

    def test_a_wrong_or_malformed_credential_is_401(self) -> None:
        for header in (
            f"Bearer {TOKEN[:-1]}",
            f"Bearer {TOKEN}x",
            f"Basic {TOKEN}",
            "Bearer",
            "Bearer   ",
            TOKEN,
            f"Token {TOKEN}",
        ):
            with self.subTest(header=header):
                response = self.post(PAIR, {"question": QUESTION}, headers={"Authorization": header})
                self.assertEqual(response.status_code, 401, response.text)
        self.assert_nothing_generated()

    def test_the_configured_token_reaches_the_handler(self) -> None:
        for header in (f"Bearer {TOKEN}", f"bearer {TOKEN}", f"BEARER {TOKEN}", f"Bearer   {TOKEN}  "):
            with self.subTest(header=header):
                response = self.post(STANDALONE, {"question": QUESTION, "conditions": ["base"]},
                                     headers={"Authorization": header})
                self.assertEqual(response.status_code, 200, response.text)

    def test_a_session_cookie_is_not_a_credential(self) -> None:
        """An administrator's browser session reaches every admin route and
        not this one: the guard never resolves a principal."""
        app.dependency_overrides[current_principal] = admin
        response = self.post(PAIR, {"question": QUESTION},
                             headers={"X-Requested-With": "SyllabusModelLab"})
        self.assertEqual(response.status_code, 401)
        self.assert_nothing_generated()

    def test_the_comparison_is_constant_time_over_fixed_length_digests(self) -> None:
        recorder = MagicMock(side_effect=hmac.compare_digest)
        with patch("app.research_benchmark_routes.hmac.compare_digest", recorder):
            self.assertFalse(routes.bearer_token_matches("Bearer x", TOKEN))
            self.assertFalse(routes.bearer_token_matches("Bearer " + "y" * 300, TOKEN))
            self.assertTrue(routes.bearer_token_matches(f"Bearer {TOKEN}", TOKEN))
        self.assertEqual(recorder.call_count, 3)
        for call in recorder.call_args_list:
            left, right = call.args
            self.assertIsInstance(left, bytes)
            self.assertIsInstance(right, bytes)
            self.assertEqual(len(left), 32)
            self.assertEqual(len(right), 32)

    def test_no_token_configured_never_matches(self) -> None:
        self.assertFalse(routes.bearer_token_matches("Bearer anything", None))
        self.assertFalse(routes.bearer_token_matches("Bearer ", ""))


# --------------------------------------------------------------------------- #
# 3. Limits
# --------------------------------------------------------------------------- #


class LimitTests(BenchmarkRouteTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.enable()

    def test_ten_bad_tokens_lock_the_client_out_even_for_the_right_one(self) -> None:
        for _ in range(10):
            self.assertEqual(
                self.post(PAIR, {"question": QUESTION}, headers={"Authorization": "Bearer nope"}).status_code,
                401,
            )
        locked = self.post(STANDALONE, {"question": QUESTION, "conditions": ["base"]})
        self.assertEqual(locked.status_code, 429)
        self.assertEqual(locked.json()["detail"]["code"], "auth_failures")
        self.assertTrue(locked.headers.get("retry-after"))
        # Another client is unaffected.
        other = self.post(STANDALONE, {"question": QUESTION, "conditions": ["base"]},
                          headers={**AUTH, "X-Forwarded-For": "10.0.0.9"})
        self.assertEqual(other.status_code, 200, other.text)

    def test_accepted_requests_are_limited_per_client_per_minute(self) -> None:
        self.enable(CSS360_BENCHMARK_REQUESTS_PER_MINUTE="2")
        body = {"question": QUESTION, "conditions": ["base"]}
        self.assertEqual(self.post(STANDALONE, body).status_code, 200)
        self.assertEqual(self.post(STANDALONE, body).status_code, 200)
        third = self.post(STANDALONE, body)
        self.assertEqual(third.status_code, 429)
        self.assertEqual(third.json()["detail"]["code"], "rate_limited")
        self.assertTrue(third.headers.get("retry-after"))
        other = self.post(STANDALONE, body, headers={**AUTH, "X-Forwarded-For": "10.0.0.9"})
        self.assertEqual(other.status_code, 200)
        self.assertEqual(len(self.service.calls), 3)

    def test_an_oversized_body_is_refused_from_its_headers(self) -> None:
        body = {"question": "q" * (routes.MAX_BODY_BYTES + 100)}
        response = self.post(PAIR, body)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"]["code"], "request_too_large")
        self.assert_nothing_generated()

    def test_a_body_without_a_length_is_refused(self) -> None:
        response = self.client.post(
            PAIR,
            content=iter([json.dumps({"question": QUESTION}).encode("utf-8")]),
            headers={**AUTH, "Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 411)
        self.assertEqual(response.json()["detail"]["code"], "length_required")
        self.assert_nothing_generated()

    def test_the_question_length_is_capped(self) -> None:
        over = self.post(STANDALONE, {"question": "q" * (routes.MAX_QUESTION_CHARS + 1), "conditions": ["base"]})
        self.assertEqual(over.status_code, 422)
        self.assertEqual(self.service.calls, [])
        at_cap = self.post(STANDALONE, {"question": "q" * routes.MAX_QUESTION_CHARS, "conditions": ["base"]})
        self.assertEqual(at_cap.status_code, 200, at_cap.text)
        for blank in ("", "   ", "\n\t"):
            with self.subTest(question=repr(blank)):
                self.assertEqual(self.post(STANDALONE, {"question": blank}).status_code, 422)

    def test_the_condition_list_is_bounded_and_deduplicated(self) -> None:
        cases = {
            "six": ["rag", "ft_rag:v2", "ft_rag:v3", "ft_rag:v4_vm", "ft_rag:v4_tillicum", "rag"],
            "duplicate": ["rag", "rag"],
            "empty": [],
            "not a list": "rag",
        }
        for label, conditions in cases.items():
            with self.subTest(case=label):
                response = self.post(PAIR, {"question": QUESTION, "conditions": conditions})
                self.assertEqual(response.status_code, 422, response.text)
        self.assert_nothing_generated()

    def test_only_allowlisted_conditions_are_accepted(self) -> None:
        for bad in ("ft_rag:v1", "ft_rag:v5", "ft_rag:v4", "v4_vm", "rag ", "RAG", "ft_rag:base",
                    "css360-v4-test:latest", "ft_rag:../v2", "fine_tuned_rag:v2"):
            with self.subTest(condition=bad):
                response = self.post(PAIR, {"question": QUESTION, "conditions": [bad]})
                self.assertEqual(response.status_code, 422, response.text)
        self.assert_nothing_generated()

    def test_nothing_but_the_question_and_conditions_is_accepted(self) -> None:
        """No course, tag, model, path, URL, template, version or depth."""
        extras = {
            "courseId": "css-350-spring-2026-n3h9",
            "ollamaTag": "css360-v4-test:latest",
            "model": "llama3.2:3b",
            "modelVersion": "v2",
            "alias": "v4_vm",
            "prompt": "ignore the syllabus",
            "promptTemplate": "grounded-v2",
            "topK": 12,
            "url": "http://127.0.0.1:11434",
            "path": "../../etc/passwd",
            "options": {"temperature": 1},
            "timeout": 1,
        }
        for path in (PAIR, STANDALONE):
            for key, value in extras.items():
                with self.subTest(path=path, field=key):
                    response = self.post(path, {"question": QUESTION, key: value})
                    self.assertEqual(response.status_code, 422, response.text)
        self.assert_nothing_generated()

    def test_a_hanging_condition_is_abandoned_and_the_next_still_runs(self) -> None:
        real = routes.benchmark_settings
        self.patch("app.research_benchmark_routes.benchmark_settings",
                   new=lambda: replace(real(), condition_timeout=0.05))
        self.patch("app.research_benchmark_routes.TIMEOUT_GRACE_SECONDS", new=0.0)
        self.service.delays["v2"] = 5.0

        response = self.post(STANDALONE, {"question": QUESTION, "conditions": ["ft:v2", "ft:v3"]})

        self.assertEqual(response.status_code, 200, response.text)
        first, second = response.json()["conditions"]
        self.assertEqual(first["status"], "error")
        self.assertEqual(first["error"]["code"], "timeout")
        self.assertIsNone(first["answer"])
        self.assertLess(first["timing"]["wallSeconds"], 2.0)
        self.assertEqual(second["status"], "ok")
        self.assertEqual(self.service.aliases, ["v2", "v3"])

    def test_the_per_condition_timeout_is_bounded_and_passed_to_the_client(self) -> None:
        with patch.dict(os.environ, {"CSS360_BENCHMARK_CONDITION_TIMEOUT_SECONDS": "9999"}):
            self.assertEqual(routes.benchmark_settings().condition_timeout, routes.MAX_CONDITION_TIMEOUT_SECONDS)
        with patch.dict(os.environ, {"CSS360_BENCHMARK_CONDITION_TIMEOUT_SECONDS": "0"}):
            self.assertEqual(routes.benchmark_settings().condition_timeout, routes.MIN_CONDITION_TIMEOUT_SECONDS)
        with patch.dict(os.environ, {"CSS360_BENCHMARK_CONDITION_TIMEOUT_SECONDS": "lots"}):
            self.assertEqual(routes.benchmark_settings().condition_timeout, routes.DEFAULT_CONDITION_TIMEOUT_SECONDS)
        self.enable(CSS360_BENCHMARK_CONDITION_TIMEOUT_SECONDS="45")
        response = self.post(STANDALONE, {"question": QUESTION, "conditions": ["base"]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["conditionTimeoutSeconds"], 45.0)
        self.assertEqual(self.service.calls[0]["timeout"], 45.0)

    def test_the_slot_refuses_rather_than_queues(self) -> None:
        async def scenario() -> None:
            async with routes.benchmark_slot(1):
                with self.assertRaises(HTTPException) as caught:
                    async with routes.benchmark_slot(1):
                        pass
                self.assertEqual(caught.exception.status_code, 429)
                self.assertEqual(caught.exception.detail["code"], "busy")
                self.assertEqual(caught.exception.headers["Retry-After"], "5")
            # Released on exit.
            async with routes.benchmark_slot(1):
                pass
            # A larger capacity allows that many.
            routes.reset_research_benchmark_state_for_tests()
            async with routes.benchmark_slot(2):
                async with routes.benchmark_slot(2):
                    with self.assertRaises(HTTPException):
                        async with routes.benchmark_slot(2):
                            pass

        asyncio.run(scenario())

    def test_a_second_request_while_one_runs_is_429_busy(self) -> None:
        self.service.block_alias = "base"
        outcome: dict[str, Any] = {}

        def first_request() -> None:
            outcome["response"] = self.post(PAIR, {"question": QUESTION, "conditions": ["rag"]})

        thread = threading.Thread(target=first_request)
        thread.start()
        try:
            self.assertTrue(self.service.entered.wait(5.0), "the first request never reached the service")
            second = self.post(STANDALONE, {"question": QUESTION, "conditions": ["ft:v2"]})
        finally:
            self.service.release.set()
            thread.join(5.0)
        self.assertEqual(second.status_code, 429)
        self.assertEqual(second.json()["detail"]["code"], "busy")
        self.assertEqual(outcome["response"].status_code, 200, outcome["response"].text)
        self.assertEqual(self.service.aliases, ["base"])

    def test_the_concurrency_and_rate_settings_are_bounded(self) -> None:
        with patch.dict(os.environ, {"CSS360_BENCHMARK_MAX_CONCURRENT": "50",
                                     "CSS360_BENCHMARK_REQUESTS_PER_MINUTE": "100000"}):
            settings = routes.benchmark_settings()
        self.assertEqual(settings.max_concurrent, routes.MAX_MAX_CONCURRENT)
        self.assertEqual(settings.requests_per_minute, routes.MAX_REQUESTS_PER_MINUTE)
        with patch.dict(os.environ, {"CSS360_BENCHMARK_MAX_CONCURRENT": "0",
                                     "CSS360_BENCHMARK_REQUESTS_PER_MINUTE": "-3"}):
            settings = routes.benchmark_settings()
        self.assertEqual(settings.max_concurrent, 1)
        self.assertEqual(settings.requests_per_minute, 1)


# --------------------------------------------------------------------------- #
# 4. One retrieval, one prompt, one decoding: the pair route
# --------------------------------------------------------------------------- #


class PairTests(BenchmarkRouteTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.enable()

    def test_all_five_conditions_by_default_from_one_retrieval_and_one_prompt(self) -> None:
        response = self.post(PAIR, {"question": QUESTION})

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["route"], "pair")
        self.assertEqual(body["courseId"], COURSE)
        self.assertEqual(body["question"], QUESTION)
        self.assertEqual([c["condition"] for c in body["conditions"]], list(routes.PAIR_CONDITIONS))
        self.assertEqual([c["kind"] for c in body["conditions"]], ["rag", "ft_rag", "ft_rag", "ft_rag", "ft_rag"])
        self.assertTrue(all(c["status"] == "ok" for c in body["conditions"]))
        self.assertEqual([c["answer"] for c in body["conditions"]],
                         [f"Answer from {a}." for a in ("base", "v2", "v3", "v4_vm", "v4_tillicum")])

        self.retrieve.assert_awaited_once_with(COURSE, QUESTION, top_k=DEFAULT_TOP_K)
        self.assertEqual(self.service.aliases, ["base", "v2", "v3", "v4_vm", "v4_tillicum"])
        expected_prompt = prepared()["prompt"]
        self.assertEqual(body["prompt"], expected_prompt)
        self.assertEqual(self.service.prompts, [expected_prompt] * 5)
        self.assertEqual({call["timeout"] for call in self.service.calls}, {120.0})

    def test_the_hashes_name_exactly_the_prompt_and_the_ordered_chunks(self) -> None:
        body = self.post(PAIR, {"question": QUESTION}).json()

        self.assertEqual(body["promptSha256"], sha(body["prompt"]))
        canonical = json.dumps(
            [{"chunkId": c["chunkId"], "section": c["section"], "text": c["text"]} for c in CHUNKS],
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        self.assertEqual(body["retrieval"]["setSha256"], sha(canonical))
        self.assertEqual(body["retrieval"], {
            "topK": DEFAULT_TOP_K, "chunkCount": 2, "chunkIds": ["c1", "c2"],
            "setSha256": sha(canonical), "facets": [],
        })
        self.assertEqual(body["retrievedChunks"], CHUNKS)
        self.assertEqual(body["promptTemplate"], {"name": PROMPT_TEMPLATE_NAME,
                                                  "sha256": prompt_template_fingerprint()})
        self.assertEqual(body["promptTemplate"]["name"], "grounded-v1")

    def test_the_decoding_block_is_the_shared_grounded_recipe(self) -> None:
        body = self.post(PAIR, {"question": QUESTION}).json()
        self.assertEqual(body["decoding"], grounded_options())
        self.assertEqual(body["decoding"]["num_predict"], GROUNDED_NUM_PREDICT)
        self.assertEqual(body["decoding"]["num_ctx"], GROUNDED_NUM_CTX)
        self.assertEqual(body["decoding"]["temperature"], 0)
        self.assertTrue(all(c["decodingMatchesSpec"] for c in body["conditions"]))
        self.assertTrue(all(c["promptEchoMatches"] for c in body["conditions"]))

    def test_reordering_the_chunks_changes_the_set_hash_and_scores_do_not(self) -> None:
        forward = routes.retrieval_set_sha256(CHUNKS)
        self.assertNotEqual(forward, routes.retrieval_set_sha256(list(reversed(CHUNKS))))
        rescored = [{**c, "score": 0.0} for c in CHUNKS]
        self.assertEqual(forward, routes.retrieval_set_sha256(rescored))

    def test_lineage_metadata_comes_from_the_record(self) -> None:
        body = self.post(PAIR, {"question": QUESTION}).json()
        by_alias = {c["alias"]: c for c in body["conditions"]}
        for alias, condition in by_alias.items():
            with self.subTest(alias=alias):
                self.assertEqual(condition["lineage"], lineage_for_alias(alias, LINEAGE))
                self.assertEqual(tuple(condition["lineage"]), LINEAGE_SUMMARY_FIELDS)
                self.assertEqual(condition["servedTag"], EXPECTED_TAGS[alias])
                self.assertIs(condition["tagMatchesLineage"], True)
        v4_vm = by_alias["v4_vm"]["lineage"]
        self.assertEqual(v4_vm["lineageId"], "css360-v4-vm")
        self.assertIsNone(v4_vm["version"])
        self.assertEqual(v4_vm["expectedTag"], "css360-v4-test:latest")
        self.assertEqual(v4_vm["adapterSha256"],
                         "1e845b5a9a1e3af37e963bb274d40b0bf77488128f435275d6366d7a666dde04")
        self.assertEqual(v4_vm["ggufBlobSha256"],
                         "3c497e878699d8446768f4ab028869cd1b4818682f2c5d9e65b3db797b68373a")
        self.assertEqual(v4_vm["ggufDtype"], "F16")
        self.assertEqual(v4_vm["adapterDtype"], "BF16")
        self.assertIn("mixed-v4", v4_vm["trainingLabel"])
        self.assertEqual(by_alias["v2"]["lineage"]["role"], "production")
        self.assertEqual(by_alias["v4_tillicum"]["lineage"]["lineageId"], "css360-v4-tillicum")
        self.assertEqual(by_alias["base"]["lineage"]["role"], "base")
        self.assertEqual(body["lineageRecord"], {
            "title": "CSS 360 model lineage", "schemaVersion": 1,
            "generatedAt": LINEAGE["generatedAt"], "courseId": COURSE,
        })

    def test_timings_are_reported_per_condition(self) -> None:
        body = self.post(PAIR, {"question": QUESTION, "conditions": ["rag"]}).json()
        timing = body["conditions"][0]["timing"]
        self.assertGreaterEqual(timing["wallSeconds"], 0.0)
        self.assertEqual(timing["generationSeconds"], 0.5)
        self.assertEqual(timing["ollama"], {
            "totalDurationNs": None, "loadDurationNs": 1000, "promptEvalCount": None,
            "promptEvalDurationNs": None, "evalCount": 12, "evalDurationNs": None,
            "doneReason": "stop",
        })
        self.assertGreaterEqual(body["requestSeconds"], 0.0)

    def test_a_subset_runs_in_the_requested_order(self) -> None:
        body = self.post(PAIR, {"question": QUESTION, "conditions": ["ft_rag:v4_tillicum", "rag"]}).json()
        self.assertEqual([c["condition"] for c in body["conditions"]], ["ft_rag:v4_tillicum", "rag"])
        self.assertEqual(self.service.aliases, ["v4_tillicum", "base"])
        self.retrieve.assert_awaited_once()

    def test_the_course_and_depth_are_fixed_server_side(self) -> None:
        self.post(PAIR, {"question": QUESTION, "conditions": ["rag"]})
        self.retrieve.assert_awaited_once_with(COURSE, QUESTION, top_k=DEFAULT_TOP_K)
        self.assertEqual(DEFAULT_TOP_K, 4)

    def test_the_question_is_whitespace_normalised_once(self) -> None:
        body = self.post(PAIR, {"question": "  When   are\noffice hours? ", "conditions": ["rag"]}).json()
        self.assertEqual(body["question"], QUESTION)
        self.retrieve.assert_awaited_once_with(COURSE, QUESTION, top_k=DEFAULT_TOP_K)

    def test_decoding_drift_is_an_error_and_not_an_answer(self) -> None:
        self.service.options["v3"] = {**grounded_options(), "num_predict": 160}
        body = self.post(PAIR, {"question": QUESTION}).json()
        by_alias = {c["alias"]: c for c in body["conditions"]}
        drifted = by_alias["v3"]
        self.assertEqual(drifted["status"], "error")
        self.assertEqual(drifted["error"]["code"], "decoding_mismatch")
        self.assertIsNone(drifted["answer"])
        self.assertIs(drifted["decodingMatchesSpec"], False)
        self.assertTrue(all(c["status"] == "ok" for a, c in by_alias.items() if a != "v3"))

    def test_retrieval_failures_are_the_requests_failure(self) -> None:
        self.retrieve.side_effect = HTTPException(status_code=404, detail='No syllabus index found for course "x".')
        response = self.post(PAIR, {"question": QUESTION})
        self.assertEqual(response.status_code, 409)
        self.assertIn("no usable syllabus index", response.json()["detail"])

        self.retrieve.side_effect = HTTPException(status_code=503, detail="Ollama embedding request timed out.")
        response = self.post(PAIR, {"question": QUESTION})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Ollama embedding request timed out.")
        self.assertEqual(self.service.calls, [])

    def test_the_lineage_record_is_required(self) -> None:
        self.patch("app.research_benchmark_routes.load_lineage",
                   side_effect=LineageUnavailable("The lineage record could not be read."))
        response = self.post(PAIR, {"question": QUESTION})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "The lineage record could not be read.")
        self.assert_nothing_generated()

    def test_an_alias_missing_from_the_record_errors_only_that_condition(self) -> None:
        record = json.loads(json.dumps(LINEAGE))
        record["artifacts"] = [a for a in record["artifacts"] if a["lineageId"] != "css360-v4-tillicum"]
        self.patch("app.research_benchmark_routes.load_lineage", return_value=record)

        body = self.post(PAIR, {"question": QUESTION}).json()
        by_alias = {c["alias"]: c for c in body["conditions"]}
        missing = by_alias["v4_tillicum"]
        self.assertEqual(missing["status"], "error")
        self.assertEqual(missing["error"]["code"], "lineage_missing")
        self.assertIn("css360-v4-tillicum", missing["error"]["message"])
        self.assertIsNone(missing["lineage"])
        self.assertNotIn("v4_tillicum", self.service.aliases)
        self.assertTrue(all(c["status"] == "ok" for a, c in by_alias.items() if a != "v4_tillicum"))


# --------------------------------------------------------------------------- #
# 5. Per-condition isolation
# --------------------------------------------------------------------------- #


class IsolationTests(BenchmarkRouteTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.enable()

    def test_one_failing_condition_does_not_fail_the_request(self) -> None:
        self.service.failures["v3"] = BenchmarkConditionError(
            "service_unavailable", "The benchmark inference service is unavailable."
        )
        response = self.post(PAIR, {"question": QUESTION})

        self.assertEqual(response.status_code, 200, response.text)
        by_alias = {c["alias"]: c for c in response.json()["conditions"]}
        failed = by_alias["v3"]
        self.assertEqual(failed["status"], "error")
        self.assertEqual(failed["error"], {"code": "service_unavailable",
                                           "message": "The benchmark inference service is unavailable."})
        self.assertIsNone(failed["answer"])
        self.assertIsNone(failed["servedTag"])
        self.assertIsNotNone(failed["lineage"])
        self.assertGreaterEqual(failed["timing"]["wallSeconds"], 0.0)
        self.assertIsNone(failed["timing"]["generationSeconds"])
        for alias, condition in by_alias.items():
            if alias != "v3":
                self.assertEqual(condition["status"], "ok", alias)
        self.assertEqual(len(self.service.calls), 5)

    def test_an_unexpected_exception_is_contained_and_logged(self) -> None:
        self.service.failures["v2"] = RuntimeError("boom at /home/testuser/.ollama/models")
        with self.assertLogs("app.research_benchmark_routes", level="ERROR") as logs:
            response = self.post(PAIR, {"question": QUESTION, "conditions": ["ft_rag:v2", "rag"]})

        self.assertEqual(response.status_code, 200)
        failed, fine = response.json()["conditions"]
        self.assertEqual(failed["error"]["code"], "unexpected")
        self.assertNotIn("boom", failed["error"]["message"])
        self.assertNotIn("testuser", json.dumps(response.json()))
        self.assertEqual(fine["status"], "ok")
        self.assertIn("boom at /home/testuser", "\n".join(logs.output))

    def test_every_condition_failing_is_still_a_200_with_five_errors(self) -> None:
        for alias in EXPECTED_TAGS:
            self.service.failures[alias] = BenchmarkConditionError("service_timeout", "The benchmark inference service timed out.")
        response = self.post(PAIR, {"question": QUESTION})
        self.assertEqual(response.status_code, 200)
        self.assertEqual([c["error"]["code"] for c in response.json()["conditions"]], ["service_timeout"] * 5)


# --------------------------------------------------------------------------- #
# 5b. Invalid and failed conditions: unscored, but timed and explained
# --------------------------------------------------------------------------- #


class InvalidAndFailedConditionTests(BenchmarkRouteTestCase):
    """A served tag the record does not name makes the condition invalid; an
    answer with no text makes the generation a failure. Neither is scored:
    both are `status: error` with no answer. Both keep what was measured —
    wall time, the service's generation time, Ollama's accounting — and the
    served tag, the flags and the lineage, so a saved result says exactly
    what happened and is distinguishable from a service that never answered.
    """

    def setUp(self) -> None:
        super().setUp()
        self.enable()

    def test_a_tag_mismatch_is_an_invalid_unscored_condition_that_keeps_its_timing(self) -> None:
        self.service.tags["v4_vm"] = "css360-v4-reimport:latest"
        response = self.post(PAIR, {"question": QUESTION})

        self.assertEqual(response.status_code, 200, response.text)
        by_alias = {c["alias"]: c for c in response.json()["conditions"]}
        invalid = by_alias["v4_vm"]
        self.assertEqual(invalid["status"], "error")
        self.assertEqual(invalid["error"]["code"], "tag_mismatch")
        self.assertIn("lineage record", invalid["error"]["message"])
        self.assertIsNone(invalid["answer"])
        # What is kept: the tag that answered, the flags, the lineage, the timing.
        self.assertEqual(invalid["servedTag"], "css360-v4-reimport:latest")
        self.assertIs(invalid["tagMatchesLineage"], False)
        self.assertIs(invalid["decodingMatchesSpec"], True)
        self.assertIs(invalid["promptEchoMatches"], True)
        self.assertEqual(invalid["lineage"]["lineageId"], "css360-v4-vm")
        self.assertEqual(invalid["lineage"]["expectedTag"], "css360-v4-test:latest")
        self.assertGreaterEqual(invalid["timing"]["wallSeconds"], 0.0)
        self.assertEqual(invalid["timing"]["generationSeconds"], 0.5)
        self.assertEqual(invalid["timing"]["ollama"]["evalCount"], 12)
        self.assertEqual(invalid["timing"]["ollama"]["doneReason"], "stop")
        # The other four are untouched, and every condition was still run.
        for alias, condition in by_alias.items():
            if alias != "v4_vm":
                self.assertEqual(condition["status"], "ok", alias)
                self.assertIs(condition["tagMatchesLineage"], True, alias)
        self.assertEqual(len(self.service.calls), 5)

    def test_the_same_tag_in_another_spelling_is_not_a_mismatch(self) -> None:
        self.service.tags["v2"] = "css360-ft-v2"  # `:latest` implied, as Ollama reports it
        self.service.tags["base"] = "llama3.2:3b"
        body = self.post(PAIR, {"question": QUESTION, "conditions": ["rag", "ft_rag:v2"]}).json()
        for condition in body["conditions"]:
            self.assertEqual(condition["status"], "ok", condition["alias"])
            self.assertIs(condition["tagMatchesLineage"], True, condition["alias"])
            self.assertIsNotNone(condition["answer"])

    def test_an_empty_answer_is_a_failed_unscored_generation_that_keeps_its_timing(self) -> None:
        self.service.answers["v3"] = ""
        self.service.ollama["v3"] = {"evalCount": 0, "loadDurationNs": 4200, "doneReason": "stop"}
        response = self.post(PAIR, {"question": QUESTION})

        self.assertEqual(response.status_code, 200, response.text)
        by_alias = {c["alias"]: c for c in response.json()["conditions"]}
        failed = by_alias["v3"]
        self.assertEqual(failed["status"], "error")
        self.assertEqual(failed["error"]["code"], "empty_answer")
        self.assertIn("no answer text", failed["error"]["message"])
        self.assertIsNone(failed["answer"])
        # A generation that ran and produced nothing: the tag, the flags, the
        # lineage and the whole timing record survive.
        self.assertEqual(failed["servedTag"], "css360-cpu-v3-test:latest")
        self.assertIs(failed["tagMatchesLineage"], True)
        self.assertIs(failed["decodingMatchesSpec"], True)
        self.assertIs(failed["promptEchoMatches"], True)
        self.assertEqual(failed["lineage"]["lineageId"], "css360-v3")
        self.assertGreaterEqual(failed["timing"]["wallSeconds"], 0.0)
        self.assertEqual(failed["timing"]["generationSeconds"], 0.5)
        self.assertEqual(failed["timing"]["ollama"], {
            "totalDurationNs": None, "loadDurationNs": 4200, "promptEvalCount": None,
            "promptEvalDurationNs": None, "evalCount": 0, "evalDurationNs": None,
            "doneReason": "stop",
        })
        for alias, condition in by_alias.items():
            if alias != "v3":
                self.assertEqual(condition["status"], "ok", alias)
        self.assertEqual(len(self.service.calls), 5)

    def test_whitespace_only_is_an_empty_answer_on_either_route(self) -> None:
        for path, condition, blank in ((STANDALONE, "ft:v2", "   "), (PAIR, "ft_rag:v2", "\n\t "),
                                       (STANDALONE, "base", "")):
            with self.subTest(path=path, answer=repr(blank)):
                self.service.answers[routes.split_condition(condition)[1]] = blank
                body = self.post(path, {"question": QUESTION, "conditions": [condition]}).json()
                (failed,) = body["conditions"]
                self.assertEqual(failed["status"], "error")
                self.assertEqual(failed["error"]["code"], "empty_answer")
                self.assertIsNone(failed["answer"])
                self.assertEqual(failed["timing"]["generationSeconds"], 0.5)

    def test_a_tag_mismatch_outranks_an_empty_answer(self) -> None:
        """Both at once: the wrong artifact is the more fundamental fact."""
        self.service.tags["v2"] = "css360-other:latest"
        self.service.answers["v2"] = ""
        body = self.post(PAIR, {"question": QUESTION, "conditions": ["ft_rag:v2"]}).json()
        (condition,) = body["conditions"]
        self.assertEqual(condition["error"]["code"], "tag_mismatch")
        self.assertIs(condition["tagMatchesLineage"], False)
        self.assertIsNone(condition["answer"])

    def test_invalid_and_failed_conditions_are_distinguishable_from_service_failures(self) -> None:
        """Three kinds of non-answer in one request, each with its own code,
        and only the two that reached Ollama carry a generation time."""
        self.service.tags["v4_tillicum"] = "css360-v4-tillicum-retry:latest"
        self.service.answers["v4_vm"] = ""
        self.service.failures["v3"] = BenchmarkConditionError(
            "service_unavailable", "The benchmark inference service is unavailable."
        )
        response = self.post(PAIR, {"question": QUESTION})

        self.assertEqual(response.status_code, 200, response.text)
        by_alias = {c["alias"]: c for c in response.json()["conditions"]}
        self.assertEqual(
            {alias: (c["error"] or {}).get("code") for alias, c in by_alias.items()},
            {"base": None, "v2": None, "v3": "service_unavailable",
             "v4_vm": "empty_answer", "v4_tillicum": "tag_mismatch"},
        )
        self.assertIsNone(by_alias["v3"]["timing"]["generationSeconds"])
        self.assertIsNone(by_alias["v3"]["servedTag"])
        self.assertEqual(by_alias["v4_vm"]["timing"]["generationSeconds"], 0.5)
        self.assertEqual(by_alias["v4_vm"]["servedTag"], "css360-v4-test:latest")
        self.assertEqual(by_alias["v4_tillicum"]["timing"]["generationSeconds"], 0.5)
        self.assertEqual(by_alias["v4_tillicum"]["servedTag"], "css360-v4-tillicum-retry:latest")
        self.assertTrue(all(c["answer"] is None for a, c in by_alias.items() if a not in ("base", "v2")))

    def test_invalid_and_failed_entries_carry_nothing_internal(self) -> None:
        self.service.tags["v2"] = "css360-other:latest"
        self.service.answers["v3"] = ""
        self.service.digests["v4_vm"] = "e" * 64
        response = self.post(PAIR, {"question": QUESTION})
        self.assertEqual(response.status_code, 200)
        assert_private(self, response.json())


# --------------------------------------------------------------------------- #
# 5c. Digests recorded, outcomes classed, every attempt counted
# --------------------------------------------------------------------------- #


class DigestAndOutcomeTests(BenchmarkRouteTestCase):
    """Each answer names the Ollama digest that produced it, beside the tag.
    Each condition carries an outcome, and the response a summary whose
    `attempted` counts every requested condition, so a later report keeps
    failed and invalid generations in its denominator."""

    def setUp(self) -> None:
        super().setUp()
        self.enable()

    def test_every_condition_records_the_served_digest_and_compares_it_with_the_record(self) -> None:
        body = self.post(PAIR, {"question": QUESTION}).json()
        for condition in body["conditions"]:
            alias = condition["alias"]
            with self.subTest(alias=alias):
                self.assertEqual(condition["servedDigest"], full_digest(alias))
                self.assertIs(condition["digestMatchesLineage"], True)
                self.assertEqual(condition["lineage"]["ollamaDigest"], EXPECTED_DIGESTS[alias])
                self.assertEqual(condition["outcome"], "scorable")

    def test_a_digest_the_record_does_not_name_is_an_invalid_unscored_condition(self) -> None:
        """Same tag, different bytes: the answer cannot be attributed to the
        recorded artifact, so the condition is invalid. The observed digest,
        the matching tag, the lineage and the timing all stay."""
        self.service.digests["v2"] = "e" * 64
        response = self.post(PAIR, {"question": QUESTION, "conditions": ["ft_rag:v2", "rag"]})

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        rebuilt, control = body["conditions"]
        self.assertEqual(rebuilt["status"], "error")
        self.assertEqual(rebuilt["error"]["code"], "digest_mismatch")
        self.assertIn("digest", rebuilt["error"]["message"])
        self.assertEqual(rebuilt["outcome"], "invalid")
        self.assertIsNone(rebuilt["answer"])
        # Kept: the digest that was observed, the tag that did match, the
        # other flags, the lineage, and the whole timing record.
        self.assertEqual(rebuilt["servedDigest"], "e" * 64)
        self.assertIs(rebuilt["digestMatchesLineage"], False)
        self.assertEqual(rebuilt["servedTag"], "css360-ft-v2:latest")
        self.assertIs(rebuilt["tagMatchesLineage"], True)
        self.assertIs(rebuilt["decodingMatchesSpec"], True)
        self.assertIs(rebuilt["promptEchoMatches"], True)
        self.assertEqual(rebuilt["lineage"]["ollamaDigest"], EXPECTED_DIGESTS["v2"])
        self.assertGreaterEqual(rebuilt["timing"]["wallSeconds"], 0.0)
        self.assertEqual(rebuilt["timing"]["generationSeconds"], 0.5)
        self.assertEqual(rebuilt["timing"]["ollama"]["doneReason"], "stop")
        # The control is untouched and the attempt is still counted.
        self.assertEqual(control["status"], "ok")
        self.assertIs(control["digestMatchesLineage"], True)
        self.assertEqual(body["summary"], {
            "attempted": 2, "scorable": 1, "failedGenerations": 0, "invalid": 1, "errors": 0,
        })
        self.assertEqual(len(self.service.calls), 2)

    def test_a_tag_mismatch_outranks_a_digest_mismatch(self) -> None:
        self.service.tags["v2"] = "css360-other:latest"
        self.service.digests["v2"] = "e" * 64
        body = self.post(PAIR, {"question": QUESTION, "conditions": ["ft_rag:v2"]}).json()
        (condition,) = body["conditions"]
        self.assertEqual(condition["error"]["code"], "tag_mismatch")
        self.assertEqual(condition["servedDigest"], "e" * 64)
        self.assertIs(condition["digestMatchesLineage"], False)

    def test_a_digest_mismatch_outranks_an_empty_answer(self) -> None:
        self.service.digests["v2"] = "e" * 64
        self.service.answers["v2"] = ""
        body = self.post(PAIR, {"question": QUESTION, "conditions": ["ft_rag:v2"]}).json()
        (condition,) = body["conditions"]
        self.assertEqual(condition["error"]["code"], "digest_mismatch")
        self.assertEqual(condition["outcome"], "invalid")
        self.assertEqual(condition["timing"]["generationSeconds"], 0.5)

    def test_a_digest_mismatch_on_the_standalone_route_is_the_same_error(self) -> None:
        self.service.digests["base"] = "e" * 64
        body = self.post(STANDALONE, {"question": QUESTION, "conditions": ["base"]}).json()
        (condition,) = body["conditions"]
        self.assertEqual(condition["error"]["code"], "digest_mismatch")
        self.assertEqual(condition["servedDigest"], "e" * 64)
        self.assertEqual(body["summary"]["invalid"], 1)

    def test_a_missing_digest_is_recorded_as_unknown_not_as_a_mismatch(self) -> None:
        self.service.digests["v3"] = None
        body = self.post(PAIR, {"question": QUESTION, "conditions": ["ft_rag:v3"]}).json()
        (condition,) = body["conditions"]
        self.assertEqual(condition["status"], "ok")
        self.assertIsNone(condition["servedDigest"])
        self.assertIsNone(condition["digestMatchesLineage"])

    def test_digest_comparison_accepts_either_length_and_blob_prefixes(self) -> None:
        self.assertIs(routes.digest_matches("ab" * 32, "abab"), True)
        self.assertIs(routes.digest_matches("abab", "ab" * 32), True)
        self.assertIs(routes.digest_matches("sha256:" + "ab" * 32, "ABAB"), True)
        self.assertIs(routes.digest_matches("cd" * 32, "abab"), False)
        self.assertIsNone(routes.digest_matches(None, "abab"))
        self.assertIsNone(routes.digest_matches("abab", None))
        self.assertIsNone(routes.digest_matches("", "abab"))

    def test_the_digest_survives_invalid_and_failed_conditions(self) -> None:
        self.service.tags["v4_vm"] = "css360-v4-reimport:latest"
        self.service.answers["v3"] = ""
        body = self.post(PAIR, {"question": QUESTION}).json()
        by_alias = {c["alias"]: c for c in body["conditions"]}
        self.assertEqual(by_alias["v4_vm"]["servedDigest"], full_digest("v4_vm"))
        self.assertEqual(by_alias["v3"]["servedDigest"], full_digest("v3"))
        self.assertIsNone(by_alias["v3"]["answer"])

    def test_outcomes_and_summary_count_every_attempt(self) -> None:
        self.service.tags["v4_tillicum"] = "css360-v4-tillicum-retry:latest"   # invalid
        self.service.answers["v4_vm"] = ""                                       # failed generation
        self.service.failures["v3"] = BenchmarkConditionError(                  # error
            "service_unavailable", "The benchmark inference service is unavailable."
        )
        response = self.post(PAIR, {"question": QUESTION})

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(
            {c["alias"]: c["outcome"] for c in body["conditions"]},
            {"base": "scorable", "v2": "scorable", "v3": "error",
             "v4_vm": "failed_generation", "v4_tillicum": "invalid"},
        )
        self.assertEqual(body["summary"], {
            "attempted": 5, "scorable": 2, "failedGenerations": 1, "invalid": 1, "errors": 1,
        })
        summary = body["summary"]
        self.assertEqual(summary["attempted"], len(body["conditions"]))
        self.assertEqual(summary["attempted"],
                         summary["scorable"] + summary["failedGenerations"] + summary["invalid"] + summary["errors"])

    def test_every_error_code_has_an_outcome_class(self) -> None:
        cases = {
            "empty_answer": "failed_generation",
            "tag_mismatch": "invalid",
            "digest_mismatch": "invalid",
            "decoding_mismatch": "invalid",
            "prompt_integrity": "invalid",
            "alias_mismatch": "invalid",
            "service_unavailable": "error",
            "service_timeout": "error",
            "service_error": "error",
            "alias_not_mapped": "error",
            "lineage_missing": "error",
            "timeout": "error",
            "unexpected": "error",
            "malformed_response": "error",
        }
        for code, outcome in cases.items():
            with self.subTest(code=code):
                self.assertEqual(routes.outcome_for("error", code), outcome)
        self.assertEqual(routes.outcome_for("ok", None), "scorable")

    def test_a_clean_run_and_a_failed_run_both_count_every_condition(self) -> None:
        clean = self.post(STANDALONE, {"question": QUESTION}).json()
        self.assertEqual(clean["summary"], {"attempted": 5, "scorable": 5, "failedGenerations": 0, "invalid": 0, "errors": 0})
        for alias in EXPECTED_TAGS:
            self.service.answers[alias] = ""
        failed = self.post(STANDALONE, {"question": QUESTION}).json()
        self.assertEqual(failed["summary"], {"attempted": 5, "scorable": 0, "failedGenerations": 5, "invalid": 0, "errors": 0})
        self.assertTrue(all(c["outcome"] == "failed_generation" for c in failed["conditions"]))
        self.assertTrue(all(c["timing"]["generationSeconds"] == 0.5 for c in failed["conditions"]))

    def test_a_subset_counts_only_what_was_requested(self) -> None:
        body = self.post(PAIR, {"question": QUESTION, "conditions": ["rag", "ft_rag:v4_vm"]}).json()
        self.assertEqual(body["summary"]["attempted"], 2)
        self.assertEqual(body["summary"]["scorable"], 2)


# --------------------------------------------------------------------------- #
# 6. The standalone route
# --------------------------------------------------------------------------- #


class StandaloneTests(BenchmarkRouteTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.enable()

    def test_no_retrieval_and_the_bare_question_as_the_prompt(self) -> None:
        response = self.post(STANDALONE, {"question": "  When are   office hours? "})

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["route"], "standalone")
        self.assertEqual(body["question"], QUESTION)
        self.assertEqual(body["prompt"], QUESTION)
        self.assertEqual(body["promptSha256"], sha(QUESTION))
        self.assertIsNone(body["promptTemplate"])
        self.assertIsNone(body["retrieval"])
        self.assertEqual(body["retrievedChunks"], [])
        self.assertEqual(body["decoding"], grounded_options())
        self.assertEqual([c["condition"] for c in body["conditions"]], list(routes.STANDALONE_CONDITIONS))
        self.assertEqual([c["kind"] for c in body["conditions"]], ["base", "ft", "ft", "ft", "ft"])
        self.retrieve.assert_not_awaited()
        self.assertEqual(self.service.aliases, ["base", "v2", "v3", "v4_vm", "v4_tillicum"])
        self.assertEqual(self.service.prompts, [QUESTION] * 5)

    def test_the_allowlists_are_per_route(self) -> None:
        for path, bad in ((STANDALONE, "rag"), (STANDALONE, "ft_rag:v2"), (PAIR, "base"), (PAIR, "ft:v2")):
            with self.subTest(path=path, condition=bad):
                self.assertEqual(self.post(path, {"question": QUESTION, "conditions": [bad]}).status_code, 422)
        self.assert_nothing_generated()


# --------------------------------------------------------------------------- #
# 7. Privacy
# --------------------------------------------------------------------------- #

#: Nothing in this list may appear in any response body, whatever happened.
FORBIDDEN = (
    TOKEN,
    "Bearer " + TOKEN,
    SERVICE_URL,
    "127.0.0.1",
    "localhost",
    ":9002",
    ":11434",
    "http://",
    "https://",
    "CSS360_BENCHMARK",
    "FINETUNED_",
    "OLLAMA_",
    "DATABASE_URL",
    "postgres",
    "course_model_versions",
    "course_models",
    "vm:",
    "tillicum:",
    "$PROJ",
    "/gpfs",
    "~/",
    "model_artifacts",
    "training_outputs",
    ".gguf",
    ".log",
    ".env",
    "Modelfile",
    "hyak",
    "aiswe",
    "g007",
    "g004",
    "sacct",
    "testuser",
)


def assert_private(case: unittest.TestCase, body: Any) -> None:
    text = json.dumps(body)
    for secret in FORBIDDEN:
        case.assertNotIn(secret, text, f"{secret!r} appeared in a response")


class PrivacyTests(BenchmarkRouteTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.enable()

    def test_a_successful_response_carries_nothing_internal(self) -> None:
        for path in (PAIR, STANDALONE):
            with self.subTest(path=path):
                response = self.post(path, {"question": QUESTION})
                self.assertEqual(response.status_code, 200)
                assert_private(self, response.json())

    def test_refusals_carry_nothing_internal(self) -> None:
        refusals = [
            ("401", self.post(PAIR, {"question": QUESTION}, headers={"Authorization": "Bearer wrong"})),
            ("413", self.post(PAIR, {"question": "q" * (routes.MAX_BODY_BYTES + 1)})),
            ("422", self.post(PAIR, {"question": QUESTION, "courseId": "x"})),
        ]
        self.retrieve.side_effect = HTTPException(status_code=404, detail="No syllabus index found.")
        refusals.append(("409", self.post(PAIR, {"question": QUESTION})))
        for _ in range(10):
            self.post(PAIR, {"question": QUESTION}, headers={"Authorization": "Bearer wrong"})
        refusals.append(("429", self.post(PAIR, {"question": QUESTION})))
        with patch.dict(os.environ, {"CSS360_BENCHMARK_ENABLED": "false"}):
            refusals.append(("404", self.post(PAIR, {"question": QUESTION})))
        for label, response in refusals:
            with self.subTest(status=label):
                self.assertEqual(response.status_code, int(label))
                assert_private(self, response.json())
                self.assertNotIn("Authorization", json.dumps(dict(response.headers)))

    def test_an_unreadable_record_does_not_name_its_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope" / "model_lineage.json"
            research_benchmark_lineage.reset_lineage_cache()
            self.addCleanup(research_benchmark_lineage.reset_lineage_cache)
            self.patch("app.research_benchmark_routes.load_lineage",
                       new=lambda: research_benchmark_lineage.load_lineage(missing))
            response = self.post(PAIR, {"question": QUESTION})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(tmp, response.text)
        self.assertNotIn("model_lineage", response.text)
        assert_private(self, response.json())

    def test_error_entries_carry_nothing_internal(self) -> None:
        self.service.failures["v2"] = RuntimeError(
            "ConnectError: http://127.0.0.1:9002/generate via /home/testuser/css360-syllabus-bot"
        )
        self.service.failures["v3"] = BenchmarkConditionError(
            "service_error", "The benchmark inference service or its Ollama returned a server error (HTTP 500)."
        )
        with self.assertLogs("app.research_benchmark_routes", level="ERROR"):
            response = self.post(PAIR, {"question": QUESTION})
        self.assertEqual(response.status_code, 200)
        assert_private(self, response.json())

    def test_the_response_keys_are_exactly_the_documented_ones(self) -> None:
        body = self.post(PAIR, {"question": QUESTION}).json()
        self.assertEqual(set(body), {
            "route", "courseId", "question", "prompt", "promptSha256", "promptTemplate",
            "retrieval", "retrievedChunks", "decoding", "conditionTimeoutSeconds",
            "lineageRecord", "conditions", "summary", "requestSeconds",
        })
        self.assertEqual(set(body["conditions"][0]), {
            "condition", "kind", "alias", "status", "outcome", "answer", "servedTag", "servedDigest",
            "tagMatchesLineage", "digestMatchesLineage", "decodingMatchesSpec", "promptEchoMatches",
            "lineage", "timing", "error",
        })
        self.assertEqual(set(body["summary"]), {"attempted", "scorable", "failedGenerations", "invalid", "errors"})
        self.assertEqual(tuple(body["conditions"][0]["lineage"]), LINEAGE_SUMMARY_FIELDS)
        self.assertEqual(set(body["retrievedChunks"][0]), {"chunkId", "section", "text", "score"})
        self.assertEqual(set(body["conditions"][0]["timing"]), {"wallSeconds", "generationSeconds", "ollama"})

    def test_the_response_models_refuse_to_widen(self) -> None:
        """A field a future edit adds by mistake fails at construction."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            routes.LineageSummary(alias="v2", expectedTag="t", runDirectory="vm:~/x")
        with self.assertRaises(ValidationError):
            routes.ConditionTiming(wallSeconds=1.0, hostname="aiswe")


# --------------------------------------------------------------------------- #
# 8. No database, no registry, no registration
# --------------------------------------------------------------------------- #


class NoDatabaseTests(BenchmarkRouteTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.enable()

    def test_the_route_opens_no_database_connection(self) -> None:
        self.patch("app.db.db_connection", side_effect=AssertionError("the benchmark route opened a connection"))
        self.patch("app.db.connect", side_effect=AssertionError("the benchmark route opened a connection"))
        for path in (PAIR, STANDALONE):
            with self.subTest(path=path):
                response = self.post(path, {"question": QUESTION})
                self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.client.post(PAIR, json={"question": QUESTION}, headers=AUTH).headers.get_list("set-cookie"), [])

    def test_the_modules_import_neither_the_registry_nor_the_database(self) -> None:
        for module in (routes, research_benchmark_client, research_benchmark_lineage):
            source = inspect.getsource(module)
            with self.subTest(module=module.__name__):
                for forbidden in ("course_model_resolution", "from app.db", "import app.db",
                                  "db_models", "db_courses", "register", "current_version"):
                    self.assertNotIn(forbidden, source)

    def test_the_experimental_aliases_have_no_version_and_are_never_called_v5(self) -> None:
        body = self.post(PAIR, {"question": QUESTION}).json()
        by_alias = {c["alias"]: c for c in body["conditions"]}
        for alias in ("v4_vm", "v4_tillicum"):
            self.assertIsNone(by_alias[alias]["lineage"]["version"])
            self.assertEqual(by_alias[alias]["lineage"]["role"], "experimental")
        self.assertNotIn("v5", json.dumps(body))


# --------------------------------------------------------------------------- #
# 9. The administrator's routes are unchanged
# --------------------------------------------------------------------------- #


class ModelTestingUnchangedTests(unittest.TestCase):
    def test_request_and_response_shapes(self) -> None:
        self.assertEqual(set(ModelTestingGenerateRequest.model_fields),
                         {"course_id", "mode", "model_version", "question", "top_k"})
        self.assertEqual(set(ModelTestingPairRequest.model_fields),
                         {"course_id", "model_version", "question", "top_k"})
        self.assertEqual(set(ModelTestingGenerateResponse.model_fields), {
            "course_id", "mode", "model_version", "answer", "model", "response_type",
            "adapter_loaded", "generation_seconds", "sources", "retrieved_chunks",
        })
        self.assertEqual(set(ModelTestingPairResponse.model_fields), {
            "course_id", "mode", "model_version", "question", "prompt", "sources",
            "retrieved_chunks", "rag", "fine_tuned_rag",
        })

    def test_classification_and_mounting(self) -> None:
        self.assertEqual(CLASSIFICATION[("POST", "/api/model-testing/generate")], REQUIRE_ADMIN)
        self.assertEqual(CLASSIFICATION[("POST", "/api/model-testing/pair")], REQUIRE_ADMIN)
        mounted = {getattr(route, "path", None) for route in app.routes}
        self.assertIn("/api/model-testing/generate", mounted)
        self.assertIn("/api/model-testing/pair", mounted)


if __name__ == "__main__":
    unittest.main()
