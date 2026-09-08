"""The local Ollama-backed inference service, against a fake Ollama.

What is pinned here is the contract the backend already depends on — the same
`/health` and `/generate` shapes `app.py` answers with — and course isolation
at its source: a course is answered by the Ollama model mapped to it and its
version, or refused. Nothing is answered by whichever model happens to exist.

No test reaches a real Ollama. `httpx.AsyncClient` is replaced with a fake that
records every request and answers from a script.
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import Any, Dict, List, Optional
from unittest import mock

import httpx
from fastapi.testclient import TestClient

import ollama_service
from helpers import CourseAdapterError

CSS360 = "css-360-winter-2026-a7rp"
CSS350 = "css-350-spring-2026-n3h9"
CSS360_MODEL = "css360-ft-v2:latest"
CSS350_MODEL = "css350-ft-v1:latest"

QUESTION = "What is the late policy?"
ANSWER = "Late work loses 10% per day, up to three days."


class _Response:
    def __init__(self, status_code: int, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or (str(payload) if payload is not None else "")

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeOllama:
    """Stands in for `httpx.AsyncClient`. Records requests; answers from a script."""

    def __init__(
        self,
        *,
        models: Optional[List[str]] = None,
        answer: str = ANSWER,
        chat_status: int = 200,
        chat_payload: Any = None,
        chat_text: str = "",
        tags_error: Optional[BaseException] = None,
        chat_error: Optional[BaseException] = None,
    ) -> None:
        self.models = list(models or [])
        self.answer = answer
        self.chat_status = chat_status
        self.chat_payload = chat_payload
        self.chat_text = chat_text
        self.tags_error = tags_error
        self.chat_error = chat_error
        self.requests: List[Dict[str, Any]] = []

    # The object `httpx.AsyncClient(...)` returns, used as an async context.
    def __call__(self, *args: Any, **kwargs: Any) -> "FakeOllama":
        return self

    async def __aenter__(self) -> "FakeOllama":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def get(self, url: str, **kwargs: Any) -> _Response:
        self.requests.append({"method": "GET", "url": url})
        if self.tags_error is not None:
            raise self.tags_error
        if url.endswith("/api/tags"):
            return _Response(200, {"models": [{"name": name} for name in self.models]})
        return _Response(404, {"error": "unknown"})

    async def post(self, url: str, *, json: Any = None, **kwargs: Any) -> _Response:
        self.requests.append({"method": "POST", "url": url, "json": json})
        if self.chat_error is not None:
            raise self.chat_error
        if self.chat_status != 200:
            return _Response(self.chat_status, {"error": self.chat_text}, self.chat_text)
        if self.chat_payload is not None:
            return _Response(200, self.chat_payload)
        return _Response(
            200,
            {
                "model": json["model"],
                "message": {"role": "assistant", "content": self.answer},
                "done": True,
                "done_reason": "stop",
            },
        )

    @property
    def chat_requests(self) -> List[Dict[str, Any]]:
        return [r for r in self.requests if r["method"] == "POST"]


def _env(mapping: Optional[str], **extra: str) -> "mock._patch_dict":
    values = {
        "INFERENCE_PORT": "",
        "FINETUNED_KEEP_ALIVE": "",
        "FINETUNED_NUM_CTX": "",
        "FINETUNED_BASE_MODEL": "",
        "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
        ollama_service.MODEL_MAP_ENV: mapping or "",
    }
    values.update(extra)
    return mock.patch.dict(os.environ, values, clear=False)


DEFAULT_MAP = f"{CSS360}@v2={CSS360_MODEL},{CSS350}@v1={CSS350_MODEL}"


class ModelMapTests(unittest.TestCase):
    def test_one_entry(self) -> None:
        parsed = ollama_service.parse_model_map(f"{CSS360}@v2=css360-ft-v2:latest")
        self.assertEqual(parsed, {CSS360: {"v2": "css360-ft-v2:latest"}})

    def test_an_untagged_model_name_is_the_latest_tag(self) -> None:
        """`/api/tags` reports `name:latest`; the mapping must compare equal."""
        parsed = ollama_service.parse_model_map(f"{CSS360}@v2=css360-ft-v2")
        self.assertEqual(parsed[CSS360]["v2"], "css360-ft-v2:latest")

    def test_a_registry_prefix_with_a_port_keeps_its_tag_logic(self) -> None:
        self.assertEqual(
            ollama_service.normalize_ollama_model_name("localhost:5000/lab/css360-ft-v2"),
            "localhost:5000/lab/css360-ft-v2:latest",
        )
        self.assertEqual(
            ollama_service.normalize_ollama_model_name("lab/css360-ft-v2:q4"),
            "lab/css360-ft-v2:q4",
        )

    def test_entries_separate_on_commas_whitespace_and_newlines(self) -> None:
        raw = f"{CSS360}@v1=css360-ft-v1\n {CSS360}@v2=css360-ft-v2,{CSS350}@v1={CSS350_MODEL}"
        parsed = ollama_service.parse_model_map(raw)
        self.assertEqual(
            parsed,
            {
                CSS360: {"v1": "css360-ft-v1:latest", "v2": "css360-ft-v2:latest"},
                CSS350: {"v1": CSS350_MODEL},
            },
        )

    def test_an_empty_mapping_is_empty_not_an_error(self) -> None:
        self.assertEqual(ollama_service.parse_model_map(None), {})
        self.assertEqual(ollama_service.parse_model_map("  "), {})
        self.assertEqual(ollama_service.parse_model_map(","), {})

    def test_a_malformed_entry_fails_the_whole_map(self) -> None:
        """A silently dropped course would be a 409 with no hint why."""
        for bad in (
            f"{CSS360}=css360-ft-v2",  # no version
            f"{CSS360}@v2",  # no model
            f"{CSS360}@latest=css360-ft-v2",  # not a version
            f"CSS-360@v2=css360-ft-v2",  # not a course id
            f"../etc@v2=css360-ft-v2",
            f"{CSS360}@v2=",
        ):
            with self.subTest(entry=bad):
                with self.assertRaises(CourseAdapterError):
                    ollama_service.parse_model_map(bad)

    def test_one_version_mapped_to_two_models_is_refused(self) -> None:
        with self.assertRaises(CourseAdapterError):
            ollama_service.parse_model_map(
                f"{CSS360}@v2=css360-ft-v2,{CSS360}@v2=css360-other"
            )
        # The same entry twice is merely redundant.
        parsed = ollama_service.parse_model_map(
            f"{CSS360}@v2=css360-ft-v2,{CSS360}@v2=css360-ft-v2:latest"
        )
        self.assertEqual(parsed, {CSS360: {"v2": "css360-ft-v2:latest"}})


class CourseResolutionTests(unittest.TestCase):
    MAPPING = {
        CSS360: {"v1": "css360-ft-v1:latest", "v2": CSS360_MODEL},
        CSS350: {"v1": CSS350_MODEL},
    }

    def test_each_course_resolves_to_its_own_model(self) -> None:
        first = ollama_service.resolve_course_model(CSS360, "v2", mapping=self.MAPPING)
        second = ollama_service.resolve_course_model(CSS350, "v1", mapping=self.MAPPING)
        self.assertEqual(first["ollamaModel"], CSS360_MODEL)
        self.assertEqual(second["ollamaModel"], CSS350_MODEL)
        self.assertEqual(first["courseId"], CSS360)
        self.assertEqual(second["courseId"], CSS350)

    def test_an_explicit_version_wins(self) -> None:
        resolved = ollama_service.resolve_course_model(CSS360, "v1", mapping=self.MAPPING)
        self.assertEqual(resolved["version"], "v1")
        self.assertEqual(resolved["ollamaModel"], "css360-ft-v1:latest")
        self.assertEqual(resolved["versionSource"], "requested")

    def test_the_highest_mapped_version_answers_when_none_is_named(self) -> None:
        resolved = ollama_service.resolve_course_model(CSS360, None, mapping=self.MAPPING)
        self.assertEqual(resolved["version"], "v2")
        self.assertEqual(resolved["versionSource"], "highest mapped")

    def test_an_unmapped_course_is_refused_and_told_what_to_set(self) -> None:
        other = "css-430-fall-2026-zz11"
        with self.assertRaises(CourseAdapterError) as caught:
            ollama_service.resolve_course_model(other, "v1", mapping=self.MAPPING)
        message = str(caught.exception)
        self.assertIn(other, message)
        self.assertIn(ollama_service.MODEL_MAP_ENV, message)
        # Never a hint towards another course's model.
        self.assertNotIn(CSS360_MODEL, message)
        self.assertNotIn(CSS350_MODEL, message)

    def test_an_unmapped_version_is_refused_listing_what_is_mapped(self) -> None:
        with self.assertRaises(CourseAdapterError) as caught:
            ollama_service.resolve_course_model(CSS350, "v9", mapping=self.MAPPING)
        self.assertIn("v9", str(caught.exception))
        self.assertIn("v1", str(caught.exception))

    def test_a_course_id_that_could_be_a_path_is_refused(self) -> None:
        for bad in ("../secrets", "CSS-360", "", "css/360"):
            with self.subTest(course_id=bad):
                with self.assertRaises(CourseAdapterError):
                    ollama_service.resolve_course_model(bad, None, mapping=self.MAPPING)

    def test_an_empty_mapping_answers_for_nobody(self) -> None:
        with self.assertRaises(CourseAdapterError):
            ollama_service.resolve_course_model(CSS360, "v2", mapping={})


class DecodingSettingsTests(unittest.TestCase):
    """The Tillicum decoding settings, carried across in Ollama's vocabulary."""

    def test_options_match_the_gpu_service(self) -> None:
        with _env(DEFAULT_MAP):
            options = ollama_service.build_generation_options()
        self.assertEqual(options["num_predict"], ollama_service.DEFAULT_MAX_NEW_TOKENS)
        self.assertEqual(options["num_predict"], 160)
        self.assertEqual(options["temperature"], 0)  # do_sample=False
        self.assertEqual(options["repeat_penalty"], 1.05)
        # Whole context, as Transformers does. Spelled as the context size:
        # Ollama rejects the documented `-1` shorthand with HTTP 400.
        self.assertEqual(options["repeat_last_n"], options["num_ctx"])
        self.assertEqual(options["seed"], 360)
        self.assertEqual(options["num_ctx"], ollama_service.DEFAULT_NUM_CTX)

    def test_num_ctx_can_be_raised_but_not_made_useless(self) -> None:
        with _env(DEFAULT_MAP, FINETUNED_NUM_CTX="8192"):
            options = ollama_service.build_generation_options()
            self.assertEqual(options["num_ctx"], 8192)
            self.assertEqual(options["repeat_last_n"], 8192)
        with _env(DEFAULT_MAP, FINETUNED_NUM_CTX="16"):
            self.assertEqual(
                ollama_service.build_generation_options()["num_ctx"],
                ollama_service.DEFAULT_NUM_CTX,
            )
        with _env(DEFAULT_MAP, FINETUNED_NUM_CTX="lots"):
            self.assertEqual(
                ollama_service.build_generation_options()["num_ctx"],
                ollama_service.DEFAULT_NUM_CTX,
            )

    def test_the_prompt_is_one_user_turn_and_nothing_else(self) -> None:
        """What `apply_chat_template` was given on Tillicum: the question, as
        the user, with no system prompt. A grounded Fine-Tuned + RAG prompt
        travels the same way, verbatim."""
        grounded = "You are answering a student question...\n\nStudent question:\nWhen?"
        with _env(DEFAULT_MAP):
            payload = ollama_service.build_chat_request(grounded, ollama_model=CSS360_MODEL)
        self.assertEqual(payload["model"], CSS360_MODEL)
        self.assertEqual(payload["messages"], [{"role": "user", "content": grounded}])
        self.assertIs(payload["stream"], False)
        self.assertNotIn("system", payload)
        self.assertNotIn("keep_alive", payload)

    def test_keep_alive_is_passed_through_only_when_set(self) -> None:
        with _env(DEFAULT_MAP, FINETUNED_KEEP_ALIVE="30m"):
            payload = ollama_service.build_chat_request("q", ollama_model=CSS360_MODEL)
        self.assertEqual(payload["keep_alive"], "30m")


class HealthSummaryTests(unittest.TestCase):
    MAPPING = {CSS360: {"v1": "css360-ft-v1:latest", "v2": CSS360_MODEL}, CSS350: {"v1": CSS350_MODEL}}

    def test_only_courses_whose_model_exists_are_servable(self) -> None:
        summary = ollama_service.summarize_courses(
            self.MAPPING, [CSS360_MODEL, "llama3.2:3b"]
        )
        self.assertEqual(
            summary["courses"],
            [{"courseId": CSS360, "versions": ["v2"], "currentVersion": "v2"}],
        )
        by_model = {row["ollamaModel"]: row["available"] for row in summary["models"]}
        self.assertEqual(
            by_model,
            {"css360-ft-v1:latest": False, CSS360_MODEL: True, CSS350_MODEL: False},
        )

    def test_nothing_is_servable_when_ollama_could_not_be_asked(self) -> None:
        summary = ollama_service.summarize_courses(self.MAPPING, None)
        self.assertEqual(summary["courses"], [])
        self.assertTrue(all(row["available"] is False for row in summary["models"]))

    def test_the_current_version_is_the_highest_servable_one(self) -> None:
        summary = ollama_service.summarize_courses(
            self.MAPPING, ["css360-ft-v1:latest", CSS360_MODEL]
        )
        self.assertEqual(summary["courses"][0]["versions"], ["v1", "v2"])
        self.assertEqual(summary["courses"][0]["currentVersion"], "v2")


def _client() -> TestClient:
    return TestClient(ollama_service.app)


class HealthEndpointTests(unittest.TestCase):
    def test_health_when_ollama_has_the_mapped_model(self) -> None:
        fake = FakeOllama(models=[CSS360_MODEL, "llama3.2:3b"])
        with _env(DEFAULT_MAP), mock.patch.object(ollama_service.httpx, "AsyncClient", fake):
            response = _client().get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertIs(body["adapterLoaded"], True)
        self.assertEqual(body["model"], "llama3.2:3b")
        self.assertEqual(body["engine"], "ollama")
        self.assertEqual(body["port"], 9001)
        self.assertIsInstance(body["hostname"], str)
        self.assertEqual(
            body["courses"],
            [{"courseId": CSS360, "versions": ["v2"], "currentVersion": "v2"}],
        )
        self.assertIsNone(body["secondsRemaining"])
        self.assertIsNone(body["expiresAt"])
        self.assertNotIn("servingRoot", body)
        self.assertNotIn("cudaAvailable", body)
        self.assertEqual(fake.requests, [{"method": "GET", "url": "http://127.0.0.1:11434/api/tags"}])

    def test_health_when_ollama_is_down_is_200_and_says_so(self) -> None:
        """The service is up; the model is not. Same posture as the GPU
        service's `status: loading`: the operator scripts poll until
        `status == ok and adapterLoaded`, and the backend surfaces the status."""
        fake = FakeOllama(tags_error=httpx.ConnectError("connection refused"))
        with _env(DEFAULT_MAP), mock.patch.object(ollama_service.httpx, "AsyncClient", fake):
            response = _client().get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "unavailable")
        self.assertIs(body["adapterLoaded"], False)
        self.assertEqual(body["courses"], [])
        self.assertIn("Ollama", body["detail"])

    def test_health_when_the_mapped_model_was_never_created(self) -> None:
        fake = FakeOllama(models=["llama3.2:3b"])
        with _env(DEFAULT_MAP), mock.patch.object(ollama_service.httpx, "AsyncClient", fake):
            body = _client().get("/health").json()

        self.assertEqual(body["status"], "ok")
        self.assertIs(body["adapterLoaded"], False)
        self.assertEqual(body["courses"], [])
        self.assertEqual(
            {row["ollamaModel"]: row["available"] for row in body["models"]},
            {CSS360_MODEL: False, CSS350_MODEL: False},
        )

    def test_health_with_no_mapping_is_not_ready(self) -> None:
        fake = FakeOllama(models=[CSS360_MODEL])
        with _env(""), mock.patch.object(ollama_service.httpx, "AsyncClient", fake):
            body = _client().get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertIs(body["adapterLoaded"], False)
        self.assertEqual(body["courses"], [])

    def test_courses_endpoint_matches_health(self) -> None:
        fake = FakeOllama(models=[CSS360_MODEL, CSS350_MODEL])
        with _env(DEFAULT_MAP), mock.patch.object(ollama_service.httpx, "AsyncClient", fake):
            body = _client().get("/courses").json()
        self.assertEqual({c["courseId"] for c in body["courses"]}, {CSS350, CSS360})

    def test_startup_refuses_a_malformed_mapping(self) -> None:
        with _env(f"{CSS360}=css360-ft-v2"):
            with self.assertRaises(CourseAdapterError):
                with TestClient(ollama_service.app):
                    pass


class GenerateEndpointTests(unittest.TestCase):
    def _generate(self, fake: FakeOllama, body: Dict[str, Any], mapping: str = DEFAULT_MAP):
        with _env(mapping), mock.patch.object(ollama_service.httpx, "AsyncClient", fake):
            return _client().post("/generate", json=body)

    def test_successful_generation_echoes_course_and_version(self) -> None:
        fake = FakeOllama()
        response = self._generate(
            fake, {"courseId": CSS360, "modelVersion": "v2", "question": QUESTION}
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["answer"], ANSWER)
        self.assertEqual(body["model"], CSS360_MODEL)
        self.assertEqual(body["courseId"], CSS360)
        self.assertEqual(body["modelVersion"], "v2")
        self.assertIs(body["adapterLoaded"], True)
        self.assertIsInstance(body["generationSeconds"], float)
        self.assertGreaterEqual(body["generationSeconds"], 0.0)

        (sent,) = fake.chat_requests
        self.assertEqual(sent["url"], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(sent["json"]["model"], CSS360_MODEL)
        self.assertEqual(sent["json"]["messages"], [{"role": "user", "content": QUESTION}])
        self.assertIs(sent["json"]["stream"], False)
        self.assertEqual(sent["json"]["options"]["temperature"], 0)
        self.assertEqual(sent["json"]["options"]["num_predict"], 160)
        self.assertEqual(sent["json"]["options"]["repeat_penalty"], 1.05)

    def test_the_answer_is_stripped_like_the_gpu_service_decoded_it(self) -> None:
        fake = FakeOllama(answer="\n  Fridays at 5pm.  \n")
        body = self._generate(fake, {"courseId": CSS360, "question": QUESTION}).json()
        self.assertEqual(body["answer"], "Fridays at 5pm.")

    def test_each_course_is_answered_by_its_own_model(self) -> None:
        """The isolation requirement, stated directly: two courses, two models,
        and each request's model is the one mapped to *its* course."""
        fake = FakeOllama()
        first = self._generate(fake, {"courseId": CSS360, "modelVersion": "v2", "question": QUESTION})
        second = self._generate(fake, {"courseId": CSS350, "modelVersion": "v1", "question": QUESTION})

        self.assertEqual(first.json()["courseId"], CSS360)
        self.assertEqual(second.json()["courseId"], CSS350)
        self.assertEqual(
            [r["json"]["model"] for r in fake.chat_requests], [CSS360_MODEL, CSS350_MODEL]
        )
        self.assertEqual(first.json()["model"], CSS360_MODEL)
        self.assertEqual(second.json()["model"], CSS350_MODEL)

    def test_the_response_never_names_a_course_other_than_the_one_asked(self) -> None:
        """What the backend's own check relies on: the echo is the request."""
        fake = FakeOllama()
        for course_id, version in ((CSS360, "v2"), (CSS350, "v1"), (CSS360, None)):
            with self.subTest(course=course_id):
                body = self._generate(
                    fake, {"courseId": course_id, "modelVersion": version, "question": QUESTION}
                ).json()
                self.assertEqual(body["courseId"], course_id)

    def test_omitting_the_version_uses_the_highest_mapped_one(self) -> None:
        fake = FakeOllama()
        mapping = f"{CSS360}@v1=css360-ft-v1,{CSS360}@v2={CSS360_MODEL}"
        body = self._generate(fake, {"courseId": CSS360, "question": QUESTION}, mapping).json()
        self.assertEqual(body["modelVersion"], "v2")
        self.assertEqual(fake.chat_requests[0]["json"]["model"], CSS360_MODEL)

    def test_an_unsupported_course_is_a_409_and_ollama_is_never_asked(self) -> None:
        fake = FakeOllama()
        other = "css-430-fall-2026-zz11"
        response = self._generate(fake, {"courseId": other, "modelVersion": "v1", "question": QUESTION})

        self.assertEqual(response.status_code, 409)
        detail = response.json()["detail"]
        self.assertIn(other, detail)
        self.assertIn(ollama_service.MODEL_MAP_ENV, detail)
        self.assertNotIn(CSS360_MODEL, detail)
        self.assertEqual(fake.chat_requests, [])

    def test_a_version_this_host_does_not_have_is_a_409(self) -> None:
        """The backend resolved v3 from PostgreSQL; only v2 is mapped here.
        Refused — never answered by v2 while claiming to be v3."""
        fake = FakeOllama()
        response = self._generate(fake, {"courseId": CSS360, "modelVersion": "v3", "question": QUESTION})

        self.assertEqual(response.status_code, 409)
        self.assertIn("v3", response.json()["detail"])
        self.assertEqual(fake.chat_requests, [])

    def test_with_no_mapping_every_course_is_refused(self) -> None:
        fake = FakeOllama()
        response = self._generate(fake, {"courseId": CSS360, "question": QUESTION}, mapping="")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(fake.chat_requests, [])

    def test_a_blank_question_is_a_422(self) -> None:
        fake = FakeOllama()
        response = self._generate(fake, {"courseId": CSS360, "question": "   "})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(fake.chat_requests, [])

    def test_a_request_without_a_course_is_a_422(self) -> None:
        fake = FakeOllama()
        response = self._generate(fake, {"question": QUESTION})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(fake.chat_requests, [])

    def test_ollama_unreachable_is_a_503(self) -> None:
        fake = FakeOllama(chat_error=httpx.ConnectError("connection refused"))
        response = self._generate(fake, {"courseId": CSS360, "modelVersion": "v2", "question": QUESTION})
        self.assertEqual(response.status_code, 503)
        self.assertIn("unavailable", response.json()["detail"].lower())

    def test_ollama_timeout_is_a_503(self) -> None:
        fake = FakeOllama(chat_error=httpx.ReadTimeout("slow"))
        response = self._generate(fake, {"courseId": CSS360, "modelVersion": "v2", "question": QUESTION})
        self.assertEqual(response.status_code, 503)
        self.assertIn("timed out", response.json()["detail"].lower())

    def test_ollama_server_error_is_a_503(self) -> None:
        fake = FakeOllama(chat_status=500, chat_text="llama runner process has terminated")
        response = self._generate(fake, {"courseId": CSS360, "modelVersion": "v2", "question": QUESTION})
        self.assertEqual(response.status_code, 503)
        self.assertIn("500", response.json()["detail"])

    def test_a_mapped_model_that_does_not_exist_in_ollama_is_a_409(self) -> None:
        """Mapped but never `ollama create`d: an operator action away, like a
        missing published adapter on the GPU service."""
        fake = FakeOllama(chat_status=404, chat_text=f"model '{CSS360_MODEL}' not found")
        response = self._generate(fake, {"courseId": CSS360, "modelVersion": "v2", "question": QUESTION})
        self.assertEqual(response.status_code, 409)
        self.assertIn(CSS360_MODEL, response.json()["detail"])
        self.assertIn("ollama create", response.json()["detail"])

    def test_any_other_ollama_rejection_is_a_502(self) -> None:
        fake = FakeOllama(chat_status=400, chat_text="invalid options")
        response = self._generate(fake, {"courseId": CSS360, "modelVersion": "v2", "question": QUESTION})
        self.assertEqual(response.status_code, 502)

    def test_a_malformed_chat_body_is_a_502(self) -> None:
        fake = FakeOllama(chat_payload={"model": CSS360_MODEL, "done": True})
        response = self._generate(fake, {"courseId": CSS360, "modelVersion": "v2", "question": QUESTION})
        self.assertEqual(response.status_code, 502)

    def test_a_grounded_prompt_is_forwarded_verbatim_as_the_user_turn(self) -> None:
        """Fine-Tuned + RAG builds the whole grounded prompt on the backend and
        sends it as `question`. It must reach the model unchanged."""
        grounded = (
            "You are answering a student question about a course syllabus.\n\n"
            "Student question:\nWhen is the final?\n\nSyllabus context:\n"
            "[Section: Exams]\nThe final is on March 18.\n\nAnswer the student question now:"
        )
        fake = FakeOllama(answer="The final is on March 18.")
        response = self._generate(fake, {"courseId": CSS360, "modelVersion": "v2", "question": grounded})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            fake.chat_requests[0]["json"]["messages"], [{"role": "user", "content": grounded}]
        )


class BindingTests(unittest.TestCase):
    def test_the_service_listens_on_loopback_only(self) -> None:
        self.assertEqual(ollama_service.BIND_HOST, "127.0.0.1")

    def test_the_default_port_is_the_one_the_backend_expects(self) -> None:
        with _env(DEFAULT_MAP):
            self.assertEqual(ollama_service.resolve_port(), 9001)
        with _env(DEFAULT_MAP, INFERENCE_PORT="9100"):
            self.assertEqual(ollama_service.resolve_port(), 9100)
        with _env(DEFAULT_MAP, INFERENCE_PORT="port"):
            with self.assertRaises(RuntimeError):
                ollama_service.resolve_port()

    def test_main_binds_loopback_even_if_a_host_override_is_in_the_environment(self) -> None:
        fake_uvicorn = mock.MagicMock()
        with _env(DEFAULT_MAP, INFERENCE_HOST="0.0.0.0"), mock.patch.dict(
            sys.modules, {"uvicorn": fake_uvicorn}
        ):
            ollama_service.main()
        _, kwargs = fake_uvicorn.run.call_args
        self.assertEqual(kwargs["host"], "127.0.0.1")
        self.assertEqual(kwargs["port"], 9001)


if __name__ == "__main__":
    unittest.main()
