"""The four classroom routes with the answer cache on and off, and who may read it.

The routes, their request validation, their authorization and the registered
pipelines are the real ones; only the model calls beneath them and the cache's
database are faked.
"""

from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app import answer_cache
from app.auth.dependencies import current_principal
from app.auth.principal import Participant, Principal, StaffUser
from app.main import app

from tests.answer_cache_fake import FakeAnswerCache

COURSE = "css-360-winter-2026-a7rp"
OTHER = "css-350-spring-2026-n3h9"
QUESTION = "When is Homework 2 due?"
FT_IDENTITY = {"model": {"tag": "css360-v2:latest", "digest": "d", "courseVersion": "v2"}}
CACHE_KEYS = {"cached", "generatedAt", "cacheEntryId"}

GROUNDED = {
    "courseId": COURSE,
    "question": QUESTION,
    "prompt": "…",
    "sources": [{"chunkId": "c1", "sectionTitle": "Homework", "text": "Homework 2 is due Friday.", "score": 0.91}],
    "retrievedChunks": [{"chunkId": "c1", "section": "Homework", "text": "Homework 2 is due Friday.", "score": 0.91}],
    "answer": "Homework 2 is due Friday.",
    "model": "llama3.2:3b",
    "modelVersion": None,
    "adapterLoaded": None,
    "generationSeconds": None,
    "engine": None,
    "responseType": "rag",
}
FINE_TUNED = {
    "answer": "Friday.", "model": "css360-v2:latest", "adapter_loaded": True, "course_id": COURSE,
    "model_version": "v2", "generation_seconds": 1.5, "response_type": "fineTuned", "engine": "vm",
}


def participant(course_id: str) -> Principal:
    return Principal(participant=Participant(participant_id=f"p-{course_id}", course_id=course_id, session_id="s"))


def professor(*course_ids: str) -> Principal:
    return Principal(user=StaffUser(user_id="u-prof", email="prof@uw.edu", display_name="Prof",
                                    role="professor", course_ids=frozenset(course_ids), session_id="s"))


class RouteTestCase(unittest.TestCase):
    enabled = True

    def setUp(self) -> None:
        self.cache = FakeAnswerCache().install(self, enabled=self.enabled)
        self.cache.select(COURSE, QUESTION)
        self.cache.set_identity("fineTuned", COURSE, FT_IDENTITY)
        self.cache.set_identity("fineTunedRag", COURSE, FT_IDENTITY)
        self.base = self._patch("app.main.generate_base_model_response", AsyncMock(
            return_value={"answer": "I have no syllabus.", "model": "llama3.2:3b", "response_type": "base"}))
        self.rag = self._patch("app.main.generate_course_rag_answer", AsyncMock(return_value=dict(GROUNDED)))
        self.ft = self._patch("app.main.generate_finetuned_response", AsyncMock(return_value=dict(FINE_TUNED)))
        self.ft_rag = self._patch("app.main.generate_course_finetuned_rag_answer", AsyncMock(return_value={
            **GROUNDED, "model": "css360-v2:latest", "modelVersion": "v2", "adapterLoaded": True,
            "generationSeconds": 2.5, "engine": "vm", "responseType": "fineTunedRag"}))
        self._patch("app.main.resolve_current_course_model", MagicMock(return_value={"version": "v2"}))
        self.client = TestClient(app)

    def _patch(self, target: str, mock: Any) -> Any:
        patcher = patch(target, new=mock)
        patcher.start()
        self.addCleanup(patcher.stop)
        return mock

    def act_as(self, principal: Principal | None) -> None:
        previous = app.dependency_overrides.get(current_principal)
        if principal is None:
            app.dependency_overrides.pop(current_principal, None)
        else:
            app.dependency_overrides[current_principal] = lambda: principal
        self.addCleanup(lambda: app.dependency_overrides.__setitem__(current_principal, previous) if previous else None)

    def ask(self, path: str, question: str = QUESTION, course_id: str = COURSE, **extra: Any):
        return self.client.post(path, json={"courseId": course_id, "question": question, **extra},
                                headers={"X-Requested-With": "SyllabusModelLab"})


class CacheOnRouteTests(RouteTestCase):
    def test_each_route_generates_once_and_then_serves_the_stored_answer_unchanged(self) -> None:
        for path, mock in (
            ("/api/base-model/generate", self.base),
            ("/api/rag/generate", self.rag),
            ("/api/fine-tuned/generate", self.ft),
            ("/api/fine-tuned-rag/generate", self.ft_rag),
        ):
            with self.subTest(path=path):
                first = self.ask(path)
                second = self.ask(path, "  when is homework 2 due ")
                self.assertEqual((first.status_code, second.status_code), (200, 200), second.text)
                self.assertEqual(mock.await_count, 1)
                fresh, stored = first.json(), second.json()
                self.assertEqual((fresh["cached"], stored["cached"]), (False, True))
                self.assertEqual(stored["cacheEntryId"], fresh["cacheEntryId"])
                self.assertEqual(stored["generatedAt"], fresh["generatedAt"])
                # Everything a card or a rating reads is exactly what was generated.
                strip = lambda body: {k: v for k, v in body.items() if k not in CACHE_KEYS}
                self.assertEqual(strip(stored), strip(fresh))
        self.assertEqual({row["condition"] for row in self.cache.stored()}, {"base", "rag", "fineTuned", "fineTunedRag"})

    def test_sources_and_model_provenance_survive_the_cache(self) -> None:
        self.ask("/api/fine-tuned-rag/generate")
        stored = self.ask("/api/fine-tuned-rag/generate").json()
        self.assertTrue(stored["cached"])
        self.assertEqual(stored["sources"], GROUNDED["sources"])
        self.assertEqual(stored["retrievedChunks"], GROUNDED["retrievedChunks"])
        self.assertEqual((stored["model"], stored["modelVersion"], stored["engine"]), ("css360-v2:latest", "v2", "vm"))

    def test_a_partly_prepared_comparison_serves_what_exists_and_generates_the_rest(self) -> None:
        self.ask("/api/base-model/generate")
        self.ask("/api/rag/generate")
        answers = {path: self.ask(path).json() for path in (
            "/api/base-model/generate", "/api/rag/generate", "/api/fine-tuned/generate", "/api/fine-tuned-rag/generate")}
        self.assertEqual(
            [answers[p]["cached"] for p in answers], [True, True, False, False]
        )
        self.assertEqual((self.base.await_count, self.rag.await_count, self.ft.await_count, self.ft_rag.await_count), (1, 1, 1, 1))

    def test_a_question_nobody_selected_is_generated_every_time(self) -> None:
        for _ in range(2):
            body = self.ask("/api/base-model/generate", "What is my grade so far?").json()
            self.assertEqual((body["cached"], body["cacheEntryId"]), (False, None))
        self.assertEqual(self.base.await_count, 2)
        self.assertEqual(self.cache.stored(), [])

    def test_the_request_s_top_k_is_part_of_what_is_cached(self) -> None:
        self.ask("/api/rag/generate", topK=4)
        self.assertFalse(self.ask("/api/rag/generate", topK=6).json()["cached"])
        self.assertTrue(self.ask("/api/rag/generate", topK=4).json()["cached"])
        self.assertEqual([call.kwargs["top_k"] for call in self.rag.await_args_list], [4, 6])

    def test_a_course_with_no_model_still_gets_its_ordinary_refusal(self) -> None:
        from app.course_model_resolution import NoReadyCourseModel

        self.cache.set_identity("fineTuned", COURSE, "no_course_model")
        with patch("app.main.resolve_current_course_model", side_effect=NoReadyCourseModel(COURSE, "none")):
            response = self.ask("/api/fine-tuned/generate")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["code"], "model_not_activated")
        self.assertEqual(self.cache.stored(), [])

    def test_a_failed_generation_reaches_the_browser_as_before_and_is_not_stored(self) -> None:
        from fastapi import HTTPException

        self.base.side_effect = HTTPException(status_code=502, detail="Ollama returned an error.")
        response = self.ask("/api/base-model/generate")
        self.assertEqual((response.status_code, response.json()["detail"]), (502, "Ollama returned an error."))
        self.assertEqual(self.cache.stored(), [])

    def test_the_model_testing_route_is_never_cached(self) -> None:
        with patch("app.main.resolve_course_model_version", return_value={"version": "v2"}):
            for _ in range(2):
                response = self.client.post(
                    "/api/model-testing/generate",
                    json={"courseId": COURSE, "mode": "fineTuned", "modelVersion": "v2", "question": QUESTION},
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertNotIn("cached", response.json())
        self.assertEqual(self.ft.await_count, 2)
        self.assertEqual(self.cache.stored(), [])


class CacheAuthorizationTests(RouteTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.ask("/api/base-model/generate")  # stored, as the test administrator
        self.assertEqual(len(self.cache.stored()), 1)

    def test_a_participant_of_the_course_is_served_the_stored_answer(self) -> None:
        self.act_as(participant(COURSE))
        body = self.ask("/api/base-model/generate").json()
        self.assertTrue(body["cached"])

    def test_a_participant_of_another_course_cannot_read_it(self) -> None:
        self.act_as(participant(OTHER))
        self.assertEqual(self.ask("/api/base-model/generate").status_code, 403)

    def test_a_professor_without_the_course_cannot_read_it(self) -> None:
        self.act_as(professor(OTHER))
        self.assertEqual(self.ask("/api/base-model/generate").status_code, 403)

    def test_nobody_signed_in_cannot_read_it(self) -> None:
        self.act_as(None)
        self.assertEqual(self.ask("/api/base-model/generate").status_code, 401)

    def test_the_same_question_in_another_course_is_that_course_s_own_answer(self) -> None:
        self.cache.select(OTHER, QUESTION)
        self.base.return_value = {"answer": "Another course's answer.", "model": "llama3.2:3b", "response_type": "base"}
        body = self.ask("/api/base-model/generate", course_id=OTHER).json()
        self.assertEqual((body["cached"], body["answer"], body["courseId"]), (False, "Another course's answer.", OTHER))
        self.assertEqual(self.ask("/api/base-model/generate").json()["answer"], "I have no syllabus.")


class CacheOffRouteTests(RouteTestCase):
    enabled = False

    def test_with_the_cache_off_every_request_generates_and_nothing_is_read_or_stored(self) -> None:
        for path, mock in (
            ("/api/base-model/generate", self.base),
            ("/api/rag/generate", self.rag),
            ("/api/fine-tuned/generate", self.ft),
            ("/api/fine-tuned-rag/generate", self.ft_rag),
        ):
            with self.subTest(path=path):
                bodies = [self.ask(path).json() for _ in range(2)]
                self.assertEqual(mock.await_count, 2)
                for body in bodies:
                    self.assertEqual((body["cached"], body["generatedAt"], body["cacheEntryId"]), (False, None, None))
        self.assertEqual((self.cache.reads, self.cache.writes, self.cache.identity_calls), (0, 0, []))

    def test_the_flag_defaults_to_off(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(answer_cache.ENABLED_ENV, None)
            self.assertFalse(answer_cache.cache_enabled())


if __name__ == "__main__":
    unittest.main()
