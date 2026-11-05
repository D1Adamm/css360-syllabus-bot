"""Failures a browser must tell apart carry a stable code (audit finding B5).

`ClassifiedHTTPException.detail` stays the plain message, so every caller that
reads it as a string is unchanged; the HTTP body is `{"detail": {"code",
"message", ...}}`, the shape the generation queue's `generation_busy` already
used and the browser's HTTP client already parses.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app import error_classification as ec
from app.course_model_resolution import NoReadyCourseModel, PUBLIC_UNAVAILABLE_DETAIL
from app.main import app

COURSE = "css-360-winter-2026-a7rp"


class HandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = FastAPI()
        ec.install(self.app)

        @self.app.get("/classified")
        def classified() -> None:
            raise ec.ClassifiedHTTPException(
                status_code=409,
                code=ec.MODEL_NOT_ACTIVATED,
                message="Not yet.",
                headers={"Retry-After": "5"},
                extra={"courseId": COURSE},
            )

        @self.app.get("/plain")
        def plain() -> None:
            raise HTTPException(status_code=404, detail="Plain.")

        self.client = TestClient(self.app)

    def test_the_body_carries_code_message_and_extras(self) -> None:
        response = self.client.get("/classified")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json(),
            {"detail": {"code": "model_not_activated", "message": "Not yet.", "courseId": COURSE}},
        )
        self.assertEqual(response.headers["Retry-After"], "5")

    def test_a_plain_http_exception_is_untouched(self) -> None:
        response = self.client.get("/plain")
        self.assertEqual(response.json(), {"detail": "Plain."})

    def test_detail_stays_the_message_for_callers(self) -> None:
        exc = ec.ClassifiedHTTPException(503, ec.FINETUNED_UNAVAILABLE, "Down.")
        self.assertEqual(exc.detail, "Down.")
        self.assertEqual(ec.error_code(exc), "finetuned_unavailable")
        self.assertIsNone(ec.error_code(HTTPException(status_code=500, detail="x")))
        self.assertIsInstance(exc, HTTPException)

    def test_every_code_is_a_stable_snake_case_token(self) -> None:
        for code in ec.ALL_CODES:
            self.assertRegex(code, r"^[a-z][a-z0-9_]+$")


class GenerationRouteCodes(unittest.TestCase):
    """The codes the Compare page's four requests can receive."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        self._env = patch.dict(os.environ, {"FINETUNED_SERVICE_URL": "http://127.0.0.1:9"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_no_activated_version_is_model_not_activated(self) -> None:
        with patch(
            "app.main.resolve_current_course_model",
            side_effect=NoReadyCourseModel(COURSE, "nothing activated"),
        ):
            response = self.client.post(
                "/fine-tuned/generate", json={"courseId": COURSE, "question": "When?"}
            )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json()["detail"],
            {"code": "model_not_activated", "message": PUBLIC_UNAVAILABLE_DETAIL},
        )

    def test_a_question_too_long_to_rate_is_refused_before_anything_is_generated(self) -> None:
        """`schemas.QUESTION_MAX_LENGTH` is the rating route's `questionText`
        limit too (asserted in test_evaluation_integrity), so no student is
        handed four answers to a question that then cannot be recorded."""
        from app.schemas import QUESTION_MAX_LENGTH

        with patch("app.main.resolve_current_course_model") as resolve:
            response = self.client.post(
                "/fine-tuned/generate",
                json={"courseId": COURSE, "question": "x" * (QUESTION_MAX_LENGTH + 1)},
            )
        self.assertEqual(response.status_code, 422, response.text)
        resolve.assert_not_called()

    def test_an_unreachable_fine_tuned_service_is_finetuned_unavailable(self) -> None:
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.post.side_effect = httpx.ConnectError("refused")
        with patch(
            "app.main.resolve_current_course_model",
            return_value={"courseId": COURSE, "version": "v1"},
        ), patch("app.finetuned_client.httpx.AsyncClient", return_value=client):
            response = self.client.post(
                "/fine-tuned/generate", json={"courseId": COURSE, "question": "When?"}
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "finetuned_unavailable")

    def test_a_timeout_is_finetuned_timeout(self) -> None:
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.post.side_effect = httpx.ReadTimeout("slow")
        with patch(
            "app.main.resolve_current_course_model",
            return_value={"courseId": COURSE, "version": "v1"},
        ), patch("app.finetuned_client.httpx.AsyncClient", return_value=client):
            response = self.client.post(
                "/fine-tuned/generate", json={"courseId": COURSE, "question": "When?"}
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "finetuned_timeout")

    def test_an_unmapped_version_on_the_service_is_model_version_not_servable(self) -> None:
        response = httpx.Response(409, json={"detail": "not mapped"}, request=httpx.Request("POST", "http://x"))
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.post.return_value = response
        with patch(
            "app.main.resolve_current_course_model",
            return_value={"courseId": COURSE, "version": "v1"},
        ), patch("app.finetuned_client.httpx.AsyncClient", return_value=client):
            result = self.client.post(
                "/fine-tuned/generate", json={"courseId": COURSE, "question": "When?"}
            )
        self.assertEqual(result.status_code, 409)
        self.assertEqual(result.json()["detail"]["code"], "model_version_not_servable")

    def test_an_unreachable_ollama_is_ollama_unavailable(self) -> None:
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.post.side_effect = httpx.ConnectError("refused")
        with patch("app.ollama.httpx.AsyncClient", return_value=client):
            response = self.client.post(
                "/base-model/generate", json={"courseId": COURSE, "question": "When?"}
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "ollama_unavailable")


if __name__ == "__main__":
    unittest.main()
