"""Tests for the fine-tuned inference service HTTP client.

Every generate call now names a course. That is the contract change per-course
serving brought: there is no course-agnostic fine-tuned model to ask, so a
request without a course would be asking the service to choose one.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.finetuned_client import (
    check_finetuned_service_health,
    generate_finetuned_response,
)
from app.main import app


def _mock_async_client(*, response: MagicMock | None = None, side_effect=None) -> AsyncMock:
    mock_client = AsyncMock()
    if side_effect is not None:
        mock_client.post = AsyncMock(side_effect=side_effect)
        mock_client.get = AsyncMock(side_effect=side_effect)
    else:
        mock_client.post = AsyncMock(return_value=response)
        mock_client.get = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


COURSE = "css-360-winter-2026-a7rp"
OTHER_COURSE = "css-350-spring-2026-n3h9"


def _ok_generate_response(
    *,
    answer: str = "Late work is accepted with a 10% daily penalty.",
    model: str = "meta-llama/Llama-3.2-3B-Instruct",
    adapter_loaded: bool = True,
    generation_seconds: float = 1.25,
    course_id: str | None = COURSE,
    model_version: str | None = "v1",
) -> MagicMock:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '{"ok":true}'
    payload = {
        "answer": answer,
        "model": model,
        "adapterLoaded": adapter_loaded,
        "generationSeconds": generation_seconds,
    }
    if course_id is not None:
        payload["courseId"] = course_id
    if model_version is not None:
        payload["modelVersion"] = model_version
    mock_response.json.return_value = payload
    return mock_response


class FineTunedClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._env_backup = {
            key: os.environ.get(key)
            for key in ("FINETUNED_SERVICE_URL", "FINETUNED_SERVICE_TIMEOUT_SECONDS")
        }
        os.environ["FINETUNED_SERVICE_URL"] = "http://example-node:8001"
        os.environ.pop("FINETUNED_SERVICE_TIMEOUT_SECONDS", None)

    def tearDown(self) -> None:
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    async def test_successful_generation(self) -> None:
        mock_response = _ok_generate_response()
        mock_client = _mock_async_client(response=mock_response)

        with patch("app.finetuned_client.httpx.AsyncClient", return_value=mock_client):
            result = await generate_finetuned_response(
                "What is the late policy?", course_id=COURSE, model_version="v1"
            )

        self.assertEqual(
            result["answer"],
            "Late work is accepted with a 10% daily penalty.",
        )
        self.assertEqual(result["model"], "meta-llama/Llama-3.2-3B-Instruct")
        self.assertTrue(result["adapter_loaded"])
        self.assertEqual(result["generation_seconds"], 1.25)
        self.assertEqual(result["response_type"], "fineTuned")
        self.assertEqual(result["course_id"], COURSE)
        self.assertEqual(result["model_version"], "v1")
        mock_client.post.assert_awaited_once_with(
            "http://example-node:8001/generate",
            json={
                "question": "What is the late policy?",
                "courseId": COURSE,
                "modelVersion": "v1",
            },
        )

    async def test_a_response_for_another_course_is_refused(self) -> None:
        """The isolation check, at the one place both course ids are known.

        A CSS 360 request answered from the CSS 350 adapter comes back fluent,
        specific and about the wrong syllabus. Nothing downstream could tell;
        this can, because it asked.
        """
        mock_response = _ok_generate_response(course_id=OTHER_COURSE)
        mock_client = _mock_async_client(response=mock_response)

        with patch("app.finetuned_client.httpx.AsyncClient", return_value=mock_client):
            with self.assertRaises(HTTPException) as ctx:
                await generate_finetuned_response(
                    "What is the late policy?", course_id=COURSE
                )

        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn(OTHER_COURSE, ctx.exception.detail)
        self.assertIn("discarded", ctx.exception.detail)

    async def test_a_response_with_no_course_is_refused(self) -> None:
        """An older single-adapter service is refused rather than trusted."""
        mock_response = _ok_generate_response(course_id=None)
        mock_client = _mock_async_client(response=mock_response)

        with patch("app.finetuned_client.httpx.AsyncClient", return_value=mock_client):
            with self.assertRaises(HTTPException) as ctx:
                await generate_finetuned_response(
                    "What is the late policy?", course_id=COURSE
                )

        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn("no courseId", ctx.exception.detail)

    async def test_a_course_with_no_published_adapter_is_a_409(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 409
        mock_response.text = "No fine-tuned adapter is published for course ..."
        mock_client = _mock_async_client(response=mock_response)

        with patch("app.finetuned_client.httpx.AsyncClient", return_value=mock_client):
            with self.assertRaises(HTTPException) as ctx:
                await generate_finetuned_response(
                    "What is the late policy?", course_id=COURSE
                )

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn(COURSE, ctx.exception.detail)

    async def test_a_blank_course_id_is_refused_before_any_request(self) -> None:
        mock_client = _mock_async_client(response=_ok_generate_response())

        with patch("app.finetuned_client.httpx.AsyncClient", return_value=mock_client):
            with self.assertRaises(HTTPException) as ctx:
                await generate_finetuned_response("What is the late policy?", course_id="  ")

        self.assertEqual(ctx.exception.status_code, 422)
        mock_client.post.assert_not_awaited()

    async def test_service_unavailable(self) -> None:
        mock_client = _mock_async_client(
            side_effect=httpx.ConnectError("connection refused"),
        )

        with patch("app.finetuned_client.httpx.AsyncClient", return_value=mock_client):
            with self.assertRaises(HTTPException) as ctx:
                await generate_finetuned_response(
                "What is the late policy?", course_id=COURSE
            )

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("unavailable", ctx.exception.detail.lower())

    async def test_timeout(self) -> None:
        mock_client = _mock_async_client(
            side_effect=httpx.ReadTimeout("timed out"),
        )

        with patch("app.finetuned_client.httpx.AsyncClient", return_value=mock_client):
            with self.assertRaises(HTTPException) as ctx:
                await generate_finetuned_response(
                "What is the late policy?", course_id=COURSE
            )

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("timed out", ctx.exception.detail.lower())

    async def test_malformed_response(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "not-json-object"
        mock_response.json.return_value = ["unexpected", "list"]
        mock_client = _mock_async_client(response=mock_response)

        with patch("app.finetuned_client.httpx.AsyncClient", return_value=mock_client):
            with self.assertRaises(HTTPException) as ctx:
                await generate_finetuned_response(
                "What is the late policy?", course_id=COURSE
            )

        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn("malformed", ctx.exception.detail.lower())

    async def test_adapter_loaded_false(self) -> None:
        mock_response = _ok_generate_response(adapter_loaded=False)
        mock_client = _mock_async_client(response=mock_response)

        with patch("app.finetuned_client.httpx.AsyncClient", return_value=mock_client):
            with self.assertRaises(HTTPException) as ctx:
                await generate_finetuned_response(
                "What is the late policy?", course_id=COURSE
            )

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("adapter is not loaded", ctx.exception.detail.lower())

    async def test_missing_finetuned_service_url(self) -> None:
        os.environ.pop("FINETUNED_SERVICE_URL", None)

        with self.assertRaises(HTTPException) as ctx:
            await generate_finetuned_response(
                "What is the late policy?", course_id=COURSE
            )

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("FINETUNED_SERVICE_URL", ctx.exception.detail)

    async def test_non_200_response(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "internal error"
        mock_client = _mock_async_client(response=mock_response)

        with patch("app.finetuned_client.httpx.AsyncClient", return_value=mock_client):
            with self.assertRaises(HTTPException) as ctx:
                await generate_finetuned_response(
                "What is the late policy?", course_id=COURSE
            )

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("server error", ctx.exception.detail.lower())

    async def test_health_success(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"status":"ok"}'
        mock_response.json.return_value = {
            "status": "ok",
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "adapterLoaded": True,
            "hostname": "example-node",
            "port": 8001,
        }
        mock_client = _mock_async_client(response=mock_response)

        with patch("app.finetuned_client.httpx.AsyncClient", return_value=mock_client):
            result = await check_finetuned_service_health()

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["adapterLoaded"])
        self.assertEqual(result["serviceUrl"], "http://example-node:8001")
        mock_client.get.assert_awaited_once_with("http://example-node:8001/health")


class FineTunedEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = os.environ.get("FINETUNED_SERVICE_URL")
        os.environ["FINETUNED_SERVICE_URL"] = "http://example-node:8001"
        self.client = TestClient(app)

    def tearDown(self) -> None:
        if self._env_backup is None:
            os.environ.pop("FINETUNED_SERVICE_URL", None)
        else:
            os.environ["FINETUNED_SERVICE_URL"] = self._env_backup

    def test_generate_endpoint_success(self) -> None:
        async def fake_generate(question: str, *, course_id: str, model_version=None) -> dict:
            self.assertEqual(question, "What is the late policy?")
            self.assertEqual(course_id, COURSE)
            # The version the route resolved from the registry travels with the
            # request; the cluster is never asked to pick one.
            self.assertEqual(model_version, "v2")
            return {
                "answer": "See the syllabus late policy.",
                "model": "meta-llama/Llama-3.2-3B-Instruct",
                "adapter_loaded": True,
                "course_id": course_id,
                "model_version": model_version,
                "generation_seconds": 0.5,
                "response_type": "fineTuned",
            }

        with patch(
            "app.main.resolve_current_course_model",
            return_value={"courseId": COURSE, "version": "v2"},
        ), patch(
            "app.main.generate_finetuned_response",
            new=AsyncMock(side_effect=fake_generate),
        ):
            response = self.client.post(
                "/fine-tuned/generate",
                json={
                    "courseId": COURSE,
                    "question": "What is the late policy?",
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["answer"], "See the syllabus late policy.")
        self.assertEqual(body["responseType"], "fineTuned")
        self.assertEqual(body["courseId"], COURSE)
        self.assertEqual(body["modelVersion"], "v2")
        self.assertTrue(body["adapterLoaded"])
        self.assertEqual(body["generationSeconds"], 0.5)

    def test_generate_endpoint_without_a_ready_model(self) -> None:
        """A course with no trained model is told so, not given someone else's."""
        from app.course_model_resolution import NoReadyCourseModel

        with patch(
            "app.main.resolve_current_course_model",
            side_effect=NoReadyCourseModel(COURSE, "no fine-tuned model yet"),
        ), patch("app.main.generate_finetuned_response", new=AsyncMock()) as generate:
            response = self.client.post(
                "/fine-tuned/generate",
                json={"courseId": COURSE, "question": "What is the late policy?"},
            )

        self.assertEqual(response.status_code, 409)
        generate.assert_not_awaited()
        # The body is the student-facing sentence, not the operator reason.
        self.assertEqual(
            response.json()["detail"],
            "A fine-tuned model is not available for this course yet.",
        )

    def test_generate_endpoint_missing_url(self) -> None:
        os.environ.pop("FINETUNED_SERVICE_URL", None)

        with patch(
            "app.main.resolve_current_course_model",
            return_value={"courseId": COURSE, "version": "v1"},
        ):
            response = self.client.post(
                "/fine-tuned/generate",
                json={"courseId": COURSE, "question": "What is the late policy?"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("FINETUNED_SERVICE_URL", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
