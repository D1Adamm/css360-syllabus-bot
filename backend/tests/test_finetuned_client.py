"""Tests for the fine-tuned inference service HTTP client."""

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


def _ok_generate_response(
    *,
    answer: str = "Late work is accepted with a 10% daily penalty.",
    model: str = "meta-llama/Llama-3.2-3B-Instruct",
    adapter_loaded: bool = True,
    generation_seconds: float = 1.25,
) -> MagicMock:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '{"ok":true}'
    mock_response.json.return_value = {
        "answer": answer,
        "model": model,
        "adapterLoaded": adapter_loaded,
        "generationSeconds": generation_seconds,
    }
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
            result = await generate_finetuned_response("What is the late policy?")

        self.assertEqual(
            result["answer"],
            "Late work is accepted with a 10% daily penalty.",
        )
        self.assertEqual(result["model"], "meta-llama/Llama-3.2-3B-Instruct")
        self.assertTrue(result["adapter_loaded"])
        self.assertEqual(result["generation_seconds"], 1.25)
        self.assertEqual(result["response_type"], "fineTuned")
        mock_client.post.assert_awaited_once_with(
            "http://example-node:8001/generate",
            json={"question": "What is the late policy?"},
        )

    async def test_service_unavailable(self) -> None:
        mock_client = _mock_async_client(
            side_effect=httpx.ConnectError("connection refused"),
        )

        with patch("app.finetuned_client.httpx.AsyncClient", return_value=mock_client):
            with self.assertRaises(HTTPException) as ctx:
                await generate_finetuned_response("What is the late policy?")

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("unavailable", ctx.exception.detail.lower())

    async def test_timeout(self) -> None:
        mock_client = _mock_async_client(
            side_effect=httpx.ReadTimeout("timed out"),
        )

        with patch("app.finetuned_client.httpx.AsyncClient", return_value=mock_client):
            with self.assertRaises(HTTPException) as ctx:
                await generate_finetuned_response("What is the late policy?")

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
                await generate_finetuned_response("What is the late policy?")

        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn("malformed", ctx.exception.detail.lower())

    async def test_adapter_loaded_false(self) -> None:
        mock_response = _ok_generate_response(adapter_loaded=False)
        mock_client = _mock_async_client(response=mock_response)

        with patch("app.finetuned_client.httpx.AsyncClient", return_value=mock_client):
            with self.assertRaises(HTTPException) as ctx:
                await generate_finetuned_response("What is the late policy?")

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("adapter is not loaded", ctx.exception.detail.lower())

    async def test_missing_finetuned_service_url(self) -> None:
        os.environ.pop("FINETUNED_SERVICE_URL", None)

        with self.assertRaises(HTTPException) as ctx:
            await generate_finetuned_response("What is the late policy?")

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("FINETUNED_SERVICE_URL", ctx.exception.detail)

    async def test_non_200_response(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "internal error"
        mock_client = _mock_async_client(response=mock_response)

        with patch("app.finetuned_client.httpx.AsyncClient", return_value=mock_client):
            with self.assertRaises(HTTPException) as ctx:
                await generate_finetuned_response("What is the late policy?")

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
        async def fake_generate(question: str) -> dict:
            self.assertEqual(question, "What is the late policy?")
            return {
                "answer": "See the syllabus late policy.",
                "model": "meta-llama/Llama-3.2-3B-Instruct",
                "adapter_loaded": True,
                "generation_seconds": 0.5,
                "response_type": "fineTuned",
            }

        with patch(
            "app.main.generate_finetuned_response",
            new=AsyncMock(side_effect=fake_generate),
        ):
            response = self.client.post(
                "/fine-tuned/generate",
                json={
                    "courseId": "css-360-winter-2026-a7rp",
                    "question": "What is the late policy?",
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["answer"], "See the syllabus late policy.")
        self.assertEqual(body["responseType"], "fineTuned")
        self.assertEqual(body["courseId"], "css-360-winter-2026-a7rp")
        self.assertTrue(body["adapterLoaded"])
        self.assertEqual(body["generationSeconds"], 0.5)

    def test_generate_endpoint_missing_url(self) -> None:
        os.environ.pop("FINETUNED_SERVICE_URL", None)

        response = self.client.post(
            "/fine-tuned/generate",
            json={
                "courseId": "css-360-winter-2026-a7rp",
                "question": "What is the late policy?",
            },
        )

        self.assertEqual(response.status_code, 503)
        self.assertIn("FINETUNED_SERVICE_URL", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
