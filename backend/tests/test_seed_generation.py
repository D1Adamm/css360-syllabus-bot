"""Tests for AI seed generation service and temporary endpoint."""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.ollama import generate_ollama_completion
from app.seed_generation import (
    SEED_GENERATION_MODEL,
    generate_seeds_from_chunk,
)


def _valid_seeds_payload(count: int = 3) -> str:
    seeds = [
        {
            "question": f"Question {index}?",
            "answer": f"Answer {index}.",
            "category": "grading",
        }
        for index in range(1, count + 1)
    ]
    return json.dumps({"seeds": seeds})


class SeedGenerationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_seeds_success(self) -> None:
        with patch(
            "app.seed_generation.generate_ollama_completion",
            new=AsyncMock(
                return_value={
                    "answer": _valid_seeds_payload(3),
                    "model": SEED_GENERATION_MODEL,
                }
            ),
        ) as mock_generate:
            result = await generate_seeds_from_chunk(
                course_id="css-360-summer-2026-demo",
                chunk_id="chunk-late-1",
                chunk_text="Late work may be submitted within 24 hours for half credit.",
                count=3,
            )

        self.assertEqual(result["courseId"], "css-360-summer-2026-demo")
        self.assertEqual(result["chunkId"], "chunk-late-1")
        self.assertEqual(result["model"], SEED_GENERATION_MODEL)
        self.assertEqual(result["count"], 3)
        self.assertEqual(len(result["seeds"]), 3)

        first = result["seeds"][0]
        self.assertEqual(first["question"], "Question 1?")
        self.assertEqual(first["answer"], "Answer 1.")
        self.assertEqual(first["category"], "grading")
        self.assertEqual(first["sourceChunkIds"], ["chunk-late-1"])
        self.assertEqual(first["origin"], "ai_generated")
        self.assertEqual(first["status"], "generated")

        mock_generate.assert_awaited_once()
        call_kwargs = mock_generate.await_args.kwargs
        self.assertEqual(call_kwargs["model"], SEED_GENERATION_MODEL)
        self.assertEqual(call_kwargs["response_format"], "json")
        self.assertIs(call_kwargs["think"], False)

    async def test_empty_chunk_text_raises_422(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            await generate_seeds_from_chunk(
                course_id="css-360-summer-2026-demo",
                chunk_id="chunk-late-1",
                chunk_text="   ",
                count=3,
            )
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("chunkText", ctx.exception.detail)

    async def test_malformed_json_raises_502(self) -> None:
        with patch(
            "app.seed_generation.generate_ollama_completion",
            new=AsyncMock(
                return_value={
                    "answer": "not-json-at-all",
                    "model": SEED_GENERATION_MODEL,
                }
            ),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await generate_seeds_from_chunk(
                    course_id="css-360-summer-2026-demo",
                    chunk_id="chunk-late-1",
                    chunk_text="Office hours are Tuesdays at 2pm.",
                    count=3,
                )
        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn("malformed JSON", ctx.exception.detail)

    async def test_missing_question_or_answer_raises_502(self) -> None:
        payload = json.dumps(
            {
                "seeds": [
                    {"question": "What is the late policy?", "answer": "24 hours."},
                    {"question": "", "answer": "Missing question."},
                    {"question": "When are office hours?", "answer": "Tuesday."},
                ]
            }
        )
        with patch(
            "app.seed_generation.generate_ollama_completion",
            new=AsyncMock(
                return_value={
                    "answer": payload,
                    "model": SEED_GENERATION_MODEL,
                }
            ),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await generate_seeds_from_chunk(
                    course_id="css-360-summer-2026-demo",
                    chunk_id="chunk-late-1",
                    chunk_text="Late work may be submitted within 24 hours.",
                    count=3,
                )
        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn("missing a question or answer", ctx.exception.detail)

    async def test_ollama_unavailable_propagates_503(self) -> None:
        with patch(
            "app.seed_generation.generate_ollama_completion",
            new=AsyncMock(
                side_effect=HTTPException(
                    status_code=503,
                    detail="Ollama is unavailable. Start Ollama locally and try again.",
                )
            ),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await generate_seeds_from_chunk(
                    course_id="css-360-summer-2026-demo",
                    chunk_id="chunk-late-1",
                    chunk_text="Late work may be submitted within 24 hours.",
                    count=3,
                )
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("Ollama is unavailable", ctx.exception.detail)


class OllamaEmptyResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_response_field_raises_502(self) -> None:
        """qwen3 thinking mode can leave response empty while thinking is filled."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "qwen3:4b",
            "response": "",
            "thinking": '{"seeds":[{"question":"Q?","answer":"A","category":"t"}]}',
            "done": True,
            "done_reason": "stop",
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.ollama.httpx.AsyncClient", return_value=mock_client):
            with self.assertRaises(HTTPException) as ctx:
                await generate_ollama_completion(
                    "unused prompt",
                    model="qwen3:4b",
                    response_format="json",
                )

        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn("empty response", ctx.exception.detail)
        mock_client.post.assert_awaited_once()
        sent_payload = mock_client.post.await_args.kwargs["json"]
        self.assertNotIn("think", sent_payload)

    async def test_think_false_is_sent_in_payload(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "qwen3:4b",
            "response": _valid_seeds_payload(1),
            "done": True,
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.ollama.httpx.AsyncClient", return_value=mock_client):
            result = await generate_ollama_completion(
                "unused prompt",
                model="qwen3:4b",
                response_format="json",
                think=False,
            )

        self.assertIn("seeds", result["answer"])
        sent_payload = mock_client.post.await_args.kwargs["json"]
        self.assertIs(sent_payload["think"], False)
        self.assertEqual(sent_payload["model"], "qwen3:4b")
        self.assertEqual(sent_payload["format"], "json")


class SeedGenerationEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.course_id = "css-360-summer-2026-demo"
        self.url = f"/api/courses/{self.course_id}/seeds/generate"

    def test_endpoint_success(self) -> None:
        with patch(
            "app.main.generate_seeds_from_chunk",
            new=AsyncMock(
                return_value={
                    "courseId": self.course_id,
                    "chunkId": "chunk-late-1",
                    "model": SEED_GENERATION_MODEL,
                    "count": 3,
                    "seeds": [
                        {
                            "question": "Can I submit late?",
                            "answer": "Yes, within 24 hours for half credit.",
                            "category": "late policy",
                            "sourceChunkIds": ["chunk-late-1"],
                            "origin": "ai_generated",
                            "status": "generated",
                        },
                        {
                            "question": "What is the late penalty?",
                            "answer": "Half credit.",
                            "category": "late policy",
                            "sourceChunkIds": ["chunk-late-1"],
                            "origin": "ai_generated",
                            "status": "generated",
                        },
                        {
                            "question": "How long is the late window?",
                            "answer": "24 hours.",
                            "category": "late policy",
                            "sourceChunkIds": ["chunk-late-1"],
                            "origin": "ai_generated",
                            "status": "generated",
                        },
                    ],
                }
            ),
        ):
            response = self.client.post(
                self.url,
                json={
                    "chunkId": "chunk-late-1",
                    "chunkText": "Late work may be submitted within 24 hours for half credit.",
                    "count": 3,
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["courseId"], self.course_id)
        self.assertEqual(body["chunkId"], "chunk-late-1")
        self.assertEqual(body["model"], SEED_GENERATION_MODEL)
        self.assertEqual(body["count"], 3)
        self.assertEqual(len(body["seeds"]), 3)
        self.assertEqual(body["seeds"][0]["origin"], "ai_generated")
        self.assertEqual(body["seeds"][0]["status"], "generated")
        self.assertEqual(body["seeds"][0]["sourceChunkIds"], ["chunk-late-1"])

    def test_endpoint_empty_chunk_text(self) -> None:
        with patch(
            "app.seed_generation.generate_ollama_completion",
            new=AsyncMock(),
        ) as mock_generate:
            response = self.client.post(
                self.url,
                json={
                    "chunkId": "chunk-late-1",
                    "chunkText": "   ",
                    "count": 3,
                },
            )

        self.assertEqual(response.status_code, 422)
        self.assertIn("chunkText", response.json()["detail"])
        mock_generate.assert_not_called()

    def test_endpoint_malformed_json(self) -> None:
        with patch(
            "app.seed_generation.generate_ollama_completion",
            new=AsyncMock(
                return_value={
                    "answer": "{not valid json",
                    "model": SEED_GENERATION_MODEL,
                }
            ),
        ):
            response = self.client.post(
                self.url,
                json={
                    "chunkId": "chunk-late-1",
                    "chunkText": "Office hours are Tuesdays at 2pm.",
                },
            )

        self.assertEqual(response.status_code, 502)
        self.assertIn("malformed JSON", response.json()["detail"])

    def test_endpoint_missing_fields(self) -> None:
        payload = json.dumps(
            {
                "seeds": [
                    {"question": "Q1?", "answer": "A1"},
                    {"question": "Q2?", "answer": ""},
                    {"question": "Q3?", "answer": "A3"},
                ]
            }
        )
        with patch(
            "app.seed_generation.generate_ollama_completion",
            new=AsyncMock(
                return_value={
                    "answer": payload,
                    "model": SEED_GENERATION_MODEL,
                }
            ),
        ):
            response = self.client.post(
                self.url,
                json={
                    "chunkId": "chunk-late-1",
                    "chunkText": "Office hours are Tuesdays at 2pm.",
                    "count": 3,
                },
            )

        self.assertEqual(response.status_code, 502)
        self.assertIn("missing a question or answer", response.json()["detail"])

    def test_endpoint_ollama_failure(self) -> None:
        with patch(
            "app.seed_generation.generate_ollama_completion",
            new=AsyncMock(
                side_effect=HTTPException(
                    status_code=503,
                    detail="Ollama is unavailable. Start Ollama locally and try again.",
                )
            ),
        ):
            response = self.client.post(
                self.url,
                json={
                    "chunkId": "chunk-late-1",
                    "chunkText": "Office hours are Tuesdays at 2pm.",
                },
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("Ollama is unavailable", response.json()["detail"])
