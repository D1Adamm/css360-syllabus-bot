"""Tests for AI seed generation service and temporary endpoint."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.ollama import generate_ollama_completion
from app.seed_generation import (
    SEED_GENERATION_MODEL,
    VALIDATION_PROMPT_MARKER,
    generate_seeds_from_chunk,
    resolve_starter_run_status,
)
from app.storage import LocalCourseArtifactStorage


def _passing_validation_payload(score: float = 0.88) -> str:
    return json.dumps(
        {
            "grounded": 0.9,
            "correct": 0.91,
            "clear": 0.82,
            "useful": 0.86,
            "naturalStudentWording": 0.84,
            "categoryCorrect": 0.8,
            "notTrivialOrTemporary": 0.82,
            "unsupportedClaims": [],
            "reason": "Supported by the source chunk.",
        }
    )


def _rejecting_validation_payload(score: float = 0.4) -> str:
    return json.dumps(
        {
            "grounded": 0.45,
            "correct": 0.7,
            "clear": 0.65,
            "useful": 0.4,
            "naturalStudentWording": 0.7,
            "categoryCorrect": 0.68,
            "notTrivialOrTemporary": 0.45,
            "unsupportedClaims": ["Answer invents policy details not in the source."],
            "reason": "Not clearly supported by the chunk.",
        }
    )


def _starter_ollama_side_effect(
    *,
    generate_side_effect: object | None = None,
    validation_payload: str | None = None,
) -> AsyncMock:
    validation_answer = validation_payload or _passing_validation_payload()

    async def _fake_generate(prompt: str, **kwargs: object) -> dict[str, str]:
        if VALIDATION_PROMPT_MARKER in prompt:
            return {
                "answer": validation_answer,
                "model": SEED_GENERATION_MODEL,
            }
        if generate_side_effect is not None:
            if isinstance(generate_side_effect, AsyncMock):
                return await generate_side_effect(prompt, **kwargs)
            return await generate_side_effect(prompt, **kwargs)  # type: ignore[misc]
        raise AssertionError("Unexpected generation call without side effect")

    return AsyncMock(side_effect=_fake_generate)


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


def _index(course_id: str, chunks: list[dict]) -> dict:
    return {
        "courseId": course_id,
        "embeddingModel": "nomic-embed-text",
        "chunkCount": len(chunks),
        "chunks": chunks,
    }


def _chunk(
    chunk_id: str,
    section_title: str,
    text: str,
    order: int = 1,
) -> dict:
    return {
        "chunkId": chunk_id,
        "sectionTitle": section_title,
        "text": text,
        "order": order,
        "embedding": [1.0, 0.0, 0.0],
    }


class SeedGenerationServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        root = Path(self._temp_dir.name)
        self.storage = LocalCourseArtifactStorage(
            root_dir=root / "course_data",
            index_dir=root / "indexes",
        )
        self.course_id = "css-360-summer-2026-demo"
        self.chunk_id = "chunk-001"
        self.chunk_text = "Late work may be submitted within 24 hours for half credit."
        self.storage.save_index(
            self.course_id,
            _index(
                self.course_id,
                [
                    _chunk(
                        self.chunk_id,
                        "Late Policy",
                        self.chunk_text,
                        order=1,
                    ),
                    _chunk(
                        "chunk-002",
                        "Office Hours",
                        "Office hours are Tuesdays at 2pm.",
                        order=2,
                    ),
                ],
            ),
        )

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    async def test_generate_seeds_from_stored_chunk(self) -> None:
        with patch(
            "app.seed_generation.generate_starter_ollama_completion",
            new=AsyncMock(
                return_value={
                    "answer": _valid_seeds_payload(3),
                    "model": SEED_GENERATION_MODEL,
                }
            ),
        ) as mock_generate:
            result = await generate_seeds_from_chunk(
                course_id=self.course_id,
                chunk_id=self.chunk_id,
                count=3,
                storage=self.storage,
            )

        self.assertEqual(result["courseId"], self.course_id)
        self.assertEqual(result["chunkId"], self.chunk_id)
        self.assertEqual(result["model"], SEED_GENERATION_MODEL)
        self.assertEqual(result["count"], 3)
        self.assertEqual(len(result["seeds"]), 3)

        first = result["seeds"][0]
        self.assertEqual(first["question"], "Question 1?")
        self.assertEqual(first["answer"], "Answer 1.")
        self.assertEqual(first["category"], "grading")
        self.assertEqual(first["sourceChunkIds"], [self.chunk_id])
        self.assertEqual(first["origin"], "ai_generated")
        self.assertEqual(first["status"], "generated")

        mock_generate.assert_awaited_once()
        call_args = mock_generate.await_args
        prompt = call_args.args[0]
        self.assertIn(self.chunk_text, prompt)
        self.assertIn(self.chunk_id, prompt)
        call_kwargs = call_args.kwargs
        self.assertEqual(call_kwargs["model"], SEED_GENERATION_MODEL)
        self.assertEqual(call_kwargs["response_format"], "json")
        self.assertIs(call_kwargs["think"], False)

    async def test_missing_course_index_raises_404(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            await generate_seeds_from_chunk(
                course_id="css-999-missing-course",
                chunk_id=self.chunk_id,
                count=3,
                storage=self.storage,
            )
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("No syllabus index found", ctx.exception.detail)

    async def test_missing_chunk_raises_404(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            await generate_seeds_from_chunk(
                course_id=self.course_id,
                chunk_id="chunk-999",
                count=3,
                storage=self.storage,
            )
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("was not found", ctx.exception.detail)

    async def test_count_validation_raises_422(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            await generate_seeds_from_chunk(
                course_id=self.course_id,
                chunk_id=self.chunk_id,
                count=6,
                storage=self.storage,
            )
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("count must be between 1 and 5", ctx.exception.detail)

    async def test_malformed_json_raises_502(self) -> None:
        with patch(
            "app.seed_generation.generate_starter_ollama_completion",
            new=AsyncMock(
                return_value={
                    "answer": "not-json-at-all",
                    "model": SEED_GENERATION_MODEL,
                }
            ),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await generate_seeds_from_chunk(
                    course_id=self.course_id,
                    chunk_id=self.chunk_id,
                    count=3,
                    storage=self.storage,
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
            "app.seed_generation.generate_starter_ollama_completion",
            new=AsyncMock(
                return_value={
                    "answer": payload,
                    "model": SEED_GENERATION_MODEL,
                }
            ),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await generate_seeds_from_chunk(
                    course_id=self.course_id,
                    chunk_id=self.chunk_id,
                    count=3,
                    storage=self.storage,
                )
        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn("missing a question or answer", ctx.exception.detail)

    async def test_ollama_unavailable_propagates_503(self) -> None:
        with patch(
            "app.seed_generation.generate_starter_ollama_completion",
            new=AsyncMock(
                side_effect=HTTPException(
                    status_code=503,
                    detail="Ollama is unavailable. Start Ollama locally and try again.",
                )
            ),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await generate_seeds_from_chunk(
                    course_id=self.course_id,
                    chunk_id=self.chunk_id,
                    count=3,
                    storage=self.storage,
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
        self._temp_dir = tempfile.TemporaryDirectory()
        root = Path(self._temp_dir.name)
        self.storage = LocalCourseArtifactStorage(
            root_dir=root / "course_data",
            index_dir=root / "indexes",
        )
        self._storage_patch = patch(
            "app.seed_generation.get_course_artifact_storage",
            return_value=self.storage,
        )
        self._storage_patch.start()

        self.client = TestClient(app)
        self.course_id = "css-360-summer-2026-demo"
        self.chunk_id = "chunk-001"
        self.chunk_text = "Late work may be submitted within 24 hours for half credit."
        self.url = f"/api/courses/{self.course_id}/seeds/generate"
        self.storage.save_index(
            self.course_id,
            _index(
                self.course_id,
                [
                    _chunk(
                        self.chunk_id,
                        "Late Policy",
                        self.chunk_text,
                        order=1,
                    )
                ],
            ),
        )

    def tearDown(self) -> None:
        self._storage_patch.stop()
        self._temp_dir.cleanup()

    def test_endpoint_success_uses_stored_chunk(self) -> None:
        with patch(
            "app.seed_generation.generate_starter_ollama_completion",
            new=AsyncMock(
                return_value={
                    "answer": _valid_seeds_payload(3),
                    "model": SEED_GENERATION_MODEL,
                }
            ),
        ) as mock_generate:
            response = self.client.post(
                self.url,
                json={
                    "chunkId": self.chunk_id,
                    "count": 3,
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["courseId"], self.course_id)
        self.assertEqual(body["chunkId"], self.chunk_id)
        self.assertEqual(body["model"], SEED_GENERATION_MODEL)
        self.assertEqual(body["count"], 3)
        self.assertEqual(len(body["seeds"]), 3)
        self.assertEqual(body["seeds"][0]["origin"], "ai_generated")
        self.assertEqual(body["seeds"][0]["status"], "generated")
        self.assertEqual(body["seeds"][0]["sourceChunkIds"], [self.chunk_id])

        prompt = mock_generate.await_args.args[0]
        self.assertIn(self.chunk_text, prompt)
        self.assertIs(mock_generate.await_args.kwargs["think"], False)

    def test_endpoint_missing_course(self) -> None:
        response = self.client.post(
            "/api/courses/css-999-missing-course/seeds/generate",
            json={"chunkId": self.chunk_id, "count": 3},
        )
        self.assertEqual(response.status_code, 404)
        self.assertIn("No syllabus index found", response.json()["detail"])

    def test_endpoint_missing_chunk(self) -> None:
        response = self.client.post(
            self.url,
            json={"chunkId": "chunk-999", "count": 3},
        )
        self.assertEqual(response.status_code, 404)
        self.assertIn("was not found", response.json()["detail"])

    def test_endpoint_count_validation(self) -> None:
        response = self.client.post(
            self.url,
            json={"chunkId": self.chunk_id, "count": 6},
        )
        self.assertEqual(response.status_code, 422)

    def test_endpoint_rejects_chunk_text_field(self) -> None:
        """Swagger/request schema no longer accepts client-supplied chunkText."""
        with patch(
            "app.seed_generation.generate_starter_ollama_completion",
            new=AsyncMock(
                return_value={
                    "answer": _valid_seeds_payload(3),
                    "model": SEED_GENERATION_MODEL,
                }
            ),
        ) as mock_generate:
            response = self.client.post(
                self.url,
                json={
                    "chunkId": self.chunk_id,
                    "chunkText": "Client-supplied text that must be ignored.",
                    "count": 3,
                },
            )

        # Extra fields are ignored by default; stored chunk text is used.
        self.assertEqual(response.status_code, 200)
        prompt = mock_generate.await_args.args[0]
        self.assertIn(self.chunk_text, prompt)
        self.assertNotIn("Client-supplied text that must be ignored.", prompt)

    def test_endpoint_malformed_json(self) -> None:
        with patch(
            "app.seed_generation.generate_starter_ollama_completion",
            new=AsyncMock(
                return_value={
                    "answer": "{not valid json",
                    "model": SEED_GENERATION_MODEL,
                }
            ),
        ):
            response = self.client.post(
                self.url,
                json={"chunkId": self.chunk_id, "count": 3},
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
            "app.seed_generation.generate_starter_ollama_completion",
            new=AsyncMock(
                return_value={
                    "answer": payload,
                    "model": SEED_GENERATION_MODEL,
                }
            ),
        ):
            response = self.client.post(
                self.url,
                json={"chunkId": self.chunk_id, "count": 3},
            )

        self.assertEqual(response.status_code, 502)
        self.assertIn("missing a question or answer", response.json()["detail"])

    def test_endpoint_ollama_failure(self) -> None:
        with patch(
            "app.seed_generation.generate_starter_ollama_completion",
            new=AsyncMock(
                side_effect=HTTPException(
                    status_code=503,
                    detail="Ollama is unavailable. Start Ollama locally and try again.",
                )
            ),
        ):
            response = self.client.post(
                self.url,
                json={"chunkId": self.chunk_id, "count": 3},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("Ollama is unavailable", response.json()["detail"])


class StarterSeedSelectionTests(unittest.TestCase):
    def test_select_evenly_spaced_preserves_order_and_spreads(self) -> None:
        from app.seed_generation import select_evenly_spaced_chunks

        chunks = [{"chunkId": f"chunk-{i:03d}", "text": "x" * 100} for i in range(1, 11)]
        selected = select_evenly_spaced_chunks(chunks, 4)
        selected_ids = [chunk["chunkId"] for chunk in selected]

        self.assertEqual(len(selected_ids), 4)
        self.assertEqual(selected_ids, sorted(selected_ids))
        self.assertEqual(selected_ids[0], "chunk-001")
        self.assertEqual(selected_ids[-1], "chunk-010")
        # Should not be only the first four consecutive opening chunks.
        self.assertNotEqual(selected_ids, ["chunk-001", "chunk-002", "chunk-003", "chunk-004"])

    def test_normalize_question_for_dedupe(self) -> None:
        from app.seed_dedupe import normalize_question_for_dedupe

        left = normalize_question_for_dedupe("  Can I submit late??? ")
        right = normalize_question_for_dedupe("can i submit late")
        self.assertEqual(left, right)


class StarterRunStatusTests(unittest.TestCase):
    def test_save_false_ready_when_target_met(self) -> None:
        self.assertEqual(
            resolve_starter_run_status(
                target_count=50,
                final_count=50,
                saved_count=0,
                save=False,
            ),
            "ready",
        )

    def test_save_false_partial_when_under_target(self) -> None:
        self.assertEqual(
            resolve_starter_run_status(
                target_count=50,
                final_count=12,
                saved_count=0,
                save=False,
            ),
            "partial",
        )

    def test_save_false_failed_when_empty(self) -> None:
        self.assertEqual(
            resolve_starter_run_status(
                target_count=50,
                final_count=0,
                saved_count=0,
                save=False,
            ),
            "failed",
        )

    def test_save_true_requires_saved_count_for_ready(self) -> None:
        self.assertEqual(
            resolve_starter_run_status(
                target_count=50,
                final_count=50,
                saved_count=40,
                save=True,
            ),
            "partial",
        )
        self.assertEqual(
            resolve_starter_run_status(
                target_count=50,
                final_count=50,
                saved_count=50,
                save=True,
            ),
            "ready",
        )


class StarterSeedGenerationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        root = Path(self._temp_dir.name)
        self.storage = LocalCourseArtifactStorage(
            root_dir=root / "course_data",
            index_dir=root / "indexes",
        )
        self.course_id = "css-360-summer-2026-demo"
        chunks = []
        for index in range(1, 13):
            text = (
                f"Section {index} content about syllabus policies and assignments. "
                * 3
            )
            chunks.append(
                _chunk(
                    f"chunk-{index:03d}",
                    f"Section {index}",
                    text,
                    order=index,
                )
            )
        # Extremely short chunk should be skipped.
        chunks.insert(
            2,
            _chunk("chunk-short", "Short", "too short", order=99),
        )
        self.storage.save_index(self.course_id, _index(self.course_id, chunks))

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    def _unique_seed_side_effect(self) -> AsyncMock:
        call_state = {"n": 0}

        async def _fake_generate(prompt: str, **kwargs: object) -> dict[str, str]:
            call_state["n"] += 1
            n = call_state["n"]
            categories = [
                "Participation expectations",
                "Attendance",
                "Projects",
                "Grading",
                "Course resources",
                "Academic integrity",
                "Labs",
                "Collaboration",
                "Communication",
                "Feedback",
            ]
            question_types = [
                "scenario",
                "clarification",
                "direct",
                "procedure",
                "comparison",
                "scenario",
                "clarification",
                "direct",
                "procedure",
                "comparison",
            ]
            payload = {
                "seeds": [
                    {
                        "question": f"Unique question {n}a?",
                        "answer": f"Answer {n}a with enough detail.",
                        "category": categories[(2 * (n - 1)) % len(categories)],
                        "questionType": question_types[(2 * (n - 1)) % len(question_types)],
                    },
                    {
                        "question": f"Unique question {n}b?",
                        "answer": f"Answer {n}b with enough detail.",
                        "category": categories[(2 * (n - 1) + 1) % len(categories)],
                        "questionType": question_types[(2 * (n - 1) + 1) % len(question_types)],
                    },
                ]
            }
            return {
                "answer": json.dumps(payload),
                "model": SEED_GENERATION_MODEL,
            }

        return _starter_ollama_side_effect(generate_side_effect=_fake_generate)

    async def test_starter_stops_at_target_count(self) -> None:
        from app.seed_generation import generate_starter_seeds_for_course

        mock_generate = self._unique_seed_side_effect()
        with patch(
            "app.seed_generation.generate_starter_ollama_completion",
            new=mock_generate,
        ):
            result = await generate_starter_seeds_for_course(
                course_id=self.course_id,
                target_count=6,
                storage=self.storage,
            )

        self.assertLessEqual(result["progress"]["finalCount"], 6)
        self.assertGreaterEqual(result["progress"]["finalCount"], 5)
        self.assertEqual(len(result["seeds"]), result["progress"]["finalCount"])
        self.assertEqual(result["progress"]["chunksSkipped"], 1)
        self.assertEqual(result["progress"]["eligibleChunks"], 12)
        self.assertGreaterEqual(result["progress"]["chunksProcessed"], result["progress"]["finalCount"])
        self.assertEqual(
            result["progress"]["candidatesAccepted"],
            result["progress"]["finalCount"],
        )
        self.assertEqual(
            result["progress"]["candidatesValidated"],
            result["progress"]["finalCount"],
        )
        self.assertGreater(result["progress"]["ollamaCalls"], result["progress"]["generationCalls"])
        self.assertEqual(
            result["progress"]["ollamaCalls"],
            result["progress"]["planningCalls"]
            + result["progress"]["mergeCalls"]
            + result["progress"]["generationCalls"]
            + result["progress"]["validationCalls"],
        )
        self.assertGreaterEqual(result["progress"]["elapsedMs"], 0)
        self.assertEqual(result["progress"]["savedCount"], 0)
        self.assertIn(result["progress"]["status"], {"ready", "partial", "failed"})
        if result["progress"]["finalCount"] >= 6:
            self.assertEqual(result["progress"]["status"], "ready")
        self.assertGreaterEqual(result["seeds"][0]["validation"]["score"], 0.8)
        self.assertIn("components", result["seeds"][0]["validation"])
        self.assertIn("unsupportedClaims", result["seeds"][0]["validation"])
        self.assertEqual(mock_generate.await_args.kwargs["think"], False)
        self.assertEqual(mock_generate.await_args.kwargs["model"], SEED_GENERATION_MODEL)

    async def test_starter_deduplicates_normalized_questions(self) -> None:
        from app.seed_generation import generate_starter_seeds_for_course

        async def _dup_generate(prompt: str, **kwargs: object) -> dict[str, str]:
            payload = {
                "seeds": [
                    {
                        "question": "Can I submit late?",
                        "answer": "Yes, within the allowed window.",
                        "category": "late policy",
                    },
                    {
                        "question": "can i submit late???",
                        "answer": "Within 24 hours for half credit.",
                        "category": "late policy",
                    },
                ]
            }
            return {"answer": json.dumps(payload), "model": SEED_GENERATION_MODEL}

        with patch(
            "app.seed_generation.generate_starter_ollama_completion",
            new=_starter_ollama_side_effect(generate_side_effect=_dup_generate),
        ):
            result = await generate_starter_seeds_for_course(
                course_id=self.course_id,
                target_count=6,
                storage=self.storage,
            )

        self.assertEqual(result["progress"]["finalCount"], 1)
        self.assertGreaterEqual(result["progress"]["duplicatesRemoved"], 1)
        self.assertEqual(result["seeds"][0]["question"], "Can I submit late?")

    async def test_starter_validation_accepts_candidate(self) -> None:
        from app.seed_generation import generate_starter_seeds_for_course

        async def _single_generate(prompt: str, **kwargs: object) -> dict[str, str]:
            payload = {
                "seeds": [
                    {
                        "question": "What is the attendance policy?",
                        "answer": "Attendance is required for all lectures.",
                        "category": "attendance",
                    }
                ]
            }
            return {"answer": json.dumps(payload), "model": SEED_GENERATION_MODEL}

        with patch(
            "app.seed_generation.generate_starter_ollama_completion",
            new=_starter_ollama_side_effect(
                generate_side_effect=_single_generate,
                validation_payload=_passing_validation_payload(0.92),
            ),
        ):
            result = await generate_starter_seeds_for_course(
                course_id=self.course_id,
                target_count=1,
                storage=self.storage,
            )

        self.assertEqual(result["progress"]["finalCount"], 1)
        self.assertEqual(result["progress"]["candidatesAccepted"], 1)
        self.assertGreaterEqual(result["seeds"][0]["validation"]["score"], 0.8)
        self.assertGreaterEqual(
            result["seeds"][0]["validation"]["components"]["grounded"],
            0.8,
        )
        self.assertEqual(result["seeds"][0]["validation"]["unsupportedClaims"], [])

    async def test_starter_validation_rejects_candidate(self) -> None:
        from app import seed_generation
        from app.seed_generation import generate_starter_seeds_for_course

        async def _single_generate(prompt: str, **kwargs: object) -> dict[str, str]:
            payload = {
                "seeds": [
                    {
                        "question": "Can I skip every assignment?",
                        "answer": "Yes, assignments are optional.",
                        "category": "assignments",
                    }
                ]
            }
            return {"answer": json.dumps(payload), "model": SEED_GENERATION_MODEL}

        with (
            patch.object(seed_generation, "MAX_STARTER_SELECTED_CHUNKS", 1),
            patch(
                "app.seed_generation.generate_starter_ollama_completion",
                new=_starter_ollama_side_effect(
                    generate_side_effect=_single_generate,
                    validation_payload=_rejecting_validation_payload(),
                ),
            ),
        ):
            result = await generate_starter_seeds_for_course(
                course_id=self.course_id,
                target_count=1,
                storage=self.storage,
            )

        self.assertEqual(result["progress"]["finalCount"], 0)
        self.assertEqual(result["progress"]["candidatesAccepted"], 0)
        self.assertGreaterEqual(result["progress"]["candidatesRejected"], 1)
        self.assertGreaterEqual(result["progress"]["candidatesValidated"], 1)

    async def test_starter_malformed_validator_response_rejects_candidate(self) -> None:
        from app import seed_generation
        from app.seed_generation import generate_starter_seeds_for_course

        async def _single_generate(prompt: str, **kwargs: object) -> dict[str, str]:
            payload = {
                "seeds": [
                    {
                        "question": "When are office hours?",
                        "answer": "Office hours are on Tuesdays at 2pm.",
                        "category": "office hours",
                    }
                ]
            }
            return {"answer": json.dumps(payload), "model": SEED_GENERATION_MODEL}

        with (
            patch.object(seed_generation, "MAX_STARTER_SELECTED_CHUNKS", 1),
            patch(
                "app.seed_generation.generate_starter_ollama_completion",
                new=_starter_ollama_side_effect(
                    generate_side_effect=_single_generate,
                    validation_payload="{not valid validator json",
                ),
            ),
        ):
            result = await generate_starter_seeds_for_course(
                course_id=self.course_id,
                target_count=1,
                storage=self.storage,
            )

        self.assertEqual(result["progress"]["finalCount"], 0)
        self.assertGreaterEqual(result["progress"]["candidatesValidated"], 1)
        self.assertGreaterEqual(result["progress"]["candidatesRejected"], 1)

    async def test_starter_missing_course_raises_404(self) -> None:
        from app.seed_generation import generate_starter_seeds_for_course

        with self.assertRaises(HTTPException) as ctx:
            await generate_starter_seeds_for_course(
                course_id="css-999-missing-course",
                target_count=6,
                storage=self.storage,
            )
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_starter_respects_total_ollama_call_cap(self) -> None:
        from app import seed_generation
        from app.seed_generation import generate_starter_seeds_for_course

        call_count = {"n": 0}

        async def _fake_generate(prompt: str, **kwargs: object) -> dict[str, str]:
            call_count["n"] += 1
            if VALIDATION_PROMPT_MARKER in prompt:
                return {
                    "answer": _rejecting_validation_payload(),
                    "model": SEED_GENERATION_MODEL,
                }
            payload = {
                "seeds": [
                    {
                        "question": f"Question {call_count['n']}?",
                        "answer": "A sufficiently detailed answer here.",
                        "category": "general",
                    },
                    {
                        "question": f"Another question {call_count['n']}?",
                        "answer": "Another sufficiently detailed answer.",
                        "category": "general",
                    },
                ]
            }
            return {"answer": json.dumps(payload), "model": SEED_GENERATION_MODEL}

        with (
            patch.object(
                seed_generation,
                "get_starter_max_total_ollama_calls",
                return_value=3,
            ),
            patch.object(seed_generation, "MAX_STARTER_SELECTED_CHUNKS", 10),
            patch(
                "app.seed_generation.generate_starter_ollama_completion",
                new=AsyncMock(side_effect=_fake_generate),
            ),
        ):
            result = await generate_starter_seeds_for_course(
                course_id=self.course_id,
                target_count=50,
                storage=self.storage,
            )

        self.assertEqual(call_count["n"], 3)
        self.assertEqual(result["progress"]["ollamaCalls"], 3)
        self.assertEqual(result["progress"]["generationCalls"], 1)
        self.assertEqual(result["progress"]["validationCalls"], 1)
        self.assertEqual(result["progress"]["finalCount"], 0)
        self.assertGreaterEqual(result["progress"]["candidatesRejected"], 1)

    async def test_starter_validation_ollama_failure_returns_503(self) -> None:
        from app.seed_generation import generate_starter_seeds_for_course

        async def _single_generate(prompt: str, **kwargs: object) -> dict[str, str]:
            payload = {
                "seeds": [
                    {
                        "question": "What is the grading policy?",
                        "answer": "Grades are based on exams and assignments.",
                        "category": "grading",
                    }
                ]
            }
            return {"answer": json.dumps(payload), "model": SEED_GENERATION_MODEL}

        async def _failing_validate(prompt: str, **kwargs: object) -> dict[str, str]:
            if VALIDATION_PROMPT_MARKER in prompt:
                raise HTTPException(
                    status_code=503,
                    detail="Ollama is unavailable. Start Ollama locally and try again.",
                )
            return await _single_generate(prompt, **kwargs)

        with patch(
            "app.seed_generation.generate_starter_ollama_completion",
            new=AsyncMock(side_effect=_failing_validate),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await generate_starter_seeds_for_course(
                    course_id=self.course_id,
                    target_count=1,
                    storage=self.storage,
                )
        self.assertEqual(ctx.exception.status_code, 503)

    async def test_starter_save_false_does_not_persist(self) -> None:
        from app.seed_generation import generate_starter_seeds_for_course

        with (
            patch(
                "app.seed_generation.generate_starter_ollama_completion",
                new=self._unique_seed_side_effect(),
            ),
            patch(
                "app.seed_generation.persist_accepted_seeds",
                new=AsyncMock(),
            ) as mock_persist,
        ):
            result = await generate_starter_seeds_for_course(
                course_id=self.course_id,
                target_count=2,
                save=False,
                storage=self.storage,
            )

        self.assertEqual(result["progress"]["finalCount"], 2)
        self.assertNotIn("persistence", result)
        mock_persist.assert_not_called()

    async def test_starter_save_true_persists_accepted_seeds(self) -> None:
        from app.seed_generation import generate_starter_seeds_for_course

        with (
            patch(
                "app.seed_generation.generate_starter_ollama_completion",
                new=self._unique_seed_side_effect(),
            ),
            patch(
                "app.seed_generation.persist_accepted_seeds",
                new=AsyncMock(
                    return_value={
                        "generatedCount": 2,
                        "savedCount": 2,
                        "alreadyExistingCount": 0,
                        "failedToSaveCount": 0,
                    }
                ),
            ) as mock_persist,
        ):
            result = await generate_starter_seeds_for_course(
                course_id=self.course_id,
                target_count=2,
                save=True,
                storage=self.storage,
            )

        self.assertEqual(result["persistence"]["savedCount"], 2)
        mock_persist.assert_awaited_once()
        self.assertEqual(mock_persist.await_args.kwargs["course_id"], self.course_id)


class StarterSeedEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        root = Path(self._temp_dir.name)
        self.storage = LocalCourseArtifactStorage(
            root_dir=root / "course_data",
            index_dir=root / "indexes",
        )
        self._storage_patch = patch(
            "app.seed_generation.get_course_artifact_storage",
            return_value=self.storage,
        )
        self._storage_patch.start()
        self.client = TestClient(app)
        self.course_id = "css-360-summer-2026-demo"
        self.url = f"/api/courses/{self.course_id}/seeds/generate-starter"

        chunks = [
            _chunk(
                f"chunk-{index:03d}",
                f"Section {index}",
                f"Section {index} syllabus details " * 20,
                order=index,
            )
            for index in range(1, 9)
        ]
        self.storage.save_index(self.course_id, _index(self.course_id, chunks))

    def tearDown(self) -> None:
        self._storage_patch.stop()
        self._temp_dir.cleanup()

    def test_endpoint_success_with_progress(self) -> None:
        call_state = {"n": 0}

        async def _fake_generate(prompt: str, **kwargs: object) -> dict[str, str]:
            call_state["n"] += 1
            n = call_state["n"]
            payload = {
                "seeds": [
                    {
                        "question": f"Endpoint question {n}a?",
                        "answer": f"Answer {n}a with enough detail.",
                        "category": f"Category {2 * n - 1}",
                        "questionType": "scenario" if n % 2 else "direct",
                    },
                    {
                        "question": f"Endpoint question {n}b?",
                        "answer": f"Answer {n}b with enough detail.",
                        "category": f"Category {2 * n}",
                        "questionType": "clarification" if n % 2 else "procedure",
                    },
                ]
            }
            return {"answer": json.dumps(payload), "model": SEED_GENERATION_MODEL}

        with patch(
            "app.seed_generation.generate_starter_ollama_completion",
            new=_starter_ollama_side_effect(generate_side_effect=_fake_generate),
        ):
            response = self.client.post(
                self.url,
                json={"targetCount": 6},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["courseId"], self.course_id)
        self.assertEqual(body["targetCount"], 6)
        self.assertEqual(body["progress"]["finalCount"], 6)
        self.assertEqual(len(body["seeds"]), 6)
        self.assertIn("eligibleChunks", body["progress"])
        self.assertIn("selectedChunks", body["progress"])
        self.assertIn("chunksProcessed", body["progress"])
        self.assertIn("generationCalls", body["progress"])
        self.assertIn("validationCalls", body["progress"])
        self.assertIn("ollamaCalls", body["progress"])
        self.assertIn("candidatesValidated", body["progress"])
        self.assertIn("candidatesAccepted", body["progress"])
        self.assertIn("elapsedMs", body["progress"])
        self.assertIn("savedCount", body["progress"])
        self.assertIn("status", body["progress"])
        self.assertEqual(body["progress"]["savedCount"], 0)
        self.assertEqual(body["progress"]["status"], "ready")
        self.assertGreaterEqual(body["progress"]["elapsedMs"], 0)
        self.assertEqual(body["seeds"][0]["origin"], "ai_generated")
        self.assertEqual(body["seeds"][0]["status"], "generated")
        self.assertGreaterEqual(body["seeds"][0]["validation"]["score"], 0.8)

    def test_endpoint_missing_course(self) -> None:
        response = self.client.post(
            "/api/courses/css-999-missing-course/seeds/generate-starter",
            json={"targetCount": 6},
        )
        self.assertEqual(response.status_code, 404)

    def test_endpoint_target_count_validation(self) -> None:
        response = self.client.post(
            self.url,
            json={"targetCount": 51},
        )
        self.assertEqual(response.status_code, 422)

    def test_endpoint_generate_only_when_save_false(self) -> None:
        call_state = {"n": 0}

        async def _fake_generate(prompt: str, **kwargs: object) -> dict[str, str]:
            call_state["n"] += 1
            payload = {
                "seeds": [
                    {
                        "question": "Generate-only question here?",
                        "answer": "Generate-only answer with enough detail.",
                        "category": "general",
                    }
                ]
            }
            return {"answer": json.dumps(payload), "model": SEED_GENERATION_MODEL}

        with (
            patch(
                "app.seed_generation.generate_starter_ollama_completion",
                new=_starter_ollama_side_effect(generate_side_effect=_fake_generate),
            ),
            patch(
                "app.seed_generation.persist_accepted_seeds",
                new=AsyncMock(),
            ) as mock_persist,
        ):
            response = self.client.post(
                self.url,
                json={"targetCount": 1, "save": False},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json().get("persistence"))
        mock_persist.assert_not_called()

    def test_endpoint_save_true_returns_persistence_metadata(self) -> None:
        async def _fake_generate(prompt: str, **kwargs: object) -> dict[str, str]:
            payload = {
                "seeds": [
                    {
                        "question": "Saved starter question here?",
                        "answer": "Saved starter answer with enough detail.",
                        "category": "general",
                    }
                ]
            }
            return {"answer": json.dumps(payload), "model": SEED_GENERATION_MODEL}

        with (
            patch(
                "app.seed_generation.generate_starter_ollama_completion",
                new=_starter_ollama_side_effect(generate_side_effect=_fake_generate),
            ),
            patch(
                "app.seed_generation.persist_accepted_seeds",
                new=AsyncMock(
                    return_value={
                        "generatedCount": 1,
                        "savedCount": 1,
                        "alreadyExistingCount": 0,
                        "failedToSaveCount": 0,
                    }
                ),
            ),
        ):
            response = self.client.post(
                self.url,
                json={"targetCount": 1, "save": True},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["persistence"]["savedCount"], 1)
        self.assertEqual(body["persistence"]["generatedCount"], 1)

    def test_endpoint_save_true_missing_firebase_configuration(self) -> None:
        async def _fake_generate(prompt: str, **kwargs: object) -> dict[str, str]:
            payload = {
                "seeds": [
                    {
                        "question": "Question for firebase config test?",
                        "answer": "Answer for firebase config test here.",
                        "category": "general",
                    }
                ]
            }
            return {"answer": json.dumps(payload), "model": SEED_GENERATION_MODEL}

        from app.firebase_seeds import FirebaseConfigurationError

        with (
            patch(
                "app.seed_generation.generate_starter_ollama_completion",
                new=_starter_ollama_side_effect(generate_side_effect=_fake_generate),
            ),
            patch(
                "app.seed_generation.persist_accepted_seeds",
                new=AsyncMock(
                    side_effect=FirebaseConfigurationError(
                        "Firebase is not configured for seed persistence."
                    )
                ),
            ),
        ):
            response = self.client.post(
                self.url,
                json={"targetCount": 1, "save": True},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("Firebase is not configured", response.json()["detail"])
