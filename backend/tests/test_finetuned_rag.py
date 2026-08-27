"""Tests for Fine-Tuned + RAG (retrieval + remote LoRA generation)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.finetuned_rag import (
    build_finetuned_rag_prompt,
    generate_course_finetuned_rag_answer,
)
from app.main import app
from app.storage import LocalCourseArtifactStorage


def _chunk(
    chunk_id: str,
    section_title: str,
    text: str,
    embedding: list[float],
    order: int = 0,
) -> dict:
    return {
        "chunkId": chunk_id,
        "sectionTitle": section_title,
        "text": text,
        "order": order,
        "embedding": embedding,
    }


def _index(course_id: str, chunks: list[dict]) -> dict:
    return {
        "courseId": course_id,
        "embeddingModel": "nomic-embed-text",
        "chunkCount": len(chunks),
        "chunks": chunks,
    }


class BuildFinetunedRagPromptTests(unittest.TestCase):
    def test_prompt_includes_grounding_rules_and_context(self) -> None:
        prompt = build_finetuned_rag_prompt(
            "What is the late policy?",
            [
                {
                    "chunk_id": "c1",
                    "section": "Late Policy",
                    "text": "Late work loses 10% per day.",
                    "score": 0.9,
                }
            ],
        )
        self.assertIn("Answer only from the supplied syllabus context", prompt)
        self.assertIn("Do not invent policies, dates, percentages, locations, or requirements", prompt)
        self.assertIn(
            "the syllabus context provided does not contain enough information",
            prompt,
        )
        self.assertIn("factual source of truth", prompt)
        self.assertIn("Late work loses 10% per day.", prompt)
        self.assertIn("What is the late policy?", prompt)
        self.assertIn("Late Policy", prompt)


class FineTunedRagGenerationTests(unittest.IsolatedAsyncioTestCase):
    """Fine-Tuned + RAG, which resolves a course model exactly as Fine-Tuned does.

    `resolve_current_course_model` reads PostgreSQL, and no test in this suite
    has a database. It is patched with the version the course would have
    resolved to — which is also the assertion that the version reaches the
    inference client, rather than the cluster being left to choose one.
    """

    def _resolved_model(self, course_id: str, version: str = "v1"):
        return patch(
            "app.finetuned_rag.resolve_current_course_model",
            return_value={"courseId": course_id, "version": version},
        )

    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        root = Path(self._temp_dir.name)
        self.storage = LocalCourseArtifactStorage(
            root_dir=root / "course_data",
            index_dir=root / "indexes",
        )
        self.css430_id = "css-430-summer-2026-ibce"
        self.other_id = "css-360-winter-2026-demo"
        self.storage.save_index(
            self.css430_id,
            _index(
                self.css430_id,
                [
                    _chunk(
                        "css430-late-1",
                        "CSS 430 Late Policy",
                        "CSS 430 late work may be submitted within 24 hours for half credit.",
                        [1.0, 0.0, 0.0],
                    ),
                    _chunk(
                        "css430-office-1",
                        "CSS 430 Office Hours",
                        "CSS 430 office hours are Tuesdays at 2pm.",
                        [0.0, 1.0, 0.0],
                        order=1,
                    ),
                ],
            ),
        )
        self.storage.save_index(
            self.other_id,
            _index(
                self.other_id,
                [
                    _chunk(
                        "css360-late-1",
                        "CSS 360 Late Policy",
                        "CSS 360 allows one 48-hour late token per quarter.",
                        [1.0, 0.0, 0.0],
                    ),
                ],
            ),
        )
        self._env_backup = os.environ.get("FINETUNED_SERVICE_URL")
        os.environ["FINETUNED_SERVICE_URL"] = "http://example-node:8001"

    def tearDown(self) -> None:
        if self._env_backup is None:
            os.environ.pop("FINETUNED_SERVICE_URL", None)
        else:
            os.environ["FINETUNED_SERVICE_URL"] = self._env_backup
        self._temp_dir.cleanup()

    async def test_successful_retrieval_and_generation(self) -> None:
        with (
            patch(
                "app.finetuned_rag.retrieve_course_syllabus_chunks",
                new=AsyncMock(
                    return_value=(
                        "nomic-embed-text",
                        [
                            {
                                "chunk_id": "css430-late-1",
                                "section": "CSS 430 Late Policy",
                                "text": (
                                    "CSS 430 late work may be submitted within "
                                    "24 hours for half credit."
                                ),
                                "score": 0.95,
                            }
                        ],
                    )
                ),
            ),
            self._resolved_model(self.css430_id),
            patch(
                "app.finetuned_rag.generate_finetuned_response",
                new=AsyncMock(
                    return_value={
                        "answer": "Half credit within 24 hours.",
                        "model": "meta-llama/Llama-3.2-3B-Instruct",
                        "adapter_loaded": True,
                        "generation_seconds": 1.1,
                        "response_type": "fineTuned",
                    }
                ),
            ) as mock_ft,
        ):
            result = await generate_course_finetuned_rag_answer(
                course_id=self.css430_id,
                question="What is the late policy?",
                storage=self.storage,
            )

        self.assertEqual(result["courseId"], self.css430_id)
        self.assertEqual(result["answer"], "Half credit within 24 hours.")
        self.assertEqual(result["responseType"], "fineTunedRag")
        self.assertTrue(result["adapterLoaded"])
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["chunkId"], "css430-late-1")
        self.assertEqual(result["sources"][0]["sectionTitle"], "CSS 430 Late Policy")

        sent_prompt = mock_ft.await_args.args[0]
        self.assertIn("CSS 430 late work may be submitted within 24 hours", sent_prompt)
        self.assertIn("What is the late policy?", sent_prompt)
        self.assertIn("Answer only from the supplied syllabus context", sent_prompt)

    async def test_course_id_isolation(self) -> None:
        captured_course_ids: list[str] = []

        async def fake_retrieve(
            course_id: str,
            question: str,
            top_k: int = 3,
            storage=None,
        ):
            captured_course_ids.append(course_id)
            prefix = "css430" if "430" in course_id else "css360"
            return (
                "nomic-embed-text",
                [
                    {
                        "chunk_id": f"{prefix}-late-1",
                        "section": f"{prefix} Late Policy",
                        "text": f"{prefix} late policy text",
                        "score": 0.9,
                    }
                ],
            )

        with (
            patch(
                "app.finetuned_rag.retrieve_course_syllabus_chunks",
                new=AsyncMock(side_effect=fake_retrieve),
            ),
            self._resolved_model(self.css430_id),
            patch(
                "app.finetuned_rag.generate_finetuned_response",
                new=AsyncMock(
                    return_value={
                        "answer": "ok",
                        "model": "meta-llama/Llama-3.2-3B-Instruct",
                        "adapter_loaded": True,
                        "generation_seconds": 0.5,
                        "response_type": "fineTuned",
                    }
                ),
            ),
        ):
            first = await generate_course_finetuned_rag_answer(
                course_id=self.css430_id,
                question="late policy",
            )
            second = await generate_course_finetuned_rag_answer(
                course_id=self.other_id,
                question="late policy",
            )

        self.assertEqual(captured_course_ids, [self.css430_id, self.other_id])
        self.assertEqual(first["sources"][0]["chunkId"], "css430-late-1")
        self.assertEqual(second["sources"][0]["chunkId"], "css360-late-1")

    async def test_no_retrieval_results_does_not_call_model(self) -> None:
        mock_ft = AsyncMock()
        with (
            patch(
                "app.finetuned_rag.retrieve_course_syllabus_chunks",
                new=AsyncMock(return_value=("nomic-embed-text", [])),
            ),
            patch("app.finetuned_rag.generate_finetuned_response", new=mock_ft),
            self.assertRaises(HTTPException) as ctx,
        ):
            await generate_course_finetuned_rag_answer(
                course_id=self.css430_id,
                question="What is the late policy?",
            )

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("No usable syllabus context", ctx.exception.detail)
        mock_ft.assert_not_awaited()

    async def test_fine_tuned_service_unavailable(self) -> None:
        with (
            patch(
                "app.finetuned_rag.retrieve_course_syllabus_chunks",
                new=AsyncMock(
                    return_value=(
                        "nomic-embed-text",
                        [
                            {
                                "chunk_id": "css430-late-1",
                                "section": "CSS 430 Late Policy",
                                "text": "Late policy text",
                                "score": 0.9,
                            }
                        ],
                    )
                ),
            ),
            self._resolved_model(self.css430_id),
            patch(
                "app.finetuned_rag.generate_finetuned_response",
                new=AsyncMock(
                    side_effect=HTTPException(
                        status_code=503,
                        detail="Fine-tuned inference service is unavailable.",
                    )
                ),
            ),
            self.assertRaises(HTTPException) as ctx,
        ):
            await generate_course_finetuned_rag_answer(
                course_id=self.css430_id,
                question="What is the late policy?",
            )

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("unavailable", ctx.exception.detail.lower())

    async def test_malformed_fine_tuned_response(self) -> None:
        with (
            patch(
                "app.finetuned_rag.retrieve_course_syllabus_chunks",
                new=AsyncMock(
                    return_value=(
                        "nomic-embed-text",
                        [
                            {
                                "chunk_id": "css430-late-1",
                                "section": "CSS 430 Late Policy",
                                "text": "Late policy text",
                                "score": 0.9,
                            }
                        ],
                    )
                ),
            ),
            self._resolved_model(self.css430_id),
            patch(
                "app.finetuned_rag.generate_finetuned_response",
                new=AsyncMock(
                    side_effect=HTTPException(
                        status_code=502,
                        detail=(
                            "Fine-tuned service returned a malformed response "
                            "(expected a JSON object)."
                        ),
                    )
                ),
            ),
            self.assertRaises(HTTPException) as ctx,
        ):
            await generate_course_finetuned_rag_answer(
                course_id=self.css430_id,
                question="What is the late policy?",
            )

        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn("malformed", ctx.exception.detail.lower())


class FineTunedRagEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.course_id = "css-430-summer-2026-ibce"

    def test_endpoint_success_returns_sources(self) -> None:
        async def fake_generate(*, course_id: str, question: str, top_k: int = 3):
            self.assertEqual(course_id, self.course_id)
            self.assertEqual(question, "What is the late policy?")
            return {
                "courseId": course_id,
                "answer": "Half credit within 24 hours.",
                "model": "meta-llama/Llama-3.2-3B-Instruct",
                "adapterLoaded": True,
                "generationSeconds": 0.8,
                "sources": [
                    {
                        "chunkId": "css430-late-1",
                        "sectionTitle": "CSS 430 Late Policy",
                        "text": "Late policy excerpt",
                        "score": 0.95,
                    }
                ],
                "retrievedChunks": [
                    {
                        "chunkId": "css430-late-1",
                        "section": "CSS 430 Late Policy",
                        "text": "Late policy excerpt",
                        "score": 0.95,
                    }
                ],
                "responseType": "fineTunedRag",
            }

        with patch(
            "app.main.generate_course_finetuned_rag_answer",
            new=AsyncMock(side_effect=fake_generate),
        ):
            response = self.client.post(
                "/fine-tuned-rag/generate",
                json={
                    "courseId": self.course_id,
                    "question": "What is the late policy?",
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["responseType"], "fineTunedRag")
        self.assertEqual(body["courseId"], self.course_id)
        self.assertEqual(body["answer"], "Half credit within 24 hours.")
        self.assertTrue(body["adapterLoaded"])
        self.assertEqual(len(body["sources"]), 1)
        self.assertEqual(body["sources"][0]["sectionTitle"], "CSS 430 Late Policy")
        self.assertEqual(body["retrievedChunks"][0]["chunkId"], "css430-late-1")

    def test_endpoint_propagates_service_unavailable(self) -> None:
        with patch(
            "app.main.generate_course_finetuned_rag_answer",
            new=AsyncMock(
                side_effect=HTTPException(
                    status_code=503,
                    detail="Fine-tuned inference service is unavailable.",
                )
            ),
        ):
            response = self.client.post(
                "/fine-tuned-rag/generate",
                json={
                    "courseId": self.course_id,
                    "question": "What is the late policy?",
                },
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("unavailable", response.json()["detail"].lower())


if __name__ == "__main__":
    unittest.main()
