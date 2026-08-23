"""Tests for AI seed generation service and temporary endpoint."""

from __future__ import annotations

import json
import re
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
            # Batch validation prompts list multiple "### Candidate id:" blocks.
            candidate_ids = re.findall(
                r"### Candidate id:\s*(\S+)",
                prompt,
            )
            if len(candidate_ids) > 1:
                try:
                    single = json.loads(validation_answer)
                except json.JSONDecodeError:
                    single = json.loads(_passing_validation_payload())
                results = []
                for candidate_id in candidate_ids:
                    item = dict(single)
                    item["candidateId"] = candidate_id
                    results.append(item)
                return {
                    "answer": json.dumps({"results": results}),
                    "model": SEED_GENERATION_MODEL,
                }
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


async def _orthogonal_embed(texts, *, model=None):
    """Deterministic per-text embeddings so single-item cache embeds stay distinct."""
    # Fixed vocabulary axis: hash each text into a unique unit vector dimension.
    # Using content (not batch index) keeps cache backfill + candidate embeds comparable.
    vocabulary: list[str] = []
    for text in texts:
        key = str(text)
        if key not in vocabulary:
            vocabulary.append(key)
    # Grow a stable axis set for this call; include prior keys via hash buckets.
    dim = max(16, len(vocabulary) + 4)
    embeddings: list[list[float]] = []
    for text in texts:
        vector = [0.0] * dim
        # Stable non-colliding-ish bucket from text content.
        bucket = sum(ord(ch) for ch in str(text)) % dim
        vector[bucket] = 1.0
        # Disambiguate near-collisions with a second sparse bit from length.
        vector[(bucket + len(str(text)) + 1) % dim] = 0.5
        embeddings.append(vector)
    return {"embeddings": embeddings, "model": model or "test-embed"}


def _mock_fact(
    fact_id: str,
    *,
    statement: str,
    evidence_quote: str | None = None,
    source_chunk_ids: list[str] | None = None,
    importance: str = "high",
    importance_score: float = 0.9,
    ask: float = 0.85,
    complexity: int = 1,
    usefulness: float = 0.85,
    kind: str = "policy",
    scope: str = "course_wide",
) -> dict:
    quote = evidence_quote or statement
    return {
        "factId": fact_id,
        "statement": statement,
        "importance": importance,
        "importanceScore": importance_score,
        "studentAskLikelihood": ask,
        "complexity": complexity,
        "usefulnessScore": usefulness,
        "sourceChunkIds": source_chunk_ids or ["chunk-001"],
        "evidenceQuote": quote,
        "kind": kind,
        "scope": scope,
        "seriesKey": None,
        "assignmentGroup": None,
        "seriesOrdinal": None,
    }


def _mock_inventory(facts: list[dict]) -> dict:
    return {
        "model": SEED_GENERATION_MODEL,
        "facts": facts,
        "factCount": len(facts),
        "droppedCount": 0,
        "duplicatesRemoved": 0,
        "fallbackUsed": False,
        "cached": False,
        "countsByScope": {},
        "countsByKind": {},
        "countsBySeries": {},
    }


def _facts_from_index_chunks(chunks: list[dict], *, limit: int | None = None) -> list[dict]:
    kinds = [
        "late_work",
        "attendance",
        "grading",
        "contact",
        "communication",
        "requirement",
        "policy",
        "office_hours",
        "accommodation",
        "exam",
        "tools",
        "team_project",
    ]
    facts: list[dict] = []
    for index, chunk in enumerate(chunks, start=1):
        text = str(chunk.get("text") or "").strip()
        if len(text) < 80:
            continue
        statement = text[:160].strip()
        facts.append(
            _mock_fact(
                f"fact-{index:02d}",
                statement=statement,
                evidence_quote=statement[:120],
                source_chunk_ids=[str(chunk["chunkId"])],
                complexity=2 if index == 1 else 1,
                kind=kinds[(index - 1) % len(kinds)],
            )
        )
        if limit is not None and len(facts) >= limit:
            break
    return facts


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
        self.chunks = chunks
        self.storage.save_index(self.course_id, _index(self.course_id, chunks))
        self.facts = _facts_from_index_chunks(chunks)
        self._inventory_patch = patch(
            "app.seed_generation.load_or_build_fact_inventory",
            new=AsyncMock(return_value=_mock_inventory(self.facts)),
        )
        self._inventory_patch.start()
        # Give tests a predictable slot map: one slot per fact, up to plenty of capacity.
        self._allocation_patch = patch(
            "app.seed_generation.allocate_slots",
            side_effect=self._allocate_one_each,
        )
        self._allocation_patch.start()
        self._embed_patch = patch(
            "app.seed_generation.embed_ollama_texts",
            new=_orthogonal_embed,
        )
        self._embed_patch.start()
        self._plan_spy = patch(
            "app.syllabus_plan.plan_syllabus_topics",
            new=AsyncMock(side_effect=AssertionError("topic planner must not be called")),
        )
        self._plan_spy.start()

    def _allocate_one_each(self, facts, *, target_count=50):
        allocations = []
        remaining = target_count
        for fact in facts:
            slots = 1 if remaining > 0 else 0
            if slots:
                remaining -= 1
            allocations.append(
                {
                    "factId": fact["factId"],
                    "slotCount": slots,
                    "desiredSlots": 1,
                    "rankingScore": float(fact.get("usefulnessScore") or 0.5),
                    "suggestedStyles": ["factual"],
                    "reasons": ["test_allocation"],
                }
            )
        allocated = sum(item["slotCount"] for item in allocations)
        return {
            "allocations": allocations,
            "summary": {
                "targetCount": target_count,
                "allocatedSlots": allocated,
                "byScope": {},
                "byKind": {},
                "bySeries": {},
                "skippedFacts": [],
                "cappedFacts": [],
                "caps": {},
                "courseWideAllocated": allocated,
                "courseWideReserve": 0,
            },
            "ranking": [],
        }

    def tearDown(self) -> None:
        self._plan_spy.stop()
        self._embed_patch.stop()
        self._allocation_patch.stop()
        self._inventory_patch.stop()
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
            fact_ids = re.findall(r"factId:\s*(\S+)", prompt)
            if len(fact_ids) > 1:
                seeds = []
                for index, fact_id in enumerate(fact_ids):
                    seeds.append(
                        {
                            "factId": fact_id,
                            "question": f"Unique question {n}-{index}?",
                            "answer": f"Answer {n}-{index} with enough detail.",
                            "category": categories[
                                (index + n) % len(categories)
                            ],
                            "questionType": question_types[
                                (index + n) % len(question_types)
                            ],
                        }
                    )
                return {
                    "answer": json.dumps({"seeds": seeds}),
                    "model": SEED_GENERATION_MODEL,
                }
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
        self.assertEqual(result["progress"]["planningCalls"], 0)
        self.assertEqual(result["progress"]["mergeCalls"], 0)
        self.assertGreaterEqual(result["progress"]["factCount"], 1)
        self.assertGreaterEqual(result["progress"]["allocatedSlots"], result["progress"]["finalCount"])
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
            result["progress"]["factExtractionCalls"]
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
        self.assertTrue(result["seeds"][0].get("factId"))
        self.assertTrue(result["seeds"][0].get("evidenceQuote"))
        self.assertEqual(mock_generate.await_args.kwargs["think"], False)
        self.assertEqual(mock_generate.await_args.kwargs["model"], SEED_GENERATION_MODEL)

    async def test_starter_does_not_call_topic_planner(self) -> None:
        from app.seed_generation import generate_starter_seeds_for_course
        from app.syllabus_plan import plan_syllabus_topics

        with patch(
            "app.seed_generation.generate_starter_ollama_completion",
            new=self._unique_seed_side_effect(),
        ):
            await generate_starter_seeds_for_course(
                course_id=self.course_id,
                target_count=2,
                storage=self.storage,
            )

        plan_syllabus_topics.assert_not_called()

    async def test_starter_only_generates_for_allocated_slots(self) -> None:
        from app.seed_allocation import allocate_slots
        from app.seed_generation import generate_starter_seeds_for_course

        facts = [
            _mock_fact(
                "fact-keep",
                statement="Students may use one 48-hour extension per quarter.",
                evidence_quote="one 48-hour extension per quarter",
                source_chunk_ids=["chunk-001"],
                complexity=3,
                usefulness=0.9,
                kind="late_work",
            ),
            _mock_fact(
                "fact-skip",
                statement="The campus pantry is an optional food resource.",
                evidence_quote="optional food resource",
                source_chunk_ids=["chunk-002"],
                importance="low",
                importance_score=0.3,
                ask=0.2,
                usefulness=0.2,
                kind="resource",
                scope="resource",
            ),
        ]
        self._inventory_patch.stop()
        self._inventory_patch = patch(
            "app.seed_generation.load_or_build_fact_inventory",
            new=AsyncMock(return_value=_mock_inventory(facts)),
        )
        self._inventory_patch.start()
        # Use the real allocator so 0-slot resource facts are skipped.
        self._allocation_patch.stop()
        self._allocation_patch = patch(
            "app.seed_generation.allocate_slots",
            side_effect=allocate_slots,
        )
        self._allocation_patch.start()

        requested_counts: list[int] = []
        fact_ids_seen: list[str] = []

        async def _track_generate(prompt: str, **kwargs: object) -> dict[str, str]:
            if "Fact id: fact-skip" in prompt:
                raise AssertionError("0-slot fact must not generate")
            if "Fact id: fact-keep" in prompt:
                fact_ids_seen.append("fact-keep")
            # Count requested examples from prompt wording.
            if "Generate exactly 1 student" in prompt or "exactly one strong" in prompt.lower():
                requested_counts.append(1)
            elif "Generate exactly 2 student" in prompt or "exactly 2 materially" in prompt:
                requested_counts.append(2)
            elif "Generate exactly 3 student" in prompt or "exactly 3 materially" in prompt:
                requested_counts.append(3)
            payload = {
                "seeds": [
                    {
                        "question": f"Late work question {len(requested_counts)}-{index}?",
                        "answer": "Use the one 48-hour extension when allowed.",
                        "category": "late work",
                        "questionType": "direct",
                    }
                    for index in range(1, (requested_counts[-1] if requested_counts else 1) + 1)
                ]
            }
            return {"answer": json.dumps(payload), "model": SEED_GENERATION_MODEL}

        with patch(
            "app.seed_generation.generate_starter_ollama_completion",
            new=_starter_ollama_side_effect(generate_side_effect=_track_generate),
        ):
            result = await generate_starter_seeds_for_course(
                course_id=self.course_id,
                target_count=5,
                storage=self.storage,
            )

        self.assertIn("fact-keep", fact_ids_seen)
        self.assertGreaterEqual(sum(requested_counts), 1)
        self.assertLessEqual(sum(requested_counts), 5)
        self.assertLessEqual(result["progress"]["finalCount"], 5)
        for seed in result["seeds"]:
            self.assertEqual(seed["factId"], "fact-keep")
            self.assertEqual(seed["sourceChunkIds"], ["chunk-001"])
            self.assertIn("48-hour", seed["evidenceQuote"])

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
                        "question": "What does this syllabus section cover?",
                        "answer": (
                            "This section covers syllabus policies and assignments."
                        ),
                        "category": "policy",
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

        with patch(
            "app.seed_generation.generate_starter_ollama_completion",
            new=_starter_ollama_side_effect(
                generate_side_effect=_single_generate,
                validation_payload=_rejecting_validation_payload(),
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

        with patch(
            "app.seed_generation.generate_starter_ollama_completion",
            new=_starter_ollama_side_effect(
                generate_side_effect=_single_generate,
                validation_payload="{not valid validator json",
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
                candidate_ids = re.findall(r"### Candidate id:\s*(\S+)", prompt)
                if len(candidate_ids) > 1:
                    single = json.loads(_rejecting_validation_payload())
                    results = []
                    for candidate_id in candidate_ids:
                        item = dict(single)
                        item["candidateId"] = candidate_id
                        results.append(item)
                    return {
                        "answer": json.dumps({"results": results}),
                        "model": SEED_GENERATION_MODEL,
                    }
                return {
                    "answer": _rejecting_validation_payload(),
                    "model": SEED_GENERATION_MODEL,
                }
            fact_ids = re.findall(r"factId:\s*(\S+)", prompt)
            if len(fact_ids) > 1:
                seeds = [
                    {
                        "factId": fact_id,
                        "question": f"Question {call_count['n']}-{index}?",
                        "answer": "A sufficiently detailed answer here.",
                        "category": "general",
                    }
                    for index, fact_id in enumerate(fact_ids)
                ]
                return {
                    "answer": json.dumps({"seeds": seeds}),
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
                return_value=2,
            ),
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

        # Inventory is mocked (0 extraction calls). Cap allows 1 gen + 1 val.
        self.assertEqual(call_count["n"], 2)
        self.assertEqual(result["progress"]["ollamaCalls"], 2)
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
        persisted_seeds = mock_persist.await_args.kwargs["seeds"]
        self.assertTrue(persisted_seeds[0].get("factId"))
        self.assertTrue(persisted_seeds[0].get("sourceChunkIds"))


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
        facts = _facts_from_index_chunks(chunks)
        self._inventory_patch = patch(
            "app.seed_generation.load_or_build_fact_inventory",
            new=AsyncMock(return_value=_mock_inventory(facts)),
        )
        self._inventory_patch.start()

        def _allocate_one_each(facts_list, *, target_count=50):
            allocations = []
            remaining = target_count
            for fact in facts_list:
                slots = 1 if remaining > 0 else 0
                if slots:
                    remaining -= 1
                allocations.append(
                    {
                        "factId": fact["factId"],
                        "slotCount": slots,
                        "desiredSlots": 1,
                        "rankingScore": 0.8,
                        "suggestedStyles": ["factual"],
                        "reasons": ["test_allocation"],
                    }
                )
            allocated = sum(item["slotCount"] for item in allocations)
            return {
                "allocations": allocations,
                "summary": {
                    "targetCount": target_count,
                    "allocatedSlots": allocated,
                    "byScope": {},
                    "byKind": {},
                    "bySeries": {},
                    "skippedFacts": [],
                    "cappedFacts": [],
                    "caps": {},
                    "courseWideAllocated": allocated,
                    "courseWideReserve": 0,
                },
                "ranking": [],
            }

        self._allocation_patch = patch(
            "app.seed_generation.allocate_slots",
            side_effect=_allocate_one_each,
        )
        self._allocation_patch.start()
        self._embed_patch = patch(
            "app.seed_generation.embed_ollama_texts",
            new=_orthogonal_embed,
        )
        self._embed_patch.start()

    def tearDown(self) -> None:
        self._embed_patch.stop()
        self._allocation_patch.stop()
        self._inventory_patch.stop()
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
        self.assertIn("factCount", body["progress"])
        self.assertIn("allocatedSlots", body["progress"])
        self.assertIn("allocatedFactCount", body["progress"])
        self.assertIn("factExtractionCalls", body["progress"])
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
        self.assertEqual(body["progress"]["planningCalls"], 0)
        self.assertGreaterEqual(body["progress"]["elapsedMs"], 0)
        self.assertEqual(body["seeds"][0]["origin"], "ai_generated")
        self.assertEqual(body["seeds"][0]["status"], "generated")
        self.assertGreaterEqual(body["seeds"][0]["validation"]["score"], 0.8)
        self.assertTrue(body["seeds"][0].get("factId"))
        self.assertTrue(body["seeds"][0].get("sourceChunkIds"))

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
            # A save=true run also reconciles starter status afterwards,
            # which counts the course's seeds and writes the status row.
            # Stubbing persistence alone left that second round trip live, and
            # it rewrote this course in the real database on every suite run.
            patch(
                "app.main.reconcile_starter_seed_generation",
                new=AsyncMock(return_value=None),
            ) as mock_reconcile,
        ):
            response = self.client.post(
                self.url,
                json={"targetCount": 1, "save": True},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["persistence"]["savedCount"], 1)
        self.assertEqual(body["persistence"]["generatedCount"], 1)
        # The route must still reconcile; the point is that it is stubbed here,
        # not that it stopped happening.
        mock_reconcile.assert_awaited_once()

    def test_endpoint_save_true_reports_unavailable_storage(self) -> None:
        """A run that could not save must 503, not report a successful save.

        This is the failure the storage layer must never swallow: the seeds are
        gone either way, and a 200 would tell a professor their fifty examples
        were persisted when nothing was written.
        """

        async def _fake_generate(prompt: str, **kwargs: object) -> dict[str, str]:
            payload = {
                "seeds": [
                    {
                        "question": "Question for storage failure test?",
                        "answer": "Answer for storage failure test here.",
                        "category": "general",
                    }
                ]
            }
            return {"answer": json.dumps(payload), "model": SEED_GENERATION_MODEL}

        from fastapi import HTTPException

        with (
            patch(
                "app.seed_generation.generate_starter_ollama_completion",
                new=_starter_ollama_side_effect(generate_side_effect=_fake_generate),
            ),
            patch(
                "app.seed_generation.persist_accepted_seeds",
                new=AsyncMock(
                    side_effect=HTTPException(
                        status_code=503,
                        detail="PostgreSQL is unavailable while saving seed examples.",
                    )
                ),
            ),
        ):
            response = self.client.post(
                self.url,
                json={"targetCount": 1, "save": True},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("PostgreSQL is unavailable", response.json()["detail"])


class StarterRunCeilingTests(unittest.IsolatedAsyncioTestCase):
    """A short run has to say why it was short.

    Modelled on the CSS 350 regression: target 50, 140 eligible chunks, and an
    inventory of 8 facts. Nine examples came out, and the run reported that as
    "9 of 50" with nothing to distinguish a syllabus that had little to say from
    an extractor that had silently dropped most of it.

    The real allocator runs in these tests. Its arithmetic is the thing being
    described — 8 facts can never fill 50 slots, whatever the target says — so
    stubbing it would test the stub.
    """

    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        root = Path(self._temp_dir.name)
        self.storage = LocalCourseArtifactStorage(
            root_dir=root / "course_data",
            index_dir=root / "indexes",
        )
        self.course_id = "css-350-winter-2026-drlb"
        self.chunks = [
            _chunk(
                f"chunk-{index:03d}",
                "CSS 350 Syllabus",
                f"Policy {index}: coursework for unit {index} is described here "
                f"in enough detail to be a usable evidence chunk. " * 3,
                order=index,
            )
            for index in range(1, 21)
        ]
        self.storage.save_index(self.course_id, _index(self.course_id, self.chunks))
        self._embed_patch = patch(
            "app.seed_generation.embed_ollama_texts",
            new=_orthogonal_embed,
        )
        self._embed_patch.start()

    def tearDown(self) -> None:
        self._embed_patch.stop()
        self._temp_dir.cleanup()

    def _facts(self, count: int) -> list[dict]:
        kinds = ["late_work", "grading", "attendance", "exam", "policy", "contact"]
        return [
            _mock_fact(
                f"fact-{index:02d}",
                statement=(
                    f"Policy {index}: coursework for unit {index} is described "
                    "here in enough detail to be a usable evidence chunk."
                ),
                source_chunk_ids=[f"chunk-{index:03d}"],
                kind=kinds[(index - 1) % len(kinds)],
                complexity=1,
            )
            for index in range(1, count + 1)
        ]

    def _inventory_patch(self, facts: list[dict], *, extraction: dict | None = None):
        inventory = _mock_inventory(facts)
        if extraction is not None:
            inventory["extraction"] = extraction
        return patch(
            "app.seed_generation.load_or_build_fact_inventory",
            new=AsyncMock(return_value=inventory),
        )

    def _unique_generation(self) -> AsyncMock:
        state = {"n": 0}

        async def _fake_generate(prompt: str, **kwargs: object) -> dict[str, str]:
            state["n"] += 1
            n = state["n"]
            question_type = "scenario" if n % 3 == 0 else "direct"
            return {
                "answer": json.dumps(
                    {
                        "seeds": [
                            {
                                "question": f"Distinct question number {n}?",
                                "answer": f"Answer {n} with enough detail to pass.",
                                "category": f"Category {n % 7}",
                                "questionType": question_type,
                            }
                        ]
                    }
                ),
                "model": SEED_GENERATION_MODEL,
            }

        return _starter_ollama_side_effect(generate_side_effect=_fake_generate)

    async def _run(self, facts: list[dict], *, target: int, generate=None, **kwargs):
        from app.seed_generation import generate_starter_seeds_for_course

        with self._inventory_patch(facts, extraction=kwargs.pop("extraction", None)):
            with patch(
                "app.seed_generation.generate_starter_ollama_completion",
                new=generate or self._unique_generation(),
            ):
                return await generate_starter_seeds_for_course(
                    course_id=self.course_id,
                    target_count=target,
                    storage=self.storage,
                    **kwargs,
                )

    async def test_a_fact_starved_run_names_the_inventory_as_the_limit(self) -> None:
        result = await self._run(self._facts(8), target=50)
        progress = result["progress"]

        # The shape of the regression: the target was never reachable.
        self.assertLess(progress["allocatedSlots"], 50)
        self.assertEqual(progress["achievableCeiling"], progress["allocatedSlots"])
        self.assertEqual(progress["limitingFactor"], "fact_inventory")
        self.assertEqual(progress["status"], "partial")
        self.assertLessEqual(progress["finalCount"], progress["allocatedSlots"])
        self.assertEqual(len(result["seeds"]), progress["finalCount"])

    async def test_the_ceiling_is_what_the_facts_could_ever_support(self) -> None:
        """8 facts cannot exceed 8 × MAX_SLOTS_PER_FACT, whatever the target."""
        from app.seed_allocation import MAX_SLOTS_PER_FACT

        result = await self._run(self._facts(8), target=50)
        progress = result["progress"]

        self.assertLessEqual(progress["achievableCeiling"], 8 * MAX_SLOTS_PER_FACT)
        self.assertGreater(progress["achievableCeiling"], 0)

    async def test_the_scenario_minimum_cannot_exceed_the_ceiling(self) -> None:
        """The reported 15-of-11 contradiction.

        The minimum used to come from targetCount, so a run with 11 slots asked
        for 15 scenario questions — a goal that could not be met by arithmetic,
        stated as though it could.
        """
        result = await self._run(self._facts(8), target=50)
        progress = result["progress"]

        self.assertLessEqual(
            progress["scenarioOrClarificationMinimum"],
            progress["achievableCeiling"],
        )
        self.assertIn("scenarioOrClarificationActual", progress)
        self.assertLessEqual(
            progress["scenarioOrClarificationActual"], progress["finalCount"]
        )

    async def test_a_healthy_run_reports_no_limit(self) -> None:
        result = await self._run(self._facts(20), target=5)
        progress = result["progress"]

        self.assertEqual(progress["finalCount"], 5)
        self.assertEqual(progress["achievableCeiling"], 5)
        self.assertEqual(progress["limitingFactor"], "none")
        self.assertEqual(progress["status"], "ready")

    async def test_a_rejection_limited_run_is_not_blamed_on_the_inventory(self) -> None:
        """Plenty of facts, plenty of slots, and nothing survives validation."""
        rejecting = _starter_ollama_side_effect(
            generate_side_effect=self._passing_generation(),
            validation_payload=_rejecting_validation_payload(),
        )
        result = await self._run(self._facts(20), target=5, generate=rejecting)
        progress = result["progress"]

        self.assertGreaterEqual(progress["allocatedSlots"], 5)
        self.assertLess(progress["finalCount"], 5)
        self.assertEqual(progress["limitingFactor"], "validation_rejections")

    def _passing_generation(self):
        state = {"n": 0}

        async def _fake_generate(prompt: str, **kwargs: object) -> dict[str, str]:
            state["n"] += 1
            n = state["n"]
            return {
                "answer": json.dumps(
                    {
                        "seeds": [
                            {
                                "question": f"Distinct question number {n}?",
                                "answer": f"Answer {n} with enough detail to pass.",
                                "category": f"Category {n % 7}",
                                "questionType": "direct",
                            }
                        ]
                    }
                ),
                "model": SEED_GENERATION_MODEL,
            }

        return _fake_generate

    async def test_lost_extraction_batches_reach_the_run_report(self) -> None:
        """The number that would have named this regression on day one."""
        result = await self._run(
            self._facts(8),
            target=50,
            extraction={
                "batchCount": 23,
                "batchesOk": 4,
                "batchesEmpty": 0,
                "batchesParseFailed": 19,
                "batchesCallFailed": 0,
            },
        )
        progress = result["progress"]

        self.assertEqual(progress["factInventoryBatches"], 23)
        self.assertEqual(progress["factInventoryBatchFailures"], 19)

    async def test_an_inventory_without_the_new_report_still_runs(self) -> None:
        """Inventories cached before this change carry no extraction block."""
        result = await self._run(self._facts(8), target=50)
        progress = result["progress"]

        self.assertEqual(progress["factInventoryBatches"], 0)
        self.assertEqual(progress["factInventoryBatchFailures"], 0)

    async def test_established_invariants_still_hold(self) -> None:
        result = await self._run(self._facts(20), target=5)
        progress = result["progress"]

        self.assertEqual(len(result["seeds"]), progress["finalCount"])
        self.assertGreaterEqual(progress["allocatedSlots"], progress["finalCount"])
        self.assertEqual(progress["candidatesAccepted"], progress["finalCount"])
        self.assertEqual(
            progress["ollamaCalls"],
            progress["factExtractionCalls"]
            + progress["generationCalls"]
            + progress["validationCalls"],
        )
        self.assertIn(progress["status"], {"ready", "partial", "failed"})


class ScenarioStyleReachabilityTests(unittest.TestCase):
    """The prompt is what decides a seed's style, so test the prompt.

    CSS 350 produced 48 `direct` and 2 `procedure` seeds out of 50 — not
    because the model refused scenarios, but because every generation prompt
    named exactly one allowed questionType and none of them was scenario-like.
    """

    def test_question_types_survive_truncation_at_one_slot(self) -> None:
        from app.seed_generation import _question_types_for_styles

        self.assertEqual(
            _question_types_for_styles(["scenario"], slot_count=1), ["scenario"]
        )
        self.assertEqual(
            _question_types_for_styles(["clarification"], slot_count=1),
            ["clarification"],
        )
        self.assertEqual(
            _question_types_for_styles(["exception"], slot_count=1), ["scenario"]
        )

    def test_a_promoted_fact_prompt_allows_a_scenario_type(self) -> None:
        from app.seed_allocation import suggest_question_styles
        from app.seed_generation import _build_fact_seed_prompt

        fact = _mock_fact(
            "fact-01",
            statement="Late work may be submitted within 24 hours for half credit.",
            kind="late_work",
            complexity=3,
        )
        styles = suggest_question_styles(fact, 1, prefer_scenario_like=True)
        prompt = _build_fact_seed_prompt(
            fact=fact,
            chunk_texts=["Late work may be submitted within 24 hours."],
            count=1,
            suggested_styles=styles,
        )

        self.assertIn("Allowed questionType values: scenario", prompt)

    def test_an_ineligible_fact_prompt_stays_direct(self) -> None:
        from app.seed_allocation import suggest_question_styles
        from app.seed_generation import _build_fact_seed_prompt

        fact = _mock_fact(
            "fact-02",
            statement="Office hours are Tuesdays at 2pm.",
            kind="office_hours",
            complexity=1,
        )
        styles = suggest_question_styles(fact, 1, prefer_scenario_like=True)
        prompt = _build_fact_seed_prompt(
            fact=fact,
            chunk_texts=["Office hours are Tuesdays at 2pm."],
            count=1,
            suggested_styles=styles,
        )

        self.assertIn("Allowed questionType values: direct", prompt)
        self.assertNotIn("scenario", prompt.split("Required JSON shape")[0].replace(
            "direct|scenario|clarification|procedure|comparison", ""
        ))


class ScenarioBalanceRunTests(unittest.IsolatedAsyncioTestCase):
    """End-to-end: a policy-heavy course must not report zero scenarios.

    The model stub answers with whatever questionType the prompt allows, so
    these tests measure what the run *asks for*. A stub that hardcoded
    `scenario` would pass no matter how broken the prompt was.
    """

    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        root = Path(self._temp_dir.name)
        self.storage = LocalCourseArtifactStorage(
            root_dir=root / "course_data",
            index_dir=root / "indexes",
        )
        self.course_id = "css-350-winter-2026-scen"
        chunks = [
            _chunk(
                f"chunk-{index:03d}",
                "CSS 350 Syllabus",
                f"Late policy {index}: work submitted after the deadline for "
                f"unit {index} may receive half credit unless an extension "
                "was arranged in advance. " * 3,
                order=index,
            )
            for index in range(1, 41)
        ]
        self.storage.save_index(self.course_id, _index(self.course_id, chunks))
        self._embed_patch = patch(
            "app.seed_generation.embed_ollama_texts",
            new=_orthogonal_embed,
        )
        self._embed_patch.start()

    def tearDown(self) -> None:
        self._embed_patch.stop()
        self._temp_dir.cleanup()

    def _policy_facts(self, count: int) -> list[dict]:
        """Facts that already earn a scenario style: policy kinds, complexity 2.

        Spread across the policy family because allocation caps slots per kind,
        and a single-kind fixture would be testing that cap instead.
        """
        kinds = ["late_work", "attendance", "policy", "accommodation"]
        return [
            _mock_fact(
                f"fact-{index:02d}",
                statement=(
                    f"Late policy {index}: work submitted after the deadline "
                    f"for unit {index} may receive half credit unless an "
                    "extension was arranged in advance."
                ),
                source_chunk_ids=[f"chunk-{index:03d}"],
                kind=kinds[(index - 1) % len(kinds)],
                complexity=2,
            )
            for index in range(1, count + 1)
        ]

    def _prompt_obedient_generation(self) -> AsyncMock:
        """Return the first questionType the prompt actually permits."""
        state = {"n": 0}

        async def _fake_generate(prompt: str, **kwargs: object) -> dict[str, str]:
            state["n"] += 1
            n = state["n"]
            match = re.search(r"Allowed questionType values: ([^\n]+)", prompt)
            allowed = match.group(1).split(",")[0].strip() if match else "direct"
            return {
                "answer": json.dumps(
                    {
                        "seeds": [
                            {
                                "question": f"Distinct question number {n}?",
                                "answer": (
                                    f"Answer {n} with enough detail to pass checks."
                                ),
                                "category": f"Category {n % 7}",
                                "questionType": allowed,
                            }
                        ]
                    }
                ),
                "model": SEED_GENERATION_MODEL,
            }

        return _starter_ollama_side_effect(generate_side_effect=_fake_generate)

    async def _run(self, facts: list[dict], **kwargs):
        from app.seed_generation import generate_starter_seeds_for_course

        with patch(
            "app.seed_generation.load_or_build_fact_inventory",
            new=AsyncMock(return_value=_mock_inventory(facts)),
        ):
            with patch(
                "app.seed_generation.generate_starter_ollama_completion",
                new=self._prompt_obedient_generation(),
            ):
                return await generate_starter_seeds_for_course(
                    course_id=self.course_id,
                    storage=self.storage,
                    **kwargs,
                )

    async def test_a_policy_heavy_run_reports_scenarios(self) -> None:
        """The regression, in one assertion: this used to be 0."""
        result = await self._run(self._policy_facts(30), target_count=30)
        progress = result["progress"]

        self.assertGreater(progress["scenarioOrClarificationActual"], 0)

    async def test_scenario_seeds_carry_the_style_in_question_type(self) -> None:
        result = await self._run(self._policy_facts(30), target_count=30)

        scenario_seeds = [
            seed
            for seed in result["seeds"]
            if seed["questionType"] in {"scenario", "clarification"}
        ]
        self.assertTrue(scenario_seeds)
        self.assertEqual(
            len(scenario_seeds),
            result["progress"]["scenarioOrClarificationActual"],
        )

    async def test_ineligible_facts_are_never_forced_into_scenarios(self) -> None:
        """A course of simple lookups reports zero, and that is correct."""
        facts = [
            _mock_fact(
                f"fact-{index:02d}",
                statement=f"Office hours for section {index} are Tuesdays at 2pm.",
                source_chunk_ids=[f"chunk-{index:03d}"],
                kind="office_hours",
                complexity=1,
            )
            for index in range(1, 31)
        ]
        result = await self._run(facts, target_count=30)

        self.assertEqual(result["progress"]["scenarioOrClarificationActual"], 0)
        for seed in result["seeds"]:
            self.assertEqual(seed["questionType"], "direct")

    async def test_status_stays_count_based(self) -> None:
        """Missing the scenario minimum must not reclassify a full run."""
        facts = [
            _mock_fact(
                f"fact-{index:02d}",
                statement=f"Office hours for section {index} are Tuesdays at 2pm.",
                source_chunk_ids=[f"chunk-{index:03d}"],
                kind="office_hours",
                complexity=1,
            )
            for index in range(1, 11)
        ]
        result = await self._run(facts, target_count=10)
        progress = result["progress"]

        self.assertEqual(progress["finalCount"], 10)
        self.assertEqual(progress["scenarioOrClarificationActual"], 0)
        self.assertGreater(progress["scenarioOrClarificationMinimum"], 0)
        self.assertEqual(progress["status"], "ready")


class ScenarioEligibilityUrgencyTests(unittest.IsolatedAsyncioTestCase):
    """The instrumented failure: eligible facts spent before urgency arrived.

    Scenario-capable facts are policy-shaped and score highly, so allocation
    puts them first. A run that waits for the deficit to rival *total* slots
    starts preferring scenarios only in the final stretch, by which point every
    remaining fact is an office-hours time or a contact address and preference
    has nothing eligible left to apply to.

    The fixture reproduces exactly that ordering: six scenario-capable facts at
    the front, twenty-four lookups behind them.
    """

    ELIGIBLE_COUNT = 4
    LOOKUP_COUNT = 40
    TARGET = 40

    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        root = Path(self._temp_dir.name)
        self.storage = LocalCourseArtifactStorage(
            root_dir=root / "course_data",
            index_dir=root / "indexes",
        )
        self.course_id = "css-350-winter-2026-urgn"
        total = self.ELIGIBLE_COUNT + self.LOOKUP_COUNT
        chunks = [
            _chunk(
                f"chunk-{index:03d}",
                "CSS 350 Syllabus",
                f"Item {index}: this section carries enough detail about unit "
                f"{index} to serve as a usable evidence chunk for grading. " * 3,
                order=index,
            )
            for index in range(1, total + 1)
        ]
        self.storage.save_index(self.course_id, _index(self.course_id, chunks))
        self._embed_patch = patch(
            "app.seed_generation.embed_ollama_texts",
            new=_orthogonal_embed,
        )
        self._embed_patch.start()

    def tearDown(self) -> None:
        self._embed_patch.stop()
        self._temp_dir.cleanup()

    def _front_loaded_facts(self) -> list[dict]:
        """Scenario-capable facts that rank above a tail of simple lookups."""
        policy_kinds = ["late_work", "attendance", "policy", "accommodation"]
        lookup_kinds = [
            "office_hours",
            "contact",
            "deadline",
            "grading",
            "requirement",
            "submission",
            "communication",
            "other",
        ]
        facts = [
            _mock_fact(
                f"fact-{index:02d}",
                statement=(
                    f"Policy {index}: work submitted after the deadline for "
                    f"unit {index} may receive half credit unless an extension "
                    "was arranged in advance."
                ),
                source_chunk_ids=[f"chunk-{index:03d}"],
                kind=policy_kinds[(index - 1) % len(policy_kinds)],
                complexity=2,
                importance_score=0.95,
                ask=0.95,
                usefulness=0.95,
            )
            for index in range(1, self.ELIGIBLE_COUNT + 1)
        ]
        facts += [
            _mock_fact(
                f"fact-{index:02d}",
                statement=(
                    f"Office hours for section {index} are held Tuesdays at "
                    "2pm in the advising suite."
                ),
                source_chunk_ids=[f"chunk-{index:03d}"],
                kind=lookup_kinds[(index - 1) % len(lookup_kinds)],
                complexity=1,
                importance_score=0.6,
                ask=0.55,
                usefulness=0.6,
            )
            for index in range(
                self.ELIGIBLE_COUNT + 1, self.ELIGIBLE_COUNT + self.LOOKUP_COUNT + 1
            )
        ]
        return facts

    def _prompt_obedient_generation(self) -> AsyncMock:
        state = {"n": 0}

        async def _fake_generate(prompt: str, **kwargs: object) -> dict[str, str]:
            state["n"] += 1
            n = state["n"]
            match = re.search(r"Allowed questionType values: ([^\n]+)", prompt)
            allowed = match.group(1).split(",")[0].strip() if match else "direct"
            return {
                "answer": json.dumps(
                    {
                        "seeds": [
                            {
                                "question": f"Distinct question number {n}?",
                                "answer": f"Answer {n} with enough detail to pass.",
                                "category": f"Category {n % 7}",
                                "questionType": allowed,
                            }
                        ]
                    }
                ),
                "model": SEED_GENERATION_MODEL,
            }

        return _starter_ollama_side_effect(generate_side_effect=_fake_generate)

    async def _run(self, target_count: int):
        from app.seed_generation import generate_starter_seeds_for_course

        with patch(
            "app.seed_generation.load_or_build_fact_inventory",
            new=AsyncMock(return_value=_mock_inventory(self._front_loaded_facts())),
        ):
            with patch(
                "app.seed_generation.generate_starter_ollama_completion",
                new=self._prompt_obedient_generation(),
            ):
                return await generate_starter_seeds_for_course(
                    course_id=self.course_id,
                    target_count=target_count,
                    storage=self.storage,
                )

    def test_the_fixture_really_is_front_loaded(self) -> None:
        """Guard the premise: eligible facts must rank ahead of the lookups."""
        from app.seed_allocation import allocate_slots, fact_supports_scenario_like_style

        allocation = allocate_slots(
            facts=self._front_loaded_facts(), target_count=self.TARGET
        )
        facts_by_id = {
            fact["factId"]: fact for fact in self._front_loaded_facts()
        }
        eligibility = [
            fact_supports_scenario_like_style(
                facts_by_id[entry["factId"]], max(1, entry["slotCount"])
            )
            for entry in allocation["allocations"]
            if entry["slotCount"] > 0
        ]

        self.assertIn(True, eligibility)
        self.assertIn(False, eligibility)
        # Every eligible opportunity sits ahead of every ineligible one.
        last_eligible = max(i for i, ok in enumerate(eligibility) if ok)
        first_ineligible = min(i for i, ok in enumerate(eligibility) if not ok)
        self.assertLess(last_eligible, first_ineligible)

    async def test_preference_engages_while_eligible_slots_remain(self) -> None:
        """The regression. Under slot-count urgency this was 0."""
        result = await self._run(target_count=self.TARGET)
        progress = result["progress"]

        self.assertGreater(progress["scenarioOrClarificationActual"], 0)

    async def test_old_urgency_would_have_waited_too_long(self) -> None:
        """Pin the condition that made the old rule silent.

        The deficit is smaller than the slots remaining, so urgency measured
        against slot count would not fire until the run's final stretch — long
        after the six eligible facts at the front were spent.
        """
        result = await self._run(target_count=self.TARGET)
        progress = result["progress"]

        self.assertLess(
            progress["scenarioOrClarificationMinimum"],
            progress["achievableCeiling"],
        )
        # Eligible capacity is scarcer than the deficit, which is precisely what
        # slot-count urgency could not see.
        self.assertLessEqual(
            progress["scenarioOrClarificationAchievable"],
            progress["scenarioOrClarificationMinimum"],
        )
        self.assertGreater(progress["scenarioOrClarificationActual"], 0)

    async def test_scenarios_come_only_from_eligible_facts(self) -> None:
        result = await self._run(target_count=self.TARGET)

        scenario_seeds = [
            seed
            for seed in result["seeds"]
            if seed["questionType"] in {"scenario", "clarification"}
        ]
        self.assertTrue(scenario_seeds)
        # A fact can be drawn more than once (primary plus backfill), so the
        # count is not bounded by the fact count — but the source is.
        for seed in scenario_seeds:
            self.assertIn(seed["factId"], {
                f"fact-{index:02d}" for index in range(1, self.ELIGIBLE_COUNT + 1)
            })

    async def test_achievable_reports_a_structurally_unreachable_minimum(self) -> None:
        """The advisory field: four capable facts cannot meet a minimum of 12."""
        result = await self._run(target_count=self.TARGET)
        progress = result["progress"]

        self.assertGreater(progress["scenarioOrClarificationAchievable"], 0)
        self.assertGreater(
            progress["scenarioOrClarificationMinimum"],
            progress["scenarioOrClarificationAchievable"],
        )
        # Advisory only.
        self.assertIn(progress["status"], {"ready", "partial"})

    async def test_status_is_unaffected_by_the_shortfall(self) -> None:
        result = await self._run(target_count=self.TARGET)
        progress = result["progress"]

        self.assertEqual(
            progress["status"],
            resolve_starter_run_status(
                target_count=self.TARGET,
                final_count=progress["finalCount"],
                saved_count=0,
                save=False,
            ),
        )
