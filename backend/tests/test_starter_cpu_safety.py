"""CPU-safety guards for starter seed generation (backfill cap + num_predict)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from app.ollama import (
    DEFAULT_STARTER_GENERATION_NUM_PREDICT,
    DEFAULT_STARTER_INVENTORY_NUM_PREDICT,
    DEFAULT_STARTER_VALIDATION_NUM_PREDICT,
    generate_base_model_response,
    generate_ollama_completion,
    generate_starter_ollama_completion,
    get_starter_generation_num_predict,
    get_starter_inventory_num_predict,
    get_starter_validation_num_predict,
)
from app.seed_generation import starter_backfill_limit
from app.storage import LocalCourseArtifactStorage


def _timeout_error() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail="Ollama request timed out. Ensure Ollama is running and responsive.",
    )


def _chunk(chunk_id: str, text: str, *, order: int) -> dict[str, object]:
    return {
        "chunkId": chunk_id,
        "sectionTitle": f"Section {chunk_id}",
        "text": text,
        "order": order,
        "embedding": [1.0, 0.0, 0.0],
    }


def _fact(
    fact_id: str,
    *,
    kind: str,
    chunk: str,
    scope: str = "course_wide",
) -> dict[str, object]:
    statement = f"{kind.replace('_', ' ').title()} policy statement for students."
    return {
        "factId": fact_id,
        "statement": statement,
        "importance": "high",
        "importanceScore": 0.9,
        "studentAskLikelihood": 0.9,
        "complexity": 1,
        "usefulnessScore": 0.86,
        "sourceChunkIds": [chunk],
        "evidenceQuote": statement,
        "kind": kind,
        "scope": scope,
        "seriesKey": None,
        "assignmentGroup": None,
        "seriesOrdinal": None,
    }


class BackfillLimitHelperTests(unittest.TestCase):
    def test_target_count_three_allows_six_backfill(self) -> None:
        self.assertEqual(starter_backfill_limit(3), 6)

    def test_larger_targets_scale_with_two_x(self) -> None:
        self.assertEqual(starter_backfill_limit(10), 20)


class BackfillOpportunityCapTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        root = Path(self._temp.name)
        self.storage = LocalCourseArtifactStorage(
            root_dir=root / "course_data",
            index_dir=root / "indexes",
        )
        self.course_id = "css-360-backfill-cap"
        kinds = [
            "late_work",
            "attendance",
            "grading",
            "contact",
            "communication",
            "requirement",
            "policy",
            "exam",
            "tools",
            "accommodation",
            "office_hours",
            "team_project",
        ]
        chunks = [
            _chunk(
                f"chunk-{index:03d}",
                f"Section {index} syllabus policy details for students. " * 12,
                order=index,
            )
            for index in range(1, len(kinds) + 1)
        ]
        self.storage.save_index(
            self.course_id,
            {
                "courseId": self.course_id,
                "embeddingModel": "nomic-embed-text",
                "chunkCount": len(chunks),
                "chunks": chunks,
            },
        )
        self.facts = [
            _fact(
                f"fact-{index:02d}",
                kind=kinds[index - 1],
                chunk=f"chunk-{index:03d}",
            )
            for index in range(1, len(kinds) + 1)
        ]

    def tearDown(self) -> None:
        self._temp.cleanup()

    async def test_target_count_three_limits_backfill_to_six(self) -> None:
        from app.seed_generation import generate_starter_seeds_for_course

        inventory = {
            "model": "qwen3:4b",
            "facts": self.facts,
            "factCount": len(self.facts),
            "droppedCount": 0,
            "duplicatesRemoved": 0,
            "fallbackUsed": False,
            "countsByScope": {},
            "countsByKind": {},
            "countsBySeries": {},
            "cached": True,
        }

        # Force every generation opportunity to fail so the run walks the full
        # primary + backfill pool. Cap should keep that at 3 + 6 = 9 calls.
        mock_generate = AsyncMock(side_effect=_timeout_error())
        with (
            patch(
                "app.seed_generation.load_or_build_fact_inventory",
                new=AsyncMock(return_value=inventory),
            ),
            patch(
                "app.seed_generation.generate_starter_ollama_completion",
                new=mock_generate,
            ),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await generate_starter_seeds_for_course(
                    course_id=self.course_id,
                    target_count=3,
                    storage=self.storage,
                )

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(mock_generate.await_count, 9)
        for call in mock_generate.await_args_list:
            self.assertEqual(call.kwargs.get("stage"), "generation")
            self.assertEqual(
                call.kwargs.get("num_predict"),
                DEFAULT_STARTER_GENERATION_NUM_PREDICT,
            )


class StarterNumPredictEnvTests(unittest.TestCase):
    def test_invalid_values_fall_back(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "STARTER_INVENTORY_NUM_PREDICT": "nope",
                "STARTER_GENERATION_NUM_PREDICT": "0",
                "STARTER_VALIDATION_NUM_PREDICT": "-5",
            },
            clear=False,
        ):
            self.assertEqual(
                get_starter_inventory_num_predict(),
                DEFAULT_STARTER_INVENTORY_NUM_PREDICT,
            )
            self.assertEqual(
                get_starter_generation_num_predict(),
                DEFAULT_STARTER_GENERATION_NUM_PREDICT,
            )
            self.assertEqual(
                get_starter_validation_num_predict(),
                DEFAULT_STARTER_VALIDATION_NUM_PREDICT,
            )

    def test_empty_string_falls_back(self) -> None:
        with patch.dict(
            "os.environ",
            {"STARTER_INVENTORY_NUM_PREDICT": ""},
            clear=False,
        ):
            self.assertEqual(
                get_starter_inventory_num_predict(),
                DEFAULT_STARTER_INVENTORY_NUM_PREDICT,
            )

    def test_valid_override(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "STARTER_INVENTORY_NUM_PREDICT": "512",
                "STARTER_GENERATION_NUM_PREDICT": "200",
                "STARTER_VALIDATION_NUM_PREDICT": "128",
            },
            clear=False,
        ):
            self.assertEqual(get_starter_inventory_num_predict(), 512)
            self.assertEqual(get_starter_generation_num_predict(), 200)
            self.assertEqual(get_starter_validation_num_predict(), 128)


class StarterNumPredictPayloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_starter_payload_includes_num_predict(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "qwen3:4b",
            "response": '{"ok": true}',
            "done": True,
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.ollama.httpx.AsyncClient", return_value=mock_client):
            await generate_starter_ollama_completion(
                "short prompt",
                model="qwen3:4b",
                response_format="json",
                think=False,
                num_predict=384,
                stage="generation",
            )

        sent = mock_client.post.await_args.kwargs["json"]
        self.assertEqual(sent["options"], {"num_predict": 384})
        self.assertIs(sent["think"], False)
        self.assertEqual(sent["format"], "json")

    async def test_base_model_payload_omits_options(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "llama3.2:3b",
            "response": "No syllabus was provided.",
            "done": True,
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.ollama.httpx.AsyncClient", return_value=mock_client):
            await generate_base_model_response("When is the midterm?")

        sent = mock_client.post.await_args.kwargs["json"]
        self.assertNotIn("options", sent)
        self.assertNotIn("think", sent)

    async def test_plain_generate_without_num_predict_omits_options(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "llama3.2:3b",
            "response": "hello",
            "done": True,
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.ollama.httpx.AsyncClient", return_value=mock_client):
            await generate_ollama_completion("rag-style prompt")

        sent = mock_client.post.await_args.kwargs["json"]
        self.assertNotIn("options", sent)


class StarterTimeoutRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_retry_occurs_exactly_once_and_forwards_num_predict(
        self,
    ) -> None:
        with (
            patch(
                "app.ollama.get_starter_ollama_retry_delay_seconds",
                return_value=0.0,
            ),
            patch(
                "app.ollama.get_starter_ollama_timeout_seconds",
                return_value=300.0,
            ),
            patch(
                "app.ollama.generate_ollama_completion",
                new=AsyncMock(
                    side_effect=[
                        _timeout_error(),
                        {"answer": '{"ok": true}', "model": "qwen3:4b"},
                    ]
                ),
            ) as mock_generate,
        ):
            result = await generate_starter_ollama_completion(
                "prompt",
                model="qwen3:4b",
                response_format="json",
                think=False,
                num_predict=256,
                stage="validation",
            )

        self.assertEqual(result["answer"], '{"ok": true}')
        self.assertEqual(mock_generate.await_count, 2)
        for call in mock_generate.await_args_list:
            self.assertEqual(call.kwargs["num_predict"], 256)
            self.assertEqual(call.kwargs["timeout"], 300.0)
            self.assertIs(call.kwargs["think"], False)


class LiveCallSiteNumPredictTests(unittest.IsolatedAsyncioTestCase):
    async def test_inventory_call_passes_inventory_num_predict(self) -> None:
        from app.syllabus_facts import _extract_facts_for_batch

        mock_completion = AsyncMock(
            return_value={"answer": json.dumps({"facts": []}), "model": "qwen3:4b"}
        )
        batch = [
            {
                "sectionTitle": "Late Policy",
                "chunks": [
                    {
                        "chunkId": "chunk-001",
                        "text": "Late work may be submitted within 24 hours for half credit.",
                    }
                ],
            }
        ]
        with patch(
            "app.syllabus_facts.get_starter_inventory_num_predict",
            return_value=1024,
        ):
            await _extract_facts_for_batch(
                batch=batch,
                chunk_lookup={"chunk-001": batch[0]["chunks"][0]["text"]},
                completion_fn=mock_completion,
                model="qwen3:4b",
            )

        self.assertEqual(mock_completion.await_args.kwargs["num_predict"], 1024)
        self.assertEqual(mock_completion.await_args.kwargs["stage"], "inventory")
        self.assertIs(mock_completion.await_args.kwargs["think"], False)

    async def test_validation_call_passes_validation_num_predict(self) -> None:
        from app.seed_generation import _validate_candidate

        mock_completion = AsyncMock(
            return_value={
                "answer": json.dumps(
                    {
                        "grounded": 0.9,
                        "correct": 0.9,
                        "clear": 0.85,
                        "useful": 0.85,
                        "naturalStudentWording": 0.85,
                        "categoryCorrect": 0.8,
                        "notTrivialOrTemporary": 0.8,
                        "unsupportedClaims": [],
                        "reason": "ok",
                    }
                ),
                "model": "qwen3:4b",
            }
        )
        with patch(
            "app.seed_generation.get_starter_validation_num_predict",
            return_value=256,
        ):
            await _validate_candidate(
                question="When may late work be submitted?",
                answer="Within 24 hours for half credit.",
                topic_name="late work",
                question_type="direct",
                chunk_text="Late work may be submitted within 24 hours for half credit.",
                completion_fn=mock_completion,
            )

        self.assertEqual(mock_completion.await_args.kwargs["num_predict"], 256)
        self.assertEqual(mock_completion.await_args.kwargs["stage"], "validation")
        self.assertIs(mock_completion.await_args.kwargs["think"], False)


if __name__ == "__main__":
    unittest.main()
