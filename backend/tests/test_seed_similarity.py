"""Tests for semantic starter-seed deduplication."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from app.seed_similarity import (
    SEMANTIC_QUESTION_DUPLICATE_THRESHOLD,
    AcceptedEmbeddingCache,
    cosine_similarity,
    find_semantic_duplicate_question,
)


class SeedSimilarityTests(unittest.IsolatedAsyncioTestCase):
    def test_cosine_similarity_returns_expected_value(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    async def test_semantic_duplicate_rejection(self) -> None:
        with patch(
            "app.seed_similarity.embed_questions",
            new=AsyncMock(
                return_value={
                    "embeddings": [
                        [1.0, 0.0],
                        [0.99, 0.01],
                        [0.0, 1.0],
                    ],
                    "model": "nomic-embed-text",
                }
            ),
        ):
            duplicate = await find_semantic_duplicate_question(
                candidate_question="Can I turn work in late?",
                accepted_questions=[
                    "Is late work accepted?",
                    "How are final projects graded?",
                ],
            )

        self.assertIsNotNone(duplicate)
        assert duplicate is not None
        self.assertEqual(duplicate["question"], "Is late work accepted?")
        self.assertGreaterEqual(
            float(duplicate["similarity"]),
            SEMANTIC_QUESTION_DUPLICATE_THRESHOLD,
        )

    async def test_non_duplicate_question_is_allowed(self) -> None:
        with patch(
            "app.seed_similarity.embed_questions",
            new=AsyncMock(
                return_value={
                    "embeddings": [
                        [1.0, 0.0],
                        [0.3, 0.7],
                    ],
                    "model": "nomic-embed-text",
                }
            ),
        ):
            duplicate = await find_semantic_duplicate_question(
                candidate_question="What is the late policy?",
                accepted_questions=["How should I cite outside sources?"],
            )

        self.assertIsNone(duplicate)

    async def test_empty_accepted_questions_short_circuits(self) -> None:
        with patch("app.seed_similarity.embed_questions", new=AsyncMock()) as mock_embed:
            duplicate = await find_semantic_duplicate_question(
                candidate_question="What is the late policy?",
                accepted_questions=[],
            )

        self.assertIsNone(duplicate)
        mock_embed.assert_not_awaited()


class AcceptedEmbeddingCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_cache_avoids_reembedding_accepted_questions(self) -> None:
        cache = AcceptedEmbeddingCache()
        embed_calls: list[list[str]] = []

        async def _fake_embed(questions: list[str], **kwargs: object) -> dict[str, object]:
            embed_calls.append(list(questions))
            vectors = {
                "Is late work accepted?": [1.0, 0.0],
                "How are projects graded?": [0.0, 1.0],
                "Can I submit late work?": [0.99, 0.01],
                "Where are office hours?": [-1.0, 0.0],
            }
            return {
                "embeddings": [vectors[question] for question in questions],
                "model": "nomic-embed-text",
            }

        accepted = ["Is late work accepted?"]
        first = await find_semantic_duplicate_question(
            candidate_question="Can I submit late work?",
            accepted_questions=accepted,
            embed_fn=_fake_embed,
            cache=cache,
        )
        self.assertIsNotNone(first)
        self.assertEqual(embed_calls, [["Is late work accepted?"], ["Can I submit late work?"]])

        # Rejected duplicate: do not remember candidate embedding.
        self.assertEqual(len(cache.embeddings), 1)

        second = await find_semantic_duplicate_question(
            candidate_question="Where are office hours?",
            accepted_questions=accepted,
            embed_fn=_fake_embed,
            cache=cache,
        )
        self.assertIsNone(second)
        # Accepted question must not be re-embedded; only the new candidate.
        self.assertEqual(
            embed_calls,
            [
                ["Is late work accepted?"],
                ["Can I submit late work?"],
                ["Where are office hours?"],
            ],
        )

        accepted.append("Where are office hours?")
        cache.remember_last_candidate()
        self.assertEqual(len(cache.embeddings), 2)

        third = await find_semantic_duplicate_question(
            candidate_question="How are projects graded?",
            accepted_questions=accepted,
            embed_fn=_fake_embed,
            cache=cache,
        )
        self.assertIsNone(third)
        self.assertEqual(
            embed_calls[-1],
            ["How are projects graded?"],
        )
        # Still no re-embed of previously accepted questions.
        self.assertEqual(
            embed_calls,
            [
                ["Is late work accepted?"],
                ["Can I submit late work?"],
                ["Where are office hours?"],
                ["How are projects graded?"],
            ],
        )

    async def test_cached_path_rejects_at_same_threshold(self) -> None:
        cache = AcceptedEmbeddingCache()
        cache.embeddings = [[1.0, 0.0]]

        async def _fake_embed(questions: list[str], **kwargs: object) -> dict[str, object]:
            self.assertEqual(questions, ["Can I turn work in late?"])
            return {
                "embeddings": [[0.99, 0.01]],
                "model": "nomic-embed-text",
            }

        duplicate = await find_semantic_duplicate_question(
            candidate_question="Can I turn work in late?",
            accepted_questions=["Is late work accepted?"],
            embed_fn=_fake_embed,
            cache=cache,
        )

        self.assertIsNotNone(duplicate)
        assert duplicate is not None
        self.assertEqual(duplicate["question"], "Is late work accepted?")
        self.assertGreaterEqual(
            float(duplicate["similarity"]),
            SEMANTIC_QUESTION_DUPLICATE_THRESHOLD,
        )

    async def test_cached_path_allows_non_duplicate(self) -> None:
        cache = AcceptedEmbeddingCache()
        cache.embeddings = [[1.0, 0.0]]

        async def _fake_embed(questions: list[str], **kwargs: object) -> dict[str, object]:
            return {
                "embeddings": [[0.0, 1.0]],
                "model": "nomic-embed-text",
            }

        duplicate = await find_semantic_duplicate_question(
            candidate_question="Where is the syllabus?",
            accepted_questions=["Is late work accepted?"],
            embed_fn=_fake_embed,
            cache=cache,
        )
        self.assertIsNone(duplicate)
        self.assertEqual(cache.last_candidate_embedding, [0.0, 1.0])

    async def test_malformed_embeddings_fail_open(self) -> None:
        cache = AcceptedEmbeddingCache()
        with patch(
            "app.seed_similarity.embed_questions",
            new=AsyncMock(return_value={"embeddings": [], "model": "nomic-embed-text"}),
        ):
            duplicate = await find_semantic_duplicate_question(
                candidate_question="What is the late policy?",
                accepted_questions=["Is late work accepted?"],
                cache=cache,
            )
        self.assertIsNone(duplicate)
        self.assertIsNone(cache.last_candidate_embedding)

    async def test_embed_http_exception_propagates_for_caller_fail_open(self) -> None:
        cache = AcceptedEmbeddingCache()

        async def _boom(questions: list[str], **kwargs: object) -> dict[str, object]:
            raise HTTPException(status_code=503, detail="Ollama embeddings unavailable")

        with self.assertRaises(HTTPException):
            await find_semantic_duplicate_question(
                candidate_question="What is the late policy?",
                accepted_questions=["Is late work accepted?"],
                embed_fn=_boom,
                cache=cache,
            )
