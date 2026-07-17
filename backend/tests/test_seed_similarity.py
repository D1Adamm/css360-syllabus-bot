"""Tests for semantic starter-seed deduplication."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from app.seed_similarity import (
    SEMANTIC_QUESTION_DUPLICATE_THRESHOLD,
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
