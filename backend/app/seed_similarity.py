"""Semantic similarity helpers for seed deduplication."""

from __future__ import annotations

import math
from typing import Any

from app.ollama import DEFAULT_EMBED_MODEL, embed_ollama_texts
from app.seed_dedupe import normalize_question_for_dedupe

SEMANTIC_QUESTION_DUPLICATE_THRESHOLD = 0.88


class AcceptedEmbeddingCache:
    """Run-scoped cache of embeddings for accepted starter questions.

    Embeddings stay parallel to the caller's accepted_questions list. After a
    candidate is checked, ``last_candidate_embedding`` holds that candidate's
    vector so the caller can append it on accept without re-embedding.
    """

    def __init__(self) -> None:
        self.embeddings: list[list[float]] = []
        self.last_candidate_embedding: list[float] | None = None

    def remember_last_candidate(self) -> None:
        """Append the most recently checked candidate embedding, if present."""
        if self.last_candidate_embedding is None:
            return
        self.embeddings.append(self.last_candidate_embedding)
        self.last_candidate_embedding = None


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


async def embed_questions(
    questions: list[str],
    *,
    model: str = DEFAULT_EMBED_MODEL,
    embed_fn=embed_ollama_texts,
) -> dict[str, Any]:
    return await embed_fn(questions, model=model)


def _best_duplicate_match(
    *,
    candidate_embedding: list[float],
    accepted_questions: list[str],
    accepted_embeddings: list[list[float]],
    threshold: float,
) -> dict[str, Any] | None:
    best_match: dict[str, Any] | None = None
    for index, question in enumerate(accepted_questions):
        if index >= len(accepted_embeddings):
            break
        embedding = accepted_embeddings[index]
        if not isinstance(embedding, list):
            continue
        score = cosine_similarity(candidate_embedding, embedding)
        if score >= threshold and (
            best_match is None or score > float(best_match["similarity"])
        ):
            best_match = {
                "question": question,
                "similarity": score,
            }
    return best_match


async def _ensure_accepted_embeddings_cached(
    *,
    accepted_questions: list[str],
    cache: AcceptedEmbeddingCache,
    model: str,
    embed_fn,
) -> bool:
    """Backfill missing accepted embeddings. Returns False on malformed embed data."""
    if len(cache.embeddings) > len(accepted_questions):
        cache.embeddings = cache.embeddings[: len(accepted_questions)]

    missing_start = len(cache.embeddings)
    if missing_start >= len(accepted_questions):
        return True

    missing_questions = accepted_questions[missing_start:]
    result = await embed_questions(
        missing_questions,
        model=model,
        embed_fn=embed_fn,
    )
    raw_embeddings = result.get("embeddings", [])
    if not isinstance(raw_embeddings, list) or len(raw_embeddings) != len(missing_questions):
        return False

    for embedding in raw_embeddings:
        if not isinstance(embedding, list):
            return False
        cache.embeddings.append(embedding)
    return True


async def find_semantic_duplicate_question(
    *,
    candidate_question: str,
    accepted_questions: list[str],
    threshold: float = SEMANTIC_QUESTION_DUPLICATE_THRESHOLD,
    model: str = DEFAULT_EMBED_MODEL,
    embed_fn=embed_ollama_texts,
    cache: AcceptedEmbeddingCache | None = None,
) -> dict[str, Any] | None:
    cleaned_candidate = normalize_question_for_dedupe(candidate_question)
    cleaned_accepted = [
        normalize_question_for_dedupe(question)
        for question in accepted_questions
        if normalize_question_for_dedupe(question)
    ]
    if not cleaned_candidate or not cleaned_accepted:
        if cache is not None:
            cache.last_candidate_embedding = None
        return None

    if cache is not None:
        cache.last_candidate_embedding = None
        if not await _ensure_accepted_embeddings_cached(
            accepted_questions=accepted_questions,
            cache=cache,
            model=model,
            embed_fn=embed_fn,
        ):
            return None

        result = await embed_questions(
            [candidate_question],
            model=model,
            embed_fn=embed_fn,
        )
        raw_embeddings = result.get("embeddings", [])
        if not isinstance(raw_embeddings, list) or len(raw_embeddings) != 1:
            return None

        candidate_embedding = raw_embeddings[0]
        if not isinstance(candidate_embedding, list):
            return None

        cache.last_candidate_embedding = candidate_embedding
        return _best_duplicate_match(
            candidate_embedding=candidate_embedding,
            accepted_questions=accepted_questions,
            accepted_embeddings=cache.embeddings,
            threshold=threshold,
        )

    # Legacy path (no cache): embed candidate + all accepted questions together.
    result = await embed_questions(
        [candidate_question, *accepted_questions],
        model=model,
        embed_fn=embed_fn,
    )
    raw_embeddings = result.get("embeddings", [])
    if not isinstance(raw_embeddings, list) or len(raw_embeddings) != len(accepted_questions) + 1:
        return None

    candidate_embedding = raw_embeddings[0]
    if not isinstance(candidate_embedding, list):
        return None

    best_match: dict[str, Any] | None = None
    for index, question in enumerate(accepted_questions):
        embedding = raw_embeddings[index + 1]
        if not isinstance(embedding, list):
            continue
        score = cosine_similarity(candidate_embedding, embedding)
        if score >= threshold and (
            best_match is None or score > float(best_match["similarity"])
        ):
            best_match = {
                "question": question,
                "similarity": score,
            }
    return best_match
