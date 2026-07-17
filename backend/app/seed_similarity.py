"""Semantic similarity helpers for seed deduplication."""

from __future__ import annotations

import math
from typing import Any

from app.ollama import DEFAULT_EMBED_MODEL, embed_ollama_texts
from app.seed_dedupe import normalize_question_for_dedupe

SEMANTIC_QUESTION_DUPLICATE_THRESHOLD = 0.88


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


async def find_semantic_duplicate_question(
    *,
    candidate_question: str,
    accepted_questions: list[str],
    threshold: float = SEMANTIC_QUESTION_DUPLICATE_THRESHOLD,
    model: str = DEFAULT_EMBED_MODEL,
    embed_fn=embed_ollama_texts,
) -> dict[str, Any] | None:
    cleaned_candidate = normalize_question_for_dedupe(candidate_question)
    cleaned_accepted = [
        normalize_question_for_dedupe(question)
        for question in accepted_questions
        if normalize_question_for_dedupe(question)
    ]
    if not cleaned_candidate or not cleaned_accepted:
        return None

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
