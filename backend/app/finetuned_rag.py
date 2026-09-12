"""Fine-Tuned + RAG: the shared grounded path, answered by the course's adapter.

Retrieval, the grounded prompt and the generation call are `grounded_rag`;
this module names the target. The prompt builder is re-exported under the name
this path has always used so that callers and tests keep reading as before,
and it is byte-identical to the one the RAG path uses: the two conditions
differ in model weights and in nothing else.
"""

from __future__ import annotations

from typing import Any

from app.grounded_generation import build_grounded_prompt
from app.grounded_rag import fine_tuned_target, generate_grounded_answer
from app.retrieval_diversity import DEFAULT_TOP_K
from app.storage import CourseArtifactStorage


def build_finetuned_rag_prompt(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    facets: list[str] | None = None,
) -> str:
    """The grounded prompt, under the name this path has always used.

    Identical to `rag.build_rag_prompt`: both are
    `grounded_generation.build_grounded_prompt`. The fine-tuned model is
    trained on this text and answers this text.
    """
    return build_grounded_prompt(question, retrieved_chunks, facets)


def _sources_from_chunks(retrieved_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunkId": chunk["chunk_id"],
            "sectionTitle": chunk["section"],
            "text": chunk["text"],
            "score": chunk["score"],
        }
        for chunk in retrieved_chunks
    ]


def _retrieved_payload(retrieved_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunkId": chunk["chunk_id"],
            "section": chunk["section"],
            "text": chunk["text"],
            "score": chunk["score"],
        }
        for chunk in retrieved_chunks
    ]


async def generate_course_finetuned_rag_answer(
    course_id: str,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    storage: CourseArtifactStorage | None = None,
    *,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Fine-Tuned + RAG: the shared grounded path answered by the course adapter.

    `model_version` names the version to answer from instead of resolving the
    course's own. Only the administrator-only model-testing route passes it;
    the classroom route never does.
    """
    return await generate_grounded_answer(
        course_id,
        question,
        top_k=top_k,
        storage=storage,
        target=fine_tuned_target(model_version),
    )
