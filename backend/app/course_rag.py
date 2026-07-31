"""Course-scoped RAG retrieval over backend/data/indexes/{courseId}.json."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from app.course_id import assert_valid_course_id
from app.ollama import generate_ollama_completion
from app.rag import (
    OLLAMA_EMBEDDING_MODEL,
    build_rag_prompt,
    compute_cosine_similarity,
    get_embedding,
)
from app.retrieval_diversity import (
    CANDIDATE_POOL_SIZE,
    DEFAULT_TOP_K,
    MAX_CHUNK_CONTEXT_CHARS,
    MAX_TOTAL_CONTEXT_CHARS,
    apply_context_budget,
    select_diverse_course_chunks,
)
from app.storage import CourseArtifactStorage, get_course_artifact_storage


def _validate_course_id(course_id: str) -> str:
    try:
        return assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _normalize_course_chunk(chunk: dict[str, Any]) -> dict[str, Any] | None:
    chunk_id = chunk.get("chunkId") or chunk.get("id")
    section_title = chunk.get("sectionTitle") or chunk.get("section_title")
    text = chunk.get("text")
    embedding = chunk.get("embedding")

    if not isinstance(chunk_id, str) or not chunk_id.strip():
        return None
    if not isinstance(section_title, str) or not section_title.strip():
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    if not isinstance(embedding, list) or not embedding:
        return None
    if not all(isinstance(value, (int, float)) for value in embedding):
        return None

    return {
        "chunk_id": chunk_id,
        "section": section_title,
        "text": text,
        "embedding": [float(value) for value in embedding],
    }


async def retrieve_course_syllabus_chunks(
    course_id: str,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    storage: CourseArtifactStorage | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Load one course index and retrieve a diverse top-K chunk set."""
    safe_course_id = _validate_course_id(course_id)
    artifact_storage = storage or get_course_artifact_storage()
    index_data = artifact_storage.load_index(safe_course_id)

    if index_data is None:
        raise HTTPException(
            status_code=404,
            detail=f'No syllabus index found for course "{safe_course_id}".',
        )

    raw_chunks = index_data.get("chunks", [])
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise HTTPException(
            status_code=404,
            detail=f'No syllabus chunks found for course "{safe_course_id}".',
        )

    question_embedding = await get_embedding(question)
    scored_chunks: list[dict[str, Any]] = []

    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            continue
        normalized = _normalize_course_chunk(raw_chunk)
        if normalized is None:
            continue

        score = compute_cosine_similarity(question_embedding, normalized["embedding"])
        scored_chunks.append(
            {
                "chunk_id": normalized["chunk_id"],
                "section": normalized["section"],
                "text": normalized["text"],
                "score": score,
            }
        )

    if not scored_chunks:
        raise HTTPException(
            status_code=404,
            detail=f'No usable embedded chunks found for course "{safe_course_id}".',
        )

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)
    selected = select_diverse_course_chunks(
        scored_chunks,
        top_k=max(1, top_k),
        candidate_pool_size=CANDIDATE_POOL_SIZE,
    )
    # Bound the context before prompt building so CPU-only generation stays
    # inside OLLAMA_TIMEOUT_SECONDS. Sources are derived from these same chunks.
    selected = apply_context_budget(
        selected,
        per_chunk_limit=MAX_CHUNK_CONTEXT_CHARS,
        total_limit=MAX_TOTAL_CONTEXT_CHARS,
    )
    embedding_model = index_data.get("embeddingModel") or OLLAMA_EMBEDDING_MODEL
    return str(embedding_model), selected


async def generate_course_rag_answer(
    course_id: str,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    storage: CourseArtifactStorage | None = None,
) -> dict[str, Any]:
    safe_course_id = _validate_course_id(course_id)
    _, retrieved_chunks = await retrieve_course_syllabus_chunks(
        course_id=safe_course_id,
        question=question,
        top_k=top_k,
        storage=storage,
    )

    prompt = build_rag_prompt(question, retrieved_chunks)
    generation = await generate_ollama_completion(prompt)

    sources = [
        {
            "chunkId": chunk["chunk_id"],
            "sectionTitle": chunk["section"],
            "text": chunk["text"],
            "score": chunk["score"],
        }
        for chunk in retrieved_chunks
    ]

    return {
        "courseId": safe_course_id,
        "answer": generation["answer"],
        "model": generation["model"],
        "sources": sources,
        "retrievedChunks": [
            {
                "chunkId": chunk["chunk_id"],
                "section": chunk["section"],
                "text": chunk["text"],
                "score": chunk["score"],
            }
            for chunk in retrieved_chunks
        ],
        "responseType": "rag",
    }
