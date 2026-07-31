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
    MAX_FINAL_TOP_K,
    apply_context_budget,
    resolve_final_top_k,
    resolve_total_context_limit,
    select_coverage_aware_chunks,
    select_diverse_course_chunks,
)
from app.retrieval_facets import (
    FACET_CANDIDATE_POOL,
    assign_missing_facet_matches,
    extract_question_facets,
    merge_scored_candidates,
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

    normalized: dict[str, Any] = {
        "chunk_id": chunk_id,
        "section": section_title,
        "text": text,
        "embedding": [float(value) for value in embedding],
    }
    document_title = chunk.get("documentTitle") or chunk.get("document_title")
    if isinstance(document_title, str) and document_title.strip():
        normalized["document_title"] = document_title.strip()
    heading_path = chunk.get("headingPath") or chunk.get("heading_path")
    if isinstance(heading_path, list):
        normalized["heading_path"] = [str(part) for part in heading_path if str(part).strip()]
    return normalized


def _score_chunks_against_embedding(
    normalized_chunks: list[dict[str, Any]],
    query_embedding: list[float],
    *,
    matched_facet: str | None = None,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for chunk in normalized_chunks:
        score = compute_cosine_similarity(query_embedding, chunk["embedding"])
        entry: dict[str, Any] = {
            "chunk_id": chunk["chunk_id"],
            "section": chunk["section"],
            "text": chunk["text"],
            "score": score,
            "matched_facets": [matched_facet] if matched_facet else [],
        }
        scored.append(entry)
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


async def retrieve_course_syllabus_chunks(
    course_id: str,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    storage: CourseArtifactStorage | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Load one course index and retrieve a diverse, coverage-aware top-K set."""
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

    normalized_chunks: list[dict[str, Any]] = []
    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            continue
        normalized = _normalize_course_chunk(raw_chunk)
        if normalized is not None:
            normalized_chunks.append(normalized)

    if not normalized_chunks:
        raise HTTPException(
            status_code=404,
            detail=f'No usable embedded chunks found for course "{safe_course_id}".',
        )

    facets = extract_question_facets(question)
    multi_facet = len(facets) >= 2
    final_top_k = resolve_final_top_k(top_k, multi_facet=multi_facet)

    # Always retrieve with the full question embedding first.
    question_embedding = await get_embedding(question)
    full_ranked = _score_chunks_against_embedding(normalized_chunks, question_embedding)
    full_pool = full_ranked[:CANDIDATE_POOL_SIZE]

    if multi_facet:
        facet_pools: list[list[dict[str, Any]]] = [full_pool]
        for facet in facets:
            facet_embedding = await get_embedding(facet)
            facet_ranked = _score_chunks_against_embedding(
                normalized_chunks,
                facet_embedding,
                matched_facet=facet,
            )
            facet_pools.append(facet_ranked[:FACET_CANDIDATE_POOL])

        merged = merge_scored_candidates(*facet_pools)
        assign_missing_facet_matches(merged, facets)
        selected = select_coverage_aware_chunks(
            merged,
            facets=facets,
            top_k=final_top_k,
            candidate_pool_size=max(CANDIDATE_POOL_SIZE, len(merged)),
        )
    else:
        selected = select_diverse_course_chunks(
            full_ranked,
            top_k=final_top_k,
            candidate_pool_size=CANDIDATE_POOL_SIZE,
        )

    selected = selected[:MAX_FINAL_TOP_K]
    # Bound the context before prompt building so CPU-only generation stays
    # inside OLLAMA_TIMEOUT_SECONDS. Sources are derived from these same chunks.
    selected = apply_context_budget(
        selected,
        per_chunk_limit=MAX_CHUNK_CONTEXT_CHARS,
        total_limit=resolve_total_context_limit(multi_facet=multi_facet),
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
