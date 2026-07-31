"""Build and persist per-course RAG indexes with Ollama embeddings."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from app.rag import OLLAMA_EMBEDDING_MODEL, get_embedding
from app.storage import CourseArtifactStorage
from app.syllabus_chunking import (
    INDEX_VERSION,
    SyllabusChunk,
    chunk_syllabus_text,
    embedding_input_for_chunk,
    summarize_chunking,
    validate_chunks,
)
from app.syllabus_upload import SyllabusUploadError


def _embedding_input(chunk: SyllabusChunk) -> str:
    return embedding_input_for_chunk(
        section_title=chunk.section_title,
        text=chunk.text,
    )


async def embed_chunks(chunks: list[SyllabusChunk]) -> list[dict[str, Any]]:
    indexed: list[dict[str, Any]] = []
    for chunk in chunks:
        try:
            embedding = await get_embedding(_embedding_input(chunk))
        except HTTPException as exc:
            raise SyllabusUploadError(
                str(exc.detail),
                status_code=exc.status_code,
            ) from exc

        payload = chunk.to_dict()
        payload["embedding"] = embedding
        indexed.append(payload)
    return indexed


async def build_course_rag_index(
    *,
    course_id: str,
    source_file: str,
    syllabus_text: str,
    storage: CourseArtifactStorage,
    strict_validation: bool = False,
) -> dict[str, Any]:
    chunks = chunk_syllabus_text(syllabus_text)
    if not chunks:
        raise SyllabusUploadError(
            "Syllabus chunking produced no usable chunks.",
            status_code=400,
        )

    document_title = chunks[0].document_title if chunks else ""
    warnings = validate_chunks(
        chunks,
        document_title=document_title,
        source_char_count=len(syllabus_text),
        strict=strict_validation,
    )
    summary = summarize_chunking(chunks)

    try:
        indexed_chunks = await embed_chunks(chunks)
        index_data: dict[str, Any] = {
            "indexVersion": INDEX_VERSION,
            "courseId": course_id,
            "sourceFile": source_file,
            "documentTitle": document_title,
            "embeddingModel": OLLAMA_EMBEDDING_MODEL,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "chunkCount": len(indexed_chunks),
            "sectionCount": summary["sectionCount"],
            "chunks": indexed_chunks,
        }
        if warnings:
            index_data["chunkingWarnings"] = warnings
        storage.save_index(course_id, index_data)
        return index_data
    except Exception:
        storage.remove_index(course_id)
        raise
