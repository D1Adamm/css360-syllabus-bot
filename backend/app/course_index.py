"""Build and persist per-course RAG indexes with Ollama embeddings."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from app.rag import OLLAMA_EMBEDDING_MODEL, get_embedding
from app.storage import CourseArtifactStorage
from app.syllabus_chunking import SyllabusChunk, chunk_syllabus_text
from app.syllabus_upload import SyllabusUploadError


def _embedding_input(chunk: SyllabusChunk) -> str:
    return f"Section: {chunk.section_title}\n\n{chunk.text}"


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

        indexed.append(
            {
                "chunkId": chunk.chunk_id,
                "sectionTitle": chunk.section_title,
                "text": chunk.text,
                "order": chunk.order,
                "embedding": embedding,
            }
        )
    return indexed


async def build_course_rag_index(
    *,
    course_id: str,
    source_file: str,
    syllabus_text: str,
    storage: CourseArtifactStorage,
) -> dict[str, Any]:
    chunks = chunk_syllabus_text(syllabus_text)
    if not chunks:
        raise SyllabusUploadError(
            "Syllabus chunking produced no usable chunks.",
            status_code=400,
        )

    try:
        indexed_chunks = await embed_chunks(chunks)
        index_data: dict[str, Any] = {
            "courseId": course_id,
            "sourceFile": source_file,
            "embeddingModel": OLLAMA_EMBEDDING_MODEL,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "chunkCount": len(indexed_chunks),
            "chunks": indexed_chunks,
        }
        storage.save_index(course_id, index_data)
        return index_data
    except Exception:
        storage.remove_index(course_id)
        raise
