"""One grounded path for both grounded conditions: retrieve, prompt, generate.

`generate_grounded_answer` is what RAG and Fine-Tuned + RAG are. The two differ
in one argument, the generation target, and in nothing else: the same course
retrieval, the same chunks, the same `build_grounded_prompt`, the same
`/api/chat` call shape with the same decoding options, context window and
output cap. `course_rag.generate_course_rag_answer` and
`finetuned_rag.generate_course_finetuned_rag_answer` are thin wrappers that
name a target, kept so every existing caller and test reads as before.

Base and plain Fine-Tuned are not this path. They take the bare question with
no retrieval and remain the secondary, no-context comparison.

Two halves, so an evaluation can hold the retrieval fixed by construction:
`retrieve_and_prompt` runs once and yields the chunks and the prompt text;
`generate_from_prompt` answers that text with a target. The administrator's
model-testing route uses the halves to answer one retrieval with both models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from app.course_id import assert_valid_course_id
from app.course_model_resolution import (
    resolve_course_model_version,
    resolve_current_course_model,
)
from app.course_rag import retrieve_course_syllabus_chunks
from app.finetuned_client import generate_finetuned_response
from app.grounded_generation import (
    GROUNDED_TIMEOUT_SECONDS,
    build_grounded_prompt,
    grounded_options,
)
from app.ollama import generate_ollama_chat
from app.retrieval_diversity import DEFAULT_TOP_K
from app.retrieval_facets import extract_question_facets
from app.storage import CourseArtifactStorage


@dataclass(frozen=True)
class GenerationTarget:
    """Which weights answer a grounded prompt.

    `base` is the backend's own Ollama model (`OLLAMA_MODEL`). `fineTuned` is
    the course's adapter through the fine-tuned service: the course's resolved
    version when `version` is None, or an explicitly named version, which only
    the administrator's model-testing route asks for.
    """

    kind: str
    version: str | None = None


BASE_TARGET = GenerationTarget("base")


def fine_tuned_target(version: str | None = None) -> GenerationTarget:
    return GenerationTarget("fineTuned", version)


def _validate_course_id(course_id: str) -> str:
    try:
        return assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _sources(retrieved_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunkId": chunk["chunk_id"],
            "sectionTitle": chunk["section"],
            "text": chunk["text"],
            "score": chunk["score"],
        }
        for chunk in retrieved_chunks
    ]


def _retrieved(retrieved_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunkId": chunk["chunk_id"],
            "section": chunk["section"],
            "text": chunk["text"],
            "score": chunk["score"],
        }
        for chunk in retrieved_chunks
    ]


async def retrieve_and_prompt(
    course_id: str,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    storage: CourseArtifactStorage | None = None,
) -> dict[str, Any]:
    """The half that does not depend on the model: chunks, facets, prompt."""
    safe_course_id = _validate_course_id(course_id)
    trimmed = " ".join(question.split())
    if not trimmed:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    _, retrieved_chunks = await retrieve_course_syllabus_chunks(
        course_id=safe_course_id,
        question=trimmed,
        top_k=top_k,
        storage=storage,
    )
    if not retrieved_chunks:
        raise HTTPException(
            status_code=404,
            detail=(
                f'No usable syllabus context was retrieved for course "{safe_course_id}". '
                "Cannot generate a grounded answer without retrieved context."
            ),
        )

    facets = extract_question_facets(trimmed)
    return {
        "courseId": safe_course_id,
        "question": trimmed,
        "facets": facets,
        "chunks": retrieved_chunks,
        "prompt": build_grounded_prompt(trimmed, retrieved_chunks, facets),
        "sources": _sources(retrieved_chunks),
        "retrievedChunks": _retrieved(retrieved_chunks),
    }


async def generate_from_prompt(
    prompt: str,
    *,
    course_id: str,
    target: GenerationTarget,
) -> dict[str, Any]:
    """Answer one already-built prompt with one target. Same call shape either way.

    The base target is the backend's Ollama through `/api/chat`, one user
    message, the shared options. The fine-tuned target is the fine-tuned
    service, which sends the same message shape and the same options to the
    same Ollama for the course's adapter model, and whose reply the client
    checks for the course and the version.
    """
    if target.kind == "base":
        generation = await generate_ollama_chat(
            prompt,
            options=grounded_options(),
            timeout=GROUNDED_TIMEOUT_SECONDS,
        )
        return {
            "answer": generation["answer"],
            "model": generation["model"],
            "modelVersion": None,
            "adapterLoaded": None,
            "generationSeconds": None,
            "responseType": "rag",
        }

    if target.kind != "fineTuned":
        raise ValueError(f"Unknown generation target: {target.kind!r}")

    # The same course resolution the plain fine-tuned path does: retrieval is
    # course-scoped already, and without this the grounded prompt would be
    # answered by whichever adapter the service happened to hold. An explicitly
    # named version goes through the registry too.
    if target.version is None:
        resolved = resolve_current_course_model(course_id)
    else:
        resolved = resolve_course_model_version(course_id, target.version)
    generation = await generate_finetuned_response(
        prompt,
        course_id=course_id,
        model_version=resolved["version"],
    )
    return {
        "answer": generation["answer"],
        "model": generation["model"],
        "modelVersion": generation.get("model_version") or resolved["version"],
        "adapterLoaded": generation["adapter_loaded"],
        "generationSeconds": generation["generation_seconds"],
        "responseType": "fineTunedRag",
    }


async def generate_grounded_answer(
    course_id: str,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    storage: CourseArtifactStorage | None = None,
    *,
    target: GenerationTarget,
) -> dict[str, Any]:
    """Retrieve, prompt, generate: the whole grounded path for one target."""
    prepared = await retrieve_and_prompt(course_id, question, top_k=top_k, storage=storage)
    generation = await generate_from_prompt(
        prepared["prompt"], course_id=prepared["courseId"], target=target
    )
    return {
        "courseId": prepared["courseId"],
        "question": prepared["question"],
        "prompt": prepared["prompt"],
        "sources": prepared["sources"],
        "retrievedChunks": prepared["retrievedChunks"],
        **generation,
    }
