"""Generate a small batch of AI seed examples from one syllabus chunk."""

from __future__ import annotations

import json
import re
from typing import Any

from fastapi import HTTPException

from app.course_id import assert_valid_course_id
from app.ollama import generate_ollama_completion
from app.storage import CourseArtifactStorage, get_course_artifact_storage

SEED_GENERATION_MODEL = "qwen3:4b"
DEFAULT_SEED_COUNT = 3
MAX_SEED_COUNT = 5

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)


def _build_seed_prompt(chunk_id: str, chunk_text: str, count: int) -> str:
    return f"""You generate training seed examples for a syllabus Q&A chatbot.

Given ONE syllabus chunk, create exactly {count} student question/answer examples
that can be answered directly from that chunk.

Rules:
- Use only facts present in the chunk text.
- Questions should sound like real student questions.
- Answers should be concise and grounded in the chunk.
- Assign a short category label (e.g. grading, late policy, office hours, schedule).
- Return ONLY valid JSON (no markdown, no commentary).

Required JSON shape:
{{
  "seeds": [
    {{
      "question": "string",
      "answer": "string",
      "category": "string"
    }}
  ]
}}

Chunk id: {chunk_id}

Chunk text:
{chunk_text}
"""


def _extract_json_text(raw: str) -> str:
    text = raw.strip()
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()

    # Fall back to the outermost JSON object/array if the model added prose.
    object_start = text.find("{")
    array_start = text.find("[")
    if object_start == -1 and array_start == -1:
        return text

    if object_start == -1:
        start = array_start
        end = text.rfind("]")
    elif array_start == -1:
        start = object_start
        end = text.rfind("}")
    else:
        start = min(object_start, array_start)
        end = text.rfind("]") if array_start < object_start else text.rfind("}")

    if end == -1 or end <= start:
        return text
    return text[start : end + 1]


def _parse_seed_payload(raw: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(_extract_json_text(raw))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned malformed JSON for seed generation.",
        ) from exc

    if isinstance(parsed, dict):
        seeds = parsed.get("seeds")
        if seeds is None:
            raise HTTPException(
                status_code=502,
                detail="Ollama JSON response is missing a seeds array.",
            )
    elif isinstance(parsed, list):
        seeds = parsed
    else:
        raise HTTPException(
            status_code=502,
            detail="Ollama JSON response must be an object or array of seeds.",
        )

    if not isinstance(seeds, list) or not seeds:
        raise HTTPException(
            status_code=502,
            detail="Ollama JSON response must include a non-empty seeds array.",
        )

    return seeds


def _normalize_seed(
    raw_seed: Any,
    *,
    chunk_id: str,
    index: int,
) -> dict[str, Any]:
    if not isinstance(raw_seed, dict):
        raise HTTPException(
            status_code=502,
            detail=f"Seed at index {index} must be a JSON object.",
        )

    question = str(raw_seed.get("question", "")).strip()
    answer = str(raw_seed.get("answer", "")).strip()
    if not question or not answer:
        raise HTTPException(
            status_code=502,
            detail=f"Seed at index {index} is missing a question or answer.",
        )

    category = str(raw_seed.get("category", "")).strip() or "general"
    source_chunk_ids = raw_seed.get("sourceChunkIds")
    if not isinstance(source_chunk_ids, list) or not source_chunk_ids:
        source_chunk_ids = [chunk_id]
    else:
        source_chunk_ids = [str(item).strip() for item in source_chunk_ids if str(item).strip()]
        if not source_chunk_ids:
            source_chunk_ids = [chunk_id]

    return {
        "question": question,
        "answer": answer,
        "category": category,
        "sourceChunkIds": source_chunk_ids,
        "origin": "ai_generated",
        "status": "generated",
    }


def _load_chunk_text(
    *,
    course_id: str,
    chunk_id: str,
    storage: CourseArtifactStorage,
) -> str:
    """Load one stored chunk's text from the course syllabus index."""
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    cleaned_chunk_id = chunk_id.strip()
    if not cleaned_chunk_id:
        raise HTTPException(status_code=422, detail="chunkId must not be empty.")

    index_data = storage.load_index(safe_course_id)
    if index_data is None:
        raise HTTPException(
            status_code=404,
            detail=f'No syllabus index found for course "{safe_course_id}".',
        )

    raw_chunks = index_data.get("chunks", [])
    if not isinstance(raw_chunks, list):
        raise HTTPException(
            status_code=404,
            detail=f'No syllabus chunks found for course "{safe_course_id}".',
        )

    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            continue
        stored_id = raw_chunk.get("chunkId") or raw_chunk.get("id")
        if not isinstance(stored_id, str) or stored_id.strip() != cleaned_chunk_id:
            continue

        text = raw_chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(
                status_code=404,
                detail=(
                    f'Chunk "{cleaned_chunk_id}" was found for course '
                    f'"{safe_course_id}" but has no usable text.'
                ),
            )
        return text.strip()

    raise HTTPException(
        status_code=404,
        detail=(
            f'Chunk "{cleaned_chunk_id}" was not found for course "{safe_course_id}".'
        ),
    )


async def generate_seeds_from_chunk(
    *,
    course_id: str,
    chunk_id: str,
    count: int = DEFAULT_SEED_COUNT,
    storage: CourseArtifactStorage | None = None,
) -> dict[str, Any]:
    """Generate a small batch of AI seeds from one stored syllabus chunk.

    Loads chunk text from the course index. Does not persist to Firebase or
    mutate course creation flows.
    """
    cleaned_chunk_id = chunk_id.strip()
    if not cleaned_chunk_id:
        raise HTTPException(status_code=422, detail="chunkId must not be empty.")

    if count < 1 or count > MAX_SEED_COUNT:
        raise HTTPException(
            status_code=422,
            detail=f"count must be between 1 and {MAX_SEED_COUNT}.",
        )

    artifact_storage = storage or get_course_artifact_storage()
    chunk_text = _load_chunk_text(
        course_id=course_id,
        chunk_id=cleaned_chunk_id,
        storage=artifact_storage,
    )

    prompt = _build_seed_prompt(cleaned_chunk_id, chunk_text, count)
    generation = await generate_ollama_completion(
        prompt,
        model=SEED_GENERATION_MODEL,
        response_format="json",
        think=False,
    )

    raw_seeds = _parse_seed_payload(generation["answer"])
    if len(raw_seeds) < count:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned {len(raw_seeds)} seeds but {count} were requested.",
        )

    seeds = [
        _normalize_seed(raw_seed, chunk_id=cleaned_chunk_id, index=index)
        for index, raw_seed in enumerate(raw_seeds[:count])
    ]

    return {
        "courseId": course_id,
        "chunkId": cleaned_chunk_id,
        "model": generation.get("model", SEED_GENERATION_MODEL),
        "count": len(seeds),
        "seeds": seeds,
    }
