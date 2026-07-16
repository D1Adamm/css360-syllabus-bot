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

STARTER_SEEDS_PER_CHUNK = 2
DEFAULT_STARTER_TARGET_COUNT = 50
MAX_STARTER_TARGET_COUNT = 50
MAX_STARTER_SELECTED_CHUNKS = 35
MAX_STARTER_OLLAMA_CALLS = 40
MIN_STARTER_CHUNK_CHARS = 80

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
_WHITESPACE_RE = re.compile(r"\s+")


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


def _try_normalize_seed(
    raw_seed: Any,
    *,
    chunk_id: str,
    index: int,
) -> dict[str, Any] | None:
    try:
        return _normalize_seed(raw_seed, chunk_id=chunk_id, index=index)
    except HTTPException:
        return None


def normalize_question_for_dedupe(question: str) -> str:
    """Normalize a question for exact-match deduplication."""
    lowered = question.strip().lower()
    without_punct = _NON_ALNUM_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", without_punct).strip()


def _load_course_index(
    course_id: str,
    storage: CourseArtifactStorage,
) -> dict[str, Any]:
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    index_data = storage.load_index(safe_course_id)
    if index_data is None:
        raise HTTPException(
            status_code=404,
            detail=f'No syllabus index found for course "{safe_course_id}".',
        )
    return index_data


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

    index_data = _load_course_index(safe_course_id, storage)
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


def _extract_eligible_chunks(
    raw_chunks: list[Any],
    *,
    min_chars: int = MIN_STARTER_CHUNK_CHARS,
) -> tuple[list[dict[str, str]], int]:
    """Return (eligible chunks in order, skipped count)."""
    eligible: list[dict[str, str]] = []
    skipped = 0

    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            skipped += 1
            continue

        chunk_id = raw_chunk.get("chunkId") or raw_chunk.get("id")
        text = raw_chunk.get("text")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            skipped += 1
            continue
        if not isinstance(text, str) or len(text.strip()) < min_chars:
            skipped += 1
            continue

        eligible.append(
            {
                "chunkId": chunk_id.strip(),
                "text": text.strip(),
            }
        )

    return eligible, skipped


def select_evenly_spaced_chunks(
    chunks: list[dict[str, str]],
    max_count: int,
) -> list[dict[str, str]]:
    """Pick up to max_count chunks spread evenly across the list, preserving order."""
    if max_count <= 0:
        return []
    total = len(chunks)
    if total <= max_count:
        return list(chunks)
    if max_count == 1:
        return [chunks[0]]

    selected: list[tuple[int, dict[str, str]]] = []
    used_indices: set[int] = set()
    for i in range(max_count):
        idx = int(round(i * (total - 1) / (max_count - 1)))
        if idx in used_indices:
            nudged = idx
            while nudged in used_indices and nudged < total - 1:
                nudged += 1
            if nudged in used_indices:
                nudged = idx
                while nudged in used_indices and nudged > 0:
                    nudged -= 1
            idx = nudged
        if idx in used_indices:
            continue
        used_indices.add(idx)
        selected.append((idx, chunks[idx]))

    selected.sort(key=lambda item: item[0])
    return [chunk for _, chunk in selected]


async def _generate_candidates_for_chunk(
    *,
    chunk_id: str,
    chunk_text: str,
    count: int,
    require_exact_count: bool,
) -> tuple[list[dict[str, Any]], str]:
    prompt = _build_seed_prompt(chunk_id, chunk_text, count)
    generation = await generate_ollama_completion(
        prompt,
        model=SEED_GENERATION_MODEL,
        response_format="json",
        think=False,
    )
    raw_seeds = _parse_seed_payload(generation["answer"])
    if require_exact_count and len(raw_seeds) < count:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned {len(raw_seeds)} seeds but {count} were requested.",
        )

    seeds: list[dict[str, Any]] = []
    for index, raw_seed in enumerate(raw_seeds[:count]):
        if require_exact_count:
            seeds.append(_normalize_seed(raw_seed, chunk_id=chunk_id, index=index))
        else:
            normalized = _try_normalize_seed(raw_seed, chunk_id=chunk_id, index=index)
            if normalized is not None:
                seeds.append(normalized)

    model = generation.get("model", SEED_GENERATION_MODEL)
    return seeds, model


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

    seeds, model = await _generate_candidates_for_chunk(
        chunk_id=cleaned_chunk_id,
        chunk_text=chunk_text,
        count=count,
        require_exact_count=True,
    )

    return {
        "courseId": course_id,
        "chunkId": cleaned_chunk_id,
        "model": model,
        "count": len(seeds),
        "seeds": seeds,
    }


async def generate_starter_seeds_for_course(
    *,
    course_id: str,
    target_count: int = DEFAULT_STARTER_TARGET_COUNT,
    storage: CourseArtifactStorage | None = None,
) -> dict[str, Any]:
    """Generate up to target_count starter seeds across a course syllabus.

    Processes an evenly spaced subset of eligible chunks (not only the opening
    sections). Does not persist to Firebase or trigger course creation.
    """
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if target_count < 1 or target_count > MAX_STARTER_TARGET_COUNT:
        raise HTTPException(
            status_code=422,
            detail=(
                f"targetCount must be between 1 and {MAX_STARTER_TARGET_COUNT}."
            ),
        )

    artifact_storage = storage or get_course_artifact_storage()
    index_data = _load_course_index(safe_course_id, artifact_storage)
    raw_chunks = index_data.get("chunks", [])
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise HTTPException(
            status_code=404,
            detail=f'No syllabus chunks found for course "{safe_course_id}".',
        )

    eligible, chunks_skipped = _extract_eligible_chunks(raw_chunks)
    if not eligible:
        raise HTTPException(
            status_code=404,
            detail=(
                f'No eligible syllabus chunks found for course "{safe_course_id}".'
            ),
        )

    selected = select_evenly_spaced_chunks(eligible, MAX_STARTER_SELECTED_CHUNKS)

    seeds: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    chunks_processed = 0
    ollama_calls = 0
    candidates_generated = 0
    duplicates_removed = 0
    model_used = SEED_GENERATION_MODEL

    for chunk in selected:
        if len(seeds) >= target_count:
            break
        if ollama_calls >= MAX_STARTER_OLLAMA_CALLS:
            break

        chunks_processed += 1
        ollama_calls += 1
        try:
            candidates, model_used = await _generate_candidates_for_chunk(
                chunk_id=chunk["chunkId"],
                chunk_text=chunk["text"],
                count=STARTER_SEEDS_PER_CHUNK,
                require_exact_count=False,
            )
        except HTTPException as exc:
            # Infrastructure failures should abort; per-chunk model/parse issues skip.
            if exc.status_code == 503:
                raise
            continue

        candidates_generated += len(candidates)
        for candidate in candidates:
            if len(seeds) >= target_count:
                break
            normalized_question = normalize_question_for_dedupe(candidate["question"])
            if not normalized_question:
                duplicates_removed += 1
                continue
            if normalized_question in seen_questions:
                duplicates_removed += 1
                continue
            seen_questions.add(normalized_question)
            seeds.append(candidate)

    return {
        "courseId": safe_course_id,
        "model": model_used,
        "targetCount": target_count,
        "seeds": seeds,
        "progress": {
            "eligibleChunks": len(eligible),
            "selectedChunks": len(selected),
            "chunksProcessed": chunks_processed,
            "chunksSkipped": chunks_skipped,
            "ollamaCalls": ollama_calls,
            "candidatesGenerated": candidates_generated,
            "duplicatesRemoved": duplicates_removed,
            "finalCount": len(seeds),
        },
    }
