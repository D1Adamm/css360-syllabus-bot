"""Generate a small batch of AI seed examples from one syllabus chunk."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from fastapi import HTTPException

from app.course_id import assert_valid_course_id
from app.firebase_seeds import (
    FirebaseConfigurationError,
    build_chunk_section_lookup,
    persist_accepted_seeds,
)
from app.ollama import embed_ollama_texts, generate_ollama_completion
from app.seed_balance import (
    SCENARIO_LIKE_TYPES,
    compute_scenario_minimum,
    count_schedule_like,
    count_topics,
    should_prefer_scenario_or_clarification,
    would_violate_balancing,
)
from app.seed_dedupe import normalize_question_for_dedupe
from app.seed_similarity import find_semantic_duplicate_question
from app.storage import CourseArtifactStorage, get_course_artifact_storage
from app.syllabus_plan import (
    build_section_title_fallback_plan,
    merge_topics_by_overlap,
    plan_syllabus_topics,
)
from app.seed_validation import (
    VALIDATION_PROMPT_MARKER,
    build_validation_prompt,
    calibrate_validation_result,
    try_parse_validation_payload as parse_rubric_validation_payload,
    validation_result_accepts as rubric_validation_result_accepts,
)

SEED_GENERATION_MODEL = "qwen3:4b"
DEFAULT_SEED_COUNT = 3
MAX_SEED_COUNT = 5

STARTER_SEEDS_PER_CHUNK = 2
DEFAULT_STARTER_TARGET_COUNT = 50
MAX_STARTER_TARGET_COUNT = 50
MAX_STARTER_SELECTED_CHUNKS = 35
DEFAULT_STARTER_MAX_TOTAL_OLLAMA_CALLS = 140
MIN_STARTER_CHUNK_CHARS = 80
MIN_QUESTION_CHARS = 8
MAX_QUESTION_CHARS = 300
MIN_ANSWER_CHARS = 8
MAX_ANSWER_CHARS = 600
STARTER_SEED_CANDIDATES_PER_TOPIC_CALL = 4
QUESTION_TYPES = ("direct", "scenario", "clarification", "procedure", "comparison")

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)


def get_starter_max_total_ollama_calls() -> int:
    """Total Ollama call budget for starter generation plus validation."""
    raw = os.getenv("STARTER_MAX_TOTAL_OLLAMA_CALLS")
    if raw is None:
        return DEFAULT_STARTER_MAX_TOTAL_OLLAMA_CALLS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_STARTER_MAX_TOTAL_OLLAMA_CALLS
    return max(1, value)


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


def _build_validation_prompt(question: str, answer: str, chunk_text: str) -> str:
    return build_validation_prompt(
        question=question,
        answer=answer,
        topic_name="General",
        question_type="direct",
        chunk_text=chunk_text,
    )


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


def _try_parse_validation_payload(raw: str) -> dict[str, Any] | None:
    return parse_rubric_validation_payload(raw)


def validation_result_accepts(result: dict[str, Any]) -> bool:
    return rubric_validation_result_accepts(result)


def passes_programmatic_candidate_checks(candidate: dict[str, Any]) -> bool:
    question = str(candidate.get("question", "")).strip()
    answer = str(candidate.get("answer", "")).strip()
    if not question or not answer:
        return False
    if len(question) < MIN_QUESTION_CHARS or len(question) > MAX_QUESTION_CHARS:
        return False
    if len(answer) < MIN_ANSWER_CHARS or len(answer) > MAX_ANSWER_CHARS:
        return False

    source_chunk_ids = candidate.get("sourceChunkIds")
    if not isinstance(source_chunk_ids, list) or not source_chunk_ids:
        return False
    if not any(str(item).strip() for item in source_chunk_ids):
        return False
    return True


def _normalize_seed(
    raw_seed: Any,
    *,
    chunk_id: str,
    index: int,
    default_category: str | None = None,
    default_source_chunk_ids: list[str] | None = None,
    default_question_type: str | None = None,
    topic_summary: str | None = None,
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

    category = str(raw_seed.get("category", "")).strip() or (default_category or "general")
    source_chunk_ids = raw_seed.get("sourceChunkIds")
    if not isinstance(source_chunk_ids, list) or not source_chunk_ids:
        source_chunk_ids = list(default_source_chunk_ids or [chunk_id])
    else:
        source_chunk_ids = [str(item).strip() for item in source_chunk_ids if str(item).strip()]
        if not source_chunk_ids:
            source_chunk_ids = list(default_source_chunk_ids or [chunk_id])

    question_type = str(raw_seed.get("questionType", "")).strip().lower() or (
        default_question_type or "direct"
    )
    if question_type not in QUESTION_TYPES:
        question_type = default_question_type or "direct"

    return {
        "question": question,
        "answer": answer,
        "category": category,
        "sourceChunkIds": source_chunk_ids,
        "questionType": question_type,
        "topicSummary": topic_summary or "",
        "origin": "ai_generated",
        "status": "generated",
    }


def _try_normalize_seed(
    raw_seed: Any,
    *,
    chunk_id: str,
    index: int,
    default_category: str | None = None,
    default_source_chunk_ids: list[str] | None = None,
    default_question_type: str | None = None,
    topic_summary: str | None = None,
) -> dict[str, Any] | None:
    try:
        return _normalize_seed(
            raw_seed,
            chunk_id=chunk_id,
            index=index,
            default_category=default_category,
            default_source_chunk_ids=default_source_chunk_ids,
            default_question_type=default_question_type,
            topic_summary=topic_summary,
        )
    except HTTPException:
        return None


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


def _build_topic_seed_prompt(
    *,
    topic: dict[str, Any],
    chunk_texts: list[str],
    count: int,
    prefer_scenario_like: bool,
    requested_question_types: list[str] | None = None,
) -> str:
    topic_name = str(topic.get("name", "General")).strip() or "General"
    topic_summary = str(topic.get("summary", "")).strip()
    source_chunk_ids = ", ".join(topic.get("sourceChunkIds", []))
    if requested_question_types:
        allowed_types = ", ".join(requested_question_types)
        question_type_hint = f"All returned seeds must use questionType values from: {allowed_types}."
    else:
        question_type_hint = (
            "At least half of the questions should be scenario or clarification."
            if prefer_scenario_like
            else "Include a mix of direct, scenario, clarification, procedure, and comparison."
        )
    chunk_block = "\n\n".join(chunk_texts)
    return f"""You generate training seed examples for a syllabus Q&A chatbot.

Generate up to {count} student question/answer examples for this ONE syllabus topic.

Rules:
- Use only facts present in the provided chunk text.
- Topic/category must stay: {topic_name}
- Use natural student wording.
- Return a mix of question types: direct, scenario, clarification, procedure, comparison.
- {question_type_hint}
- Return ONLY valid JSON.

Required JSON shape:
{{
  "seeds": [
    {{
      "question": "string",
      "answer": "string",
      "category": "{topic_name}",
      "questionType": "direct|scenario|clarification|procedure|comparison",
      "sourceChunkIds": ["chunk-001"]
    }}
  ]
}}

Topic summary: {topic_summary}
Source chunk ids: {source_chunk_ids}

Source chunk text:
{chunk_block}
"""


async def _generate_candidates_for_topic(
    *,
    topic: dict[str, Any],
    chunk_texts: list[str],
    count: int,
    prefer_scenario_like: bool,
    requested_question_types: list[str] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    prompt = _build_topic_seed_prompt(
        topic=topic,
        chunk_texts=chunk_texts,
        count=count,
        prefer_scenario_like=prefer_scenario_like,
        requested_question_types=requested_question_types,
    )
    generation = await generate_ollama_completion(
        prompt,
        model=SEED_GENERATION_MODEL,
        response_format="json",
        think=False,
    )
    raw_seeds = _parse_seed_payload(generation["answer"])
    seeds: list[dict[str, Any]] = []
    topic_chunk_ids = [
        str(item).strip() for item in topic.get("sourceChunkIds", []) if str(item).strip()
    ]
    fallback_chunk_id = topic_chunk_ids[0] if topic_chunk_ids else "topic"
    for index, raw_seed in enumerate(raw_seeds[:count]):
        normalized = _try_normalize_seed(
            raw_seed,
            chunk_id=fallback_chunk_id,
            index=index,
            default_category=str(topic.get("name", "General")).strip() or "General",
            default_source_chunk_ids=topic_chunk_ids or [fallback_chunk_id],
            default_question_type=(
                requested_question_types[0]
                if requested_question_types
                else ("scenario" if prefer_scenario_like else "direct")
            ),
            topic_summary=str(topic.get("summary", "")).strip(),
        )
        if normalized is not None:
            seeds.append(normalized)

    model = generation.get("model", SEED_GENERATION_MODEL)
    return seeds, model


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


async def _validate_candidate(
    *,
    question: str,
    answer: str,
    topic_name: str,
    question_type: str,
    chunk_text: str,
) -> dict[str, Any] | None:
    prompt = build_validation_prompt(
        question=question,
        answer=answer,
        topic_name=topic_name,
        question_type=question_type,
        chunk_text=chunk_text,
    )
    generation = await generate_ollama_completion(
        prompt,
        model=SEED_GENERATION_MODEL,
        response_format="json",
        think=False,
    )
    return _try_parse_validation_payload(generation["answer"])


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
    save: bool = False,
    storage: CourseArtifactStorage | None = None,
) -> dict[str, Any]:
    """Generate up to target_count starter seeds across a course syllabus.

    Processes an evenly spaced subset of eligible chunks (not only the opening
    sections). When save=True, persists accepted validated seeds to Firebase.
    Does not trigger course creation.
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

    chunk_lookup = {
        str(chunk.get("chunkId") or chunk.get("id")).strip(): chunk
        for chunk in raw_chunks
        if isinstance(chunk, dict) and isinstance(chunk.get("text"), str)
    }

    seeds: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    accepted_questions: list[str] = []
    chunks_processed = 0
    generation_calls = 0
    validation_calls = 0
    planning_calls = 0
    merge_calls = 0
    ollama_calls = 0
    embedding_calls = 0
    candidates_generated = 0
    candidates_validated = 0
    candidates_accepted = 0
    candidates_rejected = 0
    candidates_rejected_invalid = 0
    candidates_rejected_validation = 0
    candidates_rejected_unsupported = 0
    candidates_rejected_balancing = 0
    duplicates_removed = 0
    semantic_duplicates_removed = 0
    model_used = SEED_GENERATION_MODEL
    max_total_calls = get_starter_max_total_ollama_calls()

    class _BudgetExhausted(Exception):
        pass

    def _budget_remaining() -> bool:
        return ollama_calls < max_total_calls

    async def _counted_completion(prompt: str, **kwargs: Any) -> dict[str, str]:
        nonlocal ollama_calls, planning_calls, merge_calls
        if not _budget_remaining():
            raise _BudgetExhausted()
        ollama_calls += 1
        if "You merge syllabus starter-seed topics." in prompt:
            merge_calls += 1
        elif "You plan course-specific syllabus starter-seed topics." in prompt:
            planning_calls += 1
        return await generate_ollama_completion(prompt, **kwargs)

    try:
        plan = await plan_syllabus_topics(
            raw_chunks=raw_chunks,
            target_count=target_count,
            completion_fn=_counted_completion,
        )
    except _BudgetExhausted:
        plan = build_section_title_fallback_plan(raw_chunks, target_count=target_count)
    except HTTPException as exc:
        if exc.status_code == 503:
            raise
        plan = build_section_title_fallback_plan(raw_chunks, target_count=target_count)

    topics = list(plan.get("topics", []))
    if not topics:
        topics = build_section_title_fallback_plan(raw_chunks, target_count=target_count)["topics"]
    else:
        fallback_topics = build_section_title_fallback_plan(
            raw_chunks,
            target_count=target_count,
        )["topics"]
        covered_chunk_ids = {
            str(chunk_id).strip()
            for topic in topics
            for chunk_id in topic.get("sourceChunkIds", [])
            if str(chunk_id).strip()
        }
        supplemental_topics = [
            topic
            for topic in fallback_topics
            if any(
                str(chunk_id).strip() not in covered_chunk_ids
                for chunk_id in topic.get("sourceChunkIds", [])
            )
        ]
        if supplemental_topics:
            topics = merge_topics_by_overlap([*topics, *supplemental_topics])

    importance_rank = {"high": 0, "medium": 1, "low": 2}
    topics.sort(key=lambda item: (importance_rank.get(str(item.get("importance")), 3), item["name"]))

    for pass_index in range(2):
        for topic in topics:
            if len(seeds) >= target_count or not _budget_remaining():
                break

            topic_chunk_ids = [
                str(item).strip() for item in topic.get("sourceChunkIds", []) if str(item).strip()
            ]
            topic_chunk_texts = []
            for chunk_id in topic_chunk_ids:
                raw_chunk = chunk_lookup.get(chunk_id)
                text = raw_chunk.get("text") if isinstance(raw_chunk, dict) else None
                if isinstance(text, str) and text.strip():
                    topic_chunk_texts.append(text.strip())
            if not topic_chunk_texts:
                continue

            topic_counts = count_topics(seeds)
            remaining_slots = target_count - len(seeds)
            topic_name = str(topic.get("name", "General")).strip() or "General"
            topic_count = topic_counts[topic_name]
            suggested_count = int(topic.get("suggestedExampleCount", 1))
            topic_target = suggested_count if pass_index == 0 else min(2, suggested_count + 1)
            topic_call_target = min(
                STARTER_SEED_CANDIDATES_PER_TOPIC_CALL,
                remaining_slots,
                max(1, topic_target - topic_count),
            )
            if topic_call_target <= 0:
                continue

            prefer_scenario_like = should_prefer_scenario_or_clarification(
                accepted_seeds=seeds,
                remaining_slots=remaining_slots,
                target_count=target_count,
            )
            requested_question_types = (
                ["scenario", "clarification"] if prefer_scenario_like else None
            )

            chunks_processed += len(topic_chunk_texts)
            ollama_calls += 1
            generation_calls += 1
            try:
                candidates, model_used = await _generate_candidates_for_topic(
                    topic=topic,
                    chunk_texts=topic_chunk_texts,
                    count=topic_call_target,
                    prefer_scenario_like=prefer_scenario_like,
                    requested_question_types=requested_question_types,
                )
            except HTTPException as exc:
                if exc.status_code == 503:
                    raise
                continue

            candidates_generated += len(candidates)
            for candidate in candidates:
                if len(seeds) >= target_count:
                    break

                if not passes_programmatic_candidate_checks(candidate):
                    candidates_rejected += 1
                    candidates_rejected_invalid += 1
                    continue

                normalized_question = normalize_question_for_dedupe(candidate["question"])
                if not normalized_question:
                    duplicates_removed += 1
                    candidates_rejected += 1
                    continue
                if normalized_question in seen_questions:
                    duplicates_removed += 1
                    candidates_rejected += 1
                    continue

                if accepted_questions:
                    embedding_calls += 1
                    try:
                        semantic_duplicate = await find_semantic_duplicate_question(
                            candidate_question=candidate["question"],
                            accepted_questions=accepted_questions,
                            embed_fn=embed_ollama_texts,
                        )
                    except HTTPException:
                        semantic_duplicate = None
                    if semantic_duplicate is not None:
                        semantic_duplicates_removed += 1
                        candidates_rejected += 1
                        continue

                if (
                    str(candidate.get("questionType", "")).strip().lower()
                    not in SCENARIO_LIKE_TYPES
                    and should_prefer_scenario_or_clarification(
                        accepted_seeds=seeds,
                        remaining_slots=target_count - len(seeds),
                        target_count=target_count,
                    )
                ):
                    candidates_rejected += 1
                    candidates_rejected_balancing += 1
                    continue

                balancing_failure = would_violate_balancing(
                    candidate=candidate,
                    accepted_seeds=seeds,
                    target_count=target_count,
                    planner_topics=topics,
                )
                if balancing_failure is not None:
                    candidates_rejected += 1
                    candidates_rejected_balancing += 1
                    continue

                if not _budget_remaining():
                    break

                ollama_calls += 1
                validation_calls += 1
                try:
                    validation = await _validate_candidate(
                        question=candidate["question"],
                        answer=candidate["answer"],
                        topic_name=str(candidate.get("category", "General")),
                        question_type=str(candidate.get("questionType", "direct")),
                        chunk_text="\n\n".join(topic_chunk_texts),
                    )
                except HTTPException as exc:
                    if exc.status_code == 503:
                        raise
                    candidates_validated += 1
                    candidates_rejected += 1
                    candidates_rejected_validation += 1
                    continue

                candidates_validated += 1
                if validation is None:
                    candidates_rejected += 1
                    candidates_rejected_validation += 1
                    continue
                validation = calibrate_validation_result(
                    result=validation,
                    question=candidate["question"],
                    answer=candidate["answer"],
                    topic_name=str(candidate.get("category", "General")),
                    question_type=str(candidate.get("questionType", "direct")),
                )
                if not validation_result_accepts(validation):
                    candidates_rejected += 1
                    candidates_rejected_validation += 1
                    if validation.get("unsupportedClaims"):
                        candidates_rejected_unsupported += 1
                    continue

                seen_questions.add(normalized_question)
                accepted_questions.append(candidate["question"])
                candidates_accepted += 1
                seeds.append(
                    {
                        **candidate,
                        "topicId": topic.get("topicId"),
                        "validation": validation,
                    }
                )

    result: dict[str, Any] = {
        "courseId": safe_course_id,
        "model": model_used,
        "targetCount": target_count,
        "seeds": seeds,
        "topicPlan": {"topics": topics},
        "progress": {
            "eligibleChunks": len(eligible),
            "selectedChunks": len(
                {chunk_id for topic in topics for chunk_id in topic.get("sourceChunkIds", [])}
            ),
            "chunksProcessed": chunks_processed,
            "chunksSkipped": chunks_skipped,
            "planningCalls": planning_calls,
            "mergeCalls": merge_calls,
            "generationCalls": generation_calls,
            "validationCalls": validation_calls,
            "ollamaCalls": ollama_calls,
            "embeddingCalls": embedding_calls,
            "candidatesGenerated": candidates_generated,
            "candidatesValidated": candidates_validated,
            "candidatesAccepted": candidates_accepted,
            "candidatesRejected": candidates_rejected,
            "duplicatesRemoved": duplicates_removed,
            "semanticDuplicatesRemoved": semantic_duplicates_removed,
            "candidatesRejectedInvalid": candidates_rejected_invalid,
            "candidatesRejectedValidation": candidates_rejected_validation,
            "candidatesRejectedUnsupportedClaims": candidates_rejected_unsupported,
            "candidatesRejectedBalancing": candidates_rejected_balancing,
            "scheduleCount": count_schedule_like(seeds),
            "scenarioOrClarificationMinimum": compute_scenario_minimum(target_count),
            "finalCount": len(seeds),
        },
    }

    if save:
        try:
            chunk_sections = build_chunk_section_lookup(raw_chunks)
            result["persistence"] = await persist_accepted_seeds(
                course_id=safe_course_id,
                seeds=seeds,
                chunk_sections=chunk_sections,
            )
        except FirebaseConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return result
