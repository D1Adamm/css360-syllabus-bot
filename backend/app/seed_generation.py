"""Generate a small batch of AI seed examples from one syllabus chunk."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from fastapi import HTTPException

from app.course_id import assert_valid_course_id
from app.firebase_seeds import (
    FirebaseConfigurationError,
    build_chunk_section_lookup,
    persist_accepted_seeds,
)
from app.seed_export import write_generation_snapshot
from app.ollama import (
    embed_ollama_texts,
    generate_starter_ollama_completion,
    is_ollama_timeout_error,
)
from app.fact_inventory_cache import load_or_build_fact_inventory
from app.seed_allocation import allocate_slots
from app.seed_balance import compute_scenario_minimum, count_schedule_like
from app.seed_batching import (
    build_batch_fact_seed_prompt,
    build_batch_validation_prompt,
    parse_batch_seed_payload,
    parse_batch_validation_payload,
)
from app.seed_dedupe import normalize_question_for_dedupe
from app.seed_prevalidation import prevalidate_candidate
from app.seed_similarity import AcceptedEmbeddingCache, find_semantic_duplicate_question
from app.storage import CourseArtifactStorage, get_course_artifact_storage
from app.syllabus_facts import FACT_EXTRACTION_PROMPT_MARKER
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
FACT_SEED_GENERATION_PROMPT_MARKER = (
    "You generate training seed examples for ONE syllabus fact."
)

# Map allocator style hints to generation questionType values.
_STYLE_TO_QUESTION_TYPE = {
    "factual": "direct",
    "policy": "direct",
    "scenario": "scenario",
    "exception": "scenario",
    "clarification": "clarification",
    "procedural": "procedure",
}

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


def resolve_starter_run_status(
    *,
    target_count: int,
    final_count: int,
    saved_count: int = 0,
    save: bool = False,
) -> str:
    """Classify a starter run as ready, partial, or failed for baseline reporting.

    When save=False, status is based on generated final_count vs target_count only.
    When save=True, both generation and persistence must reach the target for ready.
    """
    if target_count <= 0:
        return "ready"
    if final_count >= target_count and (not save or saved_count >= target_count):
        return "ready"
    if final_count > 0 or saved_count > 0:
        return "partial"
    return "failed"


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
        "reviewStatus": "generated",
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


def _category_for_fact(fact: dict[str, Any]) -> str:
    kind = str(fact.get("kind") or "").strip()
    if kind and kind != "other":
        return kind.replace("_", " ")
    scope = str(fact.get("scope") or "").strip()
    if scope and scope != "other":
        return scope.replace("_", " ")
    return "general"


def _question_types_for_styles(styles: list[str] | None, *, slot_count: int) -> list[str]:
    """Translate allocator style hints into questionType values."""
    mapped: list[str] = []
    for style in styles or []:
        question_type = _STYLE_TO_QUESTION_TYPE.get(str(style).strip().lower())
        if question_type and question_type not in mapped:
            mapped.append(question_type)
    if not mapped:
        mapped = ["direct"]
        if slot_count >= 2:
            mapped.append("scenario")
        if slot_count >= 3:
            mapped.append("clarification")
    return mapped[: max(1, slot_count)]


def _build_fact_seed_prompt(
    *,
    fact: dict[str, Any],
    chunk_texts: list[str],
    count: int,
    suggested_styles: list[str] | None = None,
) -> str:
    fact_id = str(fact.get("factId") or "").strip()
    statement = str(fact.get("statement") or "").strip()
    evidence_quote = str(fact.get("evidenceQuote") or "").strip()
    source_chunk_ids = ", ".join(
        str(item).strip()
        for item in (fact.get("sourceChunkIds") or [])
        if str(item).strip()
    )
    category = _category_for_fact(fact)
    question_types = _question_types_for_styles(suggested_styles, slot_count=count)
    allowed_types = ", ".join(question_types)

    if count <= 1:
        diversity_rule = (
            "Generate exactly ONE strong, natural student question and answer "
            "for this fact."
        )
    else:
        diversity_rule = (
            f"Generate exactly {count} materially different useful student questions "
            "for this fact (not paraphrases). Vary wording and angle using the "
            f"allowed question types ({allowed_types}) when that helps, but never "
            "invent details beyond the evidence."
        )

    chunk_block = "\n\n".join(chunk_texts)
    return f"""{FACT_SEED_GENERATION_PROMPT_MARKER}

Generate exactly {count} student question/answer example(s) for this ONE syllabus fact.

Grounding rules (strict):
- The question must be fully answerable from ONLY the fact statement, evidence quote,
  and source chunk text below.
- The answer must stay within that evidence. Do not broaden, generalize, or add
  syllabus knowledge that is not present.
- Preserve qualifiers, limits, exceptions, and conditions exactly
  (e.g. "one", "per quarter", "except", "unless", "may", "optional").
- Do NOT convert permission into obligation:
  - "may" / "can" / "allowed" must NOT become "must" / "required"
- Do NOT convert recommendations into requirements
  ("suggested" / "recommended" must NOT become "required").
- Do NOT turn response-time / nudge guidance into a guarantee that the instructor
  will reply within a fixed time.
- Do NOT invent consequences, penalties, grades, or outcomes not in the evidence.
- Prefer the wording of the evidence for numbers, deadlines, and exceptions.
- Use natural student wording for the question; keep the answer concise and faithful.
- Category must stay: {category}
- Allowed questionType values: {allowed_types}
- {diversity_rule}
- Return ONLY valid JSON (no markdown, no commentary).

Required JSON shape:
{{
  "seeds": [
    {{
      "question": "string",
      "answer": "string",
      "category": "{category}",
      "questionType": "direct|scenario|clarification|procedure|comparison",
      "sourceChunkIds": ["chunk-001"]
    }}
  ]
}}

Fact id: {fact_id}
Fact statement: {statement}
Evidence quote: {evidence_quote}
Source chunk ids: {source_chunk_ids}

Source chunk text:
{chunk_block}
"""


async def _generate_candidates_for_fact(
    *,
    fact: dict[str, Any],
    chunk_texts: list[str],
    count: int,
    suggested_styles: list[str] | None = None,
    completion_fn=None,
) -> tuple[list[dict[str, Any]], str]:
    """Generate seed candidates grounded in one allocated fact."""
    run_completion = completion_fn or generate_starter_ollama_completion
    prompt = _build_fact_seed_prompt(
        fact=fact,
        chunk_texts=chunk_texts,
        count=count,
        suggested_styles=suggested_styles,
    )
    generation = await run_completion(
        prompt,
        model=SEED_GENERATION_MODEL,
        response_format="json",
        think=False,
    )
    raw_seeds = _parse_seed_payload(generation["answer"])
    seeds: list[dict[str, Any]] = []
    for index, raw_seed in enumerate(raw_seeds[:count]):
        if not isinstance(raw_seed, dict):
            continue
        normalized = _normalize_fact_seed(
            raw_seed=raw_seed,
            fact=fact,
            index=index,
            suggested_styles=suggested_styles,
            count=count,
        )
        if normalized is not None:
            seeds.append(normalized)

    model = generation.get("model", SEED_GENERATION_MODEL)
    return seeds, model


def _normalize_fact_seed(
    *,
    raw_seed: dict[str, Any],
    fact: dict[str, Any],
    index: int,
    suggested_styles: list[str] | None,
    count: int,
) -> dict[str, Any] | None:
    fact_chunk_ids = [
        str(item).strip()
        for item in (fact.get("sourceChunkIds") or [])
        if str(item).strip()
    ]
    fallback_chunk_id = fact_chunk_ids[0] if fact_chunk_ids else "fact"
    default_question_types = _question_types_for_styles(
        suggested_styles, slot_count=count
    )
    category = _category_for_fact(fact)
    normalized = _try_normalize_seed(
        raw_seed,
        chunk_id=fallback_chunk_id,
        index=index,
        default_category=category,
        default_source_chunk_ids=fact_chunk_ids or [fallback_chunk_id],
        default_question_type=default_question_types[
            min(index, len(default_question_types) - 1)
        ],
        topic_summary=str(fact.get("statement") or "").strip(),
    )
    if normalized is None:
        return None
    normalized["sourceChunkIds"] = list(fact_chunk_ids or [fallback_chunk_id])
    normalized["factId"] = str(fact.get("factId") or "")
    normalized["evidenceQuote"] = str(fact.get("evidenceQuote") or "")
    return normalized


async def _generate_candidates_for_facts_batch(
    *,
    items: list[dict[str, Any]],
    completion_fn=None,
) -> tuple[dict[str, list[dict[str, Any]]], str, dict[str, int]]:
    """Unused by live starter generation (Phase 6 disabled). Kept for unit tests."""
    run_completion = completion_fn or generate_starter_ollama_completion
    metrics = {
        "generation_calls": 0,
        "generation_batch_calls": 0,
        "timeout_failures": 0,
        "max_generation_batch_size": 0,
    }
    model_used = SEED_GENERATION_MODEL
    results: dict[str, list[dict[str, Any]]] = {
        str(item["fact"].get("factId") or ""): [] for item in items
    }

    async def _run(batch: list[dict[str, Any]]) -> None:
        nonlocal model_used
        if not batch:
            return
        metrics["max_generation_batch_size"] = max(
            metrics["max_generation_batch_size"], len(batch)
        )
        if len(batch) == 1:
            item = batch[0]
            fact = item["fact"]
            fact_id = str(fact.get("factId") or "")
            metrics["generation_calls"] += 1
            try:
                seeds, model = await _generate_candidates_for_fact(
                    fact=fact,
                    chunk_texts=item.get("chunk_texts") or [],
                    count=1,
                    suggested_styles=list(item.get("suggested_styles") or []),
                    completion_fn=run_completion,
                )
            except HTTPException as exc:
                if is_ollama_timeout_error(exc):
                    metrics["timeout_failures"] += 1
                    return
                if exc.status_code == 503:
                    raise
                return
            model_used = model
            results[fact_id] = seeds
            return

        metrics["generation_calls"] += 1
        metrics["generation_batch_calls"] += 1
        prompt = build_batch_fact_seed_prompt(items=batch)
        try:
            generation = await run_completion(
                prompt,
                model=SEED_GENERATION_MODEL,
                response_format="json",
                think=False,
            )
        except HTTPException as exc:
            if exc.status_code == 503 and not is_ollama_timeout_error(exc):
                raise
            if is_ollama_timeout_error(exc):
                metrics["timeout_failures"] += 1
            # Split and retry smaller groups.
            mid = max(1, len(batch) // 2)
            await _run(batch[:mid])
            await _run(batch[mid:])
            return

        model_used = generation.get("model", SEED_GENERATION_MODEL)
        expected_ids = [str(item["fact"].get("factId") or "") for item in batch]
        try:
            raw_seeds = parse_batch_seed_payload(
                generation["answer"],
                expected_fact_ids=expected_ids,
            )
        except Exception:
            raw_seeds = []

        if not raw_seeds:
            mid = max(1, len(batch) // 2)
            await _run(batch[:mid])
            await _run(batch[mid:])
            return

        seeds_by_fact = {str(seed.get("factId") or ""): seed for seed in raw_seeds}
        missing = [
            item
            for item in batch
            if str(item["fact"].get("factId") or "") not in seeds_by_fact
        ]
        for item in batch:
            fact = item["fact"]
            fact_id = str(fact.get("factId") or "")
            raw_seed = seeds_by_fact.get(fact_id)
            if raw_seed is None:
                continue
            normalized = _normalize_fact_seed(
                raw_seed=raw_seed,
                fact=fact,
                index=0,
                suggested_styles=list(item.get("suggested_styles") or []),
                count=1,
            )
            if normalized is not None:
                results[fact_id] = [normalized]

        # Only re-split for facts that produced nothing, not the whole batch.
        if missing and len(missing) < len(batch):
            await _run(missing)
        elif missing:
            mid = max(1, len(missing) // 2)
            await _run(missing[:mid])
            await _run(missing[mid:])

    await _run(items)
    return results, model_used, metrics


async def _validate_candidates_batch(
    *,
    candidates: list[dict[str, Any]],
    completion_fn=None,
) -> tuple[dict[str, dict[str, Any] | None], dict[str, int]]:
    """Unused by live starter generation (Phase 6 disabled). Kept for unit tests."""
    run_completion = completion_fn or generate_starter_ollama_completion
    metrics = {
        "validation_calls": 0,
        "validation_batch_calls": 0,
        "timeout_failures": 0,
        "max_validation_batch_size": 0,
    }
    results: dict[str, dict[str, Any] | None] = {
        str(item.get("candidateId") or ""): None for item in candidates
    }

    async def _run(batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        metrics["max_validation_batch_size"] = max(
            metrics["max_validation_batch_size"], len(batch)
        )
        expected_ids = [str(item.get("candidateId") or "") for item in batch]

        if len(batch) == 1:
            item = batch[0]
            candidate_id = expected_ids[0]
            metrics["validation_calls"] += 1
            try:
                validation = await _validate_candidate(
                    question=str(item.get("question") or ""),
                    answer=str(item.get("answer") or ""),
                    topic_name=str(item.get("topic_name") or "General"),
                    question_type=str(item.get("question_type") or "direct"),
                    chunk_text=str(item.get("chunk_text") or ""),
                    completion_fn=run_completion,
                )
            except HTTPException as exc:
                if is_ollama_timeout_error(exc):
                    metrics["timeout_failures"] += 1
                    results[candidate_id] = None
                    return
                if exc.status_code == 503:
                    raise
                results[candidate_id] = None
                return
            results[candidate_id] = validation
            return

        metrics["validation_calls"] += 1
        metrics["validation_batch_calls"] += 1
        prompt = build_batch_validation_prompt(candidates=batch)
        try:
            generation = await run_completion(
                prompt,
                model=SEED_GENERATION_MODEL,
                response_format="json",
                think=False,
            )
        except HTTPException as exc:
            if exc.status_code == 503 and not is_ollama_timeout_error(exc):
                raise
            if is_ollama_timeout_error(exc):
                metrics["timeout_failures"] += 1
            mid = max(1, len(batch) // 2)
            await _run(batch[:mid])
            await _run(batch[mid:])
            return

        mapped = parse_batch_validation_payload(
            generation["answer"],
            expected_candidate_ids=expected_ids,
        )
        if mapped is None:
            mid = max(1, len(batch) // 2)
            await _run(batch[:mid])
            await _run(batch[mid:])
            return

        for candidate_id, value in mapped.items():
            results[candidate_id] = value

    await _run(candidates)
    return results, metrics


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
    generation = await generate_starter_ollama_completion(
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
    generation = await generate_starter_ollama_completion(
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
    completion_fn=None,
) -> dict[str, Any] | None:
    run_completion = completion_fn or generate_starter_ollama_completion
    prompt = build_validation_prompt(
        question=question,
        answer=answer,
        topic_name=topic_name,
        question_type=question_type,
        chunk_text=chunk_text,
    )
    generation = await run_completion(
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
    force_refresh: bool = False,
    storage: CourseArtifactStorage | None = None,
) -> dict[str, Any]:
    """Generate up to target_count starter seeds via fact inventory + allocation.

    Live path (Phase 4 / 4.5 / 5):
      chunks → cached-or-fresh fact inventory → breadth-first allocation
      → sequential fact-scoped Q/A with rejection backfill
      → programmatic checks → exact/semantic dedupe → pre-validation
      → LLM validation → accept

    Phase 6 batching helpers exist but are not used on this live path.
    Chunks are evidence only. Does not use topic planning. When save=True,
    persists accepted validated seeds to Firebase. Does not trigger course creation.
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

    started_at = time.perf_counter()

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
    embedding_cache = AcceptedEmbeddingCache()
    chunks_processed = 0
    generation_calls = 0
    validation_calls = 0
    planning_calls = 0  # Unused on fact pipeline; kept for schema compatibility.
    merge_calls = 0  # Unused on fact pipeline; kept for schema compatibility.
    fact_extraction_calls = 0
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
    candidates_rejected_prevalidation = 0
    candidates_rejected_qualifier_mismatch = 0
    candidates_rejected_modal_escalation = 0
    duplicates_removed = 0
    semantic_duplicates_removed = 0
    timeout_failures = 0
    backfill_attempts = 0
    backfill_accepted = 0
    model_used = SEED_GENERATION_MODEL
    max_total_calls = get_starter_max_total_ollama_calls()

    def _budget_remaining() -> bool:
        return ollama_calls < max_total_calls

    async def _counted_completion(prompt: str, **kwargs: Any) -> dict[str, str]:
        nonlocal ollama_calls, fact_extraction_calls
        ollama_calls += 1
        if FACT_EXTRACTION_PROMPT_MARKER in prompt:
            fact_extraction_calls += 1
        return await generate_starter_ollama_completion(prompt, **kwargs)

    async def _counted_embed(texts: list[str], **kwargs: Any) -> dict[str, Any]:
        nonlocal embedding_calls
        embedding_calls += 1
        return await embed_ollama_texts(texts, **kwargs)

    inventory = await load_or_build_fact_inventory(
        course_id=safe_course_id,
        raw_chunks=raw_chunks,
        storage=artifact_storage,
        force_refresh=force_refresh,
        completion_fn=_counted_completion,
        embed_fn=_counted_embed,
    )
    fact_inventory_cached = bool(inventory.get("cached"))
    facts = list(inventory.get("facts") or [])
    fact_count = int(inventory.get("factCount") or len(facts))
    if inventory.get("model"):
        model_used = str(inventory["model"])

    allocation = allocate_slots(facts, target_count=target_count)
    facts_by_id = {str(fact.get("factId") or ""): fact for fact in facts}

    # Unit opportunities: one generation attempt per allocated slot.
    opportunities: list[tuple[dict[str, Any], dict[str, Any], bool]] = []
    primary_fact_ids: set[str] = set()
    for alloc in allocation.get("allocations", []):
        slot_count = int(alloc.get("slotCount") or 0)
        if slot_count <= 0:
            continue
        fact = facts_by_id.get(str(alloc.get("factId") or ""))
        if fact is None:
            continue
        primary_fact_ids.add(str(fact.get("factId") or ""))
        for _ in range(slot_count):
            opportunities.append((fact, alloc, False))

    # Backfill pool: next-best eligible facts that received 0 slots.
    for alloc in allocation.get("allocations", []):
        if int(alloc.get("slotCount") or 0) > 0:
            continue
        if int(alloc.get("desiredSlots") or 0) <= 0:
            continue
        fact_id = str(alloc.get("factId") or "")
        if fact_id in primary_fact_ids:
            continue
        fact = facts_by_id.get(fact_id)
        if fact is None:
            continue
        backfill_alloc = {
            **alloc,
            "slotCount": 1,
            "suggestedStyles": list(alloc.get("suggestedStyles") or ["factual"]),
        }
        opportunities.append((fact, backfill_alloc, True))

    allocated_fact_count = len(primary_fact_ids)
    allocated_slots = int(allocation.get("summary", {}).get("allocatedSlots") or 0)
    selected_chunk_ids = {
        str(chunk_id).strip()
        for fact_id in primary_fact_ids
        for chunk_id in (facts_by_id.get(fact_id, {}).get("sourceChunkIds") or [])
        if str(chunk_id).strip()
    }

    failed_facts: set[str] = set()

    for fact, alloc, is_backfill in opportunities:
        if len(seeds) >= target_count or not _budget_remaining():
            break

        fact_id = str(fact.get("factId") or "")
        if fact_id in failed_facts:
            continue

        if is_backfill:
            backfill_attempts += 1

        fact_chunk_ids = [
            str(item).strip()
            for item in (fact.get("sourceChunkIds") or [])
            if str(item).strip()
        ]
        fact_chunk_texts: list[str] = []
        for chunk_id in fact_chunk_ids:
            raw_chunk = chunk_lookup.get(chunk_id)
            text = raw_chunk.get("text") if isinstance(raw_chunk, dict) else None
            if isinstance(text, str) and text.strip():
                fact_chunk_texts.append(text.strip())
        if not fact_chunk_texts:
            continue

        chunks_processed += len(fact_chunk_texts)
        ollama_calls += 1
        generation_calls += 1
        accepted_before = len(seeds)
        try:
            candidates, model_used = await _generate_candidates_for_fact(
                fact=fact,
                chunk_texts=fact_chunk_texts,
                count=1,
                suggested_styles=list(alloc.get("suggestedStyles") or []),
            )
        except HTTPException as exc:
            failed_facts.add(fact_id)
            if is_ollama_timeout_error(exc):
                timeout_failures += 1
                continue
            if exc.status_code == 503:
                raise
            continue

        candidates_generated += len(candidates)
        if not candidates:
            failed_facts.add(fact_id)
            continue

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
                        cache=embedding_cache,
                    )
                except HTTPException:
                    embedding_cache.last_candidate_embedding = None
                    semantic_duplicate = None
                if semantic_duplicate is not None:
                    semantic_duplicates_removed += 1
                    candidates_rejected += 1
                    continue

            # Deterministic grounding/escalation checks before LLM validation.
            prevalidation = prevalidate_candidate(
                candidate=candidate,
                fact=fact,
                source_text="\n\n".join(fact_chunk_texts),
            )
            if prevalidation is not None:
                candidates_rejected += 1
                candidates_rejected_prevalidation += 1
                category = prevalidation.get("category")
                if category == "modal_escalation":
                    candidates_rejected_modal_escalation += 1
                elif category == "qualifier_mismatch":
                    candidates_rejected_qualifier_mismatch += 1
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
                    chunk_text="\n\n".join(fact_chunk_texts),
                )
            except HTTPException as exc:
                if is_ollama_timeout_error(exc):
                    timeout_failures += 1
                    candidates_validated += 1
                    candidates_rejected += 1
                    candidates_rejected_validation += 1
                    continue
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
            embedding_cache.remember_last_candidate()
            candidates_accepted += 1
            if is_backfill:
                backfill_accepted += 1
            seeds.append(
                {
                    **candidate,
                    "factId": str(
                        candidate.get("factId") or fact.get("factId") or ""
                    ),
                    "evidenceQuote": str(
                        candidate.get("evidenceQuote")
                        or fact.get("evidenceQuote")
                        or ""
                    ),
                    "sourceChunkIds": list(
                        candidate.get("sourceChunkIds") or fact_chunk_ids
                    ),
                    "validation": validation,
                }
            )

        if len(seeds) == accepted_before:
            # No accept from this attempt — do not keep regenerating this fact.
            failed_facts.add(fact_id)

    saved_count = 0
    persistence: dict[str, Any] | None = None
    if save:
        try:
            chunk_sections = build_chunk_section_lookup(raw_chunks)
            persistence = await persist_accepted_seeds(
                course_id=safe_course_id,
                seeds=seeds,
                chunk_sections=chunk_sections,
            )
            saved_count = int(persistence.get("savedCount", 0))
        except FirebaseConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    elapsed_ms = max(0, int(round((time.perf_counter() - started_at) * 1000)))
    final_count = len(seeds)
    run_status = resolve_starter_run_status(
        target_count=target_count,
        final_count=final_count,
        saved_count=saved_count,
        save=save,
    )

    result: dict[str, Any] = {
        "courseId": safe_course_id,
        "model": model_used,
        "targetCount": target_count,
        "seeds": seeds,
        "progress": {
            "eligibleChunks": len(eligible),
            "selectedChunks": len(selected_chunk_ids),
            "chunksProcessed": chunks_processed,
            "chunksSkipped": chunks_skipped,
            "planningCalls": planning_calls,
            "mergeCalls": merge_calls,
            "factExtractionCalls": fact_extraction_calls,
            "factInventoryCached": fact_inventory_cached,
            "factCount": fact_count,
            "allocatedFactCount": allocated_fact_count,
            "allocatedSlots": allocated_slots,
            "backfillAttempts": backfill_attempts,
            "backfillAccepted": backfill_accepted,
            "generationCalls": generation_calls,
            # Phase 6 batching disabled on the live path; retained as zeroed metrics.
            "generationBatchCalls": 0,
            "maxGenerationBatchSize": 0,
            "validationCalls": validation_calls,
            "validationBatchCalls": 0,
            "maxValidationBatchSize": 0,
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
            "candidatesRejectedPreValidation": candidates_rejected_prevalidation,
            "candidatesRejectedQualifierMismatch": candidates_rejected_qualifier_mismatch,
            "candidatesRejectedModalEscalation": candidates_rejected_modal_escalation,
            "scheduleCount": count_schedule_like(seeds),
            "scenarioOrClarificationMinimum": compute_scenario_minimum(target_count),
            "timeoutFailures": timeout_failures,
            "finalCount": final_count,
            "savedCount": saved_count,
            "elapsedMs": elapsed_ms,
            "status": run_status,
        },
    }

    if final_count == 0 and timeout_failures > 0:
        raise HTTPException(
            status_code=503,
            detail=(
                "Starter seed generation timed out after retries. "
                "The syllabus and RAG index remain available."
            ),
        )

    if persistence is not None:
        result["persistence"] = persistence

    # Local artifact for review/export (does not replace Firebase course path).
    if seeds:
        snapshot_path = write_generation_snapshot(
            course_id=safe_course_id,
            seeds=seeds,
            progress=result["progress"],
        )
        result["localSnapshotPath"] = str(snapshot_path)

    return result
