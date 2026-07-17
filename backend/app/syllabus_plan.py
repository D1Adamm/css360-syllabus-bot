"""Syllabus-topic planning helpers for starter seed generation."""

from __future__ import annotations

import json
import re
from typing import Any

from fastapi import HTTPException

from app.ollama import generate_ollama_completion

SEED_GENERATION_MODEL = "qwen3:4b"

DEFAULT_PLANNING_BATCH_SIZE = 18
MIN_PLANNING_BATCH_SIZE = 15
MAX_PLANNING_BATCH_SIZE = 20
DEFAULT_DIGEST_TEXT_CHARS = 320
MIN_DIGEST_TEXT_CHARS = 250
MAX_DIGEST_TEXT_CHARS = 350

PLANNER_PROMPT_MARKER = "You plan course-specific syllabus starter-seed topics."
PLANNER_MERGE_PROMPT_MARKER = "You merge syllabus starter-seed topics."

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip())


def _extract_json_text(raw: str) -> str:
    text = raw.strip()
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()

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


def normalize_topic_name(name: str) -> str:
    lowered = name.strip().lower()
    collapsed = _NON_ALNUM_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", collapsed).strip()


def chunk_digest_from_chunk(
    raw_chunk: dict[str, Any],
    *,
    text_chars: int = DEFAULT_DIGEST_TEXT_CHARS,
) -> dict[str, str] | None:
    chunk_id = raw_chunk.get("chunkId") or raw_chunk.get("id")
    section_title = raw_chunk.get("sectionTitle") or raw_chunk.get("section_title") or "General"
    text = raw_chunk.get("text")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        return None
    if not isinstance(text, str) or not text.strip():
        return None

    normalized_text = _normalize_whitespace(text)
    clipped_text = normalized_text[: max(MIN_DIGEST_TEXT_CHARS, min(text_chars, MAX_DIGEST_TEXT_CHARS))]
    return {
        "chunkId": chunk_id.strip(),
        "sectionTitle": str(section_title).strip() or "General",
        "textDigest": clipped_text,
    }


def build_chunk_digests(
    raw_chunks: list[Any],
    *,
    text_chars: int = DEFAULT_DIGEST_TEXT_CHARS,
) -> list[dict[str, str]]:
    digests: list[dict[str, str]] = []
    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            continue
        digest = chunk_digest_from_chunk(raw_chunk, text_chars=text_chars)
        if digest is not None:
            digests.append(digest)
    return digests


def batch_chunk_digests(
    digests: list[dict[str, str]],
    *,
    batch_size: int = DEFAULT_PLANNING_BATCH_SIZE,
) -> list[list[dict[str, str]]]:
    safe_size = min(MAX_PLANNING_BATCH_SIZE, max(MIN_PLANNING_BATCH_SIZE, batch_size))
    if not digests:
        return []
    return [digests[i : i + safe_size] for i in range(0, len(digests), safe_size)]


def topic_is_schedule_heavy(name: str, summary: str = "") -> bool:
    text = f"{name} {summary}".lower()
    if "schedule" in text or "calendar" in text:
        return True
    schedule_terms = (
        "deadline",
        "deadlines",
        "date",
        "dates",
        "time",
        "times",
        "due",
        "midterm",
        "final exam",
        "class meeting",
        "office hours",
        "weekly meeting",
        "late policy",
    )
    return any(term in text for term in schedule_terms)


def _build_planner_prompt(
    digests: list[dict[str, str]],
    *,
    target_count: int,
) -> str:
    digest_lines = []
    for digest in digests:
        digest_lines.append(
            f'- {digest["chunkId"]} | {digest["sectionTitle"]} | {digest["textDigest"]}'
        )
    digest_block = "\n".join(digest_lines)
    return f"""{PLANNER_PROMPT_MARKER}

Inspect these syllabus chunk digests and propose course-specific starter-seed topics.

Rules:
- Use only the provided digest content.
- Do not invent fixed generic course categories.
- Prefer topics that would produce useful student questions.
- Mark major dates/times/deadlines/direct links as scheduleHeavy only when that is the primary focus.
- Return only valid JSON.

Target starter seed count: {target_count}

Return JSON in this shape:
{{
  "topics": [
    {{
      "name": "string",
      "importance": "high|medium|low",
      "sourceChunkIds": ["chunk-001"],
      "suggestedExampleCount": 3,
      "summary": "short summary",
      "scheduleHeavy": false
    }}
  ]
}}

Chunk digests:
{digest_block}
"""


def _build_merge_prompt(
    topics: list[dict[str, Any]],
    *,
    target_count: int,
) -> str:
    serialized_topics = json.dumps({"topics": topics}, indent=2)
    return f"""{PLANNER_MERGE_PROMPT_MARKER}

Merge overlapping syllabus topic candidates into a concise balanced plan.

Rules:
- Merge near-duplicates.
- Keep course-specific topic names.
- Preserve and union sourceChunkIds.
- Keep scheduleHeavy true only when dates/times/deadlines/links are the primary focus.
- SuggestedExampleCount values should add up close to {target_count}.
- Return only valid JSON.

Required JSON shape:
{{
  "topics": [
    {{
      "name": "string",
      "importance": "high|medium|low",
      "sourceChunkIds": ["chunk-001"],
      "suggestedExampleCount": 3,
      "summary": "short summary",
      "scheduleHeavy": false
    }}
  ]
}}

Topic candidates:
{serialized_topics}
"""


def _coerce_importance(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered in {"high", "medium", "low"}:
        return lowered
    return None


def _parse_topics_payload(raw: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(_extract_json_text(raw))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned malformed JSON for syllabus topic planning.",
        ) from exc

    if isinstance(parsed, dict):
        topics = parsed.get("topics")
    elif isinstance(parsed, list):
        topics = parsed
    else:
        topics = None

    if not isinstance(topics, list):
        raise HTTPException(
            status_code=502,
            detail="Ollama topic plan response must include a topics array.",
        )
    return topics


def _normalize_topic(raw_topic: Any, *, index: int) -> dict[str, Any] | None:
    if not isinstance(raw_topic, dict):
        return None

    name = str(raw_topic.get("name", "")).strip()
    importance = _coerce_importance(raw_topic.get("importance"))
    source_chunk_ids = raw_topic.get("sourceChunkIds")
    suggested_count = raw_topic.get("suggestedExampleCount")
    summary = str(raw_topic.get("summary", "")).strip()
    schedule_heavy = raw_topic.get("scheduleHeavy")

    if not name or importance is None or not isinstance(source_chunk_ids, list):
        return None

    cleaned_chunk_ids = sorted({str(item).strip() for item in source_chunk_ids if str(item).strip()})
    if not cleaned_chunk_ids:
        return None

    try:
        count = int(suggested_count)
    except (TypeError, ValueError):
        return None

    if count < 1:
        return None

    if not isinstance(schedule_heavy, bool):
        schedule_heavy = topic_is_schedule_heavy(name, summary)

    return {
        "topicId": f"topic-{index + 1:02d}",
        "name": name,
        "importance": importance,
        "sourceChunkIds": cleaned_chunk_ids,
        "suggestedExampleCount": count,
        "summary": summary,
        "scheduleHeavy": schedule_heavy,
    }


def deterministic_merge_topics(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}

    for topic in topics:
        normalized_name = normalize_topic_name(str(topic.get("name", "")))
        chunk_ids = tuple(sorted(topic.get("sourceChunkIds", [])))
        if not normalized_name or not chunk_ids:
            continue
        key = (normalized_name, chunk_ids)
        existing = merged.get(key)
        if existing is None:
            merged[key] = {
                **topic,
                "sourceChunkIds": list(chunk_ids),
            }
            continue

        importance_order = {"high": 3, "medium": 2, "low": 1}
        if importance_order.get(topic.get("importance", "low"), 1) > importance_order.get(
            existing.get("importance", "low"), 1
        ):
            existing["importance"] = topic["importance"]
        existing["suggestedExampleCount"] = max(
            int(existing.get("suggestedExampleCount", 1)),
            int(topic.get("suggestedExampleCount", 1)),
        )
        if len(str(topic.get("summary", "")).strip()) > len(str(existing.get("summary", "")).strip()):
            existing["summary"] = topic.get("summary", "")
        existing["scheduleHeavy"] = bool(existing.get("scheduleHeavy")) or bool(
            topic.get("scheduleHeavy")
        )

    ordered = list(merged.values())
    ordered.sort(key=lambda item: (-len(item["sourceChunkIds"]), item["name"].lower()))
    for index, topic in enumerate(ordered):
        topic["topicId"] = f"topic-{index + 1:02d}"
    return ordered


def merge_topics_by_overlap(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for topic in topics:
        normalized_name = normalize_topic_name(str(topic.get("name", "")))
        if not normalized_name:
            continue
        existing = by_name.get(normalized_name)
        if existing is None:
            by_name[normalized_name] = {
                **topic,
                "sourceChunkIds": list(sorted(set(topic.get("sourceChunkIds", [])))),
            }
            continue

        overlap = set(existing["sourceChunkIds"]) & set(topic.get("sourceChunkIds", []))
        if overlap or normalized_name == normalize_topic_name(existing["name"]):
            combined_ids = sorted(
                set(existing["sourceChunkIds"]) | set(topic.get("sourceChunkIds", []))
            )
            existing["sourceChunkIds"] = combined_ids
            existing["suggestedExampleCount"] = max(
                int(existing.get("suggestedExampleCount", 1)),
                int(topic.get("suggestedExampleCount", 1)),
            )
            if topic.get("importance") == "high":
                existing["importance"] = "high"
            elif (
                topic.get("importance") == "medium"
                and existing.get("importance") == "low"
            ):
                existing["importance"] = "medium"
            if len(str(topic.get("summary", "")).strip()) > len(str(existing.get("summary", "")).strip()):
                existing["summary"] = topic.get("summary", "")
            existing["scheduleHeavy"] = bool(existing.get("scheduleHeavy")) or bool(
                topic.get("scheduleHeavy")
            )
        else:
            by_name[f"{normalized_name}|{len(by_name)}"] = topic

    topics_list = list(by_name.values())
    topics_list.sort(key=lambda item: (-len(item.get("sourceChunkIds", [])), item["name"].lower()))
    for index, topic in enumerate(topics_list):
        topic["topicId"] = f"topic-{index + 1:02d}"
    return topics_list


def build_section_title_fallback_plan(
    raw_chunks: list[Any],
    *,
    target_count: int,
) -> dict[str, Any]:
    grouped: dict[str, list[str]] = {}
    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            continue
        chunk_id = raw_chunk.get("chunkId") or raw_chunk.get("id")
        section_title = raw_chunk.get("sectionTitle") or raw_chunk.get("section_title") or "General"
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            continue
        key = str(section_title).strip() or "General"
        grouped.setdefault(key, []).append(chunk_id.strip())

    topic_names = sorted(grouped)
    if not topic_names:
        return {"topics": []}

    base_count = max(1, target_count // len(topic_names))
    topics = []
    for index, name in enumerate(topic_names):
        topics.append(
            {
                "topicId": f"topic-{index + 1:02d}",
                "name": name,
                "importance": "medium",
                "sourceChunkIds": grouped[name],
                "suggestedExampleCount": base_count,
                "summary": name,
                "scheduleHeavy": topic_is_schedule_heavy(name),
            }
        )
    return {"topics": topics}


async def plan_syllabus_topics(
    *,
    raw_chunks: list[Any],
    target_count: int,
    batch_size: int = DEFAULT_PLANNING_BATCH_SIZE,
    completion_fn=None,
) -> dict[str, Any]:
    if completion_fn is None:
        completion_fn = generate_ollama_completion
    digests = build_chunk_digests(raw_chunks)
    if not digests:
        return {"topics": []}

    topic_candidates: list[dict[str, Any]] = []
    for batch in batch_chunk_digests(digests, batch_size=batch_size):
        prompt = _build_planner_prompt(batch, target_count=target_count)
        generation = await completion_fn(
            prompt,
            model=SEED_GENERATION_MODEL,
            response_format="json",
            think=False,
        )
        raw_topics = _parse_topics_payload(generation["answer"])
        for index, raw_topic in enumerate(raw_topics):
            normalized = _normalize_topic(raw_topic, index=index + len(topic_candidates))
            if normalized is not None:
                topic_candidates.append(normalized)

    merged_candidates = merge_topics_by_overlap(deterministic_merge_topics(topic_candidates))
    if not merged_candidates:
        return build_section_title_fallback_plan(raw_chunks, target_count=target_count)

    merge_prompt = _build_merge_prompt(merged_candidates, target_count=target_count)
    try:
        merge_generation = await completion_fn(
            merge_prompt,
            model=SEED_GENERATION_MODEL,
            response_format="json",
            think=False,
        )
        merged_raw_topics = _parse_topics_payload(merge_generation["answer"])
        merged_topics = [
            normalized
            for index, raw_topic in enumerate(merged_raw_topics)
            if (normalized := _normalize_topic(raw_topic, index=index)) is not None
        ]
    except HTTPException:
        merged_topics = merged_candidates

    final_topics = merge_topics_by_overlap(deterministic_merge_topics(merged_topics))
    if not final_topics:
        return build_section_title_fallback_plan(raw_chunks, target_count=target_count)
    return {"topics": final_topics}
