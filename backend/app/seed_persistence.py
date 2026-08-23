"""Persist and read course seed examples in PostgreSQL.

This is the seed half of the storage layer the generator, the review routes and
the training export all sit on. It replaces the Realtime Database helpers that
used to own the same job, and the change is not only where the bytes land: a
run that saves forty seeds now either saves them or does not, instead of
leaving whatever number of individual REST writes happened to succeed.

Two kinds of thing live here, and the split is deliberate:

  - Pure helpers — `derive_source_section`, `build_seed_record`,
    `summarize_existing_seed_examples`, `compute_top_up_gap`. They describe what
    a seed record is and what a top-up still owes, and they touch no storage.
    The generator uses them long before anything is written.
  - Storage functions — `fetch_course_seed_examples`, `persist_accepted_seeds`.
    They open one connection and hand the work to `db_seeds`, so seeds written
    by generation and seeds written through `/api/db` go through exactly the
    same repository and land identically.

The read still returns a mapping keyed by seed id rather than a list. That is
the shape the generator's dedupe and top-up planning already consume, and it is
the shape the review and export paths expect; keeping it means this module
changed store without changing anyone's parsing.

Async signatures are kept even though psycopg is synchronous here. The callers
are async generation paths that awaited the old helpers, and the queries are
single-row-per-seed statements against a local database — turning the whole
generation pipeline inside out to gain nothing measurable would have been a far
larger change than the migration itself.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from app.course_id import assert_valid_course_id
from app.db import db_connection, translate_db_errors
from app.db_courses import course_exists
from app.db_seeds import create_seed, list_seeds
from app.seed_dedupe import normalize_question_for_dedupe
from app.seed_validation import canonicalize_validation_result


def derive_source_section(
    source_chunk_ids: list[str],
    chunk_sections: dict[str, str],
) -> str:
    titles: list[str] = []
    seen_titles: set[str] = set()
    for chunk_id in source_chunk_ids:
        title = chunk_sections.get(chunk_id, "").strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            titles.append(title)
    if titles:
        return titles[0]
    if source_chunk_ids:
        return ", ".join(source_chunk_ids)
    return "General"


def build_chunk_section_lookup(raw_chunks: list[Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            continue
        chunk_id = raw_chunk.get("chunkId") or raw_chunk.get("id")
        section_title = raw_chunk.get("sectionTitle") or raw_chunk.get("section_title")
        if isinstance(chunk_id, str) and chunk_id.strip():
            lookup[chunk_id.strip()] = (
                str(section_title).strip() if isinstance(section_title, str) else ""
            )
    return lookup


def build_seed_record(
    seed: dict[str, Any],
    *,
    source_section: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """One generated seed as the stored record.

    Both name pairs are written — `question`/`instruction` and
    `answer`/`response`. The repository maps them onto one column each, but the
    review helper and the frontend both accept either, and emitting both is
    what lets a record round-trip through any of them unchanged.
    """
    question = str(seed["question"]).strip()
    answer = str(seed["answer"]).strip()
    normalized_key = normalize_question_for_dedupe(question)
    timestamp = created_at or datetime.now(timezone.utc).isoformat()

    validation = seed.get("validation")
    normalized_validation = None
    if isinstance(validation, dict):
        normalized_validation = canonicalize_validation_result(validation)

    return {
        "question": question,
        "instruction": question,
        "answer": answer,
        "response": answer,
        "category": str(seed.get("category", "general")).strip() or "general",
        "questionType": str(seed.get("questionType", "direct")).strip() or "direct",
        "sourceChunkIds": list(seed.get("sourceChunkIds", [])),
        "sourceSection": source_section,
        "difficulty": "Medium",
        "directlyAnswered": True,
        "origin": "ai_generated",
        # Legacy generation badge + review field (not human-approved yet).
        "status": "generated",
        "reviewStatus": "generated",
        "normalizedQuestionKey": normalized_key,
        "createdAt": timestamp,
        "validation": normalized_validation,
        "factId": (
            str(seed["factId"]).strip()
            if isinstance(seed.get("factId"), str) and str(seed.get("factId")).strip()
            else None
        ),
        "evidenceQuote": (
            str(seed["evidenceQuote"]).strip()
            if isinstance(seed.get("evidenceQuote"), str)
            and str(seed.get("evidenceQuote")).strip()
            else None
        ),
    }


def collect_normalized_question_keys(existing_seeds: dict[str, Any] | None) -> set[str]:
    keys: set[str] = set()
    if not existing_seeds:
        return keys

    for raw_seed in existing_seeds.values():
        if not isinstance(raw_seed, dict):
            continue
        stored_key = raw_seed.get("normalizedQuestionKey")
        if isinstance(stored_key, str) and stored_key.strip():
            keys.add(stored_key.strip())
        question = raw_seed.get("question") or raw_seed.get("instruction")
        if isinstance(question, str) and question.strip():
            keys.add(normalize_question_for_dedupe(question))
    return keys


def summarize_existing_seed_examples(
    existing_seeds: dict[str, Any] | None,
) -> dict[str, Any]:
    """Summarize stored seeds for top-up planning and dedupe preload."""
    count = 0
    questions: list[str] = []
    seen_keys: set[str] = set()
    fact_ids: set[str] = set()

    if not existing_seeds:
        return {
            "existingCount": 0,
            "questions": questions,
            "seenQuestionKeys": seen_keys,
            "factIds": fact_ids,
        }

    for raw_seed in existing_seeds.values():
        if not isinstance(raw_seed, dict):
            continue
        count += 1
        question = raw_seed.get("question") or raw_seed.get("instruction")
        if isinstance(question, str) and question.strip():
            cleaned = question.strip()
            questions.append(cleaned)
            seen_keys.add(normalize_question_for_dedupe(cleaned))
        stored_key = raw_seed.get("normalizedQuestionKey")
        if isinstance(stored_key, str) and stored_key.strip():
            seen_keys.add(stored_key.strip())
        fact_id = raw_seed.get("factId")
        if isinstance(fact_id, str) and fact_id.strip():
            fact_ids.add(fact_id.strip())

    return {
        "existingCount": count,
        "questions": questions,
        "seenQuestionKeys": seen_keys,
        "factIds": fact_ids,
    }


def compute_top_up_gap(
    *, existing_count: int, target_count: int
) -> dict[str, int | bool]:
    """Return how many new seeds are needed to reach target_count."""
    safe_existing = max(0, int(existing_count))
    safe_target = max(0, int(target_count))
    missing = max(0, safe_target - safe_existing)
    return {
        "existingCount": safe_existing,
        "targetCount": safe_target,
        "missingCount": missing,
        "alreadyComplete": missing == 0,
    }


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


async def fetch_course_seed_examples(course_id: str) -> dict[str, Any]:
    """Every stored seed for one course, keyed by seed id.

    Course-scoped by the repository, not by a path string, so a seed belonging
    to another course cannot be returned even if its id were guessed.
    """
    safe_course_id = assert_valid_course_id(course_id)

    with translate_db_errors("loading existing seed examples"):
        with db_connection() as connection:
            seeds = list_seeds(connection, safe_course_id)

    return {seed["id"]: seed for seed in seeds}


async def count_course_seed_examples(course_id: str) -> int | None:
    """How many seeds the course holds, or None when it could not be read.

    None is not zero, and the difference matters: a failed read must leave the
    caller free to keep the counts it already had rather than overwrite a true
    record with a false zero.
    """
    try:
        payload = await fetch_course_seed_examples(course_id)
    except (HTTPException, ValueError):
        return None
    return len(payload)


async def persist_accepted_seeds(
    *,
    course_id: str,
    seeds: list[dict[str, Any]],
    chunk_sections: dict[str, str],
) -> dict[str, int]:
    """Save accepted validated seeds for one course.

    One connection and one transaction for the whole batch, with a savepoint
    per seed. The savepoints are what keep the old per-seed accounting honest:
    a single unwritable record is counted in `failedToSaveCount` and the other
    thirty-nine still land, while a failure of the connection itself aborts
    everything rather than leaving a half-saved run behind.

    Deduplication is against what is already stored, not against this batch
    alone, so a top-up cannot reintroduce a question the course already has.
    """
    generated_count = len(seeds)
    empty = {
        "generatedCount": generated_count,
        "savedCount": 0,
        "alreadyExistingCount": 0,
        "failedToSaveCount": 0,
    }
    if generated_count == 0:
        return empty

    safe_course_id = assert_valid_course_id(course_id)

    saved_count = 0
    already_existing_count = 0
    failed_to_save_count = 0

    with translate_db_errors("saving seed examples"):
        with db_connection() as connection:
            if not course_exists(connection, safe_course_id):
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f'Course "{safe_course_id}" was not found, so its seeds '
                        "cannot be saved."
                    ),
                )

            existing = {
                seed["id"]: seed for seed in list_seeds(connection, safe_course_id)
            }
            seen_keys = collect_normalized_question_keys(existing)

            for seed in seeds:
                normalized_key = normalize_question_for_dedupe(
                    str(seed.get("question", ""))
                )
                if not normalized_key:
                    failed_to_save_count += 1
                    continue
                if normalized_key in seen_keys:
                    already_existing_count += 1
                    continue

                source_chunk_ids = seed.get("sourceChunkIds", [])
                if not isinstance(source_chunk_ids, list):
                    source_chunk_ids = []
                source_section = derive_source_section(
                    [str(item).strip() for item in source_chunk_ids if str(item).strip()],
                    chunk_sections,
                )
                record = build_seed_record(seed, source_section=source_section)

                try:
                    # A savepoint: one malformed record must not abort the
                    # transaction the other seeds are being written in.
                    with connection.transaction():
                        create_seed(connection, safe_course_id, record)
                except Exception:  # noqa: BLE001 - counted, not swallowed silently
                    failed_to_save_count += 1
                    continue

                seen_keys.add(normalized_key)
                saved_count += 1

    return {
        "generatedCount": generated_count,
        "savedCount": saved_count,
        "alreadyExistingCount": already_existing_count,
        "failedToSaveCount": failed_to_save_count,
    }
