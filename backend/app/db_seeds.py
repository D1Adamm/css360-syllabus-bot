"""PostgreSQL repository for course seed examples.

Records come back carrying both name pairs — `instruction`/`question` and
`response`/`answer` — because both are load-bearing today. `SeedExample` in the
frontend uses instruction/response, `normalizeSeedExample` accepts either, and
`apply_seed_review` reads and writes all four. Emitting both here means review,
export, and the seed table all read the same record without a translation step
in between.

Every function is course-scoped: `seed_examples` is keyed `(course_id, seed_id)`
and every statement below binds both, so no course can read or delete another's
seed even if it guesses an id.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

from app.course_id import assert_valid_course_id
from app.db_mapping import (
    as_bool,
    as_int,
    build_patch,
    optional_string,
    put_optional,
    string_list,
    to_iso,
    update_statement,
)
from app.seed_review import apply_seed_review

SEED_COLUMNS = """
    seed_id, course_id, instruction, response, category, source_section,
    difficulty, directly_answered, origin, notes, created_at, status,
    question_type, source_chunk_ids, validation, review_status, review_notes,
    reviewed_at, fact_id, evidence_quote, normalized_question_key,
    original_question, original_answer, was_edited
"""

# camelCase API field -> column, and the allowlist for patches. The dual names
# both land on the same column, so a client may send either.
SEED_PATCH_COLUMNS = {
    "instruction": "instruction",
    "question": "instruction",
    "response": "response",
    "answer": "response",
    "category": "category",
    "sourceSection": "source_section",
    "difficulty": "difficulty",
    "directlyAnswered": "directly_answered",
    "origin": "origin",
    "notes": "notes",
    "status": "status",
    "questionType": "question_type",
    "sourceChunkIds": "source_chunk_ids",
    "validation": "validation",
    "reviewStatus": "review_status",
    "reviewNotes": "review_notes",
    "reviewedAt": "reviewed_at",
    "factId": "fact_id",
    "evidenceQuote": "evidence_quote",
    "normalizedQuestionKey": "normalized_question_key",
    "originalQuestion": "original_question",
    "originalAnswer": "original_answer",
    "wasEdited": "was_edited",
}

JSONB_COLUMNS = frozenset({"source_chunk_ids", "validation"})


def _json(value: Any) -> Any:
    """Wrap a value bound for a JSONB column."""
    if value is None:
        return None
    from psycopg.types.json import Json

    return Json(value)


def _bind(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        column: _json(value) if column in JSONB_COLUMNS else value
        for column, value in parameters.items()
    }


def new_seed_id() -> str:
    """A fresh id for a seed this layer creates.

    Deliberately not shaped like a Realtime Database push id. Seeds imported
    from the old snapshot keep the push ids they arrived with — those are real
    references, and existing exports name them — while inventing lookalikes for
    new rows would make two different provenances indistinguishable.
    """
    return f"seed-{uuid.uuid4().hex}"


def map_seed(row: Mapping[str, Any]) -> dict[str, Any]:
    """One `seed_examples` row as the API/domain seed record."""
    instruction = row["instruction"]
    response = row["response"]

    record: dict[str, Any] = {
        "id": row["seed_id"],
        "courseId": row["course_id"],
        "instruction": instruction,
        "response": response,
        # Dual names: both halves of each pair are read somewhere.
        "question": instruction,
        "answer": response,
        "category": row["category"],
        "sourceSection": row["source_section"],
        "difficulty": row["difficulty"],
        "directlyAnswered": as_bool(row.get("directly_answered"), True),
        "origin": row["origin"],
        "sourceChunkIds": string_list(row.get("source_chunk_ids")),
        "wasEdited": as_bool(row.get("was_edited")),
    }

    put_optional(record, "notes", optional_string(row.get("notes")))
    put_optional(record, "createdAt", to_iso(row.get("created_at")))
    put_optional(record, "status", optional_string(row.get("status")))
    put_optional(record, "questionType", optional_string(row.get("question_type")))
    put_optional(record, "reviewStatus", optional_string(row.get("review_status")))
    put_optional(record, "reviewNotes", optional_string(row.get("review_notes")))
    put_optional(record, "reviewedAt", to_iso(row.get("reviewed_at")))
    put_optional(record, "factId", optional_string(row.get("fact_id")))
    put_optional(record, "evidenceQuote", optional_string(row.get("evidence_quote")))
    put_optional(
        record,
        "normalizedQuestionKey",
        optional_string(row.get("normalized_question_key")),
    )
    put_optional(record, "originalQuestion", optional_string(row.get("original_question")))
    put_optional(record, "originalAnswer", optional_string(row.get("original_answer")))

    validation = row.get("validation")
    if isinstance(validation, dict):
        record["validation"] = validation

    return record


def list_seeds(conn: Any, course_id: str) -> list[dict[str, Any]]:
    """Every seed for one course, newest first.

    Matches `parseSeedExamplesFromSnapshot`, which sorts by createdAt
    descending. NULLS LAST keeps undated legacy seeds at the bottom instead of
    letting them sort above everything.
    """
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {SEED_COLUMNS} FROM seed_examples
            WHERE course_id = %s
            ORDER BY created_at DESC NULLS LAST, seed_id ASC
            """,
            (safe_course_id,),
        )
        rows = cursor.fetchall()
    return [map_seed(row) for row in rows]


def get_seed(conn: Any, course_id: str, seed_id: str) -> dict[str, Any] | None:
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            f"SELECT {SEED_COLUMNS} FROM seed_examples "
            "WHERE course_id = %s AND seed_id = %s",
            (safe_course_id, seed_id),
        )
        row = cursor.fetchone()
    return map_seed(row) if row else None


def count_seeds_by_review_status(conn: Any, course_id: str) -> dict[str, int]:
    """Counts per review status, with the legacy `status` as the fallback.

    `resolve_review_status` prefers reviewStatus and falls back to status for
    records written before the review field existed; COALESCE reproduces that
    in SQL so the totals agree with what the review UI shows.
    """
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT COALESCE(review_status, status, 'generated') AS bucket,
                   COUNT(*) AS total
            FROM seed_examples
            WHERE course_id = %s
            GROUP BY bucket
            """,
            (safe_course_id,),
        )
        rows = cursor.fetchall()
    return {row["bucket"]: as_int(row["total"]) for row in rows}


def create_seed(
    conn: Any,
    course_id: str,
    seed: Mapping[str, Any],
    *,
    seed_id: str | None = None,
) -> dict[str, Any]:
    """Insert one seed, filling the same defaults the generator writes."""
    safe_course_id = assert_valid_course_id(course_id)

    instruction = optional_string(seed.get("instruction")) or optional_string(
        seed.get("question")
    )
    response = optional_string(seed.get("response")) or optional_string(
        seed.get("answer")
    )
    if not instruction or not response:
        raise ValueError("A seed needs both instruction/question and response/answer.")

    source_chunk_ids = string_list(seed.get("sourceChunkIds"))
    source_section = (
        optional_string(seed.get("sourceSection"))
        or (", ".join(source_chunk_ids) if source_chunk_ids else None)
        or "General"
    )

    parameters = {
        "seed_id": seed_id or optional_string(seed.get("id")) or new_seed_id(),
        "course_id": safe_course_id,
        "instruction": instruction,
        "response": response,
        "category": optional_string(seed.get("category")) or "general",
        "source_section": source_section,
        "difficulty": optional_string(seed.get("difficulty")) or "Medium",
        "directly_answered": as_bool(seed.get("directlyAnswered"), True),
        "origin": optional_string(seed.get("origin")) or "user",
        "notes": optional_string(seed.get("notes")),
        "created_at": optional_string(seed.get("createdAt"))
        or datetime.now(timezone.utc).isoformat(),
        "status": optional_string(seed.get("status")) or "generated",
        "question_type": optional_string(seed.get("questionType")),
        "source_chunk_ids": source_chunk_ids or None,
        "validation": seed.get("validation")
        if isinstance(seed.get("validation"), dict)
        else None,
        "review_status": optional_string(seed.get("reviewStatus")) or "generated",
        "review_notes": optional_string(seed.get("reviewNotes")),
        "reviewed_at": optional_string(seed.get("reviewedAt")),
        "fact_id": optional_string(seed.get("factId")),
        "evidence_quote": optional_string(seed.get("evidenceQuote")),
        "normalized_question_key": optional_string(seed.get("normalizedQuestionKey")),
        "original_question": optional_string(seed.get("originalQuestion")),
        "original_answer": optional_string(seed.get("originalAnswer")),
        "was_edited": as_bool(seed.get("wasEdited")),
    }

    columns = list(parameters)
    placeholders = ", ".join(f"%({column})s" for column in columns)
    with conn.cursor() as cursor:
        cursor.execute(
            f"INSERT INTO seed_examples ({', '.join(columns)}) VALUES ({placeholders})",
            _bind(parameters),
        )

    created = get_seed(conn, safe_course_id, parameters["seed_id"])
    if created is None:  # pragma: no cover - defensive
        raise ValueError("The seed could not be read back after insert.")
    return created


def update_seed(
    conn: Any,
    course_id: str,
    seed_id: str,
    patch: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Patch one seed. Returns None when the course does not hold that id."""
    safe_course_id = assert_valid_course_id(course_id)
    assignments = build_patch(patch, SEED_PATCH_COLUMNS)

    if get_seed(conn, safe_course_id, seed_id) is None:
        return None
    if not assignments:
        return get_seed(conn, safe_course_id, seed_id)

    parameters = {**assignments, "course_id": safe_course_id, "seed_id": seed_id}
    with conn.cursor() as cursor:
        cursor.execute(
            update_statement(
                table="seed_examples",
                assignments=assignments,
                key_columns=["course_id", "seed_id"],
            ),
            _bind(parameters),
        )

    return get_seed(conn, safe_course_id, seed_id)


def review_seed(
    conn: Any,
    course_id: str,
    seed_id: str,
    *,
    review_status: str,
    question: str | None = None,
    answer: str | None = None,
    review_notes: str | None = None,
) -> dict[str, Any] | None:
    """Apply a review through the existing `apply_seed_review` rules.

    The provenance logic — snapshotting originalQuestion/originalAnswer on the
    first edit, forcing `edited`, keeping grounding fields — is not restated
    here. Reimplementing it would give PostgreSQL-backed review subtly different
    behavior from the operational review route that shares the same helper.
    """
    safe_course_id = assert_valid_course_id(course_id)
    existing = get_seed(conn, safe_course_id, seed_id)
    if existing is None:
        return None

    updated = apply_seed_review(
        existing,
        review_status=review_status,
        question=question,
        answer=answer,
        review_notes=review_notes,
    )

    return update_seed(conn, safe_course_id, seed_id, updated)


def delete_seed(conn: Any, course_id: str, seed_id: str) -> bool:
    """Delete one seed, always keyed by course AND id. True when a row went."""
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            "DELETE FROM seed_examples WHERE course_id = %s AND seed_id = %s",
            (safe_course_id, seed_id),
        )
        return cursor.rowcount > 0
