"""PostgreSQL repository for course evaluations.

Matches `EvaluationRecord` in `src/types/index.ts` field for field. Nothing is
added: `comparisonId` stays required because records written before free-text
questions existed rely on it to aggregate, and `questionText`/`runId` stay
optional because only later records carry them.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

from app.course_id import assert_valid_course_id
from app.db_mapping import optional_string, put_optional, string_list, to_iso

EVALUATION_COLUMNS = """
    evaluation_id, course_id, comparison_id, most_accurate, most_helpful,
    most_concise, best_grounded, preferred_model, hallucination_flags,
    comment, created_at, run_id, question_text
"""

REQUIRED_FIELDS = (
    ("comparisonId", "comparison_id"),
    ("mostAccurate", "most_accurate"),
    ("mostHelpful", "most_helpful"),
    ("mostConcise", "most_concise"),
    ("bestGrounded", "best_grounded"),
    ("preferredModel", "preferred_model"),
)


def new_evaluation_id() -> str:
    return f"eval-{uuid.uuid4().hex}"


def map_evaluation(row: Mapping[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": row["evaluation_id"],
        "courseId": row["course_id"],
        "comparisonId": row["comparison_id"],
        "mostAccurate": row["most_accurate"],
        "mostHelpful": row["most_helpful"],
        "mostConcise": row["most_concise"],
        "bestGrounded": row["best_grounded"],
        "preferredModel": row["preferred_model"],
        "hallucinationFlags": string_list(row.get("hallucination_flags")),
        "createdAt": to_iso(row.get("created_at")),
    }
    put_optional(record, "comment", optional_string(row.get("comment")))
    put_optional(record, "runId", optional_string(row.get("run_id")))
    put_optional(record, "questionText", optional_string(row.get("question_text")))
    return record


def list_evaluations(conn: Any, course_id: str) -> list[dict[str, Any]]:
    """Newest first, matching `parseEvaluationsFromSnapshot`."""
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {EVALUATION_COLUMNS} FROM evaluations
            WHERE course_id = %s
            ORDER BY created_at DESC, evaluation_id ASC
            """,
            (safe_course_id,),
        )
        rows = cursor.fetchall()
    return [map_evaluation(row) for row in rows]


def get_evaluation(
    conn: Any, course_id: str, evaluation_id: str
) -> dict[str, Any] | None:
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            f"SELECT {EVALUATION_COLUMNS} FROM evaluations "
            "WHERE course_id = %s AND evaluation_id = %s",
            (safe_course_id, evaluation_id),
        )
        row = cursor.fetchone()
    return map_evaluation(row) if row else None


def create_evaluation(
    conn: Any,
    course_id: str,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    from psycopg.types.json import Json

    safe_course_id = assert_valid_course_id(course_id)

    parameters: dict[str, Any] = {
        "evaluation_id": optional_string(evaluation.get("id")) or new_evaluation_id(),
        "course_id": safe_course_id,
        "created_at": optional_string(evaluation.get("createdAt"))
        or datetime.now(timezone.utc).isoformat(),
        "comment": optional_string(evaluation.get("comment")),
        "run_id": optional_string(evaluation.get("runId")),
        "question_text": optional_string(evaluation.get("questionText")),
    }

    for field, column in REQUIRED_FIELDS:
        value = optional_string(evaluation.get(field))
        if not value:
            raise ValueError(f"Evaluation is missing required field '{field}'.")
        parameters[column] = value

    parameters["hallucination_flags"] = Json(
        string_list(evaluation.get("hallucinationFlags"))
    )

    columns = list(parameters)
    placeholders = ", ".join(f"%({column})s" for column in columns)
    with conn.cursor() as cursor:
        cursor.execute(
            f"INSERT INTO evaluations ({', '.join(columns)}) VALUES ({placeholders})",
            parameters,
        )

    created = get_evaluation(conn, safe_course_id, parameters["evaluation_id"])
    if created is None:  # pragma: no cover - defensive
        raise ValueError("The evaluation could not be read back after insert.")
    return created


def delete_evaluation(conn: Any, course_id: str, evaluation_id: str) -> bool:
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            "DELETE FROM evaluations WHERE course_id = %s AND evaluation_id = %s",
            (safe_course_id, evaluation_id),
        )
        return cursor.rowcount > 0


def delete_all_evaluations(conn: Any, course_id: str) -> int:
    """Clear one course's evaluations. Course-scoped, never a bare DELETE."""
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            "DELETE FROM evaluations WHERE course_id = %s", (safe_course_id,)
        )
        return max(0, cursor.rowcount)
