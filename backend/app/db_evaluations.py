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
    comment, created_at, run_id, question_text, participant_id
"""

REQUIRED_FIELDS = (
    ("comparisonId", "comparison_id"),
    ("mostAccurate", "most_accurate"),
    ("preferredModel", "preferred_model"),
)

#: Criteria the student form no longer asks for.
#:
#: Every evaluation recorded before the form was cut down carries all three, and
#: nothing here deletes or rewrites one. What changed is that a new rating may
#: omit them, so they moved out of REQUIRED_FIELDS.
#:
#: The columns stay `NOT NULL`. Making them nullable would be a migration
#: against a database holding live research data to record something the empty
#: string already records, so an unanswered criterion is stored as `''` and
#: `map_evaluation` omits it from the API record — the same "absent means not
#: asked" the TypeScript parsers apply to every other optional field.
RETIRED_FIELDS = (
    ("mostHelpful", "most_helpful"),
    ("mostConcise", "most_concise"),
    ("bestGrounded", "best_grounded"),
)


def new_evaluation_id() -> str:
    return f"eval-{uuid.uuid4().hex}"


#: The id prefix of a previewed rating. No stored row ever carries it, so a
#: record an administrator's preview echoes back cannot be mistaken for one.
PREVIEW_ID_PREFIX = "preview-"


def new_preview_evaluation_id() -> str:
    return f"{PREVIEW_ID_PREFIX}{uuid.uuid4().hex}"


def map_evaluation(row: Mapping[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": row["evaluation_id"],
        "courseId": row["course_id"],
        "comparisonId": row["comparison_id"],
        "mostAccurate": row["most_accurate"],
        "preferredModel": row["preferred_model"],
        "hallucinationFlags": string_list(row.get("hallucination_flags")),
        "createdAt": to_iso(row.get("created_at")),
    }
    # An empty retired column means the student was never asked. Omitted rather
    # than emitted as "", so aggregation counts answers and nothing else.
    for field, column in RETIRED_FIELDS:
        put_optional(record, field, optional_string(row.get(column)))
    put_optional(record, "comment", optional_string(row.get("comment")))
    put_optional(record, "runId", optional_string(row.get("run_id")))
    put_optional(record, "questionText", optional_string(row.get("question_text")))
    # The pseudonymous participant, when one was recorded. Absent on every
    # row written before participants existed; those stay anonymous as they
    # were. Staff responses carry it; a participant's own view drops it.
    participant_id = row.get("participant_id")
    if participant_id is not None:
        record["participantId"] = str(participant_id)
    return record


def list_evaluations(
    conn: Any, course_id: str, *, participant_id: str | None = None
) -> list[dict[str, Any]]:
    """Newest first, matching `parseEvaluationsFromSnapshot`.

    `participant_id` narrows the list to one participant's own ratings — what
    a student is shown. Staff pass nothing and see the course.
    """
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        if participant_id is None:
            cursor.execute(
                f"""
                SELECT {EVALUATION_COLUMNS} FROM evaluations
                WHERE course_id = %s
                ORDER BY created_at DESC, evaluation_id ASC
                """,
                (safe_course_id,),
            )
        else:
            cursor.execute(
                f"""
                SELECT {EVALUATION_COLUMNS} FROM evaluations
                WHERE course_id = %s AND participant_id = %s
                ORDER BY created_at DESC, evaluation_id ASC
                """,
                (safe_course_id, participant_id),
            )
        rows = cursor.fetchall()
    return [map_evaluation(row) for row in rows]


def count_evaluations(conn: Any, course_id: str) -> int:
    """How many ratings the course has, for the student home page."""
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) AS total FROM evaluations WHERE course_id = %s",
            (safe_course_id,),
        )
        row = cursor.fetchone()
    return int(row["total"]) if row else 0


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


def evaluation_row(
    course_id: str,
    evaluation: Mapping[str, Any],
    *,
    evaluation_id: str,
    participant_id: str | None,
) -> dict[str, Any]:
    """The columns one rating maps to, validated.

    Shared by the insert and by the administrator's preview, so the two cannot
    disagree about what a complete rating is: a body the preview accepts is a
    body the real route would have stored, and one it refuses the real route
    refuses too. `hallucination_flags` is a plain list here; the insert wraps
    it for psycopg.
    """
    safe_course_id = assert_valid_course_id(course_id)

    row: dict[str, Any] = {
        "evaluation_id": evaluation_id,
        "course_id": safe_course_id,
        "participant_id": participant_id,
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
        row[column] = value

    for field, column in RETIRED_FIELDS:
        row[column] = optional_string(evaluation.get(field)) or ""

    row["hallucination_flags"] = string_list(evaluation.get("hallucinationFlags"))
    return row


def preview_evaluation(course_id: str, evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """What `create_evaluation` would have stored, without storing it.

    The administrator's preview of the student flow ends here rather than in
    the table. The body goes through the same validation — a missing required
    field raises the same ValueError — and comes back through the same mapping
    a stored row is read with, so the preview shows exactly what a student's
    submission would have produced. No connection is taken, no participant is
    named, and the id carries a prefix no stored row ever has.
    """
    return map_evaluation(
        evaluation_row(
            course_id,
            evaluation,
            evaluation_id=new_preview_evaluation_id(),
            participant_id=None,
        )
    )


def create_evaluation(
    conn: Any,
    course_id: str,
    evaluation: Mapping[str, Any],
    *,
    participant_id: str | None = None,
) -> dict[str, Any]:
    """Insert one rating.

    `participant_id` is the session's participant, supplied by the route and
    never read from the body: attribution is a fact about who was asking, not
    a field a client fills in.
    """
    from psycopg.types.json import Json

    parameters = evaluation_row(
        course_id,
        evaluation,
        evaluation_id=optional_string(evaluation.get("id")) or new_evaluation_id(),
        participant_id=participant_id,
    )
    safe_course_id = parameters["course_id"]
    parameters["hallucination_flags"] = Json(parameters["hallucination_flags"])

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
