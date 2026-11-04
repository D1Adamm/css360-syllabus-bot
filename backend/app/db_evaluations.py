"""PostgreSQL repository for course evaluations.

Matches `EvaluationRecord` in `src/types/index.ts` field for field.
`comparisonId` stays required because records written before free-text
questions existed rely on it to aggregate, and `questionText`/`runId` stay
optional because only later records carry them.

What a rating is allowed to say (audit findings B8 and B9, 2026-10-27)
--------------------------------------------------------------------
A participant's body used to be stored as it arrived: any string for the
approach it preferred, any list for the flags, any length of comment, any
timestamp, and no record of the four answers it was choosing between — so a
rating could later be read against answers the models no longer give, and the
same comparison could be rated twice. Now:

- the criteria name one of the four approaches (`MODEL_KEYS`), flags are a
  subset of them, and text fields have the lengths the form enforces;
- `responses` — the answers the student saw, with the model, version and
  engine that produced each — is validated and stored beside the rating, and
  a criterion cannot name an approach whose answer never arrived;
- `createdAt` is the server's clock, not the browser's;
- one participant rates one comparison run once; a second attempt is refused
  with `DuplicateEvaluationError` and nothing is written.

Rows written before this keep their shape: `responses` is NULL on them, and
nothing here rewrites a stored rating.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

from app.course_id import assert_valid_course_id
from app.db_mapping import optional_string, put_optional, string_list, to_iso
from app.schemas import QUESTION_MAX_LENGTH

EVALUATION_COLUMNS = """
    evaluation_id, course_id, comparison_id, most_accurate, most_helpful,
    most_concise, best_grounded, preferred_model, hallucination_flags,
    comment, created_at, run_id, question_text, participant_id, responses
"""

#: The four approaches a rating may name: `MODEL_KEYS` in
#: `src/utils/evaluationUtils.ts`, and the keys of a comparison run's
#: `responses`.
MODEL_KEYS = ("base", "rag", "fineTuned", "fineTunedRag")

#: What the student form enforces, enforced here as well so the form is not the
#: only thing standing between a browser and the research table.
COMMENT_MAX_LENGTH = 1000
#: The generation routes refuse a longer question (`schemas.QUESTION_MAX_LENGTH`),
#: so every question that was asked can be rated.
QUESTION_TEXT_MAX_LENGTH = QUESTION_MAX_LENGTH
IDENTIFIER_MAX_LENGTH = 200
RESPONSE_TEXT_MAX_LENGTH = 20000
RESPONSE_ERROR_MAX_LENGTH = 2000
RESPONSE_SOURCES_MAX = 50
RESPONSE_LABEL_MAX_LENGTH = 300

#: The per-approach fields of a stored response snapshot, and their limits.
RESPONSE_STRING_FIELDS = (
    ("text", RESPONSE_TEXT_MAX_LENGTH),
    ("error", RESPONSE_ERROR_MAX_LENGTH),
    ("model", RESPONSE_LABEL_MAX_LENGTH),
    ("modelVersion", 40),
    ("engine", 40),
)


class DuplicateEvaluationError(ValueError):
    """This participant has already rated this comparison run."""

    def __init__(self, evaluation_id: str) -> None:
        super().__init__("This comparison has already been rated.")
        self.evaluation_id = evaluation_id

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
    # The answers rated, when the rating recorded them. Absent, not null, on
    # rows written before the column existed.
    responses = row.get("responses")
    if isinstance(responses, dict) and responses:
        record["responses"] = responses
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


def _bounded_string(
    evaluation: Mapping[str, Any], field: str, max_length: int
) -> str | None:
    value = optional_string(evaluation.get(field))
    if value is not None and len(value) > max_length:
        raise ValueError(
            f"Evaluation field '{field}' is longer than {max_length} characters."
        )
    return value


def _require_model_key(field: str, value: Any) -> str:
    key = optional_string(value)
    if not key:
        raise ValueError(f"Evaluation is missing required field '{field}'.")
    if key not in MODEL_KEYS:
        raise ValueError(
            f"Evaluation field '{field}' must name one of {', '.join(MODEL_KEYS)}; got {key!r}."
        )
    return key


def _optional_model_key(field: str, value: Any) -> str:
    key = optional_string(value)
    if key is None:
        return ""
    if key not in MODEL_KEYS:
        raise ValueError(
            f"Evaluation field '{field}' must name one of {', '.join(MODEL_KEYS)}; got {key!r}."
        )
    return key


def _model_key_list(field: str, value: Any) -> list[str]:
    keys = string_list(value)
    unknown = [key for key in keys if key not in MODEL_KEYS]
    if unknown:
        raise ValueError(
            f"Evaluation field '{field}' names unknown approaches: {', '.join(unknown)}."
        )
    # Order kept, repeats dropped: a flag is a yes/no per approach.
    seen: list[str] = []
    for key in keys:
        if key not in seen:
            seen.append(key)
    return seen


def validate_responses(raw: Any) -> dict[str, dict[str, Any]] | None:
    """The four answers a rating was made against, checked and trimmed.

    None when the body carries none (a client from before the field existed,
    or a historical import). Otherwise every one of `MODEL_KEYS` must be
    present, and each entry is reduced to the fields this table stores: the
    text, the student-facing error if the approach did not answer, its
    sources, and the model, version and engine that produced it.
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("Evaluation field 'responses' must be an object keyed by approach.")
    missing = [key for key in MODEL_KEYS if key not in raw]
    unknown = [key for key in raw if key not in MODEL_KEYS]
    if missing or unknown:
        raise ValueError(
            "Evaluation field 'responses' must have exactly the four approaches "
            f"({', '.join(MODEL_KEYS)})."
        )
    cleaned: dict[str, dict[str, Any]] = {}
    for key in MODEL_KEYS:
        entry = raw[key]
        if not isinstance(entry, Mapping):
            raise ValueError(f"Evaluation field 'responses.{key}' must be an object.")
        record: dict[str, Any] = {}
        for field, limit in RESPONSE_STRING_FIELDS:
            value = entry.get(field)
            if value is None:
                record[field] = None if field != "text" else ""
                continue
            if not isinstance(value, str):
                raise ValueError(f"Evaluation field 'responses.{key}.{field}' must be a string.")
            if len(value) > limit:
                raise ValueError(
                    f"Evaluation field 'responses.{key}.{field}' is longer than {limit} characters."
                )
            record[field] = value
        sources = string_list(entry.get("sources"))
        if len(sources) > RESPONSE_SOURCES_MAX or any(
            len(source) > RESPONSE_LABEL_MAX_LENGTH for source in sources
        ):
            raise ValueError(f"Evaluation field 'responses.{key}.sources' is too long.")
        record["sources"] = sources
        cleaned[key] = record
    return cleaned


def _answered(responses: Mapping[str, Mapping[str, Any]], key: str) -> bool:
    entry = responses.get(key) or {}
    return not entry.get("error") and bool(str(entry.get("text") or "").strip())


def _assert_rated_approaches_answered(
    row: Mapping[str, Any], responses: Mapping[str, Mapping[str, Any]] | None
) -> None:
    """A criterion cannot be won, or flagged, by an answer that never arrived."""
    if responses is None:
        return
    named: list[tuple[str, str]] = [
        ("mostAccurate", row["most_accurate"]),
        ("preferredModel", row["preferred_model"]),
    ]
    named.extend(("hallucinationFlags", key) for key in row["hallucination_flags"])
    for field, key in named:
        if not _answered(responses, key):
            raise ValueError(
                f"Evaluation field '{field}' names '{key}', but that approach did not "
                "answer this comparison and cannot be rated."
            )
    if not any(_answered(responses, key) for key in MODEL_KEYS):
        raise ValueError("None of the approaches answered, so there is nothing to rate.")


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
    refuses too. `hallucination_flags` and `responses` are plain Python here;
    the insert wraps them for psycopg.

    `createdAt` in the body is ignored: the time a rating was recorded is the
    server's to state, and a browser clock is neither trustworthy nor
    something a participant should be able to set.
    """
    safe_course_id = assert_valid_course_id(course_id)

    row: dict[str, Any] = {
        "evaluation_id": evaluation_id,
        "course_id": safe_course_id,
        "participant_id": participant_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "comment": _bounded_string(evaluation, "comment", COMMENT_MAX_LENGTH),
        "run_id": _bounded_string(evaluation, "runId", IDENTIFIER_MAX_LENGTH),
        "question_text": _bounded_string(
            evaluation, "questionText", QUESTION_TEXT_MAX_LENGTH
        ),
    }

    comparison_id = _bounded_string(evaluation, "comparisonId", IDENTIFIER_MAX_LENGTH)
    if not comparison_id:
        raise ValueError("Evaluation is missing required field 'comparisonId'.")
    row["comparison_id"] = comparison_id
    row["most_accurate"] = _require_model_key("mostAccurate", evaluation.get("mostAccurate"))
    row["preferred_model"] = _require_model_key(
        "preferredModel", evaluation.get("preferredModel")
    )

    for field, column in RETIRED_FIELDS:
        row[column] = _optional_model_key(field, evaluation.get(field))

    row["hallucination_flags"] = _model_key_list(
        "hallucinationFlags", evaluation.get("hallucinationFlags")
    )
    row["responses"] = validate_responses(evaluation.get("responses"))
    _assert_rated_approaches_answered(row, row["responses"])
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


def _lock_participant_run(conn: Any, course_id: str, participant_id: str, run_id: str) -> None:
    """Serialise this participant's ratings of this run for the transaction.

    The duplicate check below is a SELECT followed by an INSERT. Under READ
    COMMITTED two submissions arriving together — a double click, a second
    tab, a retry after a slow save — would each see no earlier row and both
    insert. A transaction-scoped advisory lock keyed on the participant and
    the run makes the second wait until the first has committed, at which
    point its SELECT sees the row. Two int4 keys, because `hashtext()` exists
    on every PostgreSQL release; the lock is released with the transaction.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT pg_advisory_xact_lock(hashtext(%s), hashtext(%s))",
            (f"evaluation:{course_id}:{participant_id}", run_id),
        )


def find_participant_run_evaluation(
    conn: Any, course_id: str, participant_id: str, run_id: str
) -> dict[str, Any] | None:
    """The rating this participant already recorded for this run, if any."""
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {EVALUATION_COLUMNS} FROM evaluations
            WHERE course_id = %s AND participant_id = %s AND run_id = %s
            ORDER BY created_at ASC, evaluation_id ASC
            LIMIT 1
            """,
            (safe_course_id, participant_id, run_id),
        )
        row = cursor.fetchone()
    return map_evaluation(row) if row else None


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
    a field a client fills in. When the rating names the comparison run it
    belongs to, a second rating of that run by the same participant is refused
    with `DuplicateEvaluationError` before anything is written.
    """
    from psycopg.types.json import Json

    parameters = evaluation_row(
        course_id,
        evaluation,
        evaluation_id=optional_string(evaluation.get("id")) or new_evaluation_id(),
        participant_id=participant_id,
    )
    safe_course_id = parameters["course_id"]

    if participant_id and parameters["run_id"]:
        _lock_participant_run(conn, safe_course_id, participant_id, parameters["run_id"])
        existing = find_participant_run_evaluation(
            conn, safe_course_id, participant_id, parameters["run_id"]
        )
        if existing is not None:
            raise DuplicateEvaluationError(str(existing["id"]))

    parameters["hallucination_flags"] = Json(parameters["hallucination_flags"])
    parameters["responses"] = (
        Json(parameters["responses"]) if parameters["responses"] is not None else None
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
