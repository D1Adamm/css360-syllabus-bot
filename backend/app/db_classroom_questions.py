"""PostgreSQL repository for the questions staff selected for classroom preparation.

A selection is a reference to one of the course's existing questions
(`seed_examples`), never a copy of it and never its training answer. It is
independent of the training review state: selecting a question does not
approve it, and approving one does not select it.

Which questions may be selected is decided by `eligibility` below, in Python
rather than SQL so the one rule is used for listing, for saving a selection
and for deciding whether an answer may be stored, and can be tested without a
database:

- rejected questions are never eligible;
- a question a student contributed is eligible only once an instructor has
  approved it (it is then already visible to the whole class), and is never
  selected for staff automatically;
- everything else staff-side (staff-curated imports, AI-generated drafts,
  behaviour examples) is eligible whatever its review state.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

from app.course_id import assert_valid_course_id

#: Origins written by students. A row with a participant on it is a student's
#: whatever its origin says; `user` rows from before participants existed
#: carry no participant and are treated as students' too.
STUDENT_ORIGINS = frozenset({"user"})

#: How each origin is named to staff, and the order the groups are listed in.
SOURCE_LABELS = {
    "prototype": "Staff-curated",
    "ai_generated": "AI-generated",
    "authored": "Behaviour example",
    "user": "Student contribution",
}
SOURCE_ORDER = ("prototype", "ai_generated", "authored", "user")

QUESTION_COLUMNS = """
    s.seed_id, s.instruction, s.origin, s.review_status, s.status,
    s.participant_id, s.category, s.created_at
"""


def review_state(row: Mapping[str, Any]) -> str:
    """`review_status`, then the legacy `status`, as `resolve_review_status` does."""
    for key in ("review_status", "status"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return "generated"


def is_student_sourced(row: Mapping[str, Any]) -> bool:
    return row.get("participant_id") is not None or row.get("origin") in STUDENT_ORIGINS


def eligibility(row: Mapping[str, Any]) -> tuple[bool, str | None]:
    """Whether staff may select this question, and why not when they may not."""
    question = row.get("instruction")
    if not isinstance(question, str) or not question.strip():
        return False, "empty"
    state = review_state(row)
    if state == "rejected":
        return False, "rejected"
    if is_student_sourced(row) and state != "approved":
        return False, "student_unapproved"
    return True, None


def map_question(row: Mapping[str, Any], *, selected: bool) -> dict[str, Any]:
    eligible, reason = eligibility(row)
    origin = row.get("origin") or "user"
    # A staff-side origin on a row a participant wrote is still a student's.
    source = "user" if is_student_sourced(row) else origin
    return {
        "seedId": row["seed_id"],
        "question": " ".join(str(row.get("instruction") or "").split()),
        "origin": origin,
        "source": source,
        "sourceLabel": SOURCE_LABELS.get(source, "Other"),
        "reviewStatus": review_state(row),
        "category": row.get("category"),
        "selected": selected,
        "eligible": eligible,
        "ineligibleReason": reason,
    }


def list_course_questions(conn: Any, course_id: str) -> list[dict[str, Any]]:
    """Every question of the course with its selection and eligibility.

    Ineligible questions are included (flagged) only when they are currently
    selected, so a selection that stopped being eligible is visible to staff
    instead of silently vanishing. Other ineligible questions are left out:
    a student's unapproved contribution is not listed to anyone here.
    """
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {QUESTION_COLUMNS}, (q.seed_id IS NOT NULL) AS selected
            FROM seed_examples s
            LEFT JOIN classroom_questions q
              ON q.course_id = s.course_id AND q.seed_id = s.seed_id
            WHERE s.course_id = %s
            ORDER BY s.created_at DESC NULLS LAST, s.seed_id ASC
            """,
            (safe_course_id,),
        )
        rows = cursor.fetchall()
    questions = [map_question(row, selected=bool(row["selected"])) for row in rows]
    return [item for item in questions if item["eligible"] or item["selected"]]


def list_selected_questions(conn: Any, course_id: str) -> list[dict[str, Any]]:
    """The selected questions that are still eligible, with their current wording.

    This is the set preparation works through and the set an answer may be
    stored or served for.
    """
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {QUESTION_COLUMNS}
            FROM classroom_questions q
            JOIN seed_examples s
              ON s.course_id = q.course_id AND s.seed_id = q.seed_id
            WHERE q.course_id = %s
            ORDER BY q.selected_at ASC, s.seed_id ASC
            """,
            (safe_course_id,),
        )
        rows = cursor.fetchall()
    questions = [map_question(row, selected=True) for row in rows]
    return [item for item in questions if item["eligible"]]


def replace_selection(
    conn: Any,
    course_id: str,
    seed_ids: Iterable[str],
    *,
    selected_by: str | None = None,
    now: datetime | None = None,
) -> list[str]:
    """Make the course's selection exactly `seed_ids`. Returns the ids kept.

    Ids that are not this course's questions, or are not eligible, are
    refused with `ValueError` and nothing is changed. Questions that stay
    selected keep their original `selected_at`, which is the order they are
    prepared in.
    """
    safe_course_id = assert_valid_course_id(course_id)
    wanted: list[str] = []
    for seed_id in seed_ids:
        if not isinstance(seed_id, str) or not seed_id.strip():
            raise ValueError("Every selected question needs its id.")
        if seed_id not in wanted:
            wanted.append(seed_id)

    with conn.cursor() as cursor:
        if wanted:
            cursor.execute(
                f"""
                SELECT {QUESTION_COLUMNS} FROM seed_examples s
                WHERE s.course_id = %s AND s.seed_id = ANY(%s)
                """,
                (safe_course_id, wanted),
            )
            found = {row["seed_id"]: row for row in cursor.fetchall()}
            for seed_id in wanted:
                row = found.get(seed_id)
                if row is None:
                    raise ValueError("A selected question does not belong to this course.")
                eligible, _reason = eligibility(row)
                if not eligible:
                    raise ValueError("A selected question cannot be prepared for the classroom.")

        cursor.execute(
            "DELETE FROM classroom_questions WHERE course_id = %s AND NOT (seed_id = ANY(%s))",
            (safe_course_id, wanted),
        )
        moment = now or datetime.now(timezone.utc)
        for position, seed_id in enumerate(wanted):
            cursor.execute(
                """
                INSERT INTO classroom_questions (course_id, seed_id, selected_at, selected_by)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (course_id, seed_id) DO NOTHING
                """,
                # A microsecond apart, so questions added together are prepared
                # in the order they were given rather than by id.
                (safe_course_id, seed_id, moment + timedelta(microseconds=position), selected_by),
            )
    return wanted
