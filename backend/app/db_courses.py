"""PostgreSQL repository for courses, metadata, and starter-generation state.

Shapes match `CourseMetadata` and `StoredStarterSeedGeneration` in
`src/types/index.ts` so the frontend can be pointed here without a parser
change. `starterSeedGeneration` is nested inside the metadata object rather
than exposed as a sibling resource — the frontend reads it off metadata, and a
different shape would mean rewriting `useCourseMetadata` for no gain.

Every function takes an open connection. Routes own the transaction; that is
what lets one request touch two tables atomically and what lets the tests run
these against a recording fake instead of a server.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from app.course_id import assert_valid_course_id
from app.db_mapping import (
    as_int,
    build_patch,
    optional_string,
    put_optional,
    to_iso,
    update_statement,
)

COURSE_COLUMNS = """
    course_id, name, title, term, instructor_name, created_at,
    syllabus_status, syllabus_file_name, syllabus_type, chunk_count
"""

STARTER_COLUMNS = """
    course_id, status, target_count, final_count, saved_count,
    failed_to_save_count, error, started_at, completed_at,
    achievable_ceiling, limiting_factor
"""

# camelCase API field -> column. Also the allowlist: a key absent here never
# reaches SQL, so no caller can name a column.
COURSE_PATCH_COLUMNS = {
    "name": "name",
    "title": "title",
    "term": "term",
    "instructorName": "instructor_name",
    "syllabusStatus": "syllabus_status",
    "syllabusFileName": "syllabus_file_name",
    "syllabusType": "syllabus_type",
    "chunkCount": "chunk_count",
}

STARTER_PATCH_COLUMNS = {
    "status": "status",
    "targetCount": "target_count",
    "finalCount": "final_count",
    "savedCount": "saved_count",
    "failedToSaveCount": "failed_to_save_count",
    "error": "error",
    "startedAt": "started_at",
    "completedAt": "completed_at",
    "achievableCeiling": "achievable_ceiling",
    "limitingFactor": "limiting_factor",
}


class CourseAlreadyExistsError(Exception):
    """Raised when creating a course id that is already stored."""


def map_course_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    """One `courses` row as the frontend's CourseMetadata."""
    return {
        "name": row["name"],
        "title": row["title"],
        "term": row["term"],
        "instructorName": row["instructor_name"],
        "createdAt": to_iso(row["created_at"]),
        "syllabusStatus": row["syllabus_status"],
        # Explicitly null rather than omitted: CourseMetadata declares these as
        # `string | null`, and `isCourseMetadata` rejects a record missing them.
        "syllabusFileName": optional_string(row.get("syllabus_file_name")),
        "syllabusType": optional_string(row.get("syllabus_type")),
        "chunkCount": as_int(row.get("chunk_count")),
    }


def map_starter_seed_generation(row: Mapping[str, Any]) -> dict[str, Any]:
    """One `starter_seed_generation` row as StoredStarterSeedGeneration."""
    record: dict[str, Any] = {}
    put_optional(record, "status", optional_string(row.get("status")))
    for field, column in (
        ("targetCount", "target_count"),
        ("finalCount", "final_count"),
        ("savedCount", "saved_count"),
        ("failedToSaveCount", "failed_to_save_count"),
    ):
        if row.get(column) is not None:
            record[field] = as_int(row.get(column))
    put_optional(record, "error", optional_string(row.get("error")))
    put_optional(record, "startedAt", to_iso(row.get("started_at")))
    put_optional(record, "completedAt", to_iso(row.get("completed_at")))
    if row.get("achievable_ceiling") is not None:
        record["achievableCeiling"] = as_int(row.get("achievable_ceiling"))
    put_optional(record, "limitingFactor", optional_string(row.get("limiting_factor")))
    return record


def get_starter_seed_generation(conn: Any, course_id: str) -> dict[str, Any] | None:
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            f"SELECT {STARTER_COLUMNS} FROM starter_seed_generation WHERE course_id = %s",
            (safe_course_id,),
        )
        row = cursor.fetchone()
    return map_starter_seed_generation(row) if row else None


def _metadata_with_starter(
    conn: Any, course_id: str, row: Mapping[str, Any]
) -> dict[str, Any]:
    metadata = map_course_metadata(row)
    starter = get_starter_seed_generation(conn, course_id)
    if starter:
        metadata["starterSeedGeneration"] = starter
    return metadata


def list_courses(conn: Any) -> list[dict[str, Any]]:
    """Every course as {courseId, metadata}, newest first.

    Ordering matches `sortCoursesNewestFirst`: createdAt descending, course id
    ascending as the tie-break, so the picker does not reshuffle between a
    server-ordered list and a client-sorted one.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            f"SELECT {COURSE_COLUMNS} FROM courses ORDER BY created_at DESC, course_id ASC"
        )
        rows = cursor.fetchall()

    return [
        {
            "courseId": row["course_id"],
            "metadata": _metadata_with_starter(conn, row["course_id"], row),
        }
        for row in rows
    ]


def get_course(conn: Any, course_id: str) -> dict[str, Any] | None:
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            f"SELECT {COURSE_COLUMNS} FROM courses WHERE course_id = %s",
            (safe_course_id,),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    return {
        "courseId": safe_course_id,
        "metadata": _metadata_with_starter(conn, safe_course_id, row),
    }


def course_exists(conn: Any, course_id: str) -> bool:
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute("SELECT 1 FROM courses WHERE course_id = %s", (safe_course_id,))
        return cursor.fetchone() is not None


def create_course(
    conn: Any,
    course_id: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Insert one course. Raises CourseAlreadyExistsError on a repeat id.

    `ON CONFLICT DO NOTHING` plus a rowcount check rather than a prior SELECT:
    the uniqueness decision belongs to the database, and a check-then-insert
    would let two concurrent creates both see an empty table.
    """
    safe_course_id = assert_valid_course_id(course_id)
    created_at = optional_string(metadata.get("createdAt")) or datetime.now(
        timezone.utc
    ).isoformat()

    parameters = {
        "course_id": safe_course_id,
        "name": metadata["name"],
        "title": metadata["title"],
        "term": metadata["term"],
        "instructor_name": metadata["instructorName"],
        "created_at": created_at,
        "syllabus_status": metadata.get("syllabusStatus") or "none",
        "syllabus_file_name": optional_string(metadata.get("syllabusFileName")),
        "syllabus_type": optional_string(metadata.get("syllabusType")),
        "chunk_count": as_int(metadata.get("chunkCount")),
    }

    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO courses (
                course_id, name, title, term, instructor_name, created_at,
                syllabus_status, syllabus_file_name, syllabus_type, chunk_count
            ) VALUES (
                %(course_id)s, %(name)s, %(title)s, %(term)s, %(instructor_name)s,
                %(created_at)s, %(syllabus_status)s, %(syllabus_file_name)s,
                %(syllabus_type)s, %(chunk_count)s
            )
            ON CONFLICT (course_id) DO NOTHING
            """,
            parameters,
        )
        if cursor.rowcount == 0:
            raise CourseAlreadyExistsError(
                f'Course "{safe_course_id}" already exists.'
            )

    created = get_course(conn, safe_course_id)
    if created is None:  # pragma: no cover - defensive
        raise CourseAlreadyExistsError(
            f'Course "{safe_course_id}" could not be read back after insert.'
        )
    return created


def update_course(
    conn: Any,
    course_id: str,
    patch: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Patch metadata fields. Returns None when the course does not exist.

    A merge, not a write: `updateCourseMetadata` in the frontend patches single
    fields (a syllabus status, a chunk count) and must not disturb the rest.
    """
    safe_course_id = assert_valid_course_id(course_id)
    assignments = build_patch(patch, COURSE_PATCH_COLUMNS)

    if not assignments:
        return get_course(conn, safe_course_id)

    if not course_exists(conn, safe_course_id):
        return None

    parameters = {**assignments, "course_id": safe_course_id}
    with conn.cursor() as cursor:
        cursor.execute(
            update_statement(
                table="courses",
                assignments=assignments,
                key_columns=["course_id"],
            ),
            parameters,
        )

    return get_course(conn, safe_course_id)


def upsert_starter_seed_generation(
    conn: Any,
    course_id: str,
    patch: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Merge starter-generation fields, creating the row if needed.

    Merge, not replace: the generation job writes `startedAt` and `status`
    first and the counts later, so a full write would erase what an earlier
    call stored. `app/starter_status.py` is what drives this on the job's
    behalf.
    """
    safe_course_id = assert_valid_course_id(course_id)
    assignments = build_patch(patch, STARTER_PATCH_COLUMNS)
    if not assignments:
        return get_starter_seed_generation(conn, safe_course_id)

    columns = ["course_id", *assignments]
    placeholders = ", ".join(f"%({column})s" for column in columns)
    updates = ", ".join(f"{column} = EXCLUDED.{column}" for column in assignments)
    parameters = {**assignments, "course_id": safe_course_id}

    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO starter_seed_generation ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT (course_id) DO UPDATE SET {updates}
            """,
            parameters,
        )

    return get_starter_seed_generation(conn, safe_course_id)
