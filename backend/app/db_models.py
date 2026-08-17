"""PostgreSQL repository for the per-course model registry.

Returns the `CourseModelRegistry` shape `parseCourseModelRegistry` already
reads: `{currentVersion, versions: {v1: {...}}}`, versions keyed by version
string. The relational split across `course_models` and
`course_model_versions` is an implementation detail that stops at this module.

Registry rows are written by whoever trains and promotes a model — see
`scripts/register_course_model.py`. The upsert here exists for that class of
backend workflow, not for the UI, which has never created a version because
nothing in the UI trains one.
"""

from __future__ import annotations

from typing import Any, Mapping

from app.course_id import assert_valid_course_id
from app.db_mapping import as_int, optional_string, put_optional, to_iso

VERSION_COLUMNS = """
    course_id, version, base_model, training_example_count, status,
    deployment, artifact_ref, created_at, updated_at, notes
"""


def map_model_version(row: Mapping[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "version": row["version"],
        "baseModel": row["base_model"],
        "trainingExampleCount": as_int(row.get("training_example_count")),
        "status": row["status"],
        # An unrecorded deployment is "unknown", never assumed either way.
        "deployment": optional_string(row.get("deployment")) or "unknown",
        "artifactRef": row["artifact_ref"],
        "createdAt": to_iso(row.get("created_at")),
    }
    put_optional(record, "updatedAt", to_iso(row.get("updated_at")))
    put_optional(record, "notes", optional_string(row.get("notes")))
    return record


def list_model_versions(conn: Any, course_id: str) -> list[dict[str, Any]]:
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {VERSION_COLUMNS} FROM course_model_versions
            WHERE course_id = %s
            ORDER BY created_at DESC, version DESC
            """,
            (safe_course_id,),
        )
        rows = cursor.fetchall()
    return [map_model_version(row) for row in rows]


def get_model_registry(conn: Any, course_id: str) -> dict[str, Any] | None:
    """The course's registry, or None when it has no model.

    None rather than an empty registry: `parseCourseModelRegistry` returns null
    for an absent record, and the UI distinguishes "no model yet" from "a model
    whose versions failed to load".
    """
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT course_id, current_version FROM course_models WHERE course_id = %s",
            (safe_course_id,),
        )
        model_row = cursor.fetchone()

    if model_row is None:
        return None

    versions = list_model_versions(conn, safe_course_id)
    if not versions:
        return None

    return {
        "courseId": safe_course_id,
        "currentVersion": model_row["current_version"],
        "versions": {version["version"]: version for version in versions},
    }


def upsert_model_version(
    conn: Any,
    course_id: str,
    version: Mapping[str, Any],
    *,
    set_current: bool = False,
) -> dict[str, Any] | None:
    """Register or refresh one version, optionally promoting it to current.

    Promotion is explicit. A newly registered version is an artifact that
    exists; making it the one professors are shown is a separate decision, and
    inferring it from recency would promote a re-registered old adapter.
    """
    safe_course_id = assert_valid_course_id(course_id)

    version_key = optional_string(version.get("version"))
    if not version_key:
        raise ValueError("A model version needs a 'version'.")

    parameters = {
        "course_id": safe_course_id,
        "version": version_key,
        "base_model": version["baseModel"],
        "training_example_count": as_int(version.get("trainingExampleCount")),
        "status": version["status"],
        "deployment": optional_string(version.get("deployment")) or "unknown",
        "artifact_ref": version["artifactRef"],
        "created_at": version["createdAt"],
        "updated_at": optional_string(version.get("updatedAt")),
        "notes": optional_string(version.get("notes")),
    }

    columns = list(parameters)
    placeholders = ", ".join(f"%({column})s" for column in columns)
    updates = ", ".join(
        f"{column} = EXCLUDED.{column}"
        for column in columns
        if column not in ("course_id", "version")
    )

    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO course_model_versions ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT (course_id, version) DO UPDATE SET {updates}
            """,
            parameters,
        )

        if set_current:
            cursor.execute(
                """
                INSERT INTO course_models (course_id, current_version)
                VALUES (%(course_id)s, %(version)s)
                ON CONFLICT (course_id)
                DO UPDATE SET current_version = EXCLUDED.current_version
                """,
                {"course_id": safe_course_id, "version": version_key},
            )

    return get_model_registry(conn, safe_course_id)


def set_current_version(conn: Any, course_id: str, version: str) -> dict[str, Any] | None:
    """Point the course at an existing version. None when it has no such version."""
    safe_course_id = assert_valid_course_id(course_id)
    version_key = optional_string(version)
    if not version_key:
        raise ValueError("A current version needs a version string.")

    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT 1 FROM course_model_versions WHERE course_id = %s AND version = %s",
            (safe_course_id, version_key),
        )
        if cursor.fetchone() is None:
            return None

        cursor.execute(
            """
            INSERT INTO course_models (course_id, current_version)
            VALUES (%(course_id)s, %(version)s)
            ON CONFLICT (course_id)
            DO UPDATE SET current_version = EXCLUDED.current_version
            """,
            {"course_id": safe_course_id, "version": version_key},
        )

    return get_model_registry(conn, safe_course_id)
