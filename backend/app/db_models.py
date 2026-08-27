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

from datetime import datetime, timezone
from typing import Any, Mapping

from app.course_id import assert_valid_course_id
from app.db_mapping import (
    as_int,
    bind_jsonb,
    optional_string,
    put_optional,
    to_iso,
)

VERSION_COLUMNS = """
    course_id, version, base_model, training_example_count, status,
    deployment, artifact_ref, created_at, updated_at, notes, run_id, provenance
"""

JSONB_COLUMNS = frozenset({"provenance"})


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
    put_optional(record, "runId", optional_string(row.get("run_id")))

    provenance = row.get("provenance")
    if isinstance(provenance, dict):
        record["provenance"] = provenance

    return record


def find_model_version_for_run(
    conn: Any, course_id: str, run_id: str
) -> dict[str, Any] | None:
    """The version already registered for this run, if there is one.

    The read half of idempotent registration: a completion callback that arrives
    twice finds the version its first delivery created and reuses it instead of
    allocating the next `vN`. `uq_course_model_versions_run` is the other half —
    it makes a second row impossible even if two callbacks race past this read,
    which a retry after a timeout genuinely can.
    """
    safe_course_id = assert_valid_course_id(course_id)
    cleaned_run_id = optional_string(run_id)
    if not cleaned_run_id:
        return None

    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {VERSION_COLUMNS} FROM course_model_versions
            WHERE course_id = %s AND run_id = %s
            """,
            (safe_course_id, cleaned_run_id),
        )
        row = cursor.fetchone()
    return map_model_version(row) if row else None


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
        "run_id": optional_string(version.get("runId")),
        "provenance": version.get("provenance"),
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
            bind_jsonb(parameters, JSONB_COLUMNS),
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


# --------------------------------------------------------------------------- #
# Publication
#
# `deployment` finally has a writer. Until now every row said `offline` forever,
# because the only thing that could have changed it — an adapter being copied
# into the cluster's serving tree — happened on a machine with no database
# connection and reported nothing back.
#
# That made it a field nobody could trust, which is why
# `resolve_current_course_model` ignored it and resolved from `current_version`
# instead. With two versions of one course that became a real outage: a finished
# run registers `v2` and moves `current_version` to it, the backend starts
# asking the cluster for `v2`, and the cluster only has the `v1` an operator
# published. Every fine-tuned answer for that course fails until somebody
# publishes `v2` — including answers `v1` was serving perfectly a moment before.
#
# So publication now reports itself, and `deployment` is the record of it.
# --------------------------------------------------------------------------- #


def find_deployed_version(conn: Any, course_id: str) -> dict[str, Any] | None:
    """The version this course has published, or None when it has none.

    At most one row per course can be `online` — `mark_version_published`
    demotes the others in the same statement pair — but the query orders anyway
    so that a row inserted by an older code path cannot make the answer depend
    on physical order.
    """
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {VERSION_COLUMNS} FROM course_model_versions
            WHERE course_id = %s AND deployment = 'online'
            ORDER BY updated_at DESC NULLS LAST, created_at DESC, version DESC
            LIMIT 1
            """,
            (safe_course_id,),
        )
        row = cursor.fetchone()
    return map_model_version(row) if row else None


def mark_version_published(
    conn: Any,
    course_id: str,
    version: str,
    *,
    now: str | None = None,
) -> dict[str, Any] | None:
    """Record that one version is the one published for this course.

    Two writes, one transaction, in this order: demote every other version of
    the course, then promote this one. A reader that arrives between them sees
    no online version and falls back to `current_version`, which is the
    behaviour of a course that has never published — degraded, not wrong. The
    opposite order would briefly show two online versions, and "which one is
    served?" would have two answers.

    Returns None when the course has no such version, so a caller can tell
    "published something that is not registered" from "published successfully".

    Idempotent: publishing the version that is already online demotes nothing it
    should not and re-promotes the same row. A publish script that ran twice, or
    a report delivered twice after a network failure, lands in the same state.
    """
    safe_course_id = assert_valid_course_id(course_id)
    version_key = optional_string(version)
    if not version_key:
        raise ValueError("A published version needs a version string.")

    stamp = now or datetime.now(timezone.utc).isoformat()

    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT 1 FROM course_model_versions "
            "WHERE course_id = %s AND version = %s",
            (safe_course_id, version_key),
        )
        if cursor.fetchone() is None:
            return None

        cursor.execute(
            """
            UPDATE course_model_versions
            SET deployment = 'offline', updated_at = %(now)s
            WHERE course_id = %(course_id)s
              AND version <> %(version)s
              AND deployment = 'online'
            """,
            {"course_id": safe_course_id, "version": version_key, "now": stamp},
        )
        cursor.execute(
            """
            UPDATE course_model_versions
            SET deployment = 'online', updated_at = %(now)s
            WHERE course_id = %(course_id)s AND version = %(version)s
            """,
            {"course_id": safe_course_id, "version": version_key, "now": stamp},
        )

    return get_model_registry(conn, safe_course_id)


def mark_version_unpublished(
    conn: Any,
    course_id: str,
    version: str,
    *,
    now: str | None = None,
) -> dict[str, Any] | None:
    """Record that a version is no longer published.

    The artifact still exists and the version stays `ready`; only the statement
    that something is serving it is withdrawn. Used when an operator removes a
    published adapter, and by nothing automatic — a serving session ending does
    not unpublish anything, because the adapter is still on the filesystem and
    the next session will load it.
    """
    safe_course_id = assert_valid_course_id(course_id)
    version_key = optional_string(version)
    if not version_key:
        raise ValueError("An unpublished version needs a version string.")

    stamp = now or datetime.now(timezone.utc).isoformat()
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE course_model_versions
            SET deployment = 'offline', updated_at = %(now)s
            WHERE course_id = %(course_id)s AND version = %(version)s
            """,
            {"course_id": safe_course_id, "version": version_key, "now": stamp},
        )
        if cursor.rowcount == 0:
            return None

    return get_model_registry(conn, safe_course_id)
