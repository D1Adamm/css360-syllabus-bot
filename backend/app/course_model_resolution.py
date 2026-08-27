"""Which fine-tuned model answers for a course, resolved from the registry.

The step that was missing from the request flow. Previously a fine-tuned request
carried a question and nothing else, the service loaded whatever single adapter
had last been promoted, and "which model answered this?" had no answer anywhere
in the system. With two courses trained, that is not a gap in reporting — it is
CSS 350 being answered by CSS 360's adapter with no way to tell.

PostgreSQL is the system of record for what a course's model is, so the
resolution happens here: `course_models.current_version` names the version, the
version row says whether it is `ready`, and the resolved version travels with the
request to the cluster. The cluster then serves that version, and the response
says which one it used.

`deployment` is deliberately not consulted. `ready` means a usable artifact
exists; `deployment` means something is currently serving it. A request that
arrives while a serving session is up should be answered from the ready model
whether or not anyone remembered to flip a deployment flag — and a request that
arrives with nothing serving fails at the connection, which is a truthful error
rather than a stale one.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from app import db_models
from app.course_id import assert_valid_course_id
from app.db import db_connection, translate_db_errors


class NoReadyCourseModel(HTTPException):
    """This course has no fine-tuned model to answer with.

    409 rather than 404: the course exists and the route is right. What is
    missing is a trained model, which is a state the professor's own page
    already describes and an admin can act on.
    """

    def __init__(self, course_id: str, detail: str) -> None:
        super().__init__(status_code=409, detail=detail)
        self.course_id = course_id


def resolve_current_course_model(course_id: str) -> dict[str, Any]:
    """The version this course's fine-tuned answers must come from.

    Raises `NoReadyCourseModel` when there is none. Deliberately not a fallback
    to some other course's model or to the base model: a fine-tuned answer that
    is not from this course's fine-tuned model is not the thing that was asked
    for, and substituting one silently is how a demo ends up showing the wrong
    course's syllabus.
    """
    safe_course_id = assert_valid_course_id(course_id)

    with translate_db_errors("reading the course model registry"):
        with db_connection() as connection:
            registry = db_models.get_model_registry(connection, safe_course_id)

    if not registry:
        raise NoReadyCourseModel(
            safe_course_id,
            f'Course "{safe_course_id}" has no fine-tuned model yet. Train one '
            "before asking the fine-tuned model a question.",
        )

    current_version = registry.get("currentVersion")
    versions = registry.get("versions") or {}
    version_record = versions.get(current_version)

    if not isinstance(version_record, dict):
        raise NoReadyCourseModel(
            safe_course_id,
            f'Course "{safe_course_id}" points at model version '
            f'"{current_version}", which is not registered. Re-register or '
            "re-point the course's current version.",
        )

    if version_record.get("status") != "ready":
        raise NoReadyCourseModel(
            safe_course_id,
            f'Course "{safe_course_id}" has no ready fine-tuned model: version '
            f'"{current_version}" is "{version_record.get("status")}".',
        )

    return {
        "courseId": safe_course_id,
        "version": current_version,
        "baseModel": version_record.get("baseModel"),
        "artifactRef": version_record.get("artifactRef"),
        "deployment": version_record.get("deployment"),
    }
