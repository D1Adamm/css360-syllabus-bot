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

Ready and published are different questions, and inference asks the second
-----------------------------------------------------------------------
`status = ready` means a usable adapter exists somewhere. `deployment = online`
means that adapter has been copied into the cluster's serving tree and can
actually be loaded. Only the second one can answer a question.

This module used to resolve from `current_version` alone and say so: at the
time, `deployment` was a field nothing ever wrote, so trusting it would have
meant refusing every course. That reasoning stopped holding the moment a course
could have two versions.

The failure it produced is exact. A finished run registers `v2` and moves
`current_version` to it, because `current_version` is what a professor is shown
and their newest model is the honest answer to "what is my model". Publishing
`v2` to the cluster is a separate, deliberate act. In between, this resolved
`v2`, sent `v2` to the cluster, and the cluster — holding only the `v1` an
operator had published — refused with a 409. Every fine-tuned answer for that
course failed, including the ones `v1` had been serving correctly a moment
earlier. Training a new version took the old one offline.

So resolution prefers the published version, and falls back to `current_version`
only when this course has never had a publication reported. The fallback is what
keeps a course that predates publication reporting working exactly as it did;
once anything has been published for a course, the answer is exact.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException

from app import db_models
from app.course_id import assert_valid_course_id
from app.db import db_connection, translate_db_errors

logger = logging.getLogger(__name__)

#: The one sentence a browser is told when a course cannot be answered by a
#: fine-tuned model. The routes that raise it need no credential and are asked
#: from the student Compare page, so this is student-facing copy: it names no
#: version, no status, and no action, because a student can take none. What was
#: actually wrong goes to the backend log instead (see `NoReadyCourseModel`).
PUBLIC_UNAVAILABLE_DETAIL = "A fine-tuned model is not available for this course yet."


class NoReadyCourseModel(HTTPException):
    """This course has no fine-tuned model to answer with.

    409 rather than 404: the course exists and the route is right. What is
    missing is a trained model, which is a state the professor's own page
    already describes and an admin can act on.

    `detail` — the HTTP body — is always `PUBLIC_UNAVAILABLE_DETAIL`. The
    specific reason is kept on `diagnostic` for the log and for tests. It used
    to be the body itself, which put "Train one before asking the fine-tuned
    model a question" in front of students, who cannot train anything.
    """

    def __init__(self, course_id: str, diagnostic: str) -> None:
        super().__init__(status_code=409, detail=PUBLIC_UNAVAILABLE_DETAIL)
        self.course_id = course_id
        self.diagnostic = diagnostic


def _unavailable(course_id: str, diagnostic: str) -> NoReadyCourseModel:
    """Log the operator-facing reason, and return the student-facing refusal."""
    logger.warning(
        "Fine-tuned model unavailable for course %s: %s", course_id, diagnostic
    )
    return NoReadyCourseModel(course_id, diagnostic)


def select_servable_version(registry: dict[str, Any]) -> tuple[str | None, str]:
    """Which version a request should be answered from, and why that one.

    The published version wins. It is the only one the cluster can actually
    load, and a newer registered version that has not been copied there yet is
    an artifact, not an answer.

    `current_version` is the fallback, for exactly one situation: a course that
    has never had a publication reported. That is every course before this
    reporting existed, and the fallback is what keeps them answering the way
    they always did. It is not a general safety net — once a course has
    published anything, the published version is the answer, including when a
    newer version has since been registered.

    Returned with its source so a caller can say which rule applied. Pure, so
    the lifecycle is testable without a database.
    """
    versions = registry.get("versions") or {}

    published = [
        version
        for version, record in versions.items()
        if isinstance(record, dict) and record.get("deployment") == "online"
    ]
    if published:
        # Deterministic when more than one row claims to be online, which
        # `mark_version_published` prevents but an older write may not have.
        published.sort(key=_version_sort_key)
        return published[-1], "published"

    current_version = registry.get("currentVersion")
    if isinstance(current_version, str) and current_version:
        return current_version, "current"

    return None, "none"


def _version_sort_key(version: str) -> tuple[int, str]:
    """`v10` after `v9`, and anything unrecognised last but stable."""
    text = str(version)
    if text.startswith("v") and text[1:].isdigit():
        return (int(text[1:]), text)
    return (10**9, text)


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
        raise _unavailable(
            safe_course_id,
            f'Course "{safe_course_id}" has no fine-tuned model yet. Train one '
            "before asking the fine-tuned model a question.",
        )

    versions = registry.get("versions") or {}
    resolved_version, source = select_servable_version(registry)

    if resolved_version is None:
        raise _unavailable(
            safe_course_id,
            f'Course "{safe_course_id}" has no model version to answer with. '
            "Register one, or publish an existing one to the cluster.",
        )

    version_record = versions.get(resolved_version)

    if not isinstance(version_record, dict):
        raise _unavailable(
            safe_course_id,
            f'Course "{safe_course_id}" points at model version '
            f'"{resolved_version}", which is not registered. Re-register or '
            "re-point the course's current version.",
        )

    if version_record.get("status") != "ready":
        raise _unavailable(
            safe_course_id,
            f'Course "{safe_course_id}" has no ready fine-tuned model: version '
            f'"{resolved_version}" is "{version_record.get("status")}".',
        )

    return {
        "courseId": safe_course_id,
        "version": resolved_version,
        "baseModel": version_record.get("baseModel"),
        "artifactRef": version_record.get("artifactRef"),
        "deployment": version_record.get("deployment"),
        "currentVersion": registry.get("currentVersion"),
        "resolvedFrom": source,
    }
