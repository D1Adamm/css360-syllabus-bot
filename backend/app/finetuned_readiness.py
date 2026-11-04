"""Can every course's activated model actually be served right now?

Audit finding B7 (2026-10-27): nothing told an operator when the fine-tuned
path was down or a course's activated version was not the one the service
could serve. Each of those was visible only as students' "temporarily
unavailable" answers, or by running the verification script with
credentials.

This module answers one question per course — is the version students are
routed to listed by the fine-tuned service's `/health`? — and reports it two
ways:

- `GET /api/admin/model-readiness` for the Admin → Overview page, on demand;
- a background check every `FINETUNED_READINESS_CHECK_SECONDS` (default 300,
  `0` disables) that writes a WARNING to the backend's journal when a problem
  appears and an INFO line when it clears. The journal is where every other
  operator signal already goes (`journalctl --user-unit aiswe-backend`), so
  this adds no infrastructure.

`assess` is pure and tested; `collect_readiness` does the reads.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Awaitable, Callable, Mapping

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from app import db_models
from app.config import is_test_mode
from app.course_model_resolution import select_servable_version
from app.db import db_connection, translate_db_errors
from app.error_classification import error_code
from app.finetuned_client import check_finetuned_service_health
from app.generation_mode import current_mode
from app.model_publication import servable_versions

logger = logging.getLogger(__name__)

DEFAULT_CHECK_SECONDS = 300.0
CHECK_SECONDS_ENV = "FINETUNED_READINESS_CHECK_SECONDS"

# Problem codes, per course. `ok` is the absence of all of them.
SERVICE_UNREACHABLE = "service_unreachable"
NOT_SERVABLE = "activated_version_not_servable"
NO_ACTIVATED_VERSION = "no_activated_version"

HealthProbe = Callable[[], Awaitable[Mapping[str, Any]]]


def readiness_check_interval() -> float:
    raw = (os.getenv(CHECK_SECONDS_ENV) or "").strip()
    if not raw:
        return DEFAULT_CHECK_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_CHECK_SECONDS
    return max(0.0, value)


def _registry_from_versions(versions: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {"versions": {str(item["version"]): dict(item) for item in versions}}


def assess(
    courses: list[Mapping[str, Any]],
    health: Mapping[str, Any] | None,
    *,
    health_error: str | None = None,
    health_error_code: str | None = None,
    mode: str = "vm",
) -> dict[str, Any]:
    """Judge every registered course against what the service says it serves.

    `courses` is `db_models.list_registered_courses`; `health` is the service's
    `/health` (None when the probe failed, with the reason in `health_error`).
    A course with no activated version is a problem of its own kind: students
    get "not available yet" for it, which may be intended, so it is reported
    rather than alarmed about.
    """
    service_ok = health is not None and str(health.get("status", "")).lower() == "ok"
    rows: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    for course in courses:
        course_id = str(course["courseId"])
        versions = list(course.get("versions") or [])
        activated, _source = select_servable_version(_registry_from_versions(versions))
        newest = str(versions[0]["version"]) if versions else None
        newest_status = str(versions[0].get("status") or "") if versions else None
        listed = servable_versions(health, course_id) if health is not None else []
        row: dict[str, Any] = {
            "courseId": course_id,
            "courseName": course.get("courseName"),
            "activatedVersion": activated,
            "newestVersion": newest,
            "newestStatus": newest_status,
            "servableVersions": listed,
            "problem": None,
            "message": None,
        }
        if activated is None:
            row["problem"] = NO_ACTIVATED_VERSION
            row["message"] = (
                "No activated version: Fine-Tuned and Fine-Tuned + RAG answer "
                "\"not available yet\" for this course."
            )
        elif health is None:
            row["problem"] = SERVICE_UNREACHABLE
            row["message"] = (
                f"Activated {activated}, but the fine-tuned service could not be "
                f"asked: {health_error or 'no answer'}."
            )
        elif activated not in listed:
            row["problem"] = NOT_SERVABLE
            row["message"] = (
                f"Activated {activated} is not servable: the fine-tuned service "
                f"lists {', '.join(listed) if listed else 'nothing'} for this course. "
                "Every Fine-Tuned and Fine-Tuned + RAG request fails until the "
                "mapping or the activation is corrected."
            )
        else:
            row["message"] = f"Activated {activated} is servable."
        rows.append(row)
        if row["problem"] and row["problem"] != NO_ACTIVATED_VERSION:
            problems.append({"courseId": course_id, "problem": row["problem"], "message": row["message"]})

    return {
        "mode": mode,
        "service": {
            "reachable": health is not None,
            "status": (health or {}).get("status") if health is not None else None,
            "ok": service_ok,
            "error": health_error,
            "errorCode": health_error_code,
        },
        "courses": rows,
        "problems": problems,
        "ok": not problems and (health is not None),
    }


def _read_registered_courses() -> list[dict[str, Any]]:
    with translate_db_errors("reading the model registry"):
        with db_connection() as connection:
            return db_models.list_registered_courses(connection)


async def collect_readiness(
    *,
    probe: HealthProbe = check_finetuned_service_health,
    read_courses: Callable[[], list[dict[str, Any]]] = _read_registered_courses,
) -> dict[str, Any]:
    """The current readiness: one registry read, one health probe."""
    courses = await run_in_threadpool(read_courses)
    health: Mapping[str, Any] | None
    error: str | None = None
    code: str | None = None
    try:
        health = await probe()
    except HTTPException as exc:
        health = None
        error = str(exc.detail)
        code = error_code(exc)
    return assess(
        courses,
        health,
        health_error=error,
        health_error_code=code,
        mode=current_mode().kind,
    )


def _problem_signature(readiness: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted((str(item["courseId"]), str(item["problem"])) for item in readiness["problems"])
    )


async def readiness_monitor(
    interval_seconds: float,
    *,
    collect: Callable[[], Awaitable[dict[str, Any]]] = collect_readiness,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    iterations: int | None = None,
) -> None:
    """Check on an interval and log only when the picture changes.

    A WARNING names every course whose activated version cannot be served (or
    the service outage behind all of them); an INFO line says when it clears.
    Nothing is repeated while nothing changes, so a quiet journal stays quiet.
    `iterations` bounds the loop for tests.
    """
    previous: tuple[tuple[str, str], ...] | None = None
    # The last reason the check itself could not run (database down, say).
    # Logged when it appears or changes, and once more when the check works
    # again — not every interval for the length of an outage.
    failing: str | None = None
    done = 0
    while iterations is None or done < iterations:
        try:
            readiness = await collect()
        except Exception as exc:  # noqa: BLE001 - the monitor must outlive one failed check
            reason = str(exc) or exc.__class__.__name__
            if reason != failing:
                logger.warning("Fine-tuned readiness check could not run: %s", reason)
                failing = reason
        else:
            if failing is not None:
                logger.info("Fine-tuned readiness check is running again.")
                failing = None
            signature = _problem_signature(readiness)
            if signature != previous:
                if signature:
                    for problem in readiness["problems"]:
                        logger.warning(
                            "Fine-tuned readiness: course %s %s: %s",
                            problem["courseId"],
                            problem["problem"],
                            problem["message"],
                        )
                elif previous:
                    logger.info("Fine-tuned readiness: every activated version is servable again.")
                previous = signature
        done += 1
        if iterations is None or done < iterations:
            await sleep(interval_seconds)


def start_readiness_monitor() -> asyncio.Task[None] | None:
    """Start the background check, or return None when it should not run.

    Not under tests, not without a database, and not when the interval is 0.
    """
    interval = readiness_check_interval()
    if is_test_mode() or interval <= 0 or not (os.getenv("DATABASE_URL") or "").strip():
        return None
    return asyncio.create_task(readiness_monitor(interval))
