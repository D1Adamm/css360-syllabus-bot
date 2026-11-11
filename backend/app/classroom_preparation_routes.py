"""Classroom preparation for one course: status, question selection, start, stop.

Course staff only: an administrator, or a professor holding a membership in
the course in the path. The professor and administrator pages call the same
four routes. Students have no route here, and selecting a question changes
nothing a student sees: it decides what is prepared, not what is suggested.

    GET  /api/courses/{course_id}/classroom-preparation            status
    PUT  /api/courses/{course_id}/classroom-preparation/questions  replace the selection
    POST /api/courses/{course_id}/classroom-preparation/start      start or resume
    POST /api/courses/{course_id}/classroom-preparation/stop       stop gracefully

Every route answers with the full status, so the page never has to guess
what a write changed.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from app import answer_cache, classroom_preparation, db_classroom_questions, db_courses
from app.auth.dependencies import require_course_staff
from app.auth.principal import Principal
from app.course_id import assert_valid_course_id
from app.db import db_connection, translate_db_errors
from app.error_classification import ClassifiedHTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/courses/{course_id}/classroom-preparation", tags=["classroom-preparation"])

#: More than any course has questions; a bound on the request body, not a target.
MAX_SELECTED_QUESTIONS = 500


class SelectionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    seed_ids: list[str] = Field(..., alias="seedIds", max_length=MAX_SELECTED_QUESTIONS)


def _safe_course_id(course_id: str) -> str:
    try:
        return assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _acting_user_id(principal: Principal) -> str | None:
    """The staff member's id when it is a real account id, else None."""
    user = getattr(principal, "user", None)
    raw = getattr(user, "user_id", None)
    try:
        return str(uuid.UUID(str(raw)))
    except (ValueError, TypeError, AttributeError):
        return None


async def _status(course_id: str) -> dict[str, Any]:
    with translate_db_errors("reading classroom preparation"):
        return await classroom_preparation.course_status(course_id)


def _require_course(course_id: str) -> None:
    with db_connection() as connection:
        if not db_courses.course_exists(connection, course_id):
            raise HTTPException(status_code=404, detail=f'Course "{course_id}" was not found.')


@router.get("")
async def get_classroom_preparation(
    course_id: str, principal: Principal = Depends(require_course_staff)
) -> dict[str, Any]:
    return await _status(_safe_course_id(course_id))


@router.put("/questions")
async def select_classroom_questions(
    course_id: str,
    request: SelectionRequest,
    principal: Principal = Depends(require_course_staff),
) -> dict[str, Any]:
    safe_course_id = _safe_course_id(course_id)
    selected_by = _acting_user_id(principal)

    def work() -> None:
        with db_connection() as connection:
            if not db_courses.course_exists(connection, safe_course_id):
                raise HTTPException(status_code=404, detail=f'Course "{safe_course_id}" was not found.')
            db_classroom_questions.replace_selection(
                connection, safe_course_id, request.seed_ids, selected_by=selected_by
            )

    try:
        with translate_db_errors("saving the classroom question selection"):
            await run_in_threadpool(work)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # Students' requests must see the new selection at once, not in 30 seconds.
    answer_cache.invalidate_selection(safe_course_id)
    return await _status(safe_course_id)


@router.post("/start")
async def start_classroom_preparation(
    course_id: str, principal: Principal = Depends(require_course_staff)
) -> dict[str, Any]:
    safe_course_id = _safe_course_id(course_id)
    try:
        with translate_db_errors("starting classroom preparation"):
            await run_in_threadpool(_require_course, safe_course_id)
            await classroom_preparation.start(safe_course_id, started_by=_acting_user_id(principal))
    except classroom_preparation.PreparationRefused as refused:
        raise ClassifiedHTTPException(
            status_code=409, code=f"preparation_{refused.code}", message=refused.message
        ) from refused
    return await _status(safe_course_id)


@router.post("/stop")
async def stop_classroom_preparation(
    course_id: str, principal: Principal = Depends(require_course_staff)
) -> dict[str, Any]:
    safe_course_id = _safe_course_id(course_id)
    await classroom_preparation.request_stop(safe_course_id)
    return await _status(safe_course_id)
