"""The application's persistence routes, under `/api/db`.

PostgreSQL is the system of record for everything the browser reads or writes,
and this is what it talks to. The `/api/db` prefix is kept rather than
collapsed into `/api/courses`: the frontend wrappers, the Nginx config and the
deployed `VITE_API_BASE_URL` all compose these paths today, and renaming them
would be a deployment change dressed up as a cleanup.

Route bodies stay thin: validate, open one connection, call repositories, map
"not found" to 404. Each request runs inside one transaction, so a route that
touches two tables either lands both or neither.

Every route names its guard in its signature (`app.auth.dependencies`). The
course in the path is what the guard authorizes; nothing in a body can name
a different one. Participants reach the read side of their own course and
the writes the student flow needs; course staff reach the course they hold a
membership in; the training queue, bulk deletion of research data and
operator corrections are administrator-only.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from app import db_admin_actions, db_courses, db_evaluations, db_memberships
from app import db_model_requests, db_models
from app import db_seeds, db_serving_sessions, db_training_runs
from app import provenance_privacy
from app.auth.dependencies import (
    JOIN_DETAIL,
    current_principal,
    require_admin,
    require_course_access,
    require_course_staff,
    require_participant,
    require_user,
)
from app.auth.principal import Principal
from app.student_visibility import contribution_payload, seeds_for_participant
from app.course_id import assert_valid_course_id
from app.course_model_resolution import assert_valid_model_version
from app.db import db_connection, translate_db_errors
from app.finetuned_client import check_finetuned_service_health
from app.db_schemas import (
    CourseActivityResponse,
    CourseCreateRequest,
    CourseListResponse,
    CourseRecord,
    CourseUpdateRequest,
    DeleteResponse,
    EvaluationCreateRequest,
    EvaluationListResponse,
    EvaluationPreviewResponse,
    EvaluationRecordModel,
    ModelRegistryResponse,
    ModelRequestCreateRequest,
    ModelVersionActivationResponse,
    ModelRequestRecord,
    ModelRequestUpdateRequest,
    SeedCreateRequest,
    SeedListResponse,
    SeedResponse,
    SeedUpdateRequest,
    StarterSeedGenerationResponse,
    StarterSeedGenerationUpdateRequest,
    TrainingRunCreateRequest,
    TrainingRunListResponse,
    TrainingRunRecord,
    TrainingRunUpdateRequest,
)
from app.seed_review import REVIEW_STATUSES
from app.schemas import SeedReviewRequest

router = APIRouter(prefix="/api/db", tags=["postgresql"])


def _safe_course_id(course_id: str) -> str:
    """Validate before any statement runs.

    Not what keeps SQL safe — every value below is a bound parameter — but it
    keeps a malformed id from reaching the database at all, and keeps the API,
    the export directories, and the training queue agreeing on what a course id
    is.
    """
    try:
        return assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _patch_fields(request: Any) -> dict[str, Any]:
    """Only the fields the caller actually sent, by API alias.

    `exclude_unset` is what makes PATCH a merge: a body that omits `notes` must
    leave the stored note alone, while a body that sends `notes: null` clears
    it. Dumping defaults instead would overwrite every unmentioned column.
    """
    return request.model_dump(by_alias=True, exclude_unset=True)


def _run(action: str, work: Callable[[Any], Any]) -> Any:
    """One connection, one transaction, driver errors mapped to 503."""
    with translate_db_errors(action):
        with db_connection() as connection:
            return work(connection)


def _student_view(principal: Principal, course_id: str) -> str | None:
    """The participant id when this request should get the student's view.

    A professor who also joined their own course as a student still gets the
    staff view — the projection exists to keep review detail and classmates'
    drafts from students, not from the instructor.
    """
    participant = principal.participant_for(course_id)
    if participant is None or principal.can_staff_course(course_id):
        return None
    return participant.participant_id


def _contributing_participant(principal: Principal, course_id: str) -> str | None:
    """The participant a contribution is attributed to, staff or not.

    Holding a participant session for a course means having walked in through
    its classroom code, and a contribution made while holding one is a student
    contribution — including an instructor's, when they try the flow their
    students will use. Without one, staff write seeds as staff.
    """
    participant = principal.participant_for(course_id)
    return participant.participant_id if participant is not None else None


def _mark_mine(seeds: list[dict[str, Any]], participant_id: str) -> list[dict[str, Any]]:
    """Flag the rows a staff member contributed as a participant, without
    projecting anything away: they still get the full record."""
    return [
        {**seed, "mine": True} if seed.get("participantId") == participant_id else seed
        for seed in seeds
    ]


# --------------------------------------------------------------------------- #
# Courses
# --------------------------------------------------------------------------- #


@router.get("/courses", response_model=CourseListResponse)
def list_courses(principal: Principal = Depends(current_principal)) -> CourseListResponse:
    """The courses this principal may open: all for an administrator, the
    memberships for a professor, every course this browser joined as a
    participant."""
    if principal.is_anonymous:
        raise HTTPException(status_code=401, detail=JOIN_DETAIL)
    scope = principal.staff_course_ids()
    visible: set[str] | None
    if scope is None:
        visible = None
    else:
        visible = set(scope) | principal.participant_course_ids()
    courses = _run(
        "listing courses", lambda connection: db_courses.list_courses(connection, visible)
    )
    return CourseListResponse(
        count=len(courses),
        courses=[CourseRecord(**course) for course in courses],
    )


@router.get("/courses/{course_id}", response_model=CourseRecord)
def get_course(course_id: str, principal: Principal = Depends(require_course_access)) -> CourseRecord:
    safe_course_id = _safe_course_id(course_id)
    course = _run(
        "reading course metadata",
        lambda connection: db_courses.get_course(connection, safe_course_id),
    )
    if course is None:
        raise HTTPException(
            status_code=404, detail=f'Course "{safe_course_id}" was not found.'
        )
    return CourseRecord(**course)


@router.post("/courses", response_model=CourseRecord, status_code=201)
def create_course(request: CourseCreateRequest, principal: Principal = Depends(require_user)) -> CourseRecord:
    safe_course_id = _safe_course_id(request.course_id)
    metadata = request.model_dump(by_alias=True)
    metadata.pop("courseId", None)
    creator = principal.user
    assert creator is not None

    def work(connection: Any) -> dict[str, Any]:
        created = db_courses.create_course(
            connection, safe_course_id, metadata, created_by=creator.user_id
        )
        # A professor who creates a course is its instructor from the first
        # moment; administrators need no membership to reach it.
        if not creator.is_admin:
            db_memberships.add_membership(
                connection,
                course_id=safe_course_id,
                user_id=creator.user_id,
                granted_by=creator.user_id,
            )
        db_admin_actions.record_action(
            connection,
            actor_user_id=creator.user_id,
            actor_role=creator.role,
            action="course.create",
            target_kind="course",
            target_id=safe_course_id,
            course_id=safe_course_id,
        )
        return created

    try:
        created = _run("creating a course", work)
    except db_courses.CourseAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return CourseRecord(**created)


@router.patch("/courses/{course_id}", response_model=CourseRecord)
def update_course(course_id: str, request: CourseUpdateRequest, principal: Principal = Depends(require_course_staff)) -> CourseRecord:
    safe_course_id = _safe_course_id(course_id)
    patch = _patch_fields(request)

    updated = _run(
        "updating course metadata",
        lambda connection: db_courses.update_course(connection, safe_course_id, patch),
    )
    if updated is None:
        raise HTTPException(
            status_code=404, detail=f'Course "{safe_course_id}" was not found.'
        )
    return CourseRecord(**updated)


@router.get(
    "/courses/{course_id}/starter-seed-generation",
    response_model=StarterSeedGenerationResponse,
)
def get_starter_seed_generation(course_id: str, principal: Principal = Depends(require_course_staff)) -> StarterSeedGenerationResponse:
    safe_course_id = _safe_course_id(course_id)
    record = _run(
        "reading starter seed generation state",
        lambda connection: db_courses.get_starter_seed_generation(
            connection, safe_course_id
        ),
    )
    return StarterSeedGenerationResponse(
        courseId=safe_course_id,
        starterSeedGeneration=record,
    )


@router.patch(
    "/courses/{course_id}/starter-seed-generation",
    response_model=StarterSeedGenerationResponse,
)
def update_starter_seed_generation(
    course_id: str,
    request: StarterSeedGenerationUpdateRequest,
    principal: Principal = Depends(require_admin),
) -> StarterSeedGenerationResponse:
    """Merge starter-generation state.

    The generation job writes this through `app/starter_status.py` rather than
    over HTTP; this route exists for admin correction and for reading a course
    back after a run.
    """
    safe_course_id = _safe_course_id(course_id)
    patch = _patch_fields(request)

    def work(connection: Any) -> dict[str, Any] | None:
        if not db_courses.course_exists(connection, safe_course_id):
            return None
        return db_courses.upsert_starter_seed_generation(
            connection, safe_course_id, patch
        )

    record = _run("updating starter seed generation state", work)
    if record is None and patch:
        raise HTTPException(
            status_code=404, detail=f'Course "{safe_course_id}" was not found.'
        )
    return StarterSeedGenerationResponse(
        courseId=safe_course_id,
        starterSeedGeneration=record,
    )


# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #


@router.get("/courses/{course_id}/seeds", response_model=SeedListResponse)
def list_course_seeds(course_id: str, principal: Principal = Depends(require_course_access)) -> SeedListResponse:
    safe_course_id = _safe_course_id(course_id)

    student = _student_view(principal, safe_course_id)

    def work(connection: Any) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
        return (
            db_seeds.list_seeds(connection, safe_course_id),
            db_seeds.count_seeds_by_review_status(connection, safe_course_id),
            db_seeds.count_seeds_by_origin(connection, safe_course_id),
        )

    seeds, counts, origins = _run("listing course seeds", work)
    if student is not None:
        seeds = seeds_for_participant(seeds, student)
    else:
        contributor = _contributing_participant(principal, safe_course_id)
        if contributor is not None:
            seeds = _mark_mine(seeds, contributor)
    return SeedListResponse(
        courseId=safe_course_id,
        count=len(seeds),
        seeds=seeds,
        reviewStatusCounts=counts,
        originCounts=origins,
    )


@router.get("/courses/{course_id}/seeds/{seed_id}", response_model=SeedResponse)
def get_course_seed(course_id: str, seed_id: str, principal: Principal = Depends(require_course_staff)) -> SeedResponse:
    safe_course_id = _safe_course_id(course_id)
    seed = _run(
        "reading a seed",
        lambda connection: db_seeds.get_seed(connection, safe_course_id, seed_id),
    )
    if seed is None:
        raise HTTPException(status_code=404, detail=f'Seed "{seed_id}" was not found.')
    return SeedResponse(courseId=safe_course_id, seedId=seed_id, seed=seed)


@router.post(
    "/courses/{course_id}/seeds", response_model=SeedResponse, status_code=201
)
def create_course_seed(course_id: str, request: SeedCreateRequest, principal: Principal = Depends(require_course_access)) -> SeedResponse:
    safe_course_id = _safe_course_id(course_id)
    payload = _patch_fields(request)
    student = _student_view(principal, safe_course_id)
    contributor = _contributing_participant(principal, safe_course_id)
    if contributor is not None:
        # A contribution: the student chooses the question and answer, the
        # server decides everything about its provenance and review state.
        # Staff who joined their own course contribute the same way.
        payload = contribution_payload(payload)

    def work(connection: Any) -> dict[str, Any]:
        if not db_courses.course_exists(connection, safe_course_id):
            raise HTTPException(
                status_code=404, detail=f'Course "{safe_course_id}" was not found.'
            )
        return db_seeds.create_seed(
            connection, safe_course_id, payload, participant_id=contributor
        )

    try:
        created = _run("creating a seed", work)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if student is not None:
        created = seeds_for_participant([created], student)[0]
    elif contributor is not None:
        created = _mark_mine([created], contributor)[0]
    return SeedResponse(
        courseId=safe_course_id, seedId=created["id"], seed=created
    )


@router.patch("/courses/{course_id}/seeds/{seed_id}", response_model=SeedResponse)
def update_course_seed(
    course_id: str,
    seed_id: str,
    request: SeedUpdateRequest,
    principal: Principal = Depends(require_course_staff),
) -> SeedResponse:
    safe_course_id = _safe_course_id(course_id)
    patch = _patch_fields(request)

    updated = _run(
        "updating a seed",
        lambda connection: db_seeds.update_seed(
            connection, safe_course_id, seed_id, patch
        ),
    )
    if updated is None:
        raise HTTPException(status_code=404, detail=f'Seed "{seed_id}" was not found.')
    return SeedResponse(courseId=safe_course_id, seedId=seed_id, seed=updated)


@router.post(
    "/courses/{course_id}/seeds/{seed_id}/review", response_model=SeedResponse
)
def review_course_seed(
    course_id: str,
    seed_id: str,
    request: SeedReviewRequest,
    principal: Principal = Depends(require_course_staff),
) -> SeedResponse:
    """Approve, reject, or edit one seed.

    Reuses `SeedReviewRequest` and `apply_seed_review`, the same helper the
    `/api/courses/{courseId}/seeds/{seedId}/review` route uses, so the two
    validate and record provenance identically.
    """
    safe_course_id = _safe_course_id(course_id)
    status = request.review_status.strip().lower()
    if status not in REVIEW_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=f"reviewStatus must be one of {sorted(REVIEW_STATUSES)}.",
        )

    def work(connection: Any) -> dict[str, Any] | None:
        stored = db_seeds.review_seed(
            connection,
            safe_course_id,
            seed_id,
            review_status=status,
            question=request.question,
            answer=request.answer,
            review_notes=request.review_notes,
        )
        # Same rule as the operational review route: an administrator acting
        # for a course's instructors is audited, a professor on their own
        # course is not.
        if stored is not None and principal.is_admin and principal.user is not None:
            db_admin_actions.record_seed_review(
                connection,
                actor_user_id=principal.user.user_id,
                actor_role=principal.user.role,
                course_id=safe_course_id,
                seed_id=seed_id,
                review_status=status,
                text_edited=request.question is not None or request.answer is not None,
                notes_changed=request.review_notes is not None,
            )
        return stored

    try:
        updated = _run("reviewing a seed", work)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if updated is None:
        raise HTTPException(status_code=404, detail=f'Seed "{seed_id}" was not found.')
    return SeedResponse(courseId=safe_course_id, seedId=seed_id, seed=updated)


@router.delete("/courses/{course_id}/seeds/{seed_id}", response_model=DeleteResponse)
def delete_course_seed(course_id: str, seed_id: str, principal: Principal = Depends(require_course_access)) -> DeleteResponse:
    safe_course_id = _safe_course_id(course_id)
    student = _student_view(principal, safe_course_id)

    def work(connection: Any) -> bool:
        if student is not None:
            # A participant removes only what they contributed. Anything else
            # in the course is not theirs to delete, approved or not.
            existing = db_seeds.get_seed(connection, safe_course_id, seed_id)
            if existing is None:
                return False
            if existing.get("participantId") != student:
                raise HTTPException(
                    status_code=403, detail="You can only remove questions you added."
                )
        return db_seeds.delete_seed(connection, safe_course_id, seed_id)

    deleted = _run("deleting a seed", work)
    if not deleted:
        raise HTTPException(status_code=404, detail=f'Seed "{seed_id}" was not found.')
    return DeleteResponse(courseId=safe_course_id, deleted=1)


# --------------------------------------------------------------------------- #
# Evaluations
# --------------------------------------------------------------------------- #


@router.get("/courses/{course_id}/evaluations", response_model=EvaluationListResponse)
def list_course_evaluations(course_id: str, principal: Principal = Depends(require_course_access)) -> EvaluationListResponse:
    """Staff see the course's ratings; a participant sees only their own."""
    safe_course_id = _safe_course_id(course_id)
    student = _student_view(principal, safe_course_id)
    evaluations = _run(
        "listing evaluations",
        lambda connection: db_evaluations.list_evaluations(
            connection, safe_course_id, participant_id=student
        ),
    )
    if student is not None:
        evaluations = [_without_participant(record) for record in evaluations]
    return EvaluationListResponse(
        courseId=safe_course_id,
        count=len(evaluations),
        evaluations=evaluations,
    )


def _without_participant(record: dict[str, Any]) -> dict[str, Any]:
    """A participant's own record, minus the id they have no use for."""
    return {key: value for key, value in record.items() if key != "participantId"}


@router.post(
    "/courses/{course_id}/evaluations",
    response_model=EvaluationRecordModel,
    status_code=201,
)
def create_course_evaluation(
    course_id: str,
    request: EvaluationCreateRequest,
    principal: Principal = Depends(require_participant),
) -> EvaluationRecordModel:
    """Record a rating, attributed to the session's participant.

    The participant comes from the cookie, never the body, and the id is
    allocated here: a client cannot choose either.
    """
    safe_course_id = _safe_course_id(course_id)
    payload = request.model_dump(by_alias=True, exclude_unset=True)
    payload.pop("id", None)
    participant = principal.participant_for(safe_course_id)
    assert participant is not None  # require_participant guarantees it

    def work(connection: Any) -> dict[str, Any]:
        if not db_courses.course_exists(connection, safe_course_id):
            raise HTTPException(
                status_code=404, detail=f'Course "{safe_course_id}" was not found.'
            )
        return db_evaluations.create_evaluation(
            connection,
            safe_course_id,
            payload,
            participant_id=participant.participant_id,
        )

    try:
        created = _run("creating an evaluation", work)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return EvaluationRecordModel(**_without_participant(created))


@router.post(
    "/courses/{course_id}/evaluations/preview",
    response_model=EvaluationPreviewResponse,
)
def preview_course_evaluation(
    course_id: str,
    request: EvaluationCreateRequest,
    principal: Principal = Depends(require_admin),
) -> EvaluationPreviewResponse:
    """An administrator's preview of the student flow, ending in no write.

    An administrator opening a course's student pages holds no participant
    session there — they never redeemed its classroom code — so the route above
    refuses them, and rightly: a rating with nobody to attribute it to is not
    research data. This route lets them finish the flow anyway. Same body,
    same validation, same 404 for a course that does not exist; the rating
    comes back the way it would have been stored, and nothing reaches the
    evaluations table. No participant is created, no cookie is set, and the id
    is marked `preview-` so it cannot be mistaken for a real one.

    The course is the one in the path, as everywhere in this file; a
    `courseId` in the body is ignored. Administrator-only, deliberately: a
    professor previewing their own course already has a way — its classroom
    code, and a real participant.
    """
    safe_course_id = _safe_course_id(course_id)
    payload = request.model_dump(by_alias=True, exclude_unset=True)
    payload.pop("id", None)

    def work(connection: Any) -> dict[str, Any]:
        if not db_courses.course_exists(connection, safe_course_id):
            raise HTTPException(
                status_code=404, detail=f'Course "{safe_course_id}" was not found.'
            )
        return db_evaluations.preview_evaluation(safe_course_id, payload)

    try:
        previewed = _run("previewing an evaluation", work)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return EvaluationPreviewResponse(
        courseId=safe_course_id, evaluation=EvaluationRecordModel(**previewed)
    )


@router.get("/courses/{course_id}/activity", response_model=CourseActivityResponse)
def get_course_activity(
    course_id: str, principal: Principal = Depends(require_course_access)
) -> CourseActivityResponse:
    """Class-wide counts for the student home page.

    Counts only. The home page used to download every evaluation — comments
    included — to show a number; a participant now never receives another
    student's rating at all.
    """
    safe_course_id = _safe_course_id(course_id)

    def work(connection: Any) -> tuple[int, int]:
        origins = db_seeds.count_seeds_by_origin(connection, safe_course_id)
        return origins.get("user", 0), db_evaluations.count_evaluations(
            connection, safe_course_id
        )

    contributed, evaluations = _run("reading course activity", work)
    return CourseActivityResponse(
        courseId=safe_course_id,
        contributedQuestions=contributed,
        evaluations=evaluations,
    )


@router.delete(
    "/courses/{course_id}/evaluations/{evaluation_id}", response_model=DeleteResponse
)
def delete_course_evaluation(course_id: str, evaluation_id: str, principal: Principal = Depends(require_admin)) -> DeleteResponse:
    safe_course_id = _safe_course_id(course_id)
    actor = principal.user
    assert actor is not None

    def work(connection: Any) -> bool:
        deleted = db_evaluations.delete_evaluation(connection, safe_course_id, evaluation_id)
        if deleted:
            db_admin_actions.record_action(
                connection,
                actor_user_id=actor.user_id,
                actor_role=actor.role,
                action="evaluation.delete",
                target_kind="evaluation",
                target_id=evaluation_id,
                course_id=safe_course_id,
            )
        return deleted

    deleted = _run("deleting an evaluation", work)
    if not deleted:
        raise HTTPException(
            status_code=404, detail=f'Evaluation "{evaluation_id}" was not found.'
        )
    return DeleteResponse(courseId=safe_course_id, deleted=1)


@router.delete("/courses/{course_id}/evaluations", response_model=DeleteResponse)
def delete_all_course_evaluations(course_id: str, principal: Principal = Depends(require_admin)) -> DeleteResponse:
    """Clear one course's evaluations. Administrator-only, and audited: this is
    research data."""
    safe_course_id = _safe_course_id(course_id)
    actor = principal.user
    assert actor is not None

    def work(connection: Any) -> int:
        deleted = db_evaluations.delete_all_evaluations(connection, safe_course_id)
        db_admin_actions.record_action(
            connection,
            actor_user_id=actor.user_id,
            actor_role=actor.role,
            action="evaluations.clear",
            target_kind="course",
            target_id=safe_course_id,
            course_id=safe_course_id,
            detail={"deleted": deleted},
        )
        return deleted

    deleted = _run("clearing evaluations", work)
    return DeleteResponse(courseId=safe_course_id, deleted=deleted)


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #


@router.get("/courses/{course_id}/model", response_model=ModelRegistryResponse)
def get_course_model(course_id: str, principal: Principal = Depends(require_course_staff)) -> ModelRegistryResponse:
    safe_course_id = _safe_course_id(course_id)
    registry = _run(
        "reading the model registry",
        lambda connection: db_models.get_model_registry(connection, safe_course_id),
    )
    if registry is None:
        raise HTTPException(
            status_code=404,
            detail=f'Course "{safe_course_id}" has no registered model.',
        )
    # Stored provenance keeps the cluster's exact run directories; the browser
    # does not get them. See `provenance_privacy`.
    return ModelRegistryResponse(**provenance_privacy.public_model_registry(registry))


def _require_ready_version(registry: dict[str, Any] | None, course_id: str, version: str) -> None:
    """Refuse a version this course cannot serve, before or after the VM is asked."""
    record = ((registry or {}).get("versions") or {}).get(version)
    if not isinstance(record, dict):
        raise HTTPException(
            status_code=404,
            detail=f'Course "{course_id}" has no registered model version "{version}".',
        )
    if record.get("status") != "ready":
        raise HTTPException(
            status_code=409,
            detail=(
                f'Version "{version}" is "{record.get("status")}", not ready. '
                "Only a ready version can be activated."
            ),
        )


def _servable_versions(health: dict[str, Any], course_id: str) -> list[str]:
    """The versions the fine-tuned service says it can answer this course with."""
    for entry in health.get("courses") or []:
        if isinstance(entry, dict) and entry.get("courseId") == course_id:
            versions = entry.get("versions")
            return [v for v in versions if isinstance(v, str)] if isinstance(versions, list) else []
    return []


@router.post(
    "/courses/{course_id}/model-versions/{version}/activate",
    response_model=ModelVersionActivationResponse,
)
async def activate_course_model_version(
    course_id: str,
    version: str,
    principal: Principal = Depends(require_admin),
) -> ModelVersionActivationResponse:
    """Make one registered version the one this course's fine-tuned answers use.

    The explicit step between "training registered a version" and "students
    get it". Inference resolves the published version, so this writes exactly
    what Tillicum's `promote_qlora_adapter.sh` writes — `mark_version_published`
    — but only after the fine-tuned service itself says it can answer this
    course at this version. On the VM that means the version is mapped in
    `FINETUNED_OLLAMA_MODELS` and Ollama has the model. A registered version
    that has not been installed and mapped is refused with 409 and nothing is
    written, so activation can never point production at a version the
    service would refuse. If the service cannot be asked, nothing is written
    either: an unverifiable activation is not one.

    Rollback is the same call for an older version, under the same check.
    Activating the version that is already active is a no-op reported as
    `unchanged`. Every change is written to `admin_actions`.

    Registration never does this. A finished training run lands `offline`, and
    ordinary registration refuses `deployment=online`.
    """
    safe_course_id = _safe_course_id(course_id)
    try:
        safe_version = assert_valid_model_version(version)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    actor = principal.user
    assert actor is not None  # require_admin

    def read_registry(connection: Any) -> dict[str, Any] | None:
        return db_models.get_model_registry(connection, safe_course_id)

    # Checked first so a typo or an unready version is refused without
    # depending on the service being up.
    registry = await run_in_threadpool(_run, "reading the model registry", read_registry)
    _require_ready_version(registry, safe_course_id, safe_version)

    try:
        health = await check_finetuned_service_health()
    except HTTPException as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not confirm that the fine-tuned service can serve "
                f"{safe_course_id} {safe_version}, so nothing was activated. "
                f"{exc.detail}"
            ),
        ) from exc

    servable = _servable_versions(health, safe_course_id)
    if safe_version not in servable:
        raise HTTPException(
            status_code=409,
            detail=(
                f"The fine-tuned service cannot serve {safe_course_id} "
                f"{safe_version}: it is not mapped, or its Ollama model is "
                "missing. Install and map it on the VM "
                "(scripts/install_finetuned_adapter.py, then "
                "scripts/aiswe_finetuned.sh set-mapping and restart), then "
                "activate it. Servable now: "
                f"{', '.join(servable) if servable else 'none'}."
            ),
        )

    def activate(connection: Any) -> tuple[dict[str, Any], str | None, bool]:
        # Re-read inside the write's transaction: the registry may have moved
        # while the service was being asked.
        current = db_models.get_model_registry(connection, safe_course_id)
        _require_ready_version(current, safe_course_id, safe_version)
        versions = (current or {}).get("versions") or {}
        already = versions[safe_version].get("deployment") == "online"
        previous = next(
            (
                key
                for key, record in versions.items()
                if key != safe_version
                and isinstance(record, dict)
                and record.get("deployment") == "online"
            ),
            None,
        )
        if already and previous is None:
            return current, None, True
        updated = db_models.mark_version_published(connection, safe_course_id, safe_version)
        db_admin_actions.record_action(
            connection,
            actor_user_id=actor.user_id,
            actor_role=actor.role,
            action="model.activate",
            target_kind="model_version",
            target_id=f"{safe_course_id}@{safe_version}",
            course_id=safe_course_id,
            detail={"version": safe_version, "previousVersion": previous},
        )
        return updated or current, previous, False

    updated, previous, unchanged = await run_in_threadpool(
        _run, "activating a model version", activate
    )
    return ModelVersionActivationResponse(
        courseId=safe_course_id,
        version=safe_version,
        previousVersion=previous,
        unchanged=unchanged,
        model=ModelRegistryResponse(**provenance_privacy.public_model_registry(updated)),
    )


# --------------------------------------------------------------------------- #
# Model requests
# --------------------------------------------------------------------------- #


@router.get("/courses/{course_id}/model-request", response_model=ModelRequestRecord)
def get_course_model_request(course_id: str, principal: Principal = Depends(require_course_staff)) -> ModelRequestRecord:
    safe_course_id = _safe_course_id(course_id)
    request_record = _run(
        "reading the model request",
        lambda connection: db_model_requests.get_model_request(
            connection, safe_course_id
        ),
    )
    if request_record is None:
        raise HTTPException(
            status_code=404,
            detail=f'Course "{safe_course_id}" has no model request.',
        )
    return ModelRequestRecord(
        **provenance_privacy.public_model_request(request_record)
    )


@router.post(
    "/courses/{course_id}/model-request",
    response_model=ModelRequestRecord,
    status_code=201,
)
def create_course_model_request(
    course_id: str,
    request: ModelRequestCreateRequest,
    principal: Principal = Depends(require_course_staff),
) -> ModelRequestRecord:
    safe_course_id = _safe_course_id(course_id)

    def work(connection: Any) -> dict[str, Any]:
        if not db_courses.course_exists(connection, safe_course_id):
            raise HTTPException(
                status_code=404, detail=f'Course "{safe_course_id}" was not found.'
            )
        return db_model_requests.create_model_request(
            connection, safe_course_id, request.approved_example_count
        )

    try:
        created = _run("creating a model request", work)
    except db_model_requests.ActiveModelRequestError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ModelRequestRecord(**provenance_privacy.public_model_request(created))


@router.patch("/courses/{course_id}/model-request", response_model=ModelRequestRecord)
def update_course_model_request(
    course_id: str,
    request: ModelRequestUpdateRequest,
    principal: Principal = Depends(require_admin),
) -> ModelRequestRecord:
    safe_course_id = _safe_course_id(course_id)
    patch = _patch_fields(request)

    updated = _run(
        "updating the model request",
        lambda connection: db_model_requests.update_model_request(
            connection, safe_course_id, patch
        ),
    )
    if updated is None:
        raise HTTPException(
            status_code=404,
            detail=f'Course "{safe_course_id}" has no model request.',
        )
    return ModelRequestRecord(**provenance_privacy.public_model_request(updated))


# --------------------------------------------------------------------------- #
# Training runs
# --------------------------------------------------------------------------- #


@router.get(
    "/courses/{course_id}/training-runs", response_model=TrainingRunListResponse
)
def list_course_training_runs(course_id: str, principal: Principal = Depends(require_admin)) -> TrainingRunListResponse:
    safe_course_id = _safe_course_id(course_id)
    runs = _run(
        "listing training runs",
        lambda connection: db_training_runs.list_training_runs(
            connection, safe_course_id
        ),
    )
    return TrainingRunListResponse(
        courseId=safe_course_id,
        count=len(runs),
        runs=[provenance_privacy.public_training_run(run) for run in runs],
    )


@router.get(
    "/courses/{course_id}/training-runs/{run_id}", response_model=TrainingRunRecord
)
def get_course_training_run(course_id: str, run_id: str, principal: Principal = Depends(require_admin)) -> TrainingRunRecord:
    safe_course_id = _safe_course_id(course_id)
    run = _run(
        "reading a training run",
        lambda connection: db_training_runs.get_training_run(
            connection, safe_course_id, run_id
        ),
    )
    if run is None:
        raise HTTPException(
            status_code=404, detail=f'Training run "{run_id}" was not found.'
        )
    return TrainingRunRecord(**provenance_privacy.public_training_run(run))


@router.post(
    "/courses/{course_id}/training-runs",
    response_model=TrainingRunRecord,
    status_code=201,
)
def enqueue_course_training_run(
    course_id: str,
    request: TrainingRunCreateRequest,
    principal: Principal = Depends(require_admin),
) -> TrainingRunRecord:
    """Queue one run, refusing while this course already has an active one."""
    safe_course_id = _safe_course_id(course_id)

    def work(connection: Any) -> dict[str, Any]:
        if not db_courses.course_exists(connection, safe_course_id):
            raise HTTPException(
                status_code=404, detail=f'Course "{safe_course_id}" was not found.'
            )
        return db_training_runs.enqueue_training_run(
            connection,
            safe_course_id,
            mode=request.mode,
            dataset_ref=request.dataset_ref,
            approved_example_count=request.approved_example_count,
            train_examples=request.train_examples,
            validation_examples=request.validation_examples,
        )

    try:
        created = _run("queueing a training run", work)
    except db_training_runs.ActiveTrainingRunError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return TrainingRunRecord(**provenance_privacy.public_training_run(created))


@router.patch(
    "/courses/{course_id}/training-runs/{run_id}", response_model=TrainingRunRecord
)
def update_course_training_run(
    course_id: str,
    run_id: str,
    request: TrainingRunUpdateRequest,
    principal: Principal = Depends(require_admin),
) -> TrainingRunRecord:
    safe_course_id = _safe_course_id(course_id)
    patch = _patch_fields(request)

    # `clearClaim` is an explicit release. Sending `claim: null` would be
    # ambiguous with "did not mention the claim", which must leave it alone.
    if patch.pop("clearClaim", False):
        patch["claim"] = None

    updated = _run(
        "updating a training run",
        lambda connection: db_training_runs.update_training_run(
            connection, safe_course_id, run_id, patch
        ),
    )
    if updated is None:
        raise HTTPException(
            status_code=404, detail=f'Training run "{run_id}" was not found.'
        )
    return TrainingRunRecord(**provenance_privacy.public_training_run(updated))



@router.get("/serving-session")
def get_current_serving_session(principal: Principal = Depends(require_admin)) -> dict[str, Any]:
    """Whether a fine-tuned serving job is up, and until when.

    The browser-facing half of the serving session. `node` and `port` are not
    included: every route on this router is reachable without a credential, and
    a compute hostname with a listening port is the one field in that record
    that describes how to reach a machine. The worker-token route
    `/api/training-queue/serving-session` returns those to the one caller that
    needs them.

    `session: null` is a normal answer — most of the time nothing is serving,
    which is the intended resting state of a research GPU allocation.
    """
    session = _run(
        "reading the current serving session",
        lambda connection: db_serving_sessions.current_serving_session(connection),
    )
    return {"session": db_serving_sessions.public_serving_session(session)}
