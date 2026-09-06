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
from app.db import db_connection, translate_db_errors
from app.db_schemas import (
    CourseActivityResponse,
    CourseCreateRequest,
    CourseListResponse,
    CourseRecord,
    CourseUpdateRequest,
    DeleteResponse,
    EvaluationCreateRequest,
    EvaluationListResponse,
    EvaluationRecordModel,
    ModelRegistryResponse,
    ModelRequestCreateRequest,
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
    memberships for a professor, the one joined course for a participant."""
    if principal.is_anonymous:
        raise HTTPException(status_code=401, detail=JOIN_DETAIL)
    scope = principal.staff_course_ids()
    visible: set[str] | None
    if scope is None:
        visible = None
    else:
        visible = set(scope)
        if principal.participant is not None:
            visible.add(principal.participant.course_id)
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
        return db_seeds.review_seed(
            connection,
            safe_course_id,
            seed_id,
            review_status=status,
            question=request.question,
            answer=request.answer,
            review_notes=request.review_notes,
        )

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
