"""PostgreSQL-backed routes, mounted in parallel under `/api/db`.

Every route here is new. Nothing under `/api/courses` changed: those still read
and write Firebase, which remains the system of record. The prefix is the whole
point — during the cutover it must be obvious from a URL alone which store
answered a request, and a shared path with a feature flag would not give that.

Route bodies stay thin: validate, open one connection, call repositories, map
"not found" to 404. Each request runs inside one transaction, so a route that
touches two tables either lands both or neither.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, HTTPException

from app import db_courses, db_evaluations, db_model_requests, db_models
from app import db_seeds, db_training_runs
from app.course_id import assert_valid_course_id
from app.db import db_connection, translate_db_errors
from app.db_schemas import (
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

    The same validator the Firebase paths use. It is not what keeps SQL safe —
    every value below is a bound parameter — but it keeps a malformed id from
    reaching the database at all, and keeps both stores agreeing on what a
    course id is.
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


# --------------------------------------------------------------------------- #
# Courses
# --------------------------------------------------------------------------- #


@router.get("/courses", response_model=CourseListResponse)
def list_courses() -> CourseListResponse:
    courses = _run("listing courses", db_courses.list_courses)
    return CourseListResponse(
        count=len(courses),
        courses=[CourseRecord(**course) for course in courses],
    )


@router.get("/courses/{course_id}", response_model=CourseRecord)
def get_course(course_id: str) -> CourseRecord:
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
def create_course(request: CourseCreateRequest) -> CourseRecord:
    safe_course_id = _safe_course_id(request.course_id)
    metadata = request.model_dump(by_alias=True)
    metadata.pop("courseId", None)

    def work(connection: Any) -> dict[str, Any]:
        return db_courses.create_course(connection, safe_course_id, metadata)

    try:
        created = _run("creating a course", work)
    except db_courses.CourseAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return CourseRecord(**created)


@router.patch("/courses/{course_id}", response_model=CourseRecord)
def update_course(course_id: str, request: CourseUpdateRequest) -> CourseRecord:
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
def get_starter_seed_generation(course_id: str) -> StarterSeedGenerationResponse:
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
) -> StarterSeedGenerationResponse:
    """Merge starter-generation state.

    Repository support only. The running generation job still writes Firebase
    and nothing calls this on its behalf.
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
def list_course_seeds(course_id: str) -> SeedListResponse:
    safe_course_id = _safe_course_id(course_id)

    def work(connection: Any) -> tuple[list[dict[str, Any]], dict[str, int]]:
        return (
            db_seeds.list_seeds(connection, safe_course_id),
            db_seeds.count_seeds_by_review_status(connection, safe_course_id),
        )

    seeds, counts = _run("listing course seeds", work)
    return SeedListResponse(
        courseId=safe_course_id,
        count=len(seeds),
        seeds=seeds,
        reviewStatusCounts=counts,
    )


@router.get("/courses/{course_id}/seeds/{seed_id}", response_model=SeedResponse)
def get_course_seed(course_id: str, seed_id: str) -> SeedResponse:
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
def create_course_seed(course_id: str, request: SeedCreateRequest) -> SeedResponse:
    safe_course_id = _safe_course_id(course_id)
    payload = _patch_fields(request)

    def work(connection: Any) -> dict[str, Any]:
        if not db_courses.course_exists(connection, safe_course_id):
            raise HTTPException(
                status_code=404, detail=f'Course "{safe_course_id}" was not found.'
            )
        return db_seeds.create_seed(connection, safe_course_id, payload)

    try:
        created = _run("creating a seed", work)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SeedResponse(
        courseId=safe_course_id, seedId=created["id"], seed=created
    )


@router.patch("/courses/{course_id}/seeds/{seed_id}", response_model=SeedResponse)
def update_course_seed(
    course_id: str,
    seed_id: str,
    request: SeedUpdateRequest,
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
) -> SeedResponse:
    """Approve, reject, or edit one seed.

    Reuses `SeedReviewRequest` and `apply_seed_review` so the PostgreSQL path
    validates and records provenance identically to the Firebase route.
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
def delete_course_seed(course_id: str, seed_id: str) -> DeleteResponse:
    safe_course_id = _safe_course_id(course_id)
    deleted = _run(
        "deleting a seed",
        lambda connection: db_seeds.delete_seed(connection, safe_course_id, seed_id),
    )
    if not deleted:
        raise HTTPException(status_code=404, detail=f'Seed "{seed_id}" was not found.')
    return DeleteResponse(courseId=safe_course_id, deleted=1)


# --------------------------------------------------------------------------- #
# Evaluations
# --------------------------------------------------------------------------- #


@router.get("/courses/{course_id}/evaluations", response_model=EvaluationListResponse)
def list_course_evaluations(course_id: str) -> EvaluationListResponse:
    safe_course_id = _safe_course_id(course_id)
    evaluations = _run(
        "listing evaluations",
        lambda connection: db_evaluations.list_evaluations(connection, safe_course_id),
    )
    return EvaluationListResponse(
        courseId=safe_course_id,
        count=len(evaluations),
        evaluations=evaluations,
    )


@router.post(
    "/courses/{course_id}/evaluations",
    response_model=EvaluationRecordModel,
    status_code=201,
)
def create_course_evaluation(
    course_id: str,
    request: EvaluationCreateRequest,
) -> EvaluationRecordModel:
    safe_course_id = _safe_course_id(course_id)
    payload = request.model_dump(by_alias=True, exclude_unset=True)

    def work(connection: Any) -> dict[str, Any]:
        if not db_courses.course_exists(connection, safe_course_id):
            raise HTTPException(
                status_code=404, detail=f'Course "{safe_course_id}" was not found.'
            )
        return db_evaluations.create_evaluation(connection, safe_course_id, payload)

    try:
        created = _run("creating an evaluation", work)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return EvaluationRecordModel(**created)


@router.delete(
    "/courses/{course_id}/evaluations/{evaluation_id}", response_model=DeleteResponse
)
def delete_course_evaluation(course_id: str, evaluation_id: str) -> DeleteResponse:
    safe_course_id = _safe_course_id(course_id)
    deleted = _run(
        "deleting an evaluation",
        lambda connection: db_evaluations.delete_evaluation(
            connection, safe_course_id, evaluation_id
        ),
    )
    if not deleted:
        raise HTTPException(
            status_code=404, detail=f'Evaluation "{evaluation_id}" was not found.'
        )
    return DeleteResponse(courseId=safe_course_id, deleted=1)


@router.delete("/courses/{course_id}/evaluations", response_model=DeleteResponse)
def delete_all_course_evaluations(course_id: str) -> DeleteResponse:
    """Clear one course's evaluations, matching the existing bulk-clear UI."""
    safe_course_id = _safe_course_id(course_id)
    deleted = _run(
        "clearing evaluations",
        lambda connection: db_evaluations.delete_all_evaluations(
            connection, safe_course_id
        ),
    )
    return DeleteResponse(courseId=safe_course_id, deleted=deleted)


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #


@router.get("/courses/{course_id}/model", response_model=ModelRegistryResponse)
def get_course_model(course_id: str) -> ModelRegistryResponse:
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
    return ModelRegistryResponse(**registry)


# --------------------------------------------------------------------------- #
# Model requests
# --------------------------------------------------------------------------- #


@router.get("/courses/{course_id}/model-request", response_model=ModelRequestRecord)
def get_course_model_request(course_id: str) -> ModelRequestRecord:
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
    return ModelRequestRecord(**request_record)


@router.post(
    "/courses/{course_id}/model-request",
    response_model=ModelRequestRecord,
    status_code=201,
)
def create_course_model_request(
    course_id: str,
    request: ModelRequestCreateRequest,
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
    return ModelRequestRecord(**created)


@router.patch("/courses/{course_id}/model-request", response_model=ModelRequestRecord)
def update_course_model_request(
    course_id: str,
    request: ModelRequestUpdateRequest,
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
    return ModelRequestRecord(**updated)


# --------------------------------------------------------------------------- #
# Training runs
# --------------------------------------------------------------------------- #


@router.get(
    "/courses/{course_id}/training-runs", response_model=TrainingRunListResponse
)
def list_course_training_runs(course_id: str) -> TrainingRunListResponse:
    safe_course_id = _safe_course_id(course_id)
    runs = _run(
        "listing training runs",
        lambda connection: db_training_runs.list_training_runs(
            connection, safe_course_id
        ),
    )
    return TrainingRunListResponse(
        courseId=safe_course_id, count=len(runs), runs=runs
    )


@router.get(
    "/courses/{course_id}/training-runs/{run_id}", response_model=TrainingRunRecord
)
def get_course_training_run(course_id: str, run_id: str) -> TrainingRunRecord:
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
    return TrainingRunRecord(**run)


@router.post(
    "/courses/{course_id}/training-runs",
    response_model=TrainingRunRecord,
    status_code=201,
)
def enqueue_course_training_run(
    course_id: str,
    request: TrainingRunCreateRequest,
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
    return TrainingRunRecord(**created)


@router.patch(
    "/courses/{course_id}/training-runs/{run_id}", response_model=TrainingRunRecord
)
def update_course_training_run(
    course_id: str,
    run_id: str,
    request: TrainingRunUpdateRequest,
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
    return TrainingRunRecord(**updated)
