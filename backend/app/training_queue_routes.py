"""The training queue, as an API the cluster runner can reach.

Why the worker does not talk to PostgreSQL directly
---------------------------------------------------
The runner executes on Tillicum, which is not the machine the database runs on
and is not on its network. The only way to let it hold a psycopg connection
would be to expose PostgreSQL beyond the VM, and a queue migration is not a
good enough reason to put a database on a public interface. So the worker keeps
the shape it already had — outbound HTTPS to one host, no client library, no
credentials for anything but that host — and this router is the one thing on
the other end of it.

That is also the pre-existing pattern for remote operations here: everything
Tillicum does to application state, it does over HTTP.

What the worker is allowed to do
--------------------------------
Only what a queue runner needs: see what is claimable, take exactly one run,
hand it back, record a scheduler job id, report a failure, and register a
finished model. It cannot read seeds, list courses, or touch a course that has
no run of its own — every write below is keyed by `(course_id, run_id)`, so a
run belonging to another course is a 404 and not a cross-course edit.

Authentication is a shared secret in `X-Training-Worker-Token`, compared
against `TRAINING_WORKER_TOKEN` with a constant-time comparison. When that
variable is unset the whole router refuses with 503 rather than defaulting to
open: an unconfigured deployment must not be an unauthenticated queue.
"""

from __future__ import annotations

import hmac
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app import db_courses, db_model_requests, db_models, db_training_runs
from app.course_id import assert_valid_course_id
from app.db import db_connection, translate_db_errors
from app.db_schemas import TrainingRunRecord

WORKER_TOKEN_ENV_VAR = "TRAINING_WORKER_TOKEN"
WORKER_TOKEN_HEADER = "X-Training-Worker-Token"

VERSION_PATTERN = re.compile(r"^v(\d+)$")

MODEL_STATUSES = ("ready", "training", "failed")
DEPLOYMENT_STATUSES = ("online", "offline", "unknown")

router = APIRouter(prefix="/api/training-queue", tags=["training-queue"])


def require_worker_token(
    x_training_worker_token: str | None = Header(default=None),
) -> None:
    """Refuse anything that is not the configured worker.

    `hmac.compare_digest` rather than `==`: the comparison is against a secret,
    and a short-circuiting compare leaks its prefix to anyone able to time the
    endpoint.
    """
    expected = os.getenv(WORKER_TOKEN_ENV_VAR, "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail=(
                "The training queue API is not configured. Set "
                f"{WORKER_TOKEN_ENV_VAR} in the backend environment."
            ),
        )

    presented = (x_training_worker_token or "").strip()
    if not presented or not hmac.compare_digest(presented, expected):
        raise HTTPException(
            status_code=401,
            detail=f"A valid {WORKER_TOKEN_HEADER} header is required.",
        )


def _safe_course_id(course_id: str) -> str:
    try:
        return assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _run(action: str, work: Callable[[Any], Any]) -> Any:
    """One connection, one transaction, driver errors mapped to 503."""
    with translate_db_errors(action):
        with db_connection() as connection:
            return work(connection)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Request/response shapes
# --------------------------------------------------------------------------- #


class QueueModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class ClaimRequest(QueueModel):
    owner: str = Field(min_length=1, max_length=200)
    lease_seconds: int = Field(
        default=db_training_runs.DEFAULT_LEASE_SECONDS,
        alias="leaseSeconds",
        gt=0,
        le=86400,
    )
    course_ids: list[str] | None = Field(default=None, alias="courseIds")


class ClaimResponse(QueueModel):
    claimed: bool
    run: TrainingRunRecord | None = None


class PendingResponse(QueueModel):
    count: int
    runs: list[TrainingRunRecord]


class ReleaseRequest(QueueModel):
    error: str | None = Field(default=None, max_length=2000)


class FailRequest(QueueModel):
    error: str = Field(min_length=1, max_length=2000)


class SubmittedRequest(QueueModel):
    job_id: str = Field(alias="jobId", min_length=1, max_length=64)
    train_examples: int = Field(default=0, alias="trainExamples", ge=0)
    validation_examples: int = Field(default=0, alias="validationExamples", ge=0)


class SubmittedResponse(QueueModel):
    run: TrainingRunRecord
    request_status: str | None = Field(default=None, alias="requestStatus")


class ModelVersionRegisterRequest(QueueModel):
    base_model: str = Field(alias="baseModel", min_length=1)
    training_example_count: int = Field(alias="trainingExampleCount", ge=0)
    artifact_ref: str = Field(alias="artifactRef", min_length=1)
    status: str = Field(default="ready")
    deployment: str = Field(default="offline")
    version: str | None = Field(default=None)
    notes: str | None = Field(default=None)
    run_id: str | None = Field(default=None, alias="runId")
    set_current: bool = Field(default=True, alias="setCurrent")


class ModelVersionRegisterResponse(QueueModel):
    course_id: str = Field(alias="courseId")
    version: str
    current_version: str = Field(alias="currentVersion")
    request_status: str | None = Field(default=None, alias="requestStatus")


# --------------------------------------------------------------------------- #
# Queue
# --------------------------------------------------------------------------- #


@router.get(
    "/pending",
    response_model=PendingResponse,
    dependencies=[Depends(require_worker_token)],
)
def list_pending_runs(course_id: str | None = None) -> PendingResponse:
    """What a worker could take right now. Claims nothing.

    Backs `--dry-run`, which has to be safe to point at the live queue.
    """
    course_ids = [_safe_course_id(course_id)] if course_id else None
    runs = _run(
        "listing claimable training runs",
        lambda connection: db_training_runs.claimable_training_runs(
            connection, now=_utc_now(), course_ids=course_ids
        ),
    )
    return PendingResponse(
        count=len(runs), runs=[TrainingRunRecord(**run) for run in runs]
    )


@router.post(
    "/claim",
    response_model=ClaimResponse,
    dependencies=[Depends(require_worker_token)],
)
def claim_next_run(request: ClaimRequest) -> ClaimResponse:
    """Take exactly one eligible run, or report that there was none.

    Not an error when nothing is claimable, and not an error when another
    worker won the race: both are `claimed: false`, because from the caller's
    side they are the same situation — there is no work for it right now. The
    atomicity that makes that safe lives in `claim_next_training_run`.
    """
    course_ids = (
        [_safe_course_id(course_id) for course_id in request.course_ids]
        if request.course_ids is not None
        else None
    )

    def work(connection: Any) -> dict[str, Any] | None:
        return db_training_runs.claim_next_training_run(
            connection,
            owner=request.owner,
            lease_seconds=request.lease_seconds,
            now=_utc_now(),
            course_ids=course_ids,
        )

    try:
        claimed = _run("claiming a training run", work)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if claimed is None:
        return ClaimResponse(claimed=False, run=None)
    return ClaimResponse(claimed=True, run=TrainingRunRecord(**claimed))


def _require_run(connection: Any, course_id: str, run_id: str) -> dict[str, Any]:
    run = db_training_runs.get_training_run(connection, course_id, run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail=f'Training run "{run_id}" was not found for this course.',
        )
    return run


@router.post(
    "/courses/{course_id}/runs/{run_id}/release",
    response_model=TrainingRunRecord,
    dependencies=[Depends(require_worker_token)],
)
def release_run(
    course_id: str, run_id: str, request: ReleaseRequest | None = None
) -> TrainingRunRecord:
    """Hand a claimed run back, still queued and now unleased."""
    safe_course_id = _safe_course_id(course_id)
    body = request or ReleaseRequest()

    def work(connection: Any) -> dict[str, Any]:
        _require_run(connection, safe_course_id, run_id)
        return db_training_runs.release_training_run(
            connection, safe_course_id, run_id, error=body.error, now=_utc_now()
        )

    return TrainingRunRecord(**_run("releasing a training run", work))


@router.post(
    "/courses/{course_id}/runs/{run_id}/submitted",
    response_model=SubmittedResponse,
    dependencies=[Depends(require_worker_token)],
)
def record_submission(
    course_id: str, run_id: str, request: SubmittedRequest
) -> SubmittedResponse:
    """Record a real scheduler job id, then point the request at the job.

    Both writes happen in one transaction, which is the whole reason this is
    an endpoint rather than two. Previously they were separate REST calls in a
    fixed order, and a crash between them left a submitted run whose request
    still said `preparing`. Now a professor either sees a job that exists or
    sees the state from before the submission — never a model that is training
    according to one record and not according to the other.

    `failureMessage` is deliberately not written: that field is professor-facing
    and describes a failed request. A successful submission is not a failure.
    """
    safe_course_id = _safe_course_id(course_id)

    def work(connection: Any) -> tuple[dict[str, Any], str | None]:
        run = _require_run(connection, safe_course_id, run_id)
        submitted = db_training_runs.mark_training_run_submitted(
            connection,
            safe_course_id,
            run_id,
            job_id=request.job_id,
            train_examples=request.train_examples,
            validation_examples=request.validation_examples,
            now=_utc_now(),
        )

        submitted_at = submitted["updatedAt"]
        updated_request = db_model_requests.update_model_request(
            connection,
            safe_course_id,
            {
                "status": "training",
                "updatedAt": submitted_at,
                "currentRunId": run_id,
                "launchError": None,
                "training": {
                    "jobId": request.job_id.strip(),
                    "mode": run["mode"],
                    "submittedAt": submitted_at,
                    "datasetRef": run.get("datasetRef") or "",
                    "trainExamples": int(request.train_examples),
                    "validationExamples": int(request.validation_examples),
                },
            },
        )
        status = updated_request["status"] if updated_request else None
        return submitted, status

    try:
        submitted, status = _run("recording a training submission", work)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return SubmittedResponse(
        run=TrainingRunRecord(**submitted), requestStatus=status
    )


@router.post(
    "/courses/{course_id}/runs/{run_id}/submission-failed",
    response_model=TrainingRunRecord,
    dependencies=[Depends(require_worker_token)],
)
def record_submission_failure(
    course_id: str, run_id: str, request: FailRequest
) -> TrainingRunRecord:
    """Nothing was submitted. Leave the run retryable, the request preparing.

    The dataset is still valid, so the run goes back to `queued` carrying an
    operator-visible error. The professor-facing status does not move:
    `launchError` is admin-only, and telling a professor their model is
    training — or that it failed — would be untrue when no job was ever
    created.
    """
    safe_course_id = _safe_course_id(course_id)

    def work(connection: Any) -> dict[str, Any]:
        _require_run(connection, safe_course_id, run_id)
        released = db_training_runs.release_training_run(
            connection, safe_course_id, run_id, error=request.error, now=_utc_now()
        )
        db_model_requests.update_model_request(
            connection,
            safe_course_id,
            {
                "launchError": request.error,
                "updatedAt": _utc_now().isoformat(),
                "currentRunId": run_id,
            },
        )
        return released

    return TrainingRunRecord(**_run("recording a submission failure", work))


@router.post(
    "/courses/{course_id}/runs/{run_id}/failed",
    response_model=TrainingRunRecord,
    dependencies=[Depends(require_worker_token)],
)
def record_training_failure(
    course_id: str, run_id: str, request: FailRequest
) -> TrainingRunRecord:
    """Training itself failed. Terminal for the run and for the request.

    Distinct from a submission failure: a job existed and did not produce a
    model, so there is nothing to retry automatically and the professor does
    need to be told. `failureMessage` carries the professor-facing text and
    `error` the operator-facing one.
    """
    safe_course_id = _safe_course_id(course_id)

    def work(connection: Any) -> dict[str, Any]:
        _require_run(connection, safe_course_id, run_id)
        failed = db_training_runs.fail_training_run(
            connection, safe_course_id, run_id, error=request.error, now=_utc_now()
        )
        db_model_requests.update_model_request(
            connection,
            safe_course_id,
            {
                "status": "failed",
                "updatedAt": _utc_now().isoformat(),
                "currentRunId": run_id,
                "failureMessage": "Training did not finish successfully.",
                "launchError": request.error,
            },
        )
        return failed

    return TrainingRunRecord(**_run("recording a training failure", work))


# --------------------------------------------------------------------------- #
# Model registration
# --------------------------------------------------------------------------- #


def next_model_version(existing_versions: list[str]) -> str:
    """`v1`, then `v2`, … — the smallest scheme that still orders."""
    highest = 0
    for key in existing_versions:
        match = VERSION_PATTERN.match(str(key))
        if match:
            highest = max(highest, int(match.group(1)))
    return f"v{highest + 1}"


@router.post(
    "/courses/{course_id}/model-versions",
    response_model=ModelVersionRegisterResponse,
    status_code=201,
    dependencies=[Depends(require_worker_token)],
)
def register_model_version(
    course_id: str, request: ModelVersionRegisterRequest
) -> ModelVersionRegisterResponse:
    """Record a finished model against one course, and mark the request ready.

    Course isolation is structural, not checked: the version row is keyed
    `(course_id, version)` and `course_models.course_id` is the primary key, so
    a version registered for CSS 360 cannot land on CSS 350 even if a caller
    asked it to — there is no shape of this request that writes another
    course's row.

    The artifact reference must stay relative. An absolute promote-script
    destination embeds a cluster home directory and a username, and admin
    surfaces display this string.
    """
    safe_course_id = _safe_course_id(course_id)

    if request.status not in MODEL_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=f"status must be one of {sorted(MODEL_STATUSES)}.",
        )
    if request.deployment not in DEPLOYMENT_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=f"deployment must be one of {sorted(DEPLOYMENT_STATUSES)}.",
        )

    artifact_ref = request.artifact_ref.strip()
    if artifact_ref.startswith("/"):
        raise HTTPException(
            status_code=422,
            detail=(
                "artifactRef must be relative. An absolute path embeds a "
                "cluster home directory and a username, which must not be "
                "stored."
            ),
        )
    if request.version is not None and not VERSION_PATTERN.match(request.version):
        raise HTTPException(
            status_code=422, detail="version must look like v1, v2, …"
        )

    def work(connection: Any) -> tuple[str, str, str | None]:
        if not db_courses.course_exists(connection, safe_course_id):
            raise HTTPException(
                status_code=404, detail=f'Course "{safe_course_id}" was not found.'
            )

        existing = db_models.list_model_versions(connection, safe_course_id)
        version_key = request.version or next_model_version(
            [item["version"] for item in existing]
        )

        now = _utc_now().isoformat()
        registry = db_models.upsert_model_version(
            connection,
            safe_course_id,
            {
                "version": version_key,
                "baseModel": request.base_model,
                "trainingExampleCount": request.training_example_count,
                "status": request.status,
                "deployment": request.deployment,
                "artifactRef": artifact_ref,
                "createdAt": now,
                "updatedAt": now,
                "notes": request.notes,
            },
            set_current=request.set_current,
        )

        request_status: str | None = None
        # A registered, ready model is what the professor's page is waiting
        # for. Anything else is not a finished model and must not flip the
        # request to ready.
        if request.status == "ready":
            patch: dict[str, Any] = {
                "status": "ready",
                "updatedAt": now,
                "failureMessage": None,
                "launchError": None,
            }
            if request.run_id:
                patch["currentRunId"] = request.run_id
            updated = db_model_requests.update_model_request(
                connection, safe_course_id, patch
            )
            request_status = updated["status"] if updated else None

            if request.run_id:
                db_training_runs.update_training_run(
                    connection,
                    safe_course_id,
                    request.run_id,
                    {"state": "succeeded", "updatedAt": now, "claim": None},
                )

        current = (registry or {}).get("currentVersion") or version_key
        return version_key, current, request_status

    try:
        version_key, current, request_status = _run(
            "registering a course model version", work
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ModelVersionRegisterResponse(
        courseId=safe_course_id,
        version=version_key,
        currentVersion=current,
        requestStatus=request_status,
    )
