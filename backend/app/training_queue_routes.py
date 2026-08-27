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
hand it back, fetch that run's prepared dataset, record a scheduler job id,
report a failure or a completion, and register a finished model. It cannot read
seeds, list courses, or touch a course that has no run of its own — every route
below is keyed by `(course_id, run_id)`, so a run belonging to another course is
a 404 and not a cross-course read or edit.

The dataset routes are the reason the manual `rsync` step is gone. They serve a
fixed set of file names out of the directory the *backend* resolves from the
run, so a worker names a run and receives that run's data; it never names a
path and there is no request shape that reaches another course's export or any
other file on the VM. See `app/dataset_artifacts.py`.

Authentication is a shared secret in `X-Training-Worker-Token`, compared
against `TRAINING_WORKER_TOKEN` with a constant-time comparison. When that
variable is unset the whole router refuses with 503 rather than defaulting to
open: an unconfigured deployment must not be an unauthenticated queue.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from app import (
    db_courses,
    db_model_requests,
    db_models,
    db_serving_sessions,
    db_training_runs,
)
from app.course_id import assert_valid_course_id
from app.dataset_artifacts import (
    DatasetArtifactError,
    UnknownDatasetFileError,
    resolve_dataset_file,
    resolve_run_dataset,
)
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


def sha256_text_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


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


class CompletionRequest(QueueModel):
    """What the cluster reports when a training job ends, either way.

    Everything except `outcome` is optional, because this payload is assembled
    on a compute node from files a failed job may never have written. A run that
    died during model load has a job id, a failure stage and nothing else; that
    still has to be reportable, or the state this endpoint exists to fix — a run
    stuck at `submitted` for days — comes straight back for failures.

    `jobId` is how a completion is matched to the submission it belongs to. It
    is also how the lost-`/submitted` case is recovered: if the run carries no
    job id because that call never landed, this one records it.
    """

    outcome: str = Field(description="succeeded | failed")
    job_id: str | None = Field(default=None, alias="jobId", max_length=64)

    # Artifact — required for a success that is to be registered.
    base_model: str | None = Field(default=None, alias="baseModel", max_length=200)
    artifact_ref: str | None = Field(default=None, alias="artifactRef", max_length=500)
    version: str | None = Field(default=None)
    # Named `register_model` rather than `register`: the plain name shadows an
    # attribute pydantic's BaseModel already carries, and the alias is what the
    # wire format uses anyway.
    register_model: bool = Field(default=True, alias="register")
    set_current: bool = Field(default=True, alias="setCurrent")

    # Dataset provenance.
    dataset_ref: str | None = Field(default=None, alias="datasetRef", max_length=500)
    dataset_version: str | None = Field(
        default=None, alias="datasetVersion", max_length=300
    )
    dataset_sha256: str | None = Field(
        default=None, alias="datasetSha256", max_length=128
    )
    dataset_checksums: dict[str, str] | None = Field(
        default=None, alias="datasetChecksums"
    )
    approved_example_count: int | None = Field(
        default=None, alias="approvedExampleCount", ge=0
    )
    train_examples: int | None = Field(default=None, alias="trainExamples", ge=0)
    validation_examples: int | None = Field(
        default=None, alias="validationExamples", ge=0
    )

    # Training length accounting — the fix that made the CSS 350 run trustworthy.
    intended_optimizer_steps: int | None = Field(
        default=None, alias="intendedOptimizerSteps", ge=0
    )
    completed_steps: int | None = Field(default=None, alias="completedSteps", ge=0)
    missing_optimizer_steps: int | None = Field(
        default=None, alias="missingOptimizerSteps", ge=0
    )
    training_length_satisfied: bool | None = Field(
        default=None, alias="trainingLengthSatisfied"
    )
    epochs: float | None = Field(default=None, ge=0)

    # Metrics and cost.
    train_loss: float | None = Field(default=None, alias="trainLoss")
    eval_loss: float | None = Field(default=None, alias="evalLoss")
    actual_gpu_hours: float | None = Field(default=None, alias="actualGpuHours", ge=0)
    gpu_count: int | None = Field(default=None, alias="gpuCount", ge=0)
    elapsed_seconds: float | None = Field(default=None, alias="elapsedSeconds", ge=0)

    # Reproducibility.
    git_commit_sha: str | None = Field(
        default=None, alias="gitCommitSha", max_length=64
    )
    resolved_config: dict[str, Any] | None = Field(
        default=None, alias="resolvedConfig"
    )
    training_metrics: dict[str, Any] | None = Field(
        default=None, alias="trainingMetrics"
    )
    evaluation_metrics: dict[str, Any] | None = Field(
        default=None, alias="evaluationMetrics"
    )
    runtime_report: dict[str, Any] | None = Field(default=None, alias="runtimeReport")
    output_ref: str | None = Field(default=None, alias="outputRef", max_length=500)

    # Failure detail.
    failure_stage: str | None = Field(
        default=None, alias="failureStage", max_length=100
    )
    error: str | None = Field(default=None, max_length=2000)

    # Timestamps the cluster knows and the backend does not.
    started_at: str | None = Field(default=None, alias="startedAt", max_length=64)
    completed_at: str | None = Field(default=None, alias="completedAt", max_length=64)


class CompletionResponse(QueueModel):
    course_id: str = Field(alias="courseId")
    run_id: str = Field(alias="runId")
    outcome: str
    run_state: str = Field(alias="runState")
    request_status: str | None = Field(default=None, alias="requestStatus")
    version: str | None = None
    current_version: str | None = Field(default=None, alias="currentVersion")
    registered: bool = False
    #: True when this delivery found an already-registered version for the run
    #: and refreshed it rather than creating one. A retried callback reports
    #: this so a worker can tell "recorded now" from "already recorded".
    already_registered: bool = Field(default=False, alias="alreadyRegistered")


class PublishVersionRequest(QueueModel):
    """What the cluster says after an adapter is in place.

    Everything here is optional and descriptive. The action is decided by the
    URL — this course, this version — so a malformed body cannot publish
    something other than what the path names.
    """

    source_ref: str | None = Field(default=None, alias="sourceRef", max_length=500)
    published_at: str | None = Field(default=None, alias="publishedAt", max_length=64)


class PublishVersionResponse(QueueModel):
    course_id: str = Field(alias="courseId")
    version: str
    deployment: str
    current_version: str | None = Field(default=None, alias="currentVersion")
    previous_version: str | None = Field(default=None, alias="previousVersion")
    unchanged: bool = False


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


class SupersededRunCallback(HTTPException):
    """A worker reported against a run the model request no longer points at.

    The window is real and narrow: an admin retries a run whose lease has
    expired (or whose job has been silent for hours), the worker that still
    held it is alive after all, and its callback arrives after the replacement
    run has become current. Without this, that late report would move the
    request's status, its Slurm metadata and `current_run_id` back onto a run
    that was deliberately retired — and, for a model registration, would
    promote the very adapter the retry existed to discard.

    409 rather than a quiet no-op: the job it describes is real, it may still
    be running on the cluster, and an operator needs to see that in the
    runner's output rather than find it later.
    """

    def __init__(self, course_id: str, run_id: str, current_run_id: str) -> None:
        super().__init__(
            status_code=409,
            detail=(
                f'Run "{run_id}" is no longer the current run for course '
                f'"{course_id}" — it was superseded by "{current_run_id}". '
                "Its report was not applied to the model request."
            ),
        )


def _require_current_run(
    connection: Any, course_id: str, run_id: str
) -> dict[str, Any] | None:
    """Take the request row and refuse a report from a superseded run.

    `lock_model_request` rather than a plain read: this runs first in the
    transaction, so a retry arriving at the same moment either commits before
    this sees it — and this then refuses — or waits behind it. The two cannot
    interleave into a state where both believe they own the request.

    A course with no request at all is not an error here. The run rows are the
    system of record for the queue; the request is the professor-facing view,
    and its absence is not something a worker can do anything about.
    """
    request = db_model_requests.lock_model_request(connection, course_id)
    if request is None:
        return None

    current_run_id = request.get("currentRunId")
    if current_run_id and current_run_id != run_id:
        raise SupersededRunCallback(course_id, run_id, current_run_id)
    return request


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
        # Before the run is written, not after: `mark_training_run_submitted`
        # would move a retired run back to a non-terminal state, which is a
        # second active run for the course as well as a stolen request.
        _require_current_run(connection, safe_course_id, run_id)
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
        updated_request = db_model_requests.update_model_request_for_run(
            connection,
            safe_course_id,
            run_id,
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
        # `release_training_run` puts the run back to `queued`. On a superseded
        # run that would resurrect it alongside its own replacement, leaving
        # the course with two active runs and the queue guard unable to say
        # which is real.
        _require_current_run(connection, safe_course_id, run_id)
        released = db_training_runs.release_training_run(
            connection, safe_course_id, run_id, error=request.error, now=_utc_now()
        )
        db_model_requests.update_model_request_for_run(
            connection,
            safe_course_id,
            run_id,
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
        # A late failure from a retired run must not tell a professor their
        # model failed while its replacement is queued and untried.
        _require_current_run(connection, safe_course_id, run_id)
        failed = db_training_runs.fail_training_run(
            connection, safe_course_id, run_id, error=request.error, now=_utc_now()
        )
        db_model_requests.update_model_request_for_run(
            connection,
            safe_course_id,
            run_id,
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
# Completion
#
# The gap this closes: job 253552 finished successfully and the application did
# not find out for days. `training_run.state` stayed `submitted` and
# `model_request.status` stayed `training`, because nothing on the cluster ever
# told the backend the job had ended — there was no route to tell it through.
# An operator noticed, ran `register_course_model.py` by hand, and the state
# caught up. That is not a workflow, it is a person compensating for a missing
# edge in a state machine.
#
# One route handles both outcomes, because the thing that must not be lost is
# the *event*, and a job that failed is exactly as important to report as one
# that succeeded — arguably more, since a silent failure is what leaves a
# professor watching "training" forever.
#
# Everything about this is built to survive being delivered more than once. The
# worker persists its payload before sending and re-sends it later if the
# backend was unreachable, so a redelivery is the normal case and not an edge
# one. See `_register_version_in_transaction` for how a second delivery finds
# the version the first one created instead of allocating v2.
# --------------------------------------------------------------------------- #

COMPLETION_OUTCOMES = ("succeeded", "failed")


class JobIdMismatch(HTTPException):
    """A completion naming a different job than the one this run submitted.

    Refused rather than accepted-and-overwritten. Two jobs reporting against one
    run means either a stale worker replaying an old payload or a genuine
    mix-up, and in both cases the run's own record of what it submitted is the
    one to trust.
    """

    def __init__(self, run_id: str, recorded: str, reported: str) -> None:
        super().__init__(
            status_code=409,
            detail=(
                f'Run "{run_id}" was submitted as Slurm job {recorded}, but this '
                f"completion reports job {reported}. Nothing was recorded."
            ),
        )


def _completion_record(
    request: CompletionRequest,
    *,
    job_id: str | None,
    received_at: str,
) -> dict[str, Any]:
    """The durable operator-facing record of how a job ended.

    Keys with nothing behind them are dropped rather than stored as null: this
    document is read by a person asking what happened, and a wall of nulls from
    a job that failed before it could measure anything is worse than a short
    record of what was actually known.
    """
    record: dict[str, Any] = {
        "outcome": request.outcome,
        "receivedAt": received_at,
    }
    optional: dict[str, Any] = {
        "jobId": job_id,
        "startedAt": request.started_at,
        "completedAt": request.completed_at,
        "outputRef": request.output_ref,
        "artifactRef": request.artifact_ref,
        "baseModel": request.base_model,
        "datasetRef": request.dataset_ref,
        "datasetVersion": request.dataset_version,
        "datasetSha256": request.dataset_sha256,
        "datasetChecksums": request.dataset_checksums,
        "approvedExampleCount": request.approved_example_count,
        "trainExamples": request.train_examples,
        "validationExamples": request.validation_examples,
        "intendedOptimizerSteps": request.intended_optimizer_steps,
        "completedSteps": request.completed_steps,
        "missingOptimizerSteps": request.missing_optimizer_steps,
        "trainingLengthSatisfied": request.training_length_satisfied,
        "epochs": request.epochs,
        "trainLoss": request.train_loss,
        "evalLoss": request.eval_loss,
        "actualGpuHours": request.actual_gpu_hours,
        "gpuCount": request.gpu_count,
        "elapsedSeconds": request.elapsed_seconds,
        "gitCommitSha": request.git_commit_sha,
        "resolvedConfig": request.resolved_config,
        "trainingMetrics": request.training_metrics,
        "evaluationMetrics": request.evaluation_metrics,
        "runtimeReport": request.runtime_report,
        "failureStage": request.failure_stage,
        "error": request.error,
    }
    for key, value in optional.items():
        if value is not None:
            record[key] = value
    return record


def _provenance_record(
    request: CompletionRequest,
    run: dict[str, Any],
    *,
    job_id: str | None,
) -> dict[str, Any]:
    """What a finished artifact was made from, stored on the version.

    Deliberately duplicated onto the version rather than left only on the run.
    The question this answers — "what is this model, exactly?" — is asked about
    a version long after anyone is looking at the run that produced it, and a
    version that can only be explained by joining to another table is one that
    stops being explicable the moment that row is archived.
    """
    record: dict[str, Any] = {
        "courseId": run.get("courseId"),
        "runId": run.get("runId"),
        "mode": run.get("mode"),
        "attempt": run.get("attempt"),
        "enqueuedAt": run.get("enqueuedAt"),
    }
    optional: dict[str, Any] = {
        "slurmJobId": job_id,
        "baseModel": request.base_model,
        "artifactRef": request.artifact_ref,
        "outputRef": request.output_ref,
        "datasetRef": request.dataset_ref or run.get("datasetRef"),
        "datasetVersion": request.dataset_version,
        "datasetSha256": request.dataset_sha256,
        "datasetChecksums": request.dataset_checksums,
        "approvedExampleCount": (
            request.approved_example_count
            if request.approved_example_count is not None
            else run.get("approvedExampleCount")
        ),
        "trainExamples": request.train_examples,
        "validationExamples": request.validation_examples,
        "intendedOptimizerSteps": request.intended_optimizer_steps,
        "completedSteps": request.completed_steps,
        "missingOptimizerSteps": request.missing_optimizer_steps,
        "trainingLengthSatisfied": request.training_length_satisfied,
        "epochs": request.epochs,
        "trainLoss": request.train_loss,
        "evalLoss": request.eval_loss,
        "actualGpuHours": request.actual_gpu_hours,
        "gpuCount": request.gpu_count,
        "elapsedSeconds": request.elapsed_seconds,
        "gitCommitSha": request.git_commit_sha,
        "resolvedConfig": request.resolved_config,
        "startedAt": request.started_at,
        "completedAt": request.completed_at,
    }
    for key, value in optional.items():
        if value is not None:
            record[key] = value
    return record


def _training_example_count(request: CompletionRequest, run: dict[str, Any]) -> int:
    """How many examples this model was actually trained on.

    The reported train count first, because it was counted on the file the job
    read. The run's enqueued count is the fallback: it was recorded at queue
    time and can have been superseded by a re-preparation since.
    """
    if request.train_examples is not None:
        return int(request.train_examples)
    return int(run.get("trainExamples") or 0)


@router.post(
    "/courses/{course_id}/runs/{run_id}/completed",
    response_model=CompletionResponse,
    dependencies=[Depends(require_worker_token)],
)
def record_completion(
    course_id: str, run_id: str, request: CompletionRequest
) -> CompletionResponse:
    """Record how a training job ended, and register its model if it produced one.

    One transaction covers the completion record, the run's terminal state, the
    model version and the request's status. Previously the last of those was a
    separate manual command run hours later, which is why a course could sit in
    `training` with a finished adapter on disk.

    A success is only registered when the artifact is real and the run actually
    ran to its intended length. `trainingLengthSatisfied: false` is refused: a
    QLoRA run that stopped short of its optimizer-step budget is the exact bug
    the step-budget fix exists to catch, and registering its adapter as a ready
    model would hide it behind a green status.
    """
    safe_course_id = _safe_course_id(course_id)

    if request.outcome not in COMPLETION_OUTCOMES:
        raise HTTPException(
            status_code=422,
            detail=f"outcome must be one of {sorted(COMPLETION_OUTCOMES)}.",
        )
    if request.version is not None and not VERSION_PATTERN.match(request.version):
        raise HTTPException(
            status_code=422, detail="version must look like v1, v2, …"
        )

    will_register = request.outcome == "succeeded" and request.register_model
    artifact_ref = (
        _validate_artifact_ref(request.artifact_ref or "") if will_register else None
    )
    if will_register:
        if not (request.base_model or "").strip():
            raise HTTPException(
                status_code=422,
                detail="baseModel is required to register a finished model.",
            )
        # Refused, not registered-with-a-warning. A short run is a defective
        # artifact, and the professor-facing status this would set is "ready".
        if request.training_length_satisfied is False or (
            request.missing_optimizer_steps is not None
            and request.missing_optimizer_steps > 0
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    "This run finished short of its intended optimizer-step "
                    f"budget ({request.completed_steps}/"
                    f"{request.intended_optimizer_steps}, "
                    f"{request.missing_optimizer_steps} missing). It was not "
                    "registered as a model. Report it as failed, or retry the "
                    "run."
                ),
            )

    def work(connection: Any) -> CompletionResponse:
        run = _require_run(connection, safe_course_id, run_id)
        # Runs first, as it does for every other callback: a report from a run
        # an admin has retired must not move the replacement's request, and must
        # not register the adapter the retry existed to discard.
        _require_current_run(connection, safe_course_id, run_id)

        recorded_job_id = (run.get("jobId") or "").strip()
        reported_job_id = (request.job_id or "").strip()
        if recorded_job_id and reported_job_id and recorded_job_id != reported_job_id:
            raise JobIdMismatch(run_id, recorded_job_id, reported_job_id)
        job_id = recorded_job_id or reported_job_id or None

        now = _utc_now().isoformat()
        completion = _completion_record(request, job_id=job_id, received_at=now)

        if request.outcome == "failed":
            error = request.error or "Training did not finish successfully."
            db_training_runs.update_training_run(
                connection,
                safe_course_id,
                run_id,
                {
                    "state": "failed",
                    "updatedAt": now,
                    "error": error,
                    "claim": None,
                    "completion": completion,
                    # Recovers the case where /submitted never landed: the job
                    # existed, the callback knows its id, and the run should
                    # carry it so an operator can find the logs.
                    **({"jobId": job_id} if job_id and not recorded_job_id else {}),
                },
            )
            updated = db_model_requests.update_model_request_for_run(
                connection,
                safe_course_id,
                run_id,
                {
                    "status": "failed",
                    "updatedAt": now,
                    "currentRunId": run_id,
                    "failureMessage": "Training did not finish successfully.",
                    "launchError": error,
                },
            )
            return CompletionResponse(
                courseId=safe_course_id,
                runId=run_id,
                outcome="failed",
                runState="failed",
                requestStatus=updated["status"] if updated else None,
            )

        # Success. The run's own record is written first so that a registration
        # which then fails validation still leaves the completion visible — an
        # operator must be able to see what the job reported even when the
        # backend declined to turn it into a model.
        db_training_runs.update_training_run(
            connection,
            safe_course_id,
            run_id,
            {
                "updatedAt": now,
                "error": None,
                "claim": None,
                "completion": completion,
                **({"trainExamples": request.train_examples} if request.train_examples is not None else {}),
                **(
                    {"validationExamples": request.validation_examples}
                    if request.validation_examples is not None
                    else {}
                ),
                **({"jobId": job_id} if job_id and not recorded_job_id else {}),
            },
        )

        if not will_register:
            # A success the caller asked not to register: the artifact exists
            # but the model registry is left alone. The run is still terminal —
            # nothing more is going to happen to it.
            db_training_runs.update_training_run(
                connection,
                safe_course_id,
                run_id,
                {"state": "succeeded", "updatedAt": now, "claim": None},
            )
            return CompletionResponse(
                courseId=safe_course_id,
                runId=run_id,
                outcome="succeeded",
                runState="succeeded",
                requestStatus=None,
                registered=False,
            )

        version_key, current, request_status, reused = _register_version_in_transaction(
            connection,
            course_id=safe_course_id,
            run_id=run_id,
            base_model=(request.base_model or "").strip(),
            training_example_count=_training_example_count(request, run),
            artifact_ref=artifact_ref or "",
            status="ready",
            deployment="offline",
            version=request.version,
            notes=None,
            provenance=_provenance_record(request, run, job_id=job_id),
            set_current=request.set_current,
        )

        return CompletionResponse(
            courseId=safe_course_id,
            runId=run_id,
            outcome="succeeded",
            runState="succeeded",
            requestStatus=request_status,
            version=version_key,
            currentVersion=current,
            registered=True,
            alreadyRegistered=reused,
        )

    try:
        return _run("recording a training completion", work)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# Prepared dataset transfer
#
# Two routes: one describing the dataset, one serving a single named file. The
# split is what makes the transfer restartable and cheap to skip — a worker
# reads the descriptor, compares `datasetSha256` against what it already has,
# and in the common case stops there without moving any bytes.
# --------------------------------------------------------------------------- #


def _dataset_for_run(course_id: str, run_id: str) -> dict[str, Any]:
    """Describe the dataset belonging to one run, or raise the right refusal.

    The run is read first and from the database, so the directory served is the
    one this course's run points at rather than anything the caller supplied.
    A run that does not exist for this course is a 404 here exactly as it is for
    every write route.
    """
    safe_course_id = _safe_course_id(course_id)
    run = _run(
        "reading a training run",
        lambda connection: _require_run(connection, safe_course_id, run_id),
    )
    try:
        return resolve_run_dataset(run)
    except DatasetArtifactError as exc:
        # 409 rather than 404: the run is real and the request was well formed.
        # What is missing is a preparation step on this side, and an operator
        # needs to see that distinction in the worker's output.
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get(
    "/courses/{course_id}/runs/{run_id}/dataset",
    dependencies=[Depends(require_worker_token)],
)
def describe_run_dataset(course_id: str, run_id: str) -> dict[str, Any]:
    """What this run will train on: counts, file digests, one dataset digest.

    Read-only and side-effect free, so `--dry-run` can call it against the live
    queue to report whether a run is actually trainable.
    """
    return _dataset_for_run(course_id, run_id)


@router.get(
    "/courses/{course_id}/runs/{run_id}/dataset/files/{name}",
    dependencies=[Depends(require_worker_token)],
)
def download_run_dataset_file(course_id: str, run_id: str, name: str) -> Response:
    """One file from this run's prepared dataset, as bytes.

    `name` is checked against a fixed allowlist before any path exists; see
    `resolve_dataset_file`. The digest travels in a header so a worker that
    streams the body straight to disk can verify it without re-reading the
    descriptor, and so a truncated response is detectable rather than silently
    shorter training data.
    """
    safe_course_id = _safe_course_id(course_id)
    # Establishes that this run belongs to this course before anything is read
    # off disk. Without it the route would serve a course's export to anyone who
    # could name the course, which is a weaker rule than every other route here.
    _run(
        "reading a training run",
        lambda connection: _require_run(connection, safe_course_id, run_id),
    )

    try:
        path = resolve_dataset_file(safe_course_id, name)
    except UnknownDatasetFileError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DatasetArtifactError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    body = path.read_bytes()
    return Response(
        content=body,
        media_type="application/octet-stream",
        headers={
            "X-Content-SHA256": sha256_text_bytes(body),
            "X-Dataset-File-Name": name,
            "Content-Length": str(len(body)),
        },
    )


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


class VersionAlreadyRegisteredForRun(HTTPException):
    """This run already has a registered version, under a different key.

    Not a failure of the callback — it is the successful outcome of a previous
    delivery, described back to a caller who asked for a specific version key
    the run does not have. Refusing rather than writing keeps one run to one
    version, which is what stops a redelivered completion becoming v2.
    """

    def __init__(self, course_id: str, run_id: str, version: str) -> None:
        super().__init__(
            status_code=409,
            detail=(
                f'Run "{run_id}" is already registered for course "{course_id}" '
                f'as version "{version}". A training run produces exactly one '
                "version; re-register it under that key or omit the version."
            ),
        )


def _register_version_in_transaction(
    connection: Any,
    *,
    course_id: str,
    run_id: str | None,
    base_model: str,
    training_example_count: int,
    artifact_ref: str,
    status: str,
    deployment: str,
    version: str | None,
    notes: str | None,
    provenance: dict[str, Any] | None,
    set_current: bool,
) -> tuple[str, str, str | None, bool]:
    """Write one course model version and move the run and request with it.

    Shared by the explicit registration route and the completion callback so
    both get the same isolation, the same ownership guard, and the same
    idempotency. Returns `(version, currentVersion, requestStatus, reused)`.

    Idempotency is keyed on the run, not on the request body. A completion
    callback delivered twice — a retry after a timeout, a worker re-run after a
    dropped login session — finds the version its first delivery created and
    refreshes that row instead of allocating the next `vN`. Without this, the
    ordinary network failure this system already sees would leave a course with
    v1 and v2 pointing at one adapter, and a professor's page showing a version
    count that means nothing.

    Course isolation is structural rather than checked: the row is keyed
    `(course_id, version)` and `course_models.course_id` is a primary key, so
    there is no shape of this call that writes another course's registry.
    """
    if not db_courses.course_exists(connection, course_id):
        raise HTTPException(
            status_code=404, detail=f'Course "{course_id}" was not found.'
        )

    # The most consequential guard here, and the reason it runs before the
    # version is written rather than only before the request is.
    #
    # A registration names a run. If that run has been superseded, this call is
    # a finished job reporting an adapter an admin has already decided not to
    # use — the exact case a retry after a training-code fix exists to produce.
    # Letting it through would register that adapter and, with `setCurrent`,
    # promote it: the course would end up serving the model the retry was meant
    # to replace, and the request would go `ready` while its replacement run sat
    # queued and untried.
    #
    # Refused rather than registered-but-not-promoted. The artifact is real and
    # an operator may still want it recorded, but that is a decision a person
    # makes deliberately with `register_course_model.py`, not one a stale
    # callback makes on their behalf.
    if run_id:
        _require_current_run(connection, course_id, run_id)

    reused = False
    version_key: str | None = None
    if run_id:
        already = db_models.find_model_version_for_run(connection, course_id, run_id)
        if already is not None:
            if version and version != already["version"]:
                raise VersionAlreadyRegisteredForRun(
                    course_id, run_id, already["version"]
                )
            version_key = already["version"]
            reused = True

    if version_key is None:
        existing = db_models.list_model_versions(connection, course_id)
        version_key = version or next_model_version(
            [item["version"] for item in existing]
        )

    now = _utc_now().isoformat()
    record: dict[str, Any] = {
        "version": version_key,
        "baseModel": base_model,
        "trainingExampleCount": training_example_count,
        "status": status,
        "deployment": deployment,
        "artifactRef": artifact_ref,
        "createdAt": now,
        "updatedAt": now,
        "notes": notes,
    }
    if run_id:
        record["runId"] = run_id
    if provenance:
        record["provenance"] = provenance

    registry = db_models.upsert_model_version(
        connection, course_id, record, set_current=set_current
    )

    request_status: str | None = None
    # A registered, ready model is what the professor's page is waiting for.
    # Anything else is not a finished model and must not flip the request to
    # ready.
    if status == "ready":
        patch: dict[str, Any] = {
            "status": "ready",
            "updatedAt": now,
            "failureMessage": None,
            "launchError": None,
        }
        if run_id:
            patch["currentRunId"] = run_id
        updated = (
            db_model_requests.update_model_request_for_run(
                connection, course_id, run_id, patch
            )
            if run_id
            else db_model_requests.update_model_request(connection, course_id, patch)
        )
        request_status = updated["status"] if updated else None

        if run_id:
            db_training_runs.update_training_run(
                connection,
                course_id,
                run_id,
                {"state": "succeeded", "updatedAt": now, "claim": None},
            )

    current = (registry or {}).get("currentVersion") or version_key
    return version_key, current, request_status, reused


def _validate_artifact_ref(artifact_ref: str) -> str:
    """Relative references only.

    An absolute promote-script destination embeds a cluster home directory and a
    username, and admin surfaces display this string.
    """
    cleaned = (artifact_ref or "").strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="artifactRef must not be empty.")
    if cleaned.startswith("/"):
        raise HTTPException(
            status_code=422,
            detail=(
                "artifactRef must be relative. An absolute path embeds a "
                "cluster home directory and a username, which must not be "
                "stored."
            ),
        )
    return cleaned


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

    The manual and recovery path — `scripts/register_course_model.py` calls
    this. Normal operation goes through the completion callback instead, which
    reaches the same helper with the same guarantees.
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

    artifact_ref = _validate_artifact_ref(request.artifact_ref)
    if request.version is not None and not VERSION_PATTERN.match(request.version):
        raise HTTPException(
            status_code=422, detail="version must look like v1, v2, …"
        )

    def work(connection: Any) -> tuple[str, str, str | None, bool]:
        return _register_version_in_transaction(
            connection,
            course_id=safe_course_id,
            run_id=request.run_id,
            base_model=request.base_model,
            training_example_count=request.training_example_count,
            artifact_ref=artifact_ref,
            status=request.status,
            deployment=request.deployment,
            version=request.version,
            notes=request.notes,
            provenance=None,
            set_current=request.set_current,
        )

    try:
        version_key, current, request_status, _reused = _run(
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


# --------------------------------------------------------------------------- #
# Serving sessions
#
# The missing piece that made starting inference a two-machine copy-and-paste:
# the compute node changes with every Slurm job, so the operator read a hostname
# off Tillicum's output and typed it into a command on the VM. Recording the
# session here means the VM can ask the backend where the service is instead,
# and the Admin page can show whether anything is serving at all.
#
# Nothing here promotes, demotes, or otherwise touches a model version. A
# session starting does not make a model `online`, and a session ending does not
# make it `offline` — those describe the artifact, and this describes a GPU
# allocation with a wall clock on it.
# --------------------------------------------------------------------------- #

SESSION_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class ServingSessionRequest(QueueModel):
    job_id: str = Field(alias="jobId", min_length=1, max_length=64)
    node: str = Field(min_length=1, max_length=253)
    port: int = Field(gt=0, lt=65536)
    state: str = Field(default="starting")
    started_at: str | None = Field(default=None, alias="startedAt", max_length=64)
    expires_at: str = Field(alias="expiresAt", max_length=64)
    detail: dict[str, Any] | None = Field(default=None)


class ServingSessionResponse(QueueModel):
    session: dict[str, Any] | None = None


def _safe_session_id(session_id: str) -> str:
    """Session ids are ours, not a caller's free text.

    The start script derives one from the Slurm job id. Constraining the shape
    keeps it a legal path segment and keeps anything surprising out of a value
    that ends up in log lines and an admin table.
    """
    candidate = (session_id or "").strip().lower()
    if not SESSION_ID_PATTERN.match(candidate):
        raise HTTPException(
            status_code=400,
            detail=(
                "A serving session id must be lowercase letters, digits and "
                "hyphens, e.g. serve-264787."
            ),
        )
    return candidate


@router.put(
    "/serving-sessions/{session_id}",
    response_model=ServingSessionResponse,
    dependencies=[Depends(require_worker_token)],
)
def upsert_serving_session(
    session_id: str, request: ServingSessionRequest
) -> ServingSessionResponse:
    """Record or refresh the session a serving job is running under.

    Idempotent by construction: the id names the Slurm job, so re-running the
    start script against an allocation that is already up refreshes one row
    rather than claiming a second service exists.
    """
    safe_session_id = _safe_session_id(session_id)
    if request.state not in db_serving_sessions.SESSION_STATES:
        raise HTTPException(
            status_code=422,
            detail=(
                "state must be one of "
                f"{sorted(db_serving_sessions.SESSION_STATES)}."
            ),
        )

    def work(connection: Any) -> dict[str, Any]:
        return db_serving_sessions.upsert_serving_session(
            connection,
            {
                "sessionId": safe_session_id,
                "jobId": request.job_id.strip(),
                "node": request.node.strip(),
                "port": request.port,
                "state": request.state,
                "startedAt": request.started_at,
                "expiresAt": request.expires_at,
                "detail": request.detail,
            },
        )

    try:
        session = _run("recording a serving session", work)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ServingSessionResponse(session=session)


@router.post(
    "/serving-sessions/{session_id}/stopped",
    response_model=ServingSessionResponse,
    dependencies=[Depends(require_worker_token)],
)
def stop_serving_session(session_id: str) -> ServingSessionResponse:
    """Mark a session stopped. Not an error when it is already gone.

    The stop script runs after `scancel`, and a session that had already expired
    on its own is exactly the case where there is nothing left to stop. Reporting
    that as a failure would make a successful stop look broken.
    """
    safe_session_id = _safe_session_id(session_id)
    session = _run(
        "stopping a serving session",
        lambda connection: db_serving_sessions.stop_serving_session(
            connection, safe_session_id, now=_utc_now()
        ),
    )
    return ServingSessionResponse(session=session)


@router.get(
    "/serving-session",
    response_model=ServingSessionResponse,
    dependencies=[Depends(require_worker_token)],
)
def get_current_serving_session() -> ServingSessionResponse:
    """The session serving right now, node and port included, or none.

    Worker-token only. This is the one response in the system that says how to
    reach a compute node, which is why it is not on the browser-facing routes;
    `/api/db/serving-session` returns the same session without them.
    """
    session = _run(
        "reading the current serving session",
        lambda connection: db_serving_sessions.current_serving_session(
            connection, now=_utc_now()
        ),
    )
    return ServingSessionResponse(session=session)


@router.post(
    "/courses/{course_id}/model-versions/{version}/published",
    response_model=PublishVersionResponse,
    dependencies=[Depends(require_worker_token)],
)
def record_model_version_published(
    course_id: str, version: str, request: PublishVersionRequest | None = None
) -> PublishVersionResponse:
    """Record that a version's adapter is now in the cluster's serving tree.

    Reported *after* the copy has succeeded and been validated, never before.
    That ordering is the whole failure-safety story for an operation spanning
    two machines: if the copy fails, nothing is reported and the database goes
    on naming the version that really is published. The opposite order would
    let a failed copy leave the application confidently routing every question
    to an adapter that is not there.

    Publication is what inference resolves from. Registering `v2` moves
    `current_version` — a professor's newest model is the honest answer to
    "what is my model" — but it does not move what answers questions, because
    the cluster does not have `v2` until somebody puts it there. Without this
    endpoint that gap was an outage: the backend asked for `v2`, the cluster had
    only `v1`, and a course that was answering fine stopped.

    Idempotent. Publishing the version that is already published is a no-op that
    reports `unchanged`, so a rerun of the promote script, or a report delivered
    twice after a network failure, cannot corrupt anything.

    Not guarded by `_require_current_run`: publication is an operator's
    deliberate act on a registered artifact, not a callback from a run that may
    have been superseded. An admin who publishes an older version on purpose —
    rolling back a bad `v2` to `v1` — is doing something this must allow.
    """
    safe_course_id = _safe_course_id(course_id)

    if not VERSION_PATTERN.match(version):
        raise HTTPException(
            status_code=422, detail="version must look like v1, v2, …"
        )

    def work(connection: Any) -> tuple[str, str | None, str | None, bool]:
        existing = db_models.list_model_versions(connection, safe_course_id)
        by_version = {item["version"]: item for item in existing}

        target = by_version.get(version)
        if target is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f'Course "{safe_course_id}" has no registered model version '
                    f'"{version}". Register it before publishing it.'
                ),
            )

        # A version that is not ready is not an artifact anyone should be
        # routed to, whatever is sitting on the cluster's filesystem.
        if target.get("status") != "ready":
            raise HTTPException(
                status_code=409,
                detail=(
                    f'Version "{version}" is "{target.get("status")}", not ready. '
                    "Only a ready version can be published."
                ),
            )

        previous = next(
            (
                item["version"]
                for item in existing
                if item.get("deployment") == "online" and item["version"] != version
            ),
            None,
        )
        already = target.get("deployment") == "online"

        registry = db_models.mark_version_published(
            connection, safe_course_id, version
        )
        current = (registry or {}).get("currentVersion")
        return current, previous, version, already

    try:
        current, previous, published, already = _run(
            "recording a published model version", work
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PublishVersionResponse(
        courseId=safe_course_id,
        version=published,
        deployment="online",
        currentVersion=current,
        previousVersion=previous,
        unchanged=already,
    )
