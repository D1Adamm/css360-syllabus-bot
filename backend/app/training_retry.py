"""Admin retry: retire a stale training run and queue its replacement.

Why this exists
---------------
A run reaches `submitted`, the cluster finishes the job, and the completion
callback never arrives. PostgreSQL still says `model_requests.status =
training` and `training_runs.state = submitted`, so `canQueueTraining` refuses
to offer a second run — correctly, because from the application's point of view
work is still outstanding. There was no supported way out of that state: the
run cannot finish (nothing is going to report it), and it cannot be replaced
(it blocks the queue). This module is the missing recovery path.

It is not a repair of one row. Retry is also the ordinary thing to do after a
run failed, or after the training code itself was fixed and the old attempt's
result is no longer wanted.

What it guarantees
------------------
One transaction, opened by the caller, doing all of:

  - the model request row is taken `FOR UPDATE` first, before anything is read
    that a decision depends on;
  - the outstanding run is retired terminally, keeping its job id, attempt
    count and timestamps, so training history is complete rather than tidy;
  - exactly one replacement run is queued, carrying the *same* prepared
    dataset — no export is rerun, no split is recomputed, no seed is touched;
  - the request points at the replacement and no longer carries the retired
    run's Slurm metadata.

Concurrency is the row lock plus the conditional INSERT underneath
`enqueue_training_run`. A second retry — a double click, a second admin — waits
on the lock, and when it proceeds it reads the pointer the first one moved: a
freshly `queued` run, which `training_run_retry_block` refuses. Two new active
runs from one course is therefore not a race that can be lost, it is a state
two independent guards each forbid.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from app import db_model_requests, db_training_runs
from app.course_id import assert_valid_course_id
from app.db_mapping import as_int

# The request status a course returns to while a run sits in the queue.
#
# `preparing` is what `queueTraining.ts` leaves a request at after an ordinary
# enqueue, and the reason is worth repeating here: nothing is training. A run is
# waiting to be picked up, and telling a professor otherwise is a lie they read
# on their own page.
QUEUED_REQUEST_STATUS = "preparing"


class RetryNotEligibleError(Exception):
    """Raised when a retry is refused. `status_code` is what the route returns.

    404 means there is nothing to retry — no request, no run. 409 means there
    is something, and it is not in a state an admin may replace.
    """

    def __init__(self, message: str, *, status_code: int = 409) -> None:
        super().__init__(message)
        self.status_code = status_code


def _carried_count(run: Mapping[str, Any], preparation: Mapping[str, Any], *keys: str) -> int:
    """The count the replacement run inherits.

    The retired run is the first source: it is what was actually queued, and
    matching it is what makes "same dataset, corrected code" true. Preparation
    is the fallback for a run written before a count was recorded — never a
    recount, because recounting would silently change the dataset the retry
    promised to preserve.
    """
    for source, key in ((run, keys[0]), (preparation, keys[-1])):
        value = as_int(source.get(key))
        if value > 0:
            return value
    return 0


def retry_training_run(
    conn: Any,
    course_id: str,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Supersede this course's current run and queue a replacement.

    Returns `{"superseded": ..., "run": ..., "request": ...}`. Raises
    `RetryNotEligibleError` rather than writing anything when the current run
    is one an admin may not replace.
    """
    safe_course_id = assert_valid_course_id(course_id)
    moment = now or datetime.now(timezone.utc)

    # First statement of the transaction, deliberately. Everything below is a
    # decision made from what this read returns.
    request = db_model_requests.lock_model_request(conn, safe_course_id)
    if request is None:
        raise RetryNotEligibleError(
            f'Course "{safe_course_id}" has no model request to retry.',
            status_code=404,
        )

    run_id = request.get("currentRunId")
    if not run_id:
        raise RetryNotEligibleError(
            f'Course "{safe_course_id}" has no training run to retry. '
            "Queue one instead.",
            status_code=404,
        )

    previous = db_training_runs.get_training_run(conn, safe_course_id, run_id)
    if previous is None:
        raise RetryNotEligibleError(
            f'Training run "{run_id}" was not found for course '
            f'"{safe_course_id}".',
            status_code=404,
        )

    block = db_training_runs.training_run_retry_block(previous, now=moment)
    if block is not None:
        raise RetryNotEligibleError(block)

    preparation = request.get("preparation")
    preparation = preparation if isinstance(preparation, Mapping) else {}

    # The dataset reference is carried, never rebuilt. If neither the run nor
    # the preparation record has one there is nothing to point a job at, and
    # queueing a run without it would fail on the cluster instead of here.
    dataset_ref = previous.get("datasetRef") or preparation.get("datasetRef") or ""
    if not str(dataset_ref).strip():
        raise RetryNotEligibleError(
            f'Course "{safe_course_id}" has no prepared dataset to retry. '
            "Prepare the training data first."
        )

    superseded = db_training_runs.supersede_training_run(
        conn, safe_course_id, run_id, now=moment
    )

    # With the previous run terminal, this course has no active run, so the
    # conditional INSERT inside `enqueue_training_run` can land. If some other
    # run is still active it raises `ActiveTrainingRunError`, which the route
    # turns into a 409 — the right answer, and one this module must not paper
    # over by picking a run to retire on its own.
    created = db_training_runs.enqueue_training_run(
        conn,
        safe_course_id,
        mode=previous.get("mode") or "full",
        dataset_ref=str(dataset_ref),
        approved_example_count=_carried_count(
            previous, preparation, "approvedExampleCount", "sourceApprovedExampleCount"
        ),
        train_examples=_carried_count(previous, preparation, "trainExamples"),
        validation_examples=_carried_count(previous, preparation, "validationExamples"),
    )

    updated_request = db_model_requests.update_model_request(
        conn,
        safe_course_id,
        {
            "status": QUEUED_REQUEST_STATUS,
            "updatedAt": moment.isoformat(),
            "currentRunId": created["runId"],
            # The retired run's Slurm job. Leaving it would show the course as
            # training under a job that finished long ago, which is the exact
            # confusion this feature exists to end.
            "training": None,
            "launchError": None,
            "failureMessage": None,
        },
    )

    return {
        "superseded": superseded,
        "run": created,
        "request": updated_request,
    }
