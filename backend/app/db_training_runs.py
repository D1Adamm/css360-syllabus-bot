"""PostgreSQL repository for the per-course training run queue.

This is the durable queue the cluster runner claims work from. Matches
`TrainingRun` exactly. The claim is the one shape that differs between storage
and API: the schema keeps `claim_owner`/`claim_claimed_at`/`claim_expires_at`
as columns because a lease gets queried — who holds this, has it expired —
while the API nests them as `claim`, because that is what `parseTrainingRun`
reads. A claim missing any of the three is reported as no claim at all,
matching `parseClaim`: a lease without an owner or an expiry cannot be reasoned
about.

Two runners are kept apart by `claim_next_training_run`, which selects one
eligible row `FOR UPDATE SKIP LOCKED` and stamps the lease in the same
statement pair. A second runner arriving mid-claim does not block on the locked
row and does not see it: it skips straight past to the next eligible run, or
finds none. The transaction is deliberately short — it covers choosing and
stamping, and nothing else. Training itself happens long after it has
committed, which is why the lease has an expiry rather than a held lock: a
runner that dies mid-job releases nothing, and the expiry is what lets the work
be taken again instead of being stranded forever.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from app.course_id import assert_valid_course_id
from app.db_mapping import (
    as_int,
    bind_jsonb,
    build_patch,
    optional_string,
    put_optional,
    to_iso,
)

RUN_COLUMNS = """
    run_id, course_id, mode, state, enqueued_at, updated_at, dataset_ref,
    approved_example_count, train_examples, validation_examples, attempt,
    job_id, claim_owner, claim_claimed_at, claim_expires_at, error, completion
"""

RUN_STATES = ("queued", "claimed", "submitted", "training", "succeeded", "failed")

# Everything else is outstanding work, and outstanding work blocks a second run.
TERMINAL_RUN_STATES = ("succeeded", "failed")

# What an admin retry writes onto the run it retires.
#
# The state stays `failed` rather than becoming a new `superseded` one. Adding
# a state is cheap in PostgreSQL — `state` is plain TEXT with no CHECK — and
# expensive everywhere else: `parseTrainingRun` in `src/lib/trainingRunDb.ts`
# drops any run whose state is not in its union, so an unrecognised state makes
# the old run disappear from the history this feature exists to preserve, on
# every browser that has not yet loaded the new bundle. `failed` is already
# terminal, already rendered, already understood by the worker, and the reason
# string says plainly what happened.
SUPERSEDED_ERROR = "Superseded by admin retry"

# States an admin may supersede outright.
#
# Only `failed`, which is terminal: nothing is running, so nothing can report
# back. Every other retryable case has to earn it, in
# `training_run_retry_block`.
RETRYABLE_RUN_STATES = ("failed",)

# States where a Slurm job may exist, so a retry needs evidence of staleness.
#
# The backend cannot see Slurm. It cannot tell "the job finished and the
# callback was lost" from "the job is still running", and those two need
# opposite answers: the first is exactly what retry is for, the second means a
# live job is about to report against a run an admin just retired.
#
# Age is the only evidence available. A run whose row has not been touched for
# `SUBMITTED_STALE_AFTER_SECONDS` is one nothing is reporting on, because a
# healthy job's own callbacks are what would have touched it.
LIVE_JOB_RUN_STATES = ("submitted", "training")

# Six hours. Long enough that no healthy run is superseded by an impatient
# click — a full QLoRA run on a course this size finishes well inside it, and a
# queue wait does not move `updated_at`. Short enough that a genuinely lost
# callback is recoverable the same working day.
#
# This is a floor, not a diagnosis. The ownership guard in
# `update_model_request_for_run` is what makes a wrong retry harmless; this
# constant is what makes a wrong retry unlikely.
SUBMITTED_STALE_AFTER_SECONDS = 6 * 60 * 60

MODES = ("smoke", "full")

# How long a claim is held before another worker may retake it. Matches the
# runner's own default so a lease means the same thing on both sides.
DEFAULT_LEASE_SECONDS = 900

# What a worker is allowed to take.
#
#   - `job_id IS NULL`: a real scheduler id means a submission already
#     happened. That run is never taken again, whatever its state says.
#   - `queued`: nobody has it.
#   - `claimed` past its expiry: whoever had it is not coming back. A NULL
#     expiry counts as expired — a lease nothing can reason about is worse
#     stranded than retaken, and the second claim still has to win the lock.
#   - `course_ids` NULL means every course; otherwise exactly those.
#
# Written once and shared by the read-only listing and the claim so the two can
# never drift into disagreeing about what is claimable.
CLAIMABLE_PREDICATE = """
    job_id IS NULL
    AND (
        state = 'queued'
        OR (
            state = 'claimed'
            AND (claim_expires_at IS NULL OR claim_expires_at <= %(now)s)
        )
    )
    AND (%(course_ids)s::text[] IS NULL OR course_id = ANY(%(course_ids)s::text[]))
"""

RUN_PATCH_COLUMNS = {
    "state": "state",
    "updatedAt": "updated_at",
    "datasetRef": "dataset_ref",
    "approvedExampleCount": "approved_example_count",
    "trainExamples": "train_examples",
    "validationExamples": "validation_examples",
    "attempt": "attempt",
    "jobId": "job_id",
    "error": "error",
    # Flattened claim fields, patchable individually by a runner taking or
    # releasing a lease.
    "claimOwner": "claim_owner",
    "claimedAt": "claim_claimed_at",
    "claimExpiresAt": "claim_expires_at",
    # What the cluster reported when the job ended. Written once, by the
    # completion callback; nothing else patches it.
    "completion": "completion",
}

JSONB_COLUMNS = frozenset({"completion"})


class ActiveTrainingRunError(Exception):
    """Raised when a course already has a run that is not finished."""


def generate_training_run_id(now: datetime | None = None) -> str:
    """`run-<utc stamp>-<random>`, matching `generateTrainingRunId`.

    Same shape as the ids the browser already writes so a queue holding both
    sorts and reads identically.
    """
    moment = now or datetime.now(timezone.utc)
    stamp = (
        moment.astimezone(timezone.utc)
        .strftime("%Y%m%dT%H%M%S")
        .lower()
    )
    suffix = secrets.token_hex(3)
    return f"run-{stamp}z-{suffix}"


def map_training_run(row: Mapping[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "runId": row["run_id"],
        "courseId": row["course_id"],
        "mode": row["mode"],
        "state": row["state"],
        "enqueuedAt": to_iso(row.get("enqueued_at")),
        "updatedAt": to_iso(row.get("updated_at")) or to_iso(row.get("enqueued_at")),
        "datasetRef": row.get("dataset_ref") or "",
        "approvedExampleCount": as_int(row.get("approved_example_count")),
        "trainExamples": as_int(row.get("train_examples")),
        "validationExamples": as_int(row.get("validation_examples")),
        "attempt": as_int(row.get("attempt")),
    }

    put_optional(record, "jobId", optional_string(row.get("job_id")))
    put_optional(record, "error", optional_string(row.get("error")))

    completion = row.get("completion")
    if isinstance(completion, dict):
        record["completion"] = completion

    owner = optional_string(row.get("claim_owner"))
    claimed_at = to_iso(row.get("claim_claimed_at"))
    expires_at = to_iso(row.get("claim_expires_at"))
    if owner and claimed_at and expires_at:
        record["claim"] = {
            "owner": owner,
            "claimedAt": claimed_at,
            "expiresAt": expires_at,
        }

    return record


def list_training_runs(conn: Any, course_id: str) -> list[dict[str, Any]]:
    """Oldest first, matching `parseTrainingRuns`' enqueuedAt ordering."""
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {RUN_COLUMNS} FROM training_runs
            WHERE course_id = %s
            ORDER BY enqueued_at ASC, run_id ASC
            """,
            (safe_course_id,),
        )
        rows = cursor.fetchall()
    return [map_training_run(row) for row in rows]


def get_training_run(conn: Any, course_id: str, run_id: str) -> dict[str, Any] | None:
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            f"SELECT {RUN_COLUMNS} FROM training_runs "
            "WHERE course_id = %s AND run_id = %s",
            (safe_course_id, run_id),
        )
        row = cursor.fetchone()
    return map_training_run(row) if row else None


def find_active_training_run(conn: Any, course_id: str) -> dict[str, Any] | None:
    safe_course_id = assert_valid_course_id(course_id)
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {RUN_COLUMNS} FROM training_runs
            WHERE course_id = %s AND NOT (state = ANY(%s))
            ORDER BY enqueued_at ASC
            LIMIT 1
            """,
            (safe_course_id, list(TERMINAL_RUN_STATES)),
        )
        row = cursor.fetchone()
    return map_training_run(row) if row else None


def _seconds_since_update(run: Mapping[str, Any], now: datetime) -> float | None:
    """How long this run's row has been untouched, or None if unknowable.

    None means "treat as stale". A row with no readable `updatedAt` is one
    nothing has reported against in a way this can see, which is the same
    conclusion a very old timestamp supports.
    """
    raw = optional_string(run.get("updatedAt")) or optional_string(
        run.get("enqueuedAt")
    )
    if not raw:
        return None
    try:
        updated_at = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    return (now - updated_at).total_seconds()


def _describe_idle(seconds: float) -> str:
    if seconds < 90:
        return "less than a minute"
    if seconds < 5400:
        return f"{seconds / 60:.0f} minutes"
    return f"{seconds / 3600:.1f} hours"


def _claim_has_expired(run: Mapping[str, Any], now: datetime) -> bool:
    """True when nothing can be said to hold this run any more.

    A missing claim, or one whose expiry cannot be parsed, counts as expired —
    the same reading `CLAIMABLE_PREDICATE` gives a NULL `claim_expires_at`. A
    lease nothing can reason about is not a lease.
    """
    claim = run.get("claim")
    if not isinstance(claim, Mapping):
        return True

    raw = optional_string(claim.get("expiresAt"))
    if not raw:
        return True

    try:
        expires_at = datetime.fromisoformat(raw)
    except ValueError:
        return True

    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at <= now


def training_run_retry_block(
    run: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> str | None:
    """Why this run may not be superseded, or None when it may be.

    A returned string is the refusal an operator reads, so each one says what
    is true of the run rather than restating the policy.

    The policy in one place, because "may an admin retry this?" is asked by the
    route, by the tests, and — in a looser advisory form — by the button.

    The conservative half is `queued` and `claimed`. A queued run is already
    exactly what a retry would produce, so superseding one destroys a healthy
    run and creates its twin. A claimed run is being worked on by a runner
    right now; the only case the backend can *prove* is stale is a lease that
    has run out, which is the same evidence `claim_next_training_run` acts on
    when it retakes work.
    """
    moment = now or datetime.now(timezone.utc)
    state = run.get("state")

    if state in RETRYABLE_RUN_STATES:
        return None

    if state in LIVE_JOB_RUN_STATES:
        idle = _seconds_since_update(run, moment)
        if idle is None or idle >= SUBMITTED_STALE_AFTER_SECONDS:
            return None
        hours = SUBMITTED_STALE_AFTER_SECONDS / 3600
        job = run.get("jobId")
        return (
            f"This run reported {_describe_idle(idle)} ago"
            + (f" and its job ({job}) may still be running" if job else "")
            + f". A run is only retried once it has been silent for {hours:.0f} "
            "hours, so a live job is not retired underneath itself."
        )

    if state == "succeeded":
        return "This run succeeded. Retrying it would discard a finished result."

    if state == "queued":
        return (
            "This run is already queued and waiting for a worker. "
            "There is nothing to retry."
        )

    if state == "claimed":
        if _claim_has_expired(run, moment):
            return None
        claim = run.get("claim") or {}
        owner = claim.get("owner") or "a worker"
        return (
            f"This run is held by {owner} until {claim.get('expiresAt')}. "
            "Wait for the lease to expire before retrying."
        )

    return f'This run is in an unrecognised state ("{state}") and was not retried.'


def enqueue_training_run(
    conn: Any,
    course_id: str,
    *,
    mode: str,
    dataset_ref: str,
    approved_example_count: int = 0,
    train_examples: int = 0,
    validation_examples: int = 0,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Queue one run, refusing while this course has an active one.

    The guarded property is that no two runs for one course are outstanding at
    once. It is a conditional INSERT evaluated at write time, so a
    double-clicked button or two admin tabs cannot both win — the loser matches
    no rows and gets `ActiveTrainingRunError`, which the route turns into a 409.

    No job id is invented. Only the cluster produces one, and a placeholder
    would make an unsubmitted run look submitted.
    """
    safe_course_id = assert_valid_course_id(course_id)
    if mode not in MODES:
        raise ValueError(f"Unknown training mode: {mode!r}. Expected one of {MODES}.")

    now = datetime.now(timezone.utc)
    parameters = {
        "run_id": run_id or generate_training_run_id(now),
        "course_id": safe_course_id,
        "mode": mode,
        "state": "queued",
        "enqueued_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "dataset_ref": dataset_ref,
        "approved_example_count": max(0, as_int(approved_example_count)),
        "train_examples": max(0, as_int(train_examples)),
        "validation_examples": max(0, as_int(validation_examples)),
        # No runner has taken it yet. The runner increments as it claims.
        "attempt": 0,
        "terminal_states": list(TERMINAL_RUN_STATES),
    }

    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO training_runs (
                run_id, course_id, mode, state, enqueued_at, updated_at,
                dataset_ref, approved_example_count, train_examples,
                validation_examples, attempt
            )
            SELECT %(run_id)s, %(course_id)s, %(mode)s, %(state)s, %(enqueued_at)s,
                   %(updated_at)s, %(dataset_ref)s, %(approved_example_count)s,
                   %(train_examples)s, %(validation_examples)s, %(attempt)s
            WHERE NOT EXISTS (
                SELECT 1 FROM training_runs
                WHERE course_id = %(course_id)s
                  AND NOT (state = ANY(%(terminal_states)s))
            )
            """,
            parameters,
        )
        if cursor.rowcount == 0:
            raise ActiveTrainingRunError(
                f'Course "{safe_course_id}" already has an active training run.'
            )

    created = get_training_run(conn, safe_course_id, parameters["run_id"])
    if created is None:  # pragma: no cover - defensive
        raise ActiveTrainingRunError(
            f'The training run for "{safe_course_id}" could not be read back.'
        )
    return created


def update_training_run(
    conn: Any,
    course_id: str,
    run_id: str,
    patch: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Patch one run. Accepts a nested `claim` or the flattened claim fields.

    A runner taking a lease sends `claim`; releasing one sends `claim: null`,
    which clears all three columns together — a half-cleared lease would leave
    an owner with no expiry, which `parseClaim` would drop and an operator
    would have to guess at.
    """
    safe_course_id = assert_valid_course_id(course_id)
    if get_training_run(conn, safe_course_id, run_id) is None:
        return None

    normalized = dict(patch)
    if "claim" in normalized:
        claim = normalized.pop("claim")
        if isinstance(claim, Mapping):
            normalized["claimOwner"] = claim.get("owner")
            normalized["claimedAt"] = claim.get("claimedAt")
            normalized["claimExpiresAt"] = claim.get("expiresAt")
        else:
            normalized["claimOwner"] = None
            normalized["claimedAt"] = None
            normalized["claimExpiresAt"] = None

    assignments = build_patch(normalized, RUN_PATCH_COLUMNS)
    if not assignments:
        return get_training_run(conn, safe_course_id, run_id)

    assignments.setdefault("updated_at", datetime.now(timezone.utc).isoformat())

    sets = ", ".join(f"{column} = %({column})s" for column in assignments)
    parameters = {**assignments, "course_id": safe_course_id, "run_id": run_id}
    with conn.cursor() as cursor:
        cursor.execute(
            f"UPDATE training_runs SET {sets} "
            "WHERE course_id = %(course_id)s AND run_id = %(run_id)s",
            bind_jsonb(parameters, JSONB_COLUMNS),
        )

    return get_training_run(conn, safe_course_id, run_id)


# --------------------------------------------------------------------------- #
# Worker-facing queue operations
#
# Everything below is what a runner calls, and each one is a single short
# transaction. None of them is held open across the work they describe.
# --------------------------------------------------------------------------- #


def claimable_training_runs(
    conn: Any,
    *,
    now: datetime | None = None,
    course_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Runs a worker could take right now, oldest first. Read-only.

    Exists for `--dry-run`, which must be safe to point at the live queue: it
    shows what would be claimed while claiming nothing another runner could
    have had.
    """
    moment = now or datetime.now(timezone.utc)
    safe_course_ids = (
        [assert_valid_course_id(course_id) for course_id in course_ids]
        if course_ids is not None
        else None
    )

    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {RUN_COLUMNS} FROM training_runs
            WHERE {CLAIMABLE_PREDICATE}
            ORDER BY enqueued_at ASC, run_id ASC
            """,
            {"now": moment, "course_ids": safe_course_ids},
        )
        rows = cursor.fetchall()
    return [map_training_run(row) for row in rows]


def claim_next_training_run(
    conn: Any,
    *,
    owner: str,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    now: datetime | None = None,
    course_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    """Atomically take exactly one eligible run, or return None.

    `FOR UPDATE SKIP LOCKED` is the whole guarantee. The row is locked by the
    SELECT, so a second worker running the identical statement at the same
    moment cannot see it and cannot wait for it — it skips to the next eligible
    run. The UPDATE that stamps the lease runs against the row already held, so
    no third party can slip between choosing and claiming.

    `attempt` goes up on every claim, including one that follows an expired
    lease. A run that keeps being retaken this way is then visible as one that
    keeps being retried, rather than one that quietly vanished.
    """
    cleaned_owner = (owner or "").strip()
    if not cleaned_owner:
        raise ValueError("A claim needs a non-empty owner.")
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive.")

    moment = now or datetime.now(timezone.utc)
    safe_course_ids = (
        [assert_valid_course_id(course_id) for course_id in course_ids]
        if course_ids is not None
        else None
    )

    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT course_id, run_id FROM training_runs
            WHERE {CLAIMABLE_PREDICATE}
            ORDER BY enqueued_at ASC, run_id ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1
            """,
            {"now": moment, "course_ids": safe_course_ids},
        )
        row = cursor.fetchone()
        if row is None:
            return None

        cursor.execute(
            """
            UPDATE training_runs SET
                state = 'claimed',
                attempt = attempt + 1,
                updated_at = %(now)s,
                claim_owner = %(owner)s,
                claim_claimed_at = %(now)s,
                claim_expires_at = %(expires_at)s
            WHERE course_id = %(course_id)s AND run_id = %(run_id)s
            """,
            {
                "now": moment,
                "owner": cleaned_owner,
                "expires_at": moment + timedelta(seconds=lease_seconds),
                "course_id": row["course_id"],
                "run_id": row["run_id"],
            },
        )

    return get_training_run(conn, row["course_id"], row["run_id"])


def release_training_run(
    conn: Any,
    course_id: str,
    run_id: str,
    *,
    error: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Put a claimed run back on the queue, clearing its lease.

    Used when a runner finishes without submitting anything. Leaving it
    `claimed` would make it wait out a lease for no reason, and marking it
    failed would be untrue — nothing was attempted.
    """
    moment = now or datetime.now(timezone.utc)
    patch: dict[str, Any] = {
        "state": "queued",
        "updatedAt": moment.isoformat(),
        "claim": None,
    }
    if error is not None:
        patch["error"] = error
    return update_training_run(conn, course_id, run_id, patch)


def fail_training_run(
    conn: Any,
    course_id: str,
    run_id: str,
    *,
    error: str,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Mark a run terminally failed and release its lease.

    Terminal on purpose: this is what a worker reports when training itself
    failed, as opposed to a submission that never happened. `TERMINAL_RUN_STATES`
    includes `failed`, so the course is free to queue another run.
    """
    moment = now or datetime.now(timezone.utc)
    return update_training_run(
        conn,
        course_id,
        run_id,
        {
            "state": "failed",
            "updatedAt": moment.isoformat(),
            "error": error,
            "claim": None,
        },
    )


def supersede_training_run(
    conn: Any,
    course_id: str,
    run_id: str,
    *,
    error: str = SUPERSEDED_ERROR,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Retire a run an admin has replaced, keeping everything it recorded.

    `fail_training_run` does the write — the retired run is terminally failed,
    which is what makes the course free to queue its replacement. The reason
    string is what distinguishes it from a run that failed on the cluster.

    Nothing else on the row is touched. `job_id`, `attempt`, `enqueued_at` and
    the counts stay exactly as they were, because the point of retiring rather
    than deleting is that an operator can still see which Slurm job this course
    was waiting on and how long it waited.
    """
    return fail_training_run(conn, course_id, run_id, error=error, now=now)


def mark_training_run_submitted(
    conn: Any,
    course_id: str,
    run_id: str,
    *,
    job_id: str,
    train_examples: int,
    validation_examples: int,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Record a real scheduler job id against a claimed run.

    The lease is cleared: the job exists on the cluster now, and nothing about
    it depends on this runner staying alive. `is_claimable` refuses anything
    carrying a `job_id`, so a submitted run is not retaken even though it is
    no longer leased.
    """
    moment = now or datetime.now(timezone.utc)
    cleaned_job_id = (job_id or "").strip()
    if not cleaned_job_id:
        raise ValueError("A submitted run needs a job id.")

    return update_training_run(
        conn,
        course_id,
        run_id,
        {
            "state": "submitted",
            "updatedAt": moment.isoformat(),
            "jobId": cleaned_job_id,
            "trainExamples": max(0, as_int(train_examples)),
            "validationExamples": max(0, as_int(validation_examples)),
            "error": None,
            "claim": None,
        },
    )
