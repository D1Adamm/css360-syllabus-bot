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
from app.db_mapping import as_int, build_patch, optional_string, put_optional, to_iso

RUN_COLUMNS = """
    run_id, course_id, mode, state, enqueued_at, updated_at, dataset_ref,
    approved_example_count, train_examples, validation_examples, attempt,
    job_id, claim_owner, claim_claimed_at, claim_expires_at, error
"""

RUN_STATES = ("queued", "claimed", "submitted", "training", "succeeded", "failed")

# Everything else is outstanding work, and outstanding work blocks a second run.
TERMINAL_RUN_STATES = ("succeeded", "failed")

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
}


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
            parameters,
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
