"""PostgreSQL repository for the per-course training run queue.

Matches `TrainingRun` exactly. The claim is the one shape that differs between
storage and API: the schema keeps `claim_owner`/`claim_claimed_at`/
`claim_expires_at` as columns because a lease gets queried — who holds this, has
it expired — while the API nests them as `claim`, because that is what
`parseTrainingRun` reads. A claim missing any of the three is reported as no
claim at all, matching `parseClaim`: a lease without an owner or an expiry
cannot be reasoned about.

Nothing here changes training orchestration. The runner on the cluster still
reads the Firebase queue; these functions exist so the same queue can be served
from PostgreSQL after a cutover.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
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

    The guard is the property `enqueueTrainingRun` protects with a Firebase
    transaction over the whole node: no two runs for one course outstanding at
    once. Here it is a conditional INSERT evaluated at write time, so a
    double-clicked button or two admin tabs cannot both win.

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


def upsert_training_run(
    conn: Any,
    course_id: str,
    run_id: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Store a run under an id chosen elsewhere, creating or refreshing it.

    Used to mirror a run that Firebase has already accepted. Deliberately not
    `enqueue_training_run`: that one guards against a second active run, and
    re-running the guard here could refuse a run the queue has already taken —
    leaving the cluster holding work the UI would never show.

    The duplicate decision belongs to whichever store owns the queue. Today
    that is Firebase.
    """
    safe_course_id = assert_valid_course_id(course_id)

    parameters = {
        "run_id": run_id,
        "course_id": safe_course_id,
        "mode": record["mode"],
        "state": record["state"],
        "enqueued_at": record["enqueuedAt"],
        "updated_at": record.get("updatedAt") or record["enqueuedAt"],
        "dataset_ref": record.get("datasetRef") or "",
        "approved_example_count": max(0, as_int(record.get("approvedExampleCount"))),
        "train_examples": max(0, as_int(record.get("trainExamples"))),
        "validation_examples": max(0, as_int(record.get("validationExamples"))),
        "attempt": max(0, as_int(record.get("attempt"))),
        "job_id": optional_string(record.get("jobId")),
        "error": optional_string(record.get("error")),
    }

    columns = list(parameters)
    placeholders = ", ".join(f"%({column})s" for column in columns)
    updates = ", ".join(
        f"{column} = EXCLUDED.{column}"
        for column in columns
        if column not in ("course_id", "run_id")
    )

    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO training_runs ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT (course_id, run_id) DO UPDATE SET {updates}
            """,
            parameters,
        )

    stored = get_training_run(conn, safe_course_id, run_id)
    if stored is None:  # pragma: no cover - defensive
        raise ActiveTrainingRunError(
            f'The mirrored training run "{run_id}" could not be read back.'
        )
    return stored
