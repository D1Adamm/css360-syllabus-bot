"""Admin retry: what it retires, what it creates, and what it refuses.

The bug this closes
-------------------
A run reaches `submitted`, the Slurm job finishes, and the completion callback
never lands. PostgreSQL is left saying `model_requests.status = training` and
`training_runs.state = submitted` forever. Nothing could clear that: the run
will not finish, and while it is outstanding the one-active-run guard correctly
refuses a replacement. There was no supported recovery path, and the only
untried option was raw SQL against production.

What is faked, and what is not
------------------------------
`FakeConnection` here is stateful, unlike the statement recorder in
`test_db_repositories.py`. It keeps `model_requests` and `training_runs` rows in
dictionaries and answers the handful of statements these repositories actually
issue, including the conditional INSERT's `NOT EXISTS` guard. That is the point:
a retry is a sequence of reads and writes across two tables, and asserting on
the SQL text would say nothing about whether the sequence leaves the course in a
state a worker can pick up.

What it cannot tell you is whether `SELECT ... FOR UPDATE` blocks a concurrent
transaction — no server is involved. Two things stand in for that. The lock is
asserted to be taken, and taken first; and the duplicate test runs the second
retry *after* the first has committed its writes, which is exactly the state the
second transaction sees when the lock releases. What it must not do — create a
second active run — is then a real assertion rather than a mocked one.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import db_model_requests, db_training_runs
from app.db_training_runs import (
    SUBMITTED_STALE_AFTER_SECONDS,
    SUPERSEDED_ERROR,
    TERMINAL_RUN_STATES,
    training_run_retry_block,
)
from app.main import app
from app.training_retry import RetryNotEligibleError, retry_training_run

COURSE = "css-350-spring-2026-n3h9"
OTHER_COURSE = "css-360-winter-2026-a7rp"

NOW = datetime(2026, 8, 26, 17, 0, tzinfo=timezone.utc)

# The real stuck state this exists for: a submitted run whose Slurm job has
# long since finished, and a request that still believes it is training.
STALE_RUN_ID = "run-20260823t064333z-3c94f0"
STALE_JOB_ID = "253552"
DATASET_REF = f"exports/{COURSE}"


# --------------------------------------------------------------------------- #
# A stateful stand-in for the two tables a retry touches.
# --------------------------------------------------------------------------- #


def _normalize(sql: str) -> str:
    return " ".join(sql.split())


class FakeCursor:
    def __init__(self, connection: "FakeConnection") -> None:
        self._connection = connection
        self._rows: list[dict[str, Any]] = []
        self.rowcount = 0

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        statement = _normalize(sql)
        self._connection.statements.append(statement)
        self._rows, self.rowcount = self._connection.run(statement, params)

    def fetchone(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self._rows)


class FakeConnection:
    """`model_requests` and `training_runs` as dictionaries."""

    def __init__(self) -> None:
        self.requests: dict[str, dict[str, Any]] = {}
        self.runs: dict[tuple[str, str], dict[str, Any]] = {}
        self.statements: list[str] = []
        self.locked: list[str] = []

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    # -- statement dispatch ------------------------------------------------- #

    def run(self, statement: str, params: Any) -> tuple[list[dict[str, Any]], int]:
        if statement.startswith("SELECT") and "FROM model_requests" in statement:
            course_id = params[0]
            if statement.endswith("FOR UPDATE"):
                self.locked.append(course_id)
            row = self.requests.get(course_id)
            return ([dict(row)] if row else [], 1 if row else 0)

        if statement.startswith("SELECT") and "FROM training_runs" in statement:
            if "AND run_id = %s" in statement:
                row = self.runs.get((params[0], params[1]))
                return ([dict(row)] if row else [], 1 if row else 0)

            if "job_id IS NULL" in statement:
                # `CLAIMABLE_PREDICATE` — what the Tillicum worker's --dry-run
                # actually reads. Reproduced rather than stubbed, because
                # "the worker can see it" is the claim being tested.
                course_ids = params.get("course_ids")
                rows = [
                    dict(row)
                    for key, row in self.runs.items()
                    if (course_ids is None or key[0] in course_ids)
                    and row["job_id"] is None
                    and (
                        row["state"] == "queued"
                        or (
                            row["state"] == "claimed"
                            and (
                                row["claim_expires_at"] is None
                                or str(row["claim_expires_at"])
                                <= params["now"].isoformat()
                            )
                        )
                    )
                ]
                rows.sort(key=lambda row: (str(row["enqueued_at"]), row["run_id"]))
                return (rows, len(rows))

            course_id = params[0] if isinstance(params, tuple) else params["course_id"]
            rows = [
                dict(row)
                for key, row in sorted(self.runs.items())
                if key[0] == course_id
            ]
            rows.sort(key=lambda row: (str(row["enqueued_at"]), row["run_id"]))
            return (rows, len(rows))

        if statement.startswith("INSERT INTO training_runs"):
            course_id = params["course_id"]
            # The conditional INSERT's guard, evaluated exactly where the real
            # one is: at write time, not before it.
            active = any(
                row["state"] not in TERMINAL_RUN_STATES
                for key, row in self.runs.items()
                if key[0] == course_id
            )
            if active:
                return ([], 0)
            self.runs[(course_id, params["run_id"])] = {
                "run_id": params["run_id"],
                "course_id": course_id,
                "mode": params["mode"],
                "state": params["state"],
                "enqueued_at": params["enqueued_at"],
                "updated_at": params["updated_at"],
                "dataset_ref": params["dataset_ref"],
                "approved_example_count": params["approved_example_count"],
                "train_examples": params["train_examples"],
                "validation_examples": params["validation_examples"],
                "attempt": params["attempt"],
                "job_id": None,
                "claim_owner": None,
                "claim_claimed_at": None,
                "claim_expires_at": None,
                "error": None,
            }
            return ([], 1)

        if statement.startswith("UPDATE training_runs SET"):
            row = self.runs.get((params["course_id"], params["run_id"]))
            if row is None:
                return ([], 0)
            for column, value in params.items():
                if column in ("course_id", "run_id"):
                    continue
                row[column] = value
            return ([], 1)

        if statement.startswith("UPDATE model_requests SET"):
            row = self.requests.get(params["course_id"])
            if row is None:
                return ([], 0)
            if "current_run_id = %(expected_run_id)s" in statement:
                # The ownership guard, reproduced rather than assumed: a write
                # from a run that no longer owns the request matches no rows.
                owner = row.get("current_run_id")
                if owner is not None and owner != params["expected_run_id"]:
                    return ([], 0)
            for column, value in params.items():
                if column in ("course_id", "expected_run_id"):
                    continue
                # `_bind` wraps JSONB in psycopg's Json adapter; unwrap so the
                # stored row stays a plain dict the mapper can read back.
                row[column] = getattr(value, "obj", value)
            return ([], 1)

        raise AssertionError(f"Unexpected statement: {statement}")

    # -- fixtures ----------------------------------------------------------- #

    def add_request(self, course_id: str = COURSE, **overrides: Any) -> None:
        self.requests[course_id] = {
            "course_id": course_id,
            "status": "training",
            "requested_at": "2026-08-20T09:00:00+00:00",
            "updated_at": "2026-08-23T06:43:33+00:00",
            "approved_example_count": 42,
            "failure_message": None,
            "preparation": {
                "preparedAt": "2026-08-23T06:40:00+00:00",
                "sourceApprovedExampleCount": 42,
                "datasetRef": f"exports/{course_id}",
                "trainExamples": 37,
                "validationExamples": 5,
                "splitSeed": 350,
            },
            "preparation_error": None,
            "training": {
                "jobId": STALE_JOB_ID,
                "mode": "full",
                "submittedAt": "2026-08-23T06:43:33+00:00",
                "datasetRef": f"exports/{course_id}",
                "trainExamples": 37,
                "validationExamples": 5,
            },
            "launch_error": None,
            "current_run_id": STALE_RUN_ID,
            **overrides,
        }

    def add_run(
        self,
        course_id: str = COURSE,
        run_id: str = STALE_RUN_ID,
        **overrides: Any,
    ) -> None:
        self.runs[(course_id, run_id)] = {
            "run_id": run_id,
            "course_id": course_id,
            "mode": "full",
            "state": "submitted",
            "enqueued_at": "2026-08-23T06:40:00+00:00",
            "updated_at": "2026-08-23T06:43:33+00:00",
            "dataset_ref": f"exports/{course_id}",
            "approved_example_count": 42,
            "train_examples": 37,
            "validation_examples": 5,
            "attempt": 1,
            "job_id": STALE_JOB_ID,
            "claim_owner": None,
            "claim_claimed_at": None,
            "claim_expires_at": None,
            "error": None,
            **overrides,
        }

    # -- assertions helpers ------------------------------------------------- #

    def course_runs(self, course_id: str = COURSE) -> list[dict[str, Any]]:
        return db_training_runs.list_training_runs(self, course_id)

    def active_runs(self, course_id: str = COURSE) -> list[dict[str, Any]]:
        return [
            run
            for run in self.course_runs(course_id)
            if run["state"] not in TERMINAL_RUN_STATES
        ]


def _stale_store(**run_overrides: Any) -> FakeConnection:
    connection = FakeConnection()
    connection.add_request()
    connection.add_run(**run_overrides)
    return connection


# --------------------------------------------------------------------------- #
# Eligibility
# --------------------------------------------------------------------------- #


class RetryEligibilityTests(unittest.TestCase):
    """The policy, isolated from any storage.

    Stated as one helper so the route, the tests and the button cannot drift
    into disagreeing about what "retryable" means.
    """

    def _run(self, **overrides: Any) -> dict[str, Any]:
        return {
            "runId": STALE_RUN_ID,
            "state": "submitted",
            "mode": "full",
            "datasetRef": DATASET_REF,
            **overrides,
        }

    def test_a_stale_submitted_run_may_be_retried(self) -> None:
        self.assertIsNone(training_run_retry_block(self._run(), now=NOW))

    def test_a_training_run_may_be_retried(self) -> None:
        self.assertIsNone(
            training_run_retry_block(self._run(state="training"), now=NOW)
        )

    def test_a_failed_run_may_be_retried(self) -> None:
        self.assertIsNone(training_run_retry_block(self._run(state="failed"), now=NOW))

    def test_a_queued_run_is_refused(self) -> None:
        """It is already exactly what a retry would produce."""
        block = training_run_retry_block(self._run(state="queued"), now=NOW)
        self.assertIsNotNone(block)
        self.assertIn("already queued", block or "")

    def test_a_live_claim_is_refused_and_names_who_holds_it(self) -> None:
        block = training_run_retry_block(
            self._run(
                state="claimed",
                claim={
                    "owner": "adam@tillicum",
                    "claimedAt": NOW.isoformat(),
                    "expiresAt": (NOW + timedelta(minutes=15)).isoformat(),
                },
            ),
            now=NOW,
        )
        self.assertIsNotNone(block)
        self.assertIn("adam@tillicum", block or "")

    def test_an_expired_claim_may_be_retried(self) -> None:
        """The one claimed case the backend can prove is stale.

        Same evidence `claim_next_training_run` acts on when it retakes work.
        """
        self.assertIsNone(
            training_run_retry_block(
                self._run(
                    state="claimed",
                    claim={
                        "owner": "adam@tillicum",
                        "claimedAt": (NOW - timedelta(hours=2)).isoformat(),
                        "expiresAt": (NOW - timedelta(hours=1)).isoformat(),
                    },
                ),
                now=NOW,
            )
        )

    def test_a_claim_with_no_lease_at_all_may_be_retried(self) -> None:
        self.assertIsNone(
            training_run_retry_block(self._run(state="claimed"), now=NOW)
        )

    def test_a_succeeded_run_is_refused(self) -> None:
        """There is a result. Replacing it is a promotion decision, not a retry."""
        block = training_run_retry_block(self._run(state="succeeded"), now=NOW)
        self.assertIsNotNone(block)
        self.assertIn("succeeded", block or "")


# --------------------------------------------------------------------------- #
# The transaction
# --------------------------------------------------------------------------- #


class RetryTransactionTests(unittest.TestCase):
    def test_a_stale_submitted_run_is_retired_and_replaced(self) -> None:
        connection = _stale_store()

        result = retry_training_run(connection, COURSE, now=NOW)

        self.assertEqual(result["superseded"]["runId"], STALE_RUN_ID)
        self.assertEqual(result["run"]["state"], "queued")
        self.assertNotEqual(result["run"]["runId"], STALE_RUN_ID)

    def test_a_stale_training_run_is_retired_and_replaced(self) -> None:
        connection = _stale_store(state="training")

        result = retry_training_run(connection, COURSE, now=NOW)

        self.assertEqual(result["superseded"]["state"], "failed")
        self.assertEqual(result["run"]["state"], "queued")

    def test_a_failed_run_is_retried(self) -> None:
        connection = _stale_store(state="failed", error="CUDA out of memory")

        result = retry_training_run(connection, COURSE, now=NOW)

        self.assertEqual(result["run"]["state"], "queued")
        # The new reason replaces the old one; the run is still terminal.
        self.assertEqual(result["superseded"]["error"], SUPERSEDED_ERROR)

    def test_the_previous_run_stays_in_history(self) -> None:
        """Never deleted. This is the whole point of retiring rather than removing."""
        connection = _stale_store()

        retry_training_run(connection, COURSE, now=NOW)

        run_ids = [run["runId"] for run in connection.course_runs()]
        self.assertIn(STALE_RUN_ID, run_ids)
        self.assertEqual(len(run_ids), 2)

    def test_the_previous_run_becomes_terminal_with_a_stated_reason(self) -> None:
        connection = _stale_store()

        retry_training_run(connection, COURSE, now=NOW)

        previous = db_training_runs.get_training_run(connection, COURSE, STALE_RUN_ID)
        self.assertIn(previous["state"], TERMINAL_RUN_STATES)
        self.assertEqual(previous["error"], SUPERSEDED_ERROR)

    def test_the_previous_run_keeps_its_job_id_and_attempt_history(self) -> None:
        """An operator asking what this course was waiting on must still find it."""
        connection = _stale_store()

        retry_training_run(connection, COURSE, now=NOW)

        previous = db_training_runs.get_training_run(connection, COURSE, STALE_RUN_ID)
        self.assertEqual(previous["jobId"], STALE_JOB_ID)
        self.assertEqual(previous["attempt"], 1)
        self.assertEqual(previous["enqueuedAt"], "2026-08-23T06:40:00+00:00")

    def test_exactly_one_new_queued_run_exists_afterwards(self) -> None:
        connection = _stale_store()

        retry_training_run(connection, COURSE, now=NOW)

        active = connection.active_runs()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["state"], "queued")

    def test_the_new_run_has_a_different_id(self) -> None:
        connection = _stale_store()

        result = retry_training_run(connection, COURSE, now=NOW)

        self.assertNotEqual(result["run"]["runId"], STALE_RUN_ID)
        self.assertTrue(result["run"]["runId"].startswith("run-"))

    def test_the_new_run_carries_the_prepared_dataset_forward(self) -> None:
        """No export is rerun, no split recomputed, no seed read."""
        connection = _stale_store()

        created = retry_training_run(connection, COURSE, now=NOW)["run"]

        self.assertEqual(created["datasetRef"], DATASET_REF)
        self.assertEqual(created["approvedExampleCount"], 42)
        self.assertEqual(created["trainExamples"], 37)
        self.assertEqual(created["validationExamples"], 5)
        self.assertEqual(created["mode"], "full")

    def test_the_new_run_starts_unclaimed_and_unsubmitted(self) -> None:
        """It has to look like an ordinary queued run to the Tillicum worker."""
        connection = _stale_store()

        created = retry_training_run(connection, COURSE, now=NOW)["run"]

        self.assertEqual(created["attempt"], 0)
        self.assertNotIn("jobId", created)
        self.assertNotIn("claim", created)

    def test_counts_fall_back_to_the_preparation_record(self) -> None:
        """A run written before counts were recorded still retries correctly.

        The fallback is the preparation record, never a recount — recounting
        would silently change the dataset the retry promised to preserve.
        """
        connection = _stale_store(
            dataset_ref="", train_examples=0, validation_examples=0,
            approved_example_count=0,
        )

        created = retry_training_run(connection, COURSE, now=NOW)["run"]

        self.assertEqual(created["datasetRef"], DATASET_REF)
        self.assertEqual(created["trainExamples"], 37)
        self.assertEqual(created["validationExamples"], 5)
        self.assertEqual(created["approvedExampleCount"], 42)

    def test_the_request_points_at_the_new_run(self) -> None:
        connection = _stale_store()

        result = retry_training_run(connection, COURSE, now=NOW)

        request = connection.requests[COURSE]
        self.assertEqual(request["current_run_id"], result["run"]["runId"])

    def test_the_stale_slurm_job_metadata_is_cleared(self) -> None:
        """The exact confusion this feature exists to end."""
        connection = _stale_store()

        retry_training_run(connection, COURSE, now=NOW)

        self.assertIsNone(connection.requests[COURSE]["training"])

    def test_the_request_moves_back_to_the_queued_status(self) -> None:
        """`preparing` — nothing is training, a run is waiting to be picked up."""
        connection = _stale_store()

        retry_training_run(connection, COURSE, now=NOW)

        self.assertEqual(connection.requests[COURSE]["status"], "preparing")

    def test_the_approved_examples_and_preparation_record_are_untouched(self) -> None:
        connection = _stale_store()
        before = dict(connection.requests[COURSE]["preparation"])

        retry_training_run(connection, COURSE, now=NOW)

        self.assertEqual(connection.requests[COURSE]["preparation"], before)
        self.assertEqual(connection.requests[COURSE]["approved_example_count"], 42)

    def test_the_request_row_is_locked_before_anything_is_decided(self) -> None:
        """The concurrency guard. First statement, not merely somewhere."""
        connection = _stale_store()

        retry_training_run(connection, COURSE, now=NOW)

        self.assertEqual(connection.locked, [COURSE])
        self.assertTrue(connection.statements[0].endswith("FOR UPDATE"))
        self.assertIn("FROM model_requests", connection.statements[0])


class RetryRefusalTests(unittest.TestCase):
    def test_a_queued_run_is_refused_and_nothing_is_written(self) -> None:
        connection = _stale_store(state="queued", job_id=None)

        with self.assertRaises(RetryNotEligibleError) as caught:
            retry_training_run(connection, COURSE, now=NOW)

        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(len(connection.course_runs()), 1)
        self.assertEqual(connection.course_runs()[0]["state"], "queued")

    def test_a_live_claim_is_refused_and_nothing_is_written(self) -> None:
        connection = _stale_store(
            state="claimed",
            job_id=None,
            claim_owner="adam@tillicum",
            claim_claimed_at=NOW.isoformat(),
            claim_expires_at=(NOW + timedelta(minutes=15)).isoformat(),
        )

        with self.assertRaises(RetryNotEligibleError):
            retry_training_run(connection, COURSE, now=NOW)

        self.assertEqual(len(connection.course_runs()), 1)
        self.assertEqual(connection.course_runs()[0]["state"], "claimed")

    def test_a_succeeded_run_is_refused(self) -> None:
        connection = _stale_store(state="succeeded")

        with self.assertRaises(RetryNotEligibleError):
            retry_training_run(connection, COURSE, now=NOW)

    def test_a_course_with_no_model_request_is_a_404(self) -> None:
        connection = FakeConnection()

        with self.assertRaises(RetryNotEligibleError) as caught:
            retry_training_run(connection, COURSE, now=NOW)

        self.assertEqual(caught.exception.status_code, 404)

    def test_a_request_with_no_current_run_is_a_404(self) -> None:
        connection = FakeConnection()
        connection.add_request(current_run_id=None)

        with self.assertRaises(RetryNotEligibleError) as caught:
            retry_training_run(connection, COURSE, now=NOW)

        self.assertEqual(caught.exception.status_code, 404)
        self.assertIn("Queue one instead", str(caught.exception))

    def test_a_dangling_current_run_pointer_is_a_404(self) -> None:
        connection = FakeConnection()
        connection.add_request()

        with self.assertRaises(RetryNotEligibleError) as caught:
            retry_training_run(connection, COURSE, now=NOW)

        self.assertEqual(caught.exception.status_code, 404)

    def test_a_request_with_no_prepared_dataset_is_refused(self) -> None:
        connection = FakeConnection()
        connection.add_request(preparation=None)
        connection.add_run(dataset_ref="")

        with self.assertRaises(RetryNotEligibleError) as caught:
            retry_training_run(connection, COURSE, now=NOW)

        self.assertIn("no prepared dataset", str(caught.exception))
        self.assertEqual(len(connection.course_runs()), 1)


class RetryConcurrencyTests(unittest.TestCase):
    def test_a_second_retry_does_not_create_a_second_active_run(self) -> None:
        """Double-clicked button, or a second admin.

        The second call is made against the state the first one committed —
        which is precisely what the second transaction sees once the row lock
        releases. It finds a freshly queued run and is refused.
        """
        connection = _stale_store()

        first = retry_training_run(connection, COURSE, now=NOW)

        with self.assertRaises(RetryNotEligibleError):
            retry_training_run(connection, COURSE, now=NOW + timedelta(seconds=1))

        active = connection.active_runs()
        self.assertEqual([run["runId"] for run in active], [first["run"]["runId"]])
        self.assertEqual(len(connection.course_runs()), 2)

    def test_another_active_run_stops_the_retry_rather_than_being_papered_over(
        self,
    ) -> None:
        """A run this retry did not retire still blocks it.

        `enqueue_training_run` refuses, and the module must not respond by
        picking another run to retire on its own.
        """
        connection = _stale_store()
        connection.add_run(run_id="run-20260824t000000z-abcdef", state="queued",
                           job_id=None, enqueued_at="2026-08-24T00:00:00+00:00")

        with self.assertRaises(db_training_runs.ActiveTrainingRunError):
            retry_training_run(connection, COURSE, now=NOW)

    def test_only_the_named_course_is_touched(self) -> None:
        connection = _stale_store()
        connection.add_request(OTHER_COURSE)
        connection.add_run(OTHER_COURSE, run_id="run-20260823t000000z-other")
        before = dict(connection.runs[(OTHER_COURSE, "run-20260823t000000z-other")])

        retry_training_run(connection, COURSE, now=NOW)

        self.assertEqual(
            connection.runs[(OTHER_COURSE, "run-20260823t000000z-other")], before
        )
        self.assertEqual(connection.requests[OTHER_COURSE]["status"], "training")
        self.assertEqual(len(connection.course_runs(OTHER_COURSE)), 1)


# --------------------------------------------------------------------------- #
# The route
# --------------------------------------------------------------------------- #


class RetryRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    @contextmanager
    def _stubbed(
        self, connection: FakeConnection, *, course_exists: bool = True
    ) -> Iterator[None]:
        @contextmanager
        def _connect(**kwargs: Any) -> Iterator[FakeConnection]:
            yield connection

        with (
            patch("app.main.db_connection", _connect),
            patch("app.main.course_exists", return_value=course_exists),
        ):
            yield

    def test_a_retry_returns_both_halves_of_what_it_did(self) -> None:
        connection = _stale_store()

        with self._stubbed(connection):
            response = self.client.post(
                f"/api/courses/{COURSE}/training-runs/retry"
            )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["courseId"], COURSE)
        self.assertEqual(payload["supersededRunId"], STALE_RUN_ID)
        self.assertEqual(payload["supersededRun"]["error"], SUPERSEDED_ERROR)
        self.assertEqual(payload["run"]["state"], "queued")
        self.assertEqual(payload["run"]["runId"], payload["runId"])
        self.assertNotEqual(payload["runId"], STALE_RUN_ID)
        self.assertEqual(payload["requestStatus"], "preparing")

    def test_an_ineligible_run_is_409(self) -> None:
        connection = _stale_store(state="queued", job_id=None)

        with self._stubbed(connection):
            response = self.client.post(
                f"/api/courses/{COURSE}/training-runs/retry"
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("already queued", response.json()["detail"])

    def test_a_course_with_nothing_to_retry_is_404(self) -> None:
        with self._stubbed(FakeConnection()):
            response = self.client.post(
                f"/api/courses/{COURSE}/training-runs/retry"
            )

        self.assertEqual(response.status_code, 404)

    def test_an_unknown_course_is_404_before_any_write(self) -> None:
        connection = _stale_store()

        with self._stubbed(connection, course_exists=False):
            response = self.client.post(
                f"/api/courses/{COURSE}/training-runs/retry"
            )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(len(connection.course_runs()), 1)

    def test_an_invalid_course_id_is_rejected_before_any_statement(self) -> None:
        connection = FakeConnection()

        with self._stubbed(connection):
            response = self.client.post(
                "/api/courses/..%2Fetc/training-runs/retry"
            )

        self.assertIn(response.status_code, (400, 404))
        self.assertEqual(connection.statements, [])

    def test_a_second_request_arriving_after_the_first_is_refused(self) -> None:
        """Two nearly simultaneous clicks, serialized as the lock serializes them."""
        connection = _stale_store()

        with self._stubbed(connection):
            first = self.client.post(f"/api/courses/{COURSE}/training-runs/retry")
            second = self.client.post(f"/api/courses/{COURSE}/training-runs/retry")

        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 409)
        self.assertEqual(len(connection.active_runs()), 1)

    def test_an_unreachable_database_is_503_not_a_false_success(self) -> None:
        from app.db import DatabaseConfigurationError

        def _boom(**kwargs: Any) -> Any:
            raise DatabaseConfigurationError("PostgreSQL is not configured.")

        with patch("app.main.db_connection", _boom):
            response = self.client.post(
                f"/api/courses/{COURSE}/training-runs/retry"
            )

        self.assertEqual(response.status_code, 503)

    def test_the_retry_leaves_a_run_the_worker_can_claim(self) -> None:
        """Requirement nine, asserted rather than assumed.

        `claimable_training_runs` is the exact read behind
        `run_training_queue.sh --once --dry-run`.
        """
        connection = _stale_store()

        with self._stubbed(connection):
            response = self.client.post(
                f"/api/courses/{COURSE}/training-runs/retry"
            )

        claimable = db_training_runs.claimable_training_runs(
            connection, now=NOW, course_ids=[COURSE]
        )
        self.assertEqual(
            [run["runId"] for run in claimable], [response.json()["runId"]]
        )


# --------------------------------------------------------------------------- #
# The late-callback race
#
# The window this closes: an admin retries a run whose job is, in fact, still
# alive on the cluster. The old job finishes and reports — after the
# replacement run has already become current. Every one of these callbacks used
# to write `current_run_id` unconditionally, so the retired run would take the
# model request back from its own replacement.
#
# Driven through the real HTTP routes against the stateful store, because the
# claim being tested is about what two records say about each other after the
# call, not about which function was invoked.
# --------------------------------------------------------------------------- #


WORKER_TOKEN = "test-worker-token"
WORKER_HEADERS = {"X-Training-Worker-Token": WORKER_TOKEN}


class LateCallbackFromSupersededRunTests(unittest.TestCase):
    """A retired run must never steal the request back from its replacement."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        self.connection = _stale_store()
        # Retry first: this is the state every test below starts from.
        self.new_run_id = retry_training_run(self.connection, COURSE, now=NOW)["run"][
            "runId"
        ]

    @contextmanager
    def _worker(self) -> Iterator[None]:
        @contextmanager
        def _connect(**kwargs: Any) -> Iterator[FakeConnection]:
            yield self.connection

        with (
            patch.dict("os.environ", {"TRAINING_WORKER_TOKEN": WORKER_TOKEN}),
            patch("app.training_queue_routes.db_connection", _connect),
            patch(
                "app.training_queue_routes.db_courses.course_exists",
                return_value=True,
            ),
        ):
            yield

    # -- the invariant, asserted the same way after every callback ---------- #

    def assert_replacement_still_owns_the_request(self) -> None:
        request = self.connection.requests[COURSE]
        self.assertEqual(request["current_run_id"], self.new_run_id)
        self.assertEqual(request["status"], "preparing")
        self.assertIsNone(request["training"])
        self.assertIsNone(request["failure_message"])

        new_run = db_training_runs.get_training_run(
            self.connection, COURSE, self.new_run_id
        )
        self.assertEqual(new_run["state"], "queued")
        self.assertNotIn("jobId", new_run)

        old_run = db_training_runs.get_training_run(
            self.connection, COURSE, STALE_RUN_ID
        )
        self.assertIn(old_run["state"], TERMINAL_RUN_STATES)
        self.assertEqual(old_run["error"], SUPERSEDED_ERROR)

        # And the invariant the queue itself depends on.
        self.assertEqual(len(self.connection.active_runs()), 1)

    def test_a_late_submitted_callback_is_refused(self) -> None:
        with self._worker():
            response = self.client.post(
                f"/api/training-queue/courses/{COURSE}/runs/{STALE_RUN_ID}/submitted",
                json={"jobId": "999999", "trainExamples": 37, "validationExamples": 5},
                headers=WORKER_HEADERS,
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("superseded", response.json()["detail"])
        self.assert_replacement_still_owns_the_request()

    def test_a_late_submission_failure_does_not_resurrect_the_retired_run(
        self,
    ) -> None:
        """`release_training_run` would put it back to `queued`.

        That is not merely a stolen pointer: it is a second active run for a
        course that is only ever allowed one.
        """
        with self._worker():
            response = self.client.post(
                f"/api/training-queue/courses/{COURSE}/runs/{STALE_RUN_ID}"
                "/submission-failed",
                json={"error": "launcher exited nonzero"},
                headers=WORKER_HEADERS,
            )

        self.assertEqual(response.status_code, 409)
        self.assert_replacement_still_owns_the_request()

    def test_a_late_training_failure_does_not_fail_the_professors_request(
        self,
    ) -> None:
        """The replacement is queued and untried. Nothing has failed yet."""
        with self._worker():
            response = self.client.post(
                f"/api/training-queue/courses/{COURSE}/runs/{STALE_RUN_ID}/failed",
                json={"error": "the job died"},
                headers=WORKER_HEADERS,
            )

        self.assertEqual(response.status_code, 409)
        self.assert_replacement_still_owns_the_request()

    def test_a_late_model_registration_is_refused_before_anything_is_written(
        self,
    ) -> None:
        """The one that would have shipped the wrong model.

        This is the CSS 350 shape exactly: the adapter from the run trained
        before the QLoRA optimizer-step fix, arriving after an admin retried
        specifically to get a corrected one. Registering it would record that
        adapter; `setCurrent` would serve it to students.
        """
        upserts: list[Any] = []

        with self._worker(), patch(
            "app.training_queue_routes.db_models.upsert_model_version",
            side_effect=lambda *a, **k: upserts.append(a),
        ):
            response = self.client.post(
                f"/api/training-queue/courses/{COURSE}/model-versions",
                json={
                    "baseModel": "meta-llama/Llama-3.2-3B-Instruct",
                    "trainingExampleCount": 37,
                    "artifactRef": "css-350-qlora/adapter",
                    "status": "ready",
                    "deployment": "offline",
                    "setCurrent": True,
                    "runId": STALE_RUN_ID,
                },
                headers=WORKER_HEADERS,
            )

        self.assertEqual(response.status_code, 409)
        # Refused before the registry write, not merely before the promotion.
        self.assertEqual(upserts, [])
        self.assert_replacement_still_owns_the_request()

    def test_the_replacement_runs_own_callbacks_still_work(self) -> None:
        """The guard must not break the ordinary path it sits on.

        A retry is only useful if the run it creates can go on to be claimed,
        submitted and reported exactly like any other.
        """
        with self._worker():
            response = self.client.post(
                f"/api/training-queue/courses/{COURSE}/runs/{self.new_run_id}"
                "/submitted",
                json={"jobId": "262148", "trainExamples": 37, "validationExamples": 5},
                headers=WORKER_HEADERS,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["run"]["jobId"], "262148")
        self.assertEqual(response.json()["requestStatus"], "training")

        request = self.connection.requests[COURSE]
        self.assertEqual(request["current_run_id"], self.new_run_id)
        self.assertEqual(request["training"]["jobId"], "262148")

        # And the retired run is untouched by any of it.
        old_run = db_training_runs.get_training_run(
            self.connection, COURSE, STALE_RUN_ID
        )
        self.assertEqual(old_run["jobId"], STALE_JOB_ID)
        self.assertEqual(old_run["error"], SUPERSEDED_ERROR)


class OwnershipGuardTests(unittest.TestCase):
    """The guard itself, at the repository boundary."""

    def test_a_write_from_the_current_run_lands(self) -> None:
        connection = _stale_store()

        updated = db_model_requests.update_model_request_for_run(
            connection, COURSE, STALE_RUN_ID, {"status": "ready"}
        )

        self.assertIsNotNone(updated)
        self.assertEqual(connection.requests[COURSE]["status"], "ready")

    def test_a_write_from_a_superseded_run_matches_no_rows(self) -> None:
        connection = _stale_store()
        connection.requests[COURSE]["current_run_id"] = "run-something-else"

        updated = db_model_requests.update_model_request_for_run(
            connection, COURSE, STALE_RUN_ID, {"status": "ready"}
        )

        self.assertIsNone(updated)
        self.assertEqual(connection.requests[COURSE]["status"], "training")

    def test_an_unclaimed_request_can_still_be_claimed(self) -> None:
        """`current_run_id IS NULL` is ownable.

        A run queued through the operational route before anything pointed at
        it is in exactly this state, and its first callback is the write that
        takes ownership. Guarding that away would break the ordinary path.
        """
        connection = _stale_store()
        connection.requests[COURSE]["current_run_id"] = None

        updated = db_model_requests.update_model_request_for_run(
            connection, COURSE, STALE_RUN_ID, {"currentRunId": STALE_RUN_ID}
        )

        self.assertIsNotNone(updated)
        self.assertEqual(connection.requests[COURSE]["current_run_id"], STALE_RUN_ID)


class LiveJobRetryPolicyTests(unittest.TestCase):
    """Retry of `submitted`/`training` now needs evidence the job is gone.

    The backend cannot see Slurm, so it cannot distinguish a lost callback from
    a running job. Silence is the only evidence it has.
    """

    def _run(self, **overrides: Any) -> dict[str, Any]:
        return {
            "runId": STALE_RUN_ID,
            "state": "submitted",
            "mode": "full",
            "datasetRef": DATASET_REF,
            "jobId": STALE_JOB_ID,
            "updatedAt": "2026-08-23T06:43:33+00:00",
            **overrides,
        }

    def test_a_long_silent_submitted_run_may_be_retried(self) -> None:
        """The real CSS 350 case: three days without a word."""
        self.assertIsNone(training_run_retry_block(self._run(), now=NOW))

    def test_a_freshly_submitted_run_is_refused(self) -> None:
        block = training_run_retry_block(
            self._run(updatedAt=(NOW - timedelta(minutes=10)).isoformat()), now=NOW
        )
        self.assertIsNotNone(block)
        self.assertIn("may still be running", block or "")
        self.assertIn(STALE_JOB_ID, block or "")

    def test_a_run_training_for_an_hour_is_refused(self) -> None:
        block = training_run_retry_block(
            self._run(
                state="training", updatedAt=(NOW - timedelta(hours=1)).isoformat()
            ),
            now=NOW,
        )
        self.assertIsNotNone(block)

    def test_the_threshold_is_the_boundary(self) -> None:
        just_under = training_run_retry_block(
            self._run(
                updatedAt=(
                    NOW - timedelta(seconds=SUBMITTED_STALE_AFTER_SECONDS - 60)
                ).isoformat()
            ),
            now=NOW,
        )
        just_over = training_run_retry_block(
            self._run(
                updatedAt=(
                    NOW - timedelta(seconds=SUBMITTED_STALE_AFTER_SECONDS + 60)
                ).isoformat()
            ),
            now=NOW,
        )
        self.assertIsNotNone(just_under)
        self.assertIsNone(just_over)

    def test_a_failed_run_is_retryable_regardless_of_age(self) -> None:
        """Terminal. Nothing is running, so there is nothing to wait for."""
        self.assertIsNone(
            training_run_retry_block(
                self._run(state="failed", updatedAt=NOW.isoformat()), now=NOW
            )
        )

    def test_an_expired_claim_is_retryable_regardless_of_age(self) -> None:
        """The lease is the proof; it does not need a second one."""
        self.assertIsNone(
            training_run_retry_block(
                self._run(
                    state="claimed",
                    jobId=None,
                    updatedAt=NOW.isoformat(),
                    claim={
                        "owner": "adam@tillicum",
                        "claimedAt": (NOW - timedelta(hours=2)).isoformat(),
                        "expiresAt": (NOW - timedelta(minutes=1)).isoformat(),
                    },
                ),
                now=NOW,
            )
        )

    def test_the_route_refuses_a_live_looking_run(self) -> None:
        # Relative to the real clock: the route takes its own `now`, which is
        # the honest thing for it to do.
        recent = datetime.now(timezone.utc) - timedelta(minutes=5)
        connection = _stale_store(updated_at=recent.isoformat())

        @contextmanager
        def _connect(**kwargs: Any) -> Iterator[FakeConnection]:
            yield connection

        with (
            patch("app.main.db_connection", _connect),
            patch("app.main.course_exists", return_value=True),
        ):
            response = TestClient(app).post(
                f"/api/courses/{COURSE}/training-runs/retry"
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("may still be running", response.json()["detail"])
        self.assertEqual(len(connection.course_runs()), 1)


if __name__ == "__main__":
    unittest.main()
