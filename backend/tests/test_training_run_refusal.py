"""A run the worker refuses for good ends `failed`, not back on the queue.

The loop this closes (audit B10, follow-up of 2026-11-04)
-------------------------------------------------------
A run is pinned to the prepared dataset's digest when it is queued. If the data
is re-prepared under it, the backend refuses to serve the worker that run's
dataset (`409 dataset_changed`). The worker used to hand such a run back with
`/release`, which put it to `queued` with the reason — and `queued` is
claimable, so the next cycle claimed it, was refused again and released it
again, for as long as nobody noticed. Nothing trained; the attempt count grew.

Now the worker reports it through `/refused`. The run becomes `failed` with the
refusal as its reason (no new state: an older browser bundle would drop one),
stops matching the claim predicate, and is what Retry already accepts. The
request is not failed — no job existed — but carries the reason on
`launchError` for the admin, the way a submission failure does.

The stateful fake connection from `test_training_retry` is reused so that the
claim predicate, the conditional INSERT and the ownership guard are the real
SQL semantics rather than stubs: "the worker cannot claim it again" and "the
admin can queue a replacement" are the facts being tested.
"""

from __future__ import annotations

import json
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_training_retry import (  # noqa: E402
    COURSE,
    NOW,
    STALE_RUN_ID,
    WORKER_HEADERS,
    WORKER_TOKEN,
    FakeConnection,
)

from app import db_training_runs  # noqa: E402
from app.db_training_runs import SUPERSEDED_ERROR  # noqa: E402
from app.main import app  # noqa: E402
from app.training_retry import retry_training_run  # noqa: E402

RUN_PATH = f"/api/training-queue/courses/{COURSE}/runs/{STALE_RUN_ID}"
REASON = (
    "Could not fetch the prepared dataset from the backend: The backend request "
    f"for /courses/{COURSE}/runs/{STALE_RUN_ID}/dataset failed with HTTP 409: The "
    "prepared training data changed after this run was queued (queued against "
    "dataset 000000000000…, the files are now 9f2c1a7b3e4d…). Nothing was "
    "served. Queue a new run for the re-prepared data, or restore the data this "
    "run was queued for. [dataset_changed]"
)


def _claimed_pinned_store(**run_overrides: Any) -> FakeConnection:
    """A queued run a worker has just claimed, pinned to data that is gone."""
    connection = FakeConnection()
    connection.add_request(status="preparing", training=None)
    connection.add_run(
        state="claimed",
        job_id=None,
        attempt=1,
        claim_owner="alice@tillicum",
        claim_claimed_at="2026-08-26T16:59:00+00:00",
        claim_expires_at="2026-08-26T17:14:00+00:00",
        dataset_sha256="0" * 64,
        **run_overrides,
    )
    return connection


class RefusalRouteTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.connection = _claimed_pinned_store()

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

    def _refuse(self, run_id: str = STALE_RUN_ID, reason: str = REASON):
        with self._worker():
            return self.client.post(
                f"/api/training-queue/courses/{COURSE}/runs/{run_id}/refused",
                json={"error": reason},
                headers=WORKER_HEADERS,
            )

    def _run(self, run_id: str = STALE_RUN_ID) -> dict[str, Any]:
        return db_training_runs.get_training_run(self.connection, COURSE, run_id)

    def _request(self) -> dict[str, Any]:
        return self.connection.requests[COURSE]


class RefusalTests(RefusalRouteTestCase):
    def test_the_run_ends_failed_with_the_refusal_as_its_reason(self) -> None:
        response = self._refuse()

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["state"], "failed")
        self.assertEqual(body["error"], REASON)
        self.assertIsNone(body["claim"])

        stored = self._run()
        self.assertEqual(stored["state"], "failed")
        self.assertEqual(stored["error"], REASON)
        self.assertNotIn("claim", stored)
        # Nothing else about the run is rewritten: its history stays legible.
        self.assertEqual(stored["attempt"], 1)
        self.assertEqual(stored["datasetSha256"], "0" * 64)
        self.assertNotIn("jobId", stored)

    def test_the_request_is_not_failed_and_carries_the_reason_for_the_admin(self) -> None:
        """No job existed, so the professor is not told one failed."""
        self._refuse()

        request = self._request()
        self.assertEqual(request["status"], "preparing")
        self.assertIsNone(request["failure_message"])
        self.assertEqual(request["launch_error"], REASON)
        self.assertEqual(request["current_run_id"], STALE_RUN_ID)

    def test_a_refused_run_is_not_claimable_on_any_later_cycle(self) -> None:
        """The loop itself: what `/pending` and `/claim` see afterwards."""
        self._refuse()

        for _ in range(3):
            self.assertEqual(
                db_training_runs.claimable_training_runs(self.connection, now=NOW), []
            )
            self.assertIsNone(
                db_training_runs.claim_next_training_run(
                    self.connection, owner="bob@tillicum", now=NOW
                )
            )

        stored = self._run()
        self.assertEqual(stored["state"], "failed")
        self.assertEqual(stored["attempt"], 1)
        self.assertEqual(stored["error"], REASON)

    def test_the_course_has_no_active_run_afterwards(self) -> None:
        """`failed` is terminal, so the course is free to queue a replacement."""
        self._refuse()
        self.assertEqual(self.connection.active_runs(), [])

    def test_a_second_refusal_of_the_same_run_is_refused_without_rewriting_it(self) -> None:
        self._refuse()
        response = self._refuse(reason="a different reason")

        self.assertEqual(response.status_code, 409, response.text)
        self.assertIn("already failed", response.json()["detail"])
        self.assertEqual(self._run()["error"], REASON)

    def test_a_run_with_a_slurm_job_cannot_be_refused(self) -> None:
        """Its job decides how it ends; a refusal would fail a live job's run."""
        self.connection.runs[(COURSE, STALE_RUN_ID)].update(
            {"state": "submitted", "job_id": "264787", "claim_owner": None,
             "claim_claimed_at": None, "claim_expires_at": None}
        )

        response = self._refuse()

        self.assertEqual(response.status_code, 409, response.text)
        self.assertIn("264787", response.json()["detail"])
        self.assertIn("cannot be refused", response.json()["detail"])
        stored = self._run()
        self.assertEqual(stored["state"], "submitted")
        self.assertNotIn("error", stored)
        self.assertIsNone(self._request()["launch_error"])

    def test_a_finished_run_is_not_reopened(self) -> None:
        self.connection.runs[(COURSE, STALE_RUN_ID)]["state"] = "succeeded"

        response = self._refuse()

        self.assertEqual(response.status_code, 409, response.text)
        self.assertIn("already succeeded", response.json()["detail"])
        self.assertEqual(self._run()["state"], "succeeded")

    def test_a_refusal_from_a_superseded_run_is_refused(self) -> None:
        """An admin retried while the worker still held the old run."""
        self.connection.requests[COURSE]["current_run_id"] = "run-20260826t170000z-replacement"

        response = self._refuse()

        self.assertEqual(response.status_code, 409, response.text)
        self.assertIn("superseded", response.json()["detail"])
        self.assertEqual(self._run()["state"], "claimed")
        self.assertIsNone(self._request()["launch_error"])

    def test_an_unknown_run_is_404(self) -> None:
        response = self._refuse(run_id="run-20260826t170000z-nothere")
        self.assertEqual(response.status_code, 404, response.text)

    def test_the_reason_is_required(self) -> None:
        with self._worker():
            response = self.client.post(
                f"{RUN_PATH}/refused", json={}, headers=WORKER_HEADERS
            )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self._run()["state"], "claimed")

    def test_the_route_needs_the_worker_token(self) -> None:
        with self._worker():
            response = self.client.post(f"{RUN_PATH}/refused", json={"error": REASON})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(self._run()["state"], "claimed")


class RecoveryAfterRefusalTests(RefusalRouteTestCase):
    """What an admin does next, against the files as they are now."""

    def setUp(self) -> None:
        super().setUp()
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        directory = root / COURSE
        directory.mkdir(parents=True)
        (directory / "train.jsonl").write_text(
            "".join(json.dumps({"instruction": f"Q{i}", "response": "A"}) + "\n" for i in range(3)),
            encoding="utf-8",
        )
        (directory / "validation.jsonl").write_text(
            json.dumps({"instruction": "Qv", "response": "A"}) + "\n", encoding="utf-8"
        )
        (directory / "manifest.json").write_text(
            json.dumps({"datasetVersion": "x", "splitSeed": 350, "trainExamples": 3, "validationExamples": 1}),
            encoding="utf-8",
        )
        export_dir = patch(
            "app.dataset_artifacts.course_export_dir",
            side_effect=lambda course_id, root=None: root_dir / course_id,
        )
        root_dir = root
        export_dir.start()
        self.addCleanup(export_dir.stop)

    def test_retry_queues_a_replacement_pinned_to_the_current_files(self) -> None:
        from app.dataset_artifacts import describe_dataset

        self._refuse()
        result = retry_training_run(self.connection, COURSE, now=NOW)

        replacement = result["run"]
        self.assertEqual(replacement["state"], "queued")
        self.assertNotEqual(replacement["runId"], STALE_RUN_ID)
        self.assertEqual(replacement["datasetSha256"], describe_dataset(COURSE)["datasetSha256"])
        self.assertEqual(replacement["trainExamples"], 3)
        self.assertEqual(replacement["validationExamples"], 1)

        # The retired run keeps its place in history, retired the way every
        # retried run is; the request moves on with the reason cleared.
        superseded = result["superseded"]
        self.assertEqual(superseded["runId"], STALE_RUN_ID)
        self.assertEqual(superseded["state"], "failed")
        self.assertEqual(superseded["error"], SUPERSEDED_ERROR)
        request = self._request()
        self.assertEqual(request["status"], "preparing")
        self.assertEqual(request["current_run_id"], replacement["runId"])
        self.assertIsNone(request["launch_error"])

        # And the worker's next cycle sees exactly the replacement.
        claimable = db_training_runs.claimable_training_runs(self.connection, now=NOW)
        self.assertEqual([run["runId"] for run in claimable], [replacement["runId"]])
        # (The fake store answers the claim's SELECT faithfully; its literal
        # SET is not emulated, so only which run was chosen is asserted.)
        claimed = db_training_runs.claim_next_training_run(
            self.connection, owner="alice@tillicum", now=NOW
        )
        self.assertEqual(claimed["runId"], replacement["runId"])

    def test_queue_training_also_works_and_keeps_the_refusal_in_history(self) -> None:
        """The other button the admin page offers for a `preparing` request
        with no active run. The refused run is left exactly as it ended."""
        from app.dataset_artifacts import describe_dataset

        self._refuse()
        created = db_training_runs.enqueue_training_run(
            self.connection,
            COURSE,
            mode="full",
            dataset_ref=f"exports/{COURSE}",
            train_examples=3,
            validation_examples=1,
            dataset_sha256=describe_dataset(COURSE)["datasetSha256"],
        )

        self.assertEqual(created["state"], "queued")
        refused = self._run()
        self.assertEqual(refused["state"], "failed")
        self.assertEqual(refused["error"], REASON)
        self.assertEqual(len(self.connection.active_runs()), 1)

    def test_the_replacement_is_served_its_dataset(self) -> None:
        """The whole point: the new run trains; the old one is never asked about again."""
        from app.dataset_artifacts import resolve_run_dataset

        self._refuse()
        replacement = retry_training_run(self.connection, COURSE, now=NOW)["run"]

        described = resolve_run_dataset(replacement)
        self.assertEqual(described["pinnedDatasetSha256"], replacement["datasetSha256"])
        self.assertEqual(described["trainExamples"], 3)


if __name__ == "__main__":
    unittest.main()
