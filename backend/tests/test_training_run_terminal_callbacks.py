"""A finished training run stays finished, whatever a worker says about it later.

The sequence this closes (audit finding B2, 2026-10-27)
-------------------------------------------------------
`sbatch` succeeds and the worker's `/submitted` call is lost, so its local run
record keeps `reported: false`. The job finishes and reports its completion
directly from the compute node; the backend accepts it, records the job id it
never heard of, marks the run `succeeded` and registers the model. On the next
`run_training_queue.sh --once`, the worker's repair step re-posts `/submitted`
for every record with a job id and `reported: false` — and the backend, which
only checked that the run existed and was the request's current run, set
`state = submitted` and the request back to `training`. The completion had
already been delivered and its pending file cleared, so nothing would ever
finish the run again: a registered, servable model showed as "training" until
an admin retried a run that had succeeded.

`succeeded` and `failed` are terminal. Every worker callback that could move a
run out of one of them — a submission report, a submission failure, a release,
a training failure, a completion with a different outcome — is now refused with
409 and writes nothing. A redelivered completion with the same outcome is the
idempotent case the reporting path is built around and is still accepted.

The worker half of the fix (marking the local record reported once a completion
is accepted) is tested in `training/test_report_training_result.py` and
`training/test_training_state.py`; the backend half here is what makes a worker
that has not been updated harmless.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

COURSE = "css-350-spring-2026-n3h9"
RUN_ID = "run-20260827t064701z-1cf650"
JOB_ID = "264787"
TOKEN = "test-worker-token"
HEADERS = {"X-Training-Worker-Token": TOKEN}
QUEUE = f"/api/training-queue/courses/{COURSE}/runs/{RUN_ID}"


def _run(*, state: str, **overrides: Any) -> dict[str, Any]:
    record = {
        "runId": RUN_ID,
        "courseId": COURSE,
        "mode": "full",
        "state": state,
        "enqueuedAt": "2026-08-27T06:40:00+00:00",
        "updatedAt": "2026-08-27T06:47:01+00:00",
        "datasetRef": f"exports/{COURSE}",
        "approvedExampleCount": 42,
        "trainExamples": 37,
        "validationExamples": 5,
        "attempt": 1,
        "jobId": JOB_ID,
    }
    record.update(overrides)
    return record


@contextmanager
def _fake_connection(**kwargs: Any) -> Iterator[object]:
    yield object()


class TerminalRunTestCase(unittest.TestCase):
    """Every repository writer is patched to record; a refused callback records nothing."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        self._env = patch.dict("os.environ", {"TRAINING_WORKER_TOKEN": TOKEN})
        self._env.start()
        self.addCleanup(self._env.stop)

        self.run_writes: list[str] = []
        self.request_writes: list[dict[str, Any]] = []

        def _record_run_write(name: str):
            def writer(connection: Any, cid: str, rid: str, *args: Any, **kwargs: Any):
                self.run_writes.append(name)
                return _run(state="submitted")

            return writer

        def _record_request_write(connection: Any, cid: str, rid: str, patch_body: dict[str, Any]):
            self.request_writes.append(dict(patch_body))
            return {"courseId": cid, "status": patch_body.get("status", "ready")}

        prefix = "app.training_queue_routes."
        patches = [
            patch(prefix + "db_connection", _fake_connection),
            patch(prefix + "db_courses.course_exists", return_value=True),
            patch(
                prefix + "db_model_requests.lock_model_request",
                return_value={"courseId": COURSE, "currentRunId": RUN_ID, "status": "ready"},
            ),
            patch(prefix + "db_model_requests.update_model_request_for_run", side_effect=_record_request_write),
            patch(prefix + "db_training_runs.mark_training_run_submitted", side_effect=_record_run_write("submitted")),
            patch(prefix + "db_training_runs.release_training_run", side_effect=_record_run_write("released")),
            patch(prefix + "db_training_runs.fail_training_run", side_effect=_record_run_write("failed")),
            patch(prefix + "db_training_runs.update_training_run", side_effect=_record_run_write("updated")),
            patch(prefix + "db_models.upsert_model_version", side_effect=lambda *a, **k: self.run_writes.append("registered")),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def _with_run(self, run: dict[str, Any]):
        return patch("app.training_queue_routes.db_training_runs.get_training_run", return_value=run)

    def _assert_refused_without_writing(self, response, state: str) -> None:
        self.assertEqual(response.status_code, 409, response.text)
        detail = response.json()["detail"]
        self.assertIn(f"already {state}", detail)
        self.assertIn("Nothing was recorded", detail)
        self.assertEqual(self.run_writes, [])
        self.assertEqual(self.request_writes, [])


class LostSubmissionAfterCompletionTests(TerminalRunTestCase):
    def test_a_late_submission_report_cannot_reopen_a_succeeded_run(self) -> None:
        """The audit's B2 sequence, at the moment the worker replays `/submitted`."""
        with self._with_run(_run(state="succeeded")):
            response = self.client.post(
                f"{QUEUE}/submitted",
                json={"jobId": JOB_ID, "trainExamples": 37, "validationExamples": 5},
                headers=HEADERS,
            )

        self._assert_refused_without_writing(response, "succeeded")

    def test_a_late_submission_report_cannot_reopen_a_failed_run(self) -> None:
        with self._with_run(_run(state="failed")):
            response = self.client.post(
                f"{QUEUE}/submitted", json={"jobId": JOB_ID}, headers=HEADERS
            )

        self._assert_refused_without_writing(response, "failed")

    def test_a_submission_for_a_run_that_is_still_live_is_recorded(self) -> None:
        """The guard is about terminal runs only; the ordinary path is unchanged."""
        with self._with_run(_run(state="claimed", jobId=None)):
            response = self.client.post(
                f"{QUEUE}/submitted",
                json={"jobId": JOB_ID, "trainExamples": 37, "validationExamples": 5},
                headers=HEADERS,
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.run_writes, ["submitted"])
        self.assertEqual(self.request_writes[0]["status"], "training")


class OtherLateCallbacksTests(TerminalRunTestCase):
    def test_a_submission_failure_cannot_requeue_a_finished_run(self) -> None:
        with self._with_run(_run(state="succeeded")):
            response = self.client.post(
                f"{QUEUE}/submission-failed", json={"error": "launcher exited nonzero"}, headers=HEADERS
            )

        self._assert_refused_without_writing(response, "succeeded")

    def test_a_training_failure_cannot_overwrite_a_success(self) -> None:
        with self._with_run(_run(state="succeeded")):
            response = self.client.post(
                f"{QUEUE}/failed", json={"error": "the job died"}, headers=HEADERS
            )

        self._assert_refused_without_writing(response, "succeeded")

    def test_a_refusal_cannot_fail_a_finished_run_again(self) -> None:
        """A refusal is terminal itself; it must not rewrite how a run already ended."""
        for state in ("succeeded", "failed"):
            self.run_writes.clear()
            self.request_writes.clear()
            with self._with_run(_run(state=state, jobId=None)):
                response = self.client.post(
                    f"{QUEUE}/refused",
                    json={"error": "The prepared training data changed [dataset_changed]"},
                    headers=HEADERS,
                )
            self._assert_refused_without_writing(response, state)

    def test_a_refusal_of_a_live_claim_fails_the_run_and_leaves_the_request_status_alone(self) -> None:
        with self._with_run(_run(state="claimed", jobId=None)):
            response = self.client.post(
                f"{QUEUE}/refused",
                json={"error": "The prepared training data changed [dataset_changed]"},
                headers=HEADERS,
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.run_writes, ["failed"])
        self.assertEqual(len(self.request_writes), 1)
        self.assertNotIn("status", self.request_writes[0])
        self.assertNotIn("failureMessage", self.request_writes[0])
        self.assertIn("dataset_changed", self.request_writes[0]["launchError"])

    def test_a_release_cannot_requeue_a_retired_run(self) -> None:
        """A worker releasing a run an admin has since retired must not resurrect it."""
        with self._with_run(_run(state="failed", error="Superseded by admin retry")):
            response = self.client.post(f"{QUEUE}/release", json={}, headers=HEADERS)

        self._assert_refused_without_writing(response, "failed")

    def test_a_release_of_a_live_claim_still_works(self) -> None:
        with self._with_run(_run(state="claimed", jobId=None)):
            response = self.client.post(f"{QUEUE}/release", json={}, headers=HEADERS)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.run_writes, ["released"])


class ConflictingCompletionTests(TerminalRunTestCase):
    def test_a_failed_completion_cannot_overwrite_a_success(self) -> None:
        with self._with_run(_run(state="succeeded")):
            response = self.client.post(
                f"{QUEUE}/completed",
                json={"outcome": "failed", "jobId": JOB_ID, "error": "late failure"},
                headers=HEADERS,
            )

        self._assert_refused_without_writing(response, "succeeded")
        self.assertIn('"failed" completion', response.json()["detail"])

    def test_a_succeeded_completion_cannot_overwrite_a_failure(self) -> None:
        with self._with_run(_run(state="failed")):
            response = self.client.post(
                f"{QUEUE}/completed",
                json={
                    "outcome": "succeeded",
                    "jobId": JOB_ID,
                    "baseModel": "meta-llama/Llama-3.2-3B-Instruct",
                    "artifactRef": f"qlora-runs/{COURSE}/x-full/adapter",
                    "trainingLengthSatisfied": True,
                },
                headers=HEADERS,
            )

        self._assert_refused_without_writing(response, "failed")

    def test_a_redelivered_failure_is_still_idempotent(self) -> None:
        """The same outcome again is the redelivery the reporting path is built for."""
        with self._with_run(_run(state="failed")):
            response = self.client.post(
                f"{QUEUE}/completed",
                json={"outcome": "failed", "jobId": JOB_ID, "error": "the job died"},
                headers=HEADERS,
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["runState"], "failed")


class RefusalShapeTests(TerminalRunTestCase):
    def test_the_refusal_is_a_409_the_worker_treats_as_final(self) -> None:
        """`training/report_training_result.py` and the worker's reconcile step
        both stop replaying on an `HTTP 409` in the message."""
        with self._with_run(_run(state="succeeded")):
            response = self.client.post(
                f"{QUEUE}/submitted", json={"jobId": JOB_ID}, headers=HEADERS
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn(RUN_ID, response.json()["detail"])
        self.assertIn(COURSE, response.json()["detail"])

    def test_a_superseded_run_is_still_reported_as_superseded_first(self) -> None:
        """The more specific refusal keeps precedence: the request points elsewhere."""
        with (
            self._with_run(_run(state="failed", error="Superseded by admin retry")),
            patch(
                "app.training_queue_routes.db_model_requests.lock_model_request",
                return_value={"courseId": COURSE, "currentRunId": "run-20260828t090000z-aa11bb"},
            ),
        ):
            response = self.client.post(
                f"{QUEUE}/submitted", json={"jobId": JOB_ID}, headers=HEADERS
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("superseded", response.json()["detail"])
        self.assertEqual(self.run_writes, [])


if __name__ == "__main__":
    unittest.main()
