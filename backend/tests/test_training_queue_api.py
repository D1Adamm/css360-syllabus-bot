"""The worker-facing training queue API.

This router is what replaced the Firebase queue. It is the only thing on the
cluster's side of the database, and everything about it is shaped by that: the
worker never holds a psycopg connection, never sees a DSN, and can reach
nothing but these endpoints.

The claim's atomicity lives in SQL and is asserted there
(`test_db_repositories.py`, `QueueClaimSqlTests`). What these cover is the
contract the worker actually programs against — including the case the old
queue had to negotiate with retries and compare-and-set, and this one resolves
in a single statement: two workers asking at the same moment, exactly one of
them getting the run.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.training_queue_routes import next_model_version

COURSE = "css-360-winter-2026-a7rp"
OTHER_COURSE = "css-350-spring-2026-n3h9"
RUN_ID = "run-20260813t120000z-a1b2c3"
TOKEN = "test-worker-token"
HEADERS = {"X-Training-Worker-Token": TOKEN}


def _run(
    *,
    course_id: str = COURSE,
    run_id: str = RUN_ID,
    state: str = "queued",
    **overrides: Any,
) -> dict[str, Any]:
    record = {
        "runId": run_id,
        "courseId": course_id,
        "mode": "full",
        "state": state,
        "enqueuedAt": "2026-08-13T12:00:00+00:00",
        "updatedAt": "2026-08-13T12:00:00+00:00",
        "datasetRef": f"exports/{course_id}",
        "approvedExampleCount": 54,
        "trainExamples": 48,
        "validationExamples": 6,
        "attempt": 0,
    }
    record.update(overrides)
    return record


@contextmanager
def _fake_connection(**kwargs: Any) -> Iterator[object]:
    yield object()


class QueueApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self._env = patch.dict("os.environ", {"TRAINING_WORKER_TOKEN": TOKEN})
        self._env.start()
        self.addCleanup(self._env.stop)

        self._connection = patch(
            "app.training_queue_routes.db_connection", _fake_connection
        )
        self._connection.start()
        self.addCleanup(self._connection.stop)


class AuthenticationTests(unittest.TestCase):
    """The queue is not a public endpoint, and an unconfigured one is closed."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_no_token_configured_refuses_rather_than_serving_openly(self) -> None:
        # conftest removes TRAINING_WORKER_TOKEN for every test by default.
        response = self.client.post(
            "/api/training-queue/claim", json={"owner": "runner@node"}
        )
        self.assertEqual(response.status_code, 503)

    def test_a_missing_header_is_401(self) -> None:
        with patch.dict("os.environ", {"TRAINING_WORKER_TOKEN": TOKEN}):
            response = self.client.post(
                "/api/training-queue/claim", json={"owner": "runner@node"}
            )
        self.assertEqual(response.status_code, 401)

    def test_a_wrong_token_is_401(self) -> None:
        with patch.dict("os.environ", {"TRAINING_WORKER_TOKEN": TOKEN}):
            response = self.client.post(
                "/api/training-queue/claim",
                json={"owner": "runner@node"},
                headers={"X-Training-Worker-Token": "not-the-token"},
            )
        self.assertEqual(response.status_code, 401)

    def test_every_queue_route_requires_the_token(self) -> None:
        """One unguarded route would be enough to lose the whole boundary."""
        spec = self.client.get("/openapi.json").json()["paths"]
        queue_paths = [path for path in spec if path.startswith("/api/training-queue")]
        self.assertGreaterEqual(len(queue_paths), 5)

        with patch.dict("os.environ", {"TRAINING_WORKER_TOKEN": TOKEN}):
            for path in queue_paths:
                for method in spec[path]:
                    with self.subTest(path=path, method=method):
                        url = (
                            path.replace("{course_id}", COURSE)
                            .replace("{run_id}", RUN_ID)
                        )
                        response = self.client.request(method.upper(), url, json={})
                        self.assertNotIn(response.status_code, (200, 201))


class ClaimTests(QueueApiTestCase):
    def test_a_queued_job_can_be_claimed(self) -> None:
        with patch(
            "app.training_queue_routes.db_training_runs.claim_next_training_run",
            return_value=_run(state="claimed", attempt=1),
        ):
            response = self.client.post(
                "/api/training-queue/claim",
                json={"owner": "runner@node", "leaseSeconds": 900},
                headers=HEADERS,
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["claimed"])
        self.assertEqual(payload["run"]["runId"], RUN_ID)
        self.assertEqual(payload["run"]["state"], "claimed")
        self.assertEqual(payload["run"]["attempt"], 1)

    def test_exactly_one_of_two_simultaneous_workers_gets_the_job(self) -> None:
        """The property the old ETag compare-and-set existed to provide.

        The repository is stubbed with a one-shot queue: whichever call arrives
        first takes the run and every later call finds nothing. That is what
        `FOR UPDATE SKIP LOCKED` does under contention, modelled at the seam
        the router depends on.
        """
        remaining = [_run(state="claimed", attempt=1)]
        owners: list[str] = []

        def _claim(connection: Any, *, owner: str, **kwargs: Any):
            owners.append(owner)
            return remaining.pop() if remaining else None

        with patch(
            "app.training_queue_routes.db_training_runs.claim_next_training_run",
            side_effect=_claim,
        ):
            first = self.client.post(
                "/api/training-queue/claim",
                json={"owner": "runner-a@node"},
                headers=HEADERS,
            )
            second = self.client.post(
                "/api/training-queue/claim",
                json={"owner": "runner-b@node"},
                headers=HEADERS,
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertTrue(first.json()["claimed"])

        # The second worker is told there is nothing for it — not handed the
        # same run, and not given an error it would retry into a double claim.
        self.assertFalse(second.json()["claimed"])
        self.assertIsNone(second.json()["run"])
        self.assertEqual(owners, ["runner-a@node", "runner-b@node"])

    def test_an_empty_queue_is_not_an_error(self) -> None:
        with patch(
            "app.training_queue_routes.db_training_runs.claim_next_training_run",
            return_value=None,
        ):
            response = self.client.post(
                "/api/training-queue/claim",
                json={"owner": "runner@node"},
                headers=HEADERS,
            )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["claimed"])

    def test_a_claim_can_be_scoped_to_one_course(self) -> None:
        seen: dict[str, Any] = {}

        def _claim(connection: Any, **kwargs: Any):
            seen.update(kwargs)
            return _run(state="claimed")

        with patch(
            "app.training_queue_routes.db_training_runs.claim_next_training_run",
            side_effect=_claim,
        ):
            self.client.post(
                "/api/training-queue/claim",
                json={"owner": "runner@node", "courseIds": [COURSE]},
                headers=HEADERS,
            )

        self.assertEqual(seen["course_ids"], [COURSE])

    def test_an_invalid_course_id_never_reaches_the_repository(self) -> None:
        with patch(
            "app.training_queue_routes.db_training_runs.claim_next_training_run"
        ) as claim:
            response = self.client.post(
                "/api/training-queue/claim",
                json={"owner": "runner@node", "courseIds": ["../etc"]},
                headers=HEADERS,
            )

        self.assertEqual(response.status_code, 400)
        claim.assert_not_called()

    def test_pending_is_read_only(self) -> None:
        with (
            patch(
                "app.training_queue_routes.db_training_runs.claimable_training_runs",
                return_value=[_run()],
            ),
            patch(
                "app.training_queue_routes.db_training_runs.claim_next_training_run"
            ) as claim,
        ):
            response = self.client.get(
                "/api/training-queue/pending", headers=HEADERS
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 1)
        claim.assert_not_called()


class LifecycleTests(QueueApiTestCase):
    def test_successful_submission_updates_run_and_request_together(self) -> None:
        """One transaction. The two records cannot disagree about the job."""
        patches: list[dict[str, Any]] = []

        def _update_request(
            connection: Any,
            course_id: str,
            run_id: str,
            patch_body: dict[str, Any],
        ):
            patches.append({"courseId": course_id, **patch_body})
            return {"courseId": course_id, "status": patch_body["status"]}

        with (
            patch(
                "app.training_queue_routes.db_training_runs.get_training_run",
                return_value=_run(state="claimed"),
            ),
            patch(
                "app.training_queue_routes.db_training_runs.mark_training_run_submitted",
                return_value=_run(state="submitted", jobId="123456"),
            ),
            patch(
                "app.training_queue_routes.db_model_requests.lock_model_request",
                return_value={"courseId": COURSE, "currentRunId": RUN_ID},
            ),
            patch(
                "app.training_queue_routes.db_model_requests.update_model_request_for_run",
                side_effect=_update_request,
            ),
        ):
            response = self.client.post(
                f"/api/training-queue/courses/{COURSE}/runs/{RUN_ID}/submitted",
                json={"jobId": "123456", "trainExamples": 48, "validationExamples": 6},
                headers=HEADERS,
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["run"]["state"], "submitted")
        self.assertEqual(payload["run"]["jobId"], "123456")
        self.assertEqual(payload["requestStatus"], "training")

        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0]["courseId"], COURSE)
        self.assertEqual(patches[0]["training"]["jobId"], "123456")
        # Professor-facing failure text belongs to failures only.
        self.assertNotIn("failureMessage", patches[0])

    def test_a_submission_that_never_happened_leaves_the_request_preparing(self) -> None:
        patches: list[dict[str, Any]] = []

        with (
            patch(
                "app.training_queue_routes.db_training_runs.get_training_run",
                return_value=_run(state="claimed"),
            ),
            patch(
                "app.training_queue_routes.db_training_runs.release_training_run",
                return_value=_run(state="queued", error="launcher exited nonzero"),
            ),
            patch(
                "app.training_queue_routes.db_model_requests.lock_model_request",
                return_value={"courseId": COURSE, "currentRunId": RUN_ID},
            ),
            patch(
                "app.training_queue_routes.db_model_requests.update_model_request_for_run",
                side_effect=lambda c, cid, rid, body: patches.append(body),
            ),
        ):
            response = self.client.post(
                f"/api/training-queue/courses/{COURSE}/runs/{RUN_ID}/submission-failed",
                json={"error": "launcher exited nonzero"},
                headers=HEADERS,
            )

        self.assertEqual(response.status_code, 200)
        # Retryable, not terminal.
        self.assertEqual(response.json()["state"], "queued")
        self.assertEqual(patches[0]["launchError"], "launcher exited nonzero")
        self.assertNotIn("status", patches[0])
        self.assertNotIn("failureMessage", patches[0])

    def test_failed_training_is_terminal_for_the_run_and_the_request(self) -> None:
        patches: list[dict[str, Any]] = []

        with (
            patch(
                "app.training_queue_routes.db_training_runs.get_training_run",
                return_value=_run(state="training", jobId="123456"),
            ),
            patch(
                "app.training_queue_routes.db_training_runs.fail_training_run",
                return_value=_run(state="failed", error="the job died"),
            ),
            patch(
                "app.training_queue_routes.db_model_requests.lock_model_request",
                return_value={"courseId": COURSE, "currentRunId": RUN_ID},
            ),
            patch(
                "app.training_queue_routes.db_model_requests.update_model_request_for_run",
                side_effect=lambda c, cid, rid, body: patches.append(body),
            ),
        ):
            response = self.client.post(
                f"/api/training-queue/courses/{COURSE}/runs/{RUN_ID}/failed",
                json={"error": "the job died"},
                headers=HEADERS,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["state"], "failed")
        self.assertEqual(patches[0]["status"], "failed")
        self.assertIn("failureMessage", patches[0])

    def test_a_run_belonging_to_another_course_is_404(self) -> None:
        """Course scoping: the lookup binds both keys, so this cannot cross over."""

        def _get(connection: Any, course_id: str, run_id: str):
            return _run() if course_id == COURSE else None

        with (
            patch(
                "app.training_queue_routes.db_training_runs.get_training_run",
                side_effect=_get,
            ),
            patch(
                "app.training_queue_routes.db_training_runs.mark_training_run_submitted"
            ) as submitted,
        ):
            response = self.client.post(
                f"/api/training-queue/courses/{OTHER_COURSE}/runs/{RUN_ID}/submitted",
                json={"jobId": "123456"},
                headers=HEADERS,
            )

        self.assertEqual(response.status_code, 404)
        submitted.assert_not_called()


class ModelRegistrationTests(QueueApiTestCase):
    BODY = {
        "baseModel": "meta-llama/Llama-3.2-3B-Instruct",
        "trainingExampleCount": 54,
        "artifactRef": "css-360-qlora/adapter",
        "status": "ready",
        "deployment": "offline",
    }

    def test_next_version_counts_up_from_what_exists(self) -> None:
        self.assertEqual(next_model_version([]), "v1")
        self.assertEqual(next_model_version(["v1"]), "v2")
        self.assertEqual(next_model_version(["v1", "v3", "junk"]), "v4")

    def _register(self, body: dict[str, Any], *, course_id: str = COURSE):
        self.registered: list[tuple[str, dict[str, Any]]] = []
        self.request_patches: list[dict[str, Any]] = []

        def _upsert(connection, cid, version, *, set_current=False):
            self.registered.append((cid, dict(version)))
            return {
                "courseId": cid,
                "currentVersion": version["version"] if set_current else "v1",
                "versions": {version["version"]: version},
            }

        with (
            patch(
                "app.training_queue_routes.db_courses.course_exists", return_value=True
            ),
            patch(
                "app.training_queue_routes.db_models.list_model_versions",
                return_value=[],
            ),
            # No version has been registered for this run yet, so registration
            # allocates a new key rather than reusing one. The reuse path — a
            # redelivered completion callback — is covered in
            # test_training_completion_api.py.
            patch(
                "app.training_queue_routes.db_models.find_model_version_for_run",
                return_value=None,
            ),
            patch(
                "app.training_queue_routes.db_models.upsert_model_version",
                side_effect=_upsert,
            ),
            patch(
                "app.training_queue_routes.db_model_requests.lock_model_request",
                return_value={"courseId": COURSE, "currentRunId": RUN_ID},
            ),
            patch(
                "app.training_queue_routes.db_model_requests.update_model_request",
                side_effect=lambda c, cid, b: (
                    self.request_patches.append(b) or {"status": b.get("status")}
                ),
            ),
            # A registration that names a run goes through the ownership guard.
            patch(
                "app.training_queue_routes.db_model_requests.update_model_request_for_run",
                side_effect=lambda c, cid, rid, b: (
                    self.request_patches.append(b) or {"status": b.get("status")}
                ),
            ),
            patch(
                "app.training_queue_routes.db_training_runs.update_training_run",
                return_value=_run(state="succeeded"),
            ),
        ):
            return self.client.post(
                f"/api/training-queue/courses/{course_id}/model-versions",
                json=body,
                headers=HEADERS,
            )

    def test_registering_a_ready_model_makes_the_request_ready(self) -> None:
        response = self._register({**self.BODY, "runId": RUN_ID})

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["courseId"], COURSE)
        self.assertEqual(payload["version"], "v1")
        self.assertEqual(payload["currentVersion"], "v1")
        self.assertEqual(payload["requestStatus"], "ready")
        self.assertEqual(self.request_patches[0]["currentRunId"], RUN_ID)

    def test_a_version_lands_only_on_the_course_in_the_path(self) -> None:
        """A CSS 360 adapter must never attach to CSS 350."""
        self._register(self.BODY, course_id=OTHER_COURSE)
        self.assertEqual([course for course, _ in self.registered], [OTHER_COURSE])

    def test_a_non_ready_registration_does_not_flip_the_request(self) -> None:
        self._register({**self.BODY, "status": "training"})
        self.assertEqual(self.request_patches, [])

    def test_an_absolute_artifact_ref_is_refused(self) -> None:
        """It would store a cluster home directory and a username."""
        response = self._register(
            {**self.BODY, "artifactRef": "/mmfs1/home/someone/css-360-qlora/adapter"}
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.registered, [])

    def test_an_unknown_status_is_refused(self) -> None:
        response = self._register({**self.BODY, "status": "promoted"})
        self.assertEqual(response.status_code, 422)

    def test_an_unknown_course_is_404(self) -> None:
        with (
            patch(
                "app.training_queue_routes.db_courses.course_exists",
                return_value=False,
            ),
            patch(
                "app.training_queue_routes.db_models.upsert_model_version"
            ) as upsert,
        ):
            response = self.client.post(
                f"/api/training-queue/courses/{COURSE}/model-versions",
                json=self.BODY,
                headers=HEADERS,
            )

        self.assertEqual(response.status_code, 404)
        upsert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
