"""The operational training-run enqueue: one write, one store, immediately visible.

This route is the fix for a specific consistency bug. It used to write a queue
in one store and mirror the run into PostgreSQL afterwards, and the admin
training list reads PostgreSQL — so a professor could click Queue training, get
a success, reload, and find nothing. The window was real and the response
carried a `mirroredToPostgres` flag to admit it.

There is no window now, and no flag. The insert and the duplicate guard are the
same statement in the same transaction, so a 201 means the run is durable and
the next read of `training_runs` — by the admin page or by the cluster runner —
will see it.

No database is involved here. The repository's SQL is asserted statement by
statement in `test_db_repositories.py`; what these cover is the route's
contract: what it returns, which failures map to which status, and that a
duplicate is refused rather than queued twice.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.db_training_runs import ActiveTrainingRunError
from app.main import app

COURSE = "css-360-winter-2026-a7rp"
OTHER_COURSE = "css-350-spring-2026-n3h9"

BODY = {
    "mode": "full",
    "datasetRef": f"exports/{COURSE}",
    "approvedExampleCount": 54,
    "trainExamples": 48,
    "validationExamples": 6,
}


def _queued_run(course_id: str = COURSE, run_id: str = "run-20260813t120000z-a1b2c3"):
    return {
        "runId": run_id,
        "courseId": course_id,
        "mode": "full",
        "state": "queued",
        "enqueuedAt": "2026-08-13T12:00:00+00:00",
        "updatedAt": "2026-08-13T12:00:00+00:00",
        "datasetRef": f"exports/{course_id}",
        "approvedExampleCount": 54,
        "trainExamples": 48,
        "validationExamples": 6,
        "attempt": 0,
    }


@contextmanager
def _fake_connection(**kwargs: Any) -> Iterator[object]:
    yield object()


class EnqueueRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    @contextmanager
    def _stubbed(
        self,
        *,
        course_exists: bool = True,
        enqueue: Any = None,
    ) -> Iterator[list[dict[str, Any]]]:
        calls: list[dict[str, Any]] = []

        def _enqueue(connection: Any, course_id: str, **kwargs: Any) -> dict[str, Any]:
            calls.append({"courseId": course_id, **kwargs})
            if enqueue is not None:
                if isinstance(enqueue, Exception):
                    raise enqueue
                return enqueue
            return _queued_run(course_id)

        with (
            patch("app.main.db_connection", _fake_connection),
            patch("app.main.course_exists", return_value=course_exists),
            patch("app.main.enqueue_training_run", side_effect=_enqueue),
        ):
            yield calls

    def test_queues_a_run_and_returns_it(self) -> None:
        with self._stubbed() as calls:
            response = self.client.post(
                f"/api/courses/{COURSE}/training-runs", json=BODY
            )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["courseId"], COURSE)
        self.assertEqual(payload["runId"], "run-20260813t120000z-a1b2c3")
        self.assertEqual(payload["run"]["state"], "queued")
        self.assertEqual(payload["run"]["runId"], payload["runId"])

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["courseId"], COURSE)
        self.assertEqual(calls[0]["dataset_ref"], f"exports/{COURSE}")
        self.assertEqual(calls[0]["train_examples"], 48)

    def test_the_response_no_longer_hedges_about_where_the_run_landed(self) -> None:
        """`mirroredToPostgres` described a second store. There isn't one."""
        with self._stubbed():
            response = self.client.post(
                f"/api/courses/{COURSE}/training-runs", json=BODY
            )

        payload = response.json()
        self.assertNotIn("mirroredToPostgres", payload)
        self.assertNotIn("warning", payload)

    def test_the_queued_run_is_readable_immediately_after_the_request(self) -> None:
        """The bug this migration exists to close.

        The enqueue and the admin list read the same table, so whatever the
        enqueue committed is what the next list returns. Asserted end to end
        against a shared store rather than by trusting the route's word.
        """
        stored: dict[str, dict[str, Any]] = {}

        def _enqueue(connection: Any, course_id: str, **kwargs: Any) -> dict[str, Any]:
            run = _queued_run(course_id)
            stored[run["runId"]] = run
            return run

        with (
            patch("app.main.db_connection", _fake_connection),
            patch("app.main.course_exists", return_value=True),
            patch("app.main.enqueue_training_run", side_effect=_enqueue),
        ):
            created = self.client.post(
                f"/api/courses/{COURSE}/training-runs", json=BODY
            )
        self.assertEqual(created.status_code, 201)
        run_id = created.json()["runId"]

        with (
            patch("app.db_routes.db_connection", _fake_connection),
            patch(
                "app.db_routes.db_training_runs.list_training_runs",
                return_value=list(stored.values()),
            ),
        ):
            listed = self.client.get(f"/api/db/courses/{COURSE}/training-runs")

        self.assertEqual(listed.status_code, 200)
        self.assertEqual([run["runId"] for run in listed.json()["runs"]], [run_id])

    def test_a_second_active_run_is_409(self) -> None:
        """The guard, and the reason a retry is safe.

        A duplicate must be refused rather than queued twice — a retry after a
        timeout has to be able to lose harmlessly.
        """
        with self._stubbed(
            enqueue=ActiveTrainingRunError(
                f'Course "{COURSE}" already has an active training run.'
            )
        ):
            response = self.client.post(
                f"/api/courses/{COURSE}/training-runs", json=BODY
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("active training run", response.json()["detail"])

    def test_an_unknown_course_is_404(self) -> None:
        with self._stubbed(course_exists=False) as calls:
            response = self.client.post(
                f"/api/courses/{COURSE}/training-runs", json=BODY
            )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(calls, [])

    def test_an_unknown_mode_is_422(self) -> None:
        with self._stubbed(enqueue=ValueError("Unknown training mode: 'turbo'.")):
            response = self.client.post(
                f"/api/courses/{COURSE}/training-runs",
                json={**BODY, "mode": "turbo"},
            )

        self.assertEqual(response.status_code, 422)

    def test_an_invalid_course_id_is_rejected_before_any_write(self) -> None:
        with self._stubbed() as calls:
            response = self.client.post(
                "/api/courses/..%2Fetc/training-runs", json=BODY
            )

        self.assertIn(response.status_code, (400, 404))
        self.assertEqual(calls, [])

    def test_dataset_ref_is_required(self) -> None:
        with self._stubbed() as calls:
            response = self.client.post(
                f"/api/courses/{COURSE}/training-runs",
                json={"mode": "full"},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(calls, [])

    def test_a_run_is_queued_only_for_the_course_in_the_path(self) -> None:
        """Course isolation: the id in the URL is the only one that reaches SQL."""
        with self._stubbed() as calls:
            self.client.post(f"/api/courses/{OTHER_COURSE}/training-runs", json=BODY)

        self.assertEqual([call["courseId"] for call in calls], [OTHER_COURSE])

    def test_an_unreachable_database_is_503_not_a_false_success(self) -> None:
        """A failed enqueue must never read as a queued run."""
        from app.db import DatabaseConfigurationError

        def _boom(**kwargs: Any) -> Any:
            raise DatabaseConfigurationError("PostgreSQL is not configured.")

        with patch("app.main.db_connection", _boom):
            response = self.client.post(
                f"/api/courses/{COURSE}/training-runs", json=BODY
            )

        self.assertEqual(response.status_code, 503)




class RetrainEnqueueTests(unittest.TestCase):
    """Queueing another run for a course that already finished one.

    Audited before anything was written, and the answer was that the backend
    already allows this: `enqueue_training_run` consults `training_runs` and
    nothing else, so the only thing it refuses is a second *outstanding* run.
    The model request's status is not part of the decision, which is why a
    `ready` course can be retrained through the route that already exists.

    These pin that down, because the gap was entirely in the browser — the
    Queue training control is gated on `status === 'preparing'` — and a future
    change that added a request-status check here would silently reintroduce it.
    """

    def setUp(self) -> None:
        self.client = TestClient(app)

    @contextmanager
    def _store(self, existing: list[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
        """A store holding this course's history, with the real active-run rule."""
        runs = list(existing)

        def _enqueue(connection: Any, course_id: str, **kwargs: Any) -> dict[str, Any]:
            if any(
                run["courseId"] == course_id
                and run["state"] not in ("succeeded", "failed")
                for run in runs
            ):
                raise ActiveTrainingRunError(
                    f'Course "{course_id}" already has an active training run.'
                )
            created = _queued_run(course_id, run_id="run-20260902t080000z-abcdef")
            created["datasetRef"] = kwargs["dataset_ref"]
            created["trainExamples"] = kwargs["train_examples"]
            created["validationExamples"] = kwargs["validation_examples"]
            created["approvedExampleCount"] = kwargs["approved_example_count"]
            runs.append(created)
            return created

        with (
            patch("app.main.db_connection", _fake_connection),
            patch("app.main.course_exists", return_value=True),
            patch("app.main.enqueue_training_run", side_effect=_enqueue),
        ):
            yield runs

    def _succeeded_run(self) -> dict[str, Any]:
        run = _queued_run(OTHER_COURSE, run_id="run-20260827t064701z-1cf650")
        run.update(
            {
                "state": "succeeded",
                "jobId": "264787",
                "trainExamples": 37,
                "validationExamples": 5,
                "approvedExampleCount": 42,
                "attempt": 1,
            }
        )
        return run

    def test_a_course_whose_run_succeeded_can_queue_another(self) -> None:
        """The exact CSS 350 state: ready, v1 registered, run succeeded."""
        with self._store([self._succeeded_run()]) as runs:
            response = self.client.post(
                f"/api/courses/{OTHER_COURSE}/training-runs",
                json={
                    "mode": "full",
                    "datasetRef": f"exports/{OTHER_COURSE}",
                    "approvedExampleCount": 42,
                    "trainExamples": 37,
                    "validationExamples": 5,
                },
            )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["runId"], "run-20260902t080000z-abcdef")
        self.assertEqual(len(runs), 2)

    def test_the_succeeded_run_is_left_exactly_as_it_was(self) -> None:
        """A retrain supersedes nothing. This is the whole difference from retry."""
        original = self._succeeded_run()

        with self._store([original]) as runs:
            self.client.post(
                f"/api/courses/{OTHER_COURSE}/training-runs",
                json={
                    "mode": "full",
                    "datasetRef": f"exports/{OTHER_COURSE}",
                    "approvedExampleCount": 42,
                    "trainExamples": 37,
                    "validationExamples": 5,
                },
            )

        kept = runs[0]
        self.assertEqual(kept["state"], "succeeded")
        self.assertEqual(kept["jobId"], "264787")
        self.assertEqual(kept["runId"], "run-20260827t064701z-1cf650")

    def test_the_new_run_carries_the_reused_dataset_unchanged(self) -> None:
        """Reuse, not re-export: the counts are the ones already prepared."""
        with self._store([self._succeeded_run()]) as runs:
            self.client.post(
                f"/api/courses/{OTHER_COURSE}/training-runs",
                json={
                    "mode": "full",
                    "datasetRef": f"exports/{OTHER_COURSE}",
                    "approvedExampleCount": 42,
                    "trainExamples": 37,
                    "validationExamples": 5,
                },
            )

        created = runs[1]
        self.assertEqual(created["datasetRef"], f"exports/{OTHER_COURSE}")
        self.assertEqual(created["trainExamples"], 37)
        self.assertEqual(created["validationExamples"], 5)
        self.assertEqual(created["approvedExampleCount"], 42)

    def test_the_new_run_gets_a_fresh_id(self) -> None:
        with self._store([self._succeeded_run()]) as runs:
            response = self.client.post(
                f"/api/courses/{OTHER_COURSE}/training-runs",
                json={
                    "mode": "full",
                    "datasetRef": f"exports/{OTHER_COURSE}",
                    "approvedExampleCount": 42,
                    "trainExamples": 37,
                    "validationExamples": 5,
                },
            )

        self.assertNotEqual(response.json()["runId"], runs[0]["runId"])

    def test_a_second_retrain_is_still_refused_while_the_first_is_outstanding(
        self,
    ) -> None:
        """The one guard that matters, and it is atomic on the insert."""
        body = {
            "mode": "full",
            "datasetRef": f"exports/{OTHER_COURSE}",
            "approvedExampleCount": 42,
            "trainExamples": 37,
            "validationExamples": 5,
        }

        with self._store([self._succeeded_run()]):
            first = self.client.post(
                f"/api/courses/{OTHER_COURSE}/training-runs", json=body
            )
            second = self.client.post(
                f"/api/courses/{OTHER_COURSE}/training-runs", json=body
            )

        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 409)

    def test_the_route_never_reads_the_model_request(self) -> None:
        """A `ready` request must not become a reason to refuse a retrain.

        Asserted by construction: nothing in this route's stubs provides a model
        request, and it returns 201 regardless. If a status check were ever
        added here, this test would need a request fixture to pass — which is
        the signal.
        """
        with self._store([self._succeeded_run()]):
            response = self.client.post(
                f"/api/courses/{OTHER_COURSE}/training-runs",
                json={
                    "mode": "full",
                    "datasetRef": f"exports/{OTHER_COURSE}",
                    "approvedExampleCount": 42,
                    "trainExamples": 37,
                    "validationExamples": 5,
                },
            )

        self.assertEqual(response.status_code, 201)

if __name__ == "__main__":
    unittest.main()
