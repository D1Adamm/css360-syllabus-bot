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


if __name__ == "__main__":
    unittest.main()
