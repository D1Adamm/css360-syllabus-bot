"""The operational training-run enqueue: Firebase queue first, PostgreSQL after.

The browser used to write the Firebase queue itself. It no longer can, but the
queue could not move: `scripts/lib/training_queue.py` on Tillicum claims work
from Firebase with an ETag compare-and-set. So the backend now writes both, in a
fixed order, and says so when only one of them landed.

Firebase is stubbed at the httpx boundary — the same place `conftest.py` blocks
real requests — so these exercise the actual URL, headers, and payload the
runner will read, without touching a database.
"""

from __future__ import annotations

import json
import os
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.firebase_training_runs import (
    ActiveTrainingRunError,
    build_run_record,
    enqueue_training_run,
)
from app.main import app

COURSE = "css-360-winter-2026-a7rp"
RUNS_URL_FRAGMENT = f"courses/{COURSE}/trainingRuns.json"

TERMINAL_RUN = {
    "courseId": COURSE,
    "mode": "full",
    "state": "succeeded",
    "enqueuedAt": "2026-08-01T00:00:00+00:00",
}

ACTIVE_RUN = {**TERMINAL_RUN, "state": "queued"}


class FakeResponse:
    def __init__(self, status_code: int, payload: Any = None, etag: str = 'W/"e1"'):
        self.status_code = status_code
        self._payload = payload
        self.headers = {"ETag": etag}

    def json(self) -> Any:
        return self._payload


class FakeFirebaseClient:
    """Records the GET/PUT pair the compare-and-set performs."""

    def __init__(self, *, current: Any = None, put_statuses: list[int] | None = None):
        self.current = current
        self.put_statuses = put_statuses or [200]
        self.gets: list[tuple[str, dict]] = []
        self.puts: list[tuple[str, Any, dict]] = []

    async def __aenter__(self) -> "FakeFirebaseClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def get(self, url: str, headers: dict | None = None) -> FakeResponse:
        self.gets.append((url, headers or {}))
        return FakeResponse(200, self.current)

    async def put(
        self, url: str, json: Any = None, headers: dict | None = None
    ) -> FakeResponse:
        self.puts.append((url, json, headers or {}))
        status = self.put_statuses[min(len(self.puts) - 1, len(self.put_statuses) - 1)]
        if status == 200:
            self.current = json
        return FakeResponse(status, json)


def _with_firebase(client: FakeFirebaseClient):
    return patch(
        "app.firebase_training_runs.httpx.AsyncClient", return_value=client
    )


def _configured():
    return patch.dict(
        os.environ,
        {"FIREBASE_DATABASE_URL": "https://example-default-rtdb.firebaseio.com"},
    )


class FirebaseQueuePayloadTests(unittest.IsolatedAsyncioTestCase):
    """1. The queue payload is exactly what Tillicum parses."""

    def test_record_carries_every_field_the_runner_requires(self) -> None:
        record = build_run_record(
            course_id=COURSE,
            mode="full",
            dataset_ref=f"exports/{COURSE}",
            approved_example_count=54,
            train_examples=48,
            validation_examples=6,
            enqueued_at="2026-08-20T10:00:00+00:00",
        )

        # `parse_run` rejects a record missing any of these four.
        for required in ("courseId", "state", "mode", "enqueuedAt"):
            self.assertIn(required, record)

        self.assertEqual(record["state"], "queued")
        self.assertEqual(record["attempt"], 0)
        self.assertEqual(record["datasetRef"], f"exports/{COURSE}")
        self.assertEqual(record["approvedExampleCount"], 54)

    def test_record_omits_run_id_and_job_id(self) -> None:
        """The key is the id, and only the cluster produces a job id."""
        record = build_run_record(
            course_id=COURSE,
            mode="smoke",
            dataset_ref="exports/x",
            approved_example_count=0,
            train_examples=0,
            validation_examples=0,
            enqueued_at="2026-08-20T10:00:00+00:00",
        )

        self.assertNotIn("runId", record)
        self.assertNotIn("jobId", record)
        self.assertNotIn("claim", record)

    async def test_writes_to_the_course_scoped_queue_node(self) -> None:
        fake = FakeFirebaseClient(current=None)

        with _configured(), _with_firebase(fake):
            queued = await enqueue_training_run(
                course_id=COURSE, mode="full", dataset_ref=f"exports/{COURSE}"
            )

        self.assertIn(RUNS_URL_FRAGMENT, fake.puts[0][0])
        # The run is stored under its own id, alongside whatever was there.
        self.assertEqual(list(fake.puts[0][1]), [queued["runId"]])

    async def test_uses_compare_and_set_headers(self) -> None:
        fake = FakeFirebaseClient(current=None)

        with _configured(), _with_firebase(fake):
            await enqueue_training_run(
                course_id=COURSE, mode="full", dataset_ref="exports/x"
            )

        self.assertEqual(fake.gets[0][1].get("X-Firebase-ETag"), "true")
        self.assertEqual(fake.puts[0][2].get("if-match"), 'W/"e1"')

    async def test_run_id_matches_the_pattern_the_runner_validates(self) -> None:
        import re

        fake = FakeFirebaseClient(current=None)
        with _configured(), _with_firebase(fake):
            queued = await enqueue_training_run(
                course_id=COURSE, mode="full", dataset_ref="exports/x"
            )

        # RUN_ID_PATTERN in scripts/lib/training_queue.py.
        self.assertRegex(queued["runId"], re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$"))

    async def test_existing_runs_are_preserved(self) -> None:
        fake = FakeFirebaseClient(current={"run-old": TERMINAL_RUN})

        with _configured(), _with_firebase(fake):
            queued = await enqueue_training_run(
                course_id=COURSE, mode="full", dataset_ref="exports/x"
            )

        written = fake.puts[0][1]
        self.assertIn("run-old", written)
        self.assertIn(queued["runId"], written)


class DuplicateSemanticsTests(unittest.IsolatedAsyncioTestCase):
    """3. The one-active-run rule survives the move to the backend."""

    async def test_refuses_while_a_run_is_outstanding(self) -> None:
        fake = FakeFirebaseClient(current={"run-1": ACTIVE_RUN})

        with _configured(), _with_firebase(fake):
            with self.assertRaises(ActiveTrainingRunError):
                await enqueue_training_run(
                    course_id=COURSE, mode="full", dataset_ref="exports/x"
                )

        self.assertEqual(fake.puts, [])

    async def test_allows_a_new_run_once_every_earlier_one_finished(self) -> None:
        fake = FakeFirebaseClient(
            current={"run-1": TERMINAL_RUN, "run-2": {**TERMINAL_RUN, "state": "failed"}}
        )

        with _configured(), _with_firebase(fake):
            queued = await enqueue_training_run(
                course_id=COURSE, mode="full", dataset_ref="exports/x"
            )

        self.assertTrue(queued["runId"])

    async def test_a_lost_race_is_retried_and_then_refused(self) -> None:
        """412 means someone else wrote first; the retry sees their run."""
        fake = FakeFirebaseClient(current=None, put_statuses=[412])

        async def _get(url: str, headers: dict | None = None) -> FakeResponse:
            fake.gets.append((url, headers or {}))
            # Second read reflects the winner's write.
            payload = None if len(fake.gets) == 1 else {"run-1": ACTIVE_RUN}
            return FakeResponse(200, payload)

        fake.get = _get  # type: ignore[assignment]

        with _configured(), _with_firebase(fake):
            with self.assertRaises(ActiveTrainingRunError):
                await enqueue_training_run(
                    course_id=COURSE, mode="full", dataset_ref="exports/x"
                )

        self.assertEqual(len(fake.gets), 2)

    async def test_rejects_an_unknown_mode_before_any_request(self) -> None:
        fake = FakeFirebaseClient(current=None)

        with _configured(), _with_firebase(fake):
            with self.assertRaises(ValueError):
                await enqueue_training_run(
                    course_id=COURSE, mode="turbo", dataset_ref="exports/x"
                )

        self.assertEqual(fake.gets, [])


class EnqueueRouteTests(unittest.TestCase):
    """2, 5, 6. The route mirrors to PostgreSQL and is honest when it cannot."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def _post(self, mirror_side_effect: Exception | None = None, **kwargs: Any):
        fake = FakeFirebaseClient(current=None)
        self.upsert = AsyncMock()

        mirrored: list[tuple[str, str, dict]] = []

        def _upsert(connection: Any, course_id: str, run_id: str, record: dict):
            if mirror_side_effect is not None:
                raise mirror_side_effect
            mirrored.append((course_id, run_id, record))
            return {"runId": run_id, **record}

        self.mirrored = mirrored
        self.fake = fake

        from contextlib import contextmanager

        @contextmanager
        def _connection(**_kwargs: Any):
            yield object()

        with (
            _configured(),
            _with_firebase(fake),
            patch("app.main.db_connection", new=_connection),
            patch("app.main.upsert_training_run", side_effect=_upsert),
        ):
            return self.client.post(
                f"/api/courses/{COURSE}/training-runs",
                json={"mode": "full", "datasetRef": f"exports/{COURSE}", **kwargs},
            )

    def test_queues_and_mirrors_the_same_run(self) -> None:
        response = self._post(approvedExampleCount=54)

        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertTrue(body["mirroredToPostgres"])
        self.assertIsNone(body["warning"])

        # The same run id reached both stores.
        firebase_run_id = list(self.fake.puts[0][1])[0]
        self.assertEqual(body["runId"], firebase_run_id)
        self.assertEqual(self.mirrored[0][1], firebase_run_id)
        self.assertEqual(self.mirrored[0][0], COURSE)
        self.assertEqual(self.mirrored[0][2]["approvedExampleCount"], 54)

    def test_a_failed_mirror_is_reported_not_hidden(self) -> None:
        """6. Never a plain success while PostgreSQL is known stale."""
        response = self._post(mirror_side_effect=RuntimeError("postgres down"))

        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertFalse(body["mirroredToPostgres"])
        self.assertIn("PostgreSQL", body["warning"])
        # The run really is queued, so the caller must not be told to retry.
        self.assertTrue(body["runId"])
        self.assertEqual(len(self.fake.puts), 1)

    def test_a_second_active_run_is_409(self) -> None:
        fake = FakeFirebaseClient(current={"run-1": ACTIVE_RUN})

        with _configured(), _with_firebase(fake):
            response = self.client.post(
                f"/api/courses/{COURSE}/training-runs",
                json={"mode": "full", "datasetRef": "exports/x"},
            )

        self.assertEqual(response.status_code, 409)

    def test_nothing_is_mirrored_when_the_queue_write_fails(self) -> None:
        """PostgreSQL must not show a run the cluster never received."""
        fake = FakeFirebaseClient(current=None, put_statuses=[500])
        upsert = AsyncMock()

        with _configured(), _with_firebase(fake), patch(
            "app.main.upsert_training_run", new=upsert
        ):
            response = self.client.post(
                f"/api/courses/{COURSE}/training-runs",
                json={"mode": "full", "datasetRef": "exports/x"},
            )

        self.assertEqual(response.status_code, 503)
        upsert.assert_not_called()

    def test_an_invalid_course_id_is_rejected_before_any_write(self) -> None:
        response = self.client.post(
            "/api/courses/CSS%20360/training-runs",
            json={"mode": "full", "datasetRef": "exports/x"},
        )
        self.assertEqual(response.status_code, 400)

    def test_dataset_ref_is_required(self) -> None:
        response = self.client.post(
            f"/api/courses/{COURSE}/training-runs", json={"mode": "full"}
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
