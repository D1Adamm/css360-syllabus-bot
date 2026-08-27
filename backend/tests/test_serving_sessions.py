"""Serving sessions: whether a GPU is running the inference service right now.

A session is not a deployment status. `course_model_versions.deployment`
describes one course's artifact and is durable; a session describes one Slurm
allocation with a wall clock on it and belongs to no course in particular, since
one allocation serves every course whose adapter it can load.

The two properties that matter here:

  - a session ends when its allocation does, whether or not anyone reported it.
    That is what makes a dropped SSH session harmless.
  - the browser-facing view carries no compute hostname and no port. Every
    `/api/db` route is reachable without a credential.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import db_serving_sessions as sessions
from app.main import app

TOKEN = "test-worker-token"
HEADERS = {"X-Training-Worker-Token": TOKEN}

NOW = datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc)


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        "session_id": "serve-264787",
        "job_id": "264787",
        "node": "g014",
        "port": 8001,
        "state": "ready",
        "started_at": NOW - timedelta(minutes=10),
        "expires_at": NOW + timedelta(hours=1),
        "updated_at": NOW - timedelta(minutes=5),
        "detail": {
            "courses": [
                {"courseId": "css-350-spring-2026-n3h9", "currentVersion": "v1"}
            ],
            "baseModel": "meta-llama/Llama-3.2-3B-Instruct",
        },
    }
    row.update(overrides)
    return row


@contextmanager
def _fake_connection(**kwargs: Any) -> Iterator[object]:
    yield object()


class SessionMappingTests(unittest.TestCase):
    def test_a_live_session_reads_as_live(self) -> None:
        record = sessions.map_serving_session(_row(), now=NOW)

        self.assertEqual(record["state"], "ready")
        self.assertTrue(record["live"])
        self.assertEqual(record["node"], "g014")
        self.assertEqual(record["port"], 8001)

    def test_a_session_past_its_wall_clock_reads_as_expired(self) -> None:
        """Nothing wrote `expired`. The allocation ended; the read says so.

        A reader that trusted the stored state would show a service as live for
        as long as it took someone to notice it was not — which, for a session
        that ends at the end of a class, is until the next time anybody looks.
        """
        record = sessions.map_serving_session(
            _row(expires_at=NOW - timedelta(minutes=1)), now=NOW
        )

        self.assertEqual(record["state"], "expired")
        self.assertEqual(record["storedState"], "ready")
        self.assertFalse(record["live"])

    def test_a_stopped_session_stays_stopped(self) -> None:
        record = sessions.map_serving_session(
            _row(state="stopped", expires_at=NOW + timedelta(hours=1)), now=NOW
        )

        self.assertEqual(record["state"], "stopped")
        self.assertFalse(record["live"])

    def test_a_naive_expiry_is_read_as_utc(self) -> None:
        record = sessions.map_serving_session(
            _row(expires_at=(NOW - timedelta(hours=1)).replace(tzinfo=None)), now=NOW
        )

        self.assertEqual(record["state"], "expired")


class PublicViewTests(unittest.TestCase):
    def test_the_browser_view_drops_the_node_and_port(self) -> None:
        """The one field in this record that says how to reach a machine.

        `/api/db` needs no credential, so a compute hostname plus a listening
        port would be public. The job id stays: `training_runs.jobId` is already
        served on those routes, and removing it here would be inconsistent
        without being protective.
        """
        record = sessions.map_serving_session(_row(), now=NOW)
        public = sessions.public_serving_session(record)

        assert public is not None
        self.assertNotIn("node", public)
        self.assertNotIn("port", public)
        self.assertEqual(public["jobId"], "264787")
        self.assertEqual(public["state"], "ready")
        self.assertTrue(public["live"])

    def test_the_browser_view_keeps_what_is_being_served(self) -> None:
        public = sessions.public_serving_session(
            sessions.map_serving_session(_row(), now=NOW)
        )

        assert public is not None
        self.assertEqual(
            public["courses"],
            [{"courseId": "css-350-spring-2026-n3h9", "currentVersion": "v1"}],
        )

    def test_no_session_is_a_normal_answer(self) -> None:
        self.assertIsNone(sessions.public_serving_session(None))


class ServingSessionApiTests(unittest.TestCase):
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

    def test_registering_a_session_records_node_port_and_expiry(self) -> None:
        recorded: list[dict[str, Any]] = []

        with patch(
            "app.training_queue_routes.db_serving_sessions.upsert_serving_session",
            side_effect=lambda connection, session: (
                recorded.append(dict(session)) or sessions.map_serving_session(_row(), now=NOW)
            ),
        ):
            response = self.client.put(
                "/api/training-queue/serving-sessions/serve-264787",
                json={
                    "jobId": "264787",
                    "node": "g014",
                    "port": 8001,
                    "state": "ready",
                    "expiresAt": "2026-08-27T14:00:00Z",
                    "detail": {"courses": []},
                },
                headers=HEADERS,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(recorded[0]["sessionId"], "serve-264787")
        self.assertEqual(recorded[0]["node"], "g014")
        self.assertEqual(recorded[0]["expiresAt"], "2026-08-27T14:00:00Z")

    def test_a_malformed_session_id_is_refused(self) -> None:
        response = self.client.put(
            "/api/training-queue/serving-sessions/Serve%20264787",
            json={
                "jobId": "264787",
                "node": "g014",
                "port": 8001,
                "expiresAt": "2026-08-27T14:00:00Z",
            },
            headers=HEADERS,
        )

        self.assertEqual(response.status_code, 400)

    def test_an_out_of_range_port_is_refused(self) -> None:
        response = self.client.put(
            "/api/training-queue/serving-sessions/serve-264787",
            json={
                "jobId": "264787",
                "node": "g014",
                "port": 99999,
                "expiresAt": "2026-08-27T14:00:00Z",
            },
            headers=HEADERS,
        )

        self.assertEqual(response.status_code, 422)

    def test_stopping_a_session_that_is_already_gone_is_not_an_error(self) -> None:
        """A session that expired on its own is exactly what "nothing to stop" is.

        Reporting that as a failure would make a successful stop look broken to
        an operator who did everything right.
        """
        with patch(
            "app.training_queue_routes.db_serving_sessions.stop_serving_session",
            return_value=None,
        ):
            response = self.client.post(
                "/api/training-queue/serving-sessions/serve-264787/stopped",
                headers=HEADERS,
            )

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["session"])

    def test_the_worker_route_returns_the_node_it_needs(self) -> None:
        """The one response in the system that says how to reach a compute node.

        Worker-token only, which is why the VM tunnel script can look a node up
        instead of the operator copying a hostname between two machines.
        """
        with patch(
            "app.training_queue_routes.db_serving_sessions.current_serving_session",
            return_value=sessions.map_serving_session(_row(), now=NOW),
        ):
            response = self.client.get(
                "/api/training-queue/serving-session", headers=HEADERS
            )

        self.assertEqual(response.status_code, 200)
        session = response.json()["session"]
        self.assertEqual(session["node"], "g014")
        self.assertEqual(session["port"], 8001)

    def test_the_worker_route_needs_the_token(self) -> None:
        response = self.client.get("/api/training-queue/serving-session")
        self.assertEqual(response.status_code, 401)


class BrowserFacingSessionRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self._connection = patch("app.db_routes.db_connection", _fake_connection)
        self._connection.start()
        self.addCleanup(self._connection.stop)

    def test_the_browser_route_hides_the_node_and_port(self) -> None:
        with patch(
            "app.db_routes.db_serving_sessions.current_serving_session",
            return_value=sessions.map_serving_session(_row(), now=NOW),
        ):
            response = self.client.get("/api/db/serving-session")

        self.assertEqual(response.status_code, 200)
        session = response.json()["session"]
        self.assertNotIn("node", session)
        self.assertNotIn("port", session)
        self.assertEqual(session["jobId"], "264787")

    def test_nothing_serving_is_a_null_session(self) -> None:
        with patch(
            "app.db_routes.db_serving_sessions.current_serving_session",
            return_value=None,
        ):
            response = self.client.get("/api/db/serving-session")

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["session"])


if __name__ == "__main__":
    unittest.main()
