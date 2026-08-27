"""The endpoint the cluster calls once an adapter is really in the serving tree.

Publication spans two machines, and the ordering is the whole safety story: the
copy happens first, is validated, and only then is this called. So the failure
this must never allow is the database saying a version is live when the cluster
does not have it — and the way that is prevented is that a failed copy never
reaches this endpoint at all.

What is left for these tests is everything that happens once a report does
arrive: that it is idempotent, that it moves exactly one course, that it refuses
to publish something that is not a ready registered version, and that it does
not require a live run the way the training callbacks do.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

CSS350 = "css-350-spring-2026-n3h9"
CSS360 = "css-360-winter-2026-a7rp"
TOKEN = "test-worker-token"
HEADERS = {"X-Training-Worker-Token": TOKEN}


def _version(
    key: str, *, status: str = "ready", deployment: str = "offline"
) -> dict[str, Any]:
    return {
        "version": key,
        "baseModel": "meta-llama/Llama-3.2-3B-Instruct",
        "trainingExampleCount": 37,
        "status": status,
        "deployment": deployment,
        "artifactRef": f"serving/{CSS350}/{key}/adapter",
        "createdAt": "2026-08-27T06:48:00+00:00",
    }


@contextmanager
def _fake_connection(**kwargs: Any) -> Iterator[object]:
    yield object()


class PublicationApiTestCase(unittest.TestCase):
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

        #: course -> {version -> record}. Mutated by the fake publish so a
        #: second call sees what the first one did.
        self.versions: dict[str, dict[str, dict[str, Any]]] = {}
        self.current: dict[str, str] = {}

    def _publish(self, connection: Any, course_id: str, version: str, **kwargs: Any):
        records = self.versions.get(course_id)
        if not records or version not in records:
            return None
        for record in records.values():
            record["deployment"] = "offline"
        records[version]["deployment"] = "online"
        return {
            "courseId": course_id,
            "currentVersion": self.current.get(course_id),
            "versions": records,
        }

    def _post(self, course_id: str, version: str, **kwargs: Any):
        with (
            patch(
                "app.training_queue_routes.db_models.list_model_versions",
                side_effect=lambda connection, cid: list(
                    self.versions.get(cid, {}).values()
                ),
            ),
            patch(
                "app.training_queue_routes.db_models.mark_version_published",
                side_effect=self._publish,
            ),
        ):
            return self.client.post(
                f"/api/training-queue/courses/{course_id}"
                f"/model-versions/{version}/published",
                json=kwargs.pop("json", {}),
                headers=kwargs.pop("headers", HEADERS),
            )

    def _seed(self, course_id: str, records: list[dict[str, Any]], current: str):
        self.versions[course_id] = {item["version"]: item for item in records}
        self.current[course_id] = current


class PublishTests(PublicationApiTestCase):
    def test_publishing_the_newly_trained_version_moves_serving_to_it(self) -> None:
        self._seed(
            CSS350, [_version("v1", deployment="online"), _version("v2")], "v2"
        )

        response = self._post(CSS350, "v2")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["version"], "v2")
        self.assertEqual(payload["deployment"], "online")
        self.assertEqual(payload["previousVersion"], "v1")
        self.assertFalse(payload["unchanged"])

    def test_the_previously_published_version_is_demoted(self) -> None:
        """Exactly one version of a course is served at a time."""
        self._seed(
            CSS350, [_version("v1", deployment="online"), _version("v2")], "v2"
        )

        self._post(CSS350, "v2")

        self.assertEqual(self.versions[CSS350]["v1"]["deployment"], "offline")
        self.assertEqual(self.versions[CSS350]["v2"]["deployment"], "online")

    def test_the_demoted_version_stays_registered_and_ready(self) -> None:
        """v1 keeps existing. Only the claim that it is serving is withdrawn."""
        self._seed(
            CSS350, [_version("v1", deployment="online"), _version("v2")], "v2"
        )

        self._post(CSS350, "v2")

        self.assertEqual(self.versions[CSS350]["v1"]["status"], "ready")
        self.assertEqual(
            self.versions[CSS350]["v1"]["artifactRef"], f"serving/{CSS350}/v1/adapter"
        )

    def test_publishing_an_older_version_is_allowed(self) -> None:
        """Rolling a bad v2 back to v1 is a thing an operator must be able to do.

        Deliberately not guarded by the current-run check the training callbacks
        use: this is a person acting on a registered artifact, not a report from
        a run that may have been superseded.
        """
        self._seed(
            CSS350, [_version("v1"), _version("v2", deployment="online")], "v2"
        )

        response = self._post(CSS350, "v1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.versions[CSS350]["v1"]["deployment"], "online")
        self.assertEqual(self.versions[CSS350]["v2"]["deployment"], "offline")

    def test_current_version_is_reported_back_unchanged(self) -> None:
        """Publishing does not move `current_version`. They answer different
        questions: newest model versus the one being served."""
        self._seed(
            CSS350, [_version("v1"), _version("v2", deployment="online")], "v2"
        )

        payload = self._post(CSS350, "v1").json()

        self.assertEqual(payload["currentVersion"], "v2")
        self.assertEqual(payload["version"], "v1")


class IdempotencyTests(PublicationApiTestCase):
    def test_publishing_twice_changes_nothing_the_second_time(self) -> None:
        """A rerun of the promote script, or a redelivered queued report."""
        self._seed(
            CSS350, [_version("v1", deployment="online"), _version("v2")], "v2"
        )

        first = self._post(CSS350, "v2")
        second = self._post(CSS350, "v2")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertFalse(first.json()["unchanged"])
        self.assertTrue(second.json()["unchanged"])
        self.assertEqual(self.versions[CSS350]["v2"]["deployment"], "online")

    def test_a_repeated_publish_does_not_demote_anything_new(self) -> None:
        self._seed(
            CSS350, [_version("v1", deployment="online"), _version("v2")], "v2"
        )

        self._post(CSS350, "v2")
        self._post(CSS350, "v2")

        online = [
            key
            for key, record in self.versions[CSS350].items()
            if record["deployment"] == "online"
        ]
        self.assertEqual(online, ["v2"])


class RefusalTests(PublicationApiTestCase):
    def test_an_unregistered_version_is_refused(self) -> None:
        """The database must never claim to serve something it does not know."""
        self._seed(CSS350, [_version("v1", deployment="online")], "v1")

        response = self._post(CSS350, "v2")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.versions[CSS350]["v1"]["deployment"], "online")

    def test_a_version_that_is_not_ready_is_refused(self) -> None:
        self._seed(
            CSS350,
            [_version("v1", deployment="online"), _version("v2", status="failed")],
            "v1",
        )

        response = self._post(CSS350, "v2")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.versions[CSS350]["v1"]["deployment"], "online")

    def test_a_refused_publication_leaves_serving_exactly_as_it_was(self) -> None:
        """The property that keeps a failed publish from taking a course down."""
        self._seed(CSS350, [_version("v1", deployment="online")], "v1")

        self._post(CSS350, "v9")

        online = [
            key
            for key, record in self.versions[CSS350].items()
            if record["deployment"] == "online"
        ]
        self.assertEqual(online, ["v1"])

    def test_a_malformed_version_is_refused_before_anything_is_read(self) -> None:
        self._seed(CSS350, [_version("v1")], "v1")

        response = self._post(CSS350, "latest")

        self.assertEqual(response.status_code, 422)

    def test_an_invalid_course_id_is_refused(self) -> None:
        response = self._post("Not A Course", "v1")

        self.assertEqual(response.status_code, 400)

    def test_the_endpoint_requires_the_worker_token(self) -> None:
        self._seed(CSS350, [_version("v1")], "v1")

        response = self._post(CSS350, "v1", headers={})

        self.assertEqual(response.status_code, 401)

    def test_an_unconfigured_backend_refuses_rather_than_opening_up(self) -> None:
        self._seed(CSS350, [_version("v1")], "v1")

        with patch.dict("os.environ", {"TRAINING_WORKER_TOKEN": ""}):
            response = self._post(CSS350, "v1")

        self.assertEqual(response.status_code, 503)


class CourseIsolationTests(PublicationApiTestCase):
    def test_publishing_for_one_course_leaves_another_untouched(self) -> None:
        self._seed(
            CSS350, [_version("v1", deployment="online"), _version("v2")], "v2"
        )
        self._seed(CSS360, [_version("v1", deployment="online")], "v1")

        self._post(CSS350, "v2")

        self.assertEqual(self.versions[CSS360]["v1"]["deployment"], "online")

    def test_a_version_registered_only_for_another_course_is_not_found(self) -> None:
        """Structural: the lookup is keyed by the course in the path."""
        self._seed(CSS350, [_version("v1"), _version("v2")], "v2")
        self._seed(CSS360, [_version("v1", deployment="online")], "v1")

        response = self._post(CSS360, "v2")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.versions[CSS360]["v1"]["deployment"], "online")


if __name__ == "__main__":
    unittest.main()
