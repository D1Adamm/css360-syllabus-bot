"""Admin → Models → Activate: a version serves only once the VM can serve it.

The invariant
-------------
Finishing or registering training must never switch production inference to a
version that has not been installed and mapped on the VM. Registration lands a
version `offline`; the only browser-reachable way to make it `online` — the
fact inference resolves — is this route, and it writes nothing unless the
fine-tuned service's own `/health` lists that exact course and version as
servable (mapped in `FINETUNED_OLLAMA_MODELS`, and the Ollama model present).

Everything here runs against an in-memory registry and a stubbed `/health`.
Resolution is the real `resolve_current_course_model`, reading the same store,
so each test can say which version a student's question would now reach.
"""

from __future__ import annotations

import copy
import unittest
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.course_model_resolution import resolve_current_course_model
from app.main import app

CSS360D = "css360d-fall-2026-q0ne"
CSS360E = "css360e-autumn-2026-c08m"
TOKEN = "test-worker-token"
WORKER_HEADERS = {"X-Training-Worker-Token": TOKEN}


def _version(key: str, *, status: str = "ready", deployment: str = "offline") -> dict[str, Any]:
    return {
        "version": key,
        "baseModel": "meta-llama/Llama-3.2-3B-Instruct",
        "trainingExampleCount": 60,
        "status": status,
        "deployment": deployment,
        "artifactRef": f"qlora-runs/{CSS360D}/{key}/adapter",
        "createdAt": "2026-09-30T06:00:00+00:00",
    }


def _health(**servable: list[str]) -> dict[str, Any]:
    """`/health` as the backend client returns it: only servable versions listed."""
    return {
        "status": "ok",
        "courses": [
            {"courseId": course, "versions": versions, "currentVersion": versions[-1]}
            for course, versions in servable.items()
            if versions
        ],
    }


@contextmanager
def _fake_connection(**kwargs: Any) -> Iterator[object]:
    yield object()


class ActivationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        #: course -> registry, the one store every patched function reads.
        self.registries: dict[str, dict[str, Any]] = {}
        self.audit: list[dict[str, Any]] = []
        self.published: list[tuple[str, str]] = []
        self.health: dict[str, Any] | Exception = _health()

        patches = [
            patch("app.db_routes.db_connection", _fake_connection),
            patch(
                "app.db_routes.db_models.get_model_registry",
                side_effect=lambda connection, cid: copy.deepcopy(self.registries.get(cid)),
            ),
            patch(
                "app.db_routes.db_models.mark_version_published",
                side_effect=self._publish,
            ),
            patch(
                "app.db_routes.db_admin_actions.record_action",
                side_effect=lambda connection, **fields: self.audit.append(fields),
            ),
            patch(
                "app.db_routes.check_finetuned_service_health",
                new=AsyncMock(side_effect=self._health_call),
            ),
            patch(
                "app.course_model_resolution._load_registry",
                side_effect=lambda cid: copy.deepcopy(self.registries.get(cid)),
            ),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    async def _health_call(self) -> dict[str, Any]:
        if isinstance(self.health, Exception):
            raise self.health
        return self.health

    def _publish(self, connection: Any, course_id: str, version: str, **kwargs: Any):
        registry = self.registries.get(course_id)
        if not registry or version not in registry["versions"]:
            return None
        for record in registry["versions"].values():
            record["deployment"] = "offline"
        registry["versions"][version]["deployment"] = "online"
        self.published.append((course_id, version))
        return copy.deepcopy(registry)

    def _seed(self, course_id: str, current: str, *records: dict[str, Any]) -> None:
        self.registries[course_id] = {
            "courseId": course_id,
            "currentVersion": current,
            "versions": {item["version"]: item for item in records},
        }

    def _activate(self, course_id: str, version: str):
        return self.client.post(
            f"/api/db/courses/{course_id}/model-versions/{version}/activate"
        )

    def _served(self, course_id: str) -> tuple[str, str]:
        resolved = resolve_current_course_model(course_id)
        return resolved["version"], resolved["resolvedFrom"]


class ActivateMappedVersionTests(ActivationTestCase):
    def test_a_mapped_and_available_version_is_activated(self) -> None:
        """CSS 360D's real first step: v1 serving through the fallback, made explicit."""
        self._seed(CSS360D, "v1", _version("v1"))
        self.health = _health(**{CSS360D: ["v1"]})
        self.assertEqual(self._served(CSS360D), ("v1", "current"))

        response = self._activate(CSS360D, "v1")

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["version"], "v1")
        self.assertIsNone(payload["previousVersion"])
        self.assertFalse(payload["unchanged"])
        self.assertEqual(payload["model"]["versions"]["v1"]["deployment"], "online")
        self.assertEqual(self.published, [(CSS360D, "v1")])
        self.assertEqual(self._served(CSS360D), ("v1", "published"))

    def test_activation_is_audited(self) -> None:
        self._seed(CSS360D, "v1", _version("v1"))
        self.health = _health(**{CSS360D: ["v1"]})

        self._activate(CSS360D, "v1")

        self.assertEqual(len(self.audit), 1)
        entry = self.audit[0]
        self.assertEqual(entry["action"], "model.activate")
        self.assertEqual(entry["course_id"], CSS360D)
        self.assertEqual(entry["target_id"], f"{CSS360D}@v1")
        self.assertEqual(entry["detail"], {"version": "v1", "previousVersion": None})
        self.assertEqual(entry["actor_role"], "admin")

    def test_activating_the_active_version_changes_nothing(self) -> None:
        self._seed(CSS360D, "v1", _version("v1", deployment="online"))
        self.health = _health(**{CSS360D: ["v1"]})

        response = self._activate(CSS360D, "v1")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["unchanged"])
        self.assertEqual(self.published, [])
        self.assertEqual(self.audit, [])


class RefuseUnservableVersionTests(ActivationTestCase):
    """Nothing is written unless the service says it can answer that version."""

    def _assert_nothing_written(self) -> None:
        self.assertEqual(self.published, [])
        self.assertEqual(self.audit, [])

    def test_an_unmapped_version_is_refused(self) -> None:
        """v2 registered and ready, but the VM maps only v1."""
        self._seed(
            CSS360D, "v2", _version("v1", deployment="online"), _version("v2")
        )
        self.health = _health(**{CSS360D: ["v1"]})

        response = self._activate(CSS360D, "v2")

        self.assertEqual(response.status_code, 409)
        self.assertIn("set-mapping", response.json()["detail"])
        self.assertIn("Servable now: v1", response.json()["detail"])
        self._assert_nothing_written()
        self.assertEqual(self._served(CSS360D), ("v1", "published"))

    def test_a_course_the_service_does_not_list_is_refused(self) -> None:
        """Mapped but its Ollama model missing: the VM omits it from `courses`."""
        self._seed(CSS360D, "v1", _version("v1"))
        self.health = _health()

        response = self._activate(CSS360D, "v1")

        self.assertEqual(response.status_code, 409)
        self.assertIn("Servable now: none", response.json()["detail"])
        self._assert_nothing_written()

    def test_another_courses_mapping_does_not_count(self) -> None:
        self._seed(CSS360D, "v1", _version("v1"))
        self.health = _health(**{CSS360E: ["v1"]})

        response = self._activate(CSS360D, "v1")

        self.assertEqual(response.status_code, 409)
        self._assert_nothing_written()

    def test_an_unreachable_service_activates_nothing(self) -> None:
        self._seed(CSS360D, "v1", _version("v1"))
        self.health = HTTPException(status_code=503, detail="Fine-tuned inference service is unavailable.")

        response = self._activate(CSS360D, "v1")

        self.assertEqual(response.status_code, 503)
        self.assertIn("nothing was activated", response.json()["detail"])
        self._assert_nothing_written()

    def test_an_unregistered_version_is_404_without_asking_the_service(self) -> None:
        self._seed(CSS360D, "v1", _version("v1"))
        self.health = AssertionError("the service must not be asked")

        response = self._activate(CSS360D, "v3")

        self.assertEqual(response.status_code, 404)
        self._assert_nothing_written()

    def test_a_version_that_is_not_ready_is_refused(self) -> None:
        self._seed(CSS360D, "v2", _version("v1"), _version("v2", status="failed"))
        self.health = _health(**{CSS360D: ["v1", "v2"]})

        response = self._activate(CSS360D, "v2")

        self.assertEqual(response.status_code, 409)
        self._assert_nothing_written()

    def test_a_malformed_version_is_422(self) -> None:
        self._seed(CSS360D, "v1", _version("v1"))

        response = self._activate(CSS360D, "latest")

        self.assertEqual(response.status_code, 422)
        self._assert_nothing_written()


class RollbackTests(ActivationTestCase):
    def test_an_older_servable_version_can_be_activated_again(self) -> None:
        self._seed(
            CSS360D, "v2", _version("v1"), _version("v2", deployment="online")
        )
        self.health = _health(**{CSS360D: ["v1", "v2"]})
        self.assertEqual(self._served(CSS360D), ("v2", "published"))

        response = self._activate(CSS360D, "v1")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["previousVersion"], "v2")
        versions = self.registries[CSS360D]["versions"]
        self.assertEqual(versions["v1"]["deployment"], "online")
        self.assertEqual(versions["v2"]["deployment"], "offline")
        self.assertEqual(self._served(CSS360D), ("v1", "published"))
        self.assertEqual(self.audit[0]["detail"], {"version": "v1", "previousVersion": "v2"})

    def test_rollback_is_refused_when_the_older_version_is_no_longer_mapped(self) -> None:
        self._seed(
            CSS360D, "v2", _version("v1"), _version("v2", deployment="online")
        )
        self.health = _health(**{CSS360D: ["v2"]})

        response = self._activate(CSS360D, "v1")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(self._served(CSS360D), ("v2", "published"))


class RegistrationDoesNotMoveTrafficTests(ActivationTestCase):
    """Registering v2 while v1 is active leaves students on v1 until v2 is activated."""

    def setUp(self) -> None:
        super().setUp()
        env = patch.dict("os.environ", {"TRAINING_WORKER_TOKEN": TOKEN})
        env.start()
        self.addCleanup(env.stop)

    def _upsert(self, connection: Any, course_id: str, record: dict[str, Any], *, set_current: bool = False):
        registry = self.registries[course_id]
        registry["versions"][record["version"]] = dict(record)
        if set_current:
            registry["currentVersion"] = record["version"]
        return copy.deepcopy(registry)

    def _register(self, course_id: str, body: dict[str, Any]):
        with (
            patch("app.training_queue_routes.db_connection", _fake_connection),
            patch("app.training_queue_routes.db_courses.course_exists", return_value=True),
            patch(
                "app.training_queue_routes.db_models.list_model_versions",
                side_effect=lambda connection, cid: list(self.registries[cid]["versions"].values()),
            ),
            patch(
                "app.training_queue_routes.db_models.upsert_model_version",
                side_effect=self._upsert,
            ),
            patch(
                "app.training_queue_routes.db_model_requests.update_model_request",
                return_value={"status": "ready"},
            ),
        ):
            return self.client.post(
                f"/api/training-queue/courses/{course_id}/model-versions",
                json=body,
                headers=WORKER_HEADERS,
            )

    BODY = {
        "baseModel": "meta-llama/Llama-3.2-3B-Instruct",
        "trainingExampleCount": 60,
        "artifactRef": f"qlora-runs/{CSS360D}/run-2-full/adapter",
        "status": "ready",
    }

    def test_registering_v2_keeps_v1_serving_until_v2_is_activated(self) -> None:
        self._seed(CSS360D, "v1", _version("v1"))
        self.health = _health(**{CSS360D: ["v1"]})
        self.assertEqual(self._activate(CSS360D, "v1").status_code, 200)

        response = self._register(CSS360D, self.BODY)

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["version"], "v2")
        self.assertEqual(self.registries[CSS360D]["currentVersion"], "v2")
        self.assertEqual(self.registries[CSS360D]["versions"]["v2"]["deployment"], "offline")
        self.assertEqual(self._served(CSS360D), ("v1", "published"))

        # Not installed on the VM yet: activation is refused, v1 keeps serving.
        self.assertEqual(self._activate(CSS360D, "v2").status_code, 409)
        self.assertEqual(self._served(CSS360D), ("v1", "published"))

        # Installed and mapped: activation moves the answer, and v1 can come back.
        self.health = _health(**{CSS360D: ["v1", "v2"]})
        self.assertEqual(self._activate(CSS360D, "v2").status_code, 200)
        self.assertEqual(self._served(CSS360D), ("v2", "published"))
        self.assertEqual(self._activate(CSS360D, "v1").status_code, 200)
        self.assertEqual(self._served(CSS360D), ("v1", "published"))

    def test_registration_cannot_publish(self) -> None:
        self._seed(CSS360D, "v1", _version("v1", deployment="online"))

        response = self._register(CSS360D, {**self.BODY, "deployment": "online"})

        self.assertEqual(response.status_code, 422)
        self.assertIn("activate it from Admin", response.json()["detail"])
        self.assertNotIn("v2", self.registries[CSS360D]["versions"])
        self.assertEqual(self._served(CSS360D), ("v1", "published"))

    def test_the_current_version_fallback_is_preserved_for_now(self) -> None:
        """Rollout step 6: an un-activated course still answers from current_version.

        Deliberately kept until CSS 360D and CSS 360E have been activated; the
        fallback is removed in a later change, which will flip this test.
        """
        self._seed(CSS360E, "v1", _version("v1"))

        self.assertEqual(self._served(CSS360E), ("v1", "current"))


if __name__ == "__main__":
    unittest.main()
