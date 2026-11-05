"""Does the fine-tuned service serve every course's activated version? (audit B7)

`assess` is pure: registered courses in, the service's `/health` in, one row
per course out with a problem code when students would be failed. The monitor
logs when that picture changes and stays quiet otherwise. The route is for
administrators only.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_db_repositories import FakeConnection  # noqa: E402

from app import db_models, finetuned_readiness as fr  # noqa: E402
from app.auth.dependencies import current_principal  # noqa: E402
from app.auth.principal import Principal, StaffUser  # noqa: E402
from app.main import app  # noqa: E402


def admin() -> Principal:
    return Principal(
        user=StaffUser(user_id="u1", email="a@uw.edu", display_name="A", role="admin", session_id="s")
    )

D = "css360d-fall-2026-q0ne"
E = "css360e-autumn-2026-c08m"


def version(key: str, *, status: str = "ready", deployment: str = "offline", created: str = "2026-10-01") -> dict[str, Any]:
    return {"version": key, "status": status, "deployment": deployment, "createdAt": created}


def course(course_id: str, *versions: dict[str, Any]) -> dict[str, Any]:
    return {"courseId": course_id, "courseName": course_id.upper(), "versions": list(versions)}


def health(**servable: list[str]) -> dict[str, Any]:
    return {
        "status": "ok",
        "adapterLoaded": True,
        "courses": [{"courseId": c, "versions": v} for c, v in servable.items()],
    }


class AssessTests(unittest.TestCase):
    def test_a_servable_activated_version_is_ok(self) -> None:
        result = fr.assess(
            [course(D, version("v1", deployment="online"))], health(**{D: ["v1"]})
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["problems"], [])
        row = result["courses"][0]
        self.assertEqual(row["activatedVersion"], "v1")
        self.assertIsNone(row["problem"])

    def test_an_activated_version_the_service_does_not_list_is_a_problem(self) -> None:
        result = fr.assess(
            [course(D, version("v2", deployment="online", created="2026-10-02"), version("v1"))],
            health(**{D: ["v1"]}),
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["problems"][0]["problem"], fr.NOT_SERVABLE)
        self.assertIn("v2", result["problems"][0]["message"])
        self.assertIn("v1", result["problems"][0]["message"])

    def test_an_unreachable_service_is_reported_once_per_activated_course(self) -> None:
        result = fr.assess(
            [course(D, version("v1", deployment="online")), course(E, version("v1"))],
            None,
            health_error="connection refused",
            health_error_code="finetuned_unavailable",
        )
        self.assertFalse(result["ok"])
        self.assertFalse(result["service"]["reachable"])
        self.assertEqual(result["service"]["errorCode"], "finetuned_unavailable")
        problems = {item["courseId"]: item["problem"] for item in result["problems"]}
        self.assertEqual(problems, {D: fr.SERVICE_UNREACHABLE})

    def test_no_activated_version_is_reported_but_not_alarmed(self) -> None:
        result = fr.assess([course(E, version("v1"))], health(**{E: ["v1"]}))
        self.assertTrue(result["ok"])
        self.assertEqual(result["courses"][0]["problem"], fr.NO_ACTIVATED_VERSION)
        self.assertEqual(result["problems"], [])

    def test_the_newest_registered_version_travels_beside_the_activated_one(self) -> None:
        result = fr.assess(
            [course(D, version("v3", status="training", created="2026-10-09"), version("v2", deployment="online"))],
            health(**{D: ["v2"]}),
        )
        row = result["courses"][0]
        self.assertEqual((row["activatedVersion"], row["newestVersion"], row["newestStatus"]), ("v2", "v3", "training"))

    def test_an_unready_online_row_is_not_activated(self) -> None:
        result = fr.assess([course(D, version("v1", status="training", deployment="online"))], health(**{D: ["v1"]}))
        self.assertEqual(result["courses"][0]["problem"], fr.NO_ACTIVATED_VERSION)


class CollectTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_failing_probe_becomes_service_unreachable(self) -> None:
        probe = AsyncMock(side_effect=HTTPException(status_code=503, detail="down"))
        result = await fr.collect_readiness(
            probe=probe, read_courses=lambda: [course(D, version("v1", deployment="online"))]
        )
        self.assertEqual(result["problems"][0]["problem"], fr.SERVICE_UNREACHABLE)
        self.assertEqual(result["service"]["error"], "down")
        self.assertIn(result["mode"], ("vm", "gpu"))


class MonitorTests(unittest.IsolatedAsyncioTestCase):
    async def test_logs_a_warning_when_a_problem_appears_and_info_when_it_clears(self) -> None:
        pictures = [
            fr.assess([course(D, version("v1", deployment="online"))], health(**{D: ["v1"]})),
            fr.assess([course(D, version("v1", deployment="online"))], health(**{D: []})),
            fr.assess([course(D, version("v1", deployment="online"))], health(**{D: []})),
            fr.assess([course(D, version("v1", deployment="online"))], health(**{D: ["v1"]})),
        ]
        collect = AsyncMock(side_effect=pictures)
        slept: list[float] = []

        async def sleep(seconds: float) -> None:
            slept.append(seconds)

        with self.assertLogs("app.finetuned_readiness", level=logging.INFO) as logs:
            await fr.readiness_monitor(15.0, collect=collect, sleep=sleep, iterations=4)

        warnings = [line for line in logs.output if line.startswith("WARNING")]
        infos = [line for line in logs.output if line.startswith("INFO")]
        self.assertEqual(len(warnings), 1)  # the unchanged second check is silent
        self.assertIn(D, warnings[0])
        self.assertIn("activated_version_not_servable", warnings[0])
        self.assertEqual(len(infos), 1)
        self.assertIn("servable again", infos[0])
        self.assertEqual(slept, [15.0, 15.0, 15.0])

    async def test_a_check_that_raises_does_not_end_the_monitor(self) -> None:
        collect = AsyncMock(side_effect=[RuntimeError("db down"), fr.assess([], health())])

        async def sleep(_seconds: float) -> None:
            return None

        with self.assertLogs("app.finetuned_readiness", level=logging.WARNING) as logs:
            await fr.readiness_monitor(1.0, collect=collect, sleep=sleep, iterations=2)
        self.assertIn("could not run", logs.output[0])
        self.assertEqual(collect.await_count, 2)

    async def test_a_long_outage_of_the_check_itself_is_logged_once_and_its_end_once(self) -> None:
        """A database outage must not write a WARNING every interval for hours."""
        collect = AsyncMock(
            side_effect=[
                RuntimeError("db down"),
                RuntimeError("db down"),
                RuntimeError("db down"),
                fr.assess([], health()),
                fr.assess([], health()),
            ]
        )

        async def sleep(_seconds: float) -> None:
            return None

        with self.assertLogs("app.finetuned_readiness", level=logging.INFO) as logs:
            await fr.readiness_monitor(1.0, collect=collect, sleep=sleep, iterations=5)
        warnings = [line for line in logs.output if line.startswith("WARNING")]
        infos = [line for line in logs.output if line.startswith("INFO")]
        self.assertEqual(len(warnings), 1)
        self.assertIn("db down", warnings[0])
        self.assertEqual(len(infos), 1)
        self.assertIn("running again", infos[0])

    def test_the_monitor_does_not_start_under_tests(self) -> None:
        self.assertIsNone(fr.start_readiness_monitor())

    def test_the_interval_comes_from_the_environment(self) -> None:
        with patch.dict("os.environ", {fr.CHECK_SECONDS_ENV: "45"}):
            self.assertEqual(fr.readiness_check_interval(), 45.0)
        with patch.dict("os.environ", {fr.CHECK_SECONDS_ENV: "nonsense"}):
            self.assertEqual(fr.readiness_check_interval(), fr.DEFAULT_CHECK_SECONDS)
        with patch.dict("os.environ", {fr.CHECK_SECONDS_ENV: "0"}):
            self.assertEqual(fr.readiness_check_interval(), 0.0)


class RegistryReadTests(unittest.TestCase):
    def test_versions_are_grouped_by_course_newest_first(self) -> None:
        rows = [
            {"course_id": D, "course_name": "CSS 360D", "version": "v2", "base_model": "b", "training_example_count": 10,
             "status": "ready", "deployment": "online", "artifact_ref": "a", "created_at": "2026-10-02", "updated_at": None,
             "notes": None, "run_id": None, "provenance": None},
            {"course_id": D, "course_name": "CSS 360D", "version": "v1", "base_model": "b", "training_example_count": 9,
             "status": "ready", "deployment": "offline", "artifact_ref": "a", "created_at": "2026-10-01", "updated_at": None,
             "notes": None, "run_id": None, "provenance": None},
            {"course_id": E, "course_name": None, "version": "v1", "base_model": "b", "training_example_count": 9,
             "status": "ready", "deployment": "online", "artifact_ref": "a", "created_at": "2026-10-01", "updated_at": None,
             "notes": None, "run_id": None, "provenance": None},
        ]
        connection = FakeConnection([rows])
        courses = db_models.list_registered_courses(connection)
        self.assertEqual([c["courseId"] for c in courses], [D, E])
        self.assertEqual([v["version"] for v in courses[0]["versions"]], ["v2", "v1"])
        self.assertEqual(courses[0]["courseName"], "CSS 360D")
        self.assertIsNone(courses[1]["courseName"])
        self.assertIn("LEFT JOIN courses", connection.sql[0])


class RouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_administrators_only(self) -> None:
        app.dependency_overrides.pop(current_principal, None)
        response = self.client.get("/api/admin/model-readiness")
        self.assertEqual(response.status_code, 401, response.text)

    def test_the_route_returns_the_assessment(self) -> None:
        picture = fr.assess([course(D, version("v1", deployment="online"))], health(**{D: ["v1"]}))
        app.dependency_overrides[current_principal] = admin
        self.addCleanup(lambda: app.dependency_overrides.pop(current_principal, None))
        with patch("app.main.collect_readiness", new=AsyncMock(return_value=picture)):
            response = self.client.get("/api/admin/model-readiness")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["ok"])
        self.assertEqual(response.json()["courses"][0]["courseId"], D)


if __name__ == "__main__":
    unittest.main()
