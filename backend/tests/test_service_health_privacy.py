"""`GET /api/fine-tuned/health` must not describe how to reach the cluster.

The leak these cover: the route copied `hostname`, `port` and `serviceUrl`
straight out of the health probe. `hostname` is whichever Tillicum compute node
a Slurm allocation landed on, `port` is what the service listens on, and
`serviceUrl` is the SSH tunnel destination this backend dials. The route needs
no credential, so all three were public.

This is the rule `db_serving_sessions.public_serving_session` already applies to
a serving session's node and port; the health endpoint was the one place that
still published the same class of fact.

Both halves are asserted: the probe still returns the exact values to the
backend, and the response built from it carries none of them.

Fixture values are synthetic (`n3129.hyak.local`, port 8412, a localhost tunnel).
Nothing here contacts a service.
"""

from __future__ import annotations

import json
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.finetuned_client import public_service_health
from app.main import app

COURSE = "css-350-spring-2026-n3h9"

#: What a running service reports about where it is. None of it may be public.
NODE_HOSTNAME = "n3129.hyak.local"
SERVICE_PORT = 8412
SERVICE_URL = "http://localhost:8412"
SERVING_ROOT = "/gpfs/projects/simswe/testuser/training_outputs/serving"

#: Exactly what `check_finetuned_service_health` hands the route.
PROBE_RESULT: dict[str, Any] = {
    "status": "ok",
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "adapterLoaded": True,
    "hostname": NODE_HOSTNAME,
    "port": SERVICE_PORT,
    "serviceUrl": SERVICE_URL,
    "courses": [
        {
            "courseId": COURSE,
            "versions": ["v1", "v2"],
            "currentVersion": "v2",
            # A field a future build of the service could add. It must not ride
            # along into a public response just because it was in the payload.
            "servingRoot": "{0}/{1}".format(SERVING_ROOT, COURSE),
        }
    ],
    "secondsRemaining": 4200.0,
}


def assert_no_topology(case: unittest.TestCase, body: Any) -> None:
    """No node name, port, tunnel URL or cluster path anywhere in the response."""
    text = json.dumps(body)
    for secret in (
        NODE_HOSTNAME,
        "n3129",
        "hyak",
        SERVICE_URL,
        "localhost",
        "127.0.0.1",
        SERVING_ROOT,
        "/gpfs/projects/simswe/",
        "testuser",
    ):
        case.assertNotIn(secret, text)
    case.assertNotIn(str(SERVICE_PORT), text)
    for key in ("hostname", "port", "serviceUrl", "servingRoot"):
        case.assertNotIn(key, text)


class PublicServiceHealthViewTests(unittest.TestCase):
    def test_the_three_topology_fields_are_dropped(self) -> None:
        public = public_service_health(PROBE_RESULT)
        self.assertNotIn("hostname", public)
        self.assertNotIn("port", public)
        self.assertNotIn("serviceUrl", public)

    def test_the_status_a_page_is_for_survives(self) -> None:
        public = public_service_health(PROBE_RESULT)
        self.assertEqual(public["status"], "ok")
        self.assertEqual(public["model"], "meta-llama/Llama-3.2-3B-Instruct")
        self.assertTrue(public["adapterLoaded"])
        self.assertEqual(public["secondsRemaining"], 4200.0)

    def test_courses_keep_their_logical_identifiers(self) -> None:
        course = public_service_health(PROBE_RESULT)["courses"][0]
        self.assertEqual(course["courseId"], COURSE)
        self.assertEqual(course["versions"], ["v1", "v2"])
        self.assertEqual(course["currentVersion"], "v2")

    def test_a_course_entry_is_rebuilt_not_forwarded(self) -> None:
        """A key the remote service adds cannot widen this response."""
        course = public_service_health(PROBE_RESULT)["courses"][0]
        self.assertEqual(set(course), {"courseId", "versions", "currentVersion"})
        self.assertNotIn("servingRoot", course)

    def test_a_service_that_reports_no_courses_is_an_empty_list(self) -> None:
        """An older single-adapter build, which reports no `courses` at all."""
        public = public_service_health(
            {"status": "ok", "model": "m", "adapterLoaded": True}
        )
        self.assertEqual(public["courses"], [])
        self.assertIsNone(public["secondsRemaining"])

    def test_a_malformed_courses_value_does_not_leak_or_raise(self) -> None:
        for courses in ("not-a-list", None, [SERVICE_URL], [{"courseId": COURSE}]):
            with self.subTest(courses=courses):
                public = public_service_health({"status": "ok", "courses": courses})
                assert_no_topology(self, public)

    def test_an_unreported_status_is_unknown_not_missing(self) -> None:
        self.assertEqual(public_service_health({})["status"], "unknown")

    def test_the_probe_result_is_not_mutated(self) -> None:
        """A read-side view. The backend's own copy keeps every field."""
        public_service_health(PROBE_RESULT)
        self.assertEqual(PROBE_RESULT["hostname"], NODE_HOSTNAME)
        self.assertEqual(PROBE_RESULT["port"], SERVICE_PORT)
        self.assertEqual(PROBE_RESULT["serviceUrl"], SERVICE_URL)


class FineTunedHealthRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def _get(self, path: str) -> Any:
        patcher = patch(
            "app.main.check_finetuned_service_health",
            new=AsyncMock(return_value=dict(PROBE_RESULT)),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        response = self.client.get(path)
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_the_public_response_carries_no_topology(self) -> None:
        assert_no_topology(self, self._get("/api/fine-tuned/health"))

    def test_the_unprefixed_alias_is_the_same_response(self) -> None:
        """Both paths are mounted; a fix on one is not a fix."""
        assert_no_topology(self, self._get("/fine-tuned/health"))

    def test_the_response_is_still_worth_reading(self) -> None:
        body = self._get("/api/fine-tuned/health")
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["model"], "meta-llama/Llama-3.2-3B-Instruct")
        self.assertTrue(body["adapterLoaded"])
        self.assertEqual(body["secondsRemaining"], 4200.0)
        self.assertEqual(body["courses"][0]["courseId"], COURSE)
        self.assertEqual(body["courses"][0]["currentVersion"], "v2")

    def test_the_response_model_refuses_to_carry_them_back(self) -> None:
        """Even a caller that passes them in gets a response without them.

        The fields are gone from `FineTunedHealthResponse`, not merely unset by
        the route, so a future edit that reintroduces the copy still cannot
        publish them.
        """
        body = self._get("/api/fine-tuned/health")
        self.assertEqual(
            set(body),
            {"status", "model", "adapterLoaded", "courses", "secondsRemaining"},
        )


class InternalProbeStaysExactTests(unittest.IsolatedAsyncioTestCase):
    """The other half of the rule: the backend still knows where the service is."""

    async def test_the_probe_returns_the_exact_host_port_and_url(self) -> None:
        from app.finetuned_client import check_finetuned_service_health

        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "status": "ok",
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "adapterLoaded": True,
            "hostname": NODE_HOSTNAME,
            "port": SERVICE_PORT,
            "courses": [],
        }

        client = MagicMock()
        client.get = AsyncMock(return_value=response)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch.dict("os.environ", {"FINETUNED_SERVICE_URL": SERVICE_URL}):
            with patch("app.finetuned_client.httpx.AsyncClient", return_value=client):
                result = await check_finetuned_service_health()

        # Unchanged: this is what the backend dials and what an operator needs.
        self.assertEqual(result["hostname"], NODE_HOSTNAME)
        self.assertEqual(result["port"], SERVICE_PORT)
        self.assertEqual(result["serviceUrl"], SERVICE_URL)
        client.get.assert_awaited_once_with("{0}/health".format(SERVICE_URL))

        # And the same record, seen from the public side.
        assert_no_topology(self, public_service_health(result))


#: An upstream body of the kind that made forwarding it unsafe. Whatever
#: answered — the service, the SSH tunnel, or a proxy in front of either — can
#: name the node, the tunnel port, and a serving root under /gpfs.
POISONED_UPSTREAM_BODY = (
    "502 Bad Gateway: upstream {0}:{1} refused the connection while proxying "
    "from localhost:8412; adapter root {2}/{3} was not readable"
).format(NODE_HOSTNAME, SERVICE_PORT, SERVING_ROOT, COURSE)


def _upstream(status_code: int, body: str = POISONED_UPSTREAM_BODY) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.text = body
    response.json.side_effect = ValueError("not json")
    return response


def _async_client(response: MagicMock) -> MagicMock:
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    client.post = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


class UpstreamErrorBodyTests(unittest.IsolatedAsyncioTestCase):
    """A failing upstream must not narrate the cluster to the browser.

    Every one of these four branches used to append `response.text[:200]` to the
    public detail. The body is written by whatever answered, so it is exactly
    the wrong thing to forward and exactly the right thing to log.
    """

    def setUp(self) -> None:
        self.client = TestClient(app)
        env = patch.dict("os.environ", {"FINETUNED_SERVICE_URL": SERVICE_URL})
        env.start()
        self.addCleanup(env.stop)

    def assert_body_absent(self, detail: Any) -> None:
        text = json.dumps(detail)
        for secret in (
            NODE_HOSTNAME,
            "n3129",
            "hyak",
            "localhost:8412",
            str(SERVICE_PORT),
            SERVING_ROOT,
            "/gpfs/projects/simswe/",
            "testuser",
            "Bad Gateway",
            "refused the connection",
        ):
            self.assertNotIn(secret, text)

    # -- the public HTTP surface -------------------------------------------- #

    def test_the_health_route_returns_the_status_code_and_nothing_else(self) -> None:
        with patch(
            "app.finetuned_client.httpx.AsyncClient",
            return_value=_async_client(_upstream(502)),
        ):
            response = self.client.get("/api/fine-tuned/health")

        self.assertEqual(response.status_code, 503)
        body = response.json()
        self.assert_body_absent(body)
        assert_no_topology(self, body)
        # Still says what happened and what the upstream code was.
        self.assertIn("health check failed", body["detail"])
        self.assertIn("HTTP 502", body["detail"])

    def test_the_generate_route_returns_the_status_code_and_nothing_else(self) -> None:
        with patch(
            "app.main.resolve_current_course_model",
            return_value={"version": "v2", "courseId": COURSE},
        ), patch(
            "app.finetuned_client.httpx.AsyncClient",
            return_value=_async_client(_upstream(500)),
        ):
            response = self.client.post(
                "/api/fine-tuned/generate",
                json={"question": "What is the late policy?", "courseId": COURSE},
            )

        self.assertEqual(response.status_code, 503)
        body = response.json()
        self.assert_body_absent(body)
        assert_no_topology(self, body)
        self.assertIn("server error", body["detail"].lower())
        self.assertIn("HTTP 500", body["detail"])

    # -- the two remaining client branches ---------------------------------- #

    async def test_a_rejected_request_does_not_quote_the_upstream(self) -> None:
        from app.finetuned_client import generate_finetuned_response

        with patch(
            "app.finetuned_client.httpx.AsyncClient",
            return_value=_async_client(_upstream(400)),
        ):
            with self.assertRaises(HTTPException) as caught:
                await generate_finetuned_response("q", course_id=COURSE)

        self.assertEqual(caught.exception.status_code, 502)
        self.assert_body_absent(caught.exception.detail)
        self.assertIn("HTTP 400", caught.exception.detail)

    async def test_an_unpublished_course_still_says_which_course(self) -> None:
        """The 409 keeps the fact a caller can act on, drops the quoted body."""
        from app.finetuned_client import generate_finetuned_response

        with patch(
            "app.finetuned_client.httpx.AsyncClient",
            return_value=_async_client(_upstream(409)),
        ):
            with self.assertRaises(HTTPException) as caught:
                await generate_finetuned_response("q", course_id=COURSE)

        self.assertEqual(caught.exception.status_code, 409)
        self.assert_body_absent(caught.exception.detail)
        self.assertIn(COURSE, caught.exception.detail)
        self.assertIn("Publish one", caught.exception.detail)

    # -- the diagnostic that replaced it ------------------------------------ #

    async def test_the_whole_body_is_kept_in_the_backend_log(self) -> None:
        """Removed from the response, not from the operator's reach."""
        from app.finetuned_client import check_finetuned_service_health

        with patch(
            "app.finetuned_client.httpx.AsyncClient",
            return_value=_async_client(_upstream(502)),
        ):
            with self.assertLogs("app.finetuned_client", level="WARNING") as logs:
                with self.assertRaises(HTTPException):
                    await check_finetuned_service_health()

        logged = "\n".join(logs.output)
        self.assertIn(POISONED_UPSTREAM_BODY, logged)
        self.assertIn(SERVICE_URL, logged)
        self.assertIn("502", logged)

    async def test_a_long_body_is_logged_well_past_the_old_200_characters(self) -> None:
        from app.finetuned_client import check_finetuned_service_health
        from app.upstream_errors import UPSTREAM_LOG_BODY_LIMIT

        body = "x" * 1500 + POISONED_UPSTREAM_BODY
        self.assertLess(len(body), UPSTREAM_LOG_BODY_LIMIT)

        with patch(
            "app.finetuned_client.httpx.AsyncClient",
            return_value=_async_client(_upstream(500, body)),
        ):
            with self.assertLogs("app.finetuned_client", level="WARNING") as logs:
                with self.assertRaises(HTTPException):
                    await check_finetuned_service_health()

        self.assertIn(POISONED_UPSTREAM_BODY, "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
