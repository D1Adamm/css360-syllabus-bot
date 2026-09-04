"""Every application route is reachable through the deployed `/api/` proxy.

The deployed Nginx forwards `location /api/` to uvicorn and nothing else, and
the frontend composes every request below `VITE_API_BASE_URL`, which carries
that prefix. So a route mounted anywhere other than `/api/...` is reachable on
the VM with curl and unreachable from every browser — which is exactly how
`/health` and `/rag/generate` failed in production before the aliases existed.

The frontend side of this contract is pinned by `dbApi.paths.test.ts` and
`operationalApi.paths.test.ts`, which assert the composed URLs against route
strings copied from this app. This is the backend side: the routes those
strings name really are under `/api/`, and nothing new has been mounted outside
it since.

The six root-level paths are deliberate. The tunnel scripts, `systemd` health
checks and on-VM `curl localhost:8001/health` use them, and each one must keep
an `/api` twin served by the same handler so the two never drift apart.
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.main import app

#: The only paths served outside `/api/`. Each is an alias of an `/api` route,
#: kept for callers on the VM itself. Anything added here must be added to the
#: README's route documentation too, and must never be relied on by a browser.
ROOT_ALIASES = frozenset(
    {
        "/health",
        "/base-model/generate",
        "/rag/generate",
        "/fine-tuned/health",
        "/fine-tuned/generate",
        "/fine-tuned-rag/generate",
    }
)


class ApiRouteShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        # The OpenAPI document lists application routes only — FastAPI's own
        # /docs, /redoc and /openapi.json are excluded from it.
        self.spec = TestClient(app).get("/openapi.json").json()["paths"]

    def test_every_application_route_is_under_api(self) -> None:
        outside = sorted(
            path
            for path in self.spec
            if not path.startswith("/api/") and path not in ROOT_ALIASES
        )
        self.assertEqual(
            outside,
            [],
            "These routes are mounted outside /api/ and cannot be reached "
            "through the deployed Nginx proxy: {0}".format(outside),
        )

    def test_the_root_aliases_are_exactly_the_documented_six(self) -> None:
        roots = {path for path in self.spec if not path.startswith("/api/")}
        self.assertEqual(roots, set(ROOT_ALIASES))

    def test_each_root_alias_has_an_api_twin_served_by_the_same_handler(self) -> None:
        endpoints_by_path: dict[str, set[object]] = {}
        for route in app.routes:
            path = getattr(route, "path", None)
            endpoint = getattr(route, "endpoint", None)
            if path and endpoint is not None:
                endpoints_by_path.setdefault(path, set()).add(endpoint)

        for alias in sorted(ROOT_ALIASES):
            twin = "/api" + alias
            with self.subTest(alias=alias):
                self.assertIn(twin, self.spec, "{0} has no /api twin".format(alias))
                self.assertEqual(
                    set(self.spec[alias]),
                    set(self.spec[twin]),
                    "{0} and {1} accept different methods".format(alias, twin),
                )
                self.assertEqual(
                    endpoints_by_path[alias],
                    endpoints_by_path[twin],
                    "{0} and {1} are served by different handlers".format(alias, twin),
                )

    def test_the_browser_facing_groups_are_all_present_under_api(self) -> None:
        """The three route families the frontend clients compose paths for."""
        families = {
            "/api/db/": "dbApi.ts persistence routes",
            "/api/courses/": "api.ts / adminApi.ts operational routes",
            "/api/training-queue/": "the worker-token queue routes",
        }
        for prefix, description in families.items():
            with self.subTest(prefix=prefix):
                self.assertTrue(
                    any(path.startswith(prefix) for path in self.spec),
                    "No route under {0} ({1})".format(prefix, description),
                )


if __name__ == "__main__":
    unittest.main()
