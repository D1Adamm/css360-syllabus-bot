"""No route ships without an authorization class.

`tests/route_classification.py` names every route and who may call it. This
test holds the application to that table in both directions — a route the
table does not know fails, and a table entry the application no longer mounts
fails — and checks that the guard each class requires is really declared on
the route, as a dependency FastAPI resolves before the handler runs.

It is deliberately static. The behavioural half, driving each route with
each kind of principal, is `test_authorization_matrix.py`.
"""

from __future__ import annotations

import unittest

import pytest

from app.main import app
from route_classification import (
    CLASSIFICATION,
    GUARD_FOR_CLASS,
    PUBLIC,
    guard_names,
    iter_api_routes,
)

pytestmark = pytest.mark.auth

#: Routes FastAPI adds itself. Documentation only; disabled outside development.
FRAMEWORK_PATHS = frozenset({"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"})


def _mounted() -> dict[tuple[str, str], object]:
    mounted: dict[tuple[str, str], object] = {}
    for route in iter_api_routes(app):
        for method in sorted(route.methods or ()):
            if method == "HEAD":
                continue
            mounted[(method, route.path)] = route
    return mounted


class RouteClassificationCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mounted = _mounted()

    def test_every_mounted_route_is_classified(self) -> None:
        unclassified = sorted(key for key in self.mounted if key not in CLASSIFICATION)
        self.assertEqual(
            unclassified,
            [],
            "These routes are mounted but not classified. Add each one to "
            "tests/route_classification.py with the guard it must declare: "
            f"{unclassified}",
        )

    def test_every_classified_route_is_mounted(self) -> None:
        stale = sorted(key for key in CLASSIFICATION if key not in self.mounted)
        self.assertEqual(stale, [], f"Classified but no longer mounted: {stale}")

    def test_every_route_declares_the_guard_its_class_requires(self) -> None:
        for key, route in sorted(self.mounted.items()):
            cls = CLASSIFICATION[key]
            required = GUARD_FOR_CLASS[cls]
            with self.subTest(method=key[0], path=key[1], cls=cls):
                if required is None:
                    continue
                self.assertIn(
                    required,
                    guard_names(route),
                    f"{key[0]} {key[1]} is classified {cls} but does not depend on {required}",
                )

    def test_public_routes_are_exactly_the_intended_few(self) -> None:
        """The allowlist is short and every entry is deliberate."""
        public = sorted(key for key, cls in CLASSIFICATION.items() if cls == PUBLIC)
        self.assertEqual(
            public,
            [
                ("GET", "/api/auth/invitations/{token}"),
                ("GET", "/api/health"),
                ("GET", "/health"),
                ("POST", "/api/auth/invitations/{token}/accept"),
                ("POST", "/api/auth/login"),
            ],
        )

    def test_public_session_establishing_posts_require_the_csrf_header(self) -> None:
        for key in (("POST", "/api/auth/login"), ("POST", "/api/auth/invitations/{token}/accept")):
            with self.subTest(route=key):
                self.assertIn("require_csrf", guard_names(self.mounted[key]))

    def test_the_documentation_routes_are_not_application_routes(self) -> None:
        paths = {path for _, path in self.mounted}
        self.assertTrue(paths.isdisjoint(FRAMEWORK_PATHS))


if __name__ == "__main__":
    unittest.main()
