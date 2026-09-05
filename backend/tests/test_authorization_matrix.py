"""Every route, every kind of principal, the outcome the classification implies.

Six principals — nobody, a participant in course A, a participant in course
B, a professor assigned to A, a professor assigned to B, an administrator —
are driven through every route the application mounts, with course A named
in the path or body. For each, the classification in
`tests/route_classification.py` says whether the guard must refuse (401 when
nobody is signed in, 403 when someone is signed in but not allowed) or must
let the request through to the handler.

"Through to the handler" is asserted as "not 401 and not 403". The handlers
themselves run against no database and with every model call stubbed, so
they answer 503, 404 or 422 — which is fine: the guard is what is under test,
and it is the only thing that can produce 401 or 403.

The worker routes are checked separately: no browser principal reaches them.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.auth.dependencies import current_principal
from app.auth.principal import ANONYMOUS, Participant, Principal, StaffUser
from app.auth.settings import CSRF_HEADER_VALUE
from app.auth.tokens import generate_token
from app.main import app
from route_classification import (
    CLASSIFICATION,
    COURSE_ACCESS_BODY,
    PUBLIC,
    REQUIRE_ADMIN,
    REQUIRE_COURSE_ACCESS,
    REQUIRE_COURSE_STAFF,
    REQUIRE_PARTICIPANT,
    REQUIRE_USER,
    SESSION_OPTIONAL,
    SIGNED_IN,
    WORKER,
    iter_api_routes,
)

pytestmark = pytest.mark.auth

COURSE_A = "css-360-winter-2026-a7rp"
COURSE_B = "css-350-spring-2026-n3h9"
CSRF = {"X-Requested-With": CSRF_HEADER_VALUE}

OK = "ok"  # anything but 401 or 403
UNAUTHENTICATED = 401
FORBIDDEN = 403


def _staff(role: str, *course_ids: str) -> Principal:
    return Principal(
        user=StaffUser(
            user_id=f"user-{role}-{'-'.join(course_ids) or 'all'}",
            email=f"{role}@uw.edu",
            display_name=role.title(),
            role=role,
            course_ids=frozenset(course_ids),
            session_id="session",
        )
    )


def _participant(course_id: str) -> Principal:
    return Principal(
        participant=Participant(
            participant_id=f"participant-{course_id}", course_id=course_id, session_id="session"
        )
    )


PRINCIPALS: dict[str, Principal] = {
    "anonymous": ANONYMOUS,
    "participant_a": _participant(COURSE_A),
    "participant_b": _participant(COURSE_B),
    "professor_a": _staff("professor", COURSE_A),
    "professor_b": _staff("professor", COURSE_B),
    "admin": _staff("admin"),
}

#: class -> principal -> expected outcome, with course A in the request.
EXPECTATIONS: dict[str, dict[str, Any]] = {
    PUBLIC: dict.fromkeys(PRINCIPALS, OK),
    SESSION_OPTIONAL: dict.fromkeys(PRINCIPALS, OK),
    SIGNED_IN: {**dict.fromkeys(PRINCIPALS, OK), "anonymous": UNAUTHENTICATED},
    REQUIRE_USER: {
        "anonymous": UNAUTHENTICATED,
        "participant_a": UNAUTHENTICATED,
        "participant_b": UNAUTHENTICATED,
        "professor_a": OK,
        "professor_b": OK,
        "admin": OK,
    },
    REQUIRE_ADMIN: {
        "anonymous": UNAUTHENTICATED,
        "participant_a": UNAUTHENTICATED,
        "participant_b": UNAUTHENTICATED,
        "professor_a": FORBIDDEN,
        "professor_b": FORBIDDEN,
        "admin": OK,
    },
    REQUIRE_COURSE_STAFF: {
        "anonymous": UNAUTHENTICATED,
        "participant_a": UNAUTHENTICATED,
        "participant_b": UNAUTHENTICATED,
        "professor_a": OK,
        "professor_b": FORBIDDEN,
        "admin": OK,
    },
    REQUIRE_COURSE_ACCESS: {
        "anonymous": UNAUTHENTICATED,
        "participant_a": OK,
        "participant_b": FORBIDDEN,
        "professor_a": OK,
        "professor_b": FORBIDDEN,
        "admin": OK,
    },
    COURSE_ACCESS_BODY: {
        "anonymous": UNAUTHENTICATED,
        "participant_a": OK,
        "participant_b": FORBIDDEN,
        "professor_a": OK,
        "professor_b": FORBIDDEN,
        "admin": OK,
    },
    REQUIRE_PARTICIPANT: {
        "anonymous": UNAUTHENTICATED,
        "participant_a": OK,
        "participant_b": FORBIDDEN,
        "professor_a": FORBIDDEN,
        "professor_b": FORBIDDEN,
        "admin": FORBIDDEN,
    },
}

PATH_VALUES = {
    "{course_id}": COURSE_A,
    "{seed_id}": "seed-1",
    "{run_id}": "run-1",
    "{evaluation_id}": "eval-1",
    "{user_id}": "user-1",
    "{invitation_id}": "inv-1",
    "{token}": generate_token(),
    "{version}": "v1",
    "{name}": "train.jsonl",
    "{session_id}": "sess-1",
}

#: Bodies that get past FastAPI's validation for the routes that need one, so
#: that a permitted principal is seen reaching the handler rather than being
#: stopped by a 422 that would look the same for everyone. Anything not listed
#: is sent an empty object.
BODIES: dict[str, dict[str, Any]] = {
    "/base-model/generate": {"courseId": COURSE_A, "question": "When are office hours?"},
    "/api/base-model/generate": {"courseId": COURSE_A, "question": "When are office hours?"},
    "/rag/generate": {"courseId": COURSE_A, "question": "When are office hours?"},
    "/api/rag/generate": {"courseId": COURSE_A, "question": "When are office hours?"},
    "/fine-tuned/generate": {"courseId": COURSE_A, "question": "When are office hours?"},
    "/api/fine-tuned/generate": {"courseId": COURSE_A, "question": "When are office hours?"},
    "/fine-tuned-rag/generate": {"courseId": COURSE_A, "question": "When are office hours?"},
    "/api/fine-tuned-rag/generate": {"courseId": COURSE_A, "question": "When are office hours?"},
    "/api/auth/login": {"email": "x@uw.edu", "password": "a password of some length"},
    "/api/auth/join": {"code": "7K4P9X"},
    "/api/auth/invitations/{token}/accept": {
        "email": "x@uw.edu", "displayName": "X", "password": "a password of some length",
    },
    "/api/auth/password": {"currentPassword": "old old old old", "newPassword": "new new new new new"},
}


def _url(path: str) -> str:
    for placeholder, value in PATH_VALUES.items():
        path = path.replace(placeholder, value)
    return path


class AuthorizationMatrixTests(unittest.TestCase):
    """The expensive part happens once per class, not once per request."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app, base_url="https://testserver")
        cls.routes = sorted(
            (method, route.path)
            for route in iter_api_routes(app)
            for method in (route.methods or ())
            if method != "HEAD"
        )

    def setUp(self) -> None:
        self.addCleanup(app.dependency_overrides.clear)
        # Nothing below may reach a model, the cluster, or the disk.
        stubs = {
            "app.main.generate_base_model_response": AsyncMock(
                side_effect=HTTPException(status_code=503, detail="stub")
            ),
            "app.main.generate_course_rag_answer": AsyncMock(
                side_effect=HTTPException(status_code=503, detail="stub")
            ),
            "app.main.generate_finetuned_response": AsyncMock(
                side_effect=HTTPException(status_code=503, detail="stub")
            ),
            "app.main.generate_course_finetuned_rag_answer": AsyncMock(
                side_effect=HTTPException(status_code=503, detail="stub")
            ),
            "app.main.resolve_current_course_model": MagicMock(return_value={"version": "v1"}),
            "app.main.check_finetuned_service_health": AsyncMock(return_value={"status": "stub"}),
            "app.main.validate_syllabus_upload": AsyncMock(
                side_effect=HTTPException(status_code=422, detail="stub")
            ),
            "app.main.generate_starter_seeds_for_course": AsyncMock(
                side_effect=HTTPException(status_code=503, detail="stub")
            ),
            "app.main.generate_seeds_from_chunk": AsyncMock(
                side_effect=HTTPException(status_code=503, detail="stub")
            ),
            "app.main.get_course_artifact_storage": MagicMock(
                return_value=MagicMock(load_index=MagicMock(return_value=None),
                                       load_extracted_text=MagicMock(return_value=None))
            ),
            "app.main.prepare_training_split": MagicMock(
                side_effect=HTTPException(status_code=503, detail="stub")
            ),
            "app.main.approved_export_status": MagicMock(
                return_value={"courseId": COURSE_A, "exists": False, "exportPath": "",
                              "exampleCount": 0, "sourceFile": ""}
            ),
        }
        for target, replacement in stubs.items():
            patcher = patch(target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _request(self, method: str, path: str, principal_name: str):
        principal = PRINCIPALS[principal_name]
        if principal is ANONYMOUS:
            app.dependency_overrides.pop(current_principal, None)
        else:
            app.dependency_overrides[current_principal] = lambda: principal
        kwargs: dict[str, Any] = {"headers": CSRF}
        if method in ("POST", "PUT", "PATCH"):
            if path.endswith("/syllabus"):
                kwargs["files"] = {"syllabus_file": ("syllabus.txt", b"hello", "text/plain")}
            else:
                kwargs["json"] = BODIES.get(path, {})
        return self.client.request(method, _url(path), **kwargs)

    def test_every_browser_route_against_every_principal(self) -> None:
        for method, path in self.routes:
            cls = CLASSIFICATION[(method, path)]
            if cls == WORKER:
                continue
            for principal_name, expected in EXPECTATIONS[cls].items():
                with self.subTest(method=method, path=path, principal=principal_name):
                    response = self._request(method, path, principal_name)
                    if expected == OK:
                        self.assertNotIn(
                            response.status_code,
                            (401, 403),
                            f"{principal_name} should reach {method} {path} ({cls}) "
                            f"but got {response.status_code}: {response.text[:200]}",
                        )
                    else:
                        self.assertEqual(
                            response.status_code,
                            expected,
                            f"{principal_name} on {method} {path} ({cls}): "
                            f"{response.status_code} {response.text[:200]}",
                        )

    def test_no_browser_principal_reaches_the_worker_routes(self) -> None:
        """Even an administrator session is not the cluster's credential."""
        for method, path in self.routes:
            if CLASSIFICATION[(method, path)] != WORKER:
                continue
            for principal_name in PRINCIPALS:
                with self.subTest(method=method, path=path, principal=principal_name):
                    response = self._request(method, path, principal_name)
                    # 503: the worker token is unconfigured under test, and the
                    # router refuses rather than serving openly.
                    self.assertEqual(response.status_code, 503)

    def test_a_participant_cannot_change_the_course_in_the_body(self) -> None:
        """The four generation routes take the course from JSON; a participant
        of B naming A is refused, and naming B is allowed."""
        for path in ("/api/base-model/generate", "/api/rag/generate",
                     "/api/fine-tuned/generate", "/api/fine-tuned-rag/generate"):
            app.dependency_overrides[current_principal] = lambda: PRINCIPALS["participant_b"]
            with self.subTest(path=path, course=COURSE_A):
                refused = self.client.post(
                    path, json={"courseId": COURSE_A, "question": "q?"}, headers=CSRF
                )
                self.assertEqual(refused.status_code, 403)
            with self.subTest(path=path, course=COURSE_B):
                allowed = self.client.post(
                    path, json={"courseId": COURSE_B, "question": "q?"}, headers=CSRF
                )
                self.assertNotIn(allowed.status_code, (401, 403))

    def test_a_malformed_course_id_is_400_for_everyone_before_authorization(self) -> None:
        for principal_name in PRINCIPALS:
            app.dependency_overrides.pop(current_principal, None)
            if PRINCIPALS[principal_name] is not ANONYMOUS:
                app.dependency_overrides[current_principal] = lambda p=PRINCIPALS[principal_name]: p
            with self.subTest(principal=principal_name):
                response = self.client.get("/api/db/courses/Bad_Id/seeds")
                self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
