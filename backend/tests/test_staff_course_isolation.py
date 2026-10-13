"""Instructors reach their own courses only; staff can ask questions without being students.

`test_authorization_matrix.py` drives every route with a professor of another
course and expects 403. This file states the same rule in the terms the
Professor interface is used in, one surface at a time, and adds what the
matrix cannot: that an explicit co-instructor is let in, that an
administrator's own assignments never narrow their access, and that a staff
comparison (the Professor "Ask questions" tab) answers on all four conditions
with no participant session and writes nothing a study would read.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import current_principal
from app.auth.principal import Principal, StaffUser
from app.auth.settings import CSRF_HEADER_VALUE, PARTICIPANT_COOKIE_NAME, STAFF_COOKIE_NAME
from app.main import app

pytestmark = pytest.mark.auth

MINE = "css360e-autumn-2026-c08m"
THEIRS = "css360d-fall-2026-q0ne"
CSRF = {"X-Requested-With": CSRF_HEADER_VALUE}

GENERATION_PATHS = (
    "/api/base-model/generate",
    "/api/rag/generate",
    "/api/fine-tuned/generate",
    "/api/fine-tuned-rag/generate",
)

#: One request per Professor-interface surface, for the course in the path.
SURFACES: dict[str, tuple[str, str, dict[str, Any] | None]] = {
    "course": ("GET", "/api/db/courses/{c}", None),
    "course edit": ("PATCH", "/api/db/courses/{c}", {"title": "x"}),
    "syllabus text": ("GET", "/api/courses/{c}/syllabus/text", None),
    "examples": ("GET", "/api/courses/{c}/seeds", None),
    "example review": ("POST", "/api/courses/{c}/seeds/s-1/review", {"reviewStatus": "approved"}),
    "model": ("GET", "/api/db/courses/{c}/model", None),
    "model request": ("GET", "/api/db/courses/{c}/model-request", None),
    "request a model": ("POST", "/api/db/courses/{c}/model-request", {"approvedExampleCount": 1}),
    "results": ("GET", "/api/db/courses/{c}/evaluations", None),
    "student invites": ("GET", "/api/courses/{c}/student-invites", None),
}


def professor(*course_ids: str, user_id: str = "u-prof") -> Principal:
    return Principal(
        user=StaffUser(
            user_id=user_id, email=f"{user_id}@uw.edu", display_name="Prof",
            role="professor", course_ids=frozenset(course_ids), session_id="s",
        )
    )


def admin(*course_ids: str) -> Principal:
    return Principal(
        user=StaffUser(
            user_id="u-admin", email="admin@uw.edu", display_name="Admin",
            role="admin", course_ids=frozenset(course_ids), session_id="s",
        )
    )


@contextmanager
def _no_database(**kwargs: Any) -> Iterator[object]:
    raise AssertionError("this request must not open a database connection")
    yield  # pragma: no cover


class IsolationCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app, base_url="https://testserver")
        self.addCleanup(app.dependency_overrides.clear)

    def patch(self, target: str, **kwargs: Any) -> Any:
        patcher = patch(target, **kwargs)
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock

    def act_as(self, principal: Principal) -> None:
        app.dependency_overrides[current_principal] = lambda: principal

    def call(self, surface: str, course_id: str):
        method, path, body = SURFACES[surface]
        return self.client.request(method, path.format(c=course_id), json=body, headers=CSRF)

    def generate(self, path: str, course_id: str):
        return self.client.post(
            path, json={"courseId": course_id, "question": "When are office hours?"}, headers=CSRF
        )


class ProfessorIsolationTests(IsolationCase):
    def test_another_instructors_course_is_refused_on_every_surface(self) -> None:
        self.act_as(professor(MINE))
        for surface in SURFACES:
            with self.subTest(surface=surface):
                self.assertEqual(self.call(surface, THEIRS).status_code, 403)
        for path in GENERATION_PATHS:
            with self.subTest(path=path):
                self.assertEqual(self.generate(path, THEIRS).status_code, 403)

    def test_their_own_course_passes_the_guard(self) -> None:
        # Past the guard the handlers meet no database (503) or a stub; never 401/403.
        self.act_as(professor(MINE))
        for surface in SURFACES:
            with self.subTest(surface=surface):
                self.assertNotIn(self.call(surface, MINE).status_code, (401, 403))

    def test_an_explicit_co_instructor_is_let_in(self) -> None:
        # Two professors assigned to the same course by an administrator.
        for who in (professor(MINE, user_id="u-one"), professor(MINE, THEIRS, user_id="u-two")):
            self.act_as(who)
            for surface in SURFACES:
                with self.subTest(user=who.user.user_id, surface=surface):
                    self.assertNotIn(self.call(surface, MINE).status_code, (401, 403))

    def test_a_professor_cannot_assign_themselves_or_anyone(self) -> None:
        add = self.patch("app.db_memberships.add_membership", return_value=True)
        remove = self.patch("app.db_memberships.remove_membership", return_value=True)
        self.act_as(professor(MINE))
        for method, path in (
            ("PUT", f"/api/admin/users/u-prof/courses/{THEIRS}"),
            ("PUT", f"/api/admin/users/u-other/courses/{MINE}"),
            ("DELETE", f"/api/admin/users/u-other/courses/{MINE}"),
        ):
            with self.subTest(method=method, path=path):
                self.assertEqual(self.client.request(method, path, headers=CSRF).status_code, 403)
        add.assert_not_called()
        remove.assert_not_called()

    def test_the_course_list_is_scoped_to_memberships(self) -> None:
        seen: list[Any] = []

        @contextmanager
        def connection(**kwargs: Any) -> Iterator[object]:
            yield object()

        self.patch("app.db_routes.db_connection", new=connection)
        self.patch("app.db_courses.list_courses",
                   side_effect=lambda conn, visible=None: seen.append(visible) or [])
        self.act_as(professor(MINE))
        self.assertEqual(self.client.get("/api/db/courses").status_code, 200)
        self.act_as(admin(MINE))
        self.assertEqual(self.client.get("/api/db/courses").status_code, 200)
        self.assertEqual(seen, [{MINE}, None])  # the professor's memberships; every course


class AdministratorTests(IsolationCase):
    def test_assignments_never_narrow_an_administrators_access(self) -> None:
        self.act_as(admin(MINE))
        for surface in SURFACES:
            with self.subTest(surface=surface):
                self.assertNotIn(self.call(surface, THEIRS).status_code, (401, 403))
        self.assertIs(admin(MINE).can_staff_course(THEIRS), True)
        self.assertEqual(admin(MINE).staff_course_ids(), None)


class StaffComparisonTests(IsolationCase):
    """The Professor "Ask questions" tab: the same four routes as Compare."""

    def setUp(self) -> None:
        super().setUp()
        self.base = AsyncMock(return_value={"answer": "Base.", "model": "llama3.2:3b", "response_type": "base"})
        self.rag = AsyncMock(return_value={
            "courseId": MINE, "answer": "RAG.", "model": "llama3.2:3b",
            "sources": [{"chunkId": "c1", "sectionTitle": "Office hours", "text": "Tue", "score": 0.9}],
            "retrievedChunks": [], "responseType": "rag",
        })
        self.fine_tuned = AsyncMock(return_value={
            "answer": "FT.", "model": "css360e-v1:latest", "response_type": "fineTuned",
            "model_version": "v1", "adapter_loaded": True, "generation_seconds": 0.4,
        })
        self.fine_tuned_rag = AsyncMock(return_value={
            "courseId": MINE, "answer": "FT+RAG.", "model": "css360e-v1:latest", "sources": [],
            "retrievedChunks": [], "responseType": "fineTunedRag", "modelVersion": "v1",
            "adapterLoaded": True, "generationSeconds": 0.6,
        })
        self.patch("app.main.generate_base_model_response", new=self.base)
        self.patch("app.main.generate_course_rag_answer", new=self.rag)
        self.resolve = self.patch("app.main.resolve_current_course_model", return_value={"version": "v1"})
        self.patch("app.main.generate_finetuned_response", new=self.fine_tuned)
        self.patch("app.main.generate_course_finetuned_rag_answer", new=self.fine_tuned_rag)
        # Anything that would put a staff question into the study's tables.
        self.patch("app.db_routes.db_connection", new=_no_database)
        self.patch("app.main.db_connection", new=_no_database)
        self.writes = [
            self.patch(target)
            for target in (
                "app.db_participants.create_participant",
                "app.db_sessions.create_session",
                "app.db_seeds.create_seed",
                "app.db_evaluations.create_evaluation",
            )
        ]

    def test_an_assigned_instructor_gets_all_four_answers_without_being_a_participant(self) -> None:
        self.act_as(professor(MINE))
        for path in GENERATION_PATHS:
            with self.subTest(path=path):
                response = self.generate(path, MINE)
                self.assertEqual(response.status_code, 200, response.text)
                # No identity is minted or changed for the person asking.
                self.assertIsNone(response.cookies.get(PARTICIPANT_COOKIE_NAME))
                self.assertIsNone(response.cookies.get(STAFF_COOKIE_NAME))
        # Every course-scoped generator was asked for this course, and the
        # fine-tuned answer for this course's resolved version.
        for stub in (self.rag, self.fine_tuned, self.fine_tuned_rag):
            self.assertEqual(stub.call_args.kwargs["course_id"], MINE)
        self.assertEqual(self.fine_tuned.call_args.kwargs["model_version"], "v1")
        self.resolve.assert_called_with(MINE)
        for write in self.writes:
            write.assert_not_called()

    def test_sources_and_the_answering_model_come_back(self) -> None:
        self.act_as(professor(MINE))
        rag = self.generate("/api/rag/generate", MINE).json()
        self.assertEqual(rag["sources"][0]["sectionTitle"], "Office hours")
        fine_tuned = self.generate("/api/fine-tuned/generate", MINE).json()
        self.assertEqual((fine_tuned["model"], fine_tuned["modelVersion"]), ("css360e-v1:latest", "v1"))

    def test_another_instructors_course_is_never_asked(self) -> None:
        self.act_as(professor(MINE))
        for path in GENERATION_PATHS:
            with self.subTest(path=path):
                self.assertEqual(self.generate(path, THEIRS).status_code, 403)
        for stub in (self.base, self.rag, self.fine_tuned, self.fine_tuned_rag):
            stub.assert_not_called()

    def test_an_administrator_can_ask_any_course(self) -> None:
        self.act_as(admin())
        for path in GENERATION_PATHS:
            with self.subTest(path=path):
                self.assertEqual(self.generate(path, THEIRS).status_code, 200)
        for write in self.writes:
            write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
