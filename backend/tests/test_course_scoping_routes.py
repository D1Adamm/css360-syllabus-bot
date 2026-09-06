"""What each principal sees in the course list, and who owns a new course.

`GET /api/db/courses` is the one route where "allowed" is a filter rather
than a yes or no: an administrator sees every course, a professor the ones
they hold a membership in, a participant the one they joined. And
`POST /api/db/courses` makes a professor the instructor of what they create,
so the course they just made does not vanish from their list.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import current_principal
from app.auth.principal import Participant, Principal, StaffUser
from app.auth.settings import CSRF_HEADER_VALUE
from app.main import app

pytestmark = pytest.mark.auth

COURSE_A = "css-360-winter-2026-a7rp"
COURSE_B = "css-350-spring-2026-n3h9"
CSRF = {"X-Requested-With": CSRF_HEADER_VALUE}

METADATA = {
    "name": "CSS 360",
    "title": "Software Engineering",
    "term": "Winter 2026",
    "instructorName": "",
    "createdAt": "2026-01-01T00:00:00+00:00",
    "syllabusStatus": "none",
    "syllabusFileName": None,
    "syllabusType": None,
    "chunkCount": 0,
}


@contextmanager
def _fake_connection(**kwargs: Any) -> Iterator[object]:
    yield object()


def staff(role: str, *course_ids: str) -> Principal:
    return Principal(
        user=StaffUser(
            user_id="u-1", email="x@uw.edu", display_name="X", role=role,
            course_ids=frozenset(course_ids), session_id="s-1",
        )
    )


class CourseScopingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app, base_url="https://testserver")
        self.addCleanup(app.dependency_overrides.clear)
        for target in ("app.db_routes.db_connection",):
            patcher = patch(target, new=_fake_connection)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.list_courses = self._patch("app.db_courses.list_courses", return_value=[])
        self.create_course = self._patch(
            "app.db_courses.create_course",
            return_value={"courseId": COURSE_A, "metadata": METADATA},
        )
        self.add_membership = self._patch("app.db_memberships.add_membership", return_value=True)
        self.record_action = self._patch("app.db_admin_actions.record_action", return_value=None)

    def _patch(self, target: str, **kwargs: Any) -> Any:
        patcher = patch(target, **kwargs)
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock

    def act_as(self, principal: Principal) -> None:
        app.dependency_overrides[current_principal] = lambda: principal

    def scope_requested(self) -> Any:
        return self.list_courses.call_args.args[1]

    # ---- listing ----

    def test_anonymous_gets_401_not_an_empty_list(self) -> None:
        self.assertEqual(self.client.get("/api/db/courses").status_code, 401)
        self.list_courses.assert_not_called()

    def test_an_admin_asks_for_every_course(self) -> None:
        self.act_as(staff("admin"))
        self.assertEqual(self.client.get("/api/db/courses").status_code, 200)
        self.assertIsNone(self.scope_requested())

    def test_a_professor_asks_only_for_memberships(self) -> None:
        self.act_as(staff("professor", COURSE_A, COURSE_B))
        self.assertEqual(self.client.get("/api/db/courses").status_code, 200)
        self.assertEqual(self.scope_requested(), {COURSE_A, COURSE_B})

    def test_a_professor_with_no_memberships_sees_nothing(self) -> None:
        self.act_as(staff("professor"))
        body = self.client.get("/api/db/courses").json()
        self.assertEqual(body, {"count": 0, "courses": []})
        self.assertEqual(self.scope_requested(), set())

    def test_a_participant_asks_only_for_the_joined_course(self) -> None:
        self.act_as(Principal(participant=Participant(participant_id="p", course_id=COURSE_B)))
        self.assertEqual(self.client.get("/api/db/courses").status_code, 200)
        self.assertEqual(self.scope_requested(), {COURSE_B})

    def test_a_professor_who_also_joined_a_course_sees_both(self) -> None:
        self.act_as(
            Principal(
                user=staff("professor", COURSE_A).user,
                participant=Participant(participant_id="p", course_id=COURSE_B),
            )
        )
        self.client.get("/api/db/courses")
        self.assertEqual(self.scope_requested(), {COURSE_A, COURSE_B})

    # ---- creation ----

    def _create(self) -> Any:
        return self.client.post(
            "/api/db/courses",
            json={"courseId": COURSE_A, "name": "CSS 360", "title": "SE",
                  "term": "Winter 2026", "instructorName": "Prof"},
            headers=CSRF,
        )

    def test_a_professor_becomes_the_instructor_of_the_course_they_create(self) -> None:
        self.act_as(staff("professor"))
        response = self._create()
        self.assertEqual(response.status_code, 201)
        self.assertEqual(self.create_course.call_args.kwargs["created_by"], "u-1")
        self.add_membership.assert_called_once()
        self.assertEqual(self.add_membership.call_args.kwargs["course_id"], COURSE_A)
        self.assertEqual(self.add_membership.call_args.kwargs["user_id"], "u-1")
        self.assertEqual(self.record_action.call_args.kwargs["action"], "course.create")

    def test_an_admin_needs_no_membership(self) -> None:
        self.act_as(staff("admin"))
        self.assertEqual(self._create().status_code, 201)
        self.add_membership.assert_not_called()

    def test_a_participant_cannot_create_a_course(self) -> None:
        self.act_as(Principal(participant=Participant(participant_id="p", course_id=COURSE_A)))
        self.assertEqual(self._create().status_code, 401)
        self.create_course.assert_not_called()


if __name__ == "__main__":
    unittest.main()
