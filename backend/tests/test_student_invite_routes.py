"""Classroom codes: `/api/courses/{course_id}/student-invites`.

The guard is `require_course_staff`, so what these mostly pin is scope: a
professor manages codes for the course in the path only when they hold a
membership in it, an administrator for any course, and nobody else at all.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import db_invitations
from app.auth.dependencies import current_principal
from app.auth.principal import Participant, Principal, StaffUser
from app.auth.settings import CSRF_HEADER_VALUE
from app.main import app

COURSE = "css-360-winter-2026-a7rp"
OTHER_COURSE = "css-350-spring-2026-n3h9"
NOW = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
CSRF = {"X-Requested-With": CSRF_HEADER_VALUE}


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


def invite(**overrides: Any) -> dict[str, Any]:
    record = {
        "invitationId": "inv-1", "kind": "student", "status": "active", "code": "7K4P9X",
        "courseId": COURSE, "createdAt": NOW.isoformat(), "useCount": 12,
        "createdBy": "u-1", "label": "Section A",
    }
    record.update(overrides)
    return record


class StudentInviteRouteTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app, base_url="https://testserver")
        self.addCleanup(app.dependency_overrides.clear)
        self.patch("app.student_invite_routes.db_connection", new=_fake_connection)
        self.patch("app.student_invite_routes._utc_now", return_value=NOW)
        self.patch("app.db_courses.course_exists", return_value=True)
        self.list = self.patch("app.db_invitations.list_student_invitations", return_value=[invite()])
        self.create = self.patch("app.db_invitations.create_student_invitation", return_value=invite())
        self.get = self.patch("app.db_invitations.get_invitation", return_value=invite())
        self.revoke = self.patch(
            "app.db_invitations.revoke", return_value=invite(status="revoked", revokedAt=NOW.isoformat())
        )
        self.revoke_active = self.patch(
            "app.db_invitations.revoke_active_student_invitations", return_value=1
        )
        self.patch("app.db_participants.count_participants_for_invitation", return_value=9)
        self.record_action = self.patch("app.db_admin_actions.record_action", return_value=None)

    def patch(self, target: str, **kwargs: Any) -> Any:
        patcher = patch(target, **kwargs)
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock

    def act_as(self, principal: Principal) -> None:
        app.dependency_overrides[current_principal] = lambda: principal

    def base(self, course_id: str = COURSE) -> str:
        return f"/api/courses/{course_id}/student-invites"


class ScopeTests(StudentInviteRouteTestCase):
    def test_anonymous_is_401(self) -> None:
        self.assertEqual(self.client.get(self.base()).status_code, 401)
        self.assertEqual(self.client.post(self.base(), json={}, headers=CSRF).status_code, 401)

    def test_a_participant_is_401_it_has_no_staff_session(self) -> None:
        self.act_as(Principal(participant=Participant(participant_id="p", course_id=COURSE)))
        self.assertEqual(self.client.get(self.base()).status_code, 401)

    def test_a_professor_manages_only_assigned_courses(self) -> None:
        self.act_as(staff("professor", COURSE))
        self.assertEqual(self.client.get(self.base(COURSE)).status_code, 200)
        self.assertEqual(self.client.get(self.base(OTHER_COURSE)).status_code, 403)
        created = self.client.post(self.base(COURSE), json={}, headers=CSRF)
        self.assertEqual(created.status_code, 201)
        refused = self.client.post(self.base(OTHER_COURSE), json={}, headers=CSRF)
        self.assertEqual(refused.status_code, 403)
        self.assertEqual(
            self.client.post(f"{self.base(OTHER_COURSE)}/inv-1/revoke", headers=CSRF).status_code,
            403,
        )

    def test_an_admin_manages_any_course(self) -> None:
        self.act_as(staff("admin"))
        self.assertEqual(self.client.get(self.base(OTHER_COURSE)).status_code, 200)
        self.assertEqual(
            self.client.post(self.base(OTHER_COURSE), json={}, headers=CSRF).status_code, 201
        )


class BehaviourTests(StudentInviteRouteTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.act_as(staff("professor", COURSE))

    def test_listing_includes_the_code_and_participant_count(self) -> None:
        body = self.client.get(self.base()).json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["invites"][0]["code"], "7K4P9X")
        self.assertEqual(body["invites"][0]["participantCount"], 9)
        self.assertEqual(body["invites"][0]["status"], "active")

    def test_creating_draws_a_code_and_audits_it(self) -> None:
        response = self.client.post(self.base(), json={"label": "Section A"}, headers=CSRF)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["code"], "7K4P9X")
        kwargs = self.create.call_args.kwargs
        self.assertEqual(kwargs["course_id"], COURSE)
        self.assertEqual(kwargs["created_by"], "u-1")
        self.assertEqual(len(kwargs["code"]), 6)
        self.assertEqual(self.record_action.call_args.kwargs["action"], "invitation.create")
        self.assertEqual(self.record_action.call_args.kwargs["course_id"], COURSE)
        self.revoke_active.assert_not_called()

    def test_replacing_retires_the_live_codes_first(self) -> None:
        response = self.client.post(self.base(), json={"replaceExisting": True}, headers=CSRF)
        self.assertEqual(response.status_code, 201)
        self.revoke_active.assert_called_once()
        self.assertEqual(self.record_action.call_args.kwargs["detail"]["replaced"], 1)

    def test_a_code_collision_is_retried_with_a_fresh_code(self) -> None:
        self.create.side_effect = [db_invitations.CodeCollisionError("taken"), invite()]
        response = self.client.post(self.base(), json={}, headers=CSRF)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(self.create.call_count, 2)
        first, second = (call.kwargs["code"] for call in self.create.call_args_list)
        self.assertNotEqual(first, second)

    def test_an_expiry_must_be_a_future_timestamp(self) -> None:
        past = (NOW - timedelta(days=1)).isoformat()
        self.assertEqual(
            self.client.post(self.base(), json={"expiresAt": past}, headers=CSRF).status_code, 422
        )
        self.assertEqual(
            self.client.post(self.base(), json={"expiresAt": "soon"}, headers=CSRF).status_code, 422
        )
        future = (NOW + timedelta(days=90)).isoformat().replace("+00:00", "Z")
        ok = self.client.post(self.base(), json={"expiresAt": future}, headers=CSRF)
        self.assertEqual(ok.status_code, 201)
        self.assertEqual(self.create.call_args.kwargs["expires_at"], NOW + timedelta(days=90))

    def test_a_missing_course_is_404(self) -> None:
        with patch("app.db_courses.course_exists", return_value=False):
            self.assertEqual(self.client.post(self.base(), json={}, headers=CSRF).status_code, 404)

    def test_revoking_a_code_of_this_course(self) -> None:
        response = self.client.post(f"{self.base()}/inv-1/revoke", headers=CSRF)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "revoked")
        self.assertEqual(self.record_action.call_args.kwargs["action"], "invitation.revoke")

    def test_a_code_from_another_course_is_not_found_here(self) -> None:
        """Naming another course's code under this course is a 404, not a cross-course revoke."""
        self.get.return_value = invite(courseId=OTHER_COURSE)
        response = self.client.post(f"{self.base()}/inv-1/revoke", headers=CSRF)
        self.assertEqual(response.status_code, 404)
        self.revoke.assert_not_called()

    def test_a_privileged_invitation_id_is_not_a_classroom_code(self) -> None:
        self.get.return_value = invite(kind="admin", code=None, courseId=None)
        self.assertEqual(
            self.client.post(f"{self.base()}/inv-1/revoke", headers=CSRF).status_code, 404
        )

    def test_revoking_twice_audits_once(self) -> None:
        self.get.return_value = invite(status="revoked")
        self.assertEqual(
            self.client.post(f"{self.base()}/inv-1/revoke", headers=CSRF).status_code, 200
        )
        self.record_action.assert_not_called()


if __name__ == "__main__":
    unittest.main()
