"""Administration: `/api/admin`.

Every route is `require_admin`. The first class proves that a professor is
refused on every one of them — including invitation creation with
`kind: admin` — before any behaviour is examined.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.auth.dependencies import current_principal
from app.auth.principal import Participant, Principal, StaffUser
from app.auth.settings import CSRF_HEADER_VALUE
from app.main import app

COURSE = "css-360-winter-2026-a7rp"
OTHER_COURSE = "css-350-spring-2026-n3h9"
NOW = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
CSRF = {"X-Requested-With": CSRF_HEADER_VALUE}
ADMIN_ID = "u-admin"
PROF_ID = "u-prof"


@contextmanager
def _fake_connection(**kwargs: Any) -> Iterator[object]:
    yield object()


def staff(role: str, user_id: str, *course_ids: str) -> Principal:
    return Principal(
        user=StaffUser(
            user_id=user_id, email=f"{user_id}@uw.edu", display_name=user_id.title(),
            role=role, course_ids=frozenset(course_ids), session_id="s-1",
        )
    )


def user_record(user_id: str = PROF_ID, role: str = "professor", disabled: bool = False) -> dict:
    return {
        "userId": user_id, "email": f"{user_id}@uw.edu", "displayName": user_id.title(),
        "role": role, "createdAt": NOW.isoformat(), "disabled": disabled,
    }


def token_invite(kind: str = "professor", **overrides: Any) -> dict[str, Any]:
    record = {
        "invitationId": "inv-1", "kind": kind, "status": "active",
        "createdAt": NOW.isoformat(), "expiresAt": (NOW + timedelta(days=7)).isoformat(),
        "useCount": 0, "maxUses": 1, "createdBy": ADMIN_ID,
    }
    record.update(overrides)
    return record


class AdminRouteTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app, base_url="https://testserver")
        self.addCleanup(app.dependency_overrides.clear)
        self.patch("app.admin_routes.db_connection", new=_fake_connection)
        self.patch("app.admin_routes._utc_now", return_value=NOW)
        self.record_action = self.patch("app.db_admin_actions.record_action", return_value=None)
        self.patch("app.db_users.list_users", return_value=[user_record(ADMIN_ID, "admin"), user_record()])
        self.patch(
            "app.db_memberships.list_memberships",
            return_value=[{"userId": PROF_ID, "courseId": COURSE, "membershipRole": "instructor"}],
        )
        self.get_user = self.patch("app.db_users.get_user", return_value=user_record())
        self.patch("app.db_memberships.list_course_ids_for_user", return_value=[COURSE])
        self.patch("app.db_courses.course_exists", return_value=True)
        self.add_membership = self.patch("app.db_memberships.add_membership", return_value=True)
        self.remove_membership = self.patch("app.db_memberships.remove_membership", return_value=True)
        self.set_disabled = self.patch(
            "app.db_users.set_disabled", return_value=user_record(disabled=True)
        )
        self.set_role = self.patch("app.db_users.set_role", return_value=user_record(role="admin"))
        self.revoke_sessions = self.patch("app.db_sessions.revoke_sessions_for_user", return_value=1)
        self.count_admins = self.patch("app.db_users.count_active_admins", return_value=2)
        self.create_token = self.patch(
            "app.db_invitations.create_token_invitation", return_value=token_invite()
        )
        self.patch("app.db_invitations.list_invitations", return_value=[token_invite()])
        self.get_invitation = self.patch("app.db_invitations.get_invitation", return_value=token_invite())
        self.revoke = self.patch(
            "app.db_invitations.revoke", return_value=token_invite(status="revoked")
        )
        self.patch("app.db_invitations.list_grants", return_value=[COURSE])
        self.patch("app.db_admin_actions.list_actions", return_value=[
            {"actionId": 1, "action": "invitation.create", "createdAt": NOW.isoformat()}
        ])

    def patch(self, target: str, **kwargs: Any) -> Any:
        patcher = patch(target, **kwargs)
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock

    def act_as(self, principal: Principal) -> None:
        app.dependency_overrides[current_principal] = lambda: principal


ADMIN_CALLS: list[tuple[str, str, dict[str, Any] | None]] = [
    ("GET", "/api/admin/users", None),
    ("PATCH", f"/api/admin/users/{PROF_ID}", {"disabled": True}),
    ("PUT", f"/api/admin/users/{PROF_ID}/courses/{COURSE}", None),
    ("DELETE", f"/api/admin/users/{PROF_ID}/courses/{COURSE}", None),
    ("POST", f"/api/admin/users/{PROF_ID}/reset-invite", None),
    ("POST", "/api/admin/invitations", {"kind": "admin"}),
    ("POST", "/api/admin/invitations", {"kind": "professor", "courseIds": [COURSE]}),
    ("GET", "/api/admin/invitations", None),
    ("POST", "/api/admin/invitations/inv-1/revoke", None),
    ("GET", "/api/admin/audit", None),
]


class AdminOnlyTests(AdminRouteTestCase):
    def _call(self, method: str, path: str, body: dict[str, Any] | None):
        return self.client.request(method, path, json=body, headers=CSRF)

    def test_anonymous_is_401_everywhere(self) -> None:
        for method, path, body in ADMIN_CALLS:
            with self.subTest(method=method, path=path):
                self.assertEqual(self._call(method, path, body).status_code, 401)

    def test_a_participant_is_401_everywhere(self) -> None:
        self.act_as(Principal(participant=Participant(participant_id="p", course_id=COURSE)))
        for method, path, body in ADMIN_CALLS:
            with self.subTest(method=method, path=path):
                self.assertEqual(self._call(method, path, body).status_code, 401)

    def test_a_professor_is_403_everywhere_including_admin_invitations(self) -> None:
        self.act_as(staff("professor", PROF_ID, COURSE))
        for method, path, body in ADMIN_CALLS:
            with self.subTest(method=method, path=path):
                self.assertEqual(self._call(method, path, body).status_code, 403)
        self.create_token.assert_not_called()
        self.record_action.assert_not_called()

    def test_an_admin_reaches_all_of_them(self) -> None:
        self.act_as(staff("admin", ADMIN_ID))
        for method, path, body in ADMIN_CALLS:
            with self.subTest(method=method, path=path):
                self.assertIn(self._call(method, path, body).status_code, (200, 201))


class UserManagementTests(AdminRouteTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.act_as(staff("admin", ADMIN_ID))

    def test_users_come_with_their_courses(self) -> None:
        body = self.client.get("/api/admin/users").json()
        by_id = {user["userId"]: user for user in body["users"]}
        self.assertEqual(by_id[PROF_ID]["courseIds"], [COURSE])
        self.assertEqual(by_id[ADMIN_ID]["courseIds"], [])
        for user in body["users"]:
            self.assertNotIn("passwordHash", user)

    def test_disabling_revokes_sessions_and_audits(self) -> None:
        response = self.client.patch(
            f"/api/admin/users/{PROF_ID}", json={"disabled": True}, headers=CSRF
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["disabled"])
        self.revoke_sessions.assert_called_once()
        self.assertEqual(self.record_action.call_args.kwargs["action"], "user.disable")

    def test_role_change_is_audited_with_before_and_after(self) -> None:
        response = self.client.patch(
            f"/api/admin/users/{PROF_ID}", json={"role": "admin"}, headers=CSRF
        )
        self.assertEqual(response.status_code, 200)
        detail = self.record_action.call_args.kwargs["detail"]
        self.assertEqual(detail, {"from": "professor", "to": "admin"})

    def test_you_cannot_disable_or_demote_yourself(self) -> None:
        for body in ({"disabled": True}, {"role": "professor"}):
            with self.subTest(body=body):
                response = self.client.patch(
                    f"/api/admin/users/{ADMIN_ID}", json=body, headers=CSRF
                )
                self.assertEqual(response.status_code, 409)

    def test_the_last_administrator_cannot_be_removed(self) -> None:
        self.get_user.return_value = user_record("u-other-admin", "admin")
        self.count_admins.return_value = 1
        response = self.client.patch(
            "/api/admin/users/u-other-admin", json={"disabled": True}, headers=CSRF
        )
        self.assertEqual(response.status_code, 409)
        self.set_disabled.assert_not_called()

    def test_unknown_role_and_empty_patch_are_422(self) -> None:
        self.assertEqual(
            self.client.patch(f"/api/admin/users/{PROF_ID}", json={"role": "student"}, headers=CSRF).status_code,
            422,
        )
        self.assertEqual(
            self.client.patch(f"/api/admin/users/{PROF_ID}", json={}, headers=CSRF).status_code, 422
        )

    def test_membership_add_and_remove_are_audited_and_course_validated(self) -> None:
        added = self.client.put(f"/api/admin/users/{PROF_ID}/courses/{COURSE}", headers=CSRF)
        self.assertEqual(added.status_code, 200)
        self.assertTrue(added.json()["member"])
        self.assertEqual(self.record_action.call_args.kwargs["action"], "membership.add")
        removed = self.client.delete(f"/api/admin/users/{PROF_ID}/courses/{COURSE}", headers=CSRF)
        self.assertEqual(removed.status_code, 200)
        self.assertFalse(removed.json()["member"])
        self.assertEqual(self.record_action.call_args.kwargs["action"], "membership.remove")
        self.assertEqual(
            self.client.put(f"/api/admin/users/{PROF_ID}/courses/Bad_Id", headers=CSRF).status_code,
            400,
        )

    def test_membership_for_a_missing_course_or_user_is_404(self) -> None:
        with patch("app.db_courses.course_exists", return_value=False):
            self.assertEqual(
                self.client.put(f"/api/admin/users/{PROF_ID}/courses/{COURSE}", headers=CSRF).status_code,
                404,
            )
        self.get_user.return_value = None
        self.assertEqual(
            self.client.put(f"/api/admin/users/nobody/courses/{COURSE}", headers=CSRF).status_code,
            404,
        )


class InvitationManagementTests(AdminRouteTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.act_as(staff("admin", ADMIN_ID))

    def test_a_professor_invitation_carries_its_courses_and_returns_the_token_once(self) -> None:
        response = self.client.post(
            "/api/admin/invitations",
            json={"kind": "professor", "courseIds": [COURSE], "label": "Dr. X"},
            headers=CSRF,
        )
        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertTrue(body["path"].startswith("/invite/"))
        self.assertEqual(body["courseIds"], [COURSE])
        self.assertEqual(body["token"], body["path"].removeprefix("/invite/"))
        kwargs = self.create_token.call_args.kwargs
        self.assertEqual(kwargs["kind"], "professor")
        self.assertEqual(kwargs["course_ids"], [COURSE])
        self.assertEqual(kwargs["expires_at"], NOW + timedelta(days=7))
        # Hashed at rest: the stored value is not the token.
        self.assertNotEqual(kwargs["token_hash"], body["token"])
        self.assertEqual(len(kwargs["token_hash"]), 64)
        audit = self.record_action.call_args.kwargs
        self.assertEqual(audit["action"], "invitation.create")
        self.assertNotIn("token", str(audit.get("detail")))

    def test_an_admin_invitation_is_short_lived_and_takes_no_courses(self) -> None:
        response = self.client.post("/api/admin/invitations", json={"kind": "admin"}, headers=CSRF)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(self.create_token.call_args.kwargs["expires_at"], NOW + timedelta(hours=24))
        refused = self.client.post(
            "/api/admin/invitations", json={"kind": "admin", "courseIds": [COURSE]}, headers=CSRF
        )
        self.assertEqual(refused.status_code, 422)

    def test_lifetimes_are_capped(self) -> None:
        too_long = self.client.post(
            "/api/admin/invitations", json={"kind": "admin", "expiresInHours": 200}, headers=CSRF
        )
        self.assertEqual(too_long.status_code, 422)
        shorter = self.client.post(
            "/api/admin/invitations", json={"kind": "admin", "expiresInHours": 2}, headers=CSRF
        )
        self.assertEqual(shorter.status_code, 201)
        self.assertEqual(self.create_token.call_args.kwargs["expires_at"], NOW + timedelta(hours=2))

    def test_only_professor_and_admin_kinds_can_be_minted_here(self) -> None:
        for kind in ("student", "reset", "root"):
            with self.subTest(kind=kind):
                response = self.client.post(
                    "/api/admin/invitations", json={"kind": kind}, headers=CSRF
                )
                self.assertEqual(response.status_code, 422)
        self.create_token.assert_not_called()

    def test_courses_must_exist(self) -> None:
        with patch("app.db_courses.course_exists", return_value=False):
            response = self.client.post(
                "/api/admin/invitations",
                json={"kind": "professor", "courseIds": [OTHER_COURSE]},
                headers=CSRF,
            )
        self.assertEqual(response.status_code, 404)

    def test_listing_never_includes_a_token(self) -> None:
        body = self.client.get("/api/admin/invitations").json()
        self.assertEqual(body["count"], 1)
        self.assertNotIn("token", body["invitations"][0])
        self.assertNotIn("tokenHash", body["invitations"][0])

    def test_listing_filters_by_status(self) -> None:
        body = self.client.get("/api/admin/invitations", params={"status": "revoked"}).json()
        self.assertEqual(body["count"], 0)

    def test_revoking_audits_once_and_refuses_classroom_codes(self) -> None:
        response = self.client.post("/api/admin/invitations/inv-1/revoke", headers=CSRF)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "revoked")
        self.assertEqual(self.record_action.call_args.kwargs["action"], "invitation.revoke")
        self.get_invitation.return_value = token_invite(kind="student", code="7K4P9X")
        self.assertEqual(
            self.client.post("/api/admin/invitations/inv-1/revoke", headers=CSRF).status_code, 404
        )

    def test_a_reset_link_targets_the_account_and_is_audited(self) -> None:
        self.create_token.return_value = token_invite("reset", targetUserId=PROF_ID)
        response = self.client.post(f"/api/admin/users/{PROF_ID}/reset-invite", headers=CSRF)
        self.assertEqual(response.status_code, 201)
        kwargs = self.create_token.call_args.kwargs
        self.assertEqual(kwargs["kind"], "reset")
        self.assertEqual(kwargs["target_user_id"], PROF_ID)
        self.assertEqual(kwargs["expires_at"], NOW + timedelta(hours=24))
        self.assertEqual(self.record_action.call_args.kwargs["action"], "user.reset_invite")
        self.get_user.return_value = user_record(disabled=True)
        self.assertEqual(
            self.client.post(f"/api/admin/users/{PROF_ID}/reset-invite", headers=CSRF).status_code, 409
        )

    def test_audit_listing(self) -> None:
        body = self.client.get("/api/admin/audit", params={"limit": 10}).json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["actions"][0]["action"], "invitation.create")


if __name__ == "__main__":
    unittest.main()
