"""The `/api/auth` router and the bootstrap invitation.

Repositories are patched out, as in `test_db_routes.py`; what these cover is
the contract the browser and the operator program against: which cookie is
set with which attributes, which failures are indistinguishable from each
other, what is throttled, and that a session comes out the other end of a
valid credential, a valid classroom code, or a valid invitation.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import db_users
from app.auth import rate_limit
from app.auth.bootstrap import AdminAlreadyExistsError, mint_bootstrap_invitation
from app.auth.dependencies import current_principal
from app.auth.passwords import hash_password
from app.auth.principal import Participant, Principal, StaffUser
from app.auth.settings import CSRF_HEADER_VALUE, PARTICIPANT_COOKIE_NAME, STAFF_COOKIE_NAME
from app.auth.tokens import generate_token
from app.auth_routes import (
    CODE_INVALID_DETAIL,
    INVITATION_INVALID_DETAIL,
    LOGIN_FAILED_DETAIL,
)
from app.main import app
from test_db_repositories import FakeConnection

COURSE = "css-360-winter-2026-a7rp"
OTHER_COURSE = "css-350-spring-2026-n3h9"
NOW = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
CSRF = {"X-Requested-With": CSRF_HEADER_VALUE}
PASSWORD = "correct horse battery staple"
PASSWORD_HASH = hash_password(PASSWORD)
USER_ID = "11111111-1111-1111-1111-111111111111"

USER_RECORD = {
    "userId": USER_ID,
    "email": "prof@uw.edu",
    "displayName": "Prof",
    "role": "professor",
    "createdAt": NOW.isoformat(),
    "disabled": False,
}


@contextmanager
def _fake_connection(**kwargs: Any) -> Iterator[object]:
    yield object()


def professor_principal(*course_ids: str, session_id: str = "sess-1") -> Principal:
    return Principal(
        user=StaffUser(
            user_id=USER_ID,
            email="prof@uw.edu",
            display_name="Prof",
            role="professor",
            course_ids=frozenset(course_ids),
            session_id=session_id,
        )
    )


def participant_principal(course_id: str, session_id: str = "sess-p") -> Principal:
    return Principal(
        participant=Participant(participant_id="part-1", course_id=course_id, session_id=session_id)
    )


def student_invitation(**overrides: Any) -> dict[str, Any]:
    record = {
        "invitationId": "inv-student",
        "kind": "student",
        "status": "active",
        "code": "7K4P9X",
        "courseId": COURSE,
        "courseName": "CSS 360",
        "createdAt": NOW.isoformat(),
        "useCount": 3,
    }
    record.update(overrides)
    return record


def token_invitation(kind: str = "professor", **overrides: Any) -> dict[str, Any]:
    record = {
        "invitationId": "inv-token",
        "kind": kind,
        "status": "active",
        "createdAt": NOW.isoformat(),
        "expiresAt": (NOW + timedelta(days=1)).isoformat(),
        "useCount": 0,
        "maxUses": 1,
        "createdBy": "admin-1",
    }
    if kind == "professor":
        record["courseIds"] = [COURSE]
    record.update(overrides)
    return record


class AuthRouteTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app, base_url="https://testserver")
        rate_limit.reset_all_limiters()
        self.addCleanup(rate_limit.reset_all_limiters)
        self.addCleanup(app.dependency_overrides.clear)
        self.patch("app.auth_routes.db_connection", new=_fake_connection)
        self.patch("app.auth_routes._utc_now", return_value=NOW)
        self.create_session = self.patch(
            "app.db_sessions.create_session", return_value={"sessionId": "sess-new", "expiresAt": "x"}
        )
        self.patch("app.db_sessions.delete_stale_sessions", return_value=0)
        self.record_login = self.patch("app.db_users.record_login", return_value=None)
        self.record_action = self.patch("app.db_admin_actions.record_action", return_value=None)

    def patch(self, target: str, **kwargs: Any) -> Any:
        patcher = patch(target, **kwargs)
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock

    def act_as(self, principal: Principal) -> None:
        app.dependency_overrides[current_principal] = lambda: principal

    def set_cookies(self) -> list[str]:
        return self.client.cookies.jar and [
            header for header in self.last_response.headers.get_list("set-cookie")
        ]


class LoginTests(AuthRouteTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.credentials = self.patch(
            "app.db_users.find_credentials",
            return_value={
                "userId": USER_ID,
                "role": "professor",
                "passwordHash": PASSWORD_HASH,
                "disabled": False,
            },
        )

    def login(self, email: str = "Prof@UW.edu", password: str = PASSWORD, **kwargs: Any):
        return self.client.post(
            "/api/auth/login", json={"email": email, "password": password}, headers=CSRF, **kwargs
        )

    def test_success_sets_a_hardened_staff_cookie(self) -> None:
        response = self.login()
        self.assertEqual(response.status_code, 204)
        cookie = next(
            header for header in response.headers.get_list("set-cookie")
            if header.startswith(f"{STAFF_COOKIE_NAME}=")
        )
        lowered = cookie.lower()
        for attribute in ("httponly", "secure", "samesite=lax", "path=/api", "max-age="):
            self.assertIn(attribute, lowered)
        self.create_session.assert_called_once()
        self.assertEqual(self.create_session.call_args.kwargs["kind"], "user")
        self.assertEqual(self.create_session.call_args.kwargs["user_id"], USER_ID)
        self.record_login.assert_called_once()
        # The email is looked up normalised.
        self.assertEqual(self.credentials.call_args.args[1], "prof@uw.edu")

    def test_wrong_password_and_unknown_email_are_the_same_401(self) -> None:
        wrong = self.login(password="not the password")
        self.credentials.return_value = None
        unknown = self.login(email="nobody@uw.edu")
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(unknown.status_code, 401)
        self.assertEqual(wrong.json(), unknown.json())
        self.assertEqual(wrong.json()["detail"], LOGIN_FAILED_DETAIL)
        self.create_session.assert_not_called()

    def test_a_disabled_account_cannot_sign_in(self) -> None:
        self.credentials.return_value["disabled"] = True
        self.assertEqual(self.login().status_code, 401)

    def test_the_csrf_header_is_required(self) -> None:
        response = self.client.post(
            "/api/auth/login", json={"email": "prof@uw.edu", "password": PASSWORD}
        )
        self.assertEqual(response.status_code, 403)

    def test_repeated_failures_are_throttled(self) -> None:
        for _ in range(rate_limit.LOGIN_PER_ACCOUNT.limit):
            self.assertEqual(self.login(password="wrong password").status_code, 401)
        blocked = self.login(password="wrong password")
        self.assertEqual(blocked.status_code, 429)
        self.assertIn("retry-after", {k.lower() for k in blocked.headers})
        # Even the correct password waits now.
        self.assertEqual(self.login().status_code, 429)

    def test_a_weaker_hash_is_upgraded_on_a_successful_login(self) -> None:
        set_hash = self.patch("app.db_users.set_password_hash", return_value=True)
        with patch("app.auth_routes.needs_rehash", return_value=True):
            self.assertEqual(self.login().status_code, 204)
        set_hash.assert_called_once()
        self.assertTrue(set_hash.call_args.args[2].startswith("scrypt$"))


class LogoutAndSessionTests(AuthRouteTestCase):
    def test_anonymous_session_is_null_null(self) -> None:
        response = self.client.get("/api/auth/session")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"user": None, "participant": None})

    def test_session_describes_both_principals_without_the_participant_id(self) -> None:
        self.act_as(
            Principal(
                user=professor_principal(COURSE, OTHER_COURSE).user,
                participant=participant_principal(COURSE).participant,
            )
        )
        body = self.client.get("/api/auth/session").json()
        self.assertEqual(body["user"]["email"], "prof@uw.edu")
        self.assertEqual(body["user"]["role"], "professor")
        self.assertEqual(body["user"]["courseIds"], sorted([COURSE, OTHER_COURSE]))
        self.assertEqual(body["participant"], {"courseId": COURSE})

    def test_logout_revokes_every_session_and_clears_both_cookies(self) -> None:
        revoke = self.patch("app.db_sessions.revoke_session", return_value=True)
        self.act_as(
            Principal(
                user=professor_principal(COURSE, session_id="s-staff").user,
                participant=participant_principal(COURSE, session_id="s-part").participant,
            )
        )
        response = self.client.post("/api/auth/logout", headers=CSRF)
        self.assertEqual(response.status_code, 204)
        revoked = sorted(call.args[1] for call in revoke.call_args_list)
        self.assertEqual(revoked, ["s-part", "s-staff"])
        cleared = " ".join(response.headers.get_list("set-cookie")).lower()
        self.assertIn(f"{STAFF_COOKIE_NAME.lower()}=", cleared)
        self.assertIn(f"{PARTICIPANT_COOKIE_NAME.lower()}=", cleared)
        self.assertIn("max-age=0", cleared)

    def test_logout_while_anonymous_is_still_a_clean_204(self) -> None:
        self.assertEqual(self.client.post("/api/auth/logout", headers=CSRF).status_code, 204)


class JoinTests(AuthRouteTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.find = self.patch("app.db_invitations.find_by_code", return_value=student_invitation())
        self.consume = self.patch(
            "app.db_invitations.consume", return_value=student_invitation(useCount=4)
        )
        self.create_participant = self.patch(
            "app.db_participants.create_participant",
            return_value={"participantId": "part-new", "courseId": COURSE},
        )
        self.touch = self.patch("app.db_participants.touch_participant", return_value=None)
        self.revoke = self.patch("app.db_sessions.revoke_session", return_value=True)

    def join(self, code: str = "7k4p9x", headers: dict[str, str] | None = None):
        # Each call is a fresh browser: a cookie a previous join set must not be
        # presented again, or the (unpatched) session lookup would run.
        self.client.cookies.clear()
        return self.client.post(
            "/api/auth/join",
            json={"code": code},
            headers=CSRF if headers is None else headers,
        )

    def test_a_valid_code_creates_a_participant_and_a_session(self) -> None:
        response = self.join()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {"courseId": COURSE, "courseName": "CSS 360", "alreadyJoined": False}
        )
        # Normalised before lookup.
        self.assertEqual(self.find.call_args.args[1], "7K4P9X")
        self.consume.assert_called_once()
        self.create_participant.assert_called_once()
        self.assertEqual(self.create_participant.call_args.kwargs["course_id"], COURSE)
        self.assertEqual(self.create_session.call_args.kwargs["kind"], "participant")
        self.assertEqual(self.create_session.call_args.kwargs["participant_id"], "part-new")
        cookie = next(
            header for header in response.headers.get_list("set-cookie")
            if header.startswith(f"{PARTICIPANT_COOKIE_NAME}=")
        )
        for attribute in ("httponly", "secure", "samesite=lax", "path=/api"):
            self.assertIn(attribute, cookie.lower())

    def test_each_redemption_is_a_new_participant(self) -> None:
        self.join()
        self.join()
        self.assertEqual(self.create_participant.call_count, 2)
        self.assertEqual(self.consume.call_count, 2)

    def test_invalid_revoked_expired_and_malformed_codes_look_the_same(self) -> None:
        responses = []
        self.find.return_value = None
        responses.append(self.join("ZZZZZZ"))
        self.find.return_value = student_invitation(status="revoked")
        responses.append(self.join())
        self.find.return_value = student_invitation(status="expired")
        responses.append(self.join())
        responses.append(self.join("not-a-code-at-all"))
        for response in responses:
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json()["detail"], CODE_INVALID_DETAIL)
        self.create_participant.assert_not_called()
        self.create_session.assert_not_called()

    def test_a_lost_consume_race_is_the_same_neutral_404(self) -> None:
        self.consume.return_value = None
        self.assertEqual(self.join().status_code, 404)
        self.create_participant.assert_not_called()

    def test_guessing_is_throttled_per_client(self) -> None:
        self.find.return_value = None
        for _ in range(rate_limit.JOIN_CODE_PER_CLIENT.limit):
            self.assertEqual(self.join().status_code, 404)
        self.assertEqual(self.join().status_code, 429)

    def test_a_correct_code_after_typos_resets_the_client_budget(self) -> None:
        good = student_invitation()
        self.find.side_effect = [None, None, good]
        self.assertEqual(self.join().status_code, 404)
        self.assertEqual(self.join().status_code, 404)
        self.assertEqual(self.join().status_code, 200)
        self.find.side_effect = None
        self.find.return_value = None
        for _ in range(rate_limit.JOIN_CODE_PER_CLIENT.limit):
            self.assertEqual(self.join().status_code, 404)
        self.assertEqual(self.join().status_code, 429)

    def test_rejoining_the_same_course_keeps_the_existing_participant(self) -> None:
        self.act_as(participant_principal(COURSE))
        response = self.join()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["alreadyJoined"])
        self.consume.assert_not_called()
        self.create_participant.assert_not_called()
        self.create_session.assert_not_called()
        self.assertFalse(
            any(h.startswith(PARTICIPANT_COOKIE_NAME) for h in response.headers.get_list("set-cookie"))
        )

    def test_joining_another_course_replaces_the_participant_session(self) -> None:
        self.act_as(participant_principal(OTHER_COURSE, session_id="old-session"))
        response = self.join()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["alreadyJoined"])
        self.create_participant.assert_called_once()
        self.revoke.assert_called_once()
        self.assertEqual(self.revoke.call_args.args[1], "old-session")

    def test_the_csrf_header_is_required(self) -> None:
        self.assertEqual(self.join(headers={}).status_code, 403)


class InvitationPreviewTests(AuthRouteTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.find = self.patch(
            "app.db_invitations.find_by_token_hash", return_value=token_invitation()
        )
        self.patch(
            "app.db_courses.get_course",
            return_value={"courseId": COURSE, "metadata": {"name": "CSS 360"}},
        )
        self.token = generate_token()

    def test_an_active_professor_invitation_previews_its_courses(self) -> None:
        response = self.client.get(f"/api/auth/invitations/{self.token}")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "professor")
        self.assertEqual(body["courses"], [{"courseId": COURSE, "name": "CSS 360"}])

    def test_used_revoked_expired_and_unknown_are_the_same_404(self) -> None:
        for record in (
            None,
            token_invitation(status="used"),
            token_invitation(status="revoked"),
            token_invitation(status="expired"),
            student_invitation(),  # never previewable through this route
        ):
            with self.subTest(record=record and record.get("status")):
                self.find.return_value = record
                response = self.client.get(f"/api/auth/invitations/{generate_token()}")
                self.assertEqual(response.status_code, 404)
                self.assertEqual(response.json()["detail"], INVITATION_INVALID_DETAIL)

    def test_an_implausible_token_never_reaches_the_database(self) -> None:
        response = self.client.get("/api/auth/invitations/abc")
        self.assertEqual(response.status_code, 404)
        self.find.assert_not_called()

    def test_a_reset_invitation_names_the_account(self) -> None:
        self.find.return_value = token_invitation("reset", targetUserId=USER_ID)
        self.patch("app.db_users.get_user", return_value=USER_RECORD)
        body = self.client.get(f"/api/auth/invitations/{self.token}").json()
        self.assertEqual(body["kind"], "reset")
        self.assertEqual(body["targetEmail"], "prof@uw.edu")


class AcceptInvitationTests(AuthRouteTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.find = self.patch(
            "app.db_invitations.find_by_token_hash", return_value=token_invitation()
        )
        self.create_user = self.patch("app.db_users.create_user", return_value=USER_RECORD)
        self.get_user = self.patch("app.db_users.get_user", return_value=USER_RECORD)
        self.add_membership = self.patch("app.db_memberships.add_membership", return_value=True)
        self.patch("app.db_memberships.list_course_ids_for_user", return_value=[COURSE])
        self.consume = self.patch(
            "app.db_invitations.consume", return_value=token_invitation(status="used")
        )
        self.set_hash = self.patch("app.db_users.set_password_hash", return_value=True)
        self.revoke_all = self.patch("app.db_sessions.revoke_sessions_for_user", return_value=1)
        self.token = generate_token()

    def accept(self, **body: Any):
        payload = {"email": "prof@uw.edu", "displayName": "Prof", "password": PASSWORD, **body}
        return self.client.post(
            f"/api/auth/invitations/{self.token}/accept", json=payload, headers=CSRF
        )

    def test_a_professor_invitation_creates_the_account_with_its_courses(self) -> None:
        response = self.accept()
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["user"]["role"], "professor")
        self.assertEqual(body["user"]["courseIds"], [COURSE])
        self.assertEqual(self.create_user.call_args.kwargs["role"], "professor")
        self.assertTrue(self.create_user.call_args.kwargs["password_hash"].startswith("scrypt$"))
        self.assertEqual(self.add_membership.call_args.kwargs["course_id"], COURSE)
        self.assertEqual(self.add_membership.call_args.kwargs["granted_by"], "admin-1")
        self.consume.assert_called_once()
        self.assertEqual(self.consume.call_args.kwargs["accepted_by_user_id"], USER_ID)
        self.record_action.assert_called_once()
        self.assertEqual(self.record_action.call_args.kwargs["action"], "invitation.accept")
        self.assertTrue(
            any(h.startswith(f"{STAFF_COOKIE_NAME}=") for h in response.headers.get_list("set-cookie"))
        )

    def test_an_admin_invitation_creates_an_administrator_with_no_memberships(self) -> None:
        self.find.return_value = token_invitation("admin")
        self.get_user.return_value = {**USER_RECORD, "role": "admin"}
        response = self.accept()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user"]["role"], "admin")
        self.assertEqual(self.create_user.call_args.kwargs["role"], "admin")
        self.add_membership.assert_not_called()

    def test_the_role_comes_from_the_invitation_not_the_request(self) -> None:
        """A professor invitation accepted with `role: admin` in the body is still a professor."""
        response = self.accept(role="admin", courseIds=[OTHER_COURSE])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.create_user.call_args.kwargs["role"], "professor")
        self.assertEqual(self.add_membership.call_args.kwargs["course_id"], COURSE)

    def test_a_weak_password_is_refused_before_anything_is_created(self) -> None:
        response = self.accept(password="short")
        self.assertEqual(response.status_code, 422)
        self.create_user.assert_not_called()

    def test_missing_email_or_name_is_a_422(self) -> None:
        self.assertEqual(self.accept(email="").status_code, 422)
        self.assertEqual(self.accept(displayName=" ").status_code, 422)
        self.create_user.assert_not_called()

    def test_an_existing_email_is_a_409(self) -> None:
        self.create_user.side_effect = db_users.UserAlreadyExistsError("taken")
        self.assertEqual(self.accept().status_code, 409)
        self.consume.assert_not_called()
        self.create_session.assert_not_called()

    def test_a_used_invitation_is_a_404_and_creates_nothing(self) -> None:
        self.find.return_value = token_invitation(status="used")
        self.assertEqual(self.accept().status_code, 404)
        self.create_user.assert_not_called()

    def test_losing_the_consume_race_creates_no_session(self) -> None:
        self.consume.return_value = None
        self.assertEqual(self.accept().status_code, 404)
        self.create_session.assert_not_called()
        self.record_action.assert_not_called()

    def test_a_reset_invitation_sets_the_password_and_ends_other_sessions(self) -> None:
        self.find.return_value = token_invitation("reset", targetUserId=USER_ID)
        response = self.accept(email=None, displayName=None)
        self.assertEqual(response.status_code, 200)
        self.create_user.assert_not_called()
        self.set_hash.assert_called_once()
        self.assertEqual(self.set_hash.call_args.args[1], USER_ID)
        self.revoke_all.assert_called_once()
        self.assertEqual(self.consume.call_args.kwargs["accepted_by_user_id"], USER_ID)
        self.assertEqual(self.record_action.call_args.kwargs["action"], "invitation.accept_reset")

    def test_the_csrf_header_is_required(self) -> None:
        response = self.client.post(
            f"/api/auth/invitations/{self.token}/accept",
            json={"email": "a@b.c", "displayName": "x", "password": PASSWORD},
        )
        self.assertEqual(response.status_code, 403)


class PasswordChangeTests(AuthRouteTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.act_as(professor_principal(COURSE, session_id="keep-me"))
        self.patch(
            "app.db_users.find_credentials",
            return_value={"userId": USER_ID, "role": "professor", "passwordHash": PASSWORD_HASH, "disabled": False},
        )
        self.set_hash = self.patch("app.db_users.set_password_hash", return_value=True)
        self.revoke_all = self.patch("app.db_sessions.revoke_sessions_for_user", return_value=2)

    def change(self, current: str = PASSWORD, new: str = "a brand new passphrase"):
        return self.client.post(
            "/api/auth/password",
            json={"currentPassword": current, "newPassword": new},
            headers=CSRF,
        )

    def test_success_rehashes_and_keeps_only_this_session(self) -> None:
        self.assertEqual(self.change().status_code, 204)
        self.set_hash.assert_called_once()
        self.assertEqual(self.revoke_all.call_args.kwargs["keep_session_id"], "keep-me")

    def test_wrong_current_password_changes_nothing(self) -> None:
        self.assertEqual(self.change(current="nope nope nope").status_code, 400)
        self.set_hash.assert_not_called()

    def test_weak_new_password_is_refused(self) -> None:
        self.assertEqual(self.change(new="short").status_code, 422)

    def test_requires_a_staff_session(self) -> None:
        app.dependency_overrides.clear()
        self.assertEqual(self.change().status_code, 401)


class BootstrapInvitationTests(unittest.TestCase):
    def _row(self) -> dict[str, Any]:
        return {
            "invitation_id": "inv-boot", "kind": "admin", "code": None, "course_id": None,
            "target_user_id": None, "label": "bootstrap", "created_by": None,
            "created_at": NOW, "expires_at": NOW + timedelta(hours=1), "max_uses": 1,
            "use_count": 0, "revoked_at": None, "revoked_by": None, "accepted_at": None,
            "accepted_by_user_id": None,
        }

    def test_mints_a_one_hour_single_use_admin_invitation_with_no_actor(self) -> None:
        conn = FakeConnection(results=[[{"total": 0}], 1, [self._row()], 1])
        minted = mint_bootstrap_invitation(conn, now=NOW)
        self.assertTrue(minted["path"].startswith("/invite/"))
        self.assertEqual(minted["kind"], "admin")
        self.assertEqual(minted["expiresAt"], (NOW + timedelta(hours=1)).isoformat())
        insert_sql, insert_params = conn.statements[1]
        self.assertIn("INSERT INTO invitations", insert_sql)
        self.assertEqual(insert_params[1], "admin")
        self.assertIsNone(insert_params[5])  # created_by
        audit_sql, audit_params = conn.statements[3]
        self.assertIn("INSERT INTO admin_actions", audit_sql)
        self.assertIsNone(audit_params["actor_user_id"])
        self.assertEqual(audit_params["action"], "bootstrap.admin_invite")
        # The token itself is returned to the operator and stored only hashed.
        self.assertNotIn(minted["token"], str(insert_params))

    def test_refuses_while_an_administrator_exists_unless_forced(self) -> None:
        with self.assertRaises(AdminAlreadyExistsError):
            mint_bootstrap_invitation(FakeConnection(results=[[{"total": 1}]]), now=NOW)
        conn = FakeConnection(results=[[{"total": 1}], 1, [self._row()], 1])
        minted = mint_bootstrap_invitation(conn, force=True, now=NOW)
        self.assertIn("token", minted)


if __name__ == "__main__":
    unittest.main()
