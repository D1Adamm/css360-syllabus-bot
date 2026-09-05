"""The guards, driven through a small FastAPI app with the session store faked.

What is asserted is the contract every protected route in the application
relies on: who gets 401, who gets 403, that course scope comes from the path
and the membership table, that expired, revoked, idle and disabled sessions
are refused, and that a cookie-authenticated mutation without the CSRF header
is refused.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator
from unittest.mock import patch

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.auth import dependencies as deps
from app.auth.principal import Principal
from app.auth.settings import (
    CSRF_HEADER_VALUE,
    PARTICIPANT_COOKIE_NAME,
    STAFF_COOKIE_NAME,
)
from app.auth.tokens import generate_token, hash_token

COURSE_A = "css-360-winter-2026-a7rp"
COURSE_B = "css-350-spring-2026-n3h9"
NOW = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
CSRF = {"X-Requested-With": CSRF_HEADER_VALUE}

app = FastAPI()


@app.get("/user")
def _user(principal: Principal = Depends(deps.require_user)) -> dict[str, Any]:
    return {"email": principal.user.email if principal.user else None}


@app.get("/admin")
def _admin(principal: Principal = Depends(deps.require_admin)) -> dict[str, Any]:
    return {"ok": True}


@app.get("/courses/{course_id}/staff")
def _staff(course_id: str, principal: Principal = Depends(deps.require_course_staff)) -> dict:
    return {"courseId": course_id}


@app.get("/courses/{course_id}/access")
def _access(course_id: str, principal: Principal = Depends(deps.require_course_access)) -> dict:
    return {"courseId": course_id}


@app.get("/courses/{course_id}/participant")
def _participant(
    course_id: str, principal: Principal = Depends(deps.require_participant)
) -> dict:
    return {"participantId": principal.participant.participant_id if principal.participant else None}


@app.post("/courses/{course_id}/write")
def _write(course_id: str, principal: Principal = Depends(deps.require_course_access)) -> dict:
    return {"written": True}


@app.post("/public")
def _public(_: None = Depends(deps.require_csrf)) -> dict:
    return {"ok": True}


@contextmanager
def _fake_connection(**kwargs: Any) -> Iterator[object]:
    yield object()


def _staff_row(
    *,
    role: str = "professor",
    expires_in: timedelta = timedelta(days=1),
    last_seen_ago: timedelta = timedelta(minutes=1),
    revoked: bool = False,
    disabled: bool = False,
) -> dict[str, Any]:
    return {
        "session_id": "sess-staff",
        "expires_at": NOW + expires_in,
        "last_seen_at": NOW - last_seen_ago,
        "revoked_at": NOW if revoked else None,
        "user_id": "user-1",
        "email": "prof@uw.edu",
        "display_name": "Prof",
        "role": role,
        "disabled_at": NOW if disabled else None,
    }


def _participant_row(
    course_id: str = COURSE_A,
    *,
    expires_in: timedelta = timedelta(days=100),
    revoked: bool = False,
) -> dict[str, Any]:
    return {
        "session_id": "sess-part",
        "expires_at": NOW + expires_in,
        "last_seen_at": NOW - timedelta(minutes=1),
        "revoked_at": NOW if revoked else None,
        "participant_id": "part-1",
        "course_id": course_id,
    }


class GuardTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app, base_url="https://testserver")
        self.staff_token = generate_token()
        self.participant_token = generate_token()
        self.staff_rows: dict[str, dict[str, Any] | None] = {}
        self.participant_rows: dict[str, dict[str, Any] | None] = {}
        self.memberships: list[str] = []
        self.revoked: list[str] = []
        self.touched: list[str] = []

        patches = [
            patch("app.auth.dependencies.db_connection", _fake_connection),
            patch("app.auth.dependencies._utc_now", return_value=NOW),
            patch(
                "app.db_sessions.find_staff_session",
                side_effect=lambda conn, token_hash: self.staff_rows.get(token_hash),
            ),
            patch(
                "app.db_sessions.find_participant_session",
                side_effect=lambda conn, token_hash: self.participant_rows.get(token_hash),
            ),
            patch(
                "app.db_memberships.list_course_ids_for_user",
                side_effect=lambda conn, user_id: list(self.memberships),
            ),
            patch(
                "app.db_sessions.revoke_session",
                side_effect=lambda conn, session_id, now=None: self.revoked.append(session_id),
            ),
            patch(
                "app.db_sessions.touch_session",
                side_effect=lambda conn, session_id, now=None: self.touched.append(session_id),
            ),
            patch("app.db_participants.touch_participant", return_value=None),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def sign_in(self, row: dict[str, Any] | None, memberships: list[str] = ()) -> None:
        self.staff_rows[hash_token(self.staff_token)] = row
        self.memberships = list(memberships)
        self.client.cookies.set(STAFF_COOKIE_NAME, self.staff_token)

    def join(self, row: dict[str, Any] | None) -> None:
        self.participant_rows[hash_token(self.participant_token)] = row
        self.client.cookies.set(PARTICIPANT_COOKIE_NAME, self.participant_token)


class AnonymousTests(GuardTestCase):
    def test_everything_protected_is_401(self) -> None:
        for path in ("/user", "/admin", f"/courses/{COURSE_A}/staff",
                     f"/courses/{COURSE_A}/access", f"/courses/{COURSE_A}/participant"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 401)

    def test_a_malformed_course_id_is_400_before_anything_else(self) -> None:
        self.assertEqual(self.client.get("/courses/Bad_Id/staff").status_code, 400)
        self.assertEqual(self.client.get("/courses/Bad_Id/access").status_code, 400)

    def test_an_unknown_cookie_is_anonymous(self) -> None:
        self.client.cookies.set(STAFF_COOKIE_NAME, generate_token())
        self.assertEqual(self.client.get("/user").status_code, 401)

    def test_an_implausible_cookie_never_reaches_the_database(self) -> None:
        with patch("app.db_sessions.find_staff_session") as lookup:
            self.client.cookies.set(STAFF_COOKIE_NAME, "not a token")
            self.assertEqual(self.client.get("/user").status_code, 401)
            lookup.assert_not_called()


class ProfessorTests(GuardTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.sign_in(_staff_row(), memberships=[COURSE_A])

    def test_is_a_user_but_not_an_admin(self) -> None:
        self.assertEqual(self.client.get("/user").json(), {"email": "prof@uw.edu"})
        self.assertEqual(self.client.get("/admin").status_code, 403)

    def test_course_scope_comes_from_the_membership_table(self) -> None:
        self.assertEqual(self.client.get(f"/courses/{COURSE_A}/staff").status_code, 200)
        self.assertEqual(self.client.get(f"/courses/{COURSE_B}/staff").status_code, 403)
        self.assertEqual(self.client.get(f"/courses/{COURSE_A}/access").status_code, 200)
        self.assertEqual(self.client.get(f"/courses/{COURSE_B}/access").status_code, 403)

    def test_staff_is_not_a_participant(self) -> None:
        self.assertEqual(self.client.get(f"/courses/{COURSE_A}/participant").status_code, 403)

    def test_a_recent_session_is_not_touched_but_an_older_one_is(self) -> None:
        self.client.get("/user")
        self.assertEqual(self.touched, [])
        self.sign_in(_staff_row(last_seen_ago=timedelta(minutes=10)), [COURSE_A])
        self.client.get("/user")
        self.assertEqual(self.touched, ["sess-staff"])


class AdminTests(GuardTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.sign_in(_staff_row(role="admin"))

    def test_admin_reaches_everything_without_memberships(self) -> None:
        self.assertEqual(self.client.get("/admin").status_code, 200)
        self.assertEqual(self.client.get(f"/courses/{COURSE_A}/staff").status_code, 200)
        self.assertEqual(self.client.get(f"/courses/{COURSE_B}/staff").status_code, 200)

    def test_admin_is_still_not_a_participant(self) -> None:
        self.assertEqual(self.client.get(f"/courses/{COURSE_A}/participant").status_code, 403)


class ParticipantTests(GuardTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.join(_participant_row(COURSE_A))

    def test_reaches_only_its_own_course(self) -> None:
        self.assertEqual(self.client.get(f"/courses/{COURSE_A}/access").status_code, 200)
        self.assertEqual(self.client.get(f"/courses/{COURSE_B}/access").status_code, 403)
        self.assertEqual(
            self.client.get(f"/courses/{COURSE_A}/participant").json(),
            {"participantId": "part-1"},
        )
        self.assertEqual(self.client.get(f"/courses/{COURSE_B}/participant").status_code, 403)

    def test_is_not_staff(self) -> None:
        self.assertEqual(self.client.get("/user").status_code, 401)
        self.assertEqual(self.client.get("/admin").status_code, 401)
        self.assertEqual(self.client.get(f"/courses/{COURSE_A}/staff").status_code, 401)

    def test_a_professor_may_also_hold_a_participant_session(self) -> None:
        self.sign_in(_staff_row(), memberships=[COURSE_B])
        self.assertEqual(self.client.get(f"/courses/{COURSE_A}/participant").status_code, 200)
        self.assertEqual(self.client.get(f"/courses/{COURSE_B}/staff").status_code, 200)
        self.assertEqual(self.client.get(f"/courses/{COURSE_A}/staff").status_code, 403)


class SessionLifecycleTests(GuardTestCase):
    def test_expired_revoked_and_disabled_sessions_are_anonymous(self) -> None:
        cases = {
            "expired": _staff_row(expires_in=timedelta(seconds=-1)),
            "revoked": _staff_row(revoked=True),
            "disabled": _staff_row(disabled=True),
        }
        for name, row in cases.items():
            with self.subTest(case=name):
                self.sign_in(row, [COURSE_A])
                self.assertEqual(self.client.get("/user").status_code, 401)

    def test_an_idle_session_is_refused_and_retired(self) -> None:
        self.sign_in(_staff_row(last_seen_ago=timedelta(hours=25)), [COURSE_A])
        self.assertEqual(self.client.get("/user").status_code, 401)
        self.assertEqual(self.revoked, ["sess-staff"])

    def test_an_expired_participant_session_is_anonymous(self) -> None:
        self.join(_participant_row(expires_in=timedelta(seconds=-1)))
        self.assertEqual(self.client.get(f"/courses/{COURSE_A}/access").status_code, 401)
        self.join(_participant_row(revoked=True))
        self.assertEqual(self.client.get(f"/courses/{COURSE_A}/access").status_code, 401)


class CsrfTests(GuardTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.join(_participant_row(COURSE_A))

    def test_a_cookie_authenticated_mutation_needs_the_header(self) -> None:
        self.assertEqual(self.client.post(f"/courses/{COURSE_A}/write").status_code, 403)
        self.assertEqual(
            self.client.post(f"/courses/{COURSE_A}/write", headers=CSRF).status_code, 200
        )

    def test_a_wrong_header_value_is_refused(self) -> None:
        response = self.client.post(
            f"/courses/{COURSE_A}/write", headers={"X-Requested-With": "XMLHttpRequest"}
        )
        self.assertEqual(response.status_code, 403)

    def test_a_foreign_origin_is_refused_even_with_the_header(self) -> None:
        response = self.client.post(
            f"/courses/{COURSE_A}/write",
            headers={**CSRF, "Origin": "https://evil.example"},
        )
        self.assertEqual(response.status_code, 403)

    def test_the_sites_own_origin_is_accepted(self) -> None:
        with patch.dict("os.environ", {"CORS_ALLOWED_ORIGINS": "https://aiswe.uwb.edu"}):
            allowed = self.client.post(
                f"/courses/{COURSE_A}/write",
                headers={**CSRF, "Origin": "https://aiswe.uwb.edu"},
            )
            self.assertEqual(allowed.status_code, 200)
            same_host = self.client.post(
                f"/courses/{COURSE_A}/write",
                headers={**CSRF, "Origin": "https://testserver"},
            )
            self.assertEqual(same_host.status_code, 200)

    def test_reads_never_need_the_header(self) -> None:
        self.assertEqual(self.client.get(f"/courses/{COURSE_A}/access").status_code, 200)

    def test_public_session_establishing_routes_need_it_too(self) -> None:
        self.client.cookies.clear()
        self.assertEqual(self.client.post("/public").status_code, 403)
        self.assertEqual(self.client.post("/public", headers=CSRF).status_code, 200)


if __name__ == "__main__":
    unittest.main()
