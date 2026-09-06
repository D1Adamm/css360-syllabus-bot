"""Opt-in: the identity SQL against a real PostgreSQL.

Skipped unless TEST_DATABASE_URL names a throwaway database that already has
`db/schema.sql` applied. Everything else in the suite runs against fakes; this
file is where the properties only a database can prove are proven:

- the constraints hold (one account per email, one session per token, a
  session is exactly one principal, an invitation has the shape its kind needs);
- a single-use invitation consumed by two connections at once is redeemed by
  exactly one of them;
- erasing a participant anonymises its ratings and contributions rather than
  deleting them, and deleting a course cascades to its codes and participants;
- the whole join flow — code, participant, session — round-trips through the
  repositories and the principal resolver.

Every test runs inside a transaction that is rolled back, so the database is
as empty afterwards as before, and tests can run in any order.

Run it with the container the runbook describes:

    TEST_DATABASE_URL=postgresql://tester:tester@127.0.0.1:55432/syllabus_bot_test \\
        .venv/bin/python -m pytest tests/test_integration_postgres.py -q
"""

from __future__ import annotations

import os
import threading
import unittest
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import pytest

from app import (
    db_admin_actions,
    db_courses,
    db_evaluations,
    db_invitations,
    db_memberships,
    db_participants,
    db_seeds,
    db_sessions,
    db_users,
)
from app.auth.codes import generate_code
from app.auth.passwords import hash_password
from app.auth.tokens import generate_token, hash_token

pytestmark = pytest.mark.auth

#: Captured at import, before the conftest fixture that strips it for every
#: other test. The fixture below puts it back for these.
DSN = (os.environ.get("TEST_DATABASE_URL") or "").strip()

COURSE_A = "css-360-winter-2026-a7rp"
COURSE_B = "css-350-spring-2026-n3h9"


def _connect() -> Any:
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(DSN, row_factory=dict_row)


@pytest.fixture()
def integration_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    if not DSN:
        pytest.skip("TEST_DATABASE_URL is not set; integration tests are opt-in.")
    monkeypatch.setenv("TEST_DATABASE_URL", DSN)
    yield


@pytest.mark.usefixtures("integration_env")
class PostgresIntegrationTests(unittest.TestCase):
    """Each test owns one connection and rolls its transaction back."""

    def setUp(self) -> None:
        self.conn = _connect()
        self.now = datetime.now(timezone.utc)
        self.addCleanup(self._teardown)
        self._seed_courses()

    def _teardown(self) -> None:
        self.conn.rollback()
        self.conn.close()

    def _seed_courses(self) -> None:
        for course_id, name in ((COURSE_A, "CSS 360"), (COURSE_B, "CSS 350")):
            db_courses.create_course(
                self.conn,
                course_id,
                {"name": name, "title": "T", "term": "Winter 2026", "instructorName": "I"},
            )

    def _user(self, email: str = "prof@uw.edu", role: str = "professor") -> dict[str, Any]:
        return db_users.create_user(
            self.conn,
            email=email,
            display_name="Prof",
            role=role,
            password_hash=hash_password("a fine passphrase"),
            now=self.now,
        )

    # ---- constraints ----

    def test_one_account_per_email_case_insensitively(self) -> None:
        self._user("Prof@UW.edu")
        with self.assertRaises(db_users.UserAlreadyExistsError):
            self._user("prof@uw.edu")

    def test_a_session_is_exactly_one_principal(self) -> None:
        user = self._user()
        participant = db_participants.create_participant(
            self.conn, course_id=COURSE_A, invitation_id=None, now=self.now
        )
        with self.conn.cursor() as cursor:
            with self.assertRaises(Exception):
                with self.conn.transaction():
                    cursor.execute(
                        "INSERT INTO auth_sessions (session_id, token_hash, principal_kind, "
                        "user_id, participant_id, created_at, expires_at, last_seen_at) "
                        "VALUES (gen_random_uuid(), 'h', 'user', %s, %s, now(), now(), now())",
                        (user["userId"], participant["participantId"]),
                    )

    def test_session_tokens_are_unique(self) -> None:
        user = self._user()
        token_hash = hash_token(generate_token())
        db_sessions.create_session(
            self.conn, kind="user", token_hash=token_hash, lifetime=timedelta(days=1),
            now=self.now, user_id=user["userId"],
        )
        with self.assertRaises(Exception):
            with self.conn.transaction():
                db_sessions.create_session(
                    self.conn, kind="user", token_hash=token_hash, lifetime=timedelta(days=1),
                    now=self.now, user_id=user["userId"],
                )

    def test_an_invitation_must_have_the_shape_of_its_kind(self) -> None:
        with self.conn.cursor() as cursor:
            # A student code without a course, and a professor token with a code.
            for statement in (
                "INSERT INTO invitations (invitation_id, kind, code, created_at) "
                "VALUES (gen_random_uuid(), 'student', 'AAAAAA', now())",
                "INSERT INTO invitations (invitation_id, kind, code, token_hash, created_at) "
                "VALUES (gen_random_uuid(), 'professor', 'AAAAAA', 'h', now())",
            ):
                with self.subTest(statement=statement[:60]):
                    with self.assertRaises(Exception):
                        with self.conn.transaction():
                            cursor.execute(statement)

    # ---- single use, under contention ----

    def test_a_single_use_invitation_is_redeemed_by_exactly_one_of_two_racers(self) -> None:
        # Committed setup so the racing connections can see it. Cleaned up below.
        admin = self._user("admin@uw.edu", "admin")
        invitation = db_invitations.create_token_invitation(
            self.conn, kind="admin", token_hash=hash_token(generate_token()),
            created_by=admin["userId"], expires_at=self.now + timedelta(hours=1), now=self.now,
        )
        self.conn.commit()
        invitation_id = invitation["invitationId"]
        outcomes: list[bool] = []
        barrier = threading.Barrier(2)

        def race() -> None:
            conn = _connect()
            try:
                barrier.wait(timeout=10)
                result = db_invitations.consume(conn, invitation_id, now=self.now)
                conn.commit()
                outcomes.append(result is not None)
            finally:
                conn.close()

        threads = [threading.Thread(target=race) for _ in range(2)]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)
            self.assertEqual(sorted(outcomes), [False, True])
            final = db_invitations.get_invitation(self.conn, invitation_id, now=self.now)
            assert final is not None
            self.assertEqual(final["useCount"], 1)
            self.assertEqual(final["status"], "used")
        finally:
            with self.conn.cursor() as cursor:
                cursor.execute("DELETE FROM invitations WHERE invitation_id = %s", (invitation_id,))
                cursor.execute("DELETE FROM users WHERE user_id = %s", (admin["userId"],))
                cursor.execute("DELETE FROM courses WHERE course_id = ANY(%s)", ([COURSE_A, COURSE_B],))
            self.conn.commit()

    # ---- erasure and cascade ----

    def test_erasing_a_participant_anonymises_its_research_rows(self) -> None:
        participant = db_participants.create_participant(
            self.conn, course_id=COURSE_A, invitation_id=None, now=self.now
        )
        pid = participant["participantId"]
        evaluation = db_evaluations.create_evaluation(
            self.conn, COURSE_A,
            {"comparisonId": "c", "mostAccurate": "rag", "preferredModel": "rag"},
            participant_id=pid,
        )
        seed = db_seeds.create_seed(
            self.conn, COURSE_A, {"instruction": "Q?", "response": "A."}, participant_id=pid
        )
        self.assertEqual(evaluation["participantId"], pid)
        self.assertEqual(seed["participantId"], pid)

        with self.conn.cursor() as cursor:
            cursor.execute("DELETE FROM participants WHERE participant_id = %s", (pid,))

        kept = db_evaluations.get_evaluation(self.conn, COURSE_A, evaluation["id"])
        assert kept is not None
        self.assertNotIn("participantId", kept)
        kept_seed = db_seeds.get_seed(self.conn, COURSE_A, seed["id"])
        assert kept_seed is not None
        self.assertNotIn("participantId", kept_seed)

    def test_deleting_a_course_takes_its_codes_participants_and_memberships(self) -> None:
        user = self._user()
        db_memberships.add_membership(
            self.conn, course_id=COURSE_A, user_id=user["userId"], granted_by=None, now=self.now
        )
        code = db_invitations.create_student_invitation(
            self.conn, course_id=COURSE_A, code=generate_code(), created_by=user["userId"], now=self.now
        )
        participant = db_participants.create_participant(
            self.conn, course_id=COURSE_A, invitation_id=code["invitationId"], now=self.now
        )
        with self.conn.cursor() as cursor:
            cursor.execute("DELETE FROM courses WHERE course_id = %s", (COURSE_A,))

        self.assertIsNone(db_invitations.get_invitation(self.conn, code["invitationId"]))
        self.assertIsNone(db_participants.get_participant(self.conn, participant["participantId"]))
        self.assertEqual(db_memberships.list_course_ids_for_user(self.conn, user["userId"]), [])
        # The account itself is untouched.
        self.assertIsNotNone(db_users.get_user(self.conn, user["userId"]))

    # ---- the join flow, end to end through the repositories ----

    def test_a_classroom_code_admits_distinct_participants_bound_to_its_course(self) -> None:
        prof = self._user()
        code = db_invitations.create_student_invitation(
            self.conn, course_id=COURSE_A, code=generate_code(), created_by=prof["userId"], now=self.now
        )
        found = db_invitations.find_by_code(self.conn, code["code"], now=self.now)
        assert found is not None
        self.assertEqual(found["status"], "active")
        self.assertEqual(found["courseName"], "CSS 360")

        participants = []
        for _ in range(3):
            self.assertIsNotNone(db_invitations.consume(self.conn, code["invitationId"], now=self.now))
            participant = db_participants.create_participant(
                self.conn, course_id=COURSE_A, invitation_id=code["invitationId"], now=self.now
            )
            token = generate_token()
            db_sessions.create_session(
                self.conn, kind="participant", token_hash=hash_token(token),
                lifetime=timedelta(days=180), now=self.now, participant_id=participant["participantId"],
            )
            row = db_sessions.find_participant_session(self.conn, hash_token(token))
            assert row is not None
            self.assertEqual(row["course_id"], COURSE_A)
            participants.append(participant["participantId"])

        self.assertEqual(len(set(participants)), 3)
        self.assertEqual(db_participants.count_participants_for_invitation(self.conn, code["invitationId"]), 3)
        refreshed = db_invitations.get_invitation(self.conn, code["invitationId"], now=self.now)
        assert refreshed is not None
        self.assertEqual(refreshed["useCount"], 3)
        self.assertEqual(refreshed["status"], "active")  # reusable: still active

        revoked = db_invitations.revoke(self.conn, code["invitationId"], revoked_by=prof["userId"], now=self.now)
        assert revoked is not None
        self.assertEqual(revoked["status"], "revoked")
        self.assertIsNone(db_invitations.consume(self.conn, code["invitationId"], now=self.now))

    def test_a_participants_view_and_a_professors_view_of_the_same_course(self) -> None:
        prof = self._user()
        db_memberships.add_membership(
            self.conn, course_id=COURSE_A, user_id=prof["userId"], granted_by=None, now=self.now
        )
        self.assertEqual(db_memberships.list_course_ids_for_user(self.conn, prof["userId"]), [COURSE_A])
        self.assertEqual(
            [c["courseId"] for c in db_courses.list_courses(self.conn, {COURSE_A})], [COURSE_A]
        )
        self.assertEqual(db_courses.list_courses(self.conn, set()), [])
        both = {c["courseId"] for c in db_courses.list_courses(self.conn, None)}
        self.assertEqual(both, {COURSE_A, COURSE_B})

    def test_the_audit_trail_accepts_every_recorded_action_shape(self) -> None:
        admin = self._user("admin@uw.edu", "admin")
        db_admin_actions.record_action(
            self.conn, actor_user_id=admin["userId"], actor_role="admin",
            action="invitation.create", target_kind="invitation", target_id="x",
            course_id=COURSE_A, detail={"kind": "student", "token": "never stored"}, now=self.now,
        )
        db_admin_actions.record_action(
            self.conn, actor_user_id=None, actor_role="operator", action="bootstrap.admin_invite", now=self.now
        )
        actions = db_admin_actions.list_actions(self.conn, limit=10)
        self.assertEqual([a["action"] for a in actions[:2]], ["bootstrap.admin_invite", "invitation.create"])
        self.assertEqual(actions[1]["detail"], {"kind": "student"})
        self.assertEqual(actions[1]["actorEmail"], "admin@uw.edu")


if __name__ == "__main__":
    unittest.main()
