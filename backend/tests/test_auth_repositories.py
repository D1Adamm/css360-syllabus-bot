"""SQL of the identity repositories, driven by the recording fake.

Same approach as `test_db_repositories.py`: no server, only the statements.
What matters here is the shape of the guards that live in SQL — the
conditional consume, the conflict-free inserts, the course-scoped lookups —
and that no secret ever leaves a mapper.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from test_db_repositories import FakeConnection

from app import (
    db_admin_actions,
    db_invitations,
    db_memberships,
    db_participants,
    db_sessions,
    db_users,
)

NOW = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
COURSE = "css-360-winter-2026-a7rp"
USER_ROW = {
    "user_id": "11111111-1111-1111-1111-111111111111",
    "email": "prof@uw.edu",
    "display_name": "Prof",
    "role": "professor",
    "created_at": NOW,
    "disabled_at": None,
    "last_login_at": None,
    "created_via_invitation_id": None,
}


class UserRepositoryTests(unittest.TestCase):
    def test_create_user_lowercases_email_and_relies_on_the_unique_index(self) -> None:
        conn = FakeConnection(results=[1, [USER_ROW]])
        created = db_users.create_user(
            conn,
            email="  Prof@UW.edu ",
            display_name="Prof",
            role="professor",
            password_hash="scrypt$…",
            now=NOW,
        )
        sql, params = conn.statements[0]
        self.assertIn("ON CONFLICT (lower(email)) DO NOTHING", sql)
        self.assertEqual(params["email"], "prof@uw.edu")
        self.assertEqual(created["email"], "prof@uw.edu")
        self.assertNotIn("passwordHash", created)
        self.assertNotIn("password_hash", created)

    def test_a_duplicate_email_is_an_error_not_a_second_account(self) -> None:
        conn = FakeConnection(results=[0])
        with self.assertRaises(db_users.UserAlreadyExistsError):
            db_users.create_user(
                conn, email="prof@uw.edu", display_name="P", role="professor",
                password_hash="h", now=NOW,
            )

    def test_unknown_roles_and_bad_emails_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            db_users.create_user(
                FakeConnection(), email="a@b", display_name="x", role="student",
                password_hash="h",
            )
        with self.assertRaises(ValueError):
            db_users.create_user(
                FakeConnection(), email="not-an-email", display_name="x",
                role="professor", password_hash="h",
            )

    def test_find_credentials_is_the_only_reader_of_the_hash(self) -> None:
        conn = FakeConnection(results=[[{
            "user_id": USER_ROW["user_id"], "role": "admin",
            "password_hash": "scrypt$…", "disabled_at": None,
        }]])
        found = db_users.find_credentials(conn, "Prof@UW.edu")
        assert found is not None
        self.assertEqual(found["passwordHash"], "scrypt$…")
        self.assertFalse(found["disabled"])
        self.assertIn("lower(email) = %s", conn.statements[0][0])
        self.assertEqual(conn.statements[0][1], ("prof@uw.edu",))

    def test_public_user_record_reports_disabled_state(self) -> None:
        record = db_users.map_user({**USER_ROW, "disabled_at": NOW})
        self.assertTrue(record["disabled"])
        self.assertEqual(record["userId"], USER_ROW["user_id"])


class MembershipRepositoryTests(unittest.TestCase):
    def test_add_membership_is_conflict_free(self) -> None:
        conn = FakeConnection(results=[1])
        self.assertTrue(
            db_memberships.add_membership(
                conn, course_id=COURSE, user_id="u", granted_by="a", now=NOW
            )
        )
        sql, params = conn.statements[0]
        self.assertIn("ON CONFLICT (course_id, user_id) DO NOTHING", sql)
        self.assertEqual(params[0], COURSE)
        self.assertEqual(params[1], "u")

    def test_add_membership_reports_an_existing_row_as_not_added(self) -> None:
        conn = FakeConnection(results=[0])
        self.assertFalse(
            db_memberships.add_membership(conn, course_id=COURSE, user_id="u", granted_by=None)
        )

    def test_course_ids_are_validated_before_sql(self) -> None:
        with self.assertRaises(ValueError):
            db_memberships.add_membership(
                FakeConnection(), course_id="Bad Id", user_id="u", granted_by=None
            )
        with self.assertRaises(ValueError):
            db_memberships.remove_membership(FakeConnection(), course_id="../x", user_id="u")

    def test_remove_binds_both_keys(self) -> None:
        conn = FakeConnection(results=[1])
        db_memberships.remove_membership(conn, course_id=COURSE, user_id="u")
        sql, params = conn.statements[0]
        self.assertIn("WHERE course_id = %s AND user_id = %s", sql)
        self.assertEqual(params, (COURSE, "u"))


class ParticipantRepositoryTests(unittest.TestCase):
    def test_create_participant_binds_the_validated_course(self) -> None:
        conn = FakeConnection(results=[
            1,
            [{
                "participant_id": "p-1", "course_id": COURSE, "invitation_id": "inv-1",
                "created_at": NOW, "last_seen_at": NOW,
            }],
        ])
        created = db_participants.create_participant(
            conn, course_id=COURSE, invitation_id="inv-1", now=NOW
        )
        self.assertEqual(created["courseId"], COURSE)
        self.assertEqual(created["participantId"], "p-1")
        insert_sql, insert_params = conn.statements[0]
        self.assertIn("INSERT INTO participants", insert_sql)
        self.assertEqual(insert_params[1], COURSE)

    def test_participant_rows_carry_nothing_identifying(self) -> None:
        record = db_participants.map_participant({
            "participant_id": "p-1", "course_id": COURSE, "invitation_id": None,
            "created_at": NOW, "last_seen_at": None,
        })
        self.assertEqual(set(record), {"participantId", "courseId", "createdAt"})

    def test_invalid_course_never_reaches_sql(self) -> None:
        conn = FakeConnection()
        with self.assertRaises(ValueError):
            db_participants.create_participant(conn, course_id="Nope!", invitation_id=None)
        self.assertEqual(conn.statements, [])


class SessionRepositoryTests(unittest.TestCase):
    def test_create_session_writes_hash_kind_and_deadline(self) -> None:
        conn = FakeConnection(results=[1])
        created = db_sessions.create_session(
            conn, kind="user", token_hash="abc", lifetime=timedelta(days=7),
            now=NOW, user_id="u-1",
        )
        sql, params = conn.statements[0]
        self.assertIn("INSERT INTO auth_sessions", sql)
        self.assertEqual(params[1], "abc")
        self.assertEqual(params[2], "user")
        self.assertEqual(params[3], "u-1")
        self.assertIsNone(params[4])
        self.assertEqual(created["expiresAt"], (NOW + timedelta(days=7)).isoformat())

    def test_a_session_is_exactly_one_principal(self) -> None:
        with self.assertRaises(ValueError):
            db_sessions.create_session(
                FakeConnection(), kind="user", token_hash="h", lifetime=timedelta(1),
                user_id="u", participant_id="p",
            )
        with self.assertRaises(ValueError):
            db_sessions.create_session(
                FakeConnection(), kind="participant", token_hash="h", lifetime=timedelta(1),
            )

    def test_lookups_filter_by_kind_and_join_the_principal(self) -> None:
        conn = FakeConnection(results=[[], []])
        db_sessions.find_staff_session(conn, "h")
        db_sessions.find_participant_session(conn, "h")
        staff_sql = conn.statements[0][0]
        participant_sql = conn.statements[1][0]
        self.assertIn("principal_kind = 'user'", staff_sql)
        self.assertIn("JOIN users", staff_sql)
        self.assertIn("principal_kind = 'participant'", participant_sql)
        self.assertIn("JOIN participants", participant_sql)

    def test_revoking_everything_but_the_current_session(self) -> None:
        conn = FakeConnection(results=[2])
        count = db_sessions.revoke_sessions_for_user(conn, "u", NOW, keep_session_id="keep")
        self.assertEqual(count, 2)
        sql, params = conn.statements[0]
        self.assertIn("session_id <> %s", sql)
        self.assertEqual(params[-1], "keep")


def _invitation_row(**overrides):
    row = {
        "invitation_id": "inv-1", "kind": "professor", "code": None, "course_id": None,
        "target_user_id": None, "label": None, "created_by": "admin-1",
        "created_at": NOW, "expires_at": NOW + timedelta(days=1), "max_uses": 1,
        "use_count": 0, "revoked_at": None, "revoked_by": None, "accepted_at": None,
        "accepted_by_user_id": None,
    }
    row.update(overrides)
    return row


class InvitationRepositoryTests(unittest.TestCase):
    def test_status_is_derived_from_the_row(self) -> None:
        self.assertEqual(db_invitations.status_of(_invitation_row(), NOW), "active")
        self.assertEqual(
            db_invitations.status_of(_invitation_row(revoked_at=NOW), NOW), "revoked"
        )
        self.assertEqual(
            db_invitations.status_of(_invitation_row(expires_at=NOW), NOW), "expired"
        )
        self.assertEqual(
            db_invitations.status_of(_invitation_row(use_count=1), NOW), "used"
        )
        reusable = _invitation_row(kind="student", max_uses=None, use_count=40, expires_at=None)
        self.assertEqual(db_invitations.status_of(reusable, NOW), "active")

    def test_consume_is_one_conditional_update(self) -> None:
        conn = FakeConnection(results=[[_invitation_row(use_count=1, accepted_at=NOW)]])
        consumed = db_invitations.consume(conn, "inv-1", now=NOW, accepted_by_user_id="u")
        assert consumed is not None
        self.assertEqual(consumed["status"], "used")
        sql, params = conn.statements[0]
        self.assertEqual(len(conn.statements), 1)
        self.assertIn("UPDATE invitations", sql)
        self.assertIn("revoked_at IS NULL", sql)
        self.assertIn("expires_at IS NULL OR expires_at > %(now)s", sql)
        self.assertIn("max_uses IS NULL OR use_count < max_uses", sql)
        self.assertIn("RETURNING", sql)
        self.assertEqual(params["invitation_id"], "inv-1")

    def test_consume_returns_none_when_nothing_was_redeemable(self) -> None:
        conn = FakeConnection(results=[[]])
        self.assertIsNone(db_invitations.consume(conn, "inv-1", now=NOW))

    def test_student_code_collision_is_reported_for_a_retry(self) -> None:
        conn = FakeConnection(results=[0])
        with self.assertRaises(db_invitations.CodeCollisionError):
            db_invitations.create_student_invitation(
                conn, course_id=COURSE, code="7K4P9X", created_by="u", now=NOW
            )
        self.assertIn("ON CONFLICT (code) WHERE code IS NOT NULL DO NOTHING", conn.statements[0][0])

    def test_student_invitation_is_reusable_by_default(self) -> None:
        conn = FakeConnection(results=[1, [_invitation_row(
            kind="student", code="7K4P9X", course_id=COURSE, max_uses=None, expires_at=None
        )]])
        created = db_invitations.create_student_invitation(
            conn, course_id=COURSE, code="7K4P9X", created_by="u", now=NOW
        )
        self.assertEqual(created["code"], "7K4P9X")
        self.assertNotIn("maxUses", created)
        self.assertEqual(created["status"], "active")
        # The fake normalises whitespace; the row is inserted with NULL uses.
        self.assertIn(", NULL)", conn.statements[0][0])

    def test_token_invitations_are_single_use_and_only_professors_get_courses(self) -> None:
        conn = FakeConnection(results=[1, 1, [_invitation_row()]])
        created = db_invitations.create_token_invitation(
            conn, kind="professor", token_hash="h", created_by="a",
            expires_at=NOW + timedelta(days=7), now=NOW, course_ids=[COURSE],
        )
        insert_sql = conn.statements[0][0]
        self.assertIn("max_uses ) VALUES", insert_sql)
        self.assertIn(", 1)", insert_sql)
        self.assertIn("INSERT INTO invitation_course_grants", conn.statements[1][0])
        self.assertEqual(created["courseIds"], [COURSE])

        with self.assertRaises(ValueError):
            db_invitations.create_token_invitation(
                FakeConnection(), kind="admin", token_hash="h", created_by="a",
                expires_at=NOW, course_ids=[COURSE],
            )
        with self.assertRaises(ValueError):
            db_invitations.create_token_invitation(
                FakeConnection(), kind="reset", token_hash="h", created_by="a", expires_at=NOW,
            )
        with self.assertRaises(ValueError):
            db_invitations.create_token_invitation(
                FakeConnection(), kind="student", token_hash="h", created_by="a", expires_at=NOW,
            )

    def test_mapped_records_never_carry_the_token_hash(self) -> None:
        record = db_invitations.map_invitation({**_invitation_row(), "token_hash": "secret"}, NOW)
        self.assertNotIn("tokenHash", record)
        self.assertNotIn("token_hash", record)
        self.assertNotIn("secret", str(record))

    def test_find_by_code_is_scoped_to_student_invitations(self) -> None:
        conn = FakeConnection(results=[[]])
        db_invitations.find_by_code(conn, "7K4P9X", now=NOW)
        sql, params = conn.statements[0]
        self.assertIn("i.kind = 'student'", sql)
        self.assertIn("JOIN courses", sql)
        self.assertEqual(params, ("7K4P9X",))

    def test_revoking_active_codes_is_course_scoped(self) -> None:
        conn = FakeConnection(results=[2])
        count = db_invitations.revoke_active_student_invitations(
            conn, COURSE, revoked_by="u", now=NOW
        )
        self.assertEqual(count, 2)
        sql, params = conn.statements[0]
        self.assertIn("course_id = %s AND kind = 'student' AND revoked_at IS NULL", sql)
        self.assertEqual(params[2], COURSE)


class AdminActionRepositoryTests(unittest.TestCase):
    def test_secrets_are_scrubbed_from_detail_at_any_depth(self) -> None:
        cleaned = db_admin_actions.scrub_detail({
            "label": "CSS 360 section A",
            "token": "never",
            "passwordHash": "never",
            "code": "7K4P9X",
            "nested": {"sessionCookie": "never", "kept": 1},
            "courseId": COURSE,
        })
        self.assertEqual(cleaned, {"label": "CSS 360 section A", "nested": {"kept": 1}, "courseId": COURSE})

    def test_record_action_binds_everything_and_names_the_actor(self) -> None:
        conn = FakeConnection(results=[1])
        db_admin_actions.record_action(
            conn, actor_user_id="a", actor_role="admin", action="invitation.create",
            target_kind="invitation", target_id="inv-1", course_id=COURSE,
            detail={"kind": "professor", "token": "never"}, now=NOW,
        )
        sql, params = conn.statements[0]
        self.assertIn("INSERT INTO admin_actions", sql)
        self.assertEqual(params["actor_user_id"], "a")
        self.assertEqual(params["action"], "invitation.create")
        self.assertNotIn("never", str(params["detail"].obj))

    def test_an_empty_action_name_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            db_admin_actions.record_action(
                FakeConnection(), actor_user_id=None, actor_role=None, action="  "
            )


if __name__ == "__main__":
    unittest.main()
