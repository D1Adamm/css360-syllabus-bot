"""Who may review an example, and whose decisions leave an audit row.

The review endpoints have always let an administrator act on any course — that
is what `require_course_staff` means. What was missing was the record: an
administrator approving, rejecting or rewriting examples in a course they are
not the instructor of is exactly the kind of privileged, cross-course action
the audit trail exists for, and it left nothing there. A professor working
through their own course's queue is doing the job the membership grants and is
not audited, exactly as before.

Both review routes — the operational `/api/courses/{id}/seeds/{seed}/review`
the browser uses and the persistence `/api/db/...` twin — follow the same rule.
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

COURSE = "css-360-winter-2026-a7rp"
OTHER_COURSE = "css-350-spring-2026-n3h9"
SEED_ID = "seed-001"
CSRF = {"X-Requested-With": CSRF_HEADER_VALUE}

STORED = {
    "id": SEED_ID,
    "courseId": COURSE,
    "instruction": "Can I get an extension?",
    "response": "One 48-hour extension per quarter.",
    "question": "Can I get an extension?",
    "answer": "One 48-hour extension per quarter.",
    "category": "Late work",
    "sourceSection": "Late Policy",
    "difficulty": "Medium",
    "directlyAnswered": True,
    "origin": "ai_generated",
    "reviewStatus": "approved",
    "status": "approved",
    "wasEdited": False,
}

OPERATIONAL = f"/api/courses/{COURSE}/seeds/{SEED_ID}/review"
PERSISTENCE = f"/api/db/courses/{COURSE}/seeds/{SEED_ID}/review"


@contextmanager
def _fake_connection(**kwargs: Any) -> Iterator[object]:
    yield object()


def staff(role: str, user_id: str, *course_ids: str) -> Principal:
    return Principal(
        user=StaffUser(
            user_id=user_id,
            email=f"{user_id}@uw.edu",
            display_name=user_id.title(),
            role=role,
            course_ids=frozenset(course_ids),
            session_id="s-1",
        )
    )


ADMIN = staff("admin", "u-admin")
INSTRUCTOR = staff("professor", "u-prof", COURSE)
OTHER_INSTRUCTOR = staff("professor", "u-other", OTHER_COURSE)
STUDENT = Principal(participant=Participant(participant_id="p-1", course_id=COURSE, session_id="s-2"))


class SeedReviewAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app, base_url="https://testserver")
        self.addCleanup(app.dependency_overrides.clear)
        for target in ("app.main.db_connection", "app.db_routes.db_connection"):
            patcher = patch(target, new=_fake_connection)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.review = self._patch("app.main.review_seed", return_value=dict(STORED))
        self.db_review = self._patch("app.db_seeds.review_seed", return_value=dict(STORED))
        self.audit = self._patch("app.db_admin_actions.record_action", return_value=None)

    def _patch(self, target: str, **kwargs: Any) -> Any:
        patcher = patch(target, **kwargs)
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock

    def act_as(self, principal: Principal) -> None:
        app.dependency_overrides[current_principal] = lambda: principal

    def review_as(self, principal: Principal, path: str, body: dict[str, Any] | None = None):
        self.act_as(principal)
        return self.client.post(path, json=body or {"reviewStatus": "approved"}, headers=CSRF)

    # ---- who may review at all ------------------------------------------------

    def test_an_administrator_reviews_any_course(self) -> None:
        for path in (OPERATIONAL, PERSISTENCE):
            with self.subTest(path=path):
                response = self.review_as(ADMIN, path)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["seed"]["reviewStatus"], "approved")

    def test_the_instructor_reviews_their_own_course(self) -> None:
        for path in (OPERATIONAL, PERSISTENCE):
            with self.subTest(path=path):
                self.assertEqual(self.review_as(INSTRUCTOR, path).status_code, 200)

    def test_a_professor_is_refused_on_a_course_they_do_not_teach(self) -> None:
        for path in (OPERATIONAL, PERSISTENCE):
            with self.subTest(path=path):
                self.assertEqual(self.review_as(OTHER_INSTRUCTOR, path).status_code, 403)
        self.review.assert_not_called()
        self.db_review.assert_not_called()

    def test_a_student_cannot_review(self) -> None:
        for path in (OPERATIONAL, PERSISTENCE):
            with self.subTest(path=path):
                self.assertEqual(self.review_as(STUDENT, path).status_code, 401)
        self.review.assert_not_called()
        self.db_review.assert_not_called()

    # ---- whose decisions are audited -----------------------------------------

    def test_an_administrators_decision_is_audited_in_the_same_transaction(self) -> None:
        response = self.review_as(
            ADMIN,
            OPERATIONAL,
            {"reviewStatus": "edited", "question": "Reworded?", "answer": "Reworded."},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.audit.call_count, 1)
        kwargs = self.audit.call_args.kwargs
        self.assertEqual(kwargs["action"], "seed.review")
        self.assertEqual(kwargs["actor_user_id"], "u-admin")
        self.assertEqual(kwargs["actor_role"], "admin")
        self.assertEqual(kwargs["target_kind"], "seed")
        self.assertEqual(kwargs["target_id"], SEED_ID)
        self.assertEqual(kwargs["course_id"], COURSE)
        self.assertEqual(
            kwargs["detail"],
            {"reviewStatus": "edited", "textEdited": True, "notesChanged": False},
        )
        # The audit row and the review share one connection, so they commit or
        # roll back together.
        self.assertIs(self.audit.call_args.args[0], self.review.call_args.args[0])

    def test_the_persistence_route_audits_the_same_way(self) -> None:
        response = self.review_as(ADMIN, PERSISTENCE, {"reviewStatus": "rejected"})
        self.assertEqual(response.status_code, 200)
        kwargs = self.audit.call_args.kwargs
        self.assertEqual(kwargs["action"], "seed.review")
        self.assertEqual(kwargs["detail"]["reviewStatus"], "rejected")
        self.assertFalse(kwargs["detail"]["textEdited"])

    def test_the_audit_detail_never_carries_the_example_text(self) -> None:
        self.review_as(
            ADMIN,
            OPERATIONAL,
            {"reviewStatus": "edited", "question": "Secret wording", "answer": "Also secret"},
        )
        detail = self.audit.call_args.kwargs["detail"]
        self.assertNotIn("Secret wording", str(detail))
        self.assertNotIn("Also secret", str(detail))

    def test_a_professors_own_review_is_not_audited(self) -> None:
        for path in (OPERATIONAL, PERSISTENCE):
            with self.subTest(path=path):
                self.assertEqual(self.review_as(INSTRUCTOR, path).status_code, 200)
        self.audit.assert_not_called()

    def test_nothing_is_audited_when_the_seed_does_not_exist(self) -> None:
        self.review.return_value = None
        self.db_review.return_value = None
        for path in (OPERATIONAL, PERSISTENCE):
            with self.subTest(path=path):
                self.assertEqual(self.review_as(ADMIN, path).status_code, 404)
        self.audit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
