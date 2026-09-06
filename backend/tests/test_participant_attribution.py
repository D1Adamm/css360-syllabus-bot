"""Research data is attributed to the session's participant, and a student sees
only what a student should.

Three properties, each with the failure it prevents:

- attribution comes from the cookie, not the body — a client cannot rate on
  behalf of another participant or choose its own evaluation id;
- a participant reads only their own ratings and only the reviewed examples
  plus their own contributions, with review detail stripped — a student never
  downloads a classmate's comment or an instructor's notes;
- a participant deletes only what they contributed.

Historical rows with no participant keep working exactly as before: they map,
list and aggregate with `participantId` simply absent.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import db_evaluations, db_seeds
from app.auth.dependencies import current_principal
from app.auth.principal import Participant, Principal, StaffUser
from app.auth.settings import CSRF_HEADER_VALUE
from app.main import app
from app.student_visibility import contribution_payload, seeds_for_participant
from test_db_repositories import FakeConnection

pytestmark = pytest.mark.auth

COURSE = "css-360-winter-2026-a7rp"
CSRF = {"X-Requested-With": CSRF_HEADER_VALUE}
ME = "participant-me"
THEM = "participant-them"
NOW = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)


@contextmanager
def _fake_connection(**kwargs: Any) -> Iterator[object]:
    yield object()


def participant(participant_id: str = ME, course_id: str = COURSE) -> Principal:
    return Principal(
        participant=Participant(participant_id=participant_id, course_id=course_id, session_id="s")
    )


def professor() -> Principal:
    return Principal(
        user=StaffUser(user_id="u-prof", email="p@uw.edu", display_name="P",
                       role="professor", course_ids=frozenset({COURSE}), session_id="s")
    )


def seed(**overrides: Any) -> dict[str, Any]:
    record = {
        "id": "seed-1", "courseId": COURSE, "instruction": "Q?", "response": "A.",
        "question": "Q?", "answer": "A.", "category": "general", "sourceSection": "General",
        "difficulty": "Medium", "directlyAnswered": True, "origin": "ai_generated",
        "sourceChunkIds": ["chunk-1"], "wasEdited": False, "reviewStatus": "generated",
        "reviewNotes": "instructor only", "validation": {"score": 0.9, "reason": "fine"},
        "evidenceQuote": "verbatim syllabus", "factId": "fact-1",
    }
    record.update(overrides)
    return record


def evaluation(**overrides: Any) -> dict[str, Any]:
    record = {
        "id": "eval-1", "courseId": COURSE, "comparisonId": "cmp-1", "mostAccurate": "rag",
        "preferredModel": "rag", "hallucinationFlags": [], "createdAt": NOW.isoformat(),
        "comment": "a private remark",
    }
    record.update(overrides)
    return record


class ProjectionTests(unittest.TestCase):
    def test_reviewed_examples_are_visible_to_every_participant(self) -> None:
        for status in ("approved", "edited"):
            visible = seeds_for_participant([seed(reviewStatus=status)], ME)
            self.assertEqual(len(visible), 1)
            self.assertFalse(visible[0]["mine"])

    def test_unreviewed_and_rejected_examples_are_hidden_unless_mine(self) -> None:
        for status in ("generated", "rejected", None):
            with self.subTest(status=status):
                theirs = seed(reviewStatus=status, participantId=THEM, origin="user")
                mine = seed(reviewStatus=status, participantId=ME, origin="user")
                self.assertEqual(seeds_for_participant([theirs], ME), [])
                visible = seeds_for_participant([mine], ME)
                self.assertEqual(len(visible), 1)
                self.assertTrue(visible[0]["mine"])

    def test_legacy_status_field_is_honoured(self) -> None:
        legacy = seed(reviewStatus=None, status="approved")
        legacy.pop("reviewStatus")
        self.assertEqual(len(seeds_for_participant([legacy], ME)), 1)

    def test_review_detail_is_stripped_from_every_visible_record(self) -> None:
        visible = seeds_for_participant([seed(reviewStatus="approved", participantId=THEM)], ME)[0]
        for field in ("reviewNotes", "validation", "evidenceQuote", "factId",
                      "sourceChunkIds", "originalQuestion", "participantId"):
            self.assertNotIn(field, visible)
        for field in ("id", "instruction", "response", "sourceSection", "origin", "reviewStatus"):
            self.assertIn(field, visible)

    def test_a_contribution_may_set_only_its_content(self) -> None:
        payload = contribution_payload({
            "instruction": "Q?", "response": "A.", "sourceSection": "Late work",
            "origin": "ai_generated", "reviewStatus": "approved", "validation": {"score": 1},
            "reviewNotes": "x", "participantId": THEM, "wasEdited": True,
        })
        self.assertEqual(
            payload, {"instruction": "Q?", "response": "A.", "sourceSection": "Late work", "origin": "user"}
        )


class AttributionRouteTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app, base_url="https://testserver")
        self.addCleanup(app.dependency_overrides.clear)
        for target in ("app.db_routes.db_connection", "app.main.db_connection"):
            patcher = patch(target, new=_fake_connection)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.patch("app.db_courses.course_exists", return_value=True)
        self.create_evaluation = self.patch(
            "app.db_evaluations.create_evaluation", return_value=evaluation(participantId=ME)
        )
        self.list_evaluations = self.patch(
            "app.db_evaluations.list_evaluations", return_value=[evaluation(participantId=ME)]
        )
        self.patch("app.db_evaluations.count_evaluations", return_value=17)
        all_seeds = [
            seed(id="approved-ai", reviewStatus="approved"),
            seed(id="draft-ai", reviewStatus="generated"),
            seed(id="mine", reviewStatus="generated", origin="user", participantId=ME),
            seed(id="theirs", reviewStatus="generated", origin="user", participantId=THEM),
        ]
        self.list_seeds = self.patch("app.db_seeds.list_seeds", return_value=all_seeds)
        # main.py binds the repository function by name, so it is patched there too.
        self.patch("app.main.list_seeds", return_value=all_seeds)
        self.patch("app.db_seeds.count_seeds_by_review_status", return_value={"approved": 1})
        self.patch("app.db_seeds.count_seeds_by_origin", return_value={"ai_generated": 2, "user": 2})
        self.create_seed = self.patch(
            "app.db_seeds.create_seed", return_value=seed(id="new", origin="user", participantId=ME)
        )
        self.get_seed = self.patch("app.db_seeds.get_seed", return_value=seed(id="mine", participantId=ME))
        self.delete_seed = self.patch("app.db_seeds.delete_seed", return_value=True)

    def patch(self, target: str, **kwargs: Any) -> Any:
        patcher = patch(target, **kwargs)
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock

    def act_as(self, principal: Principal) -> None:
        app.dependency_overrides[current_principal] = lambda: principal


class EvaluationAttributionTests(AttributionRouteTestCase):
    def test_the_participant_comes_from_the_session_not_the_body(self) -> None:
        self.act_as(participant(ME))
        response = self.client.post(
            f"/api/db/courses/{COURSE}/evaluations",
            json={"id": "chosen-by-client", "comparisonId": "c", "mostAccurate": "rag",
                  "preferredModel": "rag", "participantId": THEM},
            headers=CSRF,
        )
        self.assertEqual(response.status_code, 201)
        kwargs = self.create_evaluation.call_args.kwargs
        payload = self.create_evaluation.call_args.args[2]
        self.assertEqual(kwargs["participant_id"], ME)
        self.assertNotIn("id", payload)
        self.assertNotIn("participantId", payload)
        # The student's own view carries no participant id.
        self.assertNotIn("participantId", response.json())

    def test_a_participant_lists_only_their_own_ratings(self) -> None:
        self.act_as(participant(ME))
        response = self.client.get(f"/api/db/courses/{COURSE}/evaluations")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.list_evaluations.call_args.kwargs["participant_id"], ME)
        self.assertNotIn("participantId", response.json()["evaluations"][0])

    def test_staff_list_the_whole_course_with_attribution(self) -> None:
        self.act_as(professor())
        response = self.client.get(f"/api/db/courses/{COURSE}/evaluations")
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.list_evaluations.call_args.kwargs["participant_id"])
        self.assertEqual(response.json()["evaluations"][0]["participantId"], ME)

    def test_historical_rows_without_a_participant_still_list(self) -> None:
        self.act_as(professor())
        self.list_evaluations.return_value = [evaluation()]
        body = self.client.get(f"/api/db/courses/{COURSE}/evaluations").json()
        self.assertEqual(body["count"], 1)
        self.assertNotIn("participantId", body["evaluations"][0])

    def test_activity_is_counts_only(self) -> None:
        self.act_as(participant(ME))
        response = self.client.get(f"/api/db/courses/{COURSE}/activity")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {"courseId": COURSE, "contributedQuestions": 2, "evaluations": 17}
        )


class SeedVisibilityTests(AttributionRouteTestCase):
    def test_a_participant_sees_reviewed_examples_and_their_own(self) -> None:
        self.act_as(participant(ME))
        for path in (f"/api/db/courses/{COURSE}/seeds", f"/api/courses/{COURSE}/seeds"):
            with self.subTest(path=path):
                body = self.client.get(path).json()
                ids = [record["id"] for record in body["seeds"]]
                self.assertEqual(sorted(ids), ["approved-ai", "mine"])
                # The persistence route omits stripped fields; the operational
                # route's response model declares them and serialises null.
                # Either way no review detail reaches a student.
                for record in body["seeds"]:
                    for field in ("reviewNotes", "validation", "evidenceQuote", "factId"):
                        self.assertIsNone(record.get(field), field)
                by_id = {record["id"]: record for record in body["seeds"]}
                self.assertTrue(by_id["mine"]["mine"])
                self.assertFalse(by_id["approved-ai"]["mine"])

    def test_the_persistence_list_still_reports_class_wide_counts(self) -> None:
        self.act_as(participant(ME))
        body = self.client.get(f"/api/db/courses/{COURSE}/seeds").json()
        self.assertEqual(body["count"], 2)
        self.assertEqual(body["originCounts"], {"ai_generated": 2, "user": 2})

    def test_staff_see_everything_including_review_detail(self) -> None:
        self.act_as(professor())
        body = self.client.get(f"/api/db/courses/{COURSE}/seeds").json()
        self.assertEqual(body["count"], 4)
        self.assertIn("reviewNotes", body["seeds"][0])

    def test_a_contribution_is_forced_to_user_origin_and_attributed(self) -> None:
        self.act_as(participant(ME))
        response = self.client.post(
            f"/api/db/courses/{COURSE}/seeds",
            json={"instruction": "Q?", "response": "A.", "origin": "ai_generated",
                  "reviewStatus": "approved", "validation": {"score": 1}, "participantId": THEM},
            headers=CSRF,
        )
        self.assertEqual(response.status_code, 201)
        payload = self.create_seed.call_args.args[2]
        self.assertEqual(payload["origin"], "user")
        self.assertNotIn("reviewStatus", payload)
        self.assertNotIn("validation", payload)
        self.assertEqual(self.create_seed.call_args.kwargs["participant_id"], ME)
        self.assertTrue(response.json()["seed"]["mine"])

    def test_staff_contributions_are_stored_as_sent(self) -> None:
        self.act_as(professor())
        self.client.post(
            f"/api/db/courses/{COURSE}/seeds",
            json={"instruction": "Q?", "response": "A.", "origin": "prototype", "reviewStatus": "approved"},
            headers=CSRF,
        )
        payload = self.create_seed.call_args.args[2]
        self.assertEqual(payload["origin"], "prototype")
        self.assertEqual(payload["reviewStatus"], "approved")
        self.assertIsNone(self.create_seed.call_args.kwargs["participant_id"])

    def test_staff_who_joined_their_own_course_contribute_as_that_participant(self) -> None:
        """An instructor trying the student flow gets exactly what a student gets."""
        self.act_as(
            Principal(
                user=professor().user,
                participant=Participant(participant_id=ME, course_id=COURSE, session_id="s"),
            )
        )
        response = self.client.post(
            f"/api/db/courses/{COURSE}/seeds",
            json={"instruction": "Q?", "response": "A.", "origin": "ai_generated", "reviewStatus": "approved"},
            headers=CSRF,
        )
        self.assertEqual(response.status_code, 201)
        payload = self.create_seed.call_args.args[2]
        self.assertEqual(payload["origin"], "user")
        self.assertNotIn("reviewStatus", payload)
        self.assertEqual(self.create_seed.call_args.kwargs["participant_id"], ME)
        self.assertTrue(response.json()["seed"]["mine"])

        # The staff view of the list is not projected, but their own rows are marked.
        body = self.client.get(f"/api/db/courses/{COURSE}/seeds").json()
        self.assertEqual(body["count"], 4)
        by_id = {record["id"]: record for record in body["seeds"]}
        self.assertTrue(by_id["mine"]["mine"])
        self.assertNotIn("mine", by_id["theirs"])
        self.assertIn("reviewNotes", by_id["draft-ai"])

    def test_a_participant_deletes_only_their_own_contribution(self) -> None:
        self.act_as(participant(ME))
        self.assertEqual(
            self.client.delete(f"/api/db/courses/{COURSE}/seeds/mine", headers=CSRF).status_code, 200
        )
        self.get_seed.return_value = seed(id="theirs", participantId=THEM)
        refused = self.client.delete(f"/api/db/courses/{COURSE}/seeds/theirs", headers=CSRF)
        self.assertEqual(refused.status_code, 403)
        self.get_seed.return_value = seed(id="approved-ai", reviewStatus="approved")
        refused = self.client.delete(f"/api/db/courses/{COURSE}/seeds/approved-ai", headers=CSRF)
        self.assertEqual(refused.status_code, 403)
        self.assertEqual(self.delete_seed.call_count, 1)

    def test_staff_delete_anything_in_their_course(self) -> None:
        self.act_as(professor())
        self.get_seed.return_value = seed(id="theirs", participantId=THEM)
        self.assertEqual(
            self.client.delete(f"/api/db/courses/{COURSE}/seeds/theirs", headers=CSRF).status_code, 200
        )


class AttributionRepositoryTests(unittest.TestCase):
    def test_create_evaluation_binds_the_participant(self) -> None:
        conn = FakeConnection(results=[1, [{
            "evaluation_id": "e", "course_id": COURSE, "comparison_id": "c",
            "most_accurate": "rag", "most_helpful": "", "most_concise": "", "best_grounded": "",
            "preferred_model": "rag", "hallucination_flags": [], "comment": None,
            "created_at": NOW, "run_id": None, "question_text": None, "participant_id": ME,
        }]])
        created = db_evaluations.create_evaluation(
            conn, COURSE, {"comparisonId": "c", "mostAccurate": "rag", "preferredModel": "rag"},
            participant_id=ME,
        )
        insert_sql, params = conn.statements[0]
        self.assertIn("participant_id", insert_sql)
        self.assertEqual(params["participant_id"], ME)
        self.assertEqual(created["participantId"], ME)

    def test_a_row_without_a_participant_maps_without_the_field(self) -> None:
        record = db_evaluations.map_evaluation({
            "evaluation_id": "e", "course_id": COURSE, "comparison_id": "c",
            "most_accurate": "rag", "most_helpful": "", "most_concise": "", "best_grounded": "",
            "preferred_model": "rag", "hallucination_flags": [], "comment": None,
            "created_at": NOW, "run_id": None, "question_text": None, "participant_id": None,
        })
        self.assertNotIn("participantId", record)

    def test_listing_for_a_participant_filters_in_sql(self) -> None:
        conn = FakeConnection(results=[[]])
        db_evaluations.list_evaluations(conn, COURSE, participant_id=ME)
        sql, params = conn.statements[0]
        self.assertIn("AND participant_id = %s", sql)
        self.assertEqual(params, (COURSE, ME))

    def test_seed_creation_binds_the_participant_and_counts_group_by_origin(self) -> None:
        conn = FakeConnection(results=[1, [{
            "seed_id": "s", "course_id": COURSE, "instruction": "Q?", "response": "A.",
            "category": "general", "source_section": "General", "difficulty": "Medium",
            "directly_answered": True, "origin": "user", "participant_id": ME,
        }]])
        created = db_seeds.create_seed(conn, COURSE, {"instruction": "Q?", "response": "A."},
                                       participant_id=ME)
        self.assertEqual(conn.statements[0][1]["participant_id"], ME)
        self.assertEqual(created["participantId"], ME)

        counts_conn = FakeConnection(results=[[{"origin": "user", "total": 3}]])
        self.assertEqual(db_seeds.count_seeds_by_origin(counts_conn, COURSE), {"user": 3})
        self.assertIn("GROUP BY origin", counts_conn.statements[0][0])


if __name__ == "__main__":
    unittest.main()
