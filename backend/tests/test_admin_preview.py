"""An administrator's preview of the student flow: real answers, no research data.

The student pages are the one place an administrator has no identity of their
own. A rating is attributed to a participant, and an administrator never
redeems a classroom code, so "try this course the way a student will" used to
end in a 403 on Evaluate. Preview is the smallest thing that finishes the flow
without inventing an identity:

- the four generation routes are untouched — `authorize_course_access` already
  admits an administrator to any course, and the course still comes from the
  request and is checked before any model is asked anything;
- Evaluate submits to `POST /api/db/courses/{id}/evaluations/preview`, an
  administrator-only route that validates the rating exactly as the real one
  does and stores nothing.

What is held here, in the order the requirements list them:

1. an administrator can preview a course: the generation routes answer for the
   course they name, and the preview route accepts a rating;
2. nobody else can use the preview route — anonymous, participants of any
   course, professors of the course itself;
3. a participant's real route is untouched, cookie-driven identity included,
   and it still refuses staff who hold no participant session;
4. the preview uses the course in the path and nothing in the body;
5. a preview never reaches the evaluations table, creates no participant, and
   sets no cookie.

`test_authorization_matrix.py` drives the preview route against every kind of
principal as well, because it is classified `require_admin`; the tests here
say why those outcomes are the right ones.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app import db_evaluations
from app.auth.dependencies import current_principal
from app.auth.principal import Participant, Principal, StaffUser
from app.auth.settings import CSRF_HEADER_VALUE, PARTICIPANT_COOKIE_NAME
from app.auth.tokens import generate_token, hash_token
from app.main import app
from route_classification import CLASSIFICATION, REQUIRE_ADMIN
from test_db_repositories import FakeConnection

pytestmark = pytest.mark.auth

COURSE = "css-360-winter-2026-a7rp"
OTHER = "css-350-spring-2026-n3h9"
CSRF = {"X-Requested-With": CSRF_HEADER_VALUE}
NOW = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)

GENERATION_PATHS = (
    "/api/base-model/generate",
    "/api/rag/generate",
    "/api/fine-tuned/generate",
    "/api/fine-tuned-rag/generate",
)

#: What the Evaluate page sends: the simplified form, plus the run it rated.
RATING: dict[str, Any] = {
    "comparisonId": "question-run-1",
    "mostAccurate": "rag",
    "preferredModel": "fineTunedRag",
    "hallucinationFlags": ["base"],
    "comment": "an administrator's remark",
    "runId": "run-1",
    "questionText": "When are office hours?",
    "createdAt": "2026-09-08T12:00:00+00:00",
}

#: The row `course_exists` finds.
COURSE_FOUND = [{"?column?": 1}]
EXISTENCE_CHECK = "SELECT 1 FROM courses WHERE course_id = %s"


def stored_row(participant_id: str | None) -> dict[str, Any]:
    """The evaluations row the real route reads back after its insert."""
    return {
        "evaluation_id": "eval-stored",
        "course_id": COURSE,
        "comparison_id": RATING["comparisonId"],
        "most_accurate": RATING["mostAccurate"],
        "most_helpful": "",
        "most_concise": "",
        "best_grounded": "",
        "preferred_model": RATING["preferredModel"],
        "hallucination_flags": RATING["hallucinationFlags"],
        "comment": RATING["comment"],
        "created_at": NOW,
        "run_id": RATING["runId"],
        "question_text": RATING["questionText"],
        "participant_id": participant_id,
    }


def admin() -> Principal:
    return Principal(
        user=StaffUser(
            user_id="u-admin", email="admin@uw.edu", display_name="Admin",
            role="admin", session_id="s",
        )
    )


def professor(*course_ids: str) -> Principal:
    return Principal(
        user=StaffUser(
            user_id="u-prof", email="prof@uw.edu", display_name="Prof",
            role="professor", course_ids=frozenset(course_ids), session_id="s",
        )
    )


def participant(course_id: str) -> Principal:
    return Principal(
        participant=Participant(
            participant_id=f"participant-{course_id}", course_id=course_id, session_id="s"
        )
    )


@contextmanager
def _idle_connection(**kwargs: Any) -> Iterator[object]:
    """For the session lookup, whose repository calls are patched."""
    yield object()


class PreviewTestCase(unittest.TestCase):
    """The persistence routes against a recording connection and no database.

    Every statement a route runs lands on `self.connection`. A test that
    expects a write queues the results the write needs; a test that expects
    none reads the statements back and finds only the existence check.
    """

    def setUp(self) -> None:
        self.client = TestClient(app, base_url="https://testserver")
        self.addCleanup(app.dependency_overrides.clear)
        self.connection = FakeConnection([COURSE_FOUND])

        @contextmanager
        def fake_connection(**kwargs: Any) -> Iterator[FakeConnection]:
            yield self.connection

        self.patch("app.db_routes.db_connection", new=fake_connection)
        self.create_participant = self.patch("app.db_participants.create_participant")
        self.create_session = self.patch("app.db_sessions.create_session")

    def patch(self, target: str, **kwargs: Any) -> Any:
        patcher = patch(target, **kwargs)
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock

    def act_as(self, principal: Principal | None) -> None:
        """Install a principal, or none: the real `current_principal` then
        resolves whatever cookies the client presents — usually nothing."""
        if principal is None:
            app.dependency_overrides.pop(current_principal, None)
        else:
            app.dependency_overrides[current_principal] = lambda: principal

    def preview(self, body: dict[str, Any] = RATING, course_id: str = COURSE):
        return self.client.post(
            f"/api/db/courses/{course_id}/evaluations/preview", json=body, headers=CSRF
        )

    def rate(self, body: dict[str, Any] = RATING, course_id: str = COURSE):
        return self.client.post(
            f"/api/db/courses/{course_id}/evaluations", json=body, headers=CSRF
        )

    def inserts(self) -> list[str]:
        return [sql for sql in self.connection.sql if sql.startswith("INSERT")]


# --------------------------------------------------------------------------- #
# 1. An administrator can preview a course
# --------------------------------------------------------------------------- #


class AdministratorCanPreviewTests(PreviewTestCase):
    def test_a_rating_is_accepted_and_echoed_as_it_would_have_been_stored(self) -> None:
        self.act_as(admin())
        response = self.preview()

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIs(body["saved"], False)
        self.assertEqual(body["courseId"], COURSE)
        evaluation = body["evaluation"]
        self.assertTrue(evaluation["id"].startswith(db_evaluations.PREVIEW_ID_PREFIX))
        self.assertEqual(evaluation["courseId"], COURSE)
        for field in (
            "comparisonId", "mostAccurate", "preferredModel", "hallucinationFlags",
            "comment", "runId", "questionText", "createdAt",
        ):
            self.assertEqual(evaluation[field], RATING[field], field)
        # Not asked, so not invented — serialized as null, exactly as the real
        # route serializes a stored record without them; and nobody to
        # attribute it to.
        for field in ("mostHelpful", "mostConcise", "bestGrounded"):
            self.assertIsNone(evaluation.get(field), field)
        self.assertNotIn("participantId", evaluation)

    def test_the_generation_routes_answer_for_the_course_the_preview_names(self) -> None:
        """Base, RAG, Fine-Tuned and Fine-Tuned + RAG: the same four routes the
        student page calls, reached with a staff session and no participant."""
        self.act_as(admin())
        base = AsyncMock(return_value={"answer": "Base.", "model": "llama", "response_type": "base"})
        rag = AsyncMock(return_value={
            "courseId": COURSE, "answer": "RAG.", "model": "llama",
            "sources": [], "retrievedChunks": [], "responseType": "rag",
        })
        fine_tuned = AsyncMock(return_value={
            "answer": "FT.", "model": "ft", "response_type": "fineTuned",
            "model_version": "v1", "adapter_loaded": True, "generation_seconds": 0.4,
        })
        fine_tuned_rag = AsyncMock(return_value={
            "courseId": COURSE, "answer": "FT+RAG.", "model": "ft", "sources": [],
            "retrievedChunks": [], "responseType": "fineTunedRag", "modelVersion": "v1",
            "adapterLoaded": True, "generationSeconds": 0.6,
        })
        self.patch("app.main.generate_base_model_response", new=base)
        self.patch("app.main.generate_course_rag_answer", new=rag)
        self.patch("app.main.resolve_current_course_model", return_value={"version": "v1"})
        self.patch("app.main.generate_finetuned_response", new=fine_tuned)
        self.patch("app.main.generate_course_finetuned_rag_answer", new=fine_tuned_rag)

        for path in GENERATION_PATHS:
            with self.subTest(path=path):
                response = self.client.post(
                    path, json={"courseId": COURSE, "question": "When are office hours?"},
                    headers=CSRF,
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["courseId"], COURSE)

        for stub in (rag, fine_tuned, fine_tuned_rag):
            self.assertEqual(stub.call_args.kwargs["course_id"], COURSE)
        self.create_participant.assert_not_called()
        self.create_session.assert_not_called()


# --------------------------------------------------------------------------- #
# 2. Nobody else can use the preview
# --------------------------------------------------------------------------- #


class NobodyElseTests(PreviewTestCase):
    def test_without_a_session_the_preview_is_401(self) -> None:
        self.act_as(None)
        self.assertEqual(self.preview().status_code, 401)
        self.assertEqual(self.connection.statements, [])

    def test_a_participant_of_any_course_is_401(self) -> None:
        """A participant is not staff; the preview is not theirs to use, and
        their own route is the real one."""
        for course_id in (COURSE, OTHER):
            with self.subTest(course=course_id):
                self.act_as(participant(course_id))
                self.assertEqual(self.preview().status_code, 401)
        self.assertEqual(self.connection.statements, [])

    def test_a_professor_is_403_even_for_their_own_course(self) -> None:
        """Deliberate: a professor previews through their own classroom code,
        as a real participant, and their behaviour is unchanged here."""
        for principal in (professor(COURSE), professor(OTHER)):
            with self.subTest(courses=sorted(principal.user.course_ids)):
                self.act_as(principal)
                self.assertEqual(self.preview().status_code, 403)
        self.assertEqual(self.connection.statements, [])

    def test_the_route_is_classified_administrator_only(self) -> None:
        self.assertEqual(
            CLASSIFICATION[("POST", "/api/db/courses/{course_id}/evaluations/preview")],
            REQUIRE_ADMIN,
        )


# --------------------------------------------------------------------------- #
# 3. Student participant auth still works normally
# --------------------------------------------------------------------------- #


class ParticipantRouteUnchangedTests(PreviewTestCase):
    def test_a_participant_still_records_a_rating_attributed_from_the_session(self) -> None:
        self.act_as(participant(COURSE))
        self.connection = FakeConnection([COURSE_FOUND, 1, [stored_row(f"participant-{COURSE}")]])

        response = self.rate()

        self.assertEqual(response.status_code, 201, response.text)
        insert = self.connection.params_for("INSERT INTO evaluations")
        self.assertEqual(insert["participant_id"], f"participant-{COURSE}")
        self.assertEqual(insert["course_id"], COURSE)
        self.assertTrue(insert["evaluation_id"].startswith("eval-"))
        # The participant's own view still drops the attribution.
        self.assertNotIn("participantId", response.json())

    def test_a_participant_cookie_still_resolves_and_never_reaches_the_preview(self) -> None:
        """The middleware half: a real participant cookie, looked up the way
        production looks it up, reaches the real route and not the preview."""
        token = generate_token()
        row = {
            "session_id": "sess-part",
            "expires_at": NOW + timedelta(days=100),
            "last_seen_at": NOW - timedelta(minutes=1),
            "revoked_at": None,
            "participant_id": "part-1",
            "course_id": COURSE,
        }
        self.act_as(None)
        self.patch("app.auth.dependencies.db_connection", new=_idle_connection)
        self.patch("app.auth.dependencies._utc_now", return_value=NOW)
        self.patch(
            "app.db_sessions.find_participant_session",
            side_effect=lambda conn, token_hash: row if token_hash == hash_token(token) else None,
        )
        self.patch("app.db_sessions.touch_session", return_value=None)
        self.patch("app.db_participants.touch_participant", return_value=None)
        self.client.cookies.set(PARTICIPANT_COOKIE_NAME, token)

        self.connection = FakeConnection([COURSE_FOUND, 1, [stored_row("part-1")]])
        recorded = self.rate()
        self.assertEqual(recorded.status_code, 201, recorded.text)
        self.assertEqual(
            self.connection.params_for("INSERT INTO evaluations")["participant_id"], "part-1"
        )

        self.connection = FakeConnection([COURSE_FOUND])
        refused = self.preview()
        self.assertEqual(refused.status_code, 401)
        self.assertEqual(self.connection.statements, [])

    def test_the_real_route_still_refuses_staff_without_a_participant_session(self) -> None:
        """`require_participant` is not weakened by the preview existing."""
        for principal in (admin(), professor(COURSE)):
            with self.subTest(role=principal.user.role):
                self.act_as(principal)
                self.assertEqual(self.rate().status_code, 403)
        self.assertEqual(self.connection.statements, [])


# --------------------------------------------------------------------------- #
# 4. The preview uses the selected course only
# --------------------------------------------------------------------------- #


class SelectedCourseOnlyTests(PreviewTestCase):
    def test_a_course_in_the_body_is_ignored_in_favour_of_the_path(self) -> None:
        self.act_as(admin())
        response = self.preview({**RATING, "courseId": OTHER})

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["courseId"], COURSE)
        self.assertEqual(response.json()["evaluation"]["courseId"], COURSE)
        # The existence check — the only statement — asked about the path's course.
        self.assertEqual(self.connection.statements, [(EXISTENCE_CHECK, (COURSE,))])

    def test_a_course_that_does_not_exist_is_404(self) -> None:
        self.act_as(admin())
        self.connection = FakeConnection()
        self.assertEqual(self.preview(course_id=OTHER).status_code, 404)
        self.assertEqual(self.inserts(), [])

    def test_a_malformed_course_id_is_400_before_anything_runs(self) -> None:
        self.act_as(admin())
        self.assertEqual(self.preview(course_id="Bad_Id").status_code, 400)
        self.assertEqual(self.connection.statements, [])


# --------------------------------------------------------------------------- #
# 5. A preview cannot write an evaluation record
# --------------------------------------------------------------------------- #


class NothingIsWrittenTests(PreviewTestCase):
    def test_the_preview_runs_no_statement_but_the_existence_check(self) -> None:
        self.act_as(admin())
        self.assertEqual(self.preview().status_code, 200)

        self.assertEqual(self.connection.sql, [EXISTENCE_CHECK])
        self.assertEqual(self.inserts(), [])
        self.create_participant.assert_not_called()
        self.create_session.assert_not_called()

    def test_the_preview_sets_no_cookie(self) -> None:
        self.act_as(admin())
        response = self.preview()
        self.assertEqual(response.headers.get_list("set-cookie"), [])

    def test_the_preview_validates_exactly_as_the_real_route_does(self) -> None:
        """Same schema, same repository rule: what the real route refuses, the
        preview refuses, so an administrator sees the student's errors too."""
        bodies = (
            {"comparisonId": "c", "mostAccurate": "rag"},  # schema: a rating is missing
            {**RATING, "preferredModel": "   "},  # repository: blank once stripped
        )
        self.connection = FakeConnection([COURSE_FOUND] * (2 * len(bodies)))
        for body in bodies:
            with self.subTest(body=body):
                self.act_as(admin())
                self.assertEqual(self.preview(body).status_code, 422)
                self.act_as(participant(COURSE))
                self.assertEqual(self.rate(body).status_code, 422)
        self.assertEqual(self.inserts(), [])


class PreviewRepositoryTests(unittest.TestCase):
    """`db_evaluations.preview_evaluation` takes no connection at all."""

    def test_maps_a_rating_like_a_stored_row_without_storing_it(self) -> None:
        record = db_evaluations.preview_evaluation(COURSE, RATING)

        self.assertTrue(record["id"].startswith("preview-"))
        self.assertEqual(record["courseId"], COURSE)
        self.assertEqual(record["hallucinationFlags"], ["base"])
        self.assertEqual(record["createdAt"], RATING["createdAt"])
        self.assertNotIn("participantId", record)
        for field in ("mostHelpful", "mostConcise", "bestGrounded"):
            self.assertNotIn(field, record)

    def test_refuses_what_the_insert_refuses(self) -> None:
        for missing in ("comparisonId", "mostAccurate", "preferredModel"):
            body = {key: value for key, value in RATING.items() if key != missing}
            with self.subTest(missing=missing):
                with self.assertRaises(ValueError):
                    db_evaluations.preview_evaluation(COURSE, body)
                with self.assertRaises(ValueError):
                    db_evaluations.create_evaluation(FakeConnection([1]), COURSE, body)

    def test_the_insert_and_the_preview_build_the_same_columns(self) -> None:
        connection = FakeConnection([1, [stored_row("part-1")]])
        db_evaluations.create_evaluation(connection, COURSE, RATING, participant_id="part-1")
        inserted = connection.params_for("INSERT INTO evaluations")
        previewed = db_evaluations.evaluation_row(
            COURSE, RATING, evaluation_id="preview-x", participant_id=None
        )

        self.assertEqual(set(inserted), set(previewed))
        for column, value in previewed.items():
            if column in ("evaluation_id", "participant_id"):
                continue
            with self.subTest(column=column):
                stored = inserted[column]
                # The insert wraps the JSONB column for psycopg; same content.
                self.assertEqual(getattr(stored, "obj", stored), value)


if __name__ == "__main__":
    unittest.main()
