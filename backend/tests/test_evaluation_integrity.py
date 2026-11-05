"""What a rating may say, and that it says it once (audit findings B8 and B9).

Before 2026-11 a participant's body was stored as it arrived: any string for the
winning approach, any list of flags, any comment, the browser's own clock, and
no record of the four answers being compared — so a rating could be read later
against answers the models no longer give, and one comparison could be rated
twice. These tests pin the rules `db_evaluations` now applies and the way the
routes report them.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_db_repositories import FakeConnection  # noqa: E402

from app import db_evaluations  # noqa: E402
from app.auth.dependencies import current_principal  # noqa: E402
from app.auth.principal import Participant, Principal  # noqa: E402
from app.main import app  # noqa: E402

COURSE = "css-360-winter-2026-a7rp"
ME = "participant-1"

RESPONSES = {
    "base": {"text": "Base answer.", "error": None, "sources": [], "model": "llama3.2:3b"},
    "rag": {"text": "RAG answer.", "error": None, "sources": ["Grading"], "model": "llama3.2:3b"},
    "fineTuned": {
        "text": "Fine-tuned answer.",
        "error": None,
        "sources": [],
        "model": "css360-v1:latest",
        "modelVersion": "v1",
        "engine": "vm",
    },
    "fineTunedRag": {
        "text": "",
        "error": "This response is temporarily unavailable.",
        "sources": [],
        "model": None,
        "modelVersion": None,
        "engine": None,
    },
}

RATING: dict[str, Any] = {
    "comparisonId": "question-run-1",
    "mostAccurate": "fineTuned",
    "preferredModel": "rag",
    "hallucinationFlags": ["base"],
    "comment": "The fine-tuned one named the right week.",
    "runId": "run-1",
    "questionText": "When is the final?",
    "responses": RESPONSES,
}


def stored_row(**overrides: Any) -> dict[str, Any]:
    row = {
        "evaluation_id": "eval-1",
        "course_id": COURSE,
        "comparison_id": "question-run-1",
        "most_accurate": "fineTuned",
        "most_helpful": "",
        "most_concise": "",
        "best_grounded": "",
        "preferred_model": "rag",
        "hallucination_flags": ["base"],
        "comment": RATING["comment"],
        "created_at": "2026-11-03T12:00:00+00:00",
        "run_id": "run-1",
        "question_text": "When is the final?",
        "participant_id": ME,
        "responses": RESPONSES,
    }
    row.update(overrides)
    return row


class ValidationTests(unittest.TestCase):
    def row(self, **overrides: Any) -> dict[str, Any]:
        return db_evaluations.evaluation_row(
            COURSE, {**RATING, **overrides}, evaluation_id="eval-x", participant_id=ME
        )

    def test_a_complete_rating_is_accepted_with_its_responses(self) -> None:
        row = self.row()
        self.assertEqual(row["most_accurate"], "fineTuned")
        self.assertEqual(row["hallucination_flags"], ["base"])
        self.assertEqual(set(row["responses"]), {"base", "rag", "fineTuned", "fineTunedRag"})
        self.assertEqual(row["responses"]["fineTuned"]["modelVersion"], "v1")
        self.assertEqual(row["responses"]["fineTuned"]["engine"], "vm")
        self.assertEqual(row["responses"]["rag"]["sources"], ["Grading"])

    def test_the_criteria_must_name_one_of_the_four_approaches(self) -> None:
        for field in ("mostAccurate", "preferredModel"):
            with self.subTest(field=field), self.assertRaises(ValueError) as ctx:
                self.row(**{field: "gpt"})
            self.assertIn(field, str(ctx.exception))

    def test_the_retired_criteria_are_checked_when_sent(self) -> None:
        with self.assertRaises(ValueError):
            self.row(mostHelpful="whatever")
        self.assertEqual(self.row(mostHelpful="rag")["most_helpful"], "rag")

    def test_flags_are_a_set_of_approaches(self) -> None:
        with self.assertRaises(ValueError):
            self.row(hallucinationFlags=["base", "nope"])
        self.assertEqual(self.row(hallucinationFlags=["base", "base"])["hallucination_flags"], ["base"])

    def test_the_question_limit_is_the_generate_routes_limit(self) -> None:
        """A question the generate routes accept can always be rated."""
        from app.schemas import QUESTION_MAX_LENGTH

        self.assertEqual(db_evaluations.QUESTION_TEXT_MAX_LENGTH, QUESTION_MAX_LENGTH)

    def test_text_fields_have_the_forms_limits(self) -> None:
        with self.assertRaises(ValueError):
            self.row(comment="x" * (db_evaluations.COMMENT_MAX_LENGTH + 1))
        with self.assertRaises(ValueError):
            self.row(questionText="x" * (db_evaluations.QUESTION_TEXT_MAX_LENGTH + 1))
        with self.assertRaises(ValueError):
            self.row(runId="x" * (db_evaluations.IDENTIFIER_MAX_LENGTH + 1))

    def test_created_at_is_the_servers_clock(self) -> None:
        row = self.row(createdAt="2001-01-01T00:00:00+00:00")
        self.assertNotEqual(row["created_at"], "2001-01-01T00:00:00+00:00")
        self.assertTrue(row["created_at"].startswith("20"))

    def test_responses_must_cover_exactly_the_four_approaches(self) -> None:
        missing = {key: value for key, value in RESPONSES.items() if key != "rag"}
        with self.assertRaises(ValueError):
            self.row(responses=missing)
        with self.assertRaises(ValueError):
            self.row(responses={**RESPONSES, "gpt": {"text": "x"}})
        with self.assertRaises(ValueError):
            self.row(responses="not an object")

    def test_a_response_entry_is_reduced_to_the_stored_fields(self) -> None:
        row = self.row(responses={**RESPONSES, "base": {**RESPONSES["base"], "secret": "no"}})
        self.assertNotIn("secret", row["responses"]["base"])
        self.assertEqual(row["responses"]["base"]["modelVersion"], None)

    def test_an_answer_that_never_arrived_cannot_win_or_be_flagged(self) -> None:
        """fineTunedRag errored in RESPONSES; nothing may name it."""
        for field in ("mostAccurate", "preferredModel"):
            with self.subTest(field=field), self.assertRaises(ValueError) as ctx:
                self.row(**{field: "fineTunedRag"})
            self.assertIn("did not answer", str(ctx.exception))
        with self.assertRaises(ValueError):
            self.row(hallucinationFlags=["fineTunedRag"])

    def test_a_rating_without_responses_is_still_accepted(self) -> None:
        """An older client, or a historical import: the rule cannot be checked, so it is not."""
        row = self.row(responses=None)
        self.assertIsNone(row["responses"])

    def test_a_response_text_over_the_limit_is_refused(self) -> None:
        huge = {**RESPONSES, "base": {**RESPONSES["base"], "text": "x" * 20001}}
        with self.assertRaises(ValueError):
            self.row(responses=huge)


class WriteTests(unittest.TestCase):
    def test_responses_are_stored_as_json_and_read_back(self) -> None:
        # Duplicate check (nothing yet), the insert, the read-back.
        connection = FakeConnection([[], [], 1, [stored_row()]])
        created = db_evaluations.create_evaluation(connection, COURSE, RATING, participant_id=ME)

        inserted = connection.params_for("INSERT INTO evaluations")
        self.assertEqual(inserted["responses"].obj["fineTuned"]["modelVersion"], "v1")
        self.assertEqual(created["responses"]["fineTuned"]["engine"], "vm")
        self.assertEqual(created["participantId"], ME)

    def test_a_second_rating_of_the_same_run_by_the_same_participant_is_refused(self) -> None:
        connection = FakeConnection([[], [stored_row()]])
        with self.assertRaises(db_evaluations.DuplicateEvaluationError) as ctx:
            db_evaluations.create_evaluation(connection, COURSE, RATING, participant_id=ME)
        self.assertEqual(ctx.exception.evaluation_id, "eval-1")
        self.assertFalse(any(sql.startswith("INSERT") for sql in connection.sql))
        duplicate_check = connection.sql[1]
        self.assertIn("participant_id = %s AND run_id = %s", duplicate_check)

    def test_simultaneous_submissions_are_serialised_by_a_transaction_lock(self) -> None:
        """Two submissions at once would each see no earlier row under READ
        COMMITTED and both insert. The advisory lock is taken first, keyed on
        the participant and the run, and held to the end of the transaction, so
        the second waits and then finds the first's row."""
        connection = FakeConnection([[], [], 1, [stored_row()]])
        db_evaluations.create_evaluation(connection, COURSE, RATING, participant_id=ME)

        self.assertTrue(
            connection.sql[0].startswith("SELECT pg_advisory_xact_lock(hashtext(%s), hashtext(%s))"),
            connection.sql[0],
        )
        self.assertEqual(connection.params[0], (f"evaluation:{COURSE}:{ME}", RATING["runId"]))
        self.assertIn("FROM evaluations WHERE course_id = %s AND participant_id = %s AND run_id = %s", connection.sql[1])
        self.assertTrue(connection.sql[2].startswith("INSERT INTO evaluations"))

    def test_a_rating_without_a_run_id_is_not_checked_for_duplicates(self) -> None:
        body = {key: value for key, value in RATING.items() if key != "runId"}
        connection = FakeConnection([1, [stored_row(run_id=None)]])
        db_evaluations.create_evaluation(connection, COURSE, body, participant_id=ME)
        self.assertTrue(connection.sql[0].startswith("INSERT INTO evaluations"))

    def test_a_row_without_responses_reads_back_without_the_field(self) -> None:
        record = db_evaluations.map_evaluation(stored_row(responses=None))
        self.assertNotIn("responses", record)


def _participant() -> Principal:
    return Principal(participant=Participant(participant_id=ME, course_id=COURSE, session_id="s"))


class RouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[current_principal] = _participant
        self.addCleanup(lambda: app.dependency_overrides.pop(current_principal, None))
        self.patches = [
            patch("app.db_routes.db_connection", side_effect=lambda **kwargs: _idle()),
            patch("app.db_routes.db_courses.course_exists", return_value=True),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def rate(self, body: dict[str, Any]):
        return self.client.post(
            f"/api/db/courses/{COURSE}/evaluations",
            json=body,
            headers={"X-Requested-With": "SyllabusModelLab"},
        )

    def test_a_duplicate_is_a_409_with_a_code_and_the_earlier_id(self) -> None:
        with patch(
            "app.db_routes.db_evaluations.create_evaluation",
            side_effect=db_evaluations.DuplicateEvaluationError("eval-earlier"),
        ):
            response = self.rate(RATING)

        self.assertEqual(response.status_code, 409, response.text)
        detail = response.json()["detail"]
        self.assertEqual(detail["code"], "evaluation_already_recorded")
        self.assertEqual(detail["evaluationId"], "eval-earlier")

    def test_a_criterion_naming_an_unanswered_approach_is_422(self) -> None:
        response = self.rate({**RATING, "preferredModel": "fineTunedRag"})
        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn("did not answer", response.text)

    def test_an_unknown_approach_is_422(self) -> None:
        response = self.rate({**RATING, "mostAccurate": "gpt-4"})
        self.assertEqual(response.status_code, 422, response.text)


from contextlib import contextmanager  # noqa: E402


@contextmanager
def _idle():
    yield FakeConnection([[], [], 1, [stored_row()]])


if __name__ == "__main__":
    unittest.main()
