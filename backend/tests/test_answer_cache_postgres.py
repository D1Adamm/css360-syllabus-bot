"""Opt-in: the classroom answer cache's SQL against a real PostgreSQL.

Skipped unless TEST_DATABASE_URL names a throwaway database that has
`db/schema.sql` (or migration 004) applied. Everything else about the cache
is tested against in-memory fakes; this file proves the statements run, the
constraints hold, and the three repositories agree with those fakes.

Every test runs in a transaction that is rolled back.

    TEST_DATABASE_URL=postgresql://user:password@localhost:5432/syllabus_bot_test \\
        .venv/bin/python -m pytest tests/test_answer_cache_postgres.py -q
"""

from __future__ import annotations

import os
import unittest
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import pytest

from app import (
    db_answer_cache,
    db_classroom_preparation,
    db_classroom_questions,
    db_courses,
    db_participants,
    db_seeds,
    db_users,
)
from app.answer_cache import question_key
from app.auth.passwords import hash_password

pytestmark = pytest.mark.auth

#: Captured at import, before the conftest fixture that strips it for every
#: other test. The fixture below puts it back for these.
DSN = (os.environ.get("TEST_DATABASE_URL") or "").strip()

COURSE_A = "css-360-winter-2026-a7rp"
COURSE_B = "css-350-spring-2026-n3h9"
PRINT_1 = "f" * 64
PRINT_2 = "e" * 64


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
class AnswerCachePostgresTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = _connect()
        self.now = datetime.now(timezone.utc)
        self.addCleanup(self._teardown)
        for course_id, name in ((COURSE_A, "CSS 360"), (COURSE_B, "CSS 350")):
            db_courses.create_course(
                self.conn, course_id, {"name": name, "title": "T", "term": "Winter 2026", "instructorName": "I"}
            )

    def _teardown(self) -> None:
        self.conn.rollback()
        self.conn.close()

    # -- helpers ------------------------------------------------------------------

    def seed(self, course_id: str, question: str, *, origin: str = "prototype",
             review_status: str | None = None, participant_id: str | None = None) -> str:
        created = db_seeds.create_seed(
            self.conn,
            course_id,
            {
                "instruction": question, "response": "A TRAINING ANSWER THAT MUST NEVER BE SERVED",
                "category": "general", "sourceSection": "Syllabus", "difficulty": "easy",
                "directlyAnswered": True, "origin": origin,
                **({"reviewStatus": review_status} if review_status else {}),
            },
            participant_id=participant_id,
        )
        return created["id"]

    def entry(self, course_id: str = COURSE_A, condition: str = "base", question: str = "When is the exam?",
              fingerprint: str = PRINT_1, answer: str = "In week 11.", source: str = "prepared"):
        return db_answer_cache.insert_entry(
            self.conn,
            course_id=course_id, condition=condition, question_key=question_key(question),
            fingerprint=fingerprint, question_text=question,
            payload={"answer": answer, "model": "llama3.2:3b", "responseType": condition, "courseId": course_id,
                     "sources": [{"chunkId": "c1", "sectionTitle": "Exams", "text": "week 11", "score": 0.9}]},
            identity={"model": {"tag": "llama3.2:3b", "digest": "d1"}},
            provenance={"elapsedSeconds": 9.5, "timings": {"output_tokens": 40}},
            source=source,
        )

    # -- answer_cache -------------------------------------------------------------

    def test_an_entry_round_trips_with_its_payload_identity_and_provenance(self) -> None:
        stored = self.entry()
        found = db_answer_cache.get_entry(self.conn, COURSE_A, "base", question_key("When is the exam?"), PRINT_1)
        self.assertEqual(found, stored)
        self.assertEqual(found["payload"]["sources"][0]["chunkId"], "c1")
        self.assertEqual(found["identity"]["model"]["digest"], "d1")
        self.assertEqual(found["provenance"]["timings"], {"output_tokens": 40})
        self.assertEqual(found["source"], "prepared")
        self.assertTrue(found["generatedAt"])
        self.assertEqual(len(found["entryId"]), 32)

    def test_the_first_answer_for_a_configuration_stays(self) -> None:
        first = self.entry(answer="First.")
        self.assertIsNone(self.entry(answer="Second."))
        found = db_answer_cache.get_entry(self.conn, COURSE_A, "base", question_key("When is the exam?"), PRINT_1)
        self.assertEqual((found["entryId"], found["payload"]["answer"]), (first["entryId"], "First."))

    def test_lookup_is_exact_on_course_condition_question_and_fingerprint(self) -> None:
        self.entry()
        key = question_key("When is the exam?")
        self.assertIsNone(db_answer_cache.get_entry(self.conn, COURSE_B, "base", key, PRINT_1))
        self.assertIsNone(db_answer_cache.get_entry(self.conn, COURSE_A, "rag", key, PRINT_1))
        self.assertIsNone(db_answer_cache.get_entry(self.conn, COURSE_A, "base", key, PRINT_2))
        self.assertIsNone(db_answer_cache.get_entry(self.conn, COURSE_A, "base", question_key("When is the exam review?"), PRINT_1))

    def test_a_new_configuration_adds_a_row_and_keeps_the_old_one(self) -> None:
        old = self.entry(fingerprint=PRINT_1, answer="Old model.")
        new = self.entry(fingerprint=PRINT_2, answer="New model.")
        self.assertNotEqual(old["entryId"], new["entryId"])
        keys = db_answer_cache.list_entry_keys(self.conn, COURSE_A, [question_key("When is the exam?")])
        self.assertEqual(sorted(k["fingerprint"] for k in keys), sorted([PRINT_1, PRINT_2]))

    def test_entry_keys_are_scoped_to_the_course_and_the_questions_asked_for(self) -> None:
        self.entry(question="When is the exam?")
        self.entry(question="What is the late policy?", condition="rag")
        self.entry(course_id=COURSE_B, question="When is the exam?")
        keys = db_answer_cache.list_entry_keys(self.conn, COURSE_A, [question_key("When is the exam?")])
        self.assertEqual(keys, [{"condition": "base", "questionKey": question_key("When is the exam?"), "fingerprint": PRINT_1}])
        self.assertEqual(db_answer_cache.list_entry_keys(self.conn, COURSE_A, []), [])

    def test_unknown_conditions_and_sources_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.entry(condition="summary")
        with self.assertRaises(ValueError):
            self.entry(source="imported")
        import psycopg

        with self.assertRaises(psycopg.errors.CheckViolation), self.conn.transaction():
            with self.conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO answer_cache (entry_id, course_id, condition, question_key, fingerprint, "
                    "question_text, payload, identity, provenance, source, generated_at) "
                    "VALUES ('x', %s, 'summary', 'k', 'f', 'q', '{}', '{}', '{}', 'live', now())",
                    (COURSE_A,),
                )

    def test_deleting_a_course_removes_its_answers_selection_and_status(self) -> None:
        seed_id = self.seed(COURSE_A, "When is the exam?")
        db_classroom_questions.replace_selection(self.conn, COURSE_A, [seed_id])
        db_classroom_preparation.begin_run(self.conn, COURSE_A, boot_id="boot-1")
        self.entry()
        self.entry(course_id=COURSE_B)
        with self.conn.cursor() as cursor:
            cursor.execute("DELETE FROM courses WHERE course_id = %s", (COURSE_A,))
            for table in ("answer_cache", "classroom_questions", "classroom_preparation"):
                cursor.execute(f"SELECT count(*) AS n FROM {table} WHERE course_id = %s", (COURSE_A,))
                self.assertEqual(cursor.fetchone()["n"], 0, table)
            cursor.execute("SELECT count(*) AS n FROM answer_cache WHERE course_id = %s", (COURSE_B,))
            self.assertEqual(cursor.fetchone()["n"], 1)

    def test_the_cache_table_has_no_column_for_a_person(self) -> None:
        with self.conn.cursor() as cursor:
            cursor.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'answer_cache'"
            )
            columns = {row["column_name"] for row in cursor.fetchall()}
        self.assertEqual(
            columns,
            {"entry_id", "course_id", "condition", "question_key", "fingerprint", "question_text",
             "payload", "identity", "provenance", "source", "generated_at"},
        )

    # -- classroom_questions --------------------------------------------------------

    def test_selection_is_explicit_and_independent_of_review_state(self) -> None:
        curated = self.seed(COURSE_A, "Curated, never reviewed?")
        draft = self.seed(COURSE_A, "AI draft?", origin="ai_generated")
        approved = self.seed(COURSE_A, "Approved for training?", review_status="approved")
        # Nothing is selected merely because it exists or is approved.
        self.assertEqual(db_classroom_questions.list_selected_questions(self.conn, COURSE_A), [])
        kept = db_classroom_questions.replace_selection(self.conn, COURSE_A, [curated, draft])
        self.assertEqual(kept, [curated, draft])
        selected = db_classroom_questions.list_selected_questions(self.conn, COURSE_A)
        self.assertEqual([q["question"] for q in selected], ["Curated, never reviewed?", "AI draft?"])
        listing = {q["seedId"]: q for q in db_classroom_questions.list_course_questions(self.conn, COURSE_A)}
        self.assertEqual({k: v["selected"] for k, v in listing.items()}, {curated: True, draft: True, approved: False})
        # Selecting did not approve anything.
        self.assertNotEqual(db_seeds.get_seed(self.conn, COURSE_A, curated).get("reviewStatus"), "approved")
        # No training answer travels with a selected question.
        self.assertNotIn("TRAINING ANSWER", repr(selected) + repr(listing))

    def test_replacing_the_selection_adds_removes_and_keeps_order(self) -> None:
        a, b, c = (self.seed(COURSE_A, f"Question {n}?") for n in "abc")
        early = self.now - timedelta(hours=1)
        db_classroom_questions.replace_selection(self.conn, COURSE_A, [a, b], now=early)
        db_classroom_questions.replace_selection(self.conn, COURSE_A, [c, b], now=self.now)
        selected = db_classroom_questions.list_selected_questions(self.conn, COURSE_A)
        # b kept its earlier place; a is gone; c was added after.
        self.assertEqual([q["seedId"] for q in selected], [b, c])
        db_classroom_questions.replace_selection(self.conn, COURSE_A, [])
        self.assertEqual(db_classroom_questions.list_selected_questions(self.conn, COURSE_A), [])

    def test_ineligible_and_foreign_questions_are_refused_and_nothing_changes(self) -> None:
        good = self.seed(COURSE_A, "Good?")
        rejected = self.seed(COURSE_A, "Rejected?", review_status="rejected")
        other = self.seed(COURSE_B, "Another course's?")
        participant = db_participants.create_participant(self.conn, course_id=COURSE_A, invitation_id=None, now=self.now)
        student = self.seed(COURSE_A, "A student's?", origin="user", participant_id=participant["participantId"])
        db_classroom_questions.replace_selection(self.conn, COURSE_A, [good])
        for bad in (rejected, other, student, "no-such-seed"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                db_classroom_questions.replace_selection(self.conn, COURSE_A, [good, bad])
            self.assertEqual(
                [q["seedId"] for q in db_classroom_questions.list_selected_questions(self.conn, COURSE_A)], [good]
            )
        # The student's unapproved question is not even listed.
        self.assertNotIn(student, {q["seedId"] for q in db_classroom_questions.list_course_questions(self.conn, COURSE_A)})

    def test_an_approved_student_question_can_be_selected_but_is_never_preselected(self) -> None:
        participant = db_participants.create_participant(self.conn, course_id=COURSE_A, invitation_id=None, now=self.now)
        seed_id = self.seed(COURSE_A, "A student's approved?", origin="user", participant_id=participant["participantId"])
        db_seeds.review_seed(self.conn, COURSE_A, seed_id, review_status="approved")
        (listed,) = db_classroom_questions.list_course_questions(self.conn, COURSE_A)
        self.assertEqual((listed["sourceLabel"], listed["selected"], listed["eligible"]), ("Student contribution", False, True))
        db_classroom_questions.replace_selection(self.conn, COURSE_A, [seed_id])
        self.assertEqual(len(db_classroom_questions.list_selected_questions(self.conn, COURSE_A)), 1)

    def test_a_selected_question_that_is_later_rejected_or_deleted_stops_being_prepared(self) -> None:
        keep = self.seed(COURSE_A, "Keep?")
        later_rejected = self.seed(COURSE_A, "Later rejected?")
        later_deleted = self.seed(COURSE_A, "Later deleted?")
        db_classroom_questions.replace_selection(self.conn, COURSE_A, [keep, later_rejected, later_deleted])
        db_seeds.review_seed(self.conn, COURSE_A, later_rejected, review_status="rejected")
        db_seeds.delete_seed(self.conn, COURSE_A, later_deleted)
        self.assertEqual([q["seedId"] for q in db_classroom_questions.list_selected_questions(self.conn, COURSE_A)], [keep])
        listing = {q["seedId"]: q for q in db_classroom_questions.list_course_questions(self.conn, COURSE_A)}
        # Still shown, flagged, so staff can see why it is skipped; the deleted one is gone.
        self.assertEqual((listing[later_rejected]["selected"], listing[later_rejected]["eligible"]), (True, False))
        self.assertNotIn(later_deleted, listing)

    def test_an_edited_question_is_prepared_under_its_current_wording(self) -> None:
        seed_id = self.seed(COURSE_A, "When is teh exam?")
        db_classroom_questions.replace_selection(self.conn, COURSE_A, [seed_id])
        db_seeds.update_seed(self.conn, COURSE_A, seed_id, {"instruction": "When is the exam?"})
        (selected,) = db_classroom_questions.list_selected_questions(self.conn, COURSE_A)
        self.assertEqual(selected["question"], "When is the exam?")

    def test_who_selected_is_recorded_and_survives_the_account_being_removed(self) -> None:
        user = db_users.create_user(
            self.conn, email="prof@uw.edu", display_name="Prof", role="professor",
            password_hash=hash_password("a fine passphrase"), now=self.now,
        )
        seed_id = self.seed(COURSE_A, "Who chose this?")
        db_classroom_questions.replace_selection(self.conn, COURSE_A, [seed_id], selected_by=user["userId"])
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT selected_by FROM classroom_questions WHERE seed_id = %s", (seed_id,))
            self.assertEqual(str(cursor.fetchone()["selected_by"]), user["userId"])
            cursor.execute("DELETE FROM users WHERE user_id = %s", (user["userId"],))
            cursor.execute("SELECT selected_by FROM classroom_questions WHERE seed_id = %s", (seed_id,))
            self.assertIsNone(cursor.fetchone()["selected_by"])

    # -- classroom_preparation ------------------------------------------------------

    def test_a_run_row_follows_the_run_and_belongs_to_the_process_that_started_it(self) -> None:
        self.assertIsNone(db_classroom_preparation.get_run(self.conn, COURSE_A))
        began = db_classroom_preparation.begin_run(self.conn, COURSE_A, boot_id="boot-1")
        self.assertEqual((began["state"], began["bootId"], began["generated"]), ("running", "boot-1", 0))
        self.assertTrue(db_classroom_preparation.record_progress(self.conn, COURSE_A, boot_id="boot-1", generated=5, failed=1))
        self.assertTrue(db_classroom_preparation.mark_stopping(self.conn, COURSE_A, boot_id="boot-1"))
        self.assertEqual(db_classroom_preparation.get_run(self.conn, COURSE_A)["state"], "stopping")
        # Another process cannot end or alter this run's row.
        self.assertFalse(db_classroom_preparation.finish_run(
            self.conn, COURSE_A, boot_id="boot-2", state="completed", reason=None, message=None, generated=9, failed=0))
        self.assertFalse(db_classroom_preparation.record_progress(self.conn, COURSE_A, boot_id="boot-2", generated=9, failed=9))
        self.assertTrue(db_classroom_preparation.finish_run(
            self.conn, COURSE_A, boot_id="boot-1", state="stopped", reason="stopped_by_staff",
            message=None, generated=6, failed=1))
        row = db_classroom_preparation.get_run(self.conn, COURSE_A)
        self.assertEqual((row["state"], row["reason"], row["generated"], row["failed"]), ("stopped", "stopped_by_staff", 6, 1))
        self.assertTrue(row["finishedAt"])

    def test_a_new_run_replaces_the_previous_runs_row(self) -> None:
        db_classroom_preparation.begin_run(self.conn, COURSE_A, boot_id="boot-1")
        db_classroom_preparation.finish_run(
            self.conn, COURSE_A, boot_id="boot-1", state="failed", reason="repeated_failures",
            message="The base model did not respond.", generated=3, failed=3)
        again = db_classroom_preparation.begin_run(self.conn, COURSE_A, boot_id="boot-2")
        self.assertEqual(
            (again["state"], again["bootId"], again["reason"], again["message"], again["generated"], again["finishedAt"]),
            ("running", "boot-2", None, None, 0, None),
        )
        # A stop can only be recorded on a row that is running.
        db_classroom_preparation.finish_run(
            self.conn, COURSE_A, boot_id="boot-2", state="completed", reason=None, message=None, generated=1, failed=0)
        self.assertFalse(db_classroom_preparation.mark_stopping(self.conn, COURSE_A, boot_id="boot-2"))

    def test_a_run_cannot_end_in_an_unknown_state(self) -> None:
        db_classroom_preparation.begin_run(self.conn, COURSE_A, boot_id="boot-1")
        with self.assertRaises(ValueError):
            db_classroom_preparation.finish_run(
                self.conn, COURSE_A, boot_id="boot-1", state="paused", reason=None, message=None, generated=0, failed=0)


if __name__ == "__main__":
    unittest.main()


@pytest.mark.usefixtures("integration_env")
class EndToEndPostgresTests(unittest.IsolatedAsyncioTestCase):
    """Preparation and the cache through their real database functions.

    Only the models are fake (the four pipelines and what Ollama reports).
    These tests commit, because the code under test opens its own
    connections; everything hangs off one course that is deleted afterwards.
    """

    COURSE = "css-999-cache-e2e-zz99"

    def setUp(self) -> None:
        from tests.answer_cache_fake import FakeAnswerCache, reset_shared_state
        from tests.test_classroom_preparation import FourPipelines
        from unittest.mock import patch
        from app import answer_cache
        from app.generation_queue import GenerationQueue

        self._delete_course()
        self.addCleanup(self._delete_course)
        with _connect() as conn:
            db_courses.create_course(
                conn, self.COURSE, {"name": "CSS 999", "title": "T", "term": "Winter 2026", "instructorName": "I"}
            )
            self.seed_ids = [
                db_seeds.create_seed(
                    conn, self.COURSE,
                    {"instruction": question, "response": "A TRAINING ANSWER", "category": "general",
                     "sourceSection": "Syllabus", "difficulty": "easy", "directlyAnswered": True, "origin": "prototype"},
                )["id"]
                for question in ("When is the exam?", "What is the late policy?")
            ]
            db_classroom_questions.replace_selection(conn, self.COURSE, self.seed_ids)

        # Only the identity lookup is faked; every database function is real.
        self.identity = FakeAnswerCache()
        ft = {"model": {"tag": "css360-v2:latest", "digest": "d", "courseVersion": "v2"}}
        self.identity.set_identity("fineTuned", self.COURSE, ft)
        self.identity.set_identity("fineTunedRag", self.COURSE, ft)
        for target in ("app.answer_cache.build_identity", "app.classroom_preparation.build_identity"):
            patcher = patch(target, new=self.identity._build_identity)
            patcher.start()
            self.addCleanup(patcher.stop)
        for patcher in (
            patch.dict(os.environ, {answer_cache.ENABLED_ENV: "true", "CLASSROOM_PREP_QUIET_SECONDS": "0"}),
            patch("app.classroom_preparation.POLL_SECONDS", 0.005),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        reset_shared_state(GenerationQueue(max_concurrency=1, max_waiting=160, queue_timeout=30.0))
        self.addCleanup(reset_shared_state)
        self.pipes = FourPipelines().install(self)

    def _delete_course(self) -> None:
        with _connect() as conn, conn.cursor() as cursor:
            cursor.execute("DELETE FROM courses WHERE course_id = %s", (self.COURSE,))

    async def _finish(self) -> dict[str, Any]:
        from app import classroom_preparation as prep
        import asyncio

        run = prep.active_run()
        if run is not None and run.task is not None:
            await asyncio.wait_for(asyncio.gather(run.task, return_exceptions=True), timeout=10)
        return await prep.course_status(self.COURSE)

    async def test_prepare_stop_resume_and_serve_against_the_real_tables(self) -> None:
        import asyncio
        from app import classroom_preparation as prep
        from app.answer_cache import serve_detailed

        self.pipes.gate = asyncio.Event()
        await prep.start(self.COURSE)
        self.pipes.gate.set()
        for _ in range(2000):
            if len(self.pipes.finished) >= 3:
                break
            await asyncio.sleep(0.002)
        await prep.request_stop(self.COURSE)
        status = await self._finish()
        done = status["counts"]["readyAnswers"]
        self.assertEqual(status["state"], "stopped")
        self.assertTrue(3 <= done < 8, done)
        with _connect() as conn:
            row = db_classroom_preparation.get_run(conn, self.COURSE)
            self.assertEqual((row["state"], row["bootId"], row["generated"]), ("stopped", prep.BOOT_ID, done))

        # A prepared answer is served to a student; an unprepared one generates and fills the gap.
        hit = await serve_detailed("base", self.COURSE, "when is the exam")
        self.assertEqual(hit.outcome, "hit")
        self.assertEqual(hit.response.answer, "base: When is the exam?")
        filled = await serve_detailed("fineTunedRag", self.COURSE, "What is the late policy?")
        self.assertEqual(filled.outcome, "stored")
        unselected = await serve_detailed("base", self.COURSE, "Something nobody selected?")
        self.assertEqual(unselected.outcome, "generated")

        await prep.start(self.COURSE)
        status = await self._finish()
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["counts"]["readyAnswers"], 8)
        with _connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                "SELECT source, count(*) AS n FROM answer_cache WHERE course_id = %s GROUP BY source", (self.COURSE,)
            )
            by_source = {row["source"]: row["n"] for row in cursor.fetchall()}
            cursor.execute(
                "SELECT count(*) AS n FROM answer_cache WHERE course_id = %s AND payload::text LIKE %s",
                (self.COURSE, "%TRAINING ANSWER%"),
            )
            self.assertEqual(cursor.fetchone()["n"], 0)
        self.assertEqual(by_source, {"prepared": 7, "live": 1})

    async def test_a_restart_is_reported_from_the_row_alone(self) -> None:
        from app import classroom_preparation as prep

        with _connect() as conn:
            db_classroom_preparation.begin_run(conn, self.COURSE, boot_id="a-process-that-is-gone")
        status = await prep.course_status(self.COURSE)
        self.assertEqual((status["state"], status["running"], status["lastRun"]["state"]), ("stopped", False, "interrupted"))
        self.assertTrue(status["canStart"])
        await prep.start(self.COURSE)
        status = await self._finish()
        self.assertEqual(status["state"], "ready")
        with _connect() as conn:
            self.assertEqual(db_classroom_preparation.get_run(conn, self.COURSE)["bootId"], prep.BOOT_ID)
