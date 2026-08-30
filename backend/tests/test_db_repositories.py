"""Repository SQL, driven by a recording fake connection.

No PostgreSQL server is involved. What these check is the part that is wrong or
right before any server sees it: that every statement is course-scoped, that no
caller-supplied value is ever interpolated into SQL text, that DELETEs name both
keys, and that the conflict guards for model requests and training runs are
written as conditional writes rather than read-then-write.

The fake is deliberately dumb — it records statements and hands back queued
rows. It is not a database and cannot tell you a query returns the right
answer; `db/schema.sql` and the VM smoke test cover that.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from typing import Any

from app import db_courses, db_evaluations, db_model_requests, db_models
from app import db_seeds, db_training_runs

UTC_NOON = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
COURSE = "css-350-spring-2026-n3h9"
OTHER_COURSE = "css-360-winter-2026-a7rp"


class FakeCursor:
    def __init__(self, connection: "FakeConnection") -> None:
        self._connection = connection
        self._rows: list[dict[str, Any]] = []
        self.rowcount = 0

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self._connection.statements.append((" ".join(sql.split()), params))
        result = self._connection.next_result()
        if isinstance(result, int):
            self._rows = []
            self.rowcount = result
        else:
            self._rows = list(result)
            self.rowcount = len(self._rows)

    def fetchone(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self._rows)


class FakeConnection:
    """Records statements; returns queued results in order."""

    def __init__(self, results: list[Any] | None = None) -> None:
        self.statements: list[tuple[str, Any]] = []
        self._results = list(results or [])

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def next_result(self) -> Any:
        return self._results.pop(0) if self._results else []

    # Convenience accessors for assertions.
    @property
    def sql(self) -> list[str]:
        return [statement for statement, _ in self.statements]

    @property
    def params(self) -> list[Any]:
        return [parameters for _, parameters in self.statements]

    def all_text(self) -> str:
        return " || ".join(self.sql)

    def params_for(self, prefix: str) -> Any:
        """Parameters of the first statement starting with `prefix`.

        Repositories re-read a row before and after writing it, so the write is
        not at a fixed index; addressing it by statement keeps these tests from
        breaking when a read is added.
        """
        for statement, parameters in self.statements:
            if statement.startswith(prefix):
                return parameters
        raise AssertionError(f"No statement starting with {prefix!r} was executed.")


def seed_row(**overrides: Any) -> dict[str, Any]:
    row = {
        "seed_id": "-Oseed001",
        "course_id": COURSE,
        "instruction": "Q?",
        "response": "A.",
        "category": "general",
        "source_section": "General",
        "difficulty": "Medium",
        "directly_answered": True,
        "origin": "ai_generated",
        "notes": None,
        "created_at": UTC_NOON,
        "status": "generated",
        "question_type": "direct",
        "source_chunk_ids": ["chunk-001"],
        "validation": None,
        "review_status": "generated",
        "review_notes": None,
        "reviewed_at": None,
        "fact_id": None,
        "evidence_quote": None,
        "normalized_question_key": None,
        "original_question": None,
        "original_answer": None,
        "was_edited": False,
    }
    row.update(overrides)
    return row


class CourseScopingTests(unittest.TestCase):
    """Every read and write names the course. This is the isolation guarantee."""

    def test_seed_reads_and_writes_are_course_scoped(self) -> None:
        cases = [
            (lambda c: db_seeds.list_seeds(c, COURSE), [[]]),
            (lambda c: db_seeds.get_seed(c, COURSE, "-Oseed001"), [[]]),
            (lambda c: db_seeds.count_seeds_by_review_status(c, COURSE), [[]]),
            (lambda c: db_seeds.delete_seed(c, COURSE, "-Oseed001"), [1]),
        ]
        for call, results in cases:
            connection = FakeConnection(results)
            call(connection)
            for statement in connection.sql:
                self.assertIn("course_id = %s", statement)

    def test_delete_seed_binds_course_and_seed_together(self) -> None:
        connection = FakeConnection([1])
        db_seeds.delete_seed(connection, COURSE, "-Oseed001")

        statement, params = connection.statements[0]
        self.assertIn("DELETE FROM seed_examples", statement)
        self.assertIn("WHERE course_id = %s AND seed_id = %s", statement)
        self.assertEqual(params, (COURSE, "-Oseed001"))

    def test_delete_evaluation_binds_course_and_evaluation_together(self) -> None:
        connection = FakeConnection([1])
        db_evaluations.delete_evaluation(connection, COURSE, "-Oeval001")

        statement, params = connection.statements[0]
        self.assertIn("WHERE course_id = %s AND evaluation_id = %s", statement)
        self.assertEqual(params, (COURSE, "-Oeval001"))

    def test_bulk_evaluation_delete_is_never_unscoped(self) -> None:
        connection = FakeConnection([3])
        deleted = db_evaluations.delete_all_evaluations(connection, COURSE)

        statement, params = connection.statements[0]
        self.assertEqual(
            statement, "DELETE FROM evaluations WHERE course_id = %s"
        )
        self.assertEqual(params, (COURSE,))
        self.assertEqual(deleted, 3)

    def test_a_seed_id_from_another_course_returns_nothing(self) -> None:
        """The id alone is never enough; the pair is the key."""
        connection = FakeConnection([[]])
        found = db_seeds.get_seed(connection, OTHER_COURSE, "-Oseed001")

        self.assertIsNone(found)
        self.assertEqual(connection.params[0], (OTHER_COURSE, "-Oseed001"))

    def test_invalid_course_ids_never_reach_a_statement(self) -> None:
        connection = FakeConnection()
        for bad in ("CSS 350", "../etc", "css_350", "-leading", ""):
            with self.assertRaises(ValueError):
                db_seeds.list_seeds(connection, bad)
        self.assertEqual(connection.statements, [])


class ParameterizationTests(unittest.TestCase):
    """Values are bound; only column names this layer owns reach SQL text."""

    def test_update_seed_binds_values_and_never_interpolates_them(self) -> None:
        connection = FakeConnection([[seed_row()], 1, [seed_row(category="grading")]])
        db_seeds.update_seed(
            connection,
            COURSE,
            "-Oseed001",
            {"category": "grading'; DROP TABLE seed_examples; --"},
        )

        update = next(s for s in connection.sql if s.startswith("UPDATE"))
        self.assertNotIn("DROP TABLE", update)
        self.assertIn("category = %(category)s", update)
        parameters = connection.params[1]
        self.assertEqual(
            parameters["category"], "grading'; DROP TABLE seed_examples; --"
        )

    def test_a_field_outside_the_allowlist_cannot_name_a_column(self) -> None:
        connection = FakeConnection([[seed_row()], [seed_row()]])
        db_seeds.update_seed(
            connection,
            COURSE,
            "-Oseed001",
            {"course_id": OTHER_COURSE, "seed_id": "x", "notARealField": 1},
        )

        # Nothing was assignable, so no UPDATE ran at all.
        self.assertFalse([s for s in connection.sql if s.startswith("UPDATE")])

    def test_a_patch_cannot_move_a_seed_to_another_course(self) -> None:
        connection = FakeConnection([[seed_row()], 1, [seed_row()]])
        db_seeds.update_seed(
            connection, COURSE, "-Oseed001", {"courseId": OTHER_COURSE, "notes": "n"}
        )

        update = next(s for s in connection.sql if s.startswith("UPDATE"))
        self.assertNotIn("course_id =", update.split("WHERE")[0])
        self.assertEqual(connection.params_for("UPDATE")["course_id"], COURSE)

    def test_course_patch_binds_every_value(self) -> None:
        connection = FakeConnection([[{"course_id": COURSE}], 1, [], []])
        db_courses.update_course(connection, COURSE, {"chunkCount": 42})

        update = next(s for s in connection.sql if s.startswith("UPDATE"))
        self.assertIn("chunk_count = %(chunk_count)s", update)
        self.assertNotIn("42", update)


class SeedWriteTests(unittest.TestCase):
    def test_create_seed_requires_text_under_either_name(self) -> None:
        connection = FakeConnection()
        with self.assertRaises(ValueError):
            db_seeds.create_seed(connection, COURSE, {"question": "Q?"})
        self.assertEqual(connection.statements, [])

    def test_create_seed_accepts_the_dual_names(self) -> None:
        connection = FakeConnection([1, [seed_row()]])
        db_seeds.create_seed(connection, COURSE, {"question": "Q?", "answer": "A."})

        insert = connection.sql[0]
        self.assertIn("INSERT INTO seed_examples", insert)
        parameters = connection.params[0]
        self.assertEqual(parameters["instruction"], "Q?")
        self.assertEqual(parameters["response"], "A.")
        self.assertEqual(parameters["course_id"], COURSE)

    def test_created_seeds_get_a_distinguishable_id(self) -> None:
        """Imported push ids are real references; new rows must not mimic them."""
        connection = FakeConnection([1, [seed_row()]])
        db_seeds.create_seed(connection, COURSE, {"question": "Q?", "answer": "A."})

        seed_id = connection.params[0]["seed_id"]
        self.assertTrue(seed_id.startswith("seed-"))
        self.assertFalse(seed_id.startswith("-O"))

    def test_jsonb_values_are_wrapped_for_the_driver(self) -> None:
        from psycopg.types.json import Json

        connection = FakeConnection([1, [seed_row()]])
        db_seeds.create_seed(
            connection,
            COURSE,
            {
                "question": "Q?",
                "answer": "A.",
                "sourceChunkIds": ["chunk-001"],
                "validation": {"score": 0.9},
            },
        )

        parameters = connection.params[0]
        self.assertIsInstance(parameters["source_chunk_ids"], Json)
        self.assertIsInstance(parameters["validation"], Json)
        # Plain columns stay plain.
        self.assertIsInstance(parameters["instruction"], str)

    def test_review_reuses_the_shared_provenance_helper(self) -> None:
        """Editing text must snapshot the original, as the review helper requires."""
        edited = seed_row(instruction="New?", original_question="Q?")
        connection = FakeConnection([[seed_row()], [seed_row()], 1, [edited]])
        db_seeds.review_seed(
            connection,
            COURSE,
            "-Oseed001",
            review_status="approved",
            question="New?",
            answer="A.",
        )

        parameters = connection.params_for("UPDATE")
        self.assertEqual(parameters["instruction"], "New?")
        self.assertEqual(parameters["original_question"], "Q?")
        self.assertTrue(parameters["was_edited"])
        self.assertEqual(parameters["review_status"], "approved")

    def test_review_of_a_missing_seed_returns_none_without_writing(self) -> None:
        connection = FakeConnection([[]])
        result = db_seeds.review_seed(
            connection, COURSE, "missing", review_status="approved"
        )
        self.assertIsNone(result)
        self.assertFalse([s for s in connection.sql if s.startswith("UPDATE")])

    def test_review_status_counts_fall_back_to_legacy_status(self) -> None:
        connection = FakeConnection([[{"bucket": "approved", "total": 12}]])
        counts = db_seeds.count_seeds_by_review_status(connection, COURSE)

        self.assertIn("COALESCE(review_status, status, 'generated')", connection.sql[0])
        self.assertEqual(counts, {"approved": 12})


class CourseWriteTests(unittest.TestCase):
    def test_create_course_refuses_a_duplicate_via_the_database(self) -> None:
        connection = FakeConnection([0])
        with self.assertRaises(db_courses.CourseAlreadyExistsError):
            db_courses.create_course(
                connection,
                COURSE,
                {
                    "name": "CSS 350",
                    "title": "Management",
                    "term": "Spring 2026",
                    "instructorName": "K. Champion",
                },
            )
        self.assertIn("ON CONFLICT (course_id) DO NOTHING", connection.sql[0])

    def test_update_of_a_missing_course_returns_none(self) -> None:
        connection = FakeConnection([[]])
        self.assertIsNone(
            db_courses.update_course(connection, COURSE, {"chunkCount": 5})
        )

    def test_starter_generation_upsert_merges_rather_than_replaces(self) -> None:
        connection = FakeConnection([1, []])
        db_courses.upsert_starter_seed_generation(
            connection, COURSE, {"status": "generating"}
        )

        statement = connection.sql[0]
        self.assertIn("INSERT INTO starter_seed_generation", statement)
        self.assertIn("ON CONFLICT (course_id) DO UPDATE SET status", statement)
        # Only the field sent is touched; counts written earlier survive.
        self.assertNotIn("target_count", statement)

    def test_metadata_nests_starter_generation_when_a_row_exists(self) -> None:
        course_row = {
            "course_id": COURSE,
            "name": "CSS 350",
            "title": "Management",
            "term": "Spring 2026",
            "instructor_name": "K. Champion",
            "created_at": UTC_NOON,
            "syllabus_status": "indexed",
            "syllabus_file_name": None,
            "syllabus_type": None,
            "chunk_count": 180,
        }
        starter = {
            "course_id": COURSE,
            "status": "ready",
            "target_count": 50,
            "final_count": 50,
            "saved_count": 50,
            "failed_to_save_count": 0,
            "error": None,
            "started_at": UTC_NOON,
            "completed_at": UTC_NOON,
        }
        connection = FakeConnection([[course_row], [starter]])
        course = db_courses.get_course(connection, COURSE)

        self.assertEqual(course["courseId"], COURSE)
        self.assertEqual(
            course["metadata"]["starterSeedGeneration"]["status"], "ready"
        )

    def test_metadata_omits_starter_generation_when_absent(self) -> None:
        course_row = {
            "course_id": COURSE,
            "name": "CSS 350",
            "title": "Management",
            "term": "Spring 2026",
            "instructor_name": "K. Champion",
            "created_at": UTC_NOON,
            "syllabus_status": "none",
            "syllabus_file_name": None,
            "syllabus_type": None,
            "chunk_count": 0,
        }
        connection = FakeConnection([[course_row], []])
        course = db_courses.get_course(connection, COURSE)

        self.assertNotIn("starterSeedGeneration", course["metadata"])


class ModelRepositoryTests(unittest.TestCase):
    VERSION_ROW = {
        "course_id": OTHER_COURSE,
        "version": "v1",
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "training_example_count": 54,
        "status": "ready",
        "deployment": "offline",
        "artifact_ref": "css-360-qlora/adapter",
        "created_at": UTC_NOON,
        "updated_at": UTC_NOON,
        "notes": None,
    }

    def test_registry_nests_versions_keyed_by_version(self) -> None:
        connection = FakeConnection(
            [[{"course_id": OTHER_COURSE, "current_version": "v1"}], [self.VERSION_ROW]]
        )
        registry = db_models.get_model_registry(connection, OTHER_COURSE)

        self.assertEqual(registry["currentVersion"], "v1")
        self.assertEqual(set(registry["versions"]), {"v1"})
        self.assertEqual(registry["versions"]["v1"]["trainingExampleCount"], 54)

    def test_no_model_row_means_no_registry(self) -> None:
        connection = FakeConnection([[]])
        self.assertIsNone(db_models.get_model_registry(connection, OTHER_COURSE))

    def test_a_pointer_with_no_versions_is_not_a_registry(self) -> None:
        connection = FakeConnection(
            [[{"course_id": OTHER_COURSE, "current_version": "v1"}], []]
        )
        self.assertIsNone(db_models.get_model_registry(connection, OTHER_COURSE))

    def test_registering_a_version_does_not_promote_it_by_default(self) -> None:
        connection = FakeConnection([1, [], []])
        db_models.upsert_model_version(
            connection,
            OTHER_COURSE,
            {
                "version": "v2",
                "baseModel": "meta-llama/Llama-3.2-3B-Instruct",
                "status": "ready",
                "artifactRef": "css-360-qlora/adapter-v2",
                "createdAt": UTC_NOON.isoformat(),
            },
        )
        self.assertFalse(
            [s for s in connection.sql if "INSERT INTO course_models" in s]
        )

    def test_promotion_is_explicit(self) -> None:
        connection = FakeConnection([1, 1, [], []])
        db_models.upsert_model_version(
            connection,
            OTHER_COURSE,
            {
                "version": "v2",
                "baseModel": "m",
                "status": "ready",
                "artifactRef": "a",
                "createdAt": UTC_NOON.isoformat(),
            },
            set_current=True,
        )
        self.assertTrue(
            [s for s in connection.sql if "INSERT INTO course_models" in s]
        )

    def test_cannot_point_at_a_version_the_course_does_not_have(self) -> None:
        connection = FakeConnection([[]])
        self.assertIsNone(
            db_models.set_current_version(connection, OTHER_COURSE, "v9")
        )


class ModelRequestRepositoryTests(unittest.TestCase):
    def test_create_guards_against_a_second_active_request(self) -> None:
        connection = FakeConnection([0])
        with self.assertRaises(db_model_requests.ActiveModelRequestError):
            db_model_requests.create_model_request(connection, COURSE, 54)

        statement = connection.sql[0]
        self.assertIn("WHERE NOT EXISTS", statement)
        self.assertIn("status = ANY(%(active_statuses)s)", statement)
        self.assertEqual(
            connection.params[0]["active_statuses"],
            ["requested", "preparing", "training"],
        )

    def test_terminal_statuses_do_not_block_a_new_request(self) -> None:
        self.assertNotIn("ready", db_model_requests.ACTIVE_STATUSES)
        self.assertNotIn("failed", db_model_requests.ACTIVE_STATUSES)

    def test_update_merges_and_stamps_updated_at(self) -> None:
        existing = {
            "course_id": COURSE,
            "status": "requested",
            "requested_at": UTC_NOON,
            "updated_at": UTC_NOON,
            "approved_example_count": 54,
            "failure_message": None,
            "preparation": None,
            "preparation_error": None,
            "training": None,
            "launch_error": None,
            "current_run_id": None,
        }
        connection = FakeConnection([[existing], 1, [existing]])
        db_model_requests.update_model_request(
            connection, COURSE, {"status": "preparing"}
        )

        update = next(s for s in connection.sql if s.startswith("UPDATE"))
        self.assertIn("status = %(status)s", update)
        self.assertIn("updated_at = %(updated_at)s", update)
        # requestedAt and the approved count are never touched by a merge.
        self.assertNotIn("requested_at =", update)
        self.assertNotIn("approved_example_count =", update)

    def test_nested_blocks_are_wrapped_as_jsonb(self) -> None:
        from psycopg.types.json import Json

        existing = {
            "course_id": COURSE,
            "status": "requested",
            "requested_at": UTC_NOON,
            "updated_at": UTC_NOON,
            "approved_example_count": 54,
            "failure_message": None,
            "preparation": None,
            "preparation_error": None,
            "training": None,
            "launch_error": None,
            "current_run_id": None,
        }
        connection = FakeConnection([[existing], 1, [existing]])
        db_model_requests.update_model_request(
            connection, COURSE, {"preparation": {"datasetRef": "exports/x"}}
        )
        self.assertIsInstance(connection.params_for("UPDATE")["preparation"], Json)

    def test_update_of_a_missing_request_returns_none(self) -> None:
        connection = FakeConnection([[]])
        self.assertIsNone(
            db_model_requests.update_model_request(connection, COURSE, {"status": "ready"})
        )


def _queued_run_row(**overrides: Any) -> dict[str, Any]:
    row = {
        "run_id": "run-1",
        "course_id": COURSE,
        "mode": "full",
        "state": "queued",
        "enqueued_at": UTC_NOON,
        "updated_at": UTC_NOON,
        "dataset_ref": "exports/x",
        "approved_example_count": 0,
        "train_examples": 0,
        "validation_examples": 0,
        "attempt": 0,
        "job_id": None,
        "claim_owner": None,
        "claim_claimed_at": None,
        "claim_expires_at": None,
        "error": None,
    }
    row.update(overrides)
    return row


class TrainingRunRepositoryTests(unittest.TestCase):
    def test_enqueue_guards_against_a_second_active_run(self) -> None:
        connection = FakeConnection([0])
        with self.assertRaises(db_training_runs.ActiveTrainingRunError):
            db_training_runs.enqueue_training_run(
                connection, COURSE, mode="full", dataset_ref="exports/x"
            )

        statement = connection.sql[0]
        self.assertIn("WHERE NOT EXISTS", statement)
        self.assertIn("NOT (state = ANY(%(terminal_states)s))", statement)
        self.assertEqual(
            connection.params[0]["terminal_states"], ["succeeded", "failed"]
        )

    def test_enqueue_never_invents_a_job_id(self) -> None:
        connection = FakeConnection([1, [_queued_run_row()]])
        db_training_runs.enqueue_training_run(
            connection, COURSE, mode="smoke", dataset_ref="exports/x"
        )
        parameters = connection.params[0]
        self.assertNotIn("job_id", parameters)
        self.assertEqual(parameters["state"], "queued")
        self.assertEqual(parameters["attempt"], 0)

    def test_enqueue_rejects_an_unknown_mode(self) -> None:
        connection = FakeConnection()
        with self.assertRaises(ValueError):
            db_training_runs.enqueue_training_run(
                connection, COURSE, mode="turbo", dataset_ref="exports/x"
            )
        self.assertEqual(connection.statements, [])

    def test_a_nested_claim_patch_flattens_to_columns(self) -> None:
        run_row = {
            "run_id": "run-1",
            "course_id": COURSE,
            "mode": "full",
            "state": "queued",
            "enqueued_at": UTC_NOON,
            "updated_at": UTC_NOON,
            "dataset_ref": "exports/x",
            "approved_example_count": 0,
            "train_examples": 0,
            "validation_examples": 0,
            "attempt": 0,
            "job_id": None,
            "claim_owner": None,
            "claim_claimed_at": None,
            "claim_expires_at": None,
            "error": None,
        }
        connection = FakeConnection([[run_row], 1, [run_row]])
        db_training_runs.update_training_run(
            connection,
            COURSE,
            "run-1",
            {
                "state": "claimed",
                "claim": {
                    "owner": "runner-1",
                    "claimedAt": UTC_NOON.isoformat(),
                    "expiresAt": UTC_NOON.isoformat(),
                },
            },
        )

        parameters = connection.params_for("UPDATE")
        self.assertEqual(parameters["claim_owner"], "runner-1")
        self.assertEqual(parameters["state"], "claimed")

    def test_releasing_a_claim_clears_all_three_columns(self) -> None:
        run_row = {
            "run_id": "run-1",
            "course_id": COURSE,
            "mode": "full",
            "state": "claimed",
            "enqueued_at": UTC_NOON,
            "updated_at": UTC_NOON,
            "dataset_ref": "exports/x",
            "approved_example_count": 0,
            "train_examples": 0,
            "validation_examples": 0,
            "attempt": 1,
            "job_id": None,
            "claim_owner": "runner-1",
            "claim_claimed_at": UTC_NOON,
            "claim_expires_at": UTC_NOON,
            "error": None,
        }
        connection = FakeConnection([[run_row], 1, [run_row]])
        db_training_runs.update_training_run(
            connection, COURSE, "run-1", {"claim": None}
        )

        parameters = connection.params_for("UPDATE")
        self.assertIsNone(parameters["claim_owner"])
        self.assertIsNone(parameters["claim_claimed_at"])
        self.assertIsNone(parameters["claim_expires_at"])

    def test_run_update_is_keyed_by_course_and_run(self) -> None:
        run_row = {
            "run_id": "run-1",
            "course_id": COURSE,
            "mode": "full",
            "state": "queued",
            "enqueued_at": UTC_NOON,
            "updated_at": UTC_NOON,
            "dataset_ref": "exports/x",
            "approved_example_count": 0,
            "train_examples": 0,
            "validation_examples": 0,
            "attempt": 0,
            "job_id": None,
            "claim_owner": None,
            "claim_claimed_at": None,
            "claim_expires_at": None,
            "error": None,
        }
        connection = FakeConnection([[run_row], 1, [run_row]])
        db_training_runs.update_training_run(
            connection, COURSE, "run-1", {"state": "training"}
        )

        update = next(s for s in connection.sql if s.startswith("UPDATE"))
        self.assertIn(
            "WHERE course_id = %(course_id)s AND run_id = %(run_id)s", update
        )

    def test_active_run_lookup_excludes_terminal_states(self) -> None:
        connection = FakeConnection([[]])
        db_training_runs.find_active_training_run(connection, COURSE)

        statement = connection.sql[0]
        self.assertIn("NOT (state = ANY(%s))", statement)
        self.assertEqual(connection.params[0], (COURSE, ["succeeded", "failed"]))


class QueueClaimSqlTests(unittest.TestCase):
    """The claim is the one place two workers can collide.

    What makes it safe is in the SQL, not in the worker: a single eligible row
    is selected `FOR UPDATE SKIP LOCKED` and the lease is stamped against that
    already-locked row. These assert the statement really says so — a claim
    that lost `SKIP LOCKED` would still pass every behavioural test and start
    handing the same run to two runners under load.
    """

    def _row(self, course_id: str = COURSE, run_id: str = "run-1") -> dict[str, Any]:
        return {"course_id": course_id, "run_id": run_id}

    def test_claim_selects_one_row_for_update_skip_locked(self) -> None:
        connection = FakeConnection([[self._row()], 1, []])
        db_training_runs.claim_next_training_run(
            connection, owner="runner@node", lease_seconds=900, now=UTC_NOON
        )

        select = connection.sql[0]
        self.assertIn("FOR UPDATE SKIP LOCKED", select)
        self.assertIn("LIMIT 1", select)
        self.assertIn("ORDER BY enqueued_at ASC", select)

    def test_claim_never_takes_a_run_that_already_has_a_job(self) -> None:
        """A job id means a submission happened; retaking it double-submits."""
        connection = FakeConnection([[self._row()], 1, []])
        db_training_runs.claim_next_training_run(
            connection, owner="runner@node", now=UTC_NOON
        )
        self.assertIn("job_id IS NULL", connection.sql[0])

    def test_claim_takes_queued_runs_and_expired_leases_only(self) -> None:
        connection = FakeConnection([[self._row()], 1, []])
        db_training_runs.claim_next_training_run(
            connection, owner="runner@node", now=UTC_NOON
        )

        select = connection.sql[0]
        self.assertIn("state = 'queued'", select)
        self.assertIn("state = 'claimed'", select)
        self.assertIn("claim_expires_at <= %(now)s", select)
        # submitted / training / succeeded / failed are not claimable.
        self.assertNotIn("'submitted'", select)

    def test_claim_stamps_the_lease_on_the_locked_row(self) -> None:
        connection = FakeConnection([[self._row()], 1, []])
        db_training_runs.claim_next_training_run(
            connection, owner="runner@node", lease_seconds=600, now=UTC_NOON
        )

        update = connection.sql[1]
        self.assertIn("UPDATE training_runs SET", update)
        self.assertIn("state = 'claimed'", update)
        self.assertIn("attempt = attempt + 1", update)
        self.assertIn(
            "WHERE course_id = %(course_id)s AND run_id = %(run_id)s", update
        )

        parameters = connection.params[1]
        self.assertEqual(parameters["owner"], "runner@node")
        self.assertEqual(
            parameters["expires_at"], UTC_NOON + timedelta(seconds=600)
        )

    def test_no_eligible_row_claims_nothing_and_updates_nothing(self) -> None:
        """No work is not an error, and must not write."""
        connection = FakeConnection([[]])
        claimed = db_training_runs.claim_next_training_run(
            connection, owner="runner@node", now=UTC_NOON
        )

        self.assertIsNone(claimed)
        self.assertEqual(len(connection.sql), 1)
        self.assertFalse(
            [statement for statement in connection.sql if statement.startswith("UPDATE")]
        )

    def test_a_claim_can_be_scoped_to_named_courses(self) -> None:
        connection = FakeConnection([[self._row()], 1, []])
        db_training_runs.claim_next_training_run(
            connection,
            owner="runner@node",
            now=UTC_NOON,
            course_ids=[COURSE],
        )

        self.assertIn("course_id = ANY(%(course_ids)s::text[])", connection.sql[0])
        self.assertEqual(connection.params[0]["course_ids"], [COURSE])

    def test_an_unscoped_claim_binds_null_rather_than_naming_every_course(self) -> None:
        connection = FakeConnection([[self._row()], 1, []])
        db_training_runs.claim_next_training_run(
            connection, owner="runner@node", now=UTC_NOON
        )
        self.assertIsNone(connection.params[0]["course_ids"])

    def test_a_claim_needs_an_owner_and_a_positive_lease(self) -> None:
        with self.assertRaises(ValueError):
            db_training_runs.claim_next_training_run(FakeConnection(), owner="  ")
        with self.assertRaises(ValueError):
            db_training_runs.claim_next_training_run(
                FakeConnection(), owner="runner", lease_seconds=0
            )

    def test_release_clears_the_whole_lease_not_half_of_it(self) -> None:
        connection = FakeConnection([[{"run_id": "run-1", "course_id": COURSE,
                                       "mode": "full", "state": "claimed",
                                       "enqueued_at": UTC_NOON,
                                       "updated_at": UTC_NOON,
                                       "dataset_ref": "exports/x"}], 1, []])
        db_training_runs.release_training_run(
            connection, COURSE, "run-1", error="nothing was submitted", now=UTC_NOON
        )

        update = next(s for s in connection.sql if s.startswith("UPDATE"))
        for column in ("claim_owner", "claim_claimed_at", "claim_expires_at"):
            self.assertIn(column, update)
        self.assertIn("state", update)

    def test_a_failed_run_is_terminal_and_unleased(self) -> None:
        row = {
            "run_id": "run-1",
            "course_id": COURSE,
            "mode": "full",
            "state": "training",
            "enqueued_at": UTC_NOON,
            "updated_at": UTC_NOON,
            "dataset_ref": "exports/x",
        }
        connection = FakeConnection([[row], 1, [{**row, "state": "failed"}]])
        db_training_runs.fail_training_run(
            connection, COURSE, "run-1", error="the job died", now=UTC_NOON
        )

        update = next(s for s in connection.sql if s.startswith("UPDATE"))
        self.assertIn("claim_owner", update)
        parameters = next(
            p for s, p in connection.statements if s.startswith("UPDATE")
        )
        self.assertEqual(parameters["state"], "failed")
        self.assertEqual(parameters["error"], "the job died")
        self.assertIsNone(parameters["claim_owner"])

    def test_a_submission_records_the_job_id_and_drops_the_lease(self) -> None:
        row = {
            "run_id": "run-1",
            "course_id": COURSE,
            "mode": "full",
            "state": "claimed",
            "enqueued_at": UTC_NOON,
            "updated_at": UTC_NOON,
            "dataset_ref": "exports/x",
        }
        connection = FakeConnection([[row], 1, [{**row, "state": "submitted"}]])
        db_training_runs.mark_training_run_submitted(
            connection,
            COURSE,
            "run-1",
            job_id="123456",
            train_examples=48,
            validation_examples=6,
            now=UTC_NOON,
        )

        parameters = next(
            p for s, p in connection.statements if s.startswith("UPDATE")
        )
        self.assertEqual(parameters["state"], "submitted")
        self.assertEqual(parameters["job_id"], "123456")
        self.assertEqual(parameters["train_examples"], 48)
        self.assertIsNone(parameters["claim_owner"])

    def test_a_submission_without_a_job_id_is_refused(self) -> None:
        """A placeholder id would make an unsubmitted run look submitted."""
        with self.assertRaises(ValueError):
            db_training_runs.mark_training_run_submitted(
                FakeConnection(),
                COURSE,
                "run-1",
                job_id="   ",
                train_examples=1,
                validation_examples=1,
            )

    def test_claimable_listing_is_read_only(self) -> None:
        connection = FakeConnection([[]])
        db_training_runs.claimable_training_runs(connection, now=UTC_NOON)

        self.assertEqual(len(connection.sql), 1)
        self.assertTrue(connection.sql[0].startswith("SELECT"))
        self.assertNotIn("FOR UPDATE", connection.sql[0])


class EvaluationWriteTests(unittest.TestCase):
    """What a rating stores, now that the student form asks for less.

    `mostHelpful`, `mostConcise` and `bestGrounded` were retired from the form.
    Their columns are `NOT NULL` and stay that way — a migration against a
    database holding live research data would record nothing the empty string
    does not — so an unanswered criterion is written as `''` and read back as
    absent.
    """

    SIMPLIFIED = {
        "id": "-Oeval900",
        "comparisonId": "question-run-9",
        "mostAccurate": "fineTuned",
        "preferredModel": "fineTunedRag",
        "hallucinationFlags": ["base"],
        "createdAt": "2026-03-01T00:00:00+00:00",
    }

    def created_row(self, **overrides: Any) -> dict[str, Any]:
        row = {
            "evaluation_id": "-Oeval900",
            "course_id": COURSE,
            "comparison_id": "question-run-9",
            "most_accurate": "fineTuned",
            "most_helpful": "",
            "most_concise": "",
            "best_grounded": "",
            "preferred_model": "fineTunedRag",
            "hallucination_flags": ["base"],
            "comment": None,
            "created_at": UTC_NOON,
            "run_id": None,
            "question_text": None,
        }
        row.update(overrides)
        return row

    def test_a_rating_without_the_retired_criteria_is_accepted(self) -> None:
        connection = FakeConnection([1, [self.created_row()]])
        created = db_evaluations.create_evaluation(
            connection, COURSE, self.SIMPLIFIED
        )

        parameters = connection.params_for("INSERT INTO evaluations")
        self.assertEqual(parameters["most_helpful"], "")
        self.assertEqual(parameters["most_concise"], "")
        self.assertEqual(parameters["best_grounded"], "")
        self.assertEqual(parameters["most_accurate"], "fineTuned")
        self.assertEqual(parameters["preferred_model"], "fineTunedRag")
        # Read back, an unanswered criterion is absent rather than empty.
        for key in ("mostHelpful", "mostConcise", "bestGrounded"):
            self.assertNotIn(key, created)

    def test_a_rating_that_still_sends_them_stores_them_unchanged(self) -> None:
        connection = FakeConnection(
            [
                1,
                [
                    self.created_row(
                        most_helpful="rag", most_concise="base", best_grounded="rag"
                    )
                ],
            ]
        )
        created = db_evaluations.create_evaluation(
            connection,
            COURSE,
            {
                **self.SIMPLIFIED,
                "mostHelpful": "rag",
                "mostConcise": "base",
                "bestGrounded": "rag",
            },
        )

        parameters = connection.params_for("INSERT INTO evaluations")
        self.assertEqual(parameters["most_helpful"], "rag")
        self.assertEqual(created["bestGrounded"], "rag")

    def test_the_criteria_the_form_still_asks_for_stay_required(self) -> None:
        for missing in ("mostAccurate", "preferredModel"):
            payload = {
                key: value
                for key, value in self.SIMPLIFIED.items()
                if key != missing
            }
            with self.assertRaises(ValueError):
                db_evaluations.create_evaluation(FakeConnection([1]), COURSE, payload)

    def test_the_write_is_course_scoped(self) -> None:
        connection = FakeConnection([1, [self.created_row()]])
        db_evaluations.create_evaluation(connection, COURSE, self.SIMPLIFIED)

        parameters = connection.params_for("INSERT INTO evaluations")
        self.assertEqual(parameters["course_id"], COURSE)


class OrderingTests(unittest.TestCase):
    """Ordering matches the frontend parsers, so no list reshuffles."""

    def test_courses_sort_newest_first_then_by_id(self) -> None:
        connection = FakeConnection([[]])
        db_courses.list_courses(connection)
        self.assertIn("ORDER BY created_at DESC, course_id ASC", connection.sql[0])

    def test_seeds_sort_newest_first_with_undated_last(self) -> None:
        connection = FakeConnection([[]])
        db_seeds.list_seeds(connection, COURSE)
        self.assertIn("ORDER BY created_at DESC NULLS LAST", connection.sql[0])

    def test_evaluations_sort_newest_first(self) -> None:
        connection = FakeConnection([[]])
        db_evaluations.list_evaluations(connection, COURSE)
        self.assertIn("ORDER BY created_at DESC", connection.sql[0])

    def test_training_runs_sort_oldest_first(self) -> None:
        connection = FakeConnection([[]])
        db_training_runs.list_training_runs(connection, COURSE)
        self.assertIn("ORDER BY enqueued_at ASC", connection.sql[0])


if __name__ == "__main__":
    unittest.main()
