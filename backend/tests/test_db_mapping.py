"""Row -> API mapping for the PostgreSQL repositories.

Pure functions over dict rows: no server, no connection, no network. What these
pin down is the contract with the frontend types — a column rename or a lost
alias shows up here rather than in a browser after cutover.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from app.db_courses import map_course_metadata, map_starter_seed_generation
from app.db_evaluations import map_evaluation
from app.db_mapping import (
    build_patch,
    put_optional,
    string_list,
    to_iso,
    update_statement,
)
from app.db_model_requests import map_model_request
from app.db_models import map_model_version
from app.db_seeds import SEED_PATCH_COLUMNS, map_seed
from app.db_training_runs import generate_training_run_id, map_training_run

UTC_NOON = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)


class TimestampSerializationTests(unittest.TestCase):
    def test_aware_timestamp_serializes_as_utc(self) -> None:
        self.assertEqual(to_iso(UTC_NOON), "2026-08-13T12:00:00+00:00")

    def test_non_utc_offset_is_converted_not_relabelled(self) -> None:
        pacific = datetime(
            2026, 8, 13, 5, 0, tzinfo=timezone(timedelta(hours=-7))
        )
        self.assertEqual(to_iso(pacific), "2026-08-13T12:00:00+00:00")

    def test_naive_timestamp_is_treated_as_utc(self) -> None:
        self.assertEqual(
            to_iso(datetime(2026, 8, 13, 12, 0)), "2026-08-13T12:00:00+00:00"
        )

    def test_null_stays_null(self) -> None:
        self.assertIsNone(to_iso(None))

    def test_every_timestamp_shares_one_shape(self) -> None:
        """Frontend sorting is `localeCompare` on the string, so shape matters."""
        rendered = [
            to_iso(UTC_NOON),
            to_iso(UTC_NOON.replace(microsecond=500)),
            to_iso(datetime(2026, 1, 1, 0, 0)),
        ]
        for value in rendered:
            self.assertTrue(value.endswith("+00:00"), value)


class HelperTests(unittest.TestCase):
    def test_put_optional_skips_none_and_blank(self) -> None:
        target: dict[str, object] = {}
        put_optional(target, "a", None)
        put_optional(target, "b", "  ")
        put_optional(target, "c", "kept")
        put_optional(target, "d", 0)
        self.assertEqual(target, {"c": "kept", "d": 0})

    def test_string_list_normalizes_index_keyed_objects(self) -> None:
        self.assertEqual(
            string_list({"0": "chunk-001", "2": "chunk-003"}),
            ["chunk-001", "chunk-003"],
        )
        self.assertEqual(string_list(None), [])
        self.assertEqual(string_list(["a", "", "  ", "b"]), ["a", "b"])

    def test_build_patch_drops_unknown_fields(self) -> None:
        """The allowlist is what keeps caller keys out of SQL."""
        assignments = build_patch(
            {"category": "grading", "droppedTable": "x", "sourceSection": "Grading"},
            SEED_PATCH_COLUMNS,
        )
        self.assertEqual(
            assignments, {"category": "grading", "source_section": "Grading"}
        )

    def test_build_patch_maps_dual_names_to_one_column(self) -> None:
        self.assertEqual(
            build_patch({"question": "Q?"}, SEED_PATCH_COLUMNS),
            {"instruction": "Q?"},
        )
        self.assertEqual(
            build_patch({"answer": "A."}, SEED_PATCH_COLUMNS), {"response": "A."}
        )

    def test_update_statement_binds_every_value(self) -> None:
        sql = update_statement(
            table="seed_examples",
            assignments={"category": "grading"},
            key_columns=["course_id", "seed_id"],
        )
        self.assertEqual(
            sql,
            "UPDATE seed_examples SET category = %(category)s "
            "WHERE course_id = %(course_id)s AND seed_id = %(seed_id)s",
        )

    def test_update_statement_refuses_an_empty_patch(self) -> None:
        with self.assertRaises(ValueError):
            update_statement(
                table="courses", assignments={}, key_columns=["course_id"]
            )


class CourseMappingTests(unittest.TestCase):
    ROW = {
        "course_id": "css-350-spring-2026-n3h9",
        "name": "CSS 350",
        "title": "Management principals",
        "term": "Spring 2026",
        "instructor_name": "Kaylea Champion",
        "created_at": UTC_NOON,
        "syllabus_status": "indexed",
        "syllabus_file_name": "Css 350.txt",
        "syllabus_type": "txt",
        "chunk_count": 180,
    }

    def test_maps_snake_case_columns_to_camel_case_metadata(self) -> None:
        metadata = map_course_metadata(self.ROW)
        self.assertEqual(metadata["instructorName"], "Kaylea Champion")
        self.assertEqual(metadata["syllabusFileName"], "Css 350.txt")
        self.assertEqual(metadata["syllabusStatus"], "indexed")
        self.assertEqual(metadata["chunkCount"], 180)
        self.assertEqual(metadata["createdAt"], "2026-08-13T12:00:00+00:00")

    def test_nullable_file_fields_stay_explicitly_null(self) -> None:
        """`isCourseMetadata` requires the keys; omitting them fails the guard."""
        metadata = map_course_metadata(
            {**self.ROW, "syllabus_file_name": None, "syllabus_type": None}
        )
        self.assertIsNone(metadata["syllabusFileName"])
        self.assertIsNone(metadata["syllabusType"])
        self.assertIn("syllabusFileName", metadata)
        self.assertIn("syllabusType", metadata)

    def test_starter_generation_maps_only_what_was_written(self) -> None:
        record = map_starter_seed_generation(
            {
                "course_id": "css-350-spring-2026-n3h9",
                "status": "partial",
                "target_count": 50,
                "final_count": 9,
                "saved_count": 9,
                "failed_to_save_count": 0,
                "error": None,
                "started_at": UTC_NOON,
                "completed_at": None,
            }
        )
        self.assertEqual(record["status"], "partial")
        self.assertEqual(record["targetCount"], 50)
        # A zero count is data, not absence.
        self.assertEqual(record["failedToSaveCount"], 0)
        self.assertNotIn("error", record)
        self.assertNotIn("completedAt", record)


class SeedMappingTests(unittest.TestCase):
    ROW = {
        "seed_id": "-Ozt97PxVXZ6vRu7v4F0",
        "course_id": "css-350-spring-2026-n3h9",
        "instruction": "Can I get an extension?",
        "response": "No extension is possible.",
        "category": "late work",
        "source_section": "Late Policy",
        "difficulty": "Medium",
        "directly_answered": True,
        "origin": "ai_generated",
        "notes": None,
        "created_at": UTC_NOON,
        "status": "generated",
        "question_type": "direct",
        "source_chunk_ids": ["chunk-092"],
        "validation": {"score": 0.92, "reason": "Grounded.", "components": {}},
        "review_status": "generated",
        "review_notes": None,
        "reviewed_at": None,
        "fact_id": "fact-02",
        "evidence_quote": "No extension is possible.",
        "normalized_question_key": "can i get an extension",
        "original_question": None,
        "original_answer": None,
        "was_edited": False,
    }

    def test_emits_both_name_pairs(self) -> None:
        """instruction/question and response/answer are both load-bearing."""
        seed = map_seed(self.ROW)
        self.assertEqual(seed["instruction"], seed["question"])
        self.assertEqual(seed["response"], seed["answer"])
        self.assertEqual(seed["instruction"], "Can I get an extension?")

    def test_preserves_an_imported_push_id(self) -> None:
        self.assertEqual(map_seed(self.ROW)["id"], "-Ozt97PxVXZ6vRu7v4F0")

    def test_jsonb_columns_round_trip_as_objects(self) -> None:
        seed = map_seed(self.ROW)
        self.assertEqual(seed["sourceChunkIds"], ["chunk-092"])
        self.assertEqual(seed["validation"]["score"], 0.92)
        self.assertIn("components", seed["validation"])

    def test_absent_optional_fields_are_omitted_not_nulled(self) -> None:
        seed = map_seed(self.ROW)
        for key in ("notes", "reviewNotes", "reviewedAt", "originalQuestion"):
            self.assertNotIn(key, seed)

    def test_review_fields_survive_when_present(self) -> None:
        seed = map_seed(
            {
                **self.ROW,
                "review_status": "edited",
                "review_notes": "Tightened wording.",
                "reviewed_at": UTC_NOON,
                "original_question": "Old?",
                "original_answer": "Old.",
                "was_edited": True,
            }
        )
        self.assertEqual(seed["reviewStatus"], "edited")
        self.assertEqual(seed["reviewedAt"], "2026-08-13T12:00:00+00:00")
        self.assertEqual(seed["originalQuestion"], "Old?")
        self.assertTrue(seed["wasEdited"])

    def test_every_documented_field_survives_the_round_trip(self) -> None:
        seed = map_seed(self.ROW)
        for key in (
            "instruction",
            "response",
            "category",
            "sourceSection",
            "difficulty",
            "directlyAnswered",
            "origin",
            "createdAt",
            "status",
            "questionType",
            "sourceChunkIds",
            "validation",
            "reviewStatus",
            "factId",
            "evidenceQuote",
            "normalizedQuestionKey",
            "wasEdited",
        ):
            self.assertIn(key, seed)


class EvaluationMappingTests(unittest.TestCase):
    ROW = {
        "evaluation_id": "-Oeval001",
        "course_id": "css-360-winter-2026-a7rp",
        "comparison_id": "cmp-3",
        "most_accurate": "rag",
        "most_helpful": "fineTunedRag",
        "most_concise": "base",
        "best_grounded": "rag",
        "preferred_model": "fineTunedRag",
        "hallucination_flags": ["base", "fineTuned"],
        "comment": "RAG cited the syllabus.",
        "created_at": UTC_NOON,
        "run_id": "run-9",
        "question_text": "When is the midterm?",
    }

    def test_maps_the_evaluation_record_shape(self) -> None:
        record = map_evaluation(self.ROW)
        self.assertEqual(record["id"], "-Oeval001")
        self.assertEqual(record["comparisonId"], "cmp-3")
        self.assertEqual(record["preferredModel"], "fineTunedRag")
        self.assertEqual(record["hallucinationFlags"], ["base", "fineTuned"])
        self.assertEqual(record["questionText"], "When is the midterm?")

    def test_empty_flags_stay_a_list(self) -> None:
        record = map_evaluation({**self.ROW, "hallucination_flags": []})
        self.assertEqual(record["hallucinationFlags"], [])

    def test_optional_fields_omitted_when_absent(self) -> None:
        record = map_evaluation(
            {**self.ROW, "comment": None, "run_id": None, "question_text": None}
        )
        for key in ("comment", "runId", "questionText"):
            self.assertNotIn(key, record)


class ModelRegistryMappingTests(unittest.TestCase):
    ROW = {
        "course_id": "css-360-winter-2026-a7rp",
        "version": "v1",
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "training_example_count": 54,
        "status": "ready",
        "deployment": "offline",
        "artifact_ref": "css-360-qlora/adapter",
        "created_at": UTC_NOON,
        "updated_at": UTC_NOON,
        "notes": "QLoRA adapter.",
    }

    def test_maps_a_version_the_frontend_parser_accepts(self) -> None:
        version = map_model_version(self.ROW)
        self.assertEqual(version["baseModel"], "meta-llama/Llama-3.2-3B-Instruct")
        self.assertEqual(version["trainingExampleCount"], 54)
        self.assertEqual(version["artifactRef"], "css-360-qlora/adapter")
        self.assertEqual(version["deployment"], "offline")

    def test_unrecorded_deployment_becomes_unknown(self) -> None:
        version = map_model_version({**self.ROW, "deployment": None})
        self.assertEqual(version["deployment"], "unknown")


class ModelRequestMappingTests(unittest.TestCase):
    ROW = {
        "course_id": "css-360-winter-2026-a7rp",
        "status": "training",
        "requested_at": UTC_NOON,
        "updated_at": UTC_NOON,
        "approved_example_count": 54,
        "failure_message": None,
        "preparation": {
            "preparedAt": "2026-08-12T01:00:00+00:00",
            "datasetRef": "exports/css-360-winter-2026-a7rp",
            "trainExamples": 48,
            "validationExamples": 6,
            "splitSeed": 7,
        },
        "preparation_error": None,
        "training": {"jobId": "123456", "mode": "full", "trainExamples": 48},
        "launch_error": None,
        "current_run_id": "run-1",
    }

    def test_nested_blocks_round_trip_whole(self) -> None:
        record = map_model_request(self.ROW)
        self.assertEqual(record["preparation"]["splitSeed"], 7)
        self.assertEqual(record["training"]["jobId"], "123456")
        self.assertEqual(record["currentRunId"], "run-1")

    def test_absent_blocks_are_omitted(self) -> None:
        record = map_model_request(
            {**self.ROW, "preparation": None, "training": None, "current_run_id": None}
        )
        for key in ("preparation", "training", "currentRunId"):
            self.assertNotIn(key, record)


class TrainingRunMappingTests(unittest.TestCase):
    ROW = {
        "run_id": "run-1",
        "course_id": "css-360-winter-2026-a7rp",
        "mode": "full",
        "state": "claimed",
        "enqueued_at": UTC_NOON,
        "updated_at": UTC_NOON,
        "dataset_ref": "exports/css-360-winter-2026-a7rp",
        "approved_example_count": 54,
        "train_examples": 48,
        "validation_examples": 6,
        "attempt": 1,
        "job_id": "123456",
        "claim_owner": "tillicum-runner",
        "claim_claimed_at": UTC_NOON,
        "claim_expires_at": UTC_NOON + timedelta(hours=1),
        "error": None,
    }

    def test_claim_columns_nest_into_a_claim_object(self) -> None:
        run = map_training_run(self.ROW)
        self.assertEqual(
            run["claim"],
            {
                "owner": "tillicum-runner",
                "claimedAt": "2026-08-13T12:00:00+00:00",
                "expiresAt": "2026-08-13T13:00:00+00:00",
            },
        )

    def test_a_partial_claim_is_reported_as_no_claim(self) -> None:
        """`parseClaim` drops a lease with no owner or no expiry; so does this."""
        for missing in ("claim_owner", "claim_claimed_at", "claim_expires_at"):
            run = map_training_run({**self.ROW, missing: None})
            self.assertNotIn("claim", run, missing)

    def test_unclaimed_run_has_no_claim_key(self) -> None:
        run = map_training_run(
            {
                **self.ROW,
                "state": "queued",
                "claim_owner": None,
                "claim_claimed_at": None,
                "claim_expires_at": None,
                "job_id": None,
            }
        )
        self.assertNotIn("claim", run)
        self.assertNotIn("jobId", run)
        self.assertEqual(run["state"], "queued")

    def test_counts_and_attempt_map_through(self) -> None:
        run = map_training_run(self.ROW)
        self.assertEqual(run["approvedExampleCount"], 54)
        self.assertEqual(run["trainExamples"], 48)
        self.assertEqual(run["validationExamples"], 6)
        self.assertEqual(run["attempt"], 1)
        self.assertEqual(run["datasetRef"], "exports/css-360-winter-2026-a7rp")

    def test_generated_run_id_matches_the_browser_format(self) -> None:
        run_id = generate_training_run_id(UTC_NOON)
        self.assertTrue(run_id.startswith("run-20260813t120000z-"), run_id)


if __name__ == "__main__":
    unittest.main()
