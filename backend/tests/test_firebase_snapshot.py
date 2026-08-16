"""Tests for Firebase snapshot -> PostgreSQL row mapping.

Pure parsing tests: no PostgreSQL server, no psycopg import, no network. The
importer's SQL generation is covered here too, as text.
"""

from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.firebase_snapshot import (
    SnapshotError,
    map_course,
    map_evaluation,
    map_model_registry,
    map_model_request,
    map_seed_example,
    map_starter_seed_generation,
    map_training_run,
    parse_snapshot,
    parse_timestamp,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from import_firebase_snapshot import (  # noqa: E402
    SUMMARY_LABELS,
    TABLE_SPECS,
    _bind_row,
    build_parser,
    load_snapshot,
    print_summary,
)

VALID_METADATA = {
    "chunkCount": 180,
    "createdAt": "2026-08-13T02:15:50.410Z",
    "instructorName": "Kaylea Champion",
    "name": "CSS 350",
    "syllabusFileName": "Css 350.txt",
    "syllabusStatus": "indexed",
    "syllabusType": "txt",
    "term": "Spring 2026",
    "title": "Management principals",
}


class TimestampParsingTests(unittest.TestCase):
    def test_parses_trailing_z_as_utc(self) -> None:
        parsed = parse_timestamp(
            "2026-08-13T02:15:50.410Z", field_name="createdAt", context="ctx"
        )
        self.assertEqual(
            parsed, datetime(2026, 8, 13, 2, 15, 50, 410000, tzinfo=timezone.utc)
        )

    def test_parses_explicit_offset_with_microseconds(self) -> None:
        parsed = parse_timestamp(
            "2026-08-13T03:26:52.013570+00:00", field_name="completedAt", context="ctx"
        )
        self.assertEqual(
            parsed, datetime(2026, 8, 13, 3, 26, 52, 13570, tzinfo=timezone.utc)
        )

    def test_treats_naive_timestamp_as_utc(self) -> None:
        parsed = parse_timestamp(
            "2026-08-13T02:15:50", field_name="createdAt", context="ctx"
        )
        self.assertEqual(parsed, datetime(2026, 8, 13, 2, 15, 50, tzinfo=timezone.utc))

    def test_returns_none_for_missing_and_blank(self) -> None:
        for value in (None, "", "   "):
            self.assertIsNone(
                parse_timestamp(value, field_name="startedAt", context="ctx")
            )

    def test_raises_on_unparsable_timestamp(self) -> None:
        with self.assertRaises(SnapshotError) as caught:
            parse_timestamp("last Tuesday", field_name="createdAt", context="course 'x'")
        self.assertIn("createdAt", str(caught.exception))
        self.assertIn("course 'x'", str(caught.exception))


class CourseMappingTests(unittest.TestCase):
    def test_maps_camel_case_metadata_to_snake_case_columns(self) -> None:
        row = map_course("css-350-spring-2026-ab12", VALID_METADATA)

        self.assertEqual(row["course_id"], "css-350-spring-2026-ab12")
        self.assertEqual(row["instructor_name"], "Kaylea Champion")
        self.assertEqual(row["syllabus_file_name"], "Css 350.txt")
        self.assertEqual(row["syllabus_type"], "txt")
        self.assertEqual(row["syllabus_status"], "indexed")
        self.assertEqual(row["chunk_count"], 180)
        self.assertEqual(
            row["created_at"],
            datetime(2026, 8, 13, 2, 15, 50, 410000, tzinfo=timezone.utc),
        )

    def test_defaults_chunk_count_and_nullable_file_fields(self) -> None:
        metadata = {
            key: value
            for key, value in VALID_METADATA.items()
            if key not in {"chunkCount", "syllabusFileName", "syllabusType"}
        }
        row = map_course("css-350-spring-2026-ab12", metadata)

        self.assertEqual(row["chunk_count"], 0)
        self.assertIsNone(row["syllabus_file_name"])
        self.assertIsNone(row["syllabus_type"])

    def test_raises_naming_the_missing_required_field(self) -> None:
        metadata = {key: value for key, value in VALID_METADATA.items() if key != "term"}
        with self.assertRaises(SnapshotError) as caught:
            map_course("css-350-spring-2026-ab12", metadata)
        self.assertIn("term", str(caught.exception))
        self.assertIn("css-350-spring-2026-ab12", str(caught.exception))


class StarterSeedGenerationMappingTests(unittest.TestCase):
    def test_maps_full_block(self) -> None:
        row = map_starter_seed_generation(
            "css-350-spring-2026-ab12",
            {
                "completedAt": "2026-08-13T03:26:52.013570+00:00",
                "failedToSaveCount": 0,
                "finalCount": 9,
                "savedCount": 9,
                "startedAt": "2026-08-13T02:16:18.373692+00:00",
                "status": "partial",
                "targetCount": 50,
            },
        )

        self.assertEqual(row["status"], "partial")
        self.assertEqual(row["target_count"], 50)
        self.assertEqual(row["final_count"], 9)
        self.assertEqual(row["saved_count"], 9)
        self.assertEqual(row["failed_to_save_count"], 0)
        self.assertIsNone(row["error"])

    def test_returns_none_when_absent(self) -> None:
        self.assertIsNone(map_starter_seed_generation("css-350-spring-2026-ab12", None))

    def test_tolerates_partially_written_block(self) -> None:
        row = map_starter_seed_generation(
            "css-350-spring-2026-ab12",
            {"status": "generating", "startedAt": "2026-08-13T02:16:18.373692+00:00"},
        )

        self.assertEqual(row["status"], "generating")
        self.assertIsNone(row["target_count"])
        self.assertIsNone(row["completed_at"])


class SeedExampleMappingTests(unittest.TestCase):
    def test_prefers_instruction_and_response_over_duplicates(self) -> None:
        row = map_seed_example(
            "css-350-spring-2026-ab12",
            "-Oseed001",
            {
                "instruction": "When are office hours?",
                "question": "stale question wording",
                "response": "Tuesdays at 2pm.",
                "answer": "stale answer wording",
                "category": "office hours",
                "origin": "ai_generated",
                "sourceSection": "Office Hours",
            },
        )

        self.assertEqual(row["instruction"], "When are office hours?")
        self.assertEqual(row["response"], "Tuesdays at 2pm.")

    def test_falls_back_to_question_and_answer(self) -> None:
        row = map_seed_example(
            "css-350-spring-2026-ab12",
            "-Oseed002",
            {
                "question": "When are office hours?",
                "answer": "Tuesdays at 2pm.",
                "category": "office hours",
                "origin": "ai_generated",
                "sourceSection": "Office Hours",
            },
        )

        self.assertEqual(row["instruction"], "When are office hours?")
        self.assertEqual(row["response"], "Tuesdays at 2pm.")

    def test_preserves_firebase_push_id_over_record_id(self) -> None:
        row = map_seed_example(
            "css-350-spring-2026-ab12",
            "-Okey123",
            {
                "id": "-Ostale999",
                "question": "Q?",
                "answer": "A.",
                "category": "general",
                "origin": "ai_generated",
            },
        )

        self.assertEqual(row["seed_id"], "-Okey123")

    def test_keeps_nested_values_as_python_objects_for_jsonb(self) -> None:
        validation = {"score": 0.95, "reason": "Supported.", "unsupportedClaims": []}
        row = map_seed_example(
            "css-350-spring-2026-ab12",
            "-Oseed003",
            {
                "question": "Q?",
                "answer": "A.",
                "category": "general",
                "origin": "ai_generated",
                "sourceChunkIds": ["chunk-001", "chunk-002"],
                "validation": validation,
            },
        )

        self.assertEqual(row["source_chunk_ids"], ["chunk-001", "chunk-002"])
        self.assertEqual(row["validation"], validation)

    def test_normalizes_firebase_index_keyed_chunk_ids(self) -> None:
        row = map_seed_example(
            "css-350-spring-2026-ab12",
            "-Oseed004",
            {
                "question": "Q?",
                "answer": "A.",
                "category": "general",
                "origin": "ai_generated",
                "sourceChunkIds": {"0": "chunk-001", "2": "chunk-003"},
            },
        )

        self.assertEqual(row["source_chunk_ids"], ["chunk-001", "chunk-003"])

    def test_applies_ai_generated_defaults(self) -> None:
        row = map_seed_example(
            "css-350-spring-2026-ab12",
            "-Oseed005",
            {
                "question": "Q?",
                "answer": "A.",
                "category": "general",
                "origin": "ai_generated",
            },
        )

        self.assertEqual(row["difficulty"], "Medium")
        self.assertTrue(row["directly_answered"])
        self.assertEqual(row["source_section"], "General")
        self.assertFalse(row["was_edited"])

    def test_derives_source_section_from_chunk_ids(self) -> None:
        row = map_seed_example(
            "css-350-spring-2026-ab12",
            "-Oseed006",
            {
                "question": "Q?",
                "answer": "A.",
                "category": "general",
                "origin": "prototype",
                "difficulty": "Easy",
                "directlyAnswered": False,
                "sourceChunkIds": ["chunk-001", "chunk-002"],
            },
        )

        self.assertEqual(row["source_section"], "chunk-001, chunk-002")

    def test_maps_review_fields(self) -> None:
        row = map_seed_example(
            "css-350-spring-2026-ab12",
            "-Oseed007",
            {
                "question": "Q?",
                "answer": "A.",
                "category": "general",
                "origin": "ai_generated",
                "reviewStatus": "edited",
                "reviewNotes": "Tightened wording.",
                "reviewedAt": "2026-08-14T10:00:00Z",
                "normalizedQuestionKey": "when are office hours",
                "originalQuestion": "Old Q?",
                "originalAnswer": "Old A.",
                "wasEdited": True,
            },
        )

        self.assertEqual(row["review_status"], "edited")
        self.assertEqual(row["review_notes"], "Tightened wording.")
        self.assertEqual(row["normalized_question_key"], "when are office hours")
        self.assertEqual(
            row["reviewed_at"], datetime(2026, 8, 14, 10, 0, tzinfo=timezone.utc)
        )
        self.assertEqual(row["original_question"], "Old Q?")
        self.assertTrue(row["was_edited"])

    def test_raises_when_no_instruction_or_question(self) -> None:
        with self.assertRaises(SnapshotError) as caught:
            map_seed_example(
                "css-350-spring-2026-ab12",
                "-Oseed008",
                {"answer": "A.", "category": "general", "origin": "ai_generated"},
            )
        self.assertIn("instruction", str(caught.exception))
        self.assertIn("-Oseed008", str(caught.exception))

    def test_raises_on_unknown_origin(self) -> None:
        with self.assertRaises(SnapshotError) as caught:
            map_seed_example(
                "css-350-spring-2026-ab12",
                "-Oseed009",
                {"question": "Q?", "answer": "A.", "category": "general", "origin": "bot"},
            )
        self.assertIn("origin", str(caught.exception))

    def test_raises_when_non_ai_seed_omits_difficulty(self) -> None:
        with self.assertRaises(SnapshotError) as caught:
            map_seed_example(
                "css-350-spring-2026-ab12",
                "-Oseed010",
                {
                    "question": "Q?",
                    "answer": "A.",
                    "category": "general",
                    "origin": "prototype",
                    "sourceSection": "Grading",
                    "directlyAnswered": True,
                },
            )
        self.assertIn("difficulty", str(caught.exception))


class EvaluationMappingTests(unittest.TestCase):
    def test_maps_evaluation_record(self) -> None:
        row = map_evaluation(
            "css-350-spring-2026-ab12",
            "-Oeval001",
            {
                "comparisonId": "cmp-3",
                "mostAccurate": "rag",
                "mostHelpful": "fineTunedRag",
                "mostConcise": "base",
                "bestGrounded": "rag",
                "preferredModel": "fineTunedRag",
                "hallucinationFlags": ["base", "fineTuned"],
                "comment": "RAG cited the syllabus.",
                "createdAt": "2026-08-14T10:00:00Z",
                "runId": "run-9",
                "questionText": "When is the midterm?",
            },
        )

        self.assertEqual(row["evaluation_id"], "-Oeval001")
        self.assertEqual(row["comparison_id"], "cmp-3")
        self.assertEqual(row["most_accurate"], "rag")
        self.assertEqual(row["preferred_model"], "fineTunedRag")
        self.assertEqual(row["hallucination_flags"], ["base", "fineTuned"])
        self.assertEqual(row["question_text"], "When is the midterm?")

    def test_defaults_hallucination_flags_to_empty_list(self) -> None:
        row = map_evaluation(
            "css-350-spring-2026-ab12",
            "-Oeval002",
            {
                "comparisonId": "cmp-1",
                "mostAccurate": "rag",
                "mostHelpful": "rag",
                "mostConcise": "rag",
                "bestGrounded": "rag",
                "preferredModel": "rag",
                "createdAt": "2026-08-14T10:00:00Z",
            },
        )

        self.assertEqual(row["hallucination_flags"], [])
        self.assertIsNone(row["comment"])
        self.assertIsNone(row["run_id"])

    def test_raises_when_required_choice_missing(self) -> None:
        with self.assertRaises(SnapshotError) as caught:
            map_evaluation(
                "css-350-spring-2026-ab12",
                "-Oeval003",
                {
                    "comparisonId": "cmp-1",
                    "mostAccurate": "rag",
                    "mostHelpful": "rag",
                    "mostConcise": "rag",
                    "preferredModel": "rag",
                    "createdAt": "2026-08-14T10:00:00Z",
                },
            )
        self.assertIn("bestGrounded", str(caught.exception))


class ModelRegistryMappingTests(unittest.TestCase):
    REGISTRY = {
        "currentVersion": "v1",
        "versions": {
            "v1": {
                "artifactRef": "css-360-qlora/adapter",
                "baseModel": "meta-llama/Llama-3.2-3B-Instruct",
                "createdAt": "2026-08-11T06:22:50.979479Z",
                "deployment": "offline",
                "notes": "QLoRA adapter trained from 54 approved examples.",
                "status": "ready",
                "trainingExampleCount": 54,
                "updatedAt": "2026-08-11T06:22:50.979479Z",
                "version": "v1",
            }
        },
    }

    def test_maps_registry_and_versions(self) -> None:
        model_row, version_rows = map_model_registry(
            "css-360-summer-2026-89m4", self.REGISTRY
        )

        self.assertEqual(model_row["current_version"], "v1")
        self.assertEqual(len(version_rows), 1)
        version = version_rows[0]
        self.assertEqual(version["version"], "v1")
        self.assertEqual(version["base_model"], "meta-llama/Llama-3.2-3B-Instruct")
        self.assertEqual(version["training_example_count"], 54)
        self.assertEqual(version["artifact_ref"], "css-360-qlora/adapter")
        self.assertEqual(version["deployment"], "offline")

    def test_returns_nothing_for_absent_registry(self) -> None:
        self.assertEqual(map_model_registry("css-360-summer-2026-89m4", None), (None, []))

    def test_raises_when_current_version_has_no_entry(self) -> None:
        with self.assertRaises(SnapshotError) as caught:
            map_model_registry(
                "css-360-summer-2026-89m4",
                {"currentVersion": "v2", "versions": self.REGISTRY["versions"]},
            )
        self.assertIn("currentVersion", str(caught.exception))

    def test_raises_when_version_missing_required_field(self) -> None:
        with self.assertRaises(SnapshotError) as caught:
            map_model_registry(
                "css-360-summer-2026-89m4",
                {
                    "currentVersion": "v1",
                    "versions": {
                        "v1": {
                            "baseModel": "llama",
                            "status": "ready",
                            "deployment": "offline",
                            "trainingExampleCount": 54,
                            "createdAt": "2026-08-11T06:22:50.979479Z",
                        }
                    },
                },
            )
        self.assertIn("artifactRef", str(caught.exception))


class ModelRequestMappingTests(unittest.TestCase):
    def test_keeps_preparation_and_training_as_jsonb_objects(self) -> None:
        preparation = {
            "preparedAt": "2026-08-12T01:00:00Z",
            "sourceApprovedExampleCount": 54,
            "datasetRef": "exports/css-360-summer-2026-89m4",
            "trainExamples": 48,
            "validationExamples": 6,
            "splitSeed": 7,
        }
        training = {
            "jobId": "123456",
            "mode": "full",
            "submittedAt": "2026-08-12T02:00:00Z",
            "datasetRef": "exports/css-360-summer-2026-89m4",
            "trainExamples": 48,
            "validationExamples": 6,
        }
        row = map_model_request(
            "css-360-summer-2026-89m4",
            {
                "courseId": "css-360-summer-2026-89m4",
                "status": "training",
                "requestedAt": "2026-08-12T00:00:00Z",
                "updatedAt": "2026-08-12T02:00:00Z",
                "approvedExampleCount": 54,
                "preparation": preparation,
                "training": training,
                "currentRunId": "run-1",
            },
        )

        self.assertEqual(row["status"], "training")
        self.assertEqual(row["approved_example_count"], 54)
        self.assertEqual(row["preparation"], preparation)
        self.assertEqual(row["training"], training)
        self.assertEqual(row["current_run_id"], "run-1")
        self.assertIsNone(row["failure_message"])

    def test_returns_none_for_absent_request(self) -> None:
        self.assertIsNone(map_model_request("css-360-summer-2026-89m4", None))

    def test_tolerates_request_with_no_optional_blocks(self) -> None:
        row = map_model_request(
            "css-360-summer-2026-89m4",
            {
                "status": "requested",
                "requestedAt": "2026-08-12T00:00:00Z",
                "updatedAt": "2026-08-12T00:00:00Z",
            },
        )

        self.assertEqual(row["approved_example_count"], 0)
        self.assertIsNone(row["preparation"])
        self.assertIsNone(row["training"])

    def test_raises_when_status_missing(self) -> None:
        with self.assertRaises(SnapshotError) as caught:
            map_model_request(
                "css-360-summer-2026-89m4",
                {"requestedAt": "2026-08-12T00:00:00Z", "updatedAt": "2026-08-12T00:00:00Z"},
            )
        self.assertIn("status", str(caught.exception))


class TrainingRunMappingTests(unittest.TestCase):
    def test_flattens_claim_into_columns(self) -> None:
        row = map_training_run(
            "css-360-summer-2026-89m4",
            "run-1",
            {
                "runId": "run-1",
                "courseId": "css-360-summer-2026-89m4",
                "mode": "full",
                "state": "claimed",
                "enqueuedAt": "2026-08-12T01:00:00Z",
                "updatedAt": "2026-08-12T01:05:00Z",
                "datasetRef": "exports/css-360-summer-2026-89m4",
                "approvedExampleCount": 54,
                "trainExamples": 48,
                "validationExamples": 6,
                "attempt": 1,
                "jobId": "123456",
                "claim": {
                    "owner": "tillicum-runner",
                    "claimedAt": "2026-08-12T01:05:00Z",
                    "expiresAt": "2026-08-12T02:05:00Z",
                },
            },
        )

        self.assertEqual(row["run_id"], "run-1")
        self.assertEqual(row["state"], "claimed")
        self.assertEqual(row["dataset_ref"], "exports/css-360-summer-2026-89m4")
        self.assertEqual(row["claim_owner"], "tillicum-runner")
        self.assertEqual(
            row["claim_claimed_at"], datetime(2026, 8, 12, 1, 5, tzinfo=timezone.utc)
        )
        self.assertEqual(
            row["claim_expires_at"], datetime(2026, 8, 12, 2, 5, tzinfo=timezone.utc)
        )

    def test_defaults_counts_and_tolerates_missing_claim(self) -> None:
        row = map_training_run(
            "css-360-summer-2026-89m4",
            "run-2",
            {
                "mode": "smoke",
                "state": "queued",
                "enqueuedAt": "2026-08-12T01:00:00Z",
                "updatedAt": "2026-08-12T01:00:00Z",
                "datasetRef": "exports/css-360-summer-2026-89m4",
            },
        )

        self.assertEqual(row["approved_example_count"], 0)
        self.assertEqual(row["train_examples"], 0)
        self.assertEqual(row["validation_examples"], 0)
        self.assertEqual(row["attempt"], 0)
        self.assertIsNone(row["job_id"])
        self.assertIsNone(row["claim_owner"])

    def test_raises_when_dataset_ref_missing(self) -> None:
        with self.assertRaises(SnapshotError) as caught:
            map_training_run(
                "css-360-summer-2026-89m4",
                "run-3",
                {
                    "mode": "full",
                    "state": "queued",
                    "enqueuedAt": "2026-08-12T01:00:00Z",
                    "updatedAt": "2026-08-12T01:00:00Z",
                },
            )
        self.assertIn("datasetRef", str(caught.exception))


def _full_snapshot() -> dict:
    return {
        "css-350-spring-2026-ab12": {
            "metadata": {
                **VALID_METADATA,
                "starterSeedGeneration": {
                    "status": "partial",
                    "targetCount": 50,
                    "finalCount": 9,
                    "savedCount": 9,
                    "failedToSaveCount": 0,
                    "startedAt": "2026-08-13T02:16:18.373692+00:00",
                    "completedAt": "2026-08-13T03:26:52.013570+00:00",
                },
            },
            "seedExamples": {
                "-Oseed001": {
                    "question": "When are office hours?",
                    "answer": "Tuesdays at 2pm.",
                    "category": "office hours",
                    "origin": "ai_generated",
                },
                "-Oseed002": {
                    "question": "How is grading weighted?",
                    "answer": "40% projects.",
                    "category": "grading",
                    "origin": "ai_generated",
                },
            },
            "evaluations": {
                "-Oeval001": {
                    "comparisonId": "cmp-1",
                    "mostAccurate": "rag",
                    "mostHelpful": "rag",
                    "mostConcise": "base",
                    "bestGrounded": "rag",
                    "preferredModel": "rag",
                    "createdAt": "2026-08-14T10:00:00Z",
                }
            },
            "model": ModelRegistryMappingTests.REGISTRY,
            "modelRequest": {
                "status": "ready",
                "requestedAt": "2026-08-12T00:00:00Z",
                "updatedAt": "2026-08-12T02:00:00Z",
                "approvedExampleCount": 54,
            },
            "trainingRuns": {
                "run-1": {
                    "mode": "full",
                    "state": "succeeded",
                    "enqueuedAt": "2026-08-12T01:00:00Z",
                    "updatedAt": "2026-08-12T03:00:00Z",
                    "datasetRef": "exports/css-350-spring-2026-ab12",
                }
            },
        },
        # A course with metadata only: everything else is legitimately absent.
        "css-430-summer-2026-ibce": {"metadata": VALID_METADATA},
    }


class SnapshotParsingTests(unittest.TestCase):
    def test_counts_every_node_type(self) -> None:
        plan = parse_snapshot(_full_snapshot())

        self.assertEqual(
            plan.counts(),
            {
                "courses": 2,
                "starter_seed_generation": 1,
                "seed_examples": 2,
                "evaluations": 1,
                "course_models": 1,
                "course_model_versions": 1,
                "model_requests": 1,
                "training_runs": 1,
            },
        )

    def test_accepts_snapshot_still_wrapped_in_a_courses_root(self) -> None:
        plan = parse_snapshot({"courses": _full_snapshot()})
        self.assertEqual(plan.counts()["courses"], 2)

    def test_courses_come_first_so_foreign_keys_resolve(self) -> None:
        plan = parse_snapshot(_full_snapshot())
        course_ids = {row["course_id"] for row in plan.courses}
        for row in plan.seed_examples + plan.training_runs + plan.evaluations:
            self.assertIn(row["course_id"], course_ids)

    def test_raises_when_metadata_node_missing(self) -> None:
        with self.assertRaises(SnapshotError) as caught:
            parse_snapshot({"css-350-spring-2026-ab12": {"seedExamples": {}}})
        self.assertIn("metadata", str(caught.exception))

    def test_raises_on_invalid_course_id(self) -> None:
        with self.assertRaises(SnapshotError) as caught:
            parse_snapshot({"CSS 350": {"metadata": VALID_METADATA}})
        self.assertIn("courseId", str(caught.exception))

    def test_raises_when_root_is_not_an_object(self) -> None:
        with self.assertRaises(SnapshotError):
            parse_snapshot([{"metadata": VALID_METADATA}])


class ImporterTests(unittest.TestCase):
    """The script's argument parsing, SQL text, and output — still no server."""

    def test_dry_run_flag_defaults_off(self) -> None:
        args = build_parser().parse_args(["snapshot.json"])
        self.assertFalse(args.dry_run)
        self.assertEqual(args.snapshot, Path("snapshot.json"))

        self.assertTrue(build_parser().parse_args(["snapshot.json", "--dry-run"]).dry_run)

    def test_every_table_spec_upserts_and_never_deletes(self) -> None:
        for _, spec in TABLE_SPECS:
            sql = spec.upsert_sql()
            self.assertIn(f"INSERT INTO {spec.table}", sql)
            self.assertIn("ON CONFLICT", sql)
            self.assertNotIn("DELETE", sql.upper())
            self.assertNotIn("TRUNCATE", sql.upper())
            for column in spec.columns:
                self.assertIn(f"%({column})s", sql)

    def test_conflict_target_matches_the_primary_key(self) -> None:
        expected = {
            "courses": ["course_id"],
            "starter_seed_generation": ["course_id"],
            "seed_examples": ["course_id", "seed_id"],
            "evaluations": ["course_id", "evaluation_id"],
            "course_models": ["course_id"],
            "course_model_versions": ["course_id", "version"],
            "model_requests": ["course_id"],
            "training_runs": ["course_id", "run_id"],
        }
        self.assertEqual(
            {spec.table: spec.conflict_columns for _, spec in TABLE_SPECS}, expected
        )

    def test_courses_are_written_before_dependent_tables(self) -> None:
        order = [spec.table for _, spec in TABLE_SPECS]
        self.assertEqual(order[0], "courses")

    def test_plan_attributes_exist_on_the_snapshot_plan(self) -> None:
        plan = parse_snapshot(_full_snapshot())
        for attribute, _ in TABLE_SPECS:
            self.assertIsInstance(getattr(plan, attribute), list)

    def test_bind_row_wraps_only_non_null_json_columns(self) -> None:
        spec = dict(TABLE_SPECS)["seed_examples"]
        row = map_seed_example(
            "css-350-spring-2026-ab12",
            "-Oseed001",
            {
                "question": "Q?",
                "answer": "A.",
                "category": "general",
                "origin": "ai_generated",
                "sourceChunkIds": ["chunk-001"],
            },
        )

        # A stand-in for psycopg's Json adapter, so this needs no driver.
        bound = _bind_row(row, spec, lambda value: ("json", value))

        self.assertEqual(bound["source_chunk_ids"], ("json", ["chunk-001"]))
        self.assertIsNone(bound["validation"])
        self.assertEqual(bound["instruction"], "Q?")
        self.assertEqual(set(bound), set(spec.columns))

    def test_summary_labels_cover_every_count(self) -> None:
        plan = parse_snapshot(_full_snapshot())
        counts = plan.counts()
        self.assertEqual({key for _, key in SUMMARY_LABELS}, set(counts))

    def test_summary_prints_the_expected_lines(self) -> None:
        from contextlib import redirect_stdout
        from io import StringIO

        buffer = StringIO()
        with redirect_stdout(buffer):
            print_summary(parse_snapshot(_full_snapshot()))

        self.assertEqual(
            buffer.getvalue().splitlines(),
            [
                "Courses: 2",
                "Starter-generation records: 1",
                "Seeds: 2",
                "Evaluations: 1",
                "Models: 1",
                "Model versions: 1",
                "Model requests: 1",
                "Training runs: 1",
            ],
        )

    def test_load_snapshot_reports_invalid_json_clearly(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "courses.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(SnapshotError) as caught:
                load_snapshot(path)
            self.assertIn("not valid JSON", str(caught.exception))

    def test_load_snapshot_reports_missing_file_clearly(self) -> None:
        with self.assertRaises(SnapshotError) as caught:
            load_snapshot(Path("/nonexistent/courses.json"))
        self.assertIn("Could not read snapshot", str(caught.exception))

    def test_dry_run_writes_nothing_and_prints_counts(self) -> None:
        import tempfile
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch

        import import_firebase_snapshot

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "courses.json"
            path.write_text(json.dumps(_full_snapshot()), encoding="utf-8")

            buffer = StringIO()
            with patch.object(import_firebase_snapshot, "write_plan") as write_plan:
                with redirect_stdout(buffer):
                    exit_code = import_firebase_snapshot.main([str(path), "--dry-run"])

            write_plan.assert_not_called()

        self.assertEqual(exit_code, 0)
        self.assertIn("No database writes.", buffer.getvalue())
        self.assertIn("Courses: 2", buffer.getvalue())

    def test_malformed_snapshot_exits_nonzero_without_writing(self) -> None:
        import tempfile
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch

        import import_firebase_snapshot

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "courses.json"
            path.write_text(
                json.dumps({"css-350-spring-2026-ab12": {"seedExamples": {}}}),
                encoding="utf-8",
            )

            with patch.object(import_firebase_snapshot, "write_plan") as write_plan:
                with redirect_stdout(StringIO()):
                    exit_code = import_firebase_snapshot.main([str(path)])

            write_plan.assert_not_called()

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
