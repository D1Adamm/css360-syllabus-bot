"""API-layer tests for the PostgreSQL-backed `/api/db` routes.

The repositories are patched out, so these need no PostgreSQL server and no
Firebase network access. What they cover is the layer above the SQL: response
shapes the frontend will consume, status codes for missing and conflicting
records, course-id validation, and that a driver failure becomes a 503 whose
body carries no connection string.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import db_courses, db_model_requests, db_training_runs
from app.main import app

COURSE = "css-350-spring-2026-n3h9"
OTHER_COURSE = "css-360-winter-2026-a7rp"

METADATA = {
    "name": "CSS 350",
    "title": "Management principals",
    "term": "Spring 2026",
    "instructorName": "Kaylea Champion",
    "createdAt": "2026-08-13T02:15:50.410000+00:00",
    "syllabusStatus": "indexed",
    "syllabusFileName": "Css 350.txt",
    "syllabusType": "txt",
    "chunkCount": 180,
}

SEED = {
    "id": "-Oseed001",
    "courseId": COURSE,
    "instruction": "Can I get an extension?",
    "response": "No extension is possible.",
    "question": "Can I get an extension?",
    "answer": "No extension is possible.",
    "category": "late work",
    "sourceSection": "Late Policy",
    "difficulty": "Medium",
    "directlyAnswered": True,
    "origin": "ai_generated",
    "sourceChunkIds": ["chunk-092"],
    "wasEdited": False,
    "createdAt": "2026-08-13T03:26:50.464264+00:00",
    "reviewStatus": "generated",
    "validation": {"score": 0.92, "reason": "Grounded."},
}

EVALUATION = {
    "id": "-Oeval001",
    "courseId": COURSE,
    "comparisonId": "cmp-1",
    "mostAccurate": "rag",
    "mostHelpful": "rag",
    "mostConcise": "base",
    "bestGrounded": "rag",
    "preferredModel": "rag",
    "hallucinationFlags": ["base"],
    "createdAt": "2026-08-14T10:00:00+00:00",
}

TRAINING_RUN = {
    "runId": "run-1",
    "courseId": COURSE,
    "mode": "full",
    "state": "claimed",
    "enqueuedAt": "2026-08-12T01:00:00+00:00",
    "updatedAt": "2026-08-12T01:05:00+00:00",
    "datasetRef": "exports/css-350-spring-2026-n3h9",
    "approvedExampleCount": 54,
    "trainExamples": 48,
    "validationExamples": 6,
    "attempt": 1,
    "jobId": "123456",
    "claim": {
        "owner": "tillicum-runner",
        "claimedAt": "2026-08-12T01:05:00+00:00",
        "expiresAt": "2026-08-12T02:05:00+00:00",
    },
}


@contextmanager
def _fake_connection() -> Any:
    """Stand in for `db_connection` so no driver or server is touched."""
    yield object()


class DbRouteTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self._connection_patch = patch(
            "app.db_routes.db_connection", new=_fake_connection
        )
        self._connection_patch.start()
        self.addCleanup(self._connection_patch.stop)

    def patch_repo(self, name: str, **kwargs: Any) -> Any:
        patcher = patch(f"app.db_routes.{name}", **kwargs)
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock


class CourseRouteTests(DbRouteTestCase):
    def test_list_courses_returns_courses_with_nested_metadata(self) -> None:
        self.patch_repo(
            "db_courses.list_courses",
            return_value=[{"courseId": COURSE, "metadata": METADATA}],
        )
        response = self.client.get("/api/db/courses")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["courses"][0]["courseId"], COURSE)
        self.assertEqual(
            body["courses"][0]["metadata"]["instructorName"], "Kaylea Champion"
        )

    def test_metadata_carries_nested_starter_seed_generation(self) -> None:
        metadata = {
            **METADATA,
            "starterSeedGeneration": {
                "status": "ready",
                "targetCount": 50,
                "finalCount": 50,
                "savedCount": 50,
                "failedToSaveCount": 0,
            },
        }
        self.patch_repo(
            "db_courses.get_course",
            return_value={"courseId": COURSE, "metadata": metadata},
        )
        response = self.client.get(f"/api/db/courses/{COURSE}")

        self.assertEqual(response.status_code, 200)
        nested = response.json()["metadata"]["starterSeedGeneration"]
        self.assertEqual(nested["status"], "ready")
        self.assertEqual(nested["savedCount"], 50)

    def test_missing_course_is_404(self) -> None:
        self.patch_repo("db_courses.get_course", return_value=None)
        response = self.client.get(f"/api/db/courses/{COURSE}")

        self.assertEqual(response.status_code, 404)
        self.assertIn(COURSE, response.json()["detail"])

    def test_invalid_course_id_is_400_and_never_queries(self) -> None:
        get_course = self.patch_repo("db_courses.get_course")
        response = self.client.get("/api/db/courses/CSS%20350")

        self.assertEqual(response.status_code, 400)
        get_course.assert_not_called()

    def test_create_course_returns_201(self) -> None:
        self.patch_repo(
            "db_courses.create_course",
            return_value={"courseId": COURSE, "metadata": METADATA},
        )
        response = self.client.post(
            "/api/db/courses",
            json={
                "courseId": COURSE,
                "name": "CSS 350",
                "title": "Management principals",
                "term": "Spring 2026",
                "instructorName": "Kaylea Champion",
            },
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["courseId"], COURSE)

    def test_duplicate_course_is_409(self) -> None:
        self.patch_repo(
            "db_courses.create_course",
            side_effect=db_courses.CourseAlreadyExistsError("already exists"),
        )
        response = self.client.post(
            "/api/db/courses",
            json={
                "courseId": COURSE,
                "name": "CSS 350",
                "title": "T",
                "term": "Spring 2026",
                "instructorName": "K",
            },
        )
        self.assertEqual(response.status_code, 409)

    def test_create_course_rejects_a_missing_required_field(self) -> None:
        response = self.client.post("/api/db/courses", json={"courseId": COURSE})
        self.assertEqual(response.status_code, 422)

    def test_patch_sends_only_the_fields_supplied(self) -> None:
        """A merge: unmentioned fields must not be overwritten with defaults."""
        update = self.patch_repo(
            "db_courses.update_course",
            return_value={"courseId": COURSE, "metadata": METADATA},
        )
        response = self.client.patch(
            f"/api/db/courses/{COURSE}", json={"chunkCount": 42}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(update.call_args.args[2], {"chunkCount": 42})

    def test_patch_of_a_missing_course_is_404(self) -> None:
        self.patch_repo("db_courses.update_course", return_value=None)
        response = self.client.patch(
            f"/api/db/courses/{COURSE}", json={"chunkCount": 1}
        )
        self.assertEqual(response.status_code, 404)


class StarterGenerationRouteTests(DbRouteTestCase):
    def test_reads_starter_state(self) -> None:
        self.patch_repo(
            "db_courses.get_starter_seed_generation",
            return_value={"status": "partial", "targetCount": 50},
        )
        response = self.client.get(
            f"/api/db/courses/{COURSE}/starter-seed-generation"
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["courseId"], COURSE)
        self.assertEqual(body["starterSeedGeneration"]["status"], "partial")

    def test_absent_state_reads_as_null_not_404(self) -> None:
        """A course that never ran generation has no record; that is not an error."""
        self.patch_repo(
            "db_courses.get_starter_seed_generation", return_value=None
        )
        response = self.client.get(
            f"/api/db/courses/{COURSE}/starter-seed-generation"
        )

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["starterSeedGeneration"])

    def test_patch_merges_only_supplied_fields(self) -> None:
        self.patch_repo("db_courses.course_exists", return_value=True)
        upsert = self.patch_repo(
            "db_courses.upsert_starter_seed_generation",
            return_value={"status": "generating"},
        )
        response = self.client.patch(
            f"/api/db/courses/{COURSE}/starter-seed-generation",
            json={"status": "generating"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(upsert.call_args.args[2], {"status": "generating"})

    def test_patch_for_a_missing_course_is_404(self) -> None:
        self.patch_repo("db_courses.course_exists", return_value=False)
        response = self.client.patch(
            f"/api/db/courses/{COURSE}/starter-seed-generation",
            json={"status": "generating"},
        )
        self.assertEqual(response.status_code, 404)


class SeedRouteTests(DbRouteTestCase):
    def test_list_returns_seeds_and_review_counts(self) -> None:
        self.patch_repo("db_seeds.list_seeds", return_value=[SEED])
        self.patch_repo(
            "db_seeds.count_seeds_by_review_status",
            return_value={"generated": 41, "approved": 9},
        )
        response = self.client.get(f"/api/db/courses/{COURSE}/seeds")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["reviewStatusCounts"], {"generated": 41, "approved": 9})

    def test_seed_response_carries_both_name_pairs(self) -> None:
        self.patch_repo("db_seeds.list_seeds", return_value=[SEED])
        self.patch_repo("db_seeds.count_seeds_by_review_status", return_value={})
        seed = self.client.get(f"/api/db/courses/{COURSE}/seeds").json()["seeds"][0]

        self.assertEqual(seed["instruction"], seed["question"])
        self.assertEqual(seed["response"], seed["answer"])
        self.assertEqual(seed["sourceChunkIds"], ["chunk-092"])
        self.assertEqual(seed["validation"]["score"], 0.92)

    def test_missing_seed_is_404(self) -> None:
        self.patch_repo("db_seeds.get_seed", return_value=None)
        response = self.client.get(f"/api/db/courses/{COURSE}/seeds/nope")
        self.assertEqual(response.status_code, 404)

    def test_create_seed_requires_the_course_to_exist(self) -> None:
        self.patch_repo("db_courses.course_exists", return_value=False)
        create = self.patch_repo("db_seeds.create_seed")
        response = self.client.post(
            f"/api/db/courses/{COURSE}/seeds",
            json={"question": "Q?", "answer": "A."},
        )

        self.assertEqual(response.status_code, 404)
        create.assert_not_called()

    def test_create_seed_returns_201(self) -> None:
        self.patch_repo("db_courses.course_exists", return_value=True)
        self.patch_repo("db_seeds.create_seed", return_value=SEED)
        response = self.client.post(
            f"/api/db/courses/{COURSE}/seeds",
            json={"question": "Q?", "answer": "A.", "origin": "user"},
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["seedId"], "-Oseed001")

    def test_create_seed_without_text_is_422(self) -> None:
        self.patch_repo("db_courses.course_exists", return_value=True)
        self.patch_repo(
            "db_seeds.create_seed", side_effect=ValueError("needs both")
        )
        response = self.client.post(
            f"/api/db/courses/{COURSE}/seeds", json={"category": "grading"}
        )
        self.assertEqual(response.status_code, 422)

    def test_review_rejects_an_unknown_status_before_querying(self) -> None:
        review = self.patch_repo("db_seeds.review_seed")
        response = self.client.post(
            f"/api/db/courses/{COURSE}/seeds/-Oseed001/review",
            json={"reviewStatus": "banished"},
        )

        self.assertEqual(response.status_code, 422)
        review.assert_not_called()

    def test_review_passes_edits_through(self) -> None:
        review = self.patch_repo(
            "db_seeds.review_seed",
            return_value={**SEED, "reviewStatus": "approved", "wasEdited": True},
        )
        response = self.client.post(
            f"/api/db/courses/{COURSE}/seeds/-Oseed001/review",
            json={
                "reviewStatus": "approved",
                "question": "Edited?",
                "answer": "Edited.",
                "reviewNotes": "Tightened.",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(review.call_args.kwargs["question"], "Edited?")
        self.assertEqual(review.call_args.kwargs["review_notes"], "Tightened.")
        self.assertTrue(response.json()["seed"]["wasEdited"])

    def test_review_of_a_missing_seed_is_404(self) -> None:
        self.patch_repo("db_seeds.review_seed", return_value=None)
        response = self.client.post(
            f"/api/db/courses/{COURSE}/seeds/nope/review",
            json={"reviewStatus": "approved"},
        )
        self.assertEqual(response.status_code, 404)

    def test_delete_passes_both_keys_and_returns_a_count(self) -> None:
        delete = self.patch_repo("db_seeds.delete_seed", return_value=True)
        response = self.client.delete(f"/api/db/courses/{COURSE}/seeds/-Oseed001")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"courseId": COURSE, "deleted": 1})
        self.assertEqual(delete.call_args.args[1:], (COURSE, "-Oseed001"))

    def test_deleting_a_seed_that_is_not_this_courses_is_404(self) -> None:
        """Cross-course deletion cannot succeed: the repo is keyed on both."""
        delete = self.patch_repo("db_seeds.delete_seed", return_value=False)
        response = self.client.delete(
            f"/api/db/courses/{OTHER_COURSE}/seeds/-Oseed001"
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(delete.call_args.args[1], OTHER_COURSE)


class EvaluationRouteTests(DbRouteTestCase):
    def test_list_returns_evaluation_records(self) -> None:
        self.patch_repo(
            "db_evaluations.list_evaluations", return_value=[EVALUATION]
        )
        response = self.client.get(f"/api/db/courses/{COURSE}/evaluations")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["evaluations"][0]["preferredModel"], "rag")
        self.assertEqual(body["evaluations"][0]["hallucinationFlags"], ["base"])

    def test_create_returns_201(self) -> None:
        self.patch_repo("db_courses.course_exists", return_value=True)
        self.patch_repo(
            "db_evaluations.create_evaluation", return_value=EVALUATION
        )
        response = self.client.post(
            f"/api/db/courses/{COURSE}/evaluations",
            json={
                "comparisonId": "cmp-1",
                "mostAccurate": "rag",
                "mostHelpful": "rag",
                "mostConcise": "base",
                "bestGrounded": "rag",
                "preferredModel": "rag",
                "hallucinationFlags": ["base"],
            },
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["id"], "-Oeval001")

    def test_create_requires_every_rating(self) -> None:
        response = self.client.post(
            f"/api/db/courses/{COURSE}/evaluations",
            json={"comparisonId": "cmp-1", "mostAccurate": "rag"},
        )
        self.assertEqual(response.status_code, 422)

    def test_delete_one_is_404_when_absent(self) -> None:
        self.patch_repo("db_evaluations.delete_evaluation", return_value=False)
        response = self.client.delete(
            f"/api/db/courses/{COURSE}/evaluations/missing"
        )
        self.assertEqual(response.status_code, 404)

    def test_bulk_delete_reports_how_many_went(self) -> None:
        delete_all = self.patch_repo(
            "db_evaluations.delete_all_evaluations", return_value=7
        )
        response = self.client.delete(f"/api/db/courses/{COURSE}/evaluations")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"courseId": COURSE, "deleted": 7})
        self.assertEqual(delete_all.call_args.args[1], COURSE)


class ModelRouteTests(DbRouteTestCase):
    REGISTRY = {
        "courseId": OTHER_COURSE,
        "currentVersion": "v1",
        "versions": {
            "v1": {
                "version": "v1",
                "baseModel": "meta-llama/Llama-3.2-3B-Instruct",
                "trainingExampleCount": 54,
                "status": "ready",
                "deployment": "offline",
                "artifactRef": "css-360-qlora/adapter",
                "createdAt": "2026-08-11T06:22:50.979479+00:00",
            }
        },
    }

    def test_registry_is_nested_by_version(self) -> None:
        self.patch_repo("db_models.get_model_registry", return_value=self.REGISTRY)
        response = self.client.get(f"/api/db/courses/{OTHER_COURSE}/model")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["currentVersion"], "v1")
        self.assertEqual(body["versions"]["v1"]["trainingExampleCount"], 54)
        self.assertEqual(
            body["versions"]["v1"]["baseModel"], "meta-llama/Llama-3.2-3B-Instruct"
        )

    def test_course_without_a_model_is_404(self) -> None:
        self.patch_repo("db_models.get_model_registry", return_value=None)
        response = self.client.get(f"/api/db/courses/{COURSE}/model")
        self.assertEqual(response.status_code, 404)


class ModelRequestRouteTests(DbRouteTestCase):
    REQUEST = {
        "courseId": COURSE,
        "status": "training",
        "requestedAt": "2026-08-12T00:00:00+00:00",
        "updatedAt": "2026-08-12T02:00:00+00:00",
        "approvedExampleCount": 54,
        "preparation": {
            "preparedAt": "2026-08-12T01:00:00+00:00",
            "datasetRef": "exports/css-350-spring-2026-n3h9",
            "trainExamples": 48,
            "validationExamples": 6,
            "splitSeed": 7,
        },
        "training": {"jobId": "123456", "mode": "full"},
        "currentRunId": "run-1",
    }

    def test_get_returns_nested_preparation_and_training(self) -> None:
        self.patch_repo(
            "db_model_requests.get_model_request", return_value=self.REQUEST
        )
        response = self.client.get(f"/api/db/courses/{COURSE}/model-request")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["preparation"]["splitSeed"], 7)
        self.assertEqual(body["training"]["jobId"], "123456")
        self.assertEqual(body["currentRunId"], "run-1")

    def test_absent_request_is_404(self) -> None:
        self.patch_repo("db_model_requests.get_model_request", return_value=None)
        response = self.client.get(f"/api/db/courses/{COURSE}/model-request")
        self.assertEqual(response.status_code, 404)

    def test_create_returns_201(self) -> None:
        self.patch_repo("db_courses.course_exists", return_value=True)
        self.patch_repo(
            "db_model_requests.create_model_request",
            return_value={**self.REQUEST, "status": "requested"},
        )
        response = self.client.post(
            f"/api/db/courses/{COURSE}/model-request",
            json={"approvedExampleCount": 54},
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["status"], "requested")

    def test_a_second_active_request_is_409(self) -> None:
        self.patch_repo("db_courses.course_exists", return_value=True)
        self.patch_repo(
            "db_model_requests.create_model_request",
            side_effect=db_model_requests.ActiveModelRequestError("outstanding"),
        )
        response = self.client.post(
            f"/api/db/courses/{COURSE}/model-request",
            json={"approvedExampleCount": 54},
        )
        self.assertEqual(response.status_code, 409)

    def test_patch_merges_only_supplied_fields(self) -> None:
        update = self.patch_repo(
            "db_model_requests.update_model_request", return_value=self.REQUEST
        )
        response = self.client.patch(
            f"/api/db/courses/{COURSE}/model-request",
            json={"status": "ready"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(update.call_args.args[2], {"status": "ready"})


class TrainingRunRouteTests(DbRouteTestCase):
    def test_list_returns_runs_with_nested_claim(self) -> None:
        self.patch_repo(
            "db_training_runs.list_training_runs", return_value=[TRAINING_RUN]
        )
        response = self.client.get(f"/api/db/courses/{COURSE}/training-runs")

        self.assertEqual(response.status_code, 200)
        run = response.json()["runs"][0]
        self.assertEqual(run["claim"]["owner"], "tillicum-runner")
        self.assertEqual(run["jobId"], "123456")
        self.assertEqual(run["attempt"], 1)

    def test_get_one_run(self) -> None:
        self.patch_repo(
            "db_training_runs.get_training_run", return_value=TRAINING_RUN
        )
        response = self.client.get(f"/api/db/courses/{COURSE}/training-runs/run-1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["datasetRef"], TRAINING_RUN["datasetRef"])

    def test_missing_run_is_404(self) -> None:
        self.patch_repo("db_training_runs.get_training_run", return_value=None)
        response = self.client.get(f"/api/db/courses/{COURSE}/training-runs/nope")
        self.assertEqual(response.status_code, 404)

    def test_enqueue_returns_201(self) -> None:
        self.patch_repo("db_courses.course_exists", return_value=True)
        self.patch_repo(
            "db_training_runs.enqueue_training_run",
            return_value={**TRAINING_RUN, "state": "queued"},
        )
        response = self.client.post(
            f"/api/db/courses/{COURSE}/training-runs",
            json={"mode": "full", "datasetRef": "exports/x"},
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["state"], "queued")

    def test_a_second_active_run_is_409(self) -> None:
        self.patch_repo("db_courses.course_exists", return_value=True)
        self.patch_repo(
            "db_training_runs.enqueue_training_run",
            side_effect=db_training_runs.ActiveTrainingRunError("active"),
        )
        response = self.client.post(
            f"/api/db/courses/{COURSE}/training-runs",
            json={"mode": "full", "datasetRef": "exports/x"},
        )
        self.assertEqual(response.status_code, 409)

    def test_an_unknown_mode_is_422(self) -> None:
        self.patch_repo("db_courses.course_exists", return_value=True)
        self.patch_repo(
            "db_training_runs.enqueue_training_run",
            side_effect=ValueError("Unknown training mode: 'turbo'"),
        )
        response = self.client.post(
            f"/api/db/courses/{COURSE}/training-runs",
            json={"mode": "turbo", "datasetRef": "exports/x"},
        )
        self.assertEqual(response.status_code, 422)

    def test_clear_claim_becomes_an_explicit_null_claim(self) -> None:
        """`clearClaim` releases a lease; omitting the claim leaves it alone."""
        update = self.patch_repo(
            "db_training_runs.update_training_run",
            return_value={**TRAINING_RUN, "state": "queued"},
        )
        response = self.client.patch(
            f"/api/db/courses/{COURSE}/training-runs/run-1",
            json={"state": "queued", "clearClaim": True},
        )

        self.assertEqual(response.status_code, 200)
        patch_body = update.call_args.args[3]
        self.assertIsNone(patch_body["claim"])
        self.assertNotIn("clearClaim", patch_body)

    def test_a_patch_without_claim_does_not_touch_it(self) -> None:
        update = self.patch_repo(
            "db_training_runs.update_training_run", return_value=TRAINING_RUN
        )
        self.client.patch(
            f"/api/db/courses/{COURSE}/training-runs/run-1",
            json={"jobId": "999"},
        )

        patch_body = update.call_args.args[3]
        self.assertEqual(patch_body, {"jobId": "999"})


class DatabaseFailureTests(DbRouteTestCase):
    def test_a_driver_failure_becomes_503_without_leaking_the_dsn(self) -> None:
        import psycopg

        self.patch_repo(
            "db_courses.list_courses",
            side_effect=psycopg.OperationalError(
                "connection failed: postgresql://syllabus_app:hunter2@localhost/syllabus_lab"
            ),
        )
        response = self.client.get("/api/db/courses")

        self.assertEqual(response.status_code, 503)
        detail = response.json()["detail"]
        self.assertNotIn("hunter2", detail)
        self.assertNotIn("postgresql://", detail)
        self.assertIn("listing courses", detail)

    def test_unconfigured_database_url_reports_what_to_set(self) -> None:
        from app.db import DatabaseConfigurationError

        self.patch_repo(
            "db_courses.list_courses",
            side_effect=DatabaseConfigurationError(
                "PostgreSQL is not configured. Set DATABASE_URL in the backend environment."
            ),
        )
        response = self.client.get("/api/db/courses")

        self.assertEqual(response.status_code, 503)
        self.assertIn("DATABASE_URL", response.json()["detail"])


class ParallelDeploymentTests(unittest.TestCase):
    """The live Firebase surface must be untouched by any of this."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_firebase_seed_route_still_exists_separately(self) -> None:
        spec = self.client.get("/openapi.json").json()["paths"]
        self.assertIn("/api/courses/{course_id}/seeds", spec)
        self.assertIn("/api/db/courses/{course_id}/seeds", spec)

    def test_db_routes_are_all_under_the_db_prefix(self) -> None:
        spec = self.client.get("/openapi.json").json()["paths"]
        db_paths = [path for path in spec if path.startswith("/api/db")]
        self.assertGreaterEqual(len(db_paths), 12)
        for path in db_paths:
            self.assertTrue(path.startswith("/api/db/"), path)

    def test_health_is_unaffected(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
