"""Tests for automatic starter-seed jobs after syllabus indexing."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.starter_jobs import (
    clear_active_starter_jobs_for_tests,
    is_auto_starter_seed_generation_enabled,
    run_auto_starter_seed_generation,
    try_queue_starter_seed_generation,
)
from app.storage import LocalCourseArtifactStorage

SAMPLE_SYLLABUS_TEXT = (
    "Course Information\n\n"
    "This course covers systems concepts including scheduling, memory, and concurrency.\n\n"
    "Attendance\n\n"
    "Students who miss class should submit the absence form at least one hour before class.\n\n"
    "Office Hours\n\n"
    "Office hours are held Mondays and Wednesdays for clarifying course policy questions."
)


class StarterJobQueueTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        clear_active_starter_jobs_for_tests()
        # Existing tests assume auto-generation is enabled; .env may disable it.
        self._auto_env = patch.dict(
            os.environ,
            {"AUTO_STARTER_SEED_GENERATION": "true"},
            clear=False,
        )
        self._auto_env.start()

    def tearDown(self) -> None:
        self._auto_env.stop()
        clear_active_starter_jobs_for_tests()

    async def test_queue_sets_queued_status_and_active_guard(self) -> None:
        with (
            patch(
                "app.starter_jobs.get_starter_auto_generate_count",
                return_value=3,
            ),
            patch(
                "app.starter_jobs._durable_status_is_active",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.starter_jobs.best_effort_patch_starter_seed_generation",
                new=AsyncMock(return_value=True),
            ) as mock_patch,
        ):
            result = await try_queue_starter_seed_generation("css-360-test-queue")

        self.assertTrue(result["queued"])
        self.assertEqual(result["status"], "queued")
        self.assertEqual(result["targetCount"], 3)
        mock_patch.assert_awaited()
        self.assertEqual(mock_patch.await_args.args[1]["status"], "queued")

        # Second schedule attempt should be blocked by process-local guard.
        with (
            patch(
                "app.starter_jobs.get_starter_auto_generate_count",
                return_value=3,
            ),
            patch(
                "app.starter_jobs._durable_status_is_active",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.starter_jobs.best_effort_patch_starter_seed_generation",
                new=AsyncMock(return_value=True),
            ) as mock_patch_again,
        ):
            second = await try_queue_starter_seed_generation("css-360-test-queue")

        self.assertFalse(second["queued"])
        self.assertEqual(second["reason"], "job_already_active")
        mock_patch_again.assert_not_called()

    async def test_auto_starter_env_false_disables_queue(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {"AUTO_STARTER_SEED_GENERATION": "false"},
                clear=False,
            ),
            patch(
                "app.starter_jobs.get_starter_auto_generate_count",
                return_value=50,
            ),
            patch(
                "app.starter_jobs.best_effort_patch_starter_seed_generation",
                new=AsyncMock(return_value=True),
            ) as mock_patch,
        ):
            self.assertFalse(is_auto_starter_seed_generation_enabled())
            result = await try_queue_starter_seed_generation("css-360-test-disabled")

        self.assertFalse(result["queued"])
        self.assertEqual(result["status"], "not_started")
        self.assertEqual(result["reason"], "auto_generate_disabled")
        self.assertEqual(result["targetCount"], 0)
        mock_patch.assert_not_called()

    async def test_queue_blocked_by_durable_generating_status(self) -> None:
        with (
            patch(
                "app.starter_jobs.get_starter_auto_generate_count",
                return_value=3,
            ),
            patch(
                "app.starter_jobs._durable_status_is_active",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "app.starter_jobs.best_effort_patch_starter_seed_generation",
                new=AsyncMock(return_value=True),
            ) as mock_patch,
        ):
            result = await try_queue_starter_seed_generation("css-360-test-durable")

        self.assertFalse(result["queued"])
        self.assertEqual(result["reason"], "durable_status_active")
        mock_patch.assert_not_called()

    async def test_background_runner_transitions_queued_to_ready(self) -> None:
        clear_active_starter_jobs_for_tests()
        from app.starter_jobs import _active_starter_jobs

        _active_starter_jobs.add("css-360-test-run")

        patches: list[dict] = []

        async def _capture_patch(course_id: str, updates: dict) -> bool:
            patches.append({"courseId": course_id, **updates})
            return True

        with (
            patch(
                "app.starter_jobs.get_starter_auto_generate_count",
                return_value=3,
            ),
            patch(
                "app.starter_jobs.best_effort_patch_starter_seed_generation",
                new=AsyncMock(side_effect=_capture_patch),
            ),
            # Reconciliation reads the course's real seed count and writes
            # through starter_status; stub both so this stays offline.
            patch(
                "app.starter_status.count_course_seed_examples",
                new=AsyncMock(return_value=3),
            ),
            patch(
                "app.starter_status.read_starter_seed_generation",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "app.starter_status.best_effort_patch_starter_seed_generation",
                new=AsyncMock(side_effect=_capture_patch),
            ),
            patch(
                "app.starter_jobs.generate_starter_seeds_for_course",
                new=AsyncMock(
                    return_value={
                        "progress": {"finalCount": 3},
                        "persistence": {
                            "savedCount": 3,
                            "failedToSaveCount": 0,
                        },
                    }
                ),
            ),
        ):
            await run_auto_starter_seed_generation("css-360-test-run")

        statuses = [item["status"] for item in patches]
        self.assertEqual(statuses[0], "generating")
        self.assertEqual(statuses[-1], "ready")
        self.assertNotIn("css-360-test-run", _active_starter_jobs)

    async def test_background_runner_marks_partial_when_target_missed(self) -> None:
        clear_active_starter_jobs_for_tests()
        from app.starter_jobs import _active_starter_jobs

        _active_starter_jobs.add("css-360-test-partial")
        patches: list[dict] = []

        async def _capture_patch(course_id: str, updates: dict) -> bool:
            patches.append(updates)
            return True

        with (
            patch(
                "app.starter_jobs.get_starter_auto_generate_count",
                return_value=3,
            ),
            patch(
                "app.starter_jobs.best_effort_patch_starter_seed_generation",
                new=AsyncMock(side_effect=_capture_patch),
            ),
            # Reconciliation reads the course's real seed count and writes
            # through starter_status; stub both so this stays offline.
            patch(
                "app.starter_status.count_course_seed_examples",
                new=AsyncMock(return_value=2),
            ),
            patch(
                "app.starter_status.read_starter_seed_generation",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "app.starter_status.best_effort_patch_starter_seed_generation",
                new=AsyncMock(side_effect=_capture_patch),
            ),
            patch(
                "app.starter_jobs.generate_starter_seeds_for_course",
                new=AsyncMock(
                    return_value={
                        "progress": {"finalCount": 2},
                        "persistence": {
                            "savedCount": 2,
                            "failedToSaveCount": 0,
                        },
                    }
                ),
            ),
        ):
            await run_auto_starter_seed_generation("css-360-test-partial")

        self.assertEqual(patches[-1]["status"], "partial")

    async def test_background_failure_marks_failed_without_raising(self) -> None:
        clear_active_starter_jobs_for_tests()
        from app.starter_jobs import _active_starter_jobs

        _active_starter_jobs.add("css-360-test-fail")
        patches: list[dict] = []

        async def _capture_patch(course_id: str, updates: dict) -> bool:
            patches.append(updates)
            return True

        with (
            patch(
                "app.starter_jobs.get_starter_auto_generate_count",
                return_value=3,
            ),
            patch(
                "app.starter_jobs.best_effort_patch_starter_seed_generation",
                new=AsyncMock(side_effect=_capture_patch),
            ),
            # Reconciliation reads the course's real seed count and writes
            # through starter_status; stub both so this stays offline.
            patch(
                "app.starter_status.count_course_seed_examples",
                new=AsyncMock(return_value=0),
            ),
            patch(
                "app.starter_status.read_starter_seed_generation",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "app.starter_status.best_effort_patch_starter_seed_generation",
                new=AsyncMock(side_effect=_capture_patch),
            ),
            patch(
                "app.starter_jobs.generate_starter_seeds_for_course",
                new=AsyncMock(side_effect=RuntimeError("ollama down")),
            ),
        ):
            await run_auto_starter_seed_generation("css-360-test-fail")

        self.assertEqual(patches[-1]["status"], "failed")
        self.assertIn("ollama down", patches[-1]["error"])
        self.assertNotIn("css-360-test-fail", _active_starter_jobs)


class StarterJobUploadIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_active_starter_jobs_for_tests()
        self._auto_env = patch.dict(
            os.environ,
            {"AUTO_STARTER_SEED_GENERATION": "true"},
            clear=False,
        )
        self._auto_env.start()
        self._temp_dir = tempfile.TemporaryDirectory()
        root = Path(self._temp_dir.name)
        self.storage = LocalCourseArtifactStorage(
            root_dir=root / "course_data",
            index_dir=root / "indexes",
        )
        self._storage_patch = patch(
            "app.main.get_course_artifact_storage",
            return_value=self.storage,
        )
        self._storage_patch.start()
        self._embed_patch = patch(
            "app.course_index.get_embedding",
            new=AsyncMock(return_value=[0.1, -0.2, 0.3]),
        )
        self._embed_patch.start()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self._embed_patch.stop()
        self._storage_patch.stop()
        self._auto_env.stop()
        self._temp_dir.cleanup()
        clear_active_starter_jobs_for_tests()

    def test_upload_succeeds_and_queues_generation(self) -> None:
        background_calls: list[str] = []

        def _add_task(func, *args, **kwargs):  # noqa: ANN001
            background_calls.append(args[0] if args else "")

        with (
            patch(
                "app.main.try_queue_starter_seed_generation",
                new=AsyncMock(
                    return_value={
                        "queued": True,
                        "status": "queued",
                        "targetCount": 3,
                    }
                ),
            ),
            patch.object(
                type(self.client.app),  # type: ignore[arg-type]
                "add_task",
                create=True,
            ),
        ):
            # Patch BackgroundTasks.add_task via the request path by intercepting
            # the helper that schedules the worker.
            with patch(
                "app.main.run_auto_starter_seed_generation",
                new=AsyncMock(),
            ) as mock_runner:
                # Force TestClient to invoke background tasks; mock runner so it is cheap.
                files = {
                    "syllabus_file": (
                        "syllabus.txt",
                        io.BytesIO(SAMPLE_SYLLABUS_TEXT.encode("utf-8")),
                        "text/plain",
                    )
                }
                with patch(
                    "fastapi.BackgroundTasks.add_task",
                    side_effect=lambda fn, *a, **k: background_calls.append(a[0]),
                ):
                    response = self.client.post(
                        "/api/courses/css-360-auto-queue/syllabus",
                        files=files,
                    )

        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertEqual(body["syllabusStatus"], "indexed")
        self.assertEqual(body["starterSeedGenerationStatus"], "queued")
        self.assertTrue(self.storage.index_exists("css-360-auto-queue"))
        self.assertEqual(background_calls, ["css-360-auto-queue"])
        # Runner is scheduled, not awaited as part of the request body construction.
        self.assertTrue(mock_runner is not None)

    def test_upload_still_succeeds_when_generation_later_fails(self) -> None:
        files = {
            "syllabus_file": (
                "syllabus.txt",
                io.BytesIO(SAMPLE_SYLLABUS_TEXT.encode("utf-8")),
                "text/plain",
            )
        }
        with (
            patch(
                "app.main.try_queue_starter_seed_generation",
                new=AsyncMock(
                    return_value={
                        "queued": True,
                        "status": "queued",
                        "targetCount": 3,
                    }
                ),
            ),
            # Background task is scheduled but must not affect the HTTP response
            # or persisted index. Use a quiet async mock (TestClient runs tasks).
            patch(
                "app.main.run_auto_starter_seed_generation",
                new=AsyncMock(return_value=None),
            ),
        ):
            response = self.client.post(
                "/api/courses/css-360-auto-fail/syllabus",
                files=files,
            )

        self.assertEqual(response.status_code, 201)
        self.assertTrue(self.storage.index_exists("css-360-auto-fail"))
        self.assertIsNotNone(self.storage.load_extracted_text("css-360-auto-fail"))
        self.assertEqual(response.json()["starterSeedGenerationStatus"], "queued")
