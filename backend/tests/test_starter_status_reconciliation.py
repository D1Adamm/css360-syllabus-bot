"""starterSeedGeneration must describe the course, not one run.

The live drift these cover: CSS 350 held 50 seeds while its record said
"partial, 9 of 50", and CSS 360 held 81 while its record said 34. Both were
topped up through routes that never wrote the record at all, so it kept
describing the original post-upload run forever.

Firebase is never contacted here — the seed fetch and the metadata patch are
both stubbed.
"""

from __future__ import annotations

import os
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from app.firebase_metadata import (
    count_course_seed_examples,
    reconcile_starter_seed_generation,
    resolve_reconciled_starter_status,
)

COURSE = "css-350-spring-2026-n3h9"


def _payload(count: int) -> dict[str, Any]:
    return {f"-Oseed{index:03d}": {"question": "Q?"} for index in range(count)}


class StatusResolutionTests(unittest.TestCase):
    def test_reaching_target_is_ready(self) -> None:
        self.assertEqual(
            resolve_reconciled_starter_status(target_count=50, actual_count=50),
            "ready",
        )

    def test_exceeding_target_is_still_ready(self) -> None:
        """CSS 360 holds 81 against a target of 50."""
        self.assertEqual(
            resolve_reconciled_starter_status(target_count=50, actual_count=81),
            "ready",
        )

    def test_short_of_target_is_partial(self) -> None:
        self.assertEqual(
            resolve_reconciled_starter_status(target_count=50, actual_count=9),
            "partial",
        )

    def test_no_seeds_is_failed(self) -> None:
        self.assertEqual(
            resolve_reconciled_starter_status(target_count=50, actual_count=0),
            "failed",
        )

    def test_no_target_is_ready(self) -> None:
        self.assertEqual(
            resolve_reconciled_starter_status(target_count=0, actual_count=0),
            "ready",
        )


class SeedCountingTests(unittest.IsolatedAsyncioTestCase):
    async def test_counts_only_record_shaped_children(self) -> None:
        payload = {**_payload(3), "junk": "not a record"}
        with patch(
            "app.firebase_metadata.fetch_course_seed_examples",
            new=AsyncMock(return_value=payload),
        ):
            self.assertEqual(await count_course_seed_examples(COURSE), 3)

    async def test_empty_node_counts_as_zero(self) -> None:
        with patch(
            "app.firebase_metadata.fetch_course_seed_examples",
            new=AsyncMock(return_value={}),
        ):
            self.assertEqual(await count_course_seed_examples(COURSE), 0)

    async def test_unreadable_firebase_is_none_not_zero(self) -> None:
        """None means "cannot say"; zero would overwrite a true record."""
        with patch(
            "app.firebase_metadata.fetch_course_seed_examples",
            new=AsyncMock(side_effect=HTTPException(status_code=503, detail="down")),
        ):
            self.assertIsNone(await count_course_seed_examples(COURSE))


class ReconciliationTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.patches: list[dict[str, Any]] = []

    def _stub(self, *, seed_count: int | None, stored: dict[str, Any] | None):
        async def _capture(course_id: str, updates: dict[str, Any]) -> bool:
            self.patches.append(updates)
            return True

        fetch = (
            AsyncMock(side_effect=HTTPException(status_code=503, detail="down"))
            if seed_count is None
            else AsyncMock(return_value=_payload(seed_count))
        )
        return (
            patch("app.firebase_metadata.fetch_course_seed_examples", new=fetch),
            patch(
                "app.firebase_metadata.read_starter_seed_generation",
                new=AsyncMock(return_value=stored),
            ),
            patch(
                "app.firebase_metadata.best_effort_patch_starter_seed_generation",
                new=AsyncMock(side_effect=_capture),
            ),
        )


class StarterReconciliationScenarios(ReconciliationTestCase):
    async def _reconcile(
        self,
        *,
        seed_count: int | None,
        stored: dict[str, Any] | None = None,
        target_count: int = 50,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        fetch_patch, read_patch, write_patch = self._stub(
            seed_count=seed_count, stored=stored
        )
        with fetch_patch, read_patch, write_patch:
            return await reconcile_starter_seed_generation(
                COURSE, target_count=target_count, **kwargs
            )

    async def test_1_original_run_below_target_is_partial(self) -> None:
        applied = await self._reconcile(seed_count=9)

        self.assertEqual(applied["status"], "partial")
        self.assertEqual(applied["finalCount"], 9)
        self.assertEqual(applied["savedCount"], 9)
        self.assertEqual(applied["targetCount"], 50)

    async def test_2_top_up_to_target_becomes_ready_with_real_count(self) -> None:
        """The CSS 350 case: record said 9, course holds 50."""
        stale = {
            "status": "partial",
            "targetCount": 50,
            "finalCount": 9,
            "savedCount": 9,
        }
        applied = await self._reconcile(seed_count=50, stored=stale)

        self.assertEqual(applied["status"], "ready")
        self.assertEqual(applied["finalCount"], 50)
        self.assertEqual(applied["savedCount"], 50)

    async def test_3_top_up_beyond_target_stays_ready_with_real_count(self) -> None:
        """The CSS 360 case: record said 34, course holds 81 against a target of 50."""
        stale = {
            "status": "partial",
            "targetCount": 50,
            "finalCount": 34,
            "savedCount": 34,
        }
        applied = await self._reconcile(seed_count=81, stored=stale)

        self.assertEqual(applied["status"], "ready")
        self.assertEqual(applied["finalCount"], 81)
        self.assertEqual(applied["savedCount"], 81)
        # The count is not clamped to the target: 81 seeds exist and the
        # professor can open all of them.
        self.assertEqual(applied["targetCount"], 50)

    async def test_4_forced_failure_stays_failed_but_counts_are_true(self) -> None:
        applied = await self._reconcile(
            seed_count=12, force_status="failed", error="Ollama timed out"
        )

        self.assertEqual(applied["status"], "failed")
        self.assertEqual(applied["error"], "Ollama timed out")
        # The run failed; the 12 seeds it did save still exist.
        self.assertEqual(applied["finalCount"], 12)
        self.assertEqual(applied["savedCount"], 12)

    async def test_stored_target_wins_over_a_caller_supplied_one(self) -> None:
        stored = {"status": "partial", "targetCount": 50, "savedCount": 9}
        applied = await self._reconcile(seed_count=30, stored=stored, target_count=10)

        self.assertEqual(applied["targetCount"], 50)
        self.assertEqual(applied["status"], "partial")

    async def test_unreadable_count_writes_nothing(self) -> None:
        applied = await self._reconcile(seed_count=None)

        self.assertIsNone(applied)
        self.assertEqual(self.patches, [])

    async def test_completed_at_is_stamped_for_this_reconciliation(self) -> None:
        applied = await self._reconcile(seed_count=50)
        self.assertTrue(applied["completedAt"])

    async def test_started_at_is_left_alone_unless_supplied(self) -> None:
        """A top-up must not claim to have started the original run."""
        applied = await self._reconcile(seed_count=50)
        self.assertNotIn("startedAt", applied)

        applied = await self._reconcile(seed_count=50, started_at="2026-08-13T02:00:00Z")
        self.assertEqual(applied["startedAt"], "2026-08-13T02:00:00Z")

    async def test_no_progress_percentage_is_invented(self) -> None:
        applied = await self._reconcile(seed_count=25)
        for key in applied:
            self.assertNotIn("percent", key.lower())
            self.assertNotIn("progress", key.lower())

    async def test_5_reconciliation_only_reads_seeds_and_writes_metadata(self) -> None:
        """It must never delete or regenerate a seed."""
        seen: list[str] = []

        async def _fetch(course_id: str) -> dict[str, Any]:
            seen.append("read_seeds")
            return _payload(50)

        async def _write(course_id: str, updates: dict[str, Any]) -> bool:
            seen.append("write_metadata")
            return True

        with (
            patch(
                "app.firebase_metadata.fetch_course_seed_examples",
                new=AsyncMock(side_effect=_fetch),
            ),
            patch(
                "app.firebase_metadata.read_starter_seed_generation",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "app.firebase_metadata.best_effort_patch_starter_seed_generation",
                new=AsyncMock(side_effect=_write),
            ),
            patch(
                "app.firebase_seeds.save_course_seed_example",
                new=AsyncMock(side_effect=AssertionError("must not write seeds")),
            ),
            patch(
                "app.firebase_seeds.patch_course_seed_example",
                new=AsyncMock(side_effect=AssertionError("must not edit seeds")),
            ),
        ):
            await reconcile_starter_seed_generation(COURSE, target_count=50)

        self.assertEqual(seen, ["read_seeds", "write_metadata"])

    async def test_the_metadata_patch_touches_no_seed_fields(self) -> None:
        applied = await self._reconcile(seed_count=50)
        for key in applied:
            self.assertNotIn(key, {"seedExamples", "seeds", "id", "question", "answer"})


class AutoJobReconciliationTests(unittest.IsolatedAsyncioTestCase):
    """The post-upload job now describes the course, with the old path as fallback."""

    def setUp(self) -> None:
        from app.starter_jobs import clear_active_starter_jobs_for_tests

        clear_active_starter_jobs_for_tests()
        self.patches: list[dict[str, Any]] = []

    async def _run_job(
        self,
        *,
        seed_count: int | None,
        run_result: dict[str, Any] | None = None,
        side_effect: Exception | None = None,
        target: int = 50,
    ) -> list[dict[str, Any]]:
        from app.starter_jobs import _active_starter_jobs, run_auto_starter_seed_generation

        course_id = "css-350-spring-2026-n3h9"
        _active_starter_jobs.add(course_id)

        async def _capture(cid: str, updates: dict[str, Any]) -> bool:
            self.patches.append(updates)
            return True

        fetch = (
            AsyncMock(side_effect=HTTPException(status_code=503, detail="down"))
            if seed_count is None
            else AsyncMock(return_value=_payload(seed_count))
        )
        generate = AsyncMock(
            side_effect=side_effect,
            return_value=run_result
            or {
                "progress": {"finalCount": 9, "achievableCeiling": 9, "limitingFactor": "none"},
                "persistence": {"savedCount": 9, "failedToSaveCount": 0},
            },
        )

        with (
            # backend/.env ships AUTO_STARTER_SEED_GENERATION=false, so the
            # worker would otherwise return before doing anything.
            patch.dict(os.environ, {"AUTO_STARTER_SEED_GENERATION": "true"}),
            patch("app.starter_jobs.get_starter_auto_generate_count", return_value=target),
            patch(
                "app.starter_jobs.best_effort_patch_starter_seed_generation",
                new=AsyncMock(side_effect=_capture),
            ),
            patch(
                "app.firebase_metadata.best_effort_patch_starter_seed_generation",
                new=AsyncMock(side_effect=_capture),
            ),
            patch("app.firebase_metadata.fetch_course_seed_examples", new=fetch),
            patch(
                "app.firebase_metadata.read_starter_seed_generation",
                new=AsyncMock(return_value=None),
            ),
            patch("app.starter_jobs.generate_starter_seeds_for_course", new=generate),
        ):
            await run_auto_starter_seed_generation(course_id)

        return self.patches

    async def test_job_records_the_courses_real_count(self) -> None:
        """A run that saved 9 into a course already holding 50 reports 50."""
        patches = await self._run_job(seed_count=50)

        final = patches[-1]
        self.assertEqual(final["status"], "ready")
        self.assertEqual(final["savedCount"], 50)
        self.assertEqual(final["finalCount"], 50)

    async def test_job_still_reports_partial_when_short(self) -> None:
        patches = await self._run_job(seed_count=9)

        final = patches[-1]
        self.assertEqual(final["status"], "partial")
        self.assertEqual(final["savedCount"], 9)

    async def test_job_keeps_the_limiting_factor_detail(self) -> None:
        patches = await self._run_job(seed_count=9)

        final = patches[-1]
        self.assertEqual(final["limitingFactor"], "none")
        self.assertEqual(final["achievableCeiling"], 9)

    async def test_job_falls_back_to_run_counts_when_firebase_cannot_be_counted(
        self,
    ) -> None:
        """Unchanged pre-reconciliation behavior, not a false zero."""
        patches = await self._run_job(seed_count=None)

        final = patches[-1]
        self.assertEqual(final["status"], "partial")
        self.assertEqual(final["finalCount"], 9)
        self.assertEqual(final["savedCount"], 9)

    async def test_a_crashed_run_stays_failed(self) -> None:
        patches = await self._run_job(
            seed_count=12, side_effect=RuntimeError("ollama exploded")
        )

        final = patches[-1]
        self.assertEqual(final["status"], "failed")
        self.assertIn("ollama exploded", final["error"])

    async def test_a_crashed_run_still_reports_seeds_that_survived(self) -> None:
        patches = await self._run_job(
            seed_count=12, side_effect=RuntimeError("ollama exploded")
        )

        final = patches[-1]
        self.assertEqual(final["savedCount"], 12)
        self.assertEqual(final["finalCount"], 12)

    async def test_a_crashed_run_with_no_seeds_reports_zero(self) -> None:
        patches = await self._run_job(
            seed_count=0, side_effect=RuntimeError("ollama exploded")
        )

        final = patches[-1]
        self.assertEqual(final["status"], "failed")
        self.assertEqual(final["savedCount"], 0)


if __name__ == "__main__":
    unittest.main()


class TopUpRouteReconciliationTests(unittest.TestCase):
    """The route that caused the drift now records the reconciled state.

    Both `/seeds/top-up` and `/seeds/generate-starter` persisted seeds and wrote
    no starterSeedGeneration at all, which is why CSS 350 kept reporting a run
    from days earlier.
    """

    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from app.main import app
        from app.starter_jobs import clear_active_starter_jobs_for_tests

        clear_active_starter_jobs_for_tests()
        self.client = TestClient(app)
        self.patches: list[dict[str, Any]] = []

    def _run_result(self, *, target: int, saved: int) -> dict[str, Any]:
        return {
            "courseId": COURSE,
            "model": "qwen3:8b",
            "targetCount": target,
            "seeds": [],
            # StarterSeedProgress requires these; the values are irrelevant to
            # what this test asserts, which is the metadata write afterwards.
            "progress": {
                "eligibleChunks": 0,
                "selectedChunks": 0,
                "chunksProcessed": 0,
                "chunksSkipped": 0,
                "generationCalls": 0,
                "validationCalls": 0,
                "ollamaCalls": 0,
                "candidatesGenerated": saved,
                "candidatesValidated": saved,
                "candidatesAccepted": saved,
                "candidatesRejected": 0,
                "duplicatesRemoved": 0,
                "finalCount": saved,
                "savedCount": saved,
                "status": "ready",
            },
            "persistence": {
                "generatedCount": saved,
                "savedCount": saved,
                "alreadyExistingCount": 0,
                "failedToSaveCount": 0,
            },
        }

    def _call(
        self,
        path: str,
        *,
        body: dict[str, Any],
        seed_count: int,
        saved: int = 41,
        target: int = 50,
    ):
        async def _capture(course_id: str, updates: dict[str, Any]) -> bool:
            self.patches.append(updates)
            return True

        with (
            patch(
                "app.main.generate_starter_seeds_for_course",
                new=AsyncMock(return_value=self._run_result(target=target, saved=saved)),
            ),
            patch(
                "app.firebase_metadata.fetch_course_seed_examples",
                new=AsyncMock(return_value=_payload(seed_count)),
            ),
            patch(
                "app.firebase_metadata.read_starter_seed_generation",
                new=AsyncMock(
                    return_value={
                        "status": "partial",
                        "targetCount": target,
                        "finalCount": 9,
                        "savedCount": 9,
                    }
                ),
            ),
            patch(
                "app.firebase_metadata.best_effort_patch_starter_seed_generation",
                new=AsyncMock(side_effect=_capture),
            ),
        ):
            return self.client.post(f"/api/courses/{COURSE}{path}", json=body)

    def test_top_up_reaching_target_marks_the_course_ready(self) -> None:
        """The exact CSS 350 sequence: 9 stored, top-up adds 41, course holds 50."""
        response = self._call("/seeds/top-up", body={"save": True}, seed_count=50)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.patches), 1)
        applied = self.patches[0]
        self.assertEqual(applied["status"], "ready")
        self.assertEqual(applied["savedCount"], 50)
        self.assertEqual(applied["finalCount"], 50)

    def test_top_up_beyond_target_reports_the_real_count(self) -> None:
        """The CSS 360 sequence: the course ends up holding 81 against a target of 50."""
        response = self._call("/seeds/top-up", body={"save": True}, seed_count=81)

        self.assertEqual(response.status_code, 200)
        applied = self.patches[0]
        self.assertEqual(applied["status"], "ready")
        self.assertEqual(applied["savedCount"], 81)

    def test_top_up_still_short_stays_partial(self) -> None:
        response = self._call("/seeds/top-up", body={"save": True}, seed_count=20)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.patches[0]["status"], "partial")
        self.assertEqual(self.patches[0]["savedCount"], 20)

    def test_a_dry_run_top_up_writes_no_metadata(self) -> None:
        """save=false persists nothing, so there is nothing to reconcile."""
        response = self._call("/seeds/top-up", body={"save": False}, seed_count=9)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.patches, [])

    def test_manual_generate_starter_also_reconciles(self) -> None:
        response = self._call(
            "/seeds/generate-starter",
            body={"targetCount": 50, "save": True},
            seed_count=50,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.patches[0]["status"], "ready")
        self.assertEqual(self.patches[0]["savedCount"], 50)

    def test_reconciliation_failure_does_not_fail_the_request(self) -> None:
        """Seeds are already saved; a metadata write must not undo that."""
        with (
            patch(
                "app.main.generate_starter_seeds_for_course",
                new=AsyncMock(return_value=self._run_result(target=50, saved=41)),
            ),
            patch(
                "app.firebase_metadata.fetch_course_seed_examples",
                new=AsyncMock(
                    side_effect=HTTPException(status_code=503, detail="down")
                ),
            ),
        ):
            response = self.client.post(
                f"/api/courses/{COURSE}/seeds/top-up", json={"save": True}
            )

        self.assertEqual(response.status_code, 200)


class FirebaseNetworkIsolationTests(unittest.TestCase):
    """No backend test may reach the real Firebase database.

    `backend/.env` carries a working FIREBASE_DATABASE_URL, so an unstubbed path
    does not fail — it succeeds, against production. That is how a suite run
    kept recreating `courses/css-360-summer-2026-demo`: the generate-starter
    route stubbed seed persistence but not the starter-status reconciliation
    that follows it, and the reconciliation's read and write both went out.

    These pin the guard in `tests/conftest.py` rather than any one caller.
    """

    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from app.main import app
        from app.starter_jobs import clear_active_starter_jobs_for_tests

        clear_active_starter_jobs_for_tests()
        self.client = TestClient(app)

    def test_the_guard_blocks_an_external_request(self) -> None:
        """The guard must actually fire, or everything below proves nothing."""
        import asyncio

        import httpx

        from conftest import ExternalRequestBlocked

        async def _attempt() -> None:
            async with httpx.AsyncClient() as client:
                await client.get(
                    "https://example-default-rtdb.firebaseio.com/courses.json"
                )

        with self.assertRaises(ExternalRequestBlocked) as caught:
            asyncio.run(_attempt())
        self.assertIn("firebaseio.com", str(caught.exception))

    def test_local_requests_are_still_allowed(self) -> None:
        """Ollama and the fine-tuned tunnel are loopback; they must pass."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_firebase_is_unconfigured_during_tests(self) -> None:
        from app.firebase_seeds import (
            FirebaseConfigurationError,
            get_firebase_database_url,
        )

        with self.assertRaises(FirebaseConfigurationError):
            get_firebase_database_url()

    def test_generate_starter_with_save_makes_no_external_request(self) -> None:
        """The exact leaking path, with nothing but the guard protecting it.

        Reconciliation is deliberately NOT stubbed here: the assertion is that
        the whole route completes without a request leaving the machine.
        """
        run_result = {
            "courseId": COURSE,
            "model": "qwen3:8b",
            "targetCount": 50,
            "seeds": [],
            "progress": {
                "eligibleChunks": 0,
                "selectedChunks": 0,
                "chunksProcessed": 0,
                "chunksSkipped": 0,
                "generationCalls": 0,
                "validationCalls": 0,
                "ollamaCalls": 0,
                "candidatesGenerated": 0,
                "candidatesValidated": 0,
                "candidatesAccepted": 0,
                "candidatesRejected": 0,
                "duplicatesRemoved": 0,
                "finalCount": 0,
            },
            "persistence": {
                "generatedCount": 0,
                "savedCount": 0,
                "alreadyExistingCount": 0,
                "failedToSaveCount": 0,
            },
        }

        with patch(
            "app.main.generate_starter_seeds_for_course",
            new=AsyncMock(return_value=run_result),
        ):
            for path in ("/seeds/generate-starter", "/seeds/top-up"):
                with self.subTest(path=path):
                    response = self.client.post(
                        f"/api/courses/{COURSE}{path}",
                        json={"targetCount": 50, "save": True},
                    )
                    self.assertEqual(response.status_code, 200)

    def test_reconciliation_without_firebase_config_writes_nothing(self) -> None:
        """Unconfigured Firebase must be a no-op, not an error and not a write."""
        import asyncio

        applied = asyncio.run(
            reconcile_starter_seed_generation(COURSE, target_count=50)
        )
        self.assertIsNone(applied)

    def test_auto_job_makes_no_external_request(self) -> None:
        import asyncio

        from app.starter_jobs import (
            _active_starter_jobs,
            run_auto_starter_seed_generation,
        )

        _active_starter_jobs.add(COURSE)
        with (
            patch.dict(os.environ, {"AUTO_STARTER_SEED_GENERATION": "true"}),
            patch("app.starter_jobs.get_starter_auto_generate_count", return_value=3),
            patch(
                "app.starter_jobs.generate_starter_seeds_for_course",
                new=AsyncMock(
                    return_value={
                        "progress": {"finalCount": 3},
                        "persistence": {"savedCount": 3, "failedToSaveCount": 0},
                    }
                ),
            ),
        ):
            asyncio.run(run_auto_starter_seed_generation(COURSE))

        self.assertNotIn(COURSE, _active_starter_jobs)
