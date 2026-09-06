"""The fact-inventory endpoint must never leave a browser waiting on extraction.

The production symptom: Admin → course → Fact inventory → Inspect sat on
"Building…" indefinitely. The cause was not a hang. The cache version bump in
`fact_inventory_cache` invalidated every inventory built before it, so the
first Inspect on a course was a full rebuild — every batch of a 163-chunk
syllabus through the local model on the CPU, inside one HTTP request, with no
timeout on either side and nothing stopping a second click from starting a
second build in parallel with the first.

These tests pin the two halves of the fix: builds are shared per course, and a
caller that says `wait: false` gets an answer at once — the cached inventory,
a 202 while the build runs, or a 503 naming the failure — instead of a socket
held open for an hour. None of them call Ollama; the build is stubbed.
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app import fact_inventory_cache
from app.fact_inventory_cache import (
    FACT_INVENTORY_CACHE_VERSION,
    compute_index_fingerprint,
    fact_inventory_build_status,
    load_or_build_fact_inventory,
    peek_fact_inventory,
    reset_fact_inventory_builds_for_tests,
    start_fact_inventory_build,
)
from app.main import app
from app.storage import LocalCourseArtifactStorage

COURSE = "css-360-winter-2026-a7rp"

CHUNKS = [
    {
        "chunkId": f"chunk-{index:03d}",
        "sectionTitle": "Late Policy",
        "text": f"Policy {index}: late work is accepted for 24 hours. " * 6,
        "order": index,
    }
    for index in range(4)
]


def _inventory(fact_count: int = 3) -> dict[str, Any]:
    return {
        "model": "qwen3:4b",
        "facts": [
            {
                "factId": f"fact-{index:02d}",
                "statement": f"Statement {index}",
                "importance": "high",
                "importanceScore": 0.9,
                "studentAskLikelihood": 0.8,
                "complexity": 1,
                "usefulnessScore": 0.8,
                "sourceChunkIds": ["chunk-000"],
                "evidenceQuote": "late work is accepted for 24 hours",
                "kind": "policy",
                "scope": "course_wide",
                "seriesKey": None,
                "assignmentGroup": None,
                "seriesOrdinal": None,
            }
            for index in range(fact_count)
        ],
        "factCount": fact_count,
        "droppedCount": 0,
        "duplicatesRemoved": 0,
        "fallbackUsed": False,
        "countsByScope": {"course_wide": fact_count},
        "countsByKind": {"policy": fact_count},
        "countsBySeries": {},
    }


class _Storage:
    """A temp-dir storage with the course index already written."""

    def __enter__(self) -> LocalCourseArtifactStorage:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        storage = LocalCourseArtifactStorage(root_dir=root / "course_data", index_dir=root / "indexes")
        storage.save_index(COURSE, {"indexVersion": 2, "chunkCount": len(CHUNKS), "chunks": CHUNKS})
        return storage

    def __exit__(self, *exc: object) -> None:
        self._tmp.cleanup()


class SharedBuildTests(unittest.IsolatedAsyncioTestCase):
    """One course, one build at a time, however many callers ask."""

    def setUp(self) -> None:
        reset_fact_inventory_builds_for_tests()
        self.addCleanup(reset_fact_inventory_builds_for_tests)

    async def test_concurrent_callers_share_one_build(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        builds = {"n": 0}

        async def slow_build(**kwargs: Any) -> dict[str, Any]:
            builds["n"] += 1
            started.set()
            await release.wait()
            return _inventory()

        with _Storage() as storage, patch(
            "app.fact_inventory_cache.build_fact_inventory", new=AsyncMock(side_effect=slow_build)
        ):
            first = asyncio.ensure_future(
                load_or_build_fact_inventory(course_id=COURSE, raw_chunks=CHUNKS, storage=storage)
            )
            await started.wait()
            second = asyncio.ensure_future(
                load_or_build_fact_inventory(course_id=COURSE, raw_chunks=CHUNKS, storage=storage)
            )
            await asyncio.sleep(0)
            self.assertTrue(fact_inventory_build_status(COURSE)["building"])
            release.set()
            results = await asyncio.gather(first, second)

        self.assertEqual(builds["n"], 1)
        self.assertEqual([r["factCount"] for r in results], [3, 3])
        # Each awaiter has its own copy of the result.
        self.assertIsNot(results[0], results[1])
        self.assertFalse(fact_inventory_build_status(COURSE)["building"])

    async def test_a_failed_background_build_is_recorded_not_lost(self) -> None:
        with _Storage() as storage, patch(
            "app.fact_inventory_cache.build_fact_inventory",
            new=AsyncMock(side_effect=RuntimeError("disk full")),
        ):
            task = start_fact_inventory_build(course_id=COURSE, raw_chunks=CHUNKS, storage=storage)
            with self.assertRaises(RuntimeError):
                await task

        status = fact_inventory_build_status(COURSE)
        self.assertFalse(status["building"])
        self.assertEqual(status["lastError"], "disk full")
        self.assertEqual(fact_inventory_cache.take_fact_inventory_build_error(COURSE), "disk full")
        self.assertIsNone(fact_inventory_build_status(COURSE)["lastError"])

    async def test_peek_never_builds(self) -> None:
        build = AsyncMock(side_effect=AssertionError("peek must not build"))
        with _Storage() as storage, patch("app.fact_inventory_cache.build_fact_inventory", new=build):
            self.assertIsNone(peek_fact_inventory(course_id=COURSE, raw_chunks=CHUNKS, storage=storage))
            storage.save_fact_inventory(
                COURSE,
                {
                    "cacheVersion": FACT_INVENTORY_CACHE_VERSION,
                    "indexFingerprint": compute_index_fingerprint(CHUNKS),
                    "inventory": _inventory(2),
                },
            )
            cached = peek_fact_inventory(course_id=COURSE, raw_chunks=CHUNKS, storage=storage)
        assert cached is not None
        self.assertTrue(cached["cached"])
        self.assertEqual(cached["factCount"], 2)
        build.assert_not_called()


def _wait_until_not_building(course_id: str, timeout: float = 5.0) -> None:
    """Let a background build finish on the app's loop before polling again."""
    deadline = time.monotonic() + timeout
    while fact_inventory_build_status(course_id)["building"]:
        if time.monotonic() > deadline:
            raise AssertionError("the background build did not finish in time")
        time.sleep(0.01)


class NonBlockingRouteTests(unittest.TestCase):
    """`wait: false` answers at once, whatever state the build is in.

    The client is entered as a context manager so every request in a test runs
    on one long-lived event loop, as under uvicorn. A fresh loop per request
    would cancel the background task between polls, which is not what a
    deployment does.
    """

    def setUp(self) -> None:
        reset_fact_inventory_builds_for_tests()
        self.addCleanup(reset_fact_inventory_builds_for_tests)
        self.client = TestClient(app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self._storage_cm = _Storage()
        self.storage = self._storage_cm.__enter__()
        self.addCleanup(self._storage_cm.__exit__, None, None, None)
        patcher = patch("app.main.get_course_artifact_storage", return_value=self.storage)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _post(self, body: dict[str, Any] | None = None):
        return self.client.post(f"/api/courses/{COURSE}/facts/inventory", json=body or {})

    def test_cached_inventory_is_returned_immediately(self) -> None:
        self.storage.save_fact_inventory(
            COURSE,
            {
                "cacheVersion": FACT_INVENTORY_CACHE_VERSION,
                "indexFingerprint": compute_index_fingerprint(CHUNKS),
                "inventory": _inventory(5),
            },
        )
        build = AsyncMock(side_effect=AssertionError("must not build"))
        with patch("app.fact_inventory_cache.build_fact_inventory", new=build):
            response = self._post({"wait": False})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["factCount"], 5)
        self.assertTrue(body["cached"])
        build.assert_not_called()

    def test_a_missing_cache_starts_the_build_and_answers_202(self) -> None:
        # A threading.Event, released from the test thread, holds the build
        # open on the app's loop without any cross-thread asyncio calls.
        release = threading.Event()
        builds = {"n": 0}

        async def slow_build(**kwargs: Any) -> dict[str, Any]:
            builds["n"] += 1
            await asyncio.get_running_loop().run_in_executor(None, release.wait)
            return _inventory()

        with patch("app.fact_inventory_cache.build_fact_inventory", new=AsyncMock(side_effect=slow_build)):
            first = self._post({"wait": False})
            self.assertEqual(first.status_code, 202)
            self.assertEqual(first.json()["status"], "building")
            self.assertEqual(first.json()["courseId"], COURSE)
            self.assertIsNotNone(first.json()["startedAt"])

            # Polling joins the running build instead of starting another.
            second = self._post({"wait": False})
            self.assertEqual(second.status_code, 202)
            self.assertEqual(second.json()["startedAt"], first.json()["startedAt"])

            # A blocking caller arriving mid-build joins it too, and gets the result.
            release.set()
            done = self._post({"wait": True})
            self.assertEqual(done.status_code, 200)
            self.assertEqual(done.json()["factCount"], 3)

        self.assertEqual(builds["n"], 1)
        polled = self._post({"wait": False})
        self.assertEqual(polled.status_code, 200)
        self.assertTrue(polled.json()["cached"])

    def test_a_failed_build_is_reported_once_then_retried(self) -> None:
        with patch(
            "app.fact_inventory_cache.build_fact_inventory",
            new=AsyncMock(side_effect=RuntimeError("Ollama returned nothing usable")),
        ):
            started = self._post({"wait": False})
            self.assertEqual(started.status_code, 202)
            _wait_until_not_building(COURSE)
            failed = self._post({"wait": False})

        self.assertEqual(failed.status_code, 503)
        self.assertIn("Ollama returned nothing usable", failed.json()["detail"])
        self.assertNotIn("Traceback", failed.json()["detail"])

        # The failure was reported; the next poll starts a fresh build.
        with patch(
            "app.fact_inventory_cache.build_fact_inventory",
            new=AsyncMock(return_value=_inventory(1)),
        ):
            retried = self._post({"wait": False})
            self.assertEqual(retried.status_code, 202)
            _wait_until_not_building(COURSE)
            result = self._post({"wait": False})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["factCount"], 1)

    def test_the_default_still_waits_for_the_build(self) -> None:
        """Scripts and the allocation route keep the blocking behaviour."""
        with patch(
            "app.fact_inventory_cache.build_fact_inventory",
            new=AsyncMock(return_value=_inventory(4)),
        ):
            response = self._post({})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["factCount"], 4)
        self.assertFalse(response.json()["cached"])

    def test_missing_index_is_404_in_both_modes(self) -> None:
        self.storage.remove_index(COURSE)
        self.assertEqual(self._post({"wait": False}).status_code, 404)
        self.assertEqual(self._post({}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
