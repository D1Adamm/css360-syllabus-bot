"""Unit tests for the durable training queue and the Tillicum runner.

The queue now lives in PostgreSQL and the runner reaches it through the
backend's `/api/training-queue` router. Nothing here touches the network,
Slurm, or the cluster: the HTTP transport is injected, so every request the
queue would make is inspected instead of sent, and the endpoints it would hit
are served by an in-memory stand-in.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import urllib.parse
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "lib"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so dataclasses can resolve the module, and so
    # the runner's own `import training_queue` gets this exact module rather
    # than a second copy with different exception classes.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


queue_module = _load("training_queue", REPO_ROOT / "scripts" / "lib" / "training_queue.py")
runner = _load("run_training_queue", REPO_ROOT / "training" / "run_training_queue.py")
helpers = _load(
    "qlora_training_helpers", REPO_ROOT / "scripts" / "lib" / "qlora_training_helpers.py"
)

COURSE_A = "css-490-spring-2026-cgvl"
COURSE_B = "css-350-winter-2026-drlb"
NOW = datetime(2026, 8, 12, 18, 0, 0, tzinfo=timezone.utc)

API_BASE_URL = "https://backend.example.test"
WORKER_TOKEN = "test-worker-token"


def make_run_record(
    course_id: str = COURSE_A,
    *,
    state: str = "queued",
    mode: str = "full",
    enqueued_at: str = "2026-08-12T17:00:00Z",
    claim: dict | None = None,
    attempt: int = 0,
    train: int = 38,
    validation: int = 4,
    job_id: str = "",
) -> dict:
    record = {
        "courseId": course_id,
        "mode": mode,
        "state": state,
        "enqueuedAt": enqueued_at,
        "updatedAt": enqueued_at,
        "datasetRef": f"exports/{course_id}",
        "approvedExampleCount": 42,
        "trainExamples": train,
        "validationExamples": validation,
        "attempt": attempt,
    }
    if claim is not None:
        record["claim"] = claim
    if job_id:
        record["jobId"] = job_id
    return record


def make_request_record(
    course_id: str = COURSE_A,
    *,
    status: str = "preparing",
    current_run_id: str = "run-1",
) -> dict:
    return {
        "courseId": course_id,
        "status": status,
        "requestedAt": "2026-08-12T16:00:00Z",
        "updatedAt": "2026-08-12T16:30:00Z",
        "approvedExampleCount": 42,
        "currentRunId": current_run_id,
        "preparation": {
            "preparedAt": "2026-08-12T16:30:00Z",
            "sourceApprovedExampleCount": 42,
            "datasetRef": f"exports/{course_id}",
            "trainExamples": 38,
            "validationExamples": 4,
        },
    }


class FakeLauncher:
    def __init__(
        self,
        returncode: int = 0,
        stdout: str = "Submitted batch job 9182736\n",
        stderr: str = "",
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.calls: list[tuple[list[str], Path]] = []

    def __call__(self, command: list[str], cwd: Path) -> "runner.LauncherResult":
        self.calls.append((list(command), cwd))
        return runner.LauncherResult(
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


def boom_launcher(command: list[str], cwd: Path) -> "runner.LauncherResult":
    raise AssertionError(f"launcher must not be called: {command} cwd={cwd}")


class FakeQueueApi:
    """An in-memory stand-in for the backend's `/api/training-queue` router.

    The queue moved from a database the runner wrote directly to an HTTP API in
    front of PostgreSQL, and this models that API rather than the store behind
    it. What it has to get right is the contract the runner programs against:
    which runs `/pending` offers, that `/claim` hands out at most one and only
    to the first caller, and that a submission moves the run and the model
    request together.

    `claim` is served the way the real endpoint is — pick the oldest eligible
    run, mark it claimed, return it — because that is what a single
    `SELECT ... FOR UPDATE SKIP LOCKED` plus its UPDATE amounts to from the
    caller's side. The real atomicity is asserted against the SQL itself, in
    the backend suite.
    """

    def __init__(
        self,
        *,
        runs: dict | None = None,
        requests: dict | None = None,
    ) -> None:
        # {courseId: {runId: record}}
        self.runs: dict = runs or {}
        # {courseId: record}
        self.model_requests: dict = requests or {}
        self.calls: list[tuple[str, str]] = []
        self.writes: list[tuple[str, str, object]] = []
        self.model_versions: list[dict] = []
        self.token_seen: list[str] = []

    # -- accessors used by tests ------------------------------------------ #
    def run(self, course_id: str, run_id: str) -> dict | None:
        return self.runs.get(course_id, {}).get(run_id)

    def request(self, course_id: str) -> dict | None:
        return self.model_requests.get(course_id)

    def set_run(self, course_id: str, run_id: str, record: dict | None) -> None:
        course = self.runs.setdefault(course_id, {})
        if record is None:
            course.pop(run_id, None)
        else:
            course[run_id] = record

    def set_request(self, course_id: str, record: dict) -> None:
        self.model_requests[course_id] = record

    # -- queue semantics --------------------------------------------------- #
    def _claimable(self, course_ids: list | None, now: datetime) -> list[tuple[str, str, dict]]:
        found = []
        for course_id, runs in self.runs.items():
            if course_ids is not None and course_id not in course_ids:
                continue
            for run_id, record in runs.items():
                parsed = queue_module.parse_run(run_id, record)
                if parsed is not None and queue_module.is_claimable(parsed, now):
                    found.append((course_id, run_id, record))
        return sorted(found, key=lambda item: item[2].get("enqueuedAt", ""))

    def _api_record(self, course_id: str, run_id: str) -> dict:
        return {**self.runs[course_id][run_id], "runId": run_id}

    def _merge_request(self, course_id: str, patch: dict) -> dict:
        current = dict(self.model_requests.get(course_id) or {})
        for key, value in patch.items():
            if value is None:
                current.pop(key, None)
            else:
                current[key] = value
        self.model_requests[course_id] = current
        self.writes.append(("request", course_id, dict(patch)))
        return current

    def _merge_run(self, course_id: str, run_id: str, patch: dict) -> dict:
        current = dict(self.runs[course_id][run_id])
        for key, value in patch.items():
            if value is None:
                current.pop(key, None)
            else:
                current[key] = value
        self.runs[course_id][run_id] = current
        self.writes.append(("run", f"{course_id}/{run_id}", dict(patch)))
        return current

    # -- transport --------------------------------------------------------- #
    def transport(self, method: str, url: str, body: bytes | None, headers: dict):
        assert url.startswith(API_BASE_URL + queue_module.QUEUE_PREFIX), url
        self.token_seen.append(headers.get(queue_module.WORKER_TOKEN_HEADER, ""))

        rest = url[len(API_BASE_URL) + len(queue_module.QUEUE_PREFIX) :]
        path, _, query = rest.partition("?")
        self.calls.append((method, path + (f"?{query}" if query else "")))
        payload = json.loads(body.decode("utf-8")) if body else None

        def ok(value):
            return queue_module.HttpResponse(
                status=200, headers={}, body=json.dumps(value)
            )

        def error(status: int, detail: str):
            return queue_module.HttpResponse(
                status=status, headers={}, body=json.dumps({"detail": detail})
            )

        if method == "GET" and path == "/pending":
            course_ids = None
            if query:
                course_ids = [urllib.parse.parse_qs(query)["course_id"][0]]
            runs = [
                self._api_record(course_id, run_id)
                for course_id, run_id, _ in self._claimable(course_ids, self.now)
            ]
            return ok({"count": len(runs), "runs": runs})

        if method == "POST" and path == "/claim":
            course_ids = payload.get("courseIds")
            lease = int(payload.get("leaseSeconds") or queue_module.DEFAULT_LEASE_SECONDS)
            candidates = self._claimable(course_ids, self.now)
            if not candidates:
                return ok({"claimed": False, "run": None})
            course_id, run_id, record = candidates[0]
            self._merge_run(
                course_id,
                run_id,
                {
                    "state": "claimed",
                    "attempt": int(record.get("attempt", 0)) + 1,
                    "updatedAt": queue_module.iso(self.now),
                    "claim": {
                        "owner": payload["owner"],
                        "claimedAt": queue_module.iso(self.now),
                        "expiresAt": queue_module.iso(
                            self.now + timedelta(seconds=lease)
                        ),
                    },
                },
            )
            return ok({"claimed": True, "run": self._api_record(course_id, run_id)})

        parts = [segment for segment in path.split("/") if segment]

        if method == "POST" and len(parts) == 3 and parts[0] == "courses" and parts[2] == "model-versions":
            course_id = parts[1]
            version = payload.get("version") or f"v{len(self.model_versions) + 1}"
            self.model_versions.append({"courseId": course_id, **payload, "version": version})
            if payload.get("status") == "ready":
                self._merge_request(
                    course_id,
                    {"status": "ready", "failureMessage": None, "launchError": None},
                )
            return ok(
                {
                    "courseId": course_id,
                    "version": version,
                    "currentVersion": version,
                    "requestStatus": "ready" if payload.get("status") == "ready" else None,
                }
            )

        if method == "POST" and len(parts) == 5 and parts[0] == "courses" and parts[2] == "runs":
            course_id, run_id, action = parts[1], parts[3], parts[4]
            if self.run(course_id, run_id) is None:
                return error(404, f'Training run "{run_id}" was not found.')

            if action == "release":
                patch = {
                    "state": "queued",
                    "updatedAt": queue_module.iso(self.now),
                    "claim": None,
                }
                if payload.get("error") is not None:
                    patch["error"] = payload["error"]
                self._merge_run(course_id, run_id, patch)
                return ok(self._api_record(course_id, run_id))

            if action == "submitted":
                job_id = payload["jobId"]
                submitted_at = queue_module.iso(self.now)
                run = self._merge_run(
                    course_id,
                    run_id,
                    {
                        "state": "submitted",
                        "updatedAt": submitted_at,
                        "jobId": job_id,
                        "trainExamples": int(payload.get("trainExamples", 0)),
                        "validationExamples": int(payload.get("validationExamples", 0)),
                        "claim": None,
                        "error": None,
                    },
                )
                self._merge_request(
                    course_id,
                    {
                        "status": "training",
                        "updatedAt": submitted_at,
                        "currentRunId": run_id,
                        "launchError": None,
                        "training": {
                            "jobId": job_id,
                            "mode": run["mode"],
                            "submittedAt": submitted_at,
                            "datasetRef": run.get("datasetRef", ""),
                            "trainExamples": int(payload.get("trainExamples", 0)),
                            "validationExamples": int(payload.get("validationExamples", 0)),
                        },
                    },
                )
                return ok(
                    {
                        "run": self._api_record(course_id, run_id),
                        "requestStatus": "training",
                    }
                )

            if action == "submission-failed":
                self._merge_run(
                    course_id,
                    run_id,
                    {
                        "state": "queued",
                        "updatedAt": queue_module.iso(self.now),
                        "claim": None,
                        "error": payload["error"],
                    },
                )
                self._merge_request(
                    course_id,
                    {
                        "launchError": payload["error"],
                        "updatedAt": queue_module.iso(self.now),
                        "currentRunId": run_id,
                    },
                )
                return ok(self._api_record(course_id, run_id))

            if action == "failed":
                self._merge_run(
                    course_id,
                    run_id,
                    {
                        "state": "failed",
                        "updatedAt": queue_module.iso(self.now),
                        "claim": None,
                        "error": payload["error"],
                    },
                )
                self._merge_request(
                    course_id,
                    {
                        "status": "failed",
                        "failureMessage": "Training did not finish successfully.",
                        "launchError": payload["error"],
                    },
                )
                return ok(self._api_record(course_id, run_id))

        raise AssertionError(f"Unexpected request {method} {path}")

    #: The clock the fake stamps leases with. Tests move it to exercise expiry.
    now = NOW


def build_queue(fake: FakeQueueApi) -> "queue_module.TrainingQueue":
    return queue_module.TrainingQueue(
        queue_module.TrainingQueueApi(
            base_url=API_BASE_URL,
            worker_token=WORKER_TOKEN,
            transport=fake.transport,
        )
    )


class ParseTests(unittest.TestCase):
    def test_reads_a_stored_run(self) -> None:
        run = queue_module.parse_run("run-1", make_run_record())
        assert run is not None
        self.assertEqual(run.course_id, COURSE_A)
        self.assertEqual(run.state, "queued")
        self.assertEqual(run.train_examples, 38)
        self.assertEqual(run.job_id, "")
        self.assertFalse(run.is_terminal)

    def test_reads_a_persisted_job_id(self) -> None:
        run = queue_module.parse_run("run-1", make_run_record(job_id="9182736"))
        assert run is not None
        self.assertEqual(run.job_id, "9182736")

    def test_rejects_records_that_cannot_be_acted_on(self) -> None:
        for bad in (
            None,
            {},
            make_run_record() | {"state": "banana"},
            make_run_record() | {"mode": "gigantic"},
            make_run_record() | {"courseId": ""},
        ):
            self.assertIsNone(queue_module.parse_run("run-1", bad))

    def test_terminal_runs_are_not_claimable(self) -> None:
        for state in ("succeeded", "failed", "submitted", "training"):
            run = queue_module.parse_run("run-1", make_run_record(state=state))
            assert run is not None
            self.assertFalse(queue_module.is_claimable(run, NOW))

    def test_a_run_with_a_job_id_is_never_claimable(self) -> None:
        run = queue_module.parse_run(
            "run-1", make_run_record(state="queued", job_id="9182736")
        )
        assert run is not None
        self.assertFalse(queue_module.is_claimable(run, NOW))


class SelectionTests(unittest.TestCase):
    def test_oldest_claimable_run_wins(self) -> None:
        runs = queue_module.parse_runs(
            {
                "run-new": make_run_record(enqueued_at="2026-08-12T17:30:00Z"),
                "run-old": make_run_record(enqueued_at="2026-08-12T09:00:00Z"),
                "run-done": make_run_record(
                    state="succeeded", enqueued_at="2026-08-11T09:00:00Z"
                ),
            }
        )
        chosen = queue_module.select_next_run(runs, NOW)
        assert chosen is not None
        self.assertEqual(chosen.run_id, "run-old")


class ClaimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeQueueApi(runs={COURSE_A: {"run-1": make_run_record()}})
        self.queue = build_queue(self.fake)

    def test_claim_writes_a_lease_and_increments_the_attempt(self) -> None:
        run = self.queue.list_runs(COURSE_A)[0]
        claimed = self.queue.claim(COURSE_A, run, owner="alice@tillicum", now=NOW)

        self.assertEqual(claimed.state, "claimed")
        self.assertEqual(claimed.attempt, 1)
        assert claimed.claim is not None
        self.assertEqual(claimed.claim["owner"], "alice@tillicum")
        self.assertEqual(
            claimed.claim["expiresAt"],
            queue_module.iso(NOW + timedelta(seconds=queue_module.DEFAULT_LEASE_SECONDS)),
        )

    def test_claim_is_one_request_scoped_to_the_course(self) -> None:
        """No read-then-write pair to lose a race in.

        The old claim was a conditional PUT that had to be retried on a stale
        tag. This is a single POST; the backend chooses and locks the row.
        """
        run = self.queue.list_runs(COURSE_A)[0]
        self.queue.claim(COURSE_A, run, owner="alice@tillicum", now=NOW)

        claims = [call for call in self.fake.calls if call == ("POST", "/claim")]
        self.assertEqual(len(claims), 1)

        kind, target, patch = self.fake.writes[-1]
        self.assertEqual(kind, "run")
        self.assertEqual(target, f"{COURSE_A}/run-1")
        self.assertEqual(patch["claim"]["owner"], "alice@tillicum")

    def test_second_runner_cannot_take_an_actively_claimed_run(self) -> None:
        """Exactly one worker gets the run; the other is told there is none."""
        run = self.queue.list_runs(COURSE_A)[0]
        self.queue.claim(COURSE_A, run, owner="alice@tillicum", now=NOW)

        # Bob still holds the pre-claim view of the run, as a racing runner would.
        with self.assertRaises(queue_module.ClaimConflict):
            self.queue.claim(
                COURSE_A, run, owner="bob@tillicum", now=NOW + timedelta(seconds=1)
            )

        stored = self.fake.run(COURSE_A, "run-1")
        self.assertEqual(stored["claim"]["owner"], "alice@tillicum")
        self.assertEqual(stored["attempt"], 1)

    def test_two_runners_reading_simultaneously_still_yield_one_claim(self) -> None:
        """Both see the same queued run; only one of them ends up holding it.

        This is the case the old compare-and-set had to detect after the fact
        and retry. Here the losing worker simply gets `claimed: false` on its
        own request, so there is nothing to retry and no second lease.
        """
        alice_view = self.queue.list_runs(COURSE_A)[0]
        bob_view = self.queue.list_runs(COURSE_A)[0]
        self.assertEqual(alice_view.run_id, bob_view.run_id)

        self.queue.claim(COURSE_A, alice_view, owner="alice@tillicum", now=NOW)
        with self.assertRaises(queue_module.ClaimConflict):
            self.queue.claim(COURSE_A, bob_view, owner="bob@tillicum", now=NOW)

        stored = self.fake.run(COURSE_A, "run-1")
        self.assertEqual(stored["claim"]["owner"], "alice@tillicum")
        # One claim, one increment. Two would mean two runners believed they held it.
        self.assertEqual(stored["attempt"], 1)

    def test_expired_lease_can_be_reclaimed(self) -> None:
        """A runner that died mid-job must not strand its run forever."""
        run = self.queue.list_runs(COURSE_A)[0]
        self.queue.claim(
            COURSE_A, run, owner="alice@tillicum", lease_seconds=60, now=NOW
        )

        later = NOW + timedelta(seconds=61)
        self.fake.now = later
        stale = self.queue.list_runs(COURSE_A)[0]
        self.assertTrue(queue_module.is_claimable(stale, later))

        reclaimed = self.queue.claim(COURSE_A, stale, owner="bob@tillicum", now=later)
        self.assertEqual(reclaimed.claim["owner"], "bob@tillicum")
        # The retry is visible rather than silent.
        self.assertEqual(reclaimed.attempt, 2)

    def test_a_live_lease_is_not_offered_to_anyone_else(self) -> None:
        run = self.queue.list_runs(COURSE_A)[0]
        self.queue.claim(
            COURSE_A, run, owner="alice@tillicum", lease_seconds=600, now=NOW
        )

        self.fake.now = NOW + timedelta(seconds=599)
        self.assertEqual(self.queue.list_runs(COURSE_A), [])
        self.assertEqual(
            self.queue.discover_claimable(now=self.fake.now, course_ids=[COURSE_A]), []
        )

    def test_release_clears_the_lease(self) -> None:
        run = self.queue.list_runs(COURSE_A)[0]
        claimed = self.queue.claim(COURSE_A, run, owner="alice@tillicum", now=NOW)
        self.queue.release(COURSE_A, claimed, now=NOW)

        stored = self.fake.run(COURSE_A, "run-1")
        self.assertEqual(stored["state"], "queued")
        self.assertNotIn("claim", stored)


class RecordSubmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeQueueApi(
            runs={COURSE_A: {"run-1": make_run_record()}},
            requests={COURSE_A: make_request_record()},
        )
        self.queue = build_queue(self.fake)

    def test_persists_the_job_id_then_marks_the_request_training(self) -> None:
        run = self.queue.list_runs(COURSE_A)[0]
        claimed = self.queue.claim(COURSE_A, run, owner="alice@tillicum", now=NOW)
        submitted = self.queue.record_submission(
            COURSE_A,
            claimed,
            job_id="9182736",
            train_count=38,
            validation_count=4,
            now=NOW,
        )

        self.assertEqual(submitted.state, "submitted")
        self.assertEqual(submitted.job_id, "9182736")
        stored = self.fake.run(COURSE_A, "run-1")
        self.assertEqual(stored["state"], "submitted")
        self.assertEqual(stored["jobId"], "9182736")
        self.assertNotIn("claim", stored)

        request = self.fake.request(COURSE_A)
        self.assertEqual(request["status"], "training")
        self.assertEqual(request["training"]["jobId"], "9182736")
        self.assertEqual(request["currentRunId"], "run-1")
        self.assertNotIn("launchError", request)
        self.assertNotIn("failureMessage", request)

        run_write = next(
            payload
            for kind, target, payload in self.fake.writes
            if kind == "run" and target.endswith("/run-1") and "jobId" in payload
        )
        request_write = next(
            payload
            for kind, _target, payload in self.fake.writes
            if kind == "request" and payload.get("status") == "training"
        )
        self.assertEqual(run_write["jobId"], "9182736")
        self.assertEqual(request_write["status"], "training")

        # Both landed in one call, so neither can be seen without the other.
        submissions = [
            call
            for call in self.fake.calls
            if call[1].endswith("/submitted")
        ]
        self.assertEqual(len(submissions), 1)

    def test_refuses_to_mark_training_without_a_real_job_id(self) -> None:
        run = self.queue.list_runs(COURSE_A)[0]
        with self.assertRaises(queue_module.TrainingQueueError):
            self.queue.record_submission(
                COURSE_A, run, job_id="", train_count=38, validation_count=4, now=NOW
            )
        request = self.fake.request(COURSE_A)
        self.assertEqual(request["status"], "preparing")
        self.assertNotIn("training", request)

    def test_failure_keeps_the_request_preparing_and_hides_the_error_from_professors(
        self,
    ) -> None:
        run = self.queue.list_runs(COURSE_A)[0]
        claimed = self.queue.claim(COURSE_A, run, owner="alice@tillicum", now=NOW)
        self.queue.record_submission_failure(
            COURSE_A, claimed, error="sbatch: error: Invalid account", now=NOW
        )

        stored = self.fake.run(COURSE_A, "run-1")
        self.assertEqual(stored["state"], "queued")
        self.assertEqual(stored["error"], "sbatch: error: Invalid account")

        request = self.fake.request(COURSE_A)
        self.assertEqual(request["status"], "preparing")
        self.assertEqual(request["launchError"], "sbatch: error: Invalid account")
        self.assertNotIn("failureMessage", request)
        self.assertNotIn("training", request)


class CourseIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeQueueApi(
            runs={
                COURSE_A: {"run-a": make_run_record(COURSE_A)},
                COURSE_B: {"run-b": make_run_record(COURSE_B)},
            }
        )
        self.queue = build_queue(self.fake)

    def test_a_course_only_sees_its_own_runs(self) -> None:
        runs = self.queue.list_runs(COURSE_B)
        self.assertEqual([run.run_id for run in runs], ["run-b"])
        self.assertEqual(runs[0].course_id, COURSE_B)

    def test_limiting_to_one_course_reads_no_other(self) -> None:
        found = self.queue.discover_claimable(now=NOW, course_ids=[COURSE_B])
        self.assertEqual([course for course, _ in found], [COURSE_B])
        for _, path in self.fake.calls:
            self.assertNotIn(COURSE_A, path)

    def test_claiming_one_course_leaves_the_other_untouched(self) -> None:
        run = self.queue.list_runs(COURSE_A)[0]
        self.queue.claim(COURSE_A, run, owner="alice@tillicum", now=NOW)

        untouched = self.fake.run(COURSE_B, "run-b")
        self.assertEqual(untouched, make_run_record(COURSE_B))
        for _, target, _ in self.fake.writes:
            self.assertNotIn(COURSE_B, target)

    def test_a_bad_course_id_never_reaches_a_url(self) -> None:
        for bad in ("", "../etc", "CSS-360", "a/b", "x$y"):
            with self.assertRaises(queue_module.TrainingQueueError):
                queue_module.validate_course_id(bad)

    def test_recording_a_submission_does_not_touch_another_course(self) -> None:
        self.fake.set_request(COURSE_A, make_request_record(COURSE_A, current_run_id="run-a"))
        self.fake.set_request(COURSE_B, make_request_record(COURSE_B, current_run_id="run-b"))
        run = self.queue.list_runs(COURSE_A)[0]
        claimed = self.queue.claim(COURSE_A, run, owner="alice@tillicum", now=NOW)
        self.queue.record_submission(
            COURSE_A, claimed, job_id="1001", train_count=38, validation_count=4, now=NOW
        )

        other_run = self.fake.run(COURSE_B, "run-b")
        other_request = self.fake.request(COURSE_B)
        self.assertEqual(other_run["state"], "queued")
        self.assertNotIn("jobId", other_run)
        self.assertEqual(other_request["status"], "preparing")
        self.assertNotIn("training", other_request)
        for _kind, target, _payload in self.fake.writes:
            self.assertNotIn(COURSE_B, target)


def _prepared_export(root: Path, course_id: str, *, train: int = 38, validation: int = 4) -> None:
    export_dir = root / "data" / "exports" / course_id
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "train.jsonl").write_text(
        "".join(
            json.dumps({"instruction": f"Q{index}", "response": f"A{index}"}) + "\n"
            for index in range(train)
        ),
        encoding="utf-8",
    )
    (export_dir / "validation.jsonl").write_text(
        "".join(
            json.dumps({"instruction": f"V{index}", "response": f"A{index}"}) + "\n"
            for index in range(validation)
        ),
        encoding="utf-8",
    )
    (export_dir / "manifest.json").write_text(
        json.dumps({"courseId": course_id}), encoding="utf-8"
    )


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _prepared_export(self.root, COURSE_A)
        self.fake = FakeQueueApi(
            runs={COURSE_A: {"run-1": make_run_record()}},
            requests={COURSE_A: make_request_record()},
        )
        self.queue = build_queue(self.fake)
        self.launcher = FakeLauncher()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run_once(self, **overrides) -> tuple[int, str]:
        import io
        from contextlib import redirect_stdout

        kwargs = {
            "helpers": helpers,
            "owner": "alice@tillicum",
            "dry_run": False,
            "lease_seconds": 600,
            "course_ids": [COURSE_A],
            "now": NOW,
            "root": self.root,
            "launcher": self.launcher,
        }
        kwargs.update(overrides)
        if kwargs.get("dry_run") and "launcher" not in overrides:
            kwargs["launcher"] = boom_launcher

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = runner.run_once(self.queue, **kwargs)
        return code, buffer.getvalue()

    def _request(self) -> dict:
        return self.fake.request(COURSE_A)

    def _run_record(self) -> dict:
        return self.fake.run(COURSE_A, "run-1")

    def test_successful_submission_invokes_the_existing_launcher(self) -> None:
        code, output = self._run_once()

        self.assertEqual(code, 0)
        self.assertEqual(len(self.launcher.calls), 1)
        argv, cwd = self.launcher.calls[0]
        self.assertEqual(
            argv,
            ["./training/start_qlora_training.sh", "--course", COURSE_A, "--full", "--yes"],
        )
        self.assertEqual(cwd, self.root)
        self.assertIn(
            f"./training/start_qlora_training.sh --course {COURSE_A} --full --yes", output
        )
        self.assertIn(f"qlora-train-{COURSE_A}", output)

    def test_persists_the_real_job_id_and_marks_the_run_submitted(self) -> None:
        code, output = self._run_once()

        self.assertEqual(code, 0)
        stored = self._run_record()
        self.assertEqual(stored["state"], "submitted")
        self.assertEqual(stored["jobId"], "9182736")
        self.assertEqual(stored["attempt"], 1)
        self.assertNotIn("claim", stored)
        self.assertIn("Job ID: 9182736", output)

    def test_request_stays_preparing_until_a_job_id_exists(self) -> None:
        seen = []

        def launcher(command, cwd):
            seen.append(self._request()["status"])
            self.assertNotIn("training", self._request())
            return self.launcher(command, cwd)

        code, _ = self._run_once(launcher=launcher)

        self.assertEqual(code, 0)
        self.assertEqual(seen, ["preparing"])
        request = self._request()
        self.assertEqual(request["status"], "training")
        self.assertEqual(request["training"]["jobId"], "9182736")
        self.assertEqual(request["currentRunId"], "run-1")
        self.assertNotIn("failureMessage", request)
        self.assertNotIn("launchError", request)

    def test_launcher_failure_does_not_mark_training(self) -> None:
        failing = FakeLauncher(
            returncode=1, stdout="", stderr="sbatch: error: Invalid account\n"
        )
        code, output = self._run_once(launcher=failing)

        self.assertEqual(code, 1)
        self.assertEqual(len(failing.calls), 1)
        self.assertIn("Invalid account", output)
        stored = self._run_record()
        self.assertEqual(stored["state"], "queued")
        self.assertNotIn("jobId", stored)
        request = self._request()
        self.assertEqual(request["status"], "preparing")
        self.assertIn("Invalid account", request["launchError"])
        self.assertNotIn("training", request)
        self.assertNotIn("failureMessage", request)

    def test_missing_job_id_does_not_mark_training(self) -> None:
        silent = FakeLauncher(returncode=0, stdout="submitted something, who knows\n")
        code, output = self._run_once(launcher=silent)

        self.assertEqual(code, 1)
        self.assertIn("did not report a Slurm job ID", output)
        stored = self._run_record()
        self.assertEqual(stored["state"], "queued")
        self.assertNotIn("jobId", stored)
        request = self._request()
        self.assertEqual(request["status"], "preparing")
        self.assertNotIn("training", request)
        self.assertNotIn("failureMessage", request)

    def test_duplicate_submission_is_prevented(self) -> None:
        first, _ = self._run_once()
        self.assertEqual(first, 0)
        self.assertEqual(len(self.launcher.calls), 1)

        second, output = self._run_once()
        self.assertEqual(second, 0)
        self.assertEqual(len(self.launcher.calls), 1)
        self.assertIn("No queued training runs.", output)
        self.assertEqual(self._run_record()["jobId"], "9182736")

    def test_a_queued_run_that_already_has_a_job_id_is_not_launched(self) -> None:
        self.fake.set_run(COURSE_A, "run-1", make_run_record(job_id="225122"))
        code, output = self._run_once()

        self.assertEqual(code, 0)
        self.assertEqual(self.launcher.calls, [])
        self.assertIn("No queued training runs.", output)

    def test_submitting_one_course_leaves_another_untouched(self) -> None:
        _prepared_export(self.root, COURSE_B)
        self.fake.set_run(COURSE_B, "run-b", make_run_record(COURSE_B))
        self.fake.set_request(
            COURSE_B, make_request_record(COURSE_B, current_run_id="run-b")
        )

        code, _ = self._run_once(course_ids=[COURSE_A])
        self.assertEqual(code, 0)

        other_run = self.fake.run(COURSE_B, "run-b")
        other_request = self.fake.request(COURSE_B)
        self.assertEqual(other_run["state"], "queued")
        self.assertNotIn("jobId", other_run)
        self.assertEqual(other_request["status"], "preparing")
        self.assertNotIn("training", other_request)
        argv, _ = self.launcher.calls[0]
        self.assertIn(COURSE_A, argv)
        self.assertNotIn(COURSE_B, argv)

    def test_dry_run_writes_nothing_and_spawns_nothing(self) -> None:
        code, output = self._run_once(dry_run=True, launcher=boom_launcher)

        self.assertEqual(code, 0)
        self.assertEqual(self.fake.writes, [])
        self.assertEqual({method for method, _ in self.fake.calls}, {"GET"})
        stored = self._run_record()
        self.assertEqual(stored["state"], "queued")
        self.assertEqual(stored["attempt"], 0)
        self.assertIn("Would run:", output)
        self.assertEqual(self._request()["status"], "preparing")

    def test_the_runner_never_calls_sbatch_itself(self) -> None:
        """Submission goes through start_qlora_training.sh only.

        sbatch, ssh, and rsync are the launcher's job. This process may import
        subprocess to spawn that script; it must not invoke the cluster tools
        directly. Tests never use the real launcher — they inject one.
        """
        import ast

        queue_path = REPO_ROOT / "scripts" / "lib" / "training_queue.py"
        queue_tree = ast.parse(queue_path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(queue_tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertNotIn("subprocess", imported)

        runner_src = (REPO_ROOT / "training" / "run_training_queue.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("start_qlora_training.sh", runner_src)

        code, _ = self._run_once()
        self.assertEqual(code, 0)
        argv, _ = self.launcher.calls[0]
        self.assertEqual(argv[0], "./training/start_qlora_training.sh")
        self.assertNotIn("sbatch", argv)
        self.assertNotIn("ssh", argv)
        self.assertNotIn("rsync", argv)

    def test_refuses_and_releases_when_the_dataset_is_missing(self) -> None:
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        code, output = self._run_once(root=empty, launcher=boom_launcher)

        self.assertEqual(code, 1)
        self.assertIn("No prepared training data", output)
        stored = self._run_record()
        self.assertEqual(stored["state"], "queued")
        self.assertIn("No prepared training data", stored["error"])
        self.assertEqual(self._request()["status"], "preparing")

    def test_dry_run_refuses_a_missing_dataset_without_writing(self) -> None:
        empty = Path(self.tmp.name) / "empty-dry"
        empty.mkdir()
        code, _ = self._run_once(dry_run=True, root=empty, launcher=boom_launcher)

        self.assertEqual(code, 1)
        self.assertEqual(self.fake.writes, [])

    def test_reports_a_count_that_disagrees_with_the_prepared_data(self) -> None:
        self.fake.set_run(COURSE_A, "run-1", make_run_record(train=999))
        _, output = self._run_once()
        self.assertIn("999", output)
        self.assertIn("Warning:", output)
        self.assertEqual(self._run_record()["jobId"], "9182736")

    def test_says_so_when_the_queue_is_empty(self) -> None:
        self.fake.runs[COURSE_A] = {}
        code, output = self._run_once(launcher=boom_launcher)

        self.assertEqual(code, 0)
        self.assertIn("No queued training runs.", output)
        self.assertEqual(self.fake.writes, [])

    def test_a_run_held_by_another_runner_is_left_alone(self) -> None:
        self.fake.set_run(
            COURSE_A,
            "run-1",
            make_run_record(
                state="claimed",
                claim={
                    "owner": "bob@tillicum",
                    "claimedAt": queue_module.iso(NOW),
                    "expiresAt": queue_module.iso(NOW + timedelta(seconds=600)),
                },
            ),
        )
        code, output = self._run_once(launcher=boom_launcher)

        self.assertEqual(code, 0)
        self.assertIn("No queued training runs.", output)
        self.assertEqual(self.fake.writes, [])

    def test_a_smoke_run_reports_the_smoke_command(self) -> None:
        self.fake.set_run(COURSE_A, "run-1", make_run_record(mode="smoke"))
        _, output = self._run_once()
        self.assertIn(f"--course {COURSE_A} --smoke --yes", output)
        self.assertIn(f"qlora-smoke-{COURSE_A}", output)
        argv, _ = self.launcher.calls[0]
        self.assertIn("--smoke", argv)

    def test_existing_active_job_output_still_captures_a_real_job_id(self) -> None:
        existing = FakeLauncher(
            returncode=0,
            stdout=(
                "Existing active qlora-train-css-490-spring-2026-cgvl job found; "
                "will not submit another.\n"
                "Job ID: 225122\n"
            ),
        )
        code, output = self._run_once(launcher=existing)

        self.assertEqual(code, 0)
        self.assertEqual(self._run_record()["jobId"], "225122")
        self.assertEqual(self._request()["status"], "training")
        self.assertIn("Job ID: 225122", output)

    def test_parse_launcher_job_id_reuses_the_existing_helper(self) -> None:
        self.assertEqual(
            runner.parse_launcher_job_id("Submitted batch job 216829\n"),
            "216829",
        )
        self.assertEqual(runner.parse_launcher_job_id("Job ID: 225122\n"), "225122")
        self.assertIsNone(runner.parse_launcher_job_id("nope\n"))


class WorkerConfigurationTests(unittest.TestCase):
    """What the runner needs on Tillicum, and what it must no longer need.

    The migration's promise on this side is negative as much as positive: the
    worker gets a URL and a token, and nothing about Firebase remains — no
    database URL, no auth token, no REST paths, no module to install.
    """

    def setUp(self) -> None:
        self._saved = {
            name: os.environ.get(name)
            for name in (
                "TRAINING_API_BASE_URL",
                "VITE_API_BASE_URL",
                "TRAINING_WORKER_TOKEN",
            )
        }
        for name in self._saved:
            os.environ.pop(name, None)

    def tearDown(self) -> None:
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_no_firebase_variable_is_read_anywhere_in_the_worker(self) -> None:
        for path in TILLICUM_RUNTIME_FILES + (
            REPO_ROOT / "scripts" / "register_course_model.py",
        ):
            source = path.read_text(encoding="utf-8").lower()
            with self.subTest(path=path.name):
                self.assertNotIn("firebase", source)

    def test_the_worker_starts_from_a_url_and_a_token(self) -> None:
        os.environ["TRAINING_API_BASE_URL"] = API_BASE_URL
        os.environ["TRAINING_WORKER_TOKEN"] = WORKER_TOKEN

        self.assertEqual(queue_module.training_api_base_url(), API_BASE_URL)
        self.assertEqual(queue_module.training_worker_token(), WORKER_TOKEN)

    def test_a_missing_url_says_what_to_set(self) -> None:
        with self.assertRaises(queue_module.TrainingQueueError) as caught:
            queue_module.training_api_base_url()
        self.assertIn("TRAINING_API_BASE_URL", str(caught.exception))

    def test_a_missing_token_says_what_to_set(self) -> None:
        with self.assertRaises(queue_module.TrainingQueueError) as caught:
            queue_module.training_worker_token()
        self.assertIn("TRAINING_WORKER_TOKEN", str(caught.exception))

    def test_every_request_carries_the_worker_token(self) -> None:
        fake = FakeQueueApi(runs={COURSE_A: {"run-1": make_run_record()}})
        queue = build_queue(fake)
        run = queue.list_runs(COURSE_A)[0]
        queue.claim(COURSE_A, run, owner="alice@tillicum", now=NOW)

        self.assertTrue(fake.token_seen)
        self.assertEqual(set(fake.token_seen), {WORKER_TOKEN})

    def test_a_rejected_token_is_reported_clearly(self) -> None:
        def refusing(method, url, body, headers):
            return queue_module.HttpResponse(
                status=401, headers={}, body=json.dumps({"detail": "nope"})
            )

        queue = queue_module.TrainingQueue(
            queue_module.TrainingQueueApi(
                base_url=API_BASE_URL,
                worker_token="wrong",
                transport=refusing,
            )
        )
        with self.assertRaises(queue_module.TrainingQueueError) as caught:
            queue.list_runs(COURSE_A)
        self.assertIn("TRAINING_WORKER_TOKEN", str(caught.exception))


class TrainingFailureAndRegistrationTests(unittest.TestCase):
    """What happens after the launcher: a failed job, or a finished model."""

    def setUp(self) -> None:
        self.fake = FakeQueueApi(
            runs={COURSE_A: {"run-1": make_run_record(state="training", job_id="9182736")}},
            requests={COURSE_A: make_request_record(status="training")},
        )
        self.queue = build_queue(self.fake)

    def test_failed_training_is_terminal_and_visible_to_the_professor(self) -> None:
        run = queue_module.parse_run("run-1", self.fake.run(COURSE_A, "run-1"))
        assert run is not None
        failed = self.queue.record_training_failure(
            COURSE_A, run, error="CUDA out of memory"
        )

        self.assertEqual(failed.state, "failed")
        stored = self.fake.run(COURSE_A, "run-1")
        self.assertEqual(stored["error"], "CUDA out of memory")

        request = self.fake.request(COURSE_A)
        self.assertEqual(request["status"], "failed")
        # Professor-facing text is not the raw operator error.
        self.assertIn("failureMessage", request)
        self.assertNotIn("CUDA", request["failureMessage"])
        self.assertEqual(request["launchError"], "CUDA out of memory")

    def test_a_failed_run_frees_the_course_for_another_run(self) -> None:
        run = queue_module.parse_run("run-1", self.fake.run(COURSE_A, "run-1"))
        assert run is not None
        failed = self.queue.record_training_failure(COURSE_A, run, error="died")
        self.assertTrue(failed.is_terminal)

    def test_registering_a_model_marks_the_request_ready(self) -> None:
        result = self.queue.register_model_version(
            COURSE_A,
            base_model="meta-llama/Llama-3.2-3B-Instruct",
            training_example_count=42,
            artifact_ref="css-490-qlora/adapter",
            run_id="run-1",
        )

        self.assertEqual(result["courseId"], COURSE_A)
        self.assertEqual(result["version"], "v1")
        self.assertEqual(self.fake.request(COURSE_A)["status"], "ready")

    def test_a_model_is_registered_only_against_its_own_course(self) -> None:
        self.queue.register_model_version(
            COURSE_A,
            base_model="meta-llama/Llama-3.2-3B-Instruct",
            training_example_count=42,
            artifact_ref="css-490-qlora/adapter",
        )
        self.assertEqual(
            [entry["courseId"] for entry in self.fake.model_versions], [COURSE_A]
        )
        self.assertIsNone(self.fake.request(COURSE_B))

    def test_a_bad_course_id_never_reaches_a_registration_url(self) -> None:
        for bad in ("", "../etc", "CSS-360", "a/b"):
            with self.assertRaises(queue_module.TrainingQueueError):
                self.queue.register_model_version(
                    bad,
                    base_model="m",
                    training_example_count=1,
                    artifact_ref="x/adapter",
                )


class CliTests(unittest.TestCase):
    def test_once_is_required(self) -> None:
        self.assertEqual(runner.main([]), 2)

    def test_lease_must_be_positive(self) -> None:
        self.assertEqual(runner.main(["--once", "--lease-seconds", "0"]), 2)


TILLICUM_RUNTIME_FILES = (
    REPO_ROOT / "training" / "run_training_queue.py",
    REPO_ROOT / "scripts" / "lib" / "training_queue.py",
)


class Python39RuntimeCompatibilityTests(unittest.TestCase):
    """Tillicum login nodes run Python 3.9; these files must import there."""

    def test_runtime_files_do_not_use_pep604_unions(self) -> None:
        """Reject `T | None` (and any `X | Y`) in Tillicum runtime sources.

        `from __future__ import annotations` postpones function annotations, but
        type aliases like `Transport = Callable[..., bytes | None, ...]` are still
        evaluated at import and crash on 3.9. Keep Optional[...] instead.
        """
        import ast
        import re

        pattern = re.compile(r"\|\s*None\b")
        for path in TILLICUM_RUNTIME_FILES:
            source = path.read_text(encoding="utf-8")
            self.assertIsNone(
                pattern.search(source),
                f"{path.relative_to(REPO_ROOT)} reintroduced PEP 604 `T | None`; "
                "use Optional[T] for Python 3.9",
            )
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
                    self.fail(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno} uses `|` "
                        "(PEP 604 unions are Python 3.10+). Use Optional[...] instead."
                    )

    def test_runtime_files_have_no_other_post39_syntax(self) -> None:
        import ast

        match_cls = getattr(ast, "Match", ())
        try_star_cls = getattr(ast, "TryStar", ())
        type_alias_cls = getattr(ast, "TypeAlias", ())

        for path in TILLICUM_RUNTIME_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if match_cls and isinstance(node, match_cls):
                    self.fail(
                        f"{path.name}:{node.lineno} uses match/case (Python 3.10+)"
                    )
                if try_star_cls and isinstance(node, try_star_cls):
                    self.fail(
                        f"{path.name}:{node.lineno} uses except* (Python 3.11+)"
                    )
                if type_alias_cls and isinstance(node, type_alias_cls):
                    self.fail(
                        f"{path.name}:{node.lineno} uses a type statement (Python 3.12+)"
                    )
                if isinstance(node, ast.Call):
                    for keyword in node.keywords:
                        if keyword.arg == "strict":
                            func = node.func
                            name = getattr(func, "id", getattr(func, "attr", ""))
                            if name == "zip":
                                self.fail(
                                    f"{path.name}:{node.lineno} uses zip(..., strict=) "
                                    "(Python 3.10+)"
                                )

    def test_type_aliases_evaluate_on_python39_builtins(self) -> None:
        """Type aliases are evaluated at import; they must not use `X | Y`."""
        import ast

        path = REPO_ROOT / "scripts" / "lib" / "training_queue.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if any(
                isinstance(child, ast.BinOp) and isinstance(child.op, ast.BitOr)
                for child in ast.walk(node)
            ):
                self.fail(
                    f"training_queue.py:{node.lineno} type alias uses PEP 604 `|`; "
                    "that is evaluated at import on Tillicum's Python 3.9."
                )


if __name__ == "__main__":
    unittest.main()
