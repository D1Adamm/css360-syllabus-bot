"""Unit tests for the durable training queue and the Tillicum runner.

Nothing here touches the network, Slurm, or the cluster: the HTTP transport is
injected, so every request the queue would make is inspected instead of sent.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
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

DATABASE_URL = "https://example-default-rtdb.firebaseio.com"


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
    return record


class FakeFirebase:
    """An in-memory database that behaves like the REST API's ETag contract.

    The ETag changes on every write, which is exactly the property a claim
    depends on: a runner holding a stale tag must be refused.
    """

    def __init__(self, tree: dict | None = None) -> None:
        self.tree: dict = tree or {}
        self.versions: dict[str, int] = {}
        self.requests: list[tuple[str, str]] = []
        self.writes: list[tuple[str, str, object]] = []

    # -- path helpers ----------------------------------------------------- #
    def _path_from_url(self, url: str) -> tuple[str, str]:
        without_base = url[len(DATABASE_URL) + 1 :]
        path, _, query = without_base.partition("?")
        return path[: -len(".json")], query

    def _read(self, path: str):
        node = self.tree
        for part in [segment for segment in path.split("/") if segment]:
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    def _write(self, path: str, value) -> None:
        parts = [segment for segment in path.split("/") if segment]
        node = self.tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        if value is None:
            node.pop(parts[-1], None)
        else:
            node[parts[-1]] = value
        self.versions[path] = self.versions.get(path, 0) + 1

    def etag(self, path: str) -> str:
        return f'"v{self.versions.get(path, 0)}"'

    # -- transport -------------------------------------------------------- #
    def transport(self, method: str, url: str, body: bytes | None, headers: dict):
        path, query = self._path_from_url(url)
        self.requests.append((method, path))

        if method == "GET":
            value = self._read(path)
            if query.startswith("shallow=true") and isinstance(value, dict):
                value = {key: True for key in value}
            response_headers = {}
            if headers.get("X-Firebase-ETag") == "true":
                response_headers["etag"] = self.etag(path)
            return queue_module.HttpResponse(
                status=200,
                headers=response_headers,
                body=json.dumps(value),
            )

        payload = json.loads(body.decode("utf-8")) if body else None

        if method == "PUT":
            expected = headers.get("if-match")
            if expected is not None and expected != self.etag(path):
                return queue_module.HttpResponse(status=412, headers={}, body="null")
            self._write(path, payload)
            self.writes.append((method, path, payload))
            return queue_module.HttpResponse(
                status=200, headers={}, body=json.dumps(payload)
            )

        if method == "PATCH":
            current = self._read(path)
            merged = dict(current) if isinstance(current, dict) else {}
            for key, value in (payload or {}).items():
                if value is None:
                    merged.pop(key, None)
                else:
                    merged[key] = value
            self._write(path, merged)
            self.writes.append((method, path, payload))
            return queue_module.HttpResponse(
                status=200, headers={}, body=json.dumps(payload)
            )

        raise AssertionError(f"Unexpected method {method}")


def build_queue(fake: FakeFirebase) -> "queue_module.TrainingQueue":
    return queue_module.TrainingQueue(
        queue_module.FirebaseRest(
            database_url=DATABASE_URL,
            auth_token=None,
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
        self.assertFalse(run.is_terminal)

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
        self.fake = FakeFirebase(
            {"courses": {COURSE_A: {"trainingRuns": {"run-1": make_run_record()}}}}
        )
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

    def test_claim_is_a_conditional_write(self) -> None:
        run = self.queue.list_runs(COURSE_A)[0]
        self.queue.claim(COURSE_A, run, owner="alice@tillicum", now=NOW)

        method, path, _ = self.fake.writes[-1]
        self.assertEqual(method, "PUT")
        self.assertEqual(path, f"courses/{COURSE_A}/trainingRuns/run-1")

    def test_second_runner_cannot_take_an_actively_claimed_run(self) -> None:
        run = self.queue.list_runs(COURSE_A)[0]
        self.queue.claim(COURSE_A, run, owner="alice@tillicum", now=NOW)

        # Bob still holds the pre-claim view of the run, as a racing runner would.
        with self.assertRaises(queue_module.ClaimConflict):
            self.queue.claim(COURSE_A, run, owner="bob@tillicum", now=NOW + timedelta(seconds=1))

        stored = self.fake._read(f"courses/{COURSE_A}/trainingRuns/run-1")
        self.assertEqual(stored["claim"]["owner"], "alice@tillicum")
        self.assertEqual(stored["attempt"], 1)

    def test_a_stale_etag_loses_the_race(self) -> None:
        """Two runners reading simultaneously: only the first write may land."""
        path = f"courses/{COURSE_A}/trainingRuns/run-1"
        client = self.queue.client
        _, alice_etag = client.get_with_etag(path)
        _, bob_etag = client.get_with_etag(path)
        self.assertEqual(alice_etag, bob_etag)

        client.put_if_match(path, make_run_record(state="claimed"), alice_etag)
        with self.assertRaises(queue_module.ClaimConflict):
            client.put_if_match(path, make_run_record(state="claimed"), bob_etag)

    def test_expired_lease_can_be_reclaimed(self) -> None:
        run = self.queue.list_runs(COURSE_A)[0]
        self.queue.claim(COURSE_A, run, owner="alice@tillicum", lease_seconds=60, now=NOW)

        later = NOW + timedelta(seconds=61)
        stale = self.queue.list_runs(COURSE_A)[0]
        self.assertTrue(queue_module.is_claimable(stale, later))

        reclaimed = self.queue.claim(COURSE_A, stale, owner="bob@tillicum", now=later)
        self.assertEqual(reclaimed.claim["owner"], "bob@tillicum")
        # The retry is visible rather than silent.
        self.assertEqual(reclaimed.attempt, 2)

    def test_a_live_lease_is_not_claimable(self) -> None:
        run = self.queue.list_runs(COURSE_A)[0]
        self.queue.claim(COURSE_A, run, owner="alice@tillicum", lease_seconds=600, now=NOW)

        held = self.queue.list_runs(COURSE_A)[0]
        self.assertFalse(queue_module.is_claimable(held, NOW + timedelta(seconds=599)))

    def test_release_clears_the_lease(self) -> None:
        run = self.queue.list_runs(COURSE_A)[0]
        claimed = self.queue.claim(COURSE_A, run, owner="alice@tillicum", now=NOW)
        self.queue.release(COURSE_A, claimed, now=NOW)

        stored = self.fake._read(f"courses/{COURSE_A}/trainingRuns/run-1")
        self.assertEqual(stored["state"], "queued")
        self.assertNotIn("claim", stored)


class CourseIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeFirebase(
            {
                "courses": {
                    COURSE_A: {"trainingRuns": {"run-a": make_run_record(COURSE_A)}},
                    COURSE_B: {"trainingRuns": {"run-b": make_run_record(COURSE_B)}},
                }
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
        for _, path in self.fake.requests:
            self.assertNotIn(COURSE_A, path)

    def test_claiming_one_course_leaves_the_other_untouched(self) -> None:
        run = self.queue.list_runs(COURSE_A)[0]
        self.queue.claim(COURSE_A, run, owner="alice@tillicum", now=NOW)

        untouched = self.fake._read(f"courses/{COURSE_B}/trainingRuns/run-b")
        self.assertEqual(untouched, make_run_record(COURSE_B))
        for _, path, _ in self.fake.writes:
            self.assertNotIn(COURSE_B, path)

    def test_a_bad_course_id_never_reaches_a_path(self) -> None:
        for bad in ("", "../etc", "CSS-360", "a/b", "x$y"):
            with self.assertRaises(queue_module.TrainingQueueError):
                queue_module.course_training_runs_path(bad)


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
        self.fake = FakeFirebase(
            {"courses": {COURSE_A: {"trainingRuns": {"run-1": make_run_record()}}}}
        )
        self.queue = build_queue(self.fake)

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
        }
        kwargs.update(overrides)

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = runner.run_once(self.queue, **kwargs)
        return code, buffer.getvalue()

    def test_reports_the_command_the_existing_launcher_would_be_given(self) -> None:
        code, output = self._run_once()

        self.assertEqual(code, 0)
        self.assertIn(
            f"./training/start_qlora_training.sh --course {COURSE_A} --full --yes", output
        )
        # The job name comes from the launcher's own helper, per course.
        self.assertIn(f"qlora-train-{COURSE_A}", output)
        self.assertIn("38 train / 4 validation", output)

    def test_claims_the_run_and_releases_it_because_nothing_was_submitted(self) -> None:
        code, output = self._run_once()

        self.assertEqual(code, 0)
        methods = [method for method, _, _ in self.fake.writes]
        self.assertEqual(methods, ["PUT", "PATCH"])
        stored = self.fake._read(f"courses/{COURSE_A}/trainingRuns/run-1")
        self.assertEqual(stored["state"], "queued")
        self.assertEqual(stored["attempt"], 1)
        self.assertIn("nothing was submitted", output)

    def test_dry_run_writes_nothing_at_all(self) -> None:
        code, output = self._run_once(dry_run=True)

        self.assertEqual(code, 0)
        self.assertEqual(self.fake.writes, [])
        self.assertEqual({method for method, _ in self.fake.requests}, {"GET"})
        stored = self.fake._read(f"courses/{COURSE_A}/trainingRuns/run-1")
        self.assertEqual(stored["state"], "queued")
        self.assertEqual(stored["attempt"], 0)
        self.assertIn("Would run:", output)

    def test_the_runner_cannot_submit_or_copy_anything(self) -> None:
        """No path here shells out — the source itself has no way to.

        The launch command is produced as a string and printed. If a future
        edit gave this module a way to start a process — sbatch, ssh, rsync or
        anything else — this test is what notices.
        """
        import ast

        for path in (
            REPO_ROOT / "training" / "run_training_queue.py",
            REPO_ROOT / "scripts" / "lib" / "training_queue.py",
        ):
            tree = ast.parse(path.read_text(encoding="utf-8"))

            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])

            for forbidden in ("subprocess", "pty", "multiprocessing", "asyncio"):
                self.assertNotIn(forbidden, imported, f"{path.name} imports {forbidden}")

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                    self.assertNotIn(
                        name,
                        ("system", "popen", "spawn", "spawnv", "execv", "execvp", "run"),
                        f"{path.name} calls {name}()",
                    )

        self.assertFalse(hasattr(runner, "subprocess"))
        # The command is only ever a string handed to print().
        plan = runner.describe_planned_launch(
            queue_module.parse_run("run-1", make_run_record()), helpers=helpers
        )
        self.assertIsInstance(plan["command"], str)

    def test_refuses_and_releases_when_the_dataset_is_missing(self) -> None:
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        code, output = self._run_once(root=empty)

        self.assertEqual(code, 1)
        self.assertIn("No prepared training data", output)
        stored = self.fake._read(f"courses/{COURSE_A}/trainingRuns/run-1")
        self.assertEqual(stored["state"], "queued")
        self.assertIn("No prepared training data", stored["error"])

    def test_dry_run_refuses_a_missing_dataset_without_writing(self) -> None:
        empty = Path(self.tmp.name) / "empty-dry"
        empty.mkdir()
        code, _ = self._run_once(dry_run=True, root=empty)

        self.assertEqual(code, 1)
        self.assertEqual(self.fake.writes, [])

    def test_reports_a_count_that_disagrees_with_the_prepared_data(self) -> None:
        self.fake._write(
            f"courses/{COURSE_A}/trainingRuns/run-1", make_run_record(train=999)
        )
        _, output = self._run_once()
        self.assertIn("999", output)
        self.assertIn("Warning:", output)

    def test_says_so_when_the_queue_is_empty(self) -> None:
        self.fake._write(f"courses/{COURSE_A}/trainingRuns", {})
        code, output = self._run_once()

        self.assertEqual(code, 0)
        self.assertIn("No queued training runs.", output)
        self.assertEqual(self.fake.writes, [])

    def test_a_run_held_by_another_runner_is_left_alone(self) -> None:
        self.fake._write(
            f"courses/{COURSE_A}/trainingRuns/run-1",
            make_run_record(
                state="claimed",
                claim={
                    "owner": "bob@tillicum",
                    "claimedAt": queue_module.iso(NOW),
                    "expiresAt": queue_module.iso(NOW + timedelta(seconds=600)),
                },
            ),
        )
        code, output = self._run_once()

        self.assertEqual(code, 0)
        self.assertIn("No queued training runs.", output)
        self.assertEqual(self.fake.writes, [])

    def test_a_smoke_run_reports_the_smoke_command(self) -> None:
        self.fake._write(
            f"courses/{COURSE_A}/trainingRuns/run-1", make_run_record(mode="smoke")
        )
        _, output = self._run_once()
        self.assertIn(f"--course {COURSE_A} --smoke --yes", output)
        self.assertIn(f"qlora-smoke-{COURSE_A}", output)


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
