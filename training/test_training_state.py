"""The cluster's own durable record of what it did and what it still owes.

Two real failures made this necessary.

A job finished and the application never found out, because the only thing that
could have told it was a person running a command. So a completion is persisted
before the first send attempt and removed only once the backend has accepted it:
anything left in `pending/` is an event UWB does not know about yet.

And an ambiguous network failure around submission: `sbatch` succeeded, the
`/submitted` call timed out, and the cluster now had a real job the application
had never heard of. Re-running the worker must re-report rather than submit
again — a duplicate there is a second GPU allocation training the same adapter.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "lib"))

import training_state as state  # noqa: E402

COURSE = "css-350-spring-2026-n3h9"
RUN_ID = "run-20260827t064701z-1cf650"
OTHER_RUN_ID = "run-20260828t090000z-aa11bb"


class TrainingStateTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)


class RunRecordTests(TrainingStateTestCase):
    def test_a_run_record_holds_the_three_identifiers_reconciliation_needs(self) -> None:
        """runId, Slurm jobId, output directory — in one file, written locally.

        These are what an operator needs to work out what happened when a report
        did not land, and they have to exist independently of whether it landed.
        """
        state.write_run_record(
            self.root,
            run_id=RUN_ID,
            course_id=COURSE,
            mode="full",
            job_id="264787",
            output_dir="/gpfs/.../qlora-runs/css-350/x-full",
        )

        record = state.read_run_record(self.root, RUN_ID)
        assert record is not None
        self.assertEqual(record["runId"], RUN_ID)
        self.assertEqual(record["courseId"], COURSE)
        self.assertEqual(record["jobId"], "264787")
        self.assertEqual(record["outputDir"], "/gpfs/.../qlora-runs/css-350/x-full")

    def test_writing_twice_merges_rather_than_replaces(self) -> None:
        """The record is built in stages and a later stage must not erase an earlier.

        The worker writes the run before `sbatch`, the job id the moment it
        returns, and `reported` only once the backend has acknowledged it.
        """
        state.write_run_record(
            self.root,
            run_id=RUN_ID,
            course_id=COURSE,
            mode="full",
            output_dir="/gpfs/out",
            train_examples=37,
        )
        state.write_run_record(
            self.root, run_id=RUN_ID, course_id=COURSE, mode="full", job_id="264787"
        )

        record = state.read_run_record(self.root, RUN_ID)
        assert record is not None
        self.assertEqual(record["outputDir"], "/gpfs/out")
        self.assertEqual(record["trainExamples"], 37)
        self.assertEqual(record["jobId"], "264787")

    def test_reported_can_go_back_to_false_for_a_retried_run(self) -> None:
        state.write_run_record(
            self.root, run_id=RUN_ID, course_id=COURSE, mode="full", reported=True
        )
        state.write_run_record(
            self.root, run_id=RUN_ID, course_id=COURSE, mode="full", reported=False
        )

        record = state.read_run_record(self.root, RUN_ID)
        assert record is not None
        self.assertFalse(record["reported"])

    def test_unreported_submissions_are_exactly_the_recoverable_ones(self) -> None:
        """A job exists and the application does not know: that and nothing else."""
        state.write_run_record(
            self.root, run_id=RUN_ID, course_id=COURSE, mode="full",
            job_id="264787", reported=False,
        )
        state.write_run_record(
            self.root, run_id=OTHER_RUN_ID, course_id=COURSE, mode="full",
            job_id="264788", reported=True,
        )
        state.write_run_record(
            self.root, run_id="run-20260829t000000z-cccccc", course_id=COURSE,
            mode="full", reported=False,
        )

        outstanding = state.unreported_submissions(self.root)

        self.assertEqual([item["runId"] for item in outstanding], [RUN_ID])

    def test_no_state_directory_is_not_an_error(self) -> None:
        """A fresh checkout has none, and that is the normal first run."""
        self.assertEqual(state.list_run_records(self.root), [])
        self.assertEqual(state.unreported_submissions(self.root), [])
        self.assertIsNone(state.read_run_record(self.root, RUN_ID))

    def test_a_run_id_that_is_not_a_plain_identifier_is_refused(self) -> None:
        """These values become file names, so they are checked wherever they came from."""
        for bad in ("../escape", "run/../..", "", "a" * 200):
            with self.subTest(run_id=bad):
                with self.assertRaises(state.TrainingStateError):
                    state.run_record_path(self.root, bad)


class PendingCallbackTests(TrainingStateTestCase):
    def _queue(self, run_id: str = RUN_ID, outcome: str = "succeeded") -> Path:
        return state.queue_pending_callback(
            self.root,
            run_id=run_id,
            course_id=COURSE,
            kind="completed",
            payload={"outcome": outcome, "jobId": "264787"},
        )

    def test_a_queued_callback_survives_on_disk(self) -> None:
        self._queue()

        pending = state.list_pending_callbacks(self.root)

        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["runId"], RUN_ID)
        self.assertEqual(pending[0]["courseId"], COURSE)
        self.assertEqual(pending[0]["payload"]["outcome"], "succeeded")

    def test_clearing_removes_it(self) -> None:
        self._queue()

        self.assertTrue(state.clear_pending_callback(self.root, RUN_ID, "completed"))
        self.assertEqual(state.list_pending_callbacks(self.root), [])

    def test_clearing_something_already_gone_is_not_an_error(self) -> None:
        """Delivery can be reported more than once; the second clear is a no-op."""
        self.assertFalse(state.clear_pending_callback(self.root, RUN_ID, "completed"))

    def test_a_second_report_for_one_run_supersedes_the_first(self) -> None:
        """There is only one true answer to how a given run ended.

        Queueing behind the first would mean replaying a stale outcome after the
        corrected one.
        """
        self._queue(outcome="succeeded")
        self._queue(outcome="failed")

        pending = state.list_pending_callbacks(self.root)

        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["payload"]["outcome"], "failed")

    def test_callbacks_for_different_runs_coexist(self) -> None:
        self._queue(RUN_ID)
        self._queue(OTHER_RUN_ID)

        self.assertEqual(len(state.list_pending_callbacks(self.root)), 2)

    def test_an_unreadable_file_is_skipped_rather_than_fatal(self) -> None:
        """One corrupt payload must not stop every other one being delivered."""
        self._queue()
        broken = state.pending_dir(self.root) / "broken-completed.json"
        broken.write_text("{not json", encoding="utf-8")

        pending = state.list_pending_callbacks(self.root)

        self.assertEqual([item["runId"] for item in pending], [RUN_ID])

    def test_an_unknown_callback_kind_is_refused(self) -> None:
        with self.assertRaises(state.TrainingStateError):
            state.queue_pending_callback(
                self.root, run_id=RUN_ID, course_id=COURSE, kind="whatever", payload={}
            )


class AtomicWriteTests(TrainingStateTestCase):
    def test_a_written_file_is_complete_json(self) -> None:
        path = self.root / "state" / "example.json"
        state.write_json_atomic(path, {"a": 1, "b": [2, 3]})

        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 1, "b": [2, 3]})

    def test_no_temporary_files_are_left_behind(self) -> None:
        """A pending directory full of `.tmp` fragments is a directory nobody trusts."""
        path = self.root / "state" / "example.json"
        state.write_json_atomic(path, {"a": 1})
        state.write_json_atomic(path, {"a": 2})

        leftovers = [item.name for item in path.parent.iterdir() if item.name != "example.json"]
        self.assertEqual(leftovers, [])

    def test_a_rewrite_replaces_the_content_wholesale(self) -> None:
        path = self.root / "state" / "example.json"
        state.write_json_atomic(path, {"long": "x" * 500})
        state.write_json_atomic(path, {"short": 1})

        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"short": 1})


if __name__ == "__main__":
    unittest.main()
