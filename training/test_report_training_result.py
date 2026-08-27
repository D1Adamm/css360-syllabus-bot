"""The job reporting its own ending, including when the network is not there.

Job 253552 finished successfully and the application never found out. The fix is
that the job reports itself — but a job that ends at 02:00 against a briefly
unreachable backend must not lose the event when it gives up retrying, and must
not turn a good training run into a failed one because a network call did not go
through.

So: build the payload, persist it, try to send it, delete the persisted copy
only on acceptance. Everything here is about that ordering holding under the
failures it exists for.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "lib"))

import training_state as state  # noqa: E402
from training_queue import TrainingQueueError  # noqa: E402


def _load_reporter():
    path = REPO_ROOT / "training" / "report_training_result.py"
    spec = importlib.util.spec_from_file_location("report_training_result", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reporter = _load_reporter()

COURSE = "css-350-spring-2026-n3h9"
RUN_ID = "run-20260827t064701z-1cf650"
JOB_ID = "264787"

RUNTIME_REPORT = {
    "mode": "full",
    "modelId": "meta-llama/Llama-3.2-3B-Instruct",
    "gpuCount": 1,
    "trainExampleCount": 37,
    "validationExampleCount": 5,
    "epochs": 3.0,
    "intendedOptimizerSteps": 15,
    "completedSteps": 15,
    "missingOptimizerSteps": 0,
    "trainingLengthSatisfied": True,
    "totalElapsedSeconds": 48.2,
    "actualGpuHours": 0.0134,
    "gitCommitSha": "9941833cafe0000000000000000000000000beef",
    "slurmJobId": JOB_ID,
}


class RecordingQueue:
    """A stand-in for `TrainingQueue` that records or refuses completions."""

    def __init__(self, *, error: Exception | None = None, result: dict | None = None):
        self.error = error
        self.result = result or {
            "runState": "succeeded",
            "requestStatus": "ready",
            "version": "v1",
            "registered": True,
            "alreadyRegistered": False,
        }
        self.calls: list[tuple[str, str, dict]] = []

    def record_completion(self, course_id: str, run_id: str, payload: dict) -> dict:
        self.calls.append((course_id, run_id, payload))
        if self.error is not None:
            raise self.error
        return self.result


class ReporterTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output_dir = (
            self.root / "training_outputs" / "qlora-runs" / COURSE / "20260827T064701Z-full"
        )
        self.export_dir = self.root / "data" / "exports" / COURSE
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def write_run_output(self, *, adapter: bool = True, runtime: dict | None = None) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "runtime-report.json").write_text(
            json.dumps(runtime if runtime is not None else RUNTIME_REPORT),
            encoding="utf-8",
        )
        (self.output_dir / "training_metrics.json").write_text(
            json.dumps({"train_loss": 1.2345, "epoch": 3.0}), encoding="utf-8"
        )
        (self.output_dir / "evaluation_metrics.json").write_text(
            json.dumps({"eval_loss": 1.4567}), encoding="utf-8"
        )
        (self.output_dir / "resolved_config.json").write_text(
            json.dumps({"mode": "full", "learning_rate": 0.0002}), encoding="utf-8"
        )
        if adapter:
            adapter_dir = self.output_dir / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")

    def write_manifest(self) -> None:
        (self.export_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "courseId": COURSE,
                    "datasetVersion": f"{COURSE}-approved-split-seed360-n42",
                    "checksums": {"train.jsonl": "a" * 64, "validation.jsonl": "b" * 64},
                }
            ),
            encoding="utf-8",
        )

    def payload(self, **kwargs) -> dict:
        defaults = dict(
            course_id=COURSE,
            run_id=RUN_ID,
            outcome="succeeded",
            output_dir=self.output_dir,
            export_dir=self.export_dir,
            job_id=JOB_ID,
        )
        defaults.update(kwargs)
        return reporter.build_completion_payload(**defaults)


class PayloadTests(ReporterTestCase):
    def test_a_successful_run_reports_everything_the_registry_needs(self) -> None:
        self.write_run_output()
        self.write_manifest()

        payload = self.payload()

        self.assertEqual(payload["outcome"], "succeeded")
        self.assertEqual(payload["jobId"], JOB_ID)
        self.assertEqual(payload["baseModel"], "meta-llama/Llama-3.2-3B-Instruct")
        self.assertEqual(payload["trainExamples"], 37)
        self.assertEqual(payload["validationExamples"], 5)
        self.assertEqual(payload["intendedOptimizerSteps"], 15)
        self.assertEqual(payload["completedSteps"], 15)
        self.assertTrue(payload["trainingLengthSatisfied"])
        self.assertEqual(payload["actualGpuHours"], 0.0134)
        self.assertEqual(payload["gpuCount"], 1)
        self.assertEqual(payload["trainLoss"], 1.2345)
        self.assertEqual(payload["evalLoss"], 1.4567)
        self.assertEqual(
            payload["gitCommitSha"], "9941833cafe0000000000000000000000000beef"
        )
        self.assertEqual(payload["resolvedConfig"]["learning_rate"], 0.0002)

    def test_the_artifact_reference_is_relative_and_carries_no_username(self) -> None:
        """A stored reference outlives the account that produced it.

        `/gpfs/projects/simswe/madamk/training_outputs/...` embeds a cluster home
        directory and a username, and admin surfaces display this string.
        """
        self.write_run_output()

        payload = self.payload()

        self.assertEqual(
            payload["artifactRef"],
            f"qlora-runs/{COURSE}/20260827T064701Z-full/adapter",
        )
        self.assertFalse(payload["artifactRef"].startswith("/"))
        self.assertNotIn("gpfs", payload["artifactRef"])

    def test_dataset_checksums_come_from_the_prepared_manifest(self) -> None:
        self.write_run_output()
        self.write_manifest()

        payload = self.payload()

        self.assertEqual(
            payload["datasetVersion"], f"{COURSE}-approved-split-seed360-n42"
        )
        self.assertEqual(payload["datasetChecksums"]["train.jsonl"], "a" * 64)
        self.assertEqual(payload["datasetRef"], f"exports/{COURSE}")

    def test_a_success_with_no_adapter_is_reported_as_a_failure(self) -> None:
        """Exiting zero and producing a loadable model are different claims.

        Registering a model that cannot be loaded is worse than a failed run: it
        turns the professor's page green and puts the discovery off until
        somebody tries to serve it.
        """
        self.write_run_output(adapter=False)

        payload = self.payload()

        self.assertEqual(payload["outcome"], "failed")
        self.assertEqual(payload["failureStage"], "artifact")
        self.assertIn("no loadable adapter", payload["error"])

    def test_a_failure_that_measured_nothing_is_still_reportable(self) -> None:
        """The half of the bug that would otherwise survive.

        A job that died during model load wrote no runtime report and has no
        metrics. Silence here means a professor watches "training" indefinitely.
        """
        payload = self.payload(
            outcome="failed", failure_stage="model-load", error="no CUDA runtime"
        )

        self.assertEqual(payload["outcome"], "failed")
        self.assertEqual(payload["failureStage"], "model-load")
        self.assertEqual(payload["error"], "no CUDA runtime")
        self.assertNotIn("runtimeReport", payload)

    def test_a_short_run_reports_its_shortfall_rather_than_hiding_it(self) -> None:
        self.write_run_output(
            runtime={
                **RUNTIME_REPORT,
                "completedSteps": 12,
                "missingOptimizerSteps": 3,
                "trainingLengthSatisfied": False,
            }
        )

        payload = self.payload()

        self.assertEqual(payload["completedSteps"], 12)
        self.assertEqual(payload["missingOptimizerSteps"], 3)
        self.assertFalse(payload["trainingLengthSatisfied"])

    def test_a_long_error_is_truncated_rather_than_rejected(self) -> None:
        payload = self.payload(outcome="failed", error="x" * 5000)

        self.assertLessEqual(len(payload["error"]), 2000)


class DeliveryTests(ReporterTestCase):
    def test_the_payload_is_persisted_before_it_is_sent(self) -> None:
        """The ordering the whole design rests on.

        Persisting only after a failure would lose the event for a process
        killed between deciding to report and finishing the attempt.
        """
        self.write_run_output()
        seen_on_disk: list[int] = []

        class ObservingQueue(RecordingQueue):
            def record_completion(inner, course_id, run_id, payload):  # noqa: N805
                seen_on_disk.append(len(state.list_pending_callbacks(self.root)))
                return super().record_completion(course_id, run_id, payload)

        reporter.report_completion(
            course_id=COURSE,
            run_id=RUN_ID,
            payload=self.payload(),
            repo_root=self.root,
            queue=ObservingQueue(),
        )

        self.assertEqual(seen_on_disk, [1])

    def test_an_accepted_report_is_removed_from_the_queue(self) -> None:
        self.write_run_output()

        reporter.report_completion(
            course_id=COURSE,
            run_id=RUN_ID,
            payload=self.payload(),
            repo_root=self.root,
            queue=RecordingQueue(),
        )

        self.assertEqual(state.list_pending_callbacks(self.root), [])

    def test_an_unreachable_backend_keeps_the_report_and_exits_zero(self) -> None:
        """A network failure is not a training failure.

        A Slurm script that exited nonzero here would turn a successful run into
        a failed one in every log an operator reads.
        """
        self.write_run_output()

        code = reporter.report_completion(
            course_id=COURSE,
            run_id=RUN_ID,
            payload=self.payload(),
            repo_root=self.root,
            queue=RecordingQueue(error=TrainingQueueError("Could not reach the backend")),
        )

        self.assertEqual(code, 0)
        pending = state.list_pending_callbacks(self.root)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["payload"]["outcome"], "succeeded")

    def test_a_run_that_no_longer_exists_stops_being_retried(self) -> None:
        """A 404 is as final as a 409: there is nothing left to report against."""
        self.write_run_output()

        code = reporter.report_completion(
            course_id=COURSE,
            run_id=RUN_ID,
            payload=self.payload(),
            repo_root=self.root,
            queue=RecordingQueue(
                error=TrainingQueueError("failed with HTTP 404: run not found")
            ),
        )

        self.assertEqual(code, 0)
        self.assertEqual(state.list_pending_callbacks(self.root), [])

    def test_a_timeout_keeps_the_report_rather_than_dropping_it(self) -> None:
        """The failure this mechanism exists for stays queued."""
        self.write_run_output()

        reporter.report_completion(
            course_id=COURSE,
            run_id=RUN_ID,
            payload=self.payload(),
            repo_root=self.root,
            queue=RecordingQueue(error=TrainingQueueError("HTTP 503: unavailable")),
        )

        self.assertEqual(len(state.list_pending_callbacks(self.root)), 1)

    def test_a_superseded_run_stops_being_retried(self) -> None:
        """A 409 is final: the run was deliberately retired.

        Replaying it on every future flush would show an operator the same
        conflict forever for a run somebody already decided about.
        """
        self.write_run_output()

        code = reporter.report_completion(
            course_id=COURSE,
            run_id=RUN_ID,
            payload=self.payload(),
            repo_root=self.root,
            queue=RecordingQueue(
                error=TrainingQueueError("failed with HTTP 409: superseded")
            ),
        )

        self.assertEqual(code, 0)
        self.assertEqual(state.list_pending_callbacks(self.root), [])


class FlushTests(ReporterTestCase):
    def _queue_pending(self, run_id: str = RUN_ID, outcome: str = "succeeded") -> None:
        state.queue_pending_callback(
            self.root,
            run_id=run_id,
            course_id=COURSE,
            kind="completed",
            payload={"outcome": outcome, "jobId": JOB_ID},
        )

    def test_flushing_delivers_what_the_job_could_not(self) -> None:
        """The recovery path an operator gets without doing anything extra.

        The worker flushes at the start of every run, so the ordinary "backend
        was down when the job ended" case resolves itself the next time somebody
        runs the one command they were going to run anyway.
        """
        self._queue_pending()
        queue = RecordingQueue()

        summary = reporter.flush_pending(self.root, queue)

        self.assertEqual(summary["delivered"], 1)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(len(queue.calls), 1)
        self.assertEqual(state.list_pending_callbacks(self.root), [])

    def test_flushing_with_nothing_queued_does_nothing(self) -> None:
        queue = RecordingQueue()

        summary = reporter.flush_pending(self.root, queue)

        self.assertEqual(summary, {"pending": 0, "delivered": 0, "failed": 0, "superseded": 0})
        self.assertEqual(queue.calls, [])

    def test_a_still_unreachable_backend_leaves_the_report_queued(self) -> None:
        self._queue_pending()

        summary = reporter.flush_pending(
            self.root, RecordingQueue(error=TrainingQueueError("still down"))
        )

        self.assertEqual(summary["failed"], 1)
        self.assertEqual(len(state.list_pending_callbacks(self.root)), 1)

    def test_a_superseded_report_is_dropped_during_flush(self) -> None:
        self._queue_pending()

        summary = reporter.flush_pending(
            self.root, RecordingQueue(error=TrainingQueueError("HTTP 409: superseded"))
        )

        self.assertEqual(summary["superseded"], 1)
        self.assertEqual(state.list_pending_callbacks(self.root), [])

    def test_flushing_twice_delivers_once(self) -> None:
        """Redelivery is safe on the backend, but there is no reason to cause it."""
        self._queue_pending()
        queue = RecordingQueue()

        reporter.flush_pending(self.root, queue)
        second = reporter.flush_pending(self.root, queue)

        self.assertEqual(len(queue.calls), 1)
        self.assertEqual(second["pending"], 0)

    def test_several_queued_reports_are_all_delivered(self) -> None:
        self._queue_pending(RUN_ID)
        self._queue_pending("run-20260828t090000z-aa11bb", outcome="failed")
        queue = RecordingQueue()

        summary = reporter.flush_pending(self.root, queue)

        self.assertEqual(summary["delivered"], 2)
        self.assertEqual(len(queue.calls), 2)


class AdapterDetectionTests(ReporterTestCase):
    def test_a_peft_adapter_directory_is_recognised(self) -> None:
        """The format check that matters: this is what training writes.

        `adapter_config.json` plus `adapter_model.safetensors` is exactly what
        `PeftModel.load_adapter` reads. No conversion step exists because none is
        needed.
        """
        self.write_run_output()

        self.assertTrue(reporter.adapter_is_present(self.output_dir))

    def test_a_config_without_weights_is_not_an_adapter(self) -> None:
        self.write_run_output(adapter=False)
        adapter_dir = self.output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

        self.assertFalse(reporter.adapter_is_present(self.output_dir))

    def test_weights_without_a_config_is_not_an_adapter(self) -> None:
        self.write_run_output(adapter=False)
        adapter_dir = self.output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")

        self.assertFalse(reporter.adapter_is_present(self.output_dir))


if __name__ == "__main__":
    unittest.main()
