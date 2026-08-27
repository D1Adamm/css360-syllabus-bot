#!/usr/bin/env python3
"""Tell the application how a training job ended. Never lose the event.

Stdlib only, Python 3.9 compatible: this runs on a Tillicum compute node inside
the Slurm job, and on a login node when flushing.

The problem
-----------
Job 253552 trained successfully and the application never found out. The run
stayed `submitted` and the model request stayed `training` for days, because the
only path from "the job ended" to "the database knows" was a person noticing and
running commands by hand. That is not an edge case to tidy up; it is a missing
edge in the state machine, and the fix is that the job reports its own ending.

Ordering, and why it is this way round
--------------------------------------
    1. build the payload from what the run wrote to disk
    2. persist it under training/state/pending/
    3. try to send it
    4. on acceptance, delete the persisted copy

Persisting *before* the first attempt is the whole design. If it were written
only after a failure, a process killed between deciding to report and finishing
the attempt would take the event with it. As written, anything in `pending/` is
by definition an event UWB has not acknowledged, and `--flush` replays it.

Exit status is deliberately 0 when the backend is unreachable. The job's own
outcome is not "failed" because a network call did not go through, and a Slurm
script that failed here would turn a successful training run into a failed one.
The event is on disk; the next `run_training_queue.sh --once` sends it.

Whether a compute node can reach UWB at all is an environment fact this script
does not assume either way. If it can, the callback lands the moment the job
ends. If it cannot, the payload waits on GPFS — visible from the login node,
which certainly can — and lands on the operator's next queue run.

Usage
-----
    # inside train.slurm, after training
    python3 training/report_training_result.py \\
        --course-id css-350-spring-2026-n3h9 \\
        --run-id run-20260827t064701z-1cf650 \\
        --output-dir "$OUT_DIR" --outcome succeeded

    # after a failure, from the Slurm trap
    python3 training/report_training_result.py ... \\
        --outcome failed --failure-stage training --error "..."

    # on a login node, replay anything still pending
    python3 training/report_training_result.py --flush
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_LIB = REPO_ROOT / "scripts" / "lib"

sys.path.insert(0, str(SCRIPTS_LIB))

from training_queue import (  # noqa: E402  (path set above)
    TrainingQueue,
    TrainingQueueError,
    build_queue,
    load_env_file,
    validate_course_id,
    validate_run_id,
)
from training_state import (  # noqa: E402
    clear_pending_callback,
    list_pending_callbacks,
    queue_pending_callback,
    read_run_record,
    write_run_record,
)

RUNTIME_REPORT = "runtime-report.json"
RESOLVED_CONFIG = "resolved_config.json"
TRAINING_METRICS = "training_metrics.json"
EVALUATION_METRICS = "evaluation_metrics.json"
ADAPTER_DIRNAME = "adapter"
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHT_NAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_model.pt",
)

OUTCOMES = ("succeeded", "failed")

TRAINING_OUTPUTS_MARKER = "training_outputs/"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def relative_output_ref(path: str) -> str:
    """Strip everything before `training_outputs/`.

    The same rule `qlora_training_helpers.relative_training_output_ref` applies,
    restated here rather than imported so this file keeps its two-module import
    surface on a compute node. A stored reference must not carry a cluster home
    directory or a username: the model registry outlives both, and admin
    surfaces display the string.
    """
    normalized = str(path).replace("\\", "/").rstrip("/")
    marker_at = normalized.rfind(TRAINING_OUTPUTS_MARKER)
    if marker_at >= 0:
        return normalized[marker_at + len(TRAINING_OUTPUTS_MARKER) :]
    return normalized.lstrip("/")


def adapter_is_present(output_dir: Path) -> bool:
    """Whether the run actually wrote a loadable PEFT adapter.

    Checked here rather than trusted from the exit status because "the process
    exited 0" and "there is an adapter to serve" are different claims, and the
    second is the one that justifies registering a model.
    """
    adapter = output_dir / ADAPTER_DIRNAME
    if not (adapter / ADAPTER_CONFIG).is_file():
        return False
    return any((adapter / name).is_file() for name in ADAPTER_WEIGHT_NAMES)


def _first_number(source: Optional[Dict[str, Any]], *keys: str) -> Optional[float]:
    if not source:
        return None
    for key in keys:
        value = source.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def dataset_facts(export_dir: Path) -> Dict[str, Any]:
    """What the prepared manifest says about the data this job trained on."""
    manifest = read_json(export_dir / "manifest.json") or {}
    facts: Dict[str, Any] = {}
    version = manifest.get("datasetVersion")
    if isinstance(version, str) and version.strip():
        facts["datasetVersion"] = version.strip()
    checksums = manifest.get("checksums")
    if isinstance(checksums, dict):
        cleaned = {
            str(name): str(digest)
            for name, digest in checksums.items()
            if isinstance(digest, str)
        }
        if cleaned:
            facts["datasetChecksums"] = cleaned
    return facts


def build_completion_payload(
    *,
    course_id: str,
    run_id: str,
    outcome: str,
    output_dir: Path,
    export_dir: Optional[Path] = None,
    job_id: Optional[str] = None,
    failure_stage: Optional[str] = None,
    error: Optional[str] = None,
    started_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the completion report from what the run left on disk.

    Everything is optional on purpose. A job that died during model load wrote
    no runtime report and has no metrics; that still has to produce a reportable
    failure, or silent failures stay silent — which is the worse half of the bug
    this exists to fix.
    """
    runtime = read_json(output_dir / RUNTIME_REPORT)
    resolved = read_json(output_dir / RESOLVED_CONFIG)
    train_metrics = read_json(output_dir / TRAINING_METRICS)
    eval_metrics = read_json(output_dir / EVALUATION_METRICS)

    payload: Dict[str, Any] = {
        "outcome": outcome,
        "completedAt": utc_now_iso(),
        "outputRef": relative_output_ref(str(output_dir)),
    }

    if started_at:
        payload["startedAt"] = started_at
    if job_id:
        payload["jobId"] = str(job_id)
    elif runtime and isinstance(runtime.get("slurmJobId"), str):
        payload["jobId"] = runtime["slurmJobId"]

    if runtime:
        payload["runtimeReport"] = runtime
        for key, source_key in (
            ("baseModel", "modelId"),
            ("gitCommitSha", "gitCommitSha"),
        ):
            value = runtime.get(source_key)
            if isinstance(value, str) and value.strip():
                payload[key] = value.strip()

        for key, source_key in (
            ("trainExamples", "trainExampleCount"),
            ("validationExamples", "validationExampleCount"),
            ("gpuCount", "gpuCount"),
            ("intendedOptimizerSteps", "intendedOptimizerSteps"),
            ("completedSteps", "completedSteps"),
            ("missingOptimizerSteps", "missingOptimizerSteps"),
        ):
            value = runtime.get(source_key)
            if isinstance(value, int) and not isinstance(value, bool):
                payload[key] = value

        for key, source_key in (
            ("epochs", "epochs"),
            ("actualGpuHours", "actualGpuHours"),
            ("elapsedSeconds", "totalElapsedSeconds"),
        ):
            value = runtime.get(source_key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                payload[key] = float(value)

        satisfied = runtime.get("trainingLengthSatisfied")
        if isinstance(satisfied, bool):
            payload["trainingLengthSatisfied"] = satisfied

    if resolved:
        payload["resolvedConfig"] = resolved
    if train_metrics:
        payload["trainingMetrics"] = train_metrics
        train_loss = _first_number(train_metrics, "train_loss", "loss")
        if train_loss is not None:
            payload["trainLoss"] = train_loss
    if eval_metrics:
        payload["evaluationMetrics"] = eval_metrics
        eval_loss = _first_number(eval_metrics, "eval_loss")
        if eval_loss is not None:
            payload["evalLoss"] = eval_loss

    if export_dir is not None:
        payload.update(dataset_facts(export_dir))
    payload.setdefault("datasetRef", "exports/{0}".format(course_id))

    if outcome == "succeeded":
        if adapter_is_present(output_dir):
            payload["artifactRef"] = "{0}/{1}".format(
                relative_output_ref(str(output_dir)), ADAPTER_DIRNAME
            )
        else:
            # The job says it succeeded and there is no adapter. Reporting the
            # success anyway would register a model that cannot be loaded, so
            # this is reported as the failure it is.
            payload["outcome"] = "failed"
            payload["failureStage"] = "artifact"
            payload["error"] = (
                "Training reported success but wrote no loadable adapter to "
                "{0}/adapter.".format(relative_output_ref(str(output_dir)))
            )
            return payload

    if outcome == "failed":
        payload["failureStage"] = failure_stage or "training"
        payload["error"] = (error or "Training did not finish successfully.")[:2000]

    return payload


def send_completion(
    queue: TrainingQueue,
    *,
    course_id: str,
    run_id: str,
    payload: Dict[str, Any],
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Send one completion and clear its pending copy once it is accepted.

    Two refusals clear the pending copy as well, because both are final and
    replaying them would mean an operator sees the same error on every future
    flush for something already decided:

      - 409: the run was superseded by an admin retry. The backend deliberately
        refuses reports from a retired run.
      - 404: the run no longer exists for that course. There is nothing left to
        report against.

    Every other failure keeps the report queued. A timeout or an unreachable
    host is exactly what this mechanism exists for.
    """
    try:
        result = queue.record_completion(course_id, run_id, payload)
    except TrainingQueueError as exc:
        message = str(exc)
        if "HTTP 409" in message or "HTTP 404" in message:
            clear_pending_callback(repo_root, run_id, "completed")
            return {
                "delivered": False,
                "superseded": True,
                "detail": message,
            }
        raise

    clear_pending_callback(repo_root, run_id, "completed")
    return {"delivered": True, "result": result}


def report_completion(
    *,
    course_id: str,
    run_id: str,
    payload: Dict[str, Any],
    repo_root: Path = REPO_ROOT,
    queue: Optional[TrainingQueue] = None,
) -> int:
    """Persist, then attempt to deliver. Returns a process exit status.

    Zero even when delivery fails. The training job's outcome does not depend on
    whether a network call succeeded, and a Slurm script that exited nonzero
    here would turn a good run into a failed one in every log an operator reads.
    """
    queue_pending_callback(
        repo_root,
        run_id=run_id,
        course_id=course_id,
        kind="completed",
        payload=payload,
    )
    print("Completion recorded locally for {0} ({1}).".format(run_id, payload["outcome"]))

    try:
        client = queue if queue is not None else build_queue()
    except TrainingQueueError as exc:
        print(
            "Not sent (worker API is not configured here): {0}\n"
            "The report is queued; it will be delivered by the next "
            "./training/run_training_queue.sh --once".format(exc)
        )
        return 0

    try:
        outcome = send_completion(
            client,
            course_id=course_id,
            run_id=run_id,
            payload=payload,
            repo_root=repo_root,
        )
    except TrainingQueueError as exc:
        print(
            "Not sent ({0}).\nThe report is queued at training/state/pending/; "
            "it will be delivered by the next "
            "./training/run_training_queue.sh --once".format(exc)
        )
        return 0

    if outcome.get("superseded"):
        print("Not applied: {0}".format(outcome.get("detail")))
        return 0

    result = outcome.get("result") or {}
    print("Reported to the application.")
    print("  run state:      {0}".format(result.get("runState")))
    print("  request status: {0}".format(result.get("requestStatus")))
    if result.get("registered"):
        print(
            "  model version:  {0}{1}".format(
                result.get("version"),
                " (already registered)" if result.get("alreadyRegistered") else "",
            )
        )
    return 0


def flush_pending(
    repo_root: Path = REPO_ROOT,
    queue: Optional[TrainingQueue] = None,
) -> Dict[str, int]:
    """Deliver everything still queued locally. Safe to call at any time.

    Called at the start of every worker run, so the ordinary "backend was down
    when the job ended" case resolves itself the next time an operator does the
    one thing they were going to do anyway.
    """
    pending = list_pending_callbacks(repo_root)
    summary = {"pending": len(pending), "delivered": 0, "failed": 0, "superseded": 0}
    if not pending:
        return summary

    try:
        client = queue if queue is not None else build_queue()
    except TrainingQueueError as exc:
        print("Cannot flush {0} queued report(s): {1}".format(len(pending), exc))
        summary["failed"] = len(pending)
        return summary

    for entry in pending:
        run_id = str(entry.get("runId") or "")
        course_id = str(entry.get("courseId") or "")
        kind = str(entry.get("kind") or "")
        payload = entry.get("payload") or {}

        if kind != "completed":
            # Submission reports are replayed by the worker itself, which has
            # the run record and the duplicate-job guard. Nothing to do here.
            continue

        try:
            outcome = send_completion(
                client,
                course_id=course_id,
                run_id=run_id,
                payload=payload,
                repo_root=repo_root,
            )
        except TrainingQueueError as exc:
            print("  {0}: still undelivered ({1})".format(run_id, exc))
            summary["failed"] += 1
            continue

        if outcome.get("superseded"):
            print("  {0}: dropped, run was superseded".format(run_id))
            summary["superseded"] += 1
            continue

        result = outcome.get("result") or {}
        summary["delivered"] += 1
        print(
            "  {0}: delivered ({1}{2})".format(
                run_id,
                result.get("runState"),
                ", model {0}".format(result.get("version"))
                if result.get("version")
                else "",
            )
        )

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="report_training_result.py",
        description=(
            "Report a finished training job to the application, persisting the "
            "report first so a network failure cannot lose it."
        ),
    )
    parser.add_argument(
        "--flush",
        action="store_true",
        help="Deliver reports queued by earlier runs and exit.",
    )
    parser.add_argument("--course-id")
    parser.add_argument("--run-id", help="The PostgreSQL training run id.")
    parser.add_argument("--output-dir", help="The run's TRAINING_OUTPUT_DIR.")
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--outcome", choices=OUTCOMES, default="succeeded")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--failure-stage", default=None)
    parser.add_argument("--error", default=None)
    parser.add_argument("--started-at", default=None)
    parser.add_argument(
        "--no-register",
        action="store_true",
        help=(
            "Report the outcome without registering a model version. Used by "
            "smoke runs, which produce an adapter that is a rehearsal rather "
            "than a model anyone should be served."
        ),
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Show the payload without persisting or sending it.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    load_env_file(REPO_ROOT)

    if args.flush:
        summary = flush_pending()
        if summary["pending"] == 0:
            print("No queued training reports.")
        return 0

    missing = [
        name
        for name, value in (
            ("--course-id", args.course_id),
            ("--run-id", args.run_id),
            ("--output-dir", args.output_dir),
        )
        if not value
    ]
    if missing:
        print(
            "Missing required argument(s): {0}".format(", ".join(missing)),
            file=sys.stderr,
        )
        return 2

    try:
        course_id = validate_course_id(args.course_id)
        run_id = validate_run_id(args.run_id)
    except TrainingQueueError as exc:
        print("ERROR: {0}".format(exc), file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    export_dir = (
        Path(args.export_dir)
        if args.export_dir
        else REPO_ROOT / "data" / "exports" / course_id
    )

    record = read_run_record(REPO_ROOT, run_id) or {}
    job_id = args.job_id or os.environ.get("SLURM_JOB_ID") or record.get("jobId")

    payload = build_completion_payload(
        course_id=course_id,
        run_id=run_id,
        outcome=args.outcome,
        output_dir=output_dir,
        export_dir=export_dir,
        job_id=str(job_id) if job_id else None,
        failure_stage=args.failure_stage,
        error=args.error,
        started_at=args.started_at,
    )
    if args.no_register:
        payload["register"] = False

    if args.print_only:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    # The run record is the local half of the reconciliation pair. Written even
    # for a failure, because "which Slurm job produced which output directory
    # for which run" is exactly what an operator needs when something did not go
    # through.
    write_run_record(
        REPO_ROOT,
        run_id=run_id,
        course_id=course_id,
        mode=str(record.get("mode") or "full"),
        job_id=str(job_id) if job_id else None,
        output_dir=str(output_dir),
        reported=bool(record.get("reported")),
        extra={"lastOutcome": payload["outcome"], "lastReportedAt": utc_now_iso()},
    )

    return report_completion(course_id=course_id, run_id=run_id, payload=payload)


if __name__ == "__main__":
    raise SystemExit(main())
