#!/usr/bin/env python3
"""Claim one queued training run on Tillicum and submit it through the launcher.

Run this *on* Tillicum, in a session you logged into normally — the usual
password and two-factor prompt. Nothing here bypasses that, and nothing here
opens a connection to anywhere except the backend API over HTTPS and the local
`training/start_qlora_training.sh` script. That is the whole point of the
queue: the browser writes a run and stops, and the work is picked up later by
a person who is already logged in.

The queue lives in PostgreSQL on the application VM and is reached through
`/api/training-queue`. Set TRAINING_API_BASE_URL and TRAINING_WORKER_TOKEN in
the environment or .env.local before running this.

What one invocation does
------------------------
    1. deliver any completion reports earlier jobs queued locally
    2. re-report any submission the backend never acknowledged
    3. claim one queued run
    4. fetch that run's prepared dataset from the backend, unless the copy on
       disk already matches it byte for byte
    5. verify counts, checksums and training configuration
    6. refuse to submit a second job for a run that already has one
    7. submit through the existing launcher
    8. record the submission, locally first and then to the backend

Steps 1, 2 and 4 are new, and each replaces something an operator used to do by
hand: chase a finished job that never reported, work out whether a job had
really been submitted after a failed call, and `rsync` a dataset from the
application VM over a second Duo prompt.

Submission still reuses the existing launcher. This process never calls sbatch
itself and never promotes an adapter. `--dry-run` stays read-only: it claims
nothing, writes nothing, sends nothing, and never spawns the launcher — it will
describe the dataset it would fetch, but it will not fetch it.

Usage (from the repository root on Tillicum):

    ./training/run_training_queue.sh --once
    ./training/run_training_queue.sh --once --dry-run
    ./training/run_training_queue.sh --once --course css-360-winter-2026-a7rp
"""

from __future__ import annotations

import argparse
import importlib.util
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_LIB = REPO_ROOT / "scripts" / "lib"

sys.path.insert(0, str(SCRIPTS_LIB))

from dataset_sync import (  # noqa: E402
    DatasetSyncError,
    DatasetSyncResult,
    local_dataset_matches,
    sync_run_dataset,
    validate_descriptor,
)
from finetuned_deploy_helpers import parse_sbatch_job_id  # noqa: E402
from training_queue import (  # noqa: E402  (path set above)
    ClaimConflict,
    DEFAULT_LEASE_SECONDS,
    TrainingQueue,
    TrainingQueueError,
    TrainingRun,
    build_queue,
    load_env_file,
    utc_now,
    validate_course_id,
)
from training_state import (  # noqa: E402
    list_pending_callbacks,
    read_run_record,
    unreported_submissions,
    write_run_record,
)

START_SCRIPT = "./training/start_qlora_training.sh"
DEFAULT_LAUNCHER_TIMEOUT_SECONDS = 900
_JOB_ID_LINE_PREFIX = "job id:"


@dataclass(frozen=True)
class LauncherResult:
    returncode: int
    stdout: str
    stderr: str


Launcher = Callable[[list[str], Path], LauncherResult]


def _load_qlora_helpers() -> Any:
    """Import the existing QLoRA helpers by path.

    Reusing them is the point: the course-id rule, the export validation and the
    job-name scheme must be the ones the launcher itself uses, or this would
    approve a run the cluster scripts would then reject.
    """
    path = SCRIPTS_LIB / "qlora_training_helpers.py"
    spec = importlib.util.spec_from_file_location("qlora_training_helpers", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise TrainingQueueError(f"Missing training helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_exclude_nodes(nodes: str) -> str:
    """Reuse the deploy helper's rule rather than restating it here.

    The value becomes an sbatch argument, so it is validated wherever it enters
    — and validated by the one function that already knows what a compute
    hostname may look like.
    """
    import importlib.util

    path = SCRIPTS_LIB / "finetuned_deploy_helpers.py"
    spec = importlib.util.spec_from_file_location("finetuned_deploy_helpers", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise TrainingQueueError(f"Missing deploy helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.validate_exclude_nodes(nodes)
    except ValueError as exc:
        raise TrainingQueueError(str(exc)) from exc


def default_owner() -> str:
    import getpass

    try:
        user = getpass.getuser()
    except Exception:  # pragma: no cover - depends on the host account setup
        user = "unknown"
    return f"{user}@{socket.gethostname()}"


def course_export_dir(course_id: str, root: Path = REPO_ROOT) -> Path:
    """Where a course's prepared dataset was pushed to. Never regenerated."""
    return root / "data" / "exports" / validate_course_id(course_id)


class ValidationFailed(Exception):
    """The claimed run does not match a usable prepared dataset."""


def validate_run_against_prepared_data(
    run: TrainingRun,
    *,
    helpers: Any,
    root: Path = REPO_ROOT,
) -> tuple[dict[str, int], list[str]]:
    """Check the artifacts this run says it needs really are here.

    Returns the counts the helper found plus any discrepancies worth printing.
    A missing or unusable export is fatal — there is nothing to train on. A
    count that disagrees with what was enqueued is not: the dataset may simply
    have been re-prepared since, and an operator should be told rather than
    stopped.
    """
    helpers.validate_course_id(run.course_id)

    export_dir = course_export_dir(run.course_id, root)
    if not export_dir.is_dir():
        raise ValidationFailed(
            f"No prepared training data for {run.course_id} at {export_dir}. "
            "Prepare it and push it before running the queue."
        )

    try:
        counts = helpers.validate_course_export_dir(export_dir)
    except Exception as exc:
        raise ValidationFailed(f"Prepared training data is not usable: {exc}") from exc

    if int(counts.get("train_count", 0)) <= 0:
        raise ValidationFailed("The prepared training set is empty.")

    warnings: list[str] = []
    expected_ref = f"exports/{run.course_id}"
    if run.dataset_ref and run.dataset_ref != expected_ref:
        warnings.append(
            f"run datasetRef is {run.dataset_ref!r}, expected {expected_ref!r}"
        )
    if run.train_examples and run.train_examples != int(counts["train_count"]):
        warnings.append(
            f"enqueued train count {run.train_examples} != {counts['train_count']} on disk"
        )
    if run.validation_examples and run.validation_examples != int(
        counts["validation_count"]
    ):
        warnings.append(
            f"enqueued validation count {run.validation_examples} != "
            f"{counts['validation_count']} on disk"
        )

    return {
        "train_count": int(counts["train_count"]),
        "validation_count": int(counts["validation_count"]),
    }, warnings


def describe_planned_launch(
    run: TrainingRun, *, helpers: Any, exclude_nodes: str = ""
) -> dict[str, str]:
    """Exactly what would be invoked, using the launcher's own job naming."""
    job_name = helpers.slurm_training_job_name(course_id=run.course_id, mode=run.mode)
    command = " ".join(launcher_argv(run, exclude_nodes=exclude_nodes))
    return {
        "jobName": job_name,
        "command": command,
        "excludeNodes": exclude_nodes,
    }


def launcher_argv(run: TrainingRun, *, exclude_nodes: str = "") -> list[str]:
    """The launcher invocation, carrying the queue run id into the Slurm job.

    `--queue-run-id` is what lets the job report its own completion. Without it
    the compute node knows the course and the output directory but not which
    PostgreSQL run it is finishing, and the completion callback has nothing to
    address itself to — which is exactly why job 253552 could not report itself.
    """
    argv = [
        START_SCRIPT,
        "--course",
        run.course_id,
        f"--{run.mode}",
        "--queue-run-id",
        run.run_id,
    ]
    # Absent unless an operator asked for it on this invocation. The queue
    # worker has no opinion about node health and keeps none between runs.
    if exclude_nodes:
        argv += ["--exclude-node", exclude_nodes]
    argv.append("--yes")
    return argv


def subprocess_launcher(
    command: list[str],
    cwd: Path,
    timeout: int = DEFAULT_LAUNCHER_TIMEOUT_SECONDS,
) -> LauncherResult:
    """Run the existing launcher. Never used by ``--dry-run``."""
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(
            "utf-8", errors="replace"
        )
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(
            "utf-8", errors="replace"
        )
        return LauncherResult(
            returncode=124,
            stdout=stdout or "",
            stderr=(stderr or "") + "\nLauncher timed out.",
        )
    return LauncherResult(
        returncode=int(completed.returncode),
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


_OUTPUT_DIR_LINE_PREFIX = "output dir:"


def parse_launcher_output_dir(output: str) -> Optional[str]:
    """The versioned run directory the launcher reported, if it said one.

    Recorded locally alongside the job id so the three identifiers an operator
    needs to reconcile anything — run id, Slurm job id, output directory — are
    in one file, written before the backend was told about any of them.
    """
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(_OUTPUT_DIR_LINE_PREFIX):
            value = stripped.split(":", 1)[1].strip()
            if value:
                return value
    return None


def parse_launcher_job_id(output: str) -> Optional[str]:
    """Reuse the existing sbatch parser; also accept the launcher's Job ID line."""
    try:
        return parse_sbatch_job_id(output)
    except ValueError:
        pass
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(_JOB_ID_LINE_PREFIX):
            token = stripped.split(":", 1)[1].strip().split()[0] if ":" in stripped else ""
            if token.isdigit():
                return token
    return None


def _combined_output(result: LauncherResult) -> str:
    return "{0}\n{1}".format(result.stdout or "", result.stderr or "")


def _short_error(message: str) -> str:
    text = " ".join(message.split())
    if len(text) > 500:
        return text[:497] + "..."
    return text


def _print_run(run: TrainingRun, course_id: str) -> None:
    print(f"Run:      {run.run_id}")
    print(f"  Course:   {course_id}")
    print(f"  Mode:     {run.mode}")
    print(f"  State:    {run.state}")
    print(f"  Enqueued: {run.enqueued_at}")
    print(f"  Dataset:  {run.dataset_ref or '(none recorded)'}")
    print(
        f"  Counts:   {run.train_examples} train / {run.validation_examples} validation "
        f"from {run.approved_example_count} approved"
    )
    print(f"  Attempt:  {run.attempt}")


class DatasetUnavailable(Exception):
    """The backend could not give this run a usable dataset."""


def ensure_dataset(
    queue: TrainingQueue,
    run: TrainingRun,
    *,
    root: Path = REPO_ROOT,
) -> DatasetSyncResult:
    """Make this cluster hold exactly the dataset the backend has for this run.

    The step that removes the manual `rsync`. It is safe to repeat: a local copy
    that already matches the backend's digests is left alone, so a retried run
    costs one request rather than a transfer.

    Failures here are refusals, not crashes. The run goes back on the queue with
    an operator-readable reason — the dataset can be re-prepared on the VM and
    the same run picked up again, with no second Slurm job and nothing lost.
    """
    export_dir = course_export_dir(run.course_id, root)
    try:
        return sync_run_dataset(queue, run.course_id, run.run_id, export_dir)
    except DatasetSyncError as exc:
        raise DatasetUnavailable(str(exc)) from exc
    except TrainingQueueError as exc:
        raise DatasetUnavailable(
            f"Could not fetch the prepared dataset from the backend: {exc}"
        ) from exc


def _load_reporter() -> Any:
    """Import the completion reporter by path.

    Loaded lazily and by path rather than imported at module scope: it lives in
    `training/` alongside this file rather than on the import path, and a worker
    doing only a dry run never needs it.
    """
    path = REPO_ROOT / "training" / "report_training_result.py"
    spec = importlib.util.spec_from_file_location("report_training_result", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise TrainingQueueError(f"Missing completion reporter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flush_pending_reports(
    queue: TrainingQueue,
    *,
    root: Path = REPO_ROOT,
) -> None:
    """Deliver completion reports earlier jobs could not send.

    Run first, before anything is claimed. A job that finished while the backend
    was unreachable left its report on the shared filesystem; this is the moment
    it becomes the application's problem instead of a file nobody looks at. It
    is also why an operator needs no separate "did anything finish?" command —
    the one command they were going to run anyway resolves it.

    The worker's own queue client is reused rather than a second one built, so a
    flush uses exactly the credentials and base URL the rest of the run does.
    """
    pending = list_pending_callbacks(root)
    if not pending:
        return

    print(f"Delivering {len(pending)} queued training report(s)...")
    summary = _load_reporter().flush_pending(root, queue)
    print(
        "  delivered {0}, still queued {1}, dropped as superseded {2}".format(
            summary["delivered"], summary["failed"], summary["superseded"]
        )
    )


def reconcile_unreported_submissions(
    queue: TrainingQueue,
    *,
    root: Path = REPO_ROOT,
) -> None:
    """Re-report submissions the backend never acknowledged.

    The ambiguous-network case, made recoverable. `sbatch` succeeded and the
    `/submitted` call did not land, so the cluster has a real job the application
    has never heard of. The local run record — written before the report was
    attempted — is the evidence, and re-reporting is the correct repair.
    Submitting again would be the wrong one: it would spend a second GPU
    allocation on work already running.
    """
    outstanding = unreported_submissions(root)
    if not outstanding:
        return

    print(f"Reconciling {len(outstanding)} unreported submission(s)...")
    for record in outstanding:
        run_id = str(record.get("runId") or "")
        course_id = str(record.get("courseId") or "")
        job_id = str(record.get("jobId") or "")
        stub = TrainingRun(
            run_id=run_id,
            course_id=course_id,
            mode=str(record.get("mode") or "full"),
            state="claimed",
            enqueued_at="",
            updated_at="",
            dataset_ref=f"exports/{course_id}",
            approved_example_count=0,
            train_examples=int(record.get("trainExamples") or 0),
            validation_examples=int(record.get("validationExamples") or 0),
            attempt=0,
        )
        try:
            queue.record_submission(
                course_id,
                stub,
                job_id=job_id,
                train_count=stub.train_examples,
                validation_count=stub.validation_examples,
            )
        except TrainingQueueError as exc:
            message = str(exc)
            if "HTTP 409" in message or "HTTP 404" in message:
                # Superseded or removed. The job is real but the application has
                # deliberately moved on; recording that locally stops this being
                # retried on every future run.
                print(f"  {run_id}: not applied ({message})")
                write_run_record(
                    root,
                    run_id=run_id,
                    course_id=course_id,
                    mode=str(record.get("mode") or "full"),
                    job_id=job_id,
                    reported=True,
                    extra={"reportRefused": message[:500]},
                )
                continue
            print(f"  {run_id}: still unreported ({message})")
            continue

        print(f"  {run_id}: reported as job {job_id}")
        write_run_record(
            root,
            run_id=run_id,
            course_id=course_id,
            mode=str(record.get("mode") or "full"),
            job_id=job_id,
            reported=True,
        )


def run_once(
    queue: TrainingQueue,
    *,
    helpers: Any,
    owner: str,
    dry_run: bool,
    exclude_nodes: str = "",
    lease_seconds: int,
    course_ids: Optional[list[str]],
    now: Optional[datetime] = None,
    root: Path = REPO_ROOT,
    launcher: Optional[Launcher] = None,
) -> int:
    moment = now or utc_now()

    if not dry_run:
        # Before anything is claimed, and in this order: a completion that never
        # landed describes work already finished, and a submission that never
        # landed describes work already running. Both change what claiming a new
        # run means, and both are cheap no-ops when there is nothing outstanding.
        flush_pending_reports(queue, root=root)
        reconcile_unreported_submissions(queue, root=root)

    candidates = queue.discover_claimable(now=moment, course_ids=course_ids)
    if not candidates:
        print("No queued training runs.")
        return 0

    course_id, candidate = candidates[0]

    if dry_run:
        return _dry_run(
            candidate,
            course_id,
            queue=queue,
            helpers=helpers,
            root=root,
            exclude_nodes=exclude_nodes,
        )

    try:
        run = queue.claim(
            course_id,
            candidate,
            owner=owner,
            lease_seconds=lease_seconds,
            now=moment,
        )
    except ClaimConflict as exc:
        # Another runner got there first. Taking the next one instead would
        # race the same way, and this process is meant to do one thing.
        print(f"Not claimed: {exc}")
        return 0

    print(f"Claimed by {owner} (lease {lease_seconds}s).")
    _print_run(run, course_id)

    # Two independent records of "this run already has a job". The backend's is
    # authoritative when it has one; the local one covers the case the backend
    # cannot know about, where sbatch succeeded and the report did not land.
    # Either is enough to refuse a second submission — which matters, because a
    # duplicate here is a second GPU allocation training the same adapter.
    local_record = read_run_record(root, run.run_id) or {}
    existing_job_id = run.job_id or str(local_record.get("jobId") or "")
    if existing_job_id:
        print(f"  Already submitted as job {existing_job_id}; will not submit another.")
        submitted = queue.record_submission(
            course_id,
            run,
            job_id=existing_job_id,
            train_count=run.train_examples or int(local_record.get("trainExamples") or 0),
            validation_count=run.validation_examples
            or int(local_record.get("validationExamples") or 0),
            now=moment,
        )
        write_run_record(
            root,
            run_id=run.run_id,
            course_id=course_id,
            mode=run.mode,
            job_id=existing_job_id,
            reported=True,
        )
        print(f"  Job ID: {submitted.job_id}")
        print("  modelRequest.status=training")
        return 0

    try:
        sync = ensure_dataset(queue, run, root=root)
    except DatasetUnavailable as exc:
        print(f"  Refused: {exc}")
        queue.release(course_id, run, error=str(exc), now=moment)
        print("Released back to queued; nothing was submitted.")
        return 1

    print(f"  Dataset:  {sync.describe()}")

    try:
        counts, warnings = validate_run_against_prepared_data(
            run, helpers=helpers, root=root
        )
    except ValidationFailed as exc:
        print(f"  Refused: {exc}")
        queue.release(course_id, run, error=str(exc), now=moment)
        print("Released back to queued; nothing was submitted.")
        return 1

    plan = describe_planned_launch(run, helpers=helpers, exclude_nodes=exclude_nodes)
    for warning in warnings:
        print(f"  Warning: {warning}")
    print(f"  Prepared: {counts['train_count']} train / {counts['validation_count']} validation")
    print(f"  Job name: {plan['jobName']}")
    if plan["excludeNodes"]:
        print(f"  Excluding: {plan['excludeNodes']} (temporary, this run only)")
    print(f"  Running: {plan['command']}")

    # Written before the launcher runs, so a submission that succeeds while this
    # process is killed still leaves the run/job/output mapping behind. The job
    # id is filled in after sbatch; what matters now is that the record exists.
    write_run_record(
        root,
        run_id=run.run_id,
        course_id=course_id,
        mode=run.mode,
        dataset_sha256=sync.dataset_sha256,
        train_examples=counts["train_count"],
        validation_examples=counts["validation_count"],
        reported=False,
    )

    invoke = launcher if launcher is not None else subprocess_launcher
    result = invoke(launcher_argv(run, exclude_nodes=exclude_nodes), root)
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")

    job_id = parse_launcher_job_id(_combined_output(result))
    if job_id:
        # Locally first. If the report below fails, the next run of this command
        # finds an unreported submission and repairs it rather than submitting a
        # second job for work that is already queued on the cluster.
        write_run_record(
            root,
            run_id=run.run_id,
            course_id=course_id,
            mode=run.mode,
            job_id=job_id,
            output_dir=parse_launcher_output_dir(_combined_output(result)),
            dataset_sha256=sync.dataset_sha256,
            train_examples=counts["train_count"],
            validation_examples=counts["validation_count"],
            reported=False,
        )

        try:
            submitted = queue.record_submission(
                course_id,
                run,
                job_id=job_id,
                train_count=counts["train_count"],
                validation_count=counts["validation_count"],
                now=moment,
            )
        except TrainingQueueError as exc:
            print(f"  Job ID: {job_id}")
            print(f"  Submitted, but the application was not told: {exc}")
            print(
                "  The job is running. Re-run ./training/run_training_queue.sh "
                "--once to report it; nothing will be submitted twice."
            )
            return 1

        write_run_record(
            root,
            run_id=run.run_id,
            course_id=course_id,
            mode=run.mode,
            job_id=job_id,
            reported=True,
        )
        print(f"  Job ID: {submitted.job_id}")
        print("  trainingRun.state=submitted")
        print("  modelRequest.status=training")
        if result.returncode != 0:
            print("  Launcher exited nonzero after reporting a job id; recorded the job anyway.")
        return 0

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "launcher exited nonzero"
        error = _short_error("Submitting the training job failed: {0}".format(detail))
    else:
        error = (
            "The training job did not report a Slurm job ID; nothing was "
            "recorded as submitted."
        )
    print(f"  Submission failed: {error}")
    queue.record_submission_failure(course_id, run, error=error, now=moment)
    print("Released back to queued; modelRequest stays preparing.")
    return 1


def _dry_run(
    candidate: TrainingRun,
    course_id: str,
    *,
    queue: TrainingQueue,
    helpers: Any,
    root: Path,
    exclude_nodes: str = "",
) -> int:
    """Describe what would happen. Claims nothing, writes nothing, sends nothing.

    It does ask the backend to *describe* the dataset, which is a read. That is
    the difference between "there is a queued run" and "there is a queued run
    that could actually be trained right now", and it is the question a dry run
    is being asked.
    """
    print("Dry run — nothing is claimed, written, downloaded, or submitted.")
    _print_run(candidate, course_id)

    pending = list_pending_callbacks(root)
    if pending:
        print(
            f"  Queued reports: {len(pending)} waiting to be delivered "
            "(a real run would deliver them first)"
        )
    outstanding = unreported_submissions(root)
    if outstanding:
        print(
            f"  Unreported submissions: {len(outstanding)} "
            "(a real run would re-report them, not resubmit)"
        )

    export_dir = course_export_dir(candidate.course_id, root)
    try:
        descriptor = queue.describe_dataset(candidate.course_id, candidate.run_id)
    except TrainingQueueError as exc:
        print(f"  Would refuse: the backend has no usable dataset for this run ({exc})")
        return 1

    print(
        "  Backend dataset: {0} train / {1} validation, digest {2}".format(
            descriptor.get("trainExamples"),
            descriptor.get("validationExamples"),
            str(descriptor.get("datasetSha256") or "")[:12],
        )
    )
    try:
        described = validate_descriptor(descriptor)
        if export_dir.is_dir() and local_dataset_matches(export_dir, described):
            print("  Local copy:      already matches; nothing would be downloaded")
        else:
            print(f"  Local copy:      would download into {export_dir}")
    except DatasetSyncError as exc:
        print(f"  Would refuse: {exc}")
        return 1

    plan = describe_planned_launch(
        candidate, helpers=helpers, exclude_nodes=exclude_nodes
    )
    print(f"  Job name: {plan['jobName']}")
    if plan["excludeNodes"]:
        print(f"  Excluding: {plan['excludeNodes']} (temporary, this run only)")
    print(f"  Would run: {plan['command']}")
    print("Not submitted (--dry-run never calls the launcher).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_training_queue.py",
        description=(
            "Claim one queued training run and submit it through "
            "training/start_qlora_training.sh. --dry-run never submits."
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process at most one run and exit. Required; there is no daemon mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read only: show the next run without claiming or writing anything.",
    )
    parser.add_argument(
        "--course",
        action="append",
        dest="courses",
        metavar="COURSE_ID",
        help="Limit to one course. Repeatable.",
    )
    parser.add_argument(
        "--exclude-node",
        action="append",
        dest="exclude_nodes",
        metavar="NODE",
        help=(
            "Do not schedule this submission on NODE. Repeatable. TEMPORARY "
            "infrastructure troubleshooting only — nothing is remembered "
            "between runs, and no node is ever excluded by default."
        ),
    )
    parser.add_argument(
        "--owner",
        default="",
        help="Lease owner recorded on the claim (default: user@host).",
    )
    parser.add_argument(
        "--lease-seconds",
        type=int,
        default=DEFAULT_LEASE_SECONDS,
        help=f"How long a claim is held before it may be retaken (default {DEFAULT_LEASE_SECONDS}).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.once:
        print("--once is required: this runner processes one run and exits.", file=sys.stderr)
        return 2
    if args.lease_seconds <= 0:
        print("--lease-seconds must be positive.", file=sys.stderr)
        return 2

    load_env_file(REPO_ROOT)

    try:
        course_ids = (
            [validate_course_id(course) for course in args.courses]
            if args.courses
            else None
        )
        exclude_nodes = ""
        if args.exclude_nodes:
            exclude_nodes = _validate_exclude_nodes(",".join(args.exclude_nodes))
        helpers = _load_qlora_helpers()
        queue = build_queue()
        return run_once(
            queue,
            helpers=helpers,
            owner=args.owner.strip() or default_owner(),
            dry_run=args.dry_run,
            lease_seconds=args.lease_seconds,
            course_ids=course_ids,
            exclude_nodes=exclude_nodes,
        )
    except TrainingQueueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
