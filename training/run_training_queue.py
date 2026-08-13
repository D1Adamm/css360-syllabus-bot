#!/usr/bin/env python3
"""Claim one queued training run on Tillicum and report what it would launch.

Run this *on* Tillicum, in a session you logged into normally — the usual
password and two-factor prompt. Nothing here bypasses that, and nothing here
opens a connection to anywhere except the database over HTTPS. That is the
whole point of the queue: the browser writes a run and stops, and the work is
picked up later by a person who is already logged in.

This stage does not submit anything.
--------------------------------------
It discovers queued work, claims exactly one run, checks the run against the
prepared dataset actually present on this machine, and prints the command that
`training/start_qlora_training.sh` would be invoked with. It does not run that
command, does not call sbatch, and does not sync or copy any data. Datasets are
prepared elsewhere and pushed; nothing is regenerated here.

Because nothing is submitted, a claimed run is released back to `queued` before
this exits. Holding a lease for work that never started would make the run wait
out its expiry for no reason.

Usage (from the repository root on Tillicum):

    ./training/run_training_queue.sh --once
    ./training/run_training_queue.sh --once --dry-run
    ./training/run_training_queue.sh --once --course css-360-winter-2026-a7rp
"""

from __future__ import annotations

import argparse
import importlib.util
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_LIB = REPO_ROOT / "scripts" / "lib"

sys.path.insert(0, str(SCRIPTS_LIB))

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

START_SCRIPT = "./training/start_qlora_training.sh"


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


def describe_planned_launch(run: TrainingRun, *, helpers: Any) -> dict[str, str]:
    """Exactly what would be invoked, using the launcher's own job naming."""
    job_name = helpers.slurm_training_job_name(course_id=run.course_id, mode=run.mode)
    command = f"{START_SCRIPT} --course {run.course_id} --{run.mode} --yes"
    return {"jobName": job_name, "command": command}


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


def run_once(
    queue: TrainingQueue,
    *,
    helpers: Any,
    owner: str,
    dry_run: bool,
    lease_seconds: int,
    course_ids: list[str] | None,
    now: datetime | None = None,
    root: Path = REPO_ROOT,
) -> int:
    moment = now or utc_now()

    candidates = queue.discover_claimable(now=moment, course_ids=course_ids)
    if not candidates:
        print("No queued training runs.")
        return 0

    course_id, candidate = candidates[0]

    if dry_run:
        # Read-only on purpose: a dry run must be safe to point at the real
        # queue, which means claiming nothing another runner could have had.
        print("Dry run — nothing is claimed, written, or submitted.")
        _print_run(candidate, course_id)
        try:
            counts, warnings = validate_run_against_prepared_data(
                candidate, helpers=helpers, root=root
            )
        except ValidationFailed as exc:
            print(f"  Would refuse: {exc}")
            return 1
        plan = describe_planned_launch(candidate, helpers=helpers)
        for warning in warnings:
            print(f"  Warning: {warning}")
        print(f"  Prepared: {counts['train_count']} train / {counts['validation_count']} validation")
        print(f"  Job name: {plan['jobName']}")
        print(f"  Would run: {plan['command']}")
        print("Not submitted (this stage never calls sbatch).")
        return 0

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

    try:
        counts, warnings = validate_run_against_prepared_data(
            run, helpers=helpers, root=root
        )
    except ValidationFailed as exc:
        print(f"  Refused: {exc}")
        queue.release(course_id, run, error=str(exc), now=moment)
        print("Released back to queued; nothing was submitted.")
        return 1

    plan = describe_planned_launch(run, helpers=helpers)
    for warning in warnings:
        print(f"  Warning: {warning}")
    print(f"  Prepared: {counts['train_count']} train / {counts['validation_count']} validation")
    print(f"  Job name: {plan['jobName']}")
    print(f"  Would run: {plan['command']}")

    # Nothing was submitted, so nothing may keep holding the lease.
    queue.release(course_id, run, now=moment)
    print("Released back to queued; nothing was submitted.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_training_queue.py",
        description=(
            "Claim one queued training run and report what would be launched. "
            "This stage never submits a job."
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


def main(argv: list[str] | None = None) -> int:
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
        helpers = _load_qlora_helpers()
        queue = build_queue()
        return run_once(
            queue,
            helpers=helpers,
            owner=args.owner.strip() or default_owner(),
            dry_run=args.dry_run,
            lease_seconds=args.lease_seconds,
            course_ids=course_ids,
        )
    except TrainingQueueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
