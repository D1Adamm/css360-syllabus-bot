"""Controlled boundary for submitting a QLoRA training job.

The browser never runs ssh, rsync, or sbatch. It calls this module, which
validates everything it can locally and then shells out to the two scripts that
already exist:

  scripts/sync_training_data_to_tillicum.sh   copies ONE course's export
  training/start_qlora_training.sh            submits the Slurm job

Neither is reimplemented here. In particular `start_qlora_training.sh` already
builds a versioned run directory, refuses to write over the live inference
adapter, and declines to submit when an equally-named job is already active —
duplicating any of that would create a second implementation that could drift
from the one the cluster actually runs.

Why launching is disabled unless explicitly enabled
---------------------------------------------------
`sync_training_data_to_tillicum.sh` documents that "UW Duo/password prompts
remain interactive", and `start_qlora_training.sh` is written to run *on*
Tillicum. A web request cannot complete a Duo prompt. So on a machine without
non-interactive SSH to the cluster this would hang or fail, and a button that
silently fails is worse than one that says it is unavailable.

Real submission therefore requires TRAINING_LAUNCH_ENABLED=1 to be set
deliberately by whoever operates the backend, alongside working SSH. Otherwise
the endpoint reports that launching is not available here and changes nothing.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.course_id import assert_valid_course_id

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SYNC_SCRIPT = PROJECT_ROOT / "scripts" / "sync_training_data_to_tillicum.sh"
START_SCRIPT_RELATIVE = "training/start_qlora_training.sh"
HELPERS_PATH = PROJECT_ROOT / "scripts" / "lib" / "qlora_training_helpers.py"

VALID_MODES = ("smoke", "full")

# sbatch prints "Submitted batch job 1234567".
_SBATCH_JOB_ID = re.compile(r"Submitted batch job (\d+)")
# start_qlora_training.sh exits 0 with this when a job is already queued/running.
_EXISTING_JOB = re.compile(r"Existing active .* job found", re.IGNORECASE)

DEFAULT_TIMEOUT_SECONDS = 900


class TrainingLaunchError(Exception):
    """Base for launch failures."""


class LaunchDisabledError(TrainingLaunchError):
    """Launching is not enabled or not possible on this host."""


class LaunchValidationError(TrainingLaunchError):
    """The request or its prepared artifacts are not fit to launch."""


class LaunchExecutionError(TrainingLaunchError):
    """Sync or submission ran and failed."""


def _load_helpers() -> Any:
    """Import the existing training helpers by path.

    They live outside the backend package, and reusing them is the point:
    export validation must agree exactly with what the launcher scripts do.
    """
    spec = importlib.util.spec_from_file_location(
        "qlora_training_helpers", HELPERS_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise LaunchValidationError("Training helpers are unavailable.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def course_export_dir(course_id: str) -> Path:
    """Local prepared-dataset directory for exactly one course."""
    return PROJECT_ROOT / "data" / "exports" / course_id


def launch_enabled() -> bool:
    return os.getenv("TRAINING_LAUNCH_ENABLED", "").strip() == "1"


def tillicum_login() -> str:
    login = os.getenv("TILLICUM_LOGIN", "").strip()
    if login:
        return login
    user = os.getenv("USER", "").strip()
    return f"{user}@tillicum.hyak.uw.edu" if user else ""


def tillicum_repo_root() -> str:
    root = os.getenv("TILLICUM_REPO_ROOT", "").strip()
    if root:
        return root
    user = os.getenv("USER", "").strip()
    return f"/gpfs/projects/simswe/{user}/css360-syllabus-bot" if user else ""


@dataclass(frozen=True)
class LaunchCapability:
    """Whether this host can submit a job, and why not when it cannot."""

    enabled: bool
    reason: str


def describe_capability() -> LaunchCapability:
    if not launch_enabled():
        return LaunchCapability(
            enabled=False,
            reason=(
                "Training launch is disabled on this backend. Set "
                "TRAINING_LAUNCH_ENABLED=1 with working non-interactive SSH to "
                "the cluster to enable it."
            ),
        )
    if not SYNC_SCRIPT.exists():
        return LaunchCapability(
            enabled=False, reason="The training data sync script is missing."
        )
    if not tillicum_login() or not tillicum_repo_root():
        return LaunchCapability(
            enabled=False,
            reason="Cluster login or repository root is not configured.",
        )
    return LaunchCapability(enabled=True, reason="")


def validate_prepared_export(course_id: str) -> dict[str, int]:
    """Confirm this course's prepared train/validation files really exist.

    Delegates to the same helper the launcher scripts use, so a dataset the
    backend accepts is one the cluster scripts will also accept.
    """
    export_dir = course_export_dir(course_id)
    if not export_dir.is_dir():
        raise LaunchValidationError(
            "No prepared training data for this course. Prepare it first."
        )

    helpers = _load_helpers()
    try:
        counts = helpers.validate_course_export_dir(export_dir)
    except Exception as exc:  # helper raises ValueError/FileNotFoundError variants
        raise LaunchValidationError(f"Prepared training data is not usable: {exc}") from exc

    if int(counts.get("train_count", 0)) <= 0:
        raise LaunchValidationError("The prepared training set is empty.")
    return {
        "trainCount": int(counts.get("train_count", 0)),
        "validationCount": int(counts.get("validation_count", 0)),
    }


def _run(command: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
        check=False,
    )


@dataclass(frozen=True)
class LaunchResult:
    job_id: str
    mode: str
    submitted_at: str
    train_count: int
    validation_count: int
    already_active: bool


def launch_training(
    course_id: str,
    *,
    mode: str = "full",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    runner=_run,
) -> LaunchResult:
    """Sync one course's prepared data and submit its training job.

    `runner` is injected so tests exercise every branch without touching SSH or
    Slurm. Nothing in this function reaches the cluster on its own.
    """
    safe_course_id = assert_valid_course_id(course_id)

    if mode not in VALID_MODES:
        raise LaunchValidationError(f"Unknown training mode: {mode}")

    capability = describe_capability()
    if not capability.enabled:
        raise LaunchDisabledError(capability.reason)

    counts = validate_prepared_export(safe_course_id)

    # 1. Push only this course's export directory. The script itself refuses
    #    anything else and never uses --delete.
    sync = runner(
        [str(SYNC_SCRIPT), safe_course_id, "--yes"],
        timeout=timeout,
    )
    if sync.returncode != 0:
        raise LaunchExecutionError(
            f"Syncing training data failed: {(sync.stderr or sync.stdout).strip()[:500]}"
        )

    # 2. Submit on the cluster using the existing launcher, which owns the
    #    versioned output directory and the live-adapter guard.
    remote_command = (
        f"cd {tillicum_repo_root()} && "
        f"./{START_SCRIPT_RELATIVE} --course {safe_course_id} --{mode} --yes"
    )
    submit = runner(
        ["ssh", tillicum_login(), remote_command],
        timeout=timeout,
    )
    combined = f"{submit.stdout or ''}\n{submit.stderr or ''}"

    if submit.returncode != 0:
        raise LaunchExecutionError(
            f"Submitting the training job failed: {combined.strip()[:500]}"
        )

    match = _SBATCH_JOB_ID.search(combined)
    if not match:
        # The launcher exits 0 when an equally-named job is already active. That
        # is not a new submission and must not be reported as one.
        if _EXISTING_JOB.search(combined):
            raise LaunchExecutionError(
                "A training job for this job name is already active on the cluster."
            )
        raise LaunchExecutionError(
            "The training job did not report a Slurm job ID; nothing was submitted."
        )

    return LaunchResult(
        job_id=match.group(1),
        mode=mode,
        submitted_at=datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        train_count=counts["trainCount"],
        validation_count=counts["validationCount"],
        already_active=False,
    )
