#!/usr/bin/env python3
"""Pure helpers for QLoRA training automation scripts.

Stdlib only. Used by sync/start/promote helpers and unit tests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

COURSE_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
UNSAFE_COURSE_CHARS = re.compile(r"[./\\[\]$]")
LIVE_ADAPTER_SUFFIX = "training_outputs/css-360-qlora/adapter"
LIVE_ADAPTER_PARENT_SUFFIX = "training_outputs/css-360-qlora"
MISSING_TRAINING_OUTPUT_DIR_MESSAGE = (
    "TRAINING_OUTPUT_DIR is required.\n"
    "Use ./training/start_qlora_training.sh --course <courseId> --smoke|--full\n"
    "to create a safe versioned training run."
)
ADAPTER_WEIGHT_NAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_model.pt",
)
# Historical course-agnostic Slurm names (pre course-isolation). Still recognized
# for status / same-course active checks via qlora-job-<id>.env metadata.
LEGACY_SLURM_JOB_NAMES = {
    "smoke": "css360-qlora-smoke",
    "full": "css360-qlora-train",
}
# Slurm truncates or rejects overly long names; keep a conservative ceiling.
SLURM_JOB_NAME_MAX_LEN = 64
_QLORA_JOB_NAME_RE = re.compile(r"^qlora-(smoke|train)-(.+)$")


def validate_course_id(course_id: str) -> str:
    value = course_id.strip()
    if not value:
        raise ValueError("Course ID must not be empty.")
    if UNSAFE_COURSE_CHARS.search(value) or ".." in value:
        raise ValueError(f'Invalid courseId "{value}": path-unsafe characters.')
    if value.startswith("-") or value.endswith("-"):
        raise ValueError(f'Invalid courseId "{value}": must not begin/end with a hyphen.')
    if not COURSE_ID_RE.fullmatch(value):
        raise ValueError(
            f'Invalid courseId "{value}": use lowercase letters, numbers, and hyphens only '
            "(example: css-360-winter-2026-a7rp)."
        )
    return value


def validate_instruction_response_jsonl(path: Path) -> int:
    """Validate JSONL; return example count. Never prints record contents."""
    if not path.is_file():
        raise ValueError(f"Missing JSONL file: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"JSONL file is empty: {path}")

    count = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"Blank line not allowed in {path} at line {line_number}")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON in {path.name} at line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"JSONL record must be an object in {path.name} at line {line_number}"
            )
        instruction = payload.get("instruction")
        response = payload.get("response")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(
                f"Missing/blank instruction in {path.name} at line {line_number}"
            )
        if not isinstance(response, str) or not response.strip():
            raise ValueError(
                f"Missing/blank response in {path.name} at line {line_number}"
            )
        count += 1

    if count < 1:
        raise ValueError(f"JSONL file has no examples: {path}")
    return count


def validate_course_export_dir(export_dir: Path) -> dict[str, int]:
    """Require train/validation/manifest and return counts."""
    if not export_dir.is_dir():
        raise ValueError(f"Export directory does not exist: {export_dir}")

    train_path = export_dir / "train.jsonl"
    val_path = export_dir / "validation.jsonl"
    manifest_path = export_dir / "manifest.json"
    for required in (train_path, val_path, manifest_path):
        if not required.is_file():
            raise ValueError(f"Required file missing: {required}")

    train_count = validate_instruction_response_jsonl(train_path)
    val_count = validate_instruction_response_jsonl(val_path)

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest.json is not valid JSON: {exc.msg}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object.")

    return {
        "train_count": train_count,
        "validation_count": val_count,
        "total_count": train_count + val_count,
    }


def utc_run_id(now: datetime | None = None) -> str:
    stamp = now or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    else:
        stamp = stamp.astimezone(timezone.utc)
    return stamp.strftime("%Y%m%dT%H%M%SZ")


def versioned_training_output_dir(
    *,
    user: str,
    course_id: str,
    run_id: str,
    mode: str,
    projects_root: str = "/gpfs/projects/simswe",
) -> str:
    """Build a versioned GPFS output directory (never the live adapter path)."""
    safe_user = user.strip()
    if not safe_user or "/" in safe_user or safe_user in {".", ".."}:
        raise ValueError(f"Invalid user for output path: {user!r}")
    safe_course = validate_course_id(course_id)
    safe_run = run_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", safe_run):
        raise ValueError(f"Invalid run id: {run_id!r}")
    mode_norm = mode.strip().lower()
    if mode_norm not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'")
    suffix = f"{safe_run}-{mode_norm}"
    path = (
        f"{projects_root.rstrip('/')}/{safe_user}/training_outputs/"
        f"qlora-runs/{safe_course}/{suffix}"
    )
    if LIVE_ADAPTER_SUFFIX in path.replace("\\", "/"):
        raise ValueError("Refusing to build a path that collides with the live adapter.")
    return path


# --------------------------------------------------------------------------- #
# Slurm wall-time policy
#
# The CSS 350 full run requested 8 hours and used 48 seconds — 0.01 GPU hours
# against an 8-hour reservation. That is not just untidy: an 8-hour request sits
# behind every other job the scheduler can fit in front of it, so the habit costs
# queue time on every run for a course of any size.
#
# The estimate is deliberately crude and deliberately generous. Optimizer steps
# are the only thing that scales with the dataset, the measured cost was about
# 2.3 seconds per step on this model, and the constant below is an order of
# magnitude above that. What the request has to survive is not a bad estimate of
# steady-state throughput — it is a cold Hugging Face download, a slow shared
# filesystem, a busy node, and per-epoch checkpoint writes. Those are what
# OVERHEAD covers, and they do not scale with the data.
#
# A ceiling is kept because a QOS has one and a request above it is rejected
# rather than queued. `full` is capped at the value that is known to work today;
# a genuinely larger course raises it explicitly rather than discovering the
# limit at submission time.
# --------------------------------------------------------------------------- #

#: Fixed overhead: model download/load, tokenizer, evaluation, checkpoint writes.
WALLTIME_OVERHEAD_SECONDS = 30 * 60

#: Per-optimizer-step allowance. Measured cost is ~2.3s on Llama-3.2-3B with
#: this configuration; this is ~9x that, so a node an order of magnitude slower
#: than the one we measured still finishes inside the request.
WALLTIME_SECONDS_PER_STEP = 20

#: Never ask for less than this for a full run, whatever the estimate says.
WALLTIME_FULL_FLOOR_SECONDS = 60 * 60

#: Never ask for more without an explicit override. This is the value the
#: existing `#SBATCH --time` used and is therefore known to be accepted.
WALLTIME_FULL_CEILING_SECONDS = 8 * 60 * 60

#: Smoke runs are capped at a fixed tiny number of optimizer steps, so their
#: cost is overhead and nothing else. Unchanged from the value the debug QOS has
#: been accepting.
WALLTIME_SMOKE_SECONDS = 45 * 60


def format_slurm_walltime(total_seconds: int) -> str:
    """Seconds as `HH:MM:SS`, rounded up to the next whole minute.

    Rounded up because a request is a limit: a job that needs 61 seconds and
    asks for 1 minute is killed at 60.
    """
    if total_seconds < 60:
        total_seconds = 60
    minutes = -(-int(total_seconds) // 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:00"


def parse_slurm_walltime(value: str) -> int:
    """`HH:MM:SS` or `MM:SS` or `HH:MM` -> seconds. Raises on anything else.

    Accepts only the shapes an operator would plausibly type for an override.
    Slurm itself accepts more (`days-hours`), but a value this cannot parse is
    one this cannot bound-check, and silently passing it through would defeat
    the ceiling.
    """
    text = (value or "").strip()
    if not re.fullmatch(r"\d{1,3}(:\d{2}){1,2}", text):
        raise ValueError(
            f"Invalid Slurm wall time {value!r}: expected HH:MM:SS or HH:MM."
        )
    parts = [int(part) for part in text.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        hours, minutes, seconds = parts[0], parts[1], 0
    if minutes > 59 or seconds > 59:
        raise ValueError(f"Invalid Slurm wall time {value!r}: minutes/seconds > 59.")
    return hours * 3600 + minutes * 60 + seconds


def estimate_training_walltime_seconds(
    *,
    mode: str,
    train_examples: int,
    epochs: float = 3.0,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    gpu_count: int = 1,
    ceiling_seconds: int = WALLTIME_FULL_CEILING_SECONDS,
) -> dict[str, Any]:
    """How much wall clock to request, and why.

    Returns the chosen value together with the inputs behind it, so the launcher
    can print a line an operator can check rather than a number they have to
    trust. `clamped` says which bound was hit, which is the one case worth
    reacting to: a full run pinned at the ceiling is a course that may genuinely
    need a raised limit.
    """
    mode_norm = _normalize_training_mode(mode)

    if mode_norm == "smoke":
        return {
            "mode": mode_norm,
            "seconds": WALLTIME_SMOKE_SECONDS,
            "walltime": format_slurm_walltime(WALLTIME_SMOKE_SECONDS),
            "optimizerSteps": None,
            "clamped": None,
            "reason": "smoke runs execute a fixed tiny step budget",
        }

    examples = max(1, int(train_examples))
    batch = max(1, int(per_device_batch_size)) * max(
        1, int(gradient_accumulation_steps)
    ) * max(1, int(gpu_count))
    steps_per_epoch = max(1, -(-examples // batch))
    steps = int(math.ceil(steps_per_epoch * max(0.1, float(epochs))))

    raw = WALLTIME_OVERHEAD_SECONDS + steps * WALLTIME_SECONDS_PER_STEP
    clamped = None
    seconds = raw
    if seconds < WALLTIME_FULL_FLOOR_SECONDS:
        seconds = WALLTIME_FULL_FLOOR_SECONDS
        clamped = "floor"
    if seconds > ceiling_seconds:
        seconds = ceiling_seconds
        clamped = "ceiling"

    return {
        "mode": mode_norm,
        "seconds": seconds,
        "walltime": format_slurm_walltime(seconds),
        "optimizerSteps": steps,
        "estimatedSeconds": raw,
        "clamped": clamped,
        "reason": (
            f"{steps} optimizer steps x {WALLTIME_SECONDS_PER_STEP}s + "
            f"{WALLTIME_OVERHEAD_SECONDS // 60}m overhead"
        ),
    }


TRAINING_OUTPUTS_MARKER = "training_outputs/"


def relative_training_output_ref(path: str | Path) -> str:
    """Turn an absolute run directory into a stored, machine-independent ref.

    ``/gpfs/projects/simswe/madamk/training_outputs/qlora-runs/css-350-.../
    20260827T064701Z-full`` becomes ``qlora-runs/css-350-.../20260827T064701Z-full``.

    The stored reference must not embed a cluster home directory or a username:
    admin surfaces display it, and it is written into the model registry where it
    long outlives the account it was produced under. Everything before
    ``training_outputs/`` is exactly the machine-specific part.

    A path with no ``training_outputs/`` segment is returned with any leading
    slash stripped rather than rejected — the caller validates separately, and a
    reference that is merely unusual is better recorded than dropped.
    """
    normalized = str(path).replace("\\", "/").rstrip("/")
    marker_at = normalized.rfind(TRAINING_OUTPUTS_MARKER)
    if marker_at >= 0:
        return normalized[marker_at + len(TRAINING_OUTPUTS_MARKER):]
    return normalized.lstrip("/")


def live_adapter_dir(*, user: str, projects_root: str = "/gpfs/projects/simswe") -> str:
    safe_user = user.strip()
    if not safe_user or "/" in safe_user:
        raise ValueError(f"Invalid user for live adapter path: {user!r}")
    return f"{projects_root.rstrip('/')}/{safe_user}/{LIVE_ADAPTER_SUFFIX}"


def is_live_adapter_path(path: str | Path, *, user: str | None = None) -> bool:
    normalized = str(path).replace("\\", "/").rstrip("/")
    if normalized.endswith(LIVE_ADAPTER_SUFFIX):
        return True
    if user:
        return normalized == live_adapter_dir(user=user).rstrip("/")
    return False


def is_live_adapter_tree(path: str | Path) -> bool:
    """True if path is the live adapter dir or anywhere under css-360-qlora/."""
    normalized = str(path).replace("\\", "/").rstrip("/")
    if normalized.endswith(LIVE_ADAPTER_PARENT_SUFFIX) or normalized.endswith(
        LIVE_ADAPTER_SUFFIX
    ):
        return True
    return f"/{LIVE_ADAPTER_PARENT_SUFFIX}/" in f"{normalized}/"


def require_training_output_dir(
    raw: str | None,
    *,
    user: str,
) -> str:
    """Require a non-empty TRAINING_OUTPUT_DIR that is not the live adapter tree.

    Used by smoke/train Slurm scripts so raw ``sbatch training/train.slurm``
    cannot silently overwrite the promoted inference adapter.
    """
    if raw is None or not str(raw).strip():
        raise ValueError(MISSING_TRAINING_OUTPUT_DIR_MESSAGE)
    path = str(raw).strip().rstrip("/")
    if is_live_adapter_tree(path) or is_live_adapter_path(path, user=user):
        live = live_adapter_dir(user=user)
        raise ValueError(
            "TRAINING_OUTPUT_DIR must not target the live inference adapter tree "
            f"({live}).\n"
            "Use ./training/start_qlora_training.sh --course <courseId> --smoke|--full\n"
            "and promote explicitly with ./training/promote_qlora_adapter.sh."
        )
    return path


def validate_adapter_source(path: Path) -> Path:
    """Ensure path looks like a completed PEFT adapter directory."""
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"Adapter source directory does not exist: {path}")
    config = resolved / "adapter_config.json"
    if not config.is_file():
        raise ValueError(f"Missing adapter_config.json in {resolved}")
    weight = next((resolved / name for name in ADAPTER_WEIGHT_NAMES if (resolved / name).is_file()), None)
    if weight is None:
        raise ValueError(
            "Missing adapter weight file "
            f"(expected one of: {', '.join(ADAPTER_WEIGHT_NAMES)}) in {resolved}"
        )
    if is_live_adapter_path(resolved):
        raise ValueError(
            "Source path is the live inference adapter; refusing to promote a path onto itself."
        )
    # Refuse promoting from backup roots accidentally named oddly — allow backups.
    return resolved


SERVING_SUBDIR = "training_outputs/serving"
VERSION_RE = re.compile(r"^v[0-9]+$")


def validate_model_version(version: str) -> str:
    value = (version or "").strip()
    if not VERSION_RE.fullmatch(value):
        raise ValueError(f"Invalid model version {version!r}: expected v1, v2, …")
    return value


def course_serving_dir(
    *,
    user: str,
    course_id: str,
    projects_root: str = "/gpfs/projects/simswe",
) -> str:
    """Where one course's published adapters live.

    Per course and per version, unlike the single `css-360-qlora/adapter` path
    this replaces. That path had no course in it at all, so publishing CSS 360
    silently replaced whatever CSS 350 was being served with — undetectable from
    a request, because a request carried no course either.
    """
    safe_user = user.strip()
    if not safe_user or "/" in safe_user or safe_user in {".", ".."}:
        raise ValueError(f"Invalid user for serving path: {user!r}")
    safe_course = validate_course_id(course_id)
    return f"{projects_root.rstrip('/')}/{safe_user}/{SERVING_SUBDIR}/{safe_course}"


def course_version_adapter_dir(
    *,
    user: str,
    course_id: str,
    version: str,
    projects_root: str = "/gpfs/projects/simswe",
) -> str:
    """The adapter directory for exactly one course and one version."""
    base = course_serving_dir(
        user=user, course_id=course_id, projects_root=projects_root
    )
    return f"{base}/{validate_model_version(version)}/adapter"


def backup_destination_dir(
    *,
    user: str,
    stamp: str | None = None,
    projects_root: str = "/gpfs/projects/simswe",
) -> str:
    safe_user = user.strip()
    if not safe_user or "/" in safe_user:
        raise ValueError(f"Invalid user for backup path: {user!r}")
    run_stamp = stamp or utc_run_id()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_stamp):
        raise ValueError(f"Invalid backup stamp: {run_stamp!r}")
    return (
        f"{projects_root.rstrip('/')}/{safe_user}/training_outputs/"
        f"adapter-backups/{run_stamp}"
    )


def parse_squeue_training_line(line: str) -> dict[str, str] | None:
    """Parse ``squeue -h -o '%i %j %t %N %M %L'``."""
    stripped = line.strip()
    if not stripped:
        return None
    parts = stripped.split()
    if len(parts) < 3:
        raise ValueError(f"Unexpected squeue line: {stripped!r}")
    job_id, name, state = parts[0], parts[1], parts[2]
    node = parts[3] if len(parts) >= 4 and not parts[3].startswith("(") else ""
    elapsed = parts[4] if len(parts) >= 5 else ""
    time_left = parts[5] if len(parts) >= 6 else ""
    return {
        "job_id": job_id,
        "name": name,
        "state": state,
        "node": node.split(",")[0] if node else "",
        "elapsed": elapsed,
        "time_left": time_left,
    }


def is_active_slurm_state(state: str) -> bool:
    return state.strip().upper() in {"PD", "PENDING", "R", "RUNNING", "CF", "CONFIGURING"}


def _normalize_training_mode(mode: str) -> str:
    mode_norm = mode.strip().lower()
    if mode_norm not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'")
    return mode_norm


def _mode_token(mode: str) -> str:
    """Slurm name token: smoke stays smoke; full uses historical 'train'."""
    mode_norm = _normalize_training_mode(mode)
    return "smoke" if mode_norm == "smoke" else "train"


def slurm_training_job_name(*, course_id: str, mode: str) -> str:
    """Course-scoped Slurm job name (course id + mode).

    Format: ``qlora-{smoke|train}-{courseId}``. Submitted via ``sbatch -J`` so
    train.slurm / smoke.slurm can keep static ``#SBATCH --job-name`` defaults.
    """
    safe_course = validate_course_id(course_id)
    token = _mode_token(mode)
    name = f"qlora-{token}-{safe_course}"
    if len(name) > SLURM_JOB_NAME_MAX_LEN:
        raise ValueError(
            f"Slurm job name exceeds {SLURM_JOB_NAME_MAX_LEN} characters: {name!r}"
        )
    return name


def legacy_slurm_training_job_name(mode: str) -> str:
    """Pre-isolation job name (course-agnostic)."""
    return LEGACY_SLURM_JOB_NAMES[_normalize_training_mode(mode)]


def is_qlora_training_job_name(name: str) -> bool:
    """True for course-scoped or legacy QLoRA smoke/train Slurm names."""
    stripped = name.strip()
    if stripped in LEGACY_SLURM_JOB_NAMES.values():
        return True
    match = _QLORA_JOB_NAME_RE.fullmatch(stripped)
    if not match:
        return False
    try:
        validate_course_id(match.group(2))
    except ValueError:
        return False
    return True


def course_id_from_slurm_job_name(name: str) -> str | None:
    """Extract course id from a course-scoped job name; None for legacy names."""
    match = _QLORA_JOB_NAME_RE.fullmatch(name.strip())
    if not match:
        return None
    try:
        return validate_course_id(match.group(2))
    except ValueError:
        return None


def log_prefix_for_slurm_job_name(name: str) -> str | None:
    """Return ``smoke`` or ``train`` log-file prefix for a recognized job name."""
    stripped = name.strip()
    if stripped == LEGACY_SLURM_JOB_NAMES["smoke"] or stripped.startswith("qlora-smoke-"):
        if is_qlora_training_job_name(stripped):
            return "smoke"
        return None
    if stripped == LEGACY_SLURM_JOB_NAMES["full"] or stripped.startswith("qlora-train-"):
        if is_qlora_training_job_name(stripped):
            return "train"
        return None
    return None


def read_course_id_from_job_meta(meta_path: Path) -> str | None:
    """Read COURSE_ID from a ``qlora-job-<id>.env`` file if present."""
    if not meta_path.is_file():
        return None
    try:
        text = meta_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("COURSE_ID="):
            raw = line.split("=", 1)[1].strip()
            if not raw:
                return None
            try:
                return validate_course_id(raw)
            except ValueError:
                return raw
    return None


def select_active_training_job_line(
    lines: list[str] | tuple[str, ...],
    *,
    course_id: str,
    mode: str,
    meta_dir: Path | None = None,
) -> dict[str, str] | None:
    """Pick the first active squeue line that belongs to this course + mode.

    Matches the course-scoped job name always. Legacy course-agnostic names only
    match when ``qlora-job-<id>.env`` records the same COURSE_ID — so a CSS360
    legacy job never blocks CSS490/CSS350.
    """
    safe_course = validate_course_id(course_id)
    expected = slurm_training_job_name(course_id=safe_course, mode=mode)
    legacy = legacy_slurm_training_job_name(mode)

    for raw in lines:
        parsed = parse_squeue_training_line(raw)
        if parsed is None:
            continue
        if not is_active_slurm_state(parsed["state"]):
            continue
        name = parsed["name"]
        if name == expected:
            return parsed
        if name != legacy:
            continue
        if meta_dir is None:
            continue
        meta_course = read_course_id_from_job_meta(
            meta_dir / f"qlora-job-{parsed['job_id']}.env"
        )
        if meta_course == safe_course:
            return parsed
    return None


def _cli_validate_course(args: argparse.Namespace) -> int:
    print(validate_course_id(args.course_id))
    return 0


def _cli_validate_export(args: argparse.Namespace) -> int:
    counts = validate_course_export_dir(Path(args.path))
    print(json.dumps(counts))
    return 0


def _cli_versioned_outdir(args: argparse.Namespace) -> int:
    print(
        versioned_training_output_dir(
            user=args.user,
            course_id=args.course_id,
            run_id=args.run_id,
            mode=args.mode,
        )
    )
    return 0


def _cli_validate_adapter(args: argparse.Namespace) -> int:
    print(str(validate_adapter_source(Path(args.path))))
    return 0


def _cli_course_serving_dir(args: argparse.Namespace) -> int:
    print(
        course_serving_dir(user=args.user, course_id=args.course_id)
    )
    return 0


def _cli_course_version_adapter_dir(args: argparse.Namespace) -> int:
    print(
        course_version_adapter_dir(
            user=args.user, course_id=args.course_id, version=args.version
        )
    )
    return 0


def _cli_training_walltime(args: argparse.Namespace) -> int:
    ceiling = (
        parse_slurm_walltime(args.ceiling)
        if args.ceiling
        else WALLTIME_FULL_CEILING_SECONDS
    )
    plan = estimate_training_walltime_seconds(
        mode=args.mode,
        train_examples=args.train_examples,
        epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ceiling_seconds=ceiling,
    )
    if args.field:
        print(plan[args.field])
    else:
        print(json.dumps(plan))
    return 0


def _cli_relative_output_ref(args: argparse.Namespace) -> int:
    print(relative_training_output_ref(args.path))
    return 0


def _cli_is_live_adapter(args: argparse.Namespace) -> int:
    print("yes" if is_live_adapter_path(args.path, user=args.user) else "no")
    return 0 if is_live_adapter_path(args.path, user=args.user) else 1


def _cli_require_training_output_dir(args: argparse.Namespace) -> int:
    # Prefer explicit arg; otherwise read the process environment (Slurm usage).
    raw = args.path if args.path is not None else os.environ.get("TRAINING_OUTPUT_DIR")
    print(require_training_output_dir(raw, user=args.user))
    return 0


def _cli_parse_squeue(args: argparse.Namespace) -> int:
    parsed = parse_squeue_training_line(sys.stdin.readline())
    if parsed is None:
        raise ValueError("Empty squeue line.")
    if args.field:
        print(parsed[args.field])
    else:
        print(json.dumps(parsed))
    return 0


def _cli_slurm_job_name(args: argparse.Namespace) -> int:
    print(slurm_training_job_name(course_id=args.course_id, mode=args.mode))
    return 0


def _cli_legacy_slurm_job_name(args: argparse.Namespace) -> int:
    print(legacy_slurm_training_job_name(args.mode))
    return 0


def _cli_is_qlora_training_job_name(args: argparse.Namespace) -> int:
    ok = is_qlora_training_job_name(args.name)
    print("yes" if ok else "no")
    return 0 if ok else 1


def _cli_log_prefix_for_job_name(args: argparse.Namespace) -> int:
    prefix = log_prefix_for_slurm_job_name(args.name)
    if prefix is None:
        raise ValueError(f"Not a recognized QLoRA training job name: {args.name!r}")
    print(prefix)
    return 0


def _cli_select_active_training_job(args: argparse.Namespace) -> int:
    lines = [line.rstrip("\n") for line in sys.stdin]
    meta_dir = Path(args.meta_dir) if args.meta_dir else None
    selected = select_active_training_job_line(
        lines,
        course_id=args.course_id,
        mode=args.mode,
        meta_dir=meta_dir,
    )
    if selected is None:
        return 1
    if args.field:
        print(selected[args.field])
    else:
        # Reconstruct a stable squeue-style line for shell consumers.
        print(
            f"{selected['job_id']} {selected['name']} {selected['state']} "
            f"{selected['node'] or '(null)'} {selected['elapsed'] or '0:00'} "
            f"{selected['time_left'] or '0:00'}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate-course-id")
    p.add_argument("course_id")
    p.set_defaults(func=_cli_validate_course)

    p = sub.add_parser("validate-export-dir")
    p.add_argument("path")
    p.set_defaults(func=_cli_validate_export)

    p = sub.add_parser("versioned-outdir")
    p.add_argument("--user", required=True)
    p.add_argument("--course-id", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--mode", choices=("smoke", "full"), required=True)
    p.set_defaults(func=_cli_versioned_outdir)

    p = sub.add_parser("validate-adapter-source")
    p.add_argument("path")
    p.set_defaults(func=_cli_validate_adapter)

    p = sub.add_parser(
        "course-serving-dir",
        help="Where one course's published adapters live",
    )
    p.add_argument("--user", required=True)
    p.add_argument("--course-id", required=True)
    p.set_defaults(func=_cli_course_serving_dir)

    p = sub.add_parser(
        "course-version-adapter-dir",
        help="The adapter directory for one course and one version",
    )
    p.add_argument("--user", required=True)
    p.add_argument("--course-id", required=True)
    p.add_argument("--version", required=True)
    p.set_defaults(func=_cli_course_version_adapter_dir)

    p = sub.add_parser(
        "training-walltime",
        help="Choose a Slurm --time value from mode and dataset size",
    )
    p.add_argument("--mode", choices=("smoke", "full"), required=True)
    p.add_argument("--train-examples", type=int, default=1)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument(
        "--ceiling",
        default=None,
        help="Override the maximum request, e.g. 16:00:00",
    )
    p.add_argument(
        "--field",
        choices=("walltime", "seconds", "optimizerSteps", "clamped", "reason"),
        default=None,
    )
    p.set_defaults(func=_cli_training_walltime)

    p = sub.add_parser(
        "relative-output-ref",
        help="Strip the machine-specific prefix from a training output path",
    )
    p.add_argument("path")
    p.set_defaults(func=_cli_relative_output_ref)

    p = sub.add_parser("is-live-adapter")
    p.add_argument("path")
    p.add_argument("--user", default=None)
    p.set_defaults(func=_cli_is_live_adapter)

    p = sub.add_parser(
        "require-training-output-dir",
        help="Validate TRAINING_OUTPUT_DIR (arg or env); refuse live adapter tree",
    )
    p.add_argument("--user", required=True)
    p.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Optional path; defaults to $TRAINING_OUTPUT_DIR",
    )
    p.set_defaults(func=_cli_require_training_output_dir)

    p = sub.add_parser("parse-squeue-line")
    p.add_argument(
        "--field",
        choices=("job_id", "name", "state", "node", "elapsed", "time_left"),
        default=None,
    )
    p.set_defaults(func=_cli_parse_squeue)

    p = sub.add_parser(
        "slurm-job-name",
        help="Build course-scoped Slurm job name for smoke/full training",
    )
    p.add_argument("--course-id", required=True)
    p.add_argument("--mode", choices=("smoke", "full"), required=True)
    p.set_defaults(func=_cli_slurm_job_name)

    p = sub.add_parser(
        "legacy-slurm-job-name",
        help="Historical course-agnostic Slurm job name for a mode",
    )
    p.add_argument("--mode", choices=("smoke", "full"), required=True)
    p.set_defaults(func=_cli_legacy_slurm_job_name)

    p = sub.add_parser(
        "is-qlora-training-job-name",
        help="Exit 0 if name is a recognized QLoRA smoke/train job name",
    )
    p.add_argument("name")
    p.set_defaults(func=_cli_is_qlora_training_job_name)

    p = sub.add_parser(
        "log-prefix-for-job-name",
        help="Print smoke|train log prefix for a recognized job name",
    )
    p.add_argument("name")
    p.set_defaults(func=_cli_log_prefix_for_job_name)

    p = sub.add_parser(
        "select-active-training-job",
        help="Read squeue lines on stdin; print the active line for course+mode",
    )
    p.add_argument("--course-id", required=True)
    p.add_argument("--mode", choices=("smoke", "full"), required=True)
    p.add_argument(
        "--meta-dir",
        default=None,
        help="Directory with qlora-job-<id>.env (needed for legacy name matching)",
    )
    p.add_argument(
        "--field",
        choices=("job_id", "name", "state", "node", "elapsed", "time_left"),
        default=None,
    )
    p.set_defaults(func=_cli_select_active_training_job)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, OSError, json.JSONDecodeError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
