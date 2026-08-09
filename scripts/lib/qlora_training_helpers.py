#!/usr/bin/env python3
"""Pure helpers for QLoRA training automation scripts.

Stdlib only. Used by sync/start/promote helpers and unit tests.
"""

from __future__ import annotations

import argparse
import json
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

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, OSError, json.JSONDecodeError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
