#!/usr/bin/env python3
"""What is safe to reclaim under training_outputs/, and what never is.

Stdlib only, Python 3.9 compatible.

The problem this exists for is disk, not tidiness. Every full run writes
per-epoch checkpoints under `<run>/checkpoints/`, and a checkpoint of a
quantised 3B model with its optimizer state is orders of magnitude larger than
the 47 MB adapter anyone actually wants. After a term of runs, almost everything
under `training_outputs/` is checkpoints for adapters that already exist.

What this will propose deleting
-------------------------------
Only two things, and only ever from a completed run:

  - `<run>/checkpoints/` — intermediate Trainer state. The adapter is saved
    separately at `<run>/adapter/`, so removing checkpoints costs the ability to
    resume a finished run and nothing else.
  - smoke run directories in their entirety — a smoke run trains four examples
    for three optimizer steps. It is a rehearsal; its adapter is not a model
    anyone should be served and is never registered.

What it will never propose
--------------------------
  - anything under `serving/`. That is the published per-course adapter tree and
    the thing inference actually loads.
  - any `adapter/` directory of a full run. A registered model version points at
    one of these by reference, and the reference outlives whoever remembers
    which run produced it.
  - any run directory named by a `current.json` pointer.
  - any run this cluster still has an unreported submission or an undelivered
    completion for. Those are runs the application does not know about yet, and
    their output is the evidence.
  - a run that is not recognisably complete — no `runtime-report.json` means
    either a job still running or one that died, and neither is a candidate.

Everything is a proposal. Nothing here deletes: `plan_reclaimable` returns paths
and reasons, and the caller decides. The shell wrapper defaults to printing the
plan and requires an explicit flag to act on it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

COURSE_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
RUN_DIR_RE = re.compile(r"^(?P<stamp>[A-Za-z0-9._-]+)-(?P<mode>smoke|full)$")

QLORA_RUNS_DIRNAME = "qlora-runs"
SERVING_DIRNAME = "serving"
CHECKPOINTS_DIRNAME = "checkpoints"
ADAPTER_DIRNAME = "adapter"
RUNTIME_REPORT = "runtime-report.json"

#: Never proposed, whatever else is true of the path.
PROTECTED_DIRNAMES = frozenset({SERVING_DIRNAME, ADAPTER_DIRNAME, "adapter-backups"})


@dataclass(frozen=True)
class ReclaimCandidate:
    path: Path
    bytes: int
    kind: str
    reason: str


@dataclass(frozen=True)
class ProtectedRun:
    path: Path
    reason: str


def directory_size(path: Path) -> int:
    """Bytes under a directory. Unreadable entries count as zero.

    A permission error while sizing must not stop a plan being produced — the
    number is for an operator's judgement, not for a decision this module makes.
    """
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return "{0:.1f} {1}".format(value, unit)
        value /= 1024
    return "{0:.1f} TB".format(value)


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def published_source_refs(serving_root: Path) -> Set[str]:
    """Run references named by any course's `current.json`.

    A published pointer records the run its adapter was copied from. That run's
    output directory is the provenance of a model students are being served, so
    it is protected even though the served copy lives elsewhere.
    """
    refs: Set[str] = set()
    if not serving_root.is_dir():
        return refs

    for course_dir in serving_root.iterdir():
        if not course_dir.is_dir() or not COURSE_ID_RE.fullmatch(course_dir.name):
            continue
        pointer = read_json(course_dir / "current.json")
        if not pointer:
            continue
        source = pointer.get("sourceRef")
        if isinstance(source, str) and source.strip():
            # `qlora-runs/<course>/<run>-full/adapter` -> the run directory.
            trimmed = source.strip().rstrip("/")
            if trimmed.endswith("/" + ADAPTER_DIRNAME):
                trimmed = trimmed[: -(len(ADAPTER_DIRNAME) + 1)]
            refs.add(trimmed)
    return refs


def busy_run_ids(repo_root: Path) -> Set[str]:
    """Runs the cluster still owes the application a report for.

    Their output is the evidence for a submission or completion that has not
    landed, so nothing under them is reclaimable until it has.
    """
    busy: Set[str] = set()
    try:
        sys.path.insert(0, str(Path(repo_root) / "scripts" / "lib"))
        import training_state  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - the state module is optional here
        return busy

    for record in training_state.unreported_submissions(repo_root):
        run_id = record.get("runId")
        if isinstance(run_id, str):
            busy.add(run_id)
    for entry in training_state.list_pending_callbacks(repo_root):
        run_id = entry.get("runId")
        if isinstance(run_id, str):
            busy.add(run_id)
    return busy


def _run_is_complete(run_dir: Path) -> bool:
    return (run_dir / RUNTIME_REPORT).is_file()


def _output_ref(run_dir: Path, outputs_root: Path) -> str:
    try:
        return str(run_dir.relative_to(outputs_root)).replace("\\", "/")
    except ValueError:
        return run_dir.name


def _completion_run_id(run_dir: Path) -> Optional[str]:
    """The queue run id this directory belongs to, from its run-meta.env."""
    meta = run_dir / "run-meta.env"
    if not meta.is_file():
        return None
    try:
        text = meta.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("QUEUE_RUN_ID="):
            value = line.split("=", 1)[1].strip()
            return value or None
    return None


def plan_reclaimable(
    outputs_root: Path,
    *,
    repo_root: Optional[Path] = None,
    serving_root: Optional[Path] = None,
    include_smoke: bool = True,
) -> Dict[str, Any]:
    """Propose what could be reclaimed. Deletes nothing.

    Returns candidates and, just as importantly, the runs that were deliberately
    left alone with the reason — an operator reading a cleanup plan needs to see
    that the current model was considered and protected, not merely absent.
    """
    outputs_root = Path(outputs_root)
    runs_root = outputs_root / QLORA_RUNS_DIRNAME
    serving = (
        Path(serving_root) if serving_root is not None else outputs_root / SERVING_DIRNAME
    )

    protected_refs = published_source_refs(serving)
    busy = busy_run_ids(Path(repo_root)) if repo_root is not None else set()

    candidates: List[ReclaimCandidate] = []
    protected: List[ProtectedRun] = []

    if not runs_root.is_dir():
        return {
            "outputsRoot": str(outputs_root),
            "candidates": [],
            "protected": [],
            "reclaimableBytes": 0,
        }

    for course_dir in sorted(runs_root.iterdir()):
        if not course_dir.is_dir() or not COURSE_ID_RE.fullmatch(course_dir.name):
            continue

        for run_dir in sorted(course_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            match = RUN_DIR_RE.match(run_dir.name)
            if match is None:
                protected.append(
                    ProtectedRun(run_dir, "not a recognised run directory")
                )
                continue

            mode = match.group("mode")
            ref = _output_ref(run_dir, outputs_root)
            run_id = _completion_run_id(run_dir)

            if ref in protected_refs:
                protected.append(
                    ProtectedRun(run_dir, "a published course adapter came from here")
                )
                continue
            if run_id and run_id in busy:
                protected.append(
                    ProtectedRun(
                        run_dir, "this cluster still owes the application a report"
                    )
                )
                continue
            if not _run_is_complete(run_dir):
                protected.append(
                    ProtectedRun(
                        run_dir, "no runtime report — still running, or it died"
                    )
                )
                continue

            if mode == "smoke":
                if include_smoke:
                    candidates.append(
                        ReclaimCandidate(
                            path=run_dir,
                            bytes=directory_size(run_dir),
                            kind="smoke-run",
                            reason=(
                                "a smoke run trains four examples for three steps; "
                                "its adapter is a rehearsal and is never registered"
                            ),
                        )
                    )
                else:
                    protected.append(ProtectedRun(run_dir, "smoke runs excluded"))
                continue

            checkpoints = run_dir / CHECKPOINTS_DIRNAME
            if checkpoints.is_dir():
                candidates.append(
                    ReclaimCandidate(
                        path=checkpoints,
                        bytes=directory_size(checkpoints),
                        kind="checkpoints",
                        reason=(
                            "intermediate Trainer state; the adapter is saved "
                            "separately at adapter/ and is untouched"
                        ),
                    )
                )
            else:
                protected.append(ProtectedRun(run_dir, "nothing reclaimable"))

    return {
        "outputsRoot": str(outputs_root),
        "candidates": [
            {
                "path": str(item.path),
                "bytes": item.bytes,
                "human": human_bytes(item.bytes),
                "kind": item.kind,
                "reason": item.reason,
            }
            for item in candidates
        ],
        "protected": [
            {"path": str(item.path), "reason": item.reason} for item in protected
        ],
        "reclaimableBytes": sum(item.bytes for item in candidates),
    }


def is_deletion_safe(path: Path, outputs_root: Path) -> bool:
    """Final gate before anything is removed.

    Re-checked at deletion time rather than trusted from the plan: a plan can be
    printed, read, and acted on minutes later, and in between an adapter may
    have been published from a run that was a candidate.
    """
    resolved = Path(path).resolve()
    root = Path(outputs_root).resolve()

    if resolved == root or root not in resolved.parents:
        return False
    if any(part in PROTECTED_DIRNAMES for part in resolved.parts):
        return False
    if resolved.name == CHECKPOINTS_DIRNAME:
        return True
    match = RUN_DIR_RE.match(resolved.name)
    return match is not None and match.group("mode") == "smoke"


def _print_plan(plan: Dict[str, Any]) -> None:
    print("Training output retention plan")
    print("Root: {0}".format(plan["outputsRoot"]))
    print()

    if not plan["candidates"]:
        print("Nothing is reclaimable.")
    else:
        print("Reclaimable ({0} total):".format(human_bytes(plan["reclaimableBytes"])))
        for item in plan["candidates"]:
            print("  {0}  [{1}]".format(item["path"], item["human"]))
            print("    {0}".format(item["reason"]))

    if plan["protected"]:
        print()
        print("Deliberately left alone:")
        for item in plan["protected"]:
            print("  {0}".format(item["path"]))
            print("    {0}".format(item["reason"]))

    print()
    print("Never considered: serving/, every adapter/, and adapter-backups/.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs_root")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--serving-root", default=None)
    parser.add_argument(
        "--no-smoke",
        action="store_true",
        help="Leave smoke run directories alone as well.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    plan = plan_reclaimable(
        Path(args.outputs_root),
        repo_root=Path(args.repo_root) if args.repo_root else None,
        serving_root=Path(args.serving_root) if args.serving_root else None,
        include_smoke=not args.no_smoke,
    )

    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        _print_plan(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
