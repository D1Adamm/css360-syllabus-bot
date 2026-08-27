"""Durable local record of what this cluster did, and what it still owes UWB.

Stdlib only, Python 3.9 compatible: this runs on Tillicum login and compute
nodes.

Why the cluster keeps its own record
------------------------------------
Two failures made this necessary, and both actually happened.

The first: a job finished and the application never found out. `training_run`
sat at `submitted` and `model_request` at `training` for days, because the only
thing that could have told the backend was a person running a command. A
completion has to be reported by the job itself — but a job that ends at 02:00
against a backend that is briefly unreachable cannot retry forever, and must not
lose the event when it gives up. So the payload is written to disk *before* the
first send attempt, and removed only once the backend has accepted it. Anything
left in `pending/` is an event the application does not know about yet, and the
next worker run flushes it.

The second: an ambiguous network failure around submission. `sbatch` succeeds,
the `/submitted` call times out, and now the cluster has a real job the backend
has never heard of. Re-running the worker must not submit a second job for the
same run. The run record here — written immediately after `sbatch`, before the
backend is told — is what makes that recoverable: the mapping from run id to
Slurm job id to output directory exists locally even when the report did not
land, so the next run reconciles instead of duplicating.

Layout
------
    training/state/runs/<runId>.json         one per run this cluster submitted
    training/state/pending/<runId>-<kind>.json   callbacks not yet accepted

Both directories are gitignored (they are under `training/`), hold no secrets,
and are safe to delete: losing them costs reconciliation information, not data.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

STATE_DIRNAME = "state"
RUNS_DIRNAME = "runs"
PENDING_DIRNAME = "pending"

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class TrainingStateError(Exception):
    """The local state directory could not be read or written."""


def _safe_id(value: str, what: str) -> str:
    """Refuse anything that is not a plain identifier.

    These values become file names. They arrive from the backend and from Slurm
    rather than from a user, but a value that becomes a path is checked wherever
    it came from — the cost is one regex and the alternative is a payload that
    can name `../`.
    """
    candidate = (value or "").strip()
    if not SAFE_ID.match(candidate):
        raise TrainingStateError("Invalid {0}: {1!r}".format(what, value))
    return candidate


def state_root(repo_root: Path) -> Path:
    return Path(repo_root) / "training" / STATE_DIRNAME


def runs_dir(repo_root: Path) -> Path:
    return state_root(repo_root) / RUNS_DIRNAME


def pending_dir(repo_root: Path) -> Path:
    return state_root(repo_root) / PENDING_DIRNAME


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write a JSON file that is either wholly old or wholly new.

    A temp file in the destination directory, then `os.replace`. The point is
    the pending directory: a process killed mid-write must not leave a truncated
    callback payload that the next flush would either fail to parse or, worse,
    parse into something incomplete and send.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=".{0}.".format(path.name),
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, str(path))
    except OSError as exc:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise TrainingStateError(
            "Could not write {0}: {1}".format(path, exc)
        ) from exc


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


# --------------------------------------------------------------------------- #
# Run records
# --------------------------------------------------------------------------- #


def run_record_path(repo_root: Path, run_id: str) -> Path:
    return runs_dir(repo_root) / "{0}.json".format(_safe_id(run_id, "runId"))


def write_run_record(
    repo_root: Path,
    *,
    run_id: str,
    course_id: str,
    mode: str,
    job_id: Optional[str] = None,
    output_dir: Optional[str] = None,
    dataset_sha256: Optional[str] = None,
    train_examples: Optional[int] = None,
    validation_examples: Optional[int] = None,
    reported: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Record what this cluster knows about one run, merging onto any existing.

    Merged rather than replaced because the record is built in stages: the
    worker writes the run and its output directory before `sbatch`, the job id
    the moment `sbatch` returns, and `reported` only once the backend has
    acknowledged the submission. A later stage must not erase an earlier one.
    """
    path = run_record_path(repo_root, run_id)
    existing = read_json(path) or {}

    record: Dict[str, Any] = dict(existing)
    record.update(
        {
            "runId": run_id,
            "courseId": course_id,
            "mode": mode,
            "updatedAt": utc_now_iso(),
        }
    )
    record.setdefault("createdAt", record["updatedAt"])

    for key, value in (
        ("jobId", job_id),
        ("outputDir", output_dir),
        ("datasetSha256", dataset_sha256),
        ("trainExamples", train_examples),
        ("validationExamples", validation_examples),
    ):
        if value is not None:
            record[key] = value

    # Explicit rather than "only when true": the flag has to be able to go from
    # true back to false when a run is retried and its submission re-reported.
    record["reported"] = bool(reported)

    if extra:
        record.update(extra)

    write_json_atomic(path, record)
    return path


def read_run_record(repo_root: Path, run_id: str) -> Optional[Dict[str, Any]]:
    return read_json(run_record_path(repo_root, run_id))


def list_run_records(repo_root: Path) -> List[Dict[str, Any]]:
    directory = runs_dir(repo_root)
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.glob("*.json")):
        record = read_json(path)
        if record:
            records.append(record)
    return records


def unreported_submissions(repo_root: Path) -> List[Dict[str, Any]]:
    """Runs this cluster submitted whose submission the backend never confirmed.

    The recovery list for the ambiguous-network case: `sbatch` succeeded and
    `/submitted` did not. Each of these is a real Slurm job the application does
    not know about, and re-submitting instead of re-reporting would burn a
    second GPU allocation on work already running.
    """
    return [
        record
        for record in list_run_records(repo_root)
        if record.get("jobId") and not record.get("reported")
    ]


# --------------------------------------------------------------------------- #
# Pending callbacks
# --------------------------------------------------------------------------- #

#: What a pending report can be about.
#:
#: `published` describes a course and a version rather than a run, and is filed
#: under a synthetic key built from those two. It is here for the same reason
#: the other two are: the operation spans two machines, and the half that
#: already happened must survive the half that could not be delivered.
CALLBACK_KINDS = ("submitted", "completed", "published")


def pending_callback_path(repo_root: Path, run_id: str, kind: str) -> Path:
    safe_kind = _safe_id(kind, "callback kind")
    if safe_kind not in CALLBACK_KINDS:
        raise TrainingStateError("Unknown callback kind: {0!r}".format(kind))
    return pending_dir(repo_root) / "{0}-{1}.json".format(
        _safe_id(run_id, "runId"), safe_kind
    )


def queue_pending_callback(
    repo_root: Path,
    *,
    run_id: str,
    course_id: str,
    kind: str,
    payload: Dict[str, Any],
) -> Path:
    """Persist a callback that has not been accepted yet.

    Written before the first send attempt, not after the first failure. A
    process that dies between "decided to report" and "failed to report" is
    indistinguishable from one that never tried, and only the file on disk makes
    the difference recoverable.

    One file per (run, kind), overwritten: a second completion payload for the
    same run supersedes the first rather than queueing behind it. There is only
    one true answer to how a given run ended.
    """
    path = pending_callback_path(repo_root, run_id, kind)
    write_json_atomic(
        path,
        {
            "runId": run_id,
            "courseId": course_id,
            "kind": kind,
            "queuedAt": utc_now_iso(),
            "payload": payload,
        },
    )
    return path


def clear_pending_callback(repo_root: Path, run_id: str, kind: str) -> bool:
    """Drop a callback the backend has accepted. True when one was removed."""
    path = pending_callback_path(repo_root, run_id, kind)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise TrainingStateError(
            "Could not clear {0}: {1}".format(path, exc)
        ) from exc


def list_pending_callbacks(repo_root: Path) -> List[Dict[str, Any]]:
    """Everything the application has not been told, oldest first.

    Sorted by queue time so a submission is replayed before the completion of
    the same run — the backend accepts them in either order, but an operator
    reading the output should see the story in the order it happened.
    """
    directory = pending_dir(repo_root)
    if not directory.is_dir():
        return []

    entries = []
    for path in sorted(directory.glob("*.json")):
        record = read_json(path)
        if not record or record.get("kind") not in CALLBACK_KINDS:
            continue
        if not isinstance(record.get("payload"), dict):
            continue
        record["path"] = str(path)
        entries.append(record)

    return sorted(entries, key=lambda item: str(item.get("queuedAt") or ""))
