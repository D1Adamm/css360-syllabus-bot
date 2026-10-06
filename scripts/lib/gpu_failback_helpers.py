"""What the classroom GPU mode switch and its failback watchdog share.

Stdlib only, and nothing here decides anything: the lock every writer of the
generation mode file holds, and the watchdog's own state file and log, which
`classroom_gpu_mode.sh status` reads.

The lock
--------
`scripts/classroom_gpu_mode.py` (`start`, `stop`) and
`scripts/gpu_failback_watchdog.py` both replace the mode file. An advisory
`flock` on `<mode file>.lock` makes "is the file still what I judged? then
replace it" one step, which is what stops the watchdog from overwriting a GPU
session a human activated while it was checking the old one. The lock is held
for the write only (milliseconds), never across a network call, and the
kernel releases it when its holder dies, so there is no stale lock to clean.

The state file and the log
--------------------------
`gpu-failback-state.json`: the last check, the current run of consecutive
failed checks (tied to one mode file by its fingerprint), and the last
automatic failback. Rewritten atomically on every check.

`gpu-failback.log`: one JSON object per line, appended only when the watchdog
switched the mode or finished a warm-up. Never rewritten.
"""

from __future__ import annotations

import datetime as dt
import errno
import fcntl
import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

TIMER_UNIT = "aiswe-gpu-failback.timer"
SERVICE_UNIT = "aiswe-gpu-failback.service"
STATE_DIR_ENV = "AISWE_GPU_FAILBACK_STATE_DIR"
STATE_FILE = "gpu-failback-state.json"
LOG_FILE = "gpu-failback.log"
RUN_LOCK_FILE = "gpu-failback.run.lock"
#: Written into the mode file's `switchedBy` when the watchdog switched.
SWITCHED_BY = "gpu-failback-watchdog"


class LockBusy(Exception):
    """Another process held the lock for longer than the caller would wait."""


def state_dir() -> Path:
    raw = (os.environ.get(STATE_DIR_ENV) or "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state") / "css360-syllabus-bot"


def state_path() -> Path:
    return state_dir() / STATE_FILE


def log_path() -> Path:
    return state_dir() / LOG_FILE


def lock_path(mode_file: Path) -> Path:
    return mode_file.with_name(mode_file.name + ".lock")


def fingerprint(text: str) -> str:
    """Identifies one mode file: one activation of GPU mode, to the byte."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@contextmanager
def file_lock(path: Path, *, timeout: float) -> Iterator[bool]:
    """Hold an exclusive `flock` on `path`. Raises LockBusy after `timeout` seconds.

    Yields False, without a lock, on a filesystem that cannot lock at all
    (some NFS mounts): the callers' own "unchanged?" check still applies, and
    refusing to switch back to the VM for want of a lock would be worse.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = True
                break
            except OSError as exc:
                if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                    held = False  # ENOLCK, ENOTSUP: no locking here
                    break
                if time.monotonic() >= deadline:
                    raise LockBusy(str(path)) from exc
                time.sleep(0.05)
        try:
            yield held
        finally:
            if held:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def mode_lock(mode_file: Path, *, timeout: float) -> Any:
    """The lock every writer of the generation mode file holds for its write."""
    return file_lock(lock_path(mode_file), timeout=timeout)


def run_lock(*, timeout: float = 0.0) -> Any:
    """One watchdog run at a time, so two runs never count the same failure twice."""
    return file_lock(state_dir() / RUN_LOCK_FILE, timeout=timeout)


def read_mode_file(gm: Any) -> tuple[Path, str | None, Any]:
    """(path, raw text or None, parsed GenerationMode or the error string).

    `gm` is the backend's `app.generation_mode`, so the scripts and the backend
    cannot disagree about what a mode file means.
    """
    path = gm.mode_file_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return path, None, gm.VM_MODE
    except OSError as exc:
        return path, None, f"cannot read: {exc}"
    try:
        return path, text, gm.parse_mode(text)
    except ValueError as exc:
        return path, text, f"invalid ({exc})"


def read_state() -> dict[str, Any]:
    """The watchdog's state; empty when absent or unreadable (a torn write, a first run)."""
    try:
        data = json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_state(state: dict[str, Any]) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".gpu-failback-state.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(state, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def append_log(record: dict[str, Any]) -> None:
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def read_log(limit: int = 20) -> list[dict[str, Any]]:
    try:
        lines = log_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records = []
    for line in lines[-limit:]:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def parse_time(value: Any) -> dt.datetime | None:
    """An `expiresAt`: ISO 8601 (with `Z` or an offset; naive means UTC) or epoch seconds."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return dt.datetime.fromtimestamp(float(value), dt.timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        moment = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=dt.timezone.utc)


def failback_record(mode_text: str | None) -> dict[str, Any]:
    """The `failback` object the watchdog left in a VM mode file, or {}."""
    if not mode_text:
        return {}
    try:
        data = json.loads(mode_text)
    except ValueError:
        return {}
    record = data.get("failback") if isinstance(data, dict) else None
    return record if isinstance(record, dict) else {}
