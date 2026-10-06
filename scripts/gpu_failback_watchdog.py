#!/usr/bin/env python3
"""Automatic GPU -> VM failback for classroom GPU mode. One check per run.

    ./scripts/gpu_failback_watchdog.sh run [--dry-run]

Run every 15 seconds by the `aiswe-gpu-failback.timer` user unit
(`./scripts/gpu_failback_watchdog.sh install`). It protects one thing: a class
must not be left pointing at a GPU that is gone. That happened once: the
Tillicum job expired, the mode file still said GPU, and every question failed
until someone ran `stop`.

What one run does
-----------------
VM mode (or no mode file, or one the backend ignores): nothing.

GPU mode: fail back to the VM when either

  - the allocation is over: the mode file's `expiresAt` is within
    `--expiry-margin` seconds of now or past it, or the GPU service itself
    reports that little time left. Reason `expired`, acted on at once.
  - the GPU does not answer `/health` through the tunnel, or answers in a
    state the backend could not use (still loading, another build's decoding,
    the course no longer published), on `--failures` consecutive runs.
    Reasons `gpu_unreachable` and `gpu_unhealthy`. One failed check alone
    changes nothing; a healthy check starts the count again.

Failing back is, in this order:

  1. replace the mode file with VM mode: one atomic write, under the lock
     `start` and `stop` write under, and only if the file is still, byte for
     byte, the one that was judged. New requests generate on the VM from that
     moment. A GPU session a human activated meanwhile is left alone.
  2. log it (`gpu-failback.log`, and the journal).
  3. load the VM's base, embedding and course models, as `stop` does, and log
     whether that worked. If it did not, the mode stays VM.

What it never does: submit, restart or cancel a Tillicum job, open or close a
tunnel, sign in, or switch *to* GPU mode. After a failback, `stop` still does
the full manual check (four conditions, tunnel, the job reminder).

Safe to run at any time and any number of times. Exit status: 0, or 1 when a
failback's VM warm-up failed.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib.util
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND = REPO_ROOT / "backend"
LIB = REPO_ROOT / "scripts" / "lib"

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _load(name: str, path: Path):
    """Load a script as a module, once: every user shares the one module object."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


helpers = _load("gpu_failback_helpers", LIB / "gpu_failback_helpers.py")

# The backend's own rules for what a mode file means. Nothing heavier is
# imported until GPU mode is actually on: in VM mode, which is nearly always,
# a run is a file read.
from app import generation_mode as gm  # noqa: E402

DEFAULT_FAILURES = 3
DEFAULT_HEALTH_TIMEOUT = 5.0
DEFAULT_EXPIRY_MARGIN = 30.0
#: A failed check extends the run of failures only if the previous one was
#: this recent; a count left over from an hour ago is not "consecutive".
DEFAULT_STREAK_WINDOW = 120.0
DEFAULT_LOCK_TIMEOUT = 10.0
DEFAULT_KEEP_ALIVE = "4h"
DEFAULT_WARM_TIMEOUT = 120.0
#: A warm-up interrupted by a restart is finished by the next run, once.
MAX_WARMUP_ATTEMPTS = 2

REASON_EXPIRED = "expired"
REASON_UNREACHABLE = "gpu_unreachable"
REASON_UNHEALTHY = "gpu_unhealthy"

_LEVELS = {"error": 3, "warning": 4, "info": 6}


@dataclass
class Settings:
    failures: int = DEFAULT_FAILURES
    health_timeout: float = DEFAULT_HEALTH_TIMEOUT
    expiry_margin: float = DEFAULT_EXPIRY_MARGIN
    streak_window: float = DEFAULT_STREAK_WINDOW
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT
    keep_alive: str = DEFAULT_KEEP_ALIVE
    warm_timeout: float = DEFAULT_WARM_TIMEOUT
    dry_run: bool = False


def emit(level: str, message: str) -> None:
    """One line. Under systemd, with the priority prefix the journal reads."""
    prefix = f"<{_LEVELS[level]}>" if os.environ.get("JOURNAL_STREAM") else ""
    stream = sys.stdout if level == "info" else sys.stderr
    print(f"{prefix}gpu-failback: {message}", file=stream, flush=True)


def iso(moment: dt.datetime) -> str:
    return moment.astimezone(dt.timezone.utc).isoformat(timespec="seconds")


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def switch_script() -> Any:
    """scripts/classroom_gpu_mode.py: the mode file writer, the GPU checks, the VM warm-up."""
    return _load("classroom_gpu_mode", REPO_ROOT / "scripts" / "classroom_gpu_mode.py")


def save(state: dict[str, Any]) -> bool:
    try:
        helpers.write_state(state)
        return True
    except OSError as exc:
        emit("error", f"could not write {helpers.state_path()}: {exc}")
        return False


def log_event(record: dict[str, Any]) -> None:
    try:
        helpers.append_log(record)
    except OSError as exc:
        emit("error", f"could not append to {helpers.log_path()}: {exc}")


def current_streak(state: dict[str, Any], fingerprint: str, now: dt.datetime, window: float) -> int:
    """Consecutive failed checks so far, against this mode file, recent enough to count."""
    streak = state.get("streak")
    if not isinstance(streak, dict) or streak.get("fingerprint") != fingerprint:
        return 0
    last = helpers.parse_time(streak.get("lastFailureAt"))
    if last is None or (now - last).total_seconds() > window:
        return 0
    try:
        return max(0, int(streak.get("failures") or 0))
    except (TypeError, ValueError):
        return 0


def health_problems(cgm: Any, health: dict[str, Any], details: dict[str, Any]) -> list[str]:
    """Why the backend could not generate on this GPU service right now; empty when it can."""
    course, version = details.get("courseId"), details.get("modelVersion")
    try:
        if course and version:
            # What `start` required of the service, less the time-left margin.
            return cgm.gpu_health_problems(health, course_id=str(course), version=str(version), min_minutes=0)
        # A mode file that names no course (written by hand): judge the service alone.
        if health.get("status") != "ok" or health.get("adapterLoaded") is not True:
            return [f"status={health.get('status')!r}"]
        return []
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return [f"malformed /health answer ({exc!r})"]


def warm_vm(cgm: Any, settings: Settings, course_id: str, version: str) -> tuple[bool, list[str]]:
    """Load the VM models `stop` loads. (everything resident?, the lines worth logging)."""
    base_model, tag, finetuned_ollama = cgm.vm_model_names(course_id, version)
    problems: list[str] = []
    if tag is None:
        course = f"{course_id} {version}".strip() or "no course was recorded, so none"
        problems.append(f"{course} is not mapped in {cgm.FINETUNED_ENV}: the fine-tuned model was not warmed")
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            warmed = cgm.warm_vm_models(cgm.Config(), base_model=base_model, finetuned_ollama=finetuned_ollama,
                                        finetuned_tag=tag, keep_alive=settings.keep_alive,
                                        timeout=settings.warm_timeout)
    except Exception as exc:  # noqa: BLE001 - the mode is already VM; a crash here is logged, not raised
        warmed = False
        problems.append(f"the warm-up crashed: {exc!r}")
    lines = [line for line in output.getvalue().splitlines() if line.startswith(("ok ", "PASS", "FAIL"))]
    problems += [line for line in lines if line.startswith("FAIL")]
    if not warmed and not problems:
        problems.append("warming the VM models failed")
    return (warmed and tag is not None), (problems or lines)


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #


def run_once(settings: Settings, *, transport: Any = None, clock: Callable[[], dt.datetime] = utc_now,
             warm: Callable[..., tuple[bool, list[str]]] | None = None) -> int:
    now = clock()
    _, text, mode = helpers.read_mode_file(gm)
    state = helpers.read_state()

    if not isinstance(mode, gm.GenerationMode) or not mode.is_gpu or text is None:
        return _vm_mode(settings, state, text, now, clock, warm)

    cgm = switch_script()
    transport = transport or cgm.checks.Transport()
    details = mode.details
    fingerprint = helpers.fingerprint(text)
    session = f"job {details.get('jobId') or '?'} on {details.get('node') or '?'}"

    expires = helpers.parse_time(details.get("expiresAt"))
    if expires is not None and (expires - now).total_seconds() <= settings.expiry_margin:
        left = (expires - now).total_seconds()
        detail = (f"the allocation ended at {iso(expires)}" if left <= 0
                  else f"the allocation ends at {iso(expires)}, in {left:.0f}s")
        return _fail_back(settings, cgm, state, text=text, mode=mode, reason=REASON_EXPIRED, detail=detail,
                          failures=0, now=now, clock=clock, warm=warm)

    url = f"{mode.gpu_url}/health"
    result = transport.request("GET", url, timeout=settings.health_timeout)
    health = result.body if result.ok and isinstance(result.body, dict) else None
    remaining = health.get("secondsRemaining") if health else None
    if isinstance(remaining, (int, float)) and not isinstance(remaining, bool) and remaining <= settings.expiry_margin:
        return _fail_back(settings, cgm, state, text=text, mode=mode, reason=REASON_EXPIRED,
                          detail=f"the GPU service reports {remaining:.0f}s left on its allocation",
                          failures=0, now=now, clock=clock, warm=warm)

    if health is None:
        problems = [f"no answer from {url} ({result.error or 'HTTP ' + str(result.status)})"]
    else:
        problems = health_problems(cgm, health, details)
    if not problems:
        left = f", {remaining / 60:.0f} min left" if isinstance(remaining, (int, float)) else ""
        state["streak"] = None
        state["lastCheck"] = {"at": iso(now), "mode": gm.GPU, "outcome": "healthy",
                              "detail": f"GPU mode, GPU healthy ({session}{left})"}
        if not settings.dry_run:
            save(state)
        emit("info", state["lastCheck"]["detail"])
        return 0

    failures = current_streak(state, fingerprint, now, settings.streak_window) + 1
    problem = "; ".join(problems)[:500]
    state["streak"] = {"fingerprint": fingerprint, "failures": failures, "lastFailureAt": iso(now), "problem": problem}
    if failures < settings.failures:
        state["lastCheck"] = {"at": iso(now), "mode": gm.GPU, "outcome": "check_failed",
                              "detail": f"GPU mode, GPU check FAILED {failures}/{settings.failures} ({session}): {problem}"}
        if not settings.dry_run:
            save(state)
        emit("warning", state["lastCheck"]["detail"] + f"; failing back to the VM after {settings.failures} in a row")
        return 0

    return _fail_back(settings, cgm, state, text=text, mode=mode,
                      reason=REASON_UNREACHABLE if health is None else REASON_UNHEALTHY,
                      detail=f"{failures} consecutive failed checks: {problem}",
                      failures=failures, now=now, clock=clock, warm=warm)


def _vm_mode(settings: Settings, state: dict[str, Any], text: str | None, now: dt.datetime,
             clock: Callable[[], dt.datetime], warm: Any) -> int:
    """Nothing to protect. The one thing left to do: a warm-up an earlier run was killed in."""
    state["streak"] = None
    state["lastCheck"] = {"at": iso(now), "mode": gm.VM, "outcome": "vm_mode", "detail": "VM mode, nothing to do"}
    last = state.get("lastFailback")
    if isinstance(last, dict) and last.get("vmWarmup") == "pending" and not settings.dry_run:
        ours = text is not None and helpers.fingerprint(text) == last.get("modeFingerprint")
        if ours and int(last.get("warmupAttempts") or 0) < MAX_WARMUP_ATTEMPTS:
            emit("warning", f"finishing the VM warm-up of the failback at {last.get('at')}, which an earlier run did not complete")
            return _warm_up(settings, switch_script(), state, clock, warm)
        # A human has switched since (their `start`/`stop` did its own checks),
        # or the warm-up keeps being interrupted: record that it never finished.
        last["vmWarmup"] = "failed"
        last["vmWarmupDetail"] = ["the warm-up did not finish (interrupted" + ("" if ours else "; the mode was switched by hand since") + ")"]
        log_event({"event": "failback_warmup", "at": iso(now), "failbackAt": last.get("at"), "reason": last.get("reason"),
                   "jobId": last.get("jobId"), "node": last.get("node"), "ok": False, "detail": last["vmWarmupDetail"]})
        emit("error", f"the VM warm-up of the failback at {last.get('at')} did not finish; the mode is VM")
    if not settings.dry_run:
        save(state)
    emit("info", "VM mode, nothing to do")
    return 0


def _fail_back(settings: Settings, cgm: Any, state: dict[str, Any], *, text: str, mode: Any, reason: str, detail: str,
               failures: int, now: dt.datetime, clock: Callable[[], dt.datetime], warm: Any) -> int:
    details = mode.details
    record = {
        "at": iso(now), "reason": reason, "detail": detail,
        "jobId": str(details.get("jobId") or ""), "node": str(details.get("node") or ""),
        "courseId": str(details.get("courseId") or ""), "modelVersion": str(details.get("modelVersion") or ""),
        "expiresAt": str(details.get("expiresAt") or ""), "gpuUrl": mode.gpu_url,
        "gpuModeSince": str(details.get("since") or ""), "consecutiveFailures": failures,
    }
    session = f"job {record['jobId'] or '?'} on {record['node'] or '?'}, {record['courseId'] or '?'}@{record['modelVersion'] or '?'}"

    if settings.dry_run:
        emit("warning", f"DRY RUN: would fail back to the VM now ({reason}: {detail}; {session}). Nothing was changed.")
        return 0

    new_text = json.dumps({"mode": gm.VM, "since": iso(now), "switchedBy": helpers.SWITCHED_BY, "failback": record},
                          indent=2) + "\n"
    if gm.parse_mode(new_text).is_gpu:  # never write anything but VM mode
        raise RuntimeError("refusing to write a mode file that is not VM mode")

    # 1. The switch: before anything slow, and only over the file that was judged.
    try:
        switched = cgm.replace_mode_file_if_unchanged(text, new_text, lock_timeout=settings.lock_timeout)
    except cgm.failback.LockBusy:
        state["lastCheck"] = {"at": iso(now), "mode": gm.GPU, "outcome": "deferred",
                              "detail": f"GPU mode, failback due ({reason}) but the mode file is locked by a manual command; retrying next run"}
        save(state)
        emit("warning", state["lastCheck"]["detail"])
        return 0
    if not switched:
        state["streak"] = None
        state["lastCheck"] = {"at": iso(now), "mode": gm.GPU, "outcome": "superseded",
                              "detail": "the mode file was changed by a manual start/stop during the check; left as it is"}
        save(state)
        emit("warning", f"failback due ({reason}), but " + state["lastCheck"]["detail"])
        return 0

    # 2. The record: written before the warm-up, so a run killed during it leaves a trace.
    state["streak"] = None
    state["lastFailback"] = {**record, "vmWarmup": "pending", "warmupAttempts": 0,
                             "modeFingerprint": helpers.fingerprint(new_text)}
    state["lastCheck"] = {"at": iso(now), "mode": gm.VM, "outcome": "failback",
                          "detail": f"FAILED BACK to the VM ({reason})"}
    log_event({"event": "failback", **record})
    save(state)
    emit("warning", f"AUTOMATIC FAILBACK: generation switched to the VM. Reason: {reason} ({detail}). Was: {session}. "
                    "The Tillicum job and the tunnel were not touched.")

    # 3. The VM models.
    return _warm_up(settings, cgm, state, clock, warm)


def _warm_up(settings: Settings, cgm: Any, state: dict[str, Any], clock: Callable[[], dt.datetime], warm: Any) -> int:
    last = state["lastFailback"]
    last["warmupAttempts"] = int(last.get("warmupAttempts") or 0) + 1
    save(state)
    ok, lines = (warm or warm_vm)(cgm, settings, str(last.get("courseId") or ""), str(last.get("modelVersion") or ""))
    finished = iso(clock())
    last["vmWarmup"] = "ok" if ok else "failed"
    last["vmWarmupAt"] = finished
    last["vmWarmupDetail"] = [str(line)[:300] for line in lines][:10]
    log_event({"event": "failback_warmup", "at": finished, "failbackAt": last.get("at"), "reason": last.get("reason"),
               "jobId": last.get("jobId"), "node": last.get("node"), "courseId": last.get("courseId"),
               "modelVersion": last.get("modelVersion"), "ok": ok, "detail": last["vmWarmupDetail"]})
    save(state)
    if ok:
        emit("warning", "VM warm-up after the failback succeeded: base, embedding and course models are resident.")
        return 0
    emit("error", "VM warm-up after the failback FAILED (the mode stays VM): " + "; ".join(last["vmWarmupDetail"]))
    return 1


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def _env(name: str, default: Any, cast: Callable[[str], Any]) -> Any:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        emit("error", f"ignoring {name}={raw!r} (not a valid value); using {default}")
        return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--failures", type=int, default=_env("AISWE_GPU_FAILBACK_FAILURES", DEFAULT_FAILURES, int),
                        help="Consecutive failed health checks before failing back (default %(default)s).")
    parser.add_argument("--health-timeout", type=float,
                        default=_env("AISWE_GPU_FAILBACK_HEALTH_TIMEOUT", DEFAULT_HEALTH_TIMEOUT, float),
                        help="Seconds to wait for the GPU service's /health (default %(default)s).")
    parser.add_argument("--expiry-margin", type=float,
                        default=_env("AISWE_GPU_FAILBACK_EXPIRY_MARGIN", DEFAULT_EXPIRY_MARGIN, float),
                        help="Fail back this many seconds before expiresAt, so the switch happens before "
                             "the job is killed rather than after (default %(default)s; 0 = only once it is past).")
    parser.add_argument("--streak-window", type=float, default=DEFAULT_STREAK_WINDOW,
                        help="A failed check counts as consecutive only within this many seconds of the last (default %(default)s).")
    parser.add_argument("--lock-timeout", type=float, default=DEFAULT_LOCK_TIMEOUT,
                        help="Seconds to wait for a manual start/stop to finish writing the mode file (default %(default)s).")
    parser.add_argument("--keep-alive", default=_env("AISWE_GPU_FAILBACK_KEEP_ALIVE", DEFAULT_KEEP_ALIVE, str),
                        help="keep_alive for the warmed VM models (default %(default)s, as `stop`).")
    parser.add_argument("--warm-timeout", type=float, default=DEFAULT_WARM_TIMEOUT,
                        help="Per-model load timeout during the VM warm-up (default %(default)s).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check and say what would happen; write nothing (no mode file, state or log).")
    return parser


def main(argv: list[str] | None = None, *, transport: Any = None, clock: Callable[[], dt.datetime] = utc_now,
         warm: Callable[..., tuple[bool, list[str]]] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings(failures=max(1, args.failures), health_timeout=args.health_timeout,
                        expiry_margin=max(0.0, args.expiry_margin), streak_window=args.streak_window,
                        lock_timeout=args.lock_timeout, keep_alive=args.keep_alive,
                        warm_timeout=args.warm_timeout, dry_run=args.dry_run)
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(helpers.run_lock())
        except helpers.LockBusy:
            emit("info", "another watchdog run is in progress; nothing done")
            return 0
        return run_once(settings, transport=transport, clock=clock, warm=warm)


if __name__ == "__main__":
    sys.exit(main())
