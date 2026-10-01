#!/usr/bin/env python3
"""Classroom load test: N students pressing Compare at once, measured end to end.

    # On the VM, against the backend directly (loopback), as an administrator:
    AISWE_VERIFY_PASSWORD=... backend/.venv/bin/python scripts/classroom_load_test.py \\
        --admin-email you@uw.edu --course-id css360e-autumn-2026-c08m \\
        --levels 1,5,10,20,30 --ollama-url http://127.0.0.1:11434 \\
        --ollama-url http://127.0.0.1:11435 --out load-$(date -u +%Y%m%dT%H%M%SZ).json --yes

    # On a laptop, against scripts/classroom_load_harness.py (no database, no sign-in):
    backend/.venv/bin/python scripts/classroom_load_test.py --no-auth \\
        --base-url http://127.0.0.1:8011 --course-id css-360-winter-2026-a7rp --levels 1,5 --yes

Each simulated student makes exactly the requests the Compare page makes
(`src/context/comparisonRunner.ts`): Base then RAG in sequence, with
Fine-Tuned and Fine-Tuned + RAG alongside that chain. Every student in a level
starts at the same moment (or spread over `--ramp` seconds), which is the
worst case a class produces when the instructor says "go".

Recorded per request: condition, HTTP status, latency, and the backend's busy
code when it refused for capacity. No answer text is kept, only its length.
Recorded per level: wall time, throughput, per-condition quantiles, outcome
counts, and — when sampled — CPU, RAM, which Ollama models were resident, how
many times a model was (re)loaded, and the backend's generation-queue depth.

This is a load generator. Pointed at production it is thirty students'
worth of CPU for several minutes, so it prints its plan and needs `--yes`.
The generate routes write nothing to the database; signing in as an
administrator creates one session, ended at exit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
LIB = REPO_ROOT / "scripts" / "lib"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, LIB / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


checks = _load("finetuned_production_checks")

CONDITIONS = ("base", "rag", "fineTuned", "fineTunedRag")
DEFAULT_LEVELS = (1, 5, 10, 20, 30)
#: A class is about thirty; forty leaves room for a large section and no more.
MAX_STUDENTS = 40
#: Nginx's `proxy_read_timeout` on `location /api/` in production.
DEFAULT_REQUEST_TIMEOUT = 300.0
#: The backend's capacity refusal (`app.generation_queue`).
BUSY_CODE = "generation_busy"

#: Generic syllabus questions a student asks in the first minutes of class.
#: Nothing personal; varied so no two concurrent prompts are identical.
QUESTIONS = (
    "When is the final exam?",
    "What is the late work policy?",
    "How is the final grade calculated?",
    "When are the instructor's office hours?",
    "Can I use AI tools on the assignments?",
    "What happens if I miss a quiz?",
    "How do I contact the instructor?",
    "What is the attendance policy?",
    "When does the course meet?",
    "What textbook is required?",
)

Sample = dict[str, Any]


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def parse_levels(raw: str) -> list[int]:
    levels: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            level = int(token)
        except ValueError as exc:
            raise ValueError(f"Levels must be integers, got {token!r}") from exc
        if level < 1:
            raise ValueError("A level must be at least one student.")
        if level > MAX_STUDENTS:
            raise ValueError(f"At most {MAX_STUDENTS} simulated students per level.")
        if level not in levels:
            levels.append(level)
    if not levels:
        raise ValueError("No levels given.")
    return levels


# --------------------------------------------------------------------------- #
# One student
# --------------------------------------------------------------------------- #


def _busy_code(body: Any) -> str | None:
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict) and isinstance(detail.get("code"), str):
            return detail["code"]
    return None


def classify(result: Any) -> str:
    """One word per request outcome: ok, busy, timeout, http_<status>, network."""
    if result.ok:
        body = result.body if isinstance(result.body, dict) else {}
        answer = body.get("answer")
        return "ok" if isinstance(answer, str) and answer.strip() else "empty"
    if result.timed_out:
        return "timeout"
    if result.status == 0:
        return "network"
    if _busy_code(result.body) == BUSY_CODE:
        return "busy"
    return f"http_{result.status}"


def run_student(
    send: Callable[[str, str, str], Any],
    *,
    student: int,
    question: str,
    clock: Callable[[], float],
) -> list[Sample]:
    """The Compare page's schedule for one student; returns one sample per condition.

    Like the page, all four requests carry one comparison id
    (`X-Comparison-Id`), which the backend's generation queue orders by.
    """
    samples: list[Sample] = []
    lock = threading.Lock()
    comparison_id = f"load-{uuid.uuid4().hex[:16]}-{student}"

    def one(condition: str) -> None:
        started = clock()
        result = send(condition, question, comparison_id)
        ended = clock()
        body = result.body if isinstance(result.body, dict) else {}
        answer = body.get("answer") if result.ok else None
        sample = {
            "student": student,
            "condition": condition,
            "status": result.status,
            "outcome": classify(result),
            "ok": classify(result) == "ok",
            "timedOut": bool(result.timed_out),
            "seconds": round(result.seconds, 3),
            "start": round(started, 3),
            "end": round(ended, 3),
            "answerChars": len(answer.strip()) if isinstance(answer, str) else 0,
        }
        with lock:
            samples.append(sample)

    def chain() -> None:
        one("base")
        one("rag")

    threads = [
        threading.Thread(target=chain),
        threading.Thread(target=one, args=("fineTuned",)),
        threading.Thread(target=one, args=("fineTunedRag",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return samples


# --------------------------------------------------------------------------- #
# Resource sampling
# --------------------------------------------------------------------------- #


class SystemSampler:
    """CPU and RAM of this machine, from psutil or /proc. Meaningful only on the VM."""

    def __init__(self) -> None:
        self._psutil = None
        try:
            import psutil  # type: ignore

            self._psutil = psutil
            psutil.cpu_percent(None)
        except Exception:  # noqa: BLE001 - optional dependency
            self._psutil = None
        self._last_cpu = self._proc_cpu()

    @staticmethod
    def _proc_cpu() -> tuple[int, int] | None:
        try:
            with open("/proc/stat", encoding="utf-8") as handle:
                fields = [int(value) for value in handle.readline().split()[1:]]
        except (OSError, ValueError):
            return None
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
        return idle, sum(fields)

    @property
    def available(self) -> bool:
        return self._psutil is not None or self._last_cpu is not None

    def sample(self) -> dict[str, float] | None:
        if self._psutil is not None:
            memory = self._psutil.virtual_memory()
            return {
                "cpuPercent": float(self._psutil.cpu_percent(None)),
                "memUsedGiB": (memory.total - memory.available) / 2**30,
                "memAvailableGiB": memory.available / 2**30,
            }
        current = self._proc_cpu()
        if current is None or self._last_cpu is None:
            return None
        idle_delta = current[0] - self._last_cpu[0]
        total_delta = current[1] - self._last_cpu[1]
        self._last_cpu = current
        cpu = 100.0 * (1 - idle_delta / total_delta) if total_delta > 0 else 0.0
        meminfo: dict[str, int] = {}
        try:
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    key, _, rest = line.partition(":")
                    meminfo[key] = int(rest.split()[0]) * 1024
        except (OSError, ValueError, IndexError):
            return {"cpuPercent": cpu}
        total = meminfo.get("MemTotal", 0)
        available = meminfo.get("MemAvailable", 0)
        return {
            "cpuPercent": cpu,
            "memUsedGiB": (total - available) / 2**30,
            "memAvailableGiB": available / 2**30,
        }


def resident_models(transport: Any, ollama_url: str) -> list[str] | None:
    result = transport.request("GET", ollama_url.rstrip("/") + "/api/ps", timeout=3.0)
    if not result.ok or not isinstance(result.body, dict):
        return None
    models = result.body.get("models")
    if not isinstance(models, list):
        return None
    return sorted(str(m.get("name")) for m in models if isinstance(m, dict) and m.get("name"))


def count_loads(timeline: list[list[str] | None]) -> int:
    """How many times a model became resident that was not resident one sample before.

    The first observation is the starting state, not a load. Sampling once a
    second can miss a swap that completes in under a second; on the CPU VM a
    load takes several, so it does not.
    """
    loads = 0
    previous: set[str] | None = None
    for names in timeline:
        if names is None:
            continue
        current = set(names)
        if previous is not None:
            loads += len(current - previous)
        previous = current
    return loads


class Monitor:
    """Samples once a second while a level runs."""

    def __init__(
        self,
        *,
        transport: Any,
        ollama_urls: list[str],
        system: SystemSampler | None,
        queue_status: Callable[[], dict[str, Any] | None] | None,
        interval: float = 1.0,
    ) -> None:
        self.transport = transport
        self.ollama_urls = ollama_urls
        self.system = system
        self.queue_status = queue_status
        self.interval = interval
        self.system_samples: list[dict[str, float]] = []
        self.resident: dict[str, list[list[str] | None]] = {url: [] for url in ollama_urls}
        self.queue_samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _tick(self) -> None:
        if self.system is not None:
            sample = self.system.sample()
            if sample:
                self.system_samples.append(sample)
        for url in self.ollama_urls:
            self.resident[url].append(resident_models(self.transport, url))
        if self.queue_status is not None:
            status = self.queue_status()
            if status:
                self.queue_samples.append(status)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._tick()
            self._stop.wait(self.interval)

    def __enter__(self) -> "Monitor":
        if self.system is not None:
            # Start the CPU delta now, not at the end of the previous level:
            # otherwise the first sample spans the idle cooldown.
            self.system.sample()
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        self._tick()

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.system_samples:
            cpu = [s["cpuPercent"] for s in self.system_samples if "cpuPercent" in s]
            used = [s["memUsedGiB"] for s in self.system_samples if "memUsedGiB" in s]
            avail = [s["memAvailableGiB"] for s in self.system_samples if "memAvailableGiB" in s]
            out["cpuPercentMean"] = round(statistics.fmean(cpu), 1) if cpu else None
            out["cpuPercentMax"] = round(max(cpu), 1) if cpu else None
            out["memUsedGiBMax"] = round(max(used), 2) if used else None
            out["memAvailableGiBMin"] = round(min(avail), 2) if avail else None
        out["ollama"] = {
            url: {
                "modelLoads": count_loads(timeline),
                "residentAtEnd": next((t for t in reversed(timeline) if t is not None), None),
                "everResident": sorted({n for t in timeline if t for n in t}),
            }
            for url, timeline in self.resident.items()
        }
        if self.queue_samples:
            out["queue"] = {
                "maxWaiting": max(int(s.get("waiting", 0)) for s in self.queue_samples),
                "maxActive": max(int(s.get("active", 0)) for s in self.queue_samples),
            }
        return out


# --------------------------------------------------------------------------- #
# Levels
# --------------------------------------------------------------------------- #


def summarize_level(samples: list[Sample], *, students: int) -> dict[str, Any]:
    wall = (max(s["end"] for s in samples) - min(s["start"] for s in samples)) if samples else 0.0
    per_condition: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = [s for s in samples if s["condition"] == condition]
        latencies = checks.summarize_latencies(rows)
        outcomes: dict[str, int] = {}
        for row in rows:
            outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
        latencies["outcomes"] = outcomes
        per_condition[condition] = latencies
    by_student: dict[int, list[Sample]] = {}
    for sample in samples:
        by_student.setdefault(sample["student"], []).append(sample)
    complete = sum(1 for rows in by_student.values() if len(rows) == 4 and all(r["ok"] for r in rows))
    ok = sum(1 for s in samples if s["ok"])
    totals: dict[str, int] = {}
    for sample in samples:
        totals[sample["outcome"]] = totals.get(sample["outcome"], 0) + 1
    return {
        "students": students,
        "requests": len(samples),
        "wallSeconds": round(wall, 1),
        "okRequests": ok,
        "successRate": round(ok / len(samples), 3) if samples else None,
        "outcomes": totals,
        "completeComparisons": complete,
        "generationsPerMinute": round(ok / wall * 60, 2) if wall > 0 else None,
        "comparisonsPerMinute": round(complete / wall * 60, 2) if wall > 0 else None,
        "perCondition": per_condition,
    }


def question_for(student: int, *, level: int, unique: bool) -> str:
    """The question one simulated student asks.

    By default students cycle through `QUESTIONS`, so several ask the same
    thing, as a class does. Ollama's runners keep a prompt cache, so a
    repeated question is cheaper the second time, provided the runner was not
    swapped out in between. `unique` gives every student in every level its
    own question, which measures the worst case: no prompt is ever seen twice.
    """
    base = QUESTIONS[student % len(QUESTIONS)]
    if not unique:
        return base
    return f"{base.rstrip('?')} (student {level}-{student})?"


def run_level(
    send: Callable[[str, str, str], Any],
    *,
    students: int,
    ramp: float,
    monitor_factory: Callable[[], Monitor | None],
    log: Callable[[str], None],
    unique: bool = False,
) -> dict[str, Any]:
    origin = time.perf_counter()

    def clock() -> float:
        return time.perf_counter() - origin

    def student(index: int) -> list[Sample]:
        if ramp > 0 and students > 1:
            time.sleep(ramp * index / (students - 1))
        return run_student(send, student=index, question=question_for(index, level=students, unique=unique), clock=clock)

    monitor = monitor_factory()
    samples: list[Sample] = []
    if monitor is not None:
        monitor.__enter__()
    try:
        with ThreadPoolExecutor(max_workers=students) as pool:
            for rows in pool.map(student, range(students)):
                samples.extend(rows)
    finally:
        if monitor is not None:
            monitor.__exit__(None, None, None)
    summary = summarize_level(samples, students=students)
    if monitor is not None:
        summary["resources"] = monitor.summary()
    summary["samples"] = sorted(samples, key=lambda s: (s["student"], CONDITIONS.index(s["condition"])))
    log(
        f"  {students:>2} students: {summary['okRequests']}/{summary['requests']} ok, "
        f"{summary['completeComparisons']} complete, wall {summary['wallSeconds']}s, "
        f"outcomes {summary['outcomes']}"
    )
    return summary


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}{suffix}"
    return f"{value}{suffix}"


def render_table(report: dict[str, Any]) -> str:
    lines = [
        "| students | ok / requests | complete | wall | gen/min | busy | timeout | other fail | p50 base / rag / ft / ft+rag | p95 base / rag / ft / ft+rag | model loads | CPU mean/max | RAM used max |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for level in report["levels"]:
        outcomes = level["outcomes"]
        busy = outcomes.get("busy", 0)
        timeouts = outcomes.get("timeout", 0)
        other = level["requests"] - level["okRequests"] - busy - timeouts
        pc = level["perCondition"]
        p50 = " / ".join(_fmt(pc[c]["median"], "s") for c in CONDITIONS)
        p95 = " / ".join(_fmt(pc[c]["p95"], "s") for c in CONDITIONS)
        resources = level.get("resources") or {}
        loads = sum(v["modelLoads"] for v in (resources.get("ollama") or {}).values()) if resources.get("ollama") else None
        cpu = f"{_fmt(resources.get('cpuPercentMean'), '%')}/{_fmt(resources.get('cpuPercentMax'), '%')}"
        lines.append(
            f"| {level['students']} | {level['okRequests']}/{level['requests']} | {level['completeComparisons']} "
            f"| {level['wallSeconds']}s | {_fmt(level['generationsPerMinute'])} | {busy} | {timeouts} | {other} "
            f"| {p50} | {p95} | {_fmt(loads)} | {cpu} | {_fmt(resources.get('memUsedGiBMax'), ' GiB')} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=checks.DEFAULT_BACKEND_URL, help="Backend origin (default %(default)s).")
    parser.add_argument("--course-id", required=True, help="The course every simulated student asks about.")
    parser.add_argument("--levels", default=",".join(str(n) for n in DEFAULT_LEVELS), help="Comma-separated student counts (default %(default)s).")
    parser.add_argument("--ramp", type=float, default=0.0, help="Spread each level's starts over this many seconds (default 0: all at once).")
    parser.add_argument("--cooldown", type=float, default=15.0, help="Seconds between levels, so one level's backlog does not leak into the next (default %(default)s).")
    parser.add_argument("--timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT, help="Per-request client timeout (default %(default)s, Nginx's).")
    parser.add_argument("--unique-questions", action="store_true", help="Give every student its own question, so no prompt repeats (worst case for Ollama's prompt cache).")
    parser.add_argument("--ollama-url", action="append", default=[], help="Sample this Ollama's /api/ps once a second (repeatable).")
    parser.add_argument("--no-system", action="store_true", help="Do not sample CPU and RAM (they describe this machine, so only mean something on the VM).")
    parser.add_argument("--admin-email", help=f"Sign in as this administrator (password from ${checks.PASSWORD_ENV} or a prompt).")
    parser.add_argument("--no-auth", action="store_true", help="Send no credentials (for scripts/classroom_load_harness.py only).")
    parser.add_argument("--out", help="Write the full JSON report here.")
    parser.add_argument("--yes", action="store_true", help="Run without the confirmation prompt.")
    return parser


def main(argv: list[str] | None = None, *, transport: Any = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        levels = parse_levels(args.levels)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    total = sum(levels)
    print(
        f"Plan: {args.base_url} course {args.course_id}; levels {levels} "
        f"({total} comparisons, {total * 4} generations), ramp {args.ramp}s, cooldown {args.cooldown}s."
    )
    if not args.yes:
        if not sys.stdin.isatty():
            print("ERROR: pass --yes to run non-interactively.", file=sys.stderr)
            return 2
        if input("This loads the target's CPU for minutes. Continue? [y/N] ").strip().lower() != "y":
            print("Cancelled.")
            return 1

    transport = transport or checks.Transport()
    session = checks.BackendSession(args.base_url, transport, timeout=args.timeout)
    if not args.no_auth:
        credentials = checks.resolve_admin_credentials(args.admin_email)
        if credentials is None:
            print("ERROR: give --admin-email (or --no-auth for the local harness).", file=sys.stderr)
            return 2
        login = session.login(*credentials)
        if not login.ok or not session.signed_in:
            print(f"ERROR: sign-in failed (HTTP {login.status} {login.error or ''}).", file=sys.stderr)
            return 1

    def send(condition: str, question: str, comparison_id: str) -> Any:
        return checks.generate(
            session, condition, args.course_id, question=question, timeout=args.timeout, comparison_id=comparison_id
        )

    def queue_status() -> dict[str, Any] | None:
        result = session.get("/api/admin/generation-queue", timeout=3.0)
        return result.body if result.ok and isinstance(result.body, dict) else None

    system = None if args.no_system else SystemSampler()
    if system is not None and not system.available:
        system = None

    def monitor_factory() -> Monitor:
        return Monitor(transport=transport, ollama_urls=args.ollama_url, system=system, queue_status=queue_status)

    report: dict[str, Any] = {
        "startedAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "baseUrl": args.base_url,
        "courseId": args.course_id,
        "ramp": args.ramp,
        "uniqueQuestions": args.unique_questions,
        "comparisonIdHeader": True,
        "timeout": args.timeout,
        "host": os.uname().nodename if hasattr(os, "uname") else None,
        "levels": [],
    }
    try:
        for index, students in enumerate(levels):
            if index:
                time.sleep(args.cooldown)
            report["levels"].append(
                run_level(
                    send,
                    students=students,
                    ramp=args.ramp,
                    monitor_factory=monitor_factory,
                    log=print,
                    unique=args.unique_questions,
                )
            )
    finally:
        session.logout()

    report["finishedAt"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    print()
    print(render_table(report))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
