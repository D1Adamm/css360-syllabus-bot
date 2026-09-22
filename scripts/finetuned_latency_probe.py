#!/usr/bin/env python3
"""A small classroom-readiness latency probe for the fine-tuned paths on the UWB VM.

    AISWE_VERIFY_PASSWORD=... backend/.venv/bin/python scripts/finetuned_latency_probe.py --admin-email you@uw.edu
    ... --out latency-$(date -u +%Y%m%dT%H%M%SZ).json        # keep the numbers (no answers are saved)
    ... --concurrency 1,2,4 --allow-concurrency-4             # heavier; explicit opt-in

Defaults are deliberately tiny: two courses x two modes (Fine-Tuned and
Fine-Tuned + RAG), one first request per cell measured on its own, then two
requests each at concurrency 1 and 2. Concurrency 4 is the ceiling and needs
`--allow-concurrency-4`; nothing here can be told to go higher. Every request
asks the same generic question, so no student data is involved, and only
timings, status codes and answer lengths are recorded.

Read the numbers with this in mind: the VM's Ollama is one CPU process shared
with Base and RAG, and the fine-tuned service serialises generations behind a
lock. Concurrency 2 therefore measures queueing, not parallel throughput, and
that is the honest classroom number — the second student waits for the first.
The first request per cell is "first" rather than strictly "cold": it pays
the model load only if the model was not resident (FINETUNED_KEEP_ALIVE).
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
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
    # Registered before execution: dataclasses under `from __future__ import
    # annotations` resolve their types through sys.modules[cls.__module__].
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


checks = _load("finetuned_production_checks")

MAX_CONCURRENCY = 4
DEFAULT_CONCURRENCY = (1, 2)
DEFAULT_REQUESTS = 2
DEFAULT_MODES = ("fineTuned", "fineTunedRag")

#: A sample: what one request produced, minus its text.
Sample = dict[str, Any]
Sender = Callable[[str, str], Sample]


def parse_concurrency(raw: str, *, allow_four: bool) -> list[int]:
    levels: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            level = int(token)
        except ValueError as exc:
            raise ValueError(f"Concurrency must be integers, got {token!r}") from exc
        if level < 1:
            raise ValueError("Concurrency must be at least 1.")
        if level > MAX_CONCURRENCY:
            raise ValueError(f"Concurrency above {MAX_CONCURRENCY} is not supported by this probe; it is a shared CPU host.")
        if level > 2 and not allow_four:
            raise ValueError(f"Concurrency {level} needs --allow-concurrency-4 (explicit opt-in on a shared CPU host).")
        if level not in levels:
            levels.append(level)
    if not levels:
        raise ValueError("No concurrency level given.")
    return levels


def build_plan(*, courses: list[str], modes: list[str], concurrency: list[int], requests: int) -> dict[str, Any]:
    if requests < 1:
        raise ValueError("--requests must be at least 1.")
    if requests > 10:
        raise ValueError("--requests above 10 per cell is not a small probe; lower it.")
    for mode in modes:
        if mode not in checks.MODE_ROUTES:
            raise ValueError(f"Unknown mode {mode!r}; choose from {', '.join(checks.MODE_ROUTES)}.")
    cells = [{"course": course, "mode": mode} for course in courses for mode in modes]
    total = len(cells) * (1 + requests * len(concurrency))
    return {"cells": cells, "concurrency": concurrency, "requests": requests, "totalRequests": total}


def make_sender(session: Any, *, timeout: float) -> Sender:
    def send(course: str, mode: str) -> Sample:
        result = checks.generate(session, mode, course, timeout=timeout)
        ok, detail = checks.judge_generation(result, course_id=course, mode=mode)
        return {"ok": ok, "seconds": result.seconds, "timedOut": result.timed_out, "status": result.status, "detail": detail}

    return send


def run_cell(cell: dict[str, str], plan: dict[str, Any], send: Sender, *, log: Callable[[str], None] = lambda _: None) -> dict[str, Any]:
    """The first request alone, then `requests` requests at each concurrency level."""
    course, mode = cell["course"], cell["mode"]
    first = send(course, mode)
    log(f"  {course} {mode} first: {'ok' if first['ok'] else 'FAIL'} {first['seconds']:.1f}s {first['detail']}")
    levels: dict[str, Any] = {}
    for level in plan["concurrency"]:
        n = plan["requests"]
        with ThreadPoolExecutor(max_workers=level) as pool:
            samples = list(pool.map(lambda _i: send(course, mode), range(n)))
        summary = checks.summarize_latencies(samples)
        levels[str(level)] = summary
        log(
            f"  {course} {mode} c={level}: ok {summary['ok']}/{summary['requests']}, timeouts {summary['timeouts']}, "
            f"median {checks.format_seconds(summary['median'])}, p95 {checks.format_seconds(summary['p95'])}, max {checks.format_seconds(summary['max'])}"
        )
    return {"course": course, "mode": mode, "first": first, "byConcurrency": levels}


def run_probe(plan: dict[str, Any], send: Sender, *, log: Callable[[str], None] = lambda _: None) -> dict[str, Any]:
    started = dt.datetime.now(dt.timezone.utc)
    cells = [run_cell(cell, plan, send, log=log) for cell in plan["cells"]]
    totals = {"requests": 0, "ok": 0, "failed": 0, "timeouts": 0}
    for cell in cells:
        totals["requests"] += 1
        totals["ok"] += 1 if cell["first"]["ok"] else 0
        totals["failed"] += 0 if cell["first"]["ok"] else 1
        totals["timeouts"] += 1 if cell["first"]["timedOut"] else 0
        for summary in cell["byConcurrency"].values():
            for key in totals:
                totals[key] += summary[key]
    return {
        "startedAt": started.isoformat(timespec="seconds"),
        "finishedAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "question": checks.PROBE_QUESTION,
        "plan": {"concurrency": plan["concurrency"], "requestsPerLevel": plan["requests"], "totalRequests": plan["totalRequests"]},
        "note": "The VM's Ollama is one shared CPU process and the fine-tuned service serialises generations; concurrency measures queueing.",
        "cells": cells,
        "totals": totals,
    }


def render_table(report: dict[str, Any]) -> str:
    lines = ["course                      mode          first    c   n  ok  to  median   p95     max"]
    for cell in report["cells"]:
        first = cell["first"]
        first_text = f"{first['seconds']:.1f}s" if first["ok"] else "FAIL"
        for level, s in cell["byConcurrency"].items():
            lines.append(
                f"{cell['course']:<27} {cell['mode']:<13} {first_text:<8} {level:>1} {s['requests']:>3} {s['ok']:>3} {s['timeouts']:>3}  "
                f"{checks.format_seconds(s['median']):<8} {checks.format_seconds(s['p95']):<7} {checks.format_seconds(s['max'])}"
            )
    t = report["totals"]
    lines.append(f"total: {t['requests']} requests, {t['ok']} ok, {t['failed']} failed, {t['timeouts']} timeouts")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend-url", default=checks.DEFAULT_BACKEND_URL)
    parser.add_argument("--course", action="append", help="course id (default: CSS 350 and CSS 360)")
    parser.add_argument("--mode", action="append", help="fineTuned, fineTunedRag, base or rag (default: the two fine-tuned modes)")
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS, help="requests per cell per concurrency level (default 2, max 10)")
    parser.add_argument("--concurrency", default=",".join(str(c) for c in DEFAULT_CONCURRENCY), help="comma-separated levels (default 1,2; 4 needs --allow-concurrency-4)")
    parser.add_argument("--allow-concurrency-4", action="store_true")
    parser.add_argument("--admin-email", help=f"administrator email (or ${checks.EMAIL_ENV}); password from ${checks.PASSWORD_ENV} or a prompt")
    parser.add_argument("--timeout", type=float, default=180.0, help="per-request timeout in seconds")
    parser.add_argument("--out", help="write the report (timings only, no answer text) to this JSON file")
    return parser


def main(argv: list[str] | None = None, *, transport: Any = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        concurrency = parse_concurrency(args.concurrency, allow_four=args.allow_concurrency_4)
        plan = build_plan(courses=list(args.course or checks.KNOWN_COURSES), modes=list(args.mode or DEFAULT_MODES), concurrency=concurrency, requests=args.requests)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    credentials = checks.resolve_admin_credentials(args.admin_email)
    if credentials is None:
        print(f"ERROR: the classroom routes need an administrator session; pass --admin-email and set {checks.PASSWORD_ENV} (or answer the prompt).", file=sys.stderr)
        return 2
    session = checks.BackendSession(args.backend_url, transport, timeout=args.timeout)
    login = session.login(*credentials)
    if not login.ok or not session.signed_in:
        print(f"ERROR: sign-in failed (HTTP {login.status} {login.error or ''}).", file=sys.stderr)
        return 1
    print(f"probe: {plan['totalRequests']} requests over {len(plan['cells'])} cells, concurrency {plan['concurrency']}, {plan['requests']} per level; question is generic; answers are not kept")
    try:
        report = run_probe(plan, make_sender(session, timeout=args.timeout), log=print)
    finally:
        session.logout()
    print(render_table(report))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"written: {args.out}")
    return 0 if report["totals"]["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
