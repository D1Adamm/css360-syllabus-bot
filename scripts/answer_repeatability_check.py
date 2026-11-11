#!/usr/bin/env python3
"""Phase 0 measurement for the Compare answer cache: one request at a time.

    # On the VM, against the backend directly (loopback), as an administrator,
    # outside class time:
    AISWE_VERIFY_PASSWORD=... backend/.venv/bin/python scripts/answer_repeatability_check.py \\
        --admin-email you@uw.edu --course-id css360e-autumn-2026-c08m \\
        --questions 10 --out repeatability-$(date -u +%Y%m%dT%H%M%SZ).json --yes

    # On a laptop, against scripts/classroom_load_harness.py (no database, no sign-in):
    backend/.venv/bin/python scripts/answer_repeatability_check.py --no-auth \\
        --base-url http://127.0.0.1:8011 --course-id css-360-winter-2026-a7rp \\
        --questions-file questions.txt --passes question,batched --yes

It answers two questions the cache design depends on, with real requests to
the four classroom routes:

1. **Does the same question get the same answer twice?** Every question is
   asked once per pass and the answers are compared by hash. Greedy decoding
   with a fixed seed should repeat on one runtime; this measures whether it
   does on this one.
2. **Does the order of work matter?** A `question` pass asks all four
   approaches for one question before the next question. A `batched` pass
   takes a few questions at a time and asks the two retrieval approaches for
   all of them, then the two direct ones, so each Ollama server sees prompts
   of one kind in a row and can reuse the rules that open every grounded
   prompt. The same questions are asked in both orders and the time compared.

It is not a load test. Exactly one request is in flight at any moment, it
waits while anyone else is using the generation queue, it stops after three
failures in a row, and it refuses to run in classroom GPU mode (those answers
come from a different engine). It still occupies the VM's CPU for as long as
it runs, so it prints its plan and needs `--yes`.

Nothing is written to the database: the generate routes write nothing, and
signing in as an administrator creates one session, ended at exit. No answer
or question text is kept, only hashes, lengths and timings.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

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
#: Within a batch: retrieval approaches first, then the direct ones. Base and
#: RAG share one Ollama server and the two fine-tuned approaches the other, so
#: this keeps prompts of one kind consecutive on each server.
BATCHED_CONDITION_ORDER = ("rag", "fineTunedRag", "base", "fineTuned")
ORDERS = ("question", "batched")
#: question, batched, batched, question: each order twice, with any slow drift
#: over the run shared between them.
DEFAULT_PASSES = "question,batched,batched,question"
DEFAULT_QUESTIONS = 10
DEFAULT_BATCH_SIZE = 5
MAX_QUESTIONS = 30
MAX_PASSES = 6
#: Nginx's `proxy_read_timeout` on `location /api/` in production.
DEFAULT_REQUEST_TIMEOUT = 300.0
#: Stop rather than keep asking a server that is refusing or failing.
MAX_CONSECUTIVE_FAILURES = 3
BUSY_CODE = "generation_busy"
#: The seed origins staff can choose from for classroom preparation.
ELIGIBLE_ORIGINS = ("prototype", "ai_generated")

Sample = dict[str, Any]
Step = tuple[int, str]


class Abort(Exception):
    """The run cannot continue; the message says why."""


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def parse_passes(raw: str) -> list[str]:
    passes = [token.strip() for token in raw.split(",") if token.strip()]
    if not passes:
        raise ValueError("No passes given.")
    for name in passes:
        if name not in ORDERS:
            raise ValueError(f"A pass is one of {', '.join(ORDERS)}; got {name!r}.")
    if len(passes) > MAX_PASSES:
        raise ValueError(f"At most {MAX_PASSES} passes.")
    return passes


def build_pass(order: str, question_count: int, batch_size: int) -> list[Step]:
    """Every (question index, condition) once, in the given order of work."""
    if order == "question":
        return [(q, condition) for q in range(question_count) for condition in CONDITIONS]
    if order == "batched":
        steps: list[Step] = []
        for start in range(0, question_count, batch_size):
            batch = range(start, min(start + batch_size, question_count))
            for condition in BATCHED_CONDITION_ORDER:
                steps.extend((q, condition) for q in batch)
        return steps
    raise ValueError(f"Unknown order {order!r}.")


def _question_key(question: str) -> str:
    return " ".join(question.split()).casefold()


def _review_status(seed: dict[str, Any]) -> str:
    for key in ("reviewStatus", "status"):
        value = seed.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return "generated"


def eligible_questions(seeds: Iterable[Any]) -> list[dict[str, str]]:
    """The course questions staff could select for preparation, one per wording.

    Staff-curated and AI-generated examples that were not rejected. Anything a
    participant contributed is left out, whatever its origin says.
    """
    seen: set[str] = set()
    eligible: list[dict[str, str]] = []
    for seed in seeds:
        if not isinstance(seed, dict):
            continue
        if seed.get("origin") not in ELIGIBLE_ORIGINS or seed.get("participantId"):
            continue
        if _review_status(seed) == "rejected":
            continue
        raw = seed.get("question") or seed.get("instruction")
        question = " ".join(raw.split()) if isinstance(raw, str) else ""
        key = _question_key(question)
        if not question or key in seen:
            continue
        seen.add(key)
        eligible.append({"ref": str(seed.get("id") or f"seed-{len(eligible)}"), "question": question})
    return sorted(eligible, key=lambda item: item["ref"])


def pick_questions(eligible: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    """`count` questions spread evenly over the list; the same ones every run."""
    if count >= len(eligible):
        return list(eligible)
    return [eligible[(index * len(eligible)) // count] for index in range(count)]


def read_questions_file(path: Path) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        question = " ".join(line.split())
        key = _question_key(question)
        if not question or question.startswith("#") or key in seen:
            continue
        seen.add(key)
        questions.append({"ref": f"line-{len(questions) + 1}", "question": question})
    return questions


# --------------------------------------------------------------------------- #
# One request
# --------------------------------------------------------------------------- #


def answer_hash(answer: str) -> str:
    """The answer's identity: its text with runs of whitespace collapsed."""
    return hashlib.sha256(" ".join(answer.split()).encode("utf-8")).hexdigest()


def _busy_code(body: Any) -> str | None:
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict) and isinstance(detail.get("code"), str):
            return detail["code"]
    return None


def classify(result: Any) -> str:
    """One word per request outcome: ok, empty, busy, timeout, network, http_<status>."""
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


def sample_of(result: Any, *, pass_index: int, order: str, question: dict[str, str], condition: str) -> Sample:
    outcome = classify(result)
    body = result.body if isinstance(result.body, dict) else {}
    answer = body.get("answer") if outcome == "ok" else None
    sample: Sample = {
        "pass": pass_index,
        "order": order,
        "question": question["ref"],
        "condition": condition,
        "status": result.status,
        "outcome": outcome,
        "seconds": round(result.seconds, 3),
        "answerSha256": answer_hash(answer) if isinstance(answer, str) else None,
        "answerChars": len(answer.strip()) if isinstance(answer, str) else 0,
    }
    for key in ("model", "modelVersion", "engine"):
        value = body.get(key)
        if isinstance(value, str) and value:
            sample[key] = value
    return sample


def answered_on_gpu(sample: Sample) -> bool:
    """Whether the backend says this answer came from the Tillicum GPU service.

    The fine-tuned routes report an `engine`; Base and RAG report it only in
    the model name (`… (Tillicum GPU)`, `app.ollama.generate_ollama_chat`).
    """
    return sample.get("engine") == "gpu" or "Tillicum GPU" in str(sample.get("model") or "")


# --------------------------------------------------------------------------- #
# Staying out of the way
# --------------------------------------------------------------------------- #


def queue_is_idle(status: dict[str, Any] | None) -> bool | None:
    """True/False from the queue snapshot, or None when it could not be read."""
    if not isinstance(status, dict):
        return None
    try:
        return int(status.get("active", 0)) == 0 and int(status.get("waiting", 0)) == 0
    except (TypeError, ValueError):
        return None


def wait_until_idle(
    queue_status: Callable[[], dict[str, Any] | None],
    *,
    max_wait: float,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    poll: float = 5.0,
    log: Callable[[str], None] = print,
) -> float:
    """Wait while anyone else is generating. Returns the seconds waited.

    Between this script's requests nothing of its own is in the queue, so any
    activity there is somebody else's and they go first.
    """
    started = clock()
    announced = False
    while True:
        idle = queue_is_idle(queue_status())
        if idle is None:
            raise Abort("The generation queue status could not be read, so the run stopped.")
        if idle:
            return clock() - started
        if clock() - started >= max_wait:
            raise Abort(f"The generation queue stayed busy for {int(max_wait)} s; someone is using the site.")
        if not announced:
            log("  Someone else is generating; waiting for the queue to be idle…")
            announced = True
        sleep(poll)


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def run_passes(
    send: Callable[[str, str], Any],
    *,
    questions: list[dict[str, str]],
    passes: list[str],
    batch_size: int,
    before_request: Callable[[], None] | None = None,
    samples: list[Sample] | None = None,
    log: Callable[[str], None] = print,
) -> list[Sample]:
    """Ask every question in every pass, strictly one request at a time.

    Samples are appended to `samples` as they arrive, so a caller that passes
    its own list still has the finished ones when the run aborts.
    """
    samples = [] if samples is None else samples
    failures = 0
    for pass_index, order in enumerate(passes):
        steps = build_pass(order, len(questions), batch_size)
        started = time.perf_counter()
        ok = 0
        for question_index, condition in steps:
            if before_request is not None:
                before_request()
            question = questions[question_index]
            result = send(condition, question["question"])
            sample = sample_of(result, pass_index=pass_index, order=order, question=question, condition=condition)
            samples.append(sample)
            if answered_on_gpu(sample):
                raise Abort("An answer came from the Tillicum GPU: the mode changed during the run.")
            if sample["outcome"] == "ok":
                ok += 1
                failures = 0
            else:
                failures += 1
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    raise Abort(
                        f"{failures} requests in a row failed (last: {condition}, {sample['outcome']}); "
                        "stopped rather than keep asking."
                    )
        log(
            f"  pass {pass_index + 1}/{len(passes)} ({order}): {ok}/{len(steps)} ok "
            f"in {time.perf_counter() - started:.0f}s"
        )
    return samples


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #


def summarize_repeatability(samples: list[Sample]) -> dict[str, Any]:
    """Per condition: of the questions answered at least twice, how many always
    got the identical answer."""
    summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        hashes: dict[str, list[str]] = {}
        for sample in samples:
            if sample["condition"] == condition and sample["outcome"] == "ok":
                hashes.setdefault(sample["question"], []).append(sample["answerSha256"])
        compared = {ref: values for ref, values in hashes.items() if len(values) >= 2}
        identical = sum(1 for values in compared.values() if len(set(values)) == 1)
        varying = sorted(ref for ref, values in compared.items() if len(set(values)) > 1)
        summary[condition] = {
            "questionsCompared": len(compared),
            "identical": identical,
            "rate": round(identical / len(compared), 3) if compared else None,
            "mostDistinctAnswers": max((len(set(values)) for values in compared.values()), default=0),
            "varyingQuestions": varying,
        }
    return summary


def summarize_orders(samples: list[Sample], passes: list[str]) -> dict[str, Any]:
    """Seconds per answer in each order of work, over the same set of answers.

    Only (question, condition) pairs that succeeded in every pass are counted,
    so both orders are timed on identical work.
    """
    by_pair: dict[tuple[str, str], list[Sample]] = {}
    for sample in samples:
        if sample["outcome"] == "ok":
            by_pair.setdefault((sample["question"], sample["condition"]), []).append(sample)
    complete = {pair for pair, rows in by_pair.items() if len(rows) == len(passes)}

    orders: dict[str, Any] = {}
    for order in ORDERS:
        if order not in passes:
            continue
        rows = [s for pair in complete for s in by_pair[pair] if s["order"] == order]
        per_condition: dict[str, Any] = {}
        for condition in CONDITIONS:
            seconds = [s["seconds"] for s in rows if s["condition"] == condition]
            per_condition[condition] = {
                "answers": len(seconds),
                "meanSeconds": round(statistics.fmean(seconds), 2) if seconds else None,
                "medianSeconds": round(statistics.median(seconds), 2) if seconds else None,
            }
        seconds = [s["seconds"] for s in rows]
        orders[order] = {
            "passes": passes.count(order),
            "answers": len(seconds),
            "totalSeconds": round(sum(seconds), 1),
            "meanSeconds": round(statistics.fmean(seconds), 2) if seconds else None,
            "perCondition": per_condition,
        }

    comparison: dict[str, Any] = {"pairsCompared": len(complete)}
    question, batched = orders.get("question"), orders.get("batched")
    if question and batched and question["meanSeconds"] and batched["meanSeconds"]:
        saving = 1 - batched["meanSeconds"] / question["meanSeconds"]
        comparison["batchedSavingPercent"] = round(100 * saving, 1)
    return {"orders": orders, "comparison": comparison}


def render_report(report: dict[str, Any]) -> str:
    lines = ["Same answer every time?", "| approach | questions compared | always identical | rate |", "|---|---|---|---|"]
    for condition in CONDITIONS:
        row = report["repeatability"][condition]
        rate = "-" if row["rate"] is None else f"{100 * row['rate']:.0f}%"
        lines.append(f"| {condition} | {row['questionsCompared']} | {row['identical']} | {rate} |")

    timing = report["orderTiming"]
    lines += ["", "Seconds per answer by order of work (same answers in each order)"]
    lines += ["| order | answers | mean | base | rag | fineTuned | fineTunedRag |", "|---|---|---|---|---|---|---|"]
    for order, row in timing["orders"].items():
        per = " | ".join(
            "-" if row["perCondition"][c]["meanSeconds"] is None else f"{row['perCondition'][c]['meanSeconds']:.1f}s"
            for c in CONDITIONS
        )
        mean = "-" if row["meanSeconds"] is None else f"{row['meanSeconds']:.1f}s"
        lines.append(f"| {order} | {row['answers']} | {mean} | {per} |")
    saving = timing["comparison"].get("batchedSavingPercent")
    if saving is not None:
        lines.append(f"\nBatched order took {abs(saving):.1f}% {'less' if saving >= 0 else 'more'} time per answer than question order.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=checks.DEFAULT_BACKEND_URL, help="Backend origin (default %(default)s).")
    parser.add_argument("--course-id", required=True, help="The course to ask about.")
    parser.add_argument("--questions", type=int, default=DEFAULT_QUESTIONS, help=f"How many questions to ask (default %(default)s, at most {MAX_QUESTIONS}).")
    parser.add_argument("--questions-file", help="Ask these questions (one per line, # for comments) instead of the course's own examples.")
    parser.add_argument("--passes", default=DEFAULT_PASSES, help="Comma-separated orders of work, one per pass (default %(default)s).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Questions per batch in a batched pass (default %(default)s).")
    parser.add_argument("--timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT, help="Per-request client timeout (default %(default)s, Nginx's).")
    parser.add_argument("--busy-wait", type=float, default=600.0, help="Give up after waiting this long for other users to finish (default %(default)s s).")
    parser.add_argument("--no-warmup", action="store_true", help="Skip the one untimed request per approach that loads the models first.")
    parser.add_argument("--admin-email", help=f"Sign in as this administrator (password from ${checks.PASSWORD_ENV} or a prompt).")
    parser.add_argument("--no-auth", action="store_true", help="Send no credentials (for scripts/classroom_load_harness.py only; needs --questions-file).")
    parser.add_argument("--out", help="Write the full JSON report here.")
    parser.add_argument("--yes", action="store_true", help="Run without the confirmation prompt.")
    return parser


def _load_course_questions(session: Any, course_id: str) -> list[dict[str, str]]:
    result = session.get(f"/api/db/courses/{course_id}/seeds")
    if not result.ok or not isinstance(result.body, dict):
        raise Abort(f"Could not list the course's examples (HTTP {result.status} {result.error or ''}).")
    seeds = result.body.get("seeds")
    return eligible_questions(seeds if isinstance(seeds, list) else [])


def main(argv: list[str] | None = None, *, transport: Any = None, sleep: Callable[[float], None] = time.sleep) -> int:
    args = build_parser().parse_args(argv)
    try:
        passes = parse_passes(args.passes)
        if not 1 <= args.questions <= MAX_QUESTIONS:
            raise ValueError(f"--questions must be between 1 and {MAX_QUESTIONS}.")
        if args.batch_size < 1:
            raise ValueError("--batch-size must be at least 1.")
        if args.no_auth and not args.questions_file:
            raise ValueError("--no-auth needs --questions-file: there are no course examples to read.")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

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

    def queue_status() -> dict[str, Any] | None:
        result = session.get("/api/admin/generation-queue", timeout=5.0)
        return result.body if result.ok and isinstance(result.body, dict) else None

    def before_request() -> None:
        wait_until_idle(queue_status, max_wait=args.busy_wait, sleep=sleep)

    def send(condition: str, question: str) -> Any:
        return checks.generate(
            session,
            condition,
            args.course_id,
            question=question,
            timeout=args.timeout,
            comparison_id=f"repeat-{uuid.uuid4().hex[:16]}",
        )

    try:
        if not args.no_auth:
            mode = session.get("/api/admin/generation-mode", timeout=5.0)
            kind = mode.body.get("mode") if mode.ok and isinstance(mode.body, dict) else None
            if kind != "vm":
                raise Abort(
                    f"The backend is not in VM mode (reported: {kind or 'unreadable'}). "
                    "This measures the VM's own answers and does not run in GPU mode."
                )

        if args.questions_file:
            available = read_questions_file(Path(args.questions_file))
            source = "file"
        else:
            available = _load_course_questions(session, args.course_id)
            source = "course examples (staff-curated and AI-generated, not rejected, none from students)"
        questions = pick_questions(available, args.questions)
        if not questions:
            raise Abort("No questions to ask.")

        generations = len(questions) * len(CONDITIONS) * len(passes)
        warmups = 0 if args.no_warmup else len(CONDITIONS)
        print(
            f"Plan: {args.base_url} course {args.course_id}; {len(questions)} of {len(available)} questions from {source}; "
            f"passes {passes}; {generations} generations one at a time"
            + (f" plus {warmups} warm-up" if warmups else "")
            + f" (about {max(1, round((generations + warmups) * 10 / 60))} minutes at 10 s each)."
        )
        if not args.yes:
            if not sys.stdin.isatty():
                print("ERROR: pass --yes to run non-interactively.", file=sys.stderr)
                return 2
            if input("This keeps the target's CPU busy for the whole run. Continue? [y/N] ").strip().lower() != "y":
                print("Cancelled.")
                return 1

        guard = None if args.no_auth else before_request
        report: dict[str, Any] = {
            "startedAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "baseUrl": args.base_url,
            "courseId": args.course_id,
            "questionSource": "file" if args.questions_file else "course",
            "questionsAvailable": len(available),
            "questions": [q["ref"] for q in questions],
            "passes": passes,
            "batchSize": args.batch_size,
            "timeout": args.timeout,
            "host": os.uname().nodename if hasattr(os, "uname") else None,
        }

        if not args.no_warmup:
            for condition in CONDITIONS:
                if guard is not None:
                    guard()
                outcome = classify(send(condition, checks.PROBE_QUESTION))
                if outcome != "ok":
                    raise Abort(f"The warm-up request for {condition} failed ({outcome}); nothing was measured.")

        aborted: str | None = None
        samples: list[Sample] = []
        try:
            run_passes(
                send,
                questions=questions,
                passes=passes,
                batch_size=args.batch_size,
                before_request=guard,
                samples=samples,
            )
        except Abort as exc:
            aborted = str(exc)
    except Abort as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        session.logout()

    report["finishedAt"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    report["aborted"] = aborted
    report["repeatability"] = summarize_repeatability(samples)
    report["orderTiming"] = summarize_orders(samples, passes)
    report["samples"] = samples
    if aborted is None:
        print()
        print(render_report(report))
    else:
        # A partial run compares orders on unequal work, so no table is shown;
        # the samples that did finish are still written for inspection.
        print(f"ERROR: {aborted}", file=sys.stderr)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0 if aborted is None else 1


if __name__ == "__main__":
    sys.exit(main())
