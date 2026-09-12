#!/usr/bin/env python3
"""Ask every benchmark question of every configuration through the deployed
administrator-only model-testing route, and record the raw answers.

    python3 evaluation/model_version_benchmark/run_benchmark.py --preflight
    python3 evaluation/model_version_benchmark/run_benchmark.py

Authentication is the normal admin login flow: an administrator signs in with
curl and saves the session cookie to a jar (see README.md beside this file);
this script only ever passes that jar to curl. It never sees a password, never
sets a cookie, and never touches the model registry, `current_version`,
publication state, or evaluations. The route it calls is read-only by design.

Configurations are run one at a time, all questions for one configuration
before the next, so the VM's Ollama swaps models as rarely as possible. Every
question is sent with exactly the same wording to every configuration. Results
are appended to results.json after each answer, so an interrupted run resumes
where it stopped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "questions.json"
RESULTS = HERE / "results.json"
LOG = HERE / "run.log"

DEFAULT_ORIGIN = "https://aiswe.uwb.edu"
DEFAULT_JAR = "/tmp/sml.jar"
COURSE_ID = "css-360-winter-2026-a7rp"
CSRF_HEADER = "X-Requested-With: SyllabusModelLab"
#: The backend's own upstream timeouts are 120 s (fine-tuned) and 60 s (RAG);
#: this only has to outlast them.
REQUEST_TIMEOUT_SECONDS = 300

#: The five configurations, in run order. `modelVersion` is omitted for RAG,
#: which the route requires.
CONFIGURATIONS = [
    {"key": "rag", "label": "RAG", "mode": "rag", "modelVersion": None},
    {"key": "ft_v2", "label": "Fine-Tuned v2", "mode": "fineTuned", "modelVersion": "v2"},
    {"key": "ftrag_v2", "label": "Fine-Tuned + RAG v2", "mode": "fineTunedRag", "modelVersion": "v2"},
    {"key": "ft_v3", "label": "Fine-Tuned v3", "mode": "fineTuned", "modelVersion": "v3"},
    {"key": "ftrag_v3", "label": "Fine-Tuned + RAG v3", "mode": "fineTunedRag", "modelVersion": "v3"},
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    with LOG.open("a") as handle:
        handle.write(line + "\n")


def curl(method: str, origin: str, path: str, jar: str, body: dict | None = None) -> tuple[int, str, float]:
    """One request through curl with the saved session jar. Returns (status, body, seconds)."""
    command = [
        "curl", "-sS", "-b", jar, "-X", method,
        "--max-time", str(REQUEST_TIMEOUT_SECONDS),
        "-H", CSRF_HEADER, "-H", "Accept: application/json",
        "-o", "-", "-w", "\n__STATUS__%{http_code}",
        f"{origin}{path}",
    ]
    if body is not None:
        command[1:1] = ["-H", "Content-Type: application/json", "--data-binary", json.dumps(body)]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        return 0, f"curl failed ({completed.returncode}): {completed.stderr.strip()}", elapsed
    text, _, status = completed.stdout.rpartition("\n__STATUS__")
    try:
        return int(status), text, elapsed
    except ValueError:
        return 0, completed.stdout, elapsed


def preflight(origin: str, jar: str) -> bool:
    """Read-only checks: the jar holds an administrator session, and both
    versions are servable. Nothing here generates anything."""
    status, text, _ = curl("GET", origin, "/api/auth/session", jar)
    if status != 200:
        log(f"preflight: GET /api/auth/session -> {status}: {text[:300]}")
        return False
    session = json.loads(text)
    user = session.get("user") or {}
    if user.get("role") != "admin":
        log(f"preflight: the session is not an administrator: {json.dumps(session)[:300]}")
        return False
    log(f"preflight: signed in as administrator {user.get('displayName') or user.get('email')}")

    status, text, _ = curl("GET", origin, "/api/fine-tuned/health", jar)
    if status != 200:
        log(f"preflight: GET /api/fine-tuned/health -> {status}: {text[:300]}")
        return False
    health = json.loads(text)
    course = next((c for c in health.get("courses", []) if c.get("courseId") == COURSE_ID), None)
    log(f"preflight: fine-tuned service status={health.get('status')} course={json.dumps(course)}")
    versions = set((course or {}).get("versions") or [])
    wanted = {c["modelVersion"] for c in CONFIGURATIONS if c["modelVersion"]}
    if not wanted <= versions:
        log(f"preflight: service does not serve {sorted(wanted - versions)}; servable: {sorted(versions)}")
        return False
    return True


def load_results() -> list[dict]:
    if RESULTS.is_file():
        return json.loads(RESULTS.read_text())
    return []


def save_results(results: list[dict]) -> None:
    tmp = RESULTS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(RESULTS)


def done_keys(results: list[dict]) -> set[tuple[str, str]]:
    return {(r["config"], r["questionId"]) for r in results if r.get("status") == 200}


def ask(origin: str, jar: str, config: dict, question: dict) -> dict:
    body = {
        "courseId": COURSE_ID,
        "mode": config["mode"],
        "question": question["question"],
        "topK": 4,
    }
    if config["modelVersion"]:
        body["modelVersion"] = config["modelVersion"]

    attempts = 0
    while True:
        attempts += 1
        status, text, elapsed = curl("POST", origin, "/api/model-testing/generate", jar, body)
        record = {
            "config": config["key"],
            "label": config["label"],
            "mode": config["mode"],
            "requestedVersion": config["modelVersion"],
            "questionId": question["id"],
            "question": question["question"],
            "category": question["category"],
            "answerableFromSyllabus": question["answerableFromSyllabus"],
            "status": status,
            "wallSeconds": round(elapsed, 2),
            "attempts": attempts,
            "askedAt": now(),
        }
        if status == 200:
            payload = json.loads(text)
            record["response"] = payload
            # The two facts the whole comparison rests on, checked here too.
            if payload.get("courseId") != COURSE_ID:
                record["anomaly"] = f"course mismatch: {payload.get('courseId')}"
            if config["modelVersion"] and payload.get("modelVersion") != config["modelVersion"]:
                record["anomaly"] = f"version mismatch: asked {config['modelVersion']}, got {payload.get('modelVersion')}"
            return record
        record["error"] = text[:2000]
        # One retry for the failures that are about the moment rather than the
        # request: an upstream timeout, an unreachable service, a dropped
        # connection. A 4xx is the request's own fault and is recorded as is.
        if attempts < 2 and (status == 0 or status >= 500):
            log(f"  {config['key']} {question['id']}: {status} on attempt {attempts}; retrying in 15 s")
            time.sleep(15)
            continue
        return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", default=DEFAULT_ORIGIN)
    parser.add_argument("--jar", default=DEFAULT_JAR, help="curl cookie jar written by the admin login")
    parser.add_argument("--preflight", action="store_true", help="Only check the session and the service")
    parser.add_argument("--only", nargs="*", help="Configuration keys to run (default: all)")
    args = parser.parse_args()

    if not Path(args.jar).is_file():
        log(f"No cookie jar at {args.jar}. Sign in as an administrator first (see README.md).")
        return 2
    if not preflight(args.origin, args.jar):
        return 1
    if args.preflight:
        return 0

    questions = json.loads(QUESTIONS.read_text())["questions"]
    results = load_results()
    finished = done_keys(results)
    configs = [c for c in CONFIGURATIONS if not args.only or c["key"] in args.only]
    total = len(configs) * len(questions)
    log(f"run: {len(questions)} questions x {len(configs)} configurations = {total}; {len(finished)} already done")

    for config in configs:
        for question in questions:
            if (config["key"], question["id"]) in finished:
                continue
            record = ask(args.origin, args.jar, config, question)
            results = [r for r in results if not (r["config"] == config["key"] and r["questionId"] == question["id"])]
            results.append(record)
            save_results(results)
            outcome = record.get("anomaly") or (
                f"{record['response']['modelVersion']} {record['response']['model']} "
                f"gen={record['response'].get('generationSeconds')}"
                if record["status"] == 200 else f"ERROR {record['status']} {record.get('error', '')[:120]}"
            )
            log(f"  {config['key']:9s} {question['id']}: {record['status']} {record['wallSeconds']:6.1f}s  {outcome}")

    failures = [r for r in results if r.get("status") != 200 or r.get("anomaly")]
    log(f"done: {len(results)} records, {len(failures)} failures/anomalies")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
