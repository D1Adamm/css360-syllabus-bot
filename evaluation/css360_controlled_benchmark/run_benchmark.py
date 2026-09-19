#!/usr/bin/env python3
"""The CSS 360 controlled benchmark runner: one command to start, one to check.

Runs on the UWB VM against the protected research route on loopback, so the
proxy's 300-second limit never applies and the bearer token never leaves the
host. It reads the token from `backend/.env` (or `CSS360_BENCHMARK_TOKEN`) and
never prints, logs or saves it.

    python3 evaluation/css360_controlled_benchmark/run_benchmark.py preflight
    python3 evaluation/css360_controlled_benchmark/run_benchmark.py run
    python3 evaluation/css360_controlled_benchmark/run_benchmark.py status

`run` does the preflight, then a pilot of the first three questions of the
first set with every condition, verifies what it saved, and continues on its
own unless `--pilot-only` was given. It is resumable: every request's raw
response is written as soon as it arrives and every attempted condition is
appended to `records.jsonl`; a restart skips finished requests and repeats
nothing. `results/CURRENT` names the run a bare `run` resumes; `--new` starts
another.

Two question sets, never mixed: `repeat22` (the 2026-09-11 questions under the
new controlled settings) and `heldout` (new factual, unanswerable and
false-premise questions). Each question is asked on the pair route (RAG and
the four FT+RAG aliases) and on the standalone route (Base and the four FT
aliases), in groups of at most three conditions per request; the grounded
groups of a question are then compared on prompt hash, ordered-chunk hash,
template fingerprint and decoding, and a mismatch invalidates that question's
grounded comparison in the saved data.

Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS_ROOT = HERE / "results"
CURRENT_POINTER = RESULTS_ROOT / "CURRENT"

COURSE_ID = "css-360-winter-2026-a7rp"
DEFAULT_ORIGIN = "http://127.0.0.1:8001"
DEFAULT_SERVICE_HEALTH = "http://127.0.0.1:9002/health"
ROUTE_PREFIX = "/api/research/css360/benchmark"
ENV_FILE = ROOT / "backend" / ".env"
TOKEN_ENV = "CSS360_BENCHMARK_TOKEN"
LINEAGE_FILE = ROOT / "evaluation" / "model_lineage.json"
INDEX_FILE = ROOT / "backend" / "data" / "indexes" / f"{COURSE_ID}.json"
SYLLABUS_FILE = ROOT / "backend" / "course_data" / COURSE_ID / "syllabus.txt"
EXPORT_DIR = ROOT / "data" / "exports" / COURSE_ID

SETS = {
    "repeat22": HERE / "questions_repeat22.json",
    "heldout": HERE / "questions_heldout.json",
}
CONDITIONS = {
    "pair": ["rag", "ft_rag:v2", "ft_rag:v3", "ft_rag:v4_vm", "ft_rag:v4_tillicum"],
    "standalone": ["base", "ft:v2", "ft:v3", "ft:v4_vm", "ft:v4_tillicum"],
}
ALIASES = ("base", "v2", "v3", "v4_vm", "v4_tillicum")
ALIAS_LINEAGE_IDS = {"v2": "css360-v2", "v3": "css360-v3", "v4_vm": "css360-v4-vm", "v4_tillicum": "css360-v4-tillicum"}

DEFAULT_GROUP_SIZE = 3
DEFAULT_PAUSE_SECONDS = 2.0
CONDITION_TIMEOUT_SECONDS = 120.0  # the route's default; the request timeout is derived from it
REQUEST_TIMEOUT_SLACK = 60.0
MAX_ATTEMPTS = 4
RETRY_AFTER_CAP = 120

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_token(env_file: Path = ENV_FILE) -> str | None:
    """The bearer token, from the environment or the backend's .env. Never printed."""
    value = (os.environ.get(TOKEN_ENV) or "").strip()
    if value:
        return value
    if not env_file.is_file():
        return None
    token = None
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if line.startswith(f"{TOKEN_ENV}="):
            candidate = line.partition("=")[2].strip()
            if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in "\"'":
                candidate = candidate[1:-1]
            token = candidate or None  # last occurrence wins, as a loader would
    return token


def git_head(repo: Path = ROOT) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def groups(conditions: list[str], size: int) -> list[list[str]]:
    size = max(1, int(size))
    return [conditions[i:i + size] for i in range(0, len(conditions), size)]


def request_key(set_id: str, qid: str, route: str, group_index: int) -> str:
    return f"{set_id}/{qid}/{route}/g{group_index}"


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


class HttpResult:
    def __init__(self, status: int, body: Any, text: str, seconds: float, error: str | None = None,
                 headers: dict[str, str] | None = None) -> None:
        self.status = status
        self.body = body
        self.text = text
        self.seconds = seconds
        self.error = error
        self.headers = headers or {}


def http(method: str, url: str, *, body: dict | None = None, token: str | None = None, timeout: float = 30.0) -> HttpResult:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", "replace")
            status = response.status
            hdrs = {k.lower(): v for k, v in response.headers.items()}
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", "replace") if exc.fp else ""
        status = exc.code
        hdrs = {k.lower(): v for k, v in exc.headers.items()} if exc.headers else {}
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
        return HttpResult(0, None, "", time.perf_counter() - started, error=f"{exc.__class__.__name__}: {exc}")
    seconds = time.perf_counter() - started
    try:
        parsed = json.loads(text) if text else None
    except ValueError:
        parsed = None
    return HttpResult(status, parsed, text, seconds, headers=hdrs)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


def records_from_response(set_id: str, question: dict, route: str, group_index: int, group: list[str],
                          payload: dict, *, wall: float, attempts: int, asked_at: str) -> list[dict]:
    """One record per condition the response carries, in the response's order."""
    records = []
    by_condition = {c.get("condition"): c for c in payload.get("conditions", []) if isinstance(c, dict)}
    for condition in group:
        c = by_condition.get(condition)
        if c is None:
            records.append(_failed_record(set_id, question, route, group_index, condition, status=200,
                                          error="condition missing from response", wall=wall,
                                          attempts=attempts, asked_at=asked_at))
            continue
        records.append({
            "set": set_id,
            "questionId": question["id"],
            "kind": question.get("kind"),
            "route": route,
            "group": group_index,
            "condition": condition,
            "alias": c.get("alias"),
            "conditionKind": c.get("kind"),
            "status": c.get("status"),
            "outcome": c.get("outcome"),
            "error": c.get("error"),
            "answer": c.get("answer"),
            "servedTag": c.get("servedTag"),
            "servedDigest": c.get("servedDigest"),
            "tagMatchesLineage": c.get("tagMatchesLineage"),
            "digestMatchesLineage": c.get("digestMatchesLineage"),
            "decodingMatchesSpec": c.get("decodingMatchesSpec"),
            "promptEchoMatches": c.get("promptEchoMatches"),
            "lineageId": (c.get("lineage") or {}).get("lineageId"),
            "expectedTag": (c.get("lineage") or {}).get("expectedTag"),
            "timing": c.get("timing"),
            "promptSha256": payload.get("promptSha256"),
            "retrievalSetSha256": (payload.get("retrieval") or {}).get("setSha256"),
            "chunkIds": (payload.get("retrieval") or {}).get("chunkIds"),
            "templateSha256": (payload.get("promptTemplate") or {}).get("sha256"),
            "decoding": payload.get("decoding"),
            "requestStatus": 200,
            "requestWallSeconds": round(wall, 3),
            "requestAttempts": attempts,
            "askedAt": asked_at,
        })
    return records


def _failed_record(set_id: str, question: dict, route: str, group_index: int, condition: str, *,
                   status: int, error: str, wall: float, attempts: int, asked_at: str) -> dict:
    alias = "base" if condition in ("rag", "base") else condition.partition(":")[2]
    return {
        "set": set_id, "questionId": question["id"], "kind": question.get("kind"), "route": route,
        "group": group_index, "condition": condition, "alias": alias,
        "conditionKind": condition.partition(":")[0] if ":" in condition else condition,
        "status": "error", "outcome": "request_failed",
        "error": {"code": "request_failed", "message": error[:500]},
        "answer": None, "servedTag": None, "servedDigest": None, "tagMatchesLineage": None,
        "digestMatchesLineage": None, "decodingMatchesSpec": None, "promptEchoMatches": None,
        "lineageId": None, "expectedTag": None, "timing": {"wallSeconds": round(wall, 3)},
        "promptSha256": None, "retrievalSetSha256": None, "chunkIds": None, "templateSha256": None,
        "decoding": None, "requestStatus": status, "requestWallSeconds": round(wall, 3),
        "requestAttempts": attempts, "askedAt": asked_at,
    }


def records_from_failure(set_id: str, question: dict, route: str, group_index: int, group: list[str], *,
                         status: int, error: str, wall: float, attempts: int, asked_at: str) -> list[dict]:
    return [_failed_record(set_id, question, route, group_index, c, status=status, error=error, wall=wall,
                           attempts=attempts, asked_at=asked_at) for c in group]


def grounded_identity(payloads: list[dict]) -> dict:
    """Are a question's pair-route groups the same grounded comparison?

    Every group must carry the same prompt hash, ordered-chunk hash, template
    fingerprint and decoding. One group alone is trivially identical.
    """
    keys = ("promptSha256", "retrievalSetSha256", "templateSha256", "decoding")
    seen: dict[str, list[Any]] = {k: [] for k in keys}
    for p in payloads:
        seen["promptSha256"].append(p.get("promptSha256"))
        seen["retrievalSetSha256"].append((p.get("retrieval") or {}).get("setSha256"))
        seen["templateSha256"].append((p.get("promptTemplate") or {}).get("sha256"))
        seen["decoding"].append(json.dumps(p.get("decoding"), sort_keys=True))
    mismatches = {k: v for k, v in seen.items() if len({json.dumps(x, sort_keys=True) for x in v}) > 1 or None in v}
    return {
        "groups": len(payloads),
        "valid": bool(payloads) and not mismatches,
        "promptSha256": seen["promptSha256"][0] if payloads and not mismatches.get("promptSha256") else None,
        "retrievalSetSha256": seen["retrievalSetSha256"][0] if payloads and not mismatches.get("retrievalSetSha256") else None,
        "mismatches": {k: v for k, v in mismatches.items()},
    }


# --------------------------------------------------------------------------- #
# Overlap with training data
# --------------------------------------------------------------------------- #

STOPWORDS = {
    "a", "am", "an", "and", "any", "are", "as", "at", "be", "by", "can", "course", "did", "do", "does",
    "for", "from", "get", "have", "how", "i", "if", "in", "is", "it", "many", "me", "much", "my", "need",
    "of", "on", "or", "our", "should", "so", "the", "there", "this", "to", "we", "what", "when", "where",
    "which", "who", "why", "will", "with", "you", "your",
}
REJECT = {"jaccard": 0.60, "containment": 0.75, "ratio": 0.75}
REVIEW = {"jaccard": 0.45, "containment": 0.60, "ratio": 0.60}


def tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def similarity(a: str, b: str) -> dict[str, float]:
    ta, tb = tokens(a), tokens(b)
    inter = len(ta & tb)
    return {
        "jaccard": round(inter / len(ta | tb), 3) if ta and tb else 0.0,
        "containment": round(inter / min(len(ta), len(tb)), 3) if ta and tb else 0.0,
        "ratio": round(SequenceMatcher(None, a.lower(), b.lower()).ratio(), 3),
    }


def verdict_for(s: dict[str, float]) -> str:
    if any(s[k] >= REJECT[k] for k in REJECT):
        return "REJECT (too similar)"
    if any(s[k] >= REVIEW[k] for k in REVIEW):
        return "REVIEW (resolve by hand)"
    return "clearly held out"


def training_rows(export_dir: Path) -> tuple[list[dict], dict]:
    """(rows, files): every training-side record's question and response, and file facts."""
    rows: list[dict] = []
    files: dict[str, Any] = {}
    for name in ("approved-finetune.jsonl", "train.jsonl", "validation.jsonl"):
        path = export_dir / name
        if not path.is_file():
            files[name] = None
            continue
        count = 0
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except ValueError:
                continue
            count += 1
            question = str(record.get("question") or "").strip()
            instruction = str(record.get("instruction") or "").strip()
            if not question:
                # A grounded record's instruction is the rendered prompt with the
                # question inside it; a bare record's instruction is the question.
                marker = "Student question:\n"
                if marker in instruction:
                    question = instruction.split(marker, 1)[1].split("\n\n", 1)[0].strip()
                else:
                    question = instruction
            rows.append({
                "file": name,
                "format": record.get("format") or ("grounded" if "Student question:" in instruction else "bare"),
                "kind": record.get("kind"),
                "question": question,
                "response": str(record.get("response") or ""),
            })
        files[name] = {"sha256": sha256_file(path), "records": count}
    manifest = export_dir / "manifest.json"
    files["manifest.json"] = {"sha256": sha256_file(manifest)} if manifest.is_file() else None
    if manifest.is_file():
        try:
            m = json.loads(manifest.read_text(encoding="utf-8"))
            files["manifest.json"].update({k: m.get(k) for k in ("datasetVersion", "trainExamples", "validationExamples", "totalExamples", "createdAt")})
        except ValueError:
            pass
    return rows, files


def overlap_for_question(question: str, key_facts: list[str], rows: list[dict]) -> dict:
    """Nearest training question by the three measures, plus key facts stated in training responses."""
    best: dict | None = None
    for row in rows:
        s = similarity(question, row["question"])
        if best is None or max(s.values()) > max(best["measures"].values()):
            best = {"file": row["file"], "format": row["format"], "kind": row["kind"], "question": row["question"], "measures": s}
    facts_seen: list[dict] = []
    for fact in key_facts:
        needle = " ".join(fact.lower().split())
        # A key fact is a short phrase; look for its content words together in one response.
        words = [w for w in tokens(needle) if len(w) > 2]
        for row in rows:
            haystack = row["response"].lower()
            if words and all(w in haystack for w in words):
                facts_seen.append({"fact": fact, "file": row["file"], "format": row["format"], "question": row["question"]})
                break
    verdict = verdict_for(best["measures"]) if best else "no training rows"
    return {
        "questionVerdict": verdict,
        "nearest": best,
        "keyFactsInTrainingResponses": facts_seen,
        "keyFactCoverage": f"{len(facts_seen)}/{len(key_facts)}" if key_facts else "0/0",
    }


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #


def lineage_expectations(path: Path = LINEAGE_FILE) -> dict[str, dict]:
    record = json.loads(path.read_text(encoding="utf-8"))
    base = record.get("baseModel") or {}
    expected = {"base": {"tag": base.get("ollamaTag"), "digest": base.get("ollamaDigest"), "lineageId": None}}
    by_id = {a.get("lineageId"): a for a in record.get("artifacts", [])}
    for alias, lineage_id in ALIAS_LINEAGE_IDS.items():
        a = by_id.get(lineage_id) or {}
        expected[alias] = {"tag": (a.get("ollama") or {}).get("tag"), "digest": (a.get("ollama") or {}).get("digest"), "lineageId": lineage_id}
    # v1 is in the record and deliberately not an alias; note it for the report.
    v1 = by_id.get("css360-v1") or {}
    expected["_v1_note"] = {"lineageId": "css360-v1", "tag": (v1.get("ollama") or {}).get("tag"),
                            "digest": (v1.get("ollama") or {}).get("digest"), "role": v1.get("role"),
                            "includeInControlledClaims": v1.get("includeInControlledClaims"),
                            "ggufDtype": (v1.get("gguf") or {}).get("tensorDtype"),
                            "adapterDtype": (v1.get("adapter") or {}).get("safetensorsDtype")}
    return expected


def digest_prefix_matches(served: str | None, expected: str | None) -> bool | None:
    if not served or not expected:
        return None
    a, b = served.lower(), expected.lower()
    for prefix in ("sha256:", "sha256-"):
        a = a[len(prefix):] if a.startswith(prefix) else a
        b = b[len(prefix):] if b.startswith(prefix) else b
    return a.startswith(b) if len(b) <= len(a) else b.startswith(a)


def compare_service_health(health: dict, expected: dict[str, dict]) -> dict:
    rows = {r.get("alias"): r for r in health.get("aliases", []) if isinstance(r, dict)}
    out: dict[str, Any] = {"ok": True, "aliases": {}}
    for alias in ALIASES:
        row = rows.get(alias) or {}
        exp = expected.get(alias) or {}
        tag_ok = bool(row.get("ollamaModel")) and _normalize_tag(row.get("ollamaModel")) == _normalize_tag(exp.get("tag") or "")
        digest_ok = digest_prefix_matches(row.get("digest"), exp.get("digest"))
        entry = {
            "mapped": bool(row.get("mapped")), "available": bool(row.get("available")),
            "servedTag": row.get("ollamaModel"), "expectedTag": exp.get("tag"), "tagMatches": tag_ok,
            "servedDigest": row.get("digest"), "expectedDigest": exp.get("digest"), "digestMatches": digest_ok,
        }
        if not (entry["mapped"] and entry["available"] and tag_ok and digest_ok is True):
            out["ok"] = False
        out["aliases"][alias] = entry
    return out


def _normalize_tag(name: str) -> str:
    value = (name or "").strip()
    head, sep, tail = value.rpartition("/")
    if ":" not in tail:
        tail += ":latest"
    return head + sep + tail if sep else tail


def preflight(args: argparse.Namespace, *, token: str | None, log) -> dict:
    report: dict[str, Any] = {"at": now(), "ok": True, "problems": []}

    def problem(text: str) -> None:
        report["ok"] = False
        report["problems"].append(text)
        log(f"preflight: PROBLEM {text}")

    if not token:
        problem(f"no benchmark token: set {TOKEN_ENV} or enable the route with backend/scripts/css360_benchmark_env.py")

    health = http("GET", f"{args.origin}/api/health", timeout=15)
    report["backendHealth"] = {"status": health.status, "body": health.body if isinstance(health.body, dict) else None}
    if health.status != 200:
        problem(f"backend health answered {health.status or health.error}")

    probe = http("POST", f"{args.origin}{ROUTE_PREFIX}/standalone", body={"question": "x", "conditions": ["base"]}, timeout=15)
    report["routeWithoutToken"] = probe.status
    if probe.status == 404:
        problem("the research route answers 404: not enabled or not configured")
    elif probe.status != 401:
        problem(f"the research route without a token answered {probe.status or probe.error}, expected 401")

    service = http("GET", args.service_health, timeout=15)
    if service.status == 200 and isinstance(service.body, dict):
        expected = lineage_expectations()
        comparison = compare_service_health(service.body, expected)
        report["serviceHealth"] = {"status": service.body.get("status"), "servable": service.body.get("servable")}
        report["tagsAndDigests"] = comparison
        report["v1"] = expected.get("_v1_note")
        if not comparison["ok"]:
            problem("a mapped alias is missing, unavailable, or its tag/digest disagrees with evaluation/model_lineage.json")
    else:
        problem(f"benchmark service health answered {service.status or service.error}")
        report["serviceHealth"] = None

    for name, path in SETS.items():
        if not path.is_file():
            problem(f"question set {name} missing at {path.name}")
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            report.setdefault("questionSets", {})[name] = {"sha256": sha256_file(path), "count": len(data.get("questions", []))}

    report["fingerprints"] = {
        "gitHead": git_head(),
        "indexSha256": sha256_file(INDEX_FILE),
        "indexChunkCount": _index_chunk_count(INDEX_FILE),
        "syllabusSha256": sha256_file(SYLLABUS_FILE),
        "hostname": socket.gethostname(),
    }
    rows, files = training_rows(EXPORT_DIR)
    report["trainingExport"] = {"files": files, "rows": len(rows)}
    if not rows:
        log("preflight: no training export present on this host; the VM-side overlap check will be empty")
    return report


def _index_chunk_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    chunks = data.get("chunks") if isinstance(data, dict) else None
    return len(chunks) if isinstance(chunks, list) else None


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


class Run:
    def __init__(self, run_dir: Path, args: argparse.Namespace, token: str | None) -> None:
        self.dir = run_dir
        self.args = args
        self.token = token
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "responses").mkdir(exist_ok=True)
        self.log_path = self.dir / "run.log"
        self.records_path = self.dir / "records.jsonl"
        self.status_path = self.dir / "status.json"
        self.manifest_path = self.dir / "manifest.json"

    def log(self, message: str) -> None:
        line = f"{now()} {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    # -- persistence ------------------------------------------------------------

    def response_path(self, key: str) -> Path:
        return self.dir / "responses" / (key.replace("/", "__") + ".json")

    def completed_keys(self) -> set[str]:
        done = set()
        for path in (self.dir / "responses").glob("*.json"):
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if saved.get("httpStatus") == 200:
                done.add(saved["key"])
        return done

    def append_records(self, records: list[dict]) -> None:
        with self.records_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_response(self, key: str, envelope: dict) -> None:
        path = self.response_path(key)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(envelope, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)

    def write_status(self, extra: dict) -> None:
        self.status_path.write_text(json.dumps(extra, indent=1) + "\n", encoding="utf-8")

    # -- one request --------------------------------------------------------------

    def ask(self, set_id: str, question: dict, route: str, group_index: int, group: list[str]) -> tuple[list[dict], dict | None]:
        key = request_key(set_id, question["id"], route, group_index)
        url = f"{self.args.origin}{ROUTE_PREFIX}/{route}"
        body = {"question": question["question"], "conditions": group}
        timeout = len(group) * (CONDITION_TIMEOUT_SECONDS + 5.0) + REQUEST_TIMEOUT_SLACK
        attempts = 0
        while True:
            attempts += 1
            asked_at = now()
            result = http("POST", url, body=body, token=self.token, timeout=timeout)
            envelope = {
                "key": key, "set": set_id, "questionId": question["id"], "route": route, "group": group_index,
                "conditions": group, "askedAt": asked_at, "httpStatus": result.status,
                "wallSeconds": round(result.seconds, 3), "attempt": attempts,
                "error": result.error, "response": result.body if result.status == 200 else None,
                "errorBody": None if result.status == 200 else (result.text[:2000] if result.text else None),
            }
            if result.status == 200 and isinstance(result.body, dict):
                self.save_response(key, envelope)
                records = records_from_response(set_id, question, route, group_index, group, result.body,
                                                wall=result.seconds, attempts=attempts, asked_at=asked_at)
                return records, result.body
            retryable = result.status in (0, 429, 502, 503, 504)
            if retryable and attempts < MAX_ATTEMPTS:
                wait = 30.0
                if result.status == 429:
                    try:
                        wait = min(float(result.headers.get("retry-after", "30")), RETRY_AFTER_CAP)
                    except ValueError:
                        wait = 30.0
                    wait = max(wait, 5.0)
                self.log(f"  {key}: HTTP {result.status or result.error}; retrying in {wait:.0f}s (attempt {attempts})")
                time.sleep(wait)
                continue
            self.save_response(key, envelope)
            error = f"HTTP {result.status}" if result.status else (result.error or "request failed")
            if result.text and result.status:
                error += ": " + result.text[:200]
            records = records_from_failure(set_id, question, route, group_index, group, status=result.status,
                                           error=error, wall=result.seconds, attempts=attempts, asked_at=asked_at)
            return records, None

    # -- the loop ------------------------------------------------------------------

    def plan(self, sets: list[str], group_size: int) -> list[tuple[str, dict, str, int, list[str]]]:
        plan = []
        for set_id in sets:
            data = json.loads(SETS[set_id].read_text(encoding="utf-8"))
            for question in data["questions"]:
                for route in ("pair", "standalone"):
                    for gi, group in enumerate(groups(CONDITIONS[route], group_size)):
                        plan.append((set_id, question, route, gi, group))
        return plan

    def execute(self, plan: list, *, label: str) -> None:
        done = self.completed_keys()
        total = len(plan)
        started = time.perf_counter()
        completed_now = 0
        for index, (set_id, question, route, gi, group) in enumerate(plan, 1):
            key = request_key(set_id, question["id"], route, gi)
            if key in done:
                continue
            records, payload = self.ask(set_id, question, route, gi, group)
            self.append_records(records)
            completed_now += 1
            outcomes = ",".join(f"{r['condition']}={r['outcome']}" for r in records)
            self.log(f"  [{label} {index}/{total}] {key}: {outcomes} ({records[0]['requestWallSeconds']}s)")
            if route == "pair" and gi == len(groups(CONDITIONS["pair"], self.args.group_size)) - 1:
                self.record_grounded_identity(set_id, question["id"])
            elapsed = time.perf_counter() - started
            remaining = sum(1 for (s, q, r, g, _) in plan if request_key(s, q["id"], r, g) not in done) - completed_now
            self.write_status({
                "phase": label, "at": now(), "requestsDone": total - remaining,
                "requestsTotal": total, "remaining": remaining,
                "secondsPerRequest": round(elapsed / completed_now, 1) if completed_now else None,
                "etaSeconds": round(remaining * elapsed / completed_now) if completed_now else None,
                "lastKey": key,
            })
            if self.args.pause_seconds > 0:
                time.sleep(self.args.pause_seconds)

    def record_grounded_identity(self, set_id: str, qid: str) -> None:
        payloads = []
        for gi in range(len(groups(CONDITIONS["pair"], self.args.group_size))):
            path = self.response_path(request_key(set_id, qid, "pair", gi))
            if path.is_file():
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved.get("httpStatus") == 200 and saved.get("response"):
                    payloads.append(saved["response"])
        identity = grounded_identity(payloads)
        identity_path = self.dir / "grounded_identity.jsonl"
        with identity_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"set": set_id, "questionId": qid, **identity}) + "\n")
        if not identity["valid"]:
            self.log(f"  {set_id}/{qid}: GROUNDED COMPARISON INVALID: {identity['mismatches']}")

    def verify_pilot(self, plan: list) -> list[str]:
        problems = []
        for set_id, question, route, gi, group in plan:
            key = request_key(set_id, question["id"], route, gi)
            path = self.response_path(key)
            if not path.is_file():
                problems.append(f"{key}: no saved response")
                continue
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("httpStatus") != 200:
                problems.append(f"{key}: HTTP {saved.get('httpStatus')} {saved.get('error') or ''}")
        records = [json.loads(l) for l in self.records_path.read_text(encoding="utf-8").splitlines() if l.strip()] if self.records_path.is_file() else []
        keys = {request_key(s, q["id"], r, g) for s, q, r, g, _ in plan}
        pilot_records = [r for r in records if request_key(r["set"], r["questionId"], r["route"], r["group"]) in keys]
        expected = sum(len(g) for _, _, _, _, g in plan)
        if len(pilot_records) != expected:
            problems.append(f"expected {expected} condition records for the pilot, found {len(pilot_records)}")
        if not any(r.get("outcome") == "scorable" for r in pilot_records):
            problems.append("no scorable condition in the pilot")
        if (self.dir / "grounded_identity.jsonl").is_file():
            for line in (self.dir / "grounded_identity.jsonl").read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                if not item["valid"]:
                    problems.append(f"{item['set']}/{item['questionId']}: grounded comparison invalid")
        return problems


def compute_status(run_dir: Path) -> dict:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8")) if (run_dir / "status.json").is_file() else {}
    records = []
    if (run_dir / "records.jsonl").is_file():
        records = [json.loads(l) for l in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_outcome: dict[str, int] = {}
    for r in records:
        by_outcome[r.get("outcome") or "?"] = by_outcome.get(r.get("outcome") or "?", 0) + 1
    per_set: dict[str, int] = {}
    for r in records:
        per_set[r["set"]] = per_set.get(r["set"], 0) + 1
    invalid = 0
    if (run_dir / "grounded_identity.jsonl").is_file():
        invalid = sum(1 for l in (run_dir / "grounded_identity.jsonl").read_text(encoding="utf-8").splitlines() if l.strip() and not json.loads(l)["valid"])
    return {"run": run_dir.name, "status": status, "conditionRecords": len(records), "byOutcome": by_outcome,
            "bySet": per_set, "groundedComparisonsInvalid": invalid, "finished": (run_dir / "FINISHED").is_file()}


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def resolve_run_dir(args: argparse.Namespace) -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    if getattr(args, "run_id", None):
        return RESULTS_ROOT / args.run_id
    if not getattr(args, "new", False) and CURRENT_POINTER.is_file():
        name = CURRENT_POINTER.read_text(encoding="utf-8").strip()
        if name and (RESULTS_ROOT / name).is_dir() and not (RESULTS_ROOT / name / "FINISHED").is_file():
            return RESULTS_ROOT / name
    name = "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RESULTS_ROOT / name


def cmd_preflight(args: argparse.Namespace) -> int:
    token = read_token()
    report = preflight(args, token=token, log=lambda m: print(m, flush=True))
    print(json.dumps({k: v for k, v in report.items() if k != "trainingExport"}, indent=1))
    print("training export:", json.dumps(report["trainingExport"]["files"], indent=1))
    print("preflight:", "OK" if report["ok"] else "PROBLEMS: " + "; ".join(report["problems"]))
    return 0 if report["ok"] else 1


def cmd_run(args: argparse.Namespace) -> int:
    token = read_token()
    run_dir = resolve_run_dir(args)
    run = Run(run_dir, args, token)
    CURRENT_POINTER.write_text(run_dir.name + "\n", encoding="utf-8")
    run.log(f"run {run_dir.name}: sets={args.sets} group_size={args.group_size} origin={args.origin}")

    report = preflight(args, token=token, log=run.log)
    (run_dir / "preflight.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    if not report["ok"] and not args.ignore_preflight:
        run.log("preflight failed; not starting. Fix the problems above or pass --ignore-preflight.")
        return 1

    manifest = {
        "runId": run_dir.name, "startedAt": now(), "origin": args.origin, "courseId": COURSE_ID,
        "sets": args.sets, "conditions": CONDITIONS, "groupSize": args.group_size,
        "groups": {route: groups(conds, args.group_size) for route, conds in CONDITIONS.items()},
        "pauseSeconds": args.pause_seconds, "conditionTimeoutSeconds": CONDITION_TIMEOUT_SECONDS,
        "fingerprints": report.get("fingerprints"), "questionSets": report.get("questionSets"),
        "tagsAndDigests": report.get("tagsAndDigests"), "v1": report.get("v1"),
        "trainingExportFiles": (report.get("trainingExport") or {}).get("files"),
        "note": "The bearer token is read from backend/.env and is never written here.",
    }
    if not run.manifest_path.is_file():
        run.manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    # Overlap of every question with the training export on this host.
    rows, _ = training_rows(EXPORT_DIR)
    overlap = {}
    for set_id in args.sets:
        data = json.loads(SETS[set_id].read_text(encoding="utf-8"))
        overlap[set_id] = {q["id"]: overlap_for_question(q["question"], q.get("keyFacts") or [], rows) for q in data["questions"]}
    (run_dir / "overlap_vm_export.json").write_text(json.dumps({"at": now(), "trainingRows": len(rows), "sets": overlap}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    flagged = [f"{s}/{qid}" for s, items in overlap.items() for qid, o in items.items() if o["questionVerdict"] != "clearly held out"]
    run.log(f"overlap vs VM export: {len(rows)} training rows; question-level flags: {flagged or 'none'}")

    plan = run.plan(args.sets, args.group_size)
    pilot_questions = []
    for set_id, question, *_ in plan:
        if question["id"] not in pilot_questions and set_id == args.sets[0]:
            pilot_questions.append(question["id"])
        if len(pilot_questions) >= args.pilot:
            break
    pilot_plan = [p for p in plan if p[0] == args.sets[0] and p[1]["id"] in pilot_questions]
    run.log(f"pilot: {len(pilot_questions)} question(s) x {len(groups(CONDITIONS['pair'], args.group_size)) + len(groups(CONDITIONS['standalone'], args.group_size))} requests")
    run.execute(pilot_plan, label="pilot")
    problems = run.verify_pilot(pilot_plan)
    if problems:
        for p in problems:
            run.log(f"pilot: PROBLEM {p}")
        run.log("pilot verification failed; stopping. Fix and re-run (finished requests are kept).")
        return 1
    run.log("pilot verified: every request saved, every condition recorded, grounded comparisons identical")
    if args.pilot_only:
        run.log("--pilot-only: stopping after the pilot")
        return 0

    run.execute(plan, label="full")
    post = http("GET", args.service_health, timeout=15)
    if post.status == 200 and isinstance(post.body, dict):
        (run_dir / "service_health_after.json").write_text(json.dumps(post.body, indent=1) + "\n", encoding="utf-8")
    (run_dir / "FINISHED").write_text(now() + "\n", encoding="utf-8")
    summary = compute_status(run_dir)
    run.log(f"finished: {json.dumps(summary)}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    run_dir = resolve_run_dir(args) if getattr(args, "run_id", None) else (
        RESULTS_ROOT / CURRENT_POINTER.read_text(encoding="utf-8").strip() if CURRENT_POINTER.is_file() else None)
    if run_dir is None or not run_dir.is_dir():
        print("no run found under", RESULTS_ROOT)
        return 1
    status = compute_status(run_dir)
    print(json.dumps(status, indent=1))
    log = run_dir / "run.log"
    if log.is_file():
        lines = log.read_text(encoding="utf-8").splitlines()
        print("--- last log lines ---")
        for line in lines[-args.tail:]:
            print(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--origin", default=DEFAULT_ORIGIN, help="the backend, loopback on the VM")
        p.add_argument("--service-health", default=DEFAULT_SERVICE_HEALTH, help="the benchmark service's /health")

    p = sub.add_parser("preflight", help="check the backend, the route, the service tags and digests, and the files")
    common(p)
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("run", help="pilot, verify, then run every set to completion; resumable")
    common(p)
    p.add_argument("--sets", nargs="+", default=["repeat22", "heldout"], choices=list(SETS))
    p.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE, help="conditions per request (1-3)")
    p.add_argument("--pilot", type=int, default=3, help="questions in the pilot")
    p.add_argument("--pilot-only", action="store_true")
    p.add_argument("--pause-seconds", type=float, default=DEFAULT_PAUSE_SECONDS)
    p.add_argument("--run-id", help="resume or create this run directory name")
    p.add_argument("--new", action="store_true", help="start a new run instead of resuming CURRENT")
    p.add_argument("--ignore-preflight", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status", help="progress of the current run")
    p.add_argument("--run-id")
    p.add_argument("--tail", type=int, default=12)
    p.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    if getattr(args, "group_size", None) is not None and not 1 <= args.group_size <= 3:
        parser.error("--group-size must be 1, 2 or 3")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
