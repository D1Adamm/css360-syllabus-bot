"""The CSS 360 controlled benchmark: a protected research route, not a lab page.

What it is for
--------------
The evidence record (`docs/css360-model-evolution.md`) shows that the historical
benchmark of 2026-09-11 compared pipelines rather than adapters. This route is
the controlled replacement: one question, one retrieval, one prompt, answered
by the base model and by any of four fixed adapters under one decoding
recipe, with every answer labelled by the lineage record. It exists so that
a benchmark runner (Phase 2B) can collect comparable answers; it is not a
classroom route, not an administrator page, and not a way to serve either v4
experiment to anyone.

Two routes, one shape:

    POST /api/research/css360/benchmark/pair        retrieval, grounded prompt
    POST /api/research/css360/benchmark/standalone  the bare question, no retrieval

What the caller controls: the question, and which of the allowlisted
conditions to run. Nothing else. The course is fixed to CSS 360, the retrieval
depth to the production default, the prompt template to the shared grounded
template, the decoding to the shared grounded options, and the alias -> model
mapping to the benchmark service's own configuration. A request that names a
course, a tag, a model, a path, a URL, a template or a `topK` is refused as
malformed rather than ignored.

How it is protected
-------------------
The routes are mounted always, so the route table can classify them, and they
answer 404 — the same 404 as a path that does not exist — unless three things
are configured: `CSS360_BENCHMARK_ENABLED`, a `CSS360_BENCHMARK_TOKEN` of at
least 32 characters, and `CSS360_BENCHMARK_SERVICE_URL`. When they are, a
request must carry the token as a bearer credential, compared in constant time
through fixed-length digests. No cookie is read and no session or database is
consulted: an administrator's browser session is not this credential, and
this credential reaches no other route.

Then the limits: a body cap enforced before the body is read, a question
length cap, a condition-count cap (the allowlist size, no duplicates), a
per-client failure limiter on bad tokens, a per-client request limiter, a
concurrency slot that refuses rather than queues, and a bounded timeout per
condition. A condition that fails or times out is recorded as such; the others
still run and the request still answers 200.

What a response may say
-----------------------
The question, the prompt bytes and their SHA-256, the ordered chunks and the
SHA-256 of that set, the decoding options, and per condition the answer, the
tag and the Ollama digest the service used, the lineage projection, the
timings, and any error by code. A condition the service answered from a tag
the lineage record does not name, or from bytes under that tag whose digest
the record does not name, is invalid and unscored; a generation that produced
no text is failed and unscored. All of them keep their timing, their served
tag and digest and their lineage and carry an error code; none carries an
answer. Every condition also carries an `outcome`, and the response a
`summary` that counts every requested condition as an attempt, so a later
report keeps failed and invalid generations in its denominator instead of
losing them. Never an internal path, port, hostname, token, environment
variable, or database detail: the lineage projection is built by name in
`research_benchmark_lineage`, upstream bodies stop at the backend log, and
every error message here is a fixed string.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal, Mapping, Sequence

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from app.auth.rate_limit import RateLimiter, client_address
from app.grounded_generation import (
    GROUNDED_TIMEOUT_SECONDS,
    PROMPT_TEMPLATE_NAME,
    grounded_options,
    prompt_template_fingerprint,
)
from app.grounded_rag import retrieve_and_prompt
from app.research_benchmark_client import (
    BenchmarkConditionError,
    generate_benchmark_answer,
    get_benchmark_service_url,
)
from app.research_benchmark_lineage import (
    ALIASES,
    CONTROL_ALIAS,
    CSS360_COURSE_ID,
    LineageUnavailable,
    lineage_for_alias,
    lineage_record_summary,
    load_lineage,
)
from app.retrieval_diversity import DEFAULT_TOP_K

logger = logging.getLogger(__name__)

ROUTE_PREFIX = "/api/research/css360/benchmark"

ENABLED_ENV = "CSS360_BENCHMARK_ENABLED"
TOKEN_ENV = "CSS360_BENCHMARK_TOKEN"
CONDITION_TIMEOUT_ENV = "CSS360_BENCHMARK_CONDITION_TIMEOUT_SECONDS"
MAX_CONCURRENT_ENV = "CSS360_BENCHMARK_MAX_CONCURRENT"
REQUESTS_PER_MINUTE_ENV = "CSS360_BENCHMARK_REQUESTS_PER_MINUTE"

TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

#: A token shorter than this does not enable the route: `openssl rand -hex 32`
#: produces 64 characters, and nothing shorter should be a research credential.
MIN_TOKEN_LENGTH = 32

#: Limits. Fixed in code; the environment can lower nothing below what is
#: sensible and raise nothing past the bound.
MAX_BODY_BYTES = 16 * 1024
MAX_QUESTION_CHARS = 1000
MAX_CONDITIONS = 5
DEFAULT_CONDITION_TIMEOUT_SECONDS = GROUNDED_TIMEOUT_SECONDS
MIN_CONDITION_TIMEOUT_SECONDS = 1.0
MAX_CONDITION_TIMEOUT_SECONDS = 600.0
#: Allowed past the httpx timeout before the condition is abandoned outright.
TIMEOUT_GRACE_SECONDS = 5.0
DEFAULT_MAX_CONCURRENT = 1
MAX_MAX_CONCURRENT = 4
DEFAULT_REQUESTS_PER_MINUTE = 30
MAX_REQUESTS_PER_MINUTE = 600
BUSY_RETRY_AFTER_SECONDS = 5

#: Bad bearer tokens from one client. A research credential is presented by a
#: script that has it; ten misses in ten minutes is not a typo budget.
AUTH_FAILURES_PER_CLIENT = RateLimiter(limit=10, window_seconds=600)
#: Accepted requests from one client per minute. Sized for a sequential runner.
REQUESTS_PER_CLIENT = RateLimiter(limit=DEFAULT_REQUESTS_PER_MINUTE, window_seconds=60)

#: The conditions each route may run. `rag` and `base` are the control, the
#: base model; the rest name one of the four fixed adapters. Nothing outside
#: these two tuples can be asked for.
PAIR_CONDITIONS: tuple[str, ...] = (
    "rag",
    "ft_rag:v2",
    "ft_rag:v3",
    "ft_rag:v4_vm",
    "ft_rag:v4_tillicum",
)
STANDALONE_CONDITIONS: tuple[str, ...] = (
    "base",
    "ft:v2",
    "ft:v3",
    "ft:v4_vm",
    "ft:v4_tillicum",
)
assert len(PAIR_CONDITIONS) == MAX_CONDITIONS
assert len(STANDALONE_CONDITIONS) == MAX_CONDITIONS

PairCondition = Literal["rag", "ft_rag:v2", "ft_rag:v3", "ft_rag:v4_vm", "ft_rag:v4_tillicum"]
StandaloneCondition = Literal["base", "ft:v2", "ft:v3", "ft:v4_vm", "ft:v4_tillicum"]


def split_condition(condition: str) -> tuple[str, str]:
    """`ft_rag:v4_vm` -> (`ft_rag`, `v4_vm`); the two controls name `base`."""
    if condition in ("rag", "base"):
        return condition, CONTROL_ALIAS
    kind, _, alias = condition.partition(":")
    if kind not in ("ft_rag", "ft") or alias not in ALIASES or alias == CONTROL_ALIAS:
        raise ValueError(f"Unknown benchmark condition: {condition!r}")
    return kind, alias


for _condition in PAIR_CONDITIONS + STANDALONE_CONDITIONS:
    split_condition(_condition)

#: How a condition's result is classed for the report. Every requested
#: condition is one attempt; the outcome says what kind. `scorable` is an
#: answer that may be judged. `failed_generation` ran and produced no text.
#: `invalid` produced text that cannot be attributed to the alias or compared
#: with the others. `error` never produced an answer at all.
Outcome = Literal["scorable", "failed_generation", "invalid", "error"]
FAILED_GENERATION_CODES = frozenset({"empty_answer"})
INVALID_CODES = frozenset(
    {"tag_mismatch", "digest_mismatch", "decoding_mismatch", "prompt_integrity", "alias_mismatch"}
)


def outcome_for(status: str, error_code: str | None) -> str:
    if status == "ok":
        return "scorable"
    if error_code in FAILED_GENERATION_CODES:
        return "failed_generation"
    if error_code in INVALID_CODES:
        return "invalid"
    return "error"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BenchmarkSettings:
    enabled: bool
    token: str | None
    service_url: str | None
    condition_timeout: float
    max_concurrent: int
    requests_per_minute: int


def _env_flag(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in TRUE_VALUES


def _bounded_float(name: str, default: float, low: float, high: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return min(high, max(low, value))


def _bounded_int(name: str, default: int, low: int, high: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return min(high, max(low, value))


def benchmark_settings() -> BenchmarkSettings:
    """Read on every call, like every other switch in this backend."""
    token = (os.getenv(TOKEN_ENV) or "").strip() or None
    service_url = get_benchmark_service_url()
    enabled = (
        _env_flag(ENABLED_ENV)
        and token is not None
        and len(token) >= MIN_TOKEN_LENGTH
        and service_url is not None
    )
    return BenchmarkSettings(
        enabled=enabled,
        token=token if enabled else None,
        service_url=service_url if enabled else None,
        condition_timeout=_bounded_float(
            CONDITION_TIMEOUT_ENV,
            DEFAULT_CONDITION_TIMEOUT_SECONDS,
            MIN_CONDITION_TIMEOUT_SECONDS,
            MAX_CONDITION_TIMEOUT_SECONDS,
        ),
        max_concurrent=_bounded_int(
            MAX_CONCURRENT_ENV, DEFAULT_MAX_CONCURRENT, 1, MAX_MAX_CONCURRENT
        ),
        requests_per_minute=_bounded_int(
            REQUESTS_PER_MINUTE_ENV, DEFAULT_REQUESTS_PER_MINUTE, 1, MAX_REQUESTS_PER_MINUTE
        ),
    )


def benchmark_enabled() -> bool:
    return benchmark_settings().enabled


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def bearer_token_matches(authorization: str | None, expected: str | None) -> bool:
    """`Authorization: Bearer <token>` against the configured token, constant time.

    Both sides are reduced to fixed-length digests before `hmac.compare_digest`,
    so the comparison takes the same time whatever the length or content of
    what was presented, and a presented value that is not a valid credential
    shape is refused on the same path as a wrong one.
    """
    if not expected:
        return False
    scheme, _, credentials = (authorization or "").strip().partition(" ")
    presented = credentials.strip()
    if scheme.lower() != "bearer" or not presented:
        return False
    return hmac.compare_digest(_digest(presented), _digest(expected))


def _not_found() -> HTTPException:
    # Indistinguishable from a path that is not mounted.
    return HTTPException(status_code=404, detail="Not Found")


def _too_many(code: str, message: str, retry_after: float) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail={"code": code, "message": message},
        headers={"Retry-After": str(max(1, int(round(retry_after))))},
    )


def _count(limiter: RateLimiter, key: str) -> None:
    """Record one event. `RateLimiter` counts events per key; its method is
    named for the credential-guessing limiters it was written for."""
    limiter.record_failure(key)


def require_css360_benchmark_access(request: Request) -> None:
    """The guard every benchmark route declares.

    404 unless enabled and configured; then the bearer token, with a
    per-client failure limiter in front of the comparison; then the
    per-client request limiter. No principal is resolved: cookies are not
    read, and no database is opened.
    """
    settings = benchmark_settings()
    if not settings.enabled:
        raise _not_found()

    client = client_address(request)
    retry = AUTH_FAILURES_PER_CLIENT.retry_after(client)
    if retry is not None:
        raise _too_many(
            "auth_failures", "Too many failed authentications. Try again later.", retry
        )
    if not bearer_token_matches(request.headers.get("authorization"), settings.token):
        _count(AUTH_FAILURES_PER_CLIENT, client)
        raise HTTPException(
            status_code=401,
            detail="A valid bearer token is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    REQUESTS_PER_CLIENT.limit = settings.requests_per_minute
    retry = REQUESTS_PER_CLIENT.retry_after(client)
    if retry is not None:
        raise _too_many("rate_limited", "Too many benchmark requests. Try again later.", retry)
    _count(REQUESTS_PER_CLIENT, client)


# --------------------------------------------------------------------------- #
# Request size, enforced before the body is read
# --------------------------------------------------------------------------- #


class ResearchBenchmarkBodyLimit:
    """Pure ASGI middleware for the benchmark prefix and nothing else.

    FastAPI reads a request body before it resolves the route's dependencies,
    so a size check in the guard would run after the bytes were already in
    memory. This runs first, from the headers alone: a body over the cap is
    refused with 413 and an unsized body with 411, without a byte of either
    being read. While the feature is disabled, every request under the prefix
    is answered with the same 404 as an unmounted path, whatever its method
    or size.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not str(scope.get("path", "")).startswith(ROUTE_PREFIX):
            await self.app(scope, receive, send)
            return
        if not benchmark_enabled():
            await JSONResponse({"detail": "Not Found"}, status_code=404)(scope, receive, send)
            return
        if str(scope.get("method", "")).upper() in ("POST", "PUT", "PATCH"):
            headers = Headers(scope=scope)
            raw = headers.get("content-length")
            if raw is None:
                await JSONResponse(
                    {"detail": {"code": "length_required",
                                "message": "A Content-Length header is required."}},
                    status_code=411,
                )(scope, receive, send)
                return
            try:
                length = int(raw)
            except ValueError:
                length = -1
            if length < 0:
                await JSONResponse(
                    {"detail": {"code": "bad_request", "message": "Invalid Content-Length."}},
                    status_code=400,
                )(scope, receive, send)
                return
            if length > self.max_bytes:
                await JSONResponse(
                    {"detail": {"code": "request_too_large",
                                "message": f"The request body may not exceed {self.max_bytes} bytes."}},
                    status_code=413,
                )(scope, receive, send)
                return
        await self.app(scope, receive, send)


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #

_semaphore: asyncio.Semaphore | None = None
_semaphore_capacity = 0


def _slot_semaphore(capacity: int) -> asyncio.Semaphore:
    global _semaphore, _semaphore_capacity
    if _semaphore is None or _semaphore_capacity != capacity:
        _semaphore = asyncio.Semaphore(capacity)
        _semaphore_capacity = capacity
    return _semaphore


@asynccontextmanager
async def benchmark_slot(capacity: int) -> AsyncIterator[None]:
    """Hold one of `capacity` slots for the whole request, or refuse at once.

    A benchmark request can take minutes on a CPU host. Queuing a second one
    behind it would keep a connection open past the proxy's patience and hide
    the fact that the host is busy; refusing with 429 and `Retry-After` tells
    the runner exactly that. The semaphore is never waited on, so it is never
    bound to an event loop.
    """
    semaphore = _slot_semaphore(capacity)
    if semaphore.locked():
        raise _too_many(
            "busy",
            "A benchmark request is already running on this host. Try again shortly.",
            BUSY_RETRY_AFTER_SECONDS,
        )
    await semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()


def reset_research_benchmark_state_for_tests() -> None:
    global _semaphore, _semaphore_capacity
    _semaphore = None
    _semaphore_capacity = 0
    AUTH_FAILURES_PER_CLIENT.clear()
    REQUESTS_PER_CLIENT.clear()
    REQUESTS_PER_CLIENT.limit = DEFAULT_REQUESTS_PER_MINUTE


# --------------------------------------------------------------------------- #
# Hashes
# --------------------------------------------------------------------------- #


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def retrieval_set_sha256(chunks: Sequence[Mapping[str, Any]]) -> str:
    """SHA-256 of the ordered retrieved set: ids, sections and texts, in order.

    Scores are left out — a float rendered two ways would give two hashes for
    the same set — and order is kept, because the prompt renders the chunks in
    this order and a reordered set is a different prompt.
    """
    canonical = [
        {"chunkId": chunk["chunkId"], "section": chunk["section"], "text": chunk["text"]}
        for chunk in chunks
    ]
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def _normalize_tag(name: str) -> str:
    value = name.strip()
    head, sep, tail = value.rpartition("/")
    if ":" not in tail:
        tail = tail + ":latest"
    return head + sep + tail if sep else tail


def _bare_digest(value: str) -> str:
    text = value.strip().lower()
    for prefix in ("sha256:", "sha256-"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text


def digest_matches(served: str | None, expected: str | None) -> bool | None:
    """Whether the digest Ollama served is the one the record names.

    The record carries the twelve-character form `ollama list` prints; Ollama's
    API reports the full manifest digest. The shorter is compared as a prefix
    of the longer. None when either side is missing: nothing to compare.
    """
    if not served or not expected:
        return None
    left, right = _bare_digest(served), _bare_digest(expected)
    if not left or not right:
        return None
    return left.startswith(right) if len(right) <= len(left) else right.startswith(left)


# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PairRequest(_Strict):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
    conditions: list[PairCondition] | None = Field(
        default=None, min_length=1, max_length=MAX_CONDITIONS,
        description="Which conditions to run, in order. Default: all five.",
    )


class StandaloneRequest(_Strict):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
    conditions: list[StandaloneCondition] | None = Field(
        default=None, min_length=1, max_length=MAX_CONDITIONS,
        description="Which conditions to run, in order. Default: all five.",
    )


class LineageSummary(_Strict):
    alias: str
    lineageId: str | None = None
    version: str | None = None
    role: str | None = None
    trainingLabel: str | None = None
    includeInControlledClaims: bool | None = None
    expectedTag: str
    ollamaDigest: str | None = None
    baseQuantization: str | None = None
    huggingFaceId: str | None = None
    adapterSha256: str | None = None
    adapterDtype: str | None = None
    ggufBlobSha256: str | None = None
    ggufDtype: str | None = None
    ggufBytes: int | None = None
    gitCommitSha: str | None = None


class OllamaTimings(_Strict):
    totalDurationNs: int | None = None
    loadDurationNs: int | None = None
    promptEvalCount: int | None = None
    promptEvalDurationNs: int | None = None
    evalCount: int | None = None
    evalDurationNs: int | None = None
    doneReason: str | None = None


class ConditionTiming(_Strict):
    wallSeconds: float
    generationSeconds: float | None = None
    ollama: OllamaTimings | None = None


class ConditionError(_Strict):
    code: str
    message: str


class ConditionResult(_Strict):
    condition: str
    kind: str
    alias: str
    status: Literal["ok", "error"]
    outcome: Outcome = "error"
    answer: str | None = None
    servedTag: str | None = None
    servedDigest: str | None = None
    tagMatchesLineage: bool | None = None
    digestMatchesLineage: bool | None = None
    decodingMatchesSpec: bool | None = None
    promptEchoMatches: bool | None = None
    lineage: LineageSummary | None = None
    timing: ConditionTiming
    error: ConditionError | None = None


class OutcomeSummary(_Strict):
    """Every requested condition counted once, whatever became of it.

    `attempted` is the denominator a report should use. The other four sum to
    it, so a failed generation or an invalid condition is never dropped from
    the count by being unscored.
    """

    attempted: int
    scorable: int
    failedGenerations: int
    invalid: int
    errors: int


class RetrievedChunk(_Strict):
    chunkId: str
    section: str
    text: str
    score: float


class RetrievalInfo(_Strict):
    topK: int
    chunkCount: int
    chunkIds: list[str]
    setSha256: str
    facets: list[str]


class PromptTemplateInfo(_Strict):
    name: str
    sha256: str


class LineageRecordInfo(_Strict):
    title: str | None = None
    schemaVersion: int | None = None
    generatedAt: str | None = None
    courseId: str | None = None


class BenchmarkResponse(_Strict):
    route: Literal["pair", "standalone"]
    courseId: str
    question: str
    prompt: str
    promptSha256: str
    promptTemplate: PromptTemplateInfo | None = None
    retrieval: RetrievalInfo | None = None
    retrievedChunks: list[RetrievedChunk]
    decoding: dict[str, float | int]
    conditionTimeoutSeconds: float
    lineageRecord: LineageRecordInfo
    conditions: list[ConditionResult]
    summary: OutcomeSummary
    requestSeconds: float


# --------------------------------------------------------------------------- #
# Running conditions
# --------------------------------------------------------------------------- #


def _normalized_question(question: str) -> str:
    trimmed = " ".join(question.split())
    if not trimmed:
        raise HTTPException(status_code=422, detail="Question must not be empty.")
    return trimmed


def _resolve_conditions(
    requested: Sequence[str] | None, allowed: tuple[str, ...]
) -> list[str]:
    if requested is None:
        return list(allowed)
    chosen = list(requested)
    if len(set(chosen)) != len(chosen):
        raise HTTPException(
            status_code=422, detail="Each condition may be requested at most once."
        )
    return chosen


def _lineage_or_503() -> dict[str, Any]:
    try:
        return load_lineage()
    except LineageUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


async def _retrieve(question: str) -> dict[str, Any]:
    """The production retrieval and prompt, for the fixed course and depth."""
    try:
        return await retrieve_and_prompt(CSS360_COURSE_ID, question, top_k=DEFAULT_TOP_K)
    except HTTPException as exc:
        if exc.status_code == 404:
            # Not the route's 404: the course has no usable index on this host.
            raise HTTPException(
                status_code=409,
                detail="The benchmark course has no usable syllabus index on this host.",
            ) from exc
        raise


def _errored(
    condition: str,
    kind: str,
    alias: str,
    started: float,
    lineage: LineageSummary | None,
    code: str,
    message: str,
    **extra: Any,
) -> ConditionResult:
    return ConditionResult(
        condition=condition,
        kind=kind,
        alias=alias,
        status="error",
        lineage=lineage,
        timing=ConditionTiming(wallSeconds=time.perf_counter() - started),
        error=ConditionError(code=code, message=message),
        **extra,
    )


async def _run_condition(
    condition: str,
    *,
    prompt: str,
    prompt_sha: str,
    record: Mapping[str, Any],
    settings: BenchmarkSettings,
) -> ConditionResult:
    """One condition, isolated, with its outcome attached."""
    result = await _answer_condition(
        condition, prompt=prompt, prompt_sha=prompt_sha, record=record, settings=settings
    )
    outcome = outcome_for(result.status, result.error.code if result.error else None)
    return result.model_copy(update={"outcome": outcome})


def summarize_outcomes(results: Sequence[ConditionResult]) -> OutcomeSummary:
    counts = Counter(result.outcome for result in results)
    return OutcomeSummary(
        attempted=len(results),
        scorable=counts["scorable"],
        failedGenerations=counts["failed_generation"],
        invalid=counts["invalid"],
        errors=counts["error"],
    )


async def _answer_condition(
    condition: str,
    *,
    prompt: str,
    prompt_sha: str,
    record: Mapping[str, Any],
    settings: BenchmarkSettings,
) -> ConditionResult:
    """One condition, isolated: any failure becomes an error entry.

    After the service answers, five checks in order, each an error that keeps
    the timing and discards the answer: the prompt echo, the decoding
    options, the served tag against the lineage record, the served digest
    against the record's, and finally whether any answer text was produced
    at all. The first four make the condition invalid; the last makes the
    generation a failure. None of them is scorable, and the error code says
    which it was. The digest Ollama served is recorded whatever the outcome;
    a digest the service could not look up is unknown, not a mismatch.
    """
    kind, alias = split_condition(condition)
    started = time.perf_counter()
    try:
        lineage = LineageSummary(**lineage_for_alias(alias, record))
    except LineageUnavailable as exc:
        return _errored(condition, kind, alias, started, None, "lineage_missing", str(exc))

    try:
        result = await asyncio.wait_for(
            generate_benchmark_answer(alias, prompt, timeout=settings.condition_timeout),
            timeout=settings.condition_timeout + TIMEOUT_GRACE_SECONDS,
        )
    except asyncio.TimeoutError:
        return _errored(
            condition, kind, alias, started, lineage, "timeout",
            f"The condition did not finish within {settings.condition_timeout:.0f} seconds.",
        )
    except BenchmarkConditionError as exc:
        return _errored(condition, kind, alias, started, lineage, exc.code, exc.message)
    except Exception:  # noqa: BLE001 - isolation: nothing may fail the other conditions
        logger.exception("Benchmark condition %s failed unexpectedly", condition)
        return _errored(
            condition, kind, alias, started, lineage, "unexpected",
            "The condition failed unexpectedly. See the backend log.",
        )

    wall = time.perf_counter() - started
    timing = ConditionTiming(
        wallSeconds=wall,
        generationSeconds=result["generationSeconds"],
        ollama=OllamaTimings(**result["ollama"]),
    )
    served_tag = result["model"]
    served_digest = result.get("modelDigest")
    tag_matches = _normalize_tag(served_tag) == _normalize_tag(lineage.expectedTag)
    digest_ok = digest_matches(served_digest, lineage.ollamaDigest)
    decoding_matches = result["options"] == grounded_options()
    prompt_echo_matches = result["promptSha256"] == prompt_sha
    common = {
        "condition": condition,
        "kind": kind,
        "alias": alias,
        "servedTag": served_tag,
        "servedDigest": served_digest,
        "tagMatchesLineage": tag_matches,
        "digestMatchesLineage": digest_ok,
        "decodingMatchesSpec": decoding_matches,
        "promptEchoMatches": prompt_echo_matches,
        "lineage": lineage,
        "timing": timing,
    }
    if not prompt_echo_matches:
        return ConditionResult(
            status="error",
            error=ConditionError(
                code="prompt_integrity",
                message="The service answered a different prompt. The answer was discarded.",
            ),
            **common,
        )
    if not decoding_matches:
        return ConditionResult(
            status="error",
            error=ConditionError(
                code="decoding_mismatch",
                message=(
                    "The service decoded with options other than the shared grounded "
                    "options. The answer was discarded."
                ),
            ),
            **common,
        )
    if not tag_matches:
        # The answer came from an artifact other than the one the record names
        # for this alias. Labelling it with that alias's lineage would be a
        # false record, so the condition is invalid and its answer is dropped;
        # the served tag and the timing stay so the mismatch is visible.
        return ConditionResult(
            status="error",
            error=ConditionError(
                code="tag_mismatch",
                message=(
                    "The service answered from a model tag the lineage record does "
                    "not name for this alias. The condition is invalid and unscored; "
                    "its answer was discarded."
                ),
            ),
            **common,
        )
    if digest_ok is False:
        # Same name, different bytes: the tag the record names now holds a
        # model the record does not describe, so the answer cannot be
        # attributed to the recorded artifact. Invalid; the observed digest
        # and the timing stay so the change is visible in the saved result.
        return ConditionResult(
            status="error",
            error=ConditionError(
                code="digest_mismatch",
                message=(
                    "The model behind the served tag has a digest the lineage record "
                    "does not name for this alias. The condition is invalid and "
                    "unscored; its answer was discarded."
                ),
            ),
            **common,
        )
    answer = result["answer"]
    if not answer.strip():
        # The service answered and Ollama ran, but no text came back. That is
        # a failed generation, not a service failure: it keeps the generation
        # time and Ollama's accounting, and it is not scored.
        return ConditionResult(
            status="error",
            error=ConditionError(
                code="empty_answer",
                message="The model produced no answer text. The generation failed and is unscored.",
            ),
            **common,
        )
    return ConditionResult(status="ok", answer=answer, **common)


async def _run_all(
    conditions: Sequence[str], *, prompt: str, record: Mapping[str, Any], settings: BenchmarkSettings
) -> list[ConditionResult]:
    prompt_sha = sha256_text(prompt)
    results: list[ConditionResult] = []
    for condition in conditions:
        results.append(
            await _run_condition(
                condition, prompt=prompt, prompt_sha=prompt_sha, record=record, settings=settings
            )
        )
    return results


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

router = APIRouter(prefix=ROUTE_PREFIX, tags=["research-benchmark"])


@router.post("/pair", response_model=BenchmarkResponse)
async def benchmark_pair(
    request: PairRequest,
    _access: None = Depends(require_css360_benchmark_access),
) -> BenchmarkResponse:
    """One retrieval, one grounded prompt, every requested condition.

    `retrieve_and_prompt` runs once, for the fixed course at the production
    retrieval depth, and its prompt text is what every condition receives,
    byte for byte. RAG and each Fine-Tuned + RAG condition therefore share the
    chunks, their order, the prompt and the decoding options, and differ in
    the weights alone.
    """
    settings = benchmark_settings()
    question = _normalized_question(request.question)
    conditions = _resolve_conditions(request.conditions, PAIR_CONDITIONS)
    record = _lineage_or_503()
    started = time.perf_counter()

    async with benchmark_slot(settings.max_concurrent):
        prepared = await _retrieve(question)
        prompt = prepared["prompt"]
        results = await _run_all(conditions, prompt=prompt, record=record, settings=settings)

    chunks = [RetrievedChunk(**chunk) for chunk in prepared["retrievedChunks"]]
    elapsed = time.perf_counter() - started
    logger.info(
        "css360 benchmark pair: %d condition(s), %d chunk(s), %.1fs",
        len(results), len(chunks), elapsed,
    )
    return BenchmarkResponse(
        route="pair",
        courseId=CSS360_COURSE_ID,
        question=prepared["question"],
        prompt=prompt,
        promptSha256=sha256_text(prompt),
        promptTemplate=PromptTemplateInfo(
            name=PROMPT_TEMPLATE_NAME, sha256=prompt_template_fingerprint()
        ),
        retrieval=RetrievalInfo(
            topK=DEFAULT_TOP_K,
            chunkCount=len(chunks),
            chunkIds=[chunk.chunkId for chunk in chunks],
            setSha256=retrieval_set_sha256(prepared["retrievedChunks"]),
            facets=[str(facet) for facet in prepared.get("facets") or []],
        ),
        retrievedChunks=chunks,
        decoding=grounded_options(),
        conditionTimeoutSeconds=settings.condition_timeout,
        lineageRecord=LineageRecordInfo(**lineage_record_summary(record)),
        conditions=results,
        summary=summarize_outcomes(results),
        requestSeconds=elapsed,
    )


@router.post("/standalone", response_model=BenchmarkResponse)
async def benchmark_standalone(
    request: StandaloneRequest,
    _access: None = Depends(require_css360_benchmark_access),
) -> BenchmarkResponse:
    """The bare question, no retrieval, every requested condition.

    The secondary, no-context comparison: does an adapter recall what it was
    trained on when nothing is retrieved? Every condition, the base model
    included, receives the whitespace-normalised question as its one user
    turn; the production Base wrapper is not applied, so the conditions differ
    in the weights alone here too.
    """
    settings = benchmark_settings()
    question = _normalized_question(request.question)
    conditions = _resolve_conditions(request.conditions, STANDALONE_CONDITIONS)
    record = _lineage_or_503()
    started = time.perf_counter()

    async with benchmark_slot(settings.max_concurrent):
        results = await _run_all(conditions, prompt=question, record=record, settings=settings)

    elapsed = time.perf_counter() - started
    logger.info("css360 benchmark standalone: %d condition(s), %.1fs", len(results), elapsed)
    return BenchmarkResponse(
        route="standalone",
        courseId=CSS360_COURSE_ID,
        question=question,
        prompt=question,
        promptSha256=sha256_text(question),
        promptTemplate=None,
        retrieval=None,
        retrievedChunks=[],
        decoding=grounded_options(),
        conditionTimeoutSeconds=settings.condition_timeout,
        lineageRecord=LineageRecordInfo(**lineage_record_summary(record)),
        conditions=results,
        summary=summarize_outcomes(results),
        requestSeconds=elapsed,
    )
