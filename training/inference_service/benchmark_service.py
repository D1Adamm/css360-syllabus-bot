"""Benchmark-only inference on the UWB VM: fixed experiment aliases, local Ollama.

A second, separate service beside `ollama_service.py`, for the controlled
CSS 360 benchmark and for nothing else. The production service maps *courses
and registered versions* to Ollama models and answers students; this one maps
a fixed set of *experiment aliases* to Ollama tags and answers the backend's
research route. Neither reads the other's configuration. The two v4 adapters
are experiments with no registry row, and they become servable here without
being added to `FINETUNED_OLLAMA_MODELS`, which is the production mapping and
stays as it is.

Aliases, fixed in code
----------------------
    base          the unmodified base model, the control
    v2            the production adapter
    v3            the experimental CPU adapter
    v4_vm         the mixed-v4 adapter trained on the VM
    v4_tillicum   the mixed-v4 adapter trained on Tillicum

Neither v4 experiment is a version. They are aliases here and lineage ids in
`evaluation/model_lineage.json`; nothing in this service or in the backend
route calls either of them v5, registers it, or serves it to a classroom.

What a request may say
----------------------
An alias and a prompt. The alias -> Ollama tag mapping is server configuration
(`CSS360_BENCHMARK_OLLAMA_MODELS`), validated against the allowlist above at
startup. A request can name no tag, no course, no path, no URL, and nothing
about decoding: unknown fields are refused, not ignored.

One decoding recipe
-------------------
Every alias is generated with the shared grounded options — the same
`build_generation_options` the production service uses, with the context
window pinned to its default rather than read from the environment — so two
conditions in a benchmark differ in the weights and in nothing else. The
options actually sent, and the SHA-256 of the prompt actually sent, are
echoed on every response so the backend can check both.

Which bytes answered
--------------------
A tag is a name, and `ollama create` can put new bytes under an old name.
Every answer therefore carries `modelDigest`, the manifest digest Ollama
reports for the tag at the moment of the call, looked up fresh per
generation. The backend records it beside the lineage record's digest so a
saved result names the artifact and not only its label.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from helpers import CourseAdapterError
from ollama_service import (
    DEFAULT_NUM_CTX,
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    OllamaUnavailable,
    available_model_names,
    build_generation_options,
    extract_chat_answer,
    normalize_ollama_model_name,
    resolve_ollama_base_url,
)

#: Loopback only, like the production service, and on its own port so the two
#: can run side by side. There is no host override.
BIND_HOST = "127.0.0.1"
DEFAULT_PORT = 9002
PORT_ENV = "BENCHMARK_INFERENCE_PORT"

MODEL_MAP_ENV = "CSS360_BENCHMARK_OLLAMA_MODELS"
TIMEOUT_ENV = "CSS360_BENCHMARK_OLLAMA_TIMEOUT_SECONDS"
KEEP_ALIVE_ENV = "CSS360_BENCHMARK_KEEP_ALIVE"

CONTROL_ALIAS = "base"
EXPERIMENT_ALIASES: tuple[str, ...] = ("v2", "v3", "v4_vm", "v4_tillicum")
ALIASES: tuple[str, ...] = (CONTROL_ALIAS,) + EXPERIMENT_ALIASES

Alias = Literal["base", "v2", "v3", "v4_vm", "v4_tillicum"]

#: The largest grounded prompt the backend's retrieval budget can produce is
#: a few thousand tokens; this is a ceiling against abuse, not a tuning knob.
MAX_PROMPT_CHARS = 32_000


class BenchmarkConfigError(Exception):
    """The alias mapping is malformed. Raised at startup, never on a request."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def resolve_port() -> int:
    raw = (os.environ.get(PORT_ENV) or "").strip()
    if not raw:
        return DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid {PORT_ENV} value: {raw!r}") from exc
    if port < 1 or port > 65535:
        raise RuntimeError(f"Port out of range: {port}")
    return port


def resolve_timeout_seconds() -> float:
    raw = (os.environ.get(TIMEOUT_ENV) or "").strip()
    if not raw:
        return DEFAULT_OLLAMA_TIMEOUT_SECONDS
    try:
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_OLLAMA_TIMEOUT_SECONDS


def resolve_keep_alive() -> Optional[str]:
    raw = (os.environ.get(KEEP_ALIVE_ENV) or "").strip()
    return raw or None


def parse_alias_map(raw: Optional[str]) -> Dict[str, str]:
    """Parse `CSS360_BENCHMARK_OLLAMA_MODELS` into {alias: ollamaTag}.

    Entries are `alias=ollamaTag`, separated by commas, whitespace or newlines::

        base=llama3.2:3b,v2=css360-ft-v2:latest,v4_vm=css360-v4-test:latest

    Every alias must be one of the fixed allowlist. A malformed or unknown entry
    fails the whole map, so a typo cannot silently drop a condition or, worse,
    add one the benchmark has no name for.
    """
    mapping: Dict[str, str] = {}
    text = (raw or "").replace(",", " ")
    for token in text.split():
        entry = token.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise BenchmarkConfigError(
                f"Invalid {MODEL_MAP_ENV} entry {entry!r}. Expected alias=ollamaTag."
            )
        alias, tag = entry.split("=", 1)
        alias = alias.strip()
        if alias not in ALIASES:
            raise BenchmarkConfigError(
                f"Unknown benchmark alias {alias!r} in {MODEL_MAP_ENV}. "
                f"Allowed aliases: {', '.join(ALIASES)}."
            )
        try:
            safe_tag = normalize_ollama_model_name(tag)
        except CourseAdapterError as exc:
            raise BenchmarkConfigError(
                f"Invalid Ollama tag for alias {alias!r} in {MODEL_MAP_ENV}: {exc}"
            ) from exc
        if alias in mapping and mapping[alias] != safe_tag:
            raise BenchmarkConfigError(
                f"{MODEL_MAP_ENV} maps alias {alias!r} to two different tags "
                f"({mapping[alias]} and {safe_tag})."
            )
        mapping[alias] = safe_tag
    return mapping


def load_alias_map() -> Dict[str, str]:
    """The mapping as currently configured. Re-read on every call."""
    return parse_alias_map(os.environ.get(MODEL_MAP_ENV))


def resolve_alias(alias: str, *, mapping: Optional[Dict[str, str]] = None) -> str:
    """The Ollama tag for one alias, or a `KeyError`-free refusal.

    Raises `BenchmarkConfigError` for an alias outside the allowlist (which a
    validated request can never carry) and `LookupError` for an allowed alias
    the operator has not mapped. Never falls back to another alias's tag.
    """
    if alias not in ALIASES:
        raise BenchmarkConfigError(f"Unknown benchmark alias {alias!r}.")
    table = mapping if mapping is not None else load_alias_map()
    tag = table.get(alias)
    if not tag:
        raise LookupError(
            f'Benchmark alias "{alias}" is not mapped on this host. Add '
            f"{alias}=<ollama-tag> to {MODEL_MAP_ENV} and restart the service."
        )
    return tag


# --------------------------------------------------------------------------- #
# The one decoding recipe and the one call shape
# --------------------------------------------------------------------------- #


def benchmark_options() -> Dict[str, Any]:
    """The shared grounded decoding options, with the context window pinned.

    `build_generation_options` is the production service's own function, so
    the recipe cannot drift from what students get. The context window is
    passed explicitly rather than read from `FINETUNED_NUM_CTX`: a benchmark
    must not change shape because of an environment variable meant for the
    other service.
    """
    return build_generation_options(num_ctx=DEFAULT_NUM_CTX)


def build_chat_request(prompt: str, *, ollama_model: str) -> Dict[str, Any]:
    """One user turn carrying the prompt verbatim, the pinned options."""
    payload: Dict[str, Any] = {
        "model": ollama_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": benchmark_options(),
    }
    keep_alive = resolve_keep_alive()
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    return payload


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


#: Ollama's own per-generation accounting, in nanoseconds and token counts.
#: Copied by name from the chat body; nothing else in that body is forwarded.
OLLAMA_TIMING_FIELDS: tuple[tuple[str, str], ...] = (
    ("total_duration", "totalDurationNs"),
    ("load_duration", "loadDurationNs"),
    ("prompt_eval_count", "promptEvalCount"),
    ("prompt_eval_duration", "promptEvalDurationNs"),
    ("eval_count", "evalCount"),
    ("eval_duration", "evalDurationNs"),
)


def extract_ollama_timings(data: Any) -> Dict[str, Any]:
    timings: Dict[str, Any] = {}
    if not isinstance(data, dict):
        return timings
    for source, target in OLLAMA_TIMING_FIELDS:
        value = data.get(source)
        if isinstance(value, int) and not isinstance(value, bool):
            timings[target] = value
    done_reason = data.get("done_reason")
    if isinstance(done_reason, str) and done_reason.strip():
        timings["doneReason"] = done_reason.strip()
    return timings


def model_digests(tags_payload: Any) -> Dict[str, str]:
    """Tag -> manifest digest from `/api/tags`, tags in their tagged form."""
    digests: Dict[str, str] = {}
    if not isinstance(tags_payload, dict):
        return digests
    models = tags_payload.get("models")
    if not isinstance(models, list):
        return digests
    for item in models:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("model")
        digest = item.get("digest")
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(digest, str) or not digest.strip():
            continue
        try:
            digests[normalize_ollama_model_name(name)] = digest.strip()
        except CourseAdapterError:
            continue
    return digests


def summarize_aliases(
    mapping: Dict[str, str],
    present: Optional[List[str]],
    digests: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """What /health says: every allowed alias, mapped or not, available or not,
    and the digest Ollama holds for its tag right now."""
    present_set = set(present or [])
    known = digests or {}
    rows: List[Dict[str, Any]] = []
    for alias in ALIASES:
        tag = mapping.get(alias)
        available = tag is not None and present is not None and tag in present_set
        rows.append(
            {
                "alias": alias,
                "ollamaModel": tag,
                "mapped": tag is not None,
                "available": available,
                "digest": known.get(tag) if available and tag is not None else None,
            }
        )
    return {"aliases": rows, "servable": [row["alias"] for row in rows if row["available"]]}


# --------------------------------------------------------------------------- #
# Ollama client
# --------------------------------------------------------------------------- #


def _refusal(code: str, message: str) -> Dict[str, str]:
    return {"code": code, "message": message}


async def fetch_tags_payload() -> Any:
    """The raw `/api/tags` body. Raises `OllamaUnavailable` if Ollama cannot say."""
    base_url = resolve_ollama_base_url()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/api/tags")
    except httpx.TimeoutException as exc:
        raise OllamaUnavailable("Ollama timed out listing models.") from exc
    except httpx.RequestError as exc:
        raise OllamaUnavailable(
            f"Ollama is unavailable: {exc.__class__.__name__}."
        ) from exc
    if response.status_code != 200:
        raise OllamaUnavailable(f"Ollama answered HTTP {response.status_code} to /api/tags.")
    try:
        return response.json()
    except ValueError as exc:
        raise OllamaUnavailable("Ollama returned invalid JSON for /api/tags.") from exc


async def lookup_model_digest(ollama_model: str) -> Optional[str]:
    """The digest Ollama holds for a tag right now, or None if it cannot say.

    Looked up per generation and never cached: a tag re-created between two
    conditions must show up as two digests. A failed lookup does not fail the
    generation; the answer is then recorded without a digest.
    """
    try:
        digests = model_digests(await fetch_tags_payload())
    except OllamaUnavailable:
        return None
    return digests.get(normalize_ollama_model_name(ollama_model))


#: One generation at a time: a CPU host shared with the production service and
#: the backend's own Base and RAG calls, all against the one Ollama.
_generation_lock = asyncio.Lock()


async def generate_with_ollama(prompt: str, *, ollama_model: str) -> Dict[str, Any]:
    """Ask Ollama for one answer. Failures carry a machine-readable code.

    503 for an Ollama that is down, slow, or failing; 409 for a tag that is
    mapped but was never created; 502 for anything else Ollama rejected or
    for a body this service cannot read. The details are for the backend's
    log: the backend forwards none of them.
    """
    base_url = resolve_ollama_base_url()
    timeout = resolve_timeout_seconds()
    payload = build_chat_request(prompt, ollama_model=ollama_model)

    async with _generation_lock:
        digest = await lookup_model_digest(ollama_model)
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{base_url}/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=503,
                detail=_refusal(
                    "ollama_timeout",
                    f"Ollama timed out after {timeout:.0f}s generating with {ollama_model}.",
                ),
            ) from exc
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=503,
                detail=_refusal(
                    "ollama_unavailable",
                    f"Ollama is unavailable ({exc.__class__.__name__}).",
                ),
            ) from exc
        elapsed = time.perf_counter() - started

    if response.status_code == 404:
        raise HTTPException(
            status_code=409,
            detail=_refusal(
                "model_missing",
                f'Ollama tag "{ollama_model}" is mapped but does not exist on this host.',
            ),
        )
    if response.status_code >= 500:
        raise HTTPException(
            status_code=503,
            detail=_refusal(
                "ollama_error",
                f"Ollama returned HTTP {response.status_code} for {ollama_model}: "
                f"{response.text[:500]}",
            ),
        )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=_refusal(
                "ollama_rejected",
                f"Ollama rejected the request for {ollama_model} "
                f"(HTTP {response.status_code}): {response.text[:500]}",
            ),
        )

    try:
        data = response.json()
        answer = extract_chat_answer(data)
    except ValueError as exc:
        raise HTTPException(
            status_code=502, detail=_refusal("malformed_ollama_response", str(exc))
        ) from exc

    return {
        "answer": answer,
        "generationSeconds": elapsed,
        "options": payload["options"],
        "ollama": extract_ollama_timings(data),
        "modelDigest": digest,
    }


# --------------------------------------------------------------------------- #
# HTTP contract
# --------------------------------------------------------------------------- #


class GenerateRequest(BaseModel):
    """An alias and a prompt. Anything else is refused, not ignored."""

    model_config = ConfigDict(extra="forbid")

    alias: Alias = Field(..., description="One of the fixed benchmark aliases")
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=MAX_PROMPT_CHARS,
        description="The exact text to send as the one user turn",
    )


class GenerateResponse(BaseModel):
    alias: str
    model: str
    modelDigest: Optional[str] = None
    answer: str
    promptSha256: str
    options: Dict[str, Any]
    generationSeconds: float
    ollama: Dict[str, Any]


class AliasSummary(BaseModel):
    alias: str
    ollamaModel: Optional[str] = None
    mapped: bool
    available: bool
    digest: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    engine: str
    aliases: List[AliasSummary]
    servable: List[str]
    detail: Optional[str] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Fail at startup on a malformed mapping rather than on the first request.
    mapping = load_alias_map()
    print(
        "CSS 360 benchmark service: {0} of {1} aliases mapped via {2}".format(
            len(mapping), len(ALIASES), resolve_ollama_base_url()
        ),
        flush=True,
    )
    for alias in ALIASES:
        print(f"  {alias} -> {mapping.get(alias) or '(unmapped)'}", flush=True)
    yield


app = FastAPI(
    title="CSS 360 benchmark inference service (local Ollama)",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    mapping = load_alias_map()
    detail: Optional[str] = None
    digests: Dict[str, str] = {}
    try:
        payload = await fetch_tags_payload()
        present: Optional[List[str]] = available_model_names(payload)
        digests = model_digests(payload)
        status = "ok"
    except OllamaUnavailable as exc:
        present = None
        status = "unavailable"
        detail = str(exc)
    summary = summarize_aliases(mapping, present, digests)
    return HealthResponse(
        status=status,
        engine="ollama",
        aliases=[AliasSummary(**row) for row in summary["aliases"]],
        servable=summary["servable"],
        detail=detail,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest) -> GenerateResponse:
    if not body.prompt.strip():
        raise HTTPException(
            status_code=422,
            detail=_refusal("empty_prompt", "The prompt must not be blank."),
        )
    try:
        ollama_model = resolve_alias(body.alias)
    except LookupError as exc:
        raise HTTPException(
            status_code=409, detail=_refusal("alias_not_mapped", str(exc))
        ) from exc

    result = await generate_with_ollama(body.prompt, ollama_model=ollama_model)
    return GenerateResponse(
        alias=body.alias,
        model=ollama_model,
        modelDigest=result["modelDigest"],
        answer=result["answer"],
        promptSha256=prompt_sha256(body.prompt),
        options=result["options"],
        generationSeconds=result["generationSeconds"],
        ollama=result["ollama"],
    )


def main() -> None:
    import uvicorn

    port = resolve_port()
    print(f"Listening on http://{BIND_HOST}:{port} (loopback only)", flush=True)
    uvicorn.run(app, host=BIND_HOST, port=port, reload=False, workers=1)


if __name__ == "__main__":
    main()
