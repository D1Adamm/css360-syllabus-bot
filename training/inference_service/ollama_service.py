"""Per-course fine-tuned inference on the UWB VM, through the local Ollama.

The same service contract as `app.py` — `GET /health`, `GET /courses`,
`POST /generate` — with a different engine underneath. `app.py` loads Llama 3.2
in 4-bit on a Tillicum GPU and attaches one PEFT adapter per course. This module
holds no model at all: each course's adapter has been merged into an Ollama model
(`FROM llama3.2:3b` + `ADAPTER <course>.gguf`) on the VM, and a request is
answered by asking the local Ollama server for that model by name.

Why a second service rather than a switch inside the first
----------------------------------------------------------
`app.py` cannot start without CUDA, Transformers, PEFT and bitsandbytes, and its
process model — one base, adapters swapped under a lock — has no analogue here.
What the backend actually depends on is the HTTP contract, and that is what is
kept: `backend/app/finetuned_client.py` is unchanged and cannot tell the two
apart, except that this one is reachable without a GPU allocation, an SSH tunnel
or a person at a Duo prompt.

Course isolation, concretely
----------------------------
1. Every request names its course. There is no default and no fallback: an
   unmapped course is a 409, not a request served by whichever model exists.
2. The Ollama model is looked up from the course id *and* the version, both
   validated with the same rules `helpers.py` applies to a serving path.
3. The response echoes the course and version it used, and the backend refuses
   a response whose course does not match what it asked for.
4. Nothing here is answered by "the" model: with an empty mapping, every
   generate request is refused and `/health` reports nothing servable.

Prompting
---------
The Tillicum service wraps the question as one user turn with
`tokenizer.apply_chat_template(..., add_generation_prompt=True)` and decodes
greedily. `/api/chat` with one user message is the Ollama equivalent — the
model's own Llama 3.2 template is applied server-side — and the options below
carry the same decoding settings across: greedy, the same new-token cap, the
same repetition penalty applied over the whole context, the same seed.
"""

from __future__ import annotations

import asyncio
import os
import socket
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from helpers import (  # noqa: F401 - the map helpers are re-exported for callers
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_SEED,
    MODEL_MAP_ENV,
    CourseAdapterError,
    normalize_ollama_model_name,
    parse_model_map,
    validate_course_id,
    validate_model_version,
    validate_question,
)

#: Loopback only. The backend on the same VM is the one caller, and the tunnel
#: path this replaces was also a loopback listener (`127.0.0.1:9001`). There is
#: no override: a fine-tuned endpoint that answers for any course it is asked
#: about must not be reachable from another host.
BIND_HOST = "127.0.0.1"
DEFAULT_PORT = 9001

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 120.0
#: Ollama truncates a prompt that exceeds the context window from the front,
#: silently. A Fine-Tuned + RAG prompt puts its rules first and the syllabus
#: context after, so truncation would drop exactly the part that says "answer
#: only from the context". 4096 covers the largest grounded prompt the backend
#: builds with room to spare, and costs a few hundred megabytes of KV cache on
#: a 3B model.
DEFAULT_NUM_CTX = 4096
#: Reported as `model` on /health, where the Tillicum service reports its base
#: model id. The per-course Ollama models are built on top of this one.
DEFAULT_BASE_MODEL = "llama3.2:3b"


class OllamaUnavailable(Exception):
    """The local Ollama server did not answer, or answered with a server error."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def resolve_port() -> int:
    raw = (os.environ.get("INFERENCE_PORT") or "").strip()
    if not raw:
        return DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid INFERENCE_PORT value: {raw!r}") from exc
    if port < 1 or port > 65535:
        raise RuntimeError(f"Port out of range: {port}")
    return port


def resolve_ollama_base_url() -> str:
    return (os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")


def resolve_ollama_timeout_seconds() -> float:
    raw = (os.environ.get("FINETUNED_OLLAMA_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return DEFAULT_OLLAMA_TIMEOUT_SECONDS
    try:
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_OLLAMA_TIMEOUT_SECONDS


def resolve_num_ctx() -> int:
    raw = (os.environ.get("FINETUNED_NUM_CTX") or "").strip()
    if not raw:
        return DEFAULT_NUM_CTX
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_NUM_CTX
    return value if value >= 512 else DEFAULT_NUM_CTX


def resolve_keep_alive() -> Optional[str]:
    """How long Ollama keeps a fine-tuned model resident after a request.

    Ollama's default is five minutes, after which the next question pays the
    model load again — several seconds on a CPU host. Set to e.g. `30m` before
    a class. Passed through verbatim; unset means Ollama's default.
    """
    raw = (os.environ.get("FINETUNED_KEEP_ALIVE") or "").strip()
    return raw or None


def resolve_base_model() -> str:
    return (os.environ.get("FINETUNED_BASE_MODEL") or DEFAULT_BASE_MODEL).strip()


def load_model_map() -> Dict[str, Dict[str, str]]:
    """The mapping as currently configured. Re-read on every call.

    Cheap, and it means a corrected environment takes effect on restart without
    any cache to reason about — the same posture the backend takes with
    `FINETUNED_SERVICE_URL`.
    """
    return parse_model_map(os.environ.get(MODEL_MAP_ENV))


def _sorted_versions(versions: Dict[str, str]) -> List[str]:
    return sorted(versions, key=lambda name: int(name[1:]))


def resolve_course_model(
    course_id: str,
    version: Optional[str],
    *,
    mapping: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, str]:
    """Which Ollama model answers for this course, and under which version.

    An explicit version wins, for the same reason as in `helpers.py`: the
    backend resolved it from PostgreSQL, and honouring it is what makes the two
    sides checkable against each other. Without one, the highest mapped version
    answers — the equivalent of the Tillicum service's "highest published"
    fallback, and the only sensible reading of a mapping with no pointer file.

    Raises `CourseAdapterError` rather than falling back to any other course's
    model or to the base model.
    """
    safe_course_id = validate_course_id(course_id)
    table = mapping if mapping is not None else load_model_map()
    versions = table.get(safe_course_id)
    if not versions:
        raise CourseAdapterError(
            'No fine-tuned model is mapped for course "{0}" on this host. '
            "Add {1}@vN=<ollama-model> to {2} and restart the service.".format(
                safe_course_id, safe_course_id, MODEL_MAP_ENV
            )
        )
    if version:
        safe_version = validate_model_version(version)
        source = "requested"
        if safe_version not in versions:
            raise CourseAdapterError(
                'Course "{0}" has no mapped model for version {1}. Mapped '
                "versions: {2}.".format(
                    safe_course_id, safe_version, ", ".join(_sorted_versions(versions))
                )
            )
    else:
        safe_version = _sorted_versions(versions)[-1]
        source = "highest mapped"
    return {
        "courseId": safe_course_id,
        "version": safe_version,
        "ollamaModel": versions[safe_version],
        "versionSource": source,
    }


def build_generation_options(
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    seed: int = DEFAULT_SEED,
    num_ctx: Optional[int] = None,
) -> Dict[str, Any]:
    """The Tillicum decoding settings, in Ollama's vocabulary.

    `do_sample=False` is `temperature: 0`. `max_new_tokens` is `num_predict`.
    Transformers applies `repetition_penalty` to every token seen so far —
    prompt included — so the penalty window here is the whole context window
    rather than Ollama's 64-token default. (Spelled as the context size itself:
    Ollama documents `-1` as meaning the same thing, but current builds reject
    a negative value with HTTP 400.) The seed does nothing under greedy decoding
    on either side and is carried for the same reason it was set there: it
    says which run this is meant to reproduce.
    """
    context = int(num_ctx if num_ctx is not None else resolve_num_ctx())
    return {
        "num_predict": int(max_new_tokens),
        "temperature": 0,
        "repeat_penalty": float(repetition_penalty),
        "repeat_last_n": context,
        "seed": int(seed),
        "num_ctx": context,
    }


def build_chat_request(question: str, *, ollama_model: str) -> Dict[str, Any]:
    """One user turn, no system prompt — what `apply_chat_template` was given."""
    payload: Dict[str, Any] = {
        "model": ollama_model,
        "messages": [{"role": "user", "content": question}],
        "stream": False,
        "options": build_generation_options(),
    }
    keep_alive = resolve_keep_alive()
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    return payload


def extract_chat_answer(data: Any) -> str:
    """The assistant text out of a non-streaming `/api/chat` body."""
    if not isinstance(data, dict):
        raise ValueError("Ollama returned a non-object chat response.")
    message = data.get("message")
    if not isinstance(message, dict):
        raise ValueError("Ollama chat response has no message.")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Ollama chat response has no message content.")
    return content.strip()


def available_model_names(tags_payload: Any) -> List[str]:
    """Model names from `/api/tags`, in their tagged form."""
    if not isinstance(tags_payload, dict):
        return []
    models = tags_payload.get("models")
    if not isinstance(models, list):
        return []
    names: List[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("model")
        if not isinstance(name, str) or not name.strip():
            continue
        try:
            names.append(normalize_ollama_model_name(name))
        except CourseAdapterError:
            # A name this service could never have been configured with is
            # simply not one of ours; it must not turn /health into a 500.
            continue
    return names


def summarize_courses(
    mapping: Dict[str, Dict[str, str]], present: Optional[List[str]]
) -> Dict[str, Any]:
    """What /health says about the mapping against what Ollama actually has.

    `courses` follows the Tillicum shape exactly — courseId, versions,
    currentVersion — and lists only what can be answered right now: a mapped
    course whose model has not been created in Ollama yet is not a course the
    backend should be told is servable. `models` is the operator's view of the
    same facts, one row per mapping entry, with the reason a course is missing
    from `courses` visible as `available: false`.

    `present=None` means Ollama could not be asked; nothing is servable then.
    """
    present_set = set(present or [])
    courses: List[Dict[str, Any]] = []
    models: List[Dict[str, Any]] = []
    for course_id in sorted(mapping):
        versions = mapping[course_id]
        servable: List[str] = []
        for version in _sorted_versions(versions):
            model_name = versions[version]
            available = present is not None and model_name in present_set
            models.append(
                {
                    "courseId": course_id,
                    "version": version,
                    "ollamaModel": model_name,
                    "available": available,
                }
            )
            if available:
                servable.append(version)
        if servable:
            courses.append(
                {
                    "courseId": course_id,
                    "versions": servable,
                    "currentVersion": servable[-1],
                }
            )
    return {"courses": courses, "models": models}


# --------------------------------------------------------------------------- #
# Ollama client
# --------------------------------------------------------------------------- #


async def fetch_available_models() -> List[str]:
    """Names Ollama has locally. Raises `OllamaUnavailable` if it cannot say."""
    base_url = resolve_ollama_base_url()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/api/tags")
    except httpx.TimeoutException as exc:
        raise OllamaUnavailable(
            f"Ollama at {base_url} timed out listing models."
        ) from exc
    except httpx.RequestError as exc:
        raise OllamaUnavailable(
            f"Ollama at {base_url} is unavailable: {exc.__class__.__name__}."
        ) from exc
    if response.status_code != 200:
        raise OllamaUnavailable(
            f"Ollama at {base_url} answered HTTP {response.status_code} to /api/tags."
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise OllamaUnavailable(f"Ollama at {base_url} returned invalid JSON.") from exc
    return available_model_names(payload)


#: One generation at a time. Not for correctness — the model name carries the
#: course, so there is no selection step to interleave — but because this is a
#: CPU host that the backend's own Base and RAG calls already share, and the
#: backend serializes those for the same reason (`ollama_coordination`).
_generation_lock = asyncio.Lock()


async def generate_with_ollama(question: str, *, ollama_model: str) -> Dict[str, Any]:
    """Ask Ollama for one answer. Errors map onto the Tillicum service's codes.

    503 for an Ollama that is down, slow, or failing, so the backend reports it
    as "server error" and logs the body that says which. 409 for a model that
    is mapped but has not been created in Ollama — the operator action away
    from working that a missing published adapter was. 502 for anything else
    Ollama rejected.
    """
    base_url = resolve_ollama_base_url()
    timeout = resolve_ollama_timeout_seconds()
    payload = build_chat_request(question, ollama_model=ollama_model)

    async with _generation_lock:
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{base_url}/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Ollama at {base_url} timed out after {timeout:.0f}s generating "
                    f"with {ollama_model}."
                ),
            ) from exc
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Ollama at {base_url} is unavailable "
                    f"({exc.__class__.__name__}). Start Ollama and try again."
                ),
            ) from exc
        elapsed = time.perf_counter() - started

    if response.status_code == 404:
        raise HTTPException(
            status_code=409,
            detail=(
                f'Ollama model "{ollama_model}" is mapped but does not exist on this '
                "host. Create it with `ollama create` from the course's Modelfile."
            ),
        )
    if response.status_code >= 500:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Ollama returned HTTP {response.status_code} for {ollama_model}: "
                f"{response.text[:500]}"
            ),
        )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Ollama rejected the request for {ollama_model} "
                f"(HTTP {response.status_code}): {response.text[:500]}"
            ),
        )

    try:
        answer = extract_chat_answer(response.json())
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {"answer": answer, "generationSeconds": elapsed}


# --------------------------------------------------------------------------- #
# HTTP contract
# --------------------------------------------------------------------------- #


class GenerateRequest(BaseModel):
    question: str = Field(..., description="Student question to answer")
    course_id: str = Field(
        ...,
        alias="courseId",
        description=(
            "Which course's model to answer with. Required: there is no "
            "course-agnostic fine-tuned model."
        ),
    )
    model_version: Optional[str] = Field(
        default=None,
        alias="modelVersion",
        description="The registered version to use, e.g. v2. Omit for the highest mapped.",
    )

    model_config = {"populate_by_name": True}


class GenerateResponse(BaseModel):
    answer: str
    model: str
    courseId: str
    modelVersion: str
    adapterLoaded: bool
    generationSeconds: float


class CourseSummary(BaseModel):
    courseId: str
    versions: List[str]
    currentVersion: str


class ModelSummary(BaseModel):
    courseId: str
    version: str
    ollamaModel: str
    available: bool


class HealthResponse(BaseModel):
    status: str
    model: str
    engine: str
    #: True when Ollama answered and at least one mapped course can be served.
    #: Named `adapterLoaded` because every existing health check — the backend,
    #: the operator scripts — reads that key.
    adapterLoaded: bool
    hostname: str
    port: int
    ollamaUrl: str
    courses: List[CourseSummary]
    models: List[ModelSummary]
    expiresAt: Optional[float] = None
    secondsRemaining: Optional[float] = None
    detail: Optional[str] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Fail at startup on a malformed mapping rather than on the first request.
    mapping = load_model_map()
    print(
        "Local fine-tuned service: {0} mapped course(s) via {1}".format(
            len(mapping), resolve_ollama_base_url()
        ),
        flush=True,
    )
    for course_id, versions in sorted(mapping.items()):
        for version in _sorted_versions(versions):
            print(f"  {course_id} {version} -> {versions[version]}", flush=True)
    if not mapping:
        print(
            f"  (none — set {MODEL_MAP_ENV}=courseId@vN=ollamaModel)", flush=True
        )
    yield


app = FastAPI(
    title="Per-Course Fine-Tuned Inference Service (local Ollama)",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    mapping = load_model_map()
    detail: Optional[str] = None
    try:
        present: Optional[List[str]] = await fetch_available_models()
        status = "ok"
    except OllamaUnavailable as exc:
        present = None
        status = "unavailable"
        detail = str(exc)

    summary = summarize_courses(mapping, present)
    return HealthResponse(
        status=status,
        model=resolve_base_model(),
        engine="ollama",
        adapterLoaded=status == "ok" and bool(summary["courses"]),
        hostname=socket.gethostname(),
        port=resolve_port(),
        ollamaUrl=resolve_ollama_base_url(),
        courses=[CourseSummary(**item) for item in summary["courses"]],
        models=[ModelSummary(**item) for item in summary["models"]],
        expiresAt=None,
        secondsRemaining=None,
        detail=detail,
    )


@app.get("/courses")
async def courses() -> Dict[str, Any]:
    """Which courses this host can answer for, and with which versions."""
    mapping = load_model_map()
    try:
        present: Optional[List[str]] = await fetch_available_models()
    except OllamaUnavailable:
        present = None
    summary = summarize_courses(mapping, present)
    return {
        "engine": "ollama",
        "ollamaUrl": resolve_ollama_base_url(),
        "courses": summary["courses"],
        "models": summary["models"],
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest) -> GenerateResponse:
    try:
        question = validate_question(body.question)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        resolved = resolve_course_model(body.course_id, body.model_version)
    except CourseAdapterError as exc:
        # 409, as the Tillicum service answers for a course with nothing
        # published: the request is well formed and the service is healthy;
        # what is missing is a mapping, which an operator adds.
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    result = await generate_with_ollama(question, ollama_model=resolved["ollamaModel"])

    return GenerateResponse(
        answer=result["answer"],
        model=resolved["ollamaModel"],
        courseId=resolved["courseId"],
        modelVersion=resolved["version"],
        adapterLoaded=True,
        generationSeconds=result["generationSeconds"],
    )


def main() -> None:
    import uvicorn

    port = resolve_port()
    print(f"Listening on http://{BIND_HOST}:{port} (loopback only)", flush=True)
    uvicorn.run(app, host=BIND_HOST, port=port, reload=False, workers=1)


if __name__ == "__main__":
    main()
