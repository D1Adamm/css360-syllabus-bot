"""What decides a classroom answer, and the fingerprint that names it.

A stored answer may be reused only while everything that produced it is
unchanged. This module gathers that "everything" for one course and approach
into a small JSON object (the identity) and hashes it (the fingerprint). A
cached row is found by its fingerprint, so a changed model, runtime, syllabus,
prompt, decoding option or retrieval rule is simply a different fingerprint:
the old row is no longer found and nothing has to be cleared by hand.

What is in the identity
-----------------------
Every approach: the classroom profile name and this host's CPU.

Base and RAG (the backend's own Ollama): the model tag and its Ollama digest
(the digest changes when the weights, template or parameters do), the Ollama
version, and the decoding options actually sent. Base adds a hash of its
instruction wrapper.

Fine-Tuned and Fine-Tuned + RAG: the course's activated model version from
PostgreSQL, the Ollama tag the fine-tuned service maps it to and that tag's
digest and Ollama version, read from the Ollama server the service itself
reports using (`ollamaUrl` in its `/health`), plus the output cap and a hash
of the service's source file when it is on this machine.

RAG and Fine-Tuned + RAG: `top_k`, a content hash of the course's syllabus
index (chunk text and embeddings), the embedding model's tag and digest, a
hash of the rendered grounded prompt template, and a hash of the retrieval
modules' source.

Unknown means no cache
----------------------
If any part cannot be established right now (a model is not installed, the
fine-tuned service is down, the course has no activated model or no index),
there is no identity, and the caller neither reads nor writes the cache for
that approach. The reason is returned so the preparation page can say why.

Cost
----
Model digests and versions come from Ollama over HTTP and are remembered for
`IDENTITY_TTL_SECONDS`, so a cache hit makes no HTTP call. The index hash is
recomputed only when the index file changes. A write re-reads everything
fresh after the generation and stores only if nothing moved meanwhile.

Not covered: the fine-tuned service's `FINETUNED_NUM_CTX` environment
setting, which the backend cannot see. Changing it calls for re-preparing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from app.course_model_resolution import resolve_current_course_model
from app.finetuned_client import get_finetuned_service_url
from app.grounded_generation import (
    CLASSROOM_NUM_PREDICT,
    CLASSROOM_PROFILE_NAME,
    build_grounded_prompt,
    classroom_options,
)
from app.ollama import OLLAMA_BASE_URL, OLLAMA_MODEL, build_base_model_prompt
from app.rag import OLLAMA_EMBEDDING_MODEL
from app.storage import get_course_artifact_storage

logger = logging.getLogger(__name__)

#: Bump to retire every stored answer at once (a change in what an identity means).
IDENTITY_VERSION = 1
#: How long model digests and versions are trusted before asking Ollama again.
IDENTITY_TTL_SECONDS = 30.0
#: Identity lookups must never hold a student's request for long.
IDENTITY_HTTP_TIMEOUT_SECONDS = 5.0

BASE = "base"
RAG = "rag"
FINE_TUNED = "fineTuned"
FINE_TUNED_RAG = "fineTunedRag"
CONDITIONS = (BASE, RAG, FINE_TUNED, FINE_TUNED_RAG)
GROUNDED_CONDITIONS = frozenset({RAG, FINE_TUNED_RAG})
FINE_TUNED_CONDITIONS = frozenset({FINE_TUNED, FINE_TUNED_RAG})

# Why an identity could not be established (also shown, reworded, to staff).
REASON_BASE_MODEL = "base_model_unavailable"
REASON_NO_INDEX = "syllabus_not_indexed"
REASON_EMBEDDING_MODEL = "embedding_model_unavailable"
REASON_NO_COURSE_MODEL = "no_course_model"
REASON_FINETUNED_SERVICE = "finetuned_service_unavailable"
REASON_FINETUNED_MODEL = "finetuned_model_unavailable"

#: The modules that decide which chunks a question retrieves and how they are
#: handed on. Editing any of them retires stored grounded answers.
#: `tests/test_answer_identity.py` checks this list against the imports.
RETRIEVAL_MODULES = (
    "course_rag.py",
    "rag.py",
    "retrieval_diversity.py",
    "retrieval_facets.py",
    "grounded_rag.py",
)

APP_DIR = Path(__file__).resolve().parent
FINETUNED_SERVICE_SOURCE = APP_DIR.parents[1] / "training" / "inference_service" / "ollama_service.py"


@dataclass(frozen=True)
class Identity:
    """One approach's identity for one course, and its fingerprint."""

    fields: dict[str, Any]
    fingerprint: str
    #: The course model version (fine-tuned approaches), for the post-check.
    model_version: str | None = None
    #: The Ollama tag expected to answer.
    model_tag: str | None = None


@dataclass(frozen=True)
class IdentityResult:
    identity: Identity | None
    reason: str | None = None


def fingerprint_of(fields: dict[str, Any]) -> str:
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ollama_tag(name: str) -> str:
    """`nomic-embed-text` and `nomic-embed-text:latest` are the same model to Ollama."""
    cleaned = name.strip()
    return cleaned if ":" in cleaned.rsplit("/", 1)[-1] else f"{cleaned}:latest"


# --------------------------------------------------------------------------- #
# Static parts: computed once per process
# --------------------------------------------------------------------------- #

_static_lock = threading.Lock()
_static: dict[str, str] = {}


def _once(key: str, compute) -> str:
    with _static_lock:
        if key not in _static:
            _static[key] = compute()
        return _static[key]


def host_identity() -> str:
    """This machine's CPU, as far as the operating system names it."""

    def compute() -> str:
        model = ""
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        model = line.split(":", 1)[1].strip()
                        break
        except OSError:
            model = ""
        return " | ".join(part for part in (model or platform.processor(), platform.machine()) if part)

    return _once("host", compute)


def base_prompt_fingerprint() -> str:
    """The Base approach's instruction wrapper, rendered around a placeholder."""
    return _once("basePrompt", lambda: _sha256_text(build_base_model_prompt("{question}")))


def grounded_prompt_fingerprint() -> str:
    """The classroom (concise) grounded prompt template, rendered with placeholders."""
    return _once(
        "groundedPrompt",
        lambda: _sha256_text(
            build_grounded_prompt(
                "{question}",
                [{"section": "{section}", "text": "{text}"}],
                ["{part one}", "{part two}"],
                concise=True,
            )
        ),
    )


def retrieval_code_fingerprint() -> str:
    def compute() -> str:
        digest = hashlib.sha256()
        for name in RETRIEVAL_MODULES:
            digest.update(name.encode("utf-8"))
            digest.update((APP_DIR / name).read_bytes())
        return digest.hexdigest()

    return _once("retrievalCode", compute)


def finetuned_service_fingerprint() -> str:
    """The fine-tuned service's source, when it lives in this checkout."""

    def compute() -> str:
        try:
            return hashlib.sha256(FINETUNED_SERVICE_SOURCE.read_bytes()).hexdigest()
        except OSError:
            return "unknown"

    return _once("finetunedService", compute)


# --------------------------------------------------------------------------- #
# The syllabus index: hashed when its file changes
# --------------------------------------------------------------------------- #

_index_lock = threading.Lock()
_index_memo: dict[str, tuple[tuple[int, int], str | None]] = {}


def _hash_index(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    chunks = data.get("chunks") if isinstance(data, dict) else None
    if not isinstance(chunks, list) or not chunks:
        return None
    material = {"embeddingModel": data.get("embeddingModel"), "chunks": chunks}
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def index_fingerprint(course_id: str) -> str | None:
    """A hash of everything retrieval reads from the course's index, or None.

    Chunk ids, order, text, headings and embeddings, plus the embedding model
    the index names. Not the file's bytes: re-saving an identical index does
    not retire answers, re-indexing with different content does.
    """
    try:
        path = Path(get_course_artifact_storage().index_path(course_id))
        stat = path.stat()
    except (OSError, ValueError, TypeError):
        return None
    key = (stat.st_mtime_ns, stat.st_size)
    memo_key = str(path)
    with _index_lock:
        cached = _index_memo.get(memo_key)
        if cached is not None and cached[0] == key:
            return cached[1]
    value = _hash_index(path)
    with _index_lock:
        _index_memo[memo_key] = (key, value)
    return value


# --------------------------------------------------------------------------- #
# Runtime parts: asked of Ollama and the fine-tuned service, briefly remembered
# --------------------------------------------------------------------------- #

_runtime_memo: dict[str, tuple[float, Any]] = {}


def _remembered(key: str) -> tuple[bool, Any]:
    entry = _runtime_memo.get(key)
    if entry is not None and entry[0] > time.monotonic():
        return True, entry[1]
    return False, None


def _remember(key: str, value: Any) -> Any:
    _runtime_memo[key] = (time.monotonic() + IDENTITY_TTL_SECONDS, value)
    return value


def reset_identity_memo_for_tests() -> None:
    _runtime_memo.clear()
    with _index_lock:
        _index_memo.clear()
    with _static_lock:
        _static.clear()


async def _get_json(url: str) -> Any:
    async with httpx.AsyncClient(timeout=IDENTITY_HTTP_TIMEOUT_SECONDS) as client:
        response = await client.get(url)
    if response.status_code != 200:
        raise ValueError(f"HTTP {response.status_code}")
    return response.json()


async def ollama_server_identity(base_url: str, *, fresh: bool = False) -> dict[str, Any] | None:
    """`{"version": …, "digests": {tag: digest}}` for one Ollama server, or None."""
    url = base_url.rstrip("/")
    key = f"ollama:{url}"
    if not fresh:
        hit, value = _remembered(key)
        if hit:
            return value
    try:
        version = await _get_json(f"{url}/api/version")
        tags = await _get_json(f"{url}/api/tags")
        models = tags.get("models") if isinstance(tags, dict) else None
        digests = {
            ollama_tag(str(model["name"])): str(model["digest"])
            for model in (models if isinstance(models, list) else [])
            if isinstance(model, dict) and model.get("name") and model.get("digest")
        }
        number = version.get("version") if isinstance(version, dict) else None
        if not isinstance(number, str) or not number.strip():
            raise ValueError("no version")
        value: dict[str, Any] | None = {"version": number.strip(), "digests": digests}
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        logger.warning("answer identity: Ollama at %s could not be read (%s)", url, exc.__class__.__name__)
        value = None
    return _remember(key, value)


async def finetuned_service_identity(*, fresh: bool = False) -> dict[str, Any] | None:
    """`{"ollamaUrl": …, "models": {"course@version": tag}}` from the service's `/health`."""
    base_url = get_finetuned_service_url()
    if base_url is None:
        return None
    key = f"finetuned:{base_url}"
    if not fresh:
        hit, value = _remembered(key)
        if hit:
            return value
    try:
        health = await _get_json(f"{base_url}/health")
        if not isinstance(health, dict) or health.get("status") != "ok":
            raise ValueError("not ok")
        ollama_url = health.get("ollamaUrl")
        models = health.get("models")
        if not isinstance(ollama_url, str) or not ollama_url.strip() or not isinstance(models, list):
            # An older or a Tillicum build: nothing to verify a digest against.
            raise ValueError("no ollamaUrl or models")
        mapping = {
            f"{model['courseId']}@{model['version']}": ollama_tag(str(model["ollamaModel"]))
            for model in models
            if isinstance(model, dict)
            and model.get("courseId")
            and model.get("version")
            and model.get("ollamaModel")
            and model.get("available") is not False
        }
        value: dict[str, Any] | None = {"ollamaUrl": ollama_url.strip().rstrip("/"), "models": mapping}
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        logger.warning("answer identity: the fine-tuned service could not be read (%s)", exc.__class__.__name__)
        value = None
    return _remember(key, value)


async def current_course_model_version(course_id: str) -> str | None:
    """The course's activated version, or None when it has none (or the registry cannot say)."""
    try:
        resolved = await run_in_threadpool(resolve_current_course_model, course_id)
    except HTTPException:
        return None
    version = resolved.get("version") if isinstance(resolved, dict) else None
    return version if isinstance(version, str) and version else None


# --------------------------------------------------------------------------- #
# Putting an identity together
# --------------------------------------------------------------------------- #


async def _retrieval_fields(course_id: str, top_k: int, *, fresh: bool) -> tuple[dict[str, Any] | None, str | None]:
    index = await run_in_threadpool(index_fingerprint, course_id)
    if index is None:
        return None, REASON_NO_INDEX
    server = await ollama_server_identity(OLLAMA_BASE_URL, fresh=fresh)
    if server is None:
        return None, REASON_EMBEDDING_MODEL
    embedding_tag = ollama_tag(OLLAMA_EMBEDDING_MODEL)
    embedding_digest = server["digests"].get(embedding_tag)
    if embedding_digest is None:
        return None, REASON_EMBEDDING_MODEL
    return (
        {
            "topK": int(top_k),
            "index": index,
            "embedding": {"tag": embedding_tag, "digest": embedding_digest, "ollama": server["version"]},
            "prompt": grounded_prompt_fingerprint(),
            "code": retrieval_code_fingerprint(),
        },
        None,
    )


async def build_identity(
    condition: str, course_id: str, *, top_k: int, fresh: bool = False
) -> IdentityResult:
    """The identity of `condition` for `course_id` right now, or why there is none."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}.")

    fields: dict[str, Any] = {
        "v": IDENTITY_VERSION,
        "condition": condition,
        "profile": CLASSROOM_PROFILE_NAME,
        "host": host_identity(),
    }
    model_version: str | None = None

    if condition in FINE_TUNED_CONDITIONS:
        model_version = await current_course_model_version(course_id)
        if model_version is None:
            return IdentityResult(None, REASON_NO_COURSE_MODEL)
        service = await finetuned_service_identity(fresh=fresh)
        if service is None:
            return IdentityResult(None, REASON_FINETUNED_SERVICE)
        tag = service["models"].get(f"{course_id}@{model_version}")
        if tag is None:
            return IdentityResult(None, REASON_FINETUNED_MODEL)
        server = await ollama_server_identity(service["ollamaUrl"], fresh=fresh)
        digest = server["digests"].get(tag) if server is not None else None
        if server is None or digest is None:
            return IdentityResult(None, REASON_FINETUNED_MODEL)
        fields["model"] = {"tag": tag, "digest": digest, "courseVersion": model_version}
        fields["ollama"] = server["version"]
        fields["maxNewTokens"] = CLASSROOM_NUM_PREDICT
        fields["service"] = finetuned_service_fingerprint()
    else:
        server = await ollama_server_identity(OLLAMA_BASE_URL, fresh=fresh)
        tag = ollama_tag(OLLAMA_MODEL)
        digest = server["digests"].get(tag) if server is not None else None
        if server is None or digest is None:
            return IdentityResult(None, REASON_BASE_MODEL)
        fields["model"] = {"tag": tag, "digest": digest}
        fields["ollama"] = server["version"]
        fields["options"] = dict(classroom_options())

    if condition == BASE:
        fields["prompt"] = base_prompt_fingerprint()
    elif condition == FINE_TUNED:
        fields["prompt"] = "bare-question"
    else:
        retrieval, reason = await _retrieval_fields(course_id, top_k, fresh=fresh)
        if retrieval is None:
            return IdentityResult(None, reason)
        fields["retrieval"] = retrieval

    return IdentityResult(
        Identity(
            fields=fields,
            fingerprint=fingerprint_of(fields),
            model_version=model_version,
            model_tag=fields["model"]["tag"],
        )
    )
