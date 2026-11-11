"""The classroom answer cache: when a stored answer may be served or written.

The four classroom routes and classroom preparation both ask for an answer
through `serve`. With `ANSWER_CACHE_ENABLED` unset (the default) `serve` calls
the approach's real pipeline and does nothing else, so the application behaves
exactly as it did before this module existed.

When enabled, for one course, approach and question:

1. In classroom GPU mode nothing is read or stored: those answers come from a
   different engine. (Identical concurrent requests still share one
   generation.)
2. The approach's identity is established (`app.answer_identity`). If it
   cannot be, nothing is read or stored.
3. An answer may be read or stored only when the question, conservatively
   normalised, is one staff selected for classroom preparation in this course
   (`classroom_questions`). An arbitrary question a student types is answered
   live and never stored.
4. A stored row for exactly this course, approach, question and fingerprint is
   returned as the response, marked `cached`.
5. Otherwise the real pipeline runs, shared with anyone asking the same thing
   at the same moment (`app.answer_flights`). A successful answer to a
   selected question is stored, after re-reading the identity fresh and
   checking that the answer came from the model the identity names.

Never stored: a failure, an empty answer, a GPU answer, an answer whose
identity moved during the generation, or an answer to an unselected question.
Any failure of the cache itself (PostgreSQL down, a row that no longer fits
the response model) is logged and treated as a miss; the cache is never the
reason a request fails.

Matching is deliberately narrow: Unicode-normalised, case-folded, whitespace
collapsed, typographic quotes straightened, and trailing `?`, `!`, `.` and
spaces dropped. "Homework 2" and "Homework 3", or "C" and "C++", never match.
There is no semantic matching.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app import db_answer_cache, db_classroom_questions
from app.answer_flights import share
from app.answer_identity import (
    FINE_TUNED_CONDITIONS,
    Identity,
    build_identity,
    ollama_tag,
)
from app.db import db_connection
from app.generation_mode import current_mode
from app.generation_queue import (
    collect_generation_details,
    get_generation_queue,
    is_background_generation,
)
from app.retrieval_diversity import DEFAULT_TOP_K

logger = logging.getLogger(__name__)

ENABLED_ENV = "ANSWER_CACHE_ENABLED"
#: How long a course's selected-question set is trusted before it is re-read.
#: Saving a selection through the preparation page clears it at once.
SELECTION_TTL_SECONDS = 30.0

#: Response fields this module adds; never part of a stored payload.
CACHE_FIELDS = ("cached", "generatedAt", "cacheEntryId")

Pipeline = Callable[[str, str, int], Awaitable[BaseModel]]


def cache_enabled() -> bool:
    return (os.getenv(ENABLED_ENV) or "").strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# Question matching
# --------------------------------------------------------------------------- #

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def normalize_question(question: str) -> str:
    """The form two questions must share to be treated as the same question."""
    text = unicodedata.normalize("NFKC", str(question)).translate(_QUOTES)
    text = " ".join(text.split()).casefold()
    return text.rstrip("?!. ")


def question_key(question: str) -> str:
    return hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# The four pipelines, registered by app.main
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RegisteredPipeline:
    run: Pipeline
    response_model: type[BaseModel]


_pipelines: dict[str, RegisteredPipeline] = {}


def register_pipeline(condition: str, run: Pipeline, response_model: type[BaseModel]) -> None:
    """Name the real pipeline for an approach: `run(course_id, question, top_k)`."""
    if condition not in db_answer_cache.CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}.")
    _pipelines[condition] = RegisteredPipeline(run=run, response_model=response_model)


def registered_conditions() -> tuple[str, ...]:
    return tuple(c for c in db_answer_cache.CONDITIONS if c in _pipelines)


# --------------------------------------------------------------------------- #
# Which questions staff selected (briefly remembered per course)
# --------------------------------------------------------------------------- #

_selection_memo: dict[str, tuple[float, frozenset[str]]] = {}


def invalidate_selection(course_id: str | None = None) -> None:
    if course_id is None:
        _selection_memo.clear()
    else:
        _selection_memo.pop(course_id, None)


def _load_selected_keys(course_id: str) -> frozenset[str]:
    with db_connection() as connection:
        questions = db_classroom_questions.list_selected_questions(connection, course_id)
    return frozenset(question_key(item["question"]) for item in questions)


async def selected_question_keys(course_id: str) -> frozenset[str]:
    """Keys of the questions an answer may be stored or served for.

    Raises whatever the database raises; callers treat that as "none".
    """
    remembered = _selection_memo.get(course_id)
    if remembered is not None and remembered[0] > time.monotonic():
        return remembered[1]
    keys = await run_in_threadpool(_load_selected_keys, course_id)
    _selection_memo[course_id] = (time.monotonic() + SELECTION_TTL_SECONDS, keys)
    return keys


async def _is_selected(course_id: str, key: str) -> bool:
    try:
        return key in await selected_question_keys(course_id)
    except Exception:  # noqa: BLE001 - the cache must never fail a request
        logger.warning("answer cache: could not read the selected questions for %s", course_id, exc_info=True)
        return False


# --------------------------------------------------------------------------- #
# Reading and writing rows (never raising)
# --------------------------------------------------------------------------- #


def _read_entry(course_id: str, condition: str, key: str, fingerprint: str) -> dict[str, Any] | None:
    with db_connection() as connection:
        return db_answer_cache.get_entry(connection, course_id, condition, key, fingerprint)


async def _lookup(course_id: str, condition: str, key: str, fingerprint: str) -> dict[str, Any] | None:
    try:
        return await run_in_threadpool(_read_entry, course_id, condition, key, fingerprint)
    except Exception:  # noqa: BLE001 - the cache must never fail a request
        logger.warning("answer cache: read failed for %s/%s; generating live", course_id, condition, exc_info=True)
        return None


def _response_from_entry(pipeline: RegisteredPipeline, entry: dict[str, Any]) -> BaseModel | None:
    try:
        return pipeline.response_model.model_validate(
            {
                **entry["payload"],
                "cached": True,
                "generatedAt": entry["generatedAt"],
                "cacheEntryId": entry["entryId"],
            }
        )
    except Exception:  # noqa: BLE001 - a row that no longer fits is a miss
        logger.warning("answer cache: stored entry %s no longer fits its response model", entry.get("entryId"), exc_info=True)
        return None


def _insert_entry(**fields: Any) -> dict[str, Any] | None:
    with db_connection() as connection:
        return db_answer_cache.insert_entry(connection, **fields)


def _storable_payload(condition: str, response: BaseModel, identity: Identity) -> dict[str, Any] | None:
    """The response as a payload, or None when it must not be stored."""
    payload = response.model_dump(by_alias=True, mode="json")
    for key in CACHE_FIELDS:
        payload.pop(key, None)
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return None
    model = payload.get("model")
    if payload.get("engine") == "gpu" or "Tillicum GPU" in str(model or ""):
        return None
    if condition in FINE_TUNED_CONDITIONS:
        # The service says which course version and Ollama model answered;
        # both must be the ones the fingerprint was computed for.
        if payload.get("modelVersion") != identity.model_version:
            return None
        if not isinstance(model, str) or ollama_tag(model) != identity.model_tag:
            return None
    return payload


async def _store(
    *,
    condition: str,
    course_id: str,
    question: str,
    key: str,
    top_k: int,
    identity: Identity,
    response: BaseModel,
    details: dict[str, Any],
    source: str,
    elapsed_seconds: float,
) -> dict[str, Any] | None:
    """Store a fresh answer if every condition for storing it still holds."""
    try:
        payload = _storable_payload(condition, response, identity)
        if payload is None or current_mode().is_gpu:
            return None
        # Everything that decides the answer, read again now: a model rebuilt
        # or a syllabus replaced while this was generating is not stored.
        after = (await build_identity(condition, course_id, top_k=top_k, fresh=True)).identity
        if after is None or after.fingerprint != identity.fingerprint:
            logger.info("answer cache: identity changed during generation for %s/%s; not stored", course_id, condition)
            return None
        if not await _is_selected(course_id, key):
            return None
        provenance = {
            **{k: v for k, v in details.items() if k != "outcome"},
            "elapsedSeconds": round(elapsed_seconds, 3),
        }
        return await run_in_threadpool(
            lambda: _insert_entry(
                course_id=course_id,
                condition=condition,
                question_key=key,
                fingerprint=identity.fingerprint,
                question_text=" ".join(question.split()),
                payload=payload,
                identity=identity.fields,
                provenance=provenance,
                source=source,
                generated_at=datetime.now(timezone.utc),
            )
        )
    except Exception:  # noqa: BLE001 - the answer is still returned
        logger.warning("answer cache: write failed for %s/%s", course_id, condition, exc_info=True)
        return None


# --------------------------------------------------------------------------- #
# The one entry point
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Served:
    """An answer and where it came from (for classroom preparation's counts)."""

    response: BaseModel
    #: `hit` (already stored), `stored` (generated and stored now),
    #: `generated` (generated, not stored).
    outcome: str


async def serve_detailed(
    condition: str,
    course_id: str,
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    source: str = "live",
) -> Served:
    pipeline = _pipelines[condition]
    if not cache_enabled():
        return Served(await pipeline.run(course_id, question, top_k), "generated")

    background = is_background_generation()
    if not background:
        # A stored answer never reaches the generation queue; preparation still
        # needs to know students are here.
        get_generation_queue().note_interactive_activity()

    key = question_key(question)
    on_gpu = current_mode().is_gpu
    identity: Identity | None = None
    if not on_gpu:
        try:
            identity = (await build_identity(condition, course_id, top_k=top_k)).identity
        except Exception:  # noqa: BLE001 - the cache must never fail a request
            logger.warning("answer cache: identity lookup failed for %s/%s", course_id, condition, exc_info=True)
            identity = None

    storable = identity is not None and await _is_selected(course_id, key)
    if storable and identity is not None:
        entry = await _lookup(course_id, condition, key, identity.fingerprint)
        if entry is not None:
            response = _response_from_entry(pipeline, entry)
            if response is not None:
                return Served(response, "hit")

    async def work() -> Served:
        started = time.perf_counter()
        with collect_generation_details() as details:
            response = await pipeline.run(course_id, question, top_k)
        if storable and identity is not None:
            stored = await _store(
                condition=condition,
                course_id=course_id,
                question=question,
                key=key,
                top_k=top_k,
                identity=identity,
                response=response,
                details=details,
                source=source,
                elapsed_seconds=time.perf_counter() - started,
            )
            if stored is not None:
                # Generated just now, not served from the cache; the entry id
                # lets a rating point at the stored row all the same.
                response = response.model_copy(
                    update={"generated_at": stored["generatedAt"], "cache_entry_id": stored["entryId"]}
                )
                return Served(response, "stored")
        return Served(response, "generated")

    flight_key = (
        course_id,
        condition,
        key,
        int(top_k),
        "gpu" if on_gpu else "vm",
        identity.fingerprint if identity is not None else None,
    )
    return await share(flight_key, work)


async def serve(
    condition: str,
    course_id: str,
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> BaseModel:
    """The classroom answer for one approach: stored when allowed, else live."""
    return (await serve_detailed(condition, course_id, question, top_k=top_k)).response
