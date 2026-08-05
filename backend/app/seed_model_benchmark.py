"""Offline seed-model benchmark helpers (qwen3:4b vs qwen3:8b).

Reuses fact inventory, allocation, generation, prevalidation, and validation
code from the starter pipeline. Never persists seeds to Firebase.
Does not change production SEED_GENERATION_MODEL or API behavior.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from app.fact_inventory_cache import load_or_build_fact_inventory
from app.ollama import generate_starter_ollama_completion
from app.seed_allocation import allocate_slots
from app.seed_generation import (
    _generate_candidates_for_fact,
    _validate_candidate,
    passes_programmatic_candidate_checks,
    validation_result_accepts,
)
from app.seed_prevalidation import prevalidate_candidate
from app.seed_validation import calibrate_validation_result
from app.storage import CourseArtifactStorage, get_course_artifact_storage

DEFAULT_BENCHMARK_COURSE_ID = "css-360-winter-2026-a7rp"
DEFAULT_BENCHMARK_FACT_COUNT = 10
DEFAULT_MODELS = ("qwen3:4b", "qwen3:8b")

CompletionFn = Callable[..., Awaitable[dict[str, Any]]]


def select_benchmark_facts(
    facts: list[dict[str, Any]],
    *,
    count: int = DEFAULT_BENCHMARK_FACT_COUNT,
) -> list[dict[str, Any]]:
    """Select a fixed ordered list of facts via production allocate_slots.

    Prefers facts that received ``slotCount > 0``. If that yields fewer than
    ``count`` unique facts (caps / sparse inventory), backfills from remaining
    allocations with ``desiredSlots > 0`` in allocator order — the same pool
    the live pipeline uses for rejection backfill.

    Returns one entry per unique fact, up to ``count``. Each entry is
    ``{"fact": ..., "allocation": ...}``.
    """
    if count < 1:
        return []

    allocation = allocate_slots(facts, target_count=count)
    facts_by_id = {
        str(fact.get("factId") or "").strip(): fact
        for fact in facts
        if str(fact.get("factId") or "").strip()
    }

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _append(alloc: dict[str, Any]) -> None:
        fact_id = str(alloc.get("factId") or "").strip()
        if not fact_id or fact_id in seen:
            return
        fact = facts_by_id.get(fact_id)
        if fact is None:
            return
        seen.add(fact_id)
        selected.append({"fact": fact, "allocation": alloc})

    for alloc in allocation.get("allocations") or []:
        if len(selected) >= count:
            break
        if int(alloc.get("slotCount") or 0) <= 0:
            continue
        _append(alloc)

    if len(selected) < count:
        for alloc in allocation.get("allocations") or []:
            if len(selected) >= count:
                break
            if int(alloc.get("desiredSlots") or 0) <= 0:
                continue
            _append(alloc)

    return selected


def build_chunk_lookup(raw_chunks: list[Any]) -> dict[str, dict[str, Any]]:
    """Map chunkId -> chunk dict for evidence text lookup."""
    lookup: dict[str, dict[str, Any]] = {}
    for chunk in raw_chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunkId") or chunk.get("id") or "").strip()
        if chunk_id and isinstance(chunk.get("text"), str):
            lookup[chunk_id] = chunk
    return lookup


def chunk_texts_for_fact(
    fact: dict[str, Any],
    chunk_lookup: dict[str, dict[str, Any]],
) -> list[str]:
    texts: list[str] = []
    for chunk_id in fact.get("sourceChunkIds") or []:
        raw = chunk_lookup.get(str(chunk_id).strip())
        text = raw.get("text") if isinstance(raw, dict) else None
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return texts


def force_model_completion(model: str, base_fn: CompletionFn | None = None) -> CompletionFn:
    """Wrap an Ollama completion so every call uses ``model``."""

    run = base_fn or generate_starter_ollama_completion

    async def _forced(prompt: str, **kwargs: Any) -> dict[str, Any]:
        kwargs["model"] = model
        return await run(prompt, **kwargs)

    return _forced


def summarize_benchmark_candidates(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate accepted/rejected metrics for one model run."""
    accepted = [row for row in candidates if row.get("status") == "accepted"]
    rejected = [row for row in candidates if row.get("status") != "accepted"]

    scores = [
        float(row["validationScore"])
        for row in candidates
        if row.get("validationScore") is not None
    ]
    unsupported_count = sum(
        len(row.get("unsupportedClaims") or []) for row in candidates
    )
    qualifier_mismatch_count = sum(
        1 for row in candidates if row.get("qualifierMismatch")
    )
    prevalidation_rejection_count = sum(
        1 for row in rejected if row.get("rejectionStage") == "prevalidation"
    )
    accepted_times = [
        float(row["elapsedSeconds"])
        for row in accepted
        if row.get("elapsedSeconds") is not None
    ]

    return {
        "candidateCount": len(candidates),
        "acceptedCount": len(accepted),
        "rejectedCount": len(rejected),
        "averageValidationScore": (
            round(sum(scores) / len(scores), 4) if scores else None
        ),
        "unsupportedClaimCount": unsupported_count,
        "qualifierMismatchCount": qualifier_mismatch_count,
        "prevalidationRejectionCount": prevalidation_rejection_count,
        "averageTimePerAcceptedSeedSeconds": (
            round(sum(accepted_times) / len(accepted_times), 4)
            if accepted_times
            else None
        ),
    }


def compare_benchmark_summaries(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    """Side-by-side summary for two model runs."""
    keys = (
        "acceptedCount",
        "averageValidationScore",
        "unsupportedClaimCount",
        "qualifierMismatchCount",
        "prevalidationRejectionCount",
        "averageTimePerAcceptedSeedSeconds",
    )
    comparison: dict[str, Any] = {
        "leftModel": left.get("model"),
        "rightModel": right.get("model"),
        "metrics": {},
    }
    for key in keys:
        left_value = (left.get("summary") or {}).get(key)
        right_value = (right.get("summary") or {}).get(key)
        comparison["metrics"][key] = {
            "left": left_value,
            "right": right_value,
            "delta": (
                None
                if left_value is None or right_value is None
                else round(float(right_value) - float(left_value), 4)
            ),
        }
    return comparison


def _empty_candidate_row(
    *,
    model: str,
    fact: dict[str, Any],
    generation_time_seconds: float,
    status: str,
    rejection_reason: str | None,
    rejection_stage: str | None = None,
    qualifier_mismatch: bool = False,
    question: str = "",
    answer: str = "",
    unsupported_claims: list[str] | None = None,
    validation_score: float | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    generation = round(float(generation_time_seconds), 4)
    elapsed = (
        round(float(elapsed_seconds), 4)
        if elapsed_seconds is not None
        else generation
    )
    return {
        "model": model,
        "factId": str(fact.get("factId") or ""),
        "question": question,
        "answer": answer,
        "evidenceQuote": str(fact.get("evidenceQuote") or ""),
        "validationScore": validation_score,
        "rejectionReason": rejection_reason,
        "rejectionStage": rejection_stage,
        "unsupportedClaims": list(unsupported_claims or []),
        "qualifierMismatch": bool(qualifier_mismatch),
        "generationTimeSeconds": generation,
        "elapsedSeconds": elapsed,
        "status": status,
    }


async def evaluate_fact_candidate(
    *,
    model: str,
    fact: dict[str, Any],
    allocation: dict[str, Any],
    chunk_texts: list[str],
    completion_fn: CompletionFn,
) -> dict[str, Any]:
    """Generate + prevalidate + validate one fact-scoped candidate."""
    overall_started = time.perf_counter()
    if not chunk_texts:
        return _empty_candidate_row(
            model=model,
            fact=fact,
            generation_time_seconds=0.0,
            status="rejected",
            rejection_reason="missing_source_chunks",
            rejection_stage="setup",
            elapsed_seconds=0.0,
        )

    started = time.perf_counter()
    try:
        candidates, _ = await _generate_candidates_for_fact(
            fact=fact,
            chunk_texts=chunk_texts,
            count=1,
            suggested_styles=list(allocation.get("suggestedStyles") or []),
            completion_fn=completion_fn,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark must keep going
        elapsed = time.perf_counter() - started
        return _empty_candidate_row(
            model=model,
            fact=fact,
            generation_time_seconds=elapsed,
            status="rejected",
            rejection_reason=f"generation_error:{type(exc).__name__}:{exc}",
            rejection_stage="generation",
            elapsed_seconds=time.perf_counter() - overall_started,
        )
    generation_time = time.perf_counter() - started

    if not candidates:
        return _empty_candidate_row(
            model=model,
            fact=fact,
            generation_time_seconds=generation_time,
            status="rejected",
            rejection_reason="no_candidates",
            rejection_stage="generation",
            elapsed_seconds=time.perf_counter() - overall_started,
        )

    candidate = candidates[0]
    question = str(candidate.get("question") or "")
    answer = str(candidate.get("answer") or "")

    if not passes_programmatic_candidate_checks(candidate):
        return _empty_candidate_row(
            model=model,
            fact=fact,
            generation_time_seconds=generation_time,
            status="rejected",
            rejection_reason="programmatic_checks_failed",
            rejection_stage="programmatic",
            question=question,
            answer=answer,
            elapsed_seconds=time.perf_counter() - overall_started,
        )

    prevalidation = prevalidate_candidate(
        candidate=candidate,
        fact=fact,
        source_text="\n\n".join(chunk_texts),
    )
    if prevalidation is not None:
        category = str(prevalidation.get("category") or "prevalidation")
        reason = str(prevalidation.get("reason") or "prevalidation_rejected")
        return _empty_candidate_row(
            model=model,
            fact=fact,
            generation_time_seconds=generation_time,
            status="rejected",
            rejection_reason=reason,
            rejection_stage="prevalidation",
            qualifier_mismatch=category == "qualifier_mismatch",
            question=question,
            answer=answer,
            elapsed_seconds=time.perf_counter() - overall_started,
        )

    try:
        validation = await _validate_candidate(
            question=question,
            answer=answer,
            topic_name=str(candidate.get("category") or "General"),
            question_type=str(candidate.get("questionType") or "direct"),
            chunk_text="\n\n".join(chunk_texts),
            completion_fn=completion_fn,
        )
    except Exception as exc:  # noqa: BLE001
        return _empty_candidate_row(
            model=model,
            fact=fact,
            generation_time_seconds=generation_time,
            status="rejected",
            rejection_reason=f"validation_error:{type(exc).__name__}:{exc}",
            rejection_stage="validation",
            question=question,
            answer=answer,
            elapsed_seconds=time.perf_counter() - overall_started,
        )

    if validation is None:
        return _empty_candidate_row(
            model=model,
            fact=fact,
            generation_time_seconds=generation_time,
            status="rejected",
            rejection_reason="validation_parse_failed",
            rejection_stage="validation",
            question=question,
            answer=answer,
            elapsed_seconds=time.perf_counter() - overall_started,
        )

    validation = calibrate_validation_result(
        result=validation,
        question=question,
        answer=answer,
        topic_name=str(candidate.get("category") or "General"),
        question_type=str(candidate.get("questionType") or "direct"),
    )
    unsupported = list(validation.get("unsupportedClaims") or [])
    score = float(validation.get("score"))
    if not validation_result_accepts(validation):
        return _empty_candidate_row(
            model=model,
            fact=fact,
            generation_time_seconds=generation_time,
            status="rejected",
            rejection_reason=str(validation.get("reason") or "validation_rejected"),
            rejection_stage="validation",
            unsupported_claims=unsupported,
            validation_score=score,
            question=question,
            answer=answer,
            elapsed_seconds=time.perf_counter() - overall_started,
        )

    return _empty_candidate_row(
        model=model,
        fact=fact,
        generation_time_seconds=generation_time,
        status="accepted",
        rejection_reason=None,
        unsupported_claims=unsupported,
        validation_score=score,
        question=question,
        answer=answer,
        elapsed_seconds=time.perf_counter() - overall_started,
    )


async def run_model_benchmark(
    *,
    model: str,
    selected: list[dict[str, Any]],
    chunk_lookup: dict[str, dict[str, Any]],
    completion_fn: CompletionFn | None = None,
) -> dict[str, Any]:
    """Run generation+validation for each selected fact with one model."""
    forced = force_model_completion(model, completion_fn)
    rows: list[dict[str, Any]] = []
    for item in selected:
        fact = item["fact"]
        allocation = item["allocation"]
        texts = chunk_texts_for_fact(fact, chunk_lookup)
        row = await evaluate_fact_candidate(
            model=model,
            fact=fact,
            allocation=allocation,
            chunk_texts=texts,
            completion_fn=forced,
        )
        rows.append(row)

    payload = {
        "model": model,
        "factIds": [
            str(item["fact"].get("factId") or "") for item in selected
        ],
        "candidates": rows,
        "summary": summarize_benchmark_candidates(rows),
    }
    return payload


async def load_benchmark_context(
    *,
    course_id: str = DEFAULT_BENCHMARK_COURSE_ID,
    fact_count: int = DEFAULT_BENCHMARK_FACT_COUNT,
    storage: CourseArtifactStorage | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Load course index + cached inventory and select fixed benchmark facts."""
    artifact_storage = storage or get_course_artifact_storage()
    index_data = artifact_storage.load_index(course_id)
    if not index_data:
        raise FileNotFoundError(f'No index found for course "{course_id}".')

    raw_chunks = index_data.get("chunks") or []
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise ValueError(f'No syllabus chunks for course "{course_id}".')

    inventory = await load_or_build_fact_inventory(
        course_id=course_id,
        raw_chunks=raw_chunks,
        storage=artifact_storage,
        force_refresh=force_refresh,
    )
    facts = list(inventory.get("facts") or [])
    selected = select_benchmark_facts(facts, count=fact_count)
    if len(selected) < fact_count:
        raise ValueError(
            f"Only selected {len(selected)} facts; need {fact_count}."
        )

    return {
        "courseId": course_id,
        "inventory": inventory,
        "selected": selected,
        "chunkLookup": build_chunk_lookup(raw_chunks),
        "factIds": [str(item["fact"].get("factId") or "") for item in selected],
    }


def write_benchmark_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write one model benchmark payload to disk."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def format_comparison_report(
    *,
    course_id: str,
    left: dict[str, Any],
    right: dict[str, Any],
) -> str:
    """Human-readable comparison for stdout."""
    comparison = compare_benchmark_summaries(left, right)
    lines = [
        f"Course: {course_id}",
        f"Facts: {', '.join(left.get('factIds') or [])}",
        f"Left model:  {left.get('model')}",
        f"Right model: {right.get('model')}",
        "",
        f"{'metric':<40} {'left':>12} {'right':>12} {'delta':>12}",
        "-" * 80,
    ]
    for key, values in comparison["metrics"].items():
        left_v = values["left"]
        right_v = values["right"]
        delta = values["delta"]
        lines.append(
            f"{key:<40} {left_v!s:>12} {right_v!s:>12} {delta!s:>12}"
        )
    return "\n".join(lines)
