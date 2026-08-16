"""Deterministic slot allocation over a fact inventory (Phase 3).

Ranks facts and assigns question-generation slots without generating seeds
and without touching the live starter-generation pipeline.

Given the same fact inventory and ``targetCount``, output is identical.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

DEFAULT_TARGET_COUNT = 50
MAX_SLOTS_PER_FACT = 3

# Soft dominance caps as fractions of targetCount (course-agnostic).
SOURCE_CHUNK_CAP_RATIO = 0.12
SERIES_CAP_RATIO = 0.18
KIND_CAP_RATIO = 0.22
DEADLINE_KIND_CAP_RATIO = 0.25
SCOPE_CAP_RATIOS: dict[str, float] = {
    "course_wide": 0.70,
    "assignment_specific": 0.40,
    "schedule": 0.20,
    "resource": 0.12,
    "other": 0.18,
}

# Soft reservation: prefer course-wide coverage until this share is filled
# (when enough course-wide candidates exist). Not a rigid required category.
COURSE_WIDE_RESERVE_RATIO = 0.30
COURSE_WIDE_RESERVE_BOOST = 1.0

# Ranking weights (must sum conceptually with priors; not a hard taxonomy).
WEIGHT_USEFULNESS = 0.35
WEIGHT_IMPORTANCE = 0.25
WEIGHT_ASK = 0.20
WEIGHT_COMPLEXITY = 0.10

SCOPE_PRIOR: dict[str, float] = {
    "course_wide": 0.12,
    "assignment_specific": 0.0,
    "schedule": -0.05,
    "resource": -0.10,
    "other": -0.02,
}

# Soft usefulness priors for frequently asked student needs.
KIND_PRIOR: dict[str, float] = {
    "late_work": 0.10,
    "attendance": 0.08,
    "grading": 0.08,
    "accommodation": 0.08,
    "contact": 0.06,
    "communication": 0.06,
    "requirement": 0.05,
    "policy": 0.05,
    "office_hours": 0.04,
    "exam": 0.04,
    "submission": 0.05,
    "team_project": 0.02,
    "tools": 0.0,
    "deadline": -0.02,
    "resource": -0.08,
    "other": 0.0,
}

# Below this ranking score, a fact is not eligible for any slot unless the
# inventory is so sparse that fill-pass needs leftovers (it still won't
# receive slots while better facts remain).
MIN_ELIGIBLE_RANKING_SCORE = 0.38

# Diminishing returns when competing for 2nd/3rd slots on the same fact.
MULTI_SLOT_SCORE_FACTORS = (1.0, 0.85, 0.70)

_SIMPLE_LOOKUP_KINDS = frozenset({"contact", "office_hours"})
_POLICY_MULTI_KINDS = frozenset(
    {"late_work", "attendance", "policy", "accommodation", "grading", "requirement"}
)
_RESOURCE_KINDS = frozenset({"resource"})

# Styles that become a scenario/clarification questionType downstream. Kept
# here rather than derived from the generator's map so allocation stays
# independent of it.
SCENARIO_LIKE_STYLES = frozenset({"scenario", "exception", "clarification"})


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _ratio_cap(target_count: int, ratio: float, *, minimum: int = 1) -> int:
    if target_count <= 0:
        return 0
    return max(minimum, int(math.floor(target_count * ratio)))


def compute_allocation_caps(target_count: int) -> dict[str, Any]:
    """Return soft caps derived from ``targetCount`` (deterministic)."""
    return {
        "targetCount": target_count,
        # Floor at MAX_SLOTS_PER_FACT so one complex fact from a chunk can
        # still receive justified multi-slots without being truncated by the
        # ratio alone on small targets.
        "perSourceChunk": max(
            MAX_SLOTS_PER_FACT,
            _ratio_cap(target_count, SOURCE_CHUNK_CAP_RATIO),
        ),
        "perSeries": _ratio_cap(target_count, SERIES_CAP_RATIO),
        "perKind": _ratio_cap(target_count, KIND_CAP_RATIO),
        "deadlineKind": _ratio_cap(target_count, DEADLINE_KIND_CAP_RATIO),
        "perScope": {
            scope: _ratio_cap(target_count, ratio)
            for scope, ratio in SCOPE_CAP_RATIOS.items()
        },
        "courseWideReserve": _ratio_cap(target_count, COURSE_WIDE_RESERVE_RATIO),
        "maxSlotsPerFact": MAX_SLOTS_PER_FACT,
    }


def compute_ranking_score(fact: dict[str, Any]) -> float:
    """Composite ranking score used for allocation priority."""
    usefulness = float(fact.get("usefulnessScore", 0.0))
    importance = float(fact.get("importanceScore", 0.5))
    ask = float(fact.get("studentAskLikelihood", 0.5))
    complexity = max(1, int(fact.get("complexity", 1)))
    complexity_norm = min(1.0, (complexity - 1) / 4.0)

    scope = str(fact.get("scope") or "other")
    kind = str(fact.get("kind") or "other")

    score = (
        WEIGHT_USEFULNESS * usefulness
        + WEIGHT_IMPORTANCE * importance
        + WEIGHT_ASK * ask
        + WEIGHT_COMPLEXITY * complexity_norm
        + SCOPE_PRIOR.get(scope, SCOPE_PRIOR["other"])
        + KIND_PRIOR.get(kind, KIND_PRIOR["other"])
    )
    return round(_clamp01(score), 4)


def suggest_question_styles(
    fact: dict[str, Any],
    slot_count: int,
    *,
    prefer_scenario_like: bool = False,
) -> list[str]:
    """Suggest question styles justified by complexity / kind (not seeds).

    `prefer_scenario_like` reorders an already-earned scenario-like style to the
    front so it survives truncation to `slot_count`. It never adds one: a fact
    that did not qualify for `scenario`/`exception`/`clarification` below is not
    given one here, because a style the evidence cannot support only produces a
    candidate that validation will reject.

    The reordering exists because scenario-like styles are never the first entry
    in any branch. At one slot per fact — which is what breadth-first allocation
    produces whenever facts are plentiful — truncation dropped every one of them,
    making the run's scenario minimum unreachable no matter what the model did.
    """
    if slot_count <= 0:
        return []

    kind = str(fact.get("kind") or "other")
    complexity = max(1, int(fact.get("complexity", 1)))
    styles: list[str] = []

    if kind in _SIMPLE_LOOKUP_KINDS:
        styles.append("factual")
    elif kind == "deadline":
        styles.append("factual")
    elif kind in ("late_work", "attendance", "policy", "accommodation"):
        styles.append("policy")
        if complexity >= 2 or slot_count >= 2:
            styles.append("scenario")
        if complexity >= 3 or slot_count >= 3:
            styles.append("exception")
    elif kind in ("grading", "requirement", "submission"):
        styles.append("factual")
        if slot_count >= 2:
            styles.append("clarification")
    elif kind == "communication":
        styles.append("procedural")
    else:
        styles.append("factual")
        if slot_count >= 2:
            styles.append("clarification")

    # Stable unique order.
    seen: set[str] = set()
    ordered: list[str] = []
    for style in styles:
        if style not in seen:
            seen.add(style)
            ordered.append(style)

    if prefer_scenario_like:
        promoted = next(
            (style for style in ordered if style in SCENARIO_LIKE_STYLES), None
        )
        if promoted is not None:
            ordered = [promoted] + [
                style for style in ordered if style != promoted
            ]

    return ordered[: max(1, slot_count)]


def fact_supports_scenario_like_style(fact: dict[str, Any], slot_count: int) -> bool:
    """Whether this fact already earns a scenario-like style at this slot count.

    The eligibility test the generator uses before promoting one, so preference
    never becomes coercion.
    """
    return any(
        style in SCENARIO_LIKE_STYLES
        for style in suggest_question_styles(
            fact, slot_count, prefer_scenario_like=True
        )
    )


def compute_desired_slots(fact: dict[str, Any]) -> dict[str, Any]:
    """Decide how many slots a fact *wants* before caps, with reasons.

    Intentionally NOT a simplistic high=2 / medium=1 / low=0 mapping.
    """
    ranking_score = compute_ranking_score(fact)
    importance = str(fact.get("importance") or "medium")
    usefulness = float(fact.get("usefulnessScore", 0.0))
    ask = float(fact.get("studentAskLikelihood", 0.5))
    complexity = max(1, int(fact.get("complexity", 1)))
    kind = str(fact.get("kind") or "other")
    scope = str(fact.get("scope") or "other")

    reasons: list[str] = []
    desired = 0

    # Minor optional resources: usually 0 when competing with better facts.
    if scope == "resource" or kind in _RESOURCE_KINDS:
        if usefulness < 0.50 or importance == "low" or ask < 0.40:
            return {
                "desiredSlots": 0,
                "rankingScore": ranking_score,
                "reasons": ["minor_resource_deprioritized"],
                "suggestedStyles": [],
            }

    if ranking_score < MIN_ELIGIBLE_RANKING_SCORE:
        return {
            "desiredSlots": 0,
            "rankingScore": ranking_score,
            "reasons": ["below_minimum_ranking_score"],
            "suggestedStyles": [],
        }

    desired = 1
    reasons.append("eligible_base_slot")

    # High usefulness + low complexity contact/office-hours → exactly 1.
    if kind in _SIMPLE_LOOKUP_KINDS and complexity <= 1:
        reasons.append("simple_contact_or_lookup_stays_at_one")
        return {
            "desiredSlots": 1,
            "rankingScore": ranking_score,
            "reasons": reasons,
            "suggestedStyles": suggest_question_styles(fact, 1),
        }

    # Simple deadline facts stay at one slot (series caps also limit them).
    if kind == "deadline" and complexity <= 1:
        reasons.append("simple_deadline_single_slot")
        return {
            "desiredSlots": 1,
            "rankingScore": ranking_score,
            "reasons": reasons,
            "suggestedStyles": suggest_question_styles(fact, 1),
        }

    # Complexity-aware multi-slot: only when conditions/exceptions justify it.
    if complexity >= 3 and usefulness >= 0.55:
        desired = min(MAX_SLOTS_PER_FACT, 1 + (complexity // 2))
        reasons.append("high_complexity_multiple_conditions")
    elif complexity >= 2 and (ask >= 0.65 or usefulness >= 0.65):
        desired = 2
        reasons.append("moderate_complexity_with_high_ask")

    if kind in _POLICY_MULTI_KINDS and complexity >= 2:
        multi = min(MAX_SLOTS_PER_FACT, max(2, complexity))
        if multi > desired:
            desired = multi
            reasons.append("policy_with_conditions_or_exceptions")

    # Resources that cleared the minor filter still max out at 1.
    if scope == "resource" or kind in _RESOURCE_KINDS:
        if desired > 1:
            desired = 1
            reasons.append("resource_capped_at_one")

    styles = suggest_question_styles(fact, desired)
    return {
        "desiredSlots": desired,
        "rankingScore": ranking_score,
        "reasons": reasons,
        "suggestedStyles": styles,
    }


def _series_key(fact: dict[str, Any]) -> str | None:
    key = fact.get("seriesKey") or fact.get("assignmentGroup")
    if key is None:
        return None
    text = str(key).strip()
    return text or None


def _primary_source_chunk(fact: dict[str, Any]) -> str | None:
    ids = fact.get("sourceChunkIds") or []
    if not isinstance(ids, list) or not ids:
        return None
    return str(ids[0])


def _cap_blockers(
    *,
    fact: dict[str, Any],
    caps: dict[str, Any],
    scope_counts: Counter[str],
    kind_counts: Counter[str],
    series_counts: Counter[str],
    chunk_counts: Counter[str],
    is_additional_slot: bool,
    enforce_scope_caps: bool = True,
) -> list[str]:
    """Return human-readable cap reasons that currently block another slot.

    Kind / scope / series caps apply when admitting a fact's *first* slot so
    one category cannot flood the dataset with many distinct facts. Additional
    slots on an already-admitted fact are limited by desiredSlots,
    MAX_SLOTS_PER_FACT, targetCount, and source-chunk volume (so one chunk
    cannot dominate via many multi-slot facts).

    During the breadth-first pass, scope caps may be relaxed (``enforce_scope_caps``
    False) so small targets can still cover many distinct high-value facts when
    they share course_wide scope.
    """
    blockers: list[str] = []
    chunk = _primary_source_chunk(fact)

    # Source-chunk volume counts every slot.
    if chunk is not None and chunk_counts[chunk] >= caps["perSourceChunk"]:
        blockers.append(f"source_chunk_cap:{chunk}")

    if is_additional_slot:
        return blockers

    scope = str(fact.get("scope") or "other")
    kind = str(fact.get("kind") or "other")
    series = _series_key(fact)

    if enforce_scope_caps:
        scope_cap = caps["perScope"].get(scope, caps["perScope"]["other"])
        if scope_counts[scope] >= scope_cap:
            blockers.append(f"scope_cap:{scope}")

    kind_cap = caps["perKind"]
    if kind == "deadline":
        kind_cap = min(kind_cap, caps["deadlineKind"])
    if kind_counts[kind] >= kind_cap:
        blockers.append(f"kind_cap:{kind}")

    if series is not None and series_counts[series] >= caps["perSeries"]:
        blockers.append(f"series_cap:{series}")

    return blockers


def allocate_slots(
    facts: list[dict[str, Any]],
    *,
    target_count: int = DEFAULT_TARGET_COUNT,
) -> dict[str, Any]:
    """Allocate question slots over a fact inventory.

    Returns an inspectable dict with ``allocations``, ``summary``, and
    ``ranking`` debug rows. Does not generate seeds.
    """
    target = max(0, int(target_count))
    caps = compute_allocation_caps(target)

    # Prepare ranked candidates with desired-slot metadata.
    candidates: list[dict[str, Any]] = []
    for fact in facts:
        desired_meta = compute_desired_slots(fact)
        candidates.append(
            {
                "fact": fact,
                "factId": str(fact.get("factId") or ""),
                "desiredSlots": int(desired_meta["desiredSlots"]),
                "rankingScore": float(desired_meta["rankingScore"]),
                "baseReasons": list(desired_meta["reasons"]),
                "suggestedStyles": list(desired_meta["suggestedStyles"]),
                "slotCount": 0,
                "allocReasons": list(desired_meta["reasons"]),
                "capReasons": [],
            }
        )

    # Deterministic order: higher ranking first, then stable factId.
    candidates.sort(key=lambda item: (-item["rankingScore"], item["factId"]))

    scope_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    series_counts: Counter[str] = Counter()
    chunk_counts: Counter[str] = Counter()
    allocated_total = 0
    course_wide_allocated = 0
    reserve = caps["courseWideReserve"]

    def _effective_score(
        item: dict[str, Any],
        *,
        first_slot_only: bool,
    ) -> float | None:
        if item["slotCount"] >= item["desiredSlots"]:
            return None
        if item["slotCount"] >= MAX_SLOTS_PER_FACT:
            return None
        if item["desiredSlots"] <= 0:
            return None
        if first_slot_only and item["slotCount"] > 0:
            return None
        if not first_slot_only and item["slotCount"] == 0:
            # Depth pass only adds slots to facts that already have breadth coverage.
            return None

        blockers = _cap_blockers(
            fact=item["fact"],
            caps=caps,
            scope_counts=scope_counts,
            kind_counts=kind_counts,
            series_counts=series_counts,
            chunk_counts=chunk_counts,
            is_additional_slot=item["slotCount"] > 0,
            # Breadth pass relaxes scope caps so small targets can still cover
            # many distinct course_wide policies when that is where the value is.
            enforce_scope_caps=not first_slot_only,
        )
        if blockers:
            # Remember why this fact is currently blocked (last observed).
            item["capReasons"] = blockers
            return None

        factor = MULTI_SLOT_SCORE_FACTORS[
            min(item["slotCount"], len(MULTI_SLOT_SCORE_FACTORS) - 1)
        ]
        score = item["rankingScore"] * factor

        # Soft course-wide reservation boost until reserve is met.
        scope = str(item["fact"].get("scope") or "other")
        if (
            course_wide_allocated < reserve
            and scope == "course_wide"
            and item["slotCount"] == 0
        ):
            score += COURSE_WIDE_RESERVE_BOOST

        return score

    def _commit_slot(item: dict[str, Any], reason: str) -> None:
        nonlocal allocated_total, course_wide_allocated
        fact = item["fact"]
        scope = str(fact.get("scope") or "other")
        kind = str(fact.get("kind") or "other")
        series = _series_key(fact)
        chunk = _primary_source_chunk(fact)

        item["slotCount"] += 1
        allocated_total += 1
        scope_counts[scope] += 1
        kind_counts[kind] += 1
        if series is not None:
            series_counts[series] += 1
        if chunk is not None:
            chunk_counts[chunk] += 1
        if scope == "course_wide":
            course_wide_allocated += 1
        if reason not in item["allocReasons"]:
            item["allocReasons"].append(reason)
        if item["slotCount"] >= 2:
            multi_reason = "additional_slot_justified_by_complexity_or_scenarios"
            if multi_reason not in item["allocReasons"]:
                item["allocReasons"].append(multi_reason)

    def _fill_pass(*, first_slot_only: bool, pass_reason: str) -> None:
        nonlocal allocated_total
        while allocated_total < target:
            best: dict[str, Any] | None = None
            best_score = float("-inf")
            for item in candidates:
                score = _effective_score(item, first_slot_only=first_slot_only)
                if score is None:
                    continue
                if score > best_score or (
                    score == best_score
                    and best is not None
                    and (
                        item["rankingScore"] > best["rankingScore"]
                        or (
                            item["rankingScore"] == best["rankingScore"]
                            and item["factId"] < best["factId"]
                        )
                    )
                ):
                    best = item
                    best_score = score

            if best is None:
                break

            reason = pass_reason
            if (
                str(best["fact"].get("scope") or "") == "course_wide"
                and course_wide_allocated < reserve
                and best["slotCount"] == 0
            ):
                reason = "course_wide_reservation_priority"
            _commit_slot(best, reason)

    # Pass 1: breadth — one slot each to the best distinct eligible facts.
    _fill_pass(first_slot_only=True, pass_reason="breadth_first_slot")
    # Pass 2: depth — additional slots only after breadth, where complexity justifies.
    _fill_pass(first_slot_only=False, pass_reason="depth_additional_slot")

    # Build inspectable outputs.
    allocations: list[dict[str, Any]] = []
    ranking: list[dict[str, Any]] = []
    skipped_facts: list[dict[str, Any]] = []
    capped_facts: list[dict[str, Any]] = []

    for item in candidates:
        fact = item["fact"]
        styles = suggest_question_styles(fact, item["slotCount"]) or item["suggestedStyles"]
        allocation = {
            "factId": item["factId"],
            "slotCount": item["slotCount"],
            "desiredSlots": item["desiredSlots"],
            "rankingScore": item["rankingScore"],
            "suggestedStyles": styles,
            "reasons": list(item["allocReasons"]),
        }
        if item["capReasons"]:
            allocation["capReasons"] = list(item["capReasons"])
        allocations.append(allocation)

        ranking.append(
            {
                "factId": item["factId"],
                "rankingScore": item["rankingScore"],
                "desiredSlots": item["desiredSlots"],
                "slotCount": item["slotCount"],
                "importance": fact.get("importance"),
                "usefulnessScore": fact.get("usefulnessScore"),
                "studentAskLikelihood": fact.get("studentAskLikelihood"),
                "complexity": fact.get("complexity"),
                "kind": fact.get("kind"),
                "scope": fact.get("scope"),
                "seriesKey": fact.get("seriesKey"),
                "sourceChunkIds": list(fact.get("sourceChunkIds") or []),
                "reasons": list(item["allocReasons"]),
                "capReasons": list(item["capReasons"]),
            }
        )

        if item["slotCount"] == 0:
            skip_reasons = list(item["allocReasons"])
            if item["capReasons"]:
                skip_reasons = skip_reasons + [
                    f"capped:{reason}" for reason in item["capReasons"]
                ]
            elif item["desiredSlots"] > 0:
                skip_reasons.append("outcompeted_by_higher_ranked_facts")
            skipped_facts.append(
                {
                    "factId": item["factId"],
                    "desiredSlots": item["desiredSlots"],
                    "rankingScore": item["rankingScore"],
                    "reasons": skip_reasons,
                }
            )

        if item["desiredSlots"] > item["slotCount"] and item["slotCount"] >= 0:
            if item["capReasons"] or (
                item["desiredSlots"] > 0 and item["slotCount"] < item["desiredSlots"]
            ):
                # Only report as capped when caps (not just target exhaustion)
                # or target exhaustion reduced slots below desire.
                cap_reasons = list(item["capReasons"])
                if not cap_reasons and allocated_total >= target:
                    cap_reasons = ["target_count_exhausted"]
                if item["slotCount"] < item["desiredSlots"]:
                    capped_facts.append(
                        {
                            "factId": item["factId"],
                            "desiredSlots": item["desiredSlots"],
                            "slotCount": item["slotCount"],
                            "capReasons": cap_reasons,
                        }
                    )

    # Sort allocations by ranking (same order as candidates) for inspectability.
    by_scope = {scope: count for scope, count in sorted(scope_counts.items())}
    by_kind = {kind: count for kind, count in sorted(kind_counts.items())}
    by_series = {series: count for series, count in sorted(series_counts.items())}

    summary = {
        "targetCount": target,
        "allocatedSlots": allocated_total,
        "byScope": by_scope,
        "byKind": by_kind,
        "bySeries": by_series,
        "skippedFacts": skipped_facts,
        "cappedFacts": capped_facts,
        "caps": caps,
        "courseWideAllocated": course_wide_allocated,
        "courseWideReserve": reserve,
    }

    return {
        "allocations": allocations,
        "summary": summary,
        "ranking": ranking,
    }
