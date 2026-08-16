"""Balancing rules for starter seed datasets."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

SCHEDULE_DEFAULT_CAP_RATIO = 0.20
SCHEDULE_MAJOR_CAP_RATIO = 0.30
TOPIC_CAP_RATIO = 0.20
SCENARIO_MIN_RATIO = 0.30

SCENARIO_LIKE_TYPES = frozenset({"scenario", "clarification"})

_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_SCHEDULE_RE = re.compile(
    r"\b("
    r"deadline|deadlines|due|due date|due dates|submit by|when is|what time|what day|"
    r"class meeting|class meetings|calendar|schedule|schedules|date|dates|time|times|"
    r"week \d+|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"midterm|final exam|exam date|office hours"
    r")\b",
    re.IGNORECASE,
)


def get_topic_key(seed_or_topic: dict[str, Any]) -> str:
    return str(
        seed_or_topic.get("category")
        or seed_or_topic.get("topicName")
        or seed_or_topic.get("name")
        or "General"
    ).strip() or "General"


def is_schedule_like_question(
    *,
    question: str,
    category: str = "",
    topic_summary: str = "",
) -> bool:
    text = " ".join(part for part in (question, category, topic_summary) if part).strip()
    lowered = text.lower()
    if not lowered:
        return False
    if not _SCHEDULE_RE.search(lowered) and not _URL_RE.search(lowered):
        return False
    if any(term in lowered for term in ("canvas", "discord", "zoom")) and not (
        _SCHEDULE_RE.search(lowered) or _URL_RE.search(lowered)
    ):
        return False
    return True


def schedule_topics_are_major(topics: list[dict[str, Any]]) -> bool:
    total = sum(max(1, int(topic.get("suggestedExampleCount", 1))) for topic in topics) or 1
    schedule_total = sum(
        max(1, int(topic.get("suggestedExampleCount", 1)))
        for topic in topics
        if bool(topic.get("scheduleHeavy"))
    )
    has_high_schedule = any(
        topic.get("scheduleHeavy") and topic.get("importance") == "high" for topic in topics
    )
    return has_high_schedule and (schedule_total / total) >= 0.25


def compute_schedule_cap(target_count: int, topics: list[dict[str, Any]]) -> int:
    ratio = (
        SCHEDULE_MAJOR_CAP_RATIO
        if schedule_topics_are_major(topics)
        else SCHEDULE_DEFAULT_CAP_RATIO
    )
    return max(1, math.floor(target_count * ratio))


def compute_topic_cap(target_count: int) -> int:
    return max(1, math.floor(target_count * TOPIC_CAP_RATIO))


def compute_scenario_minimum(target_count: int) -> int:
    return max(1, math.ceil(target_count * SCENARIO_MIN_RATIO))


def count_question_types(seeds: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(seed.get("questionType", "")).strip().lower() for seed in seeds)


def count_topics(seeds: list[dict[str, Any]]) -> Counter[str]:
    return Counter(get_topic_key(seed) for seed in seeds)


def count_schedule_like(seeds: list[dict[str, Any]]) -> int:
    total = 0
    for seed in seeds:
        if is_schedule_like_question(
            question=str(seed.get("question", "")),
            category=get_topic_key(seed),
            topic_summary=str(seed.get("topicSummary", "")),
        ):
            total += 1
    return total


def count_scenario_or_clarification(seeds: list[dict[str, Any]]) -> int:
    """How many accepted seeds are scenario- or clarification-shaped.

    The companion to `compute_scenario_minimum`. Reporting a minimum with no
    corresponding actual meant nobody could tell whether it had been met — or,
    as happened on a fact-starved run, that it was arithmetically unreachable.
    """
    return sum(
        1
        for seed in seeds
        if str(seed.get("questionType", "")).strip().lower() in SCENARIO_LIKE_TYPES
    )


def scenario_requirement_remaining(
    *,
    accepted_seeds: list[dict[str, Any]],
    target_count: int,
    existing_scenario_count: int = 0,
) -> int:
    """How many more scenario-like seeds the run still owes.

    `existing_scenario_count` carries the seeds a top-up is adding to. Balance
    is a property of the course a student eventually sees, not of one run's
    slice of it: a course that already holds nine scenario seeds does not need
    the same thing again from the next forty.
    """
    current = max(0, existing_scenario_count) + sum(
        1
        for seed in accepted_seeds
        if str(seed.get("questionType", "")).strip().lower() in SCENARIO_LIKE_TYPES
    )
    return max(0, compute_scenario_minimum(target_count) - current)


def would_violate_balancing(
    *,
    candidate: dict[str, Any],
    accepted_seeds: list[dict[str, Any]],
    target_count: int,
    planner_topics: list[dict[str, Any]],
) -> str | None:
    topic_key = get_topic_key(candidate)
    schedule_cap = compute_schedule_cap(target_count, planner_topics)
    if is_schedule_like_question(
        question=str(candidate.get("question", "")),
        category=topic_key,
        topic_summary=str(candidate.get("topicSummary", "")),
    ) and count_schedule_like(accepted_seeds) >= schedule_cap:
        return "schedule_cap"

    topic_cap = compute_topic_cap(target_count)
    topic_counts = count_topics(accepted_seeds)
    if topic_counts[topic_key] >= topic_cap:
        return "topic_cap"

    return None


def should_prefer_scenario_or_clarification(
    *,
    accepted_seeds: list[dict[str, Any]],
    remaining_slots: int,
    target_count: int,
    existing_scenario_count: int = 0,
    eligible_slots_remaining: int | None = None,
) -> bool:
    """Whether the remaining work must now go to scenario-like questions.

    Urgency-based rather than always-on, so a run spends its early slots on
    whatever each fact suits best and converges on the minimum only when it
    would otherwise miss it.

    `eligible_slots_remaining` is what makes that urgency real. Only some facts
    can carry a scenario — policy-shaped ones with conditions — and those tend
    to rank high, so they are spent early. Measured against *total* slots the
    deficit looks comfortable until the last stretch, by which point every
    remaining fact is a contact detail or a deadline and the preference has
    nothing left to apply to. Measured against the slots that could actually
    take one, pressure is visible while those slots still exist.

    On a top-up the deficit is course-wide — see `scenario_requirement_remaining`.
    """
    if remaining_slots <= 0:
        return False
    required_remaining = scenario_requirement_remaining(
        accepted_seeds=accepted_seeds,
        target_count=target_count,
        existing_scenario_count=existing_scenario_count,
    )
    if required_remaining <= 0:
        return False

    budget = remaining_slots
    if eligible_slots_remaining is not None:
        budget = min(budget, max(0, eligible_slots_remaining))
    return required_remaining >= budget
