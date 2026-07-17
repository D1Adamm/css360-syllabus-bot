"""Rubric-based validation helpers for starter seed generation."""

from __future__ import annotations

import json
import re
from typing import Any

VALIDATION_PROMPT_MARKER = "You validate syllabus seed examples."

MIN_GROUNDED_SCORE = 0.80
MIN_CORRECT_SCORE = 0.80
MIN_CATEGORY_CORRECT_SCORE = 0.70
MIN_NOT_TRIVIAL_OR_TEMPORARY_SCORE = 0.70
MIN_OTHER_COMPONENT_SCORE = 0.60
MIN_OVERALL_VALIDATION_SCORE = 0.80

# Soft-cap for model-returned perfect scores unless every claim is explicit.
PERFECT_COMPONENT_SOFT_CAP = 0.94

VALIDATION_WEIGHTS = {
    "grounded": 0.20,
    "correct": 0.20,
    "clear": 0.10,
    "useful": 0.15,
    "naturalStudentWording": 0.15,
    "categoryCorrect": 0.10,
    "notTrivialOrTemporary": 0.10,
}

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_SCHEDULE_RE = re.compile(
    r"\b(deadline|due|date|time|schedule|office hours|midterm|final exam|when is|what time)\b",
    re.IGNORECASE,
)
_ADVICE_RE = re.compile(
    r"\b("
    r"recommend|recommended|recommendation|"
    r"suggest|suggested|suggestion|"
    r"strategy|best practice|ideally|"
    r"divide(?:\s+\w+){0,4}\s+work|allocate(?:\s+\w+){0,4}\s+(?:roles|tasks|work)"
    r")\b",
    re.IGNORECASE,
)


def _extract_json_text(raw: str) -> str:
    text = raw.strip()
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def build_validation_prompt(
    *,
    question: str,
    answer: str,
    topic_name: str,
    question_type: str,
    chunk_text: str,
) -> str:
    return f"""{VALIDATION_PROMPT_MARKER}

Evaluate whether this generated syllabus starter seed is grounded, useful, and well-categorized.
Be conservative. Most acceptable examples should score around 0.75–0.92 overall via components.
A component score of 1.0 means exceptionally strong with no meaningful weakness.
Do not assign 1.0 to any component unless every claim is explicitly supported by the source
and the example is highly useful and natural.

For grounded and correct:
- Evaluate every factual or procedural claim in the answer.
- If the answer adds advice, interpretation, or a suggested strategy not explicitly supported
  by the source, lower grounded and correct.
- List every important unsupported claim in unsupportedClaims.
- If any important unsupported claim remains, unsupportedClaims must be non-empty.

Contrasting examples:
1) Fully grounded:
   Source: "Late work may be submitted within 24 hours for half credit."
   Answer: "Late work may be submitted within 24 hours for half credit."
   Expected: high grounded/correct; unsupportedClaims=[].

2) Partly grounded but embellished:
   Source: "Team projects require weekly status updates and a final demo."
   Answer: "Team projects require weekly status updates and a final demo. Split work so one
   person owns docs, one owns coding, and one owns testing."
   Expected: lower grounded/correct; unsupportedClaims includes the work-division strategy.

3) Vague but technically related:
   Source: "Office hours are Tuesdays at 2pm."
   Answer: "You can ask questions at office hours sometime."
   Expected: lower clear/useful; keep grounded modest if specifics are lost.

4) Trivial schedule fact:
   Source: "Office hours are Tuesdays at 2pm."
   Answer: "Office hours are on Tuesdays."
   Expected: lower useful and notTrivialOrTemporary.

5) Strong student scenario:
   Source: "Late work may be submitted within 24 hours for half credit."
   Answer: "If you submit 12 hours after the deadline, you can still receive half credit."
   Expected: high useful/naturalStudentWording with strong grounded/correct.

Return only valid JSON. Do NOT include an overall score field.
{{
  "grounded": 0.0,
  "correct": 0.0,
  "clear": 0.0,
  "useful": 0.0,
  "naturalStudentWording": 0.0,
  "categoryCorrect": 0.0,
  "notTrivialOrTemporary": 0.0,
  "unsupportedClaims": ["string"],
  "reason": "short explanation"
}}

Topic/category: {topic_name}
Question type: {question_type}

Question:
{question}

Answer:
{answer}

Source chunk text:
{chunk_text}
"""


def compute_validation_score(components: dict[str, float]) -> float:
    return round(
        sum(
            float(components.get(key, 0.0)) * weight
            for key, weight in VALIDATION_WEIGHTS.items()
        ),
        4,
    )


def _normalize_unsupported_claims(raw: Any) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    claims: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            claims.append(item.strip())
    return claims


def canonicalize_validation_result(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize validation to the persisted canonical structure."""
    components_raw = result.get("components")
    if isinstance(components_raw, dict):
        components = {
            key: float(components_raw.get(key, 0.0)) for key in VALIDATION_WEIGHTS
        }
    else:
        components = {key: float(result.get(key, 0.0)) for key in VALIDATION_WEIGHTS}

    unsupported = _normalize_unsupported_claims(result.get("unsupportedClaims"))
    reason = str(result.get("reason", "")).strip() or "No reason provided."
    score = compute_validation_score(components)

    return {
        "score": score,
        "reason": reason,
        "unsupportedClaims": unsupported,
        "components": components,
    }


def calibrate_validation_result(
    *,
    result: dict[str, Any],
    question: str,
    answer: str,
    topic_name: str,
    question_type: str,
) -> dict[str, Any]:
    canonical = canonicalize_validation_result(result)
    components = dict(canonical["components"])
    unsupported = list(canonical["unsupportedClaims"])

    question_text = question.strip()
    answer_text = answer.strip()
    lowered_question = question_text.lower()
    lowered_topic = topic_name.strip().lower()
    lowered_type = question_type.strip().lower()

    # Soft-cap perfect component scores; 1.0 should be rare.
    if all(float(value) >= 0.999 for value in components.values()):
        for key in components:
            components[key] = min(float(components[key]), PERFECT_COMPONENT_SOFT_CAP)

    for key, value in list(components.items()):
        if float(value) >= 0.999:
            components[key] = min(float(value), PERFECT_COMPONENT_SOFT_CAP)

    if lowered_type == "direct":
        components["naturalStudentWording"] = min(
            float(components["naturalStudentWording"]),
            0.92,
        )
    if len(question_text) < 45:
        components["useful"] = min(float(components["useful"]), 0.9)
        components["clear"] = min(float(components["clear"]), 0.95)
    if len(answer_text) < 60:
        components["useful"] = min(float(components["useful"]), 0.88)
    if _SCHEDULE_RE.search(lowered_question) or _SCHEDULE_RE.search(lowered_topic):
        components["notTrivialOrTemporary"] = min(
            float(components["notTrivialOrTemporary"]),
            0.78,
        )

    # Advice / strategy language that is often embellishment beyond the syllabus.
    if _ADVICE_RE.search(answer_text):
        components["grounded"] = min(float(components["grounded"]), 0.78)
        components["correct"] = min(float(components["correct"]), 0.78)
        if not unsupported:
            unsupported.append(
                "Answer includes advice or strategy language that may not be "
                "explicitly supported by the source."
            )

    return {
        "score": compute_validation_score(components),
        "reason": canonical["reason"],
        "unsupportedClaims": unsupported,
        "components": components,
    }


def try_parse_validation_payload(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(_extract_json_text(raw))
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    # Model must not own the overall score.
    parsed.pop("score", None)

    components: dict[str, float] = {}
    for key in VALIDATION_WEIGHTS:
        value = parsed.get(key)
        if not isinstance(value, (int, float)):
            return None
        normalized = float(value)
        if normalized < 0.0 or normalized > 1.0:
            return None
        components[key] = normalized

    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return None

    unsupported = _normalize_unsupported_claims(parsed.get("unsupportedClaims"))

    return canonicalize_validation_result(
        {
            "components": components,
            "reason": reason.strip(),
            "unsupportedClaims": unsupported,
        }
    )


def validation_result_accepts(result: dict[str, Any]) -> bool:
    canonical = canonicalize_validation_result(result)
    components = canonical["components"]

    if canonical["unsupportedClaims"]:
        return False

    if float(components.get("grounded", 0.0)) < MIN_GROUNDED_SCORE:
        return False
    if float(components.get("correct", 0.0)) < MIN_CORRECT_SCORE:
        return False
    if float(components.get("categoryCorrect", 0.0)) < MIN_CATEGORY_CORRECT_SCORE:
        return False
    if (
        float(components.get("notTrivialOrTemporary", 0.0))
        < MIN_NOT_TRIVIAL_OR_TEMPORARY_SCORE
    ):
        return False

    for key in ("clear", "useful", "naturalStudentWording"):
        if float(components.get(key, 0.0)) < MIN_OTHER_COMPONENT_SCORE:
            return False

    return float(canonical["score"]) >= MIN_OVERALL_VALIDATION_SCORE
