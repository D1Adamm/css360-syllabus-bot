"""Phase 8 dataset quality inspection for starter / reviewed seeds."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from app.seed_dedupe import normalize_question_for_dedupe
from app.seed_review import resolve_review_status

# Coverage themes relevant to CSS 360-style syllabi. Hits are optional — we do
# not force equal counts; we only report what is present vs missing.
COVERAGE_THEMES: list[tuple[str, re.Pattern[str]]] = [
    (
        "attendance_missing_class",
        re.compile(r"\b(attend|absence|miss(?:ing)? class|participation)\b", re.I),
    ),
    (
        "late_work_extensions",
        re.compile(r"\b(late|extension|deadline|half credit|48[- ]hour)\b", re.I),
    ),
    ("grading", re.compile(r"\b(grad(?:e|ing)|weight|rubric|points?|percent)\b", re.I)),
    (
        "instructor_contact_help",
        re.compile(r"\b(office hours|instructor|email|contact|help|ta\b)\b", re.I),
    ),
    (
        "assignments_projects",
        re.compile(r"\b(assignment|homework|project|lab|deliverable)\b", re.I),
    ),
    (
        "sprint_demo_reflection",
        re.compile(r"\b(sprint|demo|reflection|standup|retrospective)\b", re.I),
    ),
    (
        "plagiarism_integrity",
        re.compile(r"\b(plagiar|academic integrity|cheat|ai[- ]generated)\b", re.I),
    ),
    (
        "accommodations",
        re.compile(r"\b(accommodat|disability|DRS|accessibil)\b", re.I),
    ),
    (
        "submission_expectations",
        re.compile(r"\b(submit|submission|turn in|upload|canvas)\b", re.I),
    ),
]

_TRIVIAL_RE = re.compile(
    r"^(what is the course|when is (the )?class|where is (the )?class)\b",
    re.I,
)
_MUST_RE = re.compile(r"\b(must|required|always)\b", re.I)
_MAY_RE = re.compile(r"\b(may|optional|suggested|recommended)\b", re.I)


def _seed_text(seed: dict[str, Any]) -> str:
    return " ".join(
        [
            str(seed.get("question") or seed.get("instruction") or ""),
            str(seed.get("answer") or seed.get("response") or ""),
            str(seed.get("evidenceQuote") or ""),
            str(seed.get("category") or ""),
            str(seed.get("kind") or ""),
        ]
    )


def _validation_score(seed: dict[str, Any]) -> float | None:
    validation = seed.get("validation")
    if not isinstance(validation, dict):
        return None
    score = validation.get("score")
    if isinstance(score, (int, float)):
        return float(score)
    return None


def inspect_seed_dataset(
    seeds: list[dict[str, Any]],
    *,
    near_duplicate_normalized_match: bool = True,
) -> dict[str, Any]:
    """Inspect a seed list for coverage, quality flags, and review mix."""
    by_review: Counter[str] = Counter()
    issues: list[dict[str, Any]] = []
    theme_hits: Counter[str] = Counter({name: 0 for name, _ in COVERAGE_THEMES})
    seen_keys: dict[str, str] = {}

    for index, seed in enumerate(seeds):
        seed_id = str(seed.get("id") or seed.get("factId") or f"index-{index}")
        question = str(seed.get("question") or seed.get("instruction") or "").strip()
        answer = str(seed.get("answer") or seed.get("response") or "").strip()
        review = resolve_review_status(seed)
        by_review[review] += 1
        text = _seed_text(seed)

        for theme, pattern in COVERAGE_THEMES:
            if pattern.search(text):
                theme_hits[theme] += 1

        score = _validation_score(seed)
        if score is not None and score < 0.8:
            issues.append(
                {
                    "seedId": seed_id,
                    "kind": "low_validation_score",
                    "detail": f"validation.score={score}",
                }
            )

        unsupported = []
        validation = seed.get("validation")
        if isinstance(validation, dict):
            raw = validation.get("unsupportedClaims") or []
            if isinstance(raw, list):
                unsupported = [str(item) for item in raw if str(item).strip()]
        if unsupported:
            issues.append(
                {
                    "seedId": seed_id,
                    "kind": "unsupported_claims",
                    "detail": "; ".join(unsupported[:3]),
                }
            )

        if question and _TRIVIAL_RE.search(question):
            issues.append(
                {
                    "seedId": seed_id,
                    "kind": "trivial_or_low_value",
                    "detail": "Question looks generic/trivial",
                }
            )

        evidence = str(seed.get("evidenceQuote") or "")
        if _MAY_RE.search(evidence) and _MUST_RE.search(answer) and not _MAY_RE.search(
            answer
        ):
            issues.append(
                {
                    "seedId": seed_id,
                    "kind": "missing_qualifiers_or_modal_escalation",
                    "detail": "Evidence uses may/optional but answer uses must/required",
                }
            )

        if near_duplicate_normalized_match and question:
            key = normalize_question_for_dedupe(question)
            if key in seen_keys:
                issues.append(
                    {
                        "seedId": seed_id,
                        "kind": "near_duplicate_question",
                        "detail": f"Matches {seen_keys[key]}",
                    }
                )
            else:
                seen_keys[key] = seed_id

        if not seed.get("factId") and seed.get("origin") == "ai_generated":
            issues.append(
                {
                    "seedId": seed_id,
                    "kind": "missing_fact_id",
                    "detail": "AI seed missing factId",
                }
            )
        if not str(seed.get("evidenceQuote") or "").strip() and seed.get(
            "origin"
        ) == "ai_generated":
            issues.append(
                {
                    "seedId": seed_id,
                    "kind": "missing_evidence_quote",
                    "detail": "AI seed missing evidenceQuote",
                }
            )

    present_themes = sorted(name for name, count in theme_hits.items() if count > 0)
    missing_themes = sorted(name for name, count in theme_hits.items() if count == 0)

    return {
        "seedCount": len(seeds),
        "reviewStatusCounts": dict(by_review),
        "coverage": {
            "themeHits": dict(theme_hits),
            "presentThemes": present_themes,
            "missingThemes": missing_themes,
            "note": (
                "Missing themes are informational only; do not force equal "
                "counts if the syllabus/dataset does not support them."
            ),
        },
        "issueCounts": dict(Counter(item["kind"] for item in issues)),
        "issues": issues,
        "approvedCount": by_review.get("approved", 0),
        "generatedCount": by_review.get("generated", 0),
        "rejectedCount": by_review.get("rejected", 0),
        "editedCount": by_review.get("edited", 0),
    }
