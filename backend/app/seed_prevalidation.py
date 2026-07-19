"""Deterministic pre-validation for starter seed candidates (Phase 5).

Rejects *obviously* ungrounded or escalated candidates BEFORE an LLM validation
call. Never auto-accepts: LLM validation remains the final acceptance gate.
"""

from __future__ import annotations

import re
from typing import Any

from app.syllabus_facts import statement_entailment_violation

# Permission / soft modality in evidence.
_SOFT_MODAL_RE = re.compile(
    r"\b("
    r"may|might|can|could|optional|suggested|recommend(?:ed|ation)?|"
    r"encouraged|allowed|permitted|available"
    r")\b",
    re.IGNORECASE,
)

# Hard obligation / absolute language in answers.
_HARD_OBLIGATION_RE = re.compile(
    r"\b("
    r"must|required|mandatory|have to|shall|obligated|need to|"
    r"are expected to|you are to"
    r")\b",
    re.IGNORECASE,
)

_ABSOLUTE_RE = re.compile(
    r"\b("
    r"always|never|everyone|everybody|guaranteed|guarantee|"
    r"certainly|definitely|without exception"
    r")\b",
    re.IGNORECASE,
)

_RESPONSE_TIME_RE = re.compile(
    r"\b(within\s+\d+\s*(?:hour|hr|day)s?|\d+\s*(?:hour|hr|day)s?\b)",
    re.IGNORECASE,
)
_RESPONSE_VERB_RE = re.compile(
    r"\b(respond|response|reply|get back|answer)\b",
    re.IGNORECASE,
)
_CONDITIONAL_NUDGE_RE = re.compile(
    r"("
    r"if\s+i\s+(do\s+not|don't)\s+respond|"
    r"nudge|"
    r"unintentional|"
    r"no\s+response|"
    r"does\s+not\s+respond"
    r")",
    re.IGNORECASE,
)

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# High-signal qualifier tokens that, when present in evidence, should not be
# contradicted by an absolute rewrite in the answer.
_LIMIT_PHRASE_RE = re.compile(
    r"\b("
    r"one|once|per\s+quarter|per\s+term|except|unless|only|"
    r"no\s+extension|not\s+allowed|half\s+credit|24\s*hours?|48\s*hours?"
    r")\b",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "").strip().lower())


def _token_set(text: str) -> set[str]:
    return {tok for tok in _NON_ALNUM_RE.sub(" ", _norm(text)).split() if len(tok) > 2}


def _evidence_bundle(fact: dict[str, Any], source_text: str = "") -> str:
    parts = [
        str(fact.get("statement") or ""),
        str(fact.get("evidenceQuote") or ""),
        source_text or "",
    ]
    return " ".join(part for part in parts if part.strip())


def _soft_modal_in_evidence(evidence: str) -> bool:
    return bool(_SOFT_MODAL_RE.search(evidence))


def _hard_obligation_in_text(text: str) -> bool:
    return bool(_HARD_OBLIGATION_RE.search(text))


def _absolute_in_text(text: str) -> bool:
    return bool(_ABSOLUTE_RE.search(text))


def detect_modal_escalation(*, answer: str, evidence: str) -> str | None:
    """Reject may/can/suggested → must/required when evidence stays soft."""
    answer_text = _norm(answer)
    evidence_text = _norm(evidence)
    if not answer_text or not evidence_text:
        return None

    # Recommendation → requirement (check before generic soft→hard escalation).
    if re.search(r"\b(suggest|recommend|encourag)", evidence_text) and re.search(
        r"\b(must|required|mandatory)\b", answer_text
    ):
        if not re.search(r"\b(must|required|mandatory)\b", evidence_text):
            return "recommendation_as_requirement"

    answer_hard = _hard_obligation_in_text(answer_text)
    evidence_hard = _hard_obligation_in_text(evidence_text)
    if answer_hard and not evidence_hard and _soft_modal_in_evidence(evidence_text):
        return "modal_escalation"

    return None


def detect_absolute_overclaim(*, answer: str, evidence: str) -> str | None:
    """Reject absolute language not present in evidence."""
    answer_text = _norm(answer)
    evidence_text = _norm(evidence)
    if not answer_text:
        return None

    for match in _ABSOLUTE_RE.finditer(answer_text):
        term = match.group(1).lower()
        if term not in evidence_text:
            return f"absolute_language:{term}"
    return None


def detect_response_time_guarantee(*, answer: str, evidence: str) -> str | None:
    """Reject turning response-time nudges into hard guarantees."""
    answer_text = _norm(answer)
    evidence_text = _norm(evidence)
    answer_promises = bool(_RESPONSE_TIME_RE.search(answer_text)) and bool(
        _RESPONSE_VERB_RE.search(answer_text)
    )
    if not answer_promises:
        return None
    if re.search(r"\b(not\s+guarantee|no\s+guarantee|not\s+guaranteed)\b", answer_text):
        return None
    if _CONDITIONAL_NUDGE_RE.search(evidence_text):
        return "response_time_guarantee"
    # Reuse entailment helper for guarantee-from-conditional patterns.
    violation = statement_entailment_violation(answer_text, evidence_text)
    if violation in {"guarantee_from_conditional", "guarantee_not_in_evidence"}:
        return "response_time_guarantee"
    return None


def detect_missing_critical_qualifier(*, answer: str, evidence: str) -> str | None:
    """Reject answers that drop an explicit numeric/exception limit from evidence.

    Conservative: only fires when evidence has a clear limit phrase and the answer
    both discusses the same topic and omits that limit while using broader wording.
    """
    evidence_text = _norm(evidence)
    answer_text = _norm(answer)
    if not evidence_text or not answer_text:
        return None

    evidence_limits = {m.group(0).lower() for m in _LIMIT_PHRASE_RE.finditer(evidence_text)}
    if not evidence_limits:
        return None

    # Focus on high-value late-work / extension limits.
    topic_overlap = bool(
        re.search(r"\b(extensions?|late|deadline|due)\b", evidence_text)
    ) and bool(re.search(r"\b(extensions?|late|deadline|due)\b", answer_text))
    if not topic_overlap:
        return None

    missing = []
    for phrase in sorted(evidence_limits):
        # Normalize multi-word phrases for containment.
        if phrase not in answer_text:
            # Allow close variants for hour counts already checked via tokens.
            tokens = phrase.split()
            if all(tok in answer_text for tok in tokens if len(tok) > 2):
                continue
            missing.append(phrase)

    # Only reject when a strong limit like "one"/"except"/"per quarter" is gone
    # and the answer uses broader plural/open wording.
    strong_missing = [
        phrase
        for phrase in missing
        if phrase
        in {
            "one",
            "once",
            "per quarter",
            "per term",
            "except",
            "unless",
            "only",
            "no extension",
            "not allowed",
        }
    ]
    if not strong_missing:
        return None

    broader = bool(
        re.search(r"\b(extensions|any|all|unlimited)\b", answer_text)
    ) or (
        "one" in strong_missing
        and re.search(r"\bextensions?\b", answer_text)
        and not re.search(r"\b(one|single|1)\b", answer_text)
    )
    if broader or ("except" in strong_missing or "unless" in strong_missing):
        return "missing_qualifier:" + ",".join(strong_missing[:3])
    return None


def detect_question_beyond_evidence(*, question: str, evidence: str) -> str | None:
    """Light check: reject questions that clearly demand info absent from evidence."""
    question_text = _norm(question)
    evidence_text = _norm(evidence)
    if not question_text or not evidence_text:
        return None

    # Asking for consequences/penalties when evidence has none.
    if re.search(r"\b(what\s+happens|penalty|punish|fail|grade\s+impact)\b", question_text):
        if not re.search(
            r"\b(penalty|fail|zero|grade|impact|consequence|no\s+credit)\b",
            evidence_text,
        ):
            return "question_asks_absent_consequence"
    return None


def prevalidate_candidate(
    *,
    candidate: dict[str, Any],
    fact: dict[str, Any],
    source_text: str = "",
) -> dict[str, str] | None:
    """Return a rejection descriptor, or None if the candidate may go to LLM validation.

    Descriptor keys:
    - reason: stable machine code
    - category: one of modal_escalation | qualifier_mismatch | prevalidation
    """
    question = str(candidate.get("question") or "").strip()
    answer = str(candidate.get("answer") or "").strip()
    if not question or not answer:
        return {"reason": "empty_qa", "category": "prevalidation"}

    evidence = _evidence_bundle(fact, source_text)

    modal = detect_modal_escalation(answer=answer, evidence=evidence)
    if modal:
        return {"reason": modal, "category": "modal_escalation"}

    absolute = detect_absolute_overclaim(answer=answer, evidence=evidence)
    if absolute:
        return {"reason": absolute, "category": "modal_escalation"}

    guarantee = detect_response_time_guarantee(answer=answer, evidence=evidence)
    if guarantee:
        return {"reason": guarantee, "category": "modal_escalation"}

    # Reuse fact-inventory entailment against the answer as a statement.
    entailment = statement_entailment_violation(answer, evidence)
    if entailment == "obligation_not_in_evidence":
        return {"reason": "obligation_not_in_evidence", "category": "modal_escalation"}
    if entailment in {"guarantee_from_conditional", "guarantee_not_in_evidence"}:
        return {"reason": entailment, "category": "modal_escalation"}

    missing = detect_missing_critical_qualifier(answer=answer, evidence=evidence)
    if missing:
        return {"reason": missing, "category": "qualifier_mismatch"}

    beyond = detect_question_beyond_evidence(question=question, evidence=evidence)
    if beyond:
        return {"reason": beyond, "category": "prevalidation"}

    return None
