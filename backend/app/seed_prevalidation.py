"""Deterministic pre-validation for starter seed candidates (Phase 5).

Rejects *obviously* ungrounded or escalated candidates BEFORE an LLM validation
call. Never auto-accepts: LLM validation remains the final acceptance gate.
"""

from __future__ import annotations

import re
from typing import Any

from app.syllabus_facts import statement_entailment_violation

# Soft / advisory modality in evidence (permission, suggestion, emphasis).
_SOFT_MODAL_RE = re.compile(
    r"\b("
    r"may|might|can|could|optional|suggested|recommend(?:ed|ation)?|"
    r"encouraged?|allowed|permitted|available|"
    r"should|ought(?:\s+to)?|"
    r"important(?:\s+to)?|"
    r"welcome\s+to|please|"
    r"feel\s+free|invited\s+to"
    r")\b",
    re.IGNORECASE,
)

# Soft/advisory cues used for recommendation → requirement escalation.
_ADVISORY_RE = re.compile(
    r"\b("
    r"suggest(?:ed|ion)?|recommend(?:ed|ation)?|encourag(?:e|ed|ement)?|"
    r"should|ought(?:\s+to)?|"
    r"important(?:\s+to)?|"
    r"please|welcome\s+to|feel\s+free|invited\s+to"
    r")\b",
    re.IGNORECASE,
)

# Permission / invitation cues that must not become obligations.
_PERMISSION_RE = re.compile(
    r"\b("
    r"may|might|can|could|optional|allowed|permitted|available|"
    r"welcome\s+to|feel\s+free|invited\s+to"
    r")\b",
    re.IGNORECASE,
)

# Hard obligation / requirement language in questions or answers.
# Includes interrogative forms ("do I need to", "must I") that assert duty.
_HARD_OBLIGATION_RE = re.compile(
    r"("
    r"\b(?:must|required|mandatory|shall|obligated)\b|"
    r"\b(?:have to|need to|required to)\b|"
    r"\bare expected to\b|"
    r"\byou are to\b|"
    r"\bdo i need to\b|"
    r"\bam i required to\b|"
    r"\bmust i\b|"
    r"\bstudents must\b|"
    r"\byou must\b"
    r")",
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

# Restrictive / fallback conditions that scope when a rule applies.
_RESTRICTIVE_CONDITION_RE = re.compile(
    r"\b("
    r"if\s+you\s+(don'?t|do\s+not)|"
    r"if\s+there\s+(is|are)\s+no|"
    r"if\s+no\b|"
    r"when\s+no|"
    r"unless|"
    r"except|"
    r"only\s+when|"
    r"only\s+if|"
    r"provided\s+that|"
    r"if\s+you\s+(cannot|can'?t)"
    r")\b",
    re.IGNORECASE,
)

# Markers that count as preserving a condition in Q/A.
_CONDITION_PRESERVED_RE = re.compile(
    r"\b("
    r"if|unless|except|only|when|provided|otherwise|"
    r"no\s+obvious|don'?t\s+see|do\s+not\s+see|"
    r"no\s+place|not\s+see|cannot\s+find|can'?t\s+find"
    r")\b",
    re.IGNORECASE,
)

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_HASH_TAG_RE = re.compile(r"#([a-z0-9\-_]{2,})", re.IGNORECASE)

# High-signal qualifier tokens that, when present in evidence, should not be
# contradicted by an absolute rewrite in the answer.
_LIMIT_PHRASE_RE = re.compile(
    r"\b("
    r"one|once|per\s+quarter|per\s+term|except|unless|only|"
    r"no\s+extension|not\s+allowed|half\s+credit|24\s*hours?|48\s*hours?"
    r")\b",
    re.IGNORECASE,
)

# Evidence says "within the first N …"; Q/A must not flip that to "before …".
# Accept digits or common number words used in syllabi ("two weeks").
_FIRST_WINDOW_COUNT = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
_WITHIN_FIRST_WINDOW_RE = re.compile(
    rf"\bwithin\s+the\s+first\s+({_FIRST_WINDOW_COUNT})\s+(weeks?|days?|hours?)\b",
    re.IGNORECASE,
)
_BEFORE_FIRST_WINDOW_RE = re.compile(
    rf"\bbefore\s+the\s+first\s+({_FIRST_WINDOW_COUNT})\s+(weeks?|days?|hours?)\b",
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


def fact_qualifier_evidence(fact: dict[str, Any]) -> str:
    """Evidence for dropped-condition / missing-qualifier checks.

    Uses only this fact's statement + evidenceQuote. Neighboring syllabus
    sentences from ``source_text`` must not invent restrictive conditions or
    numeric limits that this candidate never claimed to cover.
    """
    return _evidence_bundle(fact, source_text="")


def _soft_modal_in_evidence(evidence: str) -> bool:
    return bool(_SOFT_MODAL_RE.search(evidence))


def _hard_obligation_in_text(text: str) -> bool:
    return bool(_HARD_OBLIGATION_RE.search(text))


def modality_evidence_for_fact(fact: dict[str, Any]) -> str:
    """Evidence text used for obligation/modality checks.

    Prefer ``evidenceQuote`` over ``statement``. Full chunk ``source_text`` is
    intentionally excluded: neighboring syllabus sentences often contain unrelated
    ``must``/``required`` wording that would falsely mark soft facts as mandatory.

    When the quote is advisory/soft (important/should/recommended/…) and does not
    itself contain hard obligation language, use the quote alone even if the
    inventory ``statement`` was already overstated to ``must``.
    """
    quote = str(fact.get("evidenceQuote") or "").strip()
    statement = str(fact.get("statement") or "").strip()
    if quote:
        quote_norm = _norm(quote)
        if _ADVISORY_RE.search(quote_norm) and not _hard_obligation_in_text(quote_norm):
            return quote
        return quote
    return statement


def detect_modal_escalation(
    *,
    answer: str,
    evidence: str,
    question: str = "",
) -> str | None:
    """Reject soft/advisory evidence rewritten as hard requirements in Q or A."""
    answer_text = _norm(answer)
    question_text = _norm(question)
    evidence_text = _norm(evidence)
    qa_text = f"{question_text} {answer_text}".strip()
    if not qa_text or not evidence_text:
        return None

    qa_hard = _hard_obligation_in_text(qa_text)
    evidence_hard = _hard_obligation_in_text(evidence_text)
    if not qa_hard:
        return None

    # Soft advisory evidence must not ground must/required/need-to in Q or A.
    if (
        not evidence_hard
        and _ADVISORY_RE.search(evidence_text)
        and re.search(
            r"("
            r"\b(?:must|required|mandatory)\b|"
            r"\b(?:need to|have to|required to)\b|"
            r"\bdo i need to\b|"
            r"\bam i required to\b|"
            r"\bmust i\b"
            r")",
            qa_text,
        )
    ):
        return "recommendation_as_requirement"

    # Permission / invitation → requirement
    if not evidence_hard and _PERMISSION_RE.search(evidence_text):
        return "modal_escalation"

    # Generic soft → hard when evidence stays soft and Q/A invent obligation.
    if not evidence_hard and _soft_modal_in_evidence(evidence_text):
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


def detect_dropped_condition(
    *,
    question: str,
    answer: str,
    evidence: str,
) -> str | None:
    """Reject answers that turn a restrictive/fallback condition into a universal rule.

    Fires only for high-signal conditions (if you don't / unless / only when / except /
    …). Harmless paraphrases that keep an if/unless/when cue still pass. Mild
    eligibility phrasing such as "if you have not yet…" is intentionally not covered
    here so optional contact wording is not over-rejected.
    """
    evidence_text = _norm(evidence)
    answer_text = _norm(answer)
    question_text = _norm(question)
    combined = f"{question_text} {answer_text}".strip()
    if not evidence_text or not answer_text:
        return None

    if not _RESTRICTIVE_CONDITION_RE.search(evidence_text):
        return None

    # Condition preserved in the question or answer — allow paraphrase.
    if _CONDITION_PRESERVED_RE.search(combined):
        return None

    # Require that the answer still asserts a concrete consequent from evidence
    # (channel tag, or meaningful content overlap) rather than a vague hedge.
    evidence_tags = {m.group(1).lower() for m in _HASH_TAG_RE.finditer(evidence_text)}
    answer_tags = {m.group(1).lower() for m in _HASH_TAG_RE.finditer(answer_text)}
    concrete_tag = bool(evidence_tags and (evidence_tags & answer_tags))

    overlap = _token_set(answer_text) & _token_set(evidence_text)
    # Ignore ultra-generic overlap alone; need a specific shared content word.
    generic = {
        "the",
        "and",
        "you",
        "your",
        "ask",
        "question",
        "questions",
        "channel",
        "discord",
        "about",
        "assignment",
        "assignments",
    }
    specific_overlap = {tok for tok in overlap if tok not in generic and len(tok) > 3}

    if concrete_tag or len(specific_overlap) >= 1:
        return "dropped_condition"
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


def _window_unit(raw: str) -> str:
    return raw.lower().rstrip("s")


def _window_count(raw: str) -> str:
    text = raw.lower().strip()
    words = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    return words.get(text, text)


def detect_time_window_preposition_mismatch(
    *,
    question: str,
    answer: str,
    evidence: str,
) -> str | None:
    """Reject flipping evidence ``within the first N …`` to ``before the first N …``.

    ``before the first two weeks`` is not a faithful paraphrase of
    ``within the first two weeks`` — it changes the deadline window.
    """
    evidence_text = _norm(evidence)
    qa_text = _norm(f"{question} {answer}")
    if not evidence_text or not qa_text:
        return None

    within = _WITHIN_FIRST_WINDOW_RE.search(evidence_text)
    if within is None:
        return None
    before = _BEFORE_FIRST_WINDOW_RE.search(qa_text)
    if before is None:
        return None

    if _window_count(within.group(1)) == _window_count(before.group(1)) and _window_unit(
        within.group(2)
    ) == _window_unit(before.group(2)):
        return "time_window_preposition_mismatch"
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

    # Full bundle (incl. source chunks) for grounding checks that need context.
    evidence = _evidence_bundle(fact, source_text)
    # Modality checks use quote-focused text so chunk/statement contamination
    # cannot mask soft "important to" / "should" evidence as already-mandatory.
    modality_evidence = modality_evidence_for_fact(fact) or evidence
    # Qualifier checks stay on this fact only — neighboring chunk sentences with
    # unrelated "if you don't…" / "one per quarter" must not false-reject.
    qualifier_evidence = fact_qualifier_evidence(fact) or evidence

    modal = detect_modal_escalation(
        question=question,
        answer=answer,
        evidence=modality_evidence,
    )
    if modal:
        return {"reason": modal, "category": "modal_escalation"}

    absolute = detect_absolute_overclaim(answer=answer, evidence=evidence)
    if absolute:
        return {"reason": absolute, "category": "modal_escalation"}

    guarantee = detect_response_time_guarantee(answer=answer, evidence=evidence)
    if guarantee:
        return {"reason": guarantee, "category": "modal_escalation"}

    # Entailment also uses modality-focused evidence so an overstated inventory
    # statement containing "must" cannot ground an escalated answer.
    entailment = statement_entailment_violation(answer, modality_evidence)
    if entailment == "obligation_not_in_evidence":
        return {"reason": "obligation_not_in_evidence", "category": "modal_escalation"}
    if entailment in {"guarantee_from_conditional", "guarantee_not_in_evidence"}:
        return {"reason": entailment, "category": "modal_escalation"}
    # Question-side obligation escalation (e.g. "do I need to") against soft evidence.
    question_entailment = statement_entailment_violation(question, modality_evidence)
    if question_entailment == "obligation_not_in_evidence":
        return {"reason": "obligation_not_in_evidence", "category": "modal_escalation"}

    dropped = detect_dropped_condition(
        question=question,
        answer=answer,
        evidence=qualifier_evidence,
    )
    if dropped:
        return {"reason": dropped, "category": "qualifier_mismatch"}

    missing = detect_missing_critical_qualifier(
        answer=answer,
        evidence=qualifier_evidence,
    )
    if missing:
        return {"reason": missing, "category": "qualifier_mismatch"}

    time_window = detect_time_window_preposition_mismatch(
        question=question,
        answer=answer,
        evidence=qualifier_evidence,
    )
    if time_window:
        return {"reason": time_window, "category": "qualifier_mismatch"}

    beyond = detect_question_beyond_evidence(question=question, evidence=evidence)
    if beyond:
        return {"reason": beyond, "category": "prevalidation"}

    return None
