"""Global fact-inventory extraction for starter seed generation (Phase 2).

Builds an independently inspectable inventory of atomic student-facing facts
from stored syllabus chunks. This is the future planning unit for seed
generation, but this module DOES NOT generate seeds and is not wired into
``generate_starter_seeds_for_course``.

Design notes:
- Chunks are treated as full-text evidence, not ~320-char digests.
- Every fact must carry an ``evidenceQuote`` that verifiably appears (after
  whitespace normalization) in one of its listed source chunks.
- Importance / usefulness use course-agnostic soft priors, never a hardcoded
  per-course taxonomy.
- On LLM failure or empty output, a deterministic heuristic fallback still
  returns a non-empty inspectable inventory when reasonable.
"""

from __future__ import annotations

import json
import re
from typing import Any

from fastapi import HTTPException

from app.ollama import (
    DEFAULT_EMBED_MODEL,
    embed_ollama_texts,
    generate_starter_ollama_completion,
    get_starter_inventory_num_predict,
)
from app.seed_similarity import cosine_similarity

SEED_GENERATION_MODEL = "qwen3:4b"

FACT_EXTRACTION_PROMPT_MARKER = "You extract atomic student-facing facts from a syllabus."

# Soft scope metadata used later for balance. NOT a rigid course taxonomy.
FACT_SCOPES = frozenset(
    {"course_wide", "assignment_specific", "schedule", "resource", "other"}
)
DEFAULT_SCOPE = "other"

# Suggested (non-exhaustive) kind labels; any non-empty string is accepted.
KIND_SUGGESTIONS = (
    "policy",
    "requirement",
    "deadline",
    "grading",
    "contact",
    "office_hours",
    "attendance",
    "late_work",
    "exam",
    "tools",
    "accommodation",
    "communication",
    "team_project",
    "resource",
    "other",
)
DEFAULT_KIND = "other"

MIN_STATEMENT_CHARS = 12
MIN_EVIDENCE_CHARS = 12
DEFAULT_MAX_FACTS = 80
DEFAULT_BATCH_CHAR_BUDGET = 6000

IMPORTANCE_VALUE = {"high": 0.9, "medium": 0.6, "low": 0.3}

# Course-agnostic soft priors. Presence boosts usefulness; it never forces a
# fixed category and never assumes a syllabus contains any of these.
_HIGH_VALUE_TERMS = (
    "late",
    "extension",
    "deadline",
    "due",
    "grade",
    "grading",
    "points",
    "percent",
    "penalty",
    "exam",
    "quiz",
    "midterm",
    "final",
    "attendance",
    "absence",
    "absent",
    "office hours",
    "contact",
    "email",
    "instructor",
    "professor",
    "teaching assistant",
    " ta ",
    "accommodation",
    "disability",
    "required",
    "must",
    "submit",
    "policy",
    "plagiarism",
    "integrity",
    "participation",
    "team",
    "project",
    "quiz",
)
_LOW_VALUE_TERMS = (
    "optional",
    "pantry",
    "newsletter",
    "follow us",
    "social media",
    "bookstore",
    "welcome to",
    "land acknowledg",
)
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_ASSIGNMENT_ID_RE = re.compile(
    r"\b(assignment|task|project|homework|hw|lab|milestone|quiz|exam)\s*#?\s*\d+",
    re.IGNORECASE,
)
_SCHEDULE_RE = re.compile(
    r"\b("
    r"week\s*\d+|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|october|"
    r"november|december|calendar|schedule|\d{1,2}/\d{1,2}"
    r")\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

# Statement-to-evidence entailment: keep statements from overstating the source.
_OBLIGATION_STATEMENT_TERMS = (
    "must",
    "required",
    "mandatory",
    "have to",
    "shall",
    "obligated",
    "you are to",
    "need to",
    "are expected to",
)
# Evidence terms that legitimately ground an obligation-style statement.
_REQUIREMENT_EVIDENCE_TERMS = _OBLIGATION_STATEMENT_TERMS + (
    "important",
    "impact",
    "penalty",
    "responsible",
    "responsibility",
    "expect",
    "expected",
    "requires",
    "requirement",
    "cannot",
    "not filing",
    "you need",
    "due",
    "forbidden",
    "prohibited",
    "not permitted",
    "no extension",
    "will result",
    "results in",
)
_OPTIONAL_STATEMENT_TERMS = ("optional", "not required", "voluntary")
# Availability/permission ("you can", "appointments available") is NOT optionality.
_OPTIONAL_EVIDENCE_TERMS = (
    "optional",
    "not required",
    "no need",
    "voluntary",
    "do not have to",
    "don't have to",
    "not mandatory",
)
_RESPONSE_WITHIN_RE = re.compile(r"within\s+\d+\s*(hour|hr|day)", re.IGNORECASE)
_RESPONSE_VERB_TERMS = ("respond", "response", "reply", "get back", "answer")
_CONDITIONAL_RESPONSE_CUES = (
    "if i do not respond",
    "if i don't respond",
    "do not respond",
    "don't respond",
    "does not respond",
    "no response",
    "nudge",
    "unintentional",
)

# Importance calibration: consequence/policy signals that justify "high".
_CONSEQUENCE_TERMS = (
    "grade",
    "grading",
    "penalty",
    "fail",
    "failure",
    "zero",
    "impact",
    "no credit",
    "deadline",
    "due",
    "extension",
    "accommodation",
    "drop",
    "required",
    "must",
    "consequence",
    "points",
    "percent",
    "misconduct",
    "dishonesty",
    "plagiar",
    "cannot",
)
_HIGH_IMPORTANCE_KINDS = frozenset(
    {"grading", "late_work", "attendance", "accommodation", "policy", "exam"}
)
_IMPORTANCE_ORDER = {"low": 0, "medium": 1, "high": 2}
_IMPORTANCE_BY_ORDER = {0: "low", 1: "medium", 2: "high"}

# Assignment-series grouping metadata (course-agnostic; number-bearing labels).
_ASSIGNMENT_SERIES_RE = re.compile(
    r"((?:bot\s+project\s+)?(?:task|project|assignment|milestone|sprint|homework|hw|lab|version))"
    r"\s*#?\s*(\d+)",
    re.IGNORECASE,
)

# Deterministic salvage for high-value late-work / extension / exception language
# that LLMs sometimes omit when a batch is crowded with other content.
_LATE_WORK_POLICY_SIGNAL_RE = re.compile(
    r"(?i)\b("
    r"(?:\d+[\s-]*(?:hour|hr|day)s?[\s-]*)?extension(?:s)?|"
    r"no\s+extensions?\b|"
    r"late\s+work|"
    r"late\s+policy|"
    r"late\s+penalty|"
    r"points?\s+per\s+day|"
    r"percent(?:age)?\s+per\s+day|"
    r"one\s+per\s+(?:quarter|semester|term|course)|"
    r"not\s+eligible\s+for\s+(?:an?\s+)?extension|"
    r"extensions?\s+(?:are\s+)?not\s+(?:allowed|possible|permitted|available)"
    r")\b"
)
# Require a stronger eligibility/exclusion/penalty cue so casual mentions of
# "extension" (e.g. groupmates accommodating an extension) are not salvaged alone.
_LATE_WORK_POLICY_STRONG_RE = re.compile(
    r"(?i)\b("
    r"\d+[\s-]*(?:hour|hr|day)s?[\s-]*extension|"
    r"one\s+(?:\w+\s+){0,4}extension|"
    r"no\s+extensions?|"
    r"not\s+eligible\s+for\s+(?:an?\s+)?extension|"
    r"extensions?\s+(?:are\s+)?not\s+(?:allowed|possible|permitted|available)|"
    r"late\s+policy|"
    r"late\s+penalty|"
    r"late\s+work|"
    r"points?\s+per\s+day|"
    r"percent(?:age)?\s+per\s+day|"
    r"one\s+per\s+(?:quarter|semester|term|course)"
    r")\b"
)
# Exclude instructor response-time language (nudge-if-no-reply), which is NOT a
# late-work extension policy.
_RESPONSE_TIME_SIGNAL_RE = re.compile(
    r"(?i)\b("
    r"respond(?:s|ed|ing)?|"
    r"response|"
    r"reply|"
    r"nudge|"
    r"inbox|"
    r"message\s+queues?"
    r")\b"
)

SEMANTIC_FACT_MERGE_THRESHOLD = 0.86


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip())


def _normalize_statement_key(statement: str) -> str:
    lowered = statement.strip().lower()
    collapsed = _NON_ALNUM_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", collapsed).strip()


def _extract_json_text(raw: str) -> str:
    text = raw.strip()
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()

    object_start = text.find("{")
    array_start = text.find("[")
    if object_start == -1 and array_start == -1:
        return text
    if object_start == -1:
        start = array_start
        end = text.rfind("]")
    elif array_start == -1:
        start = object_start
        end = text.rfind("}")
    else:
        start = min(object_start, array_start)
        end = text.rfind("]") if array_start < object_start else text.rfind("}")
    if end == -1 or end <= start:
        return text
    return text[start : end + 1]


def build_chunk_lookup(raw_chunks: list[Any]) -> dict[str, str]:
    """Map chunkId -> full chunk text (evidence source)."""
    lookup: dict[str, str] = {}
    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            continue
        chunk_id = raw_chunk.get("chunkId") or raw_chunk.get("id")
        text = raw_chunk.get("text")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        lookup[chunk_id.strip()] = text.strip()
    return lookup


def build_section_groups(raw_chunks: list[Any]) -> list[dict[str, Any]]:
    """Group consecutive chunks by section title, preserving full text."""
    groups: list[dict[str, Any]] = []
    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            continue
        chunk_id = raw_chunk.get("chunkId") or raw_chunk.get("id")
        text = raw_chunk.get("text")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        section_title = (
            raw_chunk.get("sectionTitle")
            or raw_chunk.get("section_title")
            or "General"
        )
        section_title = str(section_title).strip() or "General"

        if groups and groups[-1]["sectionTitle"] == section_title:
            groups[-1]["chunks"].append({"chunkId": chunk_id.strip(), "text": text.strip()})
        else:
            groups.append(
                {
                    "sectionTitle": section_title,
                    "chunks": [{"chunkId": chunk_id.strip(), "text": text.strip()}],
                }
            )
    return groups


def _split_group_to_budget(
    group: dict[str, Any],
    *,
    char_budget: int,
) -> list[dict[str, Any]]:
    """Split one section group into subgroups that each fit under char_budget.

    Needed when every chunk shares one sectionTitle (common for wiki-exported
    syllabi) so a single group does not become one oversized LLM prompt.
    """
    packs: list[dict[str, Any]] = []
    current_chunks: list[dict[str, Any]] = []
    current_chars = 0
    for chunk in group["chunks"]:
        chunk_chars = len(chunk["text"])
        if current_chunks and current_chars + chunk_chars > char_budget:
            packs.append(
                {
                    "sectionTitle": group["sectionTitle"],
                    "chunks": current_chunks,
                }
            )
            current_chunks = []
            current_chars = 0
        current_chunks.append(chunk)
        current_chars += chunk_chars
    if current_chunks:
        packs.append(
            {
                "sectionTitle": group["sectionTitle"],
                "chunks": current_chunks,
            }
        )
    return packs


def batch_section_groups(
    groups: list[dict[str, Any]],
    *,
    char_budget: int = DEFAULT_BATCH_CHAR_BUDGET,
) -> list[list[dict[str, Any]]]:
    """Pack section groups into batches under a character budget (fewer LLM calls).

    Oversized single groups are split into chunk packs first so syllabi with one
    shared sectionTitle still produce multiple extractable batches.
    """
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for group in groups:
        for subgroup in _split_group_to_budget(group, char_budget=char_budget):
            subgroup_chars = sum(len(chunk["text"]) for chunk in subgroup["chunks"])
            if current and current_chars + subgroup_chars > char_budget:
                batches.append(current)
                current = []
                current_chars = 0
            current.append(subgroup)
            current_chars += subgroup_chars
    if current:
        batches.append(current)
    return batches


def build_fact_extraction_prompt(batch: list[dict[str, Any]]) -> str:
    chunk_lines: list[str] = []
    for group in batch:
        for chunk in group["chunks"]:
            chunk_lines.append(
                f'[{chunk["chunkId"]}] Section: {group["sectionTitle"]}\n{chunk["text"]}'
            )
    chunk_block = "\n\n".join(chunk_lines)
    kind_hint = ", ".join(KIND_SUGGESTIONS)
    scope_hint = ", ".join(sorted(FACT_SCOPES))
    return f"""{FACT_EXTRACTION_PROMPT_MARKER}

Read the syllabus chunks and list distinct, atomic student-facing facts.

What makes a good fact:
- One concrete claim a student could ask about, e.g.
  - "One 48-hour extension is allowed for one of Bot Project Tasks 1 through 7."
  - "Students must notify the instructor at least one hour before class if absent."
  - "Grade-related discussion must happen via Canvas."
- Avoid vague topic labels like "Course Resources" or "Assignments".

Actively look for these high-value student-facing facts when the syllabus contains them
(do NOT invent them if absent):
- instructor identity and contact details, and preferred contact channels
- grading structure and weightings, and how grades map to a scale
- how to ask about or dispute a grade
- accommodations (disability and religious) and how/when to request them
- assignment submission rules (where, format, deadline time zone)
- team/project expectations and group-work rules
- consequences and procedures for missing class / absences
- help and support pathways (office hours, help channels, tutoring, crisis lines)
- late work / extension eligibility AND its limitations (what is NOT eligible)
- exams/quizzes existence and format
- required tools/software students must be able to access

High-priority policy structures (extract whenever present; do not skip when a batch
also contains grading/schedule text):
- extension eligibility and limits (duration, how many, which work they apply to)
- "one per quarter/term/semester" style caps
- explicit exclusions / "no extension" / "not eligible" rules
- late penalties (points/percent lost per day, grace periods)
- conditional rules and exceptions (general rule + exception should both appear)
When a policy has both a general allowance and exclusions, prefer TWO atomic facts
(one for the allowance/limit, one for each exclusion class) rather than dropping either.
Keep each statement close to the source wording, including conditions.

Do NOT confuse these distinct "time window" ideas:
- Instructor messaging turnaround / "if no reply in N hours, nudge me" is NOT a
  late-work extension policy (and is not a guaranteed response time).
- A late-work or assignment extension of N hours/days IS a high-value policy and
  MUST be extracted when present.

STRICT grounding and entailment rules:
- Use only facts present in the provided chunk text.
- evidenceQuote MUST be copied verbatim from one listed source chunk.
- Do NOT convert availability into a requirement (e.g. "appointments are available"
  does NOT mean office hours are required).
- Do NOT convert a suggestion, encouragement, or a list into "must"
  (e.g. listing AI tools does NOT mean students must use AI tools).
- Do NOT convert a follow-up instruction into a guaranteed policy
  (e.g. "if I do not respond within 48 hours, nudge me" does NOT mean the instructor
  guarantees a response within 48 hours).
- Only say "optional" if the source explicitly says it is optional.
- Prefer wording that stays as close as possible to the explicit source meaning.

Importance must meaningfully discriminate (do NOT mark everything high):
- high: critical course-wide policies and graded consequences (grading, late/extension
  rules, attendance impact, accommodations, academic integrity, missing exams).
- medium: important operational info (contact channels, submission logistics, required
  tools, team-process expectations, individual assignment deadlines).
- low: routine tool/resource mentions and minor or optional resources.
- Do not make all assignment deadlines automatically high; they are usually medium.

Other field rules:
- studentAskLikelihood is a number from 0 to 1.
- complexity is an integer count of conditions/exceptions in the fact (minimum 1).
- scope is one of: {scope_hint}. Use it as soft metadata only.
- kind is a short label such as: {kind_hint}.
- Return only valid JSON.

Scope guidance (examples, not a fixed taxonomy):
- late policy -> course_wide
- attendance rule -> course_wide
- instructor email -> course_wide
- Task #3 deadline -> assignment_specific
- course calendar date -> schedule
- optional campus resource mention -> resource

Return JSON in this shape:
{{
  "facts": [
    {{
      "statement": "string",
      "importance": "high|medium|low",
      "studentAskLikelihood": 0.0,
      "complexity": 1,
      "sourceChunkIds": ["chunk-001"],
      "evidenceQuote": "verbatim span from a listed chunk",
      "kind": "policy",
      "scope": "course_wide"
    }}
  ]
}}

Syllabus chunks:
{chunk_block}
"""


def _coerce_importance(value: Any) -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in IMPORTANCE_VALUE:
            return lowered
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if numeric > 1.0:
            numeric = numeric / 100.0 if numeric <= 100.0 else 1.0
        if numeric >= 0.75:
            return "high"
        if numeric >= 0.45:
            return "medium"
        return "low"
    return "medium"


def _coerce_unit_float(value: Any, *, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        text = value.strip().lower()
        mapping = {"high": 0.9, "medium": 0.6, "low": 0.3}
        if text in mapping:
            return mapping[text]
        try:
            numeric = float(text)
        except ValueError:
            return default
    else:
        return default
    if numeric > 1.0:
        numeric = numeric / 100.0 if numeric <= 100.0 else 1.0
    return max(0.0, min(1.0, numeric))


def _coerce_complexity(value: Any) -> int:
    if isinstance(value, bool):
        return 1
    if isinstance(value, (int, float)):
        return max(1, min(10, int(round(float(value)))))
    if isinstance(value, str):
        text = value.strip().lower()
        mapping = {"high": 3, "medium": 2, "low": 1}
        if text in mapping:
            return mapping[text]
        try:
            return max(1, min(10, int(round(float(text)))))
        except ValueError:
            return 1
    return 1


def _coerce_scope(value: Any) -> str:
    if isinstance(value, str):
        lowered = value.strip().lower().replace("-", "_").replace(" ", "_")
        if lowered in FACT_SCOPES:
            return lowered
    return DEFAULT_SCOPE


def _coerce_kind(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip().lower().replace(" ", "_")
    return DEFAULT_KIND


def verify_evidence_quote(
    quote: str,
    source_chunk_ids: list[str],
    chunk_lookup: dict[str, str],
) -> tuple[str, list[str]] | None:
    """Return (normalized quote, verified chunk ids) or None if unverifiable.

    Matching is whitespace-normalized and case-insensitive. Listed chunks are
    checked first; if none match, the quote is searched across all chunks and,
    when found, its source id is repaired to the matching chunk.
    """
    normalized_quote = normalize_whitespace(quote)
    if len(normalized_quote) < MIN_EVIDENCE_CHARS:
        return None

    needle = normalized_quote.lower()

    verified_ids: list[str] = []
    for chunk_id in source_chunk_ids:
        text = chunk_lookup.get(chunk_id)
        if text and needle in normalize_whitespace(text).lower():
            verified_ids.append(chunk_id)
    if verified_ids:
        return normalized_quote, verified_ids

    for chunk_id, text in chunk_lookup.items():
        if needle in normalize_whitespace(text).lower():
            return normalized_quote, [chunk_id]

    return None


def statement_entailment_violation(statement: str, evidence_quote: str) -> str | None:
    """Return a reason if the statement overstates the evidence, else None.

    Enforces conservative entailment so extracted facts stay close to the source:
    availability is not turned into a requirement, lists/suggestions are not turned
    into "must", and conditional follow-ups are not turned into guarantees.
    """
    statement_text = statement.lower()
    evidence_text = evidence_quote.lower()

    if any(term in statement_text for term in _OBLIGATION_STATEMENT_TERMS):
        if not any(term in evidence_text for term in _REQUIREMENT_EVIDENCE_TERMS):
            return "obligation_not_in_evidence"

    if any(term in statement_text for term in _OPTIONAL_STATEMENT_TERMS):
        if not any(term in evidence_text for term in _OPTIONAL_EVIDENCE_TERMS):
            return "optional_not_in_evidence"

    statement_is_response_guarantee = bool(
        _RESPONSE_WITHIN_RE.search(statement_text)
    ) and any(term in statement_text for term in _RESPONSE_VERB_TERMS)
    statement_negates_guarantee = any(
        cue in statement_text
        for cue in (
            "not guarantee",
            "no guarantee",
            "not guaranteed",
            "does not respond",
            "doesn't respond",
            "do not respond",
            "don't respond",
            "cannot guarantee",
            "won't guarantee",
            "may not respond",
            "might not respond",
        )
    )
    if statement_is_response_guarantee and not statement_negates_guarantee:
        if any(cue in evidence_text for cue in _CONDITIONAL_RESPONSE_CUES):
            return "guarantee_from_conditional"
        evidence_promises_response = bool(
            _RESPONSE_WITHIN_RE.search(evidence_text)
        ) and any(term in evidence_text for term in _RESPONSE_VERB_TERMS)
        if not evidence_promises_response:
            return "guarantee_not_in_evidence"

    return None


def _min_importance(left: str, right: str) -> str:
    order = min(_IMPORTANCE_ORDER.get(left, 1), _IMPORTANCE_ORDER.get(right, 1))
    return _IMPORTANCE_BY_ORDER[order]


def calibrate_importance(fact: dict[str, Any]) -> str:
    """Recompute importance so it discriminates and does not over-rate routine info."""
    importance = fact.get("importance", "medium")
    kind = fact.get("kind", DEFAULT_KIND)
    scope = fact.get("scope", DEFAULT_SCOPE)
    text = f"{fact.get('statement', '')} {fact.get('evidenceQuote', '')}".lower()
    has_consequence = any(term in text for term in _CONSEQUENCE_TERMS)

    # Extension eligibility/exclusions and late penalties are critical policies.
    if _is_strong_late_work_policy_text(text) and not _is_response_time_text(text):
        return "high"

    # Routine resources: low unless they carry a real consequence.
    if scope == "resource" or kind == "resource":
        return "low" if not has_consequence else "medium"

    # Routine tool/software mentions are at most medium.
    if kind == "tools":
        return _min_importance(importance, "medium")

    # "high" must be justified by a policy kind or a consequence signal.
    if importance == "high" and kind not in _HIGH_IMPORTANCE_KINDS and not has_consequence:
        importance = "medium"

    # Individual assignment deadlines are useful but usually not top priority.
    if scope == "assignment_specific" and kind in {"deadline"} and importance == "high":
        importance = "medium"

    return importance


def _is_late_work_policy_text(text: str) -> bool:
    return bool(_LATE_WORK_POLICY_SIGNAL_RE.search(text))


def _is_strong_late_work_policy_text(text: str) -> bool:
    return bool(_LATE_WORK_POLICY_STRONG_RE.search(text))


def _is_response_time_text(text: str) -> bool:
    """True when 'N hours' language is about messaging turnaround, not late work."""
    lowered = text.lower()
    if "extension" in lowered or "late work" in lowered or "late policy" in lowered:
        return False
    return bool(_RESPONSE_TIME_SIGNAL_RE.search(lowered)) and bool(
        _RESPONSE_WITHIN_RE.search(lowered)
        or re.search(r"\b\d+\s*(hour|hr|day)s?\b", lowered)
    )


def _token_set(text: str) -> set[str]:
    return {tok for tok in _normalize_statement_key(text).split() if tok}


def _policy_already_covered(
    facts: list[dict[str, Any]],
    *,
    statement: str,
    evidence_quote: str,
) -> bool:
    """Return True when an existing fact already represents this policy span."""
    statement_key = _normalize_statement_key(statement)
    evidence_key = _normalize_statement_key(evidence_quote)
    statement_tokens = _token_set(statement)
    evidence_tokens = _token_set(evidence_quote)

    for fact in facts:
        existing_statement = _normalize_statement_key(fact.get("statement", ""))
        existing_evidence = _normalize_statement_key(fact.get("evidenceQuote", ""))
        if statement_key and (
            statement_key == existing_statement
            or statement_key in existing_statement
            or existing_statement in statement_key
        ):
            return True
        if evidence_key and (
            evidence_key == existing_evidence
            or evidence_key in existing_evidence
            or existing_evidence in evidence_key
        ):
            return True
        # High token overlap with an existing late-work fact => same policy.
        for existing_text in (existing_statement, existing_evidence):
            existing_tokens = _token_set(existing_text)
            for candidate_tokens in (statement_tokens, evidence_tokens):
                if not candidate_tokens or not existing_tokens:
                    continue
                overlap = len(candidate_tokens & existing_tokens) / len(
                    candidate_tokens | existing_tokens
                )
                if overlap >= 0.55 and (
                    "extension" in candidate_tokens or "late" in candidate_tokens
                ):
                    return True
    return False


def _complexity_from_policy_text(text: str) -> int:
    lowered = text.lower()
    complexity = 1
    for cue in (
        "unless",
        "except",
        "among",
        "one per",
        "no extension",
        "not eligible",
        "only",
        "provided that",
        "as long as",
        "if ",
    ):
        if cue in lowered:
            complexity += 1
    return min(complexity, 5)


def salvage_late_work_policy_facts(
    raw_chunks: list[Any],
    *,
    existing_facts: list[dict[str, Any]],
    chunk_lookup: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Deterministically recover late-work / extension / exclusion policy facts.

    Course-agnostic: triggers on policy language patterns (extension eligibility,
    exclusions, late penalties, conditional caps), never on course-specific names.
    Only adds facts whose evidence verifies against source chunks and that are not
    already covered by LLM-extracted facts.
    """
    lookup = chunk_lookup or build_chunk_lookup(raw_chunks)
    salvaged: list[dict[str, Any]] = []
    covered_pool = list(existing_facts)

    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            continue
        chunk_id = raw_chunk.get("chunkId") or raw_chunk.get("id")
        text = raw_chunk.get("text")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        chunk_id = chunk_id.strip()

        for sentence in _iter_sentences(text):
            if len(sentence) < MIN_STATEMENT_CHARS:
                continue
            if not _is_late_work_policy_text(sentence):
                continue
            if not _is_strong_late_work_policy_text(sentence):
                continue
            if _is_response_time_text(sentence):
                continue
            if _is_boilerplate(sentence):
                continue

            verified = verify_evidence_quote(sentence, [chunk_id], lookup)
            if verified is None:
                continue
            evidence_quote, verified_ids = verified
            statement = normalize_whitespace(sentence)
            if statement_entailment_violation(statement, evidence_quote) is not None:
                continue
            if _policy_already_covered(
                covered_pool, statement=statement, evidence_quote=evidence_quote
            ):
                continue

            fact = {
                "statement": statement,
                "importance": "high",
                "importanceScore": IMPORTANCE_VALUE["high"],
                "studentAskLikelihood": 0.9,
                "complexity": _complexity_from_policy_text(statement),
                "sourceChunkIds": verified_ids,
                "evidenceQuote": evidence_quote,
                "kind": "late_work",
                "scope": "course_wide",
            }
            salvaged.append(fact)
            covered_pool.append(fact)

    return salvaged


def detect_assignment_series(text: str) -> dict[str, Any] | None:
    """Detect a number-bearing assignment-series reference (e.g., 'Bot Project Task #3').

    Returns grouping metadata (seriesKey, assignmentGroup, seriesOrdinal) or None.
    Course-agnostic: keys are derived from the matched label, not hardcoded.
    """
    match = _ASSIGNMENT_SERIES_RE.search(text)
    if match is None:
        return None
    label = normalize_whitespace(match.group(1))
    ordinal = int(match.group(2))
    series_key = _NON_ALNUM_RE.sub("_", label.lower()).strip("_")
    if not series_key:
        return None
    return {
        "seriesKey": series_key,
        "assignmentGroup": label.title(),
        "seriesOrdinal": ordinal,
    }


async def semantic_merge_facts(
    facts: list[dict[str, Any]],
    *,
    embed_fn=None,
    model: str = DEFAULT_EMBED_MODEL,
    threshold: float = SEMANTIC_FACT_MERGE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Collapse semantically near-duplicate facts. Fail-open on embedding errors.

    Higher-importance facts are kept as the primary; near-duplicates are absorbed
    (union of source chunks, strongest importance/scores, longest evidence).
    """
    if len(facts) < 2:
        return facts
    if embed_fn is None:
        embed_fn = embed_ollama_texts

    try:
        result = await embed_fn([fact["statement"] for fact in facts], model=model)
    except HTTPException:
        return facts

    raw_embeddings = result.get("embeddings", [])
    if not isinstance(raw_embeddings, list) or len(raw_embeddings) != len(facts):
        return facts

    indexed = list(enumerate(facts))
    indexed.sort(
        key=lambda pair: (
            -IMPORTANCE_VALUE.get(pair[1].get("importance", "medium"), 0.6),
            -pair[1].get("studentAskLikelihood", 0.5),
            -len(pair[1]["statement"]),
        )
    )

    kept: list[dict[str, Any]] = []
    kept_embeddings: list[list[float]] = []
    for original_index, fact in indexed:
        embedding = raw_embeddings[original_index]
        if not isinstance(embedding, list):
            kept.append(fact)
            kept_embeddings.append([])
            continue
        merged_into: int | None = None
        for kept_index, kept_embedding in enumerate(kept_embeddings):
            if not kept_embedding:
                continue
            if cosine_similarity(embedding, kept_embedding) >= threshold:
                merged_into = kept_index
                break
        if merged_into is not None:
            _absorb_fact(kept[merged_into], fact)
        else:
            kept.append(fact)
            kept_embeddings.append(embedding)
    return kept


def normalize_fact(
    raw_fact: Any,
    *,
    chunk_lookup: dict[str, str],
) -> dict[str, Any] | None:
    if not isinstance(raw_fact, dict):
        return None

    statement = str(raw_fact.get("statement", "")).strip()
    if len(statement) < MIN_STATEMENT_CHARS:
        return None

    evidence_raw = raw_fact.get("evidenceQuote") or raw_fact.get("evidence") or ""
    source_raw = raw_fact.get("sourceChunkIds")
    if isinstance(source_raw, list):
        source_chunk_ids = [str(item).strip() for item in source_raw if str(item).strip()]
    elif isinstance(source_raw, str) and source_raw.strip():
        source_chunk_ids = [source_raw.strip()]
    else:
        source_chunk_ids = []

    verified = verify_evidence_quote(str(evidence_raw), source_chunk_ids, chunk_lookup)
    if verified is None:
        return None
    evidence_quote, verified_ids = verified

    normalized_statement = normalize_whitespace(statement)
    if statement_entailment_violation(normalized_statement, evidence_quote) is not None:
        return None

    importance = _coerce_importance(raw_fact.get("importance"))
    return {
        "statement": normalize_whitespace(statement),
        "importance": importance,
        "importanceScore": IMPORTANCE_VALUE[importance],
        "studentAskLikelihood": _coerce_unit_float(
            raw_fact.get("studentAskLikelihood"), default=0.5
        ),
        "complexity": _coerce_complexity(raw_fact.get("complexity")),
        "sourceChunkIds": verified_ids,
        "evidenceQuote": evidence_quote,
        "kind": _coerce_kind(raw_fact.get("kind")),
        "scope": _coerce_scope(raw_fact.get("scope")),
    }


def _priority_signal_adjustment(fact: dict[str, Any]) -> float:
    text = f"{fact.get('statement', '')} {fact.get('evidenceQuote', '')}".lower()
    adjustment = 0.0
    if any(term in text for term in _HIGH_VALUE_TERMS):
        adjustment += 0.05
    if any(term in text for term in _LOW_VALUE_TERMS) or _URL_RE.search(text):
        adjustment -= 0.05
    return adjustment


def compute_usefulness_score(fact: dict[str, Any]) -> float:
    importance_value = IMPORTANCE_VALUE.get(fact.get("importance", "medium"), 0.6)
    ask = float(fact.get("studentAskLikelihood", 0.5))
    complexity = int(fact.get("complexity", 1))
    complexity_bonus = min(0.10, 0.03 * (complexity - 1))

    score = 0.55 * importance_value + 0.35 * ask + complexity_bonus
    score += _priority_signal_adjustment(fact)
    if fact.get("scope") == "resource":
        score -= 0.15

    return round(max(0.0, min(1.0, score)), 4)


def _absorb_fact(primary: dict[str, Any], other: dict[str, Any]) -> None:
    primary["sourceChunkIds"] = sorted(
        set(primary["sourceChunkIds"]) | set(other["sourceChunkIds"])
    )
    if IMPORTANCE_VALUE[other["importance"]] > IMPORTANCE_VALUE[primary["importance"]]:
        primary["importance"] = other["importance"]
        primary["importanceScore"] = other["importanceScore"]
        primary["kind"] = other["kind"]
        primary["scope"] = other["scope"]
    primary["studentAskLikelihood"] = max(
        primary["studentAskLikelihood"], other["studentAskLikelihood"]
    )
    primary["complexity"] = max(primary["complexity"], other["complexity"])
    if len(other["evidenceQuote"]) > len(primary["evidenceQuote"]):
        primary["evidenceQuote"] = other["evidenceQuote"]


def merge_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge exact-duplicate statements and contained statements with source overlap."""
    ordered = sorted(facts, key=lambda item: len(item["statement"]), reverse=True)
    merged: list[dict[str, Any]] = []
    for fact in ordered:
        key = _normalize_statement_key(fact["statement"])
        if not key:
            continue
        duplicate_of: dict[str, Any] | None = None
        for existing in merged:
            existing_key = _normalize_statement_key(existing["statement"])
            if key == existing_key:
                duplicate_of = existing
                break
            source_overlap = set(existing["sourceChunkIds"]) & set(fact["sourceChunkIds"])
            if source_overlap and (key in existing_key or existing_key in key):
                duplicate_of = existing
                break
        if duplicate_of is not None:
            _absorb_fact(duplicate_of, fact)
        else:
            merged.append(fact)
    return merged


def _iter_sentences(text: str) -> list[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(normalized) if part.strip()]


def _is_boilerplate(sentence: str) -> bool:
    lowered = sentence.lower()
    if _URL_RE.search(lowered) and len(lowered) < 60:
        return True
    if any(term in lowered for term in ("welcome to", "land acknowledg", "follow us")):
        return True
    if len(sentence) < 25 and not any(term in lowered for term in _HIGH_VALUE_TERMS):
        return True
    return False


def _classify_scope_heuristic(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in _LOW_VALUE_TERMS):
        return "resource"
    if _ASSIGNMENT_ID_RE.search(lowered):
        return "assignment_specific"
    if _SCHEDULE_RE.search(lowered):
        return "schedule"
    if any(term in lowered for term in _HIGH_VALUE_TERMS):
        return "course_wide"
    return DEFAULT_SCOPE


def _classify_kind_heuristic(text: str) -> str:
    lowered = text.lower()
    rules = (
        (("late", "extension"), "late_work"),
        (("attendance", "absence", "absent"), "attendance"),
        (("grade", "grading", "points", "percent", "penalty"), "grading"),
        (("exam", "quiz", "midterm", "final"), "exam"),
        (("office hours",), "office_hours"),
        (("email", "contact", "instructor", "professor", "teaching assistant"), "contact"),
        (("accommodation", "disability"), "accommodation"),
        (("discord", "canvas", "slack", "software", "tool"), "tools"),
        (("team", "project", "group"), "team_project"),
        (("plagiarism", "integrity"), "policy"),
    )
    for terms, kind in rules:
        if any(term in lowered for term in terms):
            return kind
    return DEFAULT_KIND


def _importance_heuristic(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in ("must", "required", "deadline", "due", "penalty")):
        return "high"
    if any(term in lowered for term in _HIGH_VALUE_TERMS):
        return "medium"
    if any(term in lowered for term in _LOW_VALUE_TERMS):
        return "low"
    return "medium"


def build_heuristic_fallback_inventory(
    raw_chunks: list[Any],
    *,
    chunk_lookup: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Deterministic fact extraction from substantive sentences (no LLM)."""
    lookup = chunk_lookup or build_chunk_lookup(raw_chunks)
    facts: list[dict[str, Any]] = []

    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            continue
        chunk_id = raw_chunk.get("chunkId") or raw_chunk.get("id")
        text = raw_chunk.get("text")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        chunk_id = chunk_id.strip()

        sentences = _iter_sentences(text)
        substantive = [s for s in sentences if not _is_boilerplate(s)]
        high_value = [
            s for s in substantive if any(term in s.lower() for term in _HIGH_VALUE_TERMS)
        ]
        selected = high_value or substantive[:1]

        for sentence in selected:
            if len(sentence) < MIN_STATEMENT_CHARS:
                continue
            verified = verify_evidence_quote(sentence, [chunk_id], lookup)
            if verified is None:
                continue
            evidence_quote, verified_ids = verified
            importance = _importance_heuristic(sentence)
            facts.append(
                {
                    "statement": normalize_whitespace(sentence),
                    "importance": importance,
                    "importanceScore": IMPORTANCE_VALUE[importance],
                    "studentAskLikelihood": _coerce_unit_float(
                        importance, default=0.5
                    ),
                    "complexity": 1,
                    "sourceChunkIds": verified_ids,
                    "evidenceQuote": evidence_quote,
                    "kind": _classify_kind_heuristic(sentence),
                    "scope": _classify_scope_heuristic(sentence),
                }
            )

    return facts


def _finalize_inventory(
    facts: list[dict[str, Any]],
    *,
    model: str,
    fallback_used: bool,
    dropped_count: int,
    max_facts: int,
    duplicates_removed: int = 0,
) -> dict[str, Any]:
    for fact in facts:
        calibrated = calibrate_importance(fact)
        fact["importance"] = calibrated
        fact["importanceScore"] = IMPORTANCE_VALUE[calibrated]
        series = detect_assignment_series(
            f"{fact.get('statement', '')} {fact.get('evidenceQuote', '')}"
        )
        fact["seriesKey"] = series["seriesKey"] if series else None
        fact["assignmentGroup"] = series["assignmentGroup"] if series else None
        fact["seriesOrdinal"] = series["seriesOrdinal"] if series else None
        fact["usefulnessScore"] = compute_usefulness_score(fact)

    facts.sort(key=lambda item: (-item["usefulnessScore"], item["statement"].lower()))
    if max_facts > 0 and len(facts) > max_facts:
        facts = facts[:max_facts]

    for index, fact in enumerate(facts):
        fact["factId"] = f"fact-{index + 1:02d}"

    counts_by_scope: dict[str, int] = {}
    counts_by_kind: dict[str, int] = {}
    counts_by_series: dict[str, int] = {}
    for fact in facts:
        counts_by_scope[fact["scope"]] = counts_by_scope.get(fact["scope"], 0) + 1
        counts_by_kind[fact["kind"]] = counts_by_kind.get(fact["kind"], 0) + 1
        if fact.get("seriesKey"):
            counts_by_series[fact["seriesKey"]] = (
                counts_by_series.get(fact["seriesKey"], 0) + 1
            )

    ordered_fact_fields = [
        {
            "factId": fact["factId"],
            "statement": fact["statement"],
            "importance": fact["importance"],
            "importanceScore": fact["importanceScore"],
            "studentAskLikelihood": fact["studentAskLikelihood"],
            "complexity": fact["complexity"],
            "usefulnessScore": fact["usefulnessScore"],
            "sourceChunkIds": fact["sourceChunkIds"],
            "evidenceQuote": fact["evidenceQuote"],
            "kind": fact["kind"],
            "scope": fact["scope"],
            "seriesKey": fact.get("seriesKey"),
            "assignmentGroup": fact.get("assignmentGroup"),
            "seriesOrdinal": fact.get("seriesOrdinal"),
        }
        for fact in facts
    ]

    return {
        "model": model,
        "facts": ordered_fact_fields,
        "factCount": len(ordered_fact_fields),
        "droppedCount": dropped_count,
        "duplicatesRemoved": duplicates_removed,
        "fallbackUsed": fallback_used,
        "countsByScope": counts_by_scope,
        "countsByKind": counts_by_kind,
        "countsBySeries": counts_by_series,
    }


def _parse_facts_payload(raw: str) -> list[Any]:
    parsed = json.loads(_extract_json_text(raw))
    if isinstance(parsed, dict):
        facts = parsed.get("facts")
    elif isinstance(parsed, list):
        facts = parsed
    else:
        facts = None
    if not isinstance(facts, list):
        return []
    return facts


async def _extract_facts_for_batch(
    *,
    batch: list[dict[str, Any]],
    chunk_lookup: dict[str, str],
    completion_fn,
    model: str,
) -> tuple[list[dict[str, Any]], int]:
    """Return (normalized facts, dropped_count) for one batch. Never raises."""
    prompt = build_fact_extraction_prompt(batch)
    try:
        generation = await completion_fn(
            prompt,
            model=model,
            response_format="json",
            think=False,
            num_predict=get_starter_inventory_num_predict(),
            stage="inventory",
        )
    except HTTPException:
        return [], 0

    try:
        raw_facts = _parse_facts_payload(generation.get("answer", ""))
    except json.JSONDecodeError:
        return [], 0

    facts: list[dict[str, Any]] = []
    dropped = 0
    for raw_fact in raw_facts:
        normalized = normalize_fact(raw_fact, chunk_lookup=chunk_lookup)
        if normalized is None:
            dropped += 1
            continue
        facts.append(normalized)
    return facts, dropped


async def _merge_and_finalize(
    facts: list[dict[str, Any]],
    *,
    model: str,
    fallback_used: bool,
    dropped_count: int,
    max_facts: int,
    embed_fn,
    embed_model: str,
    semantic_threshold: float,
) -> dict[str, Any]:
    pre_merge_count = len(facts)
    merged = merge_facts(facts)
    merged = await semantic_merge_facts(
        merged,
        embed_fn=embed_fn,
        model=embed_model,
        threshold=semantic_threshold,
    )
    duplicates_removed = max(0, pre_merge_count - len(merged))
    return _finalize_inventory(
        merged,
        model=model,
        fallback_used=fallback_used,
        dropped_count=dropped_count,
        max_facts=max_facts,
        duplicates_removed=duplicates_removed,
    )


async def build_fact_inventory(
    *,
    raw_chunks: list[Any],
    completion_fn=None,
    model: str = SEED_GENERATION_MODEL,
    max_facts: int = DEFAULT_MAX_FACTS,
    batch_char_budget: int = DEFAULT_BATCH_CHAR_BUDGET,
    embed_fn=None,
    embed_model: str = DEFAULT_EMBED_MODEL,
    semantic_threshold: float = SEMANTIC_FACT_MERGE_THRESHOLD,
) -> dict[str, Any]:
    """Build the global fact inventory for a course. Does NOT generate seeds.

    Returns a dict with facts[], factCount, countsByScope, countsByKind,
    countsBySeries, droppedCount, duplicatesRemoved, fallbackUsed, and model.
    Deduplicates facts exactly, by containment, and semantically (via embeddings,
    fail-open). Falls back to deterministic heuristic extraction when the LLM fails
    or produces no verifiable facts.
    """
    if completion_fn is None:
        completion_fn = generate_starter_ollama_completion

    chunk_lookup = build_chunk_lookup(raw_chunks)
    if not chunk_lookup:
        return _finalize_inventory(
            [],
            model=model,
            fallback_used=False,
            dropped_count=0,
            max_facts=max_facts,
        )

    groups = build_section_groups(raw_chunks)
    batches = batch_section_groups(groups, char_budget=batch_char_budget)

    all_facts: list[dict[str, Any]] = []
    dropped_count = 0
    for batch in batches:
        facts, dropped = await _extract_facts_for_batch(
            batch=batch,
            chunk_lookup=chunk_lookup,
            completion_fn=completion_fn,
            model=model,
        )
        all_facts.extend(facts)
        dropped_count += dropped

    if all_facts:
        # Deterministic salvage recovers late-work / extension / exclusion policies
        # the LLM sometimes omits when a batch is crowded with other content.
        all_facts.extend(
            salvage_late_work_policy_facts(
                raw_chunks,
                existing_facts=all_facts,
                chunk_lookup=chunk_lookup,
            )
        )
        return await _merge_and_finalize(
            all_facts,
            model=model,
            fallback_used=False,
            dropped_count=dropped_count,
            max_facts=max_facts,
            embed_fn=embed_fn,
            embed_model=embed_model,
            semantic_threshold=semantic_threshold,
        )

    fallback_facts = build_heuristic_fallback_inventory(
        raw_chunks, chunk_lookup=chunk_lookup
    )
    fallback_facts.extend(
        salvage_late_work_policy_facts(
            raw_chunks,
            existing_facts=fallback_facts,
            chunk_lookup=chunk_lookup,
        )
    )
    return await _merge_and_finalize(
        fallback_facts,
        model=model,
        fallback_used=True,
        dropped_count=dropped_count,
        max_facts=max_facts,
        embed_fn=embed_fn,
        embed_model=embed_model,
        semantic_threshold=semantic_threshold,
    )
