"""Phase 6 batching helpers for fact-scoped generation and rubric validation.

NOTE: Live starter generation uses the Phase 5 sequential path. These helpers
remain available for experiments/unit tests but are not wired into
generate_starter_seeds_for_course.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.seed_validation import (
    VALIDATION_PROMPT_MARKER,
    VALIDATION_WEIGHTS,
    try_parse_validation_payload,
)

# Conservative sizes for qwen3:4b JSON reliability on local Ollama.
GENERATION_BATCH_SIZE = 3
VALIDATION_BATCH_SIZE = 3

BATCH_GENERATION_PROMPT_MARKER = "Generate starter seeds for multiple syllabus facts."
BATCH_VALIDATION_PROMPT_MARKER = (
    f"{VALIDATION_PROMPT_MARKER}\n"
    "You validate multiple syllabus seed examples in one response."
)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def _extract_json_text(raw: str) -> str:
    text = raw.strip()
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def chunk_items(items: list[Any], size: int) -> list[list[Any]]:
    if size < 1:
        size = 1
    return [items[index : index + size] for index in range(0, len(items), size)]


def build_batch_fact_seed_prompt(
    *,
    items: list[dict[str, Any]],
) -> str:
    """Build a multi-fact generation prompt.

    Each item must include: fact, chunk_texts, suggested_styles (optional).
    Requests exactly one Q/A per fact.
    """
    fact_blocks: list[str] = []
    for index, item in enumerate(items):
        fact = item["fact"]
        fact_id = str(fact.get("factId") or "").strip()
        statement = str(fact.get("statement") or "").strip()
        evidence_quote = str(fact.get("evidenceQuote") or "").strip()
        source_chunk_ids = ", ".join(
            str(chunk_id).strip()
            for chunk_id in (fact.get("sourceChunkIds") or [])
            if str(chunk_id).strip()
        )
        chunk_block = "\n\n".join(item.get("chunk_texts") or [])
        fact_blocks.append(
            f"""### Fact {index + 1}
factId: {fact_id}
statement: {statement}
evidenceQuote: {evidence_quote}
sourceChunkIds: {source_chunk_ids}
source chunk text:
{chunk_block}
"""
        )

    joined = "\n".join(fact_blocks)
    return f"""{BATCH_GENERATION_PROMPT_MARKER}

Generate exactly ONE student question/answer seed for EACH fact below.
Return one seed per fact. Do not invent extra seeds.

Grounding rules (strict) — apply independently to each fact:
- The question must be fully answerable from ONLY that fact's statement, evidence
  quote, and source chunk text.
- The answer must stay within that evidence. Do not broaden or add details.
- Preserve qualifiers, limits, exceptions, and conditions exactly
  (e.g. "one", "per quarter", "except", "unless", "may", "optional").
- Do NOT convert permission into obligation ("may"/"can" must NOT become "must").
- Do NOT convert recommendations into requirements.
- Do NOT turn response-time guidance into a reply-time guarantee.
- Do NOT invent consequences, penalties, grades, or outcomes not in the evidence.
- Each seed MUST include the factId of the fact it answers.
- Return ONLY valid JSON (no markdown, no commentary).

Required JSON shape:
{{
  "seeds": [
    {{
      "factId": "fact-01",
      "question": "string",
      "answer": "string",
      "category": "string",
      "questionType": "direct|scenario|clarification|procedure|comparison",
      "sourceChunkIds": ["chunk-001"]
    }}
  ]
}}

Facts:
{joined}
"""


def _recover_seed_dicts(raw: str) -> list[dict[str, Any]]:
    """Best-effort recovery of seed objects when full JSON parse fails."""
    text = _extract_json_text(raw)
    recovered: list[dict[str, Any]] = []
    for match in _OBJECT_RE.finditer(text):
        snippet = match.group(0)
        if '"question"' not in snippet or '"answer"' not in snippet:
            continue
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed.get("question") and parsed.get("answer"):
            recovered.append(parsed)
    return recovered


def parse_batch_seed_payload(
    raw: str,
    *,
    expected_fact_ids: list[str],
) -> list[dict[str, Any]]:
    """Parse batch generation JSON and map seeds to expected factIds.

    Recovers valid seed objects when possible. Seeds without a matching factId
    are assigned positionally to remaining expected facts. Extra seeds are
    dropped. Missing facts simply have no seed in the returned list.
    """
    seeds_raw: list[Any] = []
    try:
        parsed = json.loads(_extract_json_text(raw))
        if isinstance(parsed, dict):
            maybe_seeds = parsed.get("seeds")
            if isinstance(maybe_seeds, list):
                seeds_raw = maybe_seeds
        elif isinstance(parsed, list):
            seeds_raw = parsed
    except json.JSONDecodeError:
        seeds_raw = _recover_seed_dicts(raw)

    if not seeds_raw and raw.strip():
        seeds_raw = _recover_seed_dicts(raw)

    by_fact: dict[str, dict[str, Any]] = {}
    unassigned: list[dict[str, Any]] = []
    expected_set = {fact_id for fact_id in expected_fact_ids if fact_id}

    for item in seeds_raw:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if not question or not answer:
            continue
        fact_id = str(item.get("factId") or "").strip()
        if fact_id and fact_id in expected_set and fact_id not in by_fact:
            by_fact[fact_id] = item
        else:
            unassigned.append(item)

    ordered: list[dict[str, Any]] = []
    for fact_id in expected_fact_ids:
        if fact_id in by_fact:
            seed = dict(by_fact[fact_id])
            seed["factId"] = fact_id
            ordered.append(seed)
            continue
        if not unassigned:
            continue
        seed = dict(unassigned.pop(0))
        seed["factId"] = fact_id
        ordered.append(seed)

    return ordered


def build_batch_validation_prompt(
    *,
    candidates: list[dict[str, Any]],
) -> str:
    """Build a multi-candidate validation prompt.

    Each candidate dict needs: candidateId, question, answer, topic_name,
    question_type, chunk_text.
    """
    blocks: list[str] = []
    for item in candidates:
        candidate_id = str(item.get("candidateId") or "").strip()
        blocks.append(
            f"""### Candidate id: {candidate_id}
Topic/category: {item.get("topic_name") or "General"}
Question type: {item.get("question_type") or "direct"}

Question:
{item.get("question") or ""}

Answer:
{item.get("answer") or ""}

Source chunk text:
{item.get("chunk_text") or ""}
"""
        )

    joined = "\n".join(blocks)
    expected_ids = ", ".join(str(item.get("candidateId") or "") for item in candidates)
    return f"""{BATCH_VALIDATION_PROMPT_MARKER}

Evaluate each candidate independently with the same conservative rubric used for
single-seed validation. Most acceptable examples score around 0.75–0.92 overall
via components. Do not assign 1.0 unless every claim is explicitly supported.

For grounded and correct on EACH candidate:
- Evaluate every factual or procedural claim in that answer.
- List unsupported claims for that candidate only.
- If any important unsupported claim remains, unsupportedClaims must be non-empty.

Return only valid JSON. Do NOT include an overall score field.
{{
  "results": [
    {{
      "candidateId": "c0",
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
  ]
}}

Include exactly one results item per candidate id ({expected_ids}).

Candidates:
{joined}
"""


def parse_batch_validation_payload(
    raw: str,
    *,
    expected_candidate_ids: list[str],
) -> dict[str, dict[str, Any] | None] | None:
    """Map candidateId → parsed rubric result (or None if that item is bad).

    Returns None when the whole payload is unusable (caller should split/retry).
    Returns a dict when at least the envelope is usable; individual malformed
    items become None without corrupting siblings.
    """
    try:
        parsed = json.loads(_extract_json_text(raw))
    except json.JSONDecodeError:
        # Recover individual result objects when possible.
        recovered = _recover_validation_items(raw)
        if not recovered:
            return None
        return _map_validation_items(recovered, expected_candidate_ids)

    if isinstance(parsed, dict):
        results = parsed.get("results")
        if results is None and all(
            key in parsed for key in ("grounded", "correct", "reason")
        ):
            # Single-rubric shape: only valid when exactly one expected id.
            if len(expected_candidate_ids) == 1:
                single = try_parse_validation_payload(raw)
                return {expected_candidate_ids[0]: single}
            return None
        if not isinstance(results, list):
            return None
    elif isinstance(parsed, list):
        results = parsed
    else:
        return None

    if not results:
        return None

    return _map_validation_items(results, expected_candidate_ids)


def _recover_validation_items(raw: str) -> list[Any]:
    text = _extract_json_text(raw)
    recovered: list[Any] = []
    for match in _OBJECT_RE.finditer(text):
        snippet = match.group(0)
        if '"reason"' not in snippet:
            continue
        if '"grounded"' not in snippet and '"candidateId"' not in snippet:
            continue
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            recovered.append(parsed)
    return recovered


def _map_validation_items(
    results: list[Any],
    expected_candidate_ids: list[str],
) -> dict[str, dict[str, Any] | None]:
    mapped: dict[str, dict[str, Any] | None] = {
        candidate_id: None for candidate_id in expected_candidate_ids
    }
    unused: list[dict[str, Any] | None] = []

    for item in results:
        if not isinstance(item, dict):
            unused.append(None)
            continue
        candidate_id = str(item.get("candidateId") or "").strip()
        # Re-serialize item without candidateId for the single-item parser.
        rubric_payload = {
            key: item.get(key) for key in (*VALIDATION_WEIGHTS.keys(), "unsupportedClaims", "reason")
        }
        parsed = try_parse_validation_payload(json.dumps(rubric_payload))
        if candidate_id and candidate_id in mapped and mapped[candidate_id] is None:
            mapped[candidate_id] = parsed
        else:
            unused.append(parsed)

    for candidate_id in expected_candidate_ids:
        if mapped[candidate_id] is not None:
            continue
        if not unused:
            break
        mapped[candidate_id] = unused.pop(0)

    # Usable if any expected id got a parsed result OR we at least recognized the
    # envelope (all None still means "parsed but all bad" — treat as usable so
    # callers do not re-split forever on consistently bad items).
    return mapped
