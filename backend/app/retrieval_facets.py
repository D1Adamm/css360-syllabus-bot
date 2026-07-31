"""Deterministic multi-facet extraction for long syllabus questions.

Splits a student question into short retrieval facets without calling an LLM
and without hardcoding course-specific policies.
"""

from __future__ import annotations

import re

MAX_FACETS = 6
MIN_FACET_CHARS = 6
MIN_FACET_CONTENT_TOKENS = 1
# Per-facet candidate pool size (small; merged with the full-question pool).
FACET_CANDIDATE_POOL = 3

_TOKEN_PATTERN = re.compile(r"[a-z0-9]{3,}")

# Prefer structural clause separators over every "and"/"or" so short noun
# phrases like "extensions and late work" stay intact as one retrieval facet.
_SPLIT_PATTERN = re.compile(
    r"[;,?]|"
    r"\bwhether\b|"
    r"\bwhat happens\b|"
    r"\bwhat is\b|"
    r"\bwhat are\b|"
    r"\bhow (?:do|does|should|can|will)\b|"
    r"\bwhen\b|"
    r"\bcan i\b|"
    r"\bshould i\b",
    re.IGNORECASE,
)

# Split on "and"/"or" only when both sides look like clause-sized phrases.
_AND_OR_PATTERN = re.compile(r"\s+(and|or)\s+", re.IGNORECASE)

_LEADING_FILLER = re.compile(
    r"^(?:please|also|then|about|regarding|concerning|based only on the syllabus|"
    r"based on the syllabus|in the syllabus|for this course|for the course|"
    r"explain|describe|tell me|ask about|including|including whether|"
    r"which|what|how|why|with|for|from|into)\s+",
    re.IGNORECASE,
)

_GENERIC_FACET_PHRASES = frozenset(
    {
        "the syllabus",
        "this course",
        "the course",
        "syllabus",
        "course",
        "please",
        "explain",
        "tell me",
        "based only",
        "based on",
        "information",
        "details",
        "policy",
        "policies",
        "rules",
        "question",
        "questions",
    }
)

_GENERIC_TOKENS = frozenset(
    {
        "the",
        "and",
        "for",
        "this",
        "that",
        "with",
        "from",
        "about",
        "only",
        "based",
        "syllabus",
        "course",
        "please",
        "explain",
        "describe",
        "tell",
        "what",
        "how",
        "when",
        "where",
        "which",
        "whether",
        "can",
        "should",
        "does",
        "do",
        "is",
        "are",
        "was",
        "were",
        "will",
        "would",
        "could",
        "may",
        "might",
        "into",
        "onto",
        "over",
        "under",
        "after",
        "before",
        "between",
        "among",
        "using",
        "use",
        "get",
        "know",
        "need",
        "want",
        "ask",
        "asked",
        "regarding",
        "concerning",
        "including",
        "related",
        "information",
        "details",
        "anything",
        "something",
        "everything",
    }
)


def _content_tokens(text: str) -> list[str]:
    return [token for token in _TOKEN_PATTERN.findall(text.lower()) if token not in _GENERIC_TOKENS]


def _normalize_facet(fragment: str) -> str:
    cleaned = " ".join(fragment.strip().strip(".-–—:").split())
    if not cleaned:
        return ""

    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = _LEADING_FILLER.sub("", cleaned).strip(" .-–—:")
        cleaned = " ".join(cleaned.split())

    # Drop trailing filler words that add no retrieval value.
    tokens = cleaned.split()
    while tokens and tokens[-1].lower().strip(".,?!") in {
        "please",
        "thanks",
        "thank",
        "you",
        "etc",
    }:
        tokens.pop()
    return " ".join(tokens).strip(" .,?!")


def _is_useful_facet(facet: str) -> bool:
    if len(facet) < MIN_FACET_CHARS:
        return False
    lowered = facet.lower().strip()
    if lowered in _GENERIC_FACET_PHRASES:
        return False
    content = _content_tokens(facet)
    if not content:
        return False
    # Allow a single substantial content token ("extensions", "attendance").
    if len(content) == 1 and len(content[0]) < 5:
        return False
    if len(content) < MIN_FACET_CONTENT_TOKENS:
        return False
    if all(token in _GENERIC_TOKENS for token in _TOKEN_PATTERN.findall(lowered)):
        return False
    return True


def _clause_sized(text: str) -> bool:
    return len(_TOKEN_PATTERN.findall(text.lower())) >= 4


def _expand_and_or_splits(parts: list[str]) -> list[str]:
    """Optionally split long coordinated clauses without breaking short phrases."""
    expanded: list[str] = []
    for part in parts:
        pieces = _AND_OR_PATTERN.split(part)
        # pieces like [left, 'and', right, 'or', right2, ...]
        if len(pieces) == 1:
            expanded.append(part)
            continue

        buffer = pieces[0]
        index = 1
        while index + 1 < len(pieces):
            conjunction = pieces[index]
            right = pieces[index + 1]
            if _clause_sized(buffer) and _clause_sized(right):
                expanded.append(buffer)
                buffer = right
            else:
                buffer = f"{buffer} {conjunction} {right}"
            index += 2
        expanded.append(buffer)
    return expanded


def extract_question_facets(question: str, max_facets: int = MAX_FACETS) -> list[str]:
    """Split a question into short retrieval facets.

    Always deterministic: punctuation and coordination splits only. Returns an
    empty list for ordinary single-topic questions that do not yield multiple
    useful facets.
    """
    cleaned = " ".join(question.strip().split())
    if not cleaned:
        return []

    raw_parts = _expand_and_or_splits(_SPLIT_PATTERN.split(cleaned))
    facets: list[str] = []
    seen: set[str] = set()

    for part in raw_parts:
        facet = _normalize_facet(part or "")
        if not facet or not _is_useful_facet(facet):
            continue
        key = facet.lower()
        if key in seen:
            continue
        seen.add(key)
        facets.append(facet)
        if len(facets) >= max_facets:
            break

    # Multi-facet path only when at least two distinct facets remain.
    if len(facets) < 2:
        return []
    return facets


def is_multi_facet_question(question: str) -> bool:
    return len(extract_question_facets(question)) >= 2


def facet_content_tokens(facet: str) -> set[str]:
    return set(_content_tokens(facet))


def chunk_matches_facet(chunk: dict, facet: str) -> bool:
    """Lexical overlap check used for coverage assignment (not hard policy maps)."""
    tokens = facet_content_tokens(facet)
    if not tokens:
        return False
    haystack = f"{chunk.get('section', '')} {chunk.get('text', '')}".lower()
    hits = sum(1 for token in tokens if token in haystack)
    return hits >= max(1, (len(tokens) + 1) // 2)


def merge_scored_candidates(*pools: list[dict]) -> list[dict]:
    """Merge retrieval pools by chunk ID, keeping the best score and facet tags."""
    merged: dict[str, dict] = {}

    for pool in pools:
        for chunk in pool:
            chunk_id = str(chunk.get("chunk_id") or "")
            if not chunk_id:
                continue
            incoming_facets = [
                str(item) for item in (chunk.get("matched_facets") or []) if str(item).strip()
            ]
            existing = merged.get(chunk_id)
            if existing is None:
                merged[chunk_id] = {
                    **chunk,
                    "matched_facets": list(dict.fromkeys(incoming_facets)),
                }
                continue

            existing["score"] = max(float(existing.get("score") or 0.0), float(chunk.get("score") or 0.0))
            for facet in incoming_facets:
                if facet not in existing["matched_facets"]:
                    existing["matched_facets"].append(facet)

    return sorted(merged.values(), key=lambda item: float(item.get("score") or 0.0), reverse=True)


def assign_missing_facet_matches(chunks: list[dict], facets: list[str]) -> None:
    """Fill matched_facets via lexical overlap for chunks found only by full-query retrieval."""
    for chunk in chunks:
        matched = [
            str(item) for item in (chunk.get("matched_facets") or []) if str(item).strip()
        ]
        for facet in facets:
            if facet in matched:
                continue
            if chunk_matches_facet(chunk, facet):
                matched.append(facet)
        chunk["matched_facets"] = matched
