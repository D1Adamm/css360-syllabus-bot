"""Diverse syllabus chunk selection for course-scoped RAG retrieval.

Fetches a larger cosine-similarity candidate pool, then returns a smaller
final set that prefers section diversity and drops near-duplicate text.
"""

from __future__ import annotations

import re
from typing import Any

# Final chunks returned to the model/API by default.
DEFAULT_TOP_K = 5
# Ranked candidates considered before diversity filtering.
CANDIDATE_POOL_SIZE = 10
# Token Jaccard above this treats two chunks as near-duplicates.
NEAR_DUPLICATE_JACCARD = 0.82
# Soft cap: prefer at most this many chunks sharing one diversity key
# during the first selection pass.
MAX_PER_SECTION_FIRST_PASS = 1

_TOKEN_PATTERN = re.compile(r"[a-z0-9]{3,}")
_GENERIC_SECTION_TITLES = frozenset(
    {
        "general",
        "introduction",
        "course introduction",
        "syllabus",
        "overview",
        "contents",
        "table of contents",
    }
)


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.lower()))


def token_jaccard(left: str, right: str) -> float:
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def is_generic_section_title(section: str) -> bool:
    cleaned = " ".join(section.strip().lower().split())
    if not cleaned:
        return True
    if cleaned in _GENERIC_SECTION_TITLES:
        return True
    # Document-style titles often include a term and year, e.g.
    # "Software Engineering (Fall 2025)".
    if re.search(r"\b(19|20)\d{2}\b", cleaned) and len(cleaned.split()) <= 8:
        return True
    return False


def _looks_like_heading_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    if stripped.endswith(".") and not stripped.endswith("..."):
        return False
    words = stripped.split()
    if not words or len(words) > 12:
        return False
    if stripped.endswith(":"):
        return True
    if stripped.isupper() and len(words) <= 8:
        return True
    if stripped.istitle() and len(words) <= 8:
        return True
    if re.match(r"^\d+(?:\.\d+)*\s+\S", stripped):
        return True
    return False


def diversity_section_key(chunk: dict[str, Any]) -> str:
    """Prefer a meaningful subsection heading when the stored title is generic."""
    section = str(chunk.get("section") or "").strip()
    text = str(chunk.get("text") or "")
    first_line = text.split("\n", 1)[0].strip() if text else ""

    if (
        first_line
        and first_line.casefold() != section.casefold()
        and _looks_like_heading_line(first_line)
        and (is_generic_section_title(section) or not section)
    ):
        return first_line.rstrip(":")

    return section or first_line or str(chunk.get("chunk_id") or "chunk")


def _is_near_duplicate(candidate: dict[str, Any], selected: list[dict[str, Any]]) -> bool:
    candidate_text = str(candidate.get("text") or "")
    for existing in selected:
        if token_jaccard(candidate_text, str(existing.get("text") or "")) >= NEAR_DUPLICATE_JACCARD:
            return True
    return False


def select_diverse_course_chunks(
    ranked_chunks: list[dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
    candidate_pool_size: int = CANDIDATE_POOL_SIZE,
) -> list[dict[str, Any]]:
    """Select a diverse final set from cosine-ranked candidates.

    Strategy:
    1. Take the top ``candidate_pool_size`` by score.
    2. First pass: keep highest-scoring chunks with distinct section/heading keys,
       skipping near-duplicate text.
    3. Second pass: fill remaining slots with non-duplicate leftovers (section
       repeats allowed) so multi-part answers still get enough context.
    """
    final_k = max(1, top_k)
    pool_size = max(final_k, candidate_pool_size)
    candidates = ranked_chunks[:pool_size]
    if not candidates:
        return []

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    section_counts: dict[str, int] = {}

    def try_add(chunk: dict[str, Any], *, enforce_section_cap: bool) -> bool:
        chunk_id = str(chunk.get("chunk_id") or "")
        if chunk_id and chunk_id in selected_ids:
            return False
        if _is_near_duplicate(chunk, selected):
            return False

        key = diversity_section_key(chunk)
        if enforce_section_cap and section_counts.get(key, 0) >= MAX_PER_SECTION_FIRST_PASS:
            return False

        selected.append(chunk)
        if chunk_id:
            selected_ids.add(chunk_id)
        section_counts[key] = section_counts.get(key, 0) + 1
        return True

    for chunk in candidates:
        if len(selected) >= final_k:
            break
        try_add(chunk, enforce_section_cap=True)

    if len(selected) < final_k:
        for chunk in candidates:
            if len(selected) >= final_k:
                break
            try_add(chunk, enforce_section_cap=False)

    return selected
