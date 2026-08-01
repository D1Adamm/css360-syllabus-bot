"""Diverse syllabus chunk selection for course-scoped RAG retrieval.

Fetches a larger cosine-similarity candidate pool, then returns a smaller
final set that prefers section diversity and drops near-duplicate text.
The final set is also bounded in characters so CPU-only Ollama generation
stays within OLLAMA_TIMEOUT_SECONDS.
"""

from __future__ import annotations

import re
from typing import Any

# Final chunks returned for ordinary single-topic questions.
DEFAULT_TOP_K = 4
# Final chunks returned for multi-facet questions (hard cap is MAX_FINAL_TOP_K).
MULTI_FACET_TOP_K = 5
MAX_FINAL_TOP_K = 5
# Ranked candidates considered before diversity filtering.
CANDIDATE_POOL_SIZE = 10
# Token Jaccard above this treats two chunks as near-duplicates.
NEAR_DUPLICATE_JACCARD = 0.82
# Soft cap: prefer at most this many chunks sharing one diversity key
# during the first selection pass.
MAX_PER_SECTION_FIRST_PASS = 1
# Deterministic context budget applied before prompt building.
MAX_CHUNK_CONTEXT_CHARS = 900
MAX_TOTAL_CONTEXT_CHARS = 3000
MULTI_FACET_TOTAL_CONTEXT_CHARS = 4200
# Keep a truncated chunk only if this much of its budget survives.
MIN_USEFUL_CHUNK_CHARS = 200
# Relative relevance floor: keep a weak chunk only when it is close to the
# strongest selected score and/or has lexical/facet overlap with the question.
RELATIVE_SCORE_FLOOR = 0.55
# Single-topic path uses a looser floor so ordinary top-K behavior stays stable.
SINGLE_TOPIC_RELATIVE_SCORE_FLOOR = 0.40


def resolve_final_top_k(requested_top_k: int, *, multi_facet: bool) -> int:
    """Bound the final chunk count for ordinary vs multi-facet questions."""
    requested = max(1, requested_top_k)
    if multi_facet:
        return min(MAX_FINAL_TOP_K, max(requested, MULTI_FACET_TOP_K))
    return min(MAX_FINAL_TOP_K, requested)


def resolve_total_context_limit(*, multi_facet: bool) -> int:
    if multi_facet:
        return MULTI_FACET_TOTAL_CONTEXT_CHARS
    return MAX_TOTAL_CONTEXT_CHARS

_TOKEN_PATTERN = re.compile(r"[a-z0-9]{3,}")
_OVERLAP_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "this",
        "that",
        "with",
        "from",
        "about",
        "what",
        "when",
        "where",
        "which",
        "how",
        "why",
        "who",
        "are",
        "was",
        "were",
        "will",
        "would",
        "could",
        "should",
        "can",
        "may",
        "might",
        "does",
        "did",
        "have",
        "has",
        "had",
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
        "any",
        "all",
        "each",
        "some",
        "only",
        "also",
        "just",
        "than",
        "then",
        "them",
        "they",
        "their",
        "there",
        "these",
        "those",
        "your",
        "you",
        "our",
        "out",
        "not",
        "but",
        "its",
        "per",
    }
)
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


def _content_tokens(text: str) -> set[str]:
    return {token for token in _tokenize(text) if token not in _OVERLAP_STOPWORDS}


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


def truncate_chunk_text(text: str, limit: int) -> str:
    """Trim text to ``limit`` chars, preferring paragraph then sentence breaks."""
    cleaned = text.strip()
    if limit <= 0:
        return ""
    if len(cleaned) <= limit:
        return cleaned

    window = cleaned[:limit]

    paragraph_break = window.rfind("\n\n")
    if paragraph_break >= limit // 2:
        return window[:paragraph_break].strip()

    sentence_break = max(window.rfind(". "), window.rfind(".\n"), window.rfind("? "), window.rfind("! "))
    if sentence_break >= limit // 2:
        return window[: sentence_break + 1].strip()

    space_break = window.rfind(" ")
    if space_break >= limit // 2:
        return window[:space_break].strip()

    return window.strip()


def content_token_overlap(query: str, chunk: dict[str, Any]) -> float:
    """Fraction of query content tokens that appear in the chunk section/text."""
    query_tokens = _content_tokens(query)
    if not query_tokens:
        return 0.0
    haystack = _content_tokens(f"{chunk.get('section', '')} {chunk.get('text', '')}")
    if not haystack:
        return 0.0
    hits = len(query_tokens & haystack)
    return hits / len(query_tokens)


def _best_facet_overlap(chunk: dict[str, Any], facets: list[str]) -> float:
    if not facets:
        return 0.0
    return max((content_token_overlap(facet, chunk) for facet in facets), default=0.0)


def apply_relevance_floor(
    chunks: list[dict[str, Any]],
    *,
    question: str,
    facets: list[str] | None = None,
    multi_facet: bool = False,
) -> list[dict[str, Any]]:
    """Drop weakly relevant fillers that lack lexical/facet support.

    Keeps a chunk when any of the following hold:
    - it is the strongest (or tied strongest) selected candidate;
    - it has matched facets for a multi-facet question;
    - it has meaningful lexical overlap with the question or a facet;
    - its similarity score is not substantially below the strongest candidate
      (semantic near-neighbors are retained even with low literal overlap).

    Rejects chunks that have no meaningful lexical/facet overlap and are
    substantially below the strongest candidates. Does not change top-K caps.
    """
    if not chunks:
        return []

    facet_list = [str(facet).strip() for facet in (facets or []) if str(facet).strip()]
    best_score = max(float(chunk.get("score") or 0.0) for chunk in chunks)
    relative_floor = (
        RELATIVE_SCORE_FLOOR if multi_facet else SINGLE_TOPIC_RELATIVE_SCORE_FLOOR
    )
    score_cutoff = best_score * relative_floor

    kept: list[dict[str, Any]] = []
    for chunk in chunks:
        score = float(chunk.get("score") or 0.0)
        matched_facets = [
            str(item) for item in (chunk.get("matched_facets") or []) if str(item).strip()
        ]
        lexical_overlap = content_token_overlap(question, chunk)
        facet_overlap = _best_facet_overlap(chunk, facet_list)
        has_lexical = lexical_overlap > 0.0 or facet_overlap > 0.0
        has_facet = bool(matched_facets)
        near_top = score >= score_cutoff or abs(score - best_score) <= 1e-9

        if abs(score - best_score) <= 1e-9:
            retention_reason = "top_score"
            retain = True
        elif has_facet and (has_lexical or near_top):
            retention_reason = "facet_match"
            retain = True
        elif has_lexical:
            retention_reason = "lexical_overlap"
            retain = True
        elif near_top:
            # Semantically close to the best hit even if wording differs.
            retention_reason = "relative_score"
            retain = True
        else:
            retention_reason = "filtered_low_relevance"
            retain = False

        diagnostics = {
            "fullQueryScore": score,
            "matchedFacet": matched_facets[0] if matched_facets else None,
            "matchedFacets": matched_facets,
            "lexicalOverlap": round(max(lexical_overlap, facet_overlap), 4),
            "relativeScoreFloor": relative_floor,
            "nearTopScore": near_top,
            "retentionReason": retention_reason,
            "diversityKey": diversity_section_key(chunk),
        }
        annotated = {**chunk, "retrieval_diagnostics": diagnostics}
        if retain:
            if matched_facets:
                annotated["coverage_contribution"] = "facet_coverage"
            elif retention_reason == "relative_score":
                annotated["coverage_contribution"] = "semantic_neighbor"
            else:
                annotated["coverage_contribution"] = "lexical_or_diverse"
            kept.append(annotated)

    return kept if kept else [{**chunks[0], "retrieval_diagnostics": {
        "fullQueryScore": float(chunks[0].get("score") or 0.0),
        "matchedFacet": None,
        "matchedFacets": [],
        "lexicalOverlap": content_token_overlap(question, chunks[0]),
        "relativeScoreFloor": relative_floor,
        "nearTopScore": True,
        "retentionReason": "fallback_top_score",
        "diversityKey": diversity_section_key(chunks[0]),
    }, "coverage_contribution": "fallback"}]


def apply_context_budget(
    chunks: list[dict[str, Any]],
    per_chunk_limit: int = MAX_CHUNK_CONTEXT_CHARS,
    total_limit: int = MAX_TOTAL_CONTEXT_CHARS,
) -> list[dict[str, Any]]:
    """Bound RAG context size while preserving order and source metadata.

    Chunks arrive highest-scoring/most-diverse first and are kept in that order.
    Each chunk's text is truncated to ``per_chunk_limit``; the running total is
    capped at ``total_limit``. A chunk that cannot keep a useful amount of text
    is dropped entirely so returned sources always match the context sent.
    """
    budgeted: list[dict[str, Any]] = []
    remaining = max(0, total_limit)

    for chunk in chunks:
        if remaining <= 0:
            break

        allowance = min(per_chunk_limit, remaining)
        text = truncate_chunk_text(str(chunk.get("text") or ""), allowance)
        if not text:
            continue
        if len(text) < MIN_USEFUL_CHUNK_CHARS and len(text) < len(str(chunk.get("text") or "").strip()):
            # Not enough budget left to include a meaningful excerpt.
            break

        budgeted.append({**chunk, "text": text})
        remaining -= len(text)

    return budgeted


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


def select_coverage_aware_chunks(
    ranked_chunks: list[dict[str, Any]],
    facets: list[str],
    top_k: int = MULTI_FACET_TOP_K,
    candidate_pool_size: int = CANDIDATE_POOL_SIZE,
) -> list[dict[str, Any]]:
    """Prefer uncovered facets, then fill by section diversity and score.

    Near-duplicate filtering and section-key limits from the ordinary diversity
    path still apply. Facet membership comes from ``matched_facets`` on each
    chunk (set during merged retrieval), not from hardcoded policy maps.
    """
    if not facets:
        return select_diverse_course_chunks(
            ranked_chunks,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
        )

    final_k = max(1, min(MAX_FINAL_TOP_K, top_k))
    pool_size = max(final_k, candidate_pool_size)
    candidates = ranked_chunks[:pool_size]
    if not candidates:
        return []

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    section_counts: dict[str, int] = {}
    covered_facets: set[str] = set()

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
        for facet in chunk.get("matched_facets") or []:
            covered_facets.add(str(facet))
        return True

    # Pass 1: cover as many facets as possible with distinct sections.
    for facet in facets:
        if len(selected) >= final_k:
            break
        if facet in covered_facets:
            continue
        for chunk in candidates:
            matched = {str(item) for item in (chunk.get("matched_facets") or [])}
            if facet not in matched:
                continue
            if try_add(chunk, enforce_section_cap=True):
                break

    # Pass 2: ordinary section-diverse fill by score.
    if len(selected) < final_k:
        for chunk in candidates:
            if len(selected) >= final_k:
                break
            try_add(chunk, enforce_section_cap=True)

    # Pass 3: allow section repeats if slots remain and text is not near-duplicate.
    if len(selected) < final_k:
        for chunk in candidates:
            if len(selected) >= final_k:
                break
            try_add(chunk, enforce_section_cap=False)

    return selected
