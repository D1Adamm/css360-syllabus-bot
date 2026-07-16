"""Shared seed question normalization helpers."""

from __future__ import annotations

import re

_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_question_for_dedupe(question: str) -> str:
    """Normalize a question for exact-match deduplication."""
    lowered = question.strip().lower()
    without_punct = _NON_ALNUM_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", without_punct).strip()
