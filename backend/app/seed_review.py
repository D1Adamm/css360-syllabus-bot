"""Phase 8 seed review status and edit provenance helpers.

Validated AI seeds start as reviewStatus=generated. They are not human-approved
until explicitly reviewed. Edits preserve original question/answer provenance
and grounding metadata (factId, evidenceQuote, sourceChunkIds, validation).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

REVIEW_STATUSES = frozenset({"generated", "approved", "rejected", "edited"})
DEFAULT_REVIEW_STATUS = "generated"


def resolve_review_status(record: dict[str, Any]) -> str:
    """Prefer reviewStatus; fall back to legacy status field."""
    for key in ("reviewStatus", "status"):
        value = record.get(key)
        if isinstance(value, str) and value.strip().lower() in REVIEW_STATUSES:
            return value.strip().lower()
    return DEFAULT_REVIEW_STATUS


def is_approved_for_export(record: dict[str, Any]) -> bool:
    return resolve_review_status(record) == "approved"


def seed_was_edited(record: dict[str, Any]) -> bool:
    """True when the seed has been human-edited (including after later approval)."""
    if record.get("wasEdited") is True:
        return True
    if resolve_review_status(record) == "edited":
        return True
    if str(record.get("originalQuestion") or "").strip():
        return True
    if str(record.get("originalAnswer") or "").strip():
        return True
    return False


def _current_question(record: dict[str, Any]) -> str:
    return str(record.get("question") or record.get("instruction") or "").strip()


def _current_answer(record: dict[str, Any]) -> str:
    return str(record.get("answer") or record.get("response") or "").strip()


def apply_seed_review(
    record: dict[str, Any],
    *,
    review_status: str,
    question: str | None = None,
    answer: str | None = None,
    review_notes: str | None = None,
    reviewed_at: str | None = None,
) -> dict[str, Any]:
    """Return an updated seed record with review status / optional edits.

    Provenance rules:
    - Grounding fields (factId, evidenceQuote, sourceChunkIds, validation, origin)
      are never cleared by review.
    - First text edit snapshots originalQuestion / originalAnswer (and dual-name
      instruction/response mirrors) if not already present.
    - Text edits force reviewStatus to ``edited`` unless the caller is approving
      or rejecting the edited text in the same request (approved/rejected stick).
    - Approving an edited seed sets reviewStatus=approved and keeps wasEdited=true
      plus originalQuestion/originalAnswer.
    """
    status = str(review_status or "").strip().lower()
    if status not in REVIEW_STATUSES:
        raise ValueError(
            f"reviewStatus must be one of {sorted(REVIEW_STATUSES)}; got {review_status!r}"
        )

    updated = dict(record)
    current_question = _current_question(updated)
    current_answer = _current_answer(updated)
    already_edited = seed_was_edited(updated)

    next_question = (
        str(question).strip() if question is not None else current_question
    )
    next_answer = str(answer).strip() if answer is not None else current_answer
    if not next_question or not next_answer:
        raise ValueError("question and answer must be non-empty after review.")

    text_changed = (
        next_question != current_question or next_answer != current_answer
    )
    if text_changed:
        if not str(updated.get("originalQuestion") or "").strip():
            updated["originalQuestion"] = current_question
            updated["originalInstruction"] = current_question
        if not str(updated.get("originalAnswer") or "").strip():
            updated["originalAnswer"] = current_answer
            updated["originalResponse"] = current_answer
        updated["question"] = next_question
        updated["instruction"] = next_question
        updated["answer"] = next_answer
        updated["response"] = next_answer
        # Editing moves the seed into edited unless explicitly approved/rejected.
        if status == "generated":
            status = "edited"
        already_edited = True

    if status == "edited":
        already_edited = True

    updated["reviewStatus"] = status
    # Keep legacy status in sync for existing UI badges.
    updated["status"] = status
    if already_edited:
        updated["wasEdited"] = True
    if review_notes is not None:
        updated["reviewNotes"] = str(review_notes).strip()
    updated["reviewedAt"] = reviewed_at or datetime.now(timezone.utc).isoformat()

    # Grounding / provenance must survive edits.
    for key in (
        "factId",
        "evidenceQuote",
        "sourceChunkIds",
        "sourceSection",
        "validation",
        "origin",
        "normalizedQuestionKey",
        "createdAt",
        "id",
        "originalQuestion",
        "originalAnswer",
        "originalInstruction",
        "originalResponse",
        "wasEdited",
    ):
        if key in record and key not in updated:
            updated[key] = record[key]
        elif key in record and updated.get(key) in (None, "", []):
            updated[key] = record[key]

    return updated
