"""Authored behaviour examples as seeds: abstention and false-premise correction.

The mixed training objective needs examples of two behaviours no generated
seed shows: saying the syllabus does not cover something, and correcting a
question that assumes something the syllabus contradicts. Those are written
by a person, per course, in a small JSONL file, and enter training the only
way anything does: as seed rows in PostgreSQL that the export reads. The
import script beside this module (`scripts/import_behaviour_seeds.py`) is the
one writer; this module is the validation it applies, kept pure so it is
tested without a database.

A record's `questionType` names the behaviour, and the split reads the same
value back: `unanswerable` becomes a grounded abstention example,
`false_premise` a grounded correction example. Neither becomes a bare
question-answer example, because without the excerpts there is nothing to
abstain from or correct.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ORIGIN = "authored"
ABSTENTION = "unanswerable"
CORRECTION = "false_premise"
BEHAVIOUR_QUESTION_TYPES = (ABSTENTION, CORRECTION)


class BehaviourSeedError(ValueError):
    """A record that cannot become a behaviour seed, and why."""


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def behaviour_seed(
    raw: Mapping[str, Any],
    *,
    approve: bool = False,
    now: str | None = None,
) -> dict[str, Any]:
    """The seed row an authored record becomes.

    `approve` marks it approved on import, with a review note saying so;
    otherwise it waits in the review queue like a generated seed. Either way
    the origin says a person wrote it.
    """
    question = _clean(raw.get("question") or raw.get("instruction"))
    response = " ".join(str(raw.get("response") or raw.get("answer") or "").split())
    question_type = _clean(raw.get("questionType")).lower()

    if not question:
        raise BehaviourSeedError("A behaviour seed needs a question.")
    if not response:
        raise BehaviourSeedError(f"Behaviour seed {question!r} needs a response.")
    if question_type not in BEHAVIOUR_QUESTION_TYPES:
        raise BehaviourSeedError(
            f"Behaviour seed {question!r}: questionType must be one of "
            f"{list(BEHAVIOUR_QUESTION_TYPES)}, got {question_type!r}."
        )
    if "syllabus" not in response.lower():
        raise BehaviourSeedError(
            f"Behaviour seed {question!r}: the response must say what the syllabus "
            "does or does not say; it never mentions the syllabus."
        )

    stamp = now or datetime.now(timezone.utc).isoformat()
    review_status = "approved" if approve else "generated"
    seed: dict[str, Any] = {
        "instruction": question,
        "response": response,
        "questionType": question_type,
        "category": _clean(raw.get("category")) or "behaviour",
        "sourceSection": _clean(raw.get("sourceSection")) or "Authored behaviour example",
        "difficulty": _clean(raw.get("difficulty")) or "Medium",
        # An abstention does not answer the question asked; a correction does,
        # by replacing its premise.
        "directlyAnswered": question_type == CORRECTION,
        "origin": ORIGIN,
        "notes": _clean(raw.get("notes")) or None,
        "createdAt": stamp,
        "status": review_status,
        "reviewStatus": review_status,
    }
    if approve:
        seed["reviewedAt"] = stamp
        seed["reviewNotes"] = (
            "Authored behaviour example, approved on import "
            f"({'abstention' if question_type == ABSTENTION else 'false-premise correction'})."
        )
    return seed


def load_behaviour_records(path: Path) -> list[dict[str, Any]]:
    """Every record in an authored JSONL file, validated, with its line number."""
    if not path.is_file():
        raise BehaviourSeedError(f"Behaviour seed file does not exist: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BehaviourSeedError(f"{path.name} line {line_number}: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise BehaviourSeedError(f"{path.name} line {line_number}: not a JSON object")
        # Validate now so a bad line fails the whole file before any import.
        behaviour_seed(payload)
        records.append({**payload, "line": line_number})
    if not records:
        raise BehaviourSeedError(f"{path.name} holds no records.")
    return records


def normalized_question(text: str) -> str:
    return _clean(text).lower().rstrip("?.! ")
