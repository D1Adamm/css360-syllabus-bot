"""Bulk import of staff-authored seeds into one course, from a JSON file.

For an instructor or administrator who has written a set of syllabus
question/answer pairs by hand and wants them in a course's seed examples. It is
an operator tool, run from the VM (`scripts/import_manual_seeds.py`), not a
route, and it adds nothing to the seed system: every row is written by
`db_seeds.create_seed`, the insert every other path uses.

What makes the rows recognisable:

- `origin` is `prototype`, the existing origin for hand-written seeds — not
  `ai_generated` (starter generation) and not `user` (student contributions,
  which is what the student Contribute page lists).
- `notes` says they came from this import.
- Review state is the normal pending `generated`, or `approved` when the
  operator vouches for them with `--approved`.

Duplicate-safe by construction. A question already stored for the course (by
the same normalised key the generator dedupes with) is skipped, and each
imported seed's id is derived from the course and that key, so a second run —
even one racing the first — cannot insert another copy: the primary key
refuses it.

Deliberately imports nothing that generates: no Ollama client, no starter jobs,
no training, no syllabus storage. A test pins that.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app import db_courses, db_seeds
from app.course_id import assert_valid_course_id
from app.db import DatabaseConfigurationError, db_connection
from app.seed_dedupe import normalize_question_for_dedupe

#: The existing origin for hand-written seeds (see `SeedOrigin` in the frontend).
MANUAL_ORIGIN = "prototype"
MANUAL_NOTES = "Staff-authored seed (manual import)"
APPROVED_REVIEW_NOTE = "Imported as approved by the staff member who authored it."

REQUIRED_FIELDS = ("question", "answer")
OPTIONAL_FIELDS = ("category", "sourceSection")
ALLOWED_FIELDS = frozenset(REQUIRED_FIELDS + OPTIONAL_FIELDS)


class ImportValidationError(Exception):
    """The file or the course is not importable. Carries every problem found."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("\n".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class ManualSeed:
    question: str
    answer: str
    category: str | None
    source_section: str | None
    key: str


@dataclass
class ImportPlan:
    course_id: str
    review_status: str
    seeds_in_file: int
    existing_count: int
    to_insert: list[ManualSeed] = field(default_factory=list)
    duplicates: list[ManualSeed] = field(default_factory=list)


@dataclass
class ImportResult:
    inserted: int = 0
    skipped: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    total_course_seeds: int = 0
    approved_course_seeds: int = 0


# --------------------------------------------------------------------------- #
# Reading the file
# --------------------------------------------------------------------------- #


def parse_seed_file(text: str) -> list[ManualSeed]:
    """Every problem in the file at once, or the seeds it describes."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ImportValidationError([f"The file is not valid JSON: {exc}"]) from exc
    if not isinstance(data, list):
        raise ImportValidationError(["The file must contain a JSON list of seed objects."])
    if not data:
        raise ImportValidationError(["The file contains no seeds."])

    problems: list[str] = []
    seeds: list[ManualSeed] = []
    first_seen: dict[str, int] = {}

    for index, item in enumerate(data, start=1):
        where = f"Seed {index}"
        if not isinstance(item, dict):
            problems.append(f"{where}: must be an object with question and answer.")
            continue
        unknown = sorted(set(item) - ALLOWED_FIELDS)
        if unknown:
            problems.append(
                f"{where}: unknown field(s) {', '.join(unknown)} "
                f"(allowed: {', '.join(sorted(ALLOWED_FIELDS))})."
            )
        values: dict[str, str | None] = {}
        for name in REQUIRED_FIELDS:
            value = item.get(name)
            if not isinstance(value, str) or not value.strip():
                problems.append(f"{where}: '{name}' is missing or empty.")
                values[name] = None
            else:
                values[name] = " ".join(value.split()) if name == "question" else value.strip()
        for name in OPTIONAL_FIELDS:
            value = item.get(name)
            if value is None:
                values[name] = None
            elif not isinstance(value, str):
                problems.append(f"{where}: '{name}' must be text.")
                values[name] = None
            else:
                values[name] = value.strip() or None

        question, answer = values["question"], values["answer"]
        if not question or not answer:
            continue
        key = normalize_question_for_dedupe(question)
        if not key:
            problems.append(f"{where}: the question has no letters or digits.")
            continue
        if key in first_seen:
            problems.append(
                f"{where}: duplicates the question in seed {first_seen[key]} ({question!r})."
            )
            continue
        first_seen[key] = index
        seeds.append(
            ManualSeed(
                question=question,
                answer=answer,
                category=values["category"],
                source_section=values["sourceSection"],
                key=key,
            )
        )

    if problems:
        raise ImportValidationError(problems)
    return seeds


# --------------------------------------------------------------------------- #
# Planning and writing
# --------------------------------------------------------------------------- #


def manual_seed_id(course_id: str, key: str) -> str:
    """Stable per course and question, so a re-run addresses the same row."""
    digest = hashlib.sha256(f"{course_id}\n{key}".encode("utf-8")).hexdigest()[:24]
    return f"manual-{digest}"


def _existing_keys(existing: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    keys: set[str] = set()
    ids: set[str] = set()
    for seed in existing:
        ids.add(str(seed.get("id")))
        stored = seed.get("normalizedQuestionKey")
        if isinstance(stored, str) and stored.strip():
            keys.add(stored.strip())
        question = seed.get("question") or seed.get("instruction")
        if isinstance(question, str) and question.strip():
            keys.add(normalize_question_for_dedupe(question))
    return keys, ids


def plan_import(conn: Any, course_id: str, seeds: list[ManualSeed], *, approved: bool) -> ImportPlan:
    """Read-only: what an import would insert and what it would skip."""
    if not db_courses.course_exists(conn, course_id):
        raise ImportValidationError([f'Course "{course_id}" does not exist.'])
    existing = db_seeds.list_seeds(conn, course_id)
    keys, ids = _existing_keys(existing)
    plan = ImportPlan(
        course_id=course_id,
        review_status="approved" if approved else "generated",
        seeds_in_file=len(seeds),
        existing_count=len(existing),
    )
    for seed in seeds:
        if seed.key in keys or manual_seed_id(course_id, seed.key) in ids:
            plan.duplicates.append(seed)
        else:
            plan.to_insert.append(seed)
    return plan


def _seed_record(seed: ManualSeed, *, approved: bool, now: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "question": seed.question,
        "answer": seed.answer,
        "category": seed.category or "general",
        "sourceSection": seed.source_section or "General",
        "origin": MANUAL_ORIGIN,
        "notes": MANUAL_NOTES,
        "questionType": "direct",
        "normalizedQuestionKey": seed.key,
        "createdAt": now,
        "status": "approved" if approved else "generated",
        "reviewStatus": "approved" if approved else "generated",
    }
    if approved:
        record["reviewedAt"] = now
        record["reviewNotes"] = APPROVED_REVIEW_NOTE
    return record


def execute_import(conn: Any, plan: ImportPlan, *, approved: bool) -> ImportResult:
    """Insert the planned seeds, one savepoint each, so one failure stays one failure."""
    result = ImportResult(skipped=len(plan.duplicates))
    now = datetime.now(timezone.utc).isoformat()
    for seed in plan.to_insert:
        try:
            with conn.transaction():
                db_seeds.create_seed(
                    conn,
                    plan.course_id,
                    _seed_record(seed, approved=approved, now=now),
                    seed_id=manual_seed_id(plan.course_id, seed.key),
                )
            result.inserted += 1
        except Exception as exc:  # noqa: BLE001 - reported per seed, the rest continue
            result.failed.append((seed.question, str(exc).splitlines()[0] if str(exc) else type(exc).__name__))

    counts = db_seeds.count_seeds_by_review_status(conn, plan.course_id)
    result.total_course_seeds = sum(counts.values())
    result.approved_course_seeds = counts.get("approved", 0)
    return result


# --------------------------------------------------------------------------- #
# The whole run
# --------------------------------------------------------------------------- #


def describe_plan(plan: ImportPlan) -> list[str]:
    lines = [
        f"Course:                       {plan.course_id}",
        f"Seeds in file:                {plan.seeds_in_file}",
        f"Existing course seeds:        {plan.existing_count}",
        f"New seeds to insert:          {len(plan.to_insert)}",
        f"Duplicates skipped:           {len(plan.duplicates)}",
        f"Review status to assign:      {plan.review_status}",
        f"Origin to assign:             {MANUAL_ORIGIN} (staff-authored)",
    ]
    for seed in plan.duplicates:
        lines.append(f"  skip (already stored): {seed.question}")
    return lines


def describe_result(result: ImportResult) -> list[str]:
    lines = [
        f"inserted:                     {result.inserted}",
        f"skipped:                      {result.skipped}",
        f"failed:                       {len(result.failed)}",
        f"total course seeds:           {result.total_course_seeds}",
        f"approved course seeds:        {result.approved_course_seeds}",
    ]
    for question, reason in result.failed:
        lines.append(f"  failed: {question} — {reason}")
    return lines


def run_import(
    course_id: str,
    file_path: Path,
    *,
    approved: bool = False,
    dry_run: bool = False,
    confirm: Callable[[ImportPlan], bool] | None = None,
    out: Callable[[str], None] = print,
) -> int:
    """Validate, show the plan, ask, write. Returns a process exit code."""
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        out(f"error: {exc}")
        return 1
    try:
        text = Path(file_path).read_text(encoding="utf-8")
    except OSError as exc:
        out(f"error: could not read {file_path}: {exc}")
        return 1

    try:
        seeds = parse_seed_file(text)
        with db_connection() as conn:
            plan = plan_import(conn, safe_course_id, seeds, approved=approved)
            for line in describe_plan(plan):
                out(line)

            if dry_run:
                out("Dry run: nothing was written.")
                conn.rollback()
                return 0
            if not plan.to_insert:
                out("Nothing to insert: every seed in the file is already stored.")
                return 0
            if confirm is not None and not confirm(plan):
                out("Cancelled: nothing was written.")
                conn.rollback()
                return 1

            result = execute_import(conn, plan, approved=approved)
        for line in describe_result(result):
            out(line)
        return 0 if not result.failed else 2
    except ImportValidationError as exc:
        out("error: the import was not run. Fix these and try again:")
        for problem in exc.problems:
            out(f"  - {problem}")
        return 1
    except DatabaseConfigurationError as exc:
        out(f"error: {exc}")
        return 1
