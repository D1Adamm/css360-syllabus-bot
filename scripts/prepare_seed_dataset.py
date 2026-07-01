#!/usr/bin/env python3
"""Prepare a combined seed export for review and future fine-tuning."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "exports" / "seed-dataset-combined.json"
DEFAULT_OVERRIDES = PROJECT_ROOT / "data" / "reviews" / "seed-review-overrides.json"
PREPARED_DIR = PROJECT_ROOT / "data" / "prepared"

PREPARED_JSON = "seed-dataset-prepared.json"
PREPARED_JSONL = "seed-dataset-prepared.jsonl"
FINETUNING_JSONL = "seed-dataset-finetuning.jsonl"
SUMMARY_JSON = "preparation-summary.json"

SYSTEM_PROMPT = (
    "You are a helpful assistant for CSS 360. Answer syllabus questions clearly "
    "and accurately using course policy information."
)

VALID_DIFFICULTIES = {"Easy", "Medium", "Hard"}
VALID_SOURCES = {"prototype", "student"}
VALID_ANSWER_TYPES = {"Directly answered", "Not directly answered"}
VALID_REVIEW_STATUSES = {"accepted", "rejected", "needs_review"}

OVERRIDE_FIELD_MAP = {
    "question": "instruction",
    "answer": "response",
    "category": "category",
    "sourceSection": "source_section",
    "difficulty": "difficulty",
    "answerType": "answer_type",
}

KNOWN_CATEGORIES = {
    "Course Basics",
    "Communication",
    "Attendance",
    "Course Preparation",
    "Assignments",
    "Projects",
    "Standups",
    "Case Discussions",
    "Grading",
    "Late Work",
    "AI Policy",
    "Technology",
    "Office Hours",
    "Exams and Quizzes",
    "Course Expectations",
}


@dataclass
class PreparedRecord:
    id: str
    instruction: str
    response: str
    category: str
    source_section: str
    difficulty: str
    directly_answered: bool
    answer_type: str
    source: str
    created_at: str | None = None
    notes: str | None = None
    review_recommended: bool = False
    validation_warnings: list[str] = field(default_factory=list)
    review_status: str | None = None
    review_notes: str | None = None
    override_applied_fields: list[str] = field(default_factory=list)

    def to_review_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "instruction": self.instruction,
            "response": self.response,
            "question": self.instruction,
            "answer": self.response,
            "category": self.category,
            "sourceSection": self.source_section,
            "difficulty": self.difficulty,
            "directlyAnswered": self.directly_answered,
            "answerType": self.answer_type,
            "source": self.source,
            "reviewRecommended": self.review_recommended,
        }

        if self.created_at:
            payload["createdAt"] = self.created_at
        if self.notes:
            payload["notes"] = self.notes
        if self.validation_warnings:
            payload["validationWarnings"] = self.validation_warnings
        if self.review_status:
            payload["reviewStatus"] = self.review_status
        if self.review_notes:
            payload["reviewNotes"] = self.review_notes
        if self.override_applied_fields:
            payload["appliedOverrideFields"] = self.override_applied_fields

        return payload

    def to_finetuning_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "system": SYSTEM_PROMPT,
            "instruction": self.instruction,
            "input": "",
            "output": self.response,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self.instruction},
                {"role": "assistant", "content": self.response},
            ],
        }


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def normalize_for_dedup(value: str) -> str:
    lowered = value.strip().lower()
    without_punctuation = re.sub(r"[^\w\s]", "", lowered)
    return re.sub(r"\s+", " ", without_punctuation).strip()


def dedup_key(record: PreparedRecord) -> str:
    return (
        f"{normalize_for_dedup(record.instruction)}||"
        f"{normalize_for_dedup(record.response)}"
    )


def get_text(raw: dict[str, Any], primary_key: str, fallback_key: str) -> str:
    primary = raw.get(primary_key)
    fallback = raw.get(fallback_key)
    chosen = primary if primary not in (None, "") else fallback
    return normalize_whitespace(str(chosen or ""))


def get_bool(raw: dict[str, Any]) -> bool:
    if "directlyAnswered" in raw:
        return bool(raw["directlyAnswered"])
    answer_type = str(raw.get("answerType", "")).strip()
    if answer_type == "Directly answered":
        return True
    if answer_type == "Not directly answered":
        return False
    return False


def get_source(raw: dict[str, Any]) -> str:
    source = str(raw.get("source", "")).strip().lower()
    if source in VALID_SOURCES:
        return source

    origin = str(raw.get("origin", "")).strip().lower()
    if origin == "prototype":
        return "prototype"
    if origin == "user":
        return "student"

    return source


def answer_type_label(directly_answered: bool) -> str:
    return "Directly answered" if directly_answered else "Not directly answered"


def directly_answered_from_answer_type(answer_type: str) -> bool:
    if answer_type == "Directly answered":
        return True
    if answer_type == "Not directly answered":
        return False
    raise ValueError(f"unsupported answerType {answer_type!r}")


def validate_and_prepare(raw: dict[str, Any]) -> tuple[PreparedRecord | None, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    record_id = normalize_whitespace(str(raw.get("id", "")))
    instruction = get_text(raw, "instruction", "question")
    response = get_text(raw, "response", "answer")
    category = normalize_whitespace(str(raw.get("category", "")))
    source_section = normalize_whitespace(str(raw.get("sourceSection", "")))
    difficulty = normalize_whitespace(str(raw.get("difficulty", "")))
    source = get_source(raw)
    directly_answered = get_bool(raw)
    answer_type = str(raw.get("answerType", "")).strip() or answer_type_label(directly_answered)

    created_at_raw = raw.get("createdAt")
    created_at = normalize_whitespace(str(created_at_raw)) if created_at_raw else None

    notes_raw = raw.get("notes")
    notes = normalize_whitespace(str(notes_raw)) if notes_raw else None

    if not record_id:
        errors.append("missing id")
    if not instruction:
        errors.append("missing instruction/question")
    if not response:
        errors.append("missing response/answer")
    if not category:
        errors.append("missing category")
    if not source_section:
        errors.append("missing sourceSection")
    if not difficulty:
        errors.append("missing difficulty")
    if source not in VALID_SOURCES:
        errors.append(f"unsupported source {source!r}")
    if difficulty and difficulty not in VALID_DIFFICULTIES:
        errors.append(f"unsupported difficulty {difficulty!r}")
    if answer_type not in VALID_ANSWER_TYPES:
        warnings.append(f"unexpected answerType {answer_type!r}")

    if category and category not in KNOWN_CATEGORIES:
        warnings.append(f"category {category!r} is not in the known syllabus category list")

    if errors:
        return None, errors

    review_recommended = source == "student" or not directly_answered or bool(warnings)

    record = PreparedRecord(
        id=record_id,
        instruction=instruction,
        response=response,
        category=category,
        source_section=source_section,
        difficulty=difficulty,
        directly_answered=directly_answered,
        answer_type=answer_type,
        source=source,
        created_at=created_at,
        notes=notes,
        review_recommended=review_recommended,
        validation_warnings=warnings,
    )
    return record, warnings


def load_review_overrides(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if not path.exists():
        return {}, []

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    errors: list[str] = []
    overrides: dict[str, dict[str, Any]] = {}

    for example_id, override in payload.items():
        if not isinstance(override, dict):
            errors.append(f"{example_id}: override must be an object")
            continue

        status = str(override.get("status", "")).strip()
        if status not in VALID_REVIEW_STATUSES:
            errors.append(f"{example_id}: unsupported status {status!r}")
            continue

        unknown_fields = sorted(
            key for key in override if key not in {"status", "reviewNotes", *OVERRIDE_FIELD_MAP}
        )
        if unknown_fields:
            errors.append(f"{example_id}: unsupported override fields {unknown_fields}")

        if "difficulty" in override:
            difficulty = normalize_whitespace(str(override["difficulty"]))
            if difficulty not in VALID_DIFFICULTIES:
                errors.append(f"{example_id}: unsupported difficulty {difficulty!r}")

        if "answerType" in override:
            answer_type = normalize_whitespace(str(override["answerType"]))
            if answer_type not in VALID_ANSWER_TYPES:
                errors.append(f"{example_id}: unsupported answerType {answer_type!r}")

        overrides[str(example_id)] = override

    return overrides, errors


def apply_review_override(
    record: PreparedRecord,
    override: dict[str, Any],
) -> PreparedRecord:
    applied_fields: list[str] = []
    instruction = record.instruction
    response = record.response
    category = record.category
    source_section = record.source_section
    difficulty = record.difficulty
    answer_type = record.answer_type
    directly_answered = record.directly_answered

    if "question" in override:
        instruction = normalize_whitespace(str(override["question"]))
        applied_fields.append("question")
    if "answer" in override:
        response = normalize_whitespace(str(override["answer"]))
        applied_fields.append("answer")
    if "category" in override:
        category = normalize_whitespace(str(override["category"]))
        applied_fields.append("category")
    if "sourceSection" in override:
        source_section = normalize_whitespace(str(override["sourceSection"]))
        applied_fields.append("sourceSection")
    if "difficulty" in override:
        difficulty = normalize_whitespace(str(override["difficulty"]))
        applied_fields.append("difficulty")
    if "answerType" in override:
        answer_type = normalize_whitespace(str(override["answerType"]))
        directly_answered = directly_answered_from_answer_type(answer_type)
        applied_fields.append("answerType")

    status = str(override.get("status", "")).strip()
    review_notes_raw = override.get("reviewNotes")
    review_notes = (
        normalize_whitespace(str(review_notes_raw)) if review_notes_raw else None
    )

    validation_warnings = list(record.validation_warnings)
    if category not in KNOWN_CATEGORIES:
        warning = f"category {category!r} is not in the known syllabus category list"
        if warning not in validation_warnings:
            validation_warnings.append(warning)

    if status == "accepted":
        review_recommended = bool(validation_warnings)
    elif status == "needs_review":
        review_recommended = True
    else:
        review_recommended = False

    return PreparedRecord(
        id=record.id,
        instruction=instruction,
        response=response,
        category=category,
        source_section=source_section,
        difficulty=difficulty,
        directly_answered=directly_answered,
        answer_type=answer_type,
        source=record.source,
        created_at=record.created_at,
        notes=record.notes,
        review_recommended=review_recommended,
        validation_warnings=validation_warnings,
        review_status=status,
        review_notes=review_notes,
        override_applied_fields=applied_fields,
    )


def apply_review_overrides(
    records: list[PreparedRecord],
    overrides: dict[str, dict[str, Any]],
) -> tuple[list[PreparedRecord], list[dict[str, Any]], list[str]]:
    records_by_id = {record.id: record for record in records}
    updated_records: list[PreparedRecord] = []
    applied_overrides: list[dict[str, Any]] = []
    unmatched_override_ids: list[str] = []

    for example_id, override in overrides.items():
        if example_id not in records_by_id:
            unmatched_override_ids.append(example_id)
            continue

    for record in records:
        override = overrides.get(record.id)
        if override is None:
            updated_records.append(record)
            continue

        updated = apply_review_override(record, override)
        updated_records.append(updated)
        applied_overrides.append(
            {
                "id": record.id,
                "status": updated.review_status,
                "appliedFields": updated.override_applied_fields,
                "reviewNotes": updated.review_notes,
            }
        )

    return updated_records, applied_overrides, unmatched_override_ids


def load_export_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Combined export not found at {path}. "
            "Run scripts/export_seed_dataset.py first."
        )

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Expected object at index {index} in {path}")
        records.append(item)

    return records


def deduplicate_records(records: list[PreparedRecord]) -> tuple[list[PreparedRecord], int]:
    seen: set[str] = set()
    deduped: list[PreparedRecord] = []

    for record in records:
        key = dedup_key(record)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)

    removed = len(records) - len(deduped)
    return deduped, removed


def count_by(records: list[PreparedRecord], attribute: str) -> dict[str, int]:
    counter: Counter[str] = Counter(getattr(record, attribute) for record in records)
    return dict(sorted(counter.items()))


def count_review_statuses(records: list[PreparedRecord]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        if record.review_status:
            counter[record.review_status] += 1
    return dict(sorted(counter.items()))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(content + ("\n" if rows else ""), encoding="utf-8")


def print_counter(title: str, counter: dict[str, int]) -> None:
    print(title)
    if not counter:
        print("  (none)")
        return
    for key, value in counter.items():
        print(f"  {key}: {value}")


def build_summary(
    input_path: Path,
    overrides_path: Path,
    loaded_count: int,
    invalid_records: list[dict[str, str]],
    duplicate_count: int,
    prepared_records: list[PreparedRecord],
    finetuning_records: list[PreparedRecord],
    applied_overrides: list[dict[str, Any]],
    unmatched_override_ids: list[str],
    override_validation_errors: list[str],
) -> dict[str, Any]:
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "inputFile": str(input_path),
        "overrideFile": str(overrides_path) if overrides_path.exists() else None,
        "loadedCount": loaded_count,
        "invalidCount": len(invalid_records),
        "duplicateCount": duplicate_count,
        "preparedCount": len(prepared_records),
        "finetuningCount": len(finetuning_records),
        "rejectedCount": sum(
            1 for record in prepared_records if record.review_status == "rejected"
        ),
        "prototypeCount": sum(1 for record in prepared_records if record.source == "prototype"),
        "studentCount": sum(1 for record in prepared_records if record.source == "student"),
        "reviewRecommendedCount": sum(1 for record in prepared_records if record.review_recommended),
        "reviewStatusCounts": count_review_statuses(prepared_records),
        "categoryCounts": count_by(prepared_records, "category"),
        "difficultyCounts": count_by(prepared_records, "difficulty"),
        "answerTypeCounts": count_by(prepared_records, "answer_type"),
        "sourceCounts": count_by(prepared_records, "source"),
        "appliedOverrides": applied_overrides,
        "unmatchedOverrideIds": unmatched_override_ids,
        "overrideValidationErrors": override_validation_errors,
        "invalidRecords": invalid_records,
        "notes": [
            "This step prepares data for review only.",
            "Manual corrections belong in data/reviews/seed-review-overrides.json.",
            "Generated files under data/prepared/ are overwritten on each run.",
            "No train/validation/test splits are created.",
            "No model training or fine-tuning is performed.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare combined seed export for review and future fine-tuning.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Combined export JSON file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        default=DEFAULT_OVERRIDES,
        help=f"Review override JSON file (default: {DEFAULT_OVERRIDES})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PREPARED_DIR,
        help=f"Directory for prepared outputs (default: {PREPARED_DIR})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    overrides_path = args.overrides.resolve()
    output_dir = args.output_dir.resolve()

    raw_records = load_export_records(input_path)
    loaded_count = len(raw_records)

    prepared_records: list[PreparedRecord] = []
    invalid_records: list[dict[str, str]] = []

    for raw in raw_records:
        record_id = str(raw.get("id", "(missing id)"))
        prepared, issues = validate_and_prepare(raw)
        if prepared is None:
            invalid_records.append({"id": record_id, "errors": "; ".join(issues)})
            continue
        prepared_records.append(prepared)

    overrides, override_validation_errors = load_review_overrides(overrides_path)
    prepared_records, applied_overrides, unmatched_override_ids = apply_review_overrides(
        prepared_records,
        overrides,
    )

    prepared_records.sort(key=lambda record: record.id)
    final_records, duplicate_count = deduplicate_records(prepared_records)
    finetuning_records = [
        record for record in final_records if record.review_status != "rejected"
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    review_rows = [record.to_review_dict() for record in final_records]
    finetuning_rows = [record.to_finetuning_dict() for record in finetuning_records]
    summary = build_summary(
        input_path=input_path,
        overrides_path=overrides_path,
        loaded_count=loaded_count,
        invalid_records=invalid_records,
        duplicate_count=duplicate_count,
        prepared_records=final_records,
        finetuning_records=finetuning_records,
        applied_overrides=applied_overrides,
        unmatched_override_ids=unmatched_override_ids,
        override_validation_errors=override_validation_errors,
    )

    prepared_json_path = output_dir / PREPARED_JSON
    prepared_jsonl_path = output_dir / PREPARED_JSONL
    finetuning_jsonl_path = output_dir / FINETUNING_JSONL
    summary_json_path = output_dir / SUMMARY_JSON

    write_json(prepared_json_path, review_rows)
    write_jsonl(prepared_jsonl_path, review_rows)
    write_jsonl(finetuning_jsonl_path, finetuning_rows)
    write_json(summary_json_path, summary)

    print("Seed dataset preparation complete")
    print("===============================")
    print(f"Input file: {input_path}")
    print(f"Override file: {overrides_path}")
    print(f"Output directory: {output_dir}")
    print(f"Loaded from input: {loaded_count}")
    print(f"Invalid records skipped: {len(invalid_records)}")
    print(f"Duplicates removed during preparation: {duplicate_count}")
    print(f"Prepared records written: {len(final_records)}")
    print(f"Fine-tuning records written: {len(finetuning_records)}")
    print(f"Applied overrides: {len(applied_overrides)}")
    print("\nFiles written:")
    print(f"  {prepared_json_path}")
    print(f"  {prepared_jsonl_path}")
    print(f"  {finetuning_jsonl_path}")
    print(f"  {summary_json_path}")

    print("\nPreparation summary")
    print("===================")
    print(f"Prototype examples: {summary['prototypeCount']}")
    print(f"Student examples: {summary['studentCount']}")
    print(f"Review recommended: {summary['reviewRecommendedCount']}")
    print(f"Rejected examples excluded from fine-tuning: {summary['rejectedCount']}")
    print_counter("\nReview status counts:", summary["reviewStatusCounts"])
    print_counter("\nCategory counts:", summary["categoryCounts"])
    print_counter("\nDifficulty counts:", summary["difficultyCounts"])
    print_counter("\nAnswer-type counts:", summary["answerTypeCounts"])
    print_counter("\nSource counts:", summary["sourceCounts"])

    if applied_overrides:
        print("\nApplied overrides:")
        for item in applied_overrides:
            fields = ", ".join(item["appliedFields"]) or "(status only)"
            print(f"  {item['id']} [{item['status']}]: {fields}")

    if unmatched_override_ids:
        print("\nUnmatched override IDs (not present in current export):")
        for example_id in unmatched_override_ids:
            print(f"  {example_id}")

    if override_validation_errors:
        print("\nOverride validation errors:")
        for message in override_validation_errors:
            print(f"  {message}")

    if invalid_records:
        print("\nInvalid records:")
        for item in invalid_records:
            print(f"  {item['id']}: {item['errors']}")

    print(
        "\nNote: counts above reflect only the input file available at runtime. "
        "Re-run after regenerating data/exports/seed-dataset-combined.json locally."
    )
    print(
        "Edit data/reviews/seed-review-overrides.json for manual corrections. "
        "Do not edit generated files under data/prepared/."
    )
    print(
        "This script prepares data for review and future fine-tuning. "
        "It does not create train/validation/test splits and does not train a model."
    )

    exit_code = 0
    if invalid_records or override_validation_errors:
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
