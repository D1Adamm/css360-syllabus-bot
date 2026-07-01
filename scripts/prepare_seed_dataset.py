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
    loaded_count: int,
    invalid_records: list[dict[str, str]],
    duplicate_count: int,
    prepared_records: list[PreparedRecord],
) -> dict[str, Any]:
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "inputFile": str(input_path),
        "loadedCount": loaded_count,
        "invalidCount": len(invalid_records),
        "duplicateCount": duplicate_count,
        "preparedCount": len(prepared_records),
        "prototypeCount": sum(1 for record in prepared_records if record.source == "prototype"),
        "studentCount": sum(1 for record in prepared_records if record.source == "student"),
        "reviewRecommendedCount": sum(1 for record in prepared_records if record.review_recommended),
        "categoryCounts": count_by(prepared_records, "category"),
        "difficultyCounts": count_by(prepared_records, "difficulty"),
        "answerTypeCounts": count_by(prepared_records, "answer_type"),
        "sourceCounts": count_by(prepared_records, "source"),
        "invalidRecords": invalid_records,
        "notes": [
            "This step prepares data for review only.",
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
        "--output-dir",
        type=Path,
        default=PREPARED_DIR,
        help=f"Directory for prepared outputs (default: {PREPARED_DIR})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
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

    prepared_records.sort(key=lambda record: record.id)
    final_records, duplicate_count = deduplicate_records(prepared_records)

    output_dir.mkdir(parents=True, exist_ok=True)

    review_rows = [record.to_review_dict() for record in final_records]
    finetuning_rows = [record.to_finetuning_dict() for record in final_records]
    summary = build_summary(
        input_path=input_path,
        loaded_count=loaded_count,
        invalid_records=invalid_records,
        duplicate_count=duplicate_count,
        prepared_records=final_records,
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
    print(f"Output directory: {output_dir}")
    print(f"Loaded from input: {loaded_count}")
    print(f"Invalid records skipped: {len(invalid_records)}")
    print(f"Duplicates removed during preparation: {duplicate_count}")
    print(f"Prepared records written: {len(final_records)}")
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
    print_counter("\nCategory counts:", summary["categoryCounts"])
    print_counter("\nDifficulty counts:", summary["difficultyCounts"])
    print_counter("\nAnswer-type counts:", summary["answerTypeCounts"])
    print_counter("\nSource counts:", summary["sourceCounts"])

    if invalid_records:
        print("\nInvalid records:")
        for item in invalid_records:
            print(f"  {item['id']}: {item['errors']}")

    print(
        "\nNote: counts above reflect only the input file available at runtime. "
        "Re-run after regenerating data/exports/seed-dataset-combined.json locally."
    )
    print(
        "This script prepares data for review and future fine-tuning. "
        "It does not create train/validation/test splits and does not train a model."
    )

    return 1 if invalid_records else 0


if __name__ == "__main__":
    raise SystemExit(main())
