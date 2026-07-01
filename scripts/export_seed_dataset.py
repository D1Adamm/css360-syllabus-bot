#!/usr/bin/env python3
"""Read-only export of prototype and Firebase seed examples for review."""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEED_DATA_PATH = PROJECT_ROOT / "src" / "data" / "seedData.json"
EXPORT_DIR = PROJECT_ROOT / "data" / "exports"
FIREBASE_PATH = "seedExamples"

COMBINED_JSON = "seed-dataset-combined.json"
COMBINED_JSONL = "seed-dataset-combined.jsonl"
PROTOTYPE_JSONL = "seed-dataset-prototype.jsonl"
STUDENT_JSONL = "seed-dataset-student.jsonl"


@dataclass(frozen=True)
class ExportRecord:
    id: str
    question: str
    answer: str
    category: str
    source_section: str
    difficulty: str
    answer_type: str
    source: str
    created_at: str | None = None
    notes: str | None = None
    instruction: str | None = None
    response: str | None = None
    directly_answered: bool | None = None
    origin: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "category": self.category,
            "sourceSection": self.source_section,
            "difficulty": self.difficulty,
            "answerType": self.answer_type,
            "source": self.source,
        }

        if self.created_at:
            payload["createdAt"] = self.created_at
        if self.notes:
            payload["notes"] = self.notes
        if self.instruction is not None:
            payload["instruction"] = self.instruction
        if self.response is not None:
            payload["response"] = self.response
        if self.directly_answered is not None:
            payload["directlyAnswered"] = self.directly_answered
        if self.origin is not None:
            payload["origin"] = self.origin

        return payload


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def normalize_for_dedup(value: str) -> str:
    lowered = value.strip().lower()
    without_punctuation = re.sub(r"[^\w\s]", "", lowered)
    return re.sub(r"\s+", " ", without_punctuation).strip()


def dedup_key(record: ExportRecord) -> str:
    return f"{normalize_for_dedup(record.question)}||{normalize_for_dedup(record.answer)}"


def answer_type_label(directly_answered: bool) -> str:
    return "Directly answered" if directly_answered else "Not directly answered"


def map_source(origin: str) -> str:
    if origin == "prototype":
        return "prototype"
    if origin == "user":
        return "student"
    raise ValueError(f"Unsupported origin value: {origin!r}")


def to_export_record(raw: dict[str, Any], default_source: str | None = None) -> ExportRecord:
    origin = str(raw.get("origin", default_source or "")).strip()
    source = map_source(origin) if origin in {"prototype", "user"} else (default_source or origin)

    instruction = str(raw.get("instruction", raw.get("question", ""))).strip()
    response = str(raw.get("response", raw.get("answer", ""))).strip()
    directly_answered = bool(raw.get("directlyAnswered", False))

    notes = raw.get("notes")
    created_at = raw.get("createdAt")

    return ExportRecord(
        id=str(raw.get("id", "")).strip(),
        question=instruction,
        answer=response,
        category=str(raw.get("category", "")).strip(),
        source_section=str(raw.get("sourceSection", "")).strip(),
        difficulty=str(raw.get("difficulty", "")).strip(),
        answer_type=answer_type_label(directly_answered),
        source=source,
        created_at=str(created_at).strip() if created_at else None,
        notes=str(notes).strip() if notes else None,
        instruction=instruction,
        response=response,
        directly_answered=directly_answered,
        origin=origin or None,
    )


def load_prototype_examples() -> list[ExportRecord]:
    with SEED_DATA_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {SEED_DATA_PATH}")

    records: list[ExportRecord] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        record = to_export_record(item, default_source="prototype")
        if record.id and record.question and record.answer:
            records.append(record)

    return records


def get_database_url() -> str:
    database_url = (
        os.environ.get("FIREBASE_DATABASE_URL")
        or os.environ.get("VITE_FIREBASE_DATABASE_URL")
        or ""
    ).strip().rstrip("/")

    if not database_url:
        raise ValueError(
            "Missing Firebase database URL. Set FIREBASE_DATABASE_URL or "
            "VITE_FIREBASE_DATABASE_URL in your environment or .env file."
        )

    return database_url


def fetch_firebase_seed_examples() -> list[ExportRecord]:
    database_url = get_database_url()
    auth_token = os.environ.get("FIREBASE_AUTH_TOKEN", "").strip()

    request_url = f"{database_url}/{FIREBASE_PATH}.json"
    if auth_token:
        request_url = f"{request_url}?auth={auth_token}"

    request = urllib.request.Request(
        request_url,
        headers={"Accept": "application/json"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 401:
            raise ValueError(
                "Firebase returned 401 Unauthorized. Provide FIREBASE_AUTH_TOKEN "
                "if database rules require authentication."
            ) from error
        raise ValueError(f"Firebase request failed with HTTP {error.code}: {error.reason}") from error
    except urllib.error.URLError as error:
        raise ValueError(f"Could not reach Firebase database: {error.reason}") from error

    if payload is None:
        return []

    if not isinstance(payload, dict):
        raise ValueError("Unexpected Firebase payload shape for seedExamples")

    records: list[ExportRecord] = []
    for item in payload.values():
        if not isinstance(item, dict):
            continue
        origin = str(item.get("origin", "")).strip()
        if origin and origin != "user":
            continue
        record = to_export_record(item, default_source="student")
        if record.id and record.question and record.answer:
            records.append(record)

    records.sort(key=lambda entry: entry.created_at or "", reverse=True)
    return records


def deduplicate_records(records: list[ExportRecord]) -> tuple[list[ExportRecord], int]:
    seen: set[str] = set()
    deduped: list[ExportRecord] = []

    for record in records:
        key = dedup_key(record)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)

    removed = len(records) - len(deduped)
    return deduped, removed


def write_json(path: Path, records: list[ExportRecord]) -> None:
    content = [record.to_dict() for record in records]
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[ExportRecord]) -> None:
    lines = [json.dumps(record.to_dict(), ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def count_by(records: list[ExportRecord], field_name: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        value = getattr(record, field_name)
        counter[str(value)] += 1
    return counter


def print_counter(title: str, counter: Counter[str]) -> None:
    print(title)
    if not counter:
        print("  (none)")
        return

    for key in sorted(counter.keys()):
        print(f"  {key}: {counter[key]}")


def print_summary(
    prototype_count: int,
    student_count: int,
    combined_before: int,
    duplicate_count: int,
    final_records: list[ExportRecord],
) -> None:
    print("\nExport summary")
    print("==============")
    print(f"Prototype examples: {prototype_count}")
    print(f"Student examples: {student_count}")
    print(f"Combined before deduplication: {combined_before}")
    print(f"Duplicates removed: {duplicate_count}")
    print(f"Final exported count: {len(final_records)}")

    print_counter("\nCategory counts:", count_by(final_records, "category"))
    print_counter("\nDifficulty counts:", count_by(final_records, "difficulty"))
    print_counter("\nAnswer-type counts:", count_by(final_records, "answer_type"))


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")

    prototype_records = load_prototype_examples()

    try:
        student_records = fetch_firebase_seed_examples()
        firebase_access = (
            "Firebase read used the Realtime Database REST API with "
            f"{get_database_url()}/{FIREBASE_PATH}.json"
        )
        if os.environ.get("FIREBASE_AUTH_TOKEN", "").strip():
            firebase_access += " (authenticated with FIREBASE_AUTH_TOKEN)"
        else:
            firebase_access += (
                " (unauthenticated public read; temporary if database rules allow it)"
            )
    except ValueError as error:
        print(f"Warning: skipping Firebase examples: {error}", file=sys.stderr)
        student_records = []
        firebase_access = "Firebase examples were not loaded due to the error above."

    combined_before = len(prototype_records) + len(student_records)
    combined_records = [*prototype_records, *student_records]
    final_records, duplicate_count = deduplicate_records(combined_records)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    combined_json_path = EXPORT_DIR / COMBINED_JSON
    combined_jsonl_path = EXPORT_DIR / COMBINED_JSONL
    prototype_jsonl_path = EXPORT_DIR / PROTOTYPE_JSONL
    student_jsonl_path = EXPORT_DIR / STUDENT_JSONL

    write_json(combined_json_path, final_records)
    write_jsonl(combined_jsonl_path, final_records)
    write_jsonl(prototype_jsonl_path, [r for r in final_records if r.source == "prototype"])
    write_jsonl(student_jsonl_path, [r for r in final_records if r.source == "student"])

    print("Seed dataset export complete")
    print("============================")
    print(f"Prototype source: {SEED_DATA_PATH}")
    print(f"Firebase path: {FIREBASE_PATH}")
    print(f"Export directory: {EXPORT_DIR}")
    print(f"Duplicates removed: {duplicate_count}")
    print("\nFiles written:")
    print(f"  {combined_json_path}")
    print(f"  {combined_jsonl_path}")
    print(f"  {prototype_jsonl_path}")
    print(f"  {student_jsonl_path}")
    print(f"\nFirebase access: {firebase_access}")

    print_summary(
        prototype_count=len(prototype_records),
        student_count=len(student_records),
        combined_before=combined_before,
        duplicate_count=duplicate_count,
        final_records=final_records,
    )

    print(
        "\nNote: this script only prepares a local dataset export. "
        "It does not train or fine-tune a model."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
