#!/usr/bin/env python3
"""Create deterministic train/validation/test splits for fine-tuning data."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "prepared" / "seed-dataset-finetuning.jsonl"
DEFAULT_PREPARED = PROJECT_ROOT / "data" / "prepared" / "seed-dataset-prepared.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"

TRAIN_FILE = "train.jsonl"
VALIDATION_FILE = "validation.jsonl"
TEST_FILE = "test.jsonl"
SUMMARY_FILE = "split-summary.json"

DEFAULT_SEED = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

SIMILARITY_THRESHOLD = 0.55
METADATA_FIELDS = (
    "category",
    "sourceSection",
    "difficulty",
    "directlyAnswered",
    "answerType",
    "question",
    "answer",
    "response",
    "reviewStatus",
    "reviewNotes",
    "appliedOverrideFields",
    "createdAt",
    "notes",
)


def normalize_text(value: str) -> str:
    lowered = value.strip().lower()
    without_punctuation = re.sub(r"[^\w\s]", "", lowered)
    return re.sub(r"\s+", " ", without_punctuation).strip()


def token_set(value: str) -> set[str]:
    normalized = normalize_text(value)
    return {token for token in normalized.split() if token}


def question_similarity(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return len(intersection) / len(union)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found at {path}. "
            "Run scripts/prepare_seed_dataset.py first."
        )

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object on line {line_number} in {path}")
            if "id" not in record:
                raise ValueError(f"Missing id on line {line_number} in {path}")
            records.append(record)

    return records


def load_metadata_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    metadata: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(path):
        record_id = str(record["id"])
        metadata[record_id] = {
            key: record[key]
            for key in METADATA_FIELDS
            if key in record
        }
        if "category" not in metadata[record_id] and "sourceSection" in record:
            metadata[record_id]["sourceSection"] = record["sourceSection"]
    return metadata


def enrich_record(
    record: dict[str, Any],
    metadata_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    enriched = dict(record)
    metadata = metadata_map.get(str(record["id"]), {})

    for key, value in metadata.items():
        enriched.setdefault(key, value)

    instruction = str(enriched.get("instruction", enriched.get("question", "")))
    enriched.setdefault("question", instruction)
    enriched.setdefault("answer", enriched.get("output", enriched.get("response", "")))
    enriched.setdefault("category", "Unknown")
    enriched.setdefault("sourceSection", "Unknown")
    enriched.setdefault("source", "unknown")

    return enriched


def build_similarity_groups(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    policy_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        category = str(record.get("category", "Unknown"))
        source_section = str(record.get("sourceSection", "Unknown"))
        policy_groups[(category, source_section)].append(record)

    final_groups: list[list[dict[str, Any]]] = []
    for (category, source_section), policy_records in sorted(policy_groups.items()):
        remaining = sorted(
            policy_records,
            key=lambda item: (str(item["id"]), str(item.get("instruction", ""))),
        )

        while remaining:
            anchor = remaining.pop(0)
            cluster = [anchor]
            still_remaining: list[dict[str, Any]] = []

            anchor_question = str(anchor.get("instruction", anchor.get("question", "")))
            for candidate in remaining:
                candidate_question = str(
                    candidate.get("instruction", candidate.get("question", ""))
                )
                similarity = question_similarity(anchor_question, candidate_question)
                if similarity >= SIMILARITY_THRESHOLD:
                    cluster.append(candidate)
                else:
                    still_remaining.append(candidate)

            remaining = still_remaining
            cluster.sort(key=lambda item: str(item["id"]))
            final_groups.append(cluster)

    final_groups.sort(
        key=lambda group: (
            str(group[0].get("category", "Unknown")),
            str(group[0].get("sourceSection", "Unknown")),
            str(group[0]["id"]),
        )
    )
    return final_groups


def compute_targets(total: int) -> dict[str, int]:
    train = round(total * TRAIN_RATIO)
    validation = round(total * VALIDATION_RATIO)
    test = total - train - validation

    if test < 0:
        validation = max(0, validation + test)
        test = total - train - validation

    return {"train": train, "validation": validation, "test": test}


def category_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(record.get("category", "Unknown")) for record in records)
    return dict(sorted(counter.items()))


def source_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(record.get("source", "unknown")) for record in records)
    return dict(sorted(counter.items()))


def assign_groups_to_splits(
    groups: list[list[dict[str, Any]]],
    targets: dict[str, int],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    split_records: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    split_order = ["train", "validation", "test"]

    grouped_by_category: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for group in groups:
        category = str(group[0].get("category", "Unknown"))
        grouped_by_category[category].append(group)

    ordered_groups: list[list[dict[str, Any]]] = []
    random_generator = random.Random(seed)
    for category in sorted(grouped_by_category):
        category_groups = grouped_by_category[category]
        category_groups.sort(
            key=lambda group: (-len(group), str(group[0]["id"])),
        )
        random_generator.shuffle(category_groups)
        ordered_groups.extend(category_groups)

    ordered_groups.sort(key=lambda group: (-len(group), str(group[0]["id"])))

    for group in ordered_groups:
        assigned = False
        for split_name in split_order:
            projected_count = len(split_records[split_name]) + len(group)
            if projected_count <= targets[split_name]:
                split_records[split_name].extend(group)
                assigned = True
                break

        if not assigned:
            best_split = min(
                split_order,
                key=lambda split_name: (
                    abs(
                        len(split_records[split_name]) + len(group) - targets[split_name]
                    ),
                    {"train": 0, "validation": 1, "test": 2}[split_name],
                ),
            )
            split_records[best_split].extend(group)

    for split_name in split_records:
        split_records[split_name].sort(key=lambda item: str(item["id"]))

    return split_records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    path.write_text(content + ("\n" if records else ""), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_counter(title: str, counter: dict[str, int]) -> None:
    print(title)
    if not counter:
        print("  (none)")
        return
    for key, value in counter.items():
        print(f"  {key}: {value}")


def build_summary(
    input_path: Path,
    prepared_path: Path,
    output_dir: Path,
    seed: int,
    targets: dict[str, int],
    split_records: dict[str, list[dict[str, Any]]],
    groups: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    total = sum(len(records) for records in split_records.values())
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "inputFile": str(input_path),
        "preparedMetadataFile": str(prepared_path) if prepared_path.exists() else None,
        "outputDirectory": str(output_dir),
        "randomSeed": seed,
        "targetRatios": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": TEST_RATIO,
        },
        "targetCounts": targets,
        "totalCount": total,
        "splitCounts": {
            split_name: len(records) for split_name, records in split_records.items()
        },
        "categoryCountsBySplit": {
            split_name: category_counts(records)
            for split_name, records in split_records.items()
        },
        "sourceCountsBySplit": {
            split_name: source_counts(records)
            for split_name, records in split_records.items()
        },
        "idsBySplit": {
            split_name: [str(record["id"]) for record in records]
            for split_name, records in split_records.items()
        },
        "policyGroups": [
            {
                "category": str(group[0].get("category", "Unknown")),
                "sourceSection": str(group[0].get("sourceSection", "Unknown")),
                "size": len(group),
                "ids": [str(record["id"]) for record in group],
            }
            for group in groups
        ],
        "notes": [
            "Splits are deterministic for a given input file and random seed.",
            "Similar questions from the same policy area are kept in the same split when possible.",
            "The test split must never be used during training or validation tuning.",
            "This step does not fine-tune a model.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split reviewed fine-tuning JSONL into train/validation/test sets.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Fine-tuning JSONL input (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=DEFAULT_PREPARED,
        help=(
            "Prepared JSONL used to enrich category and review metadata "
            f"(default: {DEFAULT_PREPARED})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for split outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for deterministic splitting (default: {DEFAULT_SEED})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    prepared_path = args.prepared.resolve()
    output_dir = args.output_dir.resolve()
    seed = args.seed

    records = load_jsonl(input_path)
    metadata_map = load_metadata_map(prepared_path)
    enriched_records = [enrich_record(record, metadata_map) for record in records]
    enriched_records.sort(key=lambda item: str(item["id"]))

    if not enriched_records:
        raise ValueError(f"No records found in {input_path}")

    targets = compute_targets(len(enriched_records))
    groups = build_similarity_groups(enriched_records)
    split_records = assign_groups_to_splits(groups, targets, seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / TRAIN_FILE
    validation_path = output_dir / VALIDATION_FILE
    test_path = output_dir / TEST_FILE
    summary_path = output_dir / SUMMARY_FILE

    write_jsonl(train_path, split_records["train"])
    write_jsonl(validation_path, split_records["validation"])
    write_jsonl(test_path, split_records["test"])
    summary = build_summary(
        input_path=input_path,
        prepared_path=prepared_path,
        output_dir=output_dir,
        seed=seed,
        targets=targets,
        split_records=split_records,
        groups=groups,
    )
    write_json(summary_path, summary)

    print("Training dataset split complete")
    print("=============================")
    print(f"Input file: {input_path}")
    print(f"Prepared metadata file: {prepared_path}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {seed}")
    print(f"Total records: {summary['totalCount']}")
    print(
        "Target split counts: "
        f"train={targets['train']}, "
        f"validation={targets['validation']}, "
        f"test={targets['test']}"
    )
    print("\nSplit counts:")
    for split_name, count in summary["splitCounts"].items():
        print(f"  {split_name}: {count}")

    for split_name in ("train", "validation", "test"):
        print_counter(
            f"\nCategory counts ({split_name}):",
            summary["categoryCountsBySplit"][split_name],
        )
        print_counter(
            f"Source counts ({split_name}):",
            summary["sourceCountsBySplit"][split_name],
        )
        print(f"\nIDs ({split_name}):")
        for record_id in summary["idsBySplit"][split_name]:
            print(f"  {record_id}")

    print("\nFiles written:")
    print(f"  {train_path}")
    print(f"  {validation_path}")
    print(f"  {test_path}")
    print(f"  {summary_path}")
    print(
        "\nImportant: use train.jsonl for training and validation.jsonl for tuning. "
        "Keep test.jsonl untouched until final evaluation."
    )
    print(
        "Note: counts above reflect only the input file available at runtime. "
        "Re-run after regenerating data/prepared/seed-dataset-finetuning.jsonl locally."
    )
    print("This script creates splits only. It does not fine-tune a model.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
