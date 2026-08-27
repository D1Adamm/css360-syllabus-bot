"""Deterministic train/validation split for approved fine-tune exports.

Reads data/exports/{courseId}/approved-finetune.jsonl (already validated on export)
and writes train.jsonl, validation.jsonl, and manifest.json in the same directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.seed_export import (
    FinetuneJsonlValidationError,
    course_export_dir,
    validate_finetune_jsonl,
    write_json,
    write_jsonl,
)

DEFAULT_SPLIT_SEED = 360
DEFAULT_VALIDATION_FRACTION = 0.1
SOURCE_FILENAME = "approved-finetune.jsonl"
TRAIN_FILENAME = "train.jsonl"
VALIDATION_FILENAME = "validation.jsonl"
MANIFEST_FILENAME = "manifest.json"
SUMMARY_FILENAME = "approved-export-summary.json"

CHECKSUM_CHUNK_BYTES = 64 * 1024


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHECKSUM_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class TrainingSplitError(ValueError):
    """Raised when a training split cannot be prepared from the approved export."""


def approved_finetune_path(course_id: str, *, export_root: Path | None = None) -> Path:
    return course_export_dir(course_id, root=export_root) / SOURCE_FILENAME


def compute_validation_size(total: int, *, fraction: float = DEFAULT_VALIDATION_FRACTION) -> int:
    """Return validation size (~fraction), leaving at least one train example."""
    if total < 2:
        raise TrainingSplitError(
            "Need at least 2 approved examples to create a train/validation split."
        )
    validation_count = max(1, math.ceil(total * fraction))
    validation_count = min(validation_count, total - 1)
    return validation_count


def load_approved_finetune_records(
    path: Path,
) -> list[dict[str, str]]:
    """Validate and load instruction/response records from approved-finetune.jsonl."""
    if not path.is_file():
        raise TrainingSplitError(
            f'Approved export file does not exist: {path}. '
            "Export approved seeds before preparing a training split."
        )

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    try:
        validate_finetune_jsonl(path, expected_count=len(lines))
    except FinetuneJsonlValidationError as exc:
        raise TrainingSplitError(str(exc)) from exc

    records: list[dict[str, str]] = []
    for line in lines:
        payload = json.loads(line)
        records.append(
            {
                "instruction": payload["instruction"],
                "response": payload["response"],
            }
        )
    return records


def read_source_export_timestamp(
    course_id: str,
    *,
    export_root: Path | None = None,
) -> str | None:
    summary_path = course_export_dir(course_id, root=export_root) / SUMMARY_FILENAME
    if not summary_path.is_file():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    exported_at = payload.get("exportedAt")
    if isinstance(exported_at, str) and exported_at.strip():
        return exported_at.strip()
    return None


def split_records(
    records: list[dict[str, str]],
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Deterministically shuffle then split into train and validation lists."""
    total = len(records)
    validation_count = compute_validation_size(total, fraction=validation_fraction)
    shuffled = list(records)
    rng = random.Random(split_seed)
    rng.shuffle(shuffled)
    validation = shuffled[:validation_count]
    train = shuffled[validation_count:]
    if len(train) + len(validation) != total:
        raise TrainingSplitError(
            "Split counts do not add up to the source count "
            f"({len(train)} train + {len(validation)} validation != {total})."
        )
    return train, validation


def approved_export_status(
    course_id: str,
    *,
    export_root: Path | None = None,
) -> dict[str, Any]:
    """Return whether an approved-finetune.jsonl export exists for the course."""
    path = approved_finetune_path(course_id, export_root=export_root)
    exists = path.is_file()
    example_count = 0
    if exists:
        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        example_count = len(lines)
    return {
        "courseId": course_id,
        "exists": exists,
        "exportPath": str(path),
        "exampleCount": example_count,
        "sourceFile": SOURCE_FILENAME,
    }


def prepare_training_split(
    course_id: str,
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    export_root: Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Create train.jsonl, validation.jsonl, and manifest.json from the approved export."""
    out_dir = course_export_dir(course_id, root=export_root)
    source_path = out_dir / SOURCE_FILENAME
    records = load_approved_finetune_records(source_path)
    train, validation = split_records(
        records,
        split_seed=split_seed,
        validation_fraction=validation_fraction,
    )

    train_path = out_dir / TRAIN_FILENAME
    validation_path = out_dir / VALIDATION_FILENAME
    manifest_path = out_dir / MANIFEST_FILENAME

    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)

    # Re-check written counts add up (defense in depth).
    if len(train) + len(validation) != len(records):
        raise TrainingSplitError(
            "Output counts do not add up to the source count "
            f"({len(train)} + {len(validation)} != {len(records)})."
        )

    created = created_at or datetime.now(timezone.utc).isoformat()
    # Written into the manifest so the split is self-describing wherever it ends
    # up. The worker verifies each transferred file against these before letting
    # it replace anything on the cluster, and a file that was truncated in
    # transit or edited on arrival stops being trainable rather than becoming a
    # silently shorter training set.
    checksums = {
        TRAIN_FILENAME: sha256_file(train_path),
        VALIDATION_FILENAME: sha256_file(validation_path),
    }
    source_export_timestamp = read_source_export_timestamp(
        course_id,
        export_root=export_root,
    )
    dataset_version = f"{course_id}-approved-split-seed{split_seed}-n{len(records)}"

    manifest = {
        "courseId": course_id,
        "datasetVersion": dataset_version,
        "sourceFile": str(source_path),
        "sourceExportTimestamp": source_export_timestamp,
        "createdAt": created,
        "splitSeed": split_seed,
        "totalExamples": len(records),
        "trainExamples": len(train),
        "validationExamples": len(validation),
        "trainFile": str(train_path),
        "validationFile": str(validation_path),
        # SHA-256 of each written file, keyed by file name. Additive: readers
        # that predate this key ignore it, and `validate_course_export_dir` on
        # the cluster does not require it.
        "checksums": checksums,
        "checksumAlgorithm": "sha256",
    }
    write_json(manifest_path, manifest)

    return {
        "courseId": course_id,
        "manifest": manifest,
        "trainExamples": len(train),
        "validationExamples": len(validation),
        "totalExamples": len(records),
        "splitSeed": split_seed,
        "files": {
            "trainJsonl": str(train_path),
            "validationJsonl": str(validation_path),
            "manifestJson": str(manifest_path),
            "sourceJsonl": str(source_path),
        },
    }
