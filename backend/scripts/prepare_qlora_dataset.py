#!/usr/bin/env python3
"""Prepare the approved QLoRA train/validation export for a course.

Uses existing export + split logic (does not reimplement them).

Usage:
  cd backend
  .venv/bin/python scripts/prepare_qlora_dataset.py css-360-winter-2026-a7rp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config as _config  # noqa: F401,E402
from app.config import load_backend_env  # noqa: E402

load_backend_env()

from app.course_id import assert_valid_course_id  # noqa: E402
from app.firebase_seeds import (  # noqa: E402
    FirebaseConfigurationError,
    fetch_course_seed_examples,
)
from app.seed_export import export_approved_seeds  # noqa: E402
from app.seed_split import TrainingSplitError, prepare_training_split  # noqa: E402


async def _run(course_id: str) -> dict:
    safe_course_id = assert_valid_course_id(course_id)

    try:
        payload = await fetch_course_seed_examples(safe_course_id)
    except FirebaseConfigurationError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    seeds: list[dict] = []
    for seed_id, raw in (payload or {}).items():
        if isinstance(raw, dict):
            seeds.append({**raw, "id": raw.get("id") or seed_id})

    export_summary = export_approved_seeds(course_id=safe_course_id, seeds=seeds)
    approved_count = int(export_summary.get("approvedCount") or 0)
    if approved_count < 2:
        raise SystemExit(
            "ERROR: Need at least 2 approved seeds to create a train/validation split "
            f"(approvedCount={approved_count}). Approve more seeds and retry."
        )

    try:
        split_summary = prepare_training_split(course_id=safe_course_id)
    except TrainingSplitError as exc:
        raise SystemExit(f"ERROR: training split failed: {exc}") from exc
    except FileNotFoundError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    return {
        "courseId": safe_course_id,
        "approvedCount": approved_count,
        "trainExamples": split_summary["trainExamples"],
        "validationExamples": split_summary["validationExamples"],
        "totalExamples": split_summary["totalExamples"],
        "splitSeed": split_summary["splitSeed"],
        "files": {
            **export_summary.get("files", {}),
            **split_summary.get("files", {}),
        },
        "nextSteps": [
            f"./scripts/sync_training_data_to_tillicum.sh {safe_course_id}",
            (
                "On Tillicum: "
                f"./training/start_qlora_training.sh --course {safe_course_id} --smoke"
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("course_id", help="Course id, e.g. css-360-winter-2026-a7rp")
    args = parser.parse_args()
    result = asyncio.run(_run(args.course_id))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
