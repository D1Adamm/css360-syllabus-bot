#!/usr/bin/env python3
"""Import authored behaviour examples (abstention, false-premise) as seeds.

    .venv/bin/python scripts/import_behaviour_seeds.py \\
        --course-id css-360-winter-2026-a7rp \\
        --file ../training/behaviour_seeds/css-360-winter-2026-a7rp.jsonl \\
        [--approve] [--dry-run]

Reads one JSONL record per line: `question`, `response`, `questionType`
(`unanswerable` or `false_premise`), and optionally `category`,
`sourceSection`, `notes`. Each becomes a seed row with origin `authored`.
Without `--approve` the seeds wait in the review queue; with it they are
approved on import with a note saying so. A record whose question already
exists for the course, ignoring case and punctuation, is skipped.

Writes to the database named by DATABASE_URL in backend/.env. `--dry-run`
validates and reports without writing anything.

The split renders each of these as a grounded example over what the
production retriever returns for its question, and drops any whose response
cites a figure those excerpts do not contain (listed under `groundedSkipped`
in manifest.json). Check that list after preparing the split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.config import load_backend_env  # noqa: E402

load_backend_env()

from app import db_seeds  # noqa: E402
from app.behaviour_seeds import (  # noqa: E402
    BehaviourSeedError,
    behaviour_seed,
    load_behaviour_records,
    normalized_question,
)
from app.course_id import assert_valid_course_id  # noqa: E402
from app.db import db_connection  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-id", required=True)
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--approve", action="store_true", help="Approve on import")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report only")
    args = parser.parse_args(argv)

    try:
        course_id = assert_valid_course_id(args.course_id)
        records = load_behaviour_records(args.file)
    except (ValueError, BehaviourSeedError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    seeds = [behaviour_seed(record, approve=args.approve) for record in records]
    kinds = {}
    for seed in seeds:
        kinds[seed["questionType"]] = kinds.get(seed["questionType"], 0) + 1
    print(f"{len(seeds)} valid records in {args.file.name}: {kinds}")

    with db_connection() as connection:
        existing = {
            normalized_question(seed.get("instruction") or seed.get("question") or "")
            for seed in db_seeds.list_seeds(connection, course_id)
        }
        created = skipped = 0
        for record, seed in zip(records, seeds):
            key = normalized_question(seed["instruction"])
            if key in existing:
                print(f"  skip line {record['line']}: already present: {seed['instruction'][:70]}")
                skipped += 1
                continue
            if args.dry_run:
                print(f"  would create [{seed['questionType']}]: {seed['instruction'][:70]}")
            else:
                db_seeds.create_seed(connection, course_id, seed)
                print(f"  created [{seed['questionType']}]: {seed['instruction'][:70]}")
            existing.add(key)
            created += 1
        if args.dry_run:
            connection.rollback()

    verb = "would create" if args.dry_run else "created"
    print(f"{verb} {created}, skipped {skipped}, review status: "
          f"{'approved' if args.approve else 'generated (pending review)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
