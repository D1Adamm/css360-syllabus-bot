#!/usr/bin/env python3
"""Import staff-authored syllabus Q&A seeds into one course from a JSON file.

    PYTHONPATH=backend backend/.venv/bin/python scripts/import_manual_seeds.py \\
        --course css360d-fall-2026-q0ne \\
        --file data/css360d_manual_seeds.json [--approved] [--dry-run] [--yes]

The file is a JSON list of objects with `question` and `answer` (required) and
`category` and `sourceSection` (optional). Seeds are stored with the existing
staff-authored origin (`prototype`), pending review unless `--approved` is
given. Questions already stored for the course are skipped, so running the
same file twice inserts nothing the second time.

Nothing here generates: no Ollama, no starter examples, no training, and no
change to the syllabus or to any other course. The logic lives in
`backend/app/manual_seed_import.py`; this is its command line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.manual_seed_import import ImportPlan, run_import


def _confirm(plan: ImportPlan) -> bool:
    answer = input(
        f"Insert {len(plan.to_insert)} seed(s) into {plan.course_id} "
        f"as {plan.review_status}? [y/N] "
    )
    return answer.strip().lower() in {"y", "yes"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--course", required=True, help="Course id, e.g. css360d-fall-2026-q0ne")
    parser.add_argument("--file", required=True, type=Path, help="JSON list of seeds")
    parser.add_argument(
        "--approved",
        action="store_true",
        help="Store as approved (you authored and checked them) instead of pending review",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and report; write nothing")
    parser.add_argument("--yes", action="store_true", help="Do not ask for confirmation")
    args = parser.parse_args(argv)

    return run_import(
        args.course,
        args.file,
        approved=args.approved,
        dry_run=args.dry_run,
        confirm=None if args.yes else _confirm,
    )


if __name__ == "__main__":
    sys.exit(main())
