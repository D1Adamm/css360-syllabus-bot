"""Repair stale starterSeedGeneration records from the real seed count.

Reads only `courses/{courseId}/seedExamples` and
`courses/{courseId}/metadata/starterSeedGeneration`, and writes only the latter.
It never creates, edits, or deletes a seed, and never runs generation — so it is
safe on a live course and cannot change what a professor has to review.

Usage:
    .venv/bin/python scripts/reconcile_starter_generation.py <courseId> [...] [--dry-run]

    --dry-run          report what would change; write nothing
    --target-count N   fallback target when the course has no recorded one
                       (an existing targetCount always wins)

Needs FIREBASE_DATABASE_URL (and FIREBASE_AUTH_TOKEN if rules require it) in
backend/.env, exactly like the running backend.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.course_id import is_valid_course_id  # noqa: E402
from app.firebase_metadata import (  # noqa: E402
    count_course_seed_examples,
    read_starter_seed_generation,
    reconcile_starter_seed_generation,
    resolve_reconciled_starter_status,
    resolve_target_count,
)

DEFAULT_TARGET_COUNT = 50


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


async def inspect_course(course_id: str, fallback_target: int) -> dict[str, Any]:
    """What the record says, what the course holds, and what it should say."""
    actual_count = await count_course_seed_examples(course_id)
    if actual_count is None:
        return {"courseId": course_id, "readable": False}

    block = await read_starter_seed_generation(course_id) or {}
    target_count = await resolve_target_count(course_id, fallback_target)
    stored_status = block.get("status")

    # A failed record stays failed: this tool reconciles counts, it does not
    # overturn an operator-visible failure that nobody has looked at yet.
    next_status = (
        "failed"
        if stored_status == "failed"
        else resolve_reconciled_starter_status(
            target_count=target_count, actual_count=actual_count
        )
    )

    return {
        "courseId": course_id,
        "readable": True,
        "actualCount": actual_count,
        "targetCount": target_count,
        "storedStatus": stored_status,
        "storedFinalCount": _as_int(block.get("finalCount")),
        "storedSavedCount": _as_int(block.get("savedCount")),
        "nextStatus": next_status,
        "stale": (
            stored_status != next_status
            or _as_int(block.get("finalCount")) != actual_count
            or _as_int(block.get("savedCount")) != actual_count
        ),
    }


def print_report(report: dict[str, Any], *, dry_run: bool) -> None:
    course_id = report["courseId"]
    if not report["readable"]:
        print(f"{course_id}: could not read seedExamples; skipped.")
        return

    print(
        f"{course_id}: actual seeds {report['actualCount']} / "
        f"target {report['targetCount']}"
    )
    print(
        f"  stored : status={report['storedStatus']} "
        f"finalCount={report['storedFinalCount']} "
        f"savedCount={report['storedSavedCount']}"
    )
    if not report["stale"]:
        print("  already consistent; nothing to do.")
        return

    verb = "would write" if dry_run else "wrote"
    print(
        f"  {verb} : status={report['nextStatus']} "
        f"finalCount={report['actualCount']} "
        f"savedCount={report['actualCount']}"
    )


async def run(course_ids: Sequence[str], *, dry_run: bool, target: int) -> int:
    changed = 0
    for course_id in course_ids:
        report = await inspect_course(course_id, target)
        if report["readable"] and report["stale"] and not dry_run:
            await reconcile_starter_seed_generation(
                course_id,
                target_count=target,
                force_status=(
                    "failed" if report["storedStatus"] == "failed" else None
                ),
            )
        print_report(report, dry_run=dry_run)
        if report["readable"] and report["stale"]:
            changed += 1

    action = "would be reconciled" if dry_run else "reconciled"
    print(f"\n{changed} of {len(course_ids)} course(s) {action}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconcile starterSeedGeneration with the actual seedExamples "
            "count. Never creates, edits, deletes, or regenerates seeds."
        )
    )
    parser.add_argument("course_ids", nargs="+", help="Course ids to reconcile.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing anything.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=DEFAULT_TARGET_COUNT,
        help=(
            "Fallback target when the course records none. "
            f"Default {DEFAULT_TARGET_COUNT}. A recorded targetCount always wins."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    invalid = [c for c in args.course_ids if not is_valid_course_id(c)]
    if invalid:
        print(f"Invalid course id(s): {', '.join(invalid)}", file=sys.stderr)
        return 1

    return asyncio.run(
        run(args.course_ids, dry_run=args.dry_run, target=args.target_count)
    )


if __name__ == "__main__":
    raise SystemExit(main())
