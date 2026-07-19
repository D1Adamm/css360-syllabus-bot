#!/usr/bin/env python3
"""Export approved Firebase seedExamples for a course to local JSONL + metadata.

Usage:
  cd backend
  .venv/bin/python scripts/export_approved_seeds.py css-360-winter-2026-a7rp
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

from app.firebase_seeds import (  # noqa: E402
    FirebaseConfigurationError,
    fetch_course_seed_examples,
)
from app.seed_export import export_approved_seeds  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("course_id", help="Course id, e.g. css-360-winter-2026-a7rp")
    args = parser.parse_args()

    try:
        payload = await fetch_course_seed_examples(args.course_id)
    except FirebaseConfigurationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    seeds = []
    for seed_id, raw in (payload or {}).items():
        if isinstance(raw, dict):
            seeds.append({**raw, "id": raw.get("id") or seed_id})

    summary = export_approved_seeds(course_id=args.course_id, seeds=seeds)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
