#!/usr/bin/env python3
"""Delete exactly what a retention plan proposed, and only if it is still safe.

Stdlib only, Python 3.9 compatible.

Separate from `retention.py` on purpose: that module produces plans and removes
nothing, so it can be read, tested, and run against a live cluster without any
possibility of loss. This is the one file that deletes, and it is short enough
to read in full before trusting it.

Every path is re-checked with `retention.is_deletion_safe` immediately before it
is removed, rather than trusted from the plan. A plan can be printed, read and
acted on minutes later, and in between somebody may have published an adapter
from a run that was a candidate when the plan was made.

    python3 scripts/lib/apply_retention_plan.py \\
        --plan /tmp/plan.json --outputs-root /gpfs/.../training_outputs
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent


def load_retention() -> Any:
    return SourceFileLoader("retention", str(HERE / "retention.py")).load_module()


def apply_plan(
    plan: Dict[str, Any],
    outputs_root: Path,
    *,
    retention: Any,
    remove: Any = shutil.rmtree,
) -> Dict[str, Any]:
    """Remove the plan's candidates. Returns what happened, per path.

    `remove` is injectable so the tests can assert exactly which paths would be
    deleted without deleting anything.
    """
    removed: List[str] = []
    skipped: List[Dict[str, str]] = []
    freed = 0

    for item in plan.get("candidates") or []:
        path = Path(str(item.get("path") or ""))

        if not retention.is_deletion_safe(path, outputs_root):
            skipped.append({"path": str(path), "reason": "no longer safe to delete"})
            continue
        if not path.exists():
            skipped.append({"path": str(path), "reason": "already gone"})
            continue

        try:
            remove(path)
        except OSError as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
            continue

        removed.append(str(path))
        try:
            freed += int(item.get("bytes") or 0)
        except (TypeError, ValueError):
            pass

    return {"removed": removed, "skipped": skipped, "freedBytes": freed}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="Path to a plan JSON file.")
    parser.add_argument("--outputs-root", required=True)
    args = parser.parse_args(argv)

    retention = load_retention()
    outputs_root = Path(args.outputs_root)

    try:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print("Could not read the plan: {0}".format(exc), file=sys.stderr)
        return 2
    if not isinstance(plan, dict):
        print("The plan is not an object.", file=sys.stderr)
        return 2

    result = apply_plan(plan, outputs_root, retention=retention)

    for path in result["removed"]:
        print("removed {0}".format(path))
    for entry in result["skipped"]:
        print("SKIPPED {0} ({1})".format(entry["path"], entry["reason"]))

    print()
    print(
        "Removed {0} item(s), freeing {1}.".format(
            len(result["removed"]), retention.human_bytes(result["freedBytes"])
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
