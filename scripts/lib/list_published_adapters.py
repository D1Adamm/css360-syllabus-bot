#!/usr/bin/env python3
"""Print the course adapters published under a serving root.

Stdlib only, Python 3.9 compatible.

Used by `serve.slurm` and `status_finetuned_service.sh` so an operator can see,
in the job log and in the status output, which courses the service will actually
be able to answer for. A serving job that starts with nothing published is a
supported state — every course request returns a clear 409 — but it should not
be a surprise discovered by a student.

    $ python3 scripts/lib/list_published_adapters.py \\
        /gpfs/projects/simswe/$USER/training_outputs/serving
    css-350-spring-2026-n3h9  current=v1  published=v1
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

COURSE_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
VERSION_RE = re.compile(r"^v[0-9]+$")
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin", "adapter_model.pt")


def adapter_is_loadable(path: Path) -> bool:
    if not path.is_dir() or not (path / ADAPTER_CONFIG).is_file():
        return False
    return any((path / name).is_file() for name in ADAPTER_WEIGHTS)


def read_current_version(course_dir: Path) -> Optional[str]:
    pointer = course_dir / "current.json"
    if not pointer.is_file():
        return None
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    version = payload.get("version") if isinstance(payload, dict) else None
    if isinstance(version, str) and VERSION_RE.fullmatch(version.strip()):
        return version.strip()
    return None


def published_courses(serving_root: Path) -> List[Dict[str, Any]]:
    if not serving_root.is_dir():
        return []

    found: List[Dict[str, Any]] = []
    for entry in sorted(serving_root.iterdir()):
        if not entry.is_dir() or not COURSE_ID_RE.fullmatch(entry.name):
            continue
        versions = sorted(
            (
                child.name
                for child in entry.iterdir()
                if child.is_dir()
                and VERSION_RE.fullmatch(child.name)
                and adapter_is_loadable(child / "adapter")
            ),
            key=lambda name: int(name[1:]),
        )
        if not versions:
            continue
        found.append(
            {
                "courseId": entry.name,
                "versions": versions,
                "currentVersion": read_current_version(entry) or versions[-1],
            }
        )
    return found


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: list_published_adapters.py <serving-root> [--json]", file=sys.stderr)
        return 2

    root = Path(args[0])
    courses = published_courses(root)

    if "--json" in args[1:]:
        print(json.dumps({"servingRoot": str(root), "courses": courses}))
        return 0

    if not courses:
        print(f"No published course adapters under {root}")
        print(
            "Publish one with: ./training/promote_qlora_adapter.sh "
            "--course <courseId> --version <vN> <adapter-path>"
        )
        return 0

    for course in courses:
        print(
            "{0}  current={1}  published={2}".format(
                course["courseId"],
                course["currentVersion"],
                ",".join(course["versions"]),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
