#!/usr/bin/env python3
"""Register a trained course model in the per-course model registry.

The registry lives at ``courses/{courseId}/model`` in the Realtime Database,
alongside that course's metadata, examples, and evaluations. It records that a
model was produced for a course and what it was produced from. It is the only
thing the UI consults to answer "does this course have a model" — the shared
inference service's health endpoint cannot answer that, because it reports what
is loaded, not whose.

Two facts are stored separately and must stay that way:

  status      training/artifact state. Durable. A promoted adapter stays
              ``ready`` whether or not anything is serving it.
  deployment  whether that model is currently being served. Changes when a
              service starts or stops; says nothing about existence.

The artifact reference is stored relative (``css-360-qlora/adapter``) rather
than as the absolute promote-script destination, which embeds a cluster home
directory and a username. Admin surfaces show the reference; professor surfaces
never do.

This script only writes the record. It does not train, promote, or deploy
anything.

Usage:
  python scripts/register_course_model.py \\
      --course-id css-360-winter-2026-a7rp \\
      --base-model meta-llama/Llama-3.2-3B-Instruct \\
      --training-examples 54 \\
      --artifact-ref css-360-qlora/adapter \\
      --status ready --deployment offline

  # Show what would be written without writing it:
  python scripts/register_course_model.py ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

COURSE_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
VERSION_PATTERN = re.compile(r"^v(\d+)$")

MODEL_STATUSES = ("ready", "training", "failed")
DEPLOYMENT_STATUSES = ("online", "offline", "unknown")


def load_env_file() -> None:
    """Read .env.local / .env the same way the other scripts do."""
    for name in (".env.local", ".env"):
        path = PROJECT_ROOT / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def assert_valid_course_id(course_id: str) -> str:
    """Mirror the backend's validation so a bad id cannot reach a path."""
    candidate = (course_id or "").strip()
    if (
        not candidate
        or not COURSE_ID_PATTERN.match(candidate)
        or ".." in candidate
        or "/" in candidate
    ):
        raise SystemExit(f"Invalid courseId: {course_id!r}")
    return candidate


def firebase_database_url() -> str:
    url = (
        os.environ.get("FIREBASE_DATABASE_URL")
        or os.environ.get("VITE_FIREBASE_DATABASE_URL")
        or ""
    ).strip().rstrip("/")
    if not url:
        raise SystemExit(
            "Missing Firebase database URL. Set FIREBASE_DATABASE_URL or "
            "VITE_FIREBASE_DATABASE_URL in your environment or .env.local."
        )
    return url


def request_url(path: str) -> str:
    url = f"{firebase_database_url()}/{path}.json"
    token = (os.environ.get("FIREBASE_AUTH_TOKEN") or "").strip()
    return f"{url}?auth={token}" if token else url


def http_json(url: str, *, method: str, payload: Any | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Firebase returned HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach Firebase: {exc.reason}") from exc

    return json.loads(body) if body and body != "null" else None


def next_version(existing: dict[str, Any] | None) -> str:
    """`v1`, then `v2`, … — the smallest scheme that still orders."""
    if not existing:
        return "v1"

    versions = existing.get("versions")
    if not isinstance(versions, dict) or not versions:
        return "v1"

    highest = 0
    for key in versions:
        match = VERSION_PATTERN.match(str(key))
        if match:
            highest = max(highest, int(match.group(1)))
    return f"v{highest + 1}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-id", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--training-examples", type=int, required=True)
    parser.add_argument(
        "--artifact-ref",
        required=True,
        help="Relative artifact reference, e.g. css-360-qlora/adapter",
    )
    parser.add_argument("--status", choices=MODEL_STATUSES, default="ready")
    parser.add_argument("--deployment", choices=DEPLOYMENT_STATUSES, default="offline")
    parser.add_argument(
        "--version",
        help="Override the version key. Defaults to the next unused vN.",
    )
    parser.add_argument("--notes", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the record that would be written and exit.",
    )
    return parser.parse_args()


def main() -> int:
    load_env_file()
    args = parse_args()

    course_id = assert_valid_course_id(args.course_id)
    if args.training_examples < 0:
        raise SystemExit("--training-examples must not be negative")
    if not args.artifact_ref.strip():
        raise SystemExit("--artifact-ref must not be empty")
    if args.artifact_ref.strip().startswith("/"):
        raise SystemExit(
            "--artifact-ref must be relative. An absolute path embeds a cluster "
            "home directory and a username, which must not be stored."
        )

    model_path = f"courses/{course_id}/model"
    existing = http_json(request_url(model_path), method="GET")

    version_key = args.version or next_version(existing)
    if not VERSION_PATTERN.match(version_key):
        raise SystemExit(f"Version must look like v1, v2, …: {version_key!r}")

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    record = {
        "version": version_key,
        "baseModel": args.base_model,
        "trainingExampleCount": args.training_examples,
        "status": args.status,
        "deployment": args.deployment,
        "artifactRef": args.artifact_ref.strip(),
        "createdAt": now,
        "updatedAt": now,
    }
    if args.notes:
        record["notes"] = args.notes

    print(f"course:  {course_id}")
    print(f"path:    {model_path}/versions/{version_key}")
    print(json.dumps(record, indent=2))

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return 0

    # PATCH so an existing version history is preserved rather than replaced.
    http_json(
        request_url(f"{model_path}/versions/{version_key}"),
        method="PATCH",
        payload=record,
    )
    http_json(
        request_url(model_path),
        method="PATCH",
        payload={"currentVersion": version_key},
    )

    print(f"\nRegistered {course_id} model {version_key} (current).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
