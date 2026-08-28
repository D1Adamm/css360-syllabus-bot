#!/usr/bin/env python3
"""Register a trained course model in the per-course model registry.

The registry lives in PostgreSQL, in ``course_models`` and
``course_model_versions``, alongside that course's metadata, examples, and
evaluations. It records that a model was produced for a course and what it was
produced from. It is the only thing the UI consults to answer "does this course
have a model" — the shared inference service's health endpoint cannot answer
that, because it reports what is loaded, not whose.

This script runs on the cluster, which is not on the database's network, so it
writes through the backend's ``/api/training-queue/courses/{courseId}/
model-versions`` endpoint rather than holding a connection of its own. Set
TRAINING_API_BASE_URL and TRAINING_WORKER_TOKEN, the same two variables the
queue runner uses.

Two facts are stored separately and must stay that way:

  status      training/artifact state. Durable. A promoted adapter stays
              ``ready`` whether or not anything is serving it.
  deployment  whether that model is currently being served. Changes when a
              service starts or stops; says nothing about existence.

The artifact reference is stored relative (``css-360-qlora/adapter``) rather
than as the absolute promote-script destination, which embeds a cluster home
directory and a username. Admin surfaces show the reference; professor surfaces
never do.

The course a version lands on is the one named in the URL, and the rows are
keyed by it. There is no request this script can make that attaches a CSS 360
adapter to CSS 350.

This script only writes the record. It does not train, promote, or deploy
anything.

Usage:
  python scripts/register_course_model.py \\
      --course-id css-360-winter-2026-a7rp \\
      --base-model meta-llama/Llama-3.2-3B-Instruct \\
      --training-examples 48 \\  # the train split, not the approved count
      --artifact-ref css-360-qlora/adapter \\
      --status ready --deployment offline

  # Show what would be written without writing it:
  python scripts/register_course_model.py ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "lib"))

from training_queue import (  # noqa: E402  (path set above)
    TrainingQueueError,
    build_queue,
    load_env_file,
)

VERSION_PATTERN = re.compile(r"^v(\d+)$")

MODEL_STATUSES = ("ready", "training", "failed")
DEPLOYMENT_STATUSES = ("online", "offline", "unknown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-id", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--training-examples",
        type=int,
        required=True,
        help=(
            "Examples actually passed into training — the TRAIN split, not the "
            "approved count it came from. For a 42-approved course split 37/5 "
            "this is 37. CSS 350 v1 holds 42 because it was registered by hand "
            "before this was written down; that row is left alone."
        ),
    )
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
    parser.add_argument(
        "--run-id",
        help=(
            "Training run this model came from. When given with --status ready, "
            "the run is marked succeeded and the model request becomes ready."
        ),
    )
    parser.add_argument("--notes", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the record that would be written and exit.",
    )
    return parser.parse_args()


def main() -> int:
    load_env_file(PROJECT_ROOT)
    args = parse_args()

    if args.training_examples < 0:
        raise SystemExit("--training-examples must not be negative")

    artifact_ref = args.artifact_ref.strip()
    if not artifact_ref:
        raise SystemExit("--artifact-ref must not be empty")
    if artifact_ref.startswith("/"):
        raise SystemExit(
            "--artifact-ref must be relative. An absolute path embeds a cluster "
            "home directory and a username, which must not be stored."
        )
    if args.version and not VERSION_PATTERN.match(args.version):
        raise SystemExit(f"Version must look like v1, v2, …: {args.version!r}")

    record = {
        "baseModel": args.base_model,
        "trainingExampleCount": args.training_examples,
        "status": args.status,
        "deployment": args.deployment,
        "artifactRef": artifact_ref,
    }
    if args.version:
        record["version"] = args.version
    if args.notes:
        record["notes"] = args.notes
    if args.run_id:
        record["runId"] = args.run_id

    print(f"course:  {args.course_id}")
    print(json.dumps(record, indent=2))

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return 0

    try:
        queue = build_queue()
        result = queue.register_model_version(
            args.course_id,
            base_model=args.base_model,
            training_example_count=args.training_examples,
            artifact_ref=artifact_ref,
            status=args.status,
            deployment=args.deployment,
            version=args.version,
            notes=args.notes,
            run_id=args.run_id,
        )
    except TrainingQueueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    version = result.get("version")
    current = result.get("currentVersion")
    request_status = result.get("requestStatus")

    print(f"\nRegistered {args.course_id} model {version} (current: {current}).")
    if request_status:
        print(f"modelRequest.status={request_status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
