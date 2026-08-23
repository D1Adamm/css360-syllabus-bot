"""LEGACY MIGRATION TOOLING — import an old Firebase snapshot into PostgreSQL.

Kept deliberately, and deliberately isolated. Firebase is no longer a runtime
dependency of anything in this repository: the application, the backend and the
training worker all read and write PostgreSQL only, and none of them needs
Firebase configuration to start. This script is the one exception, and it does
not need Firebase either — it reads a JSON file that was exported before the
migration and writes to PostgreSQL. It talks to no network service, and nothing
in `app.main` imports it.

It exists so the archived snapshot can be replayed if the original import ever
has to be redone or audited. Do not wire it into a live code path.

Usage:
    .venv/bin/python scripts/import_firebase_snapshot.py <snapshot.json> [--dry-run]

Export the snapshot from the Firebase console (Realtime Database -> `courses`
-> Export JSON), or with the CLI:

    firebase database:get /courses -o courses.json

Behaviour:
  - `--dry-run` parses and validates the snapshot and prints the counts it
    would write, without opening a database connection.
  - A real run writes every table inside one transaction, so a failure
    anywhere leaves PostgreSQL exactly as it was.
  - Writes are upserts keyed on the Firebase ids, so the script is safe to
    re-run: existing rows are refreshed from the snapshot and rows absent from
    the snapshot are left alone. Nothing is ever deleted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import DatabaseConfigurationError, connect  # noqa: E402
from app.firebase_snapshot import (  # noqa: E402
    SnapshotError,
    SnapshotPlan,
    parse_snapshot,
)


class TableSpec:
    """How one plan list becomes rows in one table."""

    def __init__(
        self,
        *,
        table: str,
        columns: Sequence[str],
        conflict_columns: Sequence[str],
        json_columns: Sequence[str] = (),
    ) -> None:
        self.table = table
        self.columns = list(columns)
        self.conflict_columns = list(conflict_columns)
        self.json_columns = set(json_columns)

    def upsert_sql(self) -> str:
        column_list = ", ".join(self.columns)
        placeholders = ", ".join(f"%({column})s" for column in self.columns)
        conflict_list = ", ".join(self.conflict_columns)
        updatable = [
            column for column in self.columns if column not in self.conflict_columns
        ]
        if not updatable:
            action = "DO NOTHING"
        else:
            assignments = ", ".join(
                f"{column} = EXCLUDED.{column}" for column in updatable
            )
            action = f"DO UPDATE SET {assignments}"

        return (
            f"INSERT INTO {self.table} ({column_list}) VALUES ({placeholders}) "
            f"ON CONFLICT ({conflict_list}) {action}"
        )


# Ordered so foreign keys to courses(course_id) are always satisfied.
TABLE_SPECS: list[tuple[str, TableSpec]] = [
    (
        "courses",
        TableSpec(
            table="courses",
            columns=[
                "course_id",
                "name",
                "title",
                "term",
                "instructor_name",
                "created_at",
                "syllabus_status",
                "syllabus_file_name",
                "syllabus_type",
                "chunk_count",
            ],
            conflict_columns=["course_id"],
        ),
    ),
    (
        "starter_seed_generation",
        TableSpec(
            table="starter_seed_generation",
            columns=[
                "course_id",
                "status",
                "target_count",
                "final_count",
                "saved_count",
                "failed_to_save_count",
                "error",
                "started_at",
                "completed_at",
            ],
            conflict_columns=["course_id"],
        ),
    ),
    (
        "seed_examples",
        TableSpec(
            table="seed_examples",
            columns=[
                "seed_id",
                "course_id",
                "instruction",
                "response",
                "category",
                "source_section",
                "difficulty",
                "directly_answered",
                "origin",
                "notes",
                "created_at",
                "status",
                "question_type",
                "source_chunk_ids",
                "validation",
                "review_status",
                "review_notes",
                "reviewed_at",
                "fact_id",
                "evidence_quote",
                "normalized_question_key",
                "original_question",
                "original_answer",
                "was_edited",
            ],
            conflict_columns=["course_id", "seed_id"],
            json_columns=["source_chunk_ids", "validation"],
        ),
    ),
    (
        "evaluations",
        TableSpec(
            table="evaluations",
            columns=[
                "evaluation_id",
                "course_id",
                "comparison_id",
                "most_accurate",
                "most_helpful",
                "most_concise",
                "best_grounded",
                "preferred_model",
                "hallucination_flags",
                "comment",
                "created_at",
                "run_id",
                "question_text",
            ],
            conflict_columns=["course_id", "evaluation_id"],
            json_columns=["hallucination_flags"],
        ),
    ),
    (
        "course_models",
        TableSpec(
            table="course_models",
            columns=["course_id", "current_version"],
            conflict_columns=["course_id"],
        ),
    ),
    (
        "course_model_versions",
        TableSpec(
            table="course_model_versions",
            columns=[
                "course_id",
                "version",
                "base_model",
                "training_example_count",
                "status",
                "deployment",
                "artifact_ref",
                "created_at",
                "updated_at",
                "notes",
            ],
            conflict_columns=["course_id", "version"],
        ),
    ),
    (
        "model_requests",
        TableSpec(
            table="model_requests",
            columns=[
                "course_id",
                "status",
                "requested_at",
                "updated_at",
                "approved_example_count",
                "failure_message",
                "preparation",
                "preparation_error",
                "training",
                "launch_error",
                "current_run_id",
            ],
            conflict_columns=["course_id"],
            json_columns=["preparation", "training"],
        ),
    ),
    (
        "training_runs",
        TableSpec(
            table="training_runs",
            columns=[
                "run_id",
                "course_id",
                "mode",
                "state",
                "enqueued_at",
                "updated_at",
                "dataset_ref",
                "approved_example_count",
                "train_examples",
                "validation_examples",
                "attempt",
                "job_id",
                "claim_owner",
                "claim_claimed_at",
                "claim_expires_at",
                "error",
            ],
            conflict_columns=["course_id", "run_id"],
        ),
    ),
]

# Printed labels, in the order the task specifies.
SUMMARY_LABELS: list[tuple[str, str]] = [
    ("Courses", "courses"),
    ("Starter-generation records", "starter_seed_generation"),
    ("Seeds", "seed_examples"),
    ("Evaluations", "evaluations"),
    ("Models", "course_models"),
    ("Model versions", "course_model_versions"),
    ("Model requests", "model_requests"),
    ("Training runs", "training_runs"),
]


def load_snapshot(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SnapshotError(f"Could not read snapshot {path}: {exc}") from exc

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"Snapshot {path} is not valid JSON: {exc}") from exc


def _bind_row(row: dict[str, Any], spec: TableSpec, json_wrapper: Any) -> dict[str, Any]:
    """Wrap JSONB values so psycopg sends objects/arrays rather than text."""
    return {
        column: (
            json_wrapper(row[column])
            if column in spec.json_columns and row[column] is not None
            else row[column]
        )
        for column in spec.columns
    }


def write_plan(plan: SnapshotPlan) -> None:
    """Write every table in one transaction."""
    from psycopg.types.json import Json

    with connect() as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                for attribute, spec in TABLE_SPECS:
                    rows = getattr(plan, attribute)
                    if not rows:
                        continue
                    cursor.executemany(
                        spec.upsert_sql(),
                        [_bind_row(row, spec, Json) for row in rows],
                    )


def print_summary(plan: SnapshotPlan) -> None:
    counts = plan.counts()
    for label, key in SUMMARY_LABELS:
        print(f"{label}: {counts[key]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import a Firebase courses.json snapshot into PostgreSQL. "
            "Upserts on Firebase ids; never deletes."
        ),
    )
    parser.add_argument(
        "snapshot",
        type=Path,
        help="Path to the exported Firebase courses.json snapshot.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate the snapshot, print planned counts, write nothing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        plan = parse_snapshot(load_snapshot(args.snapshot))
    except SnapshotError as exc:
        print(f"Snapshot error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"Dry run: {args.snapshot} parsed successfully. No database writes.")
        print_summary(plan)
        return 0

    try:
        write_plan(plan)
    except DatabaseConfigurationError as exc:
        print(f"Database error: {exc}", file=sys.stderr)
        return 1

    print_summary(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
