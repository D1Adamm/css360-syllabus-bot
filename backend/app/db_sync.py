"""Keep PostgreSQL current for the frontend while Firebase still owns some writes.

After the frontend cutover, the browser reads only PostgreSQL. Several backend
flows still write Firebase, and must keep doing so:

  - starter generation and top-up persist seeds through `persist_accepted_seeds`
  - seed review goes through `apply_seed_review`, and `export_approved_seeds`
    reads approved seeds back out of Firebase to build the training set

If those wrote Firebase and nothing else, a professor would run a top-up, the
page would reload from PostgreSQL, and the new examples would not be there.

So this module carries state across in both directions, and each direction
exists for a different reason:

  - `sync_course_from_firebase` pulls a course into PostgreSQL after an
    operational route wrote Firebase. This is what stops the UI going stale.
  - `mirror_seed_*_to_firebase` pushes a `/api/db` seed write back to Firebase,
    so the training export and the generation deduplicator — both of which read
    Firebase — keep seeing the complete picture.

Everything here is best-effort. A sync failure must never fail the request that
triggered it: the write it is following has already succeeded and is durable in
whichever store owns it. A failed sync leaves PostgreSQL briefly behind, which
the next successful operation repairs; a raised exception would instead tell a
professor their seeds were not saved when they were.

Deliberately reuses `firebase_snapshot`'s mappers rather than restating the
camelCase -> snake_case translation. That mapping is already the tested one, and
a second copy would be a second thing to keep in step with the schema.
"""

from __future__ import annotations

import logging
from typing import Any

from app.course_id import assert_valid_course_id
from app.db import db_connection
from app.db_courses import (
    create_course,
    course_exists,
    update_course,
    upsert_starter_seed_generation,
)
from app.db_seeds import get_seed, create_seed, update_seed
from app.firebase_snapshot import (
    SnapshotError,
    map_course,
    map_seed_example,
    map_starter_seed_generation,
)

logger = logging.getLogger(__name__)


def _seed_api_record(course_id: str, seed_id: str, raw: Any) -> dict[str, Any] | None:
    """One Firebase seed as the camelCase record the repositories accept.

    Goes through the importer's mapper first so a malformed record is rejected
    the same way a snapshot import would reject it, then back out to API names.
    A seed that cannot be mapped is skipped rather than failing the whole sync —
    one bad record must not strand the other forty.
    """
    try:
        row = map_seed_example(course_id, seed_id, raw)
    except SnapshotError as exc:
        logger.warning("Skipping unmappable seed %s/%s: %s", course_id, seed_id, exc)
        return None

    return {
        "id": row["seed_id"],
        "instruction": row["instruction"],
        "response": row["response"],
        "category": row["category"],
        "sourceSection": row["source_section"],
        "difficulty": row["difficulty"],
        "directlyAnswered": row["directly_answered"],
        "origin": row["origin"],
        "notes": row["notes"],
        "createdAt": row["created_at"].isoformat() if row["created_at"] else None,
        "status": row["status"],
        "questionType": row["question_type"],
        "sourceChunkIds": row["source_chunk_ids"],
        "validation": row["validation"],
        "reviewStatus": row["review_status"],
        "reviewNotes": row["review_notes"],
        "reviewedAt": row["reviewed_at"].isoformat() if row["reviewed_at"] else None,
        "factId": row["fact_id"],
        "evidenceQuote": row["evidence_quote"],
        "normalizedQuestionKey": row["normalized_question_key"],
        "originalQuestion": row["original_question"],
        "originalAnswer": row["original_answer"],
        "wasEdited": row["was_edited"],
    }


def _upsert_seed(connection: Any, course_id: str, record: dict[str, Any]) -> None:
    seed_id = record["id"]
    if get_seed(connection, course_id, seed_id) is None:
        create_seed(connection, course_id, record, seed_id=seed_id)
    else:
        update_seed(connection, course_id, seed_id, record)


async def sync_course_from_firebase(course_id: str) -> dict[str, int] | None:
    """Refresh one course in PostgreSQL from Firebase. Best-effort.

    Returns counts of what was written, or None when the sync could not run.
    Never raises: callers invoke this after a write that already succeeded.

    Additive by design — it upserts what Firebase holds and deletes nothing.
    A seed removed in PostgreSQL but still present in Firebase reappears, which
    is the safe direction while Firebase is the rollback copy.
    """
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError:
        return None

    # Imported lazily: this module is imported by main.py at startup, and the
    # Firebase helpers raise on import-time configuration in some test paths.
    from app.firebase_metadata import fetch_course_metadata
    from app.firebase_seeds import fetch_course_seed_examples

    try:
        metadata = await fetch_course_metadata(safe_course_id)
        seeds_payload = await fetch_course_seed_examples(safe_course_id)
    except Exception:  # noqa: BLE001 - a sync failure must not fail the caller
        logger.warning(
            "Could not read Firebase while syncing course %s to PostgreSQL",
            safe_course_id,
            exc_info=True,
        )
        return None

    counts = {"course": 0, "starterSeedGeneration": 0, "seeds": 0, "skipped": 0}

    try:
        with db_connection() as connection:
            if isinstance(metadata, dict):
                try:
                    row = map_course(safe_course_id, metadata)
                except SnapshotError as exc:
                    logger.warning(
                        "Course %s metadata is not mappable: %s", safe_course_id, exc
                    )
                    row = None

                if row is not None:
                    api_metadata = {
                        "name": row["name"],
                        "title": row["title"],
                        "term": row["term"],
                        "instructorName": row["instructor_name"],
                        "createdAt": row["created_at"].isoformat(),
                        "syllabusStatus": row["syllabus_status"],
                        "syllabusFileName": row["syllabus_file_name"],
                        "syllabusType": row["syllabus_type"],
                        "chunkCount": row["chunk_count"],
                    }
                    if course_exists(connection, safe_course_id):
                        update_course(connection, safe_course_id, api_metadata)
                    else:
                        create_course(connection, safe_course_id, api_metadata)
                    counts["course"] = 1

                starter = map_starter_seed_generation(
                    safe_course_id, metadata.get("starterSeedGeneration")
                )
                if starter is not None and counts["course"]:
                    upsert_starter_seed_generation(
                        connection,
                        safe_course_id,
                        {
                            "status": starter["status"],
                            "targetCount": starter["target_count"],
                            "finalCount": starter["final_count"],
                            "savedCount": starter["saved_count"],
                            "failedToSaveCount": starter["failed_to_save_count"],
                            "error": starter["error"],
                            "startedAt": starter["started_at"],
                            "completedAt": starter["completed_at"],
                        },
                    )
                    counts["starterSeedGeneration"] = 1

            if seeds_payload and course_exists(connection, safe_course_id):
                for seed_id, raw in seeds_payload.items():
                    record = _seed_api_record(safe_course_id, seed_id, raw)
                    if record is None:
                        counts["skipped"] += 1
                        continue
                    _upsert_seed(connection, safe_course_id, record)
                    counts["seeds"] += 1
    except Exception:  # noqa: BLE001 - see module docstring
        logger.warning(
            "Could not write PostgreSQL while syncing course %s",
            safe_course_id,
            exc_info=True,
        )
        return None

    return counts


async def mirror_seed_to_firebase(course_id: str, seed: dict[str, Any]) -> bool:
    """Copy a `/api/db` seed write into Firebase. Best-effort.

    Firebase is still what `export_approved_seeds` reads to build a training
    set, and what starter generation reads to deduplicate. A seed that existed
    only in PostgreSQL would be invisible to both — a student's contribution
    could be approved in the UI and then quietly missing from the export.
    """
    from app.firebase_seeds import patch_course_seed_example

    try:
        record = {key: value for key, value in seed.items() if value is not None}
        # Firebase stores both name pairs; the repositories already emit both.
        await patch_course_seed_example(course_id, seed["id"], record)
        return True
    except Exception:  # noqa: BLE001 - see module docstring
        logger.warning(
            "Could not mirror seed %s/%s to Firebase",
            course_id,
            seed.get("id"),
            exc_info=True,
        )
        return False


async def mirror_seed_delete_to_firebase(course_id: str, seed_id: str) -> bool:
    """Remove a seed from Firebase after a `/api/db` delete. Best-effort."""
    from app.firebase_seeds import delete_course_seed_example

    try:
        await delete_course_seed_example(course_id, seed_id)
        return True
    except Exception:  # noqa: BLE001 - see module docstring
        logger.warning(
            "Could not mirror deletion of seed %s/%s to Firebase",
            course_id,
            seed_id,
            exc_info=True,
        )
        return False
