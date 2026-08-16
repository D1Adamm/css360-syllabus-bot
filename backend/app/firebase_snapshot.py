"""Parse a Firebase Realtime Database `courses.json` snapshot into SQL rows.

Pure mapping only — nothing here opens a database connection or touches the
network, so the whole camelCase -> snake_case translation is unit testable. The
importer in `scripts/import_firebase_snapshot.py` supplies the transaction.

Two rules shape the tolerance decisions below:

  - Columns that are NOT NULL in `db/schema.sql` are required. A snapshot that
    cannot fill one is a data problem the operator has to see, so it raises
    `SnapshotError` naming the course, record, and field rather than inventing
    a value or silently dropping the row.
  - Everywhere the live app already accepts a missing field, this accepts it
    too. Seed fallbacks mirror `src/utils/seedDataUtils.ts` exactly: dual
    instruction/question and response/answer names, and the ai_generated
    defaults for sourceSection, difficulty, and directlyAnswered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.course_id import is_valid_course_id

VALID_SEED_ORIGINS = frozenset({"prototype", "user", "ai_generated"})
VALID_SEED_DIFFICULTIES = frozenset({"Easy", "Medium", "Hard"})


class SnapshotError(Exception):
    """Raised when snapshot data cannot be mapped to the PostgreSQL schema."""


# --------------------------------------------------------------------------- #
# Scalar readers
# --------------------------------------------------------------------------- #


def _optional_string(value: Any) -> str | None:
    """Return a trimmed non-empty string, or None for anything else."""
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return None


def _required_string(value: Any, *, field_name: str, context: str) -> str:
    cleaned = _optional_string(value)
    if cleaned is None:
        raise SnapshotError(f"{context}: missing required field '{field_name}'.")
    return cleaned


def _optional_int(value: Any, *, field_name: str, context: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise SnapshotError(
            f"{context}: field '{field_name}' must be a number, got a boolean."
        )
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as exc:
            raise SnapshotError(
                f"{context}: field '{field_name}' is not a whole number: {value!r}."
            ) from exc
    raise SnapshotError(
        f"{context}: field '{field_name}' is not a whole number: {value!r}."
    )


def _int_or_default(value: Any, *, field_name: str, context: str, default: int) -> int:
    parsed = _optional_int(value, field_name=field_name, context=context)
    return default if parsed is None else parsed


def _required_int(value: Any, *, field_name: str, context: str) -> int:
    parsed = _optional_int(value, field_name=field_name, context=context)
    if parsed is None:
        raise SnapshotError(f"{context}: missing required field '{field_name}'.")
    return parsed


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def parse_timestamp(
    value: Any,
    *,
    field_name: str,
    context: str,
) -> datetime | None:
    """Parse an ISO 8601 timestamp into an aware UTC datetime.

    Returns None for null/blank. Accepts both shapes Firebase holds today —
    `2026-08-13T02:15:50.410Z` and `2026-08-13T03:26:52.013570+00:00` — and
    treats a timestamp with no offset as UTC, which is what every writer in
    this repo produces. Anything unparsable raises rather than importing a
    wrong instant.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    text = _optional_string(value)
    if text is None:
        return None

    normalized = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise SnapshotError(
            f"{context}: field '{field_name}' is not an ISO 8601 timestamp: {text!r}."
        ) from exc

    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _required_timestamp(value: Any, *, field_name: str, context: str) -> datetime:
    parsed = parse_timestamp(value, field_name=field_name, context=context)
    if parsed is None:
        raise SnapshotError(f"{context}: missing required field '{field_name}'.")
    return parsed


def _string_list(value: Any) -> list[str] | None:
    """Normalize a Firebase list-ish value into a list of non-empty strings."""
    if value is None:
        return None
    # Firebase drops sparse arrays to objects keyed by index.
    items = list(value.values()) if isinstance(value, dict) else value
    if not isinstance(items, list):
        return None
    cleaned = [text for text in (_optional_string(item) for item in items) if text]
    return cleaned


def _child_dict(parent: Any, key: str, *, context: str) -> dict[str, Any]:
    """Return a `{childId: record}` mapping, tolerating absent/empty nodes."""
    value = parent.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SnapshotError(f"{context}: '{key}' must be an object of records.")
    return value


# --------------------------------------------------------------------------- #
# Row mappers
# --------------------------------------------------------------------------- #


def map_course(course_id: str, metadata: Any) -> dict[str, Any]:
    """Map `courses/{courseId}/metadata` to a `courses` row."""
    context = f"course '{course_id}' metadata"
    if not isinstance(metadata, dict):
        raise SnapshotError(f"{context}: expected an object.")

    return {
        "course_id": course_id,
        "name": _required_string(metadata.get("name"), field_name="name", context=context),
        "title": _required_string(
            metadata.get("title"), field_name="title", context=context
        ),
        "term": _required_string(metadata.get("term"), field_name="term", context=context),
        "instructor_name": _required_string(
            metadata.get("instructorName"), field_name="instructorName", context=context
        ),
        "created_at": _required_timestamp(
            metadata.get("createdAt"), field_name="createdAt", context=context
        ),
        "syllabus_status": _required_string(
            metadata.get("syllabusStatus"), field_name="syllabusStatus", context=context
        ),
        "syllabus_file_name": _optional_string(metadata.get("syllabusFileName")),
        "syllabus_type": _optional_string(metadata.get("syllabusType")),
        "chunk_count": _int_or_default(
            metadata.get("chunkCount"), field_name="chunkCount", context=context, default=0
        ),
    }


def map_starter_seed_generation(course_id: str, block: Any) -> dict[str, Any] | None:
    """Map `metadata/starterSeedGeneration` to a `starter_seed_generation` row.

    Every column is nullable, matching `StoredStarterSeedGeneration`, where a
    run interrupted partway through leaves most fields unwritten.
    """
    if block is None:
        return None
    context = f"course '{course_id}' starterSeedGeneration"
    if not isinstance(block, dict):
        raise SnapshotError(f"{context}: expected an object.")

    return {
        "course_id": course_id,
        "status": _optional_string(block.get("status")),
        "target_count": _optional_int(
            block.get("targetCount"), field_name="targetCount", context=context
        ),
        "final_count": _optional_int(
            block.get("finalCount"), field_name="finalCount", context=context
        ),
        "saved_count": _optional_int(
            block.get("savedCount"), field_name="savedCount", context=context
        ),
        "failed_to_save_count": _optional_int(
            block.get("failedToSaveCount"),
            field_name="failedToSaveCount",
            context=context,
        ),
        "error": _optional_string(block.get("error")),
        "started_at": parse_timestamp(
            block.get("startedAt"), field_name="startedAt", context=context
        ),
        "completed_at": parse_timestamp(
            block.get("completedAt"), field_name="completedAt", context=context
        ),
    }


def map_seed_example(course_id: str, seed_id: str, raw: Any) -> dict[str, Any]:
    """Map one `courses/{courseId}/seedExamples/{seedId}` record.

    The Firebase push id is the primary key, so a record whose `id` field
    disagrees with its key still imports under the key the app reads it by.
    """
    context = f"course '{course_id}' seed '{seed_id}'"
    if not isinstance(raw, dict):
        raise SnapshotError(f"{context}: expected an object.")

    origin = _optional_string(raw.get("origin"))
    if origin is None:
        raise SnapshotError(f"{context}: missing required field 'origin'.")
    if origin not in VALID_SEED_ORIGINS:
        raise SnapshotError(
            f"{context}: field 'origin' must be one of "
            f"{sorted(VALID_SEED_ORIGINS)}, got {origin!r}."
        )
    is_ai_generated = origin == "ai_generated"

    instruction = _optional_string(raw.get("instruction")) or _optional_string(
        raw.get("question")
    )
    if instruction is None:
        raise SnapshotError(f"{context}: missing required 'instruction' (or 'question').")

    response = _optional_string(raw.get("response")) or _optional_string(raw.get("answer"))
    if response is None:
        raise SnapshotError(f"{context}: missing required 'response' (or 'answer').")

    source_chunk_ids = _string_list(raw.get("sourceChunkIds"))

    source_section = _optional_string(raw.get("sourceSection"))
    if source_section is None and source_chunk_ids:
        source_section = ", ".join(source_chunk_ids)
    if source_section is None and is_ai_generated:
        source_section = "General"
    if source_section is None:
        raise SnapshotError(f"{context}: missing required field 'sourceSection'.")

    difficulty = _optional_string(raw.get("difficulty"))
    if difficulty is not None and difficulty not in VALID_SEED_DIFFICULTIES:
        raise SnapshotError(
            f"{context}: field 'difficulty' must be one of "
            f"{sorted(VALID_SEED_DIFFICULTIES)}, got {difficulty!r}."
        )
    if difficulty is None:
        if not is_ai_generated:
            raise SnapshotError(f"{context}: missing required field 'difficulty'.")
        difficulty = "Medium"

    directly_answered = _optional_bool(raw.get("directlyAnswered"))
    if directly_answered is None:
        if not is_ai_generated:
            raise SnapshotError(f"{context}: missing required field 'directlyAnswered'.")
        directly_answered = True

    category = _optional_string(raw.get("category"))
    if category is None:
        raise SnapshotError(f"{context}: missing required field 'category'.")

    validation = raw.get("validation")
    if validation is not None and not isinstance(validation, dict):
        raise SnapshotError(f"{context}: field 'validation' must be an object.")

    return {
        "seed_id": seed_id,
        "course_id": course_id,
        "instruction": instruction,
        "response": response,
        "category": category,
        "source_section": source_section,
        "difficulty": difficulty,
        "directly_answered": directly_answered,
        "origin": origin,
        "notes": _optional_string(raw.get("notes")),
        "created_at": parse_timestamp(
            raw.get("createdAt"), field_name="createdAt", context=context
        ),
        "status": _optional_string(raw.get("status")),
        "question_type": _optional_string(raw.get("questionType")),
        "source_chunk_ids": source_chunk_ids,
        "validation": validation,
        "review_status": _optional_string(raw.get("reviewStatus")),
        "review_notes": _optional_string(raw.get("reviewNotes")),
        "reviewed_at": parse_timestamp(
            raw.get("reviewedAt"), field_name="reviewedAt", context=context
        ),
        "fact_id": _optional_string(raw.get("factId")),
        "evidence_quote": _optional_string(raw.get("evidenceQuote")),
        "normalized_question_key": _optional_string(raw.get("normalizedQuestionKey")),
        "original_question": _optional_string(raw.get("originalQuestion")),
        "original_answer": _optional_string(raw.get("originalAnswer")),
        "was_edited": _optional_bool(raw.get("wasEdited")) or False,
    }


def map_evaluation(course_id: str, evaluation_id: str, raw: Any) -> dict[str, Any]:
    """Map one `courses/{courseId}/evaluations/{evaluationId}` record."""
    context = f"course '{course_id}' evaluation '{evaluation_id}'"
    if not isinstance(raw, dict):
        raise SnapshotError(f"{context}: expected an object.")

    flags = _string_list(raw.get("hallucinationFlags")) or []

    return {
        "evaluation_id": evaluation_id,
        "course_id": course_id,
        "comparison_id": _required_string(
            raw.get("comparisonId"), field_name="comparisonId", context=context
        ),
        "most_accurate": _required_string(
            raw.get("mostAccurate"), field_name="mostAccurate", context=context
        ),
        "most_helpful": _required_string(
            raw.get("mostHelpful"), field_name="mostHelpful", context=context
        ),
        "most_concise": _required_string(
            raw.get("mostConcise"), field_name="mostConcise", context=context
        ),
        "best_grounded": _required_string(
            raw.get("bestGrounded"), field_name="bestGrounded", context=context
        ),
        "preferred_model": _required_string(
            raw.get("preferredModel"), field_name="preferredModel", context=context
        ),
        "hallucination_flags": flags,
        "comment": _optional_string(raw.get("comment")),
        "created_at": _required_timestamp(
            raw.get("createdAt"), field_name="createdAt", context=context
        ),
        "run_id": _optional_string(raw.get("runId")),
        "question_text": _optional_string(raw.get("questionText")),
    }


def map_model_registry(
    course_id: str,
    raw: Any,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Map `courses/{courseId}/model` to a `course_models` row plus versions.

    `current_version` is NOT NULL, so a registry with versions but no
    `currentVersion` is a data problem. A registry holding neither imports as
    no rows at all rather than a pointer to nothing.
    """
    if raw is None:
        return None, []
    context = f"course '{course_id}' model"
    if not isinstance(raw, dict):
        raise SnapshotError(f"{context}: expected an object.")

    versions_node = _child_dict(raw, "versions", context=context)
    current_version = _optional_string(raw.get("currentVersion"))

    version_rows: list[dict[str, Any]] = []
    for version_key, version_raw in versions_node.items():
        version_context = f"{context} version '{version_key}'"
        if not isinstance(version_raw, dict):
            raise SnapshotError(f"{version_context}: expected an object.")

        version_rows.append(
            {
                "course_id": course_id,
                # The registry key is authoritative; `version` inside the
                # record is a mirror of it.
                "version": version_key,
                "base_model": _required_string(
                    version_raw.get("baseModel"),
                    field_name="baseModel",
                    context=version_context,
                ),
                "training_example_count": _required_int(
                    version_raw.get("trainingExampleCount"),
                    field_name="trainingExampleCount",
                    context=version_context,
                ),
                "status": _required_string(
                    version_raw.get("status"), field_name="status", context=version_context
                ),
                "deployment": _required_string(
                    version_raw.get("deployment"),
                    field_name="deployment",
                    context=version_context,
                ),
                "artifact_ref": _required_string(
                    version_raw.get("artifactRef"),
                    field_name="artifactRef",
                    context=version_context,
                ),
                "created_at": _required_timestamp(
                    version_raw.get("createdAt"),
                    field_name="createdAt",
                    context=version_context,
                ),
                "updated_at": parse_timestamp(
                    version_raw.get("updatedAt"),
                    field_name="updatedAt",
                    context=version_context,
                ),
                "notes": _optional_string(version_raw.get("notes")),
            }
        )

    if current_version is None:
        if version_rows:
            raise SnapshotError(f"{context}: missing required field 'currentVersion'.")
        return None, []

    known_versions = {row["version"] for row in version_rows}
    if current_version not in known_versions:
        raise SnapshotError(
            f"{context}: currentVersion {current_version!r} has no matching entry "
            f"under 'versions' ({sorted(known_versions)})."
        )

    return {"course_id": course_id, "current_version": current_version}, version_rows


def map_model_request(course_id: str, raw: Any) -> dict[str, Any] | None:
    """Map `courses/{courseId}/modelRequest` to a `model_requests` row.

    `preparation` and `training` stay whole in JSONB: they are admin-only
    detail whose shape is still moving, and flattening them would mean a
    schema change every time a field is added.
    """
    if raw is None:
        return None
    context = f"course '{course_id}' modelRequest"
    if not isinstance(raw, dict):
        raise SnapshotError(f"{context}: expected an object.")

    preparation = raw.get("preparation")
    if preparation is not None and not isinstance(preparation, dict):
        raise SnapshotError(f"{context}: field 'preparation' must be an object.")

    training = raw.get("training")
    if training is not None and not isinstance(training, dict):
        raise SnapshotError(f"{context}: field 'training' must be an object.")

    return {
        "course_id": course_id,
        "status": _required_string(
            raw.get("status"), field_name="status", context=context
        ),
        "requested_at": _required_timestamp(
            raw.get("requestedAt"), field_name="requestedAt", context=context
        ),
        "updated_at": _required_timestamp(
            raw.get("updatedAt"), field_name="updatedAt", context=context
        ),
        "approved_example_count": _int_or_default(
            raw.get("approvedExampleCount"),
            field_name="approvedExampleCount",
            context=context,
            default=0,
        ),
        "failure_message": _optional_string(raw.get("failureMessage")),
        "preparation": preparation,
        "preparation_error": _optional_string(raw.get("preparationError")),
        "training": training,
        "launch_error": _optional_string(raw.get("launchError")),
        "current_run_id": _optional_string(raw.get("currentRunId")),
    }


def map_training_run(course_id: str, run_id: str, raw: Any) -> dict[str, Any]:
    """Map one `courses/{courseId}/trainingRuns/{runId}` record.

    The claim is flattened rather than stored as JSONB: a runner lease is
    queried (who holds this, has it expired?), not just displayed.
    """
    context = f"course '{course_id}' training run '{run_id}'"
    if not isinstance(raw, dict):
        raise SnapshotError(f"{context}: expected an object.")

    claim = raw.get("claim")
    if claim is not None and not isinstance(claim, dict):
        raise SnapshotError(f"{context}: field 'claim' must be an object.")
    claim = claim or {}

    return {
        "run_id": run_id,
        "course_id": course_id,
        "mode": _required_string(raw.get("mode"), field_name="mode", context=context),
        "state": _required_string(raw.get("state"), field_name="state", context=context),
        "enqueued_at": _required_timestamp(
            raw.get("enqueuedAt"), field_name="enqueuedAt", context=context
        ),
        "updated_at": _required_timestamp(
            raw.get("updatedAt"), field_name="updatedAt", context=context
        ),
        "dataset_ref": _required_string(
            raw.get("datasetRef"), field_name="datasetRef", context=context
        ),
        "approved_example_count": _int_or_default(
            raw.get("approvedExampleCount"),
            field_name="approvedExampleCount",
            context=context,
            default=0,
        ),
        "train_examples": _int_or_default(
            raw.get("trainExamples"), field_name="trainExamples", context=context, default=0
        ),
        "validation_examples": _int_or_default(
            raw.get("validationExamples"),
            field_name="validationExamples",
            context=context,
            default=0,
        ),
        "attempt": _int_or_default(
            raw.get("attempt"), field_name="attempt", context=context, default=0
        ),
        "job_id": _optional_string(raw.get("jobId")),
        "claim_owner": _optional_string(claim.get("owner")),
        "claim_claimed_at": parse_timestamp(
            claim.get("claimedAt"), field_name="claim.claimedAt", context=context
        ),
        "claim_expires_at": parse_timestamp(
            claim.get("expiresAt"), field_name="claim.expiresAt", context=context
        ),
        "error": _optional_string(raw.get("error")),
    }


# --------------------------------------------------------------------------- #
# Whole-snapshot parsing
# --------------------------------------------------------------------------- #


@dataclass
class SnapshotPlan:
    """Every row a snapshot would write, in foreign-key-safe order."""

    courses: list[dict[str, Any]] = field(default_factory=list)
    starter_seed_generation: list[dict[str, Any]] = field(default_factory=list)
    seed_examples: list[dict[str, Any]] = field(default_factory=list)
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    course_models: list[dict[str, Any]] = field(default_factory=list)
    course_model_versions: list[dict[str, Any]] = field(default_factory=list)
    model_requests: list[dict[str, Any]] = field(default_factory=list)
    training_runs: list[dict[str, Any]] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "courses": len(self.courses),
            "starter_seed_generation": len(self.starter_seed_generation),
            "seed_examples": len(self.seed_examples),
            "evaluations": len(self.evaluations),
            "course_models": len(self.course_models),
            "course_model_versions": len(self.course_model_versions),
            "model_requests": len(self.model_requests),
            "training_runs": len(self.training_runs),
        }


def extract_courses_node(payload: Any) -> dict[str, Any]:
    """Accept either a bare `courses` export or one still wrapped in a root."""
    if not isinstance(payload, dict):
        raise SnapshotError("Snapshot root must be a JSON object of courses.")

    if "courses" in payload and isinstance(payload["courses"], dict):
        return payload["courses"]
    return payload


def parse_snapshot(payload: Any) -> SnapshotPlan:
    """Map a whole `courses.json` snapshot into rows, or raise SnapshotError."""
    courses_node = extract_courses_node(payload)
    plan = SnapshotPlan()

    for course_id, course_raw in courses_node.items():
        if not is_valid_course_id(course_id):
            raise SnapshotError(
                f"Snapshot contains an invalid courseId {course_id!r}: must use "
                "lowercase letters, numbers, and hyphens only."
            )
        context = f"course '{course_id}'"
        if not isinstance(course_raw, dict):
            raise SnapshotError(f"{context}: expected an object.")

        metadata = course_raw.get("metadata")
        if metadata is None:
            raise SnapshotError(f"{context}: missing required 'metadata' node.")
        plan.courses.append(map_course(course_id, metadata))

        starter = map_starter_seed_generation(
            course_id, metadata.get("starterSeedGeneration")
        )
        if starter is not None:
            plan.starter_seed_generation.append(starter)

        for seed_id, seed_raw in _child_dict(
            course_raw, "seedExamples", context=context
        ).items():
            plan.seed_examples.append(map_seed_example(course_id, seed_id, seed_raw))

        for evaluation_id, evaluation_raw in _child_dict(
            course_raw, "evaluations", context=context
        ).items():
            plan.evaluations.append(
                map_evaluation(course_id, evaluation_id, evaluation_raw)
            )

        model_row, version_rows = map_model_registry(course_id, course_raw.get("model"))
        if model_row is not None:
            plan.course_models.append(model_row)
        plan.course_model_versions.extend(version_rows)

        request_row = map_model_request(course_id, course_raw.get("modelRequest"))
        if request_row is not None:
            plan.model_requests.append(request_row)

        for run_id, run_raw in _child_dict(
            course_raw, "trainingRuns", context=context
        ).items():
            plan.training_runs.append(map_training_run(course_id, run_id, run_raw))

    return plan
