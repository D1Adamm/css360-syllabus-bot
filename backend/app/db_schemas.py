"""Request/response models for the PostgreSQL-backed `/api/db` routes.

Separate from `schemas.py` so the parallel layer can be read — and later
removed or promoted — without picking it out of the live Firebase models.

Response records use `extra="allow"`, the same choice `SeedReviewRecord`
already makes: the repositories emit the camelCase records the frontend types
describe, and a strict model would silently drop a field the moment one is
added to the schema. Request models are strict about what they require and
permissive about the rest, because a patch body is a partial by definition.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DbRecord(BaseModel):
    """Base for records the repositories build, already camelCase."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")


# --------------------------------------------------------------------------- #
# Courses
# --------------------------------------------------------------------------- #


class StarterSeedGenerationRecord(DbRecord):
    status: str | None = None
    target_count: int | None = Field(default=None, alias="targetCount")
    final_count: int | None = Field(default=None, alias="finalCount")
    saved_count: int | None = Field(default=None, alias="savedCount")
    failed_to_save_count: int | None = Field(default=None, alias="failedToSaveCount")
    error: str | None = None
    started_at: str | None = Field(default=None, alias="startedAt")
    completed_at: str | None = Field(default=None, alias="completedAt")


class CourseMetadataRecord(DbRecord):
    name: str
    title: str
    term: str
    instructor_name: str = Field(alias="instructorName")
    created_at: str = Field(alias="createdAt")
    syllabus_status: str = Field(alias="syllabusStatus")
    syllabus_file_name: str | None = Field(default=None, alias="syllabusFileName")
    syllabus_type: str | None = Field(default=None, alias="syllabusType")
    chunk_count: int = Field(default=0, alias="chunkCount")
    starter_seed_generation: StarterSeedGenerationRecord | None = Field(
        default=None, alias="starterSeedGeneration"
    )


class CourseRecord(DbRecord):
    course_id: str = Field(alias="courseId")
    metadata: CourseMetadataRecord


class CourseListResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    count: int
    courses: list[CourseRecord]


class CourseCreateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId", min_length=1)
    name: str = Field(min_length=1)
    title: str = Field(min_length=1)
    term: str = Field(min_length=1)
    instructor_name: str = Field(alias="instructorName", min_length=1)
    created_at: str | None = Field(default=None, alias="createdAt")
    syllabus_status: str = Field(default="none", alias="syllabusStatus")
    syllabus_file_name: str | None = Field(default=None, alias="syllabusFileName")
    syllabus_type: str | None = Field(default=None, alias="syllabusType")
    chunk_count: int = Field(default=0, alias="chunkCount", ge=0)


class CourseUpdateRequest(BaseModel):
    """Every field optional: this is a merge, not a replacement."""

    model_config = ConfigDict(populate_by_name=True)

    name: str | None = None
    title: str | None = None
    term: str | None = None
    instructor_name: str | None = Field(default=None, alias="instructorName")
    syllabus_status: str | None = Field(default=None, alias="syllabusStatus")
    syllabus_file_name: str | None = Field(default=None, alias="syllabusFileName")
    syllabus_type: str | None = Field(default=None, alias="syllabusType")
    chunk_count: int | None = Field(default=None, alias="chunkCount", ge=0)


class StarterSeedGenerationUpdateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status: str | None = None
    target_count: int | None = Field(default=None, alias="targetCount", ge=0)
    final_count: int | None = Field(default=None, alias="finalCount", ge=0)
    saved_count: int | None = Field(default=None, alias="savedCount", ge=0)
    failed_to_save_count: int | None = Field(
        default=None, alias="failedToSaveCount", ge=0
    )
    error: str | None = None
    started_at: str | None = Field(default=None, alias="startedAt")
    completed_at: str | None = Field(default=None, alias="completedAt")


class StarterSeedGenerationResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    starter_seed_generation: StarterSeedGenerationRecord | None = Field(
        default=None, alias="starterSeedGeneration"
    )


# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #


class SeedRecord(DbRecord):
    id: str
    course_id: str = Field(alias="courseId")
    instruction: str
    response: str
    question: str
    answer: str
    category: str
    source_section: str = Field(alias="sourceSection")
    difficulty: str
    directly_answered: bool = Field(alias="directlyAnswered")
    origin: str
    source_chunk_ids: list[str] = Field(default_factory=list, alias="sourceChunkIds")
    was_edited: bool = Field(default=False, alias="wasEdited")


class SeedListResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    count: int
    seeds: list[SeedRecord]
    review_status_counts: dict[str, int] = Field(
        default_factory=dict, alias="reviewStatusCounts"
    )


class SeedCreateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    instruction: str | None = None
    question: str | None = None
    response: str | None = None
    answer: str | None = None
    category: str | None = None
    source_section: str | None = Field(default=None, alias="sourceSection")
    difficulty: str | None = None
    directly_answered: bool | None = Field(default=None, alias="directlyAnswered")
    origin: str | None = None
    notes: str | None = None
    created_at: str | None = Field(default=None, alias="createdAt")
    status: str | None = None
    question_type: str | None = Field(default=None, alias="questionType")
    source_chunk_ids: list[str] | None = Field(default=None, alias="sourceChunkIds")
    validation: dict[str, Any] | None = None
    review_status: str | None = Field(default=None, alias="reviewStatus")
    review_notes: str | None = Field(default=None, alias="reviewNotes")
    fact_id: str | None = Field(default=None, alias="factId")
    evidence_quote: str | None = Field(default=None, alias="evidenceQuote")
    normalized_question_key: str | None = Field(
        default=None, alias="normalizedQuestionKey"
    )


class SeedUpdateRequest(SeedCreateRequest):
    reviewed_at: str | None = Field(default=None, alias="reviewedAt")
    original_question: str | None = Field(default=None, alias="originalQuestion")
    original_answer: str | None = Field(default=None, alias="originalAnswer")
    was_edited: bool | None = Field(default=None, alias="wasEdited")


class SeedResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    seed_id: str = Field(alias="seedId")
    seed: SeedRecord


# --------------------------------------------------------------------------- #
# Evaluations
# --------------------------------------------------------------------------- #


class EvaluationRecordModel(DbRecord):
    id: str
    course_id: str = Field(alias="courseId")
    comparison_id: str = Field(alias="comparisonId")
    most_accurate: str = Field(alias="mostAccurate")
    most_helpful: str = Field(alias="mostHelpful")
    most_concise: str = Field(alias="mostConcise")
    best_grounded: str = Field(alias="bestGrounded")
    preferred_model: str = Field(alias="preferredModel")
    hallucination_flags: list[str] = Field(
        default_factory=list, alias="hallucinationFlags"
    )
    created_at: str = Field(alias="createdAt")
    comment: str | None = None
    run_id: str | None = Field(default=None, alias="runId")
    question_text: str | None = Field(default=None, alias="questionText")


class EvaluationListResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    count: int
    evaluations: list[EvaluationRecordModel]


class EvaluationCreateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str | None = None
    comparison_id: str = Field(alias="comparisonId", min_length=1)
    most_accurate: str = Field(alias="mostAccurate", min_length=1)
    most_helpful: str = Field(alias="mostHelpful", min_length=1)
    most_concise: str = Field(alias="mostConcise", min_length=1)
    best_grounded: str = Field(alias="bestGrounded", min_length=1)
    preferred_model: str = Field(alias="preferredModel", min_length=1)
    hallucination_flags: list[str] = Field(
        default_factory=list, alias="hallucinationFlags"
    )
    comment: str | None = None
    created_at: str | None = Field(default=None, alias="createdAt")
    run_id: str | None = Field(default=None, alias="runId")
    question_text: str | None = Field(default=None, alias="questionText")


class DeleteResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    deleted: int


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #


class ModelVersionRecord(DbRecord):
    version: str
    base_model: str = Field(alias="baseModel")
    training_example_count: int = Field(default=0, alias="trainingExampleCount")
    status: str
    deployment: str
    artifact_ref: str = Field(alias="artifactRef")
    created_at: str = Field(alias="createdAt")
    updated_at: str | None = Field(default=None, alias="updatedAt")
    notes: str | None = None


class ModelRegistryResponse(DbRecord):
    course_id: str = Field(alias="courseId")
    current_version: str = Field(alias="currentVersion")
    versions: dict[str, ModelVersionRecord]


# --------------------------------------------------------------------------- #
# Model requests
# --------------------------------------------------------------------------- #


class ModelRequestRecord(DbRecord):
    course_id: str = Field(alias="courseId")
    status: str
    requested_at: str = Field(alias="requestedAt")
    updated_at: str = Field(alias="updatedAt")
    approved_example_count: int = Field(default=0, alias="approvedExampleCount")
    failure_message: str | None = Field(default=None, alias="failureMessage")
    preparation: dict[str, Any] | None = None
    preparation_error: str | None = Field(default=None, alias="preparationError")
    training: dict[str, Any] | None = None
    launch_error: str | None = Field(default=None, alias="launchError")
    current_run_id: str | None = Field(default=None, alias="currentRunId")


class ModelRequestCreateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    approved_example_count: int = Field(alias="approvedExampleCount", ge=0)


class ModelRequestUpdateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status: str | None = None
    approved_example_count: int | None = Field(
        default=None, alias="approvedExampleCount", ge=0
    )
    failure_message: str | None = Field(default=None, alias="failureMessage")
    preparation: dict[str, Any] | None = None
    preparation_error: str | None = Field(default=None, alias="preparationError")
    training: dict[str, Any] | None = None
    launch_error: str | None = Field(default=None, alias="launchError")
    current_run_id: str | None = Field(default=None, alias="currentRunId")


# --------------------------------------------------------------------------- #
# Training runs
# --------------------------------------------------------------------------- #


class TrainingRunClaimRecord(DbRecord):
    owner: str
    claimed_at: str = Field(alias="claimedAt")
    expires_at: str = Field(alias="expiresAt")


class TrainingRunRecord(DbRecord):
    run_id: str = Field(alias="runId")
    course_id: str = Field(alias="courseId")
    mode: str
    state: str
    enqueued_at: str = Field(alias="enqueuedAt")
    updated_at: str = Field(alias="updatedAt")
    dataset_ref: str = Field(default="", alias="datasetRef")
    approved_example_count: int = Field(default=0, alias="approvedExampleCount")
    train_examples: int = Field(default=0, alias="trainExamples")
    validation_examples: int = Field(default=0, alias="validationExamples")
    attempt: int = 0
    job_id: str | None = Field(default=None, alias="jobId")
    claim: TrainingRunClaimRecord | None = None
    error: str | None = None


class TrainingRunListResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    count: int
    runs: list[TrainingRunRecord]


class TrainingRunCreateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    mode: str = Field(description="smoke | full")
    dataset_ref: str = Field(alias="datasetRef", min_length=1)
    approved_example_count: int = Field(
        default=0, alias="approvedExampleCount", ge=0
    )
    train_examples: int = Field(default=0, alias="trainExamples", ge=0)
    validation_examples: int = Field(default=0, alias="validationExamples", ge=0)


class TrainingRunUpdateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    state: str | None = None
    dataset_ref: str | None = Field(default=None, alias="datasetRef")
    approved_example_count: int | None = Field(
        default=None, alias="approvedExampleCount", ge=0
    )
    train_examples: int | None = Field(default=None, alias="trainExamples", ge=0)
    validation_examples: int | None = Field(
        default=None, alias="validationExamples", ge=0
    )
    attempt: int | None = Field(default=None, ge=0)
    job_id: str | None = Field(default=None, alias="jobId")
    claim: TrainingRunClaimRecord | None = None
    clear_claim: bool = Field(
        default=False,
        alias="clearClaim",
        description="Release the lease. Clears owner, claimedAt, and expiresAt together.",
    )
    error: str | None = None
