from pydantic import BaseModel, ConfigDict, Field
from typing import Any


class BaseModelGenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(
        ...,
        alias="courseId",
        min_length=1,
        description="Course id from the active course route",
    )
    question: str = Field(..., min_length=1, description="Student question to send to the base model")


class BaseModelGenerateResponse(BaseModel):
    answer: str
    model: str
    response_type: str = Field(alias="responseType")
    course_id: str | None = Field(default=None, alias="courseId")

    model_config = {"populate_by_name": True}


class FineTunedGenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(
        ...,
        alias="courseId",
        min_length=1,
        description="Course id from the active course route",
    )
    question: str = Field(
        ...,
        min_length=1,
        description="Student question to send to the fine-tuned model",
    )


class FineTunedGenerateResponse(BaseModel):
    answer: str
    model: str
    response_type: str = Field(alias="responseType")
    course_id: str | None = Field(default=None, alias="courseId")
    adapter_loaded: bool = Field(alias="adapterLoaded")
    generation_seconds: float | None = Field(default=None, alias="generationSeconds")

    model_config = {"populate_by_name": True}


class FineTunedHealthResponse(BaseModel):
    status: str
    model: str | None = None
    adapter_loaded: bool | None = Field(default=None, alias="adapterLoaded")
    hostname: str | None = None
    port: int | None = None
    service_url: str | None = Field(default=None, alias="serviceUrl")

    model_config = {"populate_by_name": True}


class RagRetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Student question for syllabus retrieval")
    top_k: int = Field(
        default=4,
        alias="topK",
        ge=1,
        le=20,
        description="Number of syllabus chunks to return",
    )
    debug: bool = Field(
        default=False,
        description="Include retrieval ranking debug metadata for development",
    )


class RagRetrieveDebugRanking(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    chunk_id: str = Field(alias="chunkId")
    section: str
    base_score: float = Field(alias="baseScore")
    score_adjustment: float = Field(alias="scoreAdjustment")
    score: float
    selected: bool


class RagRetrieveResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    chunk_id: str = Field(alias="chunkId")
    section: str
    text: str
    score: float


class RagRetrieveResponse(BaseModel):
    embedding_model: str = Field(alias="embeddingModel")
    results: list[RagRetrieveResult]
    debug_rankings: list[RagRetrieveDebugRanking] | None = Field(
        default=None,
        alias="debugRankings",
    )

    model_config = {"populate_by_name": True}


class RagGenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(
        ...,
        alias="courseId",
        min_length=1,
        description="Course id whose syllabus index should be used for retrieval",
    )
    question: str = Field(..., min_length=1, description="Student question for RAG answer generation")
    top_k: int = Field(
        default=4,
        alias="topK",
        ge=1,
        le=20,
        description="Number of syllabus chunks to retrieve for context",
    )


class RagGenerateSource(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    chunk_id: str = Field(alias="chunkId")
    section_title: str = Field(alias="sectionTitle")
    text: str
    score: float


class RagGenerateResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    answer: str
    model: str
    sources: list[RagGenerateSource]
    retrieved_chunks: list[RagRetrieveResult] = Field(alias="retrievedChunks")
    response_type: str = Field(default="rag", alias="responseType")


class FineTunedRagGenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(
        ...,
        alias="courseId",
        min_length=1,
        description="Course id whose syllabus index should be used for retrieval",
    )
    question: str = Field(
        ...,
        min_length=1,
        description="Student question for Fine-Tuned + RAG answer generation",
    )
    top_k: int = Field(
        default=4,
        alias="topK",
        ge=1,
        le=20,
        description="Number of syllabus chunks to retrieve for context",
    )


class FineTunedRagGenerateResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    answer: str
    model: str
    sources: list[RagGenerateSource]
    retrieved_chunks: list[RagRetrieveResult] = Field(alias="retrievedChunks")
    response_type: str = Field(default="fineTunedRag", alias="responseType")
    adapter_loaded: bool = Field(alias="adapterLoaded")
    generation_seconds: float | None = Field(default=None, alias="generationSeconds")


class SyllabusUploadResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    syllabus_file_name: str = Field(alias="syllabusFileName")
    syllabus_type: str = Field(alias="syllabusType")
    syllabus_status: str = Field(alias="syllabusStatus")
    file_size: int = Field(alias="fileSize")
    character_count: int = Field(alias="characterCount")
    chunk_count: int = Field(alias="chunkCount")
    starter_seed_generation_status: str | None = Field(
        default=None,
        alias="starterSeedGenerationStatus",
        description="queued when automatic starter generation was scheduled",
    )


class SyllabusTextResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    text: str
    character_count: int = Field(alias="characterCount")


class CourseChunkMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    chunk_id: str = Field(alias="chunkId")
    section_title: str = Field(alias="sectionTitle")
    text: str
    order: int
    document_title: str | None = Field(default=None, alias="documentTitle")
    heading_path: list[str] | None = Field(default=None, alias="headingPath")
    start_offset: int | None = Field(default=None, alias="startOffset")
    end_offset: int | None = Field(default=None, alias="endOffset")


class CourseChunksResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    chunk_count: int = Field(alias="chunkCount")
    chunks: list[CourseChunkMetadata]
    index_version: int | None = Field(default=None, alias="indexVersion")
    document_title: str | None = Field(default=None, alias="documentTitle")


class SeedGenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    chunk_id: str = Field(
        ...,
        alias="chunkId",
        min_length=1,
        description="Id of the stored syllabus chunk used as seed context",
    )
    count: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of seed examples to generate (default 3, max 5)",
    )


class GeneratedSeedExample(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    question: str
    answer: str
    category: str
    question_type: str | None = Field(default=None, alias="questionType")
    source_chunk_ids: list[str] = Field(alias="sourceChunkIds")
    fact_id: str | None = Field(default=None, alias="factId")
    evidence_quote: str | None = Field(default=None, alias="evidenceQuote")
    origin: str
    status: str
    review_status: str | None = Field(default=None, alias="reviewStatus")
    validation: "SeedValidationResult | None" = None


class SeedValidationComponents(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    grounded: float = Field(ge=0.0, le=1.0)
    correct: float = Field(ge=0.0, le=1.0)
    clear: float = Field(ge=0.0, le=1.0)
    useful: float = Field(ge=0.0, le=1.0)
    natural_student_wording: float = Field(alias="naturalStudentWording", ge=0.0, le=1.0)
    category_correct: float = Field(alias="categoryCorrect", ge=0.0, le=1.0)
    not_trivial_or_temporary: float = Field(
        alias="notTrivialOrTemporary",
        ge=0.0,
        le=1.0,
    )


class SeedValidationResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    score: float = Field(ge=0.0, le=1.0)
    reason: str
    unsupported_claims: list[str] = Field(
        default_factory=list,
        alias="unsupportedClaims",
    )
    components: SeedValidationComponents
    # Optional legacy/top-level mirrors for older clients.
    grounded: float | None = Field(default=None, ge=0.0, le=1.0)
    correct: float | None = Field(default=None, ge=0.0, le=1.0)
    clear: float | None = Field(default=None, ge=0.0, le=1.0)
    useful: float | None = Field(default=None, ge=0.0, le=1.0)
    natural_student_wording: float | None = Field(
        default=None,
        alias="naturalStudentWording",
        ge=0.0,
        le=1.0,
    )
    category_correct: float | None = Field(
        default=None,
        alias="categoryCorrect",
        ge=0.0,
        le=1.0,
    )
    not_trivial_or_temporary: float | None = Field(
        default=None,
        alias="notTrivialOrTemporary",
        ge=0.0,
        le=1.0,
    )


class SeedGenerateResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    chunk_id: str = Field(alias="chunkId")
    model: str
    count: int
    seeds: list[GeneratedSeedExample]


class StarterSeedGenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    target_count: int = Field(
        default=50,
        alias="targetCount",
        ge=1,
        le=50,
        description="Maximum number of starter seeds to collect (default 50, max 50)",
    )
    save: bool = Field(
        default=False,
        description=(
            "When true, persist accepted validated seeds to Firebase. "
            "Defaults to false (generate-only)."
        ),
    )
    force_refresh: bool = Field(
        default=False,
        alias="forceRefresh",
        description=(
            "When true, rebuild the fact inventory instead of reusing a valid cache."
        ),
    )
    top_up: bool = Field(
        default=False,
        alias="topUp",
        description=(
            "When true, read existing Firebase seeds first and generate only "
            "enough new seeds to reach targetCount. Existing seeds are preserved."
        ),
    )


class StarterSeedTopUpRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    target_count: int = Field(
        default=50,
        alias="targetCount",
        ge=1,
        le=50,
        description="Desired total saved seeds after top-up (default 50).",
    )
    save: bool = Field(
        default=True,
        description=(
            "When true (default), persist only newly accepted seeds to Firebase. "
            "Existing seeds are never deleted."
        ),
    )
    force_refresh: bool = Field(
        default=False,
        alias="forceRefresh",
        description=(
            "When true, rebuild the fact inventory instead of reusing a valid cache."
        ),
    )


class StarterSeedPersistence(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    generated_count: int = Field(alias="generatedCount")
    saved_count: int = Field(alias="savedCount")
    already_existing_count: int = Field(alias="alreadyExistingCount")
    failed_to_save_count: int = Field(alias="failedToSaveCount")


class StarterSeedProgress(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    eligible_chunks: int = Field(alias="eligibleChunks")
    selected_chunks: int = Field(alias="selectedChunks")
    chunks_processed: int = Field(alias="chunksProcessed")
    chunks_skipped: int = Field(alias="chunksSkipped")
    planning_calls: int = Field(alias="planningCalls", default=0)
    merge_calls: int = Field(alias="mergeCalls", default=0)
    fact_extraction_calls: int = Field(alias="factExtractionCalls", default=0)
    fact_inventory_cached: bool = Field(alias="factInventoryCached", default=False)
    fact_count: int = Field(alias="factCount", default=0)
    allocated_fact_count: int = Field(alias="allocatedFactCount", default=0)
    allocated_slots: int = Field(alias="allocatedSlots", default=0)
    backfill_attempts: int = Field(alias="backfillAttempts", default=0)
    backfill_accepted: int = Field(alias="backfillAccepted", default=0)
    generation_calls: int = Field(alias="generationCalls")
    generation_batch_calls: int = Field(alias="generationBatchCalls", default=0)
    max_generation_batch_size: int = Field(alias="maxGenerationBatchSize", default=0)
    validation_calls: int = Field(alias="validationCalls")
    validation_batch_calls: int = Field(alias="validationBatchCalls", default=0)
    max_validation_batch_size: int = Field(alias="maxValidationBatchSize", default=0)
    ollama_calls: int = Field(alias="ollamaCalls")
    embedding_calls: int = Field(alias="embeddingCalls", default=0)
    candidates_generated: int = Field(alias="candidatesGenerated")
    candidates_validated: int = Field(alias="candidatesValidated")
    candidates_accepted: int = Field(alias="candidatesAccepted")
    candidates_rejected: int = Field(alias="candidatesRejected")
    duplicates_removed: int = Field(alias="duplicatesRemoved")
    semantic_duplicates_removed: int = Field(alias="semanticDuplicatesRemoved", default=0)
    candidates_rejected_invalid: int = Field(alias="candidatesRejectedInvalid", default=0)
    candidates_rejected_validation: int = Field(
        alias="candidatesRejectedValidation",
        default=0,
    )
    candidates_rejected_unsupported_claims: int = Field(
        alias="candidatesRejectedUnsupportedClaims",
        default=0,
    )
    candidates_rejected_balancing: int = Field(
        alias="candidatesRejectedBalancing",
        default=0,
    )
    candidates_rejected_pre_validation: int = Field(
        alias="candidatesRejectedPreValidation",
        default=0,
    )
    candidates_rejected_qualifier_mismatch: int = Field(
        alias="candidatesRejectedQualifierMismatch",
        default=0,
    )
    candidates_rejected_modal_escalation: int = Field(
        alias="candidatesRejectedModalEscalation",
        default=0,
    )
    schedule_count: int = Field(alias="scheduleCount", default=0)
    scenario_or_clarification_minimum: int = Field(
        alias="scenarioOrClarificationMinimum",
        default=0,
    )
    timeout_failures: int = Field(alias="timeoutFailures", default=0)
    final_count: int = Field(alias="finalCount")
    saved_count: int = Field(
        alias="savedCount",
        default=0,
        description="Seeds saved to Firebase for this run (0 when save=false).",
    )
    elapsed_ms: int = Field(
        alias="elapsedMs",
        default=0,
        description="Wall-clock runtime of starter generation in milliseconds.",
    )
    status: str = Field(
        default="partial",
        description="Baseline outcome for this run: ready, partial, or failed.",
    )
    top_up: bool = Field(
        default=False,
        alias="topUp",
        description="True when this run was a top-up against existing Firebase seeds.",
    )
    existing_count: int = Field(
        default=0,
        alias="existingCount",
        description="Seeds already present in Firebase before this run (top-up).",
    )
    missing_count: int = Field(
        default=0,
        alias="missingCount",
        description="Seeds still needed to reach targetCount at start of top-up.",
    )
    total_count: int = Field(
        default=0,
        alias="totalCount",
        description=(
            "Projected course total after this run "
            "(existing + newly saved/accepted). Equals finalCount when not top-up."
        ),
    )


class StarterSeedGenerateResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    model: str
    target_count: int = Field(alias="targetCount")
    seeds: list[GeneratedSeedExample]
    progress: StarterSeedProgress
    persistence: StarterSeedPersistence | None = None
    local_snapshot_path: str | None = Field(default=None, alias="localSnapshotPath")


class StarterGenerationStatusResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    active: bool
    course_id: str | None = Field(default=None, alias="courseId")
    operation: str | None = Field(
        default=None,
        description="automatic | manual | top_up when a job is active",
    )
    started_at: str | None = Field(default=None, alias="startedAt")


class SeedReviewRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    review_status: str = Field(
        alias="reviewStatus",
        description="generated | approved | rejected | edited",
    )
    question: str | None = None
    answer: str | None = None
    review_notes: str | None = Field(default=None, alias="reviewNotes")


class SeedReviewRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str | None = None
    question: str | None = None
    answer: str | None = None
    instruction: str | None = None
    response: str | None = None
    fact_id: str | None = Field(default=None, alias="factId")
    evidence_quote: str | None = Field(default=None, alias="evidenceQuote")
    source_chunk_ids: list[str] | None = Field(default=None, alias="sourceChunkIds")
    origin: str | None = None
    status: str | None = None
    review_status: str | None = Field(default=None, alias="reviewStatus")
    review_notes: str | None = Field(default=None, alias="reviewNotes")
    original_question: str | None = Field(default=None, alias="originalQuestion")
    original_answer: str | None = Field(default=None, alias="originalAnswer")
    was_edited: bool | None = Field(
        default=None,
        alias="wasEdited",
        description="True when the seed was human-edited; survives later approval.",
    )
    validation: SeedValidationResult | None = None


class SeedReviewResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    seed_id: str = Field(alias="seedId")
    seed: SeedReviewRecord
    firebase_path: str = Field(alias="firebasePath")


class CourseSeedListResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    count: int
    firebase_path: str = Field(alias="firebasePath")
    seeds: list[SeedReviewRecord]


class SeedQualityCheckRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    review_statuses: list[str] | None = Field(
        default=None,
        alias="reviewStatuses",
        description="Optional filter; default inspects all fetched seeds.",
    )


class SeedQualityCheckResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    firebase_path: str = Field(alias="firebasePath")
    report: dict


class SeedExportApprovedRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # Reserved for future filters; approved-only is always enforced.
    include_edited_approved: bool = Field(
        default=True,
        alias="includeEditedApproved",
        description="Approved seeds that were previously edited are included.",
    )


class SeedExportApprovedResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    summary: dict


class ApprovedExportStatusResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    exists: bool
    export_path: str = Field(alias="exportPath")
    example_count: int = Field(alias="exampleCount")
    source_file: str = Field(alias="sourceFile")


class PrepareTrainingSplitRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    split_seed: int | None = Field(
        default=None,
        alias="splitSeed",
        description="Optional override; default is the fixed project seed 360.",
    )


class PrepareTrainingSplitResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    summary: dict


class FactInventoryRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    force_refresh: bool = Field(
        default=False,
        alias="forceRefresh",
        description="When true, rebuild the fact inventory instead of using cache.",
    )


class FactInventoryItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    fact_id: str = Field(alias="factId")
    statement: str
    importance: str
    importance_score: float = Field(alias="importanceScore", ge=0.0, le=1.0)
    student_ask_likelihood: float = Field(
        alias="studentAskLikelihood", ge=0.0, le=1.0
    )
    complexity: int = Field(ge=1)
    usefulness_score: float = Field(alias="usefulnessScore", ge=0.0, le=1.0)
    source_chunk_ids: list[str] = Field(alias="sourceChunkIds")
    evidence_quote: str = Field(alias="evidenceQuote")
    kind: str
    scope: str
    series_key: str | None = Field(alias="seriesKey", default=None)
    assignment_group: str | None = Field(alias="assignmentGroup", default=None)
    series_ordinal: int | None = Field(alias="seriesOrdinal", default=None)


class FactInventoryResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    model: str
    fact_count: int = Field(alias="factCount")
    dropped_count: int = Field(alias="droppedCount", default=0)
    duplicates_removed: int = Field(alias="duplicatesRemoved", default=0)
    fallback_used: bool = Field(alias="fallbackUsed", default=False)
    cached: bool = Field(default=False)
    counts_by_scope: dict[str, int] = Field(alias="countsByScope", default_factory=dict)
    counts_by_kind: dict[str, int] = Field(alias="countsByKind", default_factory=dict)
    counts_by_series: dict[str, int] = Field(
        alias="countsBySeries", default_factory=dict
    )
    facts: list[FactInventoryItem]


class FactAllocationRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    target_count: int = Field(alias="targetCount", default=50, ge=1, le=200)
    force_refresh: bool = Field(
        default=False,
        alias="forceRefresh",
        description="When true, rebuild the fact inventory instead of using cache.",
    )


class FactAllocationItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    fact_id: str = Field(alias="factId")
    slot_count: int = Field(alias="slotCount", ge=0)
    desired_slots: int = Field(alias="desiredSlots", ge=0)
    ranking_score: float = Field(alias="rankingScore", ge=0.0, le=1.0)
    suggested_styles: list[str] = Field(alias="suggestedStyles", default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    cap_reasons: list[str] = Field(alias="capReasons", default_factory=list)


class FactAllocationSkippedFact(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    fact_id: str = Field(alias="factId")
    desired_slots: int = Field(alias="desiredSlots", ge=0)
    ranking_score: float = Field(alias="rankingScore", ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class FactAllocationCappedFact(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    fact_id: str = Field(alias="factId")
    desired_slots: int = Field(alias="desiredSlots", ge=0)
    slot_count: int = Field(alias="slotCount", ge=0)
    cap_reasons: list[str] = Field(alias="capReasons", default_factory=list)


class FactAllocationSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    target_count: int = Field(alias="targetCount")
    allocated_slots: int = Field(alias="allocatedSlots")
    by_scope: dict[str, int] = Field(alias="byScope", default_factory=dict)
    by_kind: dict[str, int] = Field(alias="byKind", default_factory=dict)
    by_series: dict[str, int] = Field(alias="bySeries", default_factory=dict)
    skipped_facts: list[FactAllocationSkippedFact] = Field(
        alias="skippedFacts", default_factory=list
    )
    capped_facts: list[FactAllocationCappedFact] = Field(
        alias="cappedFacts", default_factory=list
    )
    caps: dict[str, Any] = Field(default_factory=dict)
    course_wide_allocated: int = Field(alias="courseWideAllocated", default=0)
    course_wide_reserve: int = Field(alias="courseWideReserve", default=0)


class FactAllocationRankingItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    fact_id: str = Field(alias="factId")
    ranking_score: float = Field(alias="rankingScore", ge=0.0, le=1.0)
    desired_slots: int = Field(alias="desiredSlots", ge=0)
    slot_count: int = Field(alias="slotCount", ge=0)
    importance: str | None = None
    usefulness_score: float | None = Field(alias="usefulnessScore", default=None)
    student_ask_likelihood: float | None = Field(
        alias="studentAskLikelihood", default=None
    )
    complexity: int | None = None
    kind: str | None = None
    scope: str | None = None
    series_key: str | None = Field(alias="seriesKey", default=None)
    source_chunk_ids: list[str] = Field(alias="sourceChunkIds", default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    cap_reasons: list[str] = Field(alias="capReasons", default_factory=list)


class FactAllocationResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    model: str
    fact_count: int = Field(alias="factCount")
    dropped_count: int = Field(alias="droppedCount", default=0)
    duplicates_removed: int = Field(alias="duplicatesRemoved", default=0)
    fallback_used: bool = Field(alias="fallbackUsed", default=False)
    facts: list[FactInventoryItem]
    allocations: list[FactAllocationItem]
    summary: FactAllocationSummary
    ranking: list[FactAllocationRankingItem] = Field(default_factory=list)

