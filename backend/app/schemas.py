from pydantic import BaseModel, ConfigDict, Field


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


class RagRetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Student question for syllabus retrieval")
    top_k: int = Field(
        default=3,
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
        default=3,
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


class CourseChunksResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    chunk_count: int = Field(alias="chunkCount")
    chunks: list[CourseChunkMetadata]


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
    origin: str
    status: str
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
    generation_calls: int = Field(alias="generationCalls")
    validation_calls: int = Field(alias="validationCalls")
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
    schedule_count: int = Field(alias="scheduleCount", default=0)
    scenario_or_clarification_minimum: int = Field(
        alias="scenarioOrClarificationMinimum",
        default=0,
    )
    timeout_failures: int = Field(alias="timeoutFailures", default=0)
    final_count: int = Field(alias="finalCount")


class StarterSeedGenerateResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    model: str
    target_count: int = Field(alias="targetCount")
    seeds: list[GeneratedSeedExample]
    progress: StarterSeedProgress
    persistence: StarterSeedPersistence | None = None

