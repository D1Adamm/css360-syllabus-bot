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
    source_chunk_ids: list[str] = Field(alias="sourceChunkIds")
    origin: str
    status: str
    validation: "SeedValidationResult | None" = None


class SeedValidationResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    grounded: bool
    correct: bool
    clear: bool
    useful: bool
    score: float = Field(ge=0.0, le=1.0)
    reason: str


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
    generation_calls: int = Field(alias="generationCalls")
    validation_calls: int = Field(alias="validationCalls")
    ollama_calls: int = Field(alias="ollamaCalls")
    candidates_generated: int = Field(alias="candidatesGenerated")
    candidates_validated: int = Field(alias="candidatesValidated")
    candidates_accepted: int = Field(alias="candidatesAccepted")
    candidates_rejected: int = Field(alias="candidatesRejected")
    duplicates_removed: int = Field(alias="duplicatesRemoved")
    final_count: int = Field(alias="finalCount")


class StarterSeedGenerateResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course_id: str = Field(alias="courseId")
    model: str
    target_count: int = Field(alias="targetCount")
    seeds: list[GeneratedSeedExample]
    progress: StarterSeedProgress
    persistence: StarterSeedPersistence | None = None

