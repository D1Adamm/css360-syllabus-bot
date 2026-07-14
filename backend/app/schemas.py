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
