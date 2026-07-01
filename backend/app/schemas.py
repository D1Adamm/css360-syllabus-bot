from pydantic import BaseModel, ConfigDict, Field


class BaseModelGenerateRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Student question to send to the base model")


class BaseModelGenerateResponse(BaseModel):
    answer: str
    model: str
    response_type: str = Field(alias="responseType")

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
    question: str = Field(..., min_length=1, description="Student question for RAG answer generation")
    top_k: int = Field(
        default=3,
        alias="topK",
        ge=1,
        le=20,
        description="Number of syllabus chunks to retrieve for context",
    )


class RagGenerateSource(BaseModel):
    section: str


class RagGenerateResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    answer: str
    model: str
    sources: list[RagGenerateSource]
    retrieved_chunks: list[RagRetrieveResult] = Field(alias="retrievedChunks")
    response_type: str = Field(default="rag", alias="responseType")
