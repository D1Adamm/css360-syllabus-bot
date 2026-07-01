import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.ollama import generate_base_model_response
from app.rag import generate_rag_answer, retrieve_syllabus_chunks
from app.schemas import (
    BaseModelGenerateRequest,
    BaseModelGenerateResponse,
    RagGenerateRequest,
    RagGenerateResponse,
    RagGenerateSource,
    RagRetrieveRequest,
    RagRetrieveResponse,
    RagRetrieveResult,
)

app = FastAPI(title="CSS360 Syllabus Model Backend")

CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "css360-syllabus-model-backend",
    }


@app.post("/base-model/generate", response_model=BaseModelGenerateResponse)
async def generate_base_model(
    request: BaseModelGenerateRequest,
) -> BaseModelGenerateResponse:
    result = await generate_base_model_response(request.question.strip())
    return BaseModelGenerateResponse(**result)


@app.post("/rag/retrieve", response_model=RagRetrieveResponse)
async def retrieve_rag_chunks(request: RagRetrieveRequest) -> RagRetrieveResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    embedding_model, results = await retrieve_syllabus_chunks(
        question=question,
        top_k=request.top_k,
    )

    return RagRetrieveResponse(
        embedding_model=embedding_model,
        results=[RagRetrieveResult(**result) for result in results],
    )


@app.post("/rag/generate", response_model=RagGenerateResponse)
async def generate_rag_response(request: RagGenerateRequest) -> RagGenerateResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    result = await generate_rag_answer(question=question, top_k=request.top_k)

    return RagGenerateResponse(
        answer=result["answer"],
        model=result["model"],
        sources=[RagGenerateSource(**source) for source in result["sources"]],
        retrieved_chunks=[RagRetrieveResult(**chunk) for chunk in result["retrieved_chunks"]],
    )
