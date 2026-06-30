import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.ollama import generate_base_model_response
from app.schemas import BaseModelGenerateRequest, BaseModelGenerateResponse

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
