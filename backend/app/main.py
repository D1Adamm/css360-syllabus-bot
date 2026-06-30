from fastapi import FastAPI

from app.ollama import generate_base_model_response
from app.schemas import BaseModelGenerateRequest, BaseModelGenerateResponse

app = FastAPI(title="CSS360 Syllabus Model Backend")


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
