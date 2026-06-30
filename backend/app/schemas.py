from pydantic import BaseModel, Field


class BaseModelGenerateRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Student question to send to the base model")


class BaseModelGenerateResponse(BaseModel):
    answer: str
    model: str
    response_type: str = Field(alias="responseType")

    model_config = {"populate_by_name": True}
