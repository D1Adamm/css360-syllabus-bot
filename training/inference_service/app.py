"""CSS 360 fine-tuned (QLoRA) inference service.

Loads Llama 3.2 3B Instruct + PEFT LoRA adapter once at startup.
Does not merge the adapter into the base model.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from helpers import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_SEED,
    assert_hf_auth_available,
    resolve_adapter_path,
    resolve_model_id,
    resolve_port,
    validate_question,
)


class GenerateRequest(BaseModel):
    question: str = Field(..., description="Student question to answer")


class GenerateResponse(BaseModel):
    answer: str
    model: str
    adapterLoaded: bool
    generationSeconds: float


class HealthResponse(BaseModel):
    status: str
    model: str
    adapterPath: str
    adapterLoaded: bool
    cudaAvailable: bool
    hostname: str
    port: int


def assert_cuda_available() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. The fine-tuned inference service requires one GPU."
        )


def _model_device(model: Any) -> Any:
    try:
        return model.device
    except Exception:  # noqa: BLE001
        return next(model.parameters()).device


class InferenceEngine:
    """Holds tokenizer + PEFT model loaded once for the process lifetime."""

    def __init__(self, *, model_id: str, adapter_path: Path) -> None:
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.adapter_loaded = False

    def load(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        assert_cuda_available()
        assert_hf_auth_available()

        if not self.adapter_path.exists():
            raise RuntimeError(f"Adapter path does not exist: {self.adapter_path}")

        use_bf16 = torch.cuda.is_bf16_supported()
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

        print(f"Loading tokenizer/model: {self.model_id}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=compute_dtype,
        )
        print(f"Loading LoRA adapter (not merged): {self.adapter_path}", flush=True)
        model = PeftModel.from_pretrained(base, str(self.adapter_path))
        model.eval()

        self.tokenizer = tokenizer
        self.model = model
        self.adapter_loaded = True
        print("Model + adapter ready.", flush=True)

    def generate(
        self,
        question: str,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        seed: int = DEFAULT_SEED,
    ) -> tuple[str, float]:
        import torch

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded")

        cleaned = validate_question(question)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        messages = [{"role": "user", "content": cleaned}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = _model_device(self.model)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }

        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generate_kwargs)
        elapsed = time.perf_counter() - start

        prompt_length = inputs["input_ids"].shape[-1]
        completion_ids = output_ids[0, prompt_length:]
        answer = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        return answer, elapsed


ENGINE = InferenceEngine(
    model_id=resolve_model_id(),
    adapter_path=resolve_adapter_path(),
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    import socket

    hostname = socket.gethostname()
    port = resolve_port()
    print(f"Starting inference service on host={hostname} port={port}", flush=True)
    try:
        ENGINE.load()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR during model load: {exc}", flush=True)
        raise
    print(f"Listening on http://0.0.0.0:{port} (hostname={hostname})", flush=True)
    yield


app = FastAPI(
    title="CSS 360 Fine-Tuned Inference Service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    import socket

    import torch

    return HealthResponse(
        status="ok" if ENGINE.adapter_loaded else "loading",
        model=ENGINE.model_id,
        adapterPath=str(ENGINE.adapter_path),
        adapterLoaded=ENGINE.adapter_loaded,
        cudaAvailable=torch.cuda.is_available(),
        hostname=socket.gethostname(),
        port=resolve_port(),
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(body: GenerateRequest) -> GenerateResponse:
    try:
        question = validate_question(body.question)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not ENGINE.adapter_loaded or ENGINE.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        answer, elapsed = ENGINE.generate(question)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    return GenerateResponse(
        answer=answer,
        model=ENGINE.model_id,
        adapterLoaded=True,
        generationSeconds=elapsed,
    )


def main() -> None:
    import socket

    import uvicorn

    host = (os.environ.get("INFERENCE_HOST") or "0.0.0.0").strip()
    port = resolve_port()
    hostname = socket.gethostname()
    print(f"Node hostname: {hostname}", flush=True)
    print(f"Listening port: {port}", flush=True)
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
