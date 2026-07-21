"""Pure helpers for the fine-tuned inference service (no GPU imports)."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 160
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_SEED = 360


def resolve_adapter_path() -> Path:
    raw = (
        os.environ.get("ADAPTER_PATH")
        or os.environ.get("LORA_ADAPTER_PATH")
        or ""
    ).strip()
    if not raw:
        user = os.environ.get("USER", "USER")
        raw = f"/gpfs/projects/simswe/{user}/training_outputs/css-360-qlora/adapter"
    return Path(raw)


def resolve_model_id() -> str:
    return (os.environ.get("MODEL_ID") or DEFAULT_MODEL_ID).strip()


def resolve_port() -> int:
    raw = (os.environ.get("INFERENCE_PORT") or os.environ.get("PORT") or "8001").strip()
    try:
        port = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid INFERENCE_PORT/PORT value: {raw!r}") from exc
    if port < 1 or port > 65535:
        raise RuntimeError(f"Port out of range: {port}")
    return port


def assert_hf_auth_available() -> None:
    token_path = (os.environ.get("HF_TOKEN_PATH") or "").strip()
    if token_path and Path(token_path).is_file() and Path(token_path).stat().st_size > 0:
        if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
            token = Path(token_path).read_text(encoding="utf-8").strip()
            if token:
                os.environ["HF_TOKEN"] = token
                os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        return
    if (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip():
        return
    try:
        from huggingface_hub import HfFolder

        cached = HfFolder.get_token()
        if cached:
            return
    except Exception:  # noqa: BLE001
        pass
    raise RuntimeError(
        "Hugging Face authentication is unavailable. "
        "Set HF_TOKEN_PATH to a non-empty token file, or export HF_TOKEN."
    )


def validate_question(question: str) -> str:
    cleaned = (question or "").strip()
    if not cleaned:
        raise ValueError("question must be a non-blank string")
    return cleaned
