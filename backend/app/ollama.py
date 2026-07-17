import os

import httpx
from fastapi import HTTPException

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))
DEFAULT_EMBED_MODEL = os.getenv("STARTER_EMBED_MODEL", "nomic-embed-text")


async def generate_ollama_completion(
    prompt: str,
    *,
    model: str | None = None,
    response_format: str | None = None,
    think: bool | None = None,
) -> dict[str, str]:
    """Generate a completion from Ollama.

    Defaults to OLLAMA_MODEL (llama3.2:3b) so Base Model and RAG stay unchanged.
    Pass model= to use a different local model (e.g. qwen3:4b for seed generation).
    Pass response_format="json" to request structured JSON from Ollama.
    Pass think=False for models like qwen3 that otherwise put output in `thinking`
    and leave `response` empty.
    """
    selected_model = model or OLLAMA_MODEL
    payload: dict[str, object] = {
        "model": selected_model,
        "prompt": prompt,
        "stream": False,
    }
    if response_format is not None:
        payload["format"] = response_format
    if think is not None:
        payload["think"] = think

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
            )
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=503,
            detail="Ollama request timed out. Ensure Ollama is running and responsive.",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail="Ollama is unavailable. Start Ollama locally and try again.",
        ) from exc

    if response.status_code >= 500:
        raise HTTPException(
            status_code=503,
            detail="Ollama returned a server error. Ensure the model is available locally.",
        )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama rejected the request: {response.text}",
        )

    data = response.json()
    answer = data.get("response", "").strip()
    if not answer:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an empty response.",
        )

    return {
        "answer": answer,
        "model": data.get("model", selected_model),
    }


async def generate_base_model_response(question: str) -> dict[str, str]:
    result = await generate_ollama_completion(question)
    return {
        **result,
        "response_type": "base",
    }


async def embed_ollama_texts(
    texts: list[str],
    *,
    model: str | None = None,
) -> dict[str, object]:
    """Create embeddings for multiple texts using Ollama."""
    selected_model = model or DEFAULT_EMBED_MODEL
    cleaned_texts = [text.strip() for text in texts if text.strip()]
    if not cleaned_texts:
        return {"embeddings": [], "model": selected_model}

    payload = {
        "model": selected_model,
        "input": cleaned_texts,
    }

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json=payload,
            )
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=503,
            detail="Ollama embedding request timed out. Ensure Ollama is running and responsive.",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail="Ollama embeddings are unavailable. Start Ollama locally and try again.",
        ) from exc

    if response.status_code >= 500:
        raise HTTPException(
            status_code=503,
            detail="Ollama returned a server error for embeddings.",
        )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama rejected the embedding request: {response.text}",
        )

    data = response.json()
    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list):
        raise HTTPException(
            status_code=502,
            detail="Ollama returned malformed embeddings.",
        )
    return {
        "embeddings": embeddings,
        "model": data.get("model", selected_model),
    }
