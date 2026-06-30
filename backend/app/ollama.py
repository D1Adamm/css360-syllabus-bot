import os

import httpx
from fastapi import HTTPException

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))


async def generate_base_model_response(question: str) -> dict[str, str]:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": question,
        "stream": False,
    }

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
        "model": data.get("model", OLLAMA_MODEL),
        "response_type": "base",
    }
