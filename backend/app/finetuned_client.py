"""HTTP client for the remote fine-tuned (QLoRA) inference service.

The service URL comes from FINETUNED_SERVICE_URL (e.g. a Tillicum node that
changes between Slurm jobs). No hostnames are hardcoded here.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import HTTPException

DEFAULT_FINETUNED_TIMEOUT_SECONDS = 120.0


def get_finetuned_service_url() -> str | None:
    """Return the configured fine-tuned service base URL, or None if unset."""
    raw = os.getenv("FINETUNED_SERVICE_URL", "").strip()
    if not raw:
        return None
    return raw.rstrip("/")


def get_finetuned_timeout_seconds() -> float:
    raw = os.getenv("FINETUNED_SERVICE_TIMEOUT_SECONDS")
    if raw is None or not raw.strip():
        return DEFAULT_FINETUNED_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_FINETUNED_TIMEOUT_SECONDS
    return max(1.0, value)


def require_finetuned_service_url() -> str:
    """Return the configured URL or raise a clear 503 if missing."""
    url = get_finetuned_service_url()
    if url is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Fine-tuned inference service is not configured. "
                "Set FINETUNED_SERVICE_URL to the running service base URL "
                "(for example http://<tillicum-node>:<port>) and try again."
            ),
        )
    return url


def _validate_generate_payload(data: Any) -> dict[str, Any]:
    """Validate the remote /generate JSON body; raise 502 on malformed data."""
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=502,
            detail="Fine-tuned service returned a malformed response (expected a JSON object).",
        )

    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise HTTPException(
            status_code=502,
            detail="Fine-tuned service returned an empty or missing answer.",
        )

    model = data.get("model")
    if not isinstance(model, str) or not model.strip():
        raise HTTPException(
            status_code=502,
            detail="Fine-tuned service returned a malformed response (missing model).",
        )

    adapter_loaded = data.get("adapterLoaded")
    if not isinstance(adapter_loaded, bool):
        raise HTTPException(
            status_code=502,
            detail=(
                "Fine-tuned service returned a malformed response "
                "(adapterLoaded must be a boolean)."
            ),
        )
    if not adapter_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "Fine-tuned service is reachable but the LoRA adapter is not loaded. "
                "Check ADAPTER_PATH on the inference service and restart it."
            ),
        )

    generation_seconds = data.get("generationSeconds")
    if generation_seconds is not None and not isinstance(generation_seconds, (int, float)):
        raise HTTPException(
            status_code=502,
            detail=(
                "Fine-tuned service returned a malformed response "
                "(generationSeconds must be a number when present)."
            ),
        )

    return {
        "answer": answer.strip(),
        "model": model.strip(),
        "adapter_loaded": adapter_loaded,
        "generation_seconds": (
            float(generation_seconds) if generation_seconds is not None else None
        ),
    }


async def check_finetuned_service_health() -> dict[str, Any]:
    """Probe GET {FINETUNED_SERVICE_URL}/health.

    Returns a small status dict. Raises HTTPException on misconfiguration or
    when the service is unreachable / unhealthy.
    """
    base_url = require_finetuned_service_url()
    timeout = get_finetuned_timeout_seconds()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{base_url}/health")
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Fine-tuned inference service timed out while checking health. "
                "Ensure the Slurm job is running and FINETUNED_SERVICE_URL is correct."
            ),
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Fine-tuned inference service is unavailable. "
                "Check that the service is running and FINETUNED_SERVICE_URL points to it."
            ),
        ) from exc

    if response.status_code != 200:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Fine-tuned inference service health check failed "
                f"(HTTP {response.status_code}): {response.text[:200]}"
            ),
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="Fine-tuned service health endpoint returned invalid JSON.",
        ) from exc

    if not isinstance(data, dict):
        raise HTTPException(
            status_code=502,
            detail="Fine-tuned service health endpoint returned a malformed response.",
        )

    return {
        "status": data.get("status", "unknown"),
        "model": data.get("model"),
        "adapterLoaded": data.get("adapterLoaded"),
        "hostname": data.get("hostname"),
        "port": data.get("port"),
        "serviceUrl": base_url,
    }


async def generate_finetuned_response(question: str) -> dict[str, Any]:
    """Call POST {FINETUNED_SERVICE_URL}/generate and return a validated result.

    Never falls back to simulated text. Failures raise HTTPException.
    """
    trimmed = question.strip()
    if not trimmed:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    base_url = require_finetuned_service_url()
    timeout = get_finetuned_timeout_seconds()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url}/generate",
                json={"question": trimmed},
            )
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Fine-tuned inference service timed out. "
                "The model may still be loading or the node may be unreachable."
            ),
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Fine-tuned inference service is unavailable. "
                "Check that the service is running and FINETUNED_SERVICE_URL points to it."
            ),
        ) from exc

    if response.status_code >= 500:
        raise HTTPException(
            status_code=503,
            detail=(
                "Fine-tuned inference service returned a server error. "
                f"(HTTP {response.status_code}): {response.text[:200]}"
            ),
        )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=(
                "Fine-tuned inference service rejected the request. "
                f"(HTTP {response.status_code}): {response.text[:200]}"
            ),
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="Fine-tuned service returned invalid JSON.",
        ) from exc

    validated = _validate_generate_payload(data)
    return {
        "answer": validated["answer"],
        "model": validated["model"],
        "adapter_loaded": validated["adapter_loaded"],
        "generation_seconds": validated["generation_seconds"],
        "response_type": "fineTuned",
    }
