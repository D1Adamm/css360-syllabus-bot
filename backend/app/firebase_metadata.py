"""Firebase helpers for course metadata updates (starter seed job status)."""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import HTTPException

from app.course_id import assert_valid_course_id
from app.firebase_seeds import (
    FIREBASE_REQUEST_TIMEOUT_SECONDS,
    FirebaseConfigurationError,
    _request_url,
)


def course_metadata_path(course_id: str) -> str:
    safe_course_id = assert_valid_course_id(course_id)
    return f"courses/{safe_course_id}/metadata"


def starter_seed_generation_path(course_id: str) -> str:
    return f"{course_metadata_path(course_id)}/starterSeedGeneration"


async def fetch_course_metadata(course_id: str) -> dict[str, Any] | None:
    assert_valid_course_id(course_id)
    url = _request_url(course_metadata_path(course_id))

    try:
        async with httpx.AsyncClient(timeout=FIREBASE_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.get(url)
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Firebase is unavailable while reading course metadata.",
        ) from exc

    if response.status_code == 401:
        raise HTTPException(
            status_code=503,
            detail=(
                "Firebase rejected the metadata request (401). Provide "
                "FIREBASE_AUTH_TOKEN if database rules require authentication."
            ),
        )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=503,
            detail=f"Firebase metadata request failed with HTTP {response.status_code}.",
        )

    payload = response.json()
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=503,
            detail="Firebase returned an unexpected metadata payload shape.",
        )
    return payload


async def patch_starter_seed_generation(
    course_id: str,
    updates: dict[str, Any],
) -> None:
    """Patch courses/{courseId}/metadata/starterSeedGeneration (merge)."""
    assert_valid_course_id(course_id)
    url = _request_url(starter_seed_generation_path(course_id))

    try:
        async with httpx.AsyncClient(timeout=FIREBASE_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.patch(url, json=updates)
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Firebase is unavailable while updating starter seed status.",
        ) from exc

    if response.status_code == 401:
        raise HTTPException(
            status_code=503,
            detail=(
                "Firebase rejected the starter-seed status update (401). Provide "
                "FIREBASE_AUTH_TOKEN if database rules require authentication."
            ),
        )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=503,
            detail=(
                "Firebase starter-seed status update failed with "
                f"HTTP {response.status_code}."
            ),
        )


async def read_starter_seed_generation_status(course_id: str) -> str | None:
    """Return the current starterSeedGeneration.status, or None if missing."""
    try:
        metadata = await fetch_course_metadata(course_id)
    except (FirebaseConfigurationError, HTTPException):
        return None

    if not metadata:
        return None
    block = metadata.get("starterSeedGeneration")
    if not isinstance(block, dict):
        return None
    status = block.get("status")
    return status if isinstance(status, str) else None


async def best_effort_patch_starter_seed_generation(
    course_id: str,
    updates: dict[str, Any],
) -> bool:
    """Write starter status without failing the caller when Firebase is unavailable."""
    try:
        await patch_starter_seed_generation(course_id, updates)
        return True
    except (FirebaseConfigurationError, HTTPException, ValueError):
        return False
