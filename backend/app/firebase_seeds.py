"""Persist accepted AI-generated seeds to Firebase Realtime Database."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import HTTPException

from app.course_id import assert_valid_course_id
from app.seed_dedupe import normalize_question_for_dedupe
from app.seed_validation import canonicalize_validation_result

FIREBASE_REQUEST_TIMEOUT_SECONDS = 30.0


class FirebaseConfigurationError(Exception):
    """Raised when Firebase persistence is requested but not configured."""


def get_firebase_database_url() -> str:
    database_url = os.getenv("FIREBASE_DATABASE_URL", "").strip().rstrip("/")
    if not database_url:
        raise FirebaseConfigurationError(
            "Firebase is not configured for seed persistence. "
            "Set FIREBASE_DATABASE_URL in the backend environment."
        )
    return database_url


def get_firebase_auth_token() -> str | None:
    token = os.getenv("FIREBASE_AUTH_TOKEN", "").strip()
    return token or None


def course_seed_examples_path(course_id: str) -> str:
    safe_course_id = assert_valid_course_id(course_id)
    return f"courses/{safe_course_id}/seedExamples"


def course_seed_example_path(course_id: str, example_id: str) -> str:
    return f"{course_seed_examples_path(course_id)}/{example_id}"


def _request_url(path: str) -> str:
    base_url = get_firebase_database_url()
    auth_token = get_firebase_auth_token()
    url = f"{base_url}/{path}.json"
    if auth_token:
        return f"{url}?auth={auth_token}"
    return url


def derive_source_section(
    source_chunk_ids: list[str],
    chunk_sections: dict[str, str],
) -> str:
    titles: list[str] = []
    seen_titles: set[str] = set()
    for chunk_id in source_chunk_ids:
        title = chunk_sections.get(chunk_id, "").strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            titles.append(title)
    if titles:
        return titles[0]
    if source_chunk_ids:
        return ", ".join(source_chunk_ids)
    return "General"


def build_chunk_section_lookup(raw_chunks: list[Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            continue
        chunk_id = raw_chunk.get("chunkId") or raw_chunk.get("id")
        section_title = raw_chunk.get("sectionTitle") or raw_chunk.get("section_title")
        if isinstance(chunk_id, str) and chunk_id.strip():
            lookup[chunk_id.strip()] = (
                str(section_title).strip() if isinstance(section_title, str) else ""
            )
    return lookup


def build_firebase_seed_record(
    seed: dict[str, Any],
    *,
    source_section: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    question = str(seed["question"]).strip()
    answer = str(seed["answer"]).strip()
    normalized_key = normalize_question_for_dedupe(question)
    timestamp = created_at or datetime.now(timezone.utc).isoformat()

    validation = seed.get("validation")
    normalized_validation = None
    if isinstance(validation, dict):
        normalized_validation = canonicalize_validation_result(validation)

    return {
        "question": question,
        "instruction": question,
        "answer": answer,
        "response": answer,
        "category": str(seed.get("category", "general")).strip() or "general",
        "questionType": str(seed.get("questionType", "direct")).strip() or "direct",
        "sourceChunkIds": list(seed.get("sourceChunkIds", [])),
        "sourceSection": source_section,
        "difficulty": "Medium",
        "directlyAnswered": True,
        "origin": "ai_generated",
        # Legacy generation badge + Phase 8 review field (not human-approved yet).
        "status": "generated",
        "reviewStatus": "generated",
        "normalizedQuestionKey": normalized_key,
        "createdAt": timestamp,
        "validation": normalized_validation,
        "factId": (
            str(seed["factId"]).strip()
            if isinstance(seed.get("factId"), str) and str(seed.get("factId")).strip()
            else None
        ),
        "evidenceQuote": (
            str(seed["evidenceQuote"]).strip()
            if isinstance(seed.get("evidenceQuote"), str)
            and str(seed.get("evidenceQuote")).strip()
            else None
        ),
    }


def collect_normalized_question_keys(existing_seeds: dict[str, Any] | None) -> set[str]:
    keys: set[str] = set()
    if not existing_seeds:
        return keys

    for raw_seed in existing_seeds.values():
        if not isinstance(raw_seed, dict):
            continue
        stored_key = raw_seed.get("normalizedQuestionKey")
        if isinstance(stored_key, str) and stored_key.strip():
            keys.add(stored_key.strip())
        question = raw_seed.get("question") or raw_seed.get("instruction")
        if isinstance(question, str) and question.strip():
            keys.add(normalize_question_for_dedupe(question))
    return keys


async def fetch_course_seed_examples(course_id: str) -> dict[str, Any]:
    assert_valid_course_id(course_id)
    url = _request_url(course_seed_examples_path(course_id))

    try:
        async with httpx.AsyncClient(timeout=FIREBASE_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.get(url)
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=503,
            detail="Firebase request timed out while loading existing seed examples.",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail="Firebase is unavailable. Check FIREBASE_DATABASE_URL and network access.",
        ) from exc

    if response.status_code == 401:
        raise HTTPException(
            status_code=503,
            detail=(
                "Firebase rejected the request (401). Provide FIREBASE_AUTH_TOKEN "
                "if database rules require authentication."
            ),
        )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=503,
            detail=f"Firebase request failed with HTTP {response.status_code}.",
        )

    payload = response.json()
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=503,
            detail="Firebase returned an unexpected seedExamples payload shape.",
        )
    return payload


async def save_course_seed_example(
    course_id: str,
    record: dict[str, Any],
) -> str:
    """Create one seed example and return its Firebase push id."""
    assert_valid_course_id(course_id)
    url = _request_url(course_seed_examples_path(course_id))

    try:
        async with httpx.AsyncClient(timeout=FIREBASE_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=record)
            if response.status_code >= 400:
                raise HTTPException(
                    status_code=503,
                    detail=f"Firebase write failed with HTTP {response.status_code}.",
                )

            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
                raise HTTPException(
                    status_code=503,
                    detail="Firebase did not return a push id for the new seed example.",
                )
            push_id = payload["name"].strip()
            if not push_id:
                raise HTTPException(
                    status_code=503,
                    detail="Firebase returned an empty push id for the new seed example.",
                )

            stored_record = {**record, "id": push_id}
            put_url = _request_url(course_seed_example_path(course_id, push_id))
            put_response = await client.put(put_url, json=stored_record)
            if put_response.status_code >= 400:
                raise HTTPException(
                    status_code=503,
                    detail=f"Firebase finalize write failed with HTTP {put_response.status_code}.",
                )
    except HTTPException:
        raise
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=503,
            detail="Firebase request timed out while saving a seed example.",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail="Firebase is unavailable while saving seed examples.",
        ) from exc

    return push_id


async def persist_accepted_seeds(
    *,
    course_id: str,
    seeds: list[dict[str, Any]],
    chunk_sections: dict[str, str],
) -> dict[str, int]:
    """Save accepted validated seeds under courses/{courseId}/seedExamples."""
    generated_count = len(seeds)
    saved_count = 0
    already_existing_count = 0
    failed_to_save_count = 0

    if generated_count == 0:
        return {
            "generatedCount": 0,
            "savedCount": 0,
            "alreadyExistingCount": 0,
            "failedToSaveCount": 0,
        }

    existing = await fetch_course_seed_examples(course_id)
    seen_keys = collect_normalized_question_keys(existing)

    for seed in seeds:
        normalized_key = normalize_question_for_dedupe(str(seed.get("question", "")))
        if not normalized_key:
            failed_to_save_count += 1
            continue
        if normalized_key in seen_keys:
            already_existing_count += 1
            continue

        source_chunk_ids = seed.get("sourceChunkIds", [])
        if not isinstance(source_chunk_ids, list):
            source_chunk_ids = []
        source_section = derive_source_section(
            [str(item).strip() for item in source_chunk_ids if str(item).strip()],
            chunk_sections,
        )
        record = build_firebase_seed_record(seed, source_section=source_section)

        try:
            await save_course_seed_example(course_id, record)
        except HTTPException:
            failed_to_save_count += 1
            continue

        seen_keys.add(normalized_key)
        saved_count += 1

    return {
        "generatedCount": generated_count,
        "savedCount": saved_count,
        "alreadyExistingCount": already_existing_count,
        "failedToSaveCount": failed_to_save_count,
    }


async def patch_course_seed_example(
    course_id: str,
    seed_id: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Overwrite one seed example at courses/{courseId}/seedExamples/{seedId}."""
    safe_course_id = assert_valid_course_id(course_id)
    cleaned_id = str(seed_id).strip()
    if not cleaned_id:
        raise HTTPException(status_code=422, detail="seedId must not be empty.")

    stored = {**record, "id": cleaned_id}
    url = _request_url(course_seed_example_path(safe_course_id, cleaned_id))
    try:
        async with httpx.AsyncClient(timeout=FIREBASE_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.put(url, json=stored)
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=503,
            detail="Firebase request timed out while updating a seed example.",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail="Firebase is unavailable while updating seed examples.",
        ) from exc

    if response.status_code == 401:
        raise HTTPException(
            status_code=503,
            detail=(
                "Firebase rejected the request (401). Provide FIREBASE_AUTH_TOKEN "
                "if database rules require authentication."
            ),
        )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=503,
            detail=f"Firebase update failed with HTTP {response.status_code}.",
        )
    return stored
