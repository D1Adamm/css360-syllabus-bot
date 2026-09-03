"""HTTP client for the remote fine-tuned (QLoRA) inference service.

The service URL comes from FINETUNED_SERVICE_URL (e.g. a Tillicum node that
changes between Slurm jobs). No hostnames are hardcoded here.

Course isolation at the boundary
--------------------------------
Every generate request names its course, and the response is refused unless it
names the same one back. That check is not ceremony: training is per course, and
the failure it guards against — CSS 350's question answered by CSS 360's adapter
— produces a plausible-looking answer with nothing wrong on its face. A wrong
answer that looks right is exactly the kind of failure that has to be caught by
a machine rather than by a reader.

The service is the other half: it resolves the adapter from the course id on the
request and has no course-agnostic default to fall back to.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping

import httpx
from fastapi import HTTPException

from app.upstream_errors import log_upstream_failure

DEFAULT_FINETUNED_TIMEOUT_SECONDS = 120.0


def get_finetuned_service_url() -> str | None:
    """Return the configured fine-tuned service base URL, or None if unset."""
    raw = os.getenv("FINETUNED_SERVICE_URL", "").strip()
    if not raw:
        return None
    return raw.rstrip("/")


logger = logging.getLogger(__name__)

def _log_upstream_failure(
    action: str, *, url: str, status_code: int, body: Any
) -> None:
    """Record an upstream failure server-side, in full.

    The body of a failed response from the inference service is the useful half
    of the diagnostic and the unsafe half of the answer. It is written by
    whatever answered — the service, or an SSH tunnel, or a proxy in front of
    it — so it can name a compute node, a listening port, a serving root under
    `/gpfs`, or the tunnel's own `localhost:<port>`. None of that may reach a
    browser through `/api/fine-tuned/*`, which needs no credential.

    So it goes here instead. An operator reading the backend log gets the whole
    body and the URL that produced it; the client gets the status code, which is
    the part it can act on.

    The shared implementation lives in `upstream_errors` — the local Ollama
    server needs the same rule — and this module's own logger is passed in so
    the record still says which upstream failed.
    """
    log_upstream_failure(
        logger, action, url=url, status_code=status_code, body=body
    )


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


def _validate_generate_payload(
    data: Any, *, expected_course_id: str | None = None
) -> dict[str, Any]:
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

    # The isolation check. A service that answered for a different course than
    # the one asked for has loaded the wrong adapter, and the answer must not
    # reach a student — it would be fluent, specific, and about another course's
    # syllabus. 502 because the remote gave a response this backend cannot use.
    returned_course_id = data.get("courseId")
    if expected_course_id is not None:
        if not isinstance(returned_course_id, str) or not returned_course_id.strip():
            raise HTTPException(
                status_code=502,
                detail=(
                    "Fine-tuned service returned no courseId. This backend "
                    "requires a per-course service; the remote may be running "
                    "an older single-adapter build."
                ),
            )
        if returned_course_id.strip() != expected_course_id:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Fine-tuned service answered for course "
                    f'"{returned_course_id.strip()}" when asked for '
                    f'"{expected_course_id}". The answer was discarded.'
                ),
            )

    model_version = data.get("modelVersion")

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
        "course_id": (
            returned_course_id.strip()
            if isinstance(returned_course_id, str) and returned_course_id.strip()
            else None
        ),
        "model_version": (
            model_version.strip()
            if isinstance(model_version, str) and model_version.strip()
            else None
        ),
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
        _log_upstream_failure(
            "health check",
            url=base_url,
            status_code=response.status_code,
            body=response.text,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Fine-tuned inference service health check failed "
                f"(HTTP {response.status_code}). See the backend log for the "
                "service's own response."
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

    courses = data.get("courses")
    return {
        "status": data.get("status", "unknown"),
        "model": data.get("model"),
        "adapterLoaded": data.get("adapterLoaded"),
        "hostname": data.get("hostname"),
        "port": data.get("port"),
        "serviceUrl": base_url,
        # Which courses the running service can actually answer for. Absent from
        # an older single-adapter build, which reads as an empty list rather
        # than an error: health is about reachability, and a version mismatch
        # surfaces on the first generate call with a message that says so.
        "courses": courses if isinstance(courses, list) else [],
        "secondsRemaining": data.get("secondsRemaining"),
    }


#: What one course entry in a public health response may say. The service also
#: knows where each adapter lives on disk; a browser is told which courses can
#: be answered and with which versions, which is the whole of what a status page
#: can act on.
PUBLIC_COURSE_FIELDS = ("courseId", "versions", "currentVersion")


def public_service_health(health: Mapping[str, Any]) -> dict[str, Any]:
    """The browser-safe view of the inference service's health.

    `hostname`, `port` and `serviceUrl` are dropped. They describe how to reach
    a machine rather than what the application is doing, and
    `GET /api/fine-tuned/health` is reachable without a credential — the same
    reasoning `db_serving_sessions.public_serving_session` already applies to a
    serving session's compute node and port. The values here are worse in one
    respect: `serviceUrl` is the SSH tunnel destination this backend dials, and
    `hostname` is the Tillicum compute node a Slurm allocation happened to land
    on.

    What survives is what a status page is for: whether the service answered,
    which base model it is running, whether an adapter is loaded, which courses
    it can serve and at which versions, and how much wall clock the allocation
    has left.

    The per-course entries are rebuilt from a fixed field list rather than
    forwarded, because they come from a remote service: a future build of it
    cannot widen this response by adding a key.

    `check_finetuned_service_health` is unchanged and still returns the exact
    hostname, port and URL. Nothing internal reads this view.
    """
    record: dict[str, Any] = {
        "status": health.get("status", "unknown"),
        "model": health.get("model"),
        "adapterLoaded": health.get("adapterLoaded"),
        "secondsRemaining": health.get("secondsRemaining"),
    }

    courses = health.get("courses")
    record["courses"] = [
        {
            field: course.get(field)
            for field in PUBLIC_COURSE_FIELDS
            if course.get(field) is not None
        }
        for course in (courses if isinstance(courses, list) else [])
        if isinstance(course, Mapping)
    ]
    return record


async def generate_finetuned_response(
    question: str,
    *,
    course_id: str,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Call POST {FINETUNED_SERVICE_URL}/generate and return a validated result.

    `course_id` is required and is sent with the request. There is no
    course-agnostic fine-tuned model to fall back to: each course has its own
    adapter, and a request that did not say which one would be asking the
    service to guess.

    `model_version` is the version the backend resolved from its own registry.
    Sending it makes the two sides checkable against each other — the response
    reports the version actually used, and a mismatch is visible rather than
    silent.

    Never falls back to simulated text. Failures raise HTTPException.
    """
    trimmed = question.strip()
    if not trimmed:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    safe_course_id = (course_id or "").strip()
    if not safe_course_id:
        raise HTTPException(
            status_code=422,
            detail="A course id is required to generate a fine-tuned answer.",
        )

    base_url = require_finetuned_service_url()
    timeout = get_finetuned_timeout_seconds()

    body: dict[str, Any] = {"question": trimmed, "courseId": safe_course_id}
    if model_version:
        body["modelVersion"] = model_version

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{base_url}/generate", json=body)
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
        _log_upstream_failure(
            "generation",
            url=base_url,
            status_code=response.status_code,
            body=response.text,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Fine-tuned inference service returned a server error "
                f"(HTTP {response.status_code}). See the backend log for the "
                "service's own response."
            ),
        )

    if response.status_code == 409:
        # The service is healthy and this course simply has nothing published.
        # Forwarded as 409 rather than flattened into 502 because it is an
        # operator action away from being fixed, and the detail says which.
        _log_upstream_failure(
            "generation",
            url=base_url,
            status_code=response.status_code,
            body=response.text,
        )
        raise HTTPException(
            status_code=409,
            detail=(
                "No fine-tuned adapter is published on the inference service "
                f'for course "{safe_course_id}". '
                "Publish one for this course, then try again."
            ),
        )

    if response.status_code >= 400:
        _log_upstream_failure(
            "generation",
            url=base_url,
            status_code=response.status_code,
            body=response.text,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "Fine-tuned inference service rejected the request "
                f"(HTTP {response.status_code}). See the backend log for the "
                "service's own response."
            ),
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="Fine-tuned service returned invalid JSON.",
        ) from exc

    validated = _validate_generate_payload(data, expected_course_id=safe_course_id)
    return {
        "answer": validated["answer"],
        "model": validated["model"],
        "adapter_loaded": validated["adapter_loaded"],
        "course_id": validated["course_id"],
        "model_version": validated["model_version"],
        "generation_seconds": validated["generation_seconds"],
        "response_type": "fineTuned",
    }
