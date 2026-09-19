"""HTTP client for the benchmark-only inference service.

The production fine-tuned client (`finetuned_client.py`) talks to
`FINETUNED_SERVICE_URL`, resolves versions through the registry, and answers
students. This client talks to `CSS360_BENCHMARK_SERVICE_URL`, names an alias
rather than a course and version, and answers nothing but the research route.
The two never share a URL, a mapping, or a code path, so an experiment
cannot become servable to a classroom by way of configuration.

Failures are `BenchmarkConditionError`s with a fixed code and a fixed message,
not `HTTPException`s: the route records them per condition and carries on.
Upstream bodies go to the backend log in full and to the caller never, the
rule `upstream_errors` states for every service this backend calls.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

import httpx

from app.upstream_errors import log_upstream_failure

logger = logging.getLogger(__name__)

SERVICE_URL_ENV = "CSS360_BENCHMARK_SERVICE_URL"

#: Ollama's per-generation accounting, as the service reports it. Copied by
#: name; an unknown key is dropped.
OLLAMA_TIMING_INT_FIELDS: tuple[str, ...] = (
    "totalDurationNs",
    "loadDurationNs",
    "promptEvalCount",
    "promptEvalDurationNs",
    "evalCount",
    "evalDurationNs",
)


class BenchmarkConditionError(Exception):
    """One condition could not be answered. `code` is stable; `message` is public."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def get_benchmark_service_url() -> str | None:
    raw = (os.getenv(SERVICE_URL_ENV) or "").strip()
    return raw.rstrip("/") or None


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _project_ollama_timings(value: Any) -> dict[str, Any]:
    timings: dict[str, Any] = {}
    if not isinstance(value, dict):
        return timings
    for key in OLLAMA_TIMING_INT_FIELDS:
        number = value.get(key)
        if isinstance(number, int) and not isinstance(number, bool):
            timings[key] = number
    done_reason = value.get("doneReason")
    if isinstance(done_reason, str) and done_reason.strip():
        timings["doneReason"] = done_reason.strip()
    return timings


def _validate_generate_payload(
    data: Any, *, expected_alias: str, expected_prompt_sha256: str
) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise BenchmarkConditionError(
            "malformed_response", "The benchmark service returned a malformed response."
        )
    alias = data.get("alias")
    if alias != expected_alias:
        # An answer from the wrong alias would look exactly like one from the
        # right alias. Discarded.
        raise BenchmarkConditionError(
            "alias_mismatch",
            "The benchmark service answered for a different alias. The answer was discarded.",
        )
    echoed = data.get("promptSha256")
    if not isinstance(echoed, str) or echoed.lower() != expected_prompt_sha256:
        raise BenchmarkConditionError(
            "prompt_integrity",
            "The benchmark service did not receive the prompt bytes that were sent. "
            "The answer was discarded.",
        )
    answer = data.get("answer")
    if not isinstance(answer, str):
        raise BenchmarkConditionError(
            "malformed_response", "The benchmark service returned no answer text."
        )
    model = data.get("model")
    if not isinstance(model, str) or not model.strip():
        raise BenchmarkConditionError(
            "malformed_response", "The benchmark service named no model."
        )
    options = data.get("options")
    if not isinstance(options, dict):
        raise BenchmarkConditionError(
            "malformed_response", "The benchmark service reported no decoding options."
        )
    generation_seconds = data.get("generationSeconds")
    if not isinstance(generation_seconds, (int, float)) or isinstance(generation_seconds, bool):
        raise BenchmarkConditionError(
            "malformed_response", "The benchmark service reported no generation time."
        )
    # The digest Ollama held for the tag at the moment of the call. Optional:
    # a service whose digest lookup failed still answered, and the route
    # records the answer without one rather than losing it.
    digest = data.get("modelDigest")
    model_digest = digest.strip() if isinstance(digest, str) and digest.strip() else None
    return {
        "alias": alias,
        "model": model.strip(),
        "modelDigest": model_digest,
        "answer": answer,
        "promptSha256": echoed.lower(),
        "options": dict(options),
        "generationSeconds": float(generation_seconds),
        "ollama": _project_ollama_timings(data.get("ollama")),
    }


async def generate_benchmark_answer(
    alias: str, prompt: str, *, timeout: float
) -> dict[str, Any]:
    """POST {CSS360_BENCHMARK_SERVICE_URL}/generate for one alias and one prompt.

    The prompt is sent exactly as given and the service's echo of its SHA-256
    is checked against the local one, so a saved answer is known to have been
    produced from the recorded prompt bytes.
    """
    base_url = get_benchmark_service_url()
    if base_url is None:
        raise BenchmarkConditionError(
            "service_not_configured", "The benchmark inference service is not configured."
        )
    expected_sha = prompt_sha256(prompt)
    body = {"alias": alias, "prompt": prompt}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{base_url}/generate", json=body)
    except httpx.TimeoutException as exc:
        raise BenchmarkConditionError(
            "service_timeout", "The benchmark inference service timed out."
        ) from exc
    except httpx.RequestError as exc:
        raise BenchmarkConditionError(
            "service_unavailable", "The benchmark inference service is unavailable."
        ) from exc

    if response.status_code != 200:
        log_upstream_failure(
            logger,
            f"benchmark generation ({alias})",
            url=base_url,
            status_code=response.status_code,
            body=response.text,
        )
    if response.status_code >= 500:
        raise BenchmarkConditionError(
            "service_error",
            "The benchmark inference service or its Ollama returned a server error "
            f"(HTTP {response.status_code}).",
        )
    if response.status_code == 409:
        raise BenchmarkConditionError(
            "alias_not_mapped",
            f'Alias "{alias}" is not mapped to a model on the benchmark service, '
            "or its model has not been created.",
        )
    if response.status_code >= 400:
        raise BenchmarkConditionError(
            "service_rejected",
            f"The benchmark inference service rejected the request (HTTP {response.status_code}).",
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise BenchmarkConditionError(
            "malformed_response", "The benchmark inference service returned invalid JSON."
        ) from exc
    return _validate_generate_payload(
        data, expected_alias=alias, expected_prompt_sha256=expected_sha
    )
