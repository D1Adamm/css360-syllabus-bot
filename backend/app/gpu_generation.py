"""Classroom GPU mode: answering on the Tillicum GPU service, held to production settings.

Every GPU answer carries the settings it was decoded with (`decoding`, in
Ollama's vocabulary). They are compared with the options the VM would have
sent Ollama for the same request, and an answer computed any other way is
refused rather than shown: a GPU service that capped at 256 instead of 128, or
rendered a different prompt, is exactly the drift classroom GPU mode must not
introduce. An answer with no `decoding` at all comes from an older build of the
service, which did not honour the output cap, and is refused for that reason.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import httpx
from fastapi import HTTPException

from app.grounded_generation import GROUNDED_NUM_PREDICT, grounded_options
from app.upstream_errors import log_upstream_failure

logger = logging.getLogger(__name__)

#: The GPU service's prompt rendering: Ollama's llama3.2 chat template, which is
#: what the VM's `/api/chat` calls use.
PRODUCTION_PROMPT_FORMAT = "ollama-llama3.2-chat"
DECODING_KEYS = ("num_predict", "temperature", "repeat_penalty", "repeat_last_n", "seed", "num_ctx")


def expected_decoding(options: Mapping[str, Any] | None = None, *, max_new_tokens: int | None = None) -> dict[str, Any]:
    """What the VM decodes with for this request: Ollama options plus the prompt format.

    `options` is what Base/RAG would send Ollama. The fine-tuned service builds
    the same dictionary from its own constants (pinned equal to
    `grounded_options` by a test), with the request's output cap.
    """
    base = dict(options) if options is not None else {
        **grounded_options(),
        "num_predict": int(max_new_tokens) if max_new_tokens is not None else GROUNDED_NUM_PREDICT,
    }
    expected = {key: base[key] for key in DECODING_KEYS}
    expected["promptFormat"] = PRODUCTION_PROMPT_FORMAT
    return expected


def check_decoding(reported: Any, expected: Mapping[str, Any]) -> None:
    """Refuse a GPU answer decoded differently from production (502)."""
    if not isinstance(reported, Mapping):
        raise HTTPException(
            status_code=502,
            detail=(
                "The GPU service did not report its decoding settings, so it is an "
                "older build that does not match production (it ignored the output "
                "cap). Update it on Tillicum and restart the job."
            ),
        )
    differences = {
        key: (reported.get(key), value)
        for key, value in expected.items()
        if _normalise(reported.get(key)) != _normalise(value)
    }
    if differences:
        logger.error("GPU decoding differs from production: %s", differences)
        raise HTTPException(
            status_code=502,
            detail=(
                "The GPU service decoded this answer with settings that differ from "
                "production, so it was discarded: "
                + ", ".join(f"{key} {got!r} != {want!r}" for key, (got, want) in sorted(differences.items()))
            ),
        )


def _normalise(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    return value


async def generate_base_on_gpu(
    prompt: str,
    *,
    gpu_url: str,
    options: Mapping[str, Any],
    timeout: float,
    action: str,
) -> tuple[str, dict[str, Any]]:
    """Base and RAG on the GPU: the base model with adapters off. Returns (answer, body)."""
    payload = {"question": prompt, "target": "base", "maxNewTokens": int(options["num_predict"])}
    url = f"{gpu_url}/generate"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=503, detail="The GPU service timed out.") from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "The GPU service is unavailable (classroom GPU mode). The Tillicum job "
                "or its tunnel may have ended; switch back with "
                "./scripts/classroom_gpu_mode.sh stop."
            ),
        ) from exc

    if response.status_code >= 400:
        log_upstream_failure(logger, f"{action} generation (GPU)", url=url, status_code=response.status_code, body=response.text)
        raise HTTPException(
            status_code=503 if response.status_code >= 500 else 502,
            detail=f"The GPU service refused the {action} request (HTTP {response.status_code}). See the backend log.",
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="The GPU service returned invalid JSON.") from exc
    if not isinstance(data, dict) or data.get("target") != "base":
        raise HTTPException(status_code=502, detail="The GPU service did not answer with the base model.")
    check_decoding(data.get("decoding"), expected_decoding(options))
    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise HTTPException(status_code=502, detail="The GPU service returned an empty response.")
    return answer.strip(), data
