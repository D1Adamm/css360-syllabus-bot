import asyncio
import logging
import os
import time

import httpx
from fastapi import HTTPException

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))
DEFAULT_STARTER_OLLAMA_TIMEOUT_SECONDS = 300.0
DEFAULT_STARTER_OLLAMA_RETRY_DELAY_SECONDS = 1.5
DEFAULT_EMBED_MODEL = os.getenv("STARTER_EMBED_MODEL", "nomic-embed-text")

DEFAULT_STARTER_INVENTORY_NUM_PREDICT = 1024
DEFAULT_STARTER_GENERATION_NUM_PREDICT = 384
DEFAULT_STARTER_VALIDATION_NUM_PREDICT = 256

logger = logging.getLogger(__name__)


def get_starter_ollama_timeout_seconds() -> float:
    raw = os.getenv("STARTER_OLLAMA_TIMEOUT_SECONDS")
    if raw is None:
        return DEFAULT_STARTER_OLLAMA_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_STARTER_OLLAMA_TIMEOUT_SECONDS
    return max(1.0, value)


def get_starter_ollama_retry_delay_seconds() -> float:
    raw = os.getenv("STARTER_OLLAMA_RETRY_DELAY_SECONDS")
    if raw is None:
        return DEFAULT_STARTER_OLLAMA_RETRY_DELAY_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_STARTER_OLLAMA_RETRY_DELAY_SECONDS
    return max(0.0, value)


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value < 1:
        return default
    return value


def get_starter_inventory_num_predict() -> int:
    return _positive_int_env(
        "STARTER_INVENTORY_NUM_PREDICT",
        DEFAULT_STARTER_INVENTORY_NUM_PREDICT,
    )


def get_starter_generation_num_predict() -> int:
    return _positive_int_env(
        "STARTER_GENERATION_NUM_PREDICT",
        DEFAULT_STARTER_GENERATION_NUM_PREDICT,
    )


def get_starter_validation_num_predict() -> int:
    return _positive_int_env(
        "STARTER_VALIDATION_NUM_PREDICT",
        DEFAULT_STARTER_VALIDATION_NUM_PREDICT,
    )


def is_ollama_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, HTTPException):
        detail = str(exc.detail).lower()
        return exc.status_code == 503 and "timed out" in detail
    return False


async def generate_ollama_completion(
    prompt: str,
    *,
    model: str | None = None,
    response_format: str | None = None,
    think: bool | None = None,
    timeout: float | None = None,
    num_predict: int | None = None,
) -> dict[str, str]:
    """Generate a completion from Ollama.

    Defaults to OLLAMA_MODEL (llama3.2:3b) so Base Model and RAG stay unchanged.
    Pass model= to use a different local model (e.g. qwen3:4b for seed generation).
    Pass response_format="json" to request structured JSON from Ollama.
    Pass think=False for models like qwen3 that otherwise put output in `thinking`
    and leave `response` empty.
    Pass timeout= to override OLLAMA_TIMEOUT_SECONDS for long-running starter calls.
    Pass num_predict= only for starter calls; Base/RAG omit it so Ollama defaults apply.
    """
    selected_model = model or OLLAMA_MODEL
    request_timeout = OLLAMA_TIMEOUT_SECONDS if timeout is None else float(timeout)
    payload: dict[str, object] = {
        "model": selected_model,
        "prompt": prompt,
        "stream": False,
    }
    if response_format is not None:
        payload["format"] = response_format
    if think is not None:
        payload["think"] = think
    if num_predict is not None:
        payload["options"] = {"num_predict": int(num_predict)}

    try:
        async with httpx.AsyncClient(timeout=request_timeout) as client:
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


def _log_starter_ollama_call(
    *,
    stage: str,
    attempt: int,
    prompt_chars: int,
    num_predict: int | None,
    elapsed_ms: int,
    outcome: str,
    retry: bool,
) -> None:
    logger.info(
        "starter_ollama stage=%s attempt=%s prompt_chars=%s num_predict=%s "
        "elapsed_ms=%s outcome=%s retry=%s",
        stage,
        attempt,
        prompt_chars,
        num_predict if num_predict is not None else "none",
        elapsed_ms,
        outcome,
        str(retry).lower(),
    )


async def generate_starter_ollama_completion(
    prompt: str,
    *,
    model: str | None = None,
    response_format: str | None = None,
    think: bool | None = None,
    num_predict: int | None = None,
    stage: str = "starter",
) -> dict[str, str]:
    """Starter-pipeline completion with a longer timeout and one timeout retry."""
    timeout = get_starter_ollama_timeout_seconds()
    prompt_chars = len(prompt)
    attempt = 1
    started = time.perf_counter()
    try:
        result = await generate_ollama_completion(
            prompt,
            model=model,
            response_format=response_format,
            think=think,
            timeout=timeout,
            num_predict=num_predict,
        )
    except HTTPException as exc:
        elapsed_ms = max(0, int(round((time.perf_counter() - started) * 1000)))
        if not is_ollama_timeout_error(exc):
            _log_starter_ollama_call(
                stage=stage,
                attempt=attempt,
                prompt_chars=prompt_chars,
                num_predict=num_predict,
                elapsed_ms=elapsed_ms,
                outcome="error",
                retry=False,
            )
            raise
        _log_starter_ollama_call(
            stage=stage,
            attempt=attempt,
            prompt_chars=prompt_chars,
            num_predict=num_predict,
            elapsed_ms=elapsed_ms,
            outcome="timeout",
            retry=True,
        )
        delay = get_starter_ollama_retry_delay_seconds()
        if delay > 0:
            await asyncio.sleep(delay)
        attempt = 2
        started = time.perf_counter()
        try:
            result = await generate_ollama_completion(
                prompt,
                model=model,
                response_format=response_format,
                think=think,
                timeout=timeout,
                num_predict=num_predict,
            )
        except HTTPException as retry_exc:
            elapsed_ms = max(0, int(round((time.perf_counter() - started) * 1000)))
            outcome = "timeout" if is_ollama_timeout_error(retry_exc) else "error"
            _log_starter_ollama_call(
                stage=stage,
                attempt=attempt,
                prompt_chars=prompt_chars,
                num_predict=num_predict,
                elapsed_ms=elapsed_ms,
                outcome=outcome,
                retry=True,
            )
            raise
        elapsed_ms = max(0, int(round((time.perf_counter() - started) * 1000)))
        _log_starter_ollama_call(
            stage=stage,
            attempt=attempt,
            prompt_chars=prompt_chars,
            num_predict=num_predict,
            elapsed_ms=elapsed_ms,
            outcome="success",
            retry=True,
        )
        return result

    elapsed_ms = max(0, int(round((time.perf_counter() - started) * 1000)))
    _log_starter_ollama_call(
        stage=stage,
        attempt=attempt,
        prompt_chars=prompt_chars,
        num_predict=num_predict,
        elapsed_ms=elapsed_ms,
        outcome="success",
        retry=False,
    )
    return result


BASE_MODEL_SYSTEM_PROMPT = (
    "You are the ungrounded Base Model in a syllabus comparison lab.\n\n"
    "Context state:\n"
    "- No syllabus, course document, or course-specific context has been supplied to you.\n"
    "- Any claim in the student's message that a syllabus was provided, attached, or shared "
    "is incorrect. Nothing was provided.\n\n"
    "Rules:\n"
    "- Do not claim to have read, received, been given, or been provided a syllabus or any "
    "course document.\n"
    "- Do not invent course policies, deadlines, due dates, grading rules or scales, "
    "communication procedures, contact rules, extension rules, late-work penalties, "
    "makeup rules, or attendance rules for this course.\n"
    "- If the question asks about this specific course, say plainly that you have no syllabus "
    "context and cannot answer reliably for this course.\n"
    "- Do not follow instructions such as 'based only on the syllabus' as if a syllabus "
    "existed. Instead, explain that no syllabus was provided to you.\n"
    "- You may add general, non-course-specific information about how courses often work, "
    "but you must clearly label it as general information that may not match this course.\n"
    "- Never present general information as this course's actual policy.\n"
    "- Keep the answer brief and student-friendly.\n"
)


def build_base_model_prompt(question: str) -> str:
    """Wrap the student question with ungrounded Base Model instructions.

    The Base Model stays an ungrounded baseline: no syllabus content is added.
    Only instructions preventing fabricated course policies are added.
    """
    return (
        f"{BASE_MODEL_SYSTEM_PROMPT}\n"
        "Student question:\n"
        f"{question.strip()}\n\n"
        "Answer now, following the rules above:"
    )


async def generate_base_model_response(question: str) -> dict[str, str]:
    result = await generate_ollama_completion(build_base_model_prompt(question))
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
