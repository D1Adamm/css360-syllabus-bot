"""Machine-readable reasons on the failures a browser or worker has to tell apart.

Audit findings B5 and B7 (2026-10-27): every failure of a generation route
reached the student as "temporarily unavailable", whether the fine-tuned
service was down for a minute, no version was activated for the course, the
activated version was not mapped on the VM, or the local Ollama had stopped.
Those need different words for a student and different actions from an
operator, and the browser could not tell them apart from a plain-text
`detail` it is not allowed to show.

The convention the generation queue already uses (`generation_busy`) is a
`detail` object, `{"code": ..., "message": ...}`. The browser's HTTP client
reads both halves (`src/lib/httpClient.ts`), and `src/lib/errorMessages.ts`
turns the code into copy for whoever is reading: a student is told what they
can do, a professor what to ask for, an administrator the reason itself.

`ClassifiedHTTPException` keeps `.detail` as the plain message, so every
caller and test that reads `exc.detail` as a string is unchanged. Only the
HTTP body gains the code, through the handler `install` registers.
"""

from __future__ import annotations

from typing import Any, Mapping

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# The fine-tuned inference service (VM-local, or the Tillicum fallback).
FINETUNED_NOT_CONFIGURED = "finetuned_not_configured"
FINETUNED_UNAVAILABLE = "finetuned_unavailable"
FINETUNED_TIMEOUT = "finetuned_timeout"
FINETUNED_UPSTREAM_ERROR = "finetuned_upstream_error"
FINETUNED_REJECTED = "finetuned_rejected"
FINETUNED_BAD_RESPONSE = "finetuned_bad_response"

# The course's model registry versus what the service can serve.
MODEL_NOT_ACTIVATED = "model_not_activated"
MODEL_VERSION_NOT_SERVABLE = "model_version_not_servable"

# The local Ollama server (Base, RAG, embeddings).
OLLAMA_UNAVAILABLE = "ollama_unavailable"
OLLAMA_TIMEOUT = "ollama_timeout"
OLLAMA_UPSTREAM_ERROR = "ollama_upstream_error"
OLLAMA_BAD_RESPONSE = "ollama_bad_response"

# Retrieval.
SYLLABUS_NOT_INDEXED = "syllabus_not_indexed"

# Evaluations.
EVALUATION_ALREADY_RECORDED = "evaluation_already_recorded"

# Training data.
TRAINING_RUN_ACTIVE = "training_run_active"
DATASET_NOT_PREPARED = "dataset_not_prepared"
DATASET_CHANGED = "dataset_changed"

#: Every code this module hands out, for tests and for the browser's map.
ALL_CODES = frozenset(
    {
        FINETUNED_NOT_CONFIGURED,
        FINETUNED_UNAVAILABLE,
        FINETUNED_TIMEOUT,
        FINETUNED_UPSTREAM_ERROR,
        FINETUNED_REJECTED,
        FINETUNED_BAD_RESPONSE,
        MODEL_NOT_ACTIVATED,
        MODEL_VERSION_NOT_SERVABLE,
        OLLAMA_UNAVAILABLE,
        OLLAMA_TIMEOUT,
        OLLAMA_UPSTREAM_ERROR,
        OLLAMA_BAD_RESPONSE,
        SYLLABUS_NOT_INDEXED,
        EVALUATION_ALREADY_RECORDED,
        TRAINING_RUN_ACTIVE,
        DATASET_NOT_PREPARED,
        DATASET_CHANGED,
    }
)


class ClassifiedHTTPException(HTTPException):
    """An HTTPException whose body also carries a stable `code`.

    `detail` is the message, as every other HTTPException's is, so logging,
    tests and the callers that wrap a detail into a longer sentence keep
    working. `extra` adds fields to the body (an id, a list of versions) and
    is never read as a message.
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        headers: Mapping[str, str] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            status_code=status_code, detail=message, headers=dict(headers) if headers else None
        )
        self.code = code
        self.extra = dict(extra or {})

    def body(self) -> dict[str, Any]:
        content: dict[str, Any] = {"code": self.code, "message": str(self.detail)}
        for key, value in self.extra.items():
            content.setdefault(key, value)
        return content


def error_code(exc: BaseException) -> str | None:
    """The code of a classified exception, else None."""
    code = getattr(exc, "code", None)
    return code if isinstance(code, str) and code else None


async def classified_exception_handler(
    request: Request, exc: ClassifiedHTTPException
) -> JSONResponse:
    del request
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.body()}, headers=exc.headers
    )


def install(app: FastAPI) -> None:
    """Register the handler. A plain HTTPException is left to FastAPI's own."""
    app.add_exception_handler(ClassifiedHTTPException, classified_exception_handler)  # type: ignore[arg-type]
