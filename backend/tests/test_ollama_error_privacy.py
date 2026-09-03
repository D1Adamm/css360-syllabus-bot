"""A failing local Ollama server must not narrate the host to the browser.

Three call sites forwarded `response.text` straight into a public error detail:
Base generation (`ollama.generate_base_model_response`), batch embeddings
(`ollama.embed_ollama_texts`), and the RAG retrieval embedding
(`rag.get_embedding`). Ollama's own 4xx bodies routinely name the model cache it
searched and the address it is listening on, so a browser asking a syllabus
question could be handed the operator's home directory.

Same rule as the fine-tuned service: the body is a log line, the response gets
the operation and the status code.

Poisoned bodies here are synthetic (`testuser`, port 11434 on localhost) and
cover the POSIX, macOS and Windows spellings of a model cache. Nothing contacts
a server.
"""

from __future__ import annotations

import json
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from app import ollama, rag

#: What a local Ollama server can put in a 4xx body.
POISONED_BODY = (
    '{"error":"model nomic-embed-text not found, searched '
    '/home/testuser/.ollama/models/manifests, '
    '/Users/testuser/.ollama/models/blobs and '
    'C:\\Users\\testuser\\.ollama\\models; '
    'server listening on localhost:11434"}'
)

#: Every value that must not survive to a public response.
LEAKY_STRINGS = (
    "localhost:11434",
    "11434",
    "/home/testuser/",
    "/Users/testuser/",
    "C:\\Users\\testuser",
    ".ollama/models",
    ".ollama\\models",
    "testuser",
    "searched",
    "not found",
)


def _response(status_code: int, body: str = POISONED_BODY) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.text = body
    response.json.side_effect = ValueError("not json")
    return response


def _async_client(response: MagicMock) -> MagicMock:
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    client.get = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


class OllamaUpstreamBodyTests(unittest.IsolatedAsyncioTestCase):
    def assert_clean(self, detail: Any) -> None:
        text = json.dumps(detail)
        for leak in LEAKY_STRINGS:
            self.assertNotIn(leak, text)

    # -- base model generation ---------------------------------------------- #

    async def test_a_rejected_base_request_says_base_and_the_code_only(self) -> None:
        with patch(
            "app.ollama.httpx.AsyncClient", return_value=_async_client(_response(400))
        ):
            with self.assertRaises(HTTPException) as caught:
                await ollama.generate_base_model_response("What is the late policy?")

        self.assertEqual(caught.exception.status_code, 502)
        self.assert_clean(caught.exception.detail)
        # Still says which operation failed, and how.
        self.assertIn("base model request", caught.exception.detail)
        self.assertIn("HTTP 400", caught.exception.detail)

    async def test_the_base_body_is_kept_in_the_backend_log(self) -> None:
        with patch(
            "app.ollama.httpx.AsyncClient", return_value=_async_client(_response(404))
        ):
            with self.assertLogs("app.ollama", level="WARNING") as logs:
                with self.assertRaises(HTTPException):
                    await ollama.generate_base_model_response("q")

        logged = "\n".join(logs.output)
        self.assertIn("/home/testuser/.ollama/models/manifests", logged)
        self.assertIn("localhost:11434", logged)
        self.assertIn("base model generation", logged)

    # -- batch embeddings ---------------------------------------------------- #

    async def test_a_rejected_embedding_request_is_clean(self) -> None:
        with patch(
            "app.ollama.httpx.AsyncClient", return_value=_async_client(_response(422))
        ):
            with self.assertRaises(HTTPException) as caught:
                await ollama.embed_ollama_texts(["a chunk of syllabus text"])

        self.assertEqual(caught.exception.status_code, 502)
        self.assert_clean(caught.exception.detail)
        self.assertIn("embedding request", caught.exception.detail)
        self.assertIn("HTTP 422", caught.exception.detail)

    async def test_the_embedding_body_is_kept_in_the_backend_log(self) -> None:
        with patch(
            "app.ollama.httpx.AsyncClient", return_value=_async_client(_response(422))
        ):
            with self.assertLogs("app.ollama", level="WARNING") as logs:
                with self.assertRaises(HTTPException):
                    await ollama.embed_ollama_texts(["text"])

        logged = "\n".join(logs.output)
        self.assertIn("C:\\Users\\testuser", logged)
        self.assertIn("batch embedding", logged)

    # -- RAG retrieval embedding --------------------------------------------- #

    async def test_a_rejected_rag_embedding_says_rag_and_the_code_only(self) -> None:
        """The one a student's syllabus question actually reaches."""
        with patch(
            "app.rag.httpx.AsyncClient", return_value=_async_client(_response(400))
        ):
            with self.assertRaises(HTTPException) as caught:
                await rag.get_embedding("Can I get an extension?")

        self.assertEqual(caught.exception.status_code, 502)
        self.assert_clean(caught.exception.detail)
        self.assertIn("RAG embedding request", caught.exception.detail)
        self.assertIn("HTTP 400", caught.exception.detail)

    async def test_the_rag_body_is_kept_in_the_backend_log(self) -> None:
        with patch(
            "app.rag.httpx.AsyncClient", return_value=_async_client(_response(400))
        ):
            with self.assertLogs("app.rag", level="WARNING") as logs:
                with self.assertRaises(HTTPException):
                    await rag.get_embedding("q")

        logged = "\n".join(logs.output)
        self.assertIn("/Users/testuser/.ollama/models/blobs", logged)
        self.assertIn("RAG embedding", logged)

    # -- Base and RAG stay distinguishable ------------------------------------ #

    async def test_base_and_rag_failures_are_told_apart(self) -> None:
        """An operator reading a 502 must know which subsystem to look at."""
        with patch(
            "app.ollama.httpx.AsyncClient", return_value=_async_client(_response(400))
        ):
            with self.assertRaises(HTTPException) as base:
                await ollama.generate_base_model_response("q")

        with patch(
            "app.rag.httpx.AsyncClient", return_value=_async_client(_response(400))
        ):
            with self.assertRaises(HTTPException) as retrieval:
                await rag.get_embedding("q")

        self.assertIn("base model", base.exception.detail)
        self.assertNotIn("RAG", base.exception.detail)
        self.assertIn("RAG", retrieval.exception.detail)
        self.assertNotIn("base model", retrieval.exception.detail)

    # -- the paths that never quoted a body are untouched ---------------------- #

    async def test_a_5xx_still_reports_a_server_error_without_a_body(self) -> None:
        """These never forwarded the body and must keep their wording."""
        with patch(
            "app.ollama.httpx.AsyncClient", return_value=_async_client(_response(500))
        ):
            with self.assertRaises(HTTPException) as caught:
                await ollama.generate_base_model_response("q")

        self.assertEqual(caught.exception.status_code, 503)
        self.assert_clean(caught.exception.detail)
        self.assertIn("server error", caught.exception.detail.lower())

    async def test_a_long_body_is_logged_past_the_old_forwarded_length(self) -> None:
        from app.upstream_errors import UPSTREAM_LOG_BODY_LIMIT

        body = "x" * 1200 + POISONED_BODY
        self.assertLess(len(body), UPSTREAM_LOG_BODY_LIMIT)

        with patch(
            "app.ollama.httpx.AsyncClient",
            return_value=_async_client(_response(400, body)),
        ):
            with self.assertLogs("app.ollama", level="WARNING") as logs:
                with self.assertRaises(HTTPException):
                    await ollama.generate_base_model_response("q")

        self.assertIn("localhost:11434", "\n".join(logs.output))


class SuccessPathUnchangedTests(unittest.IsolatedAsyncioTestCase):
    """Nothing above is on a success path. Assert that rather than assume it."""

    async def test_base_generation_still_returns_its_answer(self) -> None:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "response": "  No extension is possible.  ",
            "model": "llama3.2:3b",
        }

        with patch(
            "app.ollama.httpx.AsyncClient", return_value=_async_client(response)
        ):
            result = await ollama.generate_base_model_response("q")

        self.assertEqual(result["answer"], "No extension is possible.")
        self.assertEqual(result["model"], "llama3.2:3b")

    async def test_batch_embeddings_still_return_their_vectors(self) -> None:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "model": "nomic-embed-text",
        }

        with patch(
            "app.ollama.httpx.AsyncClient", return_value=_async_client(response)
        ):
            result = await ollama.embed_ollama_texts(["a", "b"])

        self.assertEqual(result["embeddings"], [[0.1, 0.2], [0.3, 0.4]])

    async def test_rag_embedding_still_returns_its_vector(self) -> None:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"embedding": [0.5, 0.6, 0.7]}

        with patch("app.rag.httpx.AsyncClient", return_value=_async_client(response)):
            self.assertEqual(await rag.get_embedding("q"), [0.5, 0.6, 0.7])


if __name__ == "__main__":
    unittest.main()
