"""One grounded path, two targets: the RAG-versus-Fine-Tuned + RAG comparison.

What is pinned: both conditions retrieve through the same function, build the
same prompt text, and generate through the same call shape with the same
options; the fine-tuned target resolves the course's version (or an explicit
one) before the service is asked; the two public wrappers are this path; and
an empty retrieval refuses both alike.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from app.grounded_generation import GROUNDED_OPTIONS, GROUNDED_TIMEOUT_SECONDS, build_grounded_prompt
from app.grounded_rag import (
    BASE_TARGET,
    GenerationTarget,
    fine_tuned_target,
    generate_from_prompt,
    generate_grounded_answer,
    retrieve_and_prompt,
)

COURSE = "css-360-winter-2026-a7rp"
CHUNKS = [
    {"chunk_id": "chunk-067", "section": "Late Policy", "text": "one 48-hour extension per quarter", "score": 0.9},
    {"chunk_id": "chunk-022", "section": "Assignments", "text": "There will be no exams.", "score": 0.7},
]


def _retrieval(chunks=CHUNKS):
    return patch(
        "app.grounded_rag.retrieve_course_syllabus_chunks",
        new=AsyncMock(return_value=("nomic-embed-text", list(chunks))),
    )


def _chat(answer="Base answer."):
    return patch(
        "app.grounded_rag.generate_ollama_chat",
        new=AsyncMock(return_value={"answer": answer, "model": "llama3.2:3b"}),
    )


def _client(answer="Adapter answer.", version="v2"):
    return patch(
        "app.grounded_rag.generate_finetuned_response",
        new=AsyncMock(
            return_value={
                "answer": answer,
                "model": f"css360-ft-{version}:latest",
                "adapter_loaded": True,
                "course_id": COURSE,
                "model_version": version,
                "generation_seconds": 0.5,
                "response_type": "fineTuned",
            }
        ),
    )


class RetrieveAndPromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_prepares_the_shared_prompt_from_the_retrieved_chunks(self) -> None:
        with _retrieval() as retrieve:
            prepared = await retrieve_and_prompt(COURSE, "  Can I get an  extension? ", top_k=3)

        retrieve.assert_awaited_once_with(
            course_id=COURSE, question="Can I get an extension?", top_k=3, storage=None
        )
        self.assertEqual(prepared["courseId"], COURSE)
        self.assertEqual(prepared["question"], "Can I get an extension?")
        self.assertEqual(
            prepared["prompt"],
            build_grounded_prompt("Can I get an extension?", CHUNKS, prepared["facets"]),
        )
        self.assertEqual([s["chunkId"] for s in prepared["sources"]], ["chunk-067", "chunk-022"])
        self.assertEqual([c["chunkId"] for c in prepared["retrievedChunks"]], ["chunk-067", "chunk-022"])

    async def test_an_empty_retrieval_is_a_404_before_any_model_is_asked(self) -> None:
        with _retrieval([]), _chat() as chat, _client() as client:
            for target in (BASE_TARGET, fine_tuned_target()):
                with self.subTest(target=target.kind):
                    with self.assertRaises(HTTPException) as ctx:
                        await generate_grounded_answer(COURSE, "Q?", target=target)
                    self.assertEqual(ctx.exception.status_code, 404)
        chat.assert_not_awaited()
        client.assert_not_awaited()

    async def test_a_blank_question_and_a_bad_course_are_refused(self) -> None:
        with self.assertRaises(HTTPException) as blank:
            await retrieve_and_prompt(COURSE, "   ")
        self.assertEqual(blank.exception.status_code, 422)
        with self.assertRaises(HTTPException) as bad:
            await retrieve_and_prompt("Bad_Id", "Q?")
        self.assertEqual(bad.exception.status_code, 400)


class GenerateFromPromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_base_target_sends_the_prompt_with_the_shared_options(self) -> None:
        with _chat() as chat:
            result = await generate_from_prompt("PROMPT", course_id=COURSE, target=BASE_TARGET)

        chat.assert_awaited_once_with(
            "PROMPT", options=dict(GROUNDED_OPTIONS), timeout=GROUNDED_TIMEOUT_SECONDS
        )
        self.assertEqual(result["answer"], "Base answer.")
        self.assertEqual(result["model"], "llama3.2:3b")
        self.assertEqual(result["responseType"], "rag")
        self.assertIsNone(result["modelVersion"])
        self.assertIsNone(result["adapterLoaded"])

    async def test_the_fine_tuned_target_resolves_the_courses_version_first(self) -> None:
        with (
            patch("app.grounded_rag.resolve_current_course_model", return_value={"version": "v2"}) as current,
            patch("app.grounded_rag.resolve_course_model_version") as explicit,
            _client() as client,
        ):
            result = await generate_from_prompt("PROMPT", course_id=COURSE, target=fine_tuned_target())

        current.assert_called_once_with(COURSE)
        explicit.assert_not_called()
        client.assert_awaited_once_with("PROMPT", course_id=COURSE, model_version="v2")
        self.assertEqual(result["responseType"], "fineTunedRag")
        self.assertEqual(result["modelVersion"], "v2")
        self.assertTrue(result["adapterLoaded"])
        self.assertEqual(result["generationSeconds"], 0.5)

    async def test_an_explicit_version_goes_through_the_registry_check(self) -> None:
        with (
            patch("app.grounded_rag.resolve_current_course_model") as current,
            patch(
                "app.grounded_rag.resolve_course_model_version",
                return_value={"courseId": COURSE, "version": "v3"},
            ) as explicit,
            _client(version="v3") as client,
        ):
            result = await generate_from_prompt("PROMPT", course_id=COURSE, target=fine_tuned_target("v3"))

        explicit.assert_called_once_with(COURSE, "v3")
        current.assert_not_called()
        self.assertEqual(client.await_args.kwargs["model_version"], "v3")
        self.assertEqual(result["modelVersion"], "v3")

    async def test_an_unknown_target_is_a_programming_error(self) -> None:
        with self.assertRaises(ValueError):
            await generate_from_prompt("PROMPT", course_id=COURSE, target=GenerationTarget("other"))


class SameInputsBothTargetsTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_conditions_see_byte_identical_prompts_and_chunks(self) -> None:
        with (
            _retrieval(),
            _chat() as chat,
            patch("app.grounded_rag.resolve_current_course_model", return_value={"version": "v2"}),
            _client() as client,
        ):
            rag = await generate_grounded_answer(COURSE, "Can I get an extension?", target=BASE_TARGET)
            fine_tuned = await generate_grounded_answer(
                COURSE, "Can I get an extension?", target=fine_tuned_target()
            )

        self.assertEqual(chat.await_args.args[0], client.await_args.args[0])
        self.assertEqual(rag["prompt"], fine_tuned["prompt"])
        self.assertEqual(rag["retrievedChunks"], fine_tuned["retrievedChunks"])
        self.assertEqual(rag["sources"], fine_tuned["sources"])
        self.assertEqual(rag["responseType"], "rag")
        self.assertEqual(fine_tuned["responseType"], "fineTunedRag")
        self.assertEqual(rag["answer"], "Base answer.")
        self.assertEqual(fine_tuned["answer"], "Adapter answer.")


class PublicWrapperTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_rag_wrapper_is_the_shared_path_with_the_base_target(self) -> None:
        from app.course_rag import generate_course_rag_answer

        with patch("app.grounded_rag.generate_grounded_answer", new=AsyncMock(return_value={"ok": 1})) as shared:
            await generate_course_rag_answer(course_id=COURSE, question="Q?", top_k=6)
        shared.assert_awaited_once_with(COURSE, "Q?", top_k=6, storage=None, target=BASE_TARGET)

    async def test_the_fine_tuned_wrapper_is_the_shared_path_with_the_course_target(self) -> None:
        from app.finetuned_rag import generate_course_finetuned_rag_answer

        with patch("app.finetuned_rag.generate_grounded_answer", new=AsyncMock(return_value={"ok": 1})) as shared:
            await generate_course_finetuned_rag_answer(course_id=COURSE, question="Q?", top_k=4)
            await generate_course_finetuned_rag_answer(
                course_id=COURSE, question="Q?", top_k=4, model_version="v3"
            )
        first, second = shared.await_args_list
        self.assertEqual(first.kwargs["target"], fine_tuned_target())
        self.assertEqual(second.kwargs["target"], fine_tuned_target("v3"))


if __name__ == "__main__":
    unittest.main()
