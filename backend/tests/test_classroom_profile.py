"""The classroom profile: short answers on the student Compare page, and nothing else changed.

The four conditions keep their experimental distinction — Base gets no syllabus,
RAG and Fine-Tuned + RAG get the same retrieved excerpts, plain Fine-Tuned gets
the bare question — and only the output is constrained: one shared length
instruction where a condition already carries instructions, and one shared
output cap. The controlled profile the benchmark, the model-testing route and
the training exports use must not move.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.grounded_generation import (
    CLASSROOM_NUM_PREDICT,
    CONCISE_ANSWER_INSTRUCTION,
    PROMPT_TEMPLATE_NAME,
    RULES,
    build_grounded_prompt,
    classroom_options,
    grounded_options,
    prompt_template_fingerprint,
)
from app.grounded_rag import BASE_TARGET, fine_tuned_target, generate_from_prompt
from app.main import app
from app.ollama import build_base_model_prompt

COURSE = "css-360-winter-2026-a7rp"
CHUNKS = [
    {
        "chunk_id": "c1",
        "section": "Schedule",
        "text": "Class meets Tuesdays and Thursdays, 3:30 to 5:30 pm.",
        "score": 0.9,
    }
]

#: The controlled template's fingerprint. If this moves, the benchmark and every
#: training export changed meaning — which the classroom profile must never do.
CONTROLLED_FINGERPRINT = prompt_template_fingerprint()


class ControlledProfileUnchangedTests(unittest.TestCase):
    def test_the_controlled_options_keep_the_256_token_cap(self) -> None:
        self.assertEqual(grounded_options()["num_predict"], 256)

    def test_the_controlled_prompt_keeps_its_length_rule(self) -> None:
        prompt = build_grounded_prompt("When is class?", CHUNKS)
        self.assertIn("Write two to five sentences", prompt)
        self.assertNotIn(CONCISE_ANSWER_INSTRUCTION, prompt)
        self.assertEqual(PROMPT_TEMPLATE_NAME, "grounded-v1")
        self.assertEqual(prompt_template_fingerprint(), CONTROLLED_FINGERPRINT)

    def test_the_classroom_options_differ_only_in_the_cap(self) -> None:
        classroom = classroom_options()
        self.assertEqual(classroom["num_predict"], CLASSROOM_NUM_PREDICT)
        self.assertEqual(CLASSROOM_NUM_PREDICT, 128)
        self.assertEqual(
            {k: v for k, v in classroom.items() if k != "num_predict"},
            {k: v for k, v in grounded_options().items() if k != "num_predict"},
        )


class ClassroomPromptTests(unittest.TestCase):
    def test_the_concise_grounded_prompt_changes_only_the_length_rule(self) -> None:
        controlled = build_grounded_prompt("When is class?", CHUNKS).splitlines()
        concise = build_grounded_prompt("When is class?", CHUNKS, concise=True).splitlines()
        changed = [(a, b) for a, b in zip(controlled, concise) if a != b]
        self.assertEqual(len(controlled), len(concise))
        self.assertEqual(len(changed), 1)
        before, after = changed[0]
        self.assertEqual(before, f"- {RULES[-1]}")
        self.assertTrue(after.startswith(f"- {CONCISE_ANSWER_INSTRUCTION}"))
        # The same excerpts, word for word: no condition gains context.
        self.assertIn(CHUNKS[0]["text"], "\n".join(concise))

    def test_base_carries_the_same_instruction_and_still_no_syllabus(self) -> None:
        prompt = build_base_model_prompt("When is class?")
        self.assertIn(CONCISE_ANSWER_INSTRUCTION, prompt)
        self.assertIn("No syllabus, course document, or course-specific context", prompt)
        self.assertNotIn("Syllabus excerpts:", prompt)
        self.assertNotIn("Tuesdays", prompt)


class ClassroomRouteTests(unittest.TestCase):
    """The four student routes use the profile (conftest signs in an admin)."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        storage = MagicMock(load_index=MagicMock(return_value={"chunks": []}))
        for target, kwargs in (
            (
                "app.grounded_rag.retrieve_course_syllabus_chunks",
                {"new": AsyncMock(return_value=("", CHUNKS))},
            ),
            ("app.main.get_course_artifact_storage", {"return_value": storage}),
            (
                "app.grounded_rag.resolve_current_course_model",
                {"return_value": {"version": "v2"}},
            ),
            ("app.main.resolve_current_course_model", {"return_value": {"version": "v2"}}),
        ):
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.chat = self._patch(
            "app.grounded_rag.generate_ollama_chat",
            AsyncMock(return_value={"answer": "Tuesdays and Thursdays.", "model": "llama3.2:3b"}),
        )
        self.base_chat = self._patch(
            "app.ollama.generate_ollama_chat",
            AsyncMock(return_value={"answer": "I have no syllabus.", "model": "llama3.2:3b"}),
        )
        finetuned_result = {
            "answer": "Tuesdays.",
            "model": "css360-ft-v2",
            "adapter_loaded": True,
            "course_id": COURSE,
            "model_version": "v2",
            "generation_seconds": 1.0,
            "response_type": "fineTuned",
        }
        self.ft_rag_client = self._patch(
            "app.grounded_rag.generate_finetuned_response",
            AsyncMock(return_value=finetuned_result),
        )
        self.ft_client = self._patch(
            "app.main.generate_finetuned_response", AsyncMock(return_value=finetuned_result)
        )

    def _patch(self, target: str, mock: Any) -> Any:
        patcher = patch(target, new=mock)
        patcher.start()
        self.addCleanup(patcher.stop)
        return mock

    def _ask(self, path: str, **extra: Any) -> Any:
        body = {"courseId": COURSE, "question": "When is class?", **extra}
        response = self.client.post(path, json=body)
        self.assertEqual(response.status_code, 200, response.text)
        return response

    def test_base_route_uses_the_classroom_cap(self) -> None:
        self._ask("/api/base-model/generate")
        kwargs = self.base_chat.await_args.kwargs
        self.assertEqual(kwargs["options"]["num_predict"], 128)
        self.assertIn(CONCISE_ANSWER_INSTRUCTION, self.base_chat.await_args.args[0])

    def test_rag_route_uses_the_concise_prompt_and_cap(self) -> None:
        self._ask("/api/rag/generate", topK=4)
        prompt = self.chat.await_args.args[0]
        self.assertIn(CONCISE_ANSWER_INSTRUCTION, prompt)
        self.assertNotIn("two to five sentences", prompt)
        self.assertEqual(self.chat.await_args.kwargs["options"]["num_predict"], 128)

    def test_fine_tuned_route_sends_the_bare_question_with_the_cap(self) -> None:
        self._ask("/api/fine-tuned/generate")
        args, kwargs = self.ft_client.await_args
        self.assertEqual(args[0], "When is class?")
        self.assertEqual(kwargs["max_new_tokens"], 128)

    def test_fine_tuned_rag_route_uses_the_concise_prompt_and_cap(self) -> None:
        self._ask("/api/fine-tuned-rag/generate", topK=4)
        args, kwargs = self.ft_rag_client.await_args
        self.assertIn(CONCISE_ANSWER_INSTRUCTION, args[0])
        self.assertEqual(kwargs["max_new_tokens"], 128)


class ControlledCallersUnchangedTests(unittest.IsolatedAsyncioTestCase):
    async def test_without_the_flag_both_targets_use_the_controlled_profile(self) -> None:
        with patch(
            "app.grounded_rag.generate_ollama_chat",
            new=AsyncMock(return_value={"answer": "a", "model": "m"}),
        ) as chat:
            await generate_from_prompt("PROMPT", course_id=COURSE, target=BASE_TARGET)
        self.assertEqual(chat.await_args.kwargs["options"], grounded_options())

        with (
            patch(
                "app.grounded_rag.resolve_current_course_model",
                return_value={"version": "v2"},
            ),
            patch(
                "app.grounded_rag.generate_finetuned_response",
                new=AsyncMock(
                    return_value={
                        "answer": "a",
                        "model": "m",
                        "adapter_loaded": True,
                        "generation_seconds": 1.0,
                    }
                ),
            ) as client,
        ):
            await generate_from_prompt("PROMPT", course_id=COURSE, target=fine_tuned_target())
        self.assertIsNone(client.await_args.kwargs["max_new_tokens"])


if __name__ == "__main__":
    unittest.main()
