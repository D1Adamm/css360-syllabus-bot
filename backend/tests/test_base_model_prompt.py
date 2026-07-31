"""Tests for the ungrounded Base Model instruction prompt."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.ollama import build_base_model_prompt, generate_base_model_response

SYLLABUS_CLAIM_QUESTION = (
    "Based only on the syllabus, explain the late policy and the extension rules."
)


class BaseModelPromptTests(unittest.TestCase):
    def test_prompt_states_no_syllabus_context_was_supplied(self) -> None:
        prompt = build_base_model_prompt("What is the late policy?")
        lowered = prompt.lower()

        self.assertIn("no syllabus", lowered)
        self.assertIn("course-specific context has been supplied", lowered)

    def test_prompt_forbids_claiming_syllabus_access(self) -> None:
        prompt = build_base_model_prompt(SYLLABUS_CLAIM_QUESTION).lower()

        self.assertIn("do not claim to have read, received", prompt)
        self.assertIn("been provided a syllabus", prompt)

    def test_prompt_forbids_inventing_course_policies(self) -> None:
        prompt = build_base_model_prompt(SYLLABUS_CLAIM_QUESTION).lower()

        for forbidden in (
            "do not invent course policies",
            "deadlines",
            "grading rules",
            "communication procedures",
            "extension rules",
            "makeup rules",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertIn(forbidden, prompt)

    def test_prompt_requires_admitting_it_cannot_answer_course_questions(self) -> None:
        prompt = build_base_model_prompt(SYLLABUS_CLAIM_QUESTION).lower()

        self.assertIn("no syllabus context and cannot answer reliably", prompt)

    def test_prompt_allows_clearly_labeled_general_information(self) -> None:
        prompt = build_base_model_prompt(SYLLABUS_CLAIM_QUESTION).lower()

        self.assertIn("general, non-course-specific information", prompt)
        self.assertIn("clearly label it as general", prompt)

    def test_syllabus_framing_does_not_imply_a_syllabus_was_given(self) -> None:
        prompt = build_base_model_prompt(SYLLABUS_CLAIM_QUESTION).lower()

        self.assertIn("is incorrect. nothing was provided", prompt)
        self.assertIn("as if a syllabus", prompt)

    def test_student_question_is_preserved_verbatim(self) -> None:
        prompt = build_base_model_prompt(f"  {SYLLABUS_CLAIM_QUESTION}  ")

        self.assertIn(SYLLABUS_CLAIM_QUESTION, prompt)

    def test_prompt_adds_no_syllabus_content(self) -> None:
        """Base Model must stay an ungrounded baseline: no retrieved context."""
        prompt = build_base_model_prompt(SYLLABUS_CLAIM_QUESTION).lower()

        self.assertNotIn("syllabus context:", prompt)
        self.assertNotIn("[section:", prompt)
        self.assertNotIn("source sections:", prompt)


class BaseModelGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_generation_sends_the_instruction_prompt(self) -> None:
        completion = AsyncMock(
            return_value={"answer": "No syllabus was provided.", "model": "llama3.2:3b"}
        )
        with patch("app.ollama.generate_ollama_completion", new=completion):
            result = await generate_base_model_response(SYLLABUS_CLAIM_QUESTION)

        sent_prompt = completion.await_args.args[0]
        self.assertIn("No syllabus, course document", sent_prompt)
        self.assertIn(SYLLABUS_CLAIM_QUESTION, sent_prompt)
        self.assertEqual(result["response_type"], "base")


class BaseModelEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ollama_patch = patch(
            "app.ollama.generate_ollama_completion",
            new=AsyncMock(
                return_value={
                    "answer": "No syllabus was provided to me.",
                    "model": "llama3.2:3b",
                }
            ),
        )
        self.completion = self._ollama_patch.start()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self._ollama_patch.stop()

    def test_endpoint_wraps_question_with_grounding_rules(self) -> None:
        response = self.client.post(
            "/base-model/generate",
            json={
                "courseId": "css-430-summer-2026-ibce",
                "question": SYLLABUS_CLAIM_QUESTION,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["responseType"], "base")

        sent_prompt = self.completion.await_args.args[0]
        self.assertIn("No syllabus, course document", sent_prompt)
        self.assertIn("Do not invent course policies", sent_prompt)
        self.assertIn(SYLLABUS_CLAIM_QUESTION, sent_prompt)


if __name__ == "__main__":
    unittest.main()
