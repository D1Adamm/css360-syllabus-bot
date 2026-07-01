import unittest

from app.rag import (
    _is_late_policy_question,
    _should_skip_chunk_for_selection,
    build_rag_prompt,
    read_syllabus,
    split_syllabus_into_chunks,
)


class RagLatePolicyTests(unittest.TestCase):
    def test_late_policy_chunk_preserves_tasks_1_through_6(self) -> None:
        chunks = split_syllabus_into_chunks(read_syllabus())
        late_policy_chunk = next(
            chunk for chunk in chunks if chunk["section_title"] == "Late Policy"
        )

        self.assertIn("Bot Project Tasks 1-6", late_policy_chunk["text"])
        self.assertIn("these 6 project tasks", late_policy_chunk["text"])
        self.assertNotIn("Bot Project Tasks 1-5", late_policy_chunk["text"])

    def test_build_rag_prompt_requires_exact_numeric_preservation(self) -> None:
        prompt = build_rag_prompt(
            "What is the late policy for bot project tasks?",
            [
                {
                    "section": "Late Policy",
                    "text": (
                        "With respect to Bot Project Tasks 1-6, you may choose one 48-hour "
                        "extension per quarter."
                    ),
                }
            ],
        )

        self.assertIn("Preserve exact numbers", prompt)
        self.assertIn("Bot Project Tasks 1-6", prompt)
        self.assertIn("Do not narrow, widen, renumber, or paraphrase them.", prompt)

    def test_late_policy_question_skips_bot_task_chunks_after_late_policy_selected(self) -> None:
        question = "What is the late policy for bot project tasks?"
        selected = [{"section": "Late Policy", "chunk_id": "late-policy-001", "text": "...", "score": 1.0}]
        bot_task_chunk = {
            "section": "Bot Project Task #5",
            "chunk_id": "bot-project-task-5-001",
            "text": "...",
            "score": 0.8,
        }

        self.assertTrue(_is_late_policy_question(question))
        self.assertTrue(_should_skip_chunk_for_selection(question, bot_task_chunk, selected))


class RagPromptScopeTests(unittest.TestCase):
    def test_build_rag_prompt_requires_section_scope_fidelity(self) -> None:
        prompt = build_rag_prompt(
            "What is the difference between open lab and office hours?",
            [
                {
                    "section": "Class format and structure",
                    "text": "Each session is scheduled to run for a maximum of 120 minutes.",
                },
                {
                    "section": "Office Hours",
                    "text": "Open lab periods will include time for you to ask your questions.",
                },
            ],
        )

        self.assertIn("Keep each fact tied to the section it comes from", prompt)
        self.assertIn("Do not attribute class session rules to open lab", prompt)
        self.assertIn("[Section: Class format and structure]", prompt)
        self.assertIn("[Section: Office Hours]", prompt)

    def test_build_rag_prompt_keeps_separate_conditions_separate(self) -> None:
        prompt = build_rag_prompt(
            "What happens if I miss class?",
            [
                {
                    "section": "Impact of Missing Class",
                    "text": "Missing too many classes can make full in-class credit impossible.",
                },
                {
                    "section": "Your Presence in Class",
                    "text": "Missing the first two class sessions may result in being dropped.",
                },
            ],
        )

        self.assertIn("Keep separate conditions separate", prompt)
        self.assertIn("Do not merge consequences from different sentences", prompt)
        self.assertIn("without combining unrelated conditions", prompt)
        self.assertIn("beyond the first two", prompt)
        self.assertIn("first two class sessions", prompt)
        self.assertIn("repeated absences", prompt)
        self.assertIn("omit a secondary detail", prompt)

    def test_build_rag_prompt_preserves_attendance_rule_scope(self) -> None:
        prompt = build_rag_prompt(
            "What happens if I miss too many classes?",
            [
                {
                    "section": "Impact of Missing Class",
                    "text": (
                        "Missing too many classes can make full credit for in-class "
                        "activities impossible."
                    ),
                },
                {
                    "section": "Your Presence in Class",
                    "text": (
                        "Missing the first two class sessions may result in being dropped "
                        "if the course is full."
                    ),
                },
            ],
        )

        self.assertIn("Do not combine a dropped-from-the-course consequence", prompt)
        self.assertIn("one specific assignment, or all assignments", prompt)

    def test_build_rag_prompt_forbids_internal_label_references(self) -> None:
        prompt = build_rag_prompt(
            "Do I need a textbook?",
            [{"section": "Textbook", "text": "We do not have a required textbook."}],
        )

        self.assertIn("Do not mention internal context labels", prompt)
        self.assertIn("Section:", prompt)
        self.assertIn("natural student-facing prose", prompt)


if __name__ == "__main__":
    unittest.main()
