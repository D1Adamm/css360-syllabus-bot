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

        self.assertIn("Copy numbers, dates, times, percentages, ranges, task numbers, and deadlines", prompt)
        self.assertIn("Bot Project Tasks 1-6", prompt)
        self.assertIn("Do not round, widen, narrow, renumber, or reorder them.", prompt)

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
    """`build_rag_prompt` is `grounded_generation.build_grounded_prompt`.

    The detailed wording is pinned in `test_grounded_generation.py` and
    `test_rag_quality_pass.py`; here, the scope guarantees this file always
    held, restated for the shared template.
    """

    def test_build_rag_prompt_is_the_shared_grounded_prompt(self) -> None:
        from app.grounded_generation import build_grounded_prompt

        chunks = [
            {"section": "Class format and structure", "text": "Each session runs 120 minutes."},
            {"section": "Office Hours", "text": "Book at least 24 hours in advance."},
        ]
        self.assertEqual(
            build_rag_prompt("How long is class?", chunks),
            build_grounded_prompt("How long is class?", chunks),
        )

    def test_build_rag_prompt_requires_section_scope_fidelity(self) -> None:
        prompt = build_rag_prompt(
            "How long is class, and how do I book office hours?",
            [
                {"section": "Class format and structure", "text": "Each session runs 120 minutes."},
                {"section": "Office Hours", "text": "Book at least 24 hours in advance."},
            ],
        )
        self.assertIn("Keep each rule attached to the assignment, session, or situation it is stated for", prompt)
        self.assertIn("Do not extend a rule to other situations or merge consequences", prompt)
        self.assertIn("[Section: Class format and structure]", prompt)
        self.assertIn("[Section: Office Hours]", prompt)

    def test_build_rag_prompt_keeps_separate_conditions_separate(self) -> None:
        prompt = build_rag_prompt(
            "What happens if I miss class?",
            [{"section": "Your Presence in Class", "text": "Two separate rules apply."}],
        )
        self.assertIn("keep separate conditions separate", prompt)
        self.assertIn("unless the excerpts do", prompt)

    def test_build_rag_prompt_forbids_internal_label_references(self) -> None:
        prompt = build_rag_prompt(
            "When is the reflection due?",
            [{"section": "Reflection", "text": "Sunday, December 14, 11:59 p.m."}],
        )
        self.assertIn("Do not mention excerpts, sections, context, retrieval, or AI", prompt)
        self.assertIn("Write two to five sentences of natural prose addressed to the student", prompt)
        self.assertIn("[Section: Reflection]", prompt)

if __name__ == "__main__":
    unittest.main()
