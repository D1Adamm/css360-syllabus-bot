"""The shared grounded prompt and decoding recipe.

One template for RAG and Fine-Tuned + RAG, one set of decoding options for
every condition, and the guarantees the template carries: exact figures,
source order, scope, abstention, false-premise correction, multi-part
handling, plain prose. The two older builder names are the same function.
"""

from __future__ import annotations

import hashlib
import unittest

from app.finetuned_rag import build_finetuned_rag_prompt
from app.grounded_generation import (
    GROUNDED_NUM_CTX,
    GROUNDED_NUM_PREDICT,
    GROUNDED_OPTIONS,
    PROMPT_TEMPLATE_NAME,
    RULES,
    build_grounded_prompt,
    grounded_options,
    prompt_template_fingerprint,
    render_excerpts,
    render_requested_parts,
)
from app.rag import build_rag_prompt
from app.retrieval_diversity import MAX_CHUNK_CONTEXT_CHARS, MAX_FINAL_TOP_K

CHUNKS = [
    {"section": "Late Policy", "text": "one 48-hour extension per quarter, no questions asked."},
    {"section": "Assignments", "text": "There will be no exams."},
]


class SharedTemplateTests(unittest.TestCase):
    def test_the_two_builder_names_are_the_same_prompt(self) -> None:
        question = "When is it due, and can I get an extension?"
        facets = ["when is it due", "can I get an extension"]
        shared = build_grounded_prompt(question, CHUNKS, facets)
        self.assertEqual(build_rag_prompt(question, CHUNKS, facets), shared)
        self.assertEqual(build_finetuned_rag_prompt(question, CHUNKS, facets), shared)
        self.assertEqual(build_rag_prompt(question, CHUNKS), build_finetuned_rag_prompt(question, CHUNKS))

    def test_layout_is_rules_excerpts_parts_question_answer(self) -> None:
        prompt = build_grounded_prompt("Q?", CHUNKS, ["part one", "part two"])
        positions = [
            prompt.index("Rules:"),
            prompt.index("Syllabus excerpts:"),
            prompt.index("Requested parts (address each exactly once):"),
            prompt.index("Student question:\nQ?"),
        ]
        self.assertEqual(positions, sorted(positions))
        self.assertTrue(prompt.endswith("Answer:"))
        for rule in RULES:
            self.assertIn(f"- {rule}", prompt)

    def test_excerpts_are_section_blocks_separated_by_blank_lines(self) -> None:
        self.assertEqual(
            render_excerpts(CHUNKS),
            "[Section: Late Policy]\none 48-hour extension per quarter, no questions asked."
            "\n\n[Section: Assignments]\nThere will be no exams.",
        )
        # Retrieval's `section` key, a stored chunk's `sectionTitle`, and nothing.
        self.assertIn("[Section: Grading]", render_excerpts([{"sectionTitle": "Grading", "text": "x"}]))
        self.assertIn("[Section: Untitled]", render_excerpts([{"text": "x"}]))

    def test_requested_parts_only_for_two_or_more_facets(self) -> None:
        self.assertEqual(render_requested_parts(None), "")
        self.assertEqual(render_requested_parts(["only one"]), "")
        self.assertEqual(render_requested_parts(["a", " a "]), "")
        block = render_requested_parts([" first  part", "second part"])
        self.assertEqual(block, "Requested parts (address each exactly once):\n- first part\n- second part\n\n")
        self.assertNotIn("Requested parts", build_grounded_prompt("Q?", CHUNKS))

    def test_question_whitespace_is_normalised_and_text_kept_verbatim(self) -> None:
        prompt = build_grounded_prompt("  When   is it\ndue? ", CHUNKS)
        self.assertIn("Student question:\nWhen is it due?", prompt)
        self.assertIn("one 48-hour extension per quarter, no questions asked.", prompt)

    def test_the_template_is_course_generic(self) -> None:
        prompt = build_grounded_prompt("Q?", CHUNKS)
        for leaked in ("CSS 360", "CSS360", "Bot Project", "first two class sessions", "tasks 1-6"):
            self.assertNotIn(leaked, prompt)

    def test_abstention_and_false_premise_rules_are_present(self) -> None:
        prompt = build_grounded_prompt("Q?", CHUNKS)
        self.assertIn("say that the syllabus does not say", prompt)
        self.assertIn("If the question assumes something the excerpts contradict", prompt)
        self.assertIn("Do not mention excerpts, sections, context, retrieval, or AI", prompt)

    def test_the_largest_retrieval_budget_fits_the_context_window(self) -> None:
        """Five full-budget chunks plus the rules must leave room for the answer.

        Conservative token estimate of one token per three characters.
        """
        chunks = [{"section": f"Section {i}", "text": "x" * MAX_CHUNK_CONTEXT_CHARS} for i in range(MAX_FINAL_TOP_K)]
        prompt = build_grounded_prompt("Q?" * 40, chunks, ["part one", "part two", "part three"])
        self.assertLess(len(prompt) / 3, GROUNDED_NUM_CTX - GROUNDED_NUM_PREDICT)

    def test_fingerprint_names_this_exact_text(self) -> None:
        rendered = build_grounded_prompt(
            "{question}", [{"section": "{section}", "text": "{text}"}], ["{part one}", "{part two}"]
        )
        self.assertEqual(prompt_template_fingerprint(), hashlib.sha256(rendered.encode()).hexdigest())
        self.assertEqual(PROMPT_TEMPLATE_NAME, "grounded-v1")


class DecodingOptionsTests(unittest.TestCase):
    def test_options_are_greedy_with_a_fixed_window_and_cap(self) -> None:
        options = grounded_options()
        self.assertEqual(
            options,
            {
                "num_predict": 256,
                "temperature": 0,
                "repeat_penalty": 1.05,
                "repeat_last_n": 4096,
                "seed": 360,
                "num_ctx": 4096,
            },
        )
        self.assertEqual(dict(GROUNDED_OPTIONS), options)

    def test_each_call_gets_its_own_dict(self) -> None:
        first = grounded_options()
        first["num_predict"] = 1
        self.assertEqual(grounded_options()["num_predict"], 256)
        with self.assertRaises(TypeError):
            GROUNDED_OPTIONS["num_predict"] = 1  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
