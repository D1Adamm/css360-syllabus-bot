"""The one grounded prompt and the one decoding recipe, shared by RAG and Fine-Tuned + RAG.

Before this module, the two grounded conditions were built by two different
prompt templates (about thirty course-flavoured rules in one, ten generic rules
in the other), sent through two different Ollama endpoints with two different
decoding recipes (sampling defaults against greedy), under two different context
windows and output caps. A comparison of the two therefore measured the
pipelines, not the weights. The v2-vs-v3 benchmark of 2026-09-11 found the two
fine-tuned versions answering identically and the plain fine-tuned models
recalling none of their training facts; nothing in that result could be
attributed to the adapter, because the adapter was never the only difference.

This module is what the two conditions now share:

- `build_grounded_prompt` renders the question, the retrieved excerpts and the
  optional multi-part list into one text. It is the text a student's RAG
  request is answered from, the text a Fine-Tuned + RAG request is answered
  from, and the text every grounded training example is rendered with, so the
  adapter is trained on exactly the distribution it is deployed on.
- `grounded_options` is the decoding recipe both conditions send to Ollama's
  `/api/chat`: greedy, a fixed output cap, a fixed context window. The local
  fine-tuned service builds the same dictionary from its own constants, and a
  test in the backend suite pins the two equal.

Deliberately dependency-free: the training script renders prompts at export
time through the backend, and the fingerprint below is what a manifest records
so a run can say which template its examples were rendered with.
"""

from __future__ import annotations

import hashlib
from types import MappingProxyType
from typing import Any, Mapping, Sequence

#: Named in manifests and runtime reports. Bump when the text below changes so
#: that a dataset rendered with an older template is distinguishable.
PROMPT_TEMPLATE_NAME = "grounded-v1"

#: Context window. Ollama truncates a longer prompt from the front, which for
#: a grounded prompt would drop the rules first; 4096 covers the largest prompt
#: the retrieval budget can produce (about 1,600 tokens) with room for the
#: answer.
GROUNDED_NUM_CTX = 4096
#: Output cap. 160 cut answers mid-sentence in the 2026-09-11 benchmark; the
#: longest correct grounded answers there were under 200 tokens.
GROUNDED_NUM_PREDICT = 256
GROUNDED_REPEAT_PENALTY = 1.05
GROUNDED_SEED = 360
#: What the backend allows a grounded generation, on either path. The
#: fine-tuned service already allows 120 s; the base path used to allow 60.
GROUNDED_TIMEOUT_SECONDS = 120.0


def grounded_options() -> dict[str, int | float]:
    """The decoding options, as a fresh dict a caller may hand to httpx.

    `temperature: 0` is greedy decoding. The repetition penalty applies over
    the whole context window, as the Tillicum service's Transformers settings
    did. The seed does nothing under greedy decoding and is carried so the
    request says which run it reproduces.
    """
    return {
        "num_predict": GROUNDED_NUM_PREDICT,
        "temperature": 0,
        "repeat_penalty": GROUNDED_REPEAT_PENALTY,
        "repeat_last_n": GROUNDED_NUM_CTX,
        "seed": GROUNDED_SEED,
        "num_ctx": GROUNDED_NUM_CTX,
    }


#: Read-only view for comparisons and tests.
GROUNDED_OPTIONS: Mapping[str, int | float] = MappingProxyType(grounded_options())


PREAMBLE = (
    "You are answering a student's question about their course. Answer using "
    "only the syllabus excerpts below."
)

RULES: tuple[str, ...] = (
    "Use only what the excerpts say. Do not add policies, dates, numbers, names, "
    "reasons, conditions, or procedures from general knowledge.",
    "Copy numbers, dates, times, percentages, ranges, task numbers, and deadlines "
    "exactly as written, and keep numbered or ordered lists in the order shown. "
    "Do not round, widen, narrow, renumber, or reorder them.",
    "Keep each rule attached to the assignment, session, or situation it is stated "
    "for, and keep separate conditions separate. Do not extend a rule to other "
    "situations or merge consequences unless the excerpts do.",
    "If the excerpts do not contain the answer, say that the syllabus does not say, "
    "then briefly give what it does say about the topic if that helps. Do not guess.",
    "If the question assumes something the excerpts contradict, say so plainly and "
    "give what the syllabus actually says instead.",
    "Answer every part of a multi-part question exactly once. For a part the "
    "excerpts do not cover, say so.",
    "Write two to five sentences of natural prose addressed to the student. Do not "
    "mention excerpts, sections, context, retrieval, or AI.",
)

EXCERPTS_HEADER = "Syllabus excerpts:"
REQUESTED_PARTS_HEADER = "Requested parts (address each exactly once):"
QUESTION_HEADER = "Student question:"
ANSWER_HEADER = "Answer:"


def _section_of(chunk: Mapping[str, Any]) -> str:
    for key in ("section", "sectionTitle", "section_title"):
        value = chunk.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return "Untitled"


def render_excerpts(retrieved_chunks: Sequence[Mapping[str, Any]]) -> str:
    """One `[Section: …]` block per chunk, in retrieval order, blank-line separated."""
    blocks = [
        f"[Section: {_section_of(chunk)}]\n{str(chunk.get('text') or '').strip()}"
        for chunk in retrieved_chunks
    ]
    return "\n\n".join(blocks)


def clean_facets(facets: Sequence[str] | None) -> list[str]:
    cleaned: list[str] = []
    for facet in facets or ():
        text = " ".join(str(facet).split())
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def render_requested_parts(facets: Sequence[str] | None) -> str:
    """The multi-part block, or an empty string for a single-topic question.

    Present only when the deterministic facet splitter found two or more
    parts, for the same question on both grounded paths.
    """
    cleaned = clean_facets(facets)
    if len(cleaned) < 2:
        return ""
    lines = "\n".join(f"- {facet}" for facet in cleaned)
    return f"{REQUESTED_PARTS_HEADER}\n{lines}\n\n"


def build_grounded_prompt(
    question: str,
    retrieved_chunks: Sequence[Mapping[str, Any]],
    facets: Sequence[str] | None = None,
) -> str:
    """The grounded prompt: rules, excerpts, optional parts, the question."""
    rules = "\n".join(f"- {rule}" for rule in RULES)
    return (
        f"{PREAMBLE}\n\n"
        f"Rules:\n{rules}\n\n"
        f"{EXCERPTS_HEADER}\n{render_excerpts(retrieved_chunks)}\n\n"
        f"{render_requested_parts(facets)}"
        f"{QUESTION_HEADER}\n{' '.join(str(question).split())}\n\n"
        f"{ANSWER_HEADER}"
    )


def prompt_template_fingerprint() -> str:
    """SHA-256 of the template rendered with placeholders: names this exact text."""
    rendered = build_grounded_prompt(
        "{question}",
        [{"section": "{section}", "text": "{text}"}],
        ["{part one}", "{part two}"],
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()
