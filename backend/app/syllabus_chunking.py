"""Generic syllabus chunking for multi-course RAG indexes.

Uses one algorithm for every syllabus. Does not hardcode course-specific headings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TARGET_CHUNK_CHARS = 850
MAX_CHUNK_CHARS = 1300
MIN_CHUNK_CHARS = 700
OVERLAP_CHARS = 125
DEFAULT_SECTION_TITLE = "General"

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)*\s+\S")
ALL_CAPS_WORD = re.compile(r"^[A-Z0-9][A-Z0-9\s\-/&:,']+$")


@dataclass(frozen=True)
class SyllabusChunk:
    chunk_id: str
    section_title: str
    text: str
    order: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "chunkId": self.chunk_id,
            "sectionTitle": self.section_title,
            "text": self.text,
            "order": self.order,
        }


def normalize_syllabus_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]

    cleaned: list[str] = []
    blank_run = 0
    for line in lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= 1:
                cleaned.append("")
            continue
        blank_run = 0
        cleaned.append(line.strip())

    while cleaned and cleaned[0] == "":
        cleaned.pop(0)
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    return "\n".join(cleaned)


def is_likely_heading(paragraph: str) -> bool:
    stripped = paragraph.strip()
    if not stripped or "\n" in stripped:
        return False
    if len(stripped) > 80:
        return False
    if stripped.endswith(".") and not stripped.endswith("..."):
        return False

    words = stripped.split()
    if not words or len(words) > 12:
        return False

    if NUMBERED_HEADING.match(stripped):
        return True
    if stripped.endswith(":"):
        return True
    if ALL_CAPS_WORD.match(stripped) and len(words) <= 8:
        return True
    if stripped.istitle() and len(words) <= 8:
        return True

    return False


def _split_paragraphs(text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n", text)
    return [block.strip() for block in blocks if block.strip()]


def _split_into_sections(paragraphs: list[str]) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = DEFAULT_SECTION_TITLE
    current_body: list[str] = []

    def flush() -> None:
        nonlocal current_body
        if current_body:
            sections.append((current_title, current_body))
            current_body = []

    for paragraph in paragraphs:
        if is_likely_heading(paragraph):
            flush()
            current_title = paragraph.rstrip(":")
            continue
        current_body.append(paragraph)

    flush()

    if not sections and paragraphs:
        return [(DEFAULT_SECTION_TITLE, paragraphs)]
    return sections


def _find_sentence_split(text: str, target: int) -> int:
    """Prefer a sentence boundary near target; fall back to whitespace."""
    if len(text) <= target:
        return len(text)

    window_start = max(MIN_CHUNK_CHARS, target - 120)
    window_end = min(len(text), target + 80)
    window = text[window_start:window_end]

    best = -1
    for match in SENTENCE_BOUNDARY.finditer(window):
        candidate = window_start + match.end()
        if candidate <= MAX_CHUNK_CHARS:
            best = candidate

    if best > 0:
        return best

    space = text.rfind(" ", MIN_CHUNK_CHARS, min(len(text), MAX_CHUNK_CHARS))
    if space > 0:
        return space
    return min(len(text), MAX_CHUNK_CHARS)


def _chunk_section_text(section_title: str, paragraphs: list[str]) -> list[str]:
    heading_prefix = section_title.strip()
    body = "\n\n".join(paragraphs).strip()
    if not body:
        return []

    # Keep heading attached to following text in the chunk body.
    full_text = f"{heading_prefix}\n{body}" if heading_prefix else body

    if len(full_text) <= MAX_CHUNK_CHARS:
        return [full_text]

    chunks: list[str] = []
    remaining = full_text
    first = True

    while remaining:
        if len(remaining) <= MAX_CHUNK_CHARS:
            chunks.append(remaining.strip())
            break

        split_at = _find_sentence_split(remaining, TARGET_CHUNK_CHARS)
        piece = remaining[:split_at].strip()
        if not piece:
            piece = remaining[:MAX_CHUNK_CHARS].strip()
            split_at = len(piece)

        if first:
            chunks.append(piece)
            first = False
        else:
            # Repeat section title for continuity on long sections.
            continuation = piece
            if heading_prefix and not continuation.startswith(heading_prefix):
                continuation = f"{heading_prefix}\n{continuation}"
            chunks.append(continuation)

        if split_at >= len(remaining):
            break

        overlap_start = max(0, split_at - OVERLAP_CHARS)
        # Prefer starting overlap at a word boundary.
        while overlap_start > 0 and remaining[overlap_start] not in " \n":
            overlap_start -= 1
        next_text = remaining[overlap_start:].lstrip()
        if next_text == remaining:
            next_text = remaining[split_at:].lstrip()
        remaining = next_text

    return [chunk for chunk in chunks if chunk.strip()]


def chunk_syllabus_text(text: str) -> list[SyllabusChunk]:
    normalized = normalize_syllabus_text(text)
    if not normalized.strip():
        return []

    paragraphs = _split_paragraphs(normalized)
    sections = _split_into_sections(paragraphs)

    chunks: list[SyllabusChunk] = []
    order = 1
    for section_title, section_paragraphs in sections:
        for chunk_text in _chunk_section_text(section_title, section_paragraphs):
            if not chunk_text.strip():
                continue
            chunks.append(
                SyllabusChunk(
                    chunk_id=f"chunk-{order:03d}",
                    section_title=section_title,
                    text=chunk_text.strip(),
                    order=order,
                )
            )
            order += 1

    return chunks
