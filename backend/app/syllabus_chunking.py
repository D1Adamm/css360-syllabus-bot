"""Generic syllabus chunking for multi-course RAG indexes.

Heading-aware, policy-preserving chunker. Does not hardcode course-specific
section names; uses structural signals (standalone short lines, title case,
trailing colons, numbered/Markdown headings, glued heading+prose lines).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

TARGET_CHUNK_CHARS = 850
MAX_CHUNK_CHARS = 1300
MIN_CHUNK_CHARS = 700
OVERLAP_CHARS = 125
DEFAULT_SECTION_TITLE = "General"
INDEX_VERSION = 2

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\s+(\S.*)$")
MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$")
ALL_CAPS_WORD = re.compile(r"^[A-Z0-9][A-Z0-9\s\-/&:,']+$")
TOC_NUMBER_ONLY = re.compile(r"^\d+(?:\.\d+)*\.?$")
BULLET_PREFIX = re.compile(r"^[•\-\*\u2022]\s*")
YEAR_IN_TITLE = re.compile(r"\b(?:19|20)\d{2}\b")
DATE_LINE = re.compile(
    r"(?:^|\b)(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\b",
    re.IGNORECASE,
)
WEEKDAY_DATE_LINE = re.compile(
    r"^(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    re.IGNORECASE,
)

SMALL_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "for",
        "in",
        "on",
        "at",
        "by",
        "vs",
        "via",
        "as",
        "with",
        "from",
        "into",
        "about",
    }
)

# Single-token labels that are usually body/list chrome, not section titles.
WEAK_SINGLE_WORD_HEADINGS = frozenset(
    {
        "if",
        "when",
        "note",
        "notes",
        "case",
        "task",
        "due",
        "tips",
        "steps",
        "optional",
        "read",
        "watch",
        "standup",
        "activity",
        "call",
        "because",
        "although",
        "please",
        "consider",
        "turn",
        "deliverables",
        "optional!",
        "turn in",
        "turnin",
    }
)

NAV_NOISE_PATTERNS = (
    re.compile(r"^jump\s+to\b", re.IGNORECASE),
    re.compile(r"^navigation\b", re.IGNORECASE),
    re.compile(r"^search$", re.IGNORECASE),
)

PROSE_START_MARKERS = (
    "with respect",
    "if you",
    "if i",
    "everyone ",
    "instructors ",
    "students ",
    "there will",
    "there are",
    "this course",
    "the best",
    "the assignments",
    "your messages",
    "we do not",
    "we will",
    "i prefer",
    "i will",
    "i ask",
    "i expect",
    "note that",
    "among these",
    "use canvas",
    "washington state",
    "call safecampus",
    "in this course",
    "this includes",
    "as detailed",
    "as your",
    "global events",
    "any student",
    "resources are",
    "please note",
    "although ",
    "surveys ",
    "software engineering is",
    "uw bothell",
)

# First-token allowlist for weaker glued-heading splits (non-marker remainders).
PROSE_STARTER_WORDS = frozenset(
    {
        "with",
        "if",
        "everyone",
        "instructors",
        "students",
        "there",
        "this",
        "the",
        "your",
        "we",
        "i",
        "note",
        "among",
        "use",
        "washington",
        "call",
        "in",
        "as",
        "global",
        "any",
        "resources",
        "please",
        "although",
        "surveys",
        "software",
        "when",
        "although",
        "you",
        "my",
        "our",
        "all",
        "each",
        "once",
        "after",
        "before",
        "during",
        "for",
        "on",
        "at",
        "by",
        "to",
        "from",
        "having",
        "learning",
        "adapted",
    }
)

SENTENCE_LIKE_PHRASES = (
    " will ",
    " are ",
    " is ",
    " have ",
    " has ",
    " must ",
    " should ",
    " would ",
    " can ",
    " may ",
    " expect ",
    " include ",
    " prefer ",
)


@dataclass(frozen=True)
class SyllabusChunk:
    chunk_id: str
    section_title: str
    text: str
    order: int
    document_title: str = ""
    heading_path: tuple[str, ...] = ()
    start_offset: int | None = None
    end_offset: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chunkId": self.chunk_id,
            "sectionTitle": self.section_title,
            "documentTitle": self.document_title,
            "text": self.text,
            "order": self.order,
            "headingPath": list(self.heading_path) if self.heading_path else [self.section_title],
        }
        if self.start_offset is not None:
            payload["startOffset"] = self.start_offset
        if self.end_offset is not None:
            payload["endOffset"] = self.end_offset
        return payload


@dataclass
class _Section:
    title: str
    paragraphs: list[str] = field(default_factory=list)
    heading_path: tuple[str, ...] = ()
    start_offset: int = 0


def normalize_syllabus_text(text: str) -> str:
    """Normalize newlines/NBSP and collapse excess blank lines."""
    normalized = (
        text.replace("\u00a0", " ")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
    )
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
        # Preserve leading tabs used as wiki bullets, but strip trailing space.
        cleaned.append(line.rstrip())

    while cleaned and cleaned[0] == "":
        cleaned.pop(0)
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    return "\n".join(cleaned)


def _strip_bullet(line: str) -> str:
    stripped = line.strip()
    stripped = BULLET_PREFIX.sub("", stripped)
    return stripped.strip()


def _is_nav_noise(line: str) -> bool:
    stripped = _strip_bullet(line)
    if not stripped:
        return False
    compact = re.sub(r"\s+", " ", stripped)
    if compact.lower() in {"contents", "table of contents", "toc"}:
        return True
    return any(pattern.search(compact) for pattern in NAV_NOISE_PATTERNS)


def _looks_like_sentence(text: str) -> bool:
    lowered = f" {text.lower()} "
    if any(phrase in lowered for phrase in SENTENCE_LIKE_PHRASES):
        return True
    words = text.split()
    if len(words) >= 12:
        return True
    return False


def _is_title_case_phrase(text: str) -> bool:
    words = [part for part in re.split(r"\s+", text.strip()) if part]
    if not words or len(words) > 12:
        return False
    alpha_words = 0
    for word in words:
        core = re.sub(r"^[^\w]+|[^\w]+$", "", word)
        if not core:
            continue
        lower = core.lower()
        if lower in SMALL_WORDS:
            continue
        alpha_words += 1
        if core[0].isdigit():
            continue
        if not core[0].isupper():
            return False
    return alpha_words >= 1


def _is_weak_heading_token(text: str) -> bool:
    cleaned = _clean_heading_title(text).lower().rstrip("!?.")
    return cleaned in WEAK_SINGLE_WORD_HEADINGS


def _is_strict_heading(text: str) -> bool:
    """Conservative heading check used for glued inline splits."""
    stripped = _strip_bullet(text).rstrip(":").strip()
    if not stripped or "\n" in stripped:
        return False
    if len(stripped) > 70:
        return False
    if stripped.endswith((".", "?", "!")):
        return False
    if DATE_LINE.search(stripped):
        return False
    if WEEKDAY_DATE_LINE.match(stripped):
        return False
    if re.match(r"^\d{1,2}:\d{2}\b", stripped):
        return False
    if "—" in stripped or " – " in stripped:
        return False
    if "/" in stripped and len(stripped) < 40:
        return False
    if _looks_like_sentence(stripped):
        return False
    words = stripped.split()
    if not words or len(words) > 10:
        return False
    if len(words) == 1 and (_is_weak_heading_token(stripped) or len(stripped) < 4):
        return False
    if not _is_title_case_phrase(stripped) and not ALL_CAPS_WORD.match(stripped):
        return False
    # Reject prefixes that still contain lowercase content words.
    for word in words:
        core = re.sub(r"^[^\w]+|[^\w]+$", "", word)
        if not core or core.lower() in SMALL_WORDS or core[0].isdigit():
            continue
        if core[0].islower():
            return False
    return True


def is_likely_heading(paragraph: str) -> bool:
    """Return True when a single-line paragraph looks like a section heading."""
    stripped = _strip_bullet(paragraph)
    if not stripped or "\n" in stripped:
        return False
    if _is_nav_noise(stripped):
        return False
    if len(stripped) > 80:
        return False
    if stripped.endswith(".") and not stripped.endswith("..."):
        return False
    if stripped.endswith("?") and len(stripped.split()) <= 6:
        return False
    if DATE_LINE.search(stripped):
        return False
    if WEEKDAY_DATE_LINE.match(stripped):
        return False
    if re.match(r"^\d{1,2}:\d{2}\b", stripped):
        return False
    if "—" in stripped or " – " in stripped:
        return False
    if "/" in stripped and len(stripped) < 40:
        return False
    if _looks_like_sentence(stripped) and not stripped.endswith(":"):
        return False

    words = stripped.split()
    if not words or len(words) > 12:
        return False

    md = MARKDOWN_HEADING.match(stripped)
    if md:
        return _is_strict_heading(md.group(2)) or _is_mixed_heading(md.group(2))

    numbered = NUMBERED_HEADING.match(stripped)
    if numbered:
        rest = numbered.group(2).strip()
        return (_is_strict_heading(rest) or _is_mixed_heading(rest)) and len(rest.split()) <= 10

    if stripped.endswith(":"):
        label = stripped.rstrip(":").strip()
        return bool(label) and len(label.split()) <= 10 and not _looks_like_sentence(label)

    if ":" in stripped:
        label, rest = stripped.split(":", 1)
        label = label.strip()
        rest = rest.strip()
        if (
            label
            and rest
            and len(stripped) <= 80
            and (_is_strict_heading(label) or _is_mixed_heading(label))
            and not _looks_like_prose_start(rest)
            and not _looks_like_sentence(rest)
        ):
            return True

    if _is_weak_heading_token(stripped):
        return False

    if ALL_CAPS_WORD.match(stripped) and len(words) <= 8:
        return True

    if _is_strict_heading(stripped):
        return True

    return _is_mixed_heading(stripped)


def _is_mixed_heading(text: str) -> bool:
    """Allow mixed-case headings such as 'Class format and structure'."""
    stripped = _strip_bullet(text).rstrip(":").strip()
    words = stripped.split()
    if not (
        len(words) >= 2
        and len(words) <= 8
        and stripped[0].isupper()
        and not _looks_like_sentence(stripped)
    ):
        return False
    upper = sum(1 for word in words if word[:1].isupper())
    if upper >= max(2, (len(words) + 1) // 2):
        return True
    # Single leading capital with lowercase/small-word remainder
    # (e.g. "Class format and structure").
    if upper >= 1 and all(
        word[:1].islower() or word.lower() in SMALL_WORDS or word[:1].isupper()
        for word in words[1:]
    ):
        return not any(phrase in f" {stripped.lower()} " for phrase in SENTENCE_LIKE_PHRASES)
    return False


def _looks_like_prose_start(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 12:
        return False
    if not stripped[0].isupper():
        return False
    if not re.search(r"[a-z]", stripped):
        return False
    lowered = stripped.lower()
    if any(lowered.startswith(marker) for marker in PROSE_START_MARKERS):
        return True
    if any(phrase in f" {lowered} " for phrase in SENTENCE_LIKE_PHRASES):
        return True
    # Long remainder after a short heading-like prefix is usually body prose.
    return len(stripped) >= 50 and " " in stripped and not _is_title_case_phrase(
        " ".join(stripped.split()[:4])
    )


def _heading_level(title: str) -> int:
    md = MARKDOWN_HEADING.match(title.strip())
    if md:
        return len(md.group(1))
    numbered = re.match(r"^(\d+(?:\.\d+)*)\s+\S", title.strip())
    if numbered:
        return numbered.group(1).count(".") + 1
    return 1


def _clean_heading_title(title: str) -> str:
    stripped = _strip_bullet(title)
    md = MARKDOWN_HEADING.match(stripped)
    if md:
        stripped = md.group(2).strip()
    return stripped.rstrip(":").strip()


def try_split_inline_heading(line: str) -> tuple[str, str] | None:
    """Split glued 'Heading Body' / 'Heading: body' lines when structurally clear."""
    stripped = line.strip()
    if not stripped or len(stripped) < 8:
        return None

    md = MARKDOWN_HEADING.match(stripped)
    if md:
        title = md.group(2).strip()
        if _is_strict_heading(title) and not _looks_like_prose_start(title):
            return _clean_heading_title(title), ""
        return None

    colon = re.match(r"^([^:\n]{1,70}):\s+(.+)$", stripped)
    if colon:
        label = colon.group(1).strip()
        rest = colon.group(2).strip()
        if not label or _is_weak_heading_token(label):
            return None
        # Course-code subtitles like "CSS360: Software Engineering" are not section starts.
        if re.fullmatch(r"[A-Za-z]{2,}\d{2,}", label.replace(" ", "")):
            return None
        if _is_strict_heading(label) and rest:
            # Topical subtitle lists stay in the heading ("In Class: Discuss, Listen, Co-Work").
            if (
                len(rest) <= 60
                and not _looks_like_prose_start(rest)
                and ("," in rest or " and " in rest.lower())
                and (_is_title_case_phrase(rest) or rest[0].isupper())
                and not any(phrase in f" {rest.lower()} " for phrase in SENTENCE_LIKE_PHRASES)
            ):
                return _clean_heading_title(stripped), ""
            return _clean_heading_title(label), rest

    words = stripped.split()
    if len(words) < 3:
        return None

    max_prefix = min(10, len(words) - 1)
    marker_matches: list[tuple[str, str]] = []
    prose_matches: list[tuple[str, str]] = []

    for n in range(max_prefix, 0, -1):
        prefix = " ".join(words[:n])
        rest = " ".join(words[n:])
        if not _is_strict_heading(prefix):
            continue
        if _is_weak_heading_token(prefix):
            continue
        lowered_rest = rest.lower()
        if any(lowered_rest.startswith(marker) for marker in PROSE_START_MARKERS):
            marker_matches.append((_clean_heading_title(prefix), rest))
            continue
        if _looks_like_prose_start(rest):
            prose_matches.append((_clean_heading_title(prefix), rest))

    # Prefer the longest prefix whose remainder starts with a clear prose marker.
    if marker_matches:
        return marker_matches[0]
    # Weaker fallback: only when the remainder's first word is a common sentence starter.
    for prefix, rest in prose_matches:
        if len(prefix.split()) < 2:
            continue
        first = rest.split()[0].lower().rstrip(".,:;!?")
        if first not in PROSE_STARTER_WORDS:
            continue
        return prefix, rest
    return None


def _is_toc_entry(line: str) -> bool:
    """Return True for Contents-table chrome that should not become body sections."""
    stripped = _strip_bullet(line)
    if not stripped:
        return True
    if TOC_NUMBER_ONLY.match(stripped):
        return True
    # Real body often starts as heading + substantial prose on the same line.
    split = try_split_inline_heading(stripped)
    if split and split[1].strip():
        return False
    if len(stripped) > 100 and _looks_like_sentence(stripped):
        return False
    # Short labels inside Contents (including mixed-case) are TOC entries.
    if len(stripped) <= 90 and not _looks_like_sentence(stripped):
        words = stripped.split()
        if words and stripped[0].isupper() and len(words) <= 12:
            return True
        if NUMBERED_HEADING.match(stripped):
            return True
    return False


def _is_course_code_subtitle(line: str) -> bool:
    stripped = _strip_bullet(line)
    match = re.match(r"^([A-Za-z]{2,}\s?\d{2,})\s*:\s*(.+)$", stripped)
    if not match:
        return False
    rest = match.group(2).strip()
    return len(rest) <= 80 and not _looks_like_prose_start(rest)


def _looks_like_document_title(line: str) -> bool:
    stripped = _clean_heading_title(line)
    if not stripped or len(stripped) > 90:
        return False
    words = stripped.split()
    if not words or len(words) > 12:
        return False
    if YEAR_IN_TITLE.search(stripped):
        return True
    if re.search(r"\((?:fall|winter|spring|summer)\b", stripped, re.IGNORECASE):
        return True
    return False


def _iter_logical_lines(text: str) -> list[tuple[int, str]]:
    """Return (start_offset, line) pairs for non-empty logical lines."""
    result: list[tuple[int, str]] = []
    offset = 0
    for raw_line in text.split("\n"):
        line_start = offset
        offset += len(raw_line) + 1
        if not raw_line.strip():
            continue
        result.append((line_start, raw_line.strip()))
    return result


def extract_document_title(lines: list[str]) -> str:
    for line in lines:
        if _is_nav_noise(line):
            continue
        candidate = _strip_bullet(line)
        split = try_split_inline_heading(candidate)
        if split and split[0]:
            candidate = split[0]
        if _looks_like_document_title(candidate) or is_likely_heading(candidate):
            return _clean_heading_title(candidate)
        break
    return ""


def _update_heading_path(path: list[str], title: str) -> list[str]:
    level = _heading_level(title)
    cleaned = _clean_heading_title(title)
    if level <= 1:
        return [cleaned]
    trimmed = path[: max(0, level - 1)]
    if not trimmed:
        return [cleaned]
    return [*trimmed, cleaned]


def split_into_sections(text: str) -> tuple[str, list[_Section]]:
    """Split normalized syllabus text into heading-scoped sections."""
    normalized = normalize_syllabus_text(text)
    logical = _iter_logical_lines(normalized)
    if not logical:
        return "", []

    document_title = ""
    sections: list[_Section] = []
    current = _Section(title=DEFAULT_SECTION_TITLE, heading_path=(DEFAULT_SECTION_TITLE,))
    heading_path: list[str] = [DEFAULT_SECTION_TITLE]
    in_toc = False
    seen_body = False

    def flush() -> None:
        nonlocal current
        body = [paragraph for paragraph in current.paragraphs if paragraph.strip()]
        if body:
            current.paragraphs = body
            sections.append(current)
        current = _Section(
            title=heading_path[-1] if heading_path else DEFAULT_SECTION_TITLE,
            heading_path=tuple(heading_path) if heading_path else (DEFAULT_SECTION_TITLE,),
        )

    def start_section(title: str, offset: int) -> None:
        nonlocal heading_path, current, seen_body
        cleaned = _clean_heading_title(title)
        if document_title and cleaned == document_title:
            # Repeated document title lines are navigation noise, not sections.
            return
        flush()
        heading_path = _update_heading_path(heading_path, cleaned)
        current = _Section(
            title=cleaned,
            heading_path=tuple(heading_path),
            start_offset=offset,
        )
        seen_body = True

    for offset, line in logical:
        if _is_nav_noise(line):
            if _strip_bullet(line).lower() in {"contents", "table of contents", "toc"}:
                in_toc = True
            continue

        if _is_course_code_subtitle(line):
            continue

        if in_toc:
            if _is_toc_entry(line):
                continue
            # First non-TOC content ends the contents block.
            in_toc = False

        # Capture document title once; do not treat it as a body section heading.
        if not document_title and not seen_body:
            candidate = line
            split = try_split_inline_heading(line)
            title_candidate = split[0] if split else _clean_heading_title(candidate)
            if _looks_like_document_title(title_candidate):
                document_title = _clean_heading_title(title_candidate)
                if split and split[1]:
                    # Unusual: title glued to body on first line.
                    start_section(DEFAULT_SECTION_TITLE, offset)
                    current.paragraphs.append(split[1])
                    seen_body = True
                continue

        if document_title and _clean_heading_title(line) == document_title:
            continue

        split = try_split_inline_heading(line)
        if split:
            title, body = split
            start_section(title, offset)
            if body:
                current.paragraphs.append(body)
            continue

        if is_likely_heading(line):
            start_section(line, offset)
            continue

        if not seen_body and current.title == DEFAULT_SECTION_TITLE and not current.paragraphs:
            current.start_offset = offset
        current.paragraphs.append(line)
        seen_body = True

    flush()

    if not sections and logical:
        sections = [
            _Section(
                title=DEFAULT_SECTION_TITLE,
                paragraphs=[line for _, line in logical],
                heading_path=(DEFAULT_SECTION_TITLE,),
                start_offset=logical[0][0],
            )
        ]

    return document_title, sections


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


def _find_paragraph_split(text: str, target: int) -> int | None:
    """Prefer a blank-line / paragraph boundary near the target size."""
    if len(text) <= target:
        return len(text)
    window_start = max(200, target - 200)
    window_end = min(len(text), MAX_CHUNK_CHARS)
    best = -1
    index = text.find("\n\n", window_start)
    while index != -1 and index < window_end:
        candidate = index + 2
        if candidate <= MAX_CHUNK_CHARS:
            best = candidate
        index = text.find("\n\n", index + 2)
    return best if best > 0 else None


def _chunk_section_text(section_title: str, paragraphs: list[str]) -> list[str]:
    heading_prefix = section_title.strip()
    body = "\n\n".join(paragraphs).strip()
    if not body:
        return []

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

        para_split = _find_paragraph_split(remaining, TARGET_CHUNK_CHARS)
        split_at = para_split if para_split else _find_sentence_split(remaining, TARGET_CHUNK_CHARS)
        piece = remaining[:split_at].strip()
        if not piece:
            piece = remaining[:MAX_CHUNK_CHARS].strip()
            split_at = len(piece)

        if first:
            chunks.append(piece)
            first = False
        else:
            continuation = piece
            if heading_prefix and not continuation.startswith(heading_prefix):
                continuation = f"{heading_prefix}\n{continuation}"
            chunks.append(continuation)

        if split_at >= len(remaining):
            break

        overlap_start = max(0, split_at - OVERLAP_CHARS)
        while overlap_start > 0 and remaining[overlap_start] not in " \n":
            overlap_start -= 1
        next_text = remaining[overlap_start:].lstrip()
        if next_text == remaining:
            next_text = remaining[split_at:].lstrip()
        # Prefer not to begin an overlapped continuation mid-sentence.
        if next_text and next_text[0].islower():
            sentence_match = SENTENCE_BOUNDARY.search(next_text)
            if sentence_match and sentence_match.end() < len(next_text):
                next_text = next_text[sentence_match.end() :].lstrip()
            else:
                next_text = remaining[split_at:].lstrip()
        # Keep overlap within the same section only (already true here).
        remaining = next_text

    return [chunk for chunk in chunks if chunk.strip()]


def embedding_input_for_chunk(
    *,
    section_title: str,
    text: str,
) -> str:
    """Build the string embedded for retrieval (heading + body)."""
    title = section_title.strip() or DEFAULT_SECTION_TITLE
    body = text.strip()
    # Avoid duplicating the heading when chunk text already starts with it.
    if body.lower().startswith(title.lower()):
        remainder = body[len(title) :].lstrip("\n :")
        if remainder:
            return f"Section: {title}\n\n{remainder}"
    return f"Section: {title}\n\n{body}"


def validate_chunks(
    chunks: list[SyllabusChunk],
    *,
    document_title: str = "",
    source_char_count: int = 0,
    strict: bool = False,
) -> list[str]:
    """Return human-readable validation warnings (or raise if strict)."""
    warnings: list[str] = []
    if not chunks:
        warnings.append("No chunks produced.")
        if strict:
            raise ValueError(warnings[0])
        return warnings

    if document_title:
        titled = sum(1 for chunk in chunks if chunk.section_title == document_title)
        ratio = titled / len(chunks)
        if ratio >= 0.5:
            warnings.append(
                f"{titled}/{len(chunks)} chunks use the document title as sectionTitle "
                f"({ratio:.0%}); heading detection likely failed."
            )

    mid_sentence = 0
    for chunk in chunks:
        body = chunk.text
        if chunk.section_title and body.startswith(chunk.section_title):
            body = body[len(chunk.section_title) :].lstrip("\n :")
        first = body.lstrip()
        if first and first[0].islower():
            mid_sentence += 1
        if len(chunk.text) < 40:
            warnings.append(f"{chunk.chunk_id} is unusually small ({len(chunk.text)} chars).")
        if len(chunk.text) > MAX_CHUNK_CHARS + 120:
            warnings.append(f"{chunk.chunk_id} is unusually large ({len(chunk.text)} chars).")

    if mid_sentence >= max(2, len(chunks) // 5):
        warnings.append(
            f"{mid_sentence} chunks appear to start mid-sentence (lowercase start)."
        )

    if source_char_count >= 2000:
        meaningful = {
            chunk.section_title
            for chunk in chunks
            if chunk.section_title not in {DEFAULT_SECTION_TITLE, document_title, ""}
        }
        if len(meaningful) < 2:
            warnings.append(
                "Few meaningful headings detected in a long document; "
                "section titles may be under-specified."
            )

    seen_text: dict[str, str] = {}
    for chunk in chunks:
        key = re.sub(r"\s+", " ", chunk.text.strip().lower())
        if key in seen_text:
            warnings.append(
                f"Duplicate chunk text: {chunk.chunk_id} matches {seen_text[key]}."
            )
        else:
            seen_text[key] = chunk.chunk_id

    if strict and warnings:
        raise ValueError("; ".join(warnings))
    return warnings


def chunk_syllabus_text(text: str) -> list[SyllabusChunk]:
    normalized = normalize_syllabus_text(text)
    if not normalized.strip():
        return []

    document_title, sections = split_into_sections(normalized)
    chunks: list[SyllabusChunk] = []
    order = 1

    for section in sections:
        piece_texts = _chunk_section_text(section.title, section.paragraphs)
        # Approximate offsets within the section body for metadata.
        cursor = section.start_offset
        for chunk_text in piece_texts:
            if not chunk_text.strip():
                continue
            end = cursor + len(chunk_text)
            chunks.append(
                SyllabusChunk(
                    chunk_id=f"chunk-{order:03d}",
                    section_title=section.title,
                    text=chunk_text.strip(),
                    order=order,
                    document_title=document_title,
                    heading_path=section.heading_path or (section.title,),
                    start_offset=cursor,
                    end_offset=end,
                )
            )
            order += 1
            cursor = max(cursor, end - OVERLAP_CHARS)

    return chunks


def summarize_chunking(chunks: list[SyllabusChunk]) -> dict[str, Any]:
    titles = [chunk.section_title for chunk in chunks]
    unique_titles = []
    seen = set()
    for title in titles:
        if title not in seen:
            unique_titles.append(title)
            seen.add(title)
    return {
        "chunkCount": len(chunks),
        "documentTitle": chunks[0].document_title if chunks else "",
        "sectionTitles": unique_titles,
        "sectionCount": len(unique_titles),
    }
