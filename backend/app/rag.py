import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException

from app.upstream_errors import log_upstream_failure

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
REPO_ROOT = BACKEND_DIR.parent
# Fixture used by legacy CSS 360 chunking unit tests (not the live multi-course API).
SYLLABUS_PATH = REPO_ROOT / "docs" / "syllabus.txt"
INDEX_PATH = BACKEND_DIR / "data" / "syllabus_index.json"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))

CHUNKING_VERSION = "3"
MAX_CHUNK_CHARS = 1800
MAX_PARAGRAPHS_PER_CHUNK = 4
MIN_PARAGRAPHS_PER_CHUNK = 2
DEFAULT_TOP_K = 3
RAG_DEBUG = os.getenv("RAG_DEBUG", "").lower() in ("1", "true", "yes")

BROAD_SECTIONS = frozenset(
    {
        "Course Introduction",
        "Course Websites",
        "Grading",
        "Class Goals",
        "Before Class",
        "In Class Goals",
        "Optional Reading",
        "Optional Materials",
        "Optional Resources",
        "Administrative Notes",
        "Schedule",
    }
)
BROAD_SECTION_PENALTY = 0.04
SINGLE_PARAGRAPH_SECTIONS = frozenset({"Course absence form"})
SINGLE_PARAGRAPH_RETURN_SECTIONS = {
    "Course absence form": "Course Websites",
}
IMPLICIT_SECTION_BOUNDARIES = (
    "Standup participation is equivalent",
    "Lab completion is equivalent",
    "Grade Questions",
    "Mapping Percentage to the 4.0 Scale",
)

INLINE_HEADING_PREFIXES = (
    "Impact of Missing Class",
    "Your Presence in Class",
    "Devices in Class",
    "Office Hours",
    "Religious Accommodations",
    "Student Conduct",
    "Safety",
    "Use of AI Tools",
    "Academic Dishonesty",
    "Disability Resources",
    "Mental Health",
    "Other Student Support",
    "Teaching and learning after periods of disruption",
    "Grade Questions",
    "Late Policy",
    "Mapping Percentage to the 4.0 Scale",
)

NUMBERED_HEADING_PATTERN = re.compile(r"^\d+(?:\.\d+)*\s+\S")
BOT_TASK_HEADING_PATTERN = re.compile(r"^Bot Project Task #\d+$", re.IGNORECASE)
SPRINT_HEADING_PATTERN = re.compile(r"^Sprint \d+ --")
SENTENCE_STARTERS = (
    "if ",
    "you ",
    "we ",
    "i ",
    "i'll ",
    "as ",
    "the ",
    "this ",
    "that ",
    "there ",
    "in ",
    "on ",
    "at ",
    "for ",
    "when ",
    "although ",
    "because ",
    "students ",
    "each ",
    "all ",
    "any ",
    "unless ",
    "surveys ",
    "going ",
    "open ",
    "see ",
    "do ",
    "prepare ",
    "divide ",
    "write ",
    "read ",
    "recall ",
    "demonstrate ",
    "engage ",
    "learn ",
    "identify ",
    "set ",
    "watch ",
    "discuss ",
    "deploy ",
    "version ",
    "analyze ",
    "release ",
    "tag ",
    "review ",
    "examine ",
    "hands-on",
    "time for",
    "devise ",
    "finish ",
    "introduce ",
    "host ",
)

SENTENCE_PHRASES = (
    " will ",
    " are ",
    " is ",
    " have ",
    " has ",
    " expect ",
    " include ",
    " should ",
    " would ",
    " can ",
    " if ",
    " that you ",
)

SUBSECTION_LABEL_PATTERN = re.compile(
    r"^(Task|Due|Deliverables|Turn In|Turn in|Tips|Steps|Standup|Case|Optional|Class Discussion)$",
    re.IGNORECASE,
)
DATE_LINE_PATTERN = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
    re.IGNORECASE,
)
EXPLICIT_HEADING_PATTERN = re.compile(
    r"^(Overview and Learning Objectives|Class format and structure|"
    r"Websites and Technology Expectations|Note about this Syllabus|Assignments|"
    r"Class Goals|Class Materials|Before Class|In Class Goals|Optional Reading|Optional Materials|"
    r"Cold Calling|Standup Meetings|Sprint Planning Meetings|Projects|"
    r"Case discussion|Assessment for case study discussion|Notes on Group Work|"
    r"Turnaround time commitment|Weather Notes|Course Websites|Schedule|Grading|"
    r"In Class: Discuss, Listen, Co-Work|Out of Class: Read, Watch, Write, and Code|"
    r"Project: Developing a Discord Bot as a Team|Demonstration of Bot 2\.0|"
    r"Pre-made Demos|Bot Feedback|Reflection on Software Engineering and the Bot Project|"
    r"Credit and Notes|Administrative Notes|Resources are available for you|"
    r"Contact|Course absence form|Impact of Missing Class|"
    r"Your Presence in Class|Devices in Class|Office Hours|Religious Accommodations|"
    r"Student Conduct|Safety|Use of AI Tools|Academic Dishonesty|Disability Resources|"
    r"Mental Health|Other Student Support|Teaching and learning after periods of disruption|"
    r"Grade Questions|Late Policy|Mapping Percentage to the 4\.0 Scale|Textbook|Course Meetings)$",
    re.IGNORECASE,
)

_index_cache: dict[str, Any] | None = None


def get_syllabus_path() -> Path:
    return SYLLABUS_PATH


def get_index_path() -> Path:
    return INDEX_PATH


def read_syllabus() -> str:
    syllabus_path = get_syllabus_path()
    if not syllabus_path.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Syllabus file not found at {syllabus_path}.",
        )

    try:
        return syllabus_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read syllabus file at {syllabus_path}.",
        ) from exc


def _iter_line_units(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _split_inline_heading(line: str) -> list[str]:
    for heading in INLINE_HEADING_PREFIXES:
        if line.startswith(heading):
            remainder = line[len(heading) :].strip()
            if remainder:
                return [heading, remainder]
            return [heading]

    if line.startswith("Contact:") and len(line) > len("Contact:"):
        remainder = line[len("Contact:") :].strip()
        return ["Contact", remainder] if remainder else ["Contact"]

    if line.startswith("Course absence form:") and len(line) > len("Course absence form:"):
        remainder = line[len("Course absence form:") :].strip()
        return ["Course absence form", remainder] if remainder else ["Course absence form"]

    return [line]


def _is_implicit_section_boundary(line: str) -> bool:
    return any(line.startswith(marker) for marker in IMPLICIT_SECTION_BOUNDARIES)


def _preprocess_syllabus_lines(text: str) -> list[str]:
    processed_lines: list[str] = []
    for line in _iter_line_units(text):
        processed_lines.extend(_split_inline_heading(line))
    return processed_lines


def _is_contents_table_entry(line: str) -> bool:
    return bool(NUMBERED_HEADING_PATTERN.match(line.strip()))


def _looks_like_sentence(line: str) -> bool:
    lowered = line.lower()
    if any(lowered.startswith(prefix) for prefix in SENTENCE_STARTERS):
        return True
    return any(phrase in lowered for phrase in SENTENCE_PHRASES)


def _is_label_style_colon_heading(line: str) -> bool:
    match = re.match(r"^([^:\n]{1,55}):$", line)
    if not match:
        return False

    label = match.group(1).strip()
    if not label or _looks_like_sentence(f"{label}:"):
        return False

    return len(label.split()) <= 8


def _is_section_heading(line: str) -> bool:
    if not line:
        return False

    if SUBSECTION_LABEL_PATTERN.match(line):
        return False

    if DATE_LINE_PATTERN.search(line):
        return False

    if line[0].isdigit():
        return False

    if "/" in line and len(line) < 40:
        return False

    if line == "Contents":
        return True

    if EXPLICIT_HEADING_PATTERN.match(line):
        return True

    if NUMBERED_HEADING_PATTERN.match(line):
        return True

    if BOT_TASK_HEADING_PATTERN.match(line):
        return True

    if SPRINT_HEADING_PATTERN.match(line):
        return True

    if ":" in line and not _is_label_style_colon_heading(line):
        return False

    if _is_label_style_colon_heading(line):
        return True

    if len(line) > 70:
        return False

    if line.endswith((".", "?", "!")):
        return False

    if _looks_like_sentence(line):
        return False

    words = line.split()
    if len(words) < 2 or len(words) > 10:
        return False

    return line[0].isupper()


def _slugify_section_title(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "section"


def _embedding_input(section_title: str, chunk_text: str) -> str:
    return f"Section: {section_title}\n\n{chunk_text}"


def _make_chunk_id(section_title: str, chunk_number: int) -> str:
    return f"{_slugify_section_title(section_title)}-{chunk_number:03d}"


def split_syllabus_into_chunks(text: str) -> list[dict[str, str]]:
    lines = _preprocess_syllabus_lines(text)
    chunks: list[dict[str, str]] = []
    current_section = "Course Introduction"
    current_paragraphs: list[str] = []
    current_chars = 0
    section_chunk_counts: dict[str, int] = {}
    in_contents_table = False

    def flush_chunk() -> None:
        nonlocal current_paragraphs, current_chars

        if not current_paragraphs:
            return

        section_chunk_counts[current_section] = section_chunk_counts.get(current_section, 0) + 1
        chunk_number = section_chunk_counts[current_section]
        chunk_text = "\n\n".join(current_paragraphs)

        chunks.append(
            {
                "id": _make_chunk_id(current_section, chunk_number),
                "section_title": current_section,
                "text": chunk_text,
            }
        )
        current_paragraphs = []
        current_chars = 0

    for line in lines:
        if line == "Contents":
            flush_chunk()
            in_contents_table = True
            current_section = "Contents"
            continue

        if in_contents_table:
            if _is_contents_table_entry(line):
                continue

            in_contents_table = False
            flush_chunk()
            current_section = line.rstrip(":")
            if _is_section_heading(line):
                continue

        if _is_section_heading(line):
            flush_chunk()
            current_section = line.rstrip(":")
            continue

        if _is_implicit_section_boundary(line):
            flush_chunk()
            current_section = "Grading"

        projected_chars = current_chars + len(line) + (2 if current_paragraphs else 0)
        should_flush = current_paragraphs and (
            len(current_paragraphs) >= MAX_PARAGRAPHS_PER_CHUNK
            or (
                projected_chars > MAX_CHUNK_CHARS
                and len(current_paragraphs) >= MIN_PARAGRAPHS_PER_CHUNK
            )
        )

        if should_flush:
            flush_chunk()

        current_paragraphs.append(line)
        current_chars += len(line) + (2 if len(current_paragraphs) > 1 else 0)

        if current_section in SINGLE_PARAGRAPH_SECTIONS and len(current_paragraphs) >= 1:
            flushed_section = current_section
            flush_chunk()
            current_section = SINGLE_PARAGRAPH_RETURN_SECTIONS.get(
                flushed_section,
                current_section,
            )

    flush_chunk()

    if not chunks:
        raise HTTPException(
            status_code=500,
            detail="Syllabus chunking produced no chunks.",
        )

    return _merge_adjacent_section_chunks(chunks)


def _merge_adjacent_section_chunks(chunks: list[dict[str, str]]) -> list[dict[str, str]]:
    if not chunks:
        return chunks

    merged_chunks: list[dict[str, str]] = []

    for chunk in chunks:
        if not merged_chunks:
            merged_chunks.append(chunk)
            continue

        previous = merged_chunks[-1]
        if previous["section_title"] != chunk["section_title"]:
            merged_chunks.append(chunk)
            continue

        combined_text = f"{previous['text']}\n\n{chunk['text']}"
        if len(combined_text) > MAX_CHUNK_CHARS:
            merged_chunks.append(chunk)
            continue

        merged_chunks[-1] = {
            "id": previous["id"],
            "section_title": previous["section_title"],
            "text": combined_text,
        }

    return merged_chunks


def compute_cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    if len(vector_a) != len(vector_b) or not vector_a:
        return 0.0

    dot_product = sum(left * right for left, right in zip(vector_a, vector_b))
    norm_a = sum(value * value for value in vector_a) ** 0.5
    norm_b = sum(value * value for value in vector_b) ** 0.5

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def _syllabus_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ensure_data_directory() -> None:
    get_index_path().parent.mkdir(parents=True, exist_ok=True)


async def get_embedding(text: str) -> list[float]:
    payload = {
        "model": OLLAMA_EMBEDDING_MODEL,
        "prompt": text,
    }

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json=payload,
            )
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama embedding request timed out. Ensure Ollama is running "
                f"and the {OLLAMA_EMBEDDING_MODEL} model is available."
            ),
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama is unavailable for embeddings. Start Ollama locally and "
                f"pull the {OLLAMA_EMBEDDING_MODEL} model."
            ),
        ) from exc

    if response.status_code >= 500:
        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama returned a server error while generating embeddings. "
                f"Ensure the {OLLAMA_EMBEDDING_MODEL} model is available locally."
            ),
        )

    if response.status_code >= 400:
        # RAG retrieval's embedding call. Same rule as everywhere else: the
        # upstream's prose is a log line, not a response body.
        log_upstream_failure(
            logger,
            "RAG embedding",
            url=f"{OLLAMA_BASE_URL}/api/embeddings",
            status_code=response.status_code,
            body=response.text,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "Ollama rejected the RAG embedding request "
                f"(HTTP {response.status_code}). See the backend log for the "
                "service's own response."
            ),
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an invalid JSON response for embeddings.",
        ) from exc

    embedding = data.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an empty or invalid embedding.",
        )

    if not all(isinstance(value, (int, float)) for value in embedding):
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an embedding with non-numeric values.",
        )

    return [float(value) for value in embedding]


def _index_is_stale(index_data: dict[str, Any], syllabus_hash: str) -> bool:
    if not index_data.get("chunks"):
        return True

    if index_data.get("syllabus_content_hash") != syllabus_hash:
        return True

    if index_data.get("embedding_model") != OLLAMA_EMBEDDING_MODEL:
        return True

    if index_data.get("chunking_version") != CHUNKING_VERSION:
        return True

    return False


def _question_terms(question: str) -> set[str]:
    return set(re.findall(r"[a-z]{4,}", question.lower()))


def _retrieval_score_adjustment(question: str, section: str, text: str) -> float:
    question_lower = question.lower()
    section_lower = section.lower()
    text_lower = text.lower()
    question_words = _question_terms(question)
    adjustment = 0.0

    section_words = set(re.findall(r"[a-z]{4,}", section_lower))
    title_overlap = question_words & section_words
    if title_overlap:
        adjustment += 0.05 * len(title_overlap)

    if section_lower in question_lower or question_lower in section_lower:
        adjustment += 0.12

    if any(term in question_lower for term in ("contact", "instructor", "reach", "message", "email", "discord")):
        if section == "Contact":
            adjustment += 0.2
        if section == "Turnaround time commitment":
            adjustment += 0.04
        if section in {"Cold Calling", "Impact of Missing Class"}:
            adjustment -= 0.14
        if section == "Office Hours":
            adjustment -= 0.14

    if any(term in question_lower for term in ("miss", "absent", "absence")):
        if section in {"Impact of Missing Class", "Your Presence in Class", "Course absence form"}:
            adjustment += 0.12
        if "one hour before" in text_lower:
            adjustment += 0.14
        if "does not serve as a makeup" in text_lower or "no direct way to make up" in text_lower:
            adjustment += 0.08
        if section == "Going to be absent?":
            adjustment -= 0.05

    if "late" in question_lower and any(
        term in question_lower for term in ("policy", "extension", "penalty", "task", "bot", "project")
    ):
        if section == "Late Policy":
            adjustment += 0.22
        if BOT_TASK_HEADING_PATTERN.match(section) and not re.search(r"task\s*#?\s*\d", question_lower):
            adjustment -= 0.18

    if "policy" in question_lower and section == "Late Policy":
        adjustment += 0.08

    return adjustment


def _adjusted_similarity_score(score: float, section: str) -> float:
    if section in BROAD_SECTIONS:
        return score - BROAD_SECTION_PENALTY
    return score


def _log_retrieval_ranking(question: str, ranked_chunks: list[dict[str, Any]]) -> None:
    if not RAG_DEBUG:
        return

    preview_lines = [
        f"RAG retrieval ranking for question: {question!r}",
    ]
    for index, chunk in enumerate(ranked_chunks[:10], start=1):
        preview_lines.append(
            "  "
            f"{index}. {chunk['section']} | final={chunk['score']:.4f} "
            f"(base={chunk.get('base_score', chunk['score']):.4f}, "
            f"adj={chunk.get('score_adjustment', 0.0):+.4f})"
        )
    print("\n".join(preview_lines))


def _build_ranked_chunks(
    question: str,
    scored_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for chunk in scored_chunks:
        base_score = chunk["score"]
        adjustment = _retrieval_score_adjustment(question, chunk["section"], chunk["text"])
        chunk["base_score"] = base_score
        chunk["score_adjustment"] = adjustment
        chunk["score"] = _adjusted_similarity_score(base_score, chunk["section"]) + adjustment

    return sorted(scored_chunks, key=lambda item: item["score"], reverse=True)


def _is_late_policy_question(question: str) -> bool:
    question_lower = question.lower()
    return "late" in question_lower and any(
        term in question_lower
        for term in ("policy", "extension", "penalty", "task", "bot", "project")
    )


def _should_skip_chunk_for_selection(
    question: str,
    chunk: dict[str, Any],
    selected_chunks: list[dict[str, Any]],
) -> bool:
    if not _is_late_policy_question(question):
        return False

    if not any(item["section"] == "Late Policy" for item in selected_chunks):
        return False

    return bool(BOT_TASK_HEADING_PATTERN.match(chunk["section"]))


def _select_diverse_chunks(
    question: str,
    ranked_chunks: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    specific_chunks = [
        chunk for chunk in ranked_chunks if chunk["section"] not in BROAD_SECTIONS
    ]
    broad_chunks = [chunk for chunk in ranked_chunks if chunk["section"] in BROAD_SECTIONS]

    selected_chunks: list[dict[str, Any]] = []
    seen_sections: set[str] = set()

    for candidate_pool in (specific_chunks, broad_chunks):
        for chunk in candidate_pool:
            section = chunk["section"]
            if section in seen_sections:
                continue

            if _should_skip_chunk_for_selection(question, chunk, selected_chunks):
                continue

            seen_sections.add(section)
            selected_chunks.append(chunk)

            if len(selected_chunks) >= top_k:
                return selected_chunks

    return selected_chunks


async def build_and_save_syllabus_index() -> dict[str, Any]:
    global _index_cache

    syllabus_text = read_syllabus()
    syllabus_hash = _syllabus_content_hash(syllabus_text)
    raw_chunks = split_syllabus_into_chunks(syllabus_text)

    indexed_chunks: list[dict[str, Any]] = []
    for chunk in raw_chunks:
        embedding = await get_embedding(
            _embedding_input(chunk["section_title"], chunk["text"])
        )
        indexed_chunks.append(
            {
                "id": chunk["id"],
                "section_title": chunk["section_title"],
                "text": chunk["text"],
                "embedding": embedding,
            }
        )

    index_data = {
        "syllabus_content_hash": syllabus_hash,
        "embedding_model": OLLAMA_EMBEDDING_MODEL,
        "chunking_version": CHUNKING_VERSION,
        "chunk_count": len(indexed_chunks),
        "chunks": indexed_chunks,
    }

    _ensure_data_directory()
    index_path = get_index_path()

    try:
        index_path.write_text(
            json.dumps(index_data, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to write syllabus index to {index_path}.",
        ) from exc

    _index_cache = index_data
    return index_data


async def load_syllabus_index() -> dict[str, Any] | None:
    global _index_cache

    if _index_cache is not None:
        return _index_cache

    index_path = get_index_path()
    if not index_path.is_file():
        return None

    try:
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read syllabus index at {index_path}.",
        ) from exc

    if not isinstance(index_data, dict):
        raise HTTPException(
            status_code=500,
            detail="Syllabus index file has an invalid format.",
        )

    _index_cache = index_data
    return index_data


async def ensure_syllabus_index() -> dict[str, Any]:
    syllabus_text = read_syllabus()
    syllabus_hash = _syllabus_content_hash(syllabus_text)
    index_data = await load_syllabus_index()

    if index_data is None or _index_is_stale(index_data, syllabus_hash):
        return await build_and_save_syllabus_index()

    return index_data


async def retrieve_syllabus_chunks(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    include_debug: bool = False,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]] | None]:
    index_data = await ensure_syllabus_index()
    chunks = index_data.get("chunks", [])

    if not chunks:
        raise HTTPException(
            status_code=500,
            detail="Syllabus index contains no chunks.",
        )

    question_embedding = await get_embedding(question)
    scored_chunks: list[dict[str, Any]] = []

    for chunk in chunks:
        chunk_embedding = chunk.get("embedding")
        if not isinstance(chunk_embedding, list) or not chunk_embedding:
            continue

        score = compute_cosine_similarity(question_embedding, chunk_embedding)
        scored_chunks.append(
            {
                "chunk_id": chunk["id"],
                "section": chunk["section_title"],
                "text": chunk["text"],
                "score": score,
            }
        )

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)
    ranked_chunks = _build_ranked_chunks(question, scored_chunks)
    selected_chunks = _select_diverse_chunks(question, ranked_chunks, top_k)
    _log_retrieval_ranking(question, ranked_chunks)

    debug_rankings = None
    if include_debug:
        selected_ids = {chunk["chunk_id"] for chunk in selected_chunks}
        debug_rankings = [
            {
                "chunk_id": chunk["chunk_id"],
                "section": chunk["section"],
                "base_score": chunk["base_score"],
                "score_adjustment": chunk["score_adjustment"],
                "score": chunk["score"],
                "selected": chunk["chunk_id"] in selected_ids,
            }
            for chunk in ranked_chunks[:10]
        ]

    return index_data["embedding_model"], selected_chunks, debug_rankings


def build_rag_prompt(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    facets: list[str] | None = None,
) -> str:
    context_blocks: list[str] = []
    source_sections: list[str] = []

    for chunk in retrieved_chunks:
        section = chunk["section"]
        source_sections.append(section)
        context_blocks.append(f"[Section: {section}]\n{chunk['text']}")

    unique_sections = list(dict.fromkeys(source_sections))
    section_list = "\n".join(f"- {section}" for section in unique_sections)

    cleaned_facets = [
        " ".join(str(facet).split())
        for facet in (facets or [])
        if str(facet).strip()
    ]
    multi_facet = len(cleaned_facets) >= 2

    facet_block = ""
    multi_facet_rules = ""
    if multi_facet:
        facet_lines = "\n".join(f"- {facet}" for facet in cleaned_facets)
        facet_block = (
            "Requested parts of the question (address each exactly once):\n"
            f"{facet_lines}\n\n"
        )
        multi_facet_rules = (
            "- This is a multi-part question. Structure the answer with one concise "
            "subsection or labeled paragraph per requested part listed above.\n"
            "- Derive short labels from the wording of each requested part (do not invent "
            "section names that are not implied by the question).\n"
            "- Address every requested part exactly once. If the retrieved excerpts do not "
            "specify a part, say that the retrieved excerpts do not specify it.\n"
            "- Omit unrelated details that appear in the retrieved context but do not answer "
            "a requested part.\n"
        )

    return (
        "You are answering a student question about a course syllabus using only the "
        "excerpts below.\n\n"
        "Rules:\n"
        "- Answer only from the supplied syllabus context.\n"
        "- Do not use general knowledge to invent course policies.\n"
        "- Carefully extract all directly relevant rules, deadlines, exceptions, limits, "
        "durations, quantities, and penalties from the context.\n"
        "- Do not say information is absent when the supplied context contains a relevant policy.\n"
        "- Include important qualifiers such as deadlines, penalties, exceptions, limits, "
        "durations, and no-makeup rules when they appear in the excerpts.\n"
        "- Preserve exact numbers, numeric ranges, task numbers, dates, times, percentages, "
        "quantities, limits, durations, and deadlines exactly as written in the context. "
        "Do not narrow, widen, renumber, or paraphrase them.\n"
        "- Preserve explicit phrases that define the policy, including wording such as "
        "\"no questions asked,\" when those phrases appear in the excerpts.\n"
        "- If the context says a range such as tasks 1-6, keep that exact range in your answer.\n"
        "- Preserve numbered and ordered lists in the exact order shown in the retrieved text. "
        "Do not reverse, reorder, or renumber steps or preferences.\n"
        "- Never add qualifications, eligibility requirements, reasons, exceptions, conditions, "
        "or procedures that are not directly stated in the excerpts.\n"
        "- Never broaden a policy about one assignment, activity, session, or situation into a "
        "policy about all assignments, exams, attendance, missed work, or other activity types "
        "unless the excerpts explicitly state that broader scope.\n"
        "- Keep each fact tied to the section it comes from. Do not transfer scheduling, "
        "duration, deadline, or policy details from one section or activity to another unless "
        "the context clearly states that connection.\n"
        "- Do not attribute class session rules to open lab, office hours, or other activities "
        "unless the context explicitly applies those rules to that activity.\n"
        "- Answer using only details that clearly apply to the student's question within the "
        "scope of the supplied sections.\n"
        "- Keep separate conditions separate. Do not merge consequences from different sentences "
        "or sections into one combined rule.\n"
        "- When the syllabus gives different rules for different situations, describe each rule "
        "on its own. Do not invent connecting phrases such as 'beyond the first two' unless the "
        "syllabus explicitly uses that wording.\n"
        "- Preserve whether each rule applies to the first two class sessions, repeated absences, "
        "one specific assignment, or all assignments.\n"
        "- Do not combine a dropped-from-the-course consequence with an in-class-credit consequence "
        "unless the syllabus explicitly links them.\n"
        "- If you are unsure how two conditions relate, state them separately or omit a secondary "
        "detail rather than merging them incorrectly.\n"
        "- State each condition and its consequence exactly as scoped in the context. Do not "
        "generalize a consequence beyond the condition stated in the syllabus.\n"
        "- Answer every distinct part of a multi-part question. For any unanswered part, state "
        "that the retrieved excerpts do not specify it.\n"
        "- Omit unrelated details even when they appear in the retrieved context.\n"
        "- Before returning the answer, perform a final claim-by-claim check against the excerpts "
        "and remove any claim that is not directly supported.\n"
        f"{multi_facet_rules}"
        "- If the context does not contain the answer, clearly say that the syllabus does not "
        "provide that information.\n"
        "- Keep the answer concise and student-friendly, but do not omit important policy details.\n"
        "- Do not tell the student to email the instructor unless the supplied syllabus context "
        "specifically says to do so.\n"
        "- Do not mention that the answer was generated by AI.\n"
        "- Write the answer in natural student-facing prose. Do not mention internal context "
        "labels or framing such as Section:, Excerpt, Retrieved context, Source block, or "
        "similar prompt metadata.\n"
        "- When multiple syllabus excerpts are provided, synthesize details from all of them "
        "without combining unrelated conditions.\n\n"
        f"Student question:\n{question}\n\n"
        f"{facet_block}"
        "Syllabus context:\n"
        f"{chr(10).join(context_blocks)}\n\n"
        "Source sections:\n"
        f"{section_list}\n\n"
        "Answer the student question now:"
    )


