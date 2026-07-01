import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException

from app.ollama import generate_ollama_completion

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
REPO_ROOT = BACKEND_DIR.parent
SYLLABUS_PATH = REPO_ROOT / "docs" / "syllabus.txt"
INDEX_PATH = BACKEND_DIR / "data" / "syllabus_index.json"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))

CHUNKING_VERSION = "2"
MAX_CHUNK_CHARS = 1800
MAX_PARAGRAPHS_PER_CHUNK = 4
MIN_PARAGRAPHS_PER_CHUNK = 2
DEFAULT_TOP_K = 3

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

INLINE_HEADING_PREFIXES = (
    "Going to be absent?",
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
    r"Going to be absent\?|Contact|Course absence form|Impact of Missing Class|"
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

    flush_chunk()

    if not chunks:
        raise HTTPException(
            status_code=500,
            detail="Syllabus chunking produced no chunks.",
        )

    return chunks


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
        raise HTTPException(
            status_code=502,
            detail=f"Ollama rejected the embedding request: {response.text}",
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


def _adjusted_similarity_score(score: float, section: str) -> float:
    if section in BROAD_SECTIONS:
        return score - BROAD_SECTION_PENALTY
    return score


def _select_diverse_chunks(
    scored_chunks: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    ranked_chunks = sorted(
        scored_chunks,
        key=lambda item: _adjusted_similarity_score(item["score"], item["section"]),
        reverse=True,
    )

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
) -> tuple[str, list[dict[str, Any]]]:
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
    return index_data["embedding_model"], _select_diverse_chunks(scored_chunks, top_k)


def build_rag_prompt(question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    context_blocks: list[str] = []
    source_sections: list[str] = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        section = chunk["section"]
        source_sections.append(section)
        context_blocks.append(
            f"[Excerpt {index} | Section: {section}]\n{chunk['text']}"
        )

    unique_sections = list(dict.fromkeys(source_sections))
    section_list = "\n".join(f"- {section}" for section in unique_sections)

    return (
        "You are answering a CSS360 student question using only the syllabus excerpts below.\n\n"
        "Rules:\n"
        "- Answer only from the supplied syllabus context.\n"
        "- Do not use general knowledge to invent course policies.\n"
        "- If the context does not contain the answer, clearly say that the syllabus does not "
        "provide that information.\n"
        "- Keep the answer concise and student-friendly.\n"
        "- Do not tell the student to email the instructor unless the supplied syllabus context "
        "specifically says to do so.\n"
        "- Do not mention that the answer was generated by AI.\n\n"
        f"Student question:\n{question}\n\n"
        "Syllabus context:\n"
        f"{chr(10).join(context_blocks)}\n\n"
        "Source sections:\n"
        f"{section_list}\n\n"
        "Answer the student question now:"
    )


async def generate_rag_answer(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    _, retrieved_chunks = await retrieve_syllabus_chunks(question=question, top_k=top_k)
    prompt = build_rag_prompt(question, retrieved_chunks)
    generation = await generate_ollama_completion(prompt)

    sources: list[dict[str, Any]] = []
    seen_sections: set[str] = set()
    for chunk in retrieved_chunks:
        section = chunk["section"]
        if section in seen_sections:
            continue
        seen_sections.add(section)
        sources.append({"section": section})

    return {
        "answer": generation["answer"],
        "model": generation["model"],
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
    }
