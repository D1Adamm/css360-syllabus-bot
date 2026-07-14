"""Extract and clean readable syllabus text from PDF/TXT uploads."""

from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.syllabus_upload import SyllabusUploadError, ValidatedSyllabusUpload

# Reject near-empty or clearly unusable extraction results.
MIN_EXTRACTED_CHARACTERS = 50

UNUSABLE_TEXT_MESSAGE = (
    "The uploaded syllabus does not contain enough readable text. "
    "Please upload a text-based PDF or TXT file."
)


def clean_extracted_text(text: str) -> str:
    """Normalize extracted syllabus text while preserving paragraph structure."""
    normalized = text.replace("\x00", "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    lines = [line.rstrip() for line in normalized.split("\n")]

    cleaned_lines: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            # Keep at most one blank line between blocks (two newlines).
            if blank_run <= 1:
                cleaned_lines.append("")
            continue

        blank_run = 0
        cleaned_lines.append(line)

    # Trim leading/trailing blank lines created by collapsing.
    while cleaned_lines and cleaned_lines[0] == "":
        cleaned_lines.pop(0)
    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()

    return "\n".join(cleaned_lines)


def validate_extracted_text(text: str) -> str:
    stripped = text.strip()
    if not stripped or len(stripped) < MIN_EXTRACTED_CHARACTERS:
        raise SyllabusUploadError(UNUSABLE_TEXT_MESSAGE, status_code=400)
    return text


def extract_text_from_txt(content: bytes) -> str:
    payload = content
    if payload.startswith(b"\xef\xbb\xbf"):
        payload = payload[3:]

    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SyllabusUploadError(
            "The TXT syllabus must be valid UTF-8 text.",
            status_code=400,
        ) from exc


def extract_text_from_pdf(content: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(content))
    except (PdfReadError, OSError, ValueError) as exc:
        raise SyllabusUploadError(
            "Could not read the uploaded PDF syllabus.",
            status_code=400,
        ) from exc

    page_texts: list[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - pypdf can raise varied errors per page
            raise SyllabusUploadError(
                "Could not extract text from the uploaded PDF syllabus.",
                status_code=400,
            ) from exc

        if page_text.strip():
            page_texts.append(page_text)

    if not page_texts:
        raise SyllabusUploadError(UNUSABLE_TEXT_MESSAGE, status_code=400)

    return "\n\n".join(page_texts)


def extract_clean_syllabus_text(upload: ValidatedSyllabusUpload) -> str:
    if upload.syllabus_type == "txt":
        raw_text = extract_text_from_txt(upload.content)
    elif upload.syllabus_type == "pdf":
        raw_text = extract_text_from_pdf(upload.content)
    else:
        raise SyllabusUploadError(
            f'Unsupported syllabus type "{upload.syllabus_type}".',
            status_code=400,
        )

    cleaned = clean_extracted_text(raw_text)
    return validate_extracted_text(cleaned)
