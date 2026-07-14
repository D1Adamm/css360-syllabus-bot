"""Syllabus upload validation helpers (no text extraction yet)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from fastapi import UploadFile

from app.course_id import assert_valid_course_id

MAX_SYLLABUS_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = frozenset({"pdf", "txt"})
ALLOWED_CONTENT_TYPES = {
    "pdf": frozenset(
        {
            "application/pdf",
            "application/x-pdf",
            "binary/octet-stream",
            "application/octet-stream",
        }
    ),
    "txt": frozenset(
        {
            "text/plain",
            "text/txt",
            "application/octet-stream",
            "binary/octet-stream",
        }
    ),
}
SUSPICIOUS_FILENAME_PATTERN = re.compile(r"[\x00-\x1f\\/]|^\.\.?$|\.\.")


class SyllabusUploadError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class ValidatedSyllabusUpload:
    course_id: str
    original_filename: str
    syllabus_type: str
    content_type: str | None
    content: bytes

    @property
    def file_size(self) -> int:
        return len(self.content)


def _extension_from_filename(filename: str | None) -> str | None:
    if not filename or "." not in filename:
        return None
    return filename.rsplit(".", 1)[-1].lower()


def validate_upload_filename(filename: str | None) -> str:
    if filename is None or filename.strip() == "":
        raise SyllabusUploadError("A syllabus_file filename is required.")

    cleaned = os.path.basename(filename.strip())
    if cleaned != filename.strip() or SUSPICIOUS_FILENAME_PATTERN.search(cleaned):
        raise SyllabusUploadError("The syllabus filename is not allowed.")

    extension = _extension_from_filename(cleaned)
    if extension not in ALLOWED_EXTENSIONS:
        raise SyllabusUploadError("Only .pdf and .txt syllabus files are supported.")

    return cleaned


def validate_content_type(syllabus_type: str, content_type: str | None) -> None:
    if content_type is None or content_type.strip() == "":
        return

    normalized = content_type.split(";", 1)[0].strip().lower()
    allowed = ALLOWED_CONTENT_TYPES[syllabus_type]
    if normalized not in allowed:
        raise SyllabusUploadError(
            f'Unsupported content type "{content_type}" for .{syllabus_type} files.'
        )


async def read_upload_with_size_limit(upload: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total = 0

    while True:
        chunk = await upload.read(1024 * 64)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_SYLLABUS_BYTES:
            raise SyllabusUploadError(
                f"Syllabus file exceeds the {MAX_SYLLABUS_BYTES} byte limit.",
                status_code=413,
            )
        chunks.append(chunk)

    content = b"".join(chunks)
    if len(content) == 0:
        raise SyllabusUploadError("Syllabus file must not be empty.")
    return content


async def validate_syllabus_upload(
    course_id: str,
    syllabus_file: UploadFile,
) -> ValidatedSyllabusUpload:
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise SyllabusUploadError(str(exc), status_code=400) from exc

    original_filename = validate_upload_filename(syllabus_file.filename)
    syllabus_type = _extension_from_filename(original_filename)
    assert syllabus_type is not None  # guarded by validate_upload_filename
    validate_content_type(syllabus_type, syllabus_file.content_type)

    content = await read_upload_with_size_limit(syllabus_file)

    return ValidatedSyllabusUpload(
        course_id=safe_course_id,
        original_filename=original_filename,
        syllabus_type=syllabus_type,
        content_type=syllabus_file.content_type,
        content=content,
    )
