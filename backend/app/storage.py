"""Course artifact storage abstractions.

LocalCourseArtifactStorage is the only concrete backend for this step.
A later GCS implementation can satisfy the same CourseArtifactStorage interface.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

from app.course_id import assert_valid_course_id

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COURSE_DATA_DIR = "course_data"
ALLOWED_SYLLABUS_EXTENSIONS = frozenset({"pdf", "txt"})
EXTRACTED_TEXT_FILENAME = "syllabus.txt"


class CourseArtifactStorage(ABC):
    """Interface for course-local syllabus artifacts (local disk today, GCS later)."""

    @abstractmethod
    def ensure_course_dir(self, course_id: str) -> Path:
        raise NotImplementedError

    @abstractmethod
    def original_syllabus_path(self, course_id: str, syllabus_type: str) -> Path:
        raise NotImplementedError

    @abstractmethod
    def save_original_syllabus(
        self,
        course_id: str,
        syllabus_type: str,
        content: bytes,
    ) -> Path:
        raise NotImplementedError

    @abstractmethod
    def extracted_text_path(self, course_id: str) -> Path:
        raise NotImplementedError

    @abstractmethod
    def save_extracted_text(self, course_id: str, text: str) -> Path:
        raise NotImplementedError

    @abstractmethod
    def load_extracted_text(self, course_id: str) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def syllabus_exists(self, course_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def delete_partial_files(self, course_id: str) -> None:
        raise NotImplementedError


class LocalCourseArtifactStorage(CourseArtifactStorage):
    def __init__(self, root_dir: Path | None = None) -> None:
        if root_dir is None:
            configured = os.getenv("COURSE_DATA_DIR", DEFAULT_COURSE_DATA_DIR)
            path = Path(configured)
            root_dir = path if path.is_absolute() else BACKEND_ROOT / path
        self.root_dir = root_dir

    def ensure_course_dir(self, course_id: str) -> Path:
        safe_course_id = assert_valid_course_id(course_id)
        course_dir = self.root_dir / safe_course_id
        course_dir.mkdir(parents=True, exist_ok=True)
        return course_dir

    def original_syllabus_path(self, course_id: str, syllabus_type: str) -> Path:
        safe_course_id = assert_valid_course_id(course_id)
        normalized_type = syllabus_type.lower().lstrip(".")
        if normalized_type not in ALLOWED_SYLLABUS_EXTENSIONS:
            raise ValueError(f"Unsupported syllabus type: {syllabus_type}")
        return self.root_dir / safe_course_id / f"original.{normalized_type}"

    def save_original_syllabus(
        self,
        course_id: str,
        syllabus_type: str,
        content: bytes,
    ) -> Path:
        self.ensure_course_dir(course_id)
        destination = self.original_syllabus_path(course_id, syllabus_type)
        destination.write_bytes(content)
        return destination

    def extracted_text_path(self, course_id: str) -> Path:
        safe_course_id = assert_valid_course_id(course_id)
        return self.root_dir / safe_course_id / EXTRACTED_TEXT_FILENAME

    def save_extracted_text(self, course_id: str, text: str) -> Path:
        self.ensure_course_dir(course_id)
        destination = self.extracted_text_path(course_id)
        destination.write_text(text, encoding="utf-8")
        return destination

    def load_extracted_text(self, course_id: str) -> str | None:
        path = self.extracted_text_path(course_id)
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def syllabus_exists(self, course_id: str) -> bool:
        safe_course_id = assert_valid_course_id(course_id)
        course_dir = self.root_dir / safe_course_id
        if not course_dir.is_dir():
            return False
        return any(
            (course_dir / f"original.{extension}").is_file()
            for extension in ALLOWED_SYLLABUS_EXTENSIONS
        )

    def delete_partial_files(self, course_id: str) -> None:
        safe_course_id = assert_valid_course_id(course_id)
        course_dir = self.root_dir / safe_course_id
        if not course_dir.is_dir():
            return

        for extension in ALLOWED_SYLLABUS_EXTENSIONS:
            path = course_dir / f"original.{extension}"
            if path.exists():
                path.unlink()

        extracted = course_dir / EXTRACTED_TEXT_FILENAME
        if extracted.exists():
            extracted.unlink()

        try:
            course_dir.rmdir()
        except OSError:
            # Leave the directory if unrelated files remain.
            pass


def get_course_artifact_storage() -> CourseArtifactStorage:
    backend = os.getenv("COURSE_STORAGE_BACKEND", "local").strip().lower()
    if backend != "local":
        raise RuntimeError(
            f'Unsupported COURSE_STORAGE_BACKEND "{backend}". Only "local" is implemented.'
        )
    return LocalCourseArtifactStorage()
