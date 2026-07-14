"""Course artifact storage abstractions.

LocalCourseArtifactStorage is the only concrete backend for this step.
A later GCS implementation can satisfy the same CourseArtifactStorage interface.
"""

from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.course_id import assert_valid_course_id

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COURSE_DATA_DIR = "course_data"
DEFAULT_INDEX_DIR = BACKEND_ROOT / "data" / "indexes"
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

    @abstractmethod
    def index_path(self, course_id: str) -> Path:
        raise NotImplementedError

    @abstractmethod
    def save_index(self, course_id: str, index_data: dict[str, Any]) -> Path:
        raise NotImplementedError

    @abstractmethod
    def load_index(self, course_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def index_exists(self, course_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def remove_index(self, course_id: str) -> None:
        raise NotImplementedError


class LocalCourseArtifactStorage(CourseArtifactStorage):
    def __init__(
        self,
        root_dir: Path | None = None,
        index_dir: Path | None = None,
    ) -> None:
        if root_dir is None:
            configured = os.getenv("COURSE_DATA_DIR", DEFAULT_COURSE_DATA_DIR)
            path = Path(configured)
            root_dir = path if path.is_absolute() else BACKEND_ROOT / path
        self.root_dir = root_dir

        if index_dir is None:
            configured_index = os.getenv("COURSE_INDEX_DIR", str(DEFAULT_INDEX_DIR))
            index_path = Path(configured_index)
            index_dir = (
                index_path if index_path.is_absolute() else BACKEND_ROOT / index_path
            )
        self.index_dir = index_dir

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
        if course_dir.is_dir():
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

        self.remove_index(safe_course_id)

    def index_path(self, course_id: str) -> Path:
        safe_course_id = assert_valid_course_id(course_id)
        return self.index_dir / f"{safe_course_id}.json"

    def save_index(self, course_id: str, index_data: dict[str, Any]) -> Path:
        destination = self.index_path(course_id)
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: temp file in the same directory, then rename.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(index_data, handle, ensure_ascii=False)
            temp_name = handle.name

        os.replace(temp_name, destination)
        return destination

    def load_index(self, course_id: str) -> dict[str, Any] | None:
        path = self.index_path(course_id)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def index_exists(self, course_id: str) -> bool:
        return self.index_path(course_id).is_file()

    def remove_index(self, course_id: str) -> None:
        path = self.index_path(course_id)
        if path.exists():
            path.unlink()


def get_course_artifact_storage() -> CourseArtifactStorage:
    backend = os.getenv("COURSE_STORAGE_BACKEND", "local").strip().lower()
    if backend != "local":
        raise RuntimeError(
            f'Unsupported COURSE_STORAGE_BACKEND "{backend}". Only "local" is implemented.'
        )
    return LocalCourseArtifactStorage()
