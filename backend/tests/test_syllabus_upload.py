import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.course_id import assert_valid_course_id, is_valid_course_id
from app.main import app
from app.storage import LocalCourseArtifactStorage


class CourseIdValidationTests(unittest.TestCase):
    def test_accepts_valid_ids(self) -> None:
        self.assertTrue(is_valid_course_id("css360-default"))
        self.assertTrue(is_valid_course_id("css-430-summer-2026-a82f"))

    def test_rejects_invalid_ids(self) -> None:
        for course_id in ("", "CSS360", "-bad", "bad-", "../x", "a/b", "a.b", "a$b", "a[b]"):
            with self.subTest(course_id=course_id):
                self.assertFalse(is_valid_course_id(course_id))
                with self.assertRaises(ValueError):
                    assert_valid_course_id(course_id)


class SyllabusUploadEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.course_data_dir = Path(self._temp_dir.name)
        self.storage = LocalCourseArtifactStorage(root_dir=self.course_data_dir)
        self._storage_patch = patch(
            "app.main.get_course_artifact_storage",
            return_value=self.storage,
        )
        self._storage_patch.start()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self._storage_patch.stop()
        self._temp_dir.cleanup()

    def _upload(
        self,
        course_id: str,
        filename: str,
        content: bytes,
        content_type: str | None = None,
    ):
        files = {
            "syllabus_file": (
                filename,
                io.BytesIO(content),
                content_type or "application/octet-stream",
            )
        }
        return self.client.post(f"/api/courses/{course_id}/syllabus", files=files)

    def test_valid_pdf_upload(self) -> None:
        response = self._upload(
            "css-430-summer-2026-a82f",
            "css430-syllabus.pdf",
            b"%PDF-1.4 sample syllabus",
            "application/pdf",
        )

        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertEqual(body["courseId"], "css-430-summer-2026-a82f")
        self.assertEqual(body["syllabusFileName"], "css430-syllabus.pdf")
        self.assertEqual(body["syllabusType"], "pdf")
        self.assertEqual(body["syllabusStatus"], "uploaded")
        self.assertEqual(body["fileSize"], len(b"%PDF-1.4 sample syllabus"))

        saved = self.course_data_dir / "css-430-summer-2026-a82f" / "original.pdf"
        self.assertTrue(saved.is_file())
        self.assertEqual(saved.read_bytes(), b"%PDF-1.4 sample syllabus")

    def test_valid_txt_upload(self) -> None:
        response = self._upload(
            "course-alpha",
            "notes.txt",
            b"Office hours are on Monday.",
            "text/plain",
        )

        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertEqual(body["syllabusType"], "txt")
        saved = self.course_data_dir / "course-alpha" / "original.txt"
        self.assertTrue(saved.is_file())
        self.assertEqual(saved.read_text(encoding="utf-8"), "Office hours are on Monday.")

    def test_invalid_course_id(self) -> None:
        response = self._upload("Bad_Id", "syllabus.pdf", b"%PDF-1.4", "application/pdf")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid courseId", response.json()["detail"])
        self.assertFalse(any(self.course_data_dir.iterdir()))

    def test_unsupported_extension(self) -> None:
        response = self._upload(
            "course-alpha",
            "syllabus.docx",
            b"not allowed",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn(".pdf and .txt", response.json()["detail"])

    def test_empty_file(self) -> None:
        response = self._upload("course-alpha", "empty.pdf", b"", "application/pdf")
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["detail"].lower())

    def test_file_larger_than_10_mb(self) -> None:
        oversized = b"a" * (10 * 1024 * 1024 + 1)
        response = self._upload("course-alpha", "big.pdf", oversized, "application/pdf")
        self.assertEqual(response.status_code, 413)

    def test_file_saved_under_correct_course_directory(self) -> None:
        self._upload("course-beta", "beta.pdf", b"%PDF-beta", "application/pdf")
        self.assertTrue((self.course_data_dir / "course-beta" / "original.pdf").is_file())
        self.assertFalse((self.course_data_dir / "course-alpha" / "original.pdf").exists())

    def test_partial_file_cleanup_after_failure(self) -> None:
        with patch.object(
            self.storage,
            "save_original_syllabus",
            side_effect=OSError("disk full"),
        ):
            with patch.object(
                self.storage,
                "delete_partial_files",
                wraps=self.storage.delete_partial_files,
            ) as delete_mock:
                response = self._upload(
                    "course-gamma",
                    "gamma.pdf",
                    b"%PDF-gamma",
                    "application/pdf",
                )

        self.assertEqual(response.status_code, 500)
        delete_mock.assert_called_once_with("course-gamma")
        self.assertFalse((self.course_data_dir / "course-gamma" / "original.pdf").exists())


if __name__ == "__main__":
    unittest.main()
