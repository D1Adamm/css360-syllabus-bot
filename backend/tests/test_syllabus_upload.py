import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.course_id import assert_valid_course_id, is_valid_course_id
from app.main import app
from app.storage import LocalCourseArtifactStorage
from app.syllabus_extract import (
    UNUSABLE_TEXT_MESSAGE,
    clean_extracted_text,
    extract_text_from_pdf,
    extract_text_from_txt,
    validate_extracted_text,
)
from app.syllabus_upload import SyllabusUploadError


SAMPLE_SYLLABUS_TEXT = (
    "Course Overview\n\n"
    "This course covers operating systems concepts including processes, "
    "memory management, file systems, and concurrency.\n\n"
    "Office Hours\n\n"
    "Office hours are held on Mondays and Wednesdays."
)


def build_text_pdf(pages: list[str]) -> bytes:
    """Build a minimal text PDF with one page per string for extraction tests."""
    writer = PdfWriter()
    for page_text in pages:
        # Use a blank page and then inject a simple text content stream.
        writer.add_blank_page(width=612, height=792)
        page = writer.pages[-1]
        # Escape parentheses for PDF text operators.
        escaped = page_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = DecodedStreamObject()
        stream.set_data(
            f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1", errors="replace")
        )
        page[NameObject("/Contents")] = stream
        fonts = DictionaryObject()
        fonts[NameObject("/F1")] = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        resources = DictionaryObject()
        resources[NameObject("/Font")] = fonts
        page[NameObject("/Resources")] = resources

    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def build_empty_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


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


class TextCleaningAndExtractionTests(unittest.TestCase):
    def test_clean_line_endings_and_blank_lines(self) -> None:
        raw = "Heading\r\n\r\n\r\n\r\nParagraph one.   \n\n\n\nParagraph two.\x00"
        cleaned = clean_extracted_text(raw)
        self.assertEqual(
            cleaned,
            "Heading\n\nParagraph one.\n\nParagraph two.",
        )
        self.assertNotIn("\r", cleaned)
        self.assertNotIn("\x00", cleaned)

    def test_valid_txt_extraction_and_bom(self) -> None:
        body = ("\ufeff" + SAMPLE_SYLLABUS_TEXT).encode("utf-8")
        text = extract_text_from_txt(body)
        self.assertTrue(text.startswith("Course Overview"))
        self.assertNotIn("\ufeff", text)

    def test_invalid_txt_encoding(self) -> None:
        with self.assertRaises(SyllabusUploadError) as ctx:
            extract_text_from_txt(b"\xff\xfe\x00\x00not-utf8")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("UTF-8", ctx.exception.message)

    def test_valid_pdf_extraction(self) -> None:
        pdf_bytes = build_text_pdf(
            [
                "Course Overview This course covers operating systems concepts.",
                "Office Hours are held Monday and Wednesday each week.",
            ]
        )
        text = extract_text_from_pdf(pdf_bytes)
        self.assertIn("Course Overview", text)
        self.assertIn("Office Hours", text)

    def test_pdf_with_no_readable_text(self) -> None:
        with self.assertRaises(SyllabusUploadError) as ctx:
            extract_text_from_pdf(build_empty_pdf())
        self.assertEqual(ctx.exception.message, UNUSABLE_TEXT_MESSAGE)

    def test_empty_extracted_text_rejected(self) -> None:
        with self.assertRaises(SyllabusUploadError) as ctx:
            validate_extracted_text("   \n\n  ")
        self.assertEqual(ctx.exception.message, UNUSABLE_TEXT_MESSAGE)


class SyllabusUploadEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        root = Path(self._temp_dir.name)
        self.course_data_dir = root / "course_data"
        self.index_dir = root / "indexes"
        self.storage = LocalCourseArtifactStorage(
            root_dir=self.course_data_dir,
            index_dir=self.index_dir,
        )
        self._storage_patch = patch(
            "app.main.get_course_artifact_storage",
            return_value=self.storage,
        )
        self._storage_patch.start()
        self._embed_patch = patch(
            "app.course_index.get_embedding",
            new=AsyncMock(return_value=[0.01, 0.02, 0.03]),
        )
        self._embed_patch.start()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self._embed_patch.stop()
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
        pdf_bytes = build_text_pdf(
            [
                "Course Overview This course covers operating systems concepts in depth.",
                "Office Hours are held Monday and Wednesday each week for students.",
            ]
        )
        response = self._upload(
            "css-430-summer-2026-a82f",
            "css430-syllabus.pdf",
            pdf_bytes,
            "application/pdf",
        )

        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertEqual(body["courseId"], "css-430-summer-2026-a82f")
        self.assertEqual(body["syllabusFileName"], "css430-syllabus.pdf")
        self.assertEqual(body["syllabusType"], "pdf")
        self.assertEqual(body["syllabusStatus"], "indexed")
        self.assertEqual(body["fileSize"], len(pdf_bytes))
        self.assertGreaterEqual(body["characterCount"], 50)
        self.assertGreaterEqual(body["chunkCount"], 1)

        saved = self.course_data_dir / "css-430-summer-2026-a82f" / "original.pdf"
        self.assertTrue(saved.is_file())
        extracted = self.course_data_dir / "css-430-summer-2026-a82f" / "syllabus.txt"
        self.assertTrue(extracted.is_file())
        self.assertGreaterEqual(len(extracted.read_text(encoding="utf-8")), 50)
        self.assertTrue(self.storage.index_exists("css-430-summer-2026-a82f"))

    def test_valid_txt_upload(self) -> None:
        content = SAMPLE_SYLLABUS_TEXT.encode("utf-8")
        response = self._upload(
            "course-alpha",
            "notes.txt",
            content,
            "text/plain",
        )

        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertEqual(body["syllabusType"], "txt")
        self.assertEqual(body["syllabusStatus"], "indexed")
        self.assertEqual(body["characterCount"], len(SAMPLE_SYLLABUS_TEXT))
        self.assertGreaterEqual(body["chunkCount"], 1)

        original = self.course_data_dir / "course-alpha" / "original.txt"
        extracted = self.course_data_dir / "course-alpha" / "syllabus.txt"
        self.assertTrue(original.is_file())
        self.assertEqual(original.read_bytes(), content)
        self.assertTrue(extracted.is_file())
        self.assertEqual(extracted.read_text(encoding="utf-8"), SAMPLE_SYLLABUS_TEXT)
        self.assertTrue(self.storage.index_exists("course-alpha"))

    def test_utf8_bom_txt_upload(self) -> None:
        content = ("\ufeff" + SAMPLE_SYLLABUS_TEXT).encode("utf-8")
        response = self._upload("course-bom", "bom.txt", content, "text/plain")
        self.assertEqual(response.status_code, 201)
        extracted = (
            self.course_data_dir / "course-bom" / "syllabus.txt"
        ).read_text(encoding="utf-8")
        self.assertTrue(extracted.startswith("Course Overview"))
        self.assertNotIn("\ufeff", extracted)

    def test_invalid_txt_encoding_upload(self) -> None:
        response = self._upload(
            "course-bad-encoding",
            "bad.txt",
            b"\xff\xfe\x00\x00not-utf8-text-content-here!!!!!!",
            "text/plain",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("UTF-8", response.json()["detail"])
        self.assertFalse((self.course_data_dir / "course-bad-encoding").exists())

    def test_pdf_with_no_readable_text_upload(self) -> None:
        response = self._upload(
            "course-empty-pdf",
            "empty.pdf",
            build_empty_pdf(),
            "application/pdf",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], UNUSABLE_TEXT_MESSAGE)
        self.assertFalse((self.course_data_dir / "course-empty-pdf").exists())

    def test_empty_extracted_text_upload(self) -> None:
        response = self._upload(
            "course-short",
            "short.txt",
            b"too short",
            "text/plain",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], UNUSABLE_TEXT_MESSAGE)
        self.assertFalse((self.course_data_dir / "course-short").exists())

    def test_invalid_course_id(self) -> None:
        response = self._upload(
            "Bad_Id",
            "syllabus.pdf",
            build_text_pdf([SAMPLE_SYLLABUS_TEXT]),
            "application/pdf",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid courseId", response.json()["detail"])
        self.assertFalse(self.course_data_dir.exists() and any(self.course_data_dir.iterdir()))

    def test_unsupported_extension(self) -> None:
        response = self._upload(
            "course-alpha",
            "syllabus.docx",
            b"not allowed but long enough to avoid other checks!!!!!!!",
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
        content = SAMPLE_SYLLABUS_TEXT.encode("utf-8")
        self._upload("course-beta", "beta.txt", content, "text/plain")
        self.assertTrue((self.course_data_dir / "course-beta" / "original.txt").is_file())
        self.assertTrue((self.course_data_dir / "course-beta" / "syllabus.txt").is_file())
        self.assertFalse((self.course_data_dir / "course-alpha" / "original.txt").exists())

    def test_syllabus_txt_saved_correctly(self) -> None:
        messy = "Title\r\n\r\n\r\n\r\nBody paragraph with enough characters for validation."
        response = self._upload(
            "course-clean",
            "messy.txt",
            messy.encode("utf-8"),
            "text/plain",
        )
        self.assertEqual(response.status_code, 201)
        saved = (self.course_data_dir / "course-clean" / "syllabus.txt").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            saved,
            "Title\n\nBody paragraph with enough characters for validation.",
        )

    def test_get_syllabus_text_endpoint(self) -> None:
        content = SAMPLE_SYLLABUS_TEXT.encode("utf-8")
        upload = self._upload("course-get", "notes.txt", content, "text/plain")
        self.assertEqual(upload.status_code, 201)

        response = self.client.get("/api/courses/course-get/syllabus/text")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["courseId"], "course-get")
        self.assertEqual(body["text"], SAMPLE_SYLLABUS_TEXT)
        self.assertEqual(body["characterCount"], len(SAMPLE_SYLLABUS_TEXT))

    def test_missing_syllabus_text_returns_404(self) -> None:
        response = self.client.get("/api/courses/missing-course/syllabus/text")
        self.assertEqual(response.status_code, 404)

    def test_partial_file_cleanup_after_original_save_failure(self) -> None:
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
                    "gamma.txt",
                    SAMPLE_SYLLABUS_TEXT.encode("utf-8"),
                    "text/plain",
                )

        self.assertEqual(response.status_code, 500)
        delete_mock.assert_called_once_with("course-gamma")
        self.assertFalse((self.course_data_dir / "course-gamma").exists())

    def test_cleanup_after_extraction_failure(self) -> None:
        with patch(
            "app.main.extract_clean_syllabus_text",
            side_effect=SyllabusUploadError(UNUSABLE_TEXT_MESSAGE, status_code=400),
        ):
            response = self._upload(
                "course-extract-fail",
                "notes.txt",
                SAMPLE_SYLLABUS_TEXT.encode("utf-8"),
                "text/plain",
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], UNUSABLE_TEXT_MESSAGE)
        self.assertFalse((self.course_data_dir / "course-extract-fail").exists())


if __name__ == "__main__":
    unittest.main()
