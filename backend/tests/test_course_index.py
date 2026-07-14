import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.storage import LocalCourseArtifactStorage


SAMPLE_SYLLABUS_TEXT = (
    "Course Information\n\n"
    "This course covers systems concepts including scheduling, memory, and concurrency.\n\n"
    "Attendance\n\n"
    "Students who miss class should submit the absence form at least one hour before class.\n\n"
    "Office Hours\n\n"
    "Office hours are held Mondays and Wednesdays for clarifying course policy questions."
)


class CourseIndexEndpointTests(unittest.TestCase):
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
            new=AsyncMock(return_value=[0.1, -0.2, 0.3]),
        )
        self._embed_patch.start()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self._embed_patch.stop()
        self._storage_patch.stop()
        self._temp_dir.cleanup()

    def _upload(self, course_id: str, filename: str = "syllabus.txt") -> any:
        files = {
            "syllabus_file": (
                filename,
                io.BytesIO(SAMPLE_SYLLABUS_TEXT.encode("utf-8")),
                "text/plain",
            )
        }
        return self.client.post(f"/api/courses/{course_id}/syllabus", files=files)

    def test_embedding_generated_for_every_chunk(self) -> None:
        response = self._upload("course-embed-one")
        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertEqual(body["syllabusStatus"], "indexed")
        self.assertGreaterEqual(body["chunkCount"], 1)

        index = self.storage.load_index("course-embed-one")
        self.assertIsNotNone(index)
        assert index is not None
        self.assertEqual(index["chunkCount"], len(index["chunks"]))
        for chunk in index["chunks"]:
            self.assertEqual(chunk["embedding"], [0.1, -0.2, 0.3])
            self.assertIn("chunkId", chunk)
            self.assertIn("sectionTitle", chunk)
            self.assertIn("order", chunk)

    def test_separate_index_files_for_two_course_ids(self) -> None:
        first = self._upload("course-alpha")
        second = self._upload("course-beta")
        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)

        self.assertTrue((self.index_dir / "course-alpha.json").is_file())
        self.assertTrue((self.index_dir / "course-beta.json").is_file())
        self.assertNotEqual(
            (self.index_dir / "course-alpha.json").read_text(encoding="utf-8"),
            "",
        )

    def test_failed_embedding_generation_cleans_partial_index(self) -> None:
        with patch(
            "app.course_index.get_embedding",
            new=AsyncMock(
                side_effect=HTTPException(
                    status_code=503,
                    detail="Ollama is unavailable for embeddings.",
                )
            ),
        ):
            response = self._upload("course-embed-fail")

        self.assertEqual(response.status_code, 503)
        self.assertIn("unavailable", response.json()["detail"].lower())
        self.assertFalse(self.storage.index_exists("course-embed-fail"))
        self.assertFalse((self.course_data_dir / "course-embed-fail").exists())

    def test_index_json_format(self) -> None:
        response = self._upload("course-format", filename="Syllabus 350.txt")
        self.assertEqual(response.status_code, 201)
        index = self.storage.load_index("course-format")
        assert index is not None
        self.assertEqual(index["courseId"], "course-format")
        self.assertEqual(index["sourceFile"], "Syllabus 350.txt")
        self.assertEqual(index["embeddingModel"], "nomic-embed-text")
        self.assertIn("createdAt", index)
        self.assertEqual(index["chunkCount"], len(index["chunks"]))
        first = index["chunks"][0]
        self.assertEqual(
            set(first.keys()),
            {"chunkId", "sectionTitle", "text", "order", "embedding"},
        )

    def test_get_chunks_endpoint_omits_embeddings(self) -> None:
        upload = self._upload("course-chunks")
        self.assertEqual(upload.status_code, 201)

        response = self.client.get("/api/courses/course-chunks/chunks")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["courseId"], "course-chunks")
        self.assertGreaterEqual(body["chunkCount"], 1)
        self.assertEqual(body["chunkCount"], len(body["chunks"]))
        for chunk in body["chunks"]:
            self.assertNotIn("embedding", chunk)
            self.assertIn("chunkId", chunk)
            self.assertIn("sectionTitle", chunk)
            self.assertIn("text", chunk)
            self.assertIn("order", chunk)

    def test_missing_index_returns_404(self) -> None:
        response = self.client.get("/api/courses/missing-course/chunks")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
