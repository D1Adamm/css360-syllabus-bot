import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.storage import LocalCourseArtifactStorage


def _chunk(
    chunk_id: str,
    section_title: str,
    text: str,
    embedding: list[float],
    order: int = 0,
) -> dict:
    return {
        "chunkId": chunk_id,
        "sectionTitle": section_title,
        "text": text,
        "order": order,
        "embedding": embedding,
    }


def _index(course_id: str, chunks: list[dict]) -> dict:
    return {
        "courseId": course_id,
        "embeddingModel": "nomic-embed-text",
        "chunkCount": len(chunks),
        "chunks": chunks,
    }


class CourseSpecificRagTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        root = Path(self._temp_dir.name)
        self.storage = LocalCourseArtifactStorage(
            root_dir=root / "course_data",
            index_dir=root / "indexes",
        )
        self._storage_patch = patch(
            "app.course_rag.get_course_artifact_storage",
            return_value=self.storage,
        )
        self._storage_patch.start()
        self._embed_patch = patch(
            "app.course_rag.get_embedding",
            new=AsyncMock(return_value=[1.0, 0.0, 0.0]),
        )
        self._embed_patch.start()
        self._ollama_patch = patch(
            "app.course_rag.generate_ollama_completion",
            new=AsyncMock(
                return_value={
                    "answer": "Course-scoped RAG answer",
                    "model": "llama3.2:3b",
                }
            ),
        )
        self._ollama_patch.start()
        self._base_patch = patch(
            "app.main.generate_base_model_response",
            new=AsyncMock(
                return_value={
                    "answer": "Base answer without syllabus",
                    "model": "llama3.2:3b",
                    "response_type": "base",
                }
            ),
        )
        self._base_patch.start()
        self.client = TestClient(app)

        self.css430_id = "css-430-summer-2026-ibce"
        self.other_id = "css-360-winter-2026-demo"
        self.storage.save_index(
            self.css430_id,
            _index(
                self.css430_id,
                [
                    _chunk(
                        "css430-late-1",
                        "CSS 430 Late Policy",
                        "CSS 430 late work may be submitted within 24 hours for half credit.",
                        [1.0, 0.0, 0.0],
                        order=0,
                    ),
                    _chunk(
                        "css430-office-1",
                        "CSS 430 Office Hours",
                        "CSS 430 office hours are Tuesdays at 2pm.",
                        [0.0, 1.0, 0.0],
                        order=1,
                    ),
                ],
            ),
        )
        self.storage.save_index(
            self.other_id,
            _index(
                self.other_id,
                [
                    _chunk(
                        "css360-late-1",
                        "CSS 360 Late Policy",
                        "CSS 360 allows one 48-hour late token per quarter.",
                        [1.0, 0.0, 0.0],
                        order=0,
                    ),
                    _chunk(
                        "css360-office-1",
                        "CSS 360 Office Hours",
                        "CSS 360 office hours are Wednesdays at 10am.",
                        [0.0, 1.0, 0.0],
                        order=1,
                    ),
                ],
            ),
        )

    def tearDown(self) -> None:
        self._base_patch.stop()
        self._ollama_patch.stop()
        self._embed_patch.stop()
        self._storage_patch.stop()
        self._temp_dir.cleanup()

    def test_base_request_includes_course_id(self) -> None:
        response = self.client.post(
            "/base-model/generate",
            json={
                "courseId": self.css430_id,
                "question": "Can I submit late work?",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["courseId"], self.css430_id)
        self.assertEqual(body["answer"], "Base answer without syllabus")
        self.assertEqual(body["responseType"], "base")

    def test_rag_request_includes_course_id(self) -> None:
        response = self.client.post(
            "/rag/generate",
            json={
                "courseId": self.css430_id,
                "question": "Can I submit late work?",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["courseId"], self.css430_id)
        self.assertEqual(body["answer"], "Course-scoped RAG answer")
        self.assertEqual(body["responseType"], "rag")
        self.assertGreaterEqual(len(body["sources"]), 1)

    def test_css_430_loads_css_430_index(self) -> None:
        with patch(
            "app.course_rag.get_embedding",
            new=AsyncMock(return_value=[1.0, 0.0, 0.0]),
        ):
            response = self.client.post(
                "/rag/generate",
                json={
                    "courseId": self.css430_id,
                    "question": "What is the late policy?",
                },
            )
        self.assertEqual(response.status_code, 200)
        sources = response.json()["sources"]
        self.assertTrue(all(source["chunkId"].startswith("css430-") for source in sources))
        self.assertIn("CSS 430 Late Policy", [source["sectionTitle"] for source in sources])

    def test_other_course_loads_its_own_index(self) -> None:
        response = self.client.post(
            "/rag/generate",
            json={
                "courseId": self.other_id,
                "question": "What is the late policy?",
            },
        )
        self.assertEqual(response.status_code, 200)
        sources = response.json()["sources"]
        self.assertTrue(all(source["chunkId"].startswith("css360-") for source in sources))
        self.assertIn("CSS 360 Late Policy", [source["sectionTitle"] for source in sources])

    def test_two_courses_cannot_retrieve_each_others_chunks(self) -> None:
        css430 = self.client.post(
            "/rag/generate",
            json={"courseId": self.css430_id, "question": "late policy"},
        ).json()
        other = self.client.post(
            "/rag/generate",
            json={"courseId": self.other_id, "question": "late policy"},
        ).json()

        css430_chunk_ids = {source["chunkId"] for source in css430["sources"]}
        other_chunk_ids = {source["chunkId"] for source in other["sources"]}
        self.assertTrue(css430_chunk_ids.isdisjoint(other_chunk_ids))
        self.assertTrue(
            all("CSS 430" in source["sectionTitle"] for source in css430["sources"])
        )
        self.assertTrue(
            all("CSS 360" in source["sectionTitle"] for source in other["sources"])
        )
        self.assertFalse(
            any("CSS 360" in source["text"] for source in css430["sources"])
        )
        self.assertFalse(
            any("CSS 430" in source["text"] for source in other["sources"])
        )

    def test_missing_index_returns_404(self) -> None:
        response = self.client.post(
            "/rag/generate",
            json={
                "courseId": "course-with-no-index",
                "question": "Can I submit late work?",
            },
        )
        self.assertEqual(response.status_code, 404)
        self.assertIn("No syllabus index found", response.json()["detail"])

    def test_invalid_course_id_is_rejected(self) -> None:
        for path in ("/base-model/generate", "/rag/generate"):
            with self.subTest(path=path):
                response = self.client.post(
                    path,
                    json={
                        "courseId": "../evil",
                        "question": "Can I submit late work?",
                    },
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("Invalid courseId", response.json()["detail"])

    def test_returned_sources_come_from_selected_course(self) -> None:
        response = self.client.post(
            "/rag/generate",
            json={
                "courseId": self.css430_id,
                "question": "office hours",
                "topK": 2,
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["courseId"], self.css430_id)
        for source in body["sources"]:
            self.assertTrue(source["chunkId"].startswith("css430-"))
            self.assertIn("score", source)
            self.assertIn("text", source)
            self.assertIn("sectionTitle", source)
        for chunk in body["retrievedChunks"]:
            self.assertTrue(chunk["chunkId"].startswith("css430-"))


if __name__ == "__main__":
    unittest.main()
