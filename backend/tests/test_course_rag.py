import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.retrieval_diversity import MAX_CHUNK_CONTEXT_CHARS, MAX_TOTAL_CONTEXT_CHARS
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

    def test_multi_part_question_returns_chunks_from_multiple_sections(self) -> None:
        multi_id = "css-450-multi-section"
        self.storage.save_index(
            multi_id,
            _index(
                multi_id,
                [
                    _chunk(
                        "grade-1",
                        "Grade Questions",
                        "Grade questions must be discussed privately.",
                        [1.0, 0.0, 0.0, 0.0],
                        order=0,
                    ),
                    _chunk(
                        "contact-1",
                        "Contact",
                        "Use email or Discord for course communication.",
                        [0.85, 0.15, 0.0, 0.0],
                        order=1,
                    ),
                    _chunk(
                        "late-1",
                        "Late Policy",
                        "Late work loses credit unless an extension is approved.",
                        [0.8, 0.0, 0.2, 0.0],
                        order=2,
                    ),
                    _chunk(
                        "makeup-1",
                        "Impact of Missing Class",
                        "Missed in-class work generally cannot be made up.",
                        [0.75, 0.0, 0.0, 0.25],
                        order=3,
                    ),
                    _chunk(
                        "ai-1",
                        "Use of AI Tools",
                        "AI tools require citation when used on assignments.",
                        [0.1, 0.0, 0.0, 0.0],
                        order=4,
                    ),
                ],
            ),
        )

        with patch(
            "app.course_rag.get_embedding",
            new=AsyncMock(return_value=[1.0, 0.0, 0.0, 0.0]),
        ):
            response = self.client.post(
                "/rag/generate",
                json={
                    "courseId": multi_id,
                    "question": (
                        "How should I discuss grades, which communication channels "
                        "can I use, what is the late and extension policy, and can I "
                        "make up missed work?"
                    ),
                    "topK": 4,
                },
            )

        self.assertEqual(response.status_code, 200)
        sources = response.json()["sources"]
        sections = {source["sectionTitle"] for source in sources}
        self.assertGreaterEqual(len(sections), 3)
        self.assertTrue({"Grade Questions", "Contact", "Late Policy"} & sections)

    def test_near_duplicate_chunks_are_filtered_in_retrieval(self) -> None:
        dup_id = "css-450-dup-chunks"
        shared = (
            "Late submissions lose ten percent per day and may not be accepted "
            "after three days without an approved extension request from the instructor."
        )
        self.storage.save_index(
            dup_id,
            _index(
                dup_id,
                [
                    _chunk("dup-1", "Late Policy", shared, [1.0, 0.0], order=0),
                    _chunk(
                        "dup-2",
                        "Late Policy",
                        shared.replace("instructor", "faculty"),
                        [0.99, 0.01],
                        order=1,
                    ),
                    _chunk(
                        "makeup-1",
                        "Makeup Policy",
                        "Makeup exams require documentation before the original exam.",
                        [0.7, 0.3],
                        order=2,
                    ),
                ],
            ),
        )

        with patch(
            "app.course_rag.get_embedding",
            new=AsyncMock(return_value=[1.0, 0.0]),
        ):
            response = self.client.post(
                "/rag/generate",
                json={
                    "courseId": dup_id,
                    "question": "What is the late policy and makeup rule?",
                    "topK": 3,
                },
            )

        self.assertEqual(response.status_code, 200)
        chunk_ids = [source["chunkId"] for source in response.json()["sources"]]
        self.assertIn("dup-1", chunk_ids)
        self.assertNotIn("dup-2", chunk_ids)
        self.assertIn("makeup-1", chunk_ids)

    def _save_long_multi_section_index(self, course_id: str) -> None:
        long_body = "Detailed policy sentence describing rules and penalties. " * 60
        sections = [
            ("grade-1", "Grade Questions", [1.0, 0.0, 0.0, 0.0]),
            ("contact-1", "Contact", [0.9, 0.1, 0.0, 0.0]),
            ("late-1", "Late Policy", [0.85, 0.0, 0.15, 0.0]),
            ("makeup-1", "Makeup Policy", [0.8, 0.0, 0.0, 0.2]),
            ("ai-1", "Use of AI Tools", [0.75, 0.05, 0.05, 0.15]),
            ("office-1", "Office Hours", [0.7, 0.1, 0.1, 0.1]),
        ]
        self.storage.save_index(
            course_id,
            _index(
                course_id,
                [
                    _chunk(
                        chunk_id,
                        section,
                        f"{section}\n{long_body}",
                        embedding,
                        order=order,
                    )
                    for order, (chunk_id, section, embedding) in enumerate(sections)
                ],
            ),
        )

    def test_long_multi_part_question_is_bounded_and_diverse(self) -> None:
        course_id = "css-460-long-context"
        self._save_long_multi_section_index(course_id)

        with patch(
            "app.course_rag.get_embedding",
            new=AsyncMock(return_value=[1.0, 0.0, 0.0, 0.0]),
        ):
            response = self.client.post(
                "/rag/generate",
                json={
                    "courseId": course_id,
                    "question": (
                        "How are grade discussions handled, which communication channels "
                        "should I use, what is the late policy, how do extensions work, "
                        "and can I make up missed in-class work?"
                    ),
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        sources = body["sources"]

        # Default topK is 4, so no more than four chunks reach the model.
        self.assertLessEqual(len(sources), 4)
        # Retrieval stays diverse across sections.
        self.assertGreaterEqual(len({source["sectionTitle"] for source in sources}), 3)

        # Total context stays bounded and each chunk is individually capped.
        total_chars = sum(len(source["text"]) for source in sources)
        self.assertLessEqual(total_chars, MAX_TOTAL_CONTEXT_CHARS)
        for source in sources:
            self.assertLessEqual(len(source["text"]), MAX_CHUNK_CONTEXT_CHARS)

    def test_sources_match_chunks_actually_sent_to_the_model(self) -> None:
        course_id = "css-460-source-match"
        self._save_long_multi_section_index(course_id)

        captured: dict[str, str] = {}

        async def capture_prompt(prompt: str, **_kwargs):
            captured["prompt"] = prompt
            return {"answer": "Bounded RAG answer", "model": "llama3.2:3b"}

        with patch(
            "app.course_rag.get_embedding",
            new=AsyncMock(return_value=[1.0, 0.0, 0.0, 0.0]),
        ), patch(
            "app.course_rag.generate_ollama_completion",
            new=AsyncMock(side_effect=capture_prompt),
        ):
            response = self.client.post(
                "/rag/generate",
                json={
                    "courseId": course_id,
                    "question": "What are the grade, late, and makeup policies?",
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        prompt = captured["prompt"]

        # Every returned source text is present verbatim in the prompt context.
        for source in body["sources"]:
            self.assertIn(source["text"], prompt)
            self.assertIn(source["sectionTitle"], prompt)

        # Sources and retrievedChunks describe the same bounded set.
        self.assertEqual(
            [source["chunkId"] for source in body["sources"]],
            [chunk["chunkId"] for chunk in body["retrievedChunks"]],
        )
        for source, chunk in zip(body["sources"], body["retrievedChunks"]):
            self.assertEqual(source["text"], chunk["text"])

    def test_single_topic_retrieval_still_returns_best_section(self) -> None:
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
        self.assertGreaterEqual(len(sources), 1)
        self.assertEqual(sources[0]["sectionTitle"], "CSS 430 Late Policy")
        self.assertEqual(sources[0]["chunkId"], "css430-late-1")


if __name__ == "__main__":
    unittest.main()
