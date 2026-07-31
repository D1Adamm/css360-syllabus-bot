import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.reindex_course import main as reindex_main
from app.storage import LocalCourseArtifactStorage
from app.syllabus_chunking import INDEX_VERSION


SAMPLE = """Software Engineering (Fall 2025)

Jump to:navigation, search
Contents
\t•\t1
\t•\tLate Policy
Instructor: Ada Example
Contact: Reach instructors via Canvas for grade questions.
Late Policy With respect to projects, you may use one 48-hour extension per quarter via Canvas.
Office Hours
Hours are posted on Canvas.
"""


class ReindexCourseCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        root = Path(self._temp.name)
        self.course_data = root / "course_data"
        self.index_dir = root / "indexes"
        self.storage = LocalCourseArtifactStorage(
            root_dir=self.course_data,
            index_dir=self.index_dir,
        )
        self.course_id = "css-360-winter-2026-test"
        self.storage.save_extracted_text(self.course_id, SAMPLE)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_dry_run_does_not_write_index(self) -> None:
        with patch("app.reindex_course.get_course_artifact_storage", return_value=self.storage):
            code = reindex_main(["--course-id", self.course_id, "--dry-run"])
        self.assertEqual(code, 0)
        self.assertFalse(self.storage.index_exists(self.course_id))

    def test_reindex_writes_versioned_index(self) -> None:
        with (
            patch("app.reindex_course.get_course_artifact_storage", return_value=self.storage),
            patch(
                "app.course_index.get_embedding",
                new=AsyncMock(return_value=[0.1, 0.2, 0.3]),
            ),
        ):
            code = reindex_main(["--course-id", self.course_id])
        self.assertEqual(code, 0)
        index = self.storage.load_index(self.course_id)
        assert index is not None
        self.assertEqual(index["indexVersion"], INDEX_VERSION)
        self.assertGreaterEqual(index["chunkCount"], 2)
        sections = {chunk["sectionTitle"] for chunk in index["chunks"]}
        self.assertIn("Late Policy", sections)
        self.assertIn("Contact", sections)
        # Original syllabus preserved.
        self.assertTrue(self.storage.extracted_text_path(self.course_id).is_file())


if __name__ == "__main__":
    unittest.main()
