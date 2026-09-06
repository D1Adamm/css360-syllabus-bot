import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.db import DatabaseConfigurationError
from app.reindex_course import main as reindex_main
from app.storage import LocalCourseArtifactStorage
from app.syllabus_chunking import INDEX_VERSION


@contextmanager
def _fake_connection(**kwargs):
    yield object()


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


class CourseRecordSyncTests(unittest.TestCase):
    """The record in PostgreSQL follows the index on disk.

    The CSS 360 discrepancy: the record said 92 chunks (written at upload) while
    the rebuilt index held 163, because the rebuild never touched the database.
    """

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        root = Path(self._temp.name)
        self.storage = LocalCourseArtifactStorage(
            root_dir=root / "course_data", index_dir=root / "indexes"
        )
        self.course_id = "css-360-winter-2026-a7rp"
        self.storage.save_extracted_text(self.course_id, SAMPLE)
        self.updates: list[tuple[str, dict]] = []

        def fake_update(conn, course_id, patch):
            self.updates.append((course_id, dict(patch)))
            return {"courseId": course_id, "metadata": {"chunkCount": patch["chunkCount"]}}

        self.update_course = fake_update

    def tearDown(self) -> None:
        self._temp.cleanup()

    def _patches(self, *, db_available: bool = True):
        connection = (
            _fake_connection
            if db_available
            else self._raising_connection
        )
        return (
            patch("app.reindex_course.get_course_artifact_storage", return_value=self.storage),
            patch("app.course_index.get_embedding", new=AsyncMock(return_value=[0.1, 0.2, 0.3])),
            patch("app.reindex_course.db_connection", new=connection),
            patch("app.reindex_course.update_course", side_effect=self.update_course),
        )

    @staticmethod
    @contextmanager
    def _raising_connection(**kwargs):
        raise DatabaseConfigurationError("PostgreSQL is not configured.")
        yield  # pragma: no cover

    def test_a_rebuild_writes_the_new_chunk_count_to_the_record(self) -> None:
        with self._patches()[0], self._patches()[1], self._patches()[2], self._patches()[3]:
            code = reindex_main(["--course-id", self.course_id])
        self.assertEqual(code, 0)
        index = self.storage.load_index(self.course_id)
        assert index is not None
        self.assertEqual(len(self.updates), 1)
        course_id, patch_fields = self.updates[0]
        self.assertEqual(course_id, self.course_id)
        self.assertEqual(patch_fields["chunkCount"], index["chunkCount"])
        self.assertEqual(patch_fields["syllabusStatus"], "indexed")

    def test_sync_record_updates_from_the_existing_index_without_embedding(self) -> None:
        # An index built earlier, larger than whatever the record says.
        self.storage.save_index(
            self.course_id,
            {"indexVersion": 2, "chunkCount": 163, "chunks": [{"chunkId": f"c{i}", "text": "x", "order": i} for i in range(163)]},
        )
        embed = AsyncMock(side_effect=AssertionError("--sync-record must not embed"))
        with (
            patch("app.reindex_course.get_course_artifact_storage", return_value=self.storage),
            patch("app.course_index.get_embedding", new=embed),
            patch("app.reindex_course.db_connection", new=_fake_connection),
            patch("app.reindex_course.update_course", side_effect=self.update_course),
        ):
            code = reindex_main(["--course-id", self.course_id, "--sync-record"])
        self.assertEqual(code, 0)
        self.assertEqual(self.updates, [(self.course_id, {"chunkCount": 163, "syllabusStatus": "indexed"})])
        embed.assert_not_called()
        # The fact-inventory cache is not a casualty of a record sync.
        self.assertTrue(self.storage.index_exists(self.course_id))

    def test_sync_record_without_an_index_fails_clearly(self) -> None:
        with (
            patch("app.reindex_course.get_course_artifact_storage", return_value=self.storage),
            patch("app.reindex_course.db_connection", new=_fake_connection),
            patch("app.reindex_course.update_course", side_effect=self.update_course),
        ):
            code = reindex_main(["--course-id", self.course_id, "--sync-record"])
        self.assertEqual(code, 1)
        self.assertEqual(self.updates, [])

    def test_an_unreachable_database_does_not_undo_the_rebuild(self) -> None:
        with (
            patch("app.reindex_course.get_course_artifact_storage", return_value=self.storage),
            patch("app.course_index.get_embedding", new=AsyncMock(return_value=[0.1, 0.2, 0.3])),
            patch("app.reindex_course.db_connection", new=self._raising_connection),
        ):
            code = reindex_main(["--course-id", self.course_id])
        # The index landed; the operator is told the record did not.
        self.assertEqual(code, 0)
        self.assertTrue(self.storage.index_exists(self.course_id))
        self.assertEqual(self.updates, [])

    def test_a_missing_course_record_is_reported_not_invented(self) -> None:
        self.storage.save_index(
            self.course_id,
            {"indexVersion": 2, "chunkCount": 2, "chunks": [{"chunkId": "c0", "text": "x", "order": 0}, {"chunkId": "c1", "text": "y", "order": 1}]},
        )
        with (
            patch("app.reindex_course.get_course_artifact_storage", return_value=self.storage),
            patch("app.reindex_course.db_connection", new=_fake_connection),
            patch("app.reindex_course.update_course", return_value=None),
        ):
            code = reindex_main(["--course-id", self.course_id, "--sync-record"])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
