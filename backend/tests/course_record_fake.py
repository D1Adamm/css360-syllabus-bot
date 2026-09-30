"""The course row the syllabus upload route reads and writes, without a database.

The upload route checks that the course exists and records the syllabus state
on its row. Tests that exercise the upload pipeline itself patch those two
calls with this fake instead of a database; `records` holds every patch the
route wrote, in order, per course.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch


@contextmanager
def _fake_connection(**kwargs: Any) -> Iterator[object]:
    yield object()


class FakeCourseRecords:
    def __init__(self, existing: set[str] | None = None) -> None:
        #: None means every course id exists.
        self.existing = existing
        self.records: dict[str, list[dict[str, Any]]] = {}

    def course_exists(self, conn: Any, course_id: str) -> bool:
        return self.existing is None or course_id in self.existing

    def update_course(self, conn: Any, course_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        self.records.setdefault(course_id, []).append(dict(patch))
        return {"courseId": course_id, "metadata": {}}

    def last(self, course_id: str) -> dict[str, Any] | None:
        writes = self.records.get(course_id)
        return writes[-1] if writes else None


def patch_course_records(
    test: unittest.TestCase, existing: set[str] | None = None
) -> FakeCourseRecords:
    fake = FakeCourseRecords(existing)
    for target, replacement in (
        ("app.main.db_connection", _fake_connection),
        ("app.db_courses.course_exists", fake.course_exists),
        ("app.db_courses.update_course", fake.update_course),
    ):
        patcher = patch(target, new=replacement)
        patcher.start()
        test.addCleanup(patcher.stop)
    return fake
