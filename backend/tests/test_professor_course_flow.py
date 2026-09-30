"""A professor creates a course and then works in it, request after request.

Every route here is already covered one at a time by the authorization matrix.
What that could not catch is the sequence: the create-course page used to read
`GET /api/db/courses/{id}` for an id nobody owned yet, which every professor is
refused (403) while every administrator is answered 404 — so course creation
worked for admins and failed for all instructors.

These tests drive the professor's whole path through the real guards. The
principal is rebuilt from a membership table on every request, which is what
`app.auth.dependencies._load_staff` does against `course_memberships`, so a
membership the create route writes is visible to the very next request and
to nothing before it.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.auth.dependencies import current_principal
from app.auth.principal import ANONYMOUS, Participant, Principal, StaffUser
from app.auth.settings import CSRF_HEADER_VALUE
from app.main import app
from app.storage import LocalCourseArtifactStorage

pytestmark = pytest.mark.auth

NEW_COURSE = "css-430-fall-2026-k7q2"
OTHER_COURSE = "css-350-spring-2026-n3h9"
CSRF = {"X-Requested-With": CSRF_HEADER_VALUE}

PROFESSOR = "user-prof"
OTHER_PROFESSOR = "user-other"
ADMIN = "user-admin"

SYLLABUS_TEXT = (
    "CSS 430 Operating Systems. Course Overview: processes, threads, scheduling, "
    "memory management and file systems. Office Hours are Monday and Wednesday "
    "from 2 to 3 pm. Grading: homework 40 percent, exams 60 percent."
)

#: Exactly what the create-course page sends when the optional instructor-name
#: field is left blank. The request model once required it, which would have
#: been the next thing every professor hit.
CREATE_BODY = {
    "courseId": NEW_COURSE,
    "name": "CSS 430",
    "title": "Operating Systems",
    "term": "Fall 2026",
    "instructorName": "",
    "createdAt": "2026-09-30T00:00:00+00:00",
    "syllabusStatus": "not_uploaded",
    "syllabusFileName": "",
    "syllabusType": "",
    "chunkCount": 0,
}


def _metadata(**overrides: Any) -> dict[str, Any]:
    metadata = {
        "name": "CSS 430",
        "title": "Operating Systems",
        "term": "Fall 2026",
        "instructorName": "",
        "createdAt": "2026-09-30T00:00:00+00:00",
        "syllabusStatus": "not_uploaded",
        "syllabusFileName": None,
        "syllabusType": None,
        "chunkCount": 0,
    }
    metadata.update(overrides)
    return metadata


@contextmanager
def _fake_connection(**kwargs: Any) -> Iterator[object]:
    yield object()


class FakeCourseStore:
    """The three tables this flow touches: courses, memberships, the audit trail."""

    def __init__(self) -> None:
        self.courses: dict[str, dict[str, Any]] = {}
        self.memberships: set[tuple[str, str]] = set()
        self.actions: list[dict[str, Any]] = []

    def create_course(
        self, conn: Any, course_id: str, metadata: dict[str, Any], *, created_by: str | None = None
    ) -> dict[str, Any]:
        from app.db_courses import CourseAlreadyExistsError

        if course_id in self.courses:
            raise CourseAlreadyExistsError(f'Course "{course_id}" already exists.')
        self.courses[course_id] = {
            "metadata": _metadata(
                **{k: metadata[k] for k in ("name", "title", "term", "instructorName")}
            ),
            "created_by": created_by,
        }
        return {"courseId": course_id, "metadata": self.courses[course_id]["metadata"]}

    def course_exists(self, conn: Any, course_id: str) -> bool:
        return course_id in self.courses

    def get_course(self, conn: Any, course_id: str) -> dict[str, Any] | None:
        course = self.courses.get(course_id)
        return None if course is None else {"courseId": course_id, "metadata": course["metadata"]}

    def update_course(self, conn: Any, course_id: str, patch: dict[str, Any]) -> dict[str, Any] | None:
        course = self.courses.get(course_id)
        if course is None:
            return None
        course["metadata"].update(patch)
        return {"courseId": course_id, "metadata": course["metadata"]}

    def list_courses(self, conn: Any, visible: set[str] | None) -> list[dict[str, Any]]:
        return [
            {"courseId": course_id, "metadata": course["metadata"]}
            for course_id, course in self.courses.items()
            if visible is None or course_id in visible
        ]

    def add_membership(self, conn: Any, *, course_id: str, user_id: str, granted_by: str) -> bool:
        self.memberships.add((user_id, course_id))
        return True

    def record_action(self, conn: Any, **kwargs: Any) -> None:
        self.actions.append(kwargs)

    def course_ids_for(self, user_id: str) -> frozenset[str]:
        return frozenset(course for user, course in self.memberships if user == user_id)


class ProfessorCourseFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FakeCourseStore()
        self.client = TestClient(app, base_url="https://testserver")
        self.addCleanup(app.dependency_overrides.clear)
        self.acting_as: str | Principal = ANONYMOUS

        def principal() -> Principal:
            # Resolved per request, like the real resolver: memberships are
            # re-read every time, never carried over from the previous request.
            if isinstance(self.acting_as, Principal):
                return self.acting_as
            user_id = self.acting_as
            role = "admin" if user_id == ADMIN else "professor"
            return Principal(
                user=StaffUser(
                    user_id=user_id,
                    email=f"{user_id}@uw.edu",
                    display_name=user_id,
                    role=role,
                    course_ids=frozenset() if role == "admin" else self.store.course_ids_for(user_id),
                    session_id=f"session-{user_id}",
                )
            )

        app.dependency_overrides[current_principal] = principal

        for target, replacement in (
            ("app.db_routes.db_connection", _fake_connection),
            ("app.main.db_connection", _fake_connection),
            ("app.db_courses.course_exists", self.store.course_exists),
            ("app.db_courses.create_course", self.store.create_course),
            ("app.db_courses.get_course", self.store.get_course),
            ("app.db_courses.update_course", self.store.update_course),
            ("app.db_courses.list_courses", self.store.list_courses),
            ("app.db_memberships.add_membership", self.store.add_membership),
            ("app.db_admin_actions.record_action", self.store.record_action),
        ):
            self._patch(target, new=replacement)

        # The syllabus upload runs for real against a temporary artifact store;
        # only the embedding call and the starter-seed queue are stubbed.
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        self.artifacts = LocalCourseArtifactStorage(
            root_dir=root / "course_data", index_dir=root / "indexes"
        )
        self._patch("app.main.get_course_artifact_storage", return_value=self.artifacts)
        self._patch("app.course_index.get_embedding", new=AsyncMock(return_value=[0.01, 0.02, 0.03]))
        self._patch(
            "app.main.try_queue_starter_seed_generation",
            new=AsyncMock(return_value={"queued": False, "status": "not_started"}),
        )

    def _patch(self, target: str, **kwargs: Any) -> Any:
        patcher = patch(target, **kwargs)
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock

    # ---- requests, as the create-course page sends them ----

    def _create(self, body: dict[str, Any] | None = None) -> Any:
        return self.client.post("/api/db/courses", json=body or CREATE_BODY, headers=CSRF)

    def _upload(self, text: str = SYLLABUS_TEXT, course_id: str = NEW_COURSE) -> Any:
        return self.client.post(
            f"/api/courses/{course_id}/syllabus",
            files={"syllabus_file": ("syllabus.txt", io.BytesIO(text.encode()), "text/plain")},
            headers=CSRF,
        )

    def _mark_indexed(self, course_id: str = NEW_COURSE) -> Any:
        return self.client.patch(
            f"/api/db/courses/{course_id}",
            json={"syllabusStatus": "indexed", "syllabusFileName": "syllabus.txt",
                  "syllabusType": "txt", "chunkCount": 1},
            headers=CSRF,
        )

    # ---- the professor's path ----

    def test_a_professor_with_no_courses_creates_one_and_works_in_it(self) -> None:
        self.acting_as = PROFESSOR

        created = self._create()
        self.assertEqual(created.status_code, 201, created.text)
        self.assertEqual(created.json()["courseId"], NEW_COURSE)
        self.assertEqual(created.json()["metadata"]["instructorName"], "")

        # Owned from the first moment: creator recorded, membership granted.
        self.assertEqual(self.store.courses[NEW_COURSE]["created_by"], PROFESSOR)
        self.assertIn((PROFESSOR, NEW_COURSE), self.store.memberships)
        self.assertEqual(self.store.actions[-1]["action"], "course.create")
        self.assertEqual(self.store.actions[-1]["actor_role"], "professor")

        # Then everything the page and the course pages do next.
        uploaded = self._upload()
        self.assertEqual(uploaded.status_code, 201, uploaded.text)
        self.assertEqual(uploaded.json()["syllabusStatus"], "indexed")
        self.assertTrue(self.artifacts.index_exists(NEW_COURSE))
        # The upload itself records the syllabus on the course row.
        record = self.store.courses[NEW_COURSE]["metadata"]
        self.assertEqual(record["syllabusStatus"], "indexed")
        self.assertEqual(record["syllabusFileName"], "syllabus.txt")

        self.assertEqual(self._mark_indexed().status_code, 200)
        read = self.client.get(f"/api/db/courses/{NEW_COURSE}")
        self.assertEqual(read.status_code, 200)
        self.assertEqual(read.json()["metadata"]["syllabusStatus"], "indexed")

        listed = self.client.get("/api/db/courses").json()
        self.assertEqual([c["courseId"] for c in listed["courses"]], [NEW_COURSE])

        session = self.client.get("/api/auth/session").json()
        self.assertEqual(session["user"]["courseIds"], [NEW_COURSE])

        # A replacement syllabus is the same staff route.
        replaced = self._upload()
        self.assertEqual(replaced.status_code, 201)
        self.assertTrue(replaced.json()["replaced"])

    def test_a_failed_first_upload_leaves_a_course_the_professor_can_still_fix(self) -> None:
        self.acting_as = PROFESSOR
        self.assertEqual(self._create().status_code, 201)

        with patch(
            "app.course_index.get_embedding",
            new=AsyncMock(side_effect=HTTPException(status_code=503, detail="Embeddings unavailable.")),
        ):
            self.assertEqual(self._upload().status_code, 503)

        # Still theirs, still listed, and it says what is wrong.
        read = self.client.get(f"/api/db/courses/{NEW_COURSE}")
        self.assertEqual(read.status_code, 200)
        self.assertEqual(read.json()["metadata"]["syllabusStatus"], "index_failed")
        self.assertEqual(
            [c["courseId"] for c in self.client.get("/api/db/courses").json()["courses"]],
            [NEW_COURSE],
        )

        # The retry goes into the same course: no second course is needed.
        retried = self._upload()
        self.assertEqual(retried.status_code, 201, retried.text)
        self.assertEqual(list(self.store.courses), [NEW_COURSE])
        self.assertEqual(self.store.courses[NEW_COURSE]["metadata"]["syllabusStatus"], "indexed")

    def test_reading_a_course_before_it_exists_is_refused_to_a_professor(self) -> None:
        # Pinned because it is the reason the frontend must not probe for id
        # collisions: an unowned id is 403 for a professor, not 404.
        self.acting_as = PROFESSOR
        self.assertEqual(self.client.get(f"/api/db/courses/{NEW_COURSE}").status_code, 403)
        self.acting_as = ADMIN
        self.assertEqual(self.client.get(f"/api/db/courses/{NEW_COURSE}").status_code, 404)

    def test_an_id_that_is_already_taken_is_a_409_and_grants_nothing(self) -> None:
        self.acting_as = OTHER_PROFESSOR
        self.assertEqual(self._create().status_code, 201)

        self.acting_as = PROFESSOR
        response = self._create()
        self.assertEqual(response.status_code, 409)
        self.assertNotIn((PROFESSOR, NEW_COURSE), self.store.memberships)
        self.assertEqual(self.client.get(f"/api/db/courses/{NEW_COURSE}").status_code, 403)

    # ---- who else may not ----

    def test_another_professor_cannot_touch_the_new_course(self) -> None:
        self.acting_as = PROFESSOR
        self.assertEqual(self._create().status_code, 201)

        self.assertEqual(self._upload().status_code, 201)
        text_before = self.artifacts.load_extracted_text(NEW_COURSE)

        self.acting_as = OTHER_PROFESSOR
        self.assertEqual(self.client.get(f"/api/db/courses/{NEW_COURSE}").status_code, 403)
        self.assertEqual(self._mark_indexed().status_code, 403)
        self.assertEqual(self._upload("Replaced by someone else. " * 10).status_code, 403)
        self.assertEqual(self.artifacts.load_extracted_text(NEW_COURSE), text_before)
        self.assertEqual(self.client.get("/api/db/courses").json()["courses"], [])

    def test_nobody_signed_in_cannot_create_or_upload(self) -> None:
        self.acting_as = ANONYMOUS
        self.assertEqual(self._create().status_code, 401)
        self.assertEqual(self._upload().status_code, 401)
        self.assertEqual(self.store.courses, {})

    def test_a_student_cannot_create_a_course_or_upload_a_syllabus(self) -> None:
        self.acting_as = PROFESSOR
        self.assertEqual(self._create().status_code, 201)

        # A participant of this very course is still not its staff.
        self.acting_as = Principal(
            participant=Participant(participant_id="p-1", course_id=NEW_COURSE, session_id="s")
        )
        self.assertEqual(
            self._create({**CREATE_BODY, "courseId": OTHER_COURSE}).status_code, 401
        )
        self.assertNotIn(OTHER_COURSE, self.store.courses)
        self.assertEqual(self._upload("A student's syllabus. " * 10).status_code, 401)
        self.assertIsNone(self.artifacts.load_extracted_text(NEW_COURSE))
        self.assertEqual(self._mark_indexed().status_code, 401)
        # Reading the course they joined is theirs; its staff pages are not.
        self.assertEqual(self.client.get(f"/api/db/courses/{NEW_COURSE}").status_code, 200)

    def test_owning_a_course_grants_no_administrator_operation_on_it(self) -> None:
        self.acting_as = PROFESSOR
        self.assertEqual(self._create().status_code, 201)

        for method, path in (
            ("PATCH", f"/api/db/courses/{NEW_COURSE}/starter-seed-generation"),
            ("PATCH", f"/api/db/courses/{NEW_COURSE}/model-request"),
            ("POST", f"/api/db/courses/{NEW_COURSE}/training-runs"),
            ("POST", f"/api/db/courses/{NEW_COURSE}/evaluations/preview"),
            ("DELETE", f"/api/db/courses/{NEW_COURSE}/evaluations"),
            ("POST", f"/api/courses/{NEW_COURSE}/training/launch"),
            ("POST", f"/api/courses/{NEW_COURSE}/seeds/generate-starter"),
            ("GET", f"/api/courses/{NEW_COURSE}/chunks"),
            ("GET", "/api/admin/users"),
            ("PUT", f"/api/admin/users/{OTHER_PROFESSOR}/courses/{NEW_COURSE}"),
        ):
            with self.subTest(method=method, path=path):
                response = self.client.request(method, path, json={}, headers=CSRF)
                self.assertEqual(response.status_code, 403, response.text)

    # ---- the original file ----

    def _file(self, download: bool = False) -> Any:
        suffix = "?download=1" if download else ""
        return self.client.get(f"/api/courses/{NEW_COURSE}/syllabus/file{suffix}")

    def test_course_staff_view_and_download_the_original_file(self) -> None:
        self.acting_as = PROFESSOR
        self.assertEqual(self._create().status_code, 201)
        self.assertEqual(self._upload().status_code, 201)

        viewed = self._file()
        self.assertEqual(viewed.status_code, 200)
        self.assertEqual(viewed.content, SYLLABUS_TEXT.encode())
        self.assertTrue(viewed.headers["content-type"].startswith("text/plain"))
        self.assertTrue(viewed.headers["content-disposition"].startswith("inline"))
        self.assertIn("syllabus.txt", viewed.headers["content-disposition"])
        self.assertEqual(viewed.headers["x-content-type-options"], "nosniff")
        self.assertIn("no-store", viewed.headers["cache-control"])

        probed = self.client.head(f"/api/courses/{NEW_COURSE}/syllabus/file")
        self.assertEqual(probed.status_code, 200)
        self.assertEqual(probed.content, b"")

        downloaded = self._file(download=True)
        self.assertEqual(downloaded.status_code, 200)
        self.assertTrue(downloaded.headers["content-disposition"].startswith("attachment"))

        self.acting_as = ADMIN
        self.assertEqual(self._file().status_code, 200)

    def test_nobody_else_gets_the_original_file(self) -> None:
        self.acting_as = PROFESSOR
        self.assertEqual(self._create().status_code, 201)
        self.assertEqual(self._upload().status_code, 201)

        self.acting_as = OTHER_PROFESSOR
        self.assertEqual(self._file().status_code, 403)
        self.assertEqual(
            self.client.head(f"/api/courses/{NEW_COURSE}/syllabus/file").status_code, 403
        )
        self.acting_as = Principal(
            participant=Participant(participant_id="p-1", course_id=NEW_COURSE, session_id="s")
        )
        self.assertEqual(self._file().status_code, 401)
        self.acting_as = ANONYMOUS
        self.assertEqual(self._file().status_code, 401)

    def test_a_course_without_a_stored_original_is_a_404(self) -> None:
        self.acting_as = PROFESSOR
        self.assertEqual(self._create().status_code, 201)
        self.assertEqual(self._file().status_code, 404)
        self.assertEqual(
            self.client.head(f"/api/courses/{NEW_COURSE}/syllabus/file").status_code, 404
        )

    def test_a_failed_replacement_keeps_serving_the_previous_file(self) -> None:
        self.acting_as = PROFESSOR
        self.assertEqual(self._create().status_code, 201)
        self.assertEqual(self._upload().status_code, 201)
        with patch(
            "app.course_index.get_embedding",
            new=AsyncMock(side_effect=HTTPException(status_code=503, detail="Embeddings unavailable.")),
        ):
            self.assertEqual(self._upload("A replacement that never lands. " * 8).status_code, 503)
        self.assertEqual(self._file().content, SYLLABUS_TEXT.encode())

    def test_an_administrator_creates_a_course_without_a_membership(self) -> None:
        self.acting_as = ADMIN
        self.assertEqual(self._create().status_code, 201)
        self.assertEqual(self.store.memberships, set())
        self.assertEqual(self._upload().status_code, 201)
        self.assertEqual(self.client.get(f"/api/db/courses/{NEW_COURSE}").status_code, 200)


if __name__ == "__main__":
    unittest.main()
