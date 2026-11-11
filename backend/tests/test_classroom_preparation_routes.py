"""The four classroom-preparation routes: what they return and who may call them.

The preparation logic itself is covered in `test_classroom_preparation.py`
and the full role matrix in `test_authorization_matrix.py`; this file is the
HTTP surface: status shape, saving a selection, starting and stopping, the
refusals, and that a student has no way in.
"""

from __future__ import annotations

import os
import time
import unittest
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import answer_cache, classroom_preparation as prep
from app.auth.dependencies import current_principal
from app.auth.principal import Participant, Principal, StaffUser
from app.main import app

from tests.answer_cache_fake import FakeAnswerCache
from tests.test_classroom_preparation import FT_IDENTITY, QUESTIONS, FourPipelines

COURSE = "css-360-winter-2026-a7rp"
OTHER = "css-350-spring-2026-n3h9"
BASE = f"/api/courses/{COURSE}/classroom-preparation"
CSRF = {"X-Requested-With": "SyllabusModelLab"}
REAL_USER_ID = "3f2b8c1e-4d5a-4b6c-8d7e-9f0a1b2c3d4e"


def professor(*course_ids: str, user_id: str = REAL_USER_ID) -> Principal:
    return Principal(user=StaffUser(user_id=user_id, email="prof@uw.edu", display_name="Prof",
                                    role="professor", course_ids=frozenset(course_ids), session_id="s"))


def participant(course_id: str) -> Principal:
    return Principal(participant=Participant(participant_id=f"p-{course_id}", course_id=course_id, session_id="s"))


class PreparationRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = FakeAnswerCache().install(self)
        self.pipes = FourPipelines().install(self)
        for condition in ("fineTuned", "fineTunedRag"):
            self.cache.set_identity(condition, COURSE, FT_IDENTITY)
        self.cache.select(COURSE, *QUESTIONS)
        self.saved: list[tuple[str, list[str], str | None]] = []

        @contextmanager
        def fake_connection(**_kwargs: Any) -> Iterator[object]:
            yield object()

        def replace_selection(_conn: Any, course_id: str, seed_ids: Any, *, selected_by: str | None = None, now: Any = None):
            known = {row["seed_id"] for row in self.cache.seeds.get(course_id, [])}
            if any(seed_id not in known for seed_id in seed_ids):
                raise ValueError("A selected question does not belong to this course.")
            self.saved.append((course_id, list(seed_ids), selected_by))
            self.cache.selection[course_id] = list(seed_ids)
            return list(seed_ids)

        for target, new in (
            ("app.classroom_preparation_routes.db_connection", fake_connection),
            ("app.classroom_preparation_routes.db_courses.course_exists", lambda _conn, course_id: course_id in (COURSE, OTHER)),
            ("app.classroom_preparation_routes.db_classroom_questions.replace_selection", replace_selection),
            ("app.classroom_preparation.POLL_SECONDS", 0.005),
        ):
            patcher = patch(target, new=new)
            patcher.start()
            self.addCleanup(patcher.stop)
        env = patch.dict(os.environ, {"CLASSROOM_PREP_QUIET_SECONDS": "0"})
        env.start()
        self.addCleanup(env.stop)
        # One client for the whole test, entered so the run's task and the
        # requests share an event loop, as they do in the server.
        self.client = TestClient(app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def act_as(self, principal: Principal | None) -> None:
        if principal is None:
            app.dependency_overrides.pop(current_principal, None)
        else:
            app.dependency_overrides[current_principal] = lambda: principal

    def status(self, course: str = COURSE) -> dict[str, Any]:
        response = self.client.get(f"/api/courses/{course}/classroom-preparation")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def wait_until_idle(self) -> dict[str, Any]:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = self.status()
            if not status["running"]:
                return status
            time.sleep(0.01)
        self.fail("preparation did not finish")

    # -- status -------------------------------------------------------------------

    def test_status_says_everything_the_page_shows(self) -> None:
        status = self.status()
        self.assertEqual(
            set(status),
            {"courseId", "enabled", "state", "running", "stopping", "waiting", "current", "canStart",
             "startBlockedReason", "counts", "conditions", "message", "notices", "lastRun", "questions"},
        )
        self.assertEqual((status["state"], status["canStart"], status["lastRun"]), ("not_started", True, None))
        self.assertEqual(status["counts"]["selectedQuestions"], 3)
        self.assertEqual(status["counts"]["expectedAnswers"], 12)
        self.assertEqual(
            {key: value["label"] for key, value in status["conditions"].items()},
            {"base": "Base", "rag": "RAG", "fineTuned": "Fine-Tuned", "fineTunedRag": "Fine-Tuned + RAG"},
        )
        question = status["questions"][0]
        self.assertEqual(
            set(question),
            {"seedId", "question", "origin", "source", "sourceLabel", "reviewStatus", "category",
             "selected", "eligible", "ineligibleReason", "answers"},
        )
        # A question, its source and its readiness; never a training answer.
        self.assertNotIn("response", question)
        self.assertNotIn("answer", question)

    def test_the_database_being_down_is_a_503_not_a_crash(self) -> None:
        self.cache.fail_plan = True
        with patch("app.classroom_preparation._read_plan_rows", side_effect=__import__("app.db", fromlist=["x"]).DatabaseConfigurationError("no db")):
            response = self.client.get(BASE)
        self.assertEqual(response.status_code, 503)

    def test_a_malformed_course_id_is_a_400(self) -> None:
        self.assertEqual(self.client.get("/api/courses/Not%20A%20Course/classroom-preparation").status_code, 400)

    # -- selection ----------------------------------------------------------------

    def test_saving_a_selection_replaces_it_and_takes_effect_for_students_at_once(self) -> None:
        self.act_as(professor(COURSE))
        with patch("app.classroom_preparation_routes.answer_cache.invalidate_selection") as invalidated:
            response = self.client.put(f"{BASE}/questions", json={"seedIds": ["s2"]}, headers=CSRF)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.saved, [(COURSE, ["s2"], REAL_USER_ID)])
        invalidated.assert_called_once_with(COURSE)
        status = response.json()
        self.assertEqual(status["counts"]["selectedQuestions"], 1)
        self.assertEqual([q["selected"] for q in status["questions"]], [False, True, False])

    def test_a_selection_naming_someone_elses_question_is_refused(self) -> None:
        response = self.client.put(f"{BASE}/questions", json={"seedIds": ["s1", "not-this-course"]}, headers=CSRF)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.saved, [])
        self.assertEqual(self.status()["counts"]["selectedQuestions"], 3)

    def test_a_selection_must_be_a_list_of_ids(self) -> None:
        for body in ({}, {"seedIds": "s1"}, {"seedIds": [1, 2]}):
            with self.subTest(body=body):
                self.assertEqual(self.client.put(f"{BASE}/questions", json=body, headers=CSRF).status_code, 422)

    def test_an_identity_that_is_not_an_account_id_is_not_recorded_as_one(self) -> None:
        # conftest's administrator has a placeholder id, not a UUID.
        self.client.put(f"{BASE}/questions", json={"seedIds": ["s1"]}, headers=CSRF)
        self.assertEqual(self.saved, [(COURSE, ["s1"], None)])

    def test_an_unknown_course_is_a_404(self) -> None:
        unknown = "css-111-winter-2026-zzzz"
        response = self.client.put(f"/api/courses/{unknown}/classroom-preparation/questions", json={"seedIds": []}, headers=CSRF)
        self.assertEqual(response.status_code, 404)
        response = self.client.post(f"/api/courses/{unknown}/classroom-preparation/start", headers=CSRF)
        self.assertEqual(response.status_code, 404)

    # -- start and stop -----------------------------------------------------------

    def test_start_runs_to_completion_and_the_status_follows_it(self) -> None:
        response = self.client.post(f"{BASE}/start", headers=CSRF)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["running"])
        status = self.wait_until_idle()
        self.assertEqual((status["state"], status["counts"]["readyAnswers"]), ("ready", 12))
        self.assertEqual((status["canStart"], status["startBlockedReason"]), (False, "nothing_to_prepare"))
        self.assertEqual(len(self.pipes.finished), 12)

    def test_stop_then_resume_through_the_routes(self) -> None:
        import asyncio

        gate_holder: dict[str, Any] = {}

        async def hold() -> None:
            gate_holder["gate"] = asyncio.Event()
            self.pipes.gate = gate_holder["gate"]

        self.client.portal.call(hold)
        self.assertTrue(self.client.post(f"{BASE}/start", headers=CSRF).json()["running"])
        deadline = time.monotonic() + 5
        while self.pipes.running != 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        stopping = self.client.post(f"{BASE}/stop", headers=CSRF).json()
        self.assertEqual((stopping["state"], stopping["stopping"], stopping["canStart"]), ("stopping", True, False))
        self.client.portal.call(gate_holder["gate"].set)
        status = self.wait_until_idle()
        self.assertEqual((status["state"], status["counts"]["readyAnswers"]), ("stopped", 1))
        self.assertTrue(status["canStart"])

        self.pipes.gate = None
        self.client.post(f"{BASE}/start", headers=CSRF)
        status = self.wait_until_idle()
        self.assertEqual((status["state"], status["counts"]["readyAnswers"]), ("ready", 12))
        self.assertEqual(len(self.pipes.finished), 12)

    def test_stopping_when_nothing_is_running_just_reports_the_status(self) -> None:
        response = self.client.post(f"{BASE}/stop", headers=CSRF)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["state"], "not_started")

    def test_start_is_refused_in_plain_words_when_it_cannot_run(self) -> None:
        with patch.dict(os.environ, {answer_cache.ENABLED_ENV: ""}):
            response = self.client.post(f"{BASE}/start", headers=CSRF)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["code"], "preparation_disabled")
        self.assertIn("switched off", response.json()["detail"]["message"])

        self.cache.set_selection(COURSE, [])
        response = self.client.post(f"{BASE}/start", headers=CSRF)
        self.assertEqual((response.status_code, response.json()["detail"]["code"]), (409, "preparation_nothing_selected"))
        self.assertIsNone(prep.active_run())

    # -- who may call -------------------------------------------------------------

    def test_students_and_staff_of_other_courses_have_no_way_in(self) -> None:
        calls = (
            ("GET", BASE, None),
            ("PUT", f"{BASE}/questions", {"seedIds": []}),
            ("POST", f"{BASE}/start", None),
            ("POST", f"{BASE}/stop", None),
        )
        for principal, expected in (
            (None, 401),
            (participant(COURSE), 401),
            (participant(OTHER), 401),
            (professor(OTHER), 403),
        ):
            self.act_as(principal)
            for method, path, body in calls:
                with self.subTest(principal=principal, method=method, path=path):
                    response = self.client.request(method, path, json=body, headers=CSRF)
                    self.assertEqual(response.status_code, expected, response.text)
        self.assertEqual(self.saved, [])
        self.assertIsNone(prep.active_run())
        self.assertEqual(self.pipes.started, [])

    def test_the_courses_professor_and_an_administrator_both_reach_it(self) -> None:
        self.act_as(professor(COURSE))
        self.assertEqual(self.client.get(BASE).status_code, 200)
        # Scoped to the course in the path: the same professor cannot see another's.
        self.assertEqual(self.client.get(f"/api/courses/{OTHER}/classroom-preparation").status_code, 403)


if __name__ == "__main__":
    unittest.main()
