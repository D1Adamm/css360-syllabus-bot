"""An administrator's explicit-version generation: the model-testing route.

`POST /api/model-testing/generate` exists for one comparison the classroom
routes cannot make: the same question, the same retrieval and prompting, the
same client, answered by a version the course is not serving. Those routes
resolve the course's version from the registry and read nothing else, which is
right for a student and useless for deciding whether a newly trained version
should replace the one students get.

What is held here:

1. an administrator names a version, and that version — not the resolved one —
   is what reaches the fine-tuned service, for Fine-Tuned and Fine-Tuned + RAG;
   `rag` runs the production RAG function and has no version;
2. a version the registry cannot offer is refused with a reason, before the
   service is asked; a malformed one is refused before anything runs; and the
   refusal is never a different version;
3. nobody but an administrator reaches the route — anonymous, participants,
   professors of the course itself;
4. nothing is written: the registry is read and not touched, no participant or
   session is created, no cookie is set, no evaluation exists to save;
5. the classroom routes are unchanged: they still resolve the course's version
   from the same registry, and a `modelVersion` in their body is not read, for
   anyone.

`test_authorization_matrix.py` drives the route against every principal as
well, because it is classified `require_admin`; the tests here say why those
outcomes are the right ones.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import current_principal
from app.auth.principal import Participant, Principal, StaffUser
from app.auth.settings import CSRF_HEADER_VALUE
from app.course_model_resolution import UnservableCourseModelVersion
from app.main import app
from route_classification import CLASSIFICATION, REQUIRE_ADMIN
from test_db_repositories import FakeConnection

pytestmark = pytest.mark.auth

COURSE = "css-360-winter-2026-a7rp"
OTHER = "css-350-spring-2026-n3h9"
PATH = "/api/model-testing/generate"
CSRF = {"X-Requested-With": CSRF_HEADER_VALUE}
NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
QUESTION = "When are office hours?"

#: The first of the two statements `get_model_registry` runs, as the fake
#: records it (whitespace collapsed).
MODEL_ROW_QUERY = "SELECT course_id, current_version FROM course_models WHERE course_id = %s"


def version_row(
    version: str, *, status: str = "ready", deployment: str = "offline"
) -> dict[str, Any]:
    return {
        "version": version,
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "training_example_count": 48,
        "status": status,
        "deployment": deployment,
        "artifact_ref": f"css-360-qlora/{version}/adapter",
        "created_at": NOW,
        "updated_at": None,
        "notes": None,
        "run_id": None,
        "provenance": None,
    }


def registry_results(*, v3_status: str = "ready") -> list[Any]:
    """CSS 360 as it stands: v2 current and published; v3 ready and neither.

    The two result sets `get_model_registry` consumes, in order: the course
    row, then every version row.
    """
    return [
        [{"course_id": COURSE, "current_version": "v2"}],
        [version_row("v3", status=v3_status), version_row("v2", deployment="online")],
    ]


def admin() -> Principal:
    return Principal(
        user=StaffUser(
            user_id="u-admin", email="admin@uw.edu", display_name="Admin",
            role="admin", session_id="s",
        )
    )


def professor(*course_ids: str) -> Principal:
    return Principal(
        user=StaffUser(
            user_id="u-prof", email="prof@uw.edu", display_name="Prof",
            role="professor", course_ids=frozenset(course_ids), session_id="s",
        )
    )


def participant(course_id: str) -> Principal:
    return Principal(
        participant=Participant(
            participant_id=f"participant-{course_id}", course_id=course_id, session_id="s"
        )
    )


def fine_tuned_result(model_version: str | None) -> dict[str, Any]:
    """What `generate_finetuned_response` returns once the service answered."""
    return {
        "answer": f"Answer from {model_version}.",
        "model": f"css360-ft-{model_version}:latest",
        "adapter_loaded": True,
        "course_id": COURSE,
        "model_version": model_version,
        "generation_seconds": 0.4,
        "response_type": "fineTuned",
    }


def fine_tuned_rag_result(model_version: str | None) -> dict[str, Any]:
    """What `generate_course_finetuned_rag_answer` returns."""
    return {
        "courseId": COURSE,
        "answer": f"Grounded answer from {model_version}.",
        "model": f"css360-ft-{model_version}:latest",
        "modelVersion": model_version,
        "adapterLoaded": True,
        "generationSeconds": 0.6,
        "sources": [
            {"chunkId": "c1", "sectionTitle": "Office Hours", "text": "Tuesdays 2pm.", "score": 0.9}
        ],
        "retrievedChunks": [
            {"chunkId": "c1", "section": "Office Hours", "text": "Tuesdays 2pm.", "score": 0.9}
        ],
        "responseType": "fineTunedRag",
    }


RAG_RESULT: dict[str, Any] = {
    "courseId": COURSE,
    "answer": "Tuesdays at 2pm.",
    "model": "llama3.2:3b",
    "sources": [
        {"chunkId": "c1", "sectionTitle": "Office Hours", "text": "Tuesdays 2pm.", "score": 0.9}
    ],
    "retrievedChunks": [
        {"chunkId": "c1", "section": "Office Hours", "text": "Tuesdays 2pm.", "score": 0.9}
    ],
    "responseType": "rag",
}


class ModelTestingTestCase(unittest.TestCase):
    """The route against the real resolvers, a recording registry, and stubbed
    generation.

    `resolve_course_model_version` and `resolve_current_course_model` run for
    real and read the registry through `self.connection`, so a test can say
    which version was resolved and that only reads happened. The three
    generation functions the route dispatches to are stubbed at the seam the
    route calls them through, and record what they were asked for.
    """

    def setUp(self) -> None:
        self.client = TestClient(app, base_url="https://testserver")
        self.addCleanup(app.dependency_overrides.clear)
        self.connection = FakeConnection(registry_results())

        @contextmanager
        def fake_connection(**kwargs: Any) -> Iterator[FakeConnection]:
            yield self.connection

        self.patch("app.course_model_resolution.db_connection", new=fake_connection)

        async def fine_tuned(question: str, *, course_id: str, model_version: str | None = None):
            return fine_tuned_result(model_version)

        async def fine_tuned_rag(
            *, course_id: str, question: str, top_k: int = 4, model_version: str | None = None
        ):
            return fine_tuned_rag_result(model_version)

        self.fine_tuned = self.patch(
            "app.main.generate_finetuned_response", new=AsyncMock(side_effect=fine_tuned)
        )
        self.fine_tuned_rag = self.patch(
            "app.main.generate_course_finetuned_rag_answer",
            new=AsyncMock(side_effect=fine_tuned_rag),
        )
        self.rag = self.patch(
            "app.main.generate_course_rag_answer", new=AsyncMock(return_value=RAG_RESULT)
        )
        self.create_participant = self.patch("app.db_participants.create_participant")
        self.create_session = self.patch("app.db_sessions.create_session")

    def patch(self, target: str, **kwargs: Any) -> Any:
        patcher = patch(target, **kwargs)
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock

    def act_as(self, principal: Principal | None) -> None:
        if principal is None:
            app.dependency_overrides.pop(current_principal, None)
        else:
            app.dependency_overrides[current_principal] = lambda: principal

    def registry(self, *result_sets: list[Any]) -> None:
        """Queue one registry read per result set (each is consumed in full)."""
        queued: list[Any] = []
        for results in result_sets:
            queued.extend(results)
        self.connection = FakeConnection(queued)

    def post(self, body: dict[str, Any], path: str = PATH):
        return self.client.post(path, json=body, headers=CSRF)

    def body_for(self, mode: str, model_version: str | None = None, **extra: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"courseId": COURSE, "mode": mode, "question": QUESTION, **extra}
        if model_version is not None:
            body["modelVersion"] = model_version
        return body

    def assert_nothing_generated(self) -> None:
        self.fine_tuned.assert_not_awaited()
        self.fine_tuned_rag.assert_not_awaited()
        self.rag.assert_not_awaited()

    def writes(self) -> list[str]:
        return [sql for sql in self.connection.sql if not sql.startswith("SELECT")]


# --------------------------------------------------------------------------- #
# 1. An administrator selects the version
# --------------------------------------------------------------------------- #


class AdministratorSelectsAVersionTests(ModelTestingTestCase):
    def test_fine_tuned_answers_from_the_requested_version_not_the_resolved_one(self) -> None:
        """v2 is the course's version; v3 is ready and nothing more. Both are
        selectable, and the one asked for is the one the client is told."""
        self.act_as(admin())
        self.registry(registry_results(), registry_results())
        current = self.patch("app.main.resolve_current_course_model")

        for version in ("v2", "v3"):
            with self.subTest(version=version):
                response = self.post(self.body_for("fineTuned", version))

                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual(body["courseId"], COURSE)
                self.assertEqual(body["mode"], "fineTuned")
                self.assertEqual(body["modelVersion"], version)
                self.assertEqual(body["responseType"], "fineTuned")
                self.assertEqual(body["answer"], f"Answer from {version}.")
                sent = self.fine_tuned.await_args
                self.assertEqual(sent.args[0], QUESTION)
                self.assertEqual(sent.kwargs["course_id"], COURSE)
                self.assertEqual(sent.kwargs["model_version"], version)

        current.assert_not_called()

    def test_fine_tuned_rag_threads_the_version_through_the_production_function(self) -> None:
        self.act_as(admin())
        response = self.post(self.body_for("fineTunedRag", "v3", topK=6))

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["mode"], "fineTunedRag")
        self.assertEqual(body["modelVersion"], "v3")
        self.assertEqual(body["responseType"], "fineTunedRag")
        self.assertEqual(body["sources"][0]["sectionTitle"], "Office Hours")
        self.assertEqual(body["retrievedChunks"][0]["chunkId"], "c1")
        sent = self.fine_tuned_rag.await_args.kwargs
        self.assertEqual(sent["course_id"], COURSE)
        self.assertEqual(sent["question"], QUESTION)
        self.assertEqual(sent["top_k"], 6)
        self.assertEqual(sent["model_version"], "v3")
        self.fine_tuned.assert_not_awaited()

    def test_rag_runs_the_production_rag_function_and_has_no_version(self) -> None:
        self.act_as(admin())
        response = self.post(self.body_for("rag"))

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["mode"], "rag")
        self.assertIsNone(body["modelVersion"])
        self.assertEqual(body["responseType"], "rag")
        self.assertEqual(body["model"], "llama3.2:3b")
        self.assertEqual(body["answer"], "Tuesdays at 2pm.")
        self.assertIsNone(body["adapterLoaded"])
        self.assertEqual(
            self.rag.await_args.kwargs,
            {"course_id": COURSE, "question": QUESTION, "top_k": 4},
        )
        self.fine_tuned.assert_not_awaited()
        self.fine_tuned_rag.assert_not_awaited()
        # RAG never touches the registry.
        self.assertEqual(self.connection.statements, [])

    def test_the_response_reports_what_the_service_answered_with(self) -> None:
        self.act_as(admin())
        response = self.post(self.body_for("fineTuned", "v3"))

        body = response.json()
        self.assertEqual(body["model"], "css360-ft-v3:latest")
        self.assertIs(body["adapterLoaded"], True)
        self.assertEqual(body["generationSeconds"], 0.4)
        self.assertEqual(body["sources"], [])
        self.assertEqual(body["retrievedChunks"], [])


# --------------------------------------------------------------------------- #
# 2. Refusals are explicit, and never a different version
# --------------------------------------------------------------------------- #


class RefusalTests(ModelTestingTestCase):
    def test_a_version_the_registry_does_not_have_is_409_naming_what_it_has(self) -> None:
        self.act_as(admin())
        response = self.post(self.body_for("fineTuned", "v9"))

        self.assertEqual(response.status_code, 409, response.text)
        detail = response.json()["detail"]
        self.assertIn('"v9"', detail)
        self.assertIn("v2, v3", detail)
        self.assert_nothing_generated()

    def test_a_version_that_is_not_ready_is_409(self) -> None:
        self.act_as(admin())
        self.registry(registry_results(v3_status="training"))
        response = self.post(self.body_for("fineTuned", "v3"))

        self.assertEqual(response.status_code, 409, response.text)
        self.assertIn("training", response.json()["detail"])
        self.assert_nothing_generated()

    def test_a_course_with_no_registered_model_is_409(self) -> None:
        self.act_as(admin())
        self.registry([[]])
        response = self.post(self.body_for("fineTuned", "v2"))

        self.assertEqual(response.status_code, 409, response.text)
        self.assertIn("no fine-tuned model registered", response.json()["detail"])
        self.assert_nothing_generated()

    def test_fine_tuned_rag_propagates_the_registrys_refusal(self) -> None:
        """The composition function resolves the explicit version itself; its
        refusal reaches the caller as the same 409."""
        self.act_as(admin())
        self.fine_tuned_rag.side_effect = UnservableCourseModelVersion(
            COURSE, "v9", 'Course "x" has no registered model version "v9".'
        )
        response = self.post(self.body_for("fineTunedRag", "v9"))

        self.assertEqual(response.status_code, 409)
        self.assertIn('"v9"', response.json()["detail"])
        self.fine_tuned.assert_not_awaited()

    def test_a_malformed_version_is_422_before_anything_runs(self) -> None:
        self.act_as(admin())
        for bad in ("V2", "2", "v2.1", "latest", "v2;drop", "", " v2", "css360-ft-v2:latest"):
            with self.subTest(version=bad):
                response = self.post(self.body_for("fineTuned", bad))
                self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.connection.statements, [])
        self.assert_nothing_generated()

    def test_the_fine_tuned_modes_require_a_version(self) -> None:
        """No version is not 'the current version'. The classroom routes are
        the way to ask for that; here an omission is a mistake, not a default."""
        self.act_as(admin())
        for mode in ("fineTuned", "fineTunedRag"):
            with self.subTest(mode=mode):
                response = self.post(self.body_for(mode))
                self.assertEqual(response.status_code, 422, response.text)
                self.assertIn("modelVersion is required", response.json()["detail"])
        self.assertEqual(self.connection.statements, [])
        self.assert_nothing_generated()

    def test_rag_refuses_a_version_rather_than_ignoring_it(self) -> None:
        self.act_as(admin())
        response = self.post(self.body_for("rag", "v3"))

        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn("omit modelVersion", response.json()["detail"])
        self.assert_nothing_generated()

    def test_an_unknown_mode_is_422(self) -> None:
        self.act_as(admin())
        for mode in ("base", "fine-tuned", "FineTuned", ""):
            with self.subTest(mode=mode):
                response = self.post(self.body_for(mode, "v2"))
                self.assertEqual(response.status_code, 422, response.text)
        self.assert_nothing_generated()

    def test_a_malformed_course_id_is_400(self) -> None:
        self.act_as(admin())
        response = self.post({**self.body_for("fineTuned", "v2"), "courseId": "Bad_Id"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.connection.statements, [])
        self.assert_nothing_generated()

    def test_a_blank_question_is_422(self) -> None:
        self.act_as(admin())
        response = self.post({**self.body_for("fineTuned", "v2"), "question": "   "})

        self.assertEqual(response.status_code, 422)
        self.assert_nothing_generated()


# --------------------------------------------------------------------------- #
# 3. Nobody else
# --------------------------------------------------------------------------- #


class NobodyElseTests(ModelTestingTestCase):
    def test_without_a_session_it_is_401(self) -> None:
        self.act_as(None)
        self.assertEqual(self.post(self.body_for("fineTuned", "v3")).status_code, 401)
        self.assertEqual(self.connection.statements, [])
        self.assert_nothing_generated()

    def test_a_participant_of_any_course_is_401(self) -> None:
        for course_id in (COURSE, OTHER):
            with self.subTest(course=course_id):
                self.act_as(participant(course_id))
                self.assertEqual(self.post(self.body_for("fineTuned", "v3")).status_code, 401)
        self.assertEqual(self.connection.statements, [])
        self.assert_nothing_generated()

    def test_a_professor_is_403_even_for_their_own_course(self) -> None:
        """Deliberate: which version a course serves is an administrator's
        decision, and so is testing the one that might replace it."""
        for principal in (professor(COURSE), professor(OTHER)):
            with self.subTest(courses=sorted(principal.user.course_ids)):
                self.act_as(principal)
                for mode, version in (("fineTuned", "v3"), ("fineTunedRag", "v3"), ("rag", None)):
                    self.assertEqual(self.post(self.body_for(mode, version)).status_code, 403)
        self.assertEqual(self.connection.statements, [])
        self.assert_nothing_generated()

    def test_the_route_is_classified_administrator_only(self) -> None:
        self.assertEqual(CLASSIFICATION[("POST", PATH)], REQUIRE_ADMIN)

    def test_the_route_has_no_root_alias(self) -> None:
        """Unlike the classroom routes, nothing on the VM needs it outside
        `/api`, so it is not served there."""
        mounted = {getattr(route, "path", None) for route in app.routes}
        self.assertIn(PATH, mounted)
        self.assertNotIn("/model-testing/generate", mounted)


# --------------------------------------------------------------------------- #
# 4. Nothing is written
# --------------------------------------------------------------------------- #


class NothingIsWrittenTests(ModelTestingTestCase):
    def test_the_registry_is_read_and_not_touched(self) -> None:
        self.act_as(admin())
        response = self.post(self.body_for("fineTuned", "v3"))

        self.assertEqual(response.status_code, 200, response.text)
        # Exactly the two reads `get_model_registry` makes, for this course.
        self.assertEqual(len(self.connection.sql), 2)
        self.assertEqual(self.connection.statements[0], (MODEL_ROW_QUERY, (COURSE,)))
        self.assertTrue(self.connection.sql[1].startswith("SELECT"))
        self.assertEqual(self.writes(), [])
        self.create_participant.assert_not_called()
        self.create_session.assert_not_called()

    def test_no_cookie_is_set(self) -> None:
        self.act_as(admin())
        response = self.post(self.body_for("fineTuned", "v3"))
        self.assertEqual(response.headers.get_list("set-cookie"), [])

    def test_a_refused_version_writes_nothing_either(self) -> None:
        self.act_as(admin())
        self.assertEqual(self.post(self.body_for("fineTuned", "v9")).status_code, 409)
        self.assertEqual(self.writes(), [])


# --------------------------------------------------------------------------- #
# 5. The classroom routes are unchanged
# --------------------------------------------------------------------------- #


class ClassroomRoutesUnchangedTests(ModelTestingTestCase):
    """Same registry, same request shape, same answer as before.

    Both classroom fine-tuned routes are driven against the real
    `resolve_current_course_model` and the same recorded registry the tests
    above select v3 from. They resolve v2 — published, current — and a
    `modelVersion` in their body changes nothing, for an administrator or for
    anyone else. The explicit resolver is never consulted by them.
    """

    def setUp(self) -> None:
        super().setUp()
        self.explicit = self.patch("app.main.resolve_course_model_version")
        self.explicit_in_rag = self.patch("app.finetuned_rag.resolve_course_model_version")

    def classroom_body(self, **extra: Any) -> dict[str, Any]:
        return {"courseId": COURSE, "question": QUESTION, **extra}

    def test_the_fine_tuned_route_still_resolves_the_courses_version(self) -> None:
        for principal in (admin(), professor(COURSE), participant(COURSE)):
            with self.subTest(principal=principal):
                self.act_as(principal)
                self.registry(registry_results())
                response = self.post(
                    self.classroom_body(modelVersion="v3"), path="/api/fine-tuned/generate"
                )

                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["modelVersion"], "v2")
                self.assertEqual(self.fine_tuned.await_args.kwargs["model_version"], "v2")
                self.assertEqual(self.writes(), [])

        self.explicit.assert_not_called()
        self.explicit_in_rag.assert_not_called()

    def test_the_fine_tuned_rag_route_still_passes_no_version(self) -> None:
        for principal in (admin(), professor(COURSE), participant(COURSE)):
            with self.subTest(principal=principal):
                self.act_as(principal)
                response = self.post(
                    self.classroom_body(modelVersion="v3"), path="/api/fine-tuned-rag/generate"
                )

                self.assertEqual(response.status_code, 200, response.text)
                sent = self.fine_tuned_rag.await_args.kwargs
                self.assertNotIn("model_version", sent)
                self.assertEqual(sent["course_id"], COURSE)

        self.explicit.assert_not_called()
        self.explicit_in_rag.assert_not_called()

    def test_a_participant_cannot_select_a_version_anywhere(self) -> None:
        self.act_as(participant(COURSE))
        refused = self.post(self.body_for("fineTuned", "v3"))
        self.assertEqual(refused.status_code, 401)

        self.registry(registry_results())
        ignored = self.post(self.classroom_body(modelVersion="v3"), path="/api/fine-tuned/generate")
        self.assertEqual(ignored.status_code, 200, ignored.text)
        self.assertEqual(self.fine_tuned.await_args.kwargs["model_version"], "v2")

    def test_the_explicit_resolver_never_runs_for_a_classroom_request(self) -> None:
        self.act_as(admin())
        self.registry(registry_results())
        self.post(self.classroom_body(), path="/api/fine-tuned/generate")
        self.post(self.classroom_body(), path="/api/fine-tuned-rag/generate")

        self.explicit.assert_not_called()
        self.explicit_in_rag.assert_not_called()


if __name__ == "__main__":
    unittest.main()
