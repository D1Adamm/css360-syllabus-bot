"""Resolving which fine-tuned model answers for a course.

This is the step that did not exist. A fine-tuned request carried a question and
nothing else; the service loaded whatever single adapter had last been promoted;
and "which model answered this?" had no answer anywhere in the system. With CSS
350 and CSS 360 both trained, that is not a reporting gap — it is one course
being answered by the other's adapter with nothing able to detect it.

PostgreSQL is the system of record for what a course's model is, so resolution
happens here and the version travels with the request to the cluster.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

from app.course_model_resolution import (
    PUBLIC_UNAVAILABLE_DETAIL,
    NoReadyCourseModel,
    resolve_current_course_model,
)

COURSE = "css-350-spring-2026-n3h9"
OTHER_COURSE = "css-360-winter-2026-a7rp"


def _registry(
    *,
    current: str = "v1",
    versions: dict[str, dict[str, Any]] | None = None,
    course_id: str = COURSE,
) -> dict[str, Any]:
    return {
        "courseId": course_id,
        "currentVersion": current,
        "versions": versions
        if versions is not None
        else {
            "v1": {
                "version": "v1",
                "baseModel": "meta-llama/Llama-3.2-3B-Instruct",
                "status": "ready",
                "deployment": "offline",
                "artifactRef": "qlora-runs/css-350-spring-2026-n3h9/x-full/adapter",
                "trainingExampleCount": 37,
                "createdAt": "2026-08-27T07:00:00+00:00",
            }
        },
    }


@contextmanager
def _fake_connection(**kwargs: Any) -> Iterator[object]:
    yield object()


class ResolutionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._connection = patch(
            "app.course_model_resolution.db_connection", _fake_connection
        )
        self._connection.start()
        self.addCleanup(self._connection.stop)

    @contextmanager
    def registry(self, value):
        with patch(
            "app.course_model_resolution.db_models.get_model_registry",
            return_value=value,
        ):
            yield


class ResolutionTests(ResolutionTestCase):
    def test_a_ready_course_resolves_its_current_version(self) -> None:
        with self.registry(_registry()):
            resolved = resolve_current_course_model(COURSE)

        self.assertEqual(resolved["courseId"], COURSE)
        self.assertEqual(resolved["version"], "v1")
        self.assertEqual(resolved["baseModel"], "meta-llama/Llama-3.2-3B-Instruct")

    def test_a_course_with_no_model_is_refused(self) -> None:
        with self.registry(None):
            with self.assertRaises(NoReadyCourseModel) as caught:
                resolve_current_course_model(COURSE)

        self.assertEqual(caught.exception.status_code, 409)
        self.assertIn("no fine-tuned model yet", caught.exception.diagnostic)

    def test_a_course_whose_current_version_is_not_ready_is_refused(self) -> None:
        """`ready` is the only status that may answer a student's question.

        A `training` or `failed` version is not a usable artifact, and falling
        back to the base model would answer a fine-tuned question with something
        that is not the fine-tuned model.
        """
        registry = _registry()
        registry["versions"]["v1"]["status"] = "training"

        with self.registry(registry):
            with self.assertRaises(NoReadyCourseModel) as caught:
                resolve_current_course_model(COURSE)

        self.assertIn("training", caught.exception.diagnostic)

    def test_a_dangling_current_version_is_refused(self) -> None:
        with self.registry(_registry(current="v9")):
            with self.assertRaises(NoReadyCourseModel) as caught:
                resolve_current_course_model(COURSE)

        self.assertIn("v9", caught.exception.diagnostic)

    def test_deployment_status_does_not_gate_resolution(self) -> None:
        """`ready` and `deployed` stay distinct concepts, in both directions.

        A model that nothing is currently serving is still the model this course
        would be answered by; whether a GPU is up is discovered at the
        connection, which is a truthful error rather than a stale one.
        """
        registry = _registry()
        registry["versions"]["v1"]["deployment"] = "offline"

        with self.registry(registry):
            resolved = resolve_current_course_model(COURSE)

        self.assertEqual(resolved["version"], "v1")
        self.assertEqual(resolved["deployment"], "offline")

    def test_each_course_resolves_only_its_own_registry(self) -> None:
        """CSS 350 must never resolve to CSS 360's version, and vice versa."""
        asked: list[str] = []

        def _registry_for(connection, course_id):
            asked.append(course_id)
            if course_id == COURSE:
                return _registry(current="v1")
            return _registry(
                current="v4",
                course_id=OTHER_COURSE,
                versions={
                    "v4": {
                        "version": "v4",
                        "baseModel": "meta-llama/Llama-3.2-3B-Instruct",
                        "status": "ready",
                        "deployment": "offline",
                        "artifactRef": "qlora-runs/css-360/y-full/adapter",
                        "trainingExampleCount": 54,
                        "createdAt": "2026-08-27T07:00:00+00:00",
                    }
                },
            )

        with patch(
            "app.course_model_resolution.db_models.get_model_registry",
            side_effect=_registry_for,
        ):
            first = resolve_current_course_model(COURSE)
            second = resolve_current_course_model(OTHER_COURSE)

        self.assertEqual(asked, [COURSE, OTHER_COURSE])
        self.assertEqual(first["version"], "v1")
        self.assertEqual(second["version"], "v4")
        self.assertNotEqual(first["artifactRef"], second["artifactRef"])

    def test_an_invalid_course_id_never_reaches_the_database(self) -> None:
        with patch(
            "app.course_model_resolution.db_models.get_model_registry"
        ) as registry:
            with self.assertRaises(ValueError):
                resolve_current_course_model("../etc")

        registry.assert_not_called()


class StudentFacingRefusalTests(ResolutionTestCase):
    """What a student reads when the fine-tuned model cannot answer.

    The refusal reaches the Compare page verbatim: it is a 4xx with no
    infrastructure vocabulary, so the browser shows it as written. The audit
    found it ending "Train one before asking the fine-tuned model a question",
    which is an instruction to an operator delivered to someone who cannot act
    on it. The body is now one neutral sentence for every reason the model is
    missing, and the reason itself goes to the log.
    """

    OPERATOR_WORDS = ("train", "register", "publish", "re-point", "version")

    def _refusals(self) -> list[NoReadyCourseModel]:
        dangling = _registry(current="v9")
        not_ready = _registry()
        not_ready["versions"]["v1"]["status"] = "training"
        empty = _registry(current="", versions={})
        caught: list[NoReadyCourseModel] = []
        for registry in (None, dangling, not_ready, empty):
            with self.registry(registry):
                with self.assertRaises(NoReadyCourseModel) as raised:
                    resolve_current_course_model(COURSE)
            caught.append(raised.exception)
        return caught

    def test_every_refusal_uses_the_same_neutral_wording(self) -> None:
        for refusal in self._refusals():
            self.assertEqual(refusal.status_code, 409)
            self.assertEqual(refusal.detail, PUBLIC_UNAVAILABLE_DETAIL)

    def test_the_public_wording_gives_a_student_nothing_to_do(self) -> None:
        lowered = PUBLIC_UNAVAILABLE_DETAIL.lower()
        for word in self.OPERATOR_WORDS:
            self.assertNotIn(word, lowered)
        self.assertNotIn(COURSE, PUBLIC_UNAVAILABLE_DETAIL)

    def test_the_specific_reason_is_kept_for_operators(self) -> None:
        no_registry, dangling, not_ready, empty = self._refusals()

        self.assertIn("no fine-tuned model yet", no_registry.diagnostic)
        self.assertIn("v9", dangling.diagnostic)
        self.assertIn("training", not_ready.diagnostic)
        self.assertIn("no model version", empty.diagnostic)
        # None of which reached the body.
        for refusal in (no_registry, dangling, not_ready, empty):
            self.assertNotIn("v9", refusal.detail)
            self.assertNotIn("training", refusal.detail)

    def test_the_reason_is_logged_with_the_course(self) -> None:
        with self.registry(_registry(current="v9")):
            with self.assertLogs("app.course_model_resolution", level="WARNING") as logs:
                with self.assertRaises(NoReadyCourseModel):
                    resolve_current_course_model(COURSE)

        record = "\n".join(logs.output)
        self.assertIn(COURSE, record)
        self.assertIn("v9", record)


if __name__ == "__main__":
    unittest.main()
