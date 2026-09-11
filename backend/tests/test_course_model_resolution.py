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
    UnservableCourseModelVersion,
    assert_valid_model_version,
    resolve_course_model_version,
    resolve_current_course_model,
)
from test_db_repositories import FakeConnection

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


class ExplicitVersionResolutionTests(ResolutionTestCase):
    """`resolve_course_model_version`: the administrator's model-testing path.

    CSS 360 as it stands while v3 is under test: v2 current and published, v3
    registered and ready, neither current nor published. The normal rule must
    keep answering v2 from this registry; the explicit rule must answer v3 when
    asked for v3, v2 when asked for v2, and never anything other than what it
    was asked for.
    """

    def _css360(self, *, v3_status: str = "ready") -> dict[str, Any]:
        return _registry(
            current="v2",
            course_id=OTHER_COURSE,
            versions={
                "v2": {
                    "version": "v2",
                    "baseModel": "meta-llama/Llama-3.2-3B-Instruct",
                    "status": "ready",
                    "deployment": "online",
                    "artifactRef": "css-360-qlora/v2/adapter",
                    "trainingExampleCount": 48,
                    "createdAt": "2026-08-27T07:00:00+00:00",
                },
                "v3": {
                    "version": "v3",
                    "baseModel": "meta-llama/Llama-3.2-3B-Instruct",
                    "status": v3_status,
                    "deployment": "offline",
                    "artifactRef": "css-360-qlora/v3/adapter",
                    "trainingExampleCount": 52,
                    "createdAt": "2026-09-09T07:00:00+00:00",
                },
            },
        )

    def test_a_ready_version_that_is_neither_current_nor_published_resolves(self) -> None:
        with self.registry(self._css360()):
            resolved = resolve_course_model_version(OTHER_COURSE, "v3")

        self.assertEqual(resolved["courseId"], OTHER_COURSE)
        self.assertEqual(resolved["version"], "v3")
        self.assertEqual(resolved["resolvedFrom"], "requested")
        self.assertEqual(resolved["artifactRef"], "css-360-qlora/v3/adapter")
        self.assertEqual(resolved["deployment"], "offline")
        # Reported as it stands, and not moved.
        self.assertEqual(resolved["currentVersion"], "v2")

    def test_the_serving_version_is_selectable_too(self) -> None:
        with self.registry(self._css360()):
            resolved = resolve_course_model_version(OTHER_COURSE, "v2")

        self.assertEqual(resolved["version"], "v2")
        self.assertEqual(resolved["resolvedFrom"], "requested")

    def test_the_normal_rule_is_unaffected_by_a_testable_version_existing(self) -> None:
        """The whole point: the same registry answers v2 for the classroom."""
        with self.registry(self._css360()):
            classroom = resolve_current_course_model(OTHER_COURSE)
            under_test = resolve_course_model_version(OTHER_COURSE, "v3")

        self.assertEqual((classroom["version"], classroom["resolvedFrom"]), ("v2", "published"))
        self.assertEqual((under_test["version"], under_test["resolvedFrom"]), ("v3", "requested"))

    def test_a_version_the_registry_does_not_have_is_refused_by_name(self) -> None:
        with self.registry(self._css360()):
            with self.assertRaises(UnservableCourseModelVersion) as caught:
                resolve_course_model_version(OTHER_COURSE, "v9")

        refusal = caught.exception
        self.assertEqual(refusal.status_code, 409)
        self.assertEqual((refusal.course_id, refusal.version), (OTHER_COURSE, "v9"))
        self.assertIn('"v9"', refusal.detail)
        self.assertIn("Registered versions: v2, v3", refusal.detail)

    def test_a_version_that_is_not_ready_is_refused(self) -> None:
        for status in ("training", "failed", "queued"):
            with self.subTest(status=status):
                with self.registry(self._css360(v3_status=status)):
                    with self.assertRaises(UnservableCourseModelVersion) as caught:
                        resolve_course_model_version(OTHER_COURSE, "v3")
                self.assertIn(f'"{status}", not "ready"', caught.exception.detail)

    def test_a_course_with_no_registry_is_refused(self) -> None:
        with self.registry(None):
            with self.assertRaises(UnservableCourseModelVersion) as caught:
                resolve_course_model_version(OTHER_COURSE, "v2")

        self.assertIn("no fine-tuned model registered", caught.exception.detail)

    def test_the_refusal_is_operator_facing_not_the_student_sentence(self) -> None:
        """Only an administrator ever reads it, and they can act on it."""
        with self.registry(self._css360()):
            with self.assertRaises(UnservableCourseModelVersion) as caught:
                resolve_course_model_version(OTHER_COURSE, "v9")

        self.assertNotEqual(caught.exception.detail, PUBLIC_UNAVAILABLE_DETAIL)
        self.assertIn(OTHER_COURSE, caught.exception.detail)

    def test_a_malformed_version_never_reaches_the_database(self) -> None:
        with patch("app.course_model_resolution.db_models.get_model_registry") as registry:
            for bad in ("V2", "2", "v2.1", "latest", "", " v2", "v-2", None, 2):
                with self.subTest(version=bad):
                    with self.assertRaises(ValueError):
                        resolve_course_model_version(OTHER_COURSE, bad)  # type: ignore[arg-type]
        registry.assert_not_called()

    def test_an_invalid_course_id_never_reaches_the_database(self) -> None:
        with patch("app.course_model_resolution.db_models.get_model_registry") as registry:
            with self.assertRaises(ValueError):
                resolve_course_model_version("../etc", "v2")
        registry.assert_not_called()

    def test_version_strings_are_v_and_digits(self) -> None:
        for good in ("v1", "v2", "v10", "v360"):
            self.assertEqual(assert_valid_model_version(good), good)

    def test_nothing_is_written(self) -> None:
        """The registry is read through the repository and not touched: no
        INSERT or UPDATE, in particular none of `current_version`."""
        connection = FakeConnection(
            [
                [{"course_id": OTHER_COURSE, "current_version": "v2"}],
                [
                    {
                        "version": version,
                        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
                        "training_example_count": 48,
                        "status": "ready",
                        "deployment": deployment,
                        "artifact_ref": f"css-360-qlora/{version}/adapter",
                        "created_at": "2026-09-09T07:00:00+00:00",
                        "updated_at": None,
                        "notes": None,
                        "run_id": None,
                        "provenance": None,
                    }
                    for version, deployment in (("v3", "offline"), ("v2", "online"))
                ],
            ]
        )

        @contextmanager
        def recording(**kwargs: Any) -> Iterator[FakeConnection]:
            yield connection

        with patch("app.course_model_resolution.db_connection", recording):
            resolved = resolve_course_model_version(OTHER_COURSE, "v3")

        self.assertEqual(resolved["version"], "v3")
        self.assertEqual(resolved["currentVersion"], "v2")
        self.assertTrue(connection.sql)
        self.assertTrue(all(sql.startswith("SELECT") for sql in connection.sql), connection.sql)


if __name__ == "__main__":
    unittest.main()
