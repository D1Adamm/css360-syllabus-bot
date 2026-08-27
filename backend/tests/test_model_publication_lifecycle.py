"""Registered, published, and which one answers a question.

The outage this closes
----------------------
CSS 350 had `v1` registered and published, and it was serving. An admin trained
a new version. The completion callback registered `v2` and moved
`current_version` to it — correct, because `current_version` is what a professor
is shown and their newest model is the honest answer to "what is my model".

But `v2` was not on the cluster. Publishing is a deliberate, separate act, and
nobody had done it yet. Resolution read `current_version`, sent `v2` to the
cluster, and the cluster — holding only `v1` — refused with a 409. Every
fine-tuned answer for CSS 350 failed, including the ones `v1` had been serving
correctly a minute earlier.

Training a new version took the old one offline. That is what these test.

The two facts, kept apart
-------------------------
    status = ready      a usable adapter exists somewhere
    deployment = online this version is in the cluster's serving tree

Only the second can answer a question, so only the second is resolved from.
`current_version` remains the fallback for a course that has never published,
which is every course from before publication was reported at all.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from app.course_model_resolution import (
    NoReadyCourseModel,
    resolve_current_course_model,
    select_servable_version,
)

CSS350 = "css-350-spring-2026-n3h9"
CSS360 = "css-360-winter-2026-a7rp"


def _version(
    key: str,
    *,
    status: str = "ready",
    deployment: str = "offline",
    artifact: str | None = None,
) -> dict[str, Any]:
    return {
        "version": key,
        "baseModel": "meta-llama/Llama-3.2-3B-Instruct",
        "trainingExampleCount": 37,
        "status": status,
        "deployment": deployment,
        "artifactRef": artifact or f"serving/{CSS350}/{key}/adapter",
        "createdAt": "2026-08-27T06:48:00+00:00",
    }


@contextmanager
def _fake_connection():
    yield object()


def _registry(current: str, versions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "courseId": CSS350,
        "currentVersion": current,
        "versions": {item["version"]: item for item in versions},
    }


class SelectServableVersionTests(unittest.TestCase):
    """The rule, in isolation. No database, no HTTP."""

    def test_the_published_version_wins_over_a_newer_registered_one(self) -> None:
        """The whole fix, in one assertion.

        v1 published and serving, v2 registered an hour ago and not published.
        A question must be answered by v1.
        """
        registry = _registry(
            "v2",
            [
                _version("v1", deployment="online"),
                _version("v2", deployment="offline"),
            ],
        )

        self.assertEqual(select_servable_version(registry), ("v1", "published"))

    def test_current_version_answers_when_nothing_has_been_published(self) -> None:
        """The fallback, and the reason it exists.

        Every course from before publication reporting is in this state. They
        must keep answering exactly as they did — a rule that refused them all
        would be a worse outage than the one being fixed.
        """
        registry = _registry("v1", [_version("v1")])

        self.assertEqual(select_servable_version(registry), ("v1", "current"))

    def test_publishing_the_newer_version_moves_the_answer(self) -> None:
        registry = _registry(
            "v2",
            [
                _version("v1", deployment="offline"),
                _version("v2", deployment="online"),
            ],
        )

        self.assertEqual(select_servable_version(registry), ("v2", "published"))

    def test_the_fallback_does_not_apply_once_anything_is_published(self) -> None:
        """Not a general safety net.

        A course with a published v1 and a newer current v2 resolves v1 — the
        fallback must not quietly reintroduce the bug by preferring the newer
        version whenever it looks more current.
        """
        registry = _registry(
            "v2",
            [
                _version("v1", deployment="online"),
                _version("v2", deployment="offline"),
            ],
        )

        version, source = select_servable_version(registry)

        self.assertEqual(version, "v1")
        self.assertNotEqual(source, "current")

    def test_two_online_rows_resolve_deterministically(self) -> None:
        """`mark_version_published` prevents this; an older write may not have."""
        registry = _registry(
            "v2",
            [
                _version("v1", deployment="online"),
                _version("v2", deployment="online"),
            ],
        )

        self.assertEqual(select_servable_version(registry), ("v2", "published"))

    def test_versions_sort_numerically_not_lexically(self) -> None:
        registry = _registry(
            "v10",
            [
                _version("v9", deployment="online"),
                _version("v10", deployment="online"),
            ],
        )

        self.assertEqual(select_servable_version(registry)[0], "v10")

    def test_a_registry_with_nothing_usable_resolves_nothing(self) -> None:
        self.assertEqual(
            select_servable_version({"currentVersion": None, "versions": {}}),
            (None, "none"),
        )


class ResolutionTestCase(unittest.TestCase):
    """Through the real resolver, with the registry read stubbed."""

    def setUp(self) -> None:
        self.registries: dict[str, Any] = {}

        self._connection = patch(
            "app.course_model_resolution.db_connection",
            lambda **kwargs: _fake_connection(),
        )
        self._connection.start()
        self.addCleanup(self._connection.stop)

        self._registry = patch(
            "app.course_model_resolution.db_models.get_model_registry",
            side_effect=lambda connection, course_id: self.registries.get(course_id),
        )
        self._registry.start()
        self.addCleanup(self._registry.stop)


class TheRetrainLifecycleTests(ResolutionTestCase):
    """Step by step, in the order Stage B will actually happen."""

    def _publish(self, registry: dict[str, Any], version: str) -> dict[str, Any]:
        """What `mark_version_published` does, applied to a registry dict."""
        for record in registry["versions"].values():
            record["deployment"] = "offline"
        registry["versions"][version]["deployment"] = "online"
        return registry

    def test_the_whole_sequence(self) -> None:
        # 1. v1 exists, registered and published. It is what serves.
        self.registries[CSS350] = self._publish(
            _registry("v1", [_version("v1")]), "v1"
        )

        resolved = resolve_current_course_model(CSS350)
        self.assertEqual(resolved["version"], "v1")
        self.assertEqual(resolved["resolvedFrom"], "published")

        # 2-3. A new run finishes. v2 registers ready/offline and becomes the
        # course's current version — the professor's newest model.
        registry = self.registries[CSS350]
        registry["versions"]["v2"] = _version("v2")
        registry["currentVersion"] = "v2"

        # 4. Inference STILL resolves v1. This is the assertion the outage was.
        resolved = resolve_current_course_model(CSS350)
        self.assertEqual(resolved["version"], "v1")
        self.assertEqual(resolved["currentVersion"], "v2")

        # 5. v1 is untouched by any of it.
        self.assertEqual(registry["versions"]["v1"]["status"], "ready")
        self.assertEqual(registry["versions"]["v1"]["deployment"], "online")

        # 6. An admin deliberately publishes v2.
        self._publish(registry, "v2")

        # 7. Inference now resolves v2.
        resolved = resolve_current_course_model(CSS350)
        self.assertEqual(resolved["version"], "v2")
        self.assertEqual(resolved["resolvedFrom"], "published")

        # v1 stays registered and ready. Only its deployment changed.
        self.assertEqual(registry["versions"]["v1"]["status"], "ready")
        self.assertEqual(registry["versions"]["v1"]["deployment"], "offline")

    def test_registering_v2_alone_never_changes_what_answers(self) -> None:
        """Isolated from the sequence above, because it is the regression."""
        self.registries[CSS350] = self._publish(
            _registry("v1", [_version("v1")]), "v1"
        )
        before = resolve_current_course_model(CSS350)["version"]

        registry = self.registries[CSS350]
        registry["versions"]["v2"] = _version("v2")
        registry["currentVersion"] = "v2"

        self.assertEqual(resolve_current_course_model(CSS350)["version"], before)

    def test_the_artifact_reference_follows_the_resolved_version(self) -> None:
        """What is sent to the cluster describes v1, not v2."""
        registry = self._publish(
            _registry("v2", [_version("v1"), _version("v2")]), "v1"
        )
        self.registries[CSS350] = registry

        resolved = resolve_current_course_model(CSS350)

        self.assertEqual(resolved["artifactRef"], f"serving/{CSS350}/v1/adapter")
        self.assertEqual(resolved["deployment"], "online")

    def test_a_published_version_that_is_not_ready_is_refused(self) -> None:
        """Both facts have to hold. Published-but-broken answers nothing."""
        registry = self._publish(
            _registry("v1", [_version("v1", status="failed")]), "v1"
        )
        self.registries[CSS350] = registry

        with self.assertRaises(NoReadyCourseModel):
            resolve_current_course_model(CSS350)

    def test_a_course_with_no_registry_is_refused(self) -> None:
        with self.assertRaises(NoReadyCourseModel) as caught:
            resolve_current_course_model(CSS350)

        self.assertIn("no fine-tuned model yet", str(caught.exception.detail))


class CourseIsolationTests(ResolutionTestCase):
    """Publication is per course, and stays that way."""

    def test_publishing_for_one_course_does_not_move_another(self) -> None:
        self.registries[CSS350] = {
            "courseId": CSS350,
            "currentVersion": "v1",
            "versions": {"v1": _version("v1", deployment="online")},
        }
        self.registries[CSS360] = {
            "courseId": CSS360,
            "currentVersion": "v1",
            "versions": {
                "v1": _version(
                    "v1",
                    deployment="online",
                    artifact=f"serving/{CSS360}/v1/adapter",
                )
            },
        }

        # CSS 350 trains and publishes v2.
        self.registries[CSS350]["versions"]["v2"] = _version("v2")
        self.registries[CSS350]["currentVersion"] = "v2"
        self.registries[CSS350]["versions"]["v1"]["deployment"] = "offline"
        self.registries[CSS350]["versions"]["v2"]["deployment"] = "online"

        self.assertEqual(resolve_current_course_model(CSS350)["version"], "v2")
        # CSS 360 is exactly where it was.
        css360 = resolve_current_course_model(CSS360)
        self.assertEqual(css360["version"], "v1")
        self.assertEqual(css360["artifactRef"], f"serving/{CSS360}/v1/adapter")

    def test_a_course_with_no_model_is_never_given_another_courses(self) -> None:
        self.registries[CSS350] = self._css350_published()

        with self.assertRaises(NoReadyCourseModel) as caught:
            resolve_current_course_model(CSS360)

        self.assertNotIn("350", str(caught.exception.detail))

    def _css350_published(self) -> dict[str, Any]:
        registry = _registry("v1", [_version("v1")])
        registry["versions"]["v1"]["deployment"] = "online"
        return registry


if __name__ == "__main__":
    unittest.main()
