"""Telling the application which version the cluster is actually serving.

Publication spans two machines: the adapter is copied into Tillicum's serving
tree, and PostgreSQL has to learn that it happened, because inference resolves
the published version from there.

The ordering is the safety property, and it is asymmetric on purpose:

    copy succeeds, report succeeds  -> the new version serves
    copy succeeds, report fails     -> the OLD version keeps serving, and the
                                       report is on disk for the next queue run
    copy fails                      -> nothing is reported at all

The one state that must be impossible is the database naming a version the
cluster does not have, because that routes every question for the course at an
adapter that is not there. That is prevented by never reporting before the copy
has landed and been validated — so these tests are largely about proving the
middle row is survivable and the last row is silent.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "lib"))

import training_state  # noqa: E402
from training_queue import TrainingQueueError  # noqa: E402


def _load_reporter() -> Any:
    path = REPO_ROOT / "scripts" / "report_model_published.py"
    spec = importlib.util.spec_from_file_location("report_model_published", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reporter = _load_reporter()

CSS350 = "css-350-spring-2026-n3h9"
CSS360 = "css-360-winter-2026-a7rp"


class FakeQueue:
    """Records what would be sent, and can be told to fail like a real outage."""

    def __init__(self, *, fail: bool = False, result: Optional[Dict[str, Any]] = None):
        self.fail = fail
        self.result = result or {
            "courseId": CSS350,
            "version": "v2",
            "deployment": "online",
            "currentVersion": "v2",
            "previousVersion": "v1",
            "unchanged": False,
        }
        self.calls: List[Dict[str, Any]] = []

    def report_model_published(
        self,
        course_id: str,
        version: str,
        *,
        source_ref: Optional[str] = None,
        published_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.fail:
            raise TrainingQueueError("Could not reach the backend API: timed out")
        self.calls.append(
            {
                "courseId": course_id,
                "version": version,
                "sourceRef": source_ref,
                "publishedAt": published_at,
            }
        )
        return self.result


class ReporterTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def pending(self) -> List[Dict[str, Any]]:
        return training_state.list_pending_callbacks(self.root)


class PayloadTests(ReporterTestCase):
    def test_the_payload_names_the_course_and_version(self) -> None:
        payload = reporter.build_payload(course_id=CSS350, version="v2")

        self.assertEqual(payload["courseId"], CSS350)
        self.assertEqual(payload["version"], "v2")
        self.assertTrue(payload["publishedAt"].endswith("Z"))

    def test_a_bad_version_is_refused_before_anything_is_written(self) -> None:
        """This value becomes a URL segment and a filesystem path segment."""
        with self.assertRaises(TrainingQueueError):
            reporter.build_payload(course_id=CSS350, version="../../etc")

    def test_a_bad_course_id_is_refused(self) -> None:
        with self.assertRaises(TrainingQueueError):
            reporter.build_payload(course_id="../secrets", version="v1")

    def test_the_pending_key_is_stable_for_a_course_and_version(self) -> None:
        """So re-publishing overwrites its own entry rather than queueing a second."""
        self.assertEqual(
            reporter.pending_key(CSS350, "v2"), reporter.pending_key(CSS350, "v2")
        )
        self.assertNotEqual(
            reporter.pending_key(CSS350, "v2"), reporter.pending_key(CSS350, "v1")
        )
        self.assertNotEqual(
            reporter.pending_key(CSS350, "v2"), reporter.pending_key(CSS360, "v2")
        )


class DeliveryTests(ReporterTestCase):
    def test_a_delivered_report_leaves_nothing_pending(self) -> None:
        queue = FakeQueue()

        outcome = reporter.report_publication(
            course_id=CSS350, version="v2", repo_root=self.root, queue=queue
        )

        self.assertTrue(outcome["delivered"])
        self.assertEqual(queue.calls[0]["courseId"], CSS350)
        self.assertEqual(queue.calls[0]["version"], "v2")
        self.assertEqual(self.pending(), [])

    def test_the_source_reference_travels_with_it(self) -> None:
        queue = FakeQueue()

        reporter.report_publication(
            course_id=CSS350,
            version="v2",
            source_ref=f"qlora-runs/{CSS350}/20260902T080000Z-full/adapter",
            repo_root=self.root,
            queue=queue,
        )

        self.assertIn("qlora-runs", queue.calls[0]["sourceRef"])


class OutageTests(ReporterTestCase):
    def test_an_undeliverable_report_is_persisted_rather_than_lost(self) -> None:
        """The adapter is published. Only the application does not know yet."""
        outcome = reporter.report_publication(
            course_id=CSS350,
            version="v2",
            repo_root=self.root,
            queue=FakeQueue(fail=True),
        )

        self.assertFalse(outcome["delivered"])
        pending = self.pending()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["kind"], "published")
        self.assertEqual(pending[0]["payload"]["version"], "v2")

    def test_the_payload_is_written_before_the_attempt_not_after_it(self) -> None:
        """A process killed mid-send must not take the event with it.

        Asserted by having the fake observe the filesystem at the moment it is
        called: if persistence happened only on failure, the file would not
        exist yet here.
        """
        seen: List[int] = []
        root = self.root

        class ObservingQueue(FakeQueue):
            def report_model_published(self, *args: Any, **kwargs: Any):
                seen.append(len(training_state.list_pending_callbacks(root)))
                return super().report_model_published(*args, **kwargs)

        reporter.report_publication(
            course_id=CSS350, version="v2", repo_root=self.root, queue=ObservingQueue()
        )

        self.assertEqual(seen, [1])

    def test_a_queued_report_can_be_delivered_later(self) -> None:
        """What the next `run_training_queue.sh --once` does with it."""
        reporter.report_publication(
            course_id=CSS350,
            version="v2",
            repo_root=self.root,
            queue=FakeQueue(fail=True),
        )
        queued = self.pending()[0]

        queue = FakeQueue()
        reporter.send_publication(queue, queued["payload"], repo_root=self.root)

        self.assertEqual(queue.calls[0]["version"], "v2")
        self.assertEqual(self.pending(), [])

    def test_republishing_the_same_version_does_not_queue_two_reports(self) -> None:
        for _ in range(3):
            reporter.report_publication(
                course_id=CSS350,
                version="v2",
                repo_root=self.root,
                queue=FakeQueue(fail=True),
            )

        self.assertEqual(len(self.pending()), 1)

    def test_two_courses_queue_separately(self) -> None:
        reporter.report_publication(
            course_id=CSS350,
            version="v2",
            repo_root=self.root,
            queue=FakeQueue(fail=True),
        )
        reporter.report_publication(
            course_id=CSS360,
            version="v1",
            repo_root=self.root,
            queue=FakeQueue(fail=True),
        )

        courses = sorted(entry["courseId"] for entry in self.pending())
        self.assertEqual(courses, [CSS350, CSS360])


class CliTests(ReporterTestCase):
    def test_a_dry_run_writes_and_sends_nothing(self) -> None:
        code = reporter.main(
            ["--course-id", CSS350, "--version", "v2", "--dry-run"]
        )

        self.assertEqual(code, 0)
        self.assertEqual(training_state.list_pending_callbacks(REPO_ROOT), [])

    def test_a_malformed_version_exits_nonzero_without_writing(self) -> None:
        code = reporter.main(["--course-id", CSS350, "--version", "newest"])

        self.assertEqual(code, 2)


class PublicationOrderingTests(ReporterTestCase):
    """The property the shell script is responsible for, asserted on the script.

    The reporter is only ever invoked after the copy has landed and been
    validated. If that call moved above the copy, a failed publication could
    make the database claim a version the cluster does not have — the one state
    this whole design exists to prevent.
    """

    def setUp(self) -> None:
        super().setUp()
        self.script = (REPO_ROOT / "training" / "promote_qlora_adapter.sh").read_text(
            encoding="utf-8"
        )

    def test_the_report_comes_after_the_published_adapter_is_validated(self) -> None:
        validate_at = self.script.index('validate-adapter-source "${DEST_ADAPTER}"')
        report_at = self.script.index("scripts/report_model_published.py")

        self.assertLess(validate_at, report_at)

    def test_the_report_comes_after_the_pointer_is_moved_into_place(self) -> None:
        pointer_at = self.script.index('mv "${POINTER_TMP}" "${POINTER}"')
        report_at = self.script.index("scripts/report_model_published.py")

        self.assertLess(pointer_at, report_at)

    def test_a_failed_report_does_not_fail_the_publication(self) -> None:
        """Exiting nonzero would make an operator re-run a publish that worked."""
        tail = self.script[self.script.index("scripts/report_model_published.py") :]

        self.assertIn("The adapter IS published", tail)

    def test_publication_can_be_reported_separately(self) -> None:
        self.assertIn("--no-report", self.script)


if __name__ == "__main__":
    unittest.main()
