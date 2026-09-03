"""Public seed-export responses carry relative references, never machine paths.

The leak these cover: `seed_export` and `seed_split` work in absolute `Path`s
because they are writing files, and their summaries recorded what they wrote.
Those summaries reached the browser through `summary: dict[str, Any]` response
models, so `exportPath` was
`/Users/<developer>/css360-syllabus-bot/data/exports/<course>/approved-finetune.jsonl`
on a laptop and `/home/<account>/…` on the VM — an operator's account name on an
endpoint that needs no credential.

The frontend never wanted that form: `api.seedReview.test.ts` and
`AdminTrainingPage.test.tsx` have always fixed `exportPath` as
`data/exports/<courseId>/approved-finetune.jsonl`. These assert the backend now
returns what the frontend already expected, and that no absolute path — POSIX,
home-relative, or Windows — survives the boundary under any key.

Fixture accounts are synthetic (`testuser`, `alice`, `devuser`). Nothing here
writes a file, reads a real export, or contacts a database.
"""

from __future__ import annotations

import json
import unittest
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import export_privacy
from app.main import app
from app.seed_export import FinetuneJsonlValidationError
from app.seed_split import TrainingSplitError

COURSE = "css-350-spring-2026-n3h9"

OPERATOR_NAMES = ("testuser", "alice", "devuser")

OPERATOR_PATH_ROOTS = ("/Users/", "/home/", "/gpfs/projects/simswe/")

#: The three shapes an absolute path takes on the machines this runs on, plus
#: the Windows form, which no deployment uses but which the rule must still
#: refuse rather than pass through as "not a path we recognise".
MAC_ROOT = "/Users/devuser/css360-syllabus-bot"
LINUX_ROOT = "/home/testuser/css360-syllabus-bot"
WINDOWS_ROOT = "C:\\Users\\alice\\css360-syllabus-bot"

EXPECTED_EXPORT_REF = "data/exports/{0}/approved-finetune.jsonl".format(COURSE)


def assert_no_machine_paths(case: unittest.TestCase, body: Any) -> None:
    """No absolute path and no account name anywhere in the response."""
    text = json.dumps(body)
    for root in OPERATOR_PATH_ROOTS:
        case.assertNotIn(root, text)
    for name in OPERATOR_NAMES:
        case.assertNotIn(name, text)
    # Windows absolute paths, in both the raw and JSON-escaped spellings.
    case.assertNotIn("C:\\", text)
    case.assertNotIn("C:\\\\", text)
    case.assertNotIn("C:/", text)


class RelativeReferenceTests(unittest.TestCase):
    """The rule itself."""

    def test_a_path_in_this_checkout_becomes_repository_relative(self) -> None:
        self.assertEqual(
            export_privacy.repository_relative_path(
                "{0}/data/exports/{1}/approved-finetune.jsonl".format(MAC_ROOT, COURSE)
            ),
            EXPECTED_EXPORT_REF,
        )

    def test_the_same_holds_for_a_linux_home(self) -> None:
        self.assertEqual(
            export_privacy.repository_relative_path(
                "{0}/data/exports/{1}/train.jsonl".format(LINUX_ROOT, COURSE)
            ),
            "data/exports/{0}/train.jsonl".format(COURSE),
        )

    def test_a_windows_path_is_relativized_not_passed_through(self) -> None:
        self.assertEqual(
            export_privacy.repository_relative_path(
                "{0}\\data\\exports\\{1}\\train.jsonl".format(WINDOWS_ROOT, COURSE)
            ),
            "data/exports/{0}/train.jsonl".format(COURSE),
        )

    def test_a_cluster_output_uses_the_training_outputs_rule(self) -> None:
        """The same reduction `relative_training_output_ref` already applies."""
        self.assertEqual(
            export_privacy.repository_relative_path(
                "/gpfs/projects/simswe/testuser/training_outputs/qlora-runs/"
                "{0}/20260827T064701Z-full".format(COURSE)
            ),
            "qlora-runs/{0}/20260827T064701Z-full".format(COURSE),
        )

    def test_an_unrecognised_absolute_path_keeps_only_its_file_name(self) -> None:
        """Never an absolute path, even when no marker applies."""
        reference = export_privacy.repository_relative_path(
            "/var/lib/somewhere-else/stray.jsonl"
        )
        self.assertEqual(reference, "stray.jsonl")
        self.assertFalse(reference.startswith("/"))

    def test_a_relative_reference_is_returned_unchanged(self) -> None:
        """The identity case: what the frontend tests already expect."""
        self.assertEqual(
            export_privacy.repository_relative_path(EXPECTED_EXPORT_REF),
            EXPECTED_EXPORT_REF,
        )
        self.assertEqual(
            export_privacy.repository_relative_path("approved-finetune.jsonl"),
            "approved-finetune.jsonl",
        )

    def test_a_home_relative_path_is_reduced_too(self) -> None:
        self.assertEqual(
            export_privacy.repository_relative_path("~/exports/train.jsonl"),
            "train.jsonl",
        )

    def test_an_unlisted_key_is_still_covered(self) -> None:
        """The backstop: a renamed or new field cannot reopen the leak."""
        cleaned = export_privacy.public_export_summary(
            {"somethingNew": "{0}/data/exports/c/x.jsonl".format(MAC_ROOT)}
        )
        self.assertEqual(cleaned["somethingNew"], "data/exports/c/x.jsonl")

    def test_counts_and_text_are_left_alone(self) -> None:
        cleaned = export_privacy.public_export_summary(
            {
                "approvedCount": 42,
                "validationPassed": True,
                "note": "Only reviewStatus=approved seeds are exported.",
                "reviewStatusCounts": {"approved": 42, "rejected": 3},
            }
        )
        self.assertEqual(cleaned["approvedCount"], 42)
        self.assertTrue(cleaned["validationPassed"])
        self.assertEqual(
            cleaned["note"], "Only reviewStatus=approved seeds are exported."
        )
        self.assertEqual(cleaned["reviewStatusCounts"]["approved"], 42)

    def test_a_message_keeps_its_reason_and_names_the_file_relatively(self) -> None:
        message = export_privacy.public_message(
            "Approved export file does not exist: "
            "{0}/data/exports/{1}/approved-finetune.jsonl. "
            "Export approved seeds before preparing a training split.".format(
                MAC_ROOT, COURSE
            )
        )
        self.assertIn("does not exist", message)
        self.assertIn(EXPECTED_EXPORT_REF, message)
        self.assertNotIn("devuser", message)
        self.assertIn("Export approved seeds", message)

    def test_the_caller_s_own_summary_is_not_mutated(self) -> None:
        """A read-side view. The dict written to disk is not touched."""
        absolute = "{0}/data/exports/{1}/approved-finetune.jsonl".format(
            MAC_ROOT, COURSE
        )
        stored = {"exportPath": absolute, "files": {"finetuneJsonl": absolute}}
        export_privacy.public_export_summary(stored)
        self.assertEqual(stored["exportPath"], absolute)
        self.assertEqual(stored["files"]["finetuneJsonl"], absolute)


class ApprovedExportStatusRouteTests(unittest.TestCase):
    """`GET /api/courses/{id}/seeds/approved-export-status`."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_the_status_reports_a_relative_export_reference(self) -> None:
        patcher = patch(
            "app.main.approved_export_status",
            return_value={
                "courseId": COURSE,
                "exists": True,
                "exportPath": "{0}/data/exports/{1}/approved-finetune.jsonl".format(
                    MAC_ROOT, COURSE
                ),
                "exampleCount": 42,
                "sourceFile": "approved-finetune.jsonl",
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        response = self.client.get(
            "/api/courses/{0}/seeds/approved-export-status".format(COURSE)
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        assert_no_machine_paths(self, body)

        self.assertEqual(body["exportPath"], EXPECTED_EXPORT_REF)
        self.assertEqual(body["courseId"], COURSE)
        self.assertTrue(body["exists"])
        self.assertEqual(body["exampleCount"], 42)
        self.assertEqual(body["sourceFile"], "approved-finetune.jsonl")


class ExportApprovedRouteTests(unittest.TestCase):
    """`POST /api/courses/{id}/seeds/export-approved`."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        seeds_patch = patch("app.main._course_seed_records", return_value=[])
        seeds_patch.start()
        self.addCleanup(seeds_patch.stop)

    def _summary(self) -> dict[str, Any]:
        finetune = "{0}/data/exports/{1}/approved-finetune.jsonl".format(
            LINUX_ROOT, COURSE
        )
        return {
            "courseId": COURSE,
            "exportedAt": "2026-08-27T06:00:00+00:00",
            "inputCount": 45,
            "approvedCount": 42,
            "exportedCount": 42,
            "validatedCount": 42,
            "validationPassed": True,
            "exportPath": finetune,
            "skippedCount": 3,
            "reviewStatusCounts": {"approved": 42, "rejected": 3},
            "files": {
                "finetuneJsonl": finetune,
                "metadataJson": "{0}/data/exports/{1}/approved-metadata.json".format(
                    LINUX_ROOT, COURSE
                ),
                "summaryJson": (
                    "{0}/data/exports/{1}/approved-export-summary.json".format(
                        LINUX_ROOT, COURSE
                    )
                ),
            },
            "note": "Only reviewStatus=approved seeds are exported.",
        }

    def test_the_summary_is_relative_and_still_useful(self) -> None:
        patcher = patch("app.main.export_approved_seeds", return_value=self._summary())
        patcher.start()
        self.addCleanup(patcher.stop)

        response = self.client.post(
            "/api/courses/{0}/seeds/export-approved".format(COURSE)
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        assert_no_machine_paths(self, body)

        summary = body["summary"]
        self.assertEqual(summary["exportPath"], EXPECTED_EXPORT_REF)
        self.assertEqual(summary["files"]["finetuneJsonl"], EXPECTED_EXPORT_REF)
        self.assertEqual(
            summary["files"]["metadataJson"],
            "data/exports/{0}/approved-metadata.json".format(COURSE),
        )
        # The counts the Admin Training message is built from.
        self.assertEqual(summary["courseId"], COURSE)
        self.assertEqual(summary["approvedCount"], 42)
        self.assertEqual(summary["exportedCount"], 42)
        self.assertEqual(summary["validatedCount"], 42)
        self.assertTrue(summary["validationPassed"])
        self.assertEqual(summary["skippedCount"], 3)

    def test_a_validation_failure_names_the_file_without_the_home(self) -> None:
        patcher = patch(
            "app.main.export_approved_seeds",
            side_effect=FinetuneJsonlValidationError(
                "Approved JSONL validation failed: could not read "
                "{0}/data/exports/{1}/approved-finetune.jsonl "
                "(permission denied)".format(LINUX_ROOT, COURSE)
            ),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        response = self.client.post(
            "/api/courses/{0}/seeds/export-approved".format(COURSE)
        )
        self.assertEqual(response.status_code, 422)
        body = response.json()
        assert_no_machine_paths(self, body)
        self.assertIn("validation failed", body["detail"])
        self.assertIn(EXPECTED_EXPORT_REF, body["detail"])


class PrepareTrainingSplitRouteTests(unittest.TestCase):
    """`POST /api/courses/{id}/seeds/prepare-training-split`."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def _summary(self) -> dict[str, Any]:
        base = "{0}/data/exports/{1}".format(MAC_ROOT, COURSE)
        return {
            "courseId": COURSE,
            "manifest": {
                "courseId": COURSE,
                "datasetVersion": "{0}-approved-split-seed360-n42".format(COURSE),
                "sourceFile": "{0}/approved-finetune.jsonl".format(base),
                "sourceExportTimestamp": "2026-08-27T06:00:00+00:00",
                "createdAt": "2026-08-27T06:05:00+00:00",
                "splitSeed": 360,
                "totalExamples": 42,
                "trainExamples": 37,
                "validationExamples": 5,
                "trainFile": "{0}/train.jsonl".format(base),
                "validationFile": "{0}/validation.jsonl".format(base),
                "checksums": {"train.jsonl": "a" * 64, "validation.jsonl": "b" * 64},
                "checksumAlgorithm": "sha256",
            },
            "trainExamples": 37,
            "validationExamples": 5,
            "totalExamples": 42,
            "splitSeed": 360,
            "files": {
                "trainJsonl": "{0}/train.jsonl".format(base),
                "validationJsonl": "{0}/validation.jsonl".format(base),
                "manifestJson": "{0}/manifest.json".format(base),
                "sourceJsonl": "{0}/approved-finetune.jsonl".format(base),
            },
        }

    def test_the_nested_manifest_is_relativized_too(self) -> None:
        """Not just top-level fields — the manifest is one level down."""
        patcher = patch("app.main.prepare_training_split", return_value=self._summary())
        patcher.start()
        self.addCleanup(patcher.stop)

        response = self.client.post(
            "/api/courses/{0}/seeds/prepare-training-split".format(COURSE)
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        assert_no_machine_paths(self, body)

        summary = body["summary"]
        manifest = summary["manifest"]
        self.assertEqual(manifest["sourceFile"], EXPECTED_EXPORT_REF)
        self.assertEqual(
            manifest["trainFile"], "data/exports/{0}/train.jsonl".format(COURSE)
        )
        self.assertEqual(
            manifest["validationFile"],
            "data/exports/{0}/validation.jsonl".format(COURSE),
        )
        self.assertEqual(
            summary["files"]["trainJsonl"], "data/exports/{0}/train.jsonl".format(COURSE)
        )
        self.assertEqual(
            summary["files"]["manifestJson"],
            "data/exports/{0}/manifest.json".format(COURSE),
        )

    def test_the_split_facts_survive(self) -> None:
        patcher = patch("app.main.prepare_training_split", return_value=self._summary())
        patcher.start()
        self.addCleanup(patcher.stop)

        summary = self.client.post(
            "/api/courses/{0}/seeds/prepare-training-split".format(COURSE)
        ).json()["summary"]

        self.assertEqual(summary["trainExamples"], 37)
        self.assertEqual(summary["validationExamples"], 5)
        self.assertEqual(summary["totalExamples"], 42)
        self.assertEqual(summary["splitSeed"], 360)
        self.assertEqual(
            summary["manifest"]["datasetVersion"],
            "{0}-approved-split-seed360-n42".format(COURSE),
        )
        # The digests the cluster worker verifies against are untouched.
        self.assertEqual(summary["manifest"]["checksums"]["train.jsonl"], "a" * 64)
        self.assertEqual(summary["manifest"]["checksumAlgorithm"], "sha256")

    def test_a_missing_export_is_reported_without_the_home_directory(self) -> None:
        patcher = patch(
            "app.main.prepare_training_split",
            side_effect=TrainingSplitError(
                "Approved export file does not exist: "
                "{0}/data/exports/{1}/approved-finetune.jsonl. "
                "Export approved seeds before preparing a training split.".format(
                    MAC_ROOT, COURSE
                )
            ),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        response = self.client.post(
            "/api/courses/{0}/seeds/prepare-training-split".format(COURSE)
        )
        self.assertEqual(response.status_code, 422)
        body = response.json()
        assert_no_machine_paths(self, body)
        self.assertIn("does not exist", body["detail"])
        self.assertIn(EXPECTED_EXPORT_REF, body["detail"])


class StarterSnapshotRouteTests(unittest.TestCase):
    """`localSnapshotPath` on generate-starter and top-up."""

    def test_the_snapshot_reference_is_relative(self) -> None:
        absolute = "{0}/data/exports/{1}/generated-snapshot-20260827T060000Z.json".format(
            LINUX_ROOT, COURSE
        )
        reference = export_privacy.public_snapshot_ref(absolute)
        self.assertEqual(
            reference,
            "data/exports/{0}/generated-snapshot-20260827T060000Z.json".format(COURSE),
        )
        assert_no_machine_paths(self, {"localSnapshotPath": reference})

    def test_no_snapshot_stays_absent(self) -> None:
        """A run that saved nothing must not grow an empty string."""
        self.assertIsNone(export_privacy.public_snapshot_ref(None))
        self.assertIsNone(export_privacy.public_snapshot_ref(""))


if __name__ == "__main__":
    unittest.main()
