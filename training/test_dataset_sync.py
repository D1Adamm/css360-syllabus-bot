"""Fetching a run's dataset onto the cluster, safely and repeatably.

What replaced the manual rsync. The tests are about the three properties that
make an unattended transfer safe rather than about whether bytes move:

  - identical local data means no transfer at all, so a retried run is cheap
  - nothing partial ever replaces good data, at any point of interruption
  - a file that does not match its digest is refused, because a truncated
    training set trains successfully and produces a worse model silently
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "lib"))

import dataset_sync  # noqa: E402

COURSE = "css-350-spring-2026-n3h9"
RUN_ID = "run-20260827t064701z-1cf650"


def _jsonl(count: int, prefix: str = "Q") -> str:
    return "".join(
        json.dumps({"instruction": f"{prefix}{index}", "response": f"A{index}"}) + "\n"
        for index in range(count)
    )


def _files(train: int = 37, validation: int = 5) -> dict:
    return {
        "train.jsonl": _jsonl(train, "Q"),
        "validation.jsonl": _jsonl(validation, "V"),
        "manifest.json": json.dumps({"courseId": COURSE, "trainExamples": train}),
    }


def _descriptor(files: dict, *, train: int = 37, validation: int = 5) -> dict:
    entries = [
        {
            "name": name,
            "bytes": len(body.encode("utf-8")),
            "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        }
        for name, body in sorted(files.items())
    ]
    return {
        "courseId": COURSE,
        "runId": RUN_ID,
        "trainExamples": train,
        "validationExamples": validation,
        "datasetSha256": dataset_sync.dataset_digest(
            {item["name"]: item["sha256"] for item in entries}
        ),
        "files": entries,
    }


class FakeQueue:
    """Only the two calls `sync_run_dataset` makes."""

    def __init__(self, files: dict, *, corrupt: dict | None = None, fail_on: str | None = None):
        self.files = dict(files)
        self.corrupt = corrupt or {}
        self.fail_on = fail_on
        self.downloads: list[str] = []
        self.described = 0

    def describe_dataset(self, course_id: str, run_id: str) -> dict:
        self.described += 1
        return _descriptor(self.files)

    def download_dataset_file(self, course_id: str, run_id: str, name: str) -> bytes:
        self.downloads.append(name)
        if self.fail_on == name:
            raise OSError("connection reset mid-transfer")
        body = self.corrupt.get(name, self.files[name])
        return body.encode("utf-8")


class DatasetSyncTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.export_dir = self.root / "data" / "exports" / COURSE

    def write_local(self, files: dict) -> None:
        self.export_dir.mkdir(parents=True, exist_ok=True)
        for name, body in files.items():
            (self.export_dir / name).write_text(body, encoding="utf-8")

    def sync(self, queue) -> dataset_sync.DatasetSyncResult:
        return dataset_sync.sync_run_dataset(queue, COURSE, RUN_ID, self.export_dir)


class FreshDownloadTests(DatasetSyncTestCase):
    def test_a_missing_local_dataset_is_downloaded_and_verified(self) -> None:
        queue = FakeQueue(_files())

        result = self.sync(queue)

        self.assertFalse(result.skipped)
        self.assertEqual(sorted(result.downloaded), sorted(_files()))
        self.assertEqual(result.train_count, 37)
        self.assertEqual(result.validation_count, 5)
        self.assertEqual(
            (self.export_dir / "train.jsonl").read_text(encoding="utf-8"),
            _files()["train.jsonl"],
        )

    def test_the_reported_digest_matches_what_was_installed(self) -> None:
        queue = FakeQueue(_files())
        result = self.sync(queue)

        installed = dataset_sync.local_file_digests(
            self.export_dir, ["train.jsonl", "validation.jsonl", "manifest.json"]
        )
        self.assertEqual(result.dataset_sha256, dataset_sync.dataset_digest(installed))

    def test_no_staging_directory_is_left_behind(self) -> None:
        self.sync(FakeQueue(_files()))

        leftovers = list(self.export_dir.parent.glob(f"{dataset_sync.STAGING_PREFIX}*"))
        self.assertEqual(leftovers, [])


class SkipTests(DatasetSyncTestCase):
    def test_an_identical_local_copy_is_not_downloaded_again(self) -> None:
        """The common case for a retried run: one request, no bytes moved."""
        files = _files()
        self.write_local(files)
        queue = FakeQueue(files)

        result = self.sync(queue)

        self.assertTrue(result.skipped)
        self.assertEqual(queue.downloads, [])
        self.assertEqual(result.train_count, 37)

    def test_a_changed_local_file_forces_a_full_refetch(self) -> None:
        """A directory where two files match and one does not is not a dataset.

        Treating it as a partial hit would mean choosing which of two
        disagreeing versions to keep. There is no defensible answer, so the
        whole set is fetched again.
        """
        files = _files()
        self.write_local(files)
        (self.export_dir / "train.jsonl").write_text(_jsonl(3), encoding="utf-8")

        queue = FakeQueue(files)
        result = self.sync(queue)

        self.assertFalse(result.skipped)
        self.assertEqual(sorted(queue.downloads), sorted(files))
        self.assertEqual(result.train_count, 37)

    def test_a_missing_local_file_forces_a_refetch(self) -> None:
        files = _files()
        self.write_local(files)
        (self.export_dir / "manifest.json").unlink()

        result = self.sync(FakeQueue(files))

        self.assertFalse(result.skipped)
        self.assertTrue((self.export_dir / "manifest.json").is_file())

    def test_syncing_twice_is_a_download_then_a_skip(self) -> None:
        queue = FakeQueue(_files())

        first = self.sync(queue)
        second = self.sync(queue)

        self.assertFalse(first.skipped)
        self.assertTrue(second.skipped)
        self.assertEqual(len(queue.downloads), 3)


class CorruptionTests(DatasetSyncTestCase):
    def test_a_file_that_does_not_match_its_digest_is_refused(self) -> None:
        files = _files()
        queue = FakeQueue(files, corrupt={"train.jsonl": _jsonl(3)})

        with self.assertRaises(dataset_sync.DatasetSyncError) as caught:
            self.sync(queue)

        self.assertIn("Checksum mismatch", str(caught.exception))

    def test_a_corrupt_download_leaves_the_previous_data_untouched(self) -> None:
        """The property that makes an interrupted transfer harmless.

        Everything is verified in staging and only then replaced, so a failure
        at any point leaves the previous dataset exactly as it was. There is no
        window in which `train.jsonl` is half a file.
        """
        good = _files()
        self.write_local(good)
        original = (self.export_dir / "train.jsonl").read_text(encoding="utf-8")

        newer = _files(train=40)
        queue = FakeQueue(newer, corrupt={"validation.jsonl": "truncated"})

        with self.assertRaises(dataset_sync.DatasetSyncError):
            self.sync(queue)

        self.assertEqual(
            (self.export_dir / "train.jsonl").read_text(encoding="utf-8"), original
        )

    def test_a_transport_failure_leaves_no_staging_behind(self) -> None:
        queue = FakeQueue(_files(), fail_on="validation.jsonl")

        with self.assertRaises(dataset_sync.DatasetSyncError):
            self.sync(queue)

        leftovers = list(self.export_dir.parent.glob(f"{dataset_sync.STAGING_PREFIX}*"))
        self.assertEqual(leftovers, [])

    def test_a_failed_sync_can_simply_be_retried(self) -> None:
        files = _files()
        failing = FakeQueue(files, corrupt={"train.jsonl": "bad"})
        with self.assertRaises(dataset_sync.DatasetSyncError):
            self.sync(failing)

        result = self.sync(FakeQueue(files))

        self.assertFalse(result.skipped)
        self.assertEqual(result.train_count, 37)

    def test_a_stale_staging_directory_is_not_resumed_from(self) -> None:
        """Its contents were never verified, so it is not a resource to resume.

        A partially populated staging directory from a killed process must not
        contribute a single file to the next attempt.
        """
        stale = self.export_dir.parent / f"{dataset_sync.STAGING_PREFIX}-{COURSE}-1"
        stale.mkdir(parents=True)
        (stale / "train.jsonl").write_text("garbage", encoding="utf-8")

        result = self.sync(FakeQueue(_files()))

        self.assertEqual(result.train_count, 37)
        self.assertNotIn(
            "garbage", (self.export_dir / "train.jsonl").read_text(encoding="utf-8")
        )


class DescriptorValidationTests(DatasetSyncTestCase):
    def test_a_descriptor_naming_a_file_outside_the_set_is_refused(self) -> None:
        """A name arriving over the network never becomes a path on its own."""
        descriptor = _descriptor(_files())
        descriptor["files"].append(
            {"name": "../../.ssh/authorized_keys", "sha256": "a" * 64, "bytes": 1}
        )

        with self.assertRaises(dataset_sync.DatasetSyncError) as caught:
            dataset_sync.validate_descriptor(descriptor)

        self.assertIn("will not write", str(caught.exception))

    def test_a_descriptor_with_no_usable_checksum_is_refused(self) -> None:
        """An unverifiable file is not installed, however well formed it is."""
        descriptor = _descriptor(_files())
        descriptor["files"][0]["sha256"] = "short"

        with self.assertRaises(dataset_sync.DatasetSyncError) as caught:
            dataset_sync.validate_descriptor(descriptor)

        self.assertIn("checksum", str(caught.exception))

    def test_a_descriptor_missing_a_required_file_is_refused(self) -> None:
        files = _files()
        del files["validation.jsonl"]

        with self.assertRaises(dataset_sync.DatasetSyncError) as caught:
            dataset_sync.validate_descriptor(_descriptor(files))

        self.assertIn("validation.jsonl", str(caught.exception))

    def test_an_empty_descriptor_is_refused(self) -> None:
        with self.assertRaises(dataset_sync.DatasetSyncError):
            dataset_sync.validate_descriptor({"files": []})


class CountVerificationTests(DatasetSyncTestCase):
    def test_counts_that_disagree_with_the_descriptor_are_refused(self) -> None:
        """The check the digests cannot make.

        Digests prove the bytes arrived intact. They cannot prove that the
        backend described the same thing it prepared, and a dataset whose
        described count is wrong is one whose two records disagree.
        """
        files = _files(train=37)
        queue = FakeQueue(files)
        queue.describe_dataset = lambda course_id, run_id: {  # type: ignore[assignment]
            **_descriptor(files),
            "trainExamples": 999,
        }

        with self.assertRaises(dataset_sync.DatasetSyncError) as caught:
            self.sync(queue)

        self.assertIn("999", str(caught.exception))
        self.assertFalse((self.export_dir / "train.jsonl").exists())

    def test_an_empty_training_set_is_refused(self) -> None:
        files = {
            "train.jsonl": "",
            "validation.jsonl": _jsonl(2, "V"),
            "manifest.json": json.dumps({"courseId": COURSE}),
        }
        queue = FakeQueue(files)
        queue.describe_dataset = lambda course_id, run_id: {  # type: ignore[assignment]
            **_descriptor(files),
            "trainExamples": 0,
            "validationExamples": 2,
        }

        with self.assertRaises(dataset_sync.DatasetSyncError) as caught:
            self.sync(queue)

        self.assertIn("empty", str(caught.exception))


class ManifestChecksumTests(DatasetSyncTestCase):
    def test_manifest_checksums_are_read_when_present(self) -> None:
        self.write_local(
            {
                **_files(),
                "manifest.json": json.dumps(
                    {"courseId": COURSE, "checksums": {"train.jsonl": "a" * 64}}
                ),
            }
        )

        checksums = dataset_sync.read_manifest_checksums(self.export_dir)

        self.assertEqual(checksums, {"train.jsonl": "a" * 64})

    def test_a_manifest_without_checksums_is_not_an_error(self) -> None:
        """Manifests written before checksums existed still work.

        The descriptor is the authority during transfer; the manifest block is
        for describing the dataset to whoever reads it later on the cluster.
        """
        self.write_local(_files())

        self.assertEqual(dataset_sync.read_manifest_checksums(self.export_dir), {})


if __name__ == "__main__":
    unittest.main()
