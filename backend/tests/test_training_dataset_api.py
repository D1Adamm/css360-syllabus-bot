"""The dataset transfer endpoints, from the worker's side of the wire.

These are what removed the manual step: prepare on the VM, `rsync` the export
over SSH with a second Duo prompt, walk back to Tillicum, re-run the worker
because it had refused the run for want of local files.

What has to hold for that to be safe unattended:

  - only the configured worker can read a dataset at all
  - a run is the unit of access, so naming a course you have no run for is a 404
  - the file set is fixed, so a name that is not in it never becomes a path
  - a body arrives with a digest attached, so a truncated transfer is detectable
"""

from __future__ import annotations

import hashlib
import json
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

COURSE = "css-350-spring-2026-n3h9"
OTHER_COURSE = "css-360-winter-2026-a7rp"
RUN_ID = "run-20260827t064701z-1cf650"
TOKEN = "test-worker-token"
HEADERS = {"X-Training-Worker-Token": TOKEN}


def _jsonl(count: int, prefix: str) -> str:
    return "".join(
        json.dumps({"instruction": f"{prefix}{index}", "response": f"A{index}"}) + "\n"
        for index in range(count)
    )


def _run(course_id: str = COURSE, **overrides: Any) -> dict[str, Any]:
    record = {
        "runId": RUN_ID,
        "courseId": course_id,
        "mode": "full",
        "state": "claimed",
        "enqueuedAt": "2026-08-27T06:47:01+00:00",
        "updatedAt": "2026-08-27T06:47:01+00:00",
        "datasetRef": f"exports/{course_id}",
        "approvedExampleCount": 42,
        "trainExamples": 37,
        "validationExamples": 5,
        "attempt": 1,
    }
    record.update(overrides)
    return record


@contextmanager
def _fake_connection(**kwargs: Any) -> Iterator[object]:
    yield object()


class DatasetApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        self._env = patch.dict("os.environ", {"TRAINING_WORKER_TOKEN": TOKEN})
        self._env.start()
        self.addCleanup(self._env.stop)

        self._connection = patch(
            "app.training_queue_routes.db_connection", _fake_connection
        )
        self._connection.start()
        self.addCleanup(self._connection.stop)

        # The export root the artifact module reads. Patched rather than written
        # into the real repository tree: these tests must not create files under
        # data/exports/ that a later run would then find.
        self._export_root = patch(
            "app.dataset_artifacts.course_export_dir",
            side_effect=lambda course_id, root=None: self.root / course_id,
        )
        self._export_root.start()
        self.addCleanup(self._export_root.stop)

        self.prepare(COURSE)

    def prepare(self, course_id: str, *, train: int = 37, validation: int = 5) -> Path:
        export_dir = self.root / course_id
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "train.jsonl").write_text(_jsonl(train, "Q"), encoding="utf-8")
        (export_dir / "validation.jsonl").write_text(
            _jsonl(validation, "V"), encoding="utf-8"
        )
        (export_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "courseId": course_id,
                    "datasetVersion": f"{course_id}-approved-split-seed360-n{train + validation}",
                    "trainExamples": train,
                    "validationExamples": validation,
                    "checksums": {
                        "train.jsonl": hashlib.sha256(
                            _jsonl(train, "Q").encode("utf-8")
                        ).hexdigest(),
                    },
                }
            ),
            encoding="utf-8",
        )
        return export_dir

    @contextmanager
    def run_exists(self, run: dict[str, Any] | None) -> Iterator[None]:
        with patch(
            "app.training_queue_routes.db_training_runs.get_training_run",
            return_value=run,
        ):
            yield


class DatasetDescriptionTests(DatasetApiTestCase):
    def test_the_descriptor_names_files_counts_and_digests(self) -> None:
        with self.run_exists(_run()):
            response = self.client.get(
                f"/api/training-queue/courses/{COURSE}/runs/{RUN_ID}/dataset",
                headers=HEADERS,
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["courseId"], COURSE)
        self.assertEqual(body["runId"], RUN_ID)
        self.assertEqual(body["trainExamples"], 37)
        self.assertEqual(body["validationExamples"], 5)
        self.assertEqual(body["approvedExampleCount"], 42)
        self.assertEqual(len(body["datasetSha256"]), 64)
        self.assertEqual(
            sorted(item["name"] for item in body["files"]),
            ["manifest.json", "train.jsonl", "validation.jsonl"],
        )

    def test_an_unauthenticated_caller_gets_nothing(self) -> None:
        with self.run_exists(_run()):
            response = self.client.get(
                f"/api/training-queue/courses/{COURSE}/runs/{RUN_ID}/dataset"
            )

        self.assertEqual(response.status_code, 401)

    def test_a_wrong_token_gets_nothing(self) -> None:
        with self.run_exists(_run()):
            response = self.client.get(
                f"/api/training-queue/courses/{COURSE}/runs/{RUN_ID}/dataset",
                headers={"X-Training-Worker-Token": "not-the-token"},
            )

        self.assertEqual(response.status_code, 401)

    def test_a_run_that_does_not_belong_to_this_course_is_a_404(self) -> None:
        """The run is the unit of access, not the course.

        Without this, anyone holding the worker token could read any course's
        prepared dataset by naming it. With it, they can read the dataset for a
        run that exists for that course — which is what a worker executing that
        run needs and nothing more.
        """
        self.prepare(OTHER_COURSE)
        with self.run_exists(None):
            response = self.client.get(
                f"/api/training-queue/courses/{OTHER_COURSE}/runs/{RUN_ID}/dataset",
                headers=HEADERS,
            )

        self.assertEqual(response.status_code, 404)

    def test_an_unprepared_course_is_a_409_naming_the_missing_step(self) -> None:
        with self.run_exists(_run(OTHER_COURSE)):
            response = self.client.get(
                f"/api/training-queue/courses/{OTHER_COURSE}/runs/{RUN_ID}/dataset",
                headers=HEADERS,
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("prepared", response.json()["detail"].lower())

    def test_a_run_pointing_at_another_courses_export_is_refused(self) -> None:
        with self.run_exists(_run(datasetRef=f"exports/{OTHER_COURSE}")):
            response = self.client.get(
                f"/api/training-queue/courses/{COURSE}/runs/{RUN_ID}/dataset",
                headers=HEADERS,
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn(OTHER_COURSE, response.json()["detail"])


class DatasetDownloadTests(DatasetApiTestCase):
    def _download(self, name: str, *, course_id: str = COURSE, headers=HEADERS):
        return self.client.get(
            f"/api/training-queue/courses/{course_id}/runs/{RUN_ID}/dataset/files/{name}",
            headers=headers,
        )

    def test_a_file_arrives_with_a_matching_digest_header(self) -> None:
        with self.run_exists(_run()):
            response = self._download("train.jsonl")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.content.splitlines()), 37)
        self.assertEqual(
            response.headers["x-content-sha256"],
            hashlib.sha256(response.content).hexdigest(),
        )

    def test_the_body_is_byte_identical_to_what_was_prepared(self) -> None:
        expected = (self.root / COURSE / "train.jsonl").read_bytes()

        with self.run_exists(_run()):
            response = self._download("train.jsonl")

        self.assertEqual(response.content, expected)

    def test_every_allowlisted_file_is_downloadable(self) -> None:
        for name in ("train.jsonl", "validation.jsonl", "manifest.json"):
            with self.subTest(name=name):
                with self.run_exists(_run()):
                    response = self._download(name)
                self.assertEqual(response.status_code, 200)

    def test_a_name_outside_the_allowlist_is_a_404(self) -> None:
        for name in ("approved-finetune.jsonl", "manifest.json.bak", "secrets"):
            with self.subTest(name=name):
                with self.run_exists(_run()):
                    response = self._download(name)
                self.assertEqual(response.status_code, 404)

    def test_a_traversal_attempt_reaches_no_file_outside_the_export(self) -> None:
        """Nothing outside the course's export directory is ever served.

        The forms below are handled at two different layers, and both are fine.
        An encoded `..` arrives at the handler as a name, is compared against
        the allowlist, and is refused. A literal `..` is normalised by the URL
        layer before routing and lands on a different route entirely — the
        descriptor — so it never reaches file resolution at all.

        The assertion is therefore about the outcome rather than the status
        code: whatever path a traversal attempt takes through the stack, the
        file it was reaching for does not come back.
        """
        secret = self.root / "secret.txt"
        secret.write_text("not for the cluster", encoding="utf-8")

        for name in ("..%2Fsecret.txt", "%2E%2E%2Fsecret.txt", "..", "%2e%2e"):
            with self.subTest(name=name):
                with self.run_exists(_run()):
                    response = self._download(name)
                self.assertNotIn(b"not for the cluster", response.content)

    def test_an_encoded_traversal_name_is_refused_outright(self) -> None:
        """The forms that do reach the handler are refused by the allowlist."""
        for name in ("..%2Fsecret.txt", "%2E%2E%2Fsecret.txt"):
            with self.subTest(name=name):
                with self.run_exists(_run()):
                    response = self._download(name)
                self.assertEqual(response.status_code, 404)

    def test_downloading_needs_the_worker_token(self) -> None:
        with self.run_exists(_run()):
            response = self._download("train.jsonl", headers={})

        self.assertEqual(response.status_code, 401)

    def test_downloading_for_a_run_that_is_not_this_courses_is_a_404(self) -> None:
        self.prepare(OTHER_COURSE)
        with self.run_exists(None):
            response = self._download("train.jsonl", course_id=OTHER_COURSE)

        self.assertEqual(response.status_code, 404)

    def test_a_prepared_file_that_is_missing_is_a_409(self) -> None:
        (self.root / COURSE / "validation.jsonl").unlink()

        with self.run_exists(_run()):
            response = self._download("validation.jsonl")

        self.assertEqual(response.status_code, 409)


class UnconfiguredBackendTests(unittest.TestCase):
    def test_the_dataset_routes_are_closed_when_no_token_is_configured(self) -> None:
        """An unconfigured deployment serves nothing, rather than serving openly."""
        client = TestClient(app)
        response = client.get(
            f"/api/training-queue/courses/{COURSE}/runs/{RUN_ID}/dataset",
            headers=HEADERS,
        )
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
