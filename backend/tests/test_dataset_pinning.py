"""A queued run trains on the dataset it was queued for (audit finding B10).

The run records the prepared dataset's digest when it is queued; the worker is
refused the data when the files on disk no longer match; and the export and
split routes refuse to rewrite the files while a run is outstanding.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_db_repositories import FakeConnection  # noqa: E402

from app import db_training_runs  # noqa: E402
from app.auth.dependencies import current_principal  # noqa: E402
from app.auth.principal import Principal, StaffUser  # noqa: E402
from app.dataset_artifacts import DatasetChangedError, describe_dataset, resolve_run_dataset  # noqa: E402
from app.main import app  # noqa: E402


def admin() -> Principal:
    return Principal(
        user=StaffUser(user_id="u1", email="a@uw.edu", display_name="A", role="admin", session_id="s")
    )

COURSE = "css-360-winter-2026-a7rp"
RUN_ID = "run-20261103t000000z-abc123"


def _write_dataset(root: Path, course_id: str, *, rows: int = 3) -> Path:
    directory = root / course_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "train.jsonl").write_text(
        "".join(json.dumps({"instruction": f"Q{i}", "response": "A"}) + "\n" for i in range(rows)),
        encoding="utf-8",
    )
    (directory / "validation.jsonl").write_text(
        json.dumps({"instruction": "Qv", "response": "A"}) + "\n", encoding="utf-8"
    )
    (directory / "manifest.json").write_text(
        json.dumps({"datasetVersion": "x", "splitSeed": 360, "trainExamples": rows, "validationExamples": 1}),
        encoding="utf-8",
    )
    return directory


def _run(**overrides: Any) -> dict[str, Any]:
    run = {
        "runId": RUN_ID,
        "courseId": COURSE,
        "state": "queued",
        "datasetRef": f"exports/{COURSE}",
        "approvedExampleCount": 4,
        "trainExamples": 3,
        "validationExamples": 1,
    }
    run.update(overrides)
    return run


class ResolveRunDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        _write_dataset(self.root, COURSE)
        self._patch = patch(
            "app.dataset_artifacts.course_export_dir",
            side_effect=lambda course_id, root=None: self.root / course_id,
        )
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def test_a_run_pinned_to_the_current_files_is_served(self) -> None:
        digest = describe_dataset(COURSE)["datasetSha256"]
        described = resolve_run_dataset(_run(datasetSha256=digest))
        self.assertEqual(described["pinnedDatasetSha256"], digest)
        self.assertEqual(described["datasetSha256"], digest)

    def test_a_run_whose_files_were_re_prepared_is_refused(self) -> None:
        digest = describe_dataset(COURSE)["datasetSha256"]
        _write_dataset(self.root, COURSE, rows=3)  # same counts...
        (self.root / COURSE / "train.jsonl").write_text(
            "".join(json.dumps({"instruction": f"Q{i}", "response": "B"}) + "\n" for i in range(3)),
            encoding="utf-8",
        )  # ...different content
        with self.assertRaises(DatasetChangedError) as ctx:
            resolve_run_dataset(_run(datasetSha256=digest))
        self.assertEqual(ctx.exception.run_id, RUN_ID)
        self.assertIn("changed after this run was queued", str(ctx.exception))
        self.assertNotEqual(ctx.exception.pinned, ctx.exception.current)

    def test_a_run_queued_before_pinning_existed_is_served_as_before(self) -> None:
        described = resolve_run_dataset(_run())
        self.assertIsNone(described["pinnedDatasetSha256"])
        self.assertEqual(described["trainExamples"], 3)


class EnqueueTests(unittest.TestCase):
    def test_the_digest_is_written_onto_the_run(self) -> None:
        connection = FakeConnection([1, [{
            "run_id": RUN_ID, "course_id": COURSE, "mode": "full", "state": "queued",
            "enqueued_at": "2026-11-03T00:00:00+00:00", "updated_at": "2026-11-03T00:00:00+00:00",
            "dataset_ref": f"exports/{COURSE}", "approved_example_count": 4, "train_examples": 3,
            "validation_examples": 1, "attempt": 0, "job_id": None, "claim_owner": None,
            "claim_claimed_at": None, "claim_expires_at": None, "error": None, "completion": None,
            "dataset_sha256": "f" * 64,
        }]])
        created = db_training_runs.enqueue_training_run(
            connection, COURSE, mode="full", dataset_ref=f"exports/{COURSE}",
            train_examples=3, validation_examples=1, dataset_sha256="f" * 64,
        )
        inserted = connection.params_for("INSERT INTO training_runs")
        self.assertEqual(inserted["dataset_sha256"], "f" * 64)
        self.assertEqual(created["datasetSha256"], "f" * 64)

    def test_an_older_row_maps_without_the_field(self) -> None:
        record = db_training_runs.map_training_run({
            "run_id": RUN_ID, "course_id": COURSE, "mode": "full", "state": "queued",
            "enqueued_at": "2026-11-03T00:00:00+00:00", "updated_at": None, "dataset_ref": "x",
            "approved_example_count": 0, "train_examples": 0, "validation_examples": 0, "attempt": 0,
        })
        self.assertNotIn("datasetSha256", record)


class WorkerRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self._env = patch.dict("os.environ", {"TRAINING_WORKER_TOKEN": "tok"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_the_worker_gets_a_409_with_the_dataset_changed_code(self) -> None:
        with patch("app.training_queue_routes._run", side_effect=lambda label, fn: _run(datasetSha256="a" * 64)), patch(
            "app.training_queue_routes.resolve_run_dataset",
            side_effect=DatasetChangedError(RUN_ID, "a" * 64, "b" * 64),
        ):
            response = self.client.get(
                f"/api/training-queue/courses/{COURSE}/runs/{RUN_ID}/dataset",
                headers={"X-Training-Worker-Token": "tok"},
            )
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["detail"]["code"], "dataset_changed")
        self.assertIn("aaaaaaaaaaaa", response.json()["detail"]["message"])


class PreparationGuardTests(unittest.TestCase):
    """Export and split refuse while a worker holds the prepared files.

    A claimed, submitted or training run is using them; a merely queued run is
    pinned to its digest and goes stale instead, so it does not block.
    """

    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[current_principal] = admin
        self.addCleanup(lambda: app.dependency_overrides.pop(current_principal, None))

    def _post(self, path: str):
        return self.client.post(path, json={}, headers={"X-Requested-With": "SyllabusModelLab"})

    def test_export_is_refused_while_a_run_is_claimed(self) -> None:
        with patch("app.main.db_connection", side_effect=lambda **kwargs: _idle()), patch(
            "app.main.find_active_training_run", return_value=_run(state="claimed")
        ), patch("app.main.export_approved_seeds") as export:
            response = self._post(f"/api/courses/{COURSE}/seeds/export-approved")
        self.assertEqual(response.status_code, 409, response.text)
        detail = response.json()["detail"]
        self.assertEqual(detail["code"], "training_run_active")
        self.assertEqual(detail["runId"], RUN_ID)
        export.assert_not_called()

    def test_export_proceeds_while_a_run_is_only_queued(self) -> None:
        """The queued run is pinned to its digest; re-preparing makes it stale, not retargeted."""
        with patch("app.main.db_connection", side_effect=lambda **kwargs: _idle()), patch(
            "app.main.find_active_training_run", return_value=_run(state="queued")
        ), patch("app.main._course_seed_records", return_value=[]), patch(
            "app.main.export_approved_seeds", return_value={"courseId": COURSE, "approvedCount": 0}
        ) as export:
            response = self._post(f"/api/courses/{COURSE}/seeds/export-approved")
        self.assertEqual(response.status_code, 200, response.text)
        export.assert_called_once()

    def test_split_is_refused_while_a_run_is_submitted(self) -> None:
        with patch("app.main.db_connection", side_effect=lambda **kwargs: _idle()), patch(
            "app.main.find_active_training_run", return_value=_run(state="submitted")
        ), patch("app.main.prepare_training_split") as split:
            response = self._post(f"/api/courses/{COURSE}/seeds/prepare-training-split")
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["detail"]["code"], "training_run_active")
        split.assert_not_called()

    def test_export_proceeds_when_no_run_is_outstanding(self) -> None:
        with patch("app.main.db_connection", side_effect=lambda **kwargs: _idle()), patch(
            "app.main.find_active_training_run", return_value=None
        ), patch("app.main._course_seed_records", return_value=[]), patch(
            "app.main.export_approved_seeds", return_value={"courseId": COURSE, "approvedCount": 0}
        ) as export:
            response = self._post(f"/api/courses/{COURSE}/seeds/export-approved")
        self.assertEqual(response.status_code, 200, response.text)
        export.assert_called_once()


from contextlib import contextmanager  # noqa: E402


@contextmanager
def _idle():
    yield FakeConnection()


class BrowserEnqueueRouteTests(unittest.TestCase):
    """`POST /api/courses/{id}/training-runs` — the route the browser's Queue
    training button calls — pins the run exactly as the `/api/db` twin does."""

    BODY = {
        "mode": "full",
        "datasetRef": f"exports/{COURSE}",
        "approvedExampleCount": 4,
        "trainExamples": 3,
        "validationExamples": 1,
    }

    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[current_principal] = admin
        self.addCleanup(lambda: app.dependency_overrides.pop(current_principal, None))
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        _write_dataset(self.root, COURSE)
        self._export_dir = patch(
            "app.dataset_artifacts.course_export_dir",
            side_effect=lambda course_id, root=None: self.root / course_id,
        )
        self._export_dir.start()
        self.addCleanup(self._export_dir.stop)

    def _post(self, body: dict[str, Any]):
        calls: list[dict[str, Any]] = []

        def _enqueue(connection: Any, course_id: str, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return _run(datasetSha256=kwargs.get("dataset_sha256"), mode="full",
                        enqueuedAt="2026-11-03T00:00:00+00:00", updatedAt="2026-11-03T00:00:00+00:00",
                        attempt=0)

        with patch("app.main.db_connection", side_effect=lambda **kwargs: _idle()), patch(
            "app.main.course_exists", return_value=True
        ), patch("app.main.enqueue_training_run", side_effect=_enqueue):
            response = self.client.post(
                f"/api/courses/{COURSE}/training-runs",
                json=body,
                headers={"X-Requested-With": "SyllabusModelLab"},
            )
        return response, calls

    def test_the_run_is_pinned_to_the_files_on_disk(self) -> None:
        response, calls = self._post(self.BODY)
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["dataset_sha256"], describe_dataset(COURSE)["datasetSha256"])

    def test_stale_counts_are_refused_before_anything_is_queued(self) -> None:
        response, calls = self._post({**self.BODY, "trainExamples": 2})
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["detail"]["code"], "dataset_changed")
        self.assertEqual(calls, [])

    def test_another_courses_dataset_ref_is_refused(self) -> None:
        response, calls = self._post({**self.BODY, "datasetRef": "exports/css-350-spring-2026-n3h9"})
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
