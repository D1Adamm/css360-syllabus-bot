"""Training launch boundary.

No test here reaches the cluster. The subprocess runner is injected, so sync and
submission are simulated and every branch — including both failure paths — is
exercised without ssh, rsync, or sbatch ever running.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import training_launch
from app.main import app
from app.training_launch import (
    LaunchDisabledError,
    LaunchExecutionError,
    LaunchValidationError,
    describe_capability,
    launch_training,
    validate_prepared_export,
)

COURSE_490 = "css-490-spring-2026-cgvl"
COURSE_360 = "css-360-winter-2026-a7rp"
COURSE_350 = "css-350-winter-2026-drlb"

SUBMITTED = "Submitted batch job 9182736"


@pytest.fixture(autouse=True)
def enable_launch(monkeypatch):
    monkeypatch.setenv("TRAINING_LAUNCH_ENABLED", "1")
    monkeypatch.setenv("TILLICUM_LOGIN", "tester@tillicum.hyak.uw.edu")
    monkeypatch.setenv("TILLICUM_REPO_ROOT", "/gpfs/projects/simswe/tester/css360-syllabus-bot")


@pytest.fixture
def exports(tmp_path, monkeypatch):
    """Point the module at a temporary project root with prepared data."""
    monkeypatch.setattr(training_launch, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        training_launch, "SYNC_SCRIPT", tmp_path / "scripts" / "sync.sh"
    )
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / "sync.sh").write_text("#!/bin/sh\nexit 0\n")

    def prepare(course_id: str, *, train: int = 40, validation: int = 4) -> Path:
        out = tmp_path / "data" / "exports" / course_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "train.jsonl").write_text(
            "\n".join(
                json.dumps({"instruction": f"{course_id} q{i}", "response": f"a{i}"})
                for i in range(train)
            )
            + "\n"
        )
        (out / "validation.jsonl").write_text(
            "\n".join(
                json.dumps({"instruction": f"{course_id} vq{i}", "response": f"va{i}"})
                for i in range(validation)
            )
            + "\n"
        )
        (out / "manifest.json").write_text(json.dumps({"courseId": course_id}))
        return out

    return prepare


def recorder(results):
    """Runner that records argv and replays canned results in order."""
    calls: list[list[str]] = []

    def run(command, *, timeout):
        calls.append(list(command))
        result = results[min(len(calls) - 1, len(results) - 1)]
        return subprocess.CompletedProcess(command, result[0], result[1], result[2])

    run.calls = calls  # type: ignore[attr-defined]
    return run


def ok_runner():
    return recorder([(0, "", ""), (0, SUBMITTED, "")])


class TestPreparedArtifactValidation:
    def test_refuses_when_nothing_prepared(self, exports):
        with pytest.raises(LaunchValidationError, match="No prepared training data"):
            validate_prepared_export(COURSE_490)

    def test_refuses_when_train_file_missing(self, exports):
        out = exports(COURSE_490)
        (out / "train.jsonl").unlink()
        with pytest.raises(LaunchValidationError):
            validate_prepared_export(COURSE_490)

    def test_accepts_a_real_prepared_export(self, exports):
        exports(COURSE_490, train=40, validation=4)
        counts = validate_prepared_export(COURSE_490)
        assert counts == {"trainCount": 40, "validationCount": 4}

    def test_launch_refuses_before_preparation(self, exports):
        runner = ok_runner()
        with pytest.raises(LaunchValidationError):
            launch_training(COURSE_490, runner=runner)
        # Nothing was synced or submitted.
        assert runner.calls == []


class TestSuccessfulLaunch:
    def test_captures_the_real_slurm_job_id(self, exports):
        exports(COURSE_490)
        result = launch_training(COURSE_490, runner=ok_runner())

        assert result.job_id == "9182736"
        assert result.mode == "full"
        assert result.train_count == 40
        assert result.validation_count == 4
        assert result.submitted_at.endswith("Z")

    def test_reuses_both_existing_scripts(self, exports):
        exports(COURSE_490)
        runner = ok_runner()
        launch_training(COURSE_490, runner=runner)

        sync_call, submit_call = runner.calls
        # Sync first, then submit.
        assert sync_call[0].endswith("sync.sh")
        assert "--yes" in sync_call
        assert submit_call[0] == "ssh"
        # The existing launcher owns versioning and the live-adapter guard.
        assert "training/start_qlora_training.sh" in submit_call[2]
        assert "--full" in submit_call[2]

    def test_smoke_mode_is_available(self, exports):
        exports(COURSE_490)
        runner = ok_runner()
        launch_training(COURSE_490, mode="smoke", runner=runner)
        assert "--smoke" in runner.calls[1][2]

    def test_rejects_an_unknown_mode(self, exports):
        exports(COURSE_490)
        with pytest.raises(LaunchValidationError, match="Unknown training mode"):
            launch_training(COURSE_490, mode="turbo", runner=ok_runner())


class TestCourseIsolation:
    def test_only_the_requested_course_appears_anywhere(self, exports):
        # All three courses have prepared data sitting on disk.
        exports(COURSE_490)
        exports(COURSE_360)
        exports(COURSE_350)

        runner = ok_runner()
        launch_training(COURSE_490, runner=runner)

        flat = " ".join(" ".join(call) for call in runner.calls)
        assert COURSE_490 in flat
        assert COURSE_360 not in flat
        assert COURSE_350 not in flat

    def test_counts_come_from_the_requested_course_only(self, exports):
        exports(COURSE_490, train=7, validation=1)
        exports(COURSE_360, train=999, validation=99)

        result = launch_training(COURSE_490, runner=ok_runner())
        assert result.train_count == 7
        assert result.validation_count == 1

    def test_rejects_an_unsafe_course_id(self, exports):
        runner = ok_runner()
        with pytest.raises(ValueError):
            launch_training("../css-360-winter-2026-a7rp", runner=runner)
        assert runner.calls == []


class TestFailureHandling:
    def test_sync_failure_never_submits(self, exports):
        exports(COURSE_490)
        runner = recorder([(1, "", "rsync: connection closed")])

        with pytest.raises(LaunchExecutionError, match="Syncing training data failed"):
            launch_training(COURSE_490, runner=runner)

        # Only the sync ran; nothing was submitted.
        assert len(runner.calls) == 1

    def test_submission_failure_is_reported(self, exports):
        exports(COURSE_490)
        runner = recorder([(0, "", ""), (1, "", "sbatch: error: Invalid account")])

        with pytest.raises(LaunchExecutionError, match="Submitting the training job failed"):
            launch_training(COURSE_490, runner=runner)

    def test_an_already_active_job_is_not_reported_as_a_new_submission(self, exports):
        exports(COURSE_490)
        # The launcher exits 0 in this case, which must not look like success.
        runner = recorder(
            [(0, "", ""), (0, "Existing active css360-qlora-train job found\nJob ID: 5", "")]
        )

        with pytest.raises(LaunchExecutionError, match="already active"):
            launch_training(COURSE_490, runner=runner)

    def test_missing_job_id_is_a_failure(self, exports):
        exports(COURSE_490)
        runner = recorder([(0, "", ""), (0, "submitted something, who knows", "")])

        with pytest.raises(LaunchExecutionError, match="did not report a Slurm job ID"):
            launch_training(COURSE_490, runner=runner)


class TestCapabilityGate:
    def test_disabled_by_default(self, exports, monkeypatch):
        monkeypatch.delenv("TRAINING_LAUNCH_ENABLED", raising=False)
        exports(COURSE_490)

        assert describe_capability().enabled is False
        runner = ok_runner()
        with pytest.raises(LaunchDisabledError):
            launch_training(COURSE_490, runner=runner)
        # Disabled means nothing runs at all.
        assert runner.calls == []

    def test_requires_cluster_configuration(self, exports, monkeypatch):
        monkeypatch.setenv("TILLICUM_LOGIN", "")
        monkeypatch.setenv("USER", "")
        assert describe_capability().enabled is False


class TestEndpoints:
    def test_capability_endpoint_reports_disabled(self, monkeypatch):
        monkeypatch.delenv("TRAINING_LAUNCH_ENABLED", raising=False)
        client = TestClient(app)

        response = client.get("/api/training/launch-capability")
        assert response.status_code == 200
        assert response.json()["enabled"] is False
        assert "TRAINING_LAUNCH_ENABLED" in response.json()["reason"]

    def test_launch_endpoint_rejects_a_bad_course_id(self):
        client = TestClient(app)
        response = client.post("/api/courses/Bad_Id/training/launch", json={})
        assert response.status_code == 400

    def test_launch_endpoint_reports_disabled_as_503(self, monkeypatch):
        monkeypatch.delenv("TRAINING_LAUNCH_ENABLED", raising=False)
        client = TestClient(app)

        response = client.post(f"/api/courses/{COURSE_490}/training/launch", json={})
        assert response.status_code == 503

    def test_launch_endpoint_reports_unprepared_as_422(self, exports):
        client = TestClient(app)
        response = client.post(f"/api/courses/{COURSE_490}/training/launch", json={})
        assert response.status_code == 422
        assert "prepared" in response.json()["detail"].lower()
