"""Operator filesystem paths must not leave the browser-facing `/api/db` routes.

The leak these cover: a completion callback stores `resolved_config.json`
verbatim on the model version, and that file records where the job actually ran
— `/gpfs/projects/simswe/<netid>/training_outputs/…`. Every `/api/db` route is
reachable without a credential, so the course model endpoint was publishing the
operator's UW NetID to anyone who could name a course.

Both halves are asserted, because either one alone would be satisfied by a
broken fix: the useful provenance is still there, and no absolute path or
operator username is.

Fixture accounts are synthetic (`testuser`, `alice`). Nothing here reads a real
run, a real database, or the machine's own environment.
"""

from __future__ import annotations

import json
import unittest
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import db_training_runs, provenance_privacy
from app.main import app

COURSE = "css-350-spring-2026-n3h9"

#: The operator identities that must never appear in a public response.
OPERATOR_NETIDS = ("testuser", "alice")

#: Path roots that carry an account name on the machines this project runs on.
OPERATOR_PATH_ROOTS = ("/gpfs/projects/simswe/", "/home/", "/Users/")

#: The stored lease owner, and both halves of it. A public response must carry
#: neither the account nor the machine it was claimed from.
STORED_CLAIM_OWNER = "testuser@n3129.hyak.local"
OPERATOR_HOSTNAMES = ("n3129.hyak.local", "hyak.local")

RUN_DIR = "/gpfs/projects/simswe/testuser/training_outputs/qlora-runs/{0}/20260827T064701Z-full".format(
    COURSE
)

#: What the cluster actually writes: logical references alongside the runtime
#: locations `resolved_config.json` records.
STORED_PROVENANCE: dict[str, Any] = {
    "courseId": COURSE,
    "runId": "run-1",
    "mode": "full",
    "attempt": 1,
    "slurmJobId": "1284412",
    "baseModel": "meta-llama/Llama-3.2-3B-Instruct",
    "artifactRef": "qlora-runs/{0}/20260827T064701Z-full/adapter".format(COURSE),
    "outputRef": "qlora-runs/{0}/20260827T064701Z-full".format(COURSE),
    "datasetRef": "exports/{0}".format(COURSE),
    "datasetVersion": "{0}-approved-split-seed7-n42".format(COURSE),
    "datasetSha256": "a" * 64,
    "datasetChecksums": {"train.jsonl": "b" * 64, "validation.jsonl": "c" * 64},
    "approvedExampleCount": 42,
    "trainExamples": 37,
    "validationExamples": 5,
    "intendedOptimizerSteps": 30,
    "completedSteps": 30,
    "missingOptimizerSteps": 0,
    "trainingLengthSatisfied": True,
    "epochs": 3.0,
    "trainLoss": 0.8123,
    "evalLoss": 1.4567,
    "actualGpuHours": 0.0134,
    "gpuCount": 1,
    "elapsedSeconds": 48.2,
    "gitCommitSha": "9941833cafe0000000000000000000000000beef",
    "enqueuedAt": "2026-08-27T06:00:00+00:00",
    "startedAt": "2026-08-27T06:47:01+00:00",
    "completedAt": "2026-08-27T06:47:49+00:00",
    "resolvedConfig": {
        "mode": "full",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "train_path": "/gpfs/projects/simswe/testuser/css360-syllabus-bot/data/exports/{0}/train.jsonl".format(
            COURSE
        ),
        "validation_path": "/home/alice/exports/{0}/validation.jsonl".format(COURSE),
        "output_dir": RUN_DIR,
        "learning_rate": 0.0002,
        "num_train_epochs": 3.0,
        "seed": 7,
        "lora_r": 16,
        "lora_alpha": 32,
        "train_example_count": 37,
        "validation_example_count": 5,
        # A second level down, to prove the walk is not top-level only.
        "environment": {
            "hf_home": "/gpfs/projects/simswe/testuser/huggingface",
            "venv": "~/venvs/qlora",
            "cache_dir": "/Users/alice/.cache/huggingface",
            "torch_version": "2.4.1",
            "search_paths": [
                "/gpfs/projects/simswe/testuser/venvs/qlora/lib",
                "training_outputs/qlora-runs",
            ],
        },
    },
}

#: The same blob as a run's completion record, plus the failure text a worker
#: writes when a traceback names the directory the job died in.
STORED_COMPLETION: dict[str, Any] = {
    **{
        key: value
        for key, value in STORED_PROVENANCE.items()
        if key not in ("courseId", "runId", "mode", "attempt", "enqueuedAt")
    },
    "outcome": "succeeded",
    "receivedAt": "2026-08-27T06:48:00+00:00",
    "jobId": "1284412",
    "runtimeReport": {
        "slurmJobId": "1284412",
        "gitCommitSha": "9941833cafe0000000000000000000000000beef",
        "modelId": "meta-llama/Llama-3.2-3B-Instruct",
        "gpuCount": 1,
        "totalElapsedSeconds": 48.2,
    },
    "trainingMetrics": {"train_loss": 0.8123, "epoch": 3.0},
    "evaluationMetrics": {"eval_loss": 1.4567},
}

STORED_REGISTRY: dict[str, Any] = {
    "courseId": COURSE,
    "currentVersion": "v2",
    "versions": {
        "v2": {
            "version": "v2",
            "baseModel": "meta-llama/Llama-3.2-3B-Instruct",
            "trainingExampleCount": 37,
            "status": "ready",
            "deployment": "online",
            "artifactRef": "serving/{0}/v2/adapter".format(COURSE),
            "createdAt": "2026-08-27T06:48:00+00:00",
            "updatedAt": "2026-08-27T07:00:00+00:00",
            "runId": "run-1",
            "provenance": STORED_PROVENANCE,
        },
        # The hand-registered historical row: no provenance at all.
        "v1": {
            "version": "v1",
            "baseModel": "meta-llama/Llama-3.2-3B-Instruct",
            "trainingExampleCount": 42,
            "status": "ready",
            "deployment": "offline",
            "artifactRef": "serving/{0}/v1/adapter".format(COURSE),
            "createdAt": "2026-08-13T06:22:50+00:00",
        },
    },
}

STORED_RUN: dict[str, Any] = {
    "runId": "run-1",
    "courseId": COURSE,
    "mode": "full",
    "state": "succeeded",
    "enqueuedAt": "2026-08-27T06:00:00+00:00",
    "updatedAt": "2026-08-27T06:48:00+00:00",
    "datasetRef": "exports/{0}".format(COURSE),
    "approvedExampleCount": 42,
    "trainExamples": 37,
    "validationExamples": 5,
    "attempt": 1,
    "jobId": "1284412",
    "completion": STORED_COMPLETION,
    # What `run_training_queue.py:default_owner` actually writes:
    # `getpass.getuser()@socket.gethostname()`.
    "claim": {
        "owner": "testuser@n3129.hyak.local",
        "claimedAt": "2026-08-27T06:40:00+00:00",
        "expiresAt": "2026-08-27T07:40:00+00:00",
    },
}


def assert_no_operator_identity(case: unittest.TestCase, body: Any) -> None:
    """No path root and no account name anywhere in the serialized response."""
    text = json.dumps(body)
    for root in OPERATOR_PATH_ROOTS:
        case.assertNotIn(root, text)
    for netid in OPERATOR_NETIDS:
        case.assertNotIn(netid, text)
    for hostname in OPERATOR_HOSTNAMES:
        case.assertNotIn(hostname, text)


@contextmanager
def _fake_connection() -> Any:
    yield object()


class RedactionTests(unittest.TestCase):
    """The rule itself: absolute paths go, logical references stay."""

    def test_an_absolute_path_is_dropped_with_its_key(self) -> None:
        cleaned = provenance_privacy.public_provenance(
            {"hf_home": "/gpfs/projects/simswe/testuser/huggingface", "seed": 7}
        )
        self.assertNotIn("hf_home", cleaned)
        self.assertEqual(cleaned["seed"], 7)

    def test_a_relative_reference_survives_untouched(self) -> None:
        """The regression this rule is most likely to cause."""
        cleaned = provenance_privacy.public_provenance(
            {
                "artifactRef": "qlora-runs/css-350/20260827T064701Z-full/adapter",
                "datasetRef": "exports/css-350",
                "baseModel": "meta-llama/Llama-3.2-3B-Instruct",
            }
        )
        self.assertEqual(
            cleaned["artifactRef"],
            "qlora-runs/css-350/20260827T064701Z-full/adapter",
        )
        self.assertEqual(cleaned["datasetRef"], "exports/css-350")
        self.assertEqual(cleaned["baseModel"], "meta-llama/Llama-3.2-3B-Instruct")

    def test_path_only_keys_go_even_when_the_value_is_relative(self) -> None:
        """`datasetRef` and `artifactRef` already answer what these asked."""
        cleaned = provenance_privacy.public_provenance(
            {"train_path": "data/exports/css-350/train.jsonl", "seed": 7}
        )
        self.assertNotIn("train_path", cleaned)

    def test_a_path_inside_free_text_is_replaced_not_the_sentence(self) -> None:
        cleaned = provenance_privacy.public_provenance(
            {
                "error": (
                    "CUDA out of memory while writing "
                    "/gpfs/projects/simswe/testuser/training_outputs/run/adapter."
                )
            }
        )
        self.assertIn("CUDA out of memory", cleaned["error"])
        self.assertIn(provenance_privacy.REDACTED, cleaned["error"])
        self.assertNotIn("testuser", cleaned["error"])
        # The sentence keeps its full stop.
        self.assertTrue(cleaned["error"].endswith("."))

    def test_nesting_and_lists_are_walked(self) -> None:
        cleaned = provenance_privacy.public_provenance(
            {
                "a": {
                    "b": {
                        "c": "/Users/alice/x",
                        "keep": "relative/ok",
                        "list": ["/home/alice/y", "relative/also-ok"],
                    }
                }
            }
        )
        self.assertNotIn("c", cleaned["a"]["b"])
        self.assertEqual(cleaned["a"]["b"]["keep"], "relative/ok")
        self.assertEqual(cleaned["a"]["b"]["list"], ["relative/also-ok"])

    def test_a_home_relative_or_windows_path_is_dropped_too(self) -> None:
        cleaned = provenance_privacy.public_provenance(
            {"venv": "~/venvs/qlora", "win": "C:\\Users\\alice\\out", "seed": 7}
        )
        self.assertNotIn("venv", cleaned)
        self.assertNotIn("win", cleaned)
        self.assertEqual(cleaned["seed"], 7)

    def test_a_url_is_not_a_filesystem_path(self) -> None:
        cleaned = provenance_privacy.public_provenance(
            {"hub": "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct"}
        )
        self.assertEqual(
            cleaned["hub"], "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct"
        )

    def test_a_missing_provenance_block_is_none(self) -> None:
        self.assertIsNone(provenance_privacy.public_provenance(None))

    def test_the_stored_record_is_not_mutated(self) -> None:
        """Sanitizing is a read-side view. Nothing written is rewritten."""
        stored = {
            "resolvedConfig": {"output_dir": RUN_DIR, "seed": 7},
            "datasetRef": "exports/{0}".format(COURSE),
        }
        provenance_privacy.public_provenance(stored)
        self.assertEqual(stored["resolvedConfig"]["output_dir"], RUN_DIR)


class PublicModelEndpointTests(unittest.TestCase):
    """`GET /api/db/courses/{courseId}/model` — the reported leak."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        connection_patch = patch("app.db_routes.db_connection", new=_fake_connection)
        connection_patch.start()
        self.addCleanup(connection_patch.stop)
        registry_patch = patch(
            "app.db_routes.db_models.get_model_registry", return_value=STORED_REGISTRY
        )
        registry_patch.start()
        self.addCleanup(registry_patch.stop)

    def _body(self) -> Any:
        response = self.client.get("/api/db/courses/{0}/model".format(COURSE))
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_no_operator_path_or_netid_reaches_the_browser(self) -> None:
        assert_no_operator_identity(self, self._body())

    def test_the_unsafe_config_keys_are_gone(self) -> None:
        config = self._body()["versions"]["v2"]["provenance"]["resolvedConfig"]
        for key in ("output_dir", "train_path", "validation_path"):
            self.assertNotIn(key, config)

    def test_the_useful_provenance_survives(self) -> None:
        provenance = self._body()["versions"]["v2"]["provenance"]
        self.assertEqual(provenance["runId"], "run-1")
        self.assertEqual(provenance["courseId"], COURSE)
        self.assertEqual(
            provenance["datasetVersion"],
            "{0}-approved-split-seed7-n42".format(COURSE),
        )
        self.assertEqual(provenance["datasetRef"], "exports/{0}".format(COURSE))
        self.assertEqual(
            provenance["artifactRef"],
            "qlora-runs/{0}/20260827T064701Z-full/adapter".format(COURSE),
        )
        self.assertEqual(provenance["trainExamples"], 37)
        self.assertEqual(provenance["validationExamples"], 5)
        self.assertEqual(provenance["approvedExampleCount"], 42)
        self.assertEqual(provenance["epochs"], 3.0)
        self.assertEqual(provenance["trainLoss"], 0.8123)
        self.assertEqual(provenance["evalLoss"], 1.4567)
        self.assertEqual(provenance["actualGpuHours"], 0.0134)
        self.assertEqual(provenance["gpuCount"], 1)
        self.assertEqual(provenance["slurmJobId"], "1284412")
        self.assertEqual(
            provenance["gitCommitSha"],
            "9941833cafe0000000000000000000000000beef",
        )
        self.assertEqual(provenance["startedAt"], "2026-08-27T06:47:01+00:00")
        self.assertEqual(provenance["completedAt"], "2026-08-27T06:47:49+00:00")
        self.assertEqual(provenance["baseModel"], "meta-llama/Llama-3.2-3B-Instruct")

    def test_the_reproducible_half_of_resolved_config_survives(self) -> None:
        config = self._body()["versions"]["v2"]["provenance"]["resolvedConfig"]
        self.assertEqual(config["seed"], 7)
        self.assertEqual(config["learning_rate"], 0.0002)
        self.assertEqual(config["num_train_epochs"], 3.0)
        self.assertEqual(config["lora_r"], 16)
        self.assertEqual(config["train_example_count"], 37)
        self.assertEqual(config["model_id"], "meta-llama/Llama-3.2-3B-Instruct")

    def test_nested_config_paths_are_removed_two_levels_down(self) -> None:
        environment = self._body()["versions"]["v2"]["provenance"]["resolvedConfig"][
            "environment"
        ]
        for key in ("hf_home", "venv", "cache_dir"):
            self.assertNotIn(key, environment)
        self.assertEqual(environment["torch_version"], "2.4.1")
        self.assertEqual(environment["search_paths"], ["training_outputs/qlora-runs"])

    def test_the_version_record_itself_is_intact(self) -> None:
        versions = self._body()["versions"]
        self.assertEqual(versions["v2"]["status"], "ready")
        self.assertEqual(versions["v2"]["deployment"], "online")
        self.assertEqual(versions["v2"]["trainingExampleCount"], 37)
        self.assertEqual(
            versions["v2"]["artifactRef"], "serving/{0}/v2/adapter".format(COURSE)
        )
        # The hand-registered row has no provenance and must not grow one.
        self.assertNotIn("provenance", versions["v1"])
        self.assertEqual(versions["v1"]["trainingExampleCount"], 42)

    def test_the_stored_registry_is_left_alone(self) -> None:
        self._body()
        self.assertEqual(
            STORED_REGISTRY["versions"]["v2"]["provenance"]["resolvedConfig"][
                "output_dir"
            ],
            RUN_DIR,
        )


class PublicTrainingRunEndpointTests(unittest.TestCase):
    """The same blob is served from the training-run routes."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        connection_patch = patch("app.db_routes.db_connection", new=_fake_connection)
        connection_patch.start()
        self.addCleanup(connection_patch.stop)

    def test_the_run_list_carries_no_operator_paths(self) -> None:
        patcher = patch(
            "app.db_routes.db_training_runs.list_training_runs",
            return_value=[STORED_RUN],
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        response = self.client.get(
            "/api/db/courses/{0}/training-runs".format(COURSE)
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        assert_no_operator_identity(self, body)

        completion = body["runs"][0]["completion"]
        self.assertEqual(completion["outcome"], "succeeded")
        self.assertEqual(completion["trainExamples"], 37)
        self.assertEqual(completion["datasetRef"], "exports/{0}".format(COURSE))
        self.assertNotIn("output_dir", completion["resolvedConfig"])
        self.assertEqual(completion["resolvedConfig"]["seed"], 7)
        self.assertEqual(completion["runtimeReport"]["slurmJobId"], "1284412")
        self.assertEqual(completion["trainingMetrics"]["train_loss"], 0.8123)

    def test_a_single_run_is_sanitized_the_same_way(self) -> None:
        patcher = patch(
            "app.db_routes.db_training_runs.get_training_run", return_value=STORED_RUN
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        response = self.client.get(
            "/api/db/courses/{0}/training-runs/run-1".format(COURSE)
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        assert_no_operator_identity(self, body)
        self.assertEqual(body["runId"], "run-1")
        self.assertEqual(body["state"], "succeeded")
        self.assertEqual(body["jobId"], "1284412")

    def test_a_failure_message_keeps_its_reason_without_the_directory(self) -> None:
        failed = {
            **STORED_RUN,
            "state": "failed",
            "completion": None,
            "error": (
                "sbatch: error: unable to open "
                "/gpfs/projects/simswe/testuser/training_outputs/run/train.slurm"
            ),
        }
        patcher = patch(
            "app.db_routes.db_training_runs.get_training_run", return_value=failed
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        response = self.client.get(
            "/api/db/courses/{0}/training-runs/run-1".format(COURSE)
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        assert_no_operator_identity(self, body)
        self.assertIn("unable to open", body["error"])


class PublicModelRequestEndpointTests(unittest.TestCase):
    """The professor-facing request record, guarded at the same boundary."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        connection_patch = patch("app.db_routes.db_connection", new=_fake_connection)
        connection_patch.start()
        self.addCleanup(connection_patch.stop)

    def test_a_path_written_onto_a_request_does_not_come_back_out(self) -> None:
        record = {
            "courseId": COURSE,
            "status": "preparing",
            "requestedAt": "2026-08-27T05:00:00+00:00",
            "updatedAt": "2026-08-27T06:00:00+00:00",
            "approvedExampleCount": 42,
            "preparation": {
                "preparedAt": "2026-08-27T05:30:00+00:00",
                "datasetRef": "exports/{0}".format(COURSE),
                "trainExamples": 37,
                "validationExamples": 5,
                "splitSeed": 7,
                "sourceFile": "/Users/alice/css360-syllabus-bot/data/exports/x.jsonl",
            },
            "preparationError": (
                "Prepared training data is not usable: missing "
                "/home/alice/exports/train.jsonl"
            ),
        }
        patcher = patch(
            "app.db_routes.db_model_requests.get_model_request", return_value=record
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        response = self.client.get(
            "/api/db/courses/{0}/model-request".format(COURSE)
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        assert_no_operator_identity(self, body)

        self.assertEqual(body["status"], "preparing")
        self.assertEqual(
            body["preparation"]["datasetRef"], "exports/{0}".format(COURSE)
        )
        self.assertEqual(body["preparation"]["trainExamples"], 37)
        self.assertEqual(body["preparation"]["splitSeed"], 7)
        self.assertIn("not usable", body["preparationError"])


class ClaimOwnerTests(unittest.TestCase):
    """The lease holder is a real account. Public responses must not say so.

    `training_runs.claim_owner` is written by the cluster worker as
    `getpass.getuser()@socket.gethostname()`, so the stored value is a UW NetID
    and a machine name. The queue is read through `/api/db`, which needs no
    credential.
    """

    def setUp(self) -> None:
        self.client = TestClient(app)
        connection_patch = patch("app.db_routes.db_connection", new=_fake_connection)
        connection_patch.start()
        self.addCleanup(connection_patch.stop)

    def test_the_view_replaces_the_owner_and_keeps_the_lease(self) -> None:
        cleaned = provenance_privacy.public_training_run(STORED_RUN)
        self.assertEqual(
            cleaned["claim"]["owner"], provenance_privacy.PUBLIC_CLAIM_OWNER
        )
        self.assertNotIn("testuser", cleaned["claim"]["owner"])
        # The two facts the queue is actually read for are untouched.
        self.assertEqual(cleaned["claim"]["claimedAt"], "2026-08-27T06:40:00+00:00")
        self.assertEqual(cleaned["claim"]["expiresAt"], "2026-08-27T07:40:00+00:00")

    def test_the_claim_keeps_the_shape_the_frontend_requires(self) -> None:
        """`parseClaim` drops a claim with no owner, which would hide the lease."""
        claim = provenance_privacy.public_training_run(STORED_RUN)["claim"]
        self.assertEqual(set(claim), {"owner", "claimedAt", "expiresAt"})
        self.assertIsInstance(claim["owner"], str)
        self.assertNotEqual(claim["owner"].strip(), "")

    def test_a_run_with_no_claim_does_not_grow_one(self) -> None:
        unclaimed = {key: value for key, value in STORED_RUN.items() if key != "claim"}
        self.assertNotIn("claim", provenance_privacy.public_training_run(unclaimed))

    def test_the_stored_claim_is_not_mutated(self) -> None:
        provenance_privacy.public_training_run(STORED_RUN)
        self.assertEqual(STORED_RUN["claim"]["owner"], STORED_CLAIM_OWNER)

    def test_the_public_run_list_carries_no_account_or_hostname(self) -> None:
        patcher = patch(
            "app.db_routes.db_training_runs.list_training_runs",
            return_value=[STORED_RUN],
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        response = self.client.get(
            "/api/db/courses/{0}/training-runs".format(COURSE)
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        assert_no_operator_identity(self, body)
        self.assertEqual(
            body["runs"][0]["claim"]["owner"], provenance_privacy.PUBLIC_CLAIM_OWNER
        )

    def test_a_single_public_run_carries_no_account_or_hostname(self) -> None:
        patcher = patch(
            "app.db_routes.db_training_runs.get_training_run", return_value=STORED_RUN
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        response = self.client.get(
            "/api/db/courses/{0}/training-runs/run-1".format(COURSE)
        )
        self.assertEqual(response.status_code, 200)
        assert_no_operator_identity(self, response.json())

    def test_the_retry_refusal_does_not_name_the_holder(self) -> None:
        """The 409 body an admin reads when a live lease blocks a retry."""
        block = db_training_runs.training_run_retry_block(
            {
                "state": "claimed",
                "updatedAt": "2026-08-27T06:40:00+00:00",
                "claim": {
                    "owner": STORED_CLAIM_OWNER,
                    "claimedAt": "2026-08-27T06:40:00+00:00",
                    "expiresAt": "2999-01-01T00:00:00+00:00",
                },
            }
        )
        self.assertIsNotNone(block)
        assert_no_operator_identity(self, block)
        self.assertIn("held by a worker", block or "")
        self.assertIn("2999-01-01T00:00:00+00:00", block or "")


class MainRouteTrainingRunTests(unittest.TestCase):
    """The two run-returning routes outside `/api/db`, which return raw dicts."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    @contextmanager
    def _stubbed(self) -> Any:
        @contextmanager
        def _connect(**kwargs: Any) -> Any:
            yield object()

        with patch("app.main.db_connection", _connect), patch(
            "app.main.course_exists", return_value=True
        ):
            yield

    def test_queueing_a_run_returns_the_browser_safe_view(self) -> None:
        with self._stubbed(), patch(
            "app.main.enqueue_training_run", return_value=STORED_RUN
        ):
            response = self.client.post(
                "/api/courses/{0}/training-runs".format(COURSE),
                json={
                    "mode": "full",
                    "datasetRef": "exports/{0}".format(COURSE),
                    "approvedExampleCount": 42,
                    "trainExamples": 37,
                    "validationExamples": 5,
                },
            )

        self.assertEqual(response.status_code, 201)
        body = response.json()
        assert_no_operator_identity(self, body)
        self.assertEqual(body["runId"], "run-1")
        self.assertEqual(
            body["run"]["claim"]["owner"], provenance_privacy.PUBLIC_CLAIM_OWNER
        )
        self.assertNotIn("output_dir", body["run"]["completion"]["resolvedConfig"])

    def test_a_retry_sanitizes_both_runs_it_returns(self) -> None:
        superseded = {
            **STORED_RUN,
            "runId": "run-0",
            "state": "failed",
            "error": "Superseded by admin retry",
        }
        new_run = {
            key: value
            for key, value in STORED_RUN.items()
            if key not in ("claim", "completion")
        }
        new_run = {**new_run, "runId": "run-2", "state": "queued"}

        with self._stubbed(), patch(
            "app.main.retry_training_run",
            return_value={
                "run": new_run,
                "superseded": superseded,
                "request": {"status": "preparing"},
            },
        ):
            response = self.client.post(
                "/api/courses/{0}/training-runs/retry".format(COURSE)
            )

        self.assertEqual(response.status_code, 201)
        body = response.json()
        assert_no_operator_identity(self, body)
        self.assertEqual(body["runId"], "run-2")
        self.assertEqual(body["supersededRunId"], "run-0")
        self.assertEqual(
            body["supersededRun"]["claim"]["owner"],
            provenance_privacy.PUBLIC_CLAIM_OWNER,
        )
        self.assertEqual(body["supersededRun"]["error"], "Superseded by admin retry")


class WorkerEndpointsStayExactTests(unittest.TestCase):
    """The other half of the rule, asserted so it cannot be tidied away.

    The privacy boundary is the *public* one. `/api/training-queue` requires
    `X-Training-Worker-Token`, and its caller is the operator's own runner on
    the cluster: it needs the real lease owner to tell its own claim from
    another session's, and the real run directories to find what a job wrote.
    Sanitizing there would break the workflow while protecting nobody.
    """

    WORKER_TOKEN = "test-worker-token"
    WORKER_HEADERS = {"X-Training-Worker-Token": "test-worker-token"}

    def setUp(self) -> None:
        self.client = TestClient(app)
        env_patch = patch.dict(
            "os.environ", {"TRAINING_WORKER_TOKEN": self.WORKER_TOKEN}
        )
        env_patch.start()
        self.addCleanup(env_patch.stop)
        connection_patch = patch(
            "app.training_queue_routes.db_connection", new=_fake_connection
        )
        connection_patch.start()
        self.addCleanup(connection_patch.stop)

    def test_a_claimed_run_reports_the_real_owner_to_the_worker(self) -> None:
        patcher = patch(
            "app.training_queue_routes.db_training_runs.claim_next_training_run",
            return_value=STORED_RUN,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        response = self.client.post(
            "/api/training-queue/claim",
            json={"owner": STORED_CLAIM_OWNER, "leaseSeconds": 3600},
            headers=self.WORKER_HEADERS,
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["claimed"])
        self.assertEqual(body["run"]["claim"]["owner"], STORED_CLAIM_OWNER)

    def test_the_pending_list_keeps_the_exact_run_directories(self) -> None:
        patcher = patch(
            "app.training_queue_routes.db_training_runs.claimable_training_runs",
            return_value=[STORED_RUN],
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        response = self.client.get(
            "/api/training-queue/pending", headers=self.WORKER_HEADERS
        )
        self.assertEqual(response.status_code, 200)
        run = response.json()["runs"][0]
        self.assertEqual(run["claim"]["owner"], STORED_CLAIM_OWNER)
        self.assertEqual(
            run["completion"]["resolvedConfig"]["output_dir"], RUN_DIR
        )

    def test_the_same_run_is_sanitized_on_the_public_route(self) -> None:
        """The two views of one record, side by side."""
        public = provenance_privacy.public_training_run(STORED_RUN)
        self.assertEqual(STORED_RUN["claim"]["owner"], STORED_CLAIM_OWNER)
        self.assertNotEqual(public["claim"]["owner"], STORED_CLAIM_OWNER)
        self.assertEqual(
            STORED_RUN["completion"]["resolvedConfig"]["output_dir"], RUN_DIR
        )
        self.assertNotIn("output_dir", public["completion"]["resolvedConfig"])


if __name__ == "__main__":
    unittest.main()
