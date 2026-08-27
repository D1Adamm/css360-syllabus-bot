"""The completion callback: how a finished job tells the application it finished.

The gap this closes, precisely
------------------------------
Slurm job 253552 trained successfully. `training_run.state` stayed `submitted`
and `model_request.status` stayed `training` for days, because nothing on the
cluster had any way to say otherwise — there was no route to say it to. A person
eventually noticed and ran `register_course_model.py` by hand.

So the properties under test are not "does it write a row". They are the ones
that decide whether this can be trusted to run unattended at 02:00 against a
network that sometimes is not there:

  - a success registers a model and moves both records, in one transaction
  - a failure is just as reportable as a success, with whatever the job knew
  - a second delivery of the same report changes nothing and creates no v2
  - a report from a superseded run is refused, exactly as every other callback is
  - a run that finished short of its optimizer-step budget is not a model
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

COURSE = "css-350-spring-2026-n3h9"
OTHER_COURSE = "css-360-winter-2026-a7rp"
RUN_ID = "run-20260827t064701z-1cf650"
REPLACEMENT_RUN_ID = "run-20260828t090000z-aa11bb"
JOB_ID = "264787"
TOKEN = "test-worker-token"
HEADERS = {"X-Training-Worker-Token": TOKEN}

ARTIFACT_REF = "qlora-runs/css-350-spring-2026-n3h9/20260827T064701Z-full/adapter"
OUTPUT_REF = "qlora-runs/css-350-spring-2026-n3h9/20260827T064701Z-full"


def _run(*, state: str = "submitted", **overrides: Any) -> dict[str, Any]:
    record = {
        "runId": RUN_ID,
        "courseId": COURSE,
        "mode": "full",
        "state": state,
        "enqueuedAt": "2026-08-27T06:40:00+00:00",
        "updatedAt": "2026-08-27T06:47:01+00:00",
        "datasetRef": f"exports/{COURSE}",
        "approvedExampleCount": 42,
        "trainExamples": 37,
        "validationExamples": 5,
        "attempt": 1,
        "jobId": JOB_ID,
    }
    record.update(overrides)
    return record


def _success_payload(**overrides: Any) -> dict[str, Any]:
    """The shape `report_training_result.py` builds from a finished run."""
    payload: dict[str, Any] = {
        "outcome": "succeeded",
        "jobId": JOB_ID,
        "baseModel": "meta-llama/Llama-3.2-3B-Instruct",
        "artifactRef": ARTIFACT_REF,
        "outputRef": OUTPUT_REF,
        "datasetRef": f"exports/{COURSE}",
        "datasetVersion": f"{COURSE}-approved-split-seed360-n42",
        "datasetChecksums": {"train.jsonl": "a" * 64},
        "approvedExampleCount": 42,
        "trainExamples": 37,
        "validationExamples": 5,
        "intendedOptimizerSteps": 15,
        "completedSteps": 15,
        "missingOptimizerSteps": 0,
        "trainingLengthSatisfied": True,
        "epochs": 3.0,
        "trainLoss": 1.2345,
        "evalLoss": 1.4567,
        "actualGpuHours": 0.0134,
        "gpuCount": 1,
        "elapsedSeconds": 48.2,
        "gitCommitSha": "9941833cafe0000000000000000000000000beef",
        "startedAt": "2026-08-27T06:47:10Z",
        "completedAt": "2026-08-27T06:47:58Z",
    }
    payload.update(overrides)
    return payload


@contextmanager
def _fake_connection(**kwargs: Any) -> Iterator[object]:
    yield object()


class CompletionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

        self._env = patch.dict("os.environ", {"TRAINING_WORKER_TOKEN": TOKEN})
        self._env.start()
        self.addCleanup(self._env.stop)

        self._connection = patch(
            "app.training_queue_routes.db_connection", _fake_connection
        )
        self._connection.start()
        self.addCleanup(self._connection.stop)

        # What the fake storage recorded, in the order it was asked to.
        self.run_patches: list[dict[str, Any]] = []
        self.request_patches: list[dict[str, Any]] = []
        self.registered: list[dict[str, Any]] = []
        #: (course_id, run_id) -> already-registered version, so a redelivered
        #: callback finds what the first one wrote.
        self.versions_by_run: dict[tuple[str, str], dict[str, Any]] = {}
        #: Versions the course had before this callback — including ones with no
        #: recorded run, which is what every version registered by hand before
        #: automatic registration existed looks like.
        self.existing_versions: list[dict[str, Any]] = []

    def _post(
        self,
        payload: dict[str, Any],
        *,
        course_id: str = COURSE,
        run_id: str = RUN_ID,
        run: dict[str, Any] | None = None,
        current_run_id: str | None = RUN_ID,
        course_exists: bool = True,
    ):
        stored_run = run if run is not None else _run()

        def _upsert(connection, cid, version, *, set_current=False):
            self.registered.append({"courseId": cid, **dict(version)})
            if version.get("runId"):
                self.versions_by_run[(cid, version["runId"])] = dict(version)
            return {
                "courseId": cid,
                "currentVersion": version["version"] if set_current else "v1",
                "versions": {version["version"]: dict(version)},
            }

        def _update_run(connection, cid, rid, patch):
            self.run_patches.append(dict(patch))
            return {**stored_run, **{k: v for k, v in patch.items() if k != "claim"}}

        def _update_request_for_run(connection, cid, rid, patch):
            self.request_patches.append(dict(patch))
            return {"courseId": cid, "status": patch.get("status", "training")}

        with (
            patch(
                "app.training_queue_routes.db_training_runs.get_training_run",
                return_value=stored_run,
            ),
            patch(
                "app.training_queue_routes.db_training_runs.update_training_run",
                side_effect=_update_run,
            ),
            patch(
                "app.training_queue_routes.db_model_requests.lock_model_request",
                return_value=(
                    None
                    if current_run_id is None
                    else {"courseId": course_id, "currentRunId": current_run_id}
                ),
            ),
            patch(
                "app.training_queue_routes.db_model_requests.update_model_request_for_run",
                side_effect=_update_request_for_run,
            ),
            patch(
                "app.training_queue_routes.db_courses.course_exists",
                return_value=course_exists,
            ),
            patch(
                "app.training_queue_routes.db_models.list_model_versions",
                side_effect=lambda connection, cid: self.existing_versions
                + [
                    item
                    for (course, _run), item in self.versions_by_run.items()
                    if course == cid
                ],
            ),
            patch(
                "app.training_queue_routes.db_models.find_model_version_for_run",
                side_effect=lambda connection, cid, rid: self.versions_by_run.get(
                    (cid, rid)
                ),
            ),
            patch(
                "app.training_queue_routes.db_models.upsert_model_version",
                side_effect=_upsert,
            ),
        ):
            return self.client.post(
                f"/api/training-queue/courses/{course_id}/runs/{run_id}/completed",
                json=payload,
                headers=HEADERS,
            )


class SuccessTests(CompletionTestCase):
    def test_a_success_registers_a_model_and_moves_both_records(self) -> None:
        response = self._post(_success_payload())

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["outcome"], "succeeded")
        self.assertEqual(body["runState"], "succeeded")
        self.assertEqual(body["requestStatus"], "ready")
        self.assertEqual(body["version"], "v1")
        self.assertTrue(body["registered"])
        self.assertFalse(body["alreadyRegistered"])

    def test_the_registered_model_is_ready_and_offline(self) -> None:
        """Training success means a usable artifact exists, not that it is live.

        `deployment` stays `offline` because nothing is serving it: serving is a
        GPU allocation somebody starts, and inferring it from a finished
        training job would tell a professor their model is available when no
        process is running anywhere.
        """
        self._post(_success_payload())

        version = self.registered[0]
        self.assertEqual(version["status"], "ready")
        self.assertEqual(version["deployment"], "offline")

    def test_the_version_records_its_run_and_its_provenance(self) -> None:
        self._post(_success_payload())

        version = self.registered[0]
        self.assertEqual(version["runId"], RUN_ID)

        provenance = version["provenance"]
        self.assertEqual(provenance["courseId"], COURSE)
        self.assertEqual(provenance["runId"], RUN_ID)
        self.assertEqual(provenance["slurmJobId"], JOB_ID)
        self.assertEqual(provenance["baseModel"], "meta-llama/Llama-3.2-3B-Instruct")
        self.assertEqual(provenance["artifactRef"], ARTIFACT_REF)
        self.assertEqual(provenance["datasetVersion"], f"{COURSE}-approved-split-seed360-n42")
        self.assertEqual(provenance["datasetChecksums"], {"train.jsonl": "a" * 64})
        self.assertEqual(provenance["approvedExampleCount"], 42)
        self.assertEqual(provenance["trainExamples"], 37)
        self.assertEqual(provenance["validationExamples"], 5)
        self.assertEqual(provenance["intendedOptimizerSteps"], 15)
        self.assertEqual(provenance["completedSteps"], 15)
        self.assertTrue(provenance["trainingLengthSatisfied"])
        self.assertEqual(provenance["actualGpuHours"], 0.0134)
        self.assertEqual(provenance["gitCommitSha"], "9941833cafe0000000000000000000000000beef")
        self.assertEqual(provenance["mode"], "full")

    def test_the_run_keeps_the_full_completion_record(self) -> None:
        self._post(_success_payload())

        completion = next(
            patch_["completion"]
            for patch_ in self.run_patches
            if "completion" in patch_
        )
        self.assertEqual(completion["outcome"], "succeeded")
        self.assertEqual(completion["jobId"], JOB_ID)
        self.assertEqual(completion["actualGpuHours"], 0.0134)
        self.assertEqual(completion["evalLoss"], 1.4567)
        self.assertEqual(completion["outputRef"], OUTPUT_REF)
        self.assertIn("receivedAt", completion)

    def test_the_training_example_count_comes_from_what_was_trained_on(self) -> None:
        self._post(_success_payload(trainExamples=31))

        self.assertEqual(self.registered[0]["trainingExampleCount"], 31)

    def test_an_absolute_artifact_ref_is_refused(self) -> None:
        """A stored reference must not embed a cluster home directory."""
        response = self._post(
            _success_payload(artifactRef="/gpfs/projects/simswe/madamk/x/adapter")
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.registered, [])

    def test_a_success_with_no_base_model_is_refused(self) -> None:
        payload = _success_payload()
        del payload["baseModel"]
        response = self._post(payload)

        self.assertEqual(response.status_code, 422)
        self.assertIn("baseModel", response.json()["detail"])

    def test_register_false_records_the_outcome_without_a_model(self) -> None:
        """What a smoke run reports: a real outcome, deliberately not a model.

        A smoke run produces an adapter from four examples and three optimizer
        steps. It has to close its run — otherwise the queue strands it — but
        registering it would put a rehearsal in front of students.
        """
        response = self._post(_success_payload(register=False))

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["runState"], "succeeded")
        self.assertFalse(body["registered"])
        self.assertIsNone(body["version"])
        self.assertEqual(self.registered, [])


class ShortRunTests(CompletionTestCase):
    def test_a_run_short_of_its_step_budget_is_not_registered(self) -> None:
        """The QLoRA truncation bug must not be hidden behind a green status.

        A non-divisible dataset used to stop the trainer early — 12 of 15 steps,
        epoch 2.43 instead of 3.0 — and the run still exited zero. Registering
        that adapter as `ready` would show a professor a finished model built
        from four-fifths of the training it was configured for.
        """
        response = self._post(
            _success_payload(
                completedSteps=12,
                missingOptimizerSteps=3,
                trainingLengthSatisfied=False,
            )
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("12/15", response.json()["detail"])
        self.assertEqual(self.registered, [])

    def test_missing_steps_alone_is_enough_to_refuse(self) -> None:
        response = self._post(
            _success_payload(completedSteps=12, missingOptimizerSteps=3)
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.registered, [])


class FailureTests(CompletionTestCase):
    def test_a_failure_is_terminal_for_the_run_and_the_request(self) -> None:
        response = self._post(
            {
                "outcome": "failed",
                "jobId": JOB_ID,
                "failureStage": "training",
                "error": "CUDA out of memory during step 4.",
            }
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["runState"], "failed")
        self.assertEqual(body["requestStatus"], "failed")
        self.assertEqual(self.registered, [])

    def test_a_failure_carries_the_operator_message_and_the_professor_one(self) -> None:
        """Two audiences, two strings, neither standing in for the other."""
        self._post(
            {
                "outcome": "failed",
                "jobId": JOB_ID,
                "failureStage": "model-load",
                "error": "bitsandbytes could not find a CUDA runtime.",
            }
        )

        request_patch = self.request_patches[-1]
        self.assertEqual(
            request_patch["failureMessage"], "Training did not finish successfully."
        )
        self.assertIn("bitsandbytes", request_patch["launchError"])

    def test_a_failure_with_almost_nothing_known_is_still_reportable(self) -> None:
        """A job that died before it measured anything must still close its run.

        This is the half of the original bug that would otherwise survive: a
        silent failure leaves a professor watching "training" indefinitely,
        which is worse than a reported one.
        """
        response = self._post({"outcome": "failed"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["runState"], "failed")

    def test_the_failure_stage_is_kept_on_the_run(self) -> None:
        self._post(
            {"outcome": "failed", "failureStage": "artifact", "error": "no adapter"}
        )

        completion = next(
            patch_["completion"] for patch_ in self.run_patches if "completion" in patch_
        )
        self.assertEqual(completion["failureStage"], "artifact")


class IdempotencyTests(CompletionTestCase):
    def test_a_redelivered_success_does_not_create_a_second_version(self) -> None:
        """The property that makes "persist, send, retry later" safe.

        The worker writes its report to disk before sending it and replays
        anything unacknowledged on the next run, so a redelivery is the ordinary
        case rather than an edge one. Without this, the network failure the
        system already sees would leave a course with v1 and v2 describing one
        adapter.
        """
        first = self._post(_success_payload())
        second = self._post(_success_payload())

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json()["version"], "v1")
        self.assertEqual(second.json()["version"], "v1")
        self.assertFalse(first.json()["alreadyRegistered"])
        self.assertTrue(second.json()["alreadyRegistered"])
        self.assertEqual({item["version"] for item in self.registered}, {"v1"})

    def test_a_third_delivery_is_still_v1(self) -> None:
        for _ in range(3):
            self._post(_success_payload())

        self.assertEqual({item["version"] for item in self.registered}, {"v1"})

    def test_a_redelivery_asking_for_a_different_version_is_refused(self) -> None:
        self._post(_success_payload())
        response = self._post(_success_payload(version="v7"))

        self.assertEqual(response.status_code, 409)
        self.assertIn("already registered", response.json()["detail"])

    def test_a_second_run_for_the_same_course_gets_its_own_version(self) -> None:
        """Idempotency is per run, not per course. A retrained course gets v2."""
        self._post(_success_payload())

        second_run_id = "run-20260901t120000z-ffee00"
        response = self._post(
            _success_payload(),
            run_id=second_run_id,
            run=_run(runId=second_run_id),
            current_run_id=second_run_id,
        )

        self.assertEqual(response.json()["version"], "v2")

    def test_a_new_run_on_a_course_that_already_has_v1_registers_v2(self) -> None:
        """The state CSS 350 is actually in before this ships.

        Its v1 was registered by hand and carries no run id, so the run-keyed
        idempotency read finds nothing and the next unused version is allocated.
        A fresh training run must become v2 — never overwrite the artifact that
        is already published and referenced.
        """
        self.existing_versions = [
            {
                "version": "v1",
                "baseModel": "meta-llama/Llama-3.2-3B-Instruct",
                "status": "ready",
                "deployment": "offline",
                "artifactRef": "css-360-qlora/adapter",
                "trainingExampleCount": 37,
                "createdAt": "2026-08-27T07:00:00+00:00",
            }
        ]
        new_run_id = "run-20260902t080000z-abcdef"

        response = self._post(
            _success_payload(),
            run_id=new_run_id,
            run=_run(runId=new_run_id),
            current_run_id=new_run_id,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["version"], "v2")
        self.assertEqual([item["version"] for item in self.registered], ["v2"])

    def test_registering_v2_does_not_rewrite_the_existing_v1(self) -> None:
        """Nothing in this path touches a version other than the one allocated."""
        self.existing_versions = [
            {
                "version": "v1",
                "baseModel": "meta-llama/Llama-3.2-3B-Instruct",
                "status": "ready",
                "deployment": "offline",
                "artifactRef": "css-360-qlora/adapter",
                "trainingExampleCount": 37,
                "createdAt": "2026-08-27T07:00:00+00:00",
            }
        ]
        new_run_id = "run-20260902t080000z-abcdef"

        self._post(
            _success_payload(),
            run_id=new_run_id,
            run=_run(runId=new_run_id),
            current_run_id=new_run_id,
        )

        self.assertNotIn("v1", [item["version"] for item in self.registered])
        self.assertEqual(
            self.existing_versions[0]["artifactRef"], "css-360-qlora/adapter"
        )


class OwnershipTests(CompletionTestCase):
    """The protections added for the retry workflow, still holding here.

    A completion is the most consequential callback there is — it registers an
    adapter and promotes it — so a late report from a run an admin retired has
    to be refused by the same guard as everything else.
    """

    def test_a_superseded_run_cannot_register_a_model(self) -> None:
        response = self._post(
            _success_payload(), current_run_id=REPLACEMENT_RUN_ID
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn(REPLACEMENT_RUN_ID, response.json()["detail"])
        self.assertEqual(self.registered, [])
        self.assertEqual(self.request_patches, [])

    def test_a_superseded_run_cannot_fail_the_replacements_request(self) -> None:
        """A retired run reporting failure must not tell a professor it failed.

        Its replacement may be queued and untried; the professor's status
        belongs to the run that is current.
        """
        response = self._post(
            {"outcome": "failed", "error": "late failure"},
            current_run_id=REPLACEMENT_RUN_ID,
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.request_patches, [])

    def test_a_course_with_no_model_request_is_not_an_error(self) -> None:
        """Run rows are the queue's record; a request is the professor's view.

        Its absence is not something a worker can act on, and refusing would
        strand the run.
        """
        response = self._post(_success_payload(), current_run_id=None)

        self.assertEqual(response.status_code, 200)

    def test_a_run_that_does_not_exist_for_this_course_is_a_404(self) -> None:
        with patch(
            "app.training_queue_routes.db_training_runs.get_training_run",
            return_value=None,
        ):
            response = self.client.post(
                f"/api/training-queue/courses/{OTHER_COURSE}/runs/{RUN_ID}/completed",
                json=_success_payload(),
                headers=HEADERS,
            )

        self.assertEqual(response.status_code, 404)


class JobIdentityTests(CompletionTestCase):
    def test_a_completion_naming_a_different_job_is_refused(self) -> None:
        response = self._post(_success_payload(jobId="999999"))

        self.assertEqual(response.status_code, 409)
        self.assertIn("264787", response.json()["detail"])
        self.assertEqual(self.registered, [])

    def test_a_completion_recovers_a_submission_that_never_landed(self) -> None:
        """The ambiguous-network case, repaired from the other end.

        `sbatch` succeeded and the `/submitted` call did not, so the run carries
        no job id. The job ran anyway and now reports one. Recording it is the
        correct repair: the alternative is a finished model whose Slurm job an
        operator cannot find.
        """
        response = self._post(_success_payload(), run=_run(jobId=None))

        self.assertEqual(response.status_code, 200)
        job_patches = [
            patch_["jobId"] for patch_ in self.run_patches if "jobId" in patch_
        ]
        self.assertEqual(job_patches, [JOB_ID])

    def test_a_completion_with_no_job_id_at_all_is_accepted(self) -> None:
        payload = _success_payload()
        del payload["jobId"]
        response = self._post(payload)

        self.assertEqual(response.status_code, 200)


class ValidationTests(CompletionTestCase):
    def test_an_unknown_outcome_is_refused(self) -> None:
        response = self._post(_success_payload(outcome="maybe"))
        self.assertEqual(response.status_code, 422)

    def test_a_malformed_version_is_refused(self) -> None:
        response = self._post(_success_payload(version="latest"))
        self.assertEqual(response.status_code, 422)

    def test_the_route_needs_the_worker_token(self) -> None:
        response = self.client.post(
            f"/api/training-queue/courses/{COURSE}/runs/{RUN_ID}/completed",
            json=_success_payload(),
        )
        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
