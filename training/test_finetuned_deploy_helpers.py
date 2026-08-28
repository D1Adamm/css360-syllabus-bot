"""Unit tests for fine-tuned deploy helper parsing / .env updates."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


def _load_helpers():
    path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "lib"
        / "finetuned_deploy_helpers.py"
    )
    spec = importlib.util.spec_from_file_location("finetuned_deploy_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = _load_helpers()


class ParseSbatchJobIdTests(unittest.TestCase):
    def test_standard_output(self) -> None:
        self.assertEqual(
            helpers.parse_sbatch_job_id("Submitted batch job 216829\n"),
            "216829",
        )

    def test_ignores_noise_lines(self) -> None:
        text = "sbatch: info: ...\nSubmitted batch job 42\n"
        self.assertEqual(helpers.parse_sbatch_job_id(text), "42")

    def test_missing_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            helpers.parse_sbatch_job_id("nope\n")


class ParseSqueueTests(unittest.TestCase):
    """Real squeue rows, in the delimited format the helpers now ask for.

    The whitespace format they used to ask for could not be parsed positionally
    and was silently wrong for every pending job: `%N` is *empty* — not
    "(null)", not a placeholder — when no node is allocated, so `%i %t %N %M %L`
    collapsed from five fields to four and everything after the node shifted
    left. A job pending on a QOS limit displayed its two-hour *time limit* as
    elapsed time, an empty time-left, and `0:00` as its node.
    """

    def test_running_line(self) -> None:
        parsed = helpers.parse_squeue_job_line("216829|R|g014|00:12:03|00:47:57|")
        assert parsed is not None
        self.assertEqual(parsed["job_id"], "216829")
        self.assertEqual(parsed["state"], "R")
        self.assertEqual(parsed["node"], "g014")
        self.assertEqual(parsed["elapsed"], "00:12:03")
        self.assertEqual(parsed["time_left"], "00:47:57")
        self.assertEqual(parsed["reason"], "")

    def test_the_pending_row_that_used_to_be_misread(self) -> None:
        """The exact row from the failed two-hour serving submission."""
        parsed = helpers.parse_squeue_job_line(
            "265300|PD||0:00|2:00:00|(QOSMaxWallDurationPerJobLimit)"
        )
        assert parsed is not None

        self.assertEqual(parsed["state"], "PD")
        # None of these three were right before.
        self.assertEqual(parsed["node"], "")
        self.assertEqual(parsed["elapsed"], "0:00")
        self.assertEqual(parsed["time_left"], "2:00:00")
        self.assertEqual(parsed["reason"], "QOSMaxWallDurationPerJobLimit")

    def test_a_pending_resource_wait(self) -> None:
        parsed = helpers.parse_squeue_job_line("265301|PD||0:00|1:00:00|(Resources)")
        assert parsed is not None
        self.assertEqual(parsed["reason"], "Resources")
        self.assertEqual(parsed["node"], "")

    def test_a_configuring_job(self) -> None:
        parsed = helpers.parse_squeue_job_line("265302|CF|g007|0:02|59:58|None")
        assert parsed is not None
        self.assertEqual(parsed["state"], "CF")
        self.assertEqual(parsed["node"], "g007")

    def test_a_completing_job(self) -> None:
        parsed = helpers.parse_squeue_job_line("265323|CG|g002|48:10|0:00|None")
        assert parsed is not None
        self.assertEqual(parsed["state"], "CG")
        self.assertEqual(parsed["elapsed"], "48:10")

    def test_a_multi_node_allocation_reduces_to_one_hostname(self) -> None:
        parsed = helpers.parse_squeue_job_line("265303|R|g002,g003|1:00|59:00|None")
        assert parsed is not None
        self.assertEqual(parsed["node"], "g002")

    def test_pending_without_node(self) -> None:
        parsed = helpers.parse_squeue_job_line("216829|PD|(null)|0:00|1:00:00|")
        assert parsed is not None
        self.assertEqual(parsed["state"], "PD")
        self.assertEqual(parsed["node"], "")

    def test_whitespace_input_reads_only_what_it_can_trust(self) -> None:
        """Legacy callers still parse, but nothing is guessed past the state.

        Guessing is what produced the wrong output; refusing to guess is why a
        whitespace row now yields blank timings rather than plausible-looking
        wrong ones.
        """
        parsed = helpers.parse_squeue_job_line("216829 R g014 00:12:03 00:47:57")
        assert parsed is not None
        self.assertEqual(parsed["job_id"], "216829")
        self.assertEqual(parsed["state"], "R")
        self.assertEqual(parsed["node"], "g014")
        self.assertEqual(parsed["elapsed"], "")

    def test_a_reason_is_never_mistaken_for_a_node(self) -> None:
        parsed = helpers.parse_squeue_job_line("216829 PD (Priority)")
        assert parsed is not None
        self.assertEqual(parsed["node"], "")

    def test_a_blank_line_is_not_a_job(self) -> None:
        self.assertIsNone(helpers.parse_squeue_job_line("   "))

    def test_a_line_without_a_state_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            helpers.parse_squeue_job_line("216829|")


class StateDescriptionTests(unittest.TestCase):
    """Every state an operator asked to be able to read."""

    def test_the_states_that_matter(self) -> None:
        for code, expected in (
            ("PD", "pending"),
            ("CF", "configuring"),
            ("R", "running"),
            ("CG", "completing"),
            ("COMPLETED", "completed"),
            ("FAILED", "failed"),
            ("CD", "completed"),
            ("F", "failed"),
            ("TIMEOUT", "timed out"),
            ("NODE_FAIL", "node failure"),
        ):
            with self.subTest(code=code):
                self.assertEqual(helpers.describe_slurm_state(code), expected)

    def test_sacct_cancelled_by_user_reads_as_cancelled(self) -> None:
        self.assertEqual(
            helpers.describe_slurm_state("CANCELLED by 123456"), "cancelled"
        )

    def test_an_unknown_code_is_shown_rather_than_hidden(self) -> None:
        self.assertEqual(helpers.describe_slurm_state("RQ"), "RQ")

    def test_pending_states_are_recognised(self) -> None:
        self.assertTrue(helpers.is_pending_slurm_state("PD"))
        self.assertTrue(helpers.is_pending_slurm_state("CF"))
        self.assertFalse(helpers.is_pending_slurm_state("R"))


class PendingReasonTests(unittest.TestCase):
    def test_a_qos_wall_limit_says_the_job_will_never_start(self) -> None:
        """The reason that cost a serving session, explained rather than echoed."""
        described = helpers.describe_pending_reason("(QOSMaxWallDurationPerJobLimit)")

        self.assertIn("QOSMaxWallDurationPerJobLimit", described)
        self.assertIn("never start", described)

    def test_an_excluded_node_reason_points_at_the_exclude_list(self) -> None:
        described = helpers.describe_pending_reason("ReqNodeNotAvail")

        self.assertIn("--exclude", described)

    def test_an_ordinary_wait_is_explained_plainly(self) -> None:
        self.assertIn("higher-priority", helpers.describe_pending_reason("Priority"))

    def test_an_unknown_reason_is_passed_through(self) -> None:
        self.assertEqual(helpers.describe_pending_reason("(SomethingNew)"), "SomethingNew")

    def test_no_reason_is_empty(self) -> None:
        self.assertEqual(helpers.describe_pending_reason(""), "")


class ExcludeNodeTests(unittest.TestCase):
    """Temporary node exclusion, and the fact that it is never permanent.

    g018 failed its GPU preflight repeatedly in one session. Hyak repairs nodes;
    hardcoding an exclusion would keep the scheduler off a healthy node
    indefinitely and nobody would remember why. So the repository names no node,
    and an exclusion exists only for as long as an operator types one.
    """

    def test_the_default_is_no_exclusions(self) -> None:
        self.assertEqual(helpers.validate_exclude_nodes(""), "")
        self.assertEqual(helpers.validate_exclude_nodes("   "), "")

    def test_one_node(self) -> None:
        self.assertEqual(helpers.validate_exclude_nodes("g018"), "g018")

    def test_several_nodes(self) -> None:
        self.assertEqual(
            helpers.validate_exclude_nodes("g018,g007"), "g018,g007"
        )

    def test_whitespace_and_duplicates_are_cleaned_up(self) -> None:
        self.assertEqual(
            helpers.validate_exclude_nodes(" g018 , g018,, g007 "), "g018,g007"
        )

    def test_a_shell_metacharacter_is_refused(self) -> None:
        """The value becomes an sbatch argument."""
        for bad in ("g018;rm -rf /", "g018 && echo", "$(hostname)", "../g018"):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    helpers.validate_exclude_nodes(bad)

    def test_excluding_most_of_the_cluster_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            helpers.validate_exclude_nodes(
                ",".join(f"g{index:03d}" for index in range(20))
            )

    def test_no_node_is_named_anywhere_in_the_repository(self) -> None:
        """The regression guard for "we hardcoded g018 that one time"."""
        import re

        root = Path(__file__).resolve().parent.parent
        pattern = re.compile(r"--exclude=g\d")
        offenders = []
        for path in list((root / "training").rglob("*")) + list(
            (root / "scripts").rglob("*")
        ):
            if not path.is_file() or path.suffix not in {".sh", ".py", ".slurm"}:
                continue
            if path.name == Path(__file__).name:
                continue
            if pattern.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path))

        self.assertEqual(offenders, [])

    def test_active_and_running_helpers(self) -> None:
        self.assertTrue(helpers.is_active_slurm_state("PD"))
        self.assertTrue(helpers.is_active_slurm_state("RUNNING"))
        self.assertTrue(helpers.is_running_slurm_state("R"))
        self.assertFalse(helpers.is_running_slurm_state("PD"))


class HostnameValidationTests(unittest.TestCase):
    def test_accepts_simple_node(self) -> None:
        self.assertEqual(helpers.validate_compute_hostname("g001"), "g001")
        self.assertEqual(helpers.validate_compute_hostname("n3148"), "n3148")

    def test_rejects_injection(self) -> None:
        for bad in ("g001;rm -rf /", "g001$(reboot)", "g001`id`", "-evil", ""):
            with self.assertRaises(ValueError):
                helpers.validate_compute_hostname(bad)


class WaitForNodeStdoutTests(unittest.TestCase):
    def test_accepts_single_hostname(self) -> None:
        self.assertEqual(helpers.parse_wait_for_node_stdout("g014\n"), "g014")

    def test_pending_status_messages_contaminate_stdout(self) -> None:
        contaminated = (
            "State: PD (waiting for allocation...)\n"
            "State: RUNNING (node not reported yet...)\n"
            "g014\n"
        )
        with self.assertRaises(ValueError) as ctx:
            helpers.parse_wait_for_node_stdout(contaminated)
        self.assertIn("exactly one hostname line", str(ctx.exception))
        self.assertIn("State: PD", str(ctx.exception))

    def test_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            helpers.parse_wait_for_node_stdout("\n")


class HealthReadyTests(unittest.TestCase):
    def test_ready(self) -> None:
        self.assertTrue(
            helpers.health_payload_is_ready(
                {"status": "ok", "adapterLoaded": True, "cudaAvailable": True}
            )
        )

    def test_not_ready(self) -> None:
        self.assertFalse(
            helpers.health_payload_is_ready({"status": "ok", "adapterLoaded": False})
        )
        self.assertFalse(helpers.health_payload_is_ready({"status": "starting"}))

    def test_ready_snake_case(self) -> None:
        self.assertTrue(
            helpers.health_payload_is_ready(
                {"status": "ok", "adapter_loaded": True}
            )
        )


class UpdateEnvKeyTests(unittest.TestCase):
    def test_updates_existing_key(self) -> None:
        original = "FOO=1\nFINETUNED_SERVICE_URL=http://old\nBAR=2\n"
        updated = helpers.update_env_key(
            original, "FINETUNED_SERVICE_URL", "http://127.0.0.1:9001"
        )
        self.assertIn("FINETUNED_SERVICE_URL=http://127.0.0.1:9001\n", updated)
        self.assertIn("FOO=1\n", updated)
        self.assertIn("BAR=2\n", updated)
        self.assertNotIn("http://old", updated)

    def test_appends_missing_key(self) -> None:
        original = "FOO=1\n"
        updated = helpers.update_env_key(
            original, "FINETUNED_SERVICE_URL", "http://127.0.0.1:9001"
        )
        self.assertTrue(updated.startswith("FOO=1\n"))
        self.assertTrue(updated.endswith("FINETUNED_SERVICE_URL=http://127.0.0.1:9001\n"))

    def test_preserves_comments(self) -> None:
        original = "# keep me\nFOO=1\n"
        updated = helpers.update_env_key(original, "FINETUNED_SERVICE_URL", "x")
        self.assertTrue(updated.startswith("# keep me\n"))

    def test_update_env_file_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "backend.env"
            path.write_text("A=1\n", encoding="utf-8")
            changed = helpers.update_env_file(path, "FINETUNED_SERVICE_URL", "http://x")
            self.assertTrue(changed)
            text = path.read_text(encoding="utf-8")
            self.assertIn("A=1\n", text)
            self.assertIn("FINETUNED_SERVICE_URL=http://x\n", text)


if __name__ == "__main__":
    unittest.main()


class ServingQosTests(unittest.TestCase):
    """The serving session's wall clock has to fit the QOS it runs under.

    `serve.slurm` runs under the `debug` QOS, which caps a job at one hour. The
    script's default was two. Slurm accepted that job and then left it PENDING
    forever with `QOSMaxWallDurationPerJobLimit` — which reads like a busy
    cluster, not like a request that can never be satisfied, so it cost a
    session before anyone read the reason.

    Asserted against the script text because the logic is shell. What matters is
    that the ceiling is declared, checked before `sbatch`, and not simply a
    lower hardcoded number that would drift from the QOS again.
    """

    def setUp(self) -> None:
        root = Path(__file__).resolve().parent
        self.start = (root / "start_finetuned_service.sh").read_text(encoding="utf-8")
        self.serve = (root / "inference_service" / "serve.slurm").read_text(
            encoding="utf-8"
        )

    def test_the_qos_ceilings_are_declared_rather_than_assumed(self) -> None:
        self.assertIn("QOS_MAX_HOURS", self.start)
        self.assertIn("[debug]=1", self.start)

    def test_the_default_session_comes_from_the_qos_ceiling(self) -> None:
        # Not a second hardcoded number that can drift from the QOS.
        self.assertIn('HOURS="${SERVICE_HOURS:-${QOS_CEILING:-1}}"', self.start)

    def test_an_over_long_request_is_refused_before_sbatch(self) -> None:
        refusal_at = self.start.index("QOSMaxWallDurationPerJobLimit")
        submit_at = self.start.index("sbatch \\")

        self.assertLess(refusal_at, submit_at)

    def test_the_refusal_says_how_to_ask_for_longer(self) -> None:
        self.assertIn("SERVICE_QOS=normal", self.start)

    def test_the_qos_is_passed_explicitly_to_sbatch(self) -> None:
        # So the script's ceiling and the job's QOS cannot disagree.
        self.assertIn('--qos="${SERVICE_QOS}"', self.start)

    def test_the_slurm_script_and_the_launcher_agree_on_the_qos(self) -> None:
        self.assertIn("#SBATCH --qos=debug", self.serve)
        self.assertIn('SERVICE_QOS="${SERVICE_QOS:-debug}"', self.start)

    def test_the_slurm_default_walltime_fits_the_debug_qos(self) -> None:
        """`sbatch training/inference_service/serve.slurm` by hand must also work."""
        self.assertIn("#SBATCH --time=01:00:00", self.serve)

    def test_a_never_starting_pending_job_is_given_up_on(self) -> None:
        """Waiting out a ten-minute timeout on a job that cannot start is waste."""
        self.assertIn("cannot start", self.start)
        self.assertIn("scancel", self.start)


class ServiceExclusionWiringTests(unittest.TestCase):
    """`--exclude-node` reaches sbatch, and is absent when nobody asked."""

    def setUp(self) -> None:
        root = Path(__file__).resolve().parent
        self.start_service = (root / "start_finetuned_service.sh").read_text(
            encoding="utf-8"
        )
        self.start_training = (root / "start_qlora_training.sh").read_text(
            encoding="utf-8"
        )

    def test_both_launchers_accept_the_flag(self) -> None:
        for text in (self.start_service, self.start_training):
            self.assertIn("--exclude-node)", text)

    def test_both_default_to_no_exclusions(self) -> None:
        self.assertIn('EXCLUDE_NODES="${SERVICE_EXCLUDE_NODES:-}"', self.start_service)
        self.assertIn(
            'EXCLUDE_NODES="${TRAINING_EXCLUDE_NODES:-}"', self.start_training
        )

    def test_the_flag_is_expanded_only_when_set(self) -> None:
        # `${VAR:+--exclude=...}` contributes nothing at all when empty, so an
        # ordinary submission is byte-for-byte what it was before this existed.
        for text in (self.start_service, self.start_training):
            self.assertIn('${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"}', text)

    def test_the_value_is_validated_before_it_reaches_sbatch(self) -> None:
        for text in (self.start_service, self.start_training):
            self.assertIn("validate-exclude-nodes", text)

    def test_exclusions_are_shown_to_the_operator(self) -> None:
        for text in (self.start_service, self.start_training):
            self.assertIn("Exclud", text)

    def test_the_flag_is_documented_as_temporary(self) -> None:
        for text in (self.start_service, self.start_training):
            self.assertIn("TEMPORARY", text)
