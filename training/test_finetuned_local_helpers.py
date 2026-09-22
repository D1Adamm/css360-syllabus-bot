"""Tests for the VM-local fine-tuned service helpers, unit template and scripts.

No systemd, no Ollama, no network: `ss` output is text, the environment file is
a temp file, and the shell scripts are only parsed (`bash -n`).
"""

from __future__ import annotations

import importlib.util
import io
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = _load(REPO_ROOT / "scripts" / "lib" / "finetuned_local_helpers.py", "finetuned_local_helpers")

CSS360 = "css-360-winter-2026-a7rp"
CSS350 = "css-350-spring-2026-n3h9"

SS_TUNNEL = 'LISTEN 0 128 127.0.0.1:9001 0.0.0.0:* users:(("ssh",pid=4242,fd=5))\n'
SS_SERVICE = 'LISTEN 0 2048 127.0.0.1:9001 0.0.0.0:* users:(("python",pid=777,fd=6))\n'
SS_OTHER_PORTS = (
    "State Recv-Q Send-Q Local Address:Port Peer Address:Port Process\n"
    'LISTEN 0 4096 127.0.0.1:8001 0.0.0.0:* users:(("python",pid=100,fd=3))\n'
    'LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:(("ollama",pid=200,fd=3))\n'
    "LISTEN 0 511 *:80 *:*\n"
)


class PortOwnerTests(unittest.TestCase):
    def test_an_ssh_listener_is_the_tunnel(self) -> None:
        owner = helpers.classify_port_owner(helpers.parse_ss_listeners(SS_OTHER_PORTS + SS_TUNNEL))
        self.assertEqual(owner["kind"], "tunnel")
        self.assertEqual(owner["pid"], 4242)

    def test_a_python_listener_is_the_service(self) -> None:
        owner = helpers.classify_port_owner(helpers.parse_ss_listeners(SS_SERVICE))
        self.assertEqual((owner["kind"], owner["pid"], owner["process"]), ("service", 777, "python"))

    def test_no_listener_is_free(self) -> None:
        self.assertEqual(helpers.classify_port_owner(helpers.parse_ss_listeners(SS_OTHER_PORTS))["kind"], "free")
        self.assertEqual(helpers.classify_port_owner([])["kind"], "free")

    def test_an_unreadable_process_is_other_not_free(self) -> None:
        # Another user's process: ss shows the socket but no `users:` field.
        owner = helpers.classify_port_owner(helpers.parse_ss_listeners("LISTEN 0 128 127.0.0.1:9001 0.0.0.0:*\n"))
        self.assertEqual(owner["kind"], "other")
        self.assertIsNone(owner["pid"])

    def test_ipv6_and_wildcard_addresses_parse(self) -> None:
        rows = helpers.parse_ss_listeners('LISTEN 0 128 [::1]:9001 [::]:* users:(("python3.13",pid=9,fd=1))\nLISTEN 0 1 *:9001 *:*\n')
        self.assertEqual([r["port"] for r in rows], [9001, 9001])
        self.assertEqual(helpers.classify_port_owner(rows)["kind"], "service")


class EnvFileTests(unittest.TestCase):
    def test_parse_skips_comments_and_strips_quotes(self) -> None:
        text = '# c\n\nFINETUNED_OLLAMA_MODELS="a@v1=b"\nexport FINETUNED_KEEP_ALIVE=30m\nBAD LINE\n1BAD=x\n'
        self.assertEqual(
            helpers.parse_env_file(text),
            {"FINETUNED_OLLAMA_MODELS": "a@v1=b", "FINETUNED_KEEP_ALIVE": "30m"},
        )

    def test_upsert_replaces_in_place_and_appends_when_missing(self) -> None:
        text = "# keep\nFINETUNED_OLLAMA_MODELS=\nFINETUNED_KEEP_ALIVE=30m\n"
        updated = helpers.upsert_env_line(text, "FINETUNED_OLLAMA_MODELS", "x@v1=y")
        self.assertEqual(updated, "# keep\nFINETUNED_OLLAMA_MODELS=x@v1=y\nFINETUNED_KEEP_ALIVE=30m\n")
        appended = helpers.upsert_env_line("A=1\n", "B", "2")
        self.assertEqual(appended, "A=1\n\nB=2\n")

    def test_upsert_refuses_values_that_would_break_the_file(self) -> None:
        for bad in ("a\nb", 'x"y', "x'y"):
            with self.assertRaises(ValueError):
                helpers.upsert_env_line("", "K", bad)
        with self.assertRaises(ValueError):
            helpers.upsert_env_line("", "bad-key", "v")

    def test_validate_env_text_reports_a_malformed_mapping_without_raising(self) -> None:
        summary = helpers.validate_env_text("FINETUNED_OLLAMA_MODELS=not-an-entry\n")
        self.assertIsNotNone(summary["error"])
        self.assertEqual(summary["entries"], [])
        good = helpers.validate_env_text(f"FINETUNED_OLLAMA_MODELS={CSS360}@v2=css360-ft-v2\nFINETUNED_KEEP_ALIVE=30m\n")
        self.assertIsNone(good["error"])
        self.assertEqual(good["entries"], [{"courseId": CSS360, "version": "v2", "ollamaModel": "css360-ft-v2:latest"}])
        self.assertEqual(good["keepAlive"], "30m")


class MappingMergeTests(unittest.TestCase):
    def test_adds_a_course_and_keeps_the_rest_sorted(self) -> None:
        merged = helpers.merge_mapping(f"{CSS360}@v2=css360-ft-v2:latest", CSS350, "v1", "css350-ft-v1")
        self.assertEqual(merged, f"{CSS350}@v1=css350-ft-v1:latest,{CSS360}@v2=css360-ft-v2:latest")

    def test_the_same_entry_again_is_a_no_op(self) -> None:
        raw = f"{CSS360}@v2=css360-ft-v2:latest"
        self.assertEqual(helpers.merge_mapping(raw, CSS360, "v2", "css360-ft-v2"), raw)

    def test_repointing_a_version_needs_replace(self) -> None:
        raw = f"{CSS360}@v2=css360-ft-v2:latest"
        with self.assertRaises(helpers.CourseAdapterError):
            helpers.merge_mapping(raw, CSS360, "v2", "css360-other")
        self.assertEqual(helpers.merge_mapping(raw, CSS360, "v2", "css360-other", replace=True), f"{CSS360}@v2=css360-other:latest")

    def test_a_second_version_of_the_same_course_is_kept_beside_the_first(self) -> None:
        merged = helpers.merge_mapping(f"{CSS360}@v2=css360-ft-v2", CSS360, "v3", "css360-ft-v3")
        self.assertEqual(merged, f"{CSS360}@v2=css360-ft-v2:latest,{CSS360}@v3=css360-ft-v3:latest")

    def test_invalid_inputs_are_refused_by_the_service_rules(self) -> None:
        with self.assertRaises(helpers.CourseAdapterError):
            helpers.merge_mapping("", "../etc", "v1", "x")
        with self.assertRaises(helpers.CourseAdapterError):
            helpers.merge_mapping("", CSS350, "1", "x")
        with self.assertRaises(helpers.CourseAdapterError):
            helpers.merge_mapping("", CSS350, "v1", "has space")


class NamingTests(unittest.TestCase):
    def test_default_tag_and_gguf_name(self) -> None:
        self.assertEqual(helpers.default_ollama_tag(CSS360, "v2"), "css360-ft-v2")
        self.assertEqual(helpers.default_ollama_tag(CSS350, "v1"), "css350-ft-v1")
        self.assertEqual(helpers.gguf_filename(CSS350, "v1"), "css350-v1-lora.gguf")

    def test_a_course_without_a_number_needs_an_explicit_tag(self) -> None:
        with self.assertRaises(helpers.CourseAdapterError):
            helpers.default_ollama_tag("workshop-demo", "v1")

    def test_format_mapping_entry_normalises_the_tag(self) -> None:
        self.assertEqual(helpers.format_mapping_entry(CSS350, "v1", "css350-ft-v1"), f"{CSS350}@v1=css350-ft-v1:latest")


class CliTests(unittest.TestCase):
    def _run(self, *argv: str, stdin: str = "") -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        import sys

        real = sys.stdin
        sys.stdin = io.StringIO(stdin)
        try:
            with redirect_stdout(out), redirect_stderr(err):
                code = helpers.main(list(argv))
        finally:
            sys.stdin = real
        return code, out.getvalue(), err.getvalue()

    def test_set_mapping_writes_the_file_and_validate_env_reads_it_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "finetuned.env"
            path.write_text("FINETUNED_OLLAMA_MODELS=\nFINETUNED_KEEP_ALIVE=30m\n")
            code, out, _ = self._run("set-mapping", str(path), CSS360, "v2", "css360-ft-v2")
            self.assertEqual(code, 0)
            self.assertIn(f"FINETUNED_OLLAMA_MODELS={CSS360}@v2=css360-ft-v2:latest", out)
            code, out, _ = self._run("set-mapping", str(path), CSS360, "v2", "css360-ft-v2")
            self.assertEqual((code, out.strip()), (0, "unchanged"))
            code, out, err = self._run("set-mapping", str(path), CSS360, "v2", "css360-ft-v9")
            self.assertEqual(code, 2)
            self.assertIn("--replace", err)
            code, out, _ = self._run("set-mapping", str(path), CSS350, "v1", "css350-ft-v1", "--dry-run")
            self.assertEqual(code, 0)
            self.assertTrue(out.startswith("would set"))
            self.assertNotIn(CSS350, path.read_text())
            code, out, _ = self._run("validate-env", str(path))
            self.assertEqual(code, 0)
            self.assertEqual(out.strip(), f"{CSS360} v2 -> css360-ft-v2:latest")
            self.assertEqual(helpers.parse_env_file(path.read_text())["FINETUNED_KEEP_ALIVE"], "30m")

    def test_validate_env_fails_on_a_malformed_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "finetuned.env"
            path.write_text("FINETUNED_OLLAMA_MODELS=garbage\n")
            code, _, err = self._run("validate-env", str(path))
            self.assertEqual(code, 1)
            self.assertIn("Invalid", err)

    def test_port_owner_reads_ss_from_stdin(self) -> None:
        code, out, _ = self._run("port-owner", "--field", "kind", stdin=SS_TUNNEL)
        self.assertEqual((code, out.strip()), (0, "tunnel"))


class UnitTemplateTests(unittest.TestCase):
    """The tracked unit is what `aiswe_finetuned.sh install` renders."""

    unit = (REPO_ROOT / "training" / "inference_service" / "aiswe-finetuned.service").read_text()

    def test_it_is_a_user_unit_with_restart_and_a_restricted_env_file(self) -> None:
        self.assertIn("WantedBy=default.target", self.unit)
        self.assertIn("Restart=on-failure", self.unit)
        self.assertIn("EnvironmentFile=%h/.config/aiswe/finetuned.env", self.unit)
        self.assertIn("Environment=INFERENCE_PORT=9001", self.unit)
        self.assertIn("ExecStart=@REPO_ROOT@/backend/.venv/bin/python ollama_service.py", self.unit)
        self.assertIn("ExecStartPre=@REPO_ROOT@/scripts/aiswe_finetuned.sh preflight", self.unit)
        self.assertIn("StartLimitBurst", self.unit)

    def test_it_carries_no_mapping_no_home_path_and_no_secret(self) -> None:
        for forbidden in ("FINETUNED_OLLAMA_MODELS=", "/home/", "/Users/", "TOKEN", "PASSWORD", "DATABASE_URL"):
            self.assertNotIn(forbidden, self.unit.replace("FINETUNED_OLLAMA_MODELS   courseId", ""))

    def test_the_env_example_has_an_empty_mapping_and_a_keep_alive(self) -> None:
        text = (REPO_ROOT / "scripts" / "finetuned.env.example").read_text()
        values = helpers.parse_env_file(text)
        self.assertEqual(values["FINETUNED_OLLAMA_MODELS"], "")
        self.assertEqual(values["FINETUNED_KEEP_ALIVE"], "30m")
        self.assertIsNone(helpers.validate_env_text(text)["error"])


class ShellScriptTests(unittest.TestCase):
    scripts = (
        "scripts/aiswe_finetuned.sh",
        "scripts/start_finetuned_tunnel.sh",
        "scripts/stop_finetuned_tunnel.sh",
        "scripts/status_finetuned_tunnel.sh",
    )

    def test_scripts_parse_and_are_executable(self) -> None:
        for rel in self.scripts:
            path = REPO_ROOT / rel
            result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, f"{rel}: {result.stderr}")
            self.assertTrue(os.access(path, os.X_OK), f"{rel} is not executable")

    def test_the_tunnel_refuses_while_the_local_unit_is_active(self) -> None:
        text = (REPO_ROOT / "scripts" / "start_finetuned_tunnel.sh").read_text()
        self.assertIn('systemctl --user is-active --quiet "${LOCAL_UNIT}"', text)
        self.assertIn("aiswe_finetuned.sh stop", text)
        self.assertNotIn("pkill", text)
        self.assertNotIn("kill -9", text)

    def test_the_local_service_script_never_kills_and_refuses_the_tunnel(self) -> None:
        text = (REPO_ROOT / "scripts" / "aiswe_finetuned.sh").read_text()
        self.assertNotIn("pkill", text)
        self.assertNotIn("kill ", text)
        self.assertIn("refuse_if_tunnel_owns_port", text)
        self.assertIn("stop_finetuned_tunnel.sh", text)
        # It edits backend/.env for one non-secret key only, via the shared helper.
        self.assertIn('update-env-key "${BACKEND_ENV}" FINETUNED_SERVICE_URL', text)
        self.assertNotIn("DATABASE_URL", text)

    def test_help_exits_zero_without_touching_anything(self) -> None:
        result = subprocess.run([str(REPO_ROOT / "scripts" / "aiswe_finetuned.sh"), "--help"], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0)
        self.assertIn("install", result.stdout)


if __name__ == "__main__":
    unittest.main()
