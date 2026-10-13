"""The Tillicum fallback tunnel and the VM-local service: never both, and always back.

The 2026-10 outage: `aiswe-finetuned` was stopped (the fallback workflow said
to), nothing ever started it again, and both fine-tuned conditions failed for a
week while Base and RAG worked. These tests run the real operator scripts
(`start_finetuned_tunnel.sh`, `stop_finetuned_tunnel.sh`, `aiswe_finetuned.sh`)
against fake `systemctl`, `ssh`, `curl`, `nc` and `ss` that share one small
"world" directory: whether the unit is active, whether a tunnel is up, what
each `/health` says, and which step should fail.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
START = REPO_ROOT / "scripts" / "start_finetuned_tunnel.sh"
STOP = REPO_ROOT / "scripts" / "stop_finetuned_tunnel.sh"
LOCAL = REPO_ROOT / "scripts" / "aiswe_finetuned.sh"

D = "css360d-fall-2026-q0ne"
E = "css360e-autumn-2026-c08m"

FAKES = {
    "systemctl": r'''
        import os, sys
        w = os.environ["FAKE_WORLD"]
        args = [a for a in sys.argv[1:] if a not in ("--user", "--quiet")]
        open(os.path.join(w, "log"), "a").write("systemctl " + " ".join(args) + "\n")
        verb, unit = (args + ["", ""])[:2]
        local = unit.startswith("aiswe-finetuned")
        active = os.path.exists(os.path.join(w, "unit_active"))
        if verb == "is-active":
            state = "active" if (active or not local) else "inactive"
            if "--quiet" not in sys.argv:
                print(state)
            sys.exit(0 if state == "active" else 3)
        if verb == "is-enabled":
            print("enabled"); sys.exit(0)
        if verb == "stop" and local:
            if os.path.exists(os.path.join(w, "unit_active")):
                os.remove(os.path.join(w, "unit_active"))
            sys.exit(0)
        if verb == "start" and local:
            # The unit's ExecStartPre refuses while the tunnel holds the port.
            if os.path.exists(os.path.join(w, "start_fails")) or os.path.exists(os.path.join(w, "tunnel")):
                sys.exit(1)
            open(os.path.join(w, "unit_active"), "w").close()
            sys.exit(0)
        sys.exit(0)
    ''',
    "ssh": r'''
        import os, socket, sys
        w = os.environ["FAKE_WORLD"]
        argv = sys.argv[1:]
        open(os.path.join(w, "log"), "a").write("ssh " + " ".join(argv) + "\n")
        control = next((a.split("=", 1)[1] for a in argv if a.startswith("ControlPath=")), None)
        tunnel = os.path.join(w, "tunnel")
        if "-O" in argv:
            op = argv[argv.index("-O") + 1]
            if op == "check":
                sys.exit(0 if os.path.exists(tunnel) else 255)
            if op == "exit":
                if os.path.exists(os.path.join(w, "ssh_exit_fails")):
                    sys.exit(255)
                if os.path.exists(tunnel):
                    os.remove(tunnel)
                sys.exit(0)
        if "-L" in argv:
            if os.path.exists(os.path.join(w, "ssh_open_fails")):
                print("Permission denied (keyboard-interactive).", file=sys.stderr)
                sys.exit(255)
            open(tunnel, "w").close()
            if control:
                if os.path.exists(control):
                    os.remove(control)
                socket.socket(socket.AF_UNIX).bind(control)
            sys.exit(0)
        sys.exit(0)
    ''',
    "curl": r'''
        import json, os, sys
        w = os.environ["FAKE_WORLD"]
        url = next(a for a in reversed(sys.argv) if a.startswith("http"))
        has = lambda name: os.path.exists(os.path.join(w, name))
        def local():
            path = os.path.join(w, "local_courses.json")
            courses = json.load(open(path)) if os.path.exists(path) else [
                {"courseId": "css360d-fall-2026-q0ne", "versions": ["v1"], "currentVersion": "v1"},
                {"courseId": "css360e-autumn-2026-c08m", "versions": ["v1"], "currentVersion": "v1"}]
            return {"status": "ok", "engine": "ollama", "adapterLoaded": bool(courses), "courses": courses}
        if url.endswith(":9001/health") or url.endswith("/api/fine-tuned/health"):
            if has("tunnel"):
                if not has("gpu_ready"):
                    sys.exit(7)
                print(json.dumps({"status": "ok", "adapterLoaded": True, "courses": []}))
            elif has("unit_active"):
                print(json.dumps(local()))
            else:
                sys.exit(7)
        elif url.endswith("/api/health"):
            print(json.dumps({"status": "ok"}))
        elif url.endswith("/api/tags"):
            print(json.dumps({"models": []}))
        else:
            sys.exit(7)
    ''',
    "nc": r'''
        import os, sys
        w = os.environ["FAKE_WORLD"]
        if "9001" in sys.argv:
            up = os.path.exists(os.path.join(w, "tunnel")) or os.path.exists(os.path.join(w, "unit_active"))
            sys.exit(0 if up else 1)
        sys.exit(0)
    ''',
    "ss": r'''
        import os
        w = os.environ["FAKE_WORLD"]
        if os.path.exists(os.path.join(w, "tunnel")):
            print('LISTEN 0 128 127.0.0.1:9001 0.0.0.0:* users:(("ssh",pid=4242,fd=5))')
        elif os.path.exists(os.path.join(w, "unit_active")):
            print('LISTEN 0 2048 127.0.0.1:9001 0.0.0.0:* users:(("python",pid=2731,fd=6))')
    ''',
}


class FallbackScriptsTest(unittest.TestCase):
    def setUp(self) -> None:
        # Short: a Unix socket path must fit in ~104 bytes on macOS.
        self.root = Path(tempfile.mkdtemp(prefix="ftfb", dir="/tmp"))
        self.world = self.root / "w"
        self.world.mkdir()
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        for name, body in FAKES.items():
            path = bin_dir / name
            path.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body))
            path.chmod(0o755)
        config = self.root / "c"
        (config / "systemd" / "user").mkdir(parents=True)
        (config / "systemd" / "user" / "aiswe-finetuned.service").write_text("[Service]\n")
        (config / "aiswe").mkdir()
        (config / "aiswe" / "finetuned.env").write_text(
            f"FINETUNED_OLLAMA_MODELS={D}@v1=css360d-v1:latest,{E}@v1=css360e-v1:latest\n"
            "OLLAMA_BASE_URL=http://127.0.0.1:11435\n"
        )
        self.backend_env = self.root / "backend.env"
        self.backend_env.write_text("FINETUNED_SERVICE_URL=http://127.0.0.1:9001\n")
        self.state = self.root / "s" / "css360-syllabus-bot"
        self.env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "FAKE_WORLD": str(self.world),
            "XDG_STATE_HOME": str(self.root / "s"),
            "XDG_CONFIG_HOME": str(config),
            "BACKEND_ENV_FILE": str(self.backend_env),
            "HEALTH_TIMEOUT_SECONDS": "2",
            "BACKEND_HEALTH_TIMEOUT_SECONDS": "2",
            "HEALTH_WAIT_SECONDS": "2",
            "OLLAMA_WAIT_SECONDS": "0",
            "POLL_SECONDS": "0",
            "USER": "tester",
        }
        for name in ("OLLAMA_BASE_URL", "AISWE_FINETUNED_ENV", "INFERENCE_PORT", "LOCAL_PORT"):
            self.env.pop(name, None)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    # -- the world --------------------------------------------------------- #

    def set(self, name: str, on: bool = True) -> None:
        path = self.world / name
        if on:
            path.write_text("")
        elif path.exists():
            path.unlink()

    def has(self, name: str) -> bool:
        return (self.world / name).exists()

    @property
    def marker(self) -> Path:
        return self.state / "ft-fallback-local.env"

    def run_script(self, script: Path, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([str(script), *args], capture_output=True, text=True, env=self.env,
                              timeout=120, check=False)

    def log(self) -> str:
        path = self.world / "log"
        return path.read_text() if path.exists() else ""

    def start_fallback(self) -> subprocess.CompletedProcess:
        self.set("unit_active")
        self.set("gpu_ready")
        result = self.run_script(START, "g014", "--stop-local")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    # -- starting the fallback --------------------------------------------- #

    def test_the_tunnel_never_takes_the_port_from_a_running_service_unasked(self) -> None:
        self.set("unit_active")
        self.set("gpu_ready")
        result = self.run_script(START, "g014")
        self.assertEqual(result.returncode, 1)
        self.assertIn("rerun with --stop-local", result.stderr)
        self.assertTrue(self.has("unit_active"))
        self.assertFalse(self.has("tunnel"))
        self.assertFalse(self.marker.exists())
        self.assertNotIn("systemctl stop", self.log())

    def test_a_successful_start_records_what_to_restore(self) -> None:
        result = self.start_fallback()
        self.assertIn("Fine-tuned backend READY", result.stdout)
        self.assertTrue(self.has("tunnel"))
        self.assertFalse(self.has("unit_active"))
        marker = self.marker.read_text()
        self.assertIn("WAS=active", marker)
        self.assertIn(f"SERVABLE_BEFORE={D}@v1,{E}@v1", marker)

    def test_a_failed_ssh_sign_in_restores_the_local_service(self) -> None:
        self.set("unit_active")
        self.set("ssh_open_fails")
        result = self.run_script(START, "g014", "--stop-local")
        self.assertEqual(result.returncode, 1)
        self.assertIn("SSH tunnel setup failed", result.stderr)
        self.assertIn("returning to VM-local serving", result.stderr)
        self.assertTrue(self.has("unit_active"))
        self.assertFalse(self.marker.exists())
        self.assertIn(f"RESTORED: aiswe-finetuned serves {D}@v1,{E}@v1", result.stderr)

    def test_a_tunnel_that_never_gets_healthy_is_closed_and_the_service_restored(self) -> None:
        self.set("unit_active")  # no gpu_ready: the Tillicum service never answers
        result = self.run_script(START, "g014", "--stop-local")
        self.assertEqual(result.returncode, 1)
        self.assertIn("/health never became ready", result.stderr)
        self.assertIn("Closed the tunnel this run opened.", result.stderr)
        self.assertFalse(self.has("tunnel"))
        self.assertFalse((self.state / "ssh-ft-tunnel.sock").exists())
        self.assertTrue(self.has("unit_active"))
        self.assertFalse(self.marker.exists())

    def test_a_tunnel_that_cannot_be_closed_is_not_raced_by_the_service(self) -> None:
        self.set("unit_active")
        self.set("ssh_exit_fails")
        result = self.run_script(START, "g014", "--stop-local")
        self.assertEqual(result.returncode, 1)
        self.assertIn("aiswe-finetuned was NOT restarted", result.stderr)
        self.assertFalse(self.has("unit_active"))
        self.assertTrue(self.marker.exists())  # stop_finetuned_tunnel.sh still knows to restore

    def test_starting_twice_reuses_the_tunnel_and_keeps_the_first_record(self) -> None:
        self.start_fallback()
        first = self.marker.read_text()
        again = self.run_script(START, "g014", "--stop-local")
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
        self.assertIn("reusing", again.stdout)
        self.assertEqual(self.marker.read_text(), first)

    # -- ending it ------------------------------------------------------------- #

    def test_stopping_the_fallback_restores_and_verifies_the_service_and_is_repeatable(self) -> None:
        self.start_fallback()
        result = self.run_script(STOP)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.has("tunnel"))
        self.assertTrue(self.has("unit_active"))
        self.assertFalse(self.marker.exists())
        self.assertIn(f"RESTORED: aiswe-finetuned serves {D}@v1,{E}@v1 (servable before the fallback)", result.stdout)
        self.assertIn("served by the VM-local service again", result.stdout)

        again = self.run_script(STOP)
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
        self.assertIn("aiswe-finetuned is active; nothing to restore.", again.stdout)

    def test_a_tunnel_that_will_not_close_keeps_everything_and_starts_nothing(self) -> None:
        self.start_fallback()
        self.set("ssh_exit_fails")
        result = self.run_script(STOP)
        self.assertEqual(result.returncode, 1)
        self.assertIn("the tunnel did not close", result.stderr)
        self.assertTrue((self.state / "ssh-ft-tunnel.sock").exists())  # the handle is not thrown away
        self.assertTrue(self.marker.exists())
        self.assertFalse(self.has("unit_active"))

        self.set("ssh_exit_fails", False)
        retry = self.run_script(STOP)
        self.assertEqual(retry.returncode, 0, retry.stdout + retry.stderr)
        self.assertTrue(self.has("unit_active"))

    def test_a_restore_that_cannot_serve_the_same_course_version_is_not_success(self) -> None:
        self.start_fallback()
        # Up, healthy, and serving another course only: CSS360E's model is gone.
        (self.world / "local_courses.json").write_text(json.dumps(
            [{"courseId": D, "versions": ["v1"], "currentVersion": "v1"}]))
        result = self.run_script(STOP)
        self.assertEqual(result.returncode, 1)
        self.assertIn("RESTORE INCOMPLETE", result.stderr)
        self.assertIn(f"{E}@v1", result.stderr)
        self.assertNotIn(f"{D}@v1\n", result.stderr.split("RESTORE INCOMPLETE", 1)[1])
        self.assertTrue(self.marker.exists())  # so the next restore checks the same thing

    def test_another_version_does_not_count(self) -> None:
        self.start_fallback()
        (self.world / "local_courses.json").write_text(json.dumps(
            [{"courseId": D, "versions": ["v1"], "currentVersion": "v1"},
             {"courseId": E, "versions": ["v2"], "currentVersion": "v2"}]))
        result = self.run_script(STOP)
        self.assertEqual(result.returncode, 1)
        self.assertIn(f"{E}@v1", result.stderr)

    def test_a_unit_that_will_not_start_keeps_the_record(self) -> None:
        self.start_fallback()
        self.set("start_fails")
        result = self.run_script(STOP)
        self.assertEqual(result.returncode, 1)
        self.assertIn("RESTORE FAILED", result.stderr)
        self.assertTrue(self.marker.exists())

        self.set("start_fails", False)
        retry = self.run_script(LOCAL, "restore")
        self.assertEqual(retry.returncode, 0, retry.stdout + retry.stderr)
        self.assertFalse(self.marker.exists())

    def test_the_older_manual_workflow_is_restored_too(self) -> None:
        # Stopped by hand with plain `aiswe_finetuned.sh stop`: no record, unit enabled but down.
        result = self.run_script(STOP)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("enabled but inactive", result.stdout)
        self.assertTrue(self.has("unit_active"))
        self.assertIn(f"RESTORED: aiswe-finetuned serves {D}@v1,{E}@v1 (mapped in", result.stdout)

    def test_keep_local_stopped_says_what_it_leaves_broken(self) -> None:
        self.start_fallback()
        result = self.run_script(STOP, "--keep-local-stopped")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("FAIL for every course", result.stdout)
        self.assertFalse(self.has("unit_active"))
        self.assertTrue(self.marker.exists())

    def test_status_names_a_fallback_stop(self) -> None:
        self.start_fallback()
        result = self.run_script(LOCAL, "status")
        self.assertIn("fallback:  stopped for the Tillicum fallback", result.stdout)

    def test_mappings_are_never_touched(self) -> None:
        env_file = Path(self.env["XDG_CONFIG_HOME"]) / "aiswe" / "finetuned.env"
        before = env_file.read_text()
        self.start_fallback()
        self.run_script(STOP)
        self.assertEqual(env_file.read_text(), before)


class ClassroomGpuModeLeavesTheServiceAlone(unittest.TestCase):
    def test_gpu_mode_never_stops_the_vm_service(self) -> None:
        # It tunnels on :9101; the VM service on :9001 stays up as the failback target.
        text = (REPO_ROOT / "scripts" / "classroom_gpu_mode.py").read_text()
        self.assertNotIn("aiswe_finetuned.sh stop", text)
        self.assertNotIn("aiswe-finetuned\"", text)  # never names the unit to systemctl
        self.assertNotIn('"systemctl", "--user", "stop"', text)
        self.assertIn("DEFAULT_GPU_PORT = 9101", text)


if __name__ == "__main__":
    unittest.main()
