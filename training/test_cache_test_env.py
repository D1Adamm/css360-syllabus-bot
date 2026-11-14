"""Tests for scripts/cache_test_env.py: the isolated test instance's environment.

The properties that matter: production's password is never printed, the test
file never names production's database, production's file is never written,
and what is written reads the same to python-dotenv and to a shell.
"""

from __future__ import annotations

import importlib.util
import io
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


setup = _load("scripts/cache_test_env.py", "cache_test_env")

SECRET = "s3cr3t-P@ss$w0rd!&#%"
ENCODED = "s3cr3t-P%40ss%24w0rd%21%26%23%25"
PRODUCTION = f'''# production
export OLLAMA_MODEL=llama3.2:3b
FINETUNED_SERVICE_URL=http://127.0.0.1:9001   # the VM-local unit
GENERATION_MAX_CONCURRENCY=1
CORS_ALLOWED_ORIGINS=https://aiswe.uwb.edu
APP_PUBLIC_ORIGIN=https://aiswe.uwb.edu
DATABASE_URL="postgresql://sml_app:{ENCODED}@localhost:5432/syllabus_lab?sslmode=prefer"
TRAINING_WORKER_TOKEN=0123456789abcdef0123456789abcdef
CSS360_BENCHMARK_TOKEN=benchmark-token-value
CSS360_BENCHMARK_ENABLED=true
'''


class EnvTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.production = self.root / "prod" / "backend" / ".env"
        self.production.parent.mkdir(parents=True)
        self.production.write_text(PRODUCTION, encoding="utf-8")
        self.out = self.root / "test" / "backend" / ".env"

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = setup.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def write(self, *extra: str) -> tuple[int, str, str]:
        return self.run_cli("write", "--production-env", str(self.production), "--out", str(self.out), *extra)


class WriteTests(EnvTestCase):
    def test_the_test_file_uses_the_same_server_and_role_with_another_database(self) -> None:
        code, out, err = self.write()
        self.assertEqual((code, err), (0, ""))
        written = setup.parse_env_file(self.out)
        self.assertEqual(
            written["DATABASE_URL"],
            f"postgresql://sml_app:{ENCODED}@localhost:5432/syllabus_lab_cachetest?sslmode=prefer",
        )
        self.assertEqual(setup.database_parts(written["DATABASE_URL"], source="x")["database"], "syllabus_lab_cachetest")
        self.assertIn("syllabus_lab_cachetest on localhost:5432", out)

    def test_the_password_is_never_printed(self) -> None:
        for argv in (
            ("describe", "--production-env", str(self.production)),
            ("write", "--production-env", str(self.production), "--out", str(self.out)),
            ("guard", "--production-env", str(self.production), "--env", str(self.out)),
        ):
            code, out, err = self.run_cli(*argv)
            self.assertEqual(code, 0, err)
            for text in (out, err):
                self.assertNotIn(ENCODED, text)
                self.assertNotIn(SECRET, text)
                self.assertNotIn("0123456789abcdef", text)

    def test_only_what_decides_an_answer_is_copied_and_no_other_secret(self) -> None:
        self.write()
        written = setup.parse_env_file(self.out)
        self.assertEqual(written["OLLAMA_MODEL"], "llama3.2:3b")
        # The inline comment is not part of the value.
        self.assertEqual(written["FINETUNED_SERVICE_URL"], "http://127.0.0.1:9001")
        self.assertEqual(written["GENERATION_MAX_CONCURRENCY"], "1")
        for key in ("TRAINING_WORKER_TOKEN", "CSS360_BENCHMARK_TOKEN", "CSS360_BENCHMARK_ENABLED"):
            self.assertNotIn(key, written)
        text = self.out.read_text(encoding="utf-8")
        self.assertNotIn("benchmark-token-value", text)
        self.assertNotIn("aiswe.uwb.edu", text)

    def test_the_test_instance_is_configured_for_loopback_and_the_feature(self) -> None:
        self.write("--backend-port", "8022", "--frontend-port", "5190")
        written = setup.parse_env_file(self.out)
        self.assertEqual(written["AUTH_COOKIE_SECURE"], "false")
        self.assertEqual(written["APP_PUBLIC_ORIGIN"], "http://localhost:5190")
        self.assertEqual(written["CORS_ALLOWED_ORIGINS"], "http://localhost:5190")
        self.assertEqual(written["ANSWER_CACHE_ENABLED"], "true")
        self.assertEqual(written["AUTO_STARTER_SEED_GENERATION"], "false")
        self.assertEqual(written["FINETUNED_READINESS_CHECK_SECONDS"], "0")
        self.assertIn("Backend port 8022", self.out.read_text(encoding="utf-8"))

    def test_the_file_is_owner_only(self) -> None:
        self.write()
        self.assertEqual(stat.S_IMODE(self.out.stat().st_mode), 0o600)

    def test_an_existing_file_is_kept_unless_forced(self) -> None:
        self.write()
        self.out.write_text("MARKER=1\n", encoding="utf-8")
        code, _out, err = self.write()
        self.assertEqual(code, 1)
        self.assertIn("already exists", err)
        self.assertEqual(self.out.read_text(encoding="utf-8"), "MARKER=1\n")
        self.assertEqual(self.write("--force")[0], 0)
        self.assertIn("ANSWER_CACHE_ENABLED", self.out.read_text(encoding="utf-8"))

    def test_productions_own_file_is_never_written(self) -> None:
        before = self.production.read_text(encoding="utf-8")
        code, _out, err = self.run_cli(
            "write", "--production-env", str(self.production), "--out", str(self.production), "--force"
        )
        self.assertEqual(code, 1)
        self.assertIn("production", err)
        self.assertEqual(self.production.read_text(encoding="utf-8"), before)

    def test_the_test_database_cannot_be_productions_or_malformed(self) -> None:
        for name in ("syllabus_lab", "Bad-Name", "x", "drop table;"):
            with self.subTest(name=name):
                code, _out, err = self.write("--database", name)
                self.assertEqual(code, 1)
                self.assertFalse(self.out.exists())
                self.assertNotIn(ENCODED, err)

    def test_a_shell_reads_the_file_exactly_as_python_does(self) -> None:
        self.write()
        expected = setup.parse_env_file(self.out)["DATABASE_URL"]
        shell = subprocess.run(
            ["bash", "-c", f'set -a; source "{self.out}"; set +a; printf %s "$DATABASE_URL"'],
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(shell.stdout, expected)

    def test_a_missing_or_unusable_production_file_is_an_error_not_a_crash(self) -> None:
        code, _out, err = self.run_cli("write", "--production-env", str(self.root / "nope.env"), "--out", str(self.out))
        self.assertEqual(code, 1)
        self.assertIn("Cannot read", err)
        self.production.write_text("OLLAMA_MODEL=x\n", encoding="utf-8")
        self.assertEqual(self.write()[0], 1)
        self.production.write_text("DATABASE_URL=sqlite:///tmp/x.db\n", encoding="utf-8")
        code, _out, err = self.write()
        self.assertEqual(code, 1)
        self.assertIn("not a postgresql://", err)


class DescribeAndGuardTests(EnvTestCase):
    def test_describe_names_the_database_and_the_exact_create_command(self) -> None:
        code, out, _err = self.run_cli("describe", "--production-env", str(self.production))
        self.assertEqual(code, 0)
        self.assertIn("Production database: syllabus_lab on localhost:5432, role sml_app", out)
        self.assertIn("sudo -u postgres createdb -O sml_app syllabus_lab_cachetest", out)

    def test_guard_passes_for_the_test_database(self) -> None:
        self.write()
        code, out, _err = self.run_cli("guard", "--production-env", str(self.production), "--env", str(self.out))
        self.assertEqual(code, 0)
        self.assertIn("uses database syllabus_lab_cachetest; production is syllabus_lab", out)

    def test_guard_refuses_a_file_that_points_at_production(self) -> None:
        self.out.parent.mkdir(parents=True)
        self.out.write_text(self.production.read_text(encoding="utf-8"), encoding="utf-8")
        code, _out, err = self.run_cli("guard", "--production-env", str(self.production), "--env", str(self.out))
        self.assertEqual(code, 1)
        self.assertIn("points at the production database", err)

    def test_guard_refuses_productions_own_file_and_a_missing_one(self) -> None:
        code, _out, err = self.run_cli("guard", "--production-env", str(self.production), "--env", str(self.production))
        self.assertEqual(code, 1)
        self.assertIn("IS production", err)
        code, _out, err = self.run_cli("guard", "--production-env", str(self.production), "--env", str(self.out))
        self.assertEqual(code, 1)

    def test_guard_works_as_a_shell_condition(self) -> None:
        self.write()
        script = REPO_ROOT / "scripts" / "cache_test_env.py"
        ok = subprocess.run(
            [sys.executable, str(script), "guard", "--production-env", str(self.production), "--env", str(self.out)],
            capture_output=True, text=True,
        )
        self.assertEqual(ok.returncode, 0)
        bad = subprocess.run(
            [sys.executable, str(script), "guard", "--production-env", str(self.production), "--env", str(self.production)],
            capture_output=True, text=True,
        )
        self.assertEqual(bad.returncode, 1)


if __name__ == "__main__":
    unittest.main()
