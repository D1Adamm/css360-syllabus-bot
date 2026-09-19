"""`scripts/css360_benchmark_env.py`: the route's .env switch, safe to run twice.

What is held: each managed key ends up in the file exactly once whatever the
file held before, including duplicates an earlier `>>` left; the token is
generated once and kept across reruns, rotated only on request, never printed;
`--disable` flips the flag and nothing else; other lines and comments survive
in order; a new file is private to its owner; a short token is refused and the
file is left alone.
"""

from __future__ import annotations

import importlib.util
import io
import os
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "css360_benchmark_env.py"


def _load():
    spec = importlib.util.spec_from_file_location("css360_benchmark_env", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load()

ENABLED = "CSS360_BENCHMARK_ENABLED"
TOKEN = "CSS360_BENCHMARK_TOKEN"
URL = "CSS360_BENCHMARK_SERVICE_URL"


def _lines_for(contents: str, key: str) -> list[str]:
    return [line for line in contents.splitlines() if line.startswith(f"{key}=")]


class ApplyTests(unittest.TestCase):
    def test_enabling_an_empty_file_sets_the_three_keys_once(self) -> None:
        out = script.apply("", enabled=True)
        values = script.current_values(out)
        self.assertEqual(values[ENABLED], "true")
        self.assertEqual(values[URL], "http://127.0.0.1:9002")
        self.assertRegex(values[TOKEN], r"^[0-9a-f]{64}$")
        for key in (ENABLED, TOKEN, URL):
            self.assertEqual(len(_lines_for(out, key)), 1, key)

    def test_a_second_run_changes_nothing_and_keeps_the_token(self) -> None:
        once = script.apply("OLLAMA_MODEL=llama3.2:3b\n", enabled=True)
        twice = script.apply(once, enabled=True)
        self.assertEqual(once, twice)
        thrice = script.apply(twice, enabled=True, service_url="http://127.0.0.1:9002/")
        self.assertEqual(once, thrice)

    def test_duplicates_from_an_earlier_append_collapse_to_one(self) -> None:
        messy = (
            "DATABASE_URL=postgresql://x\n"
            f"{ENABLED}=true\n{TOKEN}={'a' * 64}\n{URL}=http://127.0.0.1:9002\n"
            f"{ENABLED}=true\n{TOKEN}={'b' * 64}\n{URL}=http://127.0.0.1:9002\n"
        )
        out = script.apply(messy, enabled=True)
        for key in (ENABLED, TOKEN, URL):
            self.assertEqual(len(_lines_for(out, key)), 1, key)
        # The last value is what a loader would have used; it is the one kept.
        self.assertEqual(script.current_values(out)[TOKEN], "b" * 64)
        self.assertTrue(out.startswith("DATABASE_URL=postgresql://x\n"))

    def test_disable_flips_the_flag_and_touches_nothing_else(self) -> None:
        enabled = script.apply("A=1\n", enabled=True)
        token = script.current_values(enabled)[TOKEN]
        disabled = script.apply(enabled, enabled=False)
        values = script.current_values(disabled)
        self.assertEqual(values[ENABLED], "false")
        self.assertEqual(values[TOKEN], token)
        self.assertEqual(values[URL], "http://127.0.0.1:9002")
        self.assertEqual(len(_lines_for(disabled, ENABLED)), 1)
        # Re-enabling reuses the token.
        again = script.apply(disabled, enabled=True)
        self.assertEqual(script.current_values(again)[TOKEN], token)
        self.assertEqual(again, enabled)

    def test_disable_on_a_file_without_the_keys_adds_only_the_flag(self) -> None:
        out = script.apply("A=1\n", enabled=False)
        self.assertEqual(script.current_values(out), {ENABLED: "false"})

    def test_rotate_replaces_the_token(self) -> None:
        first = script.apply("", enabled=True)
        rotated = script.apply(first, enabled=True, rotate_token=True)
        self.assertNotEqual(script.current_values(first)[TOKEN], script.current_values(rotated)[TOKEN])
        self.assertEqual(len(_lines_for(rotated, TOKEN)), 1)

    def test_a_supplied_token_is_used_and_a_short_one_refused(self) -> None:
        out = script.apply("", enabled=True, token="t" * 40)
        self.assertEqual(script.current_values(out)[TOKEN], "t" * 40)
        for bad in ("short", "t" * 31, "has space " + "t" * 30):
            with self.subTest(token=bad):
                with self.assertRaises(ValueError):
                    script.apply("", enabled=True, token=bad)

    def test_a_too_short_existing_token_is_replaced_on_enable(self) -> None:
        out = script.apply(f"{TOKEN}=short\n", enabled=True)
        self.assertRegex(script.current_values(out)[TOKEN], r"^[0-9a-f]{64}$")

    def test_the_service_url_is_validated_and_normalised(self) -> None:
        out = script.apply("", enabled=True, service_url="http://127.0.0.1:9102/")
        self.assertEqual(script.current_values(out)[URL], "http://127.0.0.1:9102")
        with self.assertRaises(ValueError):
            script.apply("", enabled=True, service_url="127.0.0.1:9002")

    def test_other_lines_and_comments_survive_in_order(self) -> None:
        original = "# Ollama\nOLLAMA_MODEL=llama3.2:3b\n\n# secrets\nDATABASE_URL=postgresql://x\n"
        out = script.apply(original, enabled=True)
        self.assertTrue(out.startswith(original))

    def test_quoted_and_exported_values_are_read(self) -> None:
        contents = f'export {TOKEN}="{"q" * 40}"\n{ENABLED}=\'true\'\n'
        self.assertEqual(script.current_values(contents), {TOKEN: "q" * 40, ENABLED: "true"})


class CommandLineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "backend.env"

    def run_script(self, *args: str, stdin: str = "") -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        import sys

        real_stdin = sys.stdin
        sys.stdin = io.StringIO(stdin)
        try:
            with redirect_stdout(out), redirect_stderr(err):
                code = script.main(["--env-file", str(self.path), *args])
        finally:
            sys.stdin = real_stdin
        return code, out.getvalue(), err.getvalue()

    def test_enable_creates_a_private_file_and_never_prints_the_token(self) -> None:
        code, out, err = self.run_script("--enable")
        self.assertEqual(code, 0, err)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        token = script.current_values(self.path.read_text())[TOKEN]
        self.assertNotIn(token, out)
        self.assertNotIn(token, err)
        self.assertIn("created", out)
        self.assertIn(f"{ENABLED}: true", out)
        self.assertIn("set (64 characters)", out)
        self.assertIn("Restart aiswe-backend", out)

    def test_a_second_enable_reports_unchanged(self) -> None:
        self.run_script("--enable")
        before = self.path.read_text()
        code, out, _ = self.run_script("--enable")
        self.assertEqual(code, 0)
        self.assertIn("unchanged", out)
        self.assertEqual(self.path.read_text(), before)

    def test_disable_then_show(self) -> None:
        self.run_script("--enable")
        code, out, _ = self.run_script("--disable")
        self.assertEqual(code, 0)
        self.assertIn(f"{ENABLED}: false", out)
        code, out, _ = self.run_script("--show")
        self.assertEqual(code, 0)
        self.assertIn(f"{ENABLED}: false", out)
        self.assertIn("set (64 characters)", out)
        self.assertNotIn(script.current_values(self.path.read_text())[TOKEN], out)

    def test_show_does_not_create_or_change_the_file(self) -> None:
        code, out, _ = self.run_script("--show")
        self.assertEqual(code, 0)
        self.assertFalse(self.path.exists())
        self.assertIn(f"{TOKEN}: unset", out)

    def test_an_existing_files_mode_is_preserved(self) -> None:
        self.path.write_text("A=1\n")
        os.chmod(self.path, 0o640)
        self.run_script("--enable")
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o640)
        self.assertTrue(self.path.read_text().startswith("A=1\n"))

    def test_a_token_from_stdin_is_used_and_a_short_one_leaves_the_file_alone(self) -> None:
        code, _, _ = self.run_script("--enable", "--token-from-stdin", stdin="s" * 48 + "\n")
        self.assertEqual(code, 0)
        self.assertEqual(script.current_values(self.path.read_text())[TOKEN], "s" * 48)
        before = self.path.read_text()
        code, _, err = self.run_script("--enable", "--token-from-stdin", stdin="short\n")
        self.assertEqual(code, 2)
        self.assertIn("at least 32", err)
        self.assertEqual(self.path.read_text(), before)

    def test_the_minimum_length_matches_the_route(self) -> None:
        from app.research_benchmark_routes import MIN_TOKEN_LENGTH

        self.assertEqual(script.MIN_TOKEN_LENGTH, MIN_TOKEN_LENGTH)


if __name__ == "__main__":
    unittest.main()
