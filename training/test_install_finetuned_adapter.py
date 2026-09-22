"""Tests for scripts/install_finetuned_adapter.py, with every external effect faked.

No llama.cpp, no Ollama, no `ollama` binary: the runner records the commands
it is asked to run and answers the two Ollama endpoints from canned data.
"""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("install_finetuned_adapter", REPO_ROOT / "scripts" / "install_finetuned_adapter.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


installer = _load()

CSS350 = "css-350-spring-2026-n3h9"
BASE_ID = "meta-llama/Llama-3.2-3B-Instruct"


def write_adapter(root: Path, *, base: str = BASE_ID, weights: bool = True, peft_type: str = "LORA") -> Path:
    adapter = root / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": peft_type, "base_model_name_or_path": base, "r": 8, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"]})
    )
    if weights:
        (adapter / "adapter_model.safetensors").write_bytes(b"\x00" * 64)
    return adapter


class FakeRunner:
    """Records commands; writes the GGUF the converter would; answers /api/tags."""

    def __init__(self, *, tags: dict[str, str] | None = None, convert_exit: int = 0, create_exit: int = 0, chat_text: str = "Mondays at 10.") -> None:
        self.tags = dict(tags) if tags is not None else {"llama3.2:3b": "sha256-base"}
        self.convert_exit = convert_exit
        self.create_exit = create_exit
        self.chat_text = chat_text
        self.commands: list[list[str]] = []
        self.posts: list[tuple[str, dict[str, Any]]] = []

    def run(self, command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
        self.commands.append(list(command))
        if any(part.endswith("convert_lora_to_gguf.py") for part in command):
            if self.convert_exit == 0:
                outfile = Path(command[command.index("--outfile") + 1])
                outfile.parent.mkdir(parents=True, exist_ok=True)
                outfile.write_bytes(b"GGUF" + b"\x01" * 32)
            return subprocess.CompletedProcess(command, self.convert_exit, "", "converter said no" if self.convert_exit else "")
        if command[:2] == ["ollama", "create"]:
            if self.create_exit == 0:
                name = command[2] if ":" in command[2] else command[2] + ":latest"
                self.tags[name] = "sha256-" + name
            return subprocess.CompletedProcess(command, self.create_exit, "", "create failed" if self.create_exit else "")
        raise AssertionError(f"unexpected command {command}")

    def get_json(self, url: str, timeout: float = 10.0) -> Any:
        assert url.endswith("/api/tags"), url
        return {"models": [{"name": name, "digest": digest} for name, digest in self.tags.items()]}

    def post_json(self, url: str, body: dict[str, Any], timeout: float = 180.0) -> Any:
        self.posts.append((url, body))
        return {"message": {"role": "assistant", "content": self.chat_text}}


def _args(**overrides: Any) -> Any:
    argv = ["--course", overrides.pop("course", CSS350), "--version", overrides.pop("version", "v1")]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif value is not None:
            argv += [flag, str(value)]
    return installer.build_parser().parse_args(argv)


class PlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.adapter = write_adapter(self.root / "run")
        self.converter = self.root / "llama.cpp" / "convert_lora_to_gguf.py"
        self.converter.parent.mkdir()
        self.converter.write_text("# fake\n")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _plan(self, **overrides: Any) -> dict[str, Any]:
        overrides.setdefault("adapter", str(self.adapter))
        overrides.setdefault("artifacts_root", str(self.root / "artifacts"))
        overrides.setdefault("converter", str(self.converter))
        overrides.setdefault("converter_python", None)
        return installer.build_plan(_args(**overrides))

    def test_a_valid_adapter_plans_a_versioned_gguf_tag_and_command(self) -> None:
        plan = self._plan()
        self.assertEqual(plan["tag"], "css350-ft-v1:latest")
        self.assertEqual(plan["gguf"], self.root / "artifacts" / CSS350 / "v1" / "css350-v1-lora.gguf")
        self.assertEqual(plan["modelfile"].name, "Modelfile")
        command = plan["convert"]["command"]
        self.assertEqual(command[1], str(self.converter))
        self.assertIn("--base-model-id", command)
        self.assertEqual(command[command.index("--outtype") + 1], "f16")
        self.assertEqual(command[-1], str(self.adapter))
        self.assertEqual(plan["adapter"]["r"], 8)

    def test_course_and_version_follow_the_service_rules(self) -> None:
        for bad in ({"course": "../etc"}, {"course": "CSS350"}, {"version": "1"}, {"version": "v1/"}):
            with self.assertRaises((installer.InstallError, installer.helpers.CourseAdapterError)):
                self._plan(**bad)

    def test_incomplete_or_missing_adapter_directories_are_refused(self) -> None:
        with self.assertRaises(installer.InstallError):
            self._plan(adapter=str(self.root / "nope"))
        no_weights = write_adapter(self.root / "partial", weights=False)
        with self.assertRaises(installer.InstallError) as caught:
            self._plan(adapter=str(no_weights))
        self.assertIn("weight", str(caught.exception))
        not_lora = write_adapter(self.root / "ia3", peft_type="IA3")
        with self.assertRaises(installer.InstallError):
            self._plan(adapter=str(not_lora))
        with self.assertRaises(installer.InstallError):
            self._plan(adapter=None, gguf=None)

    def test_a_base_model_mismatch_is_refused_unless_overridden(self) -> None:
        other = write_adapter(self.root / "other", base="meta-llama/Llama-3.1-8B-Instruct")
        with self.assertRaises(installer.InstallError) as caught:
            self._plan(adapter=str(other))
        self.assertIn("Llama-3.1-8B", str(caught.exception))
        plan = self._plan(adapter=str(other), allow_base_mismatch=True)
        self.assertTrue(any("WARNING" in w for w in plan["warnings"]))
        # Naming the adapter's base explicitly is the clean way; the Ollama base
        # then has no known match and is also refused without the override.
        with self.assertRaises(installer.InstallError):
            self._plan(adapter=str(self.adapter), ollama_base="mistral:7b")

    def test_an_existing_gguf_is_not_overwritten_without_replace(self) -> None:
        gguf = self.root / "artifacts" / CSS350 / "v1" / "css350-v1-lora.gguf"
        gguf.parent.mkdir(parents=True)
        gguf.write_bytes(b"old")
        with self.assertRaises(installer.InstallError) as caught:
            self._plan()
        self.assertIn("--replace", str(caught.exception))
        self.assertTrue(self._plan(replace=True)["replace"])

    def test_a_gguf_input_skips_conversion_and_needs_no_converter(self) -> None:
        gguf = self.root / "css350-v1-lora.gguf"
        gguf.write_bytes(b"GGUF")
        plan = self._plan(adapter=None, gguf=str(gguf), converter=None)
        self.assertIsNone(plan["convert"])
        self.assertEqual(plan["gguf"], gguf)

    def test_a_missing_converter_is_a_plan_warning_not_an_error(self) -> None:
        plan = self._plan(converter=None, llama_cpp_dir=str(self.root / "absent"))
        self.assertIsNone(plan["convert"]["converter"])
        self.assertIn("NOT FOUND", installer.describe_plan(plan))


class ExecuteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.adapter = write_adapter(self.root / "run")
        self.converter = self.root / "llama.cpp" / "convert_lora_to_gguf.py"
        self.converter.parent.mkdir()
        self.converter.write_text("# fake\n")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _plan(self, **overrides: Any) -> dict[str, Any]:
        overrides.setdefault("adapter", str(self.adapter))
        overrides.setdefault("artifacts_root", str(self.root / "artifacts"))
        overrides.setdefault("converter", str(self.converter))
        return installer.build_plan(_args(**overrides))

    def _execute(self, plan: dict[str, Any], runner: FakeRunner, *, dry_run: bool = False) -> tuple[dict[str, Any], str]:
        out: list[str] = []
        record = installer.execute(plan, runner, dry_run=dry_run, log=out.append)
        return record, "\n".join(out)

    def test_dry_run_runs_nothing_and_prints_the_mapping_entry(self) -> None:
        runner = FakeRunner()
        record, log = self._execute(self._plan(converter=None, llama_cpp_dir=str(self.root / "absent")), runner, dry_run=True)
        self.assertEqual(runner.commands, [])
        self.assertEqual(runner.posts, [])
        self.assertTrue(record["dryRun"])
        self.assertIn(f"{CSS350}@v1=css350-ft-v1:latest", log)
        self.assertIn("set-mapping", log)
        self.assertFalse((self.root / "artifacts").exists())

    def test_the_full_path_converts_creates_verifies_and_records(self) -> None:
        runner = FakeRunner()
        plan = self._plan(smoke=True)
        record, log = self._execute(plan, runner)
        self.assertEqual([c[0] if c[0] != "ollama" else "ollama create" for c in runner.commands][-1], "ollama create")
        self.assertTrue(any(c[1].endswith("convert_lora_to_gguf.py") for c in runner.commands))
        create = next(c for c in runner.commands if c[:2] == ["ollama", "create"])
        self.assertEqual(create[2], "css350-ft-v1")
        self.assertEqual(plan["modelfile"].read_text(), f"FROM llama3.2:3b\nADAPTER {plan['gguf']}\n")
        self.assertEqual(record["ollamaTag"], "css350-ft-v1:latest")
        self.assertEqual(record["ollamaDigest"], "sha256-css350-ft-v1:latest")
        self.assertEqual(record["conversion"]["outtype"], "f16")
        self.assertEqual(record["adapter"]["baseModel"], BASE_ID)
        self.assertEqual(record["smoke"]["characters"], len("Mondays at 10."))
        self.assertEqual(record["mappingEntry"], f"{CSS350}@v1=css350-ft-v1:latest")
        self.assertEqual(json.loads(plan["record"].read_text())["ollamaTag"], "css350-ft-v1:latest")
        self.assertEqual(len(runner.posts), 1)
        self.assertEqual(runner.posts[0][1]["options"]["num_predict"], 32)
        self.assertIn("set-mapping", log)
        # Nothing in the log or record is the answer text.
        self.assertNotIn("Mondays", log)
        self.assertNotIn("Mondays", plan["record"].read_text())

    def test_an_existing_tag_is_refused_without_replace(self) -> None:
        runner = FakeRunner(tags={"llama3.2:3b": "b", "css350-ft-v1:latest": "sha256-old"})
        with self.assertRaises(installer.InstallError) as caught:
            self._execute(self._plan(), runner)
        self.assertIn("--replace", str(caught.exception))
        self.assertEqual(runner.commands, [])
        runner2 = FakeRunner(tags={"llama3.2:3b": "b", "css350-ft-v1:latest": "sha256-old"})
        record, _ = self._execute(self._plan(replace=True), runner2)
        self.assertEqual(record["ollamaDigest"], "sha256-css350-ft-v1:latest")

    def test_a_missing_base_model_in_ollama_is_refused_before_converting(self) -> None:
        runner = FakeRunner(tags={})
        with self.assertRaises(installer.InstallError) as caught:
            self._execute(self._plan(), runner)
        self.assertIn("ollama pull", str(caught.exception))
        self.assertEqual(runner.commands, [])

    def test_a_missing_converter_is_an_error_outside_dry_run(self) -> None:
        with self.assertRaises(installer.InstallError):
            self._execute(self._plan(converter=None, llama_cpp_dir=str(self.root / "absent")), FakeRunner())

    def test_converter_and_create_failures_stop_with_their_output(self) -> None:
        with self.assertRaises(installer.InstallError) as caught:
            self._execute(self._plan(), FakeRunner(convert_exit=3))
        self.assertIn("converter said no", str(caught.exception))
        runner = FakeRunner(create_exit=1)
        with self.assertRaises(installer.InstallError) as caught:
            self._execute(self._plan(), runner)
        self.assertIn("create failed", str(caught.exception))

    def test_a_gguf_input_creates_without_converting(self) -> None:
        gguf = self.root / "from-tillicum.gguf"
        gguf.write_bytes(b"GGUF" * 8)
        runner = FakeRunner()
        record, _ = self._execute(self._plan(adapter=None, gguf=str(gguf), converter=None), runner)
        self.assertEqual([c[:2] for c in runner.commands], [["ollama", "create"]])
        self.assertIsNone(record["conversion"])
        self.assertIsNone(record["adapter"])
        self.assertEqual(record["gguf"]["path"], str(gguf))

    def test_an_empty_smoke_answer_fails(self) -> None:
        with self.assertRaises(installer.InstallError):
            self._execute(self._plan(smoke=True), FakeRunner(chat_text="   "))


class MainTests(unittest.TestCase):
    def test_main_reports_refusals_on_stderr_with_exit_1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = installer.main(["--course", CSS350, "--version", "v1", "--adapter", tmp + "/missing", "--dry-run"], runner=FakeRunner())
            self.assertEqual(code, 1)
            self.assertIn("ERROR", err.getvalue())

    def test_main_dry_run_succeeds_without_ollama_or_llama_cpp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = write_adapter(Path(tmp) / "run")
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = installer.main(
                    ["--course", CSS350, "--version", "v1", "--adapter", str(adapter), "--artifacts-root", tmp + "/art", "--llama-cpp-dir", tmp + "/none", "--dry-run"],
                    runner=FakeRunner(tags={}),
                )
            self.assertEqual(code, 0, err.getvalue())
            self.assertIn("dry run", out.getvalue())


if __name__ == "__main__":
    unittest.main()
