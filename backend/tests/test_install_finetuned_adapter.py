"""Installing an adapter for a course whose id is not in the legacy form.

`css360d-fall-2026-q0ne` and `css360e-autumn-2026-c08m` have no derivable short
label (`css-360-…` gives `css360`; these do not). The tag for them is passed
with `--tag`, and that must be enough: the GGUF filename used to re-derive the
legacy label on its own and refuse the install after the tag was settled.

Everything here is a dry run against temporary directories. Nothing is
converted, nothing reaches Ollama, and nothing is written.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_installer():
    spec = importlib.util.spec_from_file_location(
        "install_finetuned_adapter_under_test", SCRIPTS / "install_finetuned_adapter.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


installer = _load_installer()
helpers = installer.helpers

NEW_COURSES = {
    "css360d-fall-2026-q0ne": "css360d-v1",
    "css360e-autumn-2026-c08m": "css360e-v1",
}
LEGACY_COURSE = "css-360-winter-2026-a7rp"
LEGACY_COURSE_350 = "css-350-spring-2026-n3h9"


class GgufFilenameTest(unittest.TestCase):
    def test_legacy_ids_keep_their_existing_names(self) -> None:
        self.assertEqual(helpers.gguf_filename(LEGACY_COURSE, "v2"), "css360-v2-lora.gguf")
        self.assertEqual(helpers.gguf_filename(LEGACY_COURSE_350, "v1"), "css350-v1-lora.gguf")

    def test_newer_ids_use_the_full_course_id(self) -> None:
        self.assertEqual(
            helpers.gguf_filename("css360d-fall-2026-q0ne", "v1"),
            "css360d-fall-2026-q0ne-v1-lora.gguf",
        )
        self.assertEqual(
            helpers.gguf_filename("css360e-autumn-2026-c08m", "v1"),
            "css360e-autumn-2026-c08m-v1-lora.gguf",
        )

    def test_invalid_ids_and_versions_are_still_refused(self) -> None:
        with self.assertRaises(helpers.CourseAdapterError):
            helpers.gguf_filename("../etc", "v1")
        with self.assertRaises(helpers.CourseAdapterError):
            helpers.gguf_filename("css360d-fall-2026-q0ne", "latest")

    def test_default_tag_is_unchanged(self) -> None:
        # Legacy ids still get the conventional tag; newer ids still need --tag.
        self.assertEqual(helpers.default_ollama_tag(LEGACY_COURSE, "v2"), "css360-ft-v2")
        for course_id in NEW_COURSES:
            with self.assertRaisesRegex(helpers.CourseAdapterError, "pass --tag explicitly"):
                helpers.default_ollama_tag(course_id, "v1")


class DryRunInstallTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)

        self.adapter = root / "adapter"
        self.adapter.mkdir()
        (self.adapter / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "base_model_name_or_path": installer.DEFAULT_BASE_MODEL_ID,
                    "r": 16,
                    "lora_alpha": 32,
                    "target_modules": ["q_proj", "v_proj"],
                }
            ),
            encoding="utf-8",
        )
        (self.adapter / "adapter_model.safetensors").write_bytes(b"weights")

        self.converter = root / installer.CONVERTER_NAME
        self.converter.write_text("# stand-in; never run in a dry run\n", encoding="utf-8")
        self.artifacts = root / "artifacts"

    def _argv(self, course_id: str, *, tag: str | None) -> list[str]:
        argv = [
            "--course", course_id,
            "--version", "v1",
            "--adapter", str(self.adapter),
            "--artifacts-root", str(self.artifacts),
            "--converter", str(self.converter),
            "--converter-python", sys.executable,
            "--smoke",
            "--dry-run",
        ]
        if tag is not None:
            argv += ["--tag", tag]
        return argv

    def _plan(self, course_id: str, *, tag: str | None):
        return installer.build_plan(installer.build_parser().parse_args(self._argv(course_id, tag=tag)))

    def _main(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = installer.main(argv, runner=object())  # a dry run never touches the runner
        return code, out.getvalue(), err.getvalue()

    def test_an_explicit_tag_is_enough_for_newer_course_ids(self) -> None:
        for course_id, tag in NEW_COURSES.items():
            with self.subTest(course_id=course_id):
                plan = self._plan(course_id, tag=tag)
                self.assertEqual(plan["courseId"], course_id)
                self.assertEqual(
                    plan["tag"], helpers.normalize_ollama_model_name(tag)
                )
                expected = self.artifacts.resolve() / course_id / "v1" / f"{course_id}-v1-lora.gguf"
                self.assertEqual(plan["gguf"], expected)
                self.assertIn(str(expected), plan["convert"]["command"])

                code, out, err = self._main(self._argv(course_id, tag=tag))
                self.assertEqual(code, 0, err)
                self.assertIn("dry run: nothing converted, created or written.", out)
                self.assertFalse(self.artifacts.exists())

    def test_newer_course_ids_without_a_tag_are_still_refused(self) -> None:
        for course_id in NEW_COURSES:
            with self.subTest(course_id=course_id):
                code, _, err = self._main(self._argv(course_id, tag=None))
                self.assertEqual(code, 1)
                self.assertIn("pass --tag explicitly", err)

    def test_legacy_course_ids_plan_exactly_as_before(self) -> None:
        plan = self._plan(LEGACY_COURSE, tag=None)
        self.assertEqual(plan["tag"], helpers.normalize_ollama_model_name("css360-ft-v1"))
        self.assertEqual(
            plan["gguf"],
            self.artifacts.resolve() / LEGACY_COURSE / "v1" / "css360-v1-lora.gguf",
        )

        # An explicit tag on a legacy id changes the tag, not the GGUF name.
        tagged = self._plan(LEGACY_COURSE, tag="css360-ft-v9")
        self.assertEqual(tagged["gguf"].name, "css360-v1-lora.gguf")

        code, _, err = self._main(self._argv(LEGACY_COURSE, tag=None))
        self.assertEqual(code, 0, err)
        self.assertFalse(self.artifacts.exists())


class ConverterPythonTest(unittest.TestCase):
    """The converter runs under the venv's interpreter path, never its symlink target.

    On the VM `~/model_artifacts/llama.cpp/.venv/bin/python` links to
    `/usr/bin/python3.12`. Python locates its venv from the path it was started
    by, so running the resolved target loses transformers/torch/gguf and the
    conversion dies with ModuleNotFoundError.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        # Resolved once so macOS's /var → /private/var link is not mistaken
        # for the venv symlink under test.
        self.home = Path(self._tmp.name).resolve()

        self.system_python = self.home / "usr" / "bin" / "python3.12"
        self.system_python.parent.mkdir(parents=True)
        self.system_python.write_text("#!/bin/sh\n", encoding="utf-8")

        self.llama_cpp = self.home / "model_artifacts" / "llama.cpp"
        self.llama_cpp.mkdir(parents=True)
        self.converter = self.llama_cpp / installer.CONVERTER_NAME
        self.converter.write_text("# stand-in; never run in a dry run\n", encoding="utf-8")

        self.venv_python = self.llama_cpp / ".venv" / "bin" / "python"
        self.venv_python.parent.mkdir(parents=True)
        self.venv_python.symlink_to(self.system_python)

        self.adapter = self.home / "adapter"
        self.adapter.mkdir()
        (self.adapter / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA", "base_model_name_or_path": installer.DEFAULT_BASE_MODEL_ID}),
            encoding="utf-8",
        )
        (self.adapter / "adapter_model.safetensors").write_bytes(b"weights")

        # `~` is this temporary home, and no stray $CPU_TRAINING_VENV can win.
        env = {key: value for key, value in os.environ.items() if key not in {"CPU_TRAINING_VENV", "LLAMA_CPP_DIR"}}
        env["HOME"] = str(self.home)
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _command(self, *extra: str) -> list[str]:
        argv = [
            "--course", "css360d-fall-2026-q0ne",
            "--version", "v1",
            "--tag", "css360d-v1",
            "--adapter", str(self.adapter),
            "--artifacts-root", str(self.home / "artifacts"),
            "--dry-run",
            *extra,
        ]
        plan = installer.build_plan(installer.build_parser().parse_args(argv))
        return plan["convert"]["command"]

    def test_the_fixture_really_is_a_symlink_to_the_system_interpreter(self) -> None:
        self.assertTrue(self.venv_python.is_symlink())
        self.assertEqual(self.venv_python.resolve(), self.system_python)

    def test_an_explicit_venv_interpreter_keeps_its_path(self) -> None:
        self.assertEqual(
            installer.find_converter_python(str(self.venv_python)), str(self.venv_python)
        )

    def test_an_explicit_tilde_path_is_expanded_but_not_resolved(self) -> None:
        self.assertEqual(
            installer.find_converter_python("~/model_artifacts/llama.cpp/.venv/bin/python"),
            str(self.venv_python),
        )

    def test_the_discovered_venv_interpreter_keeps_its_path(self) -> None:
        self.assertEqual(
            installer.find_converter_python(None, self.converter), str(self.venv_python)
        )

    def test_the_conversion_command_runs_the_venv_path_when_passed_explicitly(self) -> None:
        command = self._command(
            "--converter", str(self.converter),
            "--converter-python", "~/model_artifacts/llama.cpp/.venv/bin/python",
        )
        self.assertEqual(command[0], str(self.venv_python))
        self.assertEqual(command[1], str(self.converter))
        self.assertNotIn(str(self.system_python), command)

    def test_the_conversion_command_runs_the_venv_path_when_discovered(self) -> None:
        # No --converter or --converter-python: the default checkout under `~`.
        command = self._command()
        self.assertEqual(command[0], str(self.venv_python))
        self.assertEqual(command[1], str(self.converter))
        self.assertNotIn(str(self.system_python), command)

    def test_a_missing_interpreter_is_still_refused(self) -> None:
        with self.assertRaisesRegex(installer.InstallError, "Converter interpreter does not exist"):
            installer.find_converter_python(str(self.home / "nope" / "python"))

    def test_a_dangling_symlink_is_still_refused(self) -> None:
        dangling = self.home / "dangling-python"
        dangling.symlink_to(self.home / "gone" / "python3.12")
        with self.assertRaisesRegex(installer.InstallError, "Converter interpreter does not exist"):
            installer.find_converter_python(str(dangling))

    def test_a_directory_is_still_refused(self) -> None:
        with self.assertRaisesRegex(installer.InstallError, "Converter interpreter is not a file"):
            installer.find_converter_python(str(self.llama_cpp))


if __name__ == "__main__":
    unittest.main()
