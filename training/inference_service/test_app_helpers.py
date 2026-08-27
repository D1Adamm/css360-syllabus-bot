"""Helpers for the per-course inference service (no GPU, no FastAPI needed).

The behaviour under test is course isolation at its source. Training is per
course; the service used to hold one adapter with no course identity anywhere in
it, so publishing CSS 360 replaced what CSS 350 was served with and no request
carried enough information to notice.

Every path here is built from a validated course id and a validated version, and
there is no answer to "serve CSS 350" that involves loading anything else — not
another course's adapter, and not a course-agnostic default.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import helpers

CSS350 = "css-350-spring-2026-n3h9"
CSS360 = "css-360-winter-2026-a7rp"


def publish(root: Path, course_id: str, version: str, *, current: bool = True) -> Path:
    """Write a loadable adapter the way `promote_qlora_adapter.sh` does."""
    adapter = root / course_id / version / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 8}), encoding="utf-8"
    )
    (adapter / "adapter_model.safetensors").write_bytes(
        f"{course_id}:{version}".encode("utf-8")
    )
    if current:
        (root / course_id / "current.json").write_text(
            json.dumps({"courseId": course_id, "version": version}), encoding="utf-8"
        )
    return adapter


class ValidationTests(unittest.TestCase):
    def test_validate_question_rejects_blank(self) -> None:
        with self.assertRaises(ValueError):
            helpers.validate_question("   ")
        self.assertEqual(helpers.validate_question("  Hello?  "), "Hello?")

    def test_a_course_id_that_could_be_a_path_is_refused(self) -> None:
        """This value becomes a directory name under the serving root."""
        for bad in ("../secrets", "css/350", "CSS-350", "", "css-350/../.."):
            with self.subTest(course_id=bad):
                with self.assertRaises(helpers.CourseAdapterError):
                    helpers.validate_course_id(bad)

    def test_a_valid_course_id_passes(self) -> None:
        self.assertEqual(helpers.validate_course_id(f"  {CSS350} "), CSS350)

    def test_a_version_must_look_like_a_version(self) -> None:
        for bad in ("latest", "v", "1", "../v1"):
            with self.subTest(version=bad):
                with self.assertRaises(helpers.CourseAdapterError):
                    helpers.validate_model_version(bad)
        self.assertEqual(helpers.validate_model_version("v12"), "v12")

    def test_resolve_port(self) -> None:
        with mock.patch.dict(os.environ, {"INFERENCE_PORT": "8123"}, clear=False):
            self.assertEqual(helpers.resolve_port(), 8123)

    def test_the_adapter_cache_bound_defaults_and_validates(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MAX_LOADED_ADAPTERS", None)
            self.assertEqual(
                helpers.resolve_max_loaded_adapters(),
                helpers.DEFAULT_MAX_LOADED_ADAPTERS,
            )
        with mock.patch.dict(os.environ, {"MAX_LOADED_ADAPTERS": "0"}, clear=False):
            with self.assertRaises(RuntimeError):
                helpers.resolve_max_loaded_adapters()


class AdapterFormatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_what_training_writes_is_what_serving_loads(self) -> None:
        """No conversion step exists because none is needed.

        `train_qlora.py` calls `save_pretrained` on the PEFT model, which writes
        `adapter_config.json` and `adapter_model.safetensors`. That is exactly
        the pair `PeftModel.load_adapter` reads. Nothing here expects GGUF or a
        merged checkpoint.
        """
        adapter = publish(self.root, CSS350, "v1")

        self.assertTrue(helpers.adapter_is_loadable(adapter))

    def test_a_directory_missing_weights_is_not_loadable(self) -> None:
        adapter = self.root / CSS350 / "v1" / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

        self.assertFalse(helpers.adapter_is_loadable(adapter))

    def test_a_directory_missing_its_config_is_not_loadable(self) -> None:
        adapter = self.root / CSS350 / "v1" / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"weights")

        self.assertFalse(helpers.adapter_is_loadable(adapter))

    def test_an_absent_directory_is_not_loadable(self) -> None:
        self.assertFalse(helpers.adapter_is_loadable(self.root / "nothing"))


class CourseResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_a_course_resolves_to_its_own_adapter(self) -> None:
        publish(self.root, CSS350, "v1")

        resolved = helpers.resolve_course_adapter(CSS350, root=self.root)

        self.assertEqual(resolved["courseId"], CSS350)
        self.assertEqual(resolved["version"], "v1")
        self.assertTrue(str(resolved["path"]).endswith(f"{CSS350}/v1/adapter"))

    def test_css350_and_css360_resolve_to_different_adapters(self) -> None:
        """The isolation requirement, stated directly.

        Two courses, two adapters, no shared path and no shared cache key.
        """
        publish(self.root, CSS350, "v1")
        publish(self.root, CSS360, "v1")

        first = helpers.resolve_course_adapter(CSS350, root=self.root)
        second = helpers.resolve_course_adapter(CSS360, root=self.root)

        self.assertNotEqual(first["path"], second["path"])
        self.assertNotEqual(first["adapterKey"], second["adapterKey"])
        self.assertEqual(
            (first["path"]).read_bytes() if first["path"].is_file() else
            (first["path"] / "adapter_model.safetensors").read_bytes(),
            f"{CSS350}:v1".encode("utf-8"),
        )

    def test_a_course_with_no_published_adapter_is_refused(self) -> None:
        """Never a fallback to another course, or to whatever is lying around."""
        publish(self.root, CSS360, "v1")

        with self.assertRaises(helpers.CourseAdapterError) as caught:
            helpers.resolve_course_adapter(CSS350, root=self.root)

        message = str(caught.exception)
        self.assertIn(CSS350, message)
        self.assertIn("promote_qlora_adapter.sh", message)
        self.assertNotIn(CSS360, message)

    def test_an_explicit_version_wins_over_the_pointer(self) -> None:
        """The backend resolves the version from PostgreSQL and sends it.

        Honouring it is what makes the two sides checkable against each other.
        """
        publish(self.root, CSS350, "v1", current=False)
        publish(self.root, CSS350, "v2")

        resolved = helpers.resolve_course_adapter(CSS350, "v1", root=self.root)

        self.assertEqual(resolved["version"], "v1")
        self.assertEqual(resolved["versionSource"], "requested")

    def test_the_pointer_decides_when_no_version_is_named(self) -> None:
        publish(self.root, CSS350, "v1", current=False)
        publish(self.root, CSS350, "v2")

        resolved = helpers.resolve_course_adapter(CSS350, root=self.root)

        self.assertEqual(resolved["version"], "v2")
        self.assertEqual(resolved["versionSource"], "current.json")

    def test_the_highest_published_version_is_the_last_resort(self) -> None:
        """For a course published before pointers existed."""
        publish(self.root, CSS350, "v1", current=False)
        publish(self.root, CSS350, "v3", current=False)

        resolved = helpers.resolve_course_adapter(CSS350, root=self.root)

        self.assertEqual(resolved["version"], "v3")
        self.assertEqual(resolved["versionSource"], "highest published")

    def test_a_version_that_was_never_published_is_refused(self) -> None:
        publish(self.root, CSS350, "v1")

        with self.assertRaises(helpers.CourseAdapterError) as caught:
            helpers.resolve_course_adapter(CSS350, "v9", root=self.root)

        self.assertIn("v9", str(caught.exception))
        self.assertIn("v1", str(caught.exception))

    def test_a_pointer_naming_an_unpublished_version_is_refused(self) -> None:
        """A dangling pointer must not silently fall through to another version.

        Serving v1 when the record says v2 would make the version reported back
        to the backend a lie.
        """
        publish(self.root, CSS350, "v1", current=False)
        (self.root / CSS350 / "current.json").write_text(
            json.dumps({"version": "v2"}), encoding="utf-8"
        )

        with self.assertRaises(helpers.CourseAdapterError):
            helpers.resolve_course_adapter(CSS350, root=self.root)

    def test_a_corrupt_pointer_falls_back_to_what_is_published(self) -> None:
        publish(self.root, CSS350, "v1", current=False)
        (self.root / CSS350 / "current.json").write_text("{not json", encoding="utf-8")

        resolved = helpers.resolve_course_adapter(CSS350, root=self.root)

        self.assertEqual(resolved["version"], "v1")

    def test_a_traversal_course_id_reaches_nothing(self) -> None:
        with self.assertRaises(helpers.CourseAdapterError):
            helpers.resolve_course_adapter("../../etc", root=self.root)


class AdapterKeyTests(unittest.TestCase):
    def test_the_key_carries_both_course_and_version(self) -> None:
        """PEFT keeps adapters in one flat namespace on the base model.

        A key naming only the course would make a promotion indistinguishable
        from the version it replaced, and a running service would keep answering
        with the old weights until someone restarted it.
        """
        self.assertEqual(helpers.adapter_key(CSS350, "v1"), f"{CSS350}@v1")
        self.assertNotEqual(
            helpers.adapter_key(CSS350, "v1"), helpers.adapter_key(CSS350, "v2")
        )
        self.assertNotEqual(
            helpers.adapter_key(CSS350, "v1"), helpers.adapter_key(CSS360, "v1")
        )


class ListingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_every_published_course_is_listed_with_its_versions(self) -> None:
        publish(self.root, CSS350, "v1", current=False)
        publish(self.root, CSS350, "v2")
        publish(self.root, CSS360, "v1")

        listed = helpers.list_available_courses(root=self.root)

        self.assertEqual(
            {item["courseId"] for item in listed}, {CSS350, CSS360}
        )
        css350 = next(item for item in listed if item["courseId"] == CSS350)
        self.assertEqual(css350["versions"], ["v1", "v2"])
        self.assertEqual(css350["currentVersion"], "v2")

    def test_an_incomplete_publication_is_not_listed(self) -> None:
        """A half-copied adapter must not look like something a course can serve."""
        (self.root / CSS350 / "v1" / "adapter").mkdir(parents=True)

        self.assertEqual(helpers.list_available_courses(root=self.root), [])

    def test_an_empty_serving_root_lists_nothing(self) -> None:
        self.assertEqual(helpers.list_available_courses(root=self.root), [])

    def test_a_missing_serving_root_is_not_an_error(self) -> None:
        self.assertEqual(
            helpers.list_available_courses(root=self.root / "nope"), []
        )


class SessionDeadlineTests(unittest.TestCase):
    def test_a_configured_deadline_is_read(self) -> None:
        with mock.patch.dict(
            os.environ, {"SERVICE_DEADLINE_EPOCH": "1756312021.4"}, clear=False
        ):
            self.assertAlmostEqual(helpers.resolve_session_deadline(), 1756312021.4)

    def test_an_absent_deadline_is_none_rather_than_a_guess(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SERVICE_DEADLINE_EPOCH", None)
            self.assertIsNone(helpers.resolve_session_deadline())

    def test_an_unparseable_deadline_is_none(self) -> None:
        with mock.patch.dict(
            os.environ, {"SERVICE_DEADLINE_EPOCH": "soon"}, clear=False
        ):
            self.assertIsNone(helpers.resolve_session_deadline())


class HuggingFaceAuthTests(unittest.TestCase):
    def test_assert_hf_auth_reads_token_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            token_path = Path(tmp) / "token"
            token_path.write_text("hf_test_token\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("HF_TOKEN", None)
                os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
                os.environ["HF_TOKEN_PATH"] = str(token_path)
                helpers.assert_hf_auth_available()
                self.assertEqual(os.environ.get("HF_TOKEN"), "hf_test_token")

    def test_assert_hf_auth_fails_without_credentials(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
            os.environ["HF_TOKEN_PATH"] = "/tmp/does-not-exist-hf-token"
            with mock.patch.dict(
                "sys.modules",
                {
                    "huggingface_hub": mock.MagicMock(
                        HfFolder=mock.MagicMock(get_token=mock.Mock(return_value=None))
                    )
                },
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    helpers.assert_hf_auth_available()
            self.assertIn("Hugging Face authentication", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
