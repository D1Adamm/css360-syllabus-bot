"""The lineage projection: the real record, a fixed field list, no paths.

`evaluation/model_lineage.json` holds run directories, log files, Modelfile
lines and cluster paths beside the hashes and tags the benchmark needs. The
projection copies by name, and this is where that is held: every alias
resolves from the real file, the projected keys are exactly the documented
ones, and no value looks like a location.
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from app.research_benchmark_lineage import (
    ALIAS_LINEAGE_IDS,
    ALIASES,
    CSS360_COURSE_ID,
    DEFAULT_LINEAGE_PATH,
    EXPERIMENT_ALIASES,
    LINEAGE_SUMMARY_FIELDS,
    LineageUnavailable,
    lineage_for_alias,
    lineage_record_summary,
    load_lineage,
    reset_lineage_cache,
)

HEX64 = re.compile(r"^[0-9a-f]{64}$")
PATH_LIKE = ("vm:", "tillicum:", "$PROJ", "/gpfs", "~/", "/home/", ".gguf", ".log", "Modelfile",
             "model_artifacts", "training_outputs", "http://", "https://")


class RealRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_lineage_cache()
        self.addCleanup(reset_lineage_cache)
        self.record = load_lineage()

    def test_the_repository_record_loads_and_is_the_css360_record(self) -> None:
        self.assertEqual(DEFAULT_LINEAGE_PATH.name, "model_lineage.json")
        self.assertEqual(self.record["courseId"], CSS360_COURSE_ID)
        self.assertEqual(lineage_record_summary(self.record)["title"], "CSS 360 model lineage")
        self.assertEqual(lineage_record_summary(self.record)["schemaVersion"], 1)

    def test_the_aliases_are_the_four_experiments_and_the_control(self) -> None:
        self.assertEqual(ALIASES, ("base", "v2", "v3", "v4_vm", "v4_tillicum"))
        self.assertEqual(EXPERIMENT_ALIASES, ("v2", "v3", "v4_vm", "v4_tillicum"))
        self.assertEqual(set(ALIAS_LINEAGE_IDS), set(EXPERIMENT_ALIASES))
        recorded = {a["lineageId"] for a in self.record["artifacts"]}
        self.assertTrue(set(ALIAS_LINEAGE_IDS.values()) <= recorded)
        # v1 is in the record and deliberately not an alias.
        self.assertIn("css360-v1", recorded)
        self.assertNotIn("css360-v1", ALIAS_LINEAGE_IDS.values())

    def test_every_alias_projects_exactly_the_documented_fields(self) -> None:
        for alias in ALIASES:
            with self.subTest(alias=alias):
                summary = lineage_for_alias(alias, self.record)
                self.assertEqual(tuple(summary), LINEAGE_SUMMARY_FIELDS)
                self.assertEqual(summary["alias"], alias)
                self.assertTrue(summary["expectedTag"])

    def test_the_experiments_carry_their_hashes_tags_and_labels(self) -> None:
        expected = {
            "v2": ("css360-v2", "v2", "production", "css360-ft-v2:latest", "F32", "F32"),
            "v3": ("css360-v3", "v3", "experimental", "css360-cpu-v3-test:latest", "BF16", "F32"),
            "v4_vm": ("css360-v4-vm", None, "experimental", "css360-v4-test:latest", "BF16", "F16"),
            "v4_tillicum": ("css360-v4-tillicum", None, "experimental",
                            "css360-v4-tillicum-test:latest", "F32", "F32"),
        }
        for alias, (lineage_id, version, role, tag, adapter_dtype, gguf_dtype) in expected.items():
            with self.subTest(alias=alias):
                summary = lineage_for_alias(alias, self.record)
                self.assertEqual(summary["lineageId"], lineage_id)
                self.assertEqual(summary["version"], version)
                self.assertEqual(summary["role"], role)
                self.assertEqual(summary["expectedTag"], tag)
                self.assertEqual(summary["adapterDtype"], adapter_dtype)
                self.assertEqual(summary["ggufDtype"], gguf_dtype)
                self.assertRegex(summary["adapterSha256"], HEX64)
                self.assertRegex(summary["ggufBlobSha256"], HEX64)
                self.assertRegex(summary["gitCommitSha"], r"^[0-9a-f]{40}$")
                self.assertIsInstance(summary["ggufBytes"], int)
                self.assertTrue(summary["trainingLabel"])
                self.assertIs(summary["includeInControlledClaims"], True)

    def test_the_gguf_hash_is_the_adapter_blob_digest_without_its_prefix(self) -> None:
        artifact = next(a for a in self.record["artifacts"] if a["lineageId"] == "css360-v4-vm")
        self.assertEqual(artifact["ollama"]["adapterBlob"],
                         "sha256-" + lineage_for_alias("v4_vm", self.record)["ggufBlobSha256"])
        self.assertEqual(artifact["adapter"]["sha256"], lineage_for_alias("v4_vm", self.record)["adapterSha256"])

    def test_the_control_is_the_base_model(self) -> None:
        base = lineage_for_alias("base", self.record)
        self.assertEqual(base["role"], "base")
        self.assertIsNone(base["lineageId"])
        self.assertEqual(base["expectedTag"], "llama3.2:3b")
        self.assertEqual(base["baseQuantization"], "Q4_K_M")
        self.assertEqual(base["huggingFaceId"], "meta-llama/Llama-3.2-3B-Instruct")
        for key in ("adapterSha256", "ggufBlobSha256", "trainingLabel", "gitCommitSha", "version"):
            self.assertIsNone(base[key])

    def test_no_projected_value_looks_like_a_location(self) -> None:
        for alias in ALIASES:
            text = json.dumps(lineage_for_alias(alias, self.record))
            for marker in PATH_LIKE:
                with self.subTest(alias=alias, marker=marker):
                    self.assertNotIn(marker, text)
        # And the record itself does hold such values, so the projection is doing work.
        raw = DEFAULT_LINEAGE_PATH.read_text(encoding="utf-8")
        self.assertIn("vm:~/", raw)
        self.assertIn("/gpfs/", raw)

    def test_an_unknown_alias_is_refused(self) -> None:
        for bad in ("v1", "v5", "base ", "css360-v4-vm", ""):
            with self.subTest(alias=bad):
                with self.assertRaises(LineageUnavailable):
                    lineage_for_alias(bad, self.record)

    def test_a_missing_artifact_is_refused_by_lineage_id(self) -> None:
        record = json.loads(json.dumps(self.record))
        record["artifacts"] = [a for a in record["artifacts"] if a["lineageId"] != "css360-v3"]
        with self.assertRaises(LineageUnavailable) as caught:
            lineage_for_alias("v3", record)
        self.assertIn("css360-v3", str(caught.exception))
        # The others are unaffected.
        self.assertEqual(lineage_for_alias("v2", record)["lineageId"], "css360-v2")

    def test_the_record_is_cached_per_path_until_reset(self) -> None:
        self.assertIs(load_lineage(), self.record)
        reset_lineage_cache()
        self.assertIsNot(load_lineage(), self.record)


class BrokenRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_lineage_cache()
        self.addCleanup(reset_lineage_cache)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "model_lineage.json"

    def _write(self, record: object) -> None:
        self.path.write_text(json.dumps(record), encoding="utf-8")

    def _assert_refused(self) -> str:
        with self.assertRaises(LineageUnavailable) as caught:
            load_lineage(self.path)
        message = str(caught.exception)
        self.assertNotIn(self.tmp.name, message)
        self.assertNotIn("model_lineage.json", message)
        return message

    def test_a_missing_file(self) -> None:
        self.assertIn("could not be read", self._assert_refused())

    def test_invalid_json(self) -> None:
        self.path.write_text("{not json", encoding="utf-8")
        self.assertIn("not valid JSON", self._assert_refused())

    def test_the_wrong_shape_schema_or_course(self) -> None:
        good = json.loads(DEFAULT_LINEAGE_PATH.read_text(encoding="utf-8"))
        cases = {
            "a list": [],
            "schema 2": {**good, "schemaVersion": 2},
            "another course": {**good, "courseId": "css-350-spring-2026-n3h9"},
            "no artifacts": {**good, "artifacts": None},
            "no base model": {**good, "baseModel": None},
        }
        for label, record in cases.items():
            with self.subTest(case=label):
                reset_lineage_cache()
                self._write(record)
                self._assert_refused()

    def test_a_valid_copy_loads(self) -> None:
        self._write(json.loads(DEFAULT_LINEAGE_PATH.read_text(encoding="utf-8")))
        self.assertEqual(load_lineage(self.path)["courseId"], CSS360_COURSE_ID)


if __name__ == "__main__":
    unittest.main()
