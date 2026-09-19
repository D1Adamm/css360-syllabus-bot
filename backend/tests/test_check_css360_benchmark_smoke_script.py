"""`scripts/check_css360_benchmark_smoke.py` against responses the route really produces.

The checker is what the VM smoke reads its verdict from, on the loopback
request and on the one through Nginx alike, so it is driven here with genuine
route responses: a clean one passes every check, and each way a condition can
be wrong fails the check that names it. Its two pinned constants, the decoding
options and the template fingerprint, are held equal to the live values.
"""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from app.grounded_generation import grounded_options, prompt_template_fingerprint
from app.research_benchmark_client import BenchmarkConditionError
from test_research_benchmark_routes import PAIR, QUESTION, STANDALONE, BenchmarkRouteTestCase

pytestmark = pytest.mark.auth

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_css360_benchmark_smoke.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_css360_benchmark_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load()


def failed(checks) -> dict[str, str]:
    return {name: detail for name, ok, detail in checks if not ok}


class PinnedConstantsTests(unittest.TestCase):
    def test_the_decoding_and_template_constants_are_the_live_values(self) -> None:
        self.assertEqual(checker.EXPECTED_DECODING, grounded_options())
        self.assertEqual(checker.EXPECTED_TEMPLATE_SHA256, prompt_template_fingerprint())
        self.assertEqual(checker.COURSE_ID, "css-360-winter-2026-a7rp")
        self.assertEqual(checker.INTEGRITY_FLAGS, ("tagMatchesLineage", "decodingMatchesSpec", "promptEchoMatches"))


class CheckerTests(BenchmarkRouteTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.enable()

    def pair(self, condition: str = "ft_rag:v2") -> dict:
        response = self.post(PAIR, {"question": QUESTION, "conditions": [condition]})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_a_clean_pair_response_passes_every_check(self) -> None:
        checks = checker.check_response(self.pair(), condition="ft_rag:v2")
        self.assertEqual(failed(checks), {})
        self.assertEqual([name for name, _, _ in checks], [
            "route and course", "prompt template", "retrieval", "prompt hash", "decoding",
            "exactly one condition", "status ok, no error, scorable",
            "flag tagMatchesLineage", "flag decodingMatchesSpec", "flag promptEchoMatches",
            "served tag is the lineage tag", "digest recorded", "digest matches lineage",
            "answer has text", "timing recorded", "summary counts one attempt",
        ])

    def test_a_clean_standalone_response_passes(self) -> None:
        response = self.post(STANDALONE, {"question": QUESTION, "conditions": ["ft:v2"]})
        checks = checker.check_response(response.json(), condition="ft:v2")
        self.assertEqual(failed(checks), {})
        self.assertIn("no retrieval", [name for name, _, _ in checks])

    def test_the_wrong_condition_fails_by_name(self) -> None:
        checks = checker.check_response(self.pair(), condition="ft_rag:v3")
        self.assertEqual(set(failed(checks)), {"exactly one condition", "route and course"} - {"route and course"})

    def test_a_tag_mismatch_fails_the_flag_and_the_status(self) -> None:
        self.service.tags["v2"] = "css360-other:latest"
        names = set(failed(checker.check_response(self.pair(), condition="ft_rag:v2")))
        self.assertIn("flag tagMatchesLineage", names)
        self.assertIn("status ok, no error, scorable", names)
        self.assertIn("served tag is the lineage tag", names)
        self.assertIn("answer has text", names)
        self.assertNotIn("flag decodingMatchesSpec", names)
        self.assertNotIn("flag promptEchoMatches", names)

    def test_a_decoding_mismatch_fails_its_flag(self) -> None:
        self.service.options["v2"] = {**grounded_options(), "num_predict": 160}
        names = set(failed(checker.check_response(self.pair(), condition="ft_rag:v2")))
        self.assertIn("flag decodingMatchesSpec", names)
        self.assertNotIn("flag tagMatchesLineage", names)

    def test_an_empty_answer_fails_the_status_and_the_answer(self) -> None:
        self.service.answers["v2"] = ""
        names = set(failed(checker.check_response(self.pair(), condition="ft_rag:v2")))
        self.assertIn("status ok, no error, scorable", names)
        self.assertIn("answer has text", names)
        self.assertIn("summary counts one attempt", names)
        # The three flags are all true for a failed generation; only the outcome is not.
        self.assertFalse({"flag tagMatchesLineage", "flag decodingMatchesSpec", "flag promptEchoMatches"} & names)

    def test_a_missing_or_foreign_digest_fails_the_digest_checks(self) -> None:
        self.service.digests["v2"] = None
        names = set(failed(checker.check_response(self.pair(), condition="ft_rag:v2")))
        self.assertEqual(names, {"digest recorded", "digest matches lineage"})
        self.service.digests["v2"] = "e" * 64
        names = set(failed(checker.check_response(self.pair(), condition="ft_rag:v2")))
        # A foreign digest is now an invalid condition: the digest check and
        # the status, answer and summary checks all say so; the flags for the
        # tag, decoding and prompt still pass, and the digest was recorded.
        self.assertEqual(names, {"digest matches lineage", "status ok, no error, scorable",
                                 "answer has text", "summary counts one attempt"})

    def test_a_service_failure_fails_status_flags_and_digest(self) -> None:
        self.service.failures["v2"] = BenchmarkConditionError("service_unavailable", "The benchmark inference service is unavailable.")
        names = set(failed(checker.check_response(self.pair(), condition="ft_rag:v2")))
        self.assertIn("status ok, no error, scorable", names)
        self.assertIn("flag tagMatchesLineage", names)
        self.assertIn("digest recorded", names)
        self.assertIn("timing recorded", names)

    def test_two_conditions_or_a_non_object_fail_early(self) -> None:
        body = self.post(PAIR, {"question": QUESTION, "conditions": ["rag", "ft_rag:v2"]}).json()
        checks = checker.check_response(body, condition="ft_rag:v2")
        self.assertIn("exactly one condition", failed(checks))
        self.assertIn("response is a JSON object", failed(checker.check_response(["nope"], condition="rag")))

    def test_main_exits_zero_only_when_everything_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "smoke.json"
            path.write_text(json.dumps(self.pair()), encoding="utf-8")
            out = io.StringIO()
            with redirect_stdout(out):
                code = checker.main([str(path), "--condition", "ft_rag:v2"])
            self.assertEqual(code, 0, out.getvalue())
            text = out.getvalue()
            self.assertEqual(text.count("PASS "), 16)
            self.assertNotIn("FAIL", text)
            self.assertIn("servedTag: 'css360-ft-v2:latest'", text)
            self.assertIn("servedDigest: 'dea74c57f25a", text)
            self.assertIn("answer: 'Answer from v2.'", text)

            self.service.answers["v2"] = ""
            path.write_text(json.dumps(self.pair()), encoding="utf-8")
            out = io.StringIO()
            with redirect_stdout(out):
                code = checker.main([str(path), "--condition", "ft_rag:v2"])
            self.assertEqual(code, 1)
            self.assertIn("FAIL status ok, no error, scorable", out.getvalue())

            out = io.StringIO()
            with redirect_stdout(out):
                code = checker.main([str(Path(tmp) / "missing.json"), "--condition", "ft_rag:v2"])
            self.assertEqual(code, 1)
            self.assertIn("FAIL could not read", out.getvalue())


if __name__ == "__main__":
    unittest.main()
