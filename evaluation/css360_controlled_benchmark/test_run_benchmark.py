"""The controlled-benchmark runner's pure parts, and the question files.

    backend/.venv/bin/python -m pytest -q evaluation/css360_controlled_benchmark

No network: `http` is replaced by a scripted fake where a test needs the
request loop. The token test proves the token reaches the Authorization header
and nothing on disk.
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


rb = _load("run_benchmark")
SYLLABUS = " ".join((ROOT / "backend/course_data/css-360-winter-2026-a7rp/syllabus.txt").read_text(encoding="utf-8").split())


def question(qid="q01", kind="factual"):
    return {"id": qid, "kind": kind, "question": "When are office hours?", "keyFacts": []}


def payload(conditions, *, prompt_sha="p" * 64, set_sha="s" * 64, template="t" * 64, decoding=None, statuses=None):
    statuses = statuses or {}
    return {
        "route": "pair", "promptSha256": prompt_sha, "promptTemplate": {"name": "grounded-v1", "sha256": template},
        "retrieval": {"topK": 4, "chunkCount": 2, "chunkIds": ["c1", "c2"], "setSha256": set_sha, "facets": []},
        "decoding": decoding or {"num_predict": 256, "temperature": 0},
        "conditions": [
            {"condition": c, "alias": "base" if c in ("rag", "base") else c.split(":")[1], "kind": c.split(":")[0],
             "status": statuses.get(c, "ok"), "outcome": "scorable" if statuses.get(c, "ok") == "ok" else "error",
             "error": None if statuses.get(c, "ok") == "ok" else {"code": "empty_answer", "message": "x"},
             "answer": f"Answer {c}", "servedTag": "tag", "servedDigest": "d" * 64, "tagMatchesLineage": True,
             "digestMatchesLineage": True, "decodingMatchesSpec": True, "promptEchoMatches": True,
             "lineage": {"lineageId": "css360-v2", "expectedTag": "tag"},
             "timing": {"wallSeconds": 1.0, "generationSeconds": 0.9, "ollama": {"doneReason": "stop"}}}
            for c in conditions
        ],
        "summary": {"attempted": len(conditions)},
    }


class PureFunctionTests(unittest.TestCase):
    def test_groups_of_three_split_five_conditions_into_three_and_two(self) -> None:
        self.assertEqual(rb.groups(rb.CONDITIONS["pair"], 3), [["rag", "ft_rag:v2", "ft_rag:v3"], ["ft_rag:v4_vm", "ft_rag:v4_tillicum"]])
        self.assertEqual(rb.groups(rb.CONDITIONS["standalone"], 2), [["base", "ft:v2"], ["ft:v3", "ft:v4_vm"], ["ft:v4_tillicum"]])
        self.assertEqual(rb.groups(["a"], 3), [["a"]])
        self.assertEqual(rb.request_key("repeat22", "q01", "pair", 0), "repeat22/q01/pair/g0")

    def test_records_from_response_one_per_condition_in_group_order(self) -> None:
        group = ["rag", "ft_rag:v2", "ft_rag:v3"]
        records = rb.records_from_response("repeat22", question(), "pair", 0, group, payload(group, statuses={"ft_rag:v2": "error"}),
                                           wall=12.5, attempts=1, asked_at="t")
        self.assertEqual([r["condition"] for r in records], group)
        self.assertEqual([r["outcome"] for r in records], ["scorable", "error", "scorable"])
        self.assertEqual(records[0]["promptSha256"], "p" * 64)
        self.assertEqual(records[0]["retrievalSetSha256"], "s" * 64)
        self.assertEqual(records[0]["chunkIds"], ["c1", "c2"])
        self.assertEqual(records[0]["requestWallSeconds"], 12.5)
        self.assertEqual(records[1]["error"]["code"], "empty_answer")

    def test_a_condition_missing_from_the_response_is_recorded_as_a_request_failure(self) -> None:
        group = ["rag", "ft_rag:v2"]
        records = rb.records_from_response("repeat22", question(), "pair", 0, group, payload(["rag"]), wall=1, attempts=1, asked_at="t")
        self.assertEqual(records[1]["outcome"], "request_failed")
        self.assertEqual(records[1]["alias"], "v2")
        self.assertIn("missing", records[1]["error"]["message"])

    def test_records_from_failure_covers_every_condition_of_the_group(self) -> None:
        records = rb.records_from_failure("heldout", question("h01"), "standalone", 1, ["ft:v4_vm", "ft:v4_tillicum"],
                                          status=503, error="HTTP 503", wall=2.0, attempts=4, asked_at="t")
        self.assertEqual([(r["condition"], r["alias"], r["outcome"], r["requestStatus"], r["requestAttempts"]) for r in records],
                         [("ft:v4_vm", "v4_vm", "request_failed", 503, 4), ("ft:v4_tillicum", "v4_tillicum", "request_failed", 503, 4)])

    def test_grounded_identity_across_groups(self) -> None:
        same = rb.grounded_identity([payload(["rag"]), payload(["ft_rag:v4_vm"])])
        self.assertTrue(same["valid"]) and self.assertEqual(same["mismatches"], {})
        self.assertEqual(same["promptSha256"], "p" * 64)
        different = rb.grounded_identity([payload(["rag"]), payload(["ft_rag:v4_vm"], set_sha="z" * 64)])
        self.assertFalse(different["valid"])
        self.assertIn("retrievalSetSha256", different["mismatches"])
        self.assertNotIn("promptSha256", different["mismatches"])
        decoding = rb.grounded_identity([payload(["rag"]), payload(["ft_rag:v2"], decoding={"num_predict": 160, "temperature": 0})])
        self.assertIn("decoding", decoding["mismatches"])
        self.assertTrue(rb.grounded_identity([payload(["rag"])])["valid"])
        self.assertFalse(rb.grounded_identity([])["valid"])

    def test_similarity_thresholds_and_fact_containment(self) -> None:
        rows = [{"file": "train.jsonl", "format": "bare", "kind": "answerable",
                 "question": "When and where does CSS 360 meet?", "response": "Tuesday and Thursday, 3:30 PM to 5:30 PM in UW2 room 131."}]
        near = rb.overlap_for_question("When and where does the CSS 360 class meet?", ["Tuesday and Thursday", "UW2 room 131", "97.0% maps to a 4.0"], rows)
        self.assertEqual(near["questionVerdict"], "REJECT (too similar)")
        self.assertEqual(near["keyFactCoverage"], "2/3")
        far = rb.overlap_for_question("Who is the guest speaker on October 28?", ["Tiffany Chen"], rows)
        self.assertEqual(far["questionVerdict"], "clearly held out")
        self.assertEqual(far["keyFactCoverage"], "0/1")

    def test_lineage_expectations_read_the_record(self) -> None:
        expected = rb.lineage_expectations()
        self.assertEqual(expected["v4_vm"], {"tag": "css360-v4-test:latest", "digest": "b8763e820e93", "lineageId": "css360-v4-vm"})
        self.assertEqual(expected["base"]["tag"], "llama3.2:3b")
        self.assertEqual(expected["_v1_note"]["tag"], "css360-ft-v1-test:latest")
        self.assertIs(expected["_v1_note"]["includeInControlledClaims"], False)

    def test_service_health_comparison(self) -> None:
        expected = rb.lineage_expectations()
        health = {"aliases": [
            {"alias": a, "ollamaModel": expected[a]["tag"].replace(":latest", ""), "mapped": True, "available": True,
             "digest": expected[a]["digest"] + "0" * 52} for a in rb.ALIASES]}
        self.assertTrue(rb.compare_service_health(health, expected)["ok"])
        health["aliases"][2]["digest"] = "e" * 64
        result = rb.compare_service_health(health, expected)
        self.assertFalse(result["ok"])
        self.assertIs(result["aliases"]["v3"]["digestMatches"], False)
        self.assertTrue(result["aliases"]["v2"]["digestMatches"])
        health["aliases"][2]["digest"] = expected["v3"]["digest"] + "1" * 52
        health["aliases"][4]["available"] = False
        self.assertFalse(rb.compare_service_health(health, expected)["ok"])

    def test_read_token_from_env_file_last_wins_and_env_var_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text("A=1\nCSS360_BENCHMARK_TOKEN='first'\nexport CSS360_BENCHMARK_TOKEN=\"second-token-value\"\n")
            with mock.patch.dict(os.environ, {"CSS360_BENCHMARK_TOKEN": ""}):
                self.assertEqual(rb.read_token(env), "second-token-value")
            with mock.patch.dict(os.environ, {"CSS360_BENCHMARK_TOKEN": "from-env"}):
                self.assertEqual(rb.read_token(env), "from-env")
            self.assertIsNone(rb.read_token(Path(tmp) / "missing"))


class FakeHttp:
    """Scripted `http`: a list of (status, body_or_None, error) per call, with recorded requests."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def __call__(self, method, url, *, body=None, token=None, timeout=30.0):
        self.calls.append({"method": method, "url": url, "body": body, "token": token, "timeout": timeout})
        status, payload_, error = self.script.pop(0)
        headers = {"retry-after": "1"} if status == 429 else {}
        return rb.HttpResult(status, payload_, json.dumps(payload_) if payload_ is not None else "", 0.01, error=error, headers=headers)


class RunLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.args = SimpleNamespace(origin="http://127.0.0.1:8001", group_size=3, pause_seconds=0, service_health="x")
        self.run = rb.Run(Path(self.tmp.name) / "run-test", self.args, "SECRET-TOKEN-" + "z" * 40)
        sleep = mock.patch.object(rb.time, "sleep")
        sleep.start()
        self.addCleanup(sleep.stop)

    def test_ask_sends_the_token_and_saves_the_raw_response(self) -> None:
        group = ["rag", "ft_rag:v2", "ft_rag:v3"]
        fake = FakeHttp([(200, payload(group), None)])
        with mock.patch.object(rb, "http", fake):
            records, body = self.run.ask("repeat22", question(), "pair", 0, group)
        self.assertEqual(fake.calls[0]["token"], self.run.token)
        self.assertEqual(fake.calls[0]["url"], "http://127.0.0.1:8001/api/research/css360/benchmark/pair")
        self.assertEqual(fake.calls[0]["body"], {"question": "When are office hours?", "conditions": group})
        self.assertGreaterEqual(fake.calls[0]["timeout"], 3 * 125)
        self.assertEqual(len(records), 3)
        self.assertEqual(self.run.completed_keys(), {"repeat22/q01/pair/g0"})
        saved = json.loads(self.run.response_path("repeat22/q01/pair/g0").read_text())
        self.assertEqual(saved["httpStatus"], 200)
        self.assertEqual(saved["response"]["promptSha256"], "p" * 64)

    def test_a_429_and_a_503_are_retried_and_a_persistent_failure_is_recorded(self) -> None:
        group = ["base", "ft:v2"]
        fake = FakeHttp([(429, {"detail": {"code": "busy"}}, None), (503, {"detail": "down"}, None), (200, payload(group), None)])
        with mock.patch.object(rb, "http", fake):
            records, body = self.run.ask("heldout", question("h01"), "standalone", 0, group)
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(records[0]["requestAttempts"], 3)
        self.assertTrue(all(r["outcome"] == "scorable" for r in records))

        fake = FakeHttp([(0, None, "ConnectionError: refused")] * rb.MAX_ATTEMPTS)
        with mock.patch.object(rb, "http", fake):
            records, body = self.run.ask("heldout", question("h02"), "standalone", 1, ["ft:v4_vm"])
        self.assertIsNone(body)
        self.assertEqual(records[0]["outcome"], "request_failed")
        self.assertEqual(records[0]["requestAttempts"], rb.MAX_ATTEMPTS)
        self.assertNotIn("heldout/h02/standalone/g1", self.run.completed_keys())

    def test_a_422_is_not_retried(self) -> None:
        fake = FakeHttp([(422, {"detail": "bad"}, None)])
        with mock.patch.object(rb, "http", fake):
            records, _ = self.run.ask("heldout", question("h01"), "standalone", 0, ["base"])
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(records[0]["requestStatus"], 422)

    def test_execute_skips_finished_requests_records_identity_and_never_writes_the_token(self) -> None:
        q1, q2 = question("q01"), question("q02")
        plan = [("repeat22", q1, "pair", 0, ["rag", "ft_rag:v2", "ft_rag:v3"]), ("repeat22", q1, "pair", 1, ["ft_rag:v4_vm", "ft_rag:v4_tillicum"]),
                ("repeat22", q2, "pair", 0, ["rag", "ft_rag:v2", "ft_rag:v3"]), ("repeat22", q2, "pair", 1, ["ft_rag:v4_vm", "ft_rag:v4_tillicum"])]
        script = [(200, payload(plan[0][4]), None), (200, payload(plan[1][4]), None),
                  (200, payload(plan[2][4]), None), (200, payload(plan[3][4], set_sha="y" * 64), None)]
        fake = FakeHttp(script)
        with mock.patch.object(rb, "http", fake):
            self.run.execute(plan, label="full")
        self.assertEqual(len(fake.calls), 4)
        identity = [json.loads(l) for l in (self.run.dir / "grounded_identity.jsonl").read_text().splitlines()]
        self.assertEqual([(i["questionId"], i["valid"]) for i in identity], [("q01", True), ("q02", False)])
        records = [json.loads(l) for l in self.run.records_path.read_text().splitlines()]
        self.assertEqual(len(records), 10)
        # Resume: nothing is asked again and nothing is duplicated.
        fake2 = FakeHttp([])
        with mock.patch.object(rb, "http", fake2):
            self.run.execute(plan, label="full")
        self.assertEqual(fake2.calls, [])
        self.assertEqual(len(self.run.records_path.read_text().splitlines()), 10)
        status = rb.compute_status(self.run.dir)
        self.assertEqual(status["conditionRecords"], 10)
        self.assertEqual(status["groundedComparisonsInvalid"], 1)
        # The token is in no file the run wrote.
        for path in self.run.dir.rglob("*"):
            if path.is_file():
                self.assertNotIn(self.run.token, path.read_text(encoding="utf-8"), path.name)

    def test_verify_pilot_reports_missing_and_failed_requests(self) -> None:
        plan = [("repeat22", question("q01"), "standalone", 0, ["base", "ft:v2", "ft:v3"])]
        self.assertTrue(any("no saved response" in p for p in self.run.verify_pilot(plan)))
        fake = FakeHttp([(200, payload(plan[0][4]), None)])
        with mock.patch.object(rb, "http", fake):
            self.run.execute(plan, label="pilot")
        self.assertEqual(self.run.verify_pilot(plan), [])


class QuestionFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repeat = json.loads((HERE / "questions_repeat22.json").read_text(encoding="utf-8"))
        cls.heldout = json.loads((HERE / "questions_heldout.json").read_text(encoding="utf-8"))
        cls.original = json.loads((ROOT / "evaluation/model_version_benchmark/questions.json").read_text(encoding="utf-8"))

    def test_repeat22_is_the_original_text_with_kinds(self) -> None:
        self.assertEqual([q["question"] for q in self.repeat["questions"]], [q["question"] for q in self.original["questions"]])
        self.assertEqual(len(self.repeat["questions"]), 22)
        kinds = {q["id"]: q["kind"] for q in self.repeat["questions"]}
        self.assertEqual(kinds["q14"], "false_premise")
        self.assertEqual({k for k, v in kinds.items() if v == "unanswerable"}, {"q19", "q20", "q21", "q22"})
        self.assertTrue(all(v in ("factual", "unanswerable", "false_premise") for v in kinds.values()))

    def test_heldout_composition_ids_and_passages(self) -> None:
        qs = self.heldout["questions"]
        self.assertEqual(len(qs), len({q["id"] for q in qs}))
        counts = {}
        for q in qs:
            counts[q["kind"]] = counts.get(q["kind"], 0) + 1
        self.assertEqual(counts, {"factual": 10, "unanswerable": 7, "false_premise": 7})
        for q in qs:
            self.assertTrue(q["supportingPassages"], q["id"])
            for passage in q["supportingPassages"]:
                self.assertIn(" ".join(passage.split()), SYLLABUS, f"{q['id']}: {passage[:60]}")
            self.assertTrue(q["keyFacts"], q["id"])
            self.assertTrue(q["referenceAnswer"], q["id"])
            if q["kind"] == "false_premise":
                self.assertTrue(q.get("premise"), q["id"])

    def test_heldout_questions_are_not_repeat22_questions(self) -> None:
        repeat_texts = {q["question"] for q in self.repeat["questions"]}
        for q in self.heldout["questions"]:
            self.assertNotIn(q["question"], repeat_texts)
            for other in self.repeat["questions"]:
                self.assertEqual(rb.verdict_for(rb.similarity(q["question"], other["question"])), "clearly held out", (q["id"], other["id"]))

    def test_every_question_carries_the_july_overlap_block(self) -> None:
        for data in (self.repeat, self.heldout):
            for q in data["questions"]:
                self.assertIn("questionVerdict", q["overlapJulySplit"], q["id"])
                self.assertIn("keyFactCoverage", q["overlapJulySplit"], q["id"])


if __name__ == "__main__":
    unittest.main()
