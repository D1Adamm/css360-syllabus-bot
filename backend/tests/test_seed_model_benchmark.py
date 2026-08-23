"""Focused tests for offline seed-model benchmark helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.seed_model_benchmark import (
    compare_benchmark_summaries,
    evaluate_fact_candidate,
    force_model_completion,
    run_model_benchmark,
    select_benchmark_facts,
    summarize_benchmark_candidates,
    write_benchmark_json,
)
from app.storage import LocalCourseArtifactStorage


def _fact(
    fact_id: str,
    *,
    usefulness: float = 0.9,
    importance: str = "high",
    importance_score: float = 0.9,
    ask: float = 0.85,
    complexity: int = 2,
    scope: str = "course_wide",
    kind: str = "late_work",
) -> dict:
    return {
        "factId": fact_id,
        "statement": f"Students may use one 48-hour extension for {fact_id}.",
        "evidenceQuote": f"Evidence for {fact_id}: one 48-hour extension allowed.",
        "sourceChunkIds": [f"chunk-{fact_id}"],
        "scope": scope,
        "kind": kind,
        "importance": importance,
        "importanceScore": importance_score,
        "usefulnessScore": usefulness,
        "studentAskLikelihood": ask,
        "complexity": complexity,
        "seriesKey": None,
        "assignmentGroup": None,
        "seriesOrdinal": None,
    }


def _passing_validation(*, score_hint: float = 0.9) -> dict:
    # Top-level component keys as required by try_parse_validation_payload.
    return {
        "reason": "ok",
        "unsupportedClaims": [],
        "grounded": score_hint,
        "correct": score_hint,
        "useful": score_hint,
        "clear": score_hint,
        "naturalStudentWording": score_hint,
        "categoryCorrect": score_hint,
        "notTrivialOrTemporary": score_hint,
    }


class SelectBenchmarkFactsTests(unittest.TestCase):
    def test_selects_fixed_unique_facts_deterministically(self) -> None:
        facts = [
            _fact(f"fact-{index:02d}", usefulness=0.95 - (index * 0.01))
            for index in range(1, 21)
        ]
        first = select_benchmark_facts(facts, count=10)
        second = select_benchmark_facts(facts, count=10)

        self.assertEqual(len(first), 10)
        self.assertEqual(
            [item["fact"]["factId"] for item in first],
            [item["fact"]["factId"] for item in second],
        )
        self.assertEqual(len({item["fact"]["factId"] for item in first}), 10)


class SummarizeBenchmarkTests(unittest.TestCase):
    def test_summary_counts_and_averages(self) -> None:
        candidates = [
            {
                "status": "accepted",
                "validationScore": 0.9,
                "unsupportedClaims": [],
                "qualifierMismatch": False,
                "rejectionStage": None,
                "generationTimeSeconds": 2.0,
                "elapsedSeconds": 2.5,
            },
            {
                "status": "rejected",
                "validationScore": 0.5,
                "unsupportedClaims": ["claim-a", "claim-b"],
                "qualifierMismatch": True,
                "rejectionStage": "prevalidation",
                "generationTimeSeconds": 1.0,
                "elapsedSeconds": 1.0,
            },
            {
                "status": "rejected",
                "validationScore": None,
                "unsupportedClaims": [],
                "qualifierMismatch": False,
                "rejectionStage": "validation",
                "generationTimeSeconds": 1.5,
                "elapsedSeconds": 2.0,
            },
        ]
        summary = summarize_benchmark_candidates(candidates)
        self.assertEqual(summary["acceptedCount"], 1)
        self.assertEqual(summary["unsupportedClaimCount"], 2)
        self.assertEqual(summary["qualifierMismatchCount"], 1)
        self.assertEqual(summary["prevalidationRejectionCount"], 1)
        self.assertEqual(summary["averageValidationScore"], 0.7)
        self.assertEqual(summary["averageTimePerAcceptedSeedSeconds"], 2.5)

    def test_comparison_delta(self) -> None:
        left = {
            "model": "qwen3:4b",
            "summary": {
                "acceptedCount": 4,
                "averageValidationScore": 0.7,
                "unsupportedClaimCount": 3,
                "qualifierMismatchCount": 2,
                "prevalidationRejectionCount": 1,
                "averageTimePerAcceptedSeedSeconds": 2.0,
            },
        }
        right = {
            "model": "qwen3:8b",
            "summary": {
                "acceptedCount": 7,
                "averageValidationScore": 0.8,
                "unsupportedClaimCount": 1,
                "qualifierMismatchCount": 0,
                "prevalidationRejectionCount": 0,
                "averageTimePerAcceptedSeedSeconds": 3.5,
            },
        }
        comparison = compare_benchmark_summaries(left, right)
        self.assertEqual(comparison["metrics"]["acceptedCount"]["delta"], 3)
        self.assertEqual(
            comparison["metrics"]["unsupportedClaimCount"]["delta"], -2
        )


class ForceModelCompletionTests(unittest.IsolatedAsyncioTestCase):
    async def test_overrides_model_kwarg(self) -> None:
        base = AsyncMock(return_value={"answer": "{}", "model": "qwen3:8b"})
        forced = force_model_completion("qwen3:4b", base)
        await forced("prompt", model="ignored", response_format="json")
        self.assertEqual(base.await_args.kwargs["model"], "qwen3:4b")


class EvaluateFactCandidateTests(unittest.IsolatedAsyncioTestCase):
    async def test_prevalidation_rejection_records_qualifier_mismatch(self) -> None:
        fact = _fact("fact-01")
        completion = AsyncMock(
            return_value={
                "answer": json.dumps(
                    {
                        "seeds": [
                            {
                                "question": "What is the absolute deadline?",
                                "answer": "Work is always due with no exceptions.",
                                "category": "policy",
                                "questionType": "direct",
                            }
                        ]
                    }
                ),
                "model": "qwen3:4b",
            }
        )
        with patch(
            "app.seed_model_benchmark.prevalidate_candidate",
            return_value={
                "reason": "missing_qualifier:unless",
                "category": "qualifier_mismatch",
            },
        ):
            row = await evaluate_fact_candidate(
                model="qwen3:4b",
                fact=fact,
                allocation={"suggestedStyles": ["policy"]},
                chunk_texts=["Evidence for fact-01"],
                completion_fn=completion,
            )

        self.assertEqual(row["status"], "rejected")
        self.assertTrue(row["qualifierMismatch"])
        self.assertEqual(row["rejectionStage"], "prevalidation")
        self.assertEqual(row["rejectionReason"], "missing_qualifier:unless")
        self.assertEqual(row["evidenceQuote"], fact["evidenceQuote"])

    async def test_accepted_candidate_records_score(self) -> None:
        fact = _fact("fact-02")

        async def _completion(prompt: str, **kwargs):
            if "You generate training seed examples" in prompt or (
                "ONE syllabus fact" in prompt
            ):
                return {
                    "answer": json.dumps(
                        {
                            "seeds": [
                                {
                                    "question": "What is the late work policy?",
                                    "answer": (
                                        "Late work may be accepted within "
                                        "the stated window."
                                    ),
                                    "category": "policy",
                                    "questionType": "direct",
                                    "sourceChunkIds": ["chunk-1"],
                                }
                            ]
                        }
                    ),
                    "model": kwargs.get("model", "qwen3:4b"),
                }
            return {
                "answer": json.dumps(_passing_validation()),
                "model": kwargs.get("model", "qwen3:4b"),
            }

        with patch(
            "app.seed_model_benchmark.prevalidate_candidate",
            return_value=None,
        ):
            row = await evaluate_fact_candidate(
                model="qwen3:4b",
                fact=fact,
                allocation={"suggestedStyles": ["policy"]},
                chunk_texts=["Evidence for fact-02"],
                completion_fn=_completion,
            )

        self.assertEqual(row["status"], "accepted")
        self.assertIsNotNone(row["validationScore"])
        self.assertIsNone(row["rejectionReason"])
        self.assertEqual(row["unsupportedClaims"], [])


class RunModelBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_facts_produce_ten_rows_and_persist_nothing(self) -> None:
        facts = [_fact(f"fact-{index:02d}") for index in range(1, 12)]
        selected = select_benchmark_facts(facts, count=5)
        self.assertEqual(len(selected), 5)
        chunk_lookup = {
            f"chunk-{fact['factId']}": {
                "chunkId": f"chunk-{fact['factId']}",
                "text": fact["evidenceQuote"],
            }
            for fact in facts
        }

        async def _fake_evaluate(**kwargs):
            fact = kwargs["fact"]
            return {
                "model": kwargs["model"],
                "factId": fact["factId"],
                "question": "Q?",
                "answer": "A.",
                "evidenceQuote": fact["evidenceQuote"],
                "validationScore": 0.8,
                "rejectionReason": None,
                "rejectionStage": None,
                "unsupportedClaims": [],
                "qualifierMismatch": False,
                "generationTimeSeconds": 0.5,
                "status": "accepted",
            }

        with patch(
            "app.seed_model_benchmark.evaluate_fact_candidate",
            new=AsyncMock(side_effect=_fake_evaluate),
        ) as mock_evaluate:
            payload = await run_model_benchmark(
                model="qwen3:8b",
                selected=selected,
                chunk_lookup=chunk_lookup,
            )

        self.assertEqual(payload["model"], "qwen3:8b")
        self.assertEqual(len(payload["candidates"]), 5)
        self.assertEqual(payload["summary"]["acceptedCount"], 5)
        self.assertEqual(mock_evaluate.await_count, 5)

    async def test_write_benchmark_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            written = write_benchmark_json(
                path,
                {"model": "qwen3:4b", "candidates": [], "summary": {}},
            )
            data = json.loads(written.read_text(encoding="utf-8"))
            self.assertEqual(data["model"], "qwen3:4b")


class LoadContextUsesCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_load_benchmark_context_reuses_cached_inventory(self) -> None:
        from app.seed_model_benchmark import load_benchmark_context

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = LocalCourseArtifactStorage(
                root_dir=root / "course_data",
                index_dir=root / "indexes",
            )
            course_id = "css-360-bench-test"
            chunks = [
                {
                    "chunkId": "chunk-1",
                    "text": "Late work may be accepted with instructor approval.",
                    "order": 0,
                }
            ]
            storage.save_index(course_id, {"chunks": chunks})

            facts = [_fact(f"fact-{index:02d}") for index in range(1, 12)]

            inventory = {
                "model": "qwen3:4b",
                "facts": facts,
                "factCount": len(facts),
                "droppedCount": 0,
            }

            async def _fake_load(**kwargs):
                return {**inventory, "cached": True}

            with patch(
                "app.seed_model_benchmark.load_or_build_fact_inventory",
                new=AsyncMock(side_effect=_fake_load),
            ) as mock_load:
                context = await load_benchmark_context(
                    course_id=course_id,
                    fact_count=10,
                    storage=storage,
                )

            mock_load.assert_awaited_once()
            self.assertEqual(len(context["selected"]), 10)
            self.assertEqual(len(context["factIds"]), 10)


if __name__ == "__main__":
    unittest.main()
