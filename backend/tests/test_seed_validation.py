"""Tests for rubric-based starter-seed validation."""

from __future__ import annotations

import json
import unittest

from app.seed_validation import (
    MIN_OVERALL_VALIDATION_SCORE,
    PERFECT_COMPONENT_SOFT_CAP,
    build_validation_prompt,
    calibrate_validation_result,
    canonicalize_validation_result,
    try_parse_validation_payload,
    validation_result_accepts,
)


def _payload(**overrides: object) -> str:
    base: dict[str, object] = {
        "grounded": 0.9,
        "correct": 0.92,
        "clear": 0.82,
        "useful": 0.88,
        "naturalStudentWording": 0.81,
        "categoryCorrect": 0.84,
        "notTrivialOrTemporary": 0.8,
        "unsupportedClaims": [],
        "reason": "Good student-style syllabus question.",
    }
    base.update(overrides)
    return json.dumps(base)


def _perfect_payload() -> str:
    return _payload(
        grounded=1.0,
        correct=1.0,
        clear=1.0,
        useful=1.0,
        naturalStudentWording=1.0,
        categoryCorrect=1.0,
        notTrivialOrTemporary=1.0,
        reason="Exceptionally strong with no weakness.",
    )


class SeedValidationTests(unittest.TestCase):
    def test_prompt_includes_conservative_guidance_and_examples(self) -> None:
        prompt = build_validation_prompt(
            question="Can I submit an assignment late?",
            answer="Late work is accepted within 24 hours for half credit.",
            topic_name="Late work policy",
            question_type="scenario",
            chunk_text="Late work is accepted within 24 hours for half credit.",
        )
        self.assertIn("Late work policy", prompt)
        self.assertIn("scenario", prompt)
        self.assertIn("0.75–0.92", prompt)
        self.assertIn("unsupportedClaims", prompt)
        self.assertIn("Partly grounded but embellished", prompt)
        self.assertIn("Do NOT include an overall score field", prompt)

    def test_parse_validation_payload_computes_server_score(self) -> None:
        raw = _payload()
        self.assertNotIn("score", json.loads(raw))
        parsed = try_parse_validation_payload(raw)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertIn("components", parsed)
        self.assertEqual(parsed["unsupportedClaims"], [])
        self.assertGreaterEqual(parsed["score"], MIN_OVERALL_VALIDATION_SCORE)

    def test_parse_ignores_model_provided_overall_score(self) -> None:
        parsed = try_parse_validation_payload(_payload(score=0.11))
        assert parsed is not None
        self.assertNotEqual(parsed["score"], 0.11)
        self.assertGreater(parsed["score"], 0.7)

    def test_bad_category_rejection(self) -> None:
        parsed = try_parse_validation_payload(_payload(categoryCorrect=0.62))
        assert parsed is not None
        self.assertFalse(validation_result_accepts(parsed))

    def test_low_grounding_rejection(self) -> None:
        parsed = try_parse_validation_payload(_payload(grounded=0.79))
        assert parsed is not None
        self.assertFalse(validation_result_accepts(parsed))

    def test_accepts_when_all_thresholds_pass(self) -> None:
        parsed = try_parse_validation_payload(_payload())
        assert parsed is not None
        self.assertTrue(validation_result_accepts(parsed))

    def test_rejects_low_natural_student_wording(self) -> None:
        parsed = try_parse_validation_payload(_payload(naturalStudentWording=0.55))
        assert parsed is not None
        self.assertFalse(validation_result_accepts(parsed))

    def test_unsupported_claims_cause_rejection(self) -> None:
        parsed = try_parse_validation_payload(
            _payload(
                unsupportedClaims=[
                    "Recommended work-division strategy not stated in the syllabus."
                ]
            )
        )
        assert parsed is not None
        self.assertFalse(validation_result_accepts(parsed))

    def test_embellished_answers_are_rejected_by_calibration(self) -> None:
        parsed = try_parse_validation_payload(
            _payload(
                grounded=1.0,
                correct=1.0,
                unsupportedClaims=[],
                reason="Looks good.",
            )
        )
        assert parsed is not None
        calibrated = calibrate_validation_result(
            result=parsed,
            question="How should we organize the team project?",
            answer=(
                "Team projects require weekly status updates and a final demo. "
                "I recommend you divide work so one person owns docs, one owns coding, "
                "and one owns testing."
            ),
            topic_name="Team Project Requirements",
            question_type="scenario",
        )
        self.assertFalse(validation_result_accepts(calibrated))
        self.assertGreater(len(calibrated["unsupportedClaims"]), 0)
        self.assertLess(calibrated["components"]["grounded"], 0.80)

    def test_perfect_scores_are_soft_capped(self) -> None:
        parsed = try_parse_validation_payload(_perfect_payload())
        assert parsed is not None
        calibrated = calibrate_validation_result(
            result=parsed,
            question="What is the late policy?",
            answer="Late work may be submitted within 24 hours for half credit.",
            topic_name="Late Work Policy",
            question_type="direct",
        )
        self.assertLess(calibrated["score"], 1.0)
        self.assertLessEqual(
            max(calibrated["components"].values()),
            PERFECT_COMPONENT_SOFT_CAP,
        )

    def test_canonical_structure_for_persistence(self) -> None:
        parsed = try_parse_validation_payload(_payload())
        assert parsed is not None
        canonical = canonicalize_validation_result(parsed)
        self.assertEqual(
            set(canonical.keys()),
            {"score", "reason", "unsupportedClaims", "components"},
        )
        self.assertEqual(
            set(canonical["components"].keys()),
            {
                "grounded",
                "correct",
                "clear",
                "useful",
                "naturalStudentWording",
                "categoryCorrect",
                "notTrivialOrTemporary",
            },
        )
