"""A rating records which stored classroom answer it was made against."""

from __future__ import annotations

import unittest

from app.db_evaluations import MODEL_KEYS, validate_responses
from app.db_schemas import EvaluationResponseSnapshot


def _responses(**overrides):
    base = {key: {"text": f"{key} answer", "error": None, "sources": [], "model": "m"} for key in MODEL_KEYS}
    for key, extra in overrides.items():
        base[key] = {**base[key], **extra}
    return base


class RatingProvenanceTests(unittest.TestCase):
    def test_a_stored_answers_id_and_whether_it_was_served_from_the_store_are_kept(self) -> None:
        cleaned = validate_responses(_responses(
            base={"cacheEntryId": "a" * 32, "cached": True},
            rag={"cacheEntryId": "b" * 32, "cached": False},
        ))
        self.assertEqual((cleaned["base"]["cacheEntryId"], cleaned["base"]["cached"]), ("a" * 32, True))
        # Generated for this request and stored then: the id, and cached false.
        self.assertEqual((cleaned["rag"]["cacheEntryId"], cleaned["rag"]["cached"]), ("b" * 32, False))

    def test_an_ordinary_live_answer_keeps_the_shape_it_always_had(self) -> None:
        cleaned = validate_responses(_responses(base={"cached": True}))
        for key in MODEL_KEYS:
            self.assertEqual(
                set(cleaned[key]), {"text", "error", "sources", "model", "modelVersion", "engine"}
            )

    def test_a_malformed_id_is_refused(self) -> None:
        for bad in (123, "", "   ", "x" * 65, ["a"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_responses(_responses(base={"cacheEntryId": bad}))

    def test_the_request_model_carries_both_fields_through(self) -> None:
        snapshot = EvaluationResponseSnapshot.model_validate(
            {"text": "x", "cacheEntryId": "a" * 32, "cached": True, "somethingElse": 1}
        )
        dumped = snapshot.model_dump(by_alias=True, exclude_unset=True)
        self.assertEqual(dumped, {"text": "x", "cacheEntryId": "a" * 32, "cached": True})


if __name__ == "__main__":
    unittest.main()
