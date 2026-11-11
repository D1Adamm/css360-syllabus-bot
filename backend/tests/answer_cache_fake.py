"""An in-memory stand-in for the answer cache's database and identity lookups.

The cache's rules (`app.answer_cache`) are tested without PostgreSQL or
Ollama: this replaces the three functions that touch the database and the one
that asks Ollama what is installed. The SQL itself is proven separately
against a real database in `test_answer_cache_postgres.py`.
"""

from __future__ import annotations

import os
import uuid
from typing import Any
from unittest.mock import patch

from app import answer_cache, answer_flights
from app.answer_identity import Identity, IdentityResult, fingerprint_of
from app.generation_queue import GenerationQueue, reset_generation_queue_for_tests


class FakeAnswerCache:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        #: course id -> the question texts staff selected.
        self.selected: dict[str, list[str]] = {}
        #: (condition, course id) -> identity fields, or a reason string.
        self.identities: dict[tuple[str, str], Any] = {}
        self.default_fields: dict[str, Any] = {"model": {"tag": "llama3.2:3b", "digest": "d1"}}
        self.fail_reads = False
        self.fail_writes = False
        self.fail_selection = False
        self.reads = 0
        self.writes = 0
        self.identity_calls: list[tuple[str, str, int, bool]] = []

    # -- what the tests arrange -------------------------------------------------

    def select(self, course_id: str, *questions: str) -> None:
        self.selected[course_id] = list(questions)
        answer_cache.invalidate_selection(course_id)

    def set_identity(self, condition: str, course_id: str, fields: Any) -> None:
        self.identities[(condition, course_id)] = fields

    def stored(self, course_id: str | None = None) -> list[dict[str, Any]]:
        return [row for row in self.rows.values() if course_id in (None, row["courseId"])]

    # -- the replaced functions -------------------------------------------------

    def _load_selected_keys(self, course_id: str) -> frozenset[str]:
        if self.fail_selection:
            raise RuntimeError("database down")
        return frozenset(answer_cache.question_key(q) for q in self.selected.get(course_id, []))

    def _read_entry(self, course_id: str, condition: str, key: str, fingerprint: str):
        self.reads += 1
        if self.fail_reads:
            raise RuntimeError("database down")
        return self.rows.get((course_id, condition, key, fingerprint))

    def _insert_entry(self, **fields: Any):
        self.writes += 1
        if self.fail_writes:
            raise RuntimeError("database down")
        key = (fields["course_id"], fields["condition"], fields["question_key"], fields["fingerprint"])
        if key in self.rows:
            return None
        row = {
            "entryId": uuid.uuid4().hex,
            "courseId": fields["course_id"],
            "condition": fields["condition"],
            "questionKey": fields["question_key"],
            "fingerprint": fields["fingerprint"],
            "questionText": fields["question_text"],
            "payload": dict(fields["payload"]),
            "identity": dict(fields["identity"]),
            "provenance": dict(fields["provenance"]),
            "source": fields["source"],
            "generatedAt": fields["generated_at"].isoformat(),
        }
        self.rows[key] = row
        return row

    async def _build_identity(self, condition: str, course_id: str, *, top_k: int, fresh: bool = False):
        self.identity_calls.append((condition, course_id, top_k, fresh))
        configured = self.identities.get((condition, course_id), self.default_fields)
        if isinstance(configured, str):
            return IdentityResult(None, configured)
        fields = {"condition": condition, "course": course_id, "topK": top_k, **configured}
        model = fields.get("model") or {}
        return IdentityResult(
            Identity(
                fields=fields,
                fingerprint=fingerprint_of(fields),
                model_version=model.get("courseVersion"),
                model_tag=model.get("tag"),
            )
        )

    # -- installation -----------------------------------------------------------

    def install(self, testcase: Any, *, enabled: bool = True) -> "FakeAnswerCache":
        for target, new in (
            ("app.answer_cache._load_selected_keys", self._load_selected_keys),
            ("app.answer_cache._read_entry", self._read_entry),
            ("app.answer_cache._insert_entry", self._insert_entry),
            ("app.answer_cache.build_identity", self._build_identity),
        ):
            patcher = patch(target, new=new)
            patcher.start()
            testcase.addCleanup(patcher.stop)
        env = patch.dict(os.environ, {answer_cache.ENABLED_ENV: "true" if enabled else ""})
        env.start()
        testcase.addCleanup(env.stop)
        reset_shared_state(GenerationQueue(max_concurrency=1, max_waiting=160, queue_timeout=30.0))
        testcase.addCleanup(reset_shared_state)
        return self


def reset_shared_state(queue: GenerationQueue | None = None) -> None:
    """No shared generations, no remembered selections, and the given queue
    (or, afterwards, the default one built on first use)."""
    answer_cache.invalidate_selection()
    answer_flights.reset_flights_for_tests()
    reset_generation_queue_for_tests(queue)
