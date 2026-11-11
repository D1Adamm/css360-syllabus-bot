"""An in-memory stand-in for the answer cache's database and identity lookups.

The cache's and preparation's rules (`app.answer_cache`,
`app.classroom_preparation`) are tested without PostgreSQL or Ollama: this
replaces the functions that touch the database and the one that asks Ollama
what is installed. The eligibility rule is the real one
(`db_classroom_questions.map_question`), applied to in-memory rows. The SQL
itself is proven separately against a real database in
`test_answer_cache_postgres.py`.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

from app import answer_cache, answer_flights, classroom_preparation, db_classroom_questions
from app.answer_identity import Identity, IdentityResult, fingerprint_of
from app.db_mapping import to_iso
from app.generation_queue import GenerationQueue, reset_generation_queue_for_tests


class FakeAnswerCache:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        #: course id -> its questions, as `seed_examples` rows.
        self.seeds: dict[str, list[dict[str, Any]]] = {}
        #: course id -> selected seed ids, in selection order.
        self.selection: dict[str, list[str]] = {}
        #: course id -> the preparation status row.
        self.runs: dict[str, dict[str, Any]] = {}
        #: (condition, course id) -> identity fields, or a reason string.
        self.identities: dict[tuple[str, str], Any] = {}
        self.default_fields: dict[str, Any] = {"model": {"tag": "llama3.2:3b", "digest": "d1"}}
        self.fail_reads = False
        self.fail_writes = False
        self.fail_selection = False
        self.fail_plan = False
        self.reads = 0
        self.writes = 0
        self.identity_calls: list[tuple[str, str, int, bool]] = []

    # -- what the tests arrange -------------------------------------------------

    def add_question(
        self,
        course_id: str,
        question: str,
        *,
        seed_id: str | None = None,
        origin: str = "prototype",
        review_status: str | None = "generated",
        participant_id: str | None = None,
        selected: bool = False,
    ) -> str:
        rows = self.seeds.setdefault(course_id, [])
        seed_id = seed_id or f"s{len(rows) + 1}"
        rows.append(
            {
                "seed_id": seed_id,
                "instruction": question,
                "origin": origin,
                "review_status": review_status,
                "status": None,
                "participant_id": participant_id,
                "category": "general",
            }
        )
        if selected:
            self.selection.setdefault(course_id, []).append(seed_id)
        answer_cache.invalidate_selection(course_id)
        return seed_id

    def select(self, course_id: str, *questions: str) -> None:
        """Make the course's questions, and its selection, exactly these."""
        self.seeds[course_id] = []
        self.selection[course_id] = []
        for question in questions:
            self.add_question(course_id, question, selected=True)
        answer_cache.invalidate_selection(course_id)

    def set_selection(self, course_id: str, seed_ids: list[str]) -> None:
        self.selection[course_id] = list(seed_ids)
        answer_cache.invalidate_selection(course_id)

    def set_identity(self, condition: str, course_id: str, fields: Any) -> None:
        self.identities[(condition, course_id)] = fields

    def stored(self, course_id: str | None = None) -> list[dict[str, Any]]:
        return [row for row in self.rows.values() if course_id in (None, row["courseId"])]

    # -- the replaced functions: answer_cache ------------------------------------

    def _questions(self, course_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        selection = self.selection.get(course_id, [])
        rows = {row["seed_id"]: row for row in self.seeds.get(course_id, [])}
        listing = [
            db_classroom_questions.map_question(row, selected=row["seed_id"] in selection)
            for row in rows.values()
        ]
        listing = [item for item in listing if item["eligible"] or item["selected"]]
        selected = [
            db_classroom_questions.map_question(rows[seed_id], selected=True)
            for seed_id in selection
            if seed_id in rows
        ]
        return listing, [item for item in selected if item["eligible"]]

    def _load_selected_keys(self, course_id: str) -> frozenset[str]:
        if self.fail_selection:
            raise RuntimeError("database down")
        return frozenset(answer_cache.question_key(q["question"]) for q in self._questions(course_id)[1])

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

    # -- the replaced functions: classroom_preparation ---------------------------

    def _read_plan_rows(self, course_id: str):
        if self.fail_plan:
            raise RuntimeError("database down")
        listing, selected = self._questions(course_id)
        return listing, selected, self.runs.get(course_id)

    def _read_entry_keys(self, course_id: str, keys: list[str]):
        wanted = set(keys)
        return [
            {"condition": row["condition"], "questionKey": row["questionKey"], "fingerprint": row["fingerprint"]}
            for row in self.rows.values()
            if row["courseId"] == course_id and row["questionKey"] in wanted
        ]

    def _begin(self, course_id: str, started_by: str | None) -> None:
        self.runs[course_id] = {
            "courseId": course_id,
            "state": "running",
            "bootId": classroom_preparation.BOOT_ID,
            "startedAt": to_iso(datetime.now(timezone.utc)),
            "stopRequestedAt": None,
            "finishedAt": None,
            "reason": None,
            "message": None,
            "generated": 0,
            "failed": 0,
        }

    def _write_stopping(self, course_id: str) -> None:
        row = self.runs.get(course_id)
        if row is not None and row["state"] == "running":
            row["state"] = "stopping"

    def _write_progress(self, run: Any) -> None:
        row = self.runs.get(run.course_id)
        if row is not None:
            row.update(generated=run.generated, failed=run.failed)

    def _write_finish(self, run: Any, state: str, reason: str | None, message: str | None) -> None:
        row = self.runs.get(run.course_id)
        if row is not None:
            row.update(
                state=state, reason=reason, message=message,
                generated=run.generated, failed=run.failed,
                finishedAt=to_iso(datetime.now(timezone.utc)),
            )

    # -- installation -----------------------------------------------------------

    def install(self, testcase: Any, *, enabled: bool = True) -> "FakeAnswerCache":
        for target, new in (
            ("app.answer_cache._load_selected_keys", self._load_selected_keys),
            ("app.answer_cache._read_entry", self._read_entry),
            ("app.answer_cache._insert_entry", self._insert_entry),
            ("app.answer_cache.build_identity", self._build_identity),
            ("app.classroom_preparation.build_identity", self._build_identity),
            ("app.classroom_preparation._read_plan_rows", self._read_plan_rows),
            ("app.classroom_preparation._read_entry_keys", self._read_entry_keys),
            ("app.classroom_preparation._begin", self._begin),
            ("app.classroom_preparation._write_stopping", self._write_stopping),
            ("app.classroom_preparation._write_progress", self._write_progress),
            ("app.classroom_preparation._write_finish", self._write_finish),
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
    """No shared generations, no remembered selections, no preparation run, and
    the given queue (or, afterwards, the default one built on first use)."""
    answer_cache.invalidate_selection()
    answer_flights.reset_flights_for_tests()
    classroom_preparation.reset_for_tests()
    reset_generation_queue_for_tests(queue)
