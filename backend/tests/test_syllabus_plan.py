"""Tests for syllabus topic planning."""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, patch

from app.syllabus_plan import (
    DEFAULT_PLANNING_BATCH_SIZE,
    build_chunk_digests,
    build_section_title_fallback_plan,
    deterministic_merge_topics,
    merge_topics_by_overlap,
    plan_syllabus_topics,
    topic_is_schedule_heavy,
)


def _chunk(chunk_id: str, section_title: str, text: str) -> dict[str, str]:
    return {
        "chunkId": chunk_id,
        "sectionTitle": section_title,
        "text": text,
    }


class SyllabusPlanTests(unittest.IsolatedAsyncioTestCase):
    def test_build_chunk_digests_normalizes_text(self) -> None:
        digests = build_chunk_digests(
            [
                _chunk(
                    "chunk-001",
                    "Late Policy",
                    "Late   work may be submitted within 24 hours.\n\nHalf credit applies.",
                )
            ]
        )
        self.assertEqual(len(digests), 1)
        self.assertEqual(digests[0]["chunkId"], "chunk-001")
        self.assertEqual(digests[0]["sectionTitle"], "Late Policy")
        self.assertNotIn("\n", digests[0]["textDigest"])
        self.assertIn("Half credit applies.", digests[0]["textDigest"])

    def test_deterministic_merge_topics_merges_same_name_and_sources(self) -> None:
        merged = deterministic_merge_topics(
            [
                {
                    "topicId": "topic-01",
                    "name": "Late Work Policy",
                    "importance": "medium",
                    "sourceChunkIds": ["chunk-001"],
                    "suggestedExampleCount": 2,
                    "summary": "late work rules",
                    "scheduleHeavy": False,
                },
                {
                    "topicId": "topic-02",
                    "name": "late work policy",
                    "importance": "high",
                    "sourceChunkIds": ["chunk-001"],
                    "suggestedExampleCount": 3,
                    "summary": "extended late work rules",
                    "scheduleHeavy": False,
                },
            ]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["importance"], "high")
        self.assertEqual(merged[0]["suggestedExampleCount"], 3)

    def test_merge_topics_by_overlap_merges_shared_name_groups(self) -> None:
        merged = merge_topics_by_overlap(
            [
                {
                    "topicId": "topic-01",
                    "name": "Attendance Policy",
                    "importance": "medium",
                    "sourceChunkIds": ["chunk-001"],
                    "suggestedExampleCount": 2,
                    "summary": "attendance expectations",
                    "scheduleHeavy": False,
                },
                {
                    "topicId": "topic-02",
                    "name": "Attendance Policy",
                    "importance": "high",
                    "sourceChunkIds": ["chunk-002"],
                    "suggestedExampleCount": 3,
                    "summary": "absence reporting",
                    "scheduleHeavy": False,
                },
            ]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(sorted(merged[0]["sourceChunkIds"]), ["chunk-001", "chunk-002"])
        self.assertEqual(merged[0]["importance"], "high")

    def test_section_title_fallback_handles_varied_structures(self) -> None:
        plan = build_section_title_fallback_plan(
            [
                _chunk("chunk-001", "Projects", "Team project details."),
                _chunk("chunk-002", "Projects", "Weekly status report expectations."),
                _chunk("chunk-003", "Academic Integrity", "Cheating policy."),
            ],
            target_count=10,
        )
        self.assertEqual(len(plan["topics"]), 2)
        topics_by_name = {topic["name"]: topic for topic in plan["topics"]}
        self.assertEqual(
            topics_by_name["Projects"]["sourceChunkIds"],
            ["chunk-001", "chunk-002"],
        )

    def test_schedule_heavy_requires_primary_schedule_signals(self) -> None:
        self.assertTrue(topic_is_schedule_heavy("Exam schedule", "Midterm dates and final exam time"))
        self.assertFalse(topic_is_schedule_heavy("Course tools", "Canvas, Discord, and Zoom access"))

    async def test_plan_syllabus_topics_batches_and_merges(self) -> None:
        raw_chunks = [
            _chunk(
                f"chunk-{index:03d}",
                f"Section {index}",
                f"Chunk {index} text about course policy and projects." * 3,
            )
            for index in range(1, DEFAULT_PLANNING_BATCH_SIZE + 3)
        ]

        first_batch = json.dumps(
            {
                "topics": [
                    {
                        "name": "Project milestones",
                        "importance": "high",
                        "sourceChunkIds": ["chunk-001", "chunk-002"],
                        "suggestedExampleCount": 4,
                        "summary": "Milestone expectations and grading.",
                        "scheduleHeavy": False,
                    }
                ]
            }
        )
        second_batch = json.dumps(
            {
                "topics": [
                    {
                        "name": "Project Milestones",
                        "importance": "medium",
                        "sourceChunkIds": ["chunk-003"],
                        "suggestedExampleCount": 2,
                        "summary": "Project planning checkpoints.",
                        "scheduleHeavy": False,
                    },
                    {
                        "name": "Late submissions",
                        "importance": "high",
                        "sourceChunkIds": ["chunk-019"],
                        "suggestedExampleCount": 2,
                        "summary": "Deadlines and late penalties.",
                        "scheduleHeavy": True,
                    },
                ]
            }
        )
        merge_response = json.dumps(
            {
                "topics": [
                    {
                        "name": "Project milestones",
                        "importance": "high",
                        "sourceChunkIds": ["chunk-001", "chunk-002", "chunk-003"],
                        "suggestedExampleCount": 6,
                        "summary": "Project checkpoints and grading expectations.",
                        "scheduleHeavy": False,
                    },
                    {
                        "name": "Late submissions",
                        "importance": "high",
                        "sourceChunkIds": ["chunk-019"],
                        "suggestedExampleCount": 2,
                        "summary": "Late deadlines and penalties.",
                        "scheduleHeavy": True,
                    },
                ]
            }
        )

        with patch(
            "app.syllabus_plan.generate_ollama_completion",
            new=AsyncMock(
                side_effect=[
                    {"answer": first_batch, "model": "qwen3:4b"},
                    {"answer": second_batch, "model": "qwen3:4b"},
                    {"answer": merge_response, "model": "qwen3:4b"},
                ]
            ),
        ) as mock_generate:
            plan = await plan_syllabus_topics(raw_chunks=raw_chunks, target_count=10)

        self.assertEqual(mock_generate.await_count, 3)
        self.assertEqual(len(plan["topics"]), 2)
        self.assertEqual(plan["topics"][0]["name"], "Project milestones")
        self.assertEqual(
            plan["topics"][0]["sourceChunkIds"],
            ["chunk-001", "chunk-002", "chunk-003"],
        )
        self.assertTrue(plan["topics"][1]["scheduleHeavy"])

    async def test_plan_syllabus_topics_falls_back_to_section_titles(self) -> None:
        raw_chunks = [
            _chunk("chunk-001", "Office Hours", "Office hours are Tuesday afternoons." * 4),
            _chunk("chunk-002", "Late Policy", "Late work is accepted within 24 hours." * 4),
        ]

        with patch(
            "app.syllabus_plan.generate_ollama_completion",
            new=AsyncMock(
                side_effect=[
                    {"answer": json.dumps({"topics": []}), "model": "qwen3:4b"},
                    {"answer": json.dumps({"topics": []}), "model": "qwen3:4b"},
                ]
            ),
        ):
            plan = await plan_syllabus_topics(raw_chunks=raw_chunks, target_count=8, batch_size=1)

        self.assertEqual(sorted(topic["name"] for topic in plan["topics"]), ["Late Policy", "Office Hours"])
