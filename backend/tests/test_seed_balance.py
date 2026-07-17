"""Tests for starter seed balancing rules."""

from __future__ import annotations

import unittest

from app.seed_balance import (
    compute_schedule_cap,
    compute_scenario_minimum,
    compute_topic_cap,
    is_schedule_like_question,
    should_prefer_scenario_or_clarification,
    would_violate_balancing,
)


def _seed(
    question: str,
    *,
    category: str,
    question_type: str,
    topic_summary: str = "",
) -> dict[str, str]:
    return {
        "question": question,
        "category": category,
        "questionType": question_type,
        "topicSummary": topic_summary,
    }


class SeedBalanceTests(unittest.TestCase):
    def test_schedule_cap_defaults_to_20_percent(self) -> None:
        topics = [
            {
                "name": "Late work",
                "importance": "medium",
                "suggestedExampleCount": 2,
                "scheduleHeavy": True,
            },
            {
                "name": "Projects",
                "importance": "high",
                "suggestedExampleCount": 8,
                "scheduleHeavy": False,
            },
        ]
        self.assertEqual(compute_schedule_cap(10, topics), 2)

    def test_schedule_cap_allows_30_percent_for_major_schedule_syllabi(self) -> None:
        topics = [
            {
                "name": "Weekly schedule",
                "importance": "high",
                "suggestedExampleCount": 4,
                "scheduleHeavy": True,
            },
            {
                "name": "Assignments",
                "importance": "medium",
                "suggestedExampleCount": 6,
                "scheduleHeavy": False,
            },
        ]
        self.assertEqual(compute_schedule_cap(10, topics), 3)

    def test_schedule_classification_ignores_tool_mentions_alone(self) -> None:
        self.assertFalse(
            is_schedule_like_question(
                question="How do I join the class Discord?",
                category="Course tools",
            )
        )
        self.assertFalse(
            is_schedule_like_question(
                question="Where is the Zoom link posted?",
                category="Course tools",
            )
        )
        self.assertTrue(
            is_schedule_like_question(
                question="When is the midterm and what time does it start?",
                category="Exam schedule",
            )
        )

    def test_topic_cap_enforcement(self) -> None:
        accepted = [
            _seed("Q1?", category="Projects", question_type="direct"),
            _seed("Q2?", category="Projects", question_type="direct"),
        ]
        result = would_violate_balancing(
            candidate=_seed("Q3?", category="Projects", question_type="clarification"),
            accepted_seeds=accepted,
            target_count=10,
            planner_topics=[],
        )
        self.assertEqual(result, "topic_cap")
        self.assertEqual(compute_topic_cap(10), 2)

    def test_schedule_cap_enforcement(self) -> None:
        planner_topics = [
            {
                "name": "Schedule",
                "importance": "medium",
                "suggestedExampleCount": 2,
                "scheduleHeavy": True,
            },
            {
                "name": "Policies",
                "importance": "high",
                "suggestedExampleCount": 8,
                "scheduleHeavy": False,
            },
        ]
        accepted = [
            _seed("When is the first exam?", category="Schedule", question_type="direct"),
            _seed("What time are office hours?", category="Schedule", question_type="direct"),
        ]
        result = would_violate_balancing(
            candidate=_seed(
                "What is the deadline for project 1?",
                category="Schedule",
                question_type="clarification",
            ),
            accepted_seeds=accepted,
            target_count=10,
            planner_topics=planner_topics,
        )
        self.assertEqual(result, "schedule_cap")

    def test_scenario_minimum_preference(self) -> None:
        accepted = [
            _seed("Direct 1?", category="Policies", question_type="direct"),
            _seed("Direct 2?", category="Policies", question_type="direct"),
            _seed("Direct 3?", category="Projects", question_type="direct"),
            _seed("Scenario 1?", category="Projects", question_type="scenario"),
            _seed("Direct 4?", category="Projects", question_type="direct"),
            _seed("Direct 5?", category="Projects", question_type="direct"),
            _seed("Direct 6?", category="Projects", question_type="direct"),
            _seed("Direct 7?", category="Projects", question_type="direct"),
        ]
        self.assertEqual(compute_scenario_minimum(10), 3)
        self.assertTrue(
            should_prefer_scenario_or_clarification(
                accepted_seeds=accepted,
                remaining_slots=2,
                target_count=10,
            )
        )
