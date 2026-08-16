"""Tests for starter seed balancing rules."""

from __future__ import annotations

import unittest

from app.seed_balance import (
    compute_schedule_cap,
    compute_scenario_minimum,
    compute_topic_cap,
    count_scenario_or_clarification,
    is_schedule_like_question,
    scenario_requirement_remaining,
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


class CourseWideScenarioBalanceTests(unittest.TestCase):
    """Balance is a property of the course, not of one top-up run's slice.

    Modelled on CSS 350: 9 existing seeds, a 41-slot generation ceiling, and an
    achievable course total of 50. Measuring the requirement against the 41 —
    or counting only the 41 — describes a course nobody has.
    """

    CSS350_EXISTING = 9
    CSS350_CEILING = 41
    CSS350_TOTAL = CSS350_EXISTING + CSS350_CEILING

    def test_css350_requirement_is_measured_against_the_whole_course(self) -> None:
        self.assertEqual(compute_scenario_minimum(self.CSS350_TOTAL), 15)
        # Not the generation slice, which is the number the run used to report.
        self.assertEqual(compute_scenario_minimum(self.CSS350_CEILING), 13)

    def test_existing_scenarios_count_toward_the_requirement(self) -> None:
        accepted = [
            _seed("Scenario 1?", category="Policies", question_type="scenario"),
            _seed("Scenario 2?", category="Policies", question_type="clarification"),
        ]

        without_existing = scenario_requirement_remaining(
            accepted_seeds=accepted,
            target_count=20,
        )
        with_existing = scenario_requirement_remaining(
            accepted_seeds=accepted,
            target_count=20,
            existing_scenario_count=4,
        )

        self.assertEqual(compute_scenario_minimum(20), 6)
        self.assertEqual(without_existing, 4)
        self.assertEqual(with_existing, 0)

    def test_css350_deficit_uses_existing_plus_accepted(self) -> None:
        """CSS 350 as it actually stood: zero scenarios anywhere."""
        remaining = scenario_requirement_remaining(
            accepted_seeds=[],
            target_count=self.CSS350_TOTAL,
            existing_scenario_count=0,
        )
        self.assertEqual(remaining, 15)

        # A course whose existing seeds already carry the balance owes nothing.
        satisfied = scenario_requirement_remaining(
            accepted_seeds=[],
            target_count=self.CSS350_TOTAL,
            existing_scenario_count=15,
        )
        self.assertEqual(satisfied, 0)

    def test_preference_is_urgency_based_not_always_on(self) -> None:
        """Early slots stay free; preference engages only when it must."""
        early = should_prefer_scenario_or_clarification(
            accepted_seeds=[],
            remaining_slots=self.CSS350_CEILING,
            target_count=self.CSS350_TOTAL,
            existing_scenario_count=0,
        )
        late = should_prefer_scenario_or_clarification(
            accepted_seeds=[],
            remaining_slots=15,
            target_count=self.CSS350_TOTAL,
            existing_scenario_count=0,
        )

        self.assertFalse(early)
        self.assertTrue(late)

    def test_existing_scenarios_relieve_the_pressure_on_a_top_up(self) -> None:
        relieved = should_prefer_scenario_or_clarification(
            accepted_seeds=[],
            remaining_slots=15,
            target_count=self.CSS350_TOTAL,
            existing_scenario_count=15,
        )
        self.assertFalse(relieved)

    def test_no_preference_once_slots_are_gone(self) -> None:
        self.assertFalse(
            should_prefer_scenario_or_clarification(
                accepted_seeds=[],
                remaining_slots=0,
                target_count=self.CSS350_TOTAL,
                existing_scenario_count=0,
            )
        )

    def test_urgency_follows_eligible_slots_not_total_slots(self) -> None:
        """The instrumented CSS 350 failure, as arithmetic.

        41 slots left, 15 owed, but only 7 of those slots belong to facts that
        could carry a scenario. Judged against the 41 the deficit looks
        comfortable and preference waits — until the 7 have been spent on
        whatever the allocator suggested and nothing eligible is left.
        """
        by_total_slots = should_prefer_scenario_or_clarification(
            accepted_seeds=[],
            remaining_slots=41,
            target_count=50,
            existing_scenario_count=0,
        )
        by_eligible_slots = should_prefer_scenario_or_clarification(
            accepted_seeds=[],
            remaining_slots=41,
            target_count=50,
            existing_scenario_count=0,
            eligible_slots_remaining=7,
        )

        self.assertFalse(by_total_slots)
        self.assertTrue(by_eligible_slots)

    def test_plentiful_eligible_slots_still_defer(self) -> None:
        """Preference stays off while eligible capacity comfortably exceeds it."""
        self.assertFalse(
            should_prefer_scenario_or_clarification(
                accepted_seeds=[],
                remaining_slots=41,
                target_count=50,
                existing_scenario_count=0,
                eligible_slots_remaining=40,
            )
        )

    def test_a_met_requirement_never_prefers(self) -> None:
        """Zero deficit must not read as urgent just because nothing is left."""
        self.assertFalse(
            should_prefer_scenario_or_clarification(
                accepted_seeds=[],
                remaining_slots=5,
                target_count=50,
                existing_scenario_count=15,
                eligible_slots_remaining=0,
            )
        )

    def test_counting_helper_reads_question_type(self) -> None:
        seeds = [
            _seed("A?", category="Policies", question_type="scenario"),
            _seed("B?", category="Policies", question_type="clarification"),
            _seed("C?", category="Policies", question_type="direct"),
            _seed("D?", category="Policies", question_type="procedure"),
            {"question": "E?"},
        ]
        self.assertEqual(count_scenario_or_clarification(seeds), 2)
