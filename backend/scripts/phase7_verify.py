#!/usr/bin/env python3
"""Phase 7 multi-course verification (inventory + allocation + optional 5-seed smoke).

Runs ONE course at a time. Does not generate 50 live seeds.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.fact_inventory_cache import load_or_build_fact_inventory  # noqa: E402
from app.seed_allocation import allocate_slots  # noqa: E402
from app.seed_generation import generate_starter_seeds_for_course  # noqa: E402
from app.storage import LocalCourseArtifactStorage, get_course_artifact_storage  # noqa: E402

# User wrote css-350-winter-2026-dr1b; local index is css-350-winter-2026-drlb.
DEFAULT_COURSES = [
    "css-350-winter-2026-drlb",
    "css-360-winter-2026-a7rp",
    "css-490-spring-2026-cgvl",
]

CATEGORY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("attendance", re.compile(r"\b(attend|absence|miss(?:ing)? class|participation)\b", re.I)),
    ("late_work_extensions", re.compile(r"\b(late|extension|deadline|due date|half credit)\b", re.I)),
    ("grading", re.compile(r"\b(grad(?:e|ing)|weight|rubric|points?|percent)\b", re.I)),
    ("instructor_contact_help", re.compile(r"\b(office hours|instructor|email|contact|help|ta\b|canvas)\b", re.I)),
    ("assignments", re.compile(r"\b(assignment|homework|project|lab|deliverable)\b", re.I)),
    ("exams_quizzes", re.compile(r"\b(exam|midterm|final|quiz|test)\b", re.I)),
    ("tools_software", re.compile(r"\b(software|tool|ide|github|zoom|discord|python|java)\b", re.I)),
    ("accommodations", re.compile(r"\b(accommodat|disability|DRS|accessibil)\b", re.I)),
    ("team_project", re.compile(r"\b(team|group|collaborat|partner)\b", re.I)),
    ("submission", re.compile(r"\b(submit|submission|turn in|upload)\b", re.I)),
]


def _fact_text(fact: dict[str, Any]) -> str:
    return " ".join(
        [
            str(fact.get("statement") or ""),
            str(fact.get("evidenceQuote") or ""),
            str(fact.get("kind") or ""),
            str(fact.get("scope") or ""),
        ]
    )


def categorize_facts(facts: list[dict[str, Any]]) -> dict[str, int]:
    found: Counter[str] = Counter()
    for fact in facts:
        text = _fact_text(fact)
        matched = False
        for label, pattern in CATEGORY_PATTERNS:
            if pattern.search(text):
                found[label] += 1
                matched = True
        if not matched:
            found["other"] += 1
    return dict(found)


def allocation_breakdown(allocation: dict[str, Any]) -> dict[str, Any]:
    allocated = [
        item
        for item in allocation.get("allocations", [])
        if int(item.get("slotCount") or 0) > 0
    ]
    by_scope: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    by_series: Counter[str] = Counter()
    multi_slot = []
    for item in allocated:
        slots = int(item.get("slotCount") or 0)
        by_scope[str(item.get("scope") or "unknown")] += slots
        by_kind[str(item.get("kind") or "unknown")] += slots
        series = item.get("seriesKey")
        if series:
            by_series[str(series)] += slots
        if slots >= 2:
            multi_slot.append(
                {
                    "factId": item.get("factId"),
                    "slotCount": slots,
                    "statement": (item.get("statement") or "")[:120],
                    "kind": item.get("kind"),
                    "scope": item.get("scope"),
                }
            )
    suspicious: list[str] = []
    total_slots = sum(by_scope.values()) or 1
    deadlineish = by_kind.get("deadline", 0) + by_kind.get("schedule", 0)
    resourceish = by_kind.get("resource", 0) + by_kind.get("tool", 0)
    if deadlineish / total_slots > 0.35:
        suspicious.append(
            f"deadline/schedule kinds are {deadlineish}/{total_slots} slots "
            f"({deadlineish / total_slots:.0%})"
        )
    if resourceish / total_slots > 0.25:
        suspicious.append(
            f"resource/tool kinds are {resourceish}/{total_slots} slots "
            f"({resourceish / total_slots:.0%})"
        )
    for series, count in by_series.most_common(3):
        if count / total_slots > 0.2:
            suspicious.append(
                f"series {series!r} has {count}/{total_slots} slots "
                f"({count / total_slots:.0%})"
            )
    return {
        "allocatedSlots": int(
            allocation.get("summary", {}).get("allocatedSlots") or total_slots
        ),
        "allocatedFactCount": len(allocated),
        "byScope": dict(by_scope),
        "byKind": dict(by_kind),
        "bySeriesTop": dict(by_series.most_common(8)),
        "multiSlotFacts": multi_slot[:12],
        "suspicious": suspicious,
    }


def isolation_paths(course_id: str, storage: LocalCourseArtifactStorage) -> dict[str, Any]:
    index_path = storage.index_dir / f"{course_id}.json"
    facts_path = storage.index_dir / f"{course_id}.facts.json"
    return {
        "indexPath": str(index_path),
        "indexExists": index_path.is_file(),
        "factsCachePath": str(facts_path),
        "factsCacheExists": facts_path.is_file(),
        # Seeds are course-scoped by the primary key, not by a path.
        "seedScope": f"seed_examples.course_id = {course_id}",
    }


def leak_scan(
    *,
    course_id: str,
    facts: list[dict[str, Any]],
    seeds: list[dict[str, Any]],
) -> list[str]:
    """Heuristic cross-course leak checks using course-number markers."""
    issues: list[str] = []
    own = re.search(r"css-(\d+)", course_id, re.I)
    own_num = own.group(1) if own else None
    foreign_nums = {"350", "360", "490"} - ({own_num} if own_num else set())
    corpus = "\n".join(
        [
            *(_fact_text(f) for f in facts),
            *(
                f"{s.get('question','')} {s.get('answer','')} {s.get('evidenceQuote','')}"
                for s in seeds
            ),
        ]
    )
    for num in sorted(foreign_nums):
        # Require explicit CSS ### mentions to avoid false positives on years/points.
        if re.search(rf"\bCSS\s*{num}\b", corpus, re.I):
            issues.append(f"Found foreign course marker CSS {num} in {course_id} corpus")
    # Known CSS 360-specific phrases that should not appear elsewhere.
    if own_num != "360":
        for phrase in (
            "48-hour extension per quarter",
            "one 48-hour extension",
        ):
            if phrase.lower() in corpus.lower():
                issues.append(
                    f"Possible CSS 360 late-work phrase leaked into {course_id}: {phrase!r}"
                )
    return issues


async def verify_course(
    course_id: str,
    *,
    run_smoke: bool,
    force_refresh: bool,
) -> dict[str, Any]:
    storage = get_course_artifact_storage()
    assert isinstance(storage, LocalCourseArtifactStorage)
    index_data = storage.load_index(course_id)
    if index_data is None:
        return {"courseId": course_id, "error": "index_not_found"}

    raw_chunks = index_data.get("chunks") or []
    print(f"\n=== {course_id} ===", flush=True)
    print(f"chunks={len(raw_chunks)} forceRefresh={force_refresh}", flush=True)

    inventory = await load_or_build_fact_inventory(
        course_id=course_id,
        raw_chunks=raw_chunks,
        storage=storage,
        force_refresh=force_refresh,
    )
    facts = list(inventory.get("facts") or [])
    print(
        f"inventory: factCount={inventory.get('factCount')} "
        f"cached={inventory.get('cached')} "
        f"countsByKind={inventory.get('countsByKind')}",
        flush=True,
    )

    allocation = allocate_slots(facts, target_count=50)
    alloc_summary = allocation_breakdown(allocation)
    print(
        f"allocation(target=50): slots={alloc_summary['allocatedSlots']} "
        f"facts={alloc_summary['allocatedFactCount']} "
        f"suspicious={alloc_summary['suspicious']}",
        flush=True,
    )

    smoke: dict[str, Any] | None = None
    seeds: list[dict[str, Any]] = []
    if run_smoke:
        print("smoke: targetCount=5 save=false ...", flush=True)
        result = await generate_starter_seeds_for_course(
            course_id=course_id,
            target_count=5,
            save=False,
            force_refresh=False,
            storage=storage,
        )
        progress = result.get("progress") or {}
        seeds = list(result.get("seeds") or [])
        smoke = {
            "factInventoryCached": progress.get("factInventoryCached"),
            "factExtractionCalls": progress.get("factExtractionCalls"),
            "generationCalls": progress.get("generationCalls"),
            "validationCalls": progress.get("validationCalls"),
            "ollamaCalls": progress.get("ollamaCalls"),
            "candidatesGenerated": progress.get("candidatesGenerated"),
            "candidatesAccepted": progress.get("candidatesAccepted"),
            "candidatesRejected": progress.get("candidatesRejected"),
            "backfillAttempts": progress.get("backfillAttempts"),
            "finalCount": progress.get("finalCount"),
            "elapsedMs": progress.get("elapsedMs"),
            "status": progress.get("status"),
            "generationBatchCalls": progress.get("generationBatchCalls"),
            "validationBatchCalls": progress.get("validationBatchCalls"),
            "exampleSeeds": [
                {
                    "factId": seed.get("factId"),
                    "question": seed.get("question"),
                    "answer": (seed.get("answer") or "")[:180],
                    "category": seed.get("category"),
                    "evidenceQuote": (seed.get("evidenceQuote") or "")[:140],
                }
                for seed in seeds
            ],
        }
        print(
            f"smoke done: finalCount={smoke['finalCount']} "
            f"ollamaCalls={smoke['ollamaCalls']} "
            f"elapsedMs={smoke['elapsedMs']} status={smoke['status']}",
            flush=True,
        )

    leaks = leak_scan(course_id=course_id, facts=facts, seeds=seeds)
    paths = isolation_paths(course_id, storage)

    # Sample top facts by usefulness for the report.
    top_facts = sorted(
        facts,
        key=lambda f: float(f.get("usefulnessScore") or 0),
        reverse=True,
    )[:12]

    return {
        "courseId": course_id,
        "chunkCount": len(raw_chunks),
        "paths": paths,
        "inventory": {
            "factCount": inventory.get("factCount"),
            "cached": inventory.get("cached"),
            "countsByScope": inventory.get("countsByScope"),
            "countsByKind": inventory.get("countsByKind"),
            "categoryHits": categorize_facts(facts),
            "topFacts": [
                {
                    "factId": f.get("factId"),
                    "kind": f.get("kind"),
                    "scope": f.get("scope"),
                    "statement": (f.get("statement") or "")[:160],
                    "usefulnessScore": f.get("usefulnessScore"),
                }
                for f in top_facts
            ],
            "qualityFlags": _inventory_quality_flags(facts),
        },
        "allocation": alloc_summary,
        "smoke": smoke,
        "isolationIssues": leaks,
    }


def _inventory_quality_flags(facts: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    kinds = Counter(str(f.get("kind") or "unknown") for f in facts)
    total = len(facts) or 1
    if kinds.get("resource", 0) / total > 0.3:
        flags.append("resources may dominate inventory")
    if kinds.get("deadline", 0) / total > 0.35:
        flags.append("deadlines may dominate inventory")
    must_from_may = 0
    for fact in facts:
        stmt = str(fact.get("statement") or "")
        evidence = str(fact.get("evidenceQuote") or "")
        if re.search(r"\bmay\b", evidence, re.I) and re.search(
            r"\b(must|required)\b", stmt, re.I
        ):
            must_from_may += 1
    if must_from_may:
        flags.append(f"{must_from_may} facts look like may→must escalation in statement")
    return flags


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--courses",
        nargs="+",
        default=DEFAULT_COURSES,
        help="Course ids to verify (one at a time)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run targetCount=5 save=false after inventory/allocation",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Rebuild fact inventory even if cache exists",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "scripts" / "phase7_summary.json"),
        help="Write JSON summary path",
    )
    args = parser.parse_args()

    reports: list[dict[str, Any]] = []
    for course_id in args.courses:
        report = await verify_course(
            course_id,
            run_smoke=args.smoke,
            force_refresh=args.force_refresh,
        )
        reports.append(report)
        # Persist incrementally so a long run is not lost.
        Path(args.out).write_text(
            json.dumps({"courses": reports}, indent=2),
            encoding="utf-8",
        )

    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
