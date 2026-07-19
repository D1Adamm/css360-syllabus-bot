"""Ad-hoc runner: build the real fact inventory for a stored course index.

Usage: .venv/bin/python scripts/run_fact_inventory.py <course_id>
Prints a JSON report to stdout and writes the full inventory to
scripts/last_fact_inventory.json.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ollama import generate_starter_ollama_completion  # noqa: E402
from app.syllabus_facts import (  # noqa: E402
    batch_section_groups,
    build_fact_inventory,
    build_section_groups,
)

INDEX_DIR = Path(__file__).resolve().parents[1] / "data" / "indexes"


async def main(course_id: str) -> None:
    index_path = INDEX_DIR / f"{course_id}.json"
    data = json.loads(index_path.read_text())
    chunks = data.get("chunks", [])
    groups = build_section_groups(chunks)
    batches = batch_section_groups(groups)
    print(
        f"Loaded {len(chunks)} chunks from {index_path.name} "
        f"({len(groups)} section groups -> {len(batches)} LLM batches)",
        flush=True,
    )

    call_count = 0

    async def tracked_completion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        print(f"  LLM batch {call_count}/{len(batches)} starting...", flush=True)
        started = time.monotonic()
        result = await generate_starter_ollama_completion(*args, **kwargs)
        elapsed = time.monotonic() - started
        answer_len = len(result.get("answer", "") or "")
        print(
            f"  LLM batch {call_count}/{len(batches)} done "
            f"({elapsed:.1f}s, answer_chars={answer_len})",
            flush=True,
        )
        return result

    started = time.monotonic()
    inventory = await build_fact_inventory(
        raw_chunks=chunks,
        completion_fn=tracked_completion,
    )
    elapsed = time.monotonic() - started

    out_path = Path(__file__).resolve().parent / "last_fact_inventory.json"
    out_path.write_text(json.dumps(inventory, indent=2))

    print(f"elapsedSeconds: {elapsed:.1f}", flush=True)
    print(f"factCount: {inventory['factCount']}", flush=True)
    print(f"droppedCount: {inventory['droppedCount']}", flush=True)
    print(f"duplicatesRemoved: {inventory['duplicatesRemoved']}", flush=True)
    print(f"fallbackUsed: {inventory['fallbackUsed']}", flush=True)
    print(f"countsByScope: {json.dumps(inventory['countsByScope'])}", flush=True)
    print(f"countsByKind: {json.dumps(inventory['countsByKind'])}", flush=True)
    print(f"countsBySeries: {json.dumps(inventory['countsBySeries'])}", flush=True)

    importance_counts: dict[str, int] = {}
    for fact in inventory["facts"]:
        importance_counts[fact["importance"]] = (
            importance_counts.get(fact["importance"], 0) + 1
        )
    print(f"countsByImportance: {json.dumps(importance_counts)}", flush=True)

    print("\n=== TOP 30 FACTS BY USEFULNESS ===", flush=True)
    for fact in inventory["facts"][:30]:
        series = (
            f" [{fact['seriesKey']}#{fact['seriesOrdinal']}]"
            if fact.get("seriesKey")
            else ""
        )
        print(
            f"- ({fact['importance']}/{fact['usefulnessScore']:.2f}) "
            f"[{fact['scope']}/{fact['kind']}]{series} {fact['statement']}",
            flush=True,
        )

    audit_terms = {
        "instructor/contact": ("instructor", "kaylea", "contact", "email"),
        "grading": ("grading", "grade scale", "percent", "weight"),
        "accommodations": ("accommodation", "disability", "religious"),
        "submission": ("canvas", "turn in", "submit", "11:59"),
        "team/project": ("group", "team", "standup", "scrum"),
        "missing class": ("absence", "miss class", "absent", "no call"),
        "help/support": ("office hours", "discord", "help", "open lab"),
        "extension limits": ("extension", "48-hour", "no extension"),
    }
    print("\n=== HIGH-VALUE AUDIT HITS ===", flush=True)
    for label, terms in audit_terms.items():
        hits = [
            f
            for f in inventory["facts"]
            if any(
                term in f["statement"].lower() or term in f["evidenceQuote"].lower()
                for term in terms
            )
        ]
        print(f"{label}: {len(hits)}", flush=True)
        for fact in hits[:2]:
            print(f"  - {fact['statement'][:140]}", flush=True)

    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    course = sys.argv[1] if len(sys.argv) > 1 else "css-360-winter-2026-a7rp"
    asyncio.run(main(course))
