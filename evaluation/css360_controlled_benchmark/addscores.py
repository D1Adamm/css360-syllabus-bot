#!/usr/bin/env python3
"""Append blind scores from compact lines on stdin to results/<run>/blind/scores_blind.jsonl.

Line format (fields separated by single spaces, then two ' | ' separated texts):

    <set>/<qid> <letter> <correctness> <groundedness> <completeness> <hallucination T|F> <abstained T|F|n> <premise c|a|e|n> | <cited passage> | <rationale>

`n` means not applicable. A line for a (question, letter) already scored replaces the earlier one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREMISE = {"c": "corrected", "a": "accepted", "e": "evaded", "n": "n/a"}
ABSTAIN = {"T": True, "F": False, "n": "n/a"}


def main() -> int:
    run = sys.argv[1]
    path = HERE / "results" / run / "blind" / "scores_blind.jsonl"
    existing = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                existing[(r["question"], r["letter"])] = r
    added = 0
    for raw in sys.stdin.read().splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        head, _, rest = raw.partition(" | ")
        cite, _, why = rest.partition(" | ")
        parts = head.split()
        if len(parts) != 8:
            print("BAD LINE:", raw, file=sys.stderr)
            return 1
        q, letter, c, g, k, h, a, p = parts
        record = {
            "question": q, "letter": letter, "correctness": int(c), "groundedness": int(g), "completeness": int(k),
            "hallucination": h == "T", "abstained": ABSTAIN[a], "premiseHandling": PREMISE[p],
            "citedPassage": cite.strip(), "rationale": why.strip(),
        }
        for field in ("correctness", "groundedness", "completeness"):
            assert 1 <= record[field] <= 5, raw
        existing[(q, letter)] = record
        added += 1
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in existing.values()), encoding="utf-8")
    print(f"scored {added} lines; total {len(existing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
