#!/usr/bin/env python3
"""Held-out questions must not be in the training data, in either format.

    python3 evaluation/check_training_overlap.py --course css-360-winter-2026-a7rp
    python3 evaluation/check_training_overlap.py --course css-360-winter-2026-a7rp \\
        --export-dir data/exports/css-360-winter-2026-a7rp \\
        --authored training/behaviour_seeds/css-360-winter-2026-a7rp.jsonl

Read-only. Compares every training-side question, the `question` of each
record in train.jsonl and validation.jsonl (for a grounded record that is the
question inside the rendered prompt, not the prompt) and every authored
behaviour seed, against every held-out question for the course:
`evaluation/held_out_questions.json` and
`evaluation/model_version_benchmark/questions.json`. Same three measures and
thresholds as `check_overlap.py`; a REJECT or REVIEW verdict exits non-zero.

`check_overlap.py` runs the opposite direction with a declared dataset
fingerprint. This one is for the moment before a training run: whatever is in
the export directory now, against the held-out sets as they are now.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

from check_overlap import nearest, read_jsonl_instructions, verdict_for  # noqa: E402

HELD_OUT_BANK = ROOT / "evaluation" / "held_out_questions.json"
BENCHMARK_BANK = ROOT / "evaluation" / "model_version_benchmark" / "questions.json"


def held_out_questions(course_id: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    bank = json.loads(HELD_OUT_BANK.read_text(encoding="utf-8"))
    for course in bank.get("courses", []):
        if course.get("courseId") != course_id:
            continue
        for item in course.get("questions", []):
            rows.append((f"held_out_questions.json:{item['id']}", item["question"]))
    if BENCHMARK_BANK.is_file():
        benchmark = json.loads(BENCHMARK_BANK.read_text(encoding="utf-8"))
        if benchmark.get("courseId") == course_id:
            for item in benchmark.get("questions", []):
                rows.append((f"model_version_benchmark/questions.json:{item['id']}", item["question"]))
    return rows


def training_questions(export_dir: Path | None, authored: Path | None) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if export_dir is not None:
        for name in ("train.jsonl", "validation.jsonl"):
            path = export_dir / name
            if path.is_file():
                rows.extend((name, question) for question in read_jsonl_instructions(path))
    if authored is not None and authored.is_file():
        rows.extend((authored.name, question) for question in read_jsonl_instructions(authored))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", required=True)
    parser.add_argument("--export-dir", type=Path, default=None,
                        help="default: data/exports/<course>")
    parser.add_argument("--authored", type=Path, default=None,
                        help="default: training/behaviour_seeds/<course>.jsonl")
    args = parser.parse_args()

    export_dir = args.export_dir or (ROOT / "data" / "exports" / args.course)
    authored = args.authored or (ROOT / "training" / "behaviour_seeds" / f"{args.course}.jsonl")
    held_out = held_out_questions(args.course)
    if not held_out:
        print(f"No held-out questions found for {args.course}; nothing to check against.")
        return 1
    candidates = training_questions(export_dir if export_dir.is_dir() else None, authored)
    if not candidates:
        print(f"No training-side questions found under {export_dir} or {authored}.")
        return 1

    print(f"=== {args.course}: {len(candidates)} training-side questions vs {len(held_out)} held-out ===")
    seen: set[str] = set()
    flagged = 0
    for label, question in candidates:
        key = question.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        scores, held_label, held_question = nearest(question, held_out)
        verdict = verdict_for(scores)
        if verdict != "clearly held out":
            flagged += 1
            print(
                f"! [{label}] {question}\n"
                f"    {verdict}: j={scores['jaccard']:.2f} c={scores['containment']:.2f} "
                f"r={scores['ratio']:.2f} vs [{held_label}] {held_question}"
            )
    print(f"checked {len(seen)} distinct questions; flagged {flagged}")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
