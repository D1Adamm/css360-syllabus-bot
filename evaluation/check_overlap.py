#!/usr/bin/env python3
"""Overlap check: held-out questions vs. the dataset that trained THAT course's adapter.

    python3 evaluation/check_overlap.py
    python3 evaluation/check_overlap.py --course css-350-spring-2026-n3h9

Read-only. Opens no database, starts no job, writes no file.

The rule this enforces
----------------------
A held-out question must not overlap with the dataset used to train that
course's adapter. Adapters are per course, so a CSS 360 training example cannot
leak into a CSS 350 answer; cross-course matches are printed as informational
and never change a verdict.

Why it refuses rather than guesses
----------------------------------
An earlier version of this script scored questions against every JSONL under
data/exports/, including stale directories for courses that are no longer
deployed. That produced confident verdicts derived from the wrong corpus. So
each course now declares the export directory AND the dataset fingerprint it
expects (dataset version and example counts). If the directory is absent, or
the manifest disagrees, the course is reported BLOCKED and no verdict is
issued for it. A missing dataset is a missing answer, not a passing grade.

Similarity
----------
Three measures, because "same intent, trivial rewording" hides from any one:

  jaccard      content-word overlap; catches reworded near-duplicates
               ("When does class meet?" / "What time are the course meetings?")
  containment  |A n B| / min(|A|,|B|); catches a trained question restated with
               padding, where jaccard is diluted by the extra words
  ratio        difflib character similarity; catches light edits

  REJECT  at jaccard >= 0.60, containment >= 0.75, or ratio >= 0.75
  REVIEW  at jaccard >= 0.45, containment >= 0.60, or ratio >= 0.60

A REVIEW verdict is not a pass. Per the research protocol, a question stays out
of the final set unless a human resolves it and records the reason.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "evaluation" / "held_out_questions.json"

#: Per course: where its prepared dataset lives, and what it must be.
#: Counts and datasetVersion come from the deployed VM, not from this checkout.
#: A course whose dataset is not present here is BLOCKED, never "clean".
DATASETS: dict[str, dict[str, object]] = {
    "css-350-spring-2026-n3h9": {
        "exportDir": "data/exports/css-350-spring-2026-n3h9",
        "datasetVersion": "css-350-spring-2026-n3h9-approved-split-seed360-n42",
        "approvedExamples": 42,
        "trainExamples": 37,
        "validationExamples": 5,
        "modelVersion": "v2",
        "trainingRunId": "run-20260827t205310z-8c3cdb",
    },
}

DATASET_FILES = ("approved-finetune.jsonl", "train.jsonl", "validation.jsonl")

STOPWORDS = {
    "a", "am", "an", "and", "any", "are", "as", "at", "be", "by", "can", "course",
    "did", "do", "does", "for", "from", "get", "have", "how", "i", "if", "in", "is",
    "it", "many", "me", "much", "my", "need", "of", "on", "or", "our", "should",
    "so", "the", "there", "this", "to", "we", "what", "when", "where", "which",
    "who", "why", "will", "with", "you", "your",
}

REJECT = {"jaccard": 0.60, "containment": 0.75, "ratio": 0.75}
REVIEW = {"jaccard": 0.45, "containment": 0.60, "ratio": 0.60}


def tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def scores(a: str, b: str) -> dict[str, float]:
    ta, tb = tokens(a), tokens(b)
    inter = len(ta & tb)
    return {
        "jaccard": inter / len(ta | tb) if ta and tb else 0.0,
        "containment": inter / min(len(ta), len(tb)) if ta and tb else 0.0,
        "ratio": SequenceMatcher(None, a.lower(), b.lower()).ratio(),
    }


def verdict_for(s: dict[str, float]) -> str:
    if any(s[k] >= REJECT[k] for k in REJECT):
        return "REJECT (too similar)"
    if any(s[k] >= REVIEW[k] for k in REVIEW):
        return "REVIEW (resolve by hand)"
    return "clearly held out"


def read_jsonl_instructions(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        question = (record.get("instruction") or record.get("question") or "").strip()
        if question:
            out.append(question)
    return out


def load_dataset(course_id: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Return ((label, question) rows, problems). Non-empty problems means BLOCKED."""
    spec = DATASETS.get(course_id)
    if spec is None:
        return [], [f"No dataset is declared for {course_id} in DATASETS."]

    export_dir = ROOT / str(spec["exportDir"])
    if not export_dir.is_dir():
        return [], [
            f"Dataset directory not present: {spec['exportDir']}",
            "This dataset lives on the UWB VM and has not been copied here.",
        ]

    problems: list[str] = []
    rows: list[tuple[str, str]] = []
    for name in DATASET_FILES:
        path = export_dir / name
        if not path.is_file():
            problems.append(f"Missing {spec['exportDir']}/{name}")
            continue
        for question in read_jsonl_instructions(path):
            rows.append((name, question))

    manifest_path = export_dir / "manifest.json"
    if not manifest_path.is_file():
        problems.append(f"Missing {spec['exportDir']}/manifest.json")
    else:
        manifest = json.loads(manifest_path.read_text())
        for key, expected in (
            ("datasetVersion", spec["datasetVersion"]),
            ("trainExamples", spec["trainExamples"]),
            ("validationExamples", spec["validationExamples"]),
        ):
            actual = manifest.get(key)
            if actual != expected:
                problems.append(
                    f"manifest {key} is {actual!r}, expected {expected!r} "
                    "- this is not the dataset the current model was trained on"
                )

    for name, expected in (
        ("approved-finetune.jsonl", spec["approvedExamples"]),
        ("train.jsonl", spec["trainExamples"]),
        ("validation.jsonl", spec["validationExamples"]),
    ):
        actual = sum(1 for label, _ in rows if label == name)
        if (export_dir / name).is_file() and actual != expected:
            problems.append(f"{name} has {actual} examples, expected {expected}")

    return rows, problems


def other_course_rows(exclude: str) -> list[tuple[str, str]]:
    """Informational corpus only. Never affects a verdict."""
    rows = []
    for path in sorted(ROOT.glob("data/exports/*/*.jsonl")):
        if path.parent.name == exclude:
            continue
        for question in read_jsonl_instructions(path):
            rows.append((f"{path.parent.name}/{path.name}", question))
    return rows


def nearest(question: str, corpus: list[tuple[str, str]]):
    best_label, best_other, best = "", "", {"jaccard": 0.0, "containment": 0.0, "ratio": 0.0}
    for label, other in corpus:
        s = scores(question, other)
        if max(s.values()) > max(best.values()):
            best, best_label, best_other = s, label, other
    return best, best_label, best_other


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", help="Only check this courseId")
    args = parser.parse_args()

    bank = json.loads(BANK.read_text())
    exit_code = 0

    for course in bank["courses"]:
        course_id = course["courseId"]
        if args.course and course_id != args.course:
            continue

        status = course.get("status", "")
        print(f"=== {course.get('courseCode', course_id)} ({course_id}) ===")
        print(f"    bank status: {status}")

        if str(status).startswith(("UNVERIFIED", "INVALID")):
            print("    SKIPPED - this block is not cleared for use; no verdicts issued.\n")
            exit_code = 1
            continue

        rows, problems = load_dataset(course_id)
        if problems:
            print("    BLOCKED - cannot check overlap:")
            for problem in problems:
                print(f"      - {problem}")
            print("    No verdicts issued. Absence of a dataset is not a pass.\n")
            exit_code = 1
            continue

        spec = DATASETS[course_id]
        print(f"    dataset: {spec['datasetVersion']}")
        print(f"    model {spec['modelVersion']} / run {spec['trainingRunId']}")
        print(f"    training-side questions: {len(rows)}")

        informational = other_course_rows(course_id)
        flagged = 0
        for item in course["questions"]:
            s, label, other = nearest(item["question"], rows)
            verdict = verdict_for(s)
            mark = " " if verdict == "clearly held out" else "!"
            if verdict != "clearly held out":
                flagged += 1
                exit_code = 1
            print(
                f"{mark} {item['id']}  j={s['jaccard']:.2f} c={s['containment']:.2f} "
                f"r={s['ratio']:.2f}  {verdict}"
            )
            if verdict != "clearly held out":
                print(f"      nearest in-dataset: [{label}] {other}")
            xs, xlabel, xother = nearest(item["question"], informational)
            if verdict_for(xs) != "clearly held out":
                print(
                    f"      (other course, informational only: j={xs['jaccard']:.2f} "
                    f"c={xs['containment']:.2f} r={xs['ratio']:.2f} [{xlabel}] {xother})"
                )
        print(f"    flagged: {flagged}\n")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
