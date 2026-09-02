#!/usr/bin/env python3
"""Read-only overlap check: held-out questions vs. every training corpus on disk.

Nothing here writes, trains, or touches the database. It exists so the
"clearly held out" verdicts in held_out_questions.json can be re-derived.

    python3 evaluation/check_overlap.py

Overlap is judged WITHIN a course: adapters are trained per course, so a CSS 360
training example cannot leak into a CSS 350 answer. Matches in other courses'
corpora are printed as informational only.

Two independent similarity measures are reported per candidate:

  jaccard  - overlap of content words (stopwords dropped), which catches
             reworded near-duplicates like "When does class meet?" vs
             "What time are the course meetings?"
  ratio    - difflib character-level similarity, which catches light edits

Thresholds are deliberately conservative:

  >= 0.60 jaccard or >= 0.75 ratio  ->  REJECT (too similar)
  >= 0.45 jaccard or >= 0.60 ratio  ->  POSSIBLE OVERLAP (resolve by hand)
  otherwise                         ->  clearly held out
"""

from __future__ import annotations

import glob
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STOPWORDS = {
    "a", "am", "an", "and", "any", "are", "as", "at", "be", "by", "can", "course",
    "did", "do", "does", "for", "from", "get", "have", "how", "i", "if", "in", "is",
    "it", "many", "me", "much", "my", "need", "of", "on", "or", "our", "should",
    "so", "the", "there", "this", "to", "we", "what", "when", "where", "which",
    "who", "why", "will", "with", "you", "your",
}

REJECT_JACCARD, REJECT_RATIO = 0.60, 0.75
REVIEW_JACCARD, REVIEW_RATIO = 0.45, 0.60


def tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_corpus() -> list[tuple[str, str, str]]:
    """(courseId, source label, question) for every training-side question on disk."""
    corpus: list[tuple[str, str, str]] = []

    for path in sorted(ROOT.glob("data/exports/*/*.jsonl")):
        course_id = path.parent.name
        label = f"{path.parent.name}/{path.name}"
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            question = (record.get("instruction") or "").strip()
            if question:
                corpus.append((course_id, label, question))

    for path in sorted(glob.glob(str(ROOT / "data/exports/*/generated-snapshot-latest.json"))):
        payload = json.loads(Path(path).read_text())
        course_id = Path(path).parent.name
        label = f"{course_id}/generated-snapshot-latest.json"
        for seed in payload.get("seeds", []):
            question = (seed.get("question") or seed.get("instruction") or "").strip()
            if question:
                corpus.append((course_id, label, question))

    smoke = ROOT / "training" / "heldout_questions.json"
    if smoke.exists():
        payload = json.loads(smoke.read_text())
        entries = payload.get("questions", payload) if isinstance(payload, dict) else payload
        for entry in entries:
            question = entry if isinstance(entry, str) else (entry.get("question") or "")
            if question.strip():
                corpus.append((
                    "css-360-winter-2026-a7rp",
                    "training/heldout_questions.json",
                    question.strip(),
                ))

    return corpus


def nearest(question: str, corpus: list[tuple[str, str, str]]) -> tuple[float, float, str, str]:
    q_tokens = tokens(question)
    best = (0.0, 0.0, "", "")
    for _course_id, label, other in corpus:
        j = jaccard(q_tokens, tokens(other))
        r = SequenceMatcher(None, question.lower(), other.lower()).ratio()
        if max(j, r) > max(best[0], best[1]):
            best = (j, r, label, other)

    return best


def verdict_for(j: float, r: float) -> str:
    if j >= REJECT_JACCARD or r >= REJECT_RATIO:
        return "REJECT (too similar)"
    if j >= REVIEW_JACCARD or r >= REVIEW_RATIO:
        return "POSSIBLE OVERLAP"
    return "clearly held out"


def main() -> int:
    corpus = load_corpus()
    bank = json.loads((ROOT / "evaluation" / "held_out_questions.json").read_text())

    print(f"Training-side questions loaded: {len(corpus)}")
    print()

    flagged = 0
    for course in bank["courses"]:
        course_id = course["courseId"]
        in_course = [row for row in corpus if row[0] == course_id]
        other_courses = [row for row in corpus if row[0] != course_id]
        print(f"=== {course['courseCode']} ({course_id}) ===")
        print(f"    in-course training questions: {len(in_course)}")
        for item in course["questions"]:
            j, r, label, other = nearest(item["question"], in_course)
            verdict = verdict_for(j, r)
            marker = " " if verdict == "clearly held out" else "!"
            if verdict != "clearly held out":
                flagged += 1
            print(f"{marker} {item['id']}  jaccard={j:.2f} ratio={r:.2f}  {verdict}")
            if verdict != "clearly held out":
                print(f"      nearest in-course: [{label}] {other}")
            xj, xr, xlabel, xother = nearest(item["question"], other_courses)
            if verdict_for(xj, xr) != "clearly held out":
                print(f"      (other-course match, informational only: "
                      f"jaccard={xj:.2f} ratio={xr:.2f} [{xlabel}] {xother})")
        print()

    print(f"Flagged for human resolution (in-course): {flagged}")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
