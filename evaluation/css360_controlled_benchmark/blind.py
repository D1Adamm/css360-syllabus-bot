#!/usr/bin/env python3
"""Blind scoring sheet: the conditions of each question behind shuffled letters.

    python3 blind.py build <run-id>            # writes results/<run-id>/blind/{sheet,key}.json
    python3 blind.py show <run-id> q01 q02 …   # prints questions for scoring, letters only
    python3 blind.py show <run-id> heldout:h01

The key (letter -> condition per question) is written once and read only by
summarize.py after scoring. `show` never prints a condition name, tag, alias,
route or digest.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LETTERS = "ABCDEFGHIJ"
CAP = 1100


def load_questions():
    out = {}
    for set_id, name in (("repeat22", "questions_repeat22.json"), ("heldout", "questions_heldout.json")):
        for q in json.loads((HERE / name).read_text(encoding="utf-8"))["questions"]:
            out[(set_id, q["id"])] = q
    return out


def build(run_id: str) -> None:
    run = HERE / "results" / run_id
    records = [json.loads(l) for l in (run / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    questions = load_questions()
    by_q: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        by_q.setdefault((r["set"], r["questionId"]), []).append(r)
    sheet, key = {}, {}
    for (set_id, qid), rows in sorted(by_q.items()):
        rng = random.Random(f"{run_id}:{set_id}:{qid}")
        rows = sorted(rows, key=lambda r: r["condition"])
        rng.shuffle(rows)
        q = questions[(set_id, qid)]
        entry = {"set": set_id, "id": qid, "kind": q["kind"], "question": q["question"],
                 "referenceAnswer": q["referenceAnswer"], "keyFacts": q["keyFacts"],
                 "premise": q.get("premise"), "sourceSection": q.get("sourceSection"),
                 "supportingPassages": q.get("supportingPassages"), "answers": {}}
        key[f"{set_id}/{qid}"] = {}
        for letter, r in zip(LETTERS, rows):
            entry["answers"][letter] = {
                "answer": r["answer"], "outcome": r["outcome"],
                "doneReason": (r.get("timing") or {}).get("ollama", {}).get("doneReason"),
            }
            key[f"{set_id}/{qid}"][letter] = r["condition"]
        sheet[f"{set_id}/{qid}"] = entry
    out = run / "blind"
    out.mkdir(exist_ok=True)
    (out / "sheet.json").write_text(json.dumps(sheet, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "key.json").write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8")
    print(f"sheet: {len(sheet)} questions, {sum(len(e['answers']) for e in sheet.values())} answers; key written separately")


def show(run_id: str, ids: list[str]) -> None:
    sheet = json.loads((HERE / "results" / run_id / "blind" / "sheet.json").read_text(encoding="utf-8"))
    for ref in ids:
        set_id, _, qid = ref.partition(":") if ":" in ref else ("repeat22", "", ref)
        e = sheet[f"{set_id}/{qid}"]
        print("=" * 90)
        print(f"{set_id}/{qid} [{e['kind']}] {e['question']}")
        print(f"REF: {e['referenceAnswer']}")
        print(f"KEY: {' | '.join(e['keyFacts'])}")
        if e.get("premise"):
            print(f"PREMISE: {e['premise']}")
        if e.get("supportingPassages"):
            for p in e["supportingPassages"]:
                print(f"PASSAGE: {p}")
        for letter, a in e["answers"].items():
            text = " ".join((a["answer"] or "").split())
            cut = "" if len(text) <= CAP else f" …[cut {len(text) - CAP} chars]"
            flag = " [LENGTH]" if a["doneReason"] == "length" else ""
            print(f"--- {letter}{flag}: {text[:CAP]}{cut}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    if sys.argv[1] == "build":
        build(sys.argv[2])
    elif sys.argv[1] == "show":
        show(sys.argv[2], sys.argv[3:])
    else:
        print(__doc__); sys.exit(2)
