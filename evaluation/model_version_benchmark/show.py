#!/usr/bin/env python3
"""Print every configuration's answer to one question side by side with the
scorer's reference, for hand scoring. Usage: show.py q01 [q02 ...] | show.py all"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ORDER = ["rag", "ft_v2", "ftrag_v2", "ft_v3", "ftrag_v3"]
results = json.loads((HERE / "results.json").read_text())
questions = {q["id"]: q for q in json.loads((HERE / "questions.json").read_text())["questions"]}
ids = sorted(questions) if sys.argv[1:] == ["all"] else sys.argv[1:]
for qid in ids:
    q = questions[qid]
    print("=" * 100)
    print(f"{qid} [{q['category']}] answerable={q['answerableFromSyllabus']} coverage={q['trainingCoverage']}")
    print("Q:", q["question"])
    print("REF:", q["referenceAnswer"])
    print("KEY:", "; ".join(q["keyFacts"]))
    for c in ORDER:
        r = next((r for r in results if r["config"] == c and r["questionId"] == qid), None)
        print("-" * 100)
        if r is None:
            print(f"[{c}] (no result)"); continue
        if r["status"] != 200:
            print(f"[{c}] HTTP {r['status']}: {r.get('error','')[:300]}"); continue
        resp = r["response"]
        secs = [s["sectionTitle"] for s in resp.get("sources", [])]
        print(f"[{c}] version={resp.get('modelVersion')} model={resp.get('model')} gen={resp.get('generationSeconds')} wall={r['wallSeconds']}s")
        if secs: print("   retrieved:", secs)
        print("   " + resp.get("answer", "").replace("\n", "\n   "))
