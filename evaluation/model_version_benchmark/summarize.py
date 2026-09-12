#!/usr/bin/env python3
"""Aggregate scores.json + results.json into the benchmark tables.

    python3 evaluation/model_version_benchmark/summarize.py            # print markdown
    python3 evaluation/model_version_benchmark/summarize.py --template # write scores.template.json from results.json

scores.json holds one record per (config, questionId) with the rubric fields
from README.md. This script only aggregates; it decides nothing about a score.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.json"
SCORES = HERE / "scores.json"
QUESTIONS = HERE / "questions.json"

CONFIG_ORDER = ["rag", "ft_v2", "ftrag_v2", "ft_v3", "ftrag_v3"]
LABELS = {
    "rag": "RAG",
    "ft_v2": "Fine-Tuned v2",
    "ftrag_v2": "Fine-Tuned + RAG v2",
    "ft_v3": "Fine-Tuned v3",
    "ftrag_v3": "Fine-Tuned + RAG v3",
}
SCALES = ("correctness", "groundedness", "relevance", "completeness")


def mean(values) -> float:
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


def load():
    results = json.loads(RESULTS.read_text())
    questions = {q["id"]: q for q in json.loads(QUESTIONS.read_text())["questions"]}
    scores = json.loads(SCORES.read_text()) if SCORES.is_file() else []
    return results, questions, scores


def write_template(results, questions) -> None:
    template = []
    for r in results:
        if r.get("status") != 200:
            continue
        q = questions[r["questionId"]]
        template.append({
            "config": r["config"],
            "questionId": r["questionId"],
            "answerableFromSyllabus": q["answerableFromSyllabus"],
            "correctness": None,
            "groundedness": None,
            "relevance": None,
            "completeness": None,
            "hallucination": None,
            "abstained": None if not q["answerableFromSyllabus"] else "n/a",
            "rationale": "",
        })
    (HERE / "scores.template.json").write_text(json.dumps(template, indent=2) + "\n")
    print(f"wrote {len(template)} score slots to scores.template.json")


def table(rows, header) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", action="store_true")
    args = parser.parse_args()
    results, questions, scores = load()
    if args.template:
        write_template(results, questions)
        return 0

    by_config = defaultdict(list)
    for s in scores:
        by_config[s["config"]].append(s)
    timing = defaultdict(list)
    for r in results:
        if r.get("status") == 200:
            timing[r["config"]].append(r["response"].get("generationSeconds") or r["wallSeconds"])

    # ---- main table -------------------------------------------------------
    rows = []
    for c in CONFIG_ORDER:
        ss = by_config.get(c, [])
        if not ss:
            continue
        ans = [s for s in ss if s["answerableFromSyllabus"]]
        unans = [s for s in ss if not s["answerableFromSyllabus"]]
        halluc = sum(1 for s in ss if s["hallucination"])
        abst = sum(1 for s in unans if s.get("abstained") is True)
        overall = mean(mean(s[k] for k in SCALES) for s in ss)
        rows.append([
            LABELS[c], len(ss),
            f"{mean(s['correctness'] for s in ss):.2f}",
            f"{mean(s['groundedness'] for s in ss):.2f}",
            f"{mean(s['relevance'] for s in ss):.2f}",
            f"{mean(s['completeness'] for s in ss):.2f}",
            f"{overall:.2f}",
            f"{halluc}/{len(ss)} ({100 * halluc / len(ss):.0f}%)",
            f"{abst}/{len(unans)}" if unans else "-",
            f"{mean(s['correctness'] for s in ans):.2f}" if ans else "-",
            f"{mean(s['correctness'] for s in unans):.2f}" if unans else "-",
            f"{mean(timing[c]):.1f}s" if timing[c] else "-",
        ])
    print("## Benchmark table (1-5 scales; hallucination = any unsupported specific claim)\n")
    print(table(rows, ["Configuration", "n", "Correct", "Grounded", "Relevant", "Complete", "Overall", "Hallucinated", "Abstained (unanswerable)", "Correct (answerable)", "Correct (unanswerable)", "Mean gen time"]))

    # ---- failures per config ---------------------------------------------
    print("\n## Failures per configuration (correctness <= 2, or a hallucination)\n")
    for c in CONFIG_ORDER:
        failed = sorted(
            (s["questionId"], s["correctness"], "H" if s["hallucination"] else "")
            for s in by_config.get(c, []) if s["correctness"] <= 2 or s["hallucination"]
        )
        print(f"- **{LABELS[c]}** ({len(failed)}): " + (", ".join(f"{q} (c={cs}{h})" for q, cs, h in failed) or "none"))

    # ---- by category ------------------------------------------------------
    print("\n## Mean correctness by category\n")
    cats = sorted({questions[s["questionId"]]["category"] for s in scores})
    rows = []
    for cat in cats:
        row = [cat, sum(1 for q in questions.values() if q["category"] == cat)]
        for c in CONFIG_ORDER:
            vals = [s["correctness"] for s in by_config.get(c, []) if questions[s["questionId"]]["category"] == cat]
            row.append(f"{mean(vals):.2f}" if vals else "-")
        rows.append(row)
    print(table(rows, ["Category", "n"] + [LABELS[c] for c in CONFIG_ORDER]))

    # ---- by training coverage --------------------------------------------
    print("\n## Mean correctness by training coverage of the asked fact (answerable questions)\n")
    rows = []
    for cov in ("train", "validation-only", "partial", "none"):
        ids = [qid for qid, q in questions.items() if q["trainingCoverage"] == cov and q["answerableFromSyllabus"]]
        if not ids:
            continue
        row = [cov, len(ids)]
        for c in CONFIG_ORDER:
            vals = [s["correctness"] for s in by_config.get(c, []) if s["questionId"] in ids]
            row.append(f"{mean(vals):.2f}" if vals else "-")
        rows.append(row)
    print(table(rows, ["Fact seen in training?", "n"] + [LABELS[c] for c in CONFIG_ORDER]))

    # ---- per-question correctness grid -----------------------------------
    print("\n## Correctness per question (H = hallucination flagged)\n")
    rows = []
    for qid in sorted(questions):
        row = [qid, questions[qid]["category"], "yes" if questions[qid]["answerableFromSyllabus"] else "NO"]
        for c in CONFIG_ORDER:
            s = next((s for s in by_config.get(c, []) if s["questionId"] == qid), None)
            row.append("-" if s is None else f"{s['correctness']}{'H' if s['hallucination'] else ''}")
        rows.append(row)
    print(table(rows, ["Q", "category", "answerable"] + [LABELS[c] for c in CONFIG_ORDER]))

    # ---- head-to-head -----------------------------------------------------
    def h2h(a, b):
        wins = losses = ties = 0
        for qid in questions:
            sa = next((s for s in by_config.get(a, []) if s["questionId"] == qid), None)
            sb = next((s for s in by_config.get(b, []) if s["questionId"] == qid), None)
            if sa is None or sb is None:
                continue
            ma, mb = mean(sa[k] for k in SCALES), mean(sb[k] for k in SCALES)
            if abs(ma - mb) < 1e-9:
                ties += 1
            elif ma > mb:
                wins += 1
            else:
                losses += 1
        return wins, ties, losses

    print("\n## Head-to-head (per-question overall score; wins-ties-losses for the first configuration)\n")
    for a, b in (("ft_v3", "ft_v2"), ("ftrag_v3", "ftrag_v2"), ("ftrag_v2", "rag"), ("ftrag_v3", "rag"), ("ft_v2", "rag"), ("ft_v3", "rag")):
        if by_config.get(a) and by_config.get(b):
            w, t, l = h2h(a, b)
            print(f"- {LABELS[a]} vs {LABELS[b]}: {w}-{t}-{l}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
