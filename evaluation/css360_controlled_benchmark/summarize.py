#!/usr/bin/env python3
"""Unblind the scores and build the tables.

    python3 summarize.py <run-id>

Reads results/<run-id>/{records.jsonl, blind/key.json, blind/scores_blind.jsonl,
overlap_vm_export.json, grounded_identity.jsonl}, writes results/<run-id>/scores.json
(every scored answer with its condition, outcome, timing and digest) and
results/<run-id>/summary.md plus summary.json. Aggregates only; decides nothing
about a score.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["base", "rag", "ft:v2", "ft_rag:v2", "ft:v3", "ft_rag:v3", "ft:v4_vm", "ft_rag:v4_vm", "ft:v4_tillicum", "ft_rag:v4_tillicum"]
LABEL = {
    "base": "Base (standalone)", "rag": "RAG", "ft:v2": "FT v2", "ft_rag:v2": "FT+RAG v2", "ft:v3": "FT v3",
    "ft_rag:v3": "FT+RAG v3", "ft:v4_vm": "FT v4-VM", "ft_rag:v4_vm": "FT+RAG v4-VM",
    "ft:v4_tillicum": "FT v4-Tillicum", "ft_rag:v4_tillicum": "FT+RAG v4-Tillicum",
}
PAIRS = [
    ("RAG vs FT+RAG v2", "rag", "ft_rag:v2"), ("RAG vs FT+RAG v3", "rag", "ft_rag:v3"),
    ("RAG vs FT+RAG v4-VM", "rag", "ft_rag:v4_vm"), ("RAG vs FT+RAG v4-Tillicum", "rag", "ft_rag:v4_tillicum"),
    ("Base vs FT v2", "base", "ft:v2"), ("Base vs FT v3", "base", "ft:v3"),
    ("Base vs FT v4-VM", "base", "ft:v4_vm"), ("Base vs FT v4-Tillicum", "base", "ft:v4_tillicum"),
    ("FT+RAG v4-VM vs FT+RAG v4-Tillicum", "ft_rag:v4_vm", "ft_rag:v4_tillicum"),
    ("FT v4-VM vs FT v4-Tillicum", "ft:v4_vm", "ft:v4_tillicum"),
    ("FT+RAG v4-VM vs FT+RAG v2", "ft_rag:v4_vm", "ft_rag:v2"), ("FT+RAG v4-Tillicum vs FT+RAG v2", "ft_rag:v4_tillicum", "ft_rag:v2"),
    ("FT+RAG v4-VM vs FT+RAG v3", "ft_rag:v4_vm", "ft_rag:v3"), ("FT v4-VM vs FT v2", "ft:v4_vm", "ft:v2"),
]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else float("nan")


def fmt(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def pct(k, n):
    return "n/a" if not n else f"{k}/{n} ({100 * k / n:.0f}%)"


def sign_test_p(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main(run_id: str) -> int:
    run = HERE / "results" / run_id
    records = [json.loads(l) for l in (run / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    key = json.loads((run / "blind" / "key.json").read_text(encoding="utf-8"))
    blind = [json.loads(l) for l in (run / "blind" / "scores_blind.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    overlap = json.loads((run / "overlap_vm_export.json").read_text(encoding="utf-8"))["sets"]
    identity = {(i["set"], i["questionId"]): i for i in (json.loads(l) for l in (run / "grounded_identity.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    questions = {}
    for set_id, name in (("repeat22", "questions_repeat22.json"), ("heldout", "questions_heldout.json")):
        for q in json.loads((HERE / name).read_text(encoding="utf-8"))["questions"]:
            questions[(set_id, q["id"])] = q
    by_rec = {(r["set"], r["questionId"], r["condition"]): r for r in records}

    rows = []
    for s in blind:
        set_id, qid = s["question"].split("/")
        condition = key[s["question"]][s["letter"]]
        r = by_rec[(set_id, qid, condition)]
        q = questions[(set_id, qid)]
        ov = overlap[set_id][qid]
        rows.append({
            "set": set_id, "questionId": qid, "kind": q["kind"], "condition": condition, "alias": r["alias"],
            "route": r["route"], "letter": s["letter"],
            "correctness": s["correctness"], "groundedness": s["groundedness"], "completeness": s["completeness"],
            "overall": round((s["correctness"] + s["groundedness"] + s["completeness"]) / 3, 3),
            "hallucination": s["hallucination"], "abstained": s["abstained"], "premiseHandling": s["premiseHandling"],
            "citedPassage": s["citedPassage"], "rationale": s["rationale"],
            "outcome": r["outcome"], "doneReason": (r.get("timing") or {}).get("ollama", {}).get("doneReason"),
            "generationSeconds": (r.get("timing") or {}).get("generationSeconds"),
            "evalCount": (r.get("timing") or {}).get("ollama", {}).get("evalCount"),
            "answerWords": len((r.get("answer") or "").split()),
            "servedTag": r["servedTag"], "servedDigest": r["servedDigest"],
            "groundedComparisonValid": identity[(set_id, qid)]["valid"] if r["route"] == "pair" else None,
            "v4TrainingQuestionVerdict": ov["questionVerdict"], "v4TrainingFactCoverage": ov["keyFactCoverage"],
            "julyQuestionVerdict": q["overlapJulySplit"]["questionVerdict"], "julyFactCoverage": q["overlapJulySplit"]["keyFactCoverage"],
            "answer": r["answer"],
        })
    rows.sort(key=lambda r: (r["set"], r["questionId"], ORDER.index(r["condition"])))
    (run / "scores.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    assert len(rows) == len(records) == 460, (len(rows), len(records))

    md = [f"# Tables for {run_id}", "", f"{len(records)} attempted conditions, {sum(1 for r in records if r['outcome'] == 'scorable')} scorable, "
          f"{sum(1 for r in rows)} scored. Grounded comparisons hash-identical: {sum(1 for i in identity.values() if i['valid'])}/{len(identity)}.", ""]
    summary = {"run": run_id, "attempted": len(records)}

    # ---- T1 per condition, per set ------------------------------------------------
    def cond_rows(subset, label):
        out = []
        for c in ORDER:
            rs = [r for r in subset if r["condition"] == c]
            if not rs:
                continue
            out.append([LABEL[c], len(rs), fmt(mean(r["correctness"] for r in rs)), fmt(mean(r["groundedness"] for r in rs)),
                        fmt(mean(r["completeness"] for r in rs)), fmt(mean(r["overall"] for r in rs)),
                        pct(sum(r["hallucination"] for r in rs), len(rs)), pct(sum(r["doneReason"] == "length" for r in rs), len(rs)),
                        fmt(mean(r["generationSeconds"] for r in rs), 1), fmt(mean(r["answerWords"] for r in rs), 0)])
        return out
    header = ["Condition", "n", "Correct", "Grounded", "Complete", "Overall", "Hallucinated", "Truncated", "Gen s", "Words"]
    for set_id, title in (("repeat22", "Repeat of the 22 questions of 2026-09-11 (controlled settings)"), ("heldout", "Held-out set (24 new questions)")):
        md += [f"## {title}", "", table(header, cond_rows([r for r in rows if r["set"] == set_id], set_id)), ""]
    md += ["## Both sets combined (46 questions)", "", table(header, cond_rows(rows, "all")), ""]
    summary["perCondition"] = {c: {s: {"n": len([r for r in rows if r["condition"] == c and r["set"] == s]),
                                      "correctness": mean(r["correctness"] for r in rows if r["condition"] == c and r["set"] == s),
                                      "overall": mean(r["overall"] for r in rows if r["condition"] == c and r["set"] == s),
                                      "hallucinated": sum(r["hallucination"] for r in rows if r["condition"] == c and r["set"] == s)}
                                  for s in ("repeat22", "heldout")} for c in ORDER}

    # ---- T2 by kind -----------------------------------------------------------------
    for set_id in ("repeat22", "heldout"):
        md += [f"## By question kind: {set_id}", ""]
        sub = [r for r in rows if r["set"] == set_id]
        kinds = [k for k in ("factual", "unanswerable", "false_premise") if any(r["kind"] == k for r in sub)]
        hdr = ["Condition"]
        for k in kinds:
            n = len({r["questionId"] for r in sub if r["kind"] == k})
            hdr += [f"{k} correct (n={n})", f"{k} halluc."]
            if k == "unanswerable":
                hdr.append("abstained")
            if k == "false_premise":
                hdr.append("corrected / accepted / evaded")
        trs = []
        for c in ORDER:
            row = [LABEL[c]]
            for k in kinds:
                rs = [r for r in sub if r["condition"] == c and r["kind"] == k]
                row += [fmt(mean(r["correctness"] for r in rs)), pct(sum(r["hallucination"] for r in rs), len(rs))]
                if k == "unanswerable":
                    row.append(pct(sum(r["abstained"] is True for r in rs), len(rs)))
                if k == "false_premise":
                    ph = Counter(r["premiseHandling"] for r in rs)
                    row.append(f"{ph['corrected']} / {ph['accepted']} / {ph['evaded']}")
            trs.append(row)
        md += [table(hdr, trs), ""]

    # ---- T3 paired comparisons ----------------------------------------------------------
    md += ["## Paired comparisons (per-question overall, wins-ties-losses for the first named)", "",
           "Grounded pairs use only questions whose pair-route groups were hash-identical (all 46 here). "
           "A sign-test p-value is shown only when at least ten non-tied pairs exist.", ""]
    paired = {}
    for scope, subset in (("repeat22", [r for r in rows if r["set"] == "repeat22"]), ("heldout", [r for r in rows if r["set"] == "heldout"]), ("both", rows)):
        trs = []
        for label, a, b in PAIRS:
            by_q = defaultdict(dict)
            for r in subset:
                if r["condition"] in (a, b) and (r["route"] == "standalone" or r["groundedComparisonValid"]):
                    by_q[(r["set"], r["questionId"])][r["condition"]] = r
            pairs = [(v[a], v[b]) for v in by_q.values() if a in v and b in v]
            wins = sum(1 for x, y in pairs if x["overall"] > y["overall"])
            losses = sum(1 for x, y in pairs if x["overall"] < y["overall"])
            ties = len(pairs) - wins - losses
            diff = mean(x["overall"] - y["overall"] for x, y in pairs)
            cdiff = mean(x["correctness"] - y["correctness"] for x, y in pairs)
            p = sign_test_p(wins, losses) if wins + losses >= 10 else None
            trs.append([label, len(pairs), f"{wins}-{ties}-{losses}", fmt(diff), fmt(cdiff), fmt(p, 3) if p is not None else "n/a"])
            paired[f"{scope}:{label}"] = {"n": len(pairs), "wins": wins, "ties": ties, "losses": losses, "meanOverallDiff": diff, "meanCorrectnessDiff": cdiff, "signTestP": p}
        md += [f"### {scope}", "", table(["Comparison", "pairs", "W-T-L", "mean overall diff", "mean correctness diff", "sign-test p"], trs), ""]
    summary["paired"] = paired

    # ---- T4 per-question grid --------------------------------------------------------------
    md += ["## Per-question correctness (H = hallucination flagged; kind: F factual, U unanswerable, P false premise)", ""]
    hdr = ["Q", "kind", "v4-train facts"] + [LABEL[c] for c in ORDER]
    trs = []
    for (set_id, qid), q in sorted(questions.items(), key=lambda kv: (kv[0][0] != "repeat22", kv[0][1])):
        cell = {r["condition"]: f"{r['correctness']}{'H' if r['hallucination'] else ''}" for r in rows if r["set"] == set_id and r["questionId"] == qid}
        ov = overlap[set_id][qid]
        flag = ov["keyFactCoverage"] + ("*" if ov["questionVerdict"] != "clearly held out" else "")
        trs.append([f"{set_id[:1]}:{qid}", q["kind"][:1].upper(), flag] + [cell.get(c, "-") for c in ORDER])
    md += [table(hdr, trs), "", "`v4-train facts`: key facts found in the v4 training responses on the VM (fraction), `*` = the question itself is REVIEW or REJECT against a v4 training question.", ""]

    # ---- T5 overlap-split means ------------------------------------------------------------
    md += ["## Training-overlap split (mean correctness on factual questions)", "",
           "`seen`: at least one key fact appears in a v4 training response, or the question is flagged against the v4 export; `unseen`: neither. "
           "For v2 and v3 the July split is the only candidate for their data and is reported alongside.", ""]
    hdr = ["Condition", "v4-seen n", "v4-seen correct", "v4-unseen n", "v4-unseen correct", "July-seen n", "July-seen correct", "July-unseen n", "July-unseen correct"]
    trs = []
    for c in ORDER:
        rs = [r for r in rows if r["condition"] == c and r["kind"] == "factual"]
        v4_seen = [r for r in rs if r["v4TrainingFactCoverage"].split("/")[0] != "0" or r["v4TrainingQuestionVerdict"] != "clearly held out"]
        v4_unseen = [r for r in rs if r not in v4_seen]
        j_seen = [r for r in rs if r["julyFactCoverage"].split("/")[0] != "0" or r["julyQuestionVerdict"] != "clearly held out"]
        j_unseen = [r for r in rs if r not in j_seen]
        trs.append([LABEL[c], len(v4_seen), fmt(mean(r["correctness"] for r in v4_seen)), len(v4_unseen), fmt(mean(r["correctness"] for r in v4_unseen)),
                    len(j_seen), fmt(mean(r["correctness"] for r in j_seen)), len(j_unseen), fmt(mean(r["correctness"] for r in j_unseen))])
    md += [table(hdr, trs), ""]

    # ---- T6 timing and outcomes ------------------------------------------------------------
    md += ["## Timing, length and outcomes by condition (both sets)", ""]
    hdr = ["Condition", "attempted", "scorable", "gen s mean", "gen s median", "gen s max", "load s mean", "eval tokens mean", "truncated (length)", "words mean"]
    trs = []
    for c in ORDER:
        rs = [r for r in records if r["condition"] == c]
        gens = [r["timing"]["generationSeconds"] for r in rs]
        loads = [(r["timing"].get("ollama") or {}).get("loadDurationNs", 0) / 1e9 for r in rs]
        evals = [(r["timing"].get("ollama") or {}).get("evalCount") for r in rs]
        trs.append([LABEL[c], len(rs), sum(r["outcome"] == "scorable" for r in rs), fmt(mean(gens), 1), fmt(statistics.median(gens), 1), fmt(max(gens), 1),
                    fmt(mean(loads), 1), fmt(mean(evals), 0), sum((r["timing"].get("ollama") or {}).get("doneReason") == "length" for r in rs),
                    fmt(mean(len((r["answer"] or "").split()) for r in rs), 0)])
    md += [table(hdr, trs), ""]
    outcomes = Counter((r["condition"], r["outcome"]) for r in records)
    summary["outcomes"] = {f"{c}:{o}": n for (c, o), n in outcomes.items()}

    (run / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (run / "summary.json").write_text(json.dumps(summary, indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
