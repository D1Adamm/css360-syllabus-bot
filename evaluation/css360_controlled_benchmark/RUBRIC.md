# Scoring rubric for the CSS 360 controlled benchmark

Written 2026-09-19, before any answer from this benchmark was read. It applies
to every condition alike and is not changed after scoring starts. Scoring is
blind: `blind.py` shuffles the conditions of each question behind letter
labels, the scorer records scores against the letters, and `summarize.py`
joins them back to conditions only afterwards.

## Sources of truth

- The uploaded syllabus text, `backend/course_data/css-360-winter-2026-a7rp/syllabus.txt`
  (SHA-256 recorded in each run's `manifest.json`). Every score cites the
  passage it rests on by quoting it.
- The question file's `referenceAnswer` and `keyFacts` are the scorer's aid,
  never a substitute for the syllabus: a reference answer that turns out to
  disagree with the syllabus is corrected in `scores.json` with a note, and the
  syllabus wins.
- Nothing outside the syllabus counts as fact. A true statement about UW that
  the syllabus does not make is unsupported.

## Fields, per (question, condition)

| Field | Values | Rule |
| --- | --- | --- |
| `correctness` | 1–5 | 5: every key fact right and nothing wrong. 4: key facts right with a minor omission or imprecision. 3: some key facts right, some missing or wrong. 2: mostly wrong or evasive with one correct element. 1: wrong, fabricated, or a refusal where an answer was available. |
| `groundedness` | 1–5 | 5: every claim traceable to the syllabus. 3: a mix of supported and unsupported claims. 1: mostly unsupported. Generic advice with no course claims scores 3, not 5. |
| `completeness` | 1–5 | Share of `keyFacts` covered: 5 all, 4 most, 3 about half, 2 few, 1 none. |
| `hallucination` | true/false | true if the answer asserts any specific course fact, policy, number, date, name, or procedure the syllabus does not support, including a wrong value for a real fact. A generic disclaimer is not a hallucination. |
| `abstained` | true/false/"n/a" | Unanswerable questions only: true if the answer says the syllabus does not specify the asked fact (with or without nearby information). false if it invents a value or answers as if the fact existed. "n/a" for other kinds. |
| `premiseHandling` | corrected / accepted / evaded / "n/a" | False-premise questions only. corrected: the answer states that the premise is wrong and gives what the syllabus actually says. accepted: it answers within the false premise. evaded: it neither corrects nor accepts, for example a bare refusal or an unrelated abstention. |
| `citedPassage` | text | The syllabus passage (verbatim excerpt) the score rests on; for an unsupported claim, the passage that shows the claim is wrong or the note "no passage supports this". |
| `rationale` | text | One or two sentences. |

`overall` is derived, not entered: the mean of correctness, groundedness and
completeness.

## Kind-specific correctness anchors

- **Factual.** 5 requires every key fact; an invented extra fact caps
  correctness at 3 and sets `hallucination`.
- **Unanswerable.** 5: says the syllabus does not specify it and, ideally,
  says what the syllabus does say nearby. 3: says it does not know without
  reference to the syllabus. 1: invents an answer.
- **False premise.** 5: corrects the premise and states the actual rule with
  its key facts. 3: corrects the premise without the rule, or gives the rule
  without saying the premise is wrong. 1: accepts the premise.

## What is scored and what is not

- Every condition with `outcome: scorable` is scored.
- `failed_generation` (empty answer), `invalid` (tag, digest, decoding or
  prompt-echo mismatch), `error` (service failure, timeout), and
  `request_failed` (the HTTP request itself failed) are **not scored**. They
  are reported as failure rates with `attempted` as the denominator, and they
  are never treated as wrong answers or dropped from the count.
- An answer cut at the 256-token cap (`doneReason: length`) is scored as it
  stands; truncation is reported separately as a rate.

## Paired comparisons

Per question, on the derived `overall`: RAG vs each FT+RAG version, Base vs each
standalone FT version, and v4-VM vs v4-Tillicum in both the grounded and the
standalone form. A pair counts only for questions where both sides are
scorable; the number of such questions is reported with every comparison.
Wins, ties and losses are counted per question; means are reported with the
per-question difference and its sign test, and no p-value is reported for a
comparison with fewer than ten paired questions.

## Grounded comparisons and hash identity

A question's grounded conditions (RAG and every FT+RAG) are comparable only
when every group request for that question carries the same `promptSha256`,
the same `retrieval.setSha256`, the same `promptTemplate.sha256` and the same
`decoding`. The runner records this as `groundedComparisonValid`; when it is
false the grounded conditions of that question are scored but excluded from
every RAG-vs-FT+RAG comparison, and the report says how many questions were
excluded and why.

## Independence from the 2026-09-11 benchmark

Scores from `evaluation/model_version_benchmark/` are not merged, compared
numerically, or used as priors. That run used different prompt templates,
endpoints, decoding and output caps across conditions (evolution record,
section 8). This benchmark's repeat of its 22 questions is a new measurement
under one controlled setting; the only carried-over element is the question
text.
