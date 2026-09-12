# CSS 360 model-version benchmark: RAG vs Fine-Tuned v2/v3 vs Fine-Tuned + RAG v2/v3

Run 2026-09-11 (UTC 06:12–06:34) against the deployed backend at
`https://aiswe.uwb.edu` through `POST /api/model-testing/generate`, with a normal
administrator session. 22 questions × 5 configurations = 110 answers, 0 request
failures, and every fine-tuned answer echoed the version it was asked for.
No registry, deployment, publication, `current_version` or evaluation state was
touched.

Registry state during the run: v2 `ready`/`online` and current; v3 `ready`/`offline`.
The local fine-tuned service mapped v2 → `css360-ft-v2:latest` and
v3 → `css360-cpu-v3-test:latest`, both served on the VM's Ollama with greedy decoding.

Scoring: one judge (Claude, in this session), every answer read against the
syllabus text with the rubric in README.md. Scores and rationales are in
`scores.json`; raw answers in `results.json`.

## Benchmark table

| Configuration | n | Correct | Grounded | Relevant | Complete | Overall | Hallucinated | Abstained (4 unanswerable) | Correct (18 answerable) | Correct (4 unanswerable) | Mean time* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAG | 22 | 3.64 | 3.86 | 4.45 | 3.27 | 3.81 | 6/22 (27%) | 3/4 | 3.67 | 3.50 | 17.3 s |
| Fine-Tuned v2 | 22 | 1.18 | 1.09 | 3.86 | 1.27 | 1.85 | 19/22 (86%) | 0/4 | 1.22 | 1.00 | 7.6 s |
| Fine-Tuned + RAG v2 | 22 | 3.64 | 3.82 | 4.73 | 3.18 | 3.84 | 7/22 (32%) | 2/4 | 3.78 | 3.00 | 11.6 s |
| Fine-Tuned v3 | 22 | 1.18 | 1.32 | 3.55 | 1.14 | 1.80 | 18/22 (82%) | 0/4 | 1.17 | 1.25 | 7.1 s |
| Fine-Tuned + RAG v3 | 22 | 3.91 | 4.23 | 4.73 | 3.36 | 4.06 | 5/22 (23%) | 3/4 | 3.89 | 4.00 | 11.4 s |

\* Fine-tuned times are the service's generation time; the RAG route reports no
generation time, so its figure is wall time including embedding and retrieval.

Head-to-head on per-question overall score (wins-ties-losses for the first named):

| Comparison | W-T-L |
| --- | --- |
| Fine-Tuned + RAG v3 vs Fine-Tuned + RAG v2 | 4-18-0 |
| Fine-Tuned v3 vs Fine-Tuned v2 | 2-17-3 |
| Fine-Tuned + RAG v2 vs RAG | 9-5-8 |
| Fine-Tuned + RAG v3 vs RAG | 11-5-6 |
| Fine-Tuned v2 vs RAG | 2-1-19 |
| Fine-Tuned v3 vs RAG | 2-1-19 |

Mean correctness by whether the asked fact appears in the local training split (answerable questions):

| Fact seen in training? | n | RAG | FT v2 | FT+RAG v2 | FT v3 | FT+RAG v3 |
| --- | --- | --- | --- | --- | --- | --- |
| train | 6 | 4.33 | 1.17 | 4.00 | 1.00 | 4.17 |
| validation-only | 1 | 5.00 | 1.00 | 5.00 | 1.00 | 5.00 |
| partial (neighbouring fact) | 3 | 2.33 | 1.67 | 4.33 | 1.67 | 4.67 |
| none | 8 | 3.50 | 1.12 | 3.25 | 1.12 | 3.25 |

Per-question correctness (H = hallucination flagged):

| Q | category | answerable | RAG | FT v2 | FT+RAG v2 | FT v3 | FT+RAG v3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q01 | instructor_contact | yes | 5 | 1 | 5 | 1H | 5 |
| q02 | meeting_times_location | yes | 4 | 2H | 5 | 1 | 5 |
| q03 | meeting_times_location | yes | 2H | 1H | 2H | 1H | 2 |
| q04 | grading | yes | 5 | 1H | 5 | 1H | 5 |
| q05 | grading | yes | 5 | 1H | 5 | 1H | 5 |
| q06 | late_work | yes | 3 | 1H | 1H | 1H | 1H |
| q07 | late_work | yes | 5 | 1H | 5 | 1 | 5 |
| q08 | attendance | yes | 2 | 2 | 2H | 1H | 2H |
| q09 | attendance | yes | 5 | 1H | 5 | 1H | 5 |
| q10 | assignments_projects | yes | 5 | 1H | 5 | 1H | 5 |
| q11 | assignments_projects | yes | 5 | 1H | 3 | 2 | 3 |
| q12 | assignments_projects | yes | 3 | 1H | 4 | 1H | 4 |
| q13 | exams | yes | 3H | 1H | 3H | 1H | 3H |
| q14 | exams (false premise) | yes | 1H | 1H | 1H | 1H | 1H |
| q15 | course_policies | yes | 2H | 1H | 5 | 1H | 5 |
| q16 | course_policies | yes | 2 | 3 | 4 | 3 | 5 |
| q17 | course_policies | yes | 5 | 1H | 5 | 1H | 5 |
| q18 | office_hours | yes | 4H | 1H | 3 | 1H | 4 |
| q19 | unanswerable | NO | 1H | 1H | 1H | 1H | 1H |
| q20 | unanswerable | NO | 5 | 1H | 5 | 1H | 5 |
| q21 | unanswerable | NO | 5 | 1H | 5 | 2H | 5 |
| q22 | unanswerable | NO | 3 | 1H | 1H | 1H | 5 |

## Failures by configuration (correctness ≤ 2 or a hallucination)

- **RAG** (8): q03, q08, q13, q14, q15, q16, q18, q19.
- **Fine-Tuned v2** (21): everything except q16.
- **Fine-Tuned + RAG v2** (7): q03, q06, q08, q13, q14, q19, q22.
- **Fine-Tuned v3** (21): everything except q16.
- **Fine-Tuned + RAG v3** (6): q03, q06, q08, q13, q14, q19.

Failures shared by every configuration: q06 (extension on Bot Feedback; the
explicit "no extension for Demo and Feedback" sentence was in the retrieved
context and still ignored), q13/q14 (both exam questions; every configuration
accepted the premise that exams exist), q19 (every configuration invented a
per-quiz point value), q08 (the drop rule lives in "Your Presence in Class",
which retrieval never surfaced), q03 (the Zoom notes are split from their date
headings by the chunker).

## Main findings

1. **Plain fine-tuning does not work as a knowledge source.** Fine-Tuned v2 and
   v3 score 1.18 on correctness and hallucinate on 86% and 82% of questions.
   Decisive detail: on the six questions whose facts are in the training set
   (instructor contact, meeting time and room, the extension policy twice, the
   no-makeup rule, office-hours booking) they score 1.17 and 1.00. The models
   answered with placeholders ("Dr. [Instructor's Name]"), fabricated values
   ("2:00 PM to 4:00 PM, room 204", "30%", "2 extensions of 7 days"), generic
   advice, or refusals ("I don't have any information about a specific class").
   That is base-model behaviour, not recall of the training examples.

2. **v2 and v3 are nearly the same model in effect.** With identical prompts and
   greedy decoding, Fine-Tuned + RAG v2 and v3 produced byte-identical answers on
   17 of 22 questions, and plain Fine-Tuned v2 and v3 on 6 (plus 4 near-identical);
   where the plain answers differ they are different base-model-like generic
   texts. Retrieval was identical for all RAG-type configurations on all 22
   questions, so every v2/v3 difference is the adapter.

3. **v3 is not worse than v2, and slightly better when grounded.** Fine-Tuned +
   RAG v3 beat v2 on 4 questions and lost none: it abstained instead of inventing
   a Zoom date (q03) and instead of naming the instructor as the TA (q22), gave
   the correct "no" on the API-token question without the misleading citation
   (q16), and answered the office-hours question more directly (q18). Overall
   4.06 vs 3.84, hallucinations 5 vs 7, abstentions on unanswerable questions
   3/4 vs 2/4. In the plain configuration the two are indistinguishable
   (2-17-3). The v3 gain is four questions in a single greedy run judged by one
   scorer; treat it as "at least as good", not as a measured improvement.

4. **Fine-Tuned + RAG versus RAG is a wash on facts.** Correctness 3.64 (RAG) vs
   3.64 (v2) vs 3.91 (v3); head-to-head 9-5-8 and 11-5-6. Both fail the same
   questions for the same reasons. Fine-Tuned + RAG is more concise (25–29 words
   vs 63) and cleaner in format; RAG is more complete on some multi-part
   questions (q11, q12). The two paths also use different prompt templates, so
   this comparison conflates the adapter with the prompt.

5. **Retrieval and premise-acceptance are the real bottlenecks.** Every
   configuration failed the same retrieval misses (q03, q08) and the same
   premise traps (q13, q14, q19). Those are properties of the chunker, the
   retriever and the 3B base model, not of any adapter.

## Research question

**Does fine-tuning provide meaningful improvement beyond RAG for syllabus
question answering, or is RAG doing essentially all of the useful work?**

RAG is doing essentially all of the useful work. Removing retrieval from the
fine-tuned models drops correctness from about 3.6–3.9 to about 1.2 and raises
hallucination from about 25% to about 85%; adding the adapter to retrieval
moves correctness by 0.0 (v2) to +0.27 (v3), which is within what this study can
resolve. The fine-tuned models show no measurable recall of the syllabus facts
they were trained on. The one place an adapter showed a plausible effect is
behavioural: v3 abstains a little more readily and is more concise when the
context does not support an answer.

## Caveats

- One judge, one run, greedy decoding: the model outputs are deterministic but
  the scores are one person's reading of the rubric. Human re-scoring of the
  110 answers in `results.json` is the natural next step before anything is
  published.
- 22 questions; the v3-over-v2 edge rests on 4 of them.
- Training coverage was judged against the July 2026 local export (54 approved
  examples); the datasets that actually trained v2 and v3 on the VM may differ.
- RAG and Fine-Tuned + RAG use different prompt templates in production, so
  their comparison measures the two production pipelines, not the adapter alone.
- Whether the Ollama models apply their adapters at all was not verifiable from
  outside the VM; the results are consistent with adapters that are inert or
  very weak. See the recommendation.

## Recommendation for v3

Do not promote or publish v3 on the strength of this run; keep v2 current and
published (no change). v3 is at worst equal and at best marginally better, but
neither version is doing what fine-tuning was meant to do, so the version
question is secondary to a diagnostic one:

1. On the VM, ask Ollama directly, with the service's exact greedy options, the
   same training-set question (for example "When and where does CSS 360 meet?")
   of `llama3.2:3b`, `css360-ft-v2:latest` and `css360-cpu-v3-test:latest`. If the
   three answers match, the adapters are inert or not applied
   (`ollama show css360-ft-v2:latest --modelfile` shows whether an ADAPTER line
   is present). If they differ but are all wrong, the adapters are too weak:
   check training steps, loss and LoRA rank/alpha for the v2 and v3 runs.
2. Until that is resolved, keep v3 mapped for testing only; the model-testing
   route makes re-running this benchmark a one-command job after any change.
3. Invest in retrieval and prompting, where every configuration failed alike:
   chunk headings that separate a schedule note from its date (q03), a policy
   section that never surfaces (q08), and no defence against false premises
   (q13, q14, q19). Those fixes help RAG and Fine-Tuned + RAG equally.
4. If fine-tuning continues, aim it at behaviour (abstention, concision,
   format) rather than knowledge, which is the only place v3 showed a gain.

## Files

| File | Contents |
| --- | --- |
| `results.json` | 110 raw answers with request, status, full response, timing |
| `scores.json` | 110 rubric scores with rationale |
| `questions.json` | the 22 questions with scorer-only references and coverage tags |
| `run.log` | timestamped progress log of the run |
| `run_benchmark.py`, `summarize.py`, `show.py` | runner, aggregation, side-by-side viewer |
