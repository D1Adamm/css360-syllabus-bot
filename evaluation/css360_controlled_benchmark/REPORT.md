# CSS 360 controlled benchmark: report

**Run:** `run-20260919T140406Z`, 2026-09-19 14:04:06 to 16:29:40 UTC on the UWB VM, 184 requests
over the research route on loopback, mean 47.4 s per request, no retries.
**Scope:** Base and RAG as controls; standalone Fine-Tuned and Fine-Tuned + RAG for v2, v3, v4
trained on the VM, and v4 trained on Tillicum. Two question sets, never mixed: the 22 questions of
2026-09-11 asked again under controlled settings (`repeat22`), and 24 new held-out questions
(`heldout`: 10 factual, 7 unanswerable, 7 false-premise).
**Nothing was retrained, registered, published or promoted.** CSS 360 v2 remained
`current_version`, `ready` and `online` throughout. The 2026-09-11 scores are in section 13 and are
not merged into anything here.

## 1. Findings in brief

1. **Retrieval does the work; no adapter changes that.** On the same prompt bytes, the same ordered
   chunks and the same decoding, RAG scores 3.35 on correctness across the 46 questions and every
   Fine-Tuned + RAG condition scores between 3.35 and 3.50. Paired per question, no grounded adapter
   beats RAG at a level the sign test can distinguish from noise (section 6).
2. **v4 adds a modest gain on held-out abstention and nothing on facts.** Fine-Tuned + RAG v4-VM is
   the best grounded condition on the held-out set (overall 3.90 against RAG's 3.53, wins 12-8-4,
   sign-test p = 0.077) and it is the only condition that declined to invent an answer on every
   held-out unanswerable question without adding a hallucination. Removing the two probes whose
   near-duplicates sit in v4's training data, it still abstains on 5 of 5 against RAG's 3 of 5. On
   the repeat set it is slightly behind RAG (7-13-2 for RAG). On factual questions it is not better
   than RAG on either set.
3. **The two v4 adapters behave differently despite identical training data.** Fine-Tuned + RAG
   v4-Tillicum runs to the 256-token cap on 41 of 46 answers (v4-VM: 0), averages 204 words
   (v4-VM: 34), and is flagged for a hallucination on 43 percent of answers (v4-VM: 20 percent),
   while its per-question overall score ties v4-VM (17-14-15). The lineage record lists the runtime
   stack, the compute dtype, the completion-loss implementation and the adapter and GGUF dtypes as
   candidate causes; this run cannot separate them.
4. **Standalone fine-tuning still does not recall course facts.** Every plain Fine-Tuned condition
   scores 1.15 to 1.37 on correctness and is flagged for hallucination on 65 to 98 percent of
   answers. The v4 adapters made this worse, not better: they answer confidently and briefly with
   wrong values where v2 and v3 sometimes refuse. The bare-question Base control scores 1.48 and
   rarely hallucinates only because it usually refuses.
5. **Seven questions defeat every grounded condition** (h01, h08, h10, q03, q08, q12, q14): the
   supporting sentence is not in the retrieved chunks, or a premise is accepted by all five models.
   These are properties of the chunker, the retriever and the base model, not of any adapter.
6. **The run itself was clean.** Every alias was served under the tag and digest the lineage record
   names, before and after the run; all 460 attempted conditions produced an answer; all 46 grounded
   comparisons were hash-identical across their two request groups.

## 2. What was run and how it was checked

| Item | Value |
| --- | --- |
| Backend commit on the VM | `1e6920c3c47f91db1064f65a97cc0edcc414b361` (main) |
| Retrieval index | `115a0dfca9aaa81f0fe2a2b91858bcf7108ea2f18e39d00c9fdc3f34238991fa`, 163 chunks, the same index the v4 training prompts were rendered over |
| Syllabus text | `a00613cfed6d041073ff36a76813798eaecd75126cb334d14288c153ccf9b9c8` |
| Prompt template | `grounded-v1`, fingerprint `c963ffb5bd97aa946e730fb916b7ddedcf3a7cab9049199be186b994752d7a93` (equal to the fingerprint in the v4 training manifests) |
| Decoding, every condition | greedy (`temperature 0`), `num_predict 256`, `repeat_penalty 1.05` over `repeat_last_n 4096`, `seed 360`, `num_ctx 4096` |
| Retrieval depth | topK 4, the production default |
| Grouping | pair route: `rag, ft_rag:v2, ft_rag:v3` then `ft_rag:v4_vm, ft_rag:v4_tillicum`; standalone likewise; 184 requests |
| Question files | `questions_repeat22.json` `e8cb109f…`, `questions_heldout.json` `1461007c…` |

**Tags and digests.** The benchmark service's `/health` was compared with `evaluation/model_lineage.json`
before the run and again after it. All five matched both times:

| Alias | Served tag | Digest (served = record) |
| --- | --- | --- |
| base | `llama3.2:3b` | `a80c4f17acd5` |
| v2 | `css360-ft-v2:latest` | `dea74c57f25a` |
| v3 | `css360-cpu-v3-test:latest` | `0d723f0db28f` |
| v4_vm | `css360-v4-test:latest` | `b8763e820e93` |
| v4_tillicum | `css360-v4-tillicum-test:latest` | `d323b6f366b6` |

Every one of the 460 condition records carries the served tag and the full digest of the model that
answered it, and every record's `tagMatchesLineage`, `digestMatchesLineage`, `decodingMatchesSpec`
and `promptEchoMatches` flags are true.

**Fair comparison.** For each question the pair route was called twice, once per group, and each
call retrieved once and answered every condition in the group from one prompt. Across a question's
two groups the runner compared the prompt SHA-256, the ordered-chunk SHA-256, the template
fingerprint and the decoding block: 46 of 46 questions were identical, so no grounded comparison had
to be invalidated. The standalone route sent the whitespace-normalised question as the one user turn
to every condition, Base included; the production Base wrapper was not applied.

**Attempts.** 460 attempted, 460 scorable, 0 `request_failed`, 0 `failed_generation`, 0 `invalid`,
0 `error`, 0 requests retried. 71 answers hit the 256-token cap (`doneReason: length`); they were
scored as they stand and the truncation rate is reported per condition in section 11.

## 3. Scoring protocol

The rubric in `RUBRIC.md` was written before any answer was read. Scoring was blind: `blind.py`
shuffled each question's ten conditions behind the letters A–J with a per-question seed, the scorer
worked from the letters, and `summarize.py` joined the letters back to conditions only after all 460
scores were recorded. Every score cites a syllabus passage (`citedPassage` in
`results/run-20260919T140406Z/scores.json`) and carries a one-line rationale.

Fields: correctness, groundedness and completeness on 1–5; `hallucination` (any asserted course
fact the syllabus does not support, including a wrong value for a real fact); `abstained` for
unanswerable questions; `premiseHandling` (corrected, accepted, evaded) for false-premise questions.
`overall` is the mean of the three scales. Two conventions applied throughout and worth knowing when
reading the tables: an answer that wrongly claims the syllabus does not specify something is a
correctness failure, not a hallucination (only positively asserted unsupported facts count); and an
invented extra fact in an otherwise right answer caps correctness at 3 and sets the flag.

Limits of the protocol: one scorer, the same model that authored the held-out questions and key
facts; blind to the condition label but not to the obvious difference between a grounded answer and
a bare one; one greedy generation per condition.

## 4. Main results

The scales are means over the questions in the set (1–5). `Hallucinated` and `Truncated` are counts
of answers. `Gen s` is the service's generation time including any model load; `Words` is the
answer length.

### Both sets, 46 questions

| Condition | n | Correct | Grounded | Complete | Overall | Hallucinated | Truncated | Gen s | Words |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base (standalone) | 46 | 1.48 | 2.78 | 1.30 | 1.86 | 1/46 (2%) | 9/46 (20%) | 14.7 | 99 |
| RAG | 46 | 3.35 | 4.28 | 3.11 | 3.58 | 8/46 (17%) | 0/46 (0%) | 19.0 | 41 |
| FT v2 | 46 | 1.30 | 1.63 | 1.22 | 1.38 | 31/46 (67%) | 4/46 (9%) | 13.0 | 70 |
| FT+RAG v2 | 46 | 3.35 | 4.04 | 3.20 | 3.53 | 12/46 (26%) | 0/46 (0%) | 20.0 | 36 |
| FT v3 | 46 | 1.30 | 1.70 | 1.24 | 1.41 | 30/46 (65%) | 6/46 (13%) | 12.5 | 66 |
| FT+RAG v3 | 46 | 3.35 | 4.22 | 3.13 | 3.57 | 9/46 (20%) | 0/46 (0%) | 19.5 | 30 |
| FT v4-VM | 46 | 1.37 | 1.41 | 1.33 | 1.37 | 41/46 (89%) | 0/46 (0%) | 8.5 | 22 |
| FT+RAG v4-VM | 46 | 3.50 | 4.22 | 3.30 | 3.67 | 9/46 (20%) | 0/46 (0%) | 19.8 | 34 |
| FT v4-Tillicum | 46 | 1.15 | 1.09 | 1.17 | 1.14 | 45/46 (98%) | 11/46 (24%) | 16.0 | 102 |
| FT+RAG v4-Tillicum | 46 | 3.35 | 3.76 | 3.83 | 3.64 | 20/46 (43%) | 41/46 (89%) | 37.3 | 204 |

### Repeat of the 22 questions of 2026-09-11, controlled settings

| Condition | n | Correct | Grounded | Complete | Overall | Hallucinated | Truncated | Gen s | Words |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base (standalone) | 22 | 1.45 | 2.91 | 1.36 | 1.91 | 0/22 (0%) | 4/22 (18%) | 14.2 | 96 |
| RAG | 22 | 3.41 | 4.27 | 3.23 | 3.64 | 5/22 (23%) | 0/22 (0%) | 19.7 | 46 |
| FT v2 | 22 | 1.09 | 1.55 | 1.05 | 1.23 | 15/22 (68%) | 2/22 (9%) | 13.3 | 74 |
| FT+RAG v2 | 22 | 3.45 | 4.05 | 3.36 | 3.62 | 5/22 (23%) | 0/22 (0%) | 20.2 | 35 |
| FT v3 | 22 | 1.18 | 1.77 | 1.14 | 1.36 | 13/22 (59%) | 3/22 (14%) | 13.0 | 71 |
| FT+RAG v3 | 22 | 3.36 | 4.09 | 3.23 | 3.56 | 5/22 (23%) | 0/22 (0%) | 19.3 | 26 |
| FT v4-VM | 22 | 1.32 | 1.36 | 1.27 | 1.32 | 20/22 (91%) | 0/22 (0%) | 8.5 | 21 |
| FT+RAG v4-VM | 22 | 3.23 | 3.95 | 3.09 | 3.42 | 5/22 (23%) | 0/22 (0%) | 19.9 | 31 |
| FT v4-Tillicum | 22 | 1.05 | 1.00 | 1.05 | 1.03 | 22/22 (100%) | 7/22 (32%) | 17.1 | 111 |
| FT+RAG v4-Tillicum | 22 | 3.09 | 3.36 | 3.77 | 3.41 | 12/22 (55%) | 20/22 (91%) | 37.6 | 203 |

### Held-out set, 24 new questions

| Condition | n | Correct | Grounded | Complete | Overall | Hallucinated | Truncated | Gen s | Words |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base (standalone) | 24 | 1.50 | 2.67 | 1.25 | 1.81 | 1/24 (4%) | 5/24 (21%) | 15.2 | 102 |
| RAG | 24 | 3.29 | 4.29 | 3.00 | 3.53 | 3/24 (12%) | 0/24 (0%) | 18.4 | 37 |
| FT v2 | 24 | 1.50 | 1.71 | 1.38 | 1.53 | 16/24 (67%) | 2/24 (8%) | 12.8 | 66 |
| FT+RAG v2 | 24 | 3.25 | 4.04 | 3.04 | 3.44 | 7/24 (29%) | 0/24 (0%) | 19.8 | 37 |
| FT v3 | 24 | 1.42 | 1.62 | 1.33 | 1.46 | 17/24 (71%) | 3/24 (12%) | 12.1 | 62 |
| FT+RAG v3 | 24 | 3.33 | 4.33 | 3.04 | 3.57 | 4/24 (17%) | 0/24 (0%) | 19.7 | 34 |
| FT v4-VM | 24 | 1.42 | 1.46 | 1.38 | 1.42 | 21/24 (88%) | 0/24 (0%) | 8.5 | 23 |
| FT+RAG v4-VM | 24 | 3.75 | 4.46 | 3.50 | 3.90 | 4/24 (17%) | 0/24 (0%) | 19.8 | 36 |
| FT v4-Tillicum | 24 | 1.25 | 1.17 | 1.29 | 1.24 | 23/24 (96%) | 4/24 (17%) | 15.0 | 94 |
| FT+RAG v4-Tillicum | 24 | 3.58 | 4.12 | 3.88 | 3.86 | 8/24 (33%) | 21/24 (88%) | 37.0 | 205 |

## 5. By question kind

The held-out set is where abstention and premise correction are measured with enough items to mean
something (7 each); the repeat set has 4 unanswerable questions and one false premise (q14).

### Held-out set

| Condition | factual correct (n=10) | factual halluc. | unanswerable correct (n=7) | unanswerable halluc. | abstained | false_premise correct (n=7) | false_premise halluc. | corrected / accepted / evaded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base (standalone) | 1.00 | 0/10 (0%) | 2.71 | 1/7 (14%) | 6/7 (86%) | 1.00 | 0/7 (0%) | 0 / 3 / 4 |
| RAG | 3.20 | 1/10 (10%) | 3.71 | 1/7 (14%) | 5/7 (71%) | 3.00 | 1/7 (14%) | 3 / 2 / 2 |
| FT v2 | 1.70 | 7/10 (70%) | 1.29 | 6/7 (86%) | 1/7 (14%) | 1.43 | 3/7 (43%) | 2 / 5 / 0 |
| FT+RAG v2 | 3.20 | 3/10 (30%) | 3.57 | 2/7 (29%) | 5/7 (71%) | 3.00 | 2/7 (29%) | 2 / 2 / 3 |
| FT v3 | 1.60 | 7/10 (70%) | 1.29 | 6/7 (86%) | 1/7 (14%) | 1.29 | 4/7 (57%) | 1 / 6 / 0 |
| FT+RAG v3 | 3.30 | 1/10 (10%) | 3.57 | 2/7 (29%) | 5/7 (71%) | 3.14 | 1/7 (14%) | 3 / 1 / 3 |
| FT v4-VM | 1.80 | 8/10 (80%) | 1.00 | 7/7 (100%) | 0/7 (0%) | 1.29 | 6/7 (86%) | 1 / 6 / 0 |
| FT+RAG v4-VM | 3.40 | 4/10 (40%) | 4.71 | 0/7 (0%) | 7/7 (100%) | 3.29 | 0/7 (0%) | 3 / 0 / 4 |
| FT v4-Tillicum | 1.60 | 9/10 (90%) | 1.00 | 7/7 (100%) | 0/7 (0%) | 1.00 | 7/7 (100%) | 0 / 7 / 0 |
| FT+RAG v4-Tillicum | 3.20 | 5/10 (50%) | 4.29 | 2/7 (29%) | 7/7 (100%) | 3.43 | 1/7 (14%) | 4 / 0 / 3 |

Two held-out unanswerable questions (u01, group size; u02, extra credit) have near-duplicate
authored abstention seeds in the v4 training export, and two false-premise questions (p03, p05) have
a key fact in a v4 training response. With those removed the rates are, out of 5:

| Condition | Abstained | Hallucinated (unanswerable) | Premise corrected |
| --- | --- | --- | --- |
| Base (standalone) | 4/5 | 1/5 | 0/5 |
| RAG | 3/5 | 1/5 | 3/5 |
| FT v2 | 1/5 | 4/5 | 1/5 |
| FT+RAG v2 | 3/5 | 2/5 | 2/5 |
| FT v3 | 1/5 | 4/5 | 0/5 |
| FT+RAG v3 | 3/5 | 2/5 | 3/5 |
| FT v4-VM | 0/5 | 5/5 | 1/5 |
| FT+RAG v4-VM | 5/5 | 0/5 | 2/5 |
| FT v4-Tillicum | 0/5 | 5/5 | 0/5 |
| FT+RAG v4-Tillicum | 5/5 | 2/5 | 3/5 |

The Base control "abstains" by refusing without reference to the syllabus (correctness 3 under the
rubric), which is why its abstention count is high and its correctness low.

### Repeat set

| Condition | factual correct (n=17) | factual halluc. | unanswerable correct (n=4) | unanswerable halluc. | abstained | false_premise correct (n=1) | false_premise halluc. | corrected / accepted / evaded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base (standalone) | 1.12 | 0/17 (0%) | 3.00 | 0/4 (0%) | 4/4 (100%) | 1.00 | 0/1 (0%) | 0 / 0 / 1 |
| RAG | 3.24 | 4/17 (24%) | 4.75 | 0/4 (0%) | 4/4 (100%) | 1.00 | 1/1 (100%) | 0 / 1 / 0 |
| FT v2 | 1.12 | 10/17 (59%) | 1.00 | 4/4 (100%) | 0/4 (0%) | 1.00 | 1/1 (100%) | 0 / 1 / 0 |
| FT+RAG v2 | 3.35 | 4/17 (24%) | 4.50 | 0/4 (0%) | 4/4 (100%) | 1.00 | 1/1 (100%) | 0 / 1 / 0 |
| FT v3 | 1.18 | 9/17 (53%) | 1.25 | 3/4 (75%) | 0/4 (0%) | 1.00 | 1/1 (100%) | 0 / 1 / 0 |
| FT+RAG v3 | 3.35 | 4/17 (24%) | 4.00 | 0/4 (0%) | 4/4 (100%) | 1.00 | 1/1 (100%) | 0 / 1 / 0 |
| FT v4-VM | 1.41 | 15/17 (88%) | 1.00 | 4/4 (100%) | 0/4 (0%) | 1.00 | 1/1 (100%) | 0 / 1 / 0 |
| FT+RAG v4-VM | 3.00 | 4/17 (24%) | 4.75 | 0/4 (0%) | 4/4 (100%) | 1.00 | 1/1 (100%) | 0 / 1 / 0 |
| FT v4-Tillicum | 1.06 | 17/17 (100%) | 1.00 | 4/4 (100%) | 0/4 (0%) | 1.00 | 1/1 (100%) | 0 / 1 / 0 |
| FT+RAG v4-Tillicum | 3.00 | 10/17 (59%) | 4.00 | 1/4 (25%) | 4/4 (100%) | 1.00 | 1/1 (100%) | 0 / 1 / 0 |

On q14 (midterm exams) all ten conditions failed: nine accepted the premise that exams exist and
Base evaded. On the four repeat unanswerable questions every grounded condition abstained on all
four; every standalone Fine-Tuned condition invented an answer on at least three.

## 6. Paired comparisons

Per question, on `overall`; wins-ties-losses are for the first-named condition. Grounded pairs use
only hash-identical questions (all 46). A sign-test p-value is shown only with at least ten non-tied
pairs; none of the grounded comparisons reaches p < 0.05 on the combined set except RAG against
Fine-Tuned + RAG v2, where RAG is ahead by 0.05 on overall and by 0.00 on correctness, a difference
of formatting and groundedness rather than of facts.

### Both sets

| Comparison | pairs | W-T-L | mean overall diff | mean correctness diff | sign-test p |
| --- | --- | --- | --- | --- | --- |
| RAG vs FT+RAG v2 | 46 | 13-29-4 | 0.05 | 0.00 | 0.049 |
| RAG vs FT+RAG v3 | 46 | 11-30-5 | 0.01 | 0.00 | 0.210 |
| RAG vs FT+RAG v4-VM | 46 | 11-21-14 | -0.09 | -0.15 | 0.690 |
| RAG vs FT+RAG v4-Tillicum | 46 | 22-6-18 | -0.07 | 0.00 | 0.636 |
| Base vs FT v2 | 46 | 30-13-3 | 0.47 | 0.17 | 0.000 |
| Base vs FT v3 | 46 | 27-15-4 | 0.44 | 0.17 | 0.000 |
| Base vs FT v4-VM | 46 | 35-3-8 | 0.49 | 0.11 | 0.000 |
| Base vs FT v4-Tillicum | 46 | 40-4-2 | 0.72 | 0.33 | 0.000 |
| FT+RAG v4-VM vs FT+RAG v4-Tillicum | 46 | 17-14-15 | 0.03 | 0.15 | 0.860 |
| FT v4-VM vs FT v4-Tillicum | 46 | 8-38-0 | 0.23 | 0.22 | n/a |
| FT+RAG v4-VM vs FT+RAG v2 | 46 | 17-20-9 | 0.14 | 0.15 | 0.169 |
| FT+RAG v4-Tillicum vs FT+RAG v2 | 46 | 19-10-17 | 0.12 | 0.00 | 0.868 |
| FT+RAG v4-VM vs FT+RAG v3 | 46 | 18-17-11 | 0.11 | 0.15 | 0.265 |
| FT v4-VM vs FT v2 | 46 | 5-29-12 | -0.01 | 0.07 | 0.143 |

### Held-out set

| Comparison | pairs | W-T-L | mean overall diff | mean correctness diff | sign-test p |
| --- | --- | --- | --- | --- | --- |
| RAG vs FT+RAG v2 | 24 | 5-18-1 | 0.08 | 0.04 | n/a |
| RAG vs FT+RAG v3 | 24 | 2-20-2 | -0.04 | -0.04 | n/a |
| RAG vs FT+RAG v4-VM | 24 | 4-8-12 | -0.38 | -0.46 | 0.077 |
| RAG vs FT+RAG v4-Tillicum | 24 | 8-5-11 | -0.33 | -0.29 | 0.648 |
| Base vs FT v2 | 24 | 13-8-3 | 0.28 | 0.00 | 0.021 |
| Base vs FT v3 | 24 | 13-8-3 | 0.35 | 0.08 | 0.021 |
| Base vs FT v4-VM | 24 | 17-3-4 | 0.39 | 0.08 | 0.007 |
| Base vs FT v4-Tillicum | 24 | 18-4-2 | 0.57 | 0.25 | 0.000 |
| FT+RAG v4-VM vs FT+RAG v4-Tillicum | 24 | 6-12-6 | 0.04 | 0.17 | 1.000 |
| FT v4-VM vs FT v4-Tillicum | 24 | 4-20-0 | 0.18 | 0.17 | n/a |
| FT+RAG v4-VM vs FT+RAG v2 | 24 | 11-10-3 | 0.46 | 0.50 | 0.057 |
| FT+RAG v4-Tillicum vs FT+RAG v2 | 24 | 11-8-5 | 0.42 | 0.33 | 0.210 |
| FT+RAG v4-VM vs FT+RAG v3 | 24 | 11-8-5 | 0.33 | 0.42 | 0.210 |
| FT v4-VM vs FT v2 | 24 | 1-17-6 | -0.11 | -0.08 | n/a |

### Repeat set

| Comparison | pairs | W-T-L | mean overall diff | mean correctness diff | sign-test p |
| --- | --- | --- | --- | --- | --- |
| RAG vs FT+RAG v2 | 22 | 8-11-3 | 0.02 | -0.05 | 0.227 |
| RAG vs FT+RAG v3 | 22 | 9-10-3 | 0.08 | 0.05 | 0.146 |
| RAG vs FT+RAG v4-VM | 22 | 7-13-2 | 0.21 | 0.18 | n/a |
| RAG vs FT+RAG v4-Tillicum | 22 | 14-1-7 | 0.23 | 0.32 | 0.189 |
| Base vs FT v2 | 22 | 17-5-0 | 0.68 | 0.36 | 0.000 |
| Base vs FT v3 | 22 | 14-7-1 | 0.55 | 0.27 | 0.001 |
| Base vs FT v4-VM | 22 | 18-0-4 | 0.59 | 0.14 | 0.004 |
| Base vs FT v4-Tillicum | 22 | 22-0-0 | 0.88 | 0.41 | 0.000 |
| FT+RAG v4-VM vs FT+RAG v4-Tillicum | 22 | 11-2-9 | 0.02 | 0.14 | 0.824 |
| FT v4-VM vs FT v4-Tillicum | 22 | 4-18-0 | 0.29 | 0.27 | n/a |
| FT+RAG v4-VM vs FT+RAG v2 | 22 | 6-10-6 | -0.20 | -0.23 | 1.000 |
| FT+RAG v4-Tillicum vs FT+RAG v2 | 22 | 8-2-12 | -0.21 | -0.36 | 0.503 |
| FT+RAG v4-VM vs FT+RAG v3 | 22 | 7-9-6 | -0.14 | -0.14 | 1.000 |
| FT v4-VM vs FT v2 | 22 | 4-12-6 | 0.09 | 0.23 | 0.754 |

Base beats every standalone Fine-Tuned condition on overall at p < 0.001 in every scope. The
correctness differences are small (0.11 to 0.33); the gap is groundedness, because a generic refusal
scores 3 there and a fabricated policy scores 1.

## 7. What v4 adds beyond RAG

Taking Fine-Tuned + RAG v4-VM as the better of the two v4 adapters:

- **Facts:** nothing. Correctness on factual questions is 3.00 (repeat) and 3.40 (held-out) against
  RAG's 3.24 and 3.20. The questions it gets right are the ones RAG gets right.
- **Abstention:** the clearest effect in the run. On held-out unanswerable questions it abstained
  7 of 7 with no hallucination, RAG 5 of 7 with one. Two of the seven are training-adjacent for v4;
  on the other five it is 5 of 5 against 3 of 5. On the repeat set both abstain 4 of 4. The mixed-v4
  data contained 20 grounded abstention examples, and this is the behaviour they targeted.
- **False premises:** a wash. Corrected 3 of 7 (2 of 5 excluding training-adjacent items) against
  RAG's 3 of 7 (3 of 5); its remaining answers evaded rather than accepted, which is the safer
  failure but not a correction. The v4 data contained 14 false-premise examples; the behaviour did
  not transfer to new premises at a rate distinguishable from RAG's.
- **Cost:** none in latency (19.8 s against 19.0 s mean generation) and none in length (34 words
  against 41).
- **Net:** on the combined set it wins 14, ties 21 and loses 11 against RAG on overall, with a mean
  difference of 0.09 in its favour and p = 0.69. That is consistent with a small real improvement in
  abstention and with no improvement at all; 46 questions cannot tell the two apart.

Fine-Tuned + RAG v4-Tillicum adds completeness (3.83 against 3.11, the highest of any condition)
by answering at length, and pays for it with the highest grounded hallucination count (20 of 46)
and truncation on 41 of 46 answers. Its correctness equals RAG's (3.35).

The plain v4 adapters, without retrieval, are the weakest conditions in the run on every scale
except brevity: v4-VM answers in 22 words with a wrong value 89 percent of the time.

## 8. v4 trained on the VM versus v4 trained on Tillicum

Same training bytes (four matching SHA-256s on both hosts), same commit `1104ad5`, same seed, hyper-
parameters, LoRA shape, 48 optimizer steps and dataset composition; same base blob at inference.
Different runtime stack (transformers 5.16 / PEFT 0.20 / TRL 1.12 on the VM against 4.47 / 0.14 /
0.13 on Tillicum), compute path (CPU float32 without autocast against CUDA bfloat16 with autocast),
completion-loss implementation (`SFTConfig.completion_only_loss` against
`DataCollatorForCompletionOnlyLM`), saved adapter dtype (BF16 against F32) and GGUF dtype (F16
against F32).

What the served models did:

| | FT+RAG v4-VM | FT+RAG v4-Tillicum | FT v4-VM | FT v4-Tillicum |
| --- | --- | --- | --- | --- |
| Overall (46) | 3.67 | 3.64 | 1.37 | 1.14 |
| Correctness | 3.50 | 3.35 | 1.37 | 1.15 |
| Groundedness | 4.22 | 3.76 | 1.41 | 1.09 |
| Completeness | 3.30 | 3.83 | 1.33 | 1.17 |
| Hallucinated | 9/46 | 20/46 | 41/46 | 45/46 |
| Truncated at 256 tokens | 0/46 | 41/46 | 0/46 | 11/46 |
| Mean generated tokens | 41 | 243 | 30 | 127 |
| Mean generation seconds | 19.8 | 37.3 | 8.5 | 16.0 |
| Paired, VM vs Tillicum (W-T-L) | 17-14-15, p = 0.86 | | 8-38-0 | |

The two adapters are indistinguishable on per-question overall score in the grounded condition and
very different in behaviour: the Tillicum adapter learned to keep writing. Its grounded answers
typically give the right facts in the first two sentences and then continue with paraphrases of
neighbouring syllabus text, self-contradictions ("the syllabus does not specify" after specifying
it) and, in 15 of 27 factual answers, an unsupported or wrong claim, until the cap stops them. The
VM adapter learned the opposite: two to four sentences and stop, which is what the grounded
template asks for. In the standalone condition the VM adapter is terse and wrong; the Tillicum
adapter is long and wrong.

The eval losses on the shared validation records (1.4251 VM, 1.4499 Tillicum) are close, and the
record qualifies even that comparison until label-mask parity between the two completion-only
implementations is verified. Nothing in this benchmark identifies which of the listed differences
produced the length behaviour; the F16 GGUF conversion of the VM adapter and the F32 conversion of
the Tillicum adapter are among the candidates, as is the loss implementation.

## 9. Where every grounded condition failed

Every one of RAG and the four Fine-Tuned + RAG conditions scored 2 or less on:

| Question | Why |
| --- | --- |
| h01, weekly hours | the "10 hours per week" sentence was not retrieved; four grounded answers said the syllabus does not say |
| h08, October 28 guest speaker | the guest-speaker line was not retrieved; the Aaron Halfaker chunk was, and three answers moved him to October 28 or October 3 |
| h10, Mythical Man-Month chapter | four grounded answers gave Chapter 2 (the October 2 reading) for September 30 |
| q03, Zoom dates | the Zoom notes are split from their date headings; every answer listed Veterans Day |
| q08, dropped for missing the first week | the drop rule in "Your Presence in Class" was not retrieved |
| q12, pull request to the class repository | the "hard fork" tip was not retrieved; answers quoted the team-branch step and called the syllabus silent |
| q14, midterm exams | every condition accepted that exams exist |

These recur across adapters because the adapters never see the missing sentence. They are the
largest single source of lost score in the run and they belong to the chunker, the retriever and
the premise-handling of a 3B model.

## 10. Training-data overlap, recorded and applied

Two checks were run and their results travel with the questions.

- **July 2026 laptop split** (54 approved, 48 train, 6 validation): the only candidate for the v2
  and v3 training data, whose bytes are not retained. Key facts of repeat questions q01, q02, q04,
  q06, q07, q09 and q17 appear in its responses; q06 and q08 are REVIEW-level near matches resolved
  by hand in the original file.
- **The v4 export on the VM** (88 approved seeds, 122 train and 15 validation rendered records,
  hashes in `manifest.json`): key facts of q01, q02, q04, q06, q07, q09, q17, q18, h03, h05, h09,
  u06, p03 and p05 appear in its responses; u01 and u02 are REJECT-level matches to authored
  abstention seeds ("How many students are in each project group?", "Is there any extra credit?");
  h03, h09, u06 and p03 are REVIEW-level token overlaps resolved by hand.

Reading the "seen" split in `summary.md`: on factual questions whose key facts appear in v4
training responses, every grounded condition scores higher (RAG 3.75 against 2.80 unseen; v4-VM
3.67 against 2.73). The gap is the same size for RAG, which was never trained on them, so it
measures which facts sit in well-formed retrievable sentences, not memorisation. No condition shows
a training-set advantage over RAG on the seen questions.

## 11. Timings and failure rates

| Condition | attempted | scorable | gen s mean | gen s median | gen s max | load s mean | eval tokens mean | truncated (length) | words mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base (standalone) | 46 | 46 | 14.7 | 11.8 | 25.3 | 5.4 | 123 | 9 | 99 |
| RAG | 46 | 46 | 19.0 | 18.7 | 24.8 | 5.4 | 52 | 0 | 41 |
| FT v2 | 46 | 46 | 13.0 | 9.4 | 27.0 | 5.5 | 88 | 4 | 70 |
| FT+RAG v2 | 46 | 46 | 20.0 | 19.5 | 33.0 | 5.5 | 45 | 0 | 36 |
| FT v3 | 46 | 46 | 12.5 | 9.3 | 27.1 | 5.5 | 83 | 6 | 66 |
| FT+RAG v3 | 46 | 46 | 19.5 | 19.2 | 24.8 | 5.5 | 38 | 0 | 30 |
| FT v4-VM | 46 | 46 | 8.5 | 8.2 | 13.7 | 5.5 | 30 | 0 | 22 |
| FT+RAG v4-VM | 46 | 46 | 19.8 | 19.3 | 26.4 | 5.5 | 41 | 0 | 34 |
| FT v4-Tillicum | 46 | 46 | 16.0 | 13.4 | 30.4 | 5.5 | 127 | 11 | 102 |
| FT+RAG v4-Tillicum | 46 | 46 | 37.3 | 38.2 | 43.9 | 5.5 | 243 | 41 | 204 |

Every request paid a model load (mean 5.5 s) because the two groups per route alternate five models
on a host that keeps fewer resident; the `keep_alive` of 30 minutes did not change that. Generation
time is otherwise a function of output length: the four short grounded conditions cluster at 19 to
20 s, Fine-Tuned + RAG v4-Tillicum at 37 s because it writes to the cap, and standalone v4-VM at
8.5 s because it writes 30 tokens. Failure rates are zero in every category (section 2); the scorer
noted repetition or running to the cap in 34 rationales, concentrated in the truncated answers.

## 12. Training history of the five artifacts

From `docs/css360-model-evolution.md` and `evaluation/model_lineage.json`. Three different clocks
are kept apart: **trainer total** is the trainer process from model load to the end of evaluation;
**Slurm elapsed** is the cluster allocation, which adds the job script's own work; **VM elapsed**
is the trainer process on the VM, which has no scheduler, so it is the same number as the trainer
total. Conversion to GGUF and import into Ollama were not timed for any artifact.

| Artifact | Host, device | Trainer total | Training only | Model load | Evaluation | Slurm elapsed (queue wait) | VM elapsed | GPU-hours | Data | Window, loss | Final eval loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 `css360-v1` | Tillicum g007, H200 | 42.36 s | 31.35 s | 7.01 s | 1.23 s | 54 s (0 s) | n/a | 0.0118 | 48 / 6, bytes unknown | 512, full sequence | 1.6536 |
| v2 `css360-v2` | Tillicum g004, H200 | 42.46 s | 30.65 s | 7.81 s | 1.19 s | 53 s (0 s) | n/a | 0.0118 | n54: 48 / 6, checksums only | 512, full sequence | 2.0056 |
| v3 `css360-v3` | VM, CPU 6 threads | 3,329.9 s (55.5 min) | 3,252.6 s | 22.6 s | 163.3 s | n/a | 3,329.9 s | n/a | 48 / 6, identity with v2 unverified | 512, full sequence | 1.9422 |
| v4-VM `css360-v4-vm` | VM, CPU 6 threads | 38,754.6 s (10.77 h) | 38,297.4 s | 30.1 s | 1,842.0 s | n/a | 38,754.6 s | n/a | n137 mixed: 122 / 15 | 2048, completion-only (`SFTConfig`) | 1.4251 |
| v4-Tillicum `css360-v4-tillicum` | Tillicum g007, H200 | 90.65 s | 78.50 s | 7.39 s | 4.05 s | 104 s (4 s) | n/a | 0.0252 | byte-identical to v4-VM | 2048, completion-only (collator) | 1.4499 |

Eval losses are comparable only within the v4 pair, and there with the label-mask qualification;
v1, v2 and v3 used different validation records and a full-sequence loss. Every adapter has the
same LoRA shape (rank 8, alpha 16, dropout 0.05, seven target modules) and the same optimiser
settings; v4 changed the data, the window and the loss.

Artifacts as served: v2 F32 adapter, F32 GGUF; v3 BF16 adapter, F32 GGUF; v4-VM BF16 adapter, F16
GGUF; v4-Tillicum F32 adapter, F32 GGUF; all over the same Q4_K_M base blob.

**v1, documented and not run.** `css360-v1`, tag `css360-ft-v1-test:latest`, digest `063ce8788f88`,
the legacy Tillicum run of 2026-07-21 at commit `27f3b15`: F32 adapter narrowed to an F16 GGUF, registry
status `ready`/`offline`, never mapped in the fine-tuned service, and marked
`includeInControlledClaims: false` in the record because its training bytes are unknown (the July
laptop split is a candidate only). It was left out of this run because the route's alias allowlist
is fixed in code, so adding it would have meant code, tests and a redeploy ahead of the v2–v4
result, and because the F16 narrowing would have made it a different kind of artifact from the
others.

## 13. The 2026-09-11 benchmark, kept separate

`evaluation/model_version_benchmark/` holds the earlier 22-question run through
`POST /api/model-testing/generate`. Its RAG and Fine-Tuned + RAG conditions used different prompt
templates, different Ollama endpoints, different decoding and different output caps (160 tokens),
and its RAG timing was wall time including retrieval; the evolution record (section 8) lists why it
is not a controlled comparison. Its scores are therefore not compared numerically with anything in
this report, not used as a baseline, and not averaged in. The `repeat22` set here re-asks the same
22 question texts under one controlled setting and is a new measurement.

## 14. Limitations

- 46 questions and one greedy pass per condition: paired sign tests on 46 pairs cannot detect
  differences of a tenth of a point, and the held-out set has 7 items per non-factual kind.
- One scorer, an LLM, who also wrote the held-out questions and the rubric; blind to condition
  labels but able to see whether an answer was grounded.
- Two held-out probes are training-adjacent for v4 and four more share a fact with v4 responses;
  they are labelled and the adjusted counts are given, but they were not replaced.
- The standalone Base control receives the bare question, not the production Base wrapper; it
  measures the weights, not the production Base condition.
- The 256-token cap truncated 89 percent of Fine-Tuned + RAG v4-Tillicum answers; a longer cap would
  change its completeness and probably its hallucination count, in an unknown direction.
- Retrieval identity is by construction within a request and verified by hash across the two groups;
  it is not verified against the production RAG route, which uses the same functions but was not
  called.

## 15. Files

| File | Contents |
| --- | --- |
| `results/run-20260919T140406Z/manifest.json`, `preflight.json` | settings, fingerprints, tag and digest verification, export hashes |
| `results/run-20260919T140406Z/responses/` | 184 raw responses |
| `results/run-20260919T140406Z/records.jsonl` | 460 condition records: outcome, answer, tag, digest, flags, timing, hashes |
| `results/run-20260919T140406Z/grounded_identity.jsonl` | per-question hash identity across groups |
| `results/run-20260919T140406Z/overlap_vm_export.json` | overlap with the v4 export on the VM |
| `results/run-20260919T140406Z/blind/` | the blind sheet, the key, the blind scores |
| `results/run-20260919T140406Z/scores.json` | every scored answer, unblinded, with cited passage and rationale |
| `results/run-20260919T140406Z/summary.md`, `summary.json` | every table, including the per-question grid |
| `RUBRIC.md`, `questions_repeat22.json`, `questions_heldout.json` | the rubric and the two sets with their overlap blocks |
