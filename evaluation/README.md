# Held-out evaluation questions

> **STATUS 2026-09-02**
>
> | Course | Status | Questions |
> | --- | --- | --- |
> | `css-350-spring-2026-n3h9` | `VERIFIED_FOR_CURRENT_MODEL` | 20 (3 unanswerable) — checked against the v2 dataset, 0 flagged |
> | `css-350-winter-2026-drlb` | `INVALID_WRONG_COURSE` | 20 — superseded, reference only, do not use |
> | `css-360-winter-2026-a7rp` | `UNVERIFIED_DO_NOT_USE` | 20 — deferred until the v2 retrain |
>
> **The authoritative source is the deployed UWB VM.** Other content under
> `backend/course_data/` and `data/exports/` in this checkout is July 2026 leftover
> from a developer laptop and is not evidence of deployed state.

## CSS 350 — verified

Checked against the exact dataset behind the live adapter: model **v2**, run
`run-20260827t205310z-8c3cdb`, dataset version
`css-350-spring-2026-n3h9-approved-split-seed360-n42` — 42 approved / 37 train /
5 validation, 84 training-side questions compared, **0 flagged**.

Re-run any time with:

```bash
python3 evaluation/check_overlap.py --course css-350-spring-2026-n3h9
```

**One data issue for the researchers.** The syllabus uploaded to this Spring 2026
course is byte-identical to the Winter 2026 document (SHA-256
`1da7f30a…6569fe8`) and its own title line reads *Management Principles (Winter
2026)*. Reference answers are grounded in that uploaded document, because it is what
the v2 adapter trained on and what RAG retrieves — so the answers are correct for the
deployed system. But the course record says Spring while its syllabus says Winter,
which is worth confirming with the instructor before results are published.

## CSS 360 — deferred

Current model is **v1**, `trainingExampleCount` 54, trained through the legacy/manual
workflow. Its exact train/validation split is not available in the VM export
directory, so these questions cannot be checked against what the live adapter saw.
CSS 360 will be retrained as **v2** on the current pipeline; validate then. The v1
split must not be reconstructed or guessed.

## Rules

- **These questions must not be added to approved examples.** They must never be
  entered through the contribute flow, the professor review queue, or any direct
  write to `seed_examples`.
- **They must not be exported into QLoRA training data.** Nothing here belongs in
  `data/exports/{courseId}/approved-finetune.jsonl`, `train.jsonl`, or
  `validation.jsonl`.
- **They must not be used to regenerate training examples.** Do not feed them to
  starter-seed generation as prompts, few-shot examples, or targets.
- **`referenceAnswer` must never be included in a model prompt.** It exists only so
  a human researcher can score whether an answer was correct. Putting it in a prompt
  turns the whole comparison into an open-book test and destroys the result.
- Ask each question through the normal student path (Compare → Evaluate → Results).
  The models should see the question text and nothing else from this file.

## Files

| File | Purpose |
| --- | --- |
| `held_out_questions.json` | The question bank, grouped by course |
| `check_overlap.py` | Read-only script that re-derives the overlap verdicts |

## Schema

`held_out_questions.json` holds a `courses` array. Each course carries its
`courseId`, a note on what training corpus exists for it on disk, and a `questions`
array. Each question has:

| Field | Meaning |
| --- | --- |
| `id` | Stable identifier, e.g. `css360-eval-001` |
| `courseId` | Repeated on each question so rows survive a flat export |
| `question` | The exact text to ask the model. Nothing else goes in the prompt |
| `category` | One of the categories listed at the top of the file |
| `answerableFromSyllabus` | `false` marks a deliberate hallucination probe |
| `referenceAnswer` | For the human scorer only. **Never prompt with this** |
| `sourceSection` | The syllabus heading the answer comes from |
| `overlapVerdict` | Result of the overlap check below |

When `answerableFromSyllabus` is `false`, the correct model behaviour is to say the
syllabus does not cover it. `referenceAnswer` then describes that expected behaviour
and states what the syllabus *does* say nearby, so a researcher can tell "admitted
uncertainty" apart from "invented a policy".

## Overlap checking

```bash
python3 evaluation/check_overlap.py
```

The script compares every question against every training-side question on disk:
the approved export, the train/validation split, the generated-seed snapshots, and
`training/heldout_questions.json`. It reads only; it writes nothing.

Overlap is judged **within a course**. Adapters are trained per course, so a CSS 360
training example cannot leak into a CSS 350 answer; cross-course near-matches are
printed as informational and do not affect a verdict.

Two measures run in parallel — Jaccard overlap of content words (catches rewordings
like "When does class meet?" vs "What time are the course meetings?") and difflib
character similarity (catches light edits). A candidate is rejected at `jaccard ≥
0.60` or `ratio ≥ 0.75`, and flagged for human resolution at `0.45` / `0.60`. The
script exits non-zero if anything is flagged. It currently exits clean.

## Course coverage

| Course | Questions | Unanswerable | Approved examples on disk |
| --- | --- | --- | --- |
| `css-360-winter-2026-a7rp` | 20 | 4 | 54 approved → 48 train / 6 validation |
| `css-350-winter-2026-drlb` | 20 | 3 | none |

CSS 350 has a syllabus and a RAG index but no approved export and no adapter, so its
questions are trivially held out today. The set is written now, before any CSS 350
examples are approved, precisely so the test is fixed before the training material
exists.

## Why the training pipeline cannot pick this up

Verified against the code, not assumed:

- Examples enter training from **PostgreSQL only**. `backend/scripts/prepare_qlora_dataset.py`
  calls `fetch_course_seed_examples(courseId)`, and `backend/app/seed_export.py`
  writes the result to `data/exports/{courseId}/`. No stage scans the repository for
  question files.
- `backend/app/seed_split.py` reads exactly one input,
  `data/exports/{courseId}/approved-finetune.jsonl`, and writes `train.jsonl`,
  `validation.jsonl`, and `manifest.json` beside it.
- `training/train_qlora.py` takes explicit `--train-file` / `--validation-file`
  paths that default under `data/exports/`. It globs nothing.
- `scripts/sync_training_data_to_tillicum.sh` rsyncs one directory,
  `data/exports/<courseId>/`, and nothing else.
- Starter-seed generation (`backend/app/seed_generation.py`) reads no files at all;
  it works from syllabus text passed in memory.

`evaluation/` sits outside every one of those paths. Note also that `data/exports/*`
is gitignored while `evaluation/` is tracked, which is the intent: the training
export is a rebuildable local artifact, and the held-out set is a committed research
record.

## Not the same as `training/heldout_questions.json`

That file is the input to the offline Tillicum comparison job
(`training/compare_inference.py`, `training/compare.slurm`). It is nine unscored
CSS 360 questions used as a smoke check that a serving session answers at all. It
has no course scoping and no reference answers, and it is not this evaluation set.
The questions here were checked against it too, and none duplicate it.
