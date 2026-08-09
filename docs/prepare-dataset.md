# Seed Dataset Preparation

> **Legacy / not used by current QLoRA training.**  
> Prefer `backend/scripts/prepare_qlora_dataset.py` and `training/README.md`.

This document describes the local preparation script that converts the combined export into review-ready and fine-tuning-ready files.

This step **only prepares data**. It does **not** train or fine-tune a model, and it does **not** modify Firebase data.

## Prerequisites

Run the export step first so this file exists:

```text
data/exports/seed-dataset-combined.json
```

```bash
python3 scripts/export_seed_dataset.py
```

The preparation script reads whatever records are present in that file at runtime. Counts are not hardcoded.

## What the preparation script does

`scripts/prepare_seed_dataset.py`:

1. Loads `data/exports/seed-dataset-combined.json`
2. Validates required fields on each record
3. Normalizes instruction/response text
4. Preserves review metadata:
   - `id`
   - `instruction` / `question`
   - `response` / `answer`
   - `category`
   - `sourceSection`
   - `difficulty`
   - `answerType`
   - `source` (`prototype` or `student`)
   - `createdAt` when available
   - `notes` when available
5. Flags records recommended for manual review
6. Loads `data/reviews/seed-review-overrides.json` when present and applies manual review overrides
7. Removes exact duplicates again using normalized question + answer text
8. Writes prepared JSON/JSONL outputs

Train/validation/test splits are intentionally **not** created here.

## Manual review overrides

Do **not** edit generated files under `data/prepared/`. They are overwritten on every run.

Instead, add review decisions to the tracked override file:

```text
data/reviews/seed-review-overrides.json
```

Supported override statuses:

- `accepted`
- `rejected`
- `needs_review`

Rejected examples are excluded from `seed-dataset-finetuning.jsonl`.

See `docs/review-overrides.md` for the full workflow and override schema.

## Output files

Generated files are written to `data/prepared/`:

| File | Description |
|------|-------------|
| `seed-dataset-prepared.json` | Review-ready dataset with metadata and validation warnings |
| `seed-dataset-prepared.jsonl` | Same records as JSONL |
| `seed-dataset-finetuning.jsonl` | Future fine-tuning format with system prompt, Alpaca-style fields, and chat `messages` |
| `preparation-summary.json` | Machine-readable counts and validation summary |

Generated prepared files are ignored by git. The `data/prepared/` directory itself is kept via `.gitkeep`.

## How to run

From the project root:

```bash
python3 scripts/prepare_seed_dataset.py
```

Optional arguments:

```bash
python3 scripts/prepare_seed_dataset.py \
  --input data/exports/seed-dataset-combined.json \
  --overrides data/reviews/seed-review-overrides.json \
  --output-dir data/prepared
```

No extra Python packages are required.

## Review recommendations

A record is marked `reviewRecommended: true` when any of the following apply:

- source is `student`
- the example is not directly answered by the syllabus
- the record has validation warnings such as an unknown category

Use `data/prepared/seed-dataset-prepared.json` to inspect generated review output.

Apply manual corrections in `data/reviews/seed-review-overrides.json`, then re-run preparation.

## Fine-tuning file format

Each line in `seed-dataset-finetuning.jsonl` includes:

- `system`
- `instruction`
- `input` (empty string for now)
- `output`
- `messages` with `system`, `user`, and `assistant` roles
- `id` and `source` metadata

This keeps the dataset compatible with common instruction-tuning workflows without starting training.

## Expected workflow

1. Export combined data:

   ```bash
   python3 scripts/export_seed_dataset.py
   ```

2. Prepare the export:

   ```bash
   python3 scripts/prepare_seed_dataset.py
   ```

3. Inspect `data/prepared/seed-dataset-prepared.json`

4. Add or update review overrides in `data/reviews/seed-review-overrides.json`

5. Re-run preparation to apply overrides

6. Use `data/prepared/seed-dataset-finetuning.jsonl` in a later training step

See `docs/split-dataset.md` for the deterministic train/validation/test split step:

```bash
python3 scripts/split_training_dataset.py
```

## Runtime counts

The script prints counts based on the input file available when you run it.

Example:

- a workspace with prototype-only export may show 55 examples
- a local machine with Firebase included may show 57 examples (55 prototype + 2 student)

Re-run the script after regenerating `data/exports/seed-dataset-combined.json` locally to refresh prepared outputs and summary counts.

## Related docs

- Export step: `docs/export-dataset.md`
- Review overrides: `docs/review-overrides.md`
- App schema reference: `docs/data-format.md`
