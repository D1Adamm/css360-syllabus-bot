# Training Dataset Splits

> **Legacy / not used by current QLoRA training.**  
> The canonical path is Firebase approved seeds →
> `backend/scripts/prepare_qlora_dataset.py` → `data/exports/<courseId>/` →
> Tillicum helpers in `training/README.md`. This document describes the older
> `data/splits/` 70/15/15 pipeline (`scripts/split_training_dataset.py`).

This document describes the deterministic train/validation/test split step for the reviewed fine-tuning dataset.

This step **creates splits only**. It does **not** fine-tune a model.

## Prerequisites

Prepare the reviewed fine-tuning dataset first:

```bash
python3 scripts/export_seed_dataset.py
python3 scripts/prepare_seed_dataset.py
```

Input file:

```text
data/prepared/seed-dataset-finetuning.jsonl
```

The split script also reads `data/prepared/seed-dataset-prepared.jsonl` when available to enrich category and review metadata for balanced splitting.

## How to run

From the project root:

```bash
python3 scripts/split_training_dataset.py
```

Optional arguments:

```bash
python3 scripts/split_training_dataset.py \
  --input data/prepared/seed-dataset-finetuning.jsonl \
  --prepared data/prepared/seed-dataset-prepared.jsonl \
  --output-dir data/splits \
  --seed 42
```

No extra Python packages are required.

## Output files

Generated files are written to `data/splits/`:

| File | Purpose |
|------|---------|
| `train.jsonl` | Training examples (~70%) |
| `validation.jsonl` | Validation examples (~15%) |
| `test.jsonl` | Held-out test examples (~15%) |
| `split-summary.json` | Counts, category/source breakdown, and IDs per split |

Generated split files are ignored by git. The `data/splits/` directory itself is kept via `.gitkeep`.

## Split behavior

- Uses a fixed random seed (`42` by default) for reproducible splits
- Targets approximately:
  - 70% train
  - 15% validation
  - 15% test
- For 57 records, a split around 40 / 9 / 8 is expected
- Groups examples by policy area using `category` and `sourceSection`
- Keeps highly similar questions from the same policy area in the same split when possible
- Preserves all metadata fields from the input record, enriched from prepared JSONL when available
- Keeps student-created examples identifiable through the `source` field

## Why the test split must stay untouched

The test split is a final held-out evaluation set.

Use it only after training and validation tuning are complete:

| Split | Allowed use |
|-------|-------------|
| `train.jsonl` | Model training |
| `validation.jsonl` | Hyperparameter tuning and early stopping |
| `test.jsonl` | Final evaluation only |

Do **not** use `test.jsonl` during training, prompt selection, or model selection. Using it early would leak evaluation signal and make reported results optimistic.

## Expected workflow

1. Export combined data
2. Prepare reviewed fine-tuning JSONL
3. Split the dataset:

   ```bash
   python3 scripts/split_training_dataset.py
   ```

4. Inspect `data/splits/split-summary.json`
5. Use `train.jsonl` and `validation.jsonl` in a later fine-tuning step
6. Reserve `test.jsonl` for final evaluation

## Runtime counts

Split counts are computed from whatever records are present in the input file at runtime.

Example:

- a prototype-only environment may split 55 records
- a local environment with reviewed student examples may split 57 records

Re-run the split after regenerating `data/prepared/seed-dataset-finetuning.jsonl` locally.

## Related docs

- Export step: `docs/export-dataset.md`
- Preparation step: `docs/prepare-dataset.md`
- Review overrides: `docs/review-overrides.md`
