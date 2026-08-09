# CSS 360 QLoRA training (Tillicum)

Fine-tunes **LoRA adapters only** on top of `meta-llama/Llama-3.2-3B-Instruct`
using the **approved** course export under `data/exports/<courseId>/`.

This is the **canonical** training workflow. Inference deployment is separate
(see `training/inference_service/README.md`).

Exports are **gitignored**. Code reaches Tillicum via `git pull`; training JSONL
must be prepared on a machine with Firebase access, then synced explicitly.

---

## Canonical workflow

### A) Machine with backend / Firebase access (local or UWB VM)

```bash
cd backend
.venv/bin/python scripts/prepare_qlora_dataset.py css-360-winter-2026-a7rp

cd ..
./scripts/sync_training_data_to_tillicum.sh css-360-winter-2026-a7rp
```

`prepare_qlora_dataset.py` calls the existing approved-export + train/validation
split logic (it does not reimplement them). Sync sends **only**
`data/exports/<courseId>/` to Tillicum (rsync; Duo remains interactive).

Required files after prepare:

- `data/exports/<courseId>/train.jsonl`
- `data/exports/<courseId>/validation.jsonl`
- `data/exports/<courseId>/manifest.json`

### B) Tillicum — train

```bash
cd /gpfs/projects/simswe/$USER/css360-syllabus-bot
git pull

./training/start_qlora_training.sh --course css-360-winter-2026-a7rp --smoke
./training/status_qlora_training.sh

# Only after smoke looks good — explicit second command:
./training/start_qlora_training.sh --course css-360-winter-2026-a7rp --full
```

Smoke and full are **never** chained automatically.

Automation writes versioned outputs under:

```text
/gpfs/projects/simswe/$USER/training_outputs/qlora-runs/<courseId>/<runId>-{smoke|full}/
```

including `adapter/`. This does **not** overwrite the live inference adapter.

### C) Explicit promotion (optional, intentional)

Only after you are satisfied with a finished full run:

```bash
./training/promote_qlora_adapter.sh \
  /gpfs/projects/simswe/$USER/training_outputs/qlora-runs/<courseId>/<runId>-full/adapter
```

This backs up the previous live adapter (if present) and replaces:

```text
/gpfs/projects/simswe/$USER/training_outputs/css-360-qlora/adapter
```

Promotion does **not** start or restart inference.

### D) Inference (separate)

Use the existing fine-tuned inference helpers:

```bash
./training/start_finetuned_service.sh
# on aiswe.uwb.edu:
./scripts/start_finetuned_tunnel.sh <NODE>
```

See `training/inference_service/README.md`.

---

## Environment (Tillicum)

Preferred venv order (smoke / train / compare Slurm):

1. `training/.venv` if present  
2. `/gpfs/projects/simswe/$USER/venvs/qlora`  

Jobs fail immediately if neither exists.

Hugging Face paths used by Slurm:

```bash
export HF_HOME=/gpfs/projects/simswe/$USER/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_TOKEN_PATH=$HF_HOME/token
```

Place a non-empty HF token at `$HF_TOKEN_PATH` before submitting jobs.
Llama 3.2 Instruct requires HF access + license acceptance.

---

## Slurm scripts (advanced)

Normal training must go through:

```bash
./training/start_qlora_training.sh --course <courseId> --smoke|--full
```

That helper always exports a **versioned** `TRAINING_OUTPUT_DIR` under
`.../training_outputs/qlora-runs/...`.

`training/train.slurm` and `training/smoke.slurm` **require** `TRAINING_OUTPUT_DIR`.
Raw `sbatch training/train.slurm` / `sbatch training/smoke.slurm` without it fails
immediately (before training). They also refuse any path under the live tree
`.../training_outputs/css-360-qlora/`.

Only `./training/promote_qlora_adapter.sh` writes the live inference adapter.

Compare (read-only evaluation against an adapter) may still be submitted as:

```bash
sbatch training/compare.slurm
```

Environment variables:

| Variable | Purpose |
| --- | --- |
| `TRAIN_FILE` / `VAL_FILE` | Train/validation JSONL paths |
| `TRAINING_OUTPUT_DIR` | **Required** for smoke/full; versioned output root (`adapter/` underneath) |
| `ADAPTER_PATH` | Compare job adapter to **read** (default: live path) |
| `COMPARISON_OUTPUT_DIR` | Compare job outputs |

---

## Default training settings (unchanged)

- Model: `meta-llama/Llama-3.2-3B-Instruct` (4-bit NF4)
- LoRA: `r=8`, `alpha=16`, `dropout=0.05`, `bias=none`, Llama attn/MLP targets
- Max length 512, LR `2e-4`, 3 epochs, batch 1, grad accum 8 (effective batch 8)
- Warmup ratio 0.1, weight decay 0.01, seed 360
- Eval + save each epoch, bf16 when supported, gradient checkpointing on

Do not change these casually; automation intentionally leaves hyperparameters alone.

---

## Status / logs

```bash
./training/status_qlora_training.sh
squeue -u $USER
ls -lt training/logs/
sacct -j JOB_ID --format=JobID,State,Elapsed,AllocTRES,Start,End
```

Runtime / metrics files live under the job `TRAINING_OUTPUT_DIR`
(`runtime-report.json`, `evaluation_metrics.json`, etc.).

---

## Local helper tests (no GPU)

```bash
python3 -m unittest \
  training.test_qlora_training_helpers \
  training.test_train_qlora_helpers \
  training.test_compare_inference_helpers \
  -v
```

---

## Legacy data docs (not used by current QLoRA training)

The older root-level pipeline in `docs/export-dataset.md`,
`docs/prepare-dataset.md`, and `docs/split-dataset.md`
(`scripts/export_seed_dataset.py` → `prepare_seed_dataset.py` →
`split_training_dataset.py` producing `data/splits/` with a 70/15/15 split)
is **legacy**. Current QLoRA training uses Firebase **approved** seeds via
`backend` export + `prepare_training_split` into `data/exports/<courseId>/`.
