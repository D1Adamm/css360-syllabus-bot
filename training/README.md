# CSS 360 QLoRA training (Tillicum)

This scaffold fine-tunes **LoRA adapters only** on top of
`meta-llama/Llama-3.2-3B-Instruct` using the approved train/validation JSONL split:

- `data/exports/css-360-winter-2026-a7rp/train.jsonl` (48 examples)
- `data/exports/css-360-winter-2026-a7rp/validation.jsonl` (6 examples)
- `data/exports/css-360-winter-2026-a7rp/manifest.json`

It does **not** change the backend/frontend export or split workflow. Inference is
not integrated yet.

## 1. Create the Python environment (on Tillicum)

From the repo root on a Tillicum login node:

```bash
python3 -m venv /gpfs/projects/simswe/$USER/venvs/css360-qlora
source /gpfs/projects/simswe/$USER/venvs/css360-qlora/bin/activate
python -m pip install --upgrade pip
pip install -r training/requirements.txt
```

Install a CUDA-enabled PyTorch build that matches the Tillicum GPU/driver stack if
the default `torch` wheel is CPU-only. Prefer the official PyTorch install
instructions for your CUDA version; do not commit CUDA wheel URLs into this repo.

Optional local venv for development:

```bash
python3 -m venv training/.venv
source training/.venv/bin/activate
pip install -r training/requirements.txt
```

## 2. Hugging Face authentication

Llama 3.2 Instruct requires Hugging Face access + acceptance of the model license.

```bash
huggingface-cli login
# or:
export HF_TOKEN=hf_...
```

Confirm you can access `meta-llama/Llama-3.2-3B-Instruct` before submitting jobs.

## 3. Cache and output paths

Slurm scripts set:

```bash
export HF_HOME=/gpfs/projects/simswe/$USER/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
```

Outputs go under:

```text
/gpfs/projects/simswe/$USER/training_outputs/
```

Keep model weights, adapters, caches, and logs **off git**.

## 4. Submit the smoke job first

From the repository root:

```bash
mkdir -p training/logs
sbatch training/smoke.slurm
```

Smoke mode:

- uses 4 train + 2 validation examples
- runs `max_steps=3`
- still saves a LoRA adapter
- measures completed step timing
- prints an **approximate** full-run duration and GPU-hour estimate

## 5. Check queue status

```bash
squeue -u $USER
```

## 6. Check logs

```bash
ls -lt training/logs/
tail -n 100 training/logs/smoke-<JOB_ID>.out
tail -n 100 training/logs/smoke-<JOB_ID>.err
```

## 7. Read `runtime-report.json`

Smoke/full outputs include `runtime-report.json` in the job output directory, for example:

```bash
cat /gpfs/projects/simswe/$USER/training_outputs/css-360-qlora-smoke/runtime-report.json
```

Important fields:

| Field | Meaning |
| --- | --- |
| `completedSteps` | Measured optimizer/training steps completed |
| `averageSecondsPerStep` | Conservative measured average (prefers excluding first-step warmup when available) |
| `estimatedOptimizerSteps` | Expected full-run steps from 48 examples × epochs × batching |
| `estimatedTrainingOnlySeconds` | Steady-state training estimate (no one-time download) |
| `estimatedConservativeTotalSeconds` | Adds model load + eval + checkpoint overhead |
| `estimatedGpuHours` | Conservative seconds ÷ 3600 × GPU count |
| `actualGpuHours` | Present after **full** runs: total elapsed hours × GPU count |
| `slurmJobId` / `gitCommitSha` | Provenance when available |

## 8. Interpreting the smoke estimate

The smoke printout looks like:

```text
Smoke benchmark:
  - Completed steps: 3
  - Training time: ...
  - Average seconds per step: ...
  - Estimated full optimizer steps: ...
  - Estimated training-only duration: ...
  - Conservative estimated total duration: ...
  - Requested GPUs: 1
  - Estimated GPU hours: ...
```

Treat this as **approximate**. First-step / warmup overhead is identified when possible.
If no steps complete, estimates are reported as `n/a` (no division by zero).

## 9. Compare with Slurm elapsed time after completion

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,AllocTRES,Start,End
seff JOB_ID
```

## 10. Submit the full job only after smoke success

```bash
sbatch training/train.slurm
```

Full training uses the complete 48/6 split and writes under:

```text
/gpfs/projects/simswe/$USER/training_outputs/css-360-qlora/
```

After completion, inspect:

```bash
cat /gpfs/projects/simswe/$USER/training_outputs/css-360-qlora/runtime-report.json
sacct -j JOB_ID --format=JobID,State,Elapsed,AllocTRES,Start,End
```

## Default training settings

- Model: `meta-llama/Llama-3.2-3B-Instruct` (4-bit NF4)
- LoRA: `r=8`, `alpha=16`, `dropout=0.05`, `bias=none`, Llama attn/MLP targets
- Max length 512, LR `2e-4`, 3 epochs, batch 1, grad accum 8
- Warmup ratio 0.1, weight decay 0.01, seed 360
- Eval + save each epoch, bf16 when supported, gradient checkpointing on

## Local helper tests (no GPU / no model download)

```bash
cd training
python -m unittest test_train_qlora_helpers.py -v
```
