# Per-course QLoRA training (Tillicum)

Fine-tunes **LoRA adapters only** on top of `meta-llama/Llama-3.2-3B-Instruct`,
using the **approved** examples of one course, exported to
`data/exports/<courseId>/`.

Training is per course throughout. Every job, output directory, adapter, and
registered version is keyed by `courseId`; nothing here is specific to any one
course.

This is the **canonical** training workflow. Inference deployment is separate
(see `training/inference_service/README.md`). The full operational reference is
[docs/tillicum-operations.md](../docs/tillicum-operations.md).

Exports are **gitignored**, and no longer need to be synced by hand: the queue
worker downloads a run's prepared dataset from the backend over the credential
it already holds. `scripts/sync_training_data_to_tillicum.sh` is kept as a
debugging tool and is not part of the normal path.

## The short version

```bash
# Tillicum, after SSH + Duo
./training/run_training_queue.sh --once
```

That claims one queued run, fetches its dataset, submits it, and records the
submission. The Slurm job reports its own completion, and a successful full run
registers its model version automatically.

---

## Canonical workflow

### A) Machine with backend / PostgreSQL access (local or UWB VM)

```bash
cd backend
.venv/bin/python scripts/prepare_qlora_dataset.py css-360-winter-2026-a7rp
```

The sync step that used to follow is gone. An administrator can do the same
thing from Admin → Training (**Create training dataset**, then **Prepare
train/validation split**), and the worker fetches the result itself.

`prepare_qlora_dataset.py` calls the existing approved-export + train/validation
split logic (it does not reimplement them). The split now also records SHA-256
checksums of `train.jsonl` and `validation.jsonl` into `manifest.json`, which is
what the cluster verifies each transferred file against.

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

### B2) Tillicum — the training queue

**This is the normal path.** An administrator queues a run from the web
application instead of trying to launch one through a backend. The run is a row
in the PostgreSQL `training_runs` table, keyed `(course_id, run_id)`, and it
waits there. Nothing about it reaches the cluster until someone runs the queue
**in a normal interactive session** — the usual login and two-factor prompt.
Nothing here bypasses that.

The cluster reaches the queue through the backend's `/api/training-queue`
endpoints over outbound HTTPS, authenticated with `TRAINING_WORKER_TOKEN`. It
never holds a database connection.

```bash
cd /gpfs/projects/simswe/$USER/css360-syllabus-bot
./training/run_training_queue.sh --once --dry-run   # read only
./training/run_training_queue.sh --once             # claims one run
```

What it does:

- delivers any completion reports earlier jobs could not send
- re-reports any submission the backend never acknowledged, rather than
  submitting a second job for work already running
- finds the oldest queued (or expired-lease) run and claims exactly one, with a
  time-limited lease, using a conditional write so two runners can never hold
  the same run
- **downloads that run's prepared dataset from the backend**, unless the local
  copy already matches it byte for byte
- validates the course id, the checksums, and the counts using the same helpers
  `start_qlora_training.sh` uses
- submits through `start_qlora_training.sh`, and records the submission locally
  before reporting it

- **reports the result when the job ends** — the Slurm job itself calls back with
  the outcome, and a successful full run registers its model version
  automatically

It never calls `sbatch` itself and never publishes an adapter. `--dry-run` claims
nothing, writes nothing, and downloads nothing.

It needs `TRAINING_API_BASE_URL` and `TRAINING_WORKER_TOKEN` in the environment
or `.env.local`, and outbound HTTPS. Nothing else — in particular it needs no
database credentials and opens no database connection.

The queue lives in PostgreSQL on the UWB VM, in `training_runs`, and this worker
reaches it through the backend's `/api/training-queue` endpoints. A claim is one
POST; the backend selects a single eligible row `FOR UPDATE SKIP LOCKED` and
stamps the lease in the same transaction, so two runners cannot take the same
run. The transaction ends there — training happens long after it has committed,
which is why the claim carries an expiry: a runner that dies mid-job releases
nothing, and the lease expiring is what lets the work be picked up again.

```bash
export TRAINING_API_BASE_URL=https://aiswe.uwb.edu
export TRAINING_WORKER_TOKEN=…   # same value as the backend's
```

### C) Explicit publication (optional, intentional)

Registering a model and serving it are separate decisions. A finished run
registers a version automatically with `status = ready` and
`deployment = offline`; publishing its adapter so a serving session can load it
is this step, and it is deliberate.

```bash
./training/promote_qlora_adapter.sh \
  --course css-350-spring-2026-n3h9 --version v1 \
  /gpfs/projects/simswe/$USER/training_outputs/qlora-runs/<courseId>/<runId>-full/adapter
```

Per course and per version:

```text
/gpfs/projects/simswe/$USER/training_outputs/serving/<courseId>/<version>/adapter
/gpfs/projects/simswe/$USER/training_outputs/serving/<courseId>/current.json
```

The legacy course-agnostic path (`training_outputs/css-360-qlora/adapter`) still
works when `--course` is omitted, but the per-course service does not read it:
that path has no course in it, so publishing one course used to replace whatever
every other course was being served with.

Publication does **not** start or restart inference.

### D) Inference (separate)

```bash
./training/start_finetuned_service.sh
./training/status_finetuned_service.sh
./training/stop_finetuned_service.sh
# on aiswe.uwb.edu:
./scripts/start_finetuned_tunnel.sh --from-backend
```

See `training/inference_service/README.md` and
[docs/tillicum-operations.md](../docs/tillicum-operations.md).

### E) Reclaiming disk (dry run first)

```bash
./training/cleanup_training_outputs.sh            # prints a plan, deletes nothing
./training/cleanup_training_outputs.sh --apply
```

Only checkpoints of completed full runs and whole smoke runs are ever proposed.
Published adapters, registered artifacts, and runs still owing a report to the
application are never touched.

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
`.../training_outputs/qlora-runs/...`, chooses the wall clock from the dataset
size rather than requesting a flat 8 hours, and passes `QUEUE_RUN_ID` into the
job so the job can report its own completion.

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
| `QUEUE_RUN_ID` | PostgreSQL run the job reports its completion against. Empty for a hand-launched job, which simply sends no callback. |
| `QLORA_WALLTIME` | Override the computed `--time` for one submission |
| `QLORA_MAX_WALLTIME` | Raise the 8-hour cap for a genuinely large course |
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

## Recovery and debugging tools

Neither is part of the normal path. Both are kept because they are the way back
when something in the automatic path has gone wrong.

**`scripts/register_course_model.py`** — registers a model version by hand.
A successful run registers its own version through the completion callback; this
exists for an artifact that was produced but never reported, or one an operator
wants recorded deliberately. **Recovery only.**

**`scripts/sync_training_data_to_tillicum.sh`** — copies one course's prepared
export to the cluster over `rsync`. The worker downloads its own dataset from the
backend, so this is not needed normally. **Debugging and recovery only**, for
when the download path itself is what you are investigating.
