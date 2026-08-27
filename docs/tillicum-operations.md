# Tillicum operations

The whole cluster-side workflow, in the order an operator meets it.

Two facts shape everything here. Tillicum is reached by an interactive login
with UW two-factor, and nothing in this repository automates, stores, or works
around that. And Tillicum is not on the application database's network, so
everything the cluster does to application state it does over HTTPS to
`aiswe.uwb.edu`, authenticated with a shared worker token that reaches the queue
endpoints and nothing else.

Everything after the login is meant to be one command.

---

## Training: the normal flow

An administrator queues a run from the Admin → Training page. Then:

```bash
ssh $USER@tillicum.hyak.uw.edu           # UW password + Duo, by hand
cd /gpfs/projects/simswe/$USER/css360-syllabus-bot
git pull origin main
./training/run_training_queue.sh --once
```

That one command:

1. delivers any completion reports earlier jobs could not send
2. re-reports any submission the backend never acknowledged
3. claims one queued run
4. **downloads that run's prepared dataset from the backend**, unless the copy
   on disk already matches it byte for byte
5. verifies checksums, counts, and the training configuration
6. refuses to submit a second job for a run that already has one
7. submits through `start_qlora_training.sh`
8. records the submission — locally first, then to the backend

The Slurm job then reports its own completion, and a successful full run
registers its model version automatically. Nothing else is required.

To see what would happen without touching anything:

```bash
./training/run_training_queue.sh --once --dry-run
```

`--dry-run` claims nothing, writes nothing, downloads nothing, and never spawns
the launcher. It does ask the backend to *describe* the dataset, which is the
difference between "there is a queued run" and "there is a queued run that could
actually be trained right now".

### What is no longer needed

The `rsync` step is gone. `scripts/sync_training_data_to_tillicum.sh` still
works and is kept as a debugging tool, but the normal path does not use it: the
worker fetches the dataset over the credential it already holds, in the same
direction as every other call it makes.

Manual `register_course_model.py` is gone from the normal path too. It remains
the recovery tool for registering an artifact by hand.

---

## Inference: the normal flow

Before a class or a demo:

```bash
ssh $USER@tillicum.hyak.uw.edu           # UW password + Duo, by hand
cd /gpfs/projects/simswe/$USER/css360-syllabus-bot
./training/start_finetuned_service.sh    # 2 hours by default; --hours 3 for longer
```

Then, on the application VM:

```bash
ssh <you>@aiswe.uwb.edu                  # UW password + Duo, by hand
cd ~/css360-syllabus-bot
./scripts/start_finetuned_tunnel.sh --from-backend
```

`--from-backend` looks the compute node up from the session Tillicum recorded,
instead of the operator reading a hostname off one machine and typing it into
another.

Status and stop:

```bash
# Tillicum
./training/status_finetuned_service.sh
./training/stop_finetuned_service.sh

# UWB VM
./scripts/status_finetuned_tunnel.sh
./scripts/stop_finetuned_tunnel.sh
```

### The one step that is still manual, and why

The tunnel is opened **from** the UWB VM **to** Tillicum, and opening it
authenticates to UW. Duo is not automated, bypassed, or stored anywhere, so that
SSH is a person at a keyboard.

What has been removed is everything around it: the operator no longer has to
discover a hostname, keep it accurate across job restarts, or edit
`backend/.env` by hand. The remaining action is one command that prompts for the
authentication it needs.

---

## Per-course serving

Training is per course, so serving is too.

```text
<SERVING_ROOT>/<courseId>/<version>/adapter/   the PEFT adapter
<SERVING_ROOT>/<courseId>/current.json         which version is current
```

Publishing an adapter for a course:

```bash
./training/promote_qlora_adapter.sh \
  --course css-350-spring-2026-n3h9 --version v1 \
  /gpfs/projects/simswe/$USER/training_outputs/qlora-runs/css-350-spring-2026-n3h9/<run>-full/adapter
```

A version is written once. Re-publishing over an existing version is refused,
because a registered model version refers to that directory by reference and
replacing it in place would make the record describe something else.

The service loads the base model once and attaches one adapter per course on top
of it, choosing per request. A LoRA adapter here is ~47 MB against a ~2.5 GB
4-bit base, so a second course costs a rounding error of GPU memory rather than
a second allocation. `MAX_LOADED_ADAPTERS` (default 4) bounds how many stay
resident.

**Adapter format.** Training writes `adapter_config.json` and
`adapter_model.safetensors` via PEFT's `save_pretrained`. That is exactly what
`PeftModel.load_adapter` reads. There is no conversion step, and none is needed
— no GGUF, no merged checkpoint.

**Course isolation** is enforced in four places:

1. Every `/generate` request names its course. There is no default.
2. The adapter path is built from a validated course id and version.
3. Adapter selection and generation happen together under one lock, because
   FastAPI runs sync handlers concurrently and an interleaving would answer one
   course's question with another's weights.
4. The response echoes the course it used, and the backend discards a response
   whose course does not match what it asked for.

---

## Slurm resource policy

The CSS 350 full run requested 8 hours and used 48 seconds. An 8-hour request
queues behind everything the scheduler can fit in front of it, so the habit cost
wall-clock latency on every run of every size.

Wall clock is now computed from the dataset:

```text
steps   = ceil(train_examples / effective_batch) * epochs
seconds = 30 minutes overhead + 20 seconds per step
request = clamp(seconds, 1 hour, 8 hours)
```

The per-step constant is about nine times the measured 2.3 s/step. The overhead
term covers what does not scale with data: a cold Hugging Face download, a slow
shared filesystem, a busy node, per-epoch checkpoint writes.

| train examples | optimizer steps | requested |
| -------------- | --------------- | --------- |
| 37 (CSS 350)   | 15              | 01:00:00  |
| 200            | 75              | 01:00:00  |
| 800            | 300             | 02:00:00  |
| 2000           | 750             | 04:40:00  |
| 5000           | 1875            | 08:00:00 (capped) |

Smoke runs keep a fixed `00:45:00` under the debug QOS: they execute three
optimizer steps, so their cost is overhead and nothing else.

A run pinned at the ceiling says so in the launcher output. Raise it
deliberately for a genuinely large course:

```bash
QLORA_MAX_WALLTIME=16:00:00 ./training/run_training_queue.sh --once
```

The runtime estimator and `runtime-report.json` are unchanged; this policy
decides what to *request*, and the report still records what was *used*.

---

## Outage and retry behaviour

Every cluster→backend operation is built to survive the backend being briefly
unreachable, because it has been.

| When it fails | What happens |
| --- | --- |
| Backend down during claim | Nothing was claimed. Re-run the command. |
| `sbatch` succeeded, `/submitted` lost | The local run record has the job id. The next run re-reports it and never submits a second job. |
| Backend down when training ends | The completion is on disk under `training/state/pending/`. The next run delivers it. |
| Response lost after the backend committed | The retried callback finds the version the first delivery created and reuses it. No v2. |
| Duplicate callback | Idempotent, by run id, enforced by a unique index as well as by a read. |
| Operator re-runs the worker | Safe. A run with a job id is never submitted again. |
| Login session drops mid-job | The Slurm job is unaffected. Its completion is reported by the job itself. |
| Run was superseded by an admin retry | The callback is refused with 409 and dropped locally, so it is not replayed forever. |

The cluster's own record lives in `training/state/`:

```text
training/state/runs/<runId>.json          runId -> Slurm jobId -> output dir
training/state/pending/<runId>-completed.json   not yet accepted by UWB
```

Both are gitignored and machine-local. Losing them costs reconciliation
information, not data.

---

## Output retention

Every full run writes per-epoch checkpoints. A checkpoint of a quantised 3B
model with optimizer state dwarfs the 47 MB adapter anyone actually wants, so
after a term most of `training_outputs/` is checkpoints for adapters that
already exist.

```bash
./training/cleanup_training_outputs.sh            # print the plan, delete nothing
./training/cleanup_training_outputs.sh --apply    # act on it, after typing DELETE
```

Dry run by default. It will only ever propose:

- `<run>/checkpoints/` from **completed** full runs
- whole **smoke** run directories

It will never propose, whatever else is true:

- anything under `serving/` — the published adapters inference loads
- any `adapter/` directory, or `adapter-backups/`
- a run a published `current.json` says a served adapter came from
- a run the cluster still owes the application a report for
- a run with no `runtime-report.json` — still going, or died

Each path is re-checked against those rules immediately before deletion rather
than trusted from the printed plan.

Nothing is deleted automatically, and nothing in this pass deleted anything.

---

## Secrets

| Secret | Where it lives | Notes |
| --- | --- | --- |
| `TRAINING_WORKER_TOKEN` | `backend/.env` on the VM, `.env.local` on Tillicum | Shared secret. Reaches the queue endpoints only; it is not a database credential and cannot be used as one. |
| Hugging Face token | `/gpfs/projects/simswe/$USER/huggingface/token` | Never in the repository. |
| UW password, Duo | Nowhere | Never stored, never automated. |
| SSH keys | `~/.ssh`, untouched | No script in this repository reads or writes them. |

### What a compute node can and cannot read

Stated precisely, because two different things are easy to conflate.

**The token does not travel through the scheduler.** `start_qlora_training.sh`
submits with `env -u TRAINING_WORKER_TOKEN sbatch …`, so it is absent from the
environment Slurm propagates to the job.

**The training job can still read it, and does.** The completion callback is
authenticated — `/completed` requires the worker token like every other queue
route — so the job needs it. It obtains it at runtime by reading `.env.local`
from the shared project filesystem, through the same `load_env_file` the login
node uses.

So the protection is the **file mode**, not the absence of the value. That is
acceptable for this Option-A baseline: anyone who can read `.env.local` on
`/gpfs/projects/simswe` already has the cluster account that can submit jobs as
you. But it has to be stated as what it is, and the file has to be locked down:

```bash
chmod 600 /gpfs/projects/simswe/$USER/css360-syllabus-bot/.env.local
```

The worker warns on stderr when that file is group- or world-readable, naming
the path and the mode and never the value.

**When compute nodes have no outbound HTTPS**, nothing breaks and no design
changes. `report_training_result.py` writes its payload to
`training/state/pending/` on GPFS *before* attempting to send, exits 0 when the
send fails, and the next `./training/run_training_queue.sh --once` on the login
node delivers it. The token is still required for that delivery; it is simply
used from the login node instead.

The dataset-download endpoint serves `manifest.json` verbatim, and that file
records absolute paths on the VM. Those paths were already going to the cluster
in the `rsync` this replaced, the route requires the worker token, and no
browser-facing route serves it.

To rotate the token:

1. `openssl rand -hex 32`
2. set `TRAINING_WORKER_TOKEN` in `backend/.env` on the VM
3. restart `aiswe-backend`
4. set the same value in `.env.local` on Tillicum
5. verify with `./training/run_training_queue.sh --once --dry-run`

Between steps 3 and 4 the worker gets 401s and does nothing else. No state is
lost; a run stays queued.

---

## Deploying this

Nothing here is run automatically. `main` is the deployed branch: review on a
feature branch, merge, then pull `main` on both hosts.

### 1. Local

```bash
git checkout -b tillicum-workflow-hardening
git add -A
git commit -m "Automate the Tillicum training and serving workflow"
git push -u origin tillicum-workflow-hardening
```

Review, then merge into `main` and push it:

```bash
git checkout main
git merge --no-ff tillicum-workflow-hardening
git push origin main
```

### 2. UWB VM

```bash
cd ~/css360-syllabus-bot
git pull origin main
```

Load the database URL from the file the backend already uses:

```bash
set -a
source backend/.env
set +a
```

Apply the migration (idempotent; safe to re-run):

```bash
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f backend/db/migrations/001_training_provenance_and_serving.sql
```

Rebuild and publish the frontend:

```bash
npm ci
npm run build
sudo cp -a dist/. /usr/share/nginx/html/
sudo restorecon -R /usr/share/nginx/html
sudo nginx -t
sudo systemctl reload nginx
```

Restart the backend and check it:

```bash
systemctl --user restart aiswe-backend
curl -s http://127.0.0.1:8001/api/health
```

The backend test suite is now safe to run here — that is one of the things this
change fixes:

```bash
backend/.venv/bin/python -m pytest backend/tests -q
```

### 3. Tillicum

```bash
cd /gpfs/projects/simswe/$USER/css360-syllabus-bot
git pull origin main
mkdir -p /gpfs/projects/simswe/$USER/training_outputs/serving
chmod 600 .env.local
./training/run_training_queue.sh --once --dry-run
```

### Optional: does a compute node reach UWB directly?

Read-only, five minutes of a debug allocation, and it changes nothing either
way — the persist-to-GPFS fallback already covers "no".

```bash
srun --account=simswe --qos=debug --time=00:05:00 --pty \
  curl -sS -o /dev/null -w '%{http_code}\n' --max-time 10 https://aiswe.uwb.edu/api/health
```

`200` means completion callbacks land the moment a job ends. Anything else means
they are written to `training/state/pending/` and delivered by the next
`./training/run_training_queue.sh --once`. No code changes on either answer.

---

## Verifying it, in two stages

Deliberately two stages. Running training and inference together for the first
time means a failure could be in either, and the two have completely different
causes.

### Stage A — serve the model CSS 350 already has

CSS 350 already has `v1` (run `run-20260827t064701z-1cf650`, Slurm job 264787,
`status = ready`, `deployment = offline`). Per-course serving can be proven
against it without spending another GPU allocation.

On Tillicum:

```bash
./training/promote_qlora_adapter.sh \
  --course css-350-spring-2026-n3h9 \
  --version v1 \
  --run-id run-20260827t064701z-1cf650 \
  /gpfs/projects/simswe/madamk/training_outputs/qlora-runs/css-350-spring-2026-n3h9/20260827T064810Z-full/adapter
```

```bash
./training/start_finetuned_service.sh
```

```bash
./training/status_finetuned_service.sh
```

On the UWB VM:

```bash
./scripts/start_finetuned_tunnel.sh --from-backend
```

```bash
curl -s http://127.0.0.1:8001/api/fine-tuned/health
```

Ask CSS 350 a fine-tuned question and check the answer names the right model:

```bash
curl -sS -X POST http://127.0.0.1:8001/api/fine-tuned/generate \
  -H 'Content-Type: application/json' \
  -d '{"courseId":"css-350-spring-2026-n3h9","question":"When does the course meet?"}'
```

Expect `"courseId": "css-350-spring-2026-n3h9"` and `"modelVersion": "v1"`.

Then prove isolation — a course with no published adapter must not quietly get
CSS 350's:

```bash
curl -sS -X POST http://127.0.0.1:8001/api/fine-tuned/generate \
  -H 'Content-Type: application/json' \
  -d '{"courseId":"css-360-winter-2026-a7rp","question":"When does the course meet?"}'
```

Expect a **409** naming CSS 360 — not an answer, and not CSS 350's adapter.

Stop when finished:

```bash
./scripts/stop_finetuned_tunnel.sh          # UWB VM
./training/stop_finetuned_service.sh        # Tillicum
```

### Stage B — the automated training lifecycle

Only after Stage A works.

1. Admin → Training: **Prepare training data** for CSS 350, then **Queue
   training**.
2. Tillicum: `./training/run_training_queue.sh --once --dry-run` — expect
   `37 train / 5 validation`, either `already matches` or `would download`, and
   `Wall clock: 01:00:00`.
3. Tillicum: `./training/run_training_queue.sh --once` — expect a job id and
   `trainingRun.state=submitted`. No rsync, no second Duo prompt.
4. `squeue -u $USER` — the job's `TIME_LIMIT` is `1:00:00`, not `8:00:00`.
5. When it finishes, Admin → **Training jobs**: state `succeeded`, `15/15`
   optimizer steps, measured GPU hours, and a registered model version.

**The new version is `v2`, not `v1`.** CSS 350's `v1` already exists, and
registration allocates the next unused version for a run it has never seen
before. Nothing overwrites `v1`, and its published adapter keeps serving until
`v2` is published deliberately.

Neither `sync_training_data_to_tillicum.sh` nor `register_course_model.py` is
run at any point.

---

## Test isolation

Running the backend tests on the VM must not reach the production database.
`env -u DATABASE_URL pytest` was **not** enough — `app.db` reloaded
`backend/.env` precisely because the variable was missing, and a route test
using a real course id rewrote that course's row.

The barrier is now structural. Under `APP_ENV=test`, or whenever pytest is
imported at all:

- `backend/.env` is not read
- `DATABASE_URL` is ignored even when set
- only `TEST_DATABASE_URL` can supply a DSN, and nothing in a deployment sets it

So this is safe on the VM:

```bash
cd ~/css360-syllabus-bot && backend/.venv/bin/python -m pytest backend/tests -q
```

and so is the form that caused the incident. See
`backend/tests/test_test_isolation.py`.
