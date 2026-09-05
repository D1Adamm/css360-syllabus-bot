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

An administrator queues a run from the Admin → Training page, using whichever
of the two controls applies:

| Course state | Control | What it does |
| --- | --- | --- |
| Request `preparing`, data prepared, no active run | **Queue training** | The first run for a course. |
| Request `ready` or `failed`, no active run | **Train new version** | Another run for a course that already finished one. Every earlier run and registered version is kept. |
| A run stuck in `submitted`/`training` and silent for 6 h | **Retry training** | Disaster recovery. Retires the stuck run and queues a replacement. |

**Train new version** and **Retry training** are not interchangeable. Retry
*retires* the run a course is waiting on — using it on a run that succeeded
would rewrite a good result as `failed` to get a side effect. Retraining
supersedes nothing.

Both reuse the dataset already prepared for the course. Nothing is re-exported
and no split is recomputed unless an administrator explicitly runs **Rebuild
dataset** and **Prepare training split** first.

Then:

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
./training/start_finetuned_service.sh    # one hour, the most the debug QOS allows
```

Then, on the application VM:

```bash
ssh <you>@aiswe.uwb.edu                  # UW password + Duo, by hand
cd ~/css360-syllabus-bot
./scripts/start_finetuned_tunnel.sh --from-backend
```

`--from-backend` looks the compute node up from the session Tillicum recorded,
instead of the operator reading a hostname off one machine and typing it into
another. It reads `TRAINING_API_BASE_URL` and `TRAINING_WORKER_TOKEN` from
`backend/.env` — the file the backend service already loads — so no `source` or
export is needed; a variable already in the environment takes precedence.

**Session length is bounded by the QOS, not by preference.** `serve.slurm` runs
under `debug`, which caps a job at one hour, and that is the default. Asking for
more is refused before submission with the reason — Slurm would otherwise accept
a two-hour job and leave it `PENDING` forever with
`QOSMaxWallDurationPerJobLimit`, which looks like a busy cluster rather than a
request that can never be satisfied. That cost a session. For a longer sitting,
submit under a QOS that permits it:

```bash
SERVICE_QOS=normal ./training/start_finetuned_service.sh --hours 3
```

Re-running the command extends nothing: an active session is reused. Start a
fresh one after it expires.

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
  --course <courseId> --version <version> \
  /gpfs/projects/simswe/$USER/training_outputs/qlora-runs/<courseId>/<run>-full/adapter
```

A version is written once. Re-publishing over an existing version is refused,
because a registered model version refers to that directory by reference and
replacing it in place would make the record describe something else.

### Registered is not published, and inference follows published

| Fact | Where | Meaning |
| --- | --- | --- |
| `status = ready` | `course_model_versions` | A usable adapter exists somewhere. |
| `deployment = online` | `course_model_versions` | This version is in the cluster's serving tree. **Inference resolves this one.** |
| `current_version` | `course_models` | The newest registered version — what a professor is shown. |
| `current.json` | `<SERVING_ROOT>/<courseId>/` | The cluster's own record of the same publication, used when a request arrives without a version. |

Training a new version makes it `ready` and moves `current_version`. It does not
move what answers questions, because the cluster does not have the new adapter
until somebody puts it there. Publishing is what changes the answer, and
`promote_qlora_adapter.sh` reports it to the backend **after** the copy has
landed and been validated.

Without that split, training a new version took the old one offline: the backend
started asking for `v2`, the cluster only had `v1`, and every fine-tuned request
for that course failed until `v2` was published.

A course that has never had a publication reported falls back to
`current_version`, which is how every course from before this reporting keeps
answering exactly as it did.

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

## When a node is unhealthy

GPU device failures happen. In one session a node repeatedly failed its
preflight:

```text
Failed to get device handle for GPU 0
nvidia-smi: No devices were found
```

The workflow handled it correctly on its own — the job failed at preflight, the
failure callback marked the run `failed` with `failureStage = preflight`, the
model request went `failed`, and **Retry training** queued a replacement that
trained successfully. Nothing was lost and no history was rewritten.

What the workaround exposed is that an operator sometimes needs to keep one
submission off one node *right now*:

```bash
./training/start_finetuned_service.sh --exclude-node g018
```
```bash
./training/run_training_queue.sh --once --exclude-node g018
```

Both accept the flag repeatedly, or a comma-separated list. It is passed
straight through as `sbatch --exclude=`.

**This is temporary troubleshooting and nothing else.** No node is named
anywhere in this repository, the default is no exclusions, and nothing is
remembered between runs — Hyak repairs nodes, and Slurm has to be free to
schedule one the moment it is healthy again. A test in
`training/test_finetuned_deploy_helpers.py` fails if a node name is ever
hardcoded into an `--exclude=`.

To default it for one shell only:

```bash
export TRAINING_EXCLUDE_NODES=g018     # training
export SERVICE_EXCLUDE_NODES=g018      # serving
```

If a node is failing repeatedly, report it to Hyak. Excluding it is a way to get
through the afternoon, not a fix.

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

## Deploying a change

Deployment of the application and the cluster checkout is documented separately:
**[deployment.md](deployment.md)**. The Tillicum half in brief:

```bash
cd /gpfs/projects/simswe/$USER/css360-syllabus-bot && git pull origin main
```
```bash
mkdir -p /gpfs/projects/simswe/$USER/training_outputs/serving
```
```bash
chmod 600 .env.local
```
```bash
./training/run_training_queue.sh --once --dry-run
```

### Compute-node connectivity: observed working, fallback still required

This was an open question. It is not any more.

A real training job finished on a compute node and reported its own completion
straight to `aiswe.uwb.edu` — the run went `succeeded`, the model request went
`ready`, and a new version registered, with **no file left in
`training/state/pending/`**. Direct callbacks work. (Details:
[verification-history.md](verification-history.md).)

That is one observation on one node, not a property of the cluster. Nodes,
routing and firewall policy differ and change, and the backend can be down for
reasons that have nothing to do with the compute node. **The persist-to-GPFS
fallback stays, permanently.** Every report is written to
`training/state/pending/` *before* the send is attempted, so the difference
between the two cases is only how quickly the application finds out.

To check it again on some other node:

```bash
srun --account=simswe --qos=debug --time=00:05:00 --pty curl -sS -o /dev/null -w '%{http_code}\n' --max-time 10 https://aiswe.uwb.edu/api/health
```

`200` means callbacks land the moment a job ends. Anything else means they wait
for the next `./training/run_training_queue.sh --once`. No code changes either
way — this is diagnostic only.

---

## Verifying a change, in two stages

Deliberately two stages. Running training and inference together for the first
time means a failure could be in either, and the two have completely different
causes.

Concrete evidence from the production run that first exercised all of this is in
[verification-history.md](verification-history.md).

### Stage A — serve a model the course already has

If a course already has a registered version, per-course serving can be proven
against it without spending another GPU allocation.

On Tillicum, publish the existing version:

```bash
./training/promote_qlora_adapter.sh --course <courseId> --version <version> --run-id <runId> /gpfs/projects/simswe/$USER/training_outputs/qlora-runs/<courseId>/<run>-full/adapter
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

Ask that course a fine-tuned question and check which model answered:

```bash
curl -sS -X POST http://127.0.0.1:8001/api/fine-tuned/generate -H 'Content-Type: application/json' -d '{"courseId":"<courseId>","question":"When does the course meet?"}'
```

Expect the response to echo `"courseId": "<courseId>"` and the version you
published.

Then prove isolation — a course with **no** published adapter must not quietly
receive another course's:

```bash
curl -sS -X POST http://127.0.0.1:8001/api/fine-tuned/generate -H 'Content-Type: application/json' -d '{"courseId":"<other-courseId>","question":"When does the course meet?"}'
```

Expect a **409** naming that course. Not an answer, and not the first course's
adapter.

Stop when finished:

```bash
./scripts/stop_finetuned_tunnel.sh
```
```bash
./training/stop_finetuned_service.sh
```

### Stage B — the automated training lifecycle

Only after Stage A works, and only once the course's published version is
recorded as `deployment: online`:

```bash
curl -s http://127.0.0.1:8001/api/db/courses/<courseId>/model
```

If it still says `offline`, re-run the Stage A publish command — it is idempotent
— so inference has a published version to hold on to while the new one trains.

1. Admin → Training: **Prepare training data**, then **Queue training** (or
   **Train new version** for a course that has already finished a run).
2. Tillicum: `./training/run_training_queue.sh --once --dry-run` — expect the
   train/validation counts, either `already matches` or `would download`, and a
   dataset-derived wall clock.
3. Tillicum: `./training/run_training_queue.sh --once` — expect a Slurm job id
   and `trainingRun.state=submitted`. No rsync, no second Duo prompt.
4. `squeue -u $USER` — the job's `TIME_LIMIT` reflects the dataset size, not a
   flat 8 hours.
5. When it finishes, Admin → **Training jobs**: `succeeded`, full optimizer-step
   count, measured GPU hours, and a registered model version.

**The new version is the next unused one**, not `v1`, for any course that already
has a version. Nothing overwrites an existing version.

**And the previously published version keeps serving** while the new one is
`ready` / `offline`. Check it:

```bash
curl -sS -X POST http://127.0.0.1:8001/api/fine-tuned/generate -H 'Content-Type: application/json' -d '{"courseId":"<courseId>","question":"When does the course meet?"}'
```

It still names the published version even though the registry's current version
has moved. Then publish the new one deliberately:

```bash
./training/promote_qlora_adapter.sh --course <courseId> --version <newVersion> --run-id <newRunId> /gpfs/projects/simswe/$USER/training_outputs/qlora-runs/<courseId>/<new-run>-full/adapter
```

Re-run the same question. It now names the new version, and the old one remains
registered and `ready` with `deployment: offline`.

Neither `sync_training_data_to_tillicum.sh` nor `register_course_model.py` is run
at any point.

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
