# Verification history

Concrete evidence that the training and serving workflow works end to end, from
the production run that first exercised all of it.

This is a **historical record**, not a runbook. It names specific courses, runs,
Slurm jobs, and nodes because that is what makes it evidence. For instructions
to follow, use [tillicum-operations.md](tillicum-operations.md), which is written
with placeholders.

The value of keeping it: the next person to change any of this should know which
properties were observed against real hardware rather than merely designed and
unit-tested.

---

## Serving, and course isolation

- A course with a published `v1` answered from its own adapter, and the response
  echoed back its own `courseId` and `modelVersion`.
- A second course, with **no** published adapter, received a **409** — not an
  answer, and not the first course's adapter. Course isolation holds at the
  serving boundary, not only in the database.

## Automatic dataset transfer

- The cluster worker downloaded the prepared dataset from the UWB backend over
  its existing worker token. No `rsync`, and no second interactive
  authentication.
- Integrity checking confirmed **42 approved / 37 train / 5 validation** against
  the checksums in `manifest.json`.

## Dataset-derived Slurm wall clock

- The run was submitted with a **1-hour** wall clock derived from the dataset
  size, rather than the flat 8 hours that had been requested previously for a
  run that took under a minute.

## Infrastructure failure, reported automatically

- A GPU preflight failure occurred on a node whose device had gone:

  ```text
  Failed to get device handle for GPU 0
  nvidia-smi: No devices were found
  ```

- The failure callback reached UWB on its own: the training run went `failed`,
  the model request went `failed`, and `failureStage` was recorded as
  `preflight`. Nothing had to be noticed by a person first.
- **Retry training** queued a replacement run. The retired run kept its state,
  its job id, and its place in the course's history — no history was rewritten to
  produce the replacement.

## Successful training and completion callback

- Slurm job **265323** completed **15/15** optimizer steps with
  `trainingLengthSatisfied = true`.
- It ran on compute node **g002** and delivered its completion callback
  **directly** to `aiswe.uwb.edu`, leaving nothing in `training/state/pending/`.

  This is why the documentation says compute-node outbound HTTPS is *observed
  working* rather than *unknown*. It is one node on one day, not a property of
  the cluster — the persist-to-GPFS fallback remains mandatory.

## Automatic model registration

- `v2` registered automatically from the completion callback.
  `scripts/register_course_model.py` was not run.
- The existing `v1` was untouched: still registered, still `ready`, and still the
  published version.

## Registered is not published

- With `currentVersion = v2`, `v2` `ready` / `offline`, and `v1`
  `ready` / `online`, inference continued resolving **`v1`**.
- This is the case that had previously been an outage: before the fix, the
  backend asked the cluster for `v2`, the cluster held only `v1`, and every
  fine-tuned request for the course failed.

## Publication switches serving without a restart

- Publishing `v2` moved it to `online` and `v1` to `offline`.
- The **already-running** inference service returned `modelVersion = v2` on the
  next request. No GPU job restart, no re-allocation.

---

## VM-local fine-tuned serving, 2026-09-22

Observed on `aiswe` after the tracked `aiswe-finetuned` unit replaced the
hand-written one (checkout `265a64d` plus the installer fix that followed).

- The unit was migrated live: `install`, three `set-mapping` calls, `check`,
  `restart`. Port 9001 changed owner from the old process to the new one with
  no tunnel involved; lingering was already on.
- `scripts/verify_finetuned_production.py` passed **28 of 28** checks as an
  administrator: unit enabled and active, a python process on 9001, all three
  mapped Ollama models present, `/health` ok with every mapped version
  servable, direct `/generate` for CSS 350 v2, CSS 360 v2 and CSS 360 v3 with
  the requested course and version echoed, and through the backend
  Fine-Tuned, Fine-Tuned + RAG, Base and RAG for both courses.
- The published adapters are the run artifacts: on Tillicum,
  `serving/css-350-spring-2026-n3h9/v2` hashes `773761a1…08fd`, identical to
  run `20260827T205338Z-full`, and `serving/css-360-winter-2026-a7rp/v2`
  hashes `8466d16e…7d59`, identical to run `20260905T040428Z-full`; both
  `current.json` pointers name those runs.
- `scripts/finetuned_latency_probe.py` at its defaults (20 requests, both
  courses, both fine-tuned modes, concurrency 1 and 2; `FINETUNED_KEEP_ALIVE`
  30m) recorded, in seconds:

  | Course | Mode | First | c=1 median / p95 | c=2 median / p95 / max |
  | --- | --- | --- | --- | --- |
  | CSS 350 | Fine-Tuned | 7.1 | 2.1 / 2.3 | 2.9 / 3.8 / 3.8 |
  | CSS 350 | Fine-Tuned + RAG | 12.4 | 3.8 / 3.9 | 6.1 / 8.0 / 8.0 |
  | CSS 360 | Fine-Tuned | 7.2 | 1.9 / 1.9 | 2.8 / 3.8 / 3.8 |
  | CSS 360 | Fine-Tuned + RAG | 10.6 | 4.0 / 4.3 | 7.0 / 8.7 / 8.7 |

  No failures, no timeouts. The first request of each cell pays the model
  load; concurrency 2 roughly doubles the median, which is the serialised
  Ollama path queueing as expected. Two students asking at once wait under
  ten seconds for a grounded answer.
- **Reboot.** `sudo reboot` at 12:03 PDT with nobody logged in afterward
  until the check itself. Ollama (system unit) was active; `aiswe-backend`
  entered active at 12:03:34 and `aiswe-finetuned` at 12:03:38, from
  lingering alone; port 9001 was owned by the new python process (pid 2737)
  with no `ssh` forward; `verify_finetuned_production.py` passed 28 of 28
  again. The first CSS 350 generation after boot took 25 s (model load), the
  rest 7 to 14 s. No Tillicum allocation, tunnel, Duo prompt or laptop was
  involved at any point. `journalctl --user` reports no journal on this VM,
  so the unit's preflight lines are not readable after the fact;
  `aiswe_finetuned.sh logs` falls back to the system journal's view.

## What was not run

Neither `scripts/sync_training_data_to_tillicum.sh` nor
`scripts/register_course_model.py` was used at any point. Both remain in the
repository as recovery and debugging tools; neither is part of the normal path.
