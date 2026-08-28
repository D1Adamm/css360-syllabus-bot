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

## What was not run

Neither `scripts/sync_training_data_to_tillicum.sh` nor
`scripts/register_course_model.py` was used at any point. Both remain in the
repository as recovery and debugging tools; neither is part of the normal path.
