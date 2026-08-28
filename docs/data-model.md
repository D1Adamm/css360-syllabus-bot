# Data model

Every piece of application state, where it is stored, and what the fields mean.

Source of truth, in this order: `backend/db/schema.sql` for storage,
`backend/app/db_schemas.py` for the API shapes, `src/types/index.ts` for what the
browser parses. This document describes those; where it disagrees with them, they
are right.

All application state lives in **PostgreSQL**. Every course-scoped table has
`course_id` in its primary key and cascades from `courses`, so deleting a course
removes everything belonging to it. The browser reaches all of it through
FastAPI under `/api/db`; nothing connects to the database directly.

---

## `courses`

One row per course. `course_id` is the partition key for the whole system.

| Column | Type | Notes |
| --- | --- | --- |
| `course_id` | TEXT PK | `^[a-z0-9]+(?:-[a-z0-9]+)*$`, e.g. `css-360-winter-2026-a7rp` |
| `name` | TEXT | Course code as displayed, e.g. `CSS 360` |
| `title` | TEXT | Course title |
| `term` | TEXT | e.g. `Winter 2026` |
| `instructor_name` | TEXT | |
| `created_at` | TIMESTAMPTZ | |
| `syllabus_status` | TEXT | See below |
| `syllabus_file_name` | TEXT NULL | Original upload name; NULL until uploaded |
| `syllabus_type` | TEXT NULL | `pdf` or `txt` |
| `chunk_count` | INTEGER | Chunks in the retrieval index. `>= 0` enforced |

`syllabus_status` values (`SyllabusStatus` in `src/types/index.ts`):
`none`, `not_uploaded`, `uploaded`, `extracted`, `indexed`, `upload_failed`,
`index_failed`, `processing`, `ready`, `error`.

The syllabus **file** is not in the database. See *Filesystem artifacts* below.

---

## `starter_seed_generation`

State of the automatic job that drafts example questions from an uploaded
syllabus. One row per course.

| Column | Type | Notes |
| --- | --- | --- |
| `course_id` | TEXT PK FK | |
| `status` | TEXT NULL | Job lifecycle state |
| `target_count` | INTEGER NULL | How many examples were asked for |
| `final_count` | INTEGER NULL | How many were produced |
| `saved_count` | INTEGER NULL | How many reached `seed_examples` |
| `failed_to_save_count` | INTEGER NULL | |
| `error` | TEXT NULL | |
| `started_at` / `completed_at` | TIMESTAMPTZ NULL | |
| `achievable_ceiling` | INTEGER NULL | Most examples this syllabus could support |
| `limiting_factor` | TEXT NULL | Why a short run was short |

`achievable_ceiling` and `limiting_factor` are operator-facing, not shown in the
UI. Without them, a course whose syllabus only supports eleven examples and
produced eleven is indistinguishable — by count alone — from one whose fact
extractor was silently failing.

---

## `seed_examples`

The example questions: generated, contributed, reviewed, and eventually
exported as training data. Primary key `(course_id, seed_id)`.

**Content**

| Column | Type | Notes |
| --- | --- | --- |
| `seed_id` | TEXT | |
| `instruction` | TEXT | The question |
| `response` | TEXT | The answer |
| `category` | TEXT | |
| `source_section` | TEXT | Syllabus section it came from |
| `difficulty` | TEXT | `Easy`, `Medium`, or `Hard` |
| `directly_answered` | BOOLEAN | Whether the syllabus answers it directly |
| `question_type` | TEXT NULL | |
| `notes` | TEXT NULL | |
| `status` | TEXT NULL | |
| `created_at` | TIMESTAMPTZ NULL | |

**Origin** — `origin` (TEXT). Exactly three values are supported:

| Value | Meaning |
| --- | --- |
| `prototype` | Seeded by hand during early development |
| `user` | Contributed by a student through the Contribute page |
| `ai_generated` | Drafted by the starter-seed job from the syllabus |

**Review** — a professor's decision about one example.

| Column | Type | Notes |
| --- | --- | --- |
| `review_status` | TEXT NULL | `generated`, `approved`, `rejected`, `edited` |
| `review_notes` | TEXT NULL | |
| `reviewed_at` | TIMESTAMPTZ NULL | |

`generated` is the starting state for a validated AI example — reviewed by
nobody yet. Only `approved` and `edited` examples are exported for training.

**Evidence** — what the example is grounded in, so a claim can be traced back to
the syllabus.

| Column | Type | Notes |
| --- | --- | --- |
| `fact_id` | TEXT NULL | Extracted fact this answers |
| `evidence_quote` | TEXT NULL | Verbatim syllabus text supporting it |
| `source_chunk_ids` | JSONB NULL | Retrieval chunks it was drawn from |
| `validation` | JSONB NULL | Automated validation scores and components |

**Edit history** — kept so a professor's correction is visible as a correction.

| Column | Type | Notes |
| --- | --- | --- |
| `original_question` | TEXT NULL | Before the professor edited it |
| `original_answer` | TEXT NULL | |
| `was_edited` | BOOLEAN | Defaults false |

**Deduplication** — `normalized_question_key` (TEXT NULL), indexed per course, so
a near-duplicate question can be found before it is stored twice.

---

## `evaluations`

A student's rating of one comparison. Primary key `(course_id, evaluation_id)`.

| Column | Type | Notes |
| --- | --- | --- |
| `evaluation_id` | TEXT | |
| `comparison_id` | TEXT | Which comparison was rated |
| `most_accurate` | TEXT | One of the four approach keys |
| `most_helpful` | TEXT | |
| `most_concise` | TEXT | |
| `best_grounded` | TEXT | |
| `preferred_model` | TEXT | Overall preference |
| `hallucination_flags` | JSONB | Approaches flagged as hallucinating. Defaults `[]` |
| `comment` | TEXT NULL | Free text |
| `created_at` | TIMESTAMPTZ | |
| `run_id` | TEXT NULL | The live comparison run, when there was one |
| `question_text` | TEXT NULL | Denormalised so results survive without the run |

Approach keys are `base`, `rag`, `fineTuned`, `fineTunedRag` (`ModelKey`).

No student identifier is recorded. There is no authentication, so nothing
collects one.

---

## `course_models` and `course_model_versions`

The per-course model registry.

**`course_models`** — one row per course.

| Column | Type | Notes |
| --- | --- | --- |
| `course_id` | TEXT PK FK | |
| `current_version` | TEXT | The newest registered version. What a professor is shown |

**`course_model_versions`** — one row per version. Primary key
`(course_id, version)`, which is what makes cross-course contamination
structurally impossible.

| Column | Type | Notes |
| --- | --- | --- |
| `version` | TEXT | `v1`, `v2`, … monotonic per course |
| `base_model` | TEXT | e.g. `meta-llama/Llama-3.2-3B-Instruct` |
| `training_example_count` | INTEGER | **The train split**, not the approved total |
| `status` | TEXT | `ready`, `training`, `failed` |
| `deployment` | TEXT | `online`, `offline`, `unknown` |
| `artifact_ref` | TEXT | Relative reference; never an absolute cluster path |
| `created_at` / `updated_at` | TIMESTAMPTZ | |
| `notes` | TEXT NULL | |
| `run_id` | TEXT NULL | The training run it came from. NULL for hand-registered versions |
| `provenance` | JSONB NULL | Full traceability — see below |

`UNIQUE (course_id, run_id) WHERE run_id IS NOT NULL` is what makes a repeated
completion callback idempotent instead of a source of `v2`, `v3`, …

**`status` versus `deployment`** — different questions, never merged:

- `status = 'ready'` means a usable adapter exists somewhere.
- `deployment = 'online'` means the adapter is in the cluster's serving tree.
  **This is what inference resolves.** A newly registered version is `ready` and
  `offline` until an operator publishes it deliberately.
- Whether a GPU is running *right now* is `serving_sessions`, not this column.

`training_example_count` is the number of examples actually passed into
training. For a course with 42 approved examples split 37/5, it is 37; the
approved and validation counts are in `provenance`. Versions registered by hand
before automatic registration existed may hold the approved count instead — those
rows are historical and are deliberately not rewritten.

`provenance` records, for one artifact: base model, dataset reference and
checksums, approved/train/validation counts, resolved training configuration,
optimizer steps intended and completed, `trainingLengthSatisfied`, losses,
measured GPU hours, Slurm job id, and git commit.

---

## `model_requests`

The professor-facing "I would like a course model" lifecycle. One row per course.

| Column | Type | Notes |
| --- | --- | --- |
| `course_id` | TEXT PK FK | |
| `status` | TEXT | `requested`, `preparing`, `training`, `ready`, `failed` |
| `requested_at` / `updated_at` | TIMESTAMPTZ | |
| `approved_example_count` | INTEGER | Approved examples at the moment of the request |
| `failure_message` | TEXT NULL | Professor-facing. Set only when failed |
| `preparation` | JSONB NULL | Dataset prep result: counts, `datasetRef`, split seed |
| `preparation_error` | TEXT NULL | Admin-only |
| `training` | JSONB NULL | Submitted job: id, mode, counts, `datasetRef` |
| `launch_error` | TEXT NULL | Admin-only |
| `current_run_id` | TEXT NULL | The run this request currently tracks |

`requested`, `preparing`, and `training` are active and block a second request;
`ready` and `failed` are terminal, so a failed request never locks a course out.

`current_run_id` is not bookkeeping. Every cluster callback is checked against
it — a report from a run the request no longer points at is refused, which is
what stops a late callback from a retired run moving state out from under its
replacement.

`preparation` and `training` stay whole in JSONB because they are admin-only
detail whose shape is still moving, and neither is ever queried by field.

---

## `training_runs`

The durable queue the cluster claims work from. Primary key
`(course_id, run_id)`.

| Column | Type | Notes |
| --- | --- | --- |
| `run_id` | TEXT | `run-<utc stamp>-<random>` |
| `mode` | TEXT | `smoke` or `full` |
| `state` | TEXT | See below |
| `enqueued_at` / `updated_at` | TIMESTAMPTZ | |
| `dataset_ref` | TEXT | Relative, e.g. `exports/{courseId}` |
| `approved_example_count` | INTEGER | |
| `train_examples` / `validation_examples` | INTEGER | |
| `attempt` | INTEGER | Incremented on every claim, including after an expired lease |
| `job_id` | TEXT NULL | Real Slurm id. Never invented; a run carrying one is never re-submitted |
| `claim_owner` | TEXT NULL | Who holds the lease |
| `claim_claimed_at` / `claim_expires_at` | TIMESTAMPTZ NULL | |
| `error` | TEXT NULL | Operator-facing |
| `completion` | JSONB NULL | What the cluster reported when the job ended |

States: `queued` → `claimed` → `submitted` → `training` → `succeeded` | `failed`.
`succeeded` and `failed` are terminal, and outstanding work in any other state
blocks a second run for the course.

A run retired by an admin retry stays `failed` with the reason
`Superseded by admin retry` rather than getting a new state — adding a state
would make older browser bundles drop it from the history the feature exists to
preserve.

The three claim columns are stored flat because a lease gets queried (who holds
this, has it expired) but are nested as `claim` in the API. A claim missing any
of the three is reported as no claim at all: a lease nobody can reason about is
not a lease.

`completion` holds optimizer steps completed against intended, losses, measured
GPU hours, git commit, dataset digests, artifact location, and failure stage.
Operator-facing, never queried by field.

---

## `serving_sessions`

Whether a GPU is serving fine-tuned inference, and until when.

| Column | Type | Notes |
| --- | --- | --- |
| `session_id` | TEXT PK | |
| `job_id` | TEXT | Slurm job holding the allocation |
| `node` | TEXT | Compute hostname. Changes every job |
| `port` | INTEGER | `0 < port < 65536` enforced |
| `state` | TEXT | `starting`, `ready`, `stopped`, `expired` |
| `started_at` / `expires_at` / `updated_at` | TIMESTAMPTZ | |
| `detail` | JSONB NULL | Published courses and versions this session can answer for |

`node` and `port` are returned to the cluster worker but **not** to the browser:
`/api/db/serving-session` omits them, so a public page cannot learn a compute
hostname.

---

## Filesystem artifacts

Not in the database, and not reproducible from it.

| Path | Machine | Contents |
| --- | --- | --- |
| `backend/course_data/{courseId}/original.(pdf\|txt)` | UWB VM | The uploaded file |
| `backend/course_data/{courseId}/syllabus.txt` | UWB VM | Extracted text |
| `backend/data/indexes/{courseId}.json` | UWB VM | Embedding index |
| `backend/data/indexes/{courseId}.facts.json` | UWB VM | Extracted-fact cache |
| `data/exports/{courseId}/` | UWB VM | `train.jsonl`, `validation.jsonl`, `manifest.json` |
| `training_outputs/qlora-runs/{courseId}/{run}-{mode}/` | Tillicum | One training run |
| `training_outputs/serving/{courseId}/{version}/adapter/` | Tillicum | A published adapter |
| `training/state/` | Tillicum | Run↔job↔output mapping, undelivered reports |

`manifest.json` carries SHA-256 checksums of `train.jsonl` and
`validation.jsonl`. The cluster verifies each transferred file against them
before letting it replace anything.

---

## API record shapes

The API is camelCase; storage is snake_case. `backend/app/db_mapping.py` and the
`db_*.py` repositories do the translation, and the mapping stops there — the
relational split across `course_models` and `course_model_versions`, for
instance, is not visible in the API, which returns
`{currentVersion, versions: {v1: {...}}}`.

Optional fields are **omitted** rather than sent as null, matching the parsers in
`src/lib/`. A field that cannot be parsed causes the record to be dropped rather
than half-read.

---

## History: the Firebase migration

This application previously stored course data in Firebase Realtime Database, at
paths like `courses/{courseId}/metadata`. **It does not any more.** PostgreSQL is
the only live database, and no current record is stored at a Firebase-style path.

Two files remain, deliberately, so the migration can be replayed or audited:

- `backend/app/firebase_snapshot.py` — parses an exported snapshot JSON file. It
  opens no connection.
- `backend/scripts/import_firebase_snapshot.py` — the one-time importer.

Nothing in the running system reaches either. Archived snapshots live under
`backups/firebase/`, which is gitignored and never deleted automatically.
