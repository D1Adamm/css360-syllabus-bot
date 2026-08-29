# Syllabus Model Lab

A teaching and research project at UW Bothell (School of STEM) for comparing how
four different AI approaches answer questions about a course syllabus.

Students ask one question and see four answers side by side, then rate them. The
ratings are the research data: they show whether retrieval, fine-tuning, or both
actually help a model answer questions about a specific course.

Not an official University of Washington service.

---

## The four approaches

Internally `base`, `rag`, `fineTuned`, and `fineTunedRag`. What everyone sees:

| Shown in the UI | What it is | Runs on |
| --- | --- | --- |
| Base | A base model with no course context at all | Local Ollama |
| RAG | Retrieval: syllabus passages are found and put in the prompt | Local Ollama |
| Fine-Tuned | A LoRA adapter fine-tuned on this course's approved examples | Tillicum GPU |
| Fine-Tuned + RAG | The fine-tuned adapter, prompted with retrieved passages | Tillicum GPU |

All four are real. Nothing is simulated.

Base and RAG share one CPU-bound local model process, so they are issued
**sequentially**; the two fine-tuned paths use a separate GPU service and
overlap with them. That ordering is load-bearing and covered by tests.

---

## Roles

Selected with a **development-only switcher** in the header. It is not
authentication and grants nothing — every route is reachable by URL. It exists so
the application can be walked through as each audience before sign-in is built.

**Students** read the syllabus, contribute example questions, compare four
answers, and rate them. They see no infrastructure: no service names, no storage
paths, no dataset internals.

**Professors** create courses, upload a syllabus, review and approve the example
questions generated for their course, request a course model, and read aggregate
results. Course management, not ML operations.

**Admins** get the technical surface: service health, per-course diagnostics, the
full dataset with validation detail, dataset preparation, the training queue, and
the model registry. Implementation detail appears here and nowhere else.

---

## Architecture

```
Browser (React 19 + TypeScript + Vite)
  │
  │  HTTPS — every path under /api
  ▼
FastAPI  ───────────────►  PostgreSQL          system of record for all
  │                                            application state
  │
  ├──►  Ollama (local)                         Base + RAG generation
  │       llama3.2:3b, nomic-embed-text        and embeddings
  │
  ├──►  local disk                             uploaded syllabi, extracted text,
  │       backend/course_data/                 per-course embedding indexes
  │       backend/data/indexes/
  │
  └──►  Tillicum GPU cluster
          ├─ fine-tuned inference    via an SSH tunnel opened by hand
          └─ training queue          via authenticated outbound HTTPS
                                     from the cluster back to FastAPI
```

**On the UWB VM:** Nginx, the React build, FastAPI, PostgreSQL, Ollama, and the
course artifacts on local disk.

**On Tillicum:** QLoRA fine-tuning and fine-tuned inference. Both need a GPU that
the VM does not have.

**Why Tillicum needs a person.** Opening the tunnel from the VM to a Tillicum
compute node authenticates to UW, and UW two-factor is deliberately not
automated, stored, or worked around. So a fine-tuned session starts with someone
running one command in a session they logged into normally. Everything after that
authentication is automatic. Base and RAG do not depend on any of it.

Deeper detail: **[docs/architecture.md](docs/architecture.md)**.

---

## Where data lives

Every table is course-scoped by `courseId` and reached only through FastAPI. The
browser never talks to PostgreSQL, and the cluster never holds a database
connection.

| Table | Holds |
| --- | --- |
| `courses` | Course metadata and syllabus status |
| `starter_seed_generation` | State of the automatic starter-seed job per course |
| `seed_examples` | Example questions, their review state, and their evidence |
| `evaluations` | Student ratings of the four answers |
| `course_models` / `course_model_versions` | The per-course model registry |
| `model_requests` | The professor-facing "I want a model" lifecycle |
| `training_runs` | The queue the cluster claims work from, and what each run reported |
| `serving_sessions` | Whether a GPU is serving fine-tuned inference, and until when |

Full field-level reference: **[docs/data-model.md](docs/data-model.md)**.

Not in the database — written to local disk by the backend:

| Path | Purpose |
| --- | --- |
| `backend/course_data/{courseId}/original.(pdf\|txt)` | The uploaded file |
| `backend/course_data/{courseId}/syllabus.txt` | Extracted text |
| `backend/data/indexes/{courseId}.json` | That course's embedding index |
| `data/exports/{courseId}/` | Prepared train/validation split for training |

---

## Routes

`/professor/reviews` and `/professor/models` are **redirects** to
`/professor/courses`, not pages — the cross-course hubs were removed from
navigation and their URLs kept so existing links resolve. Old URLs
(`/course/:courseId/*`, `/compare`, `/architecture`, …) redirect too, preserving
query strings; see `src/app/LegacyRedirects.tsx`.

```
/                                        role landing

/student                                 course list
/student/course/:courseId                home
/student/course/:courseId/syllabus       read the syllabus
/student/course/:courseId/contribute     add an example question
/student/course/:courseId/compare        ask and compare four answers
/student/course/:courseId/evaluate       rate the answers just generated

/professor/courses                       course list
/professor/courses/new                   create a course
/professor/course/:courseId              overview
/professor/course/:courseId/syllabus     syllabus
/professor/course/:courseId/examples     review queue
/professor/course/:courseId/model        course model status, request a model
/professor/course/:courseId/results      aggregate results
/professor/course/:courseId/invite       invite students (not yet implemented)

/admin                                   service health
/admin/courses                           technical course list
/admin/courses/:courseId                 course diagnostics
/admin/courses/:courseId/examples        full dataset + export
/admin/training                          dataset prep, training queue, job status
/admin/models                            model registry and published versions
/admin/system                            architecture reference

/styleguide                              design-system reference (dev only)
```

---

## Prerequisites

| Requirement | Needed for | Notes |
| --- | --- | --- |
| Node.js 20+ and npm | Frontend | |
| Python 3.11+ | Backend | 3.13 is what the checked-in venv uses |
| PostgreSQL 14+ | Everything | The backend refuses to start a request without it |
| Ollama | Base + RAG | Not needed if you only work on the UI |

**Optional, and not required for local Base/RAG development:**

- **Tillicum access** — only for fine-tuned inference and training. Without it,
  Base and RAG work normally and the two fine-tuned paths report that no
  service is configured.
- **`TRAINING_WORKER_TOKEN`** — only for the training queue API. Leave it unset
  and that router refuses every request with 503, which is the correct behaviour
  for an unconfigured deployment.
- **`qwen3:8b` / `qwen3:4b`** — only for starter-seed generation, which writes
  example questions from an uploaded syllabus. Base and RAG do not use them.

---

## Quick start from a fresh clone

### 1. Frontend dependencies

```bash
npm install
```

### 2. Python environment

```bash
python3 -m venv backend/.venv
```
```bash
backend/.venv/bin/pip install -r backend/requirements.txt
```

### 3. PostgreSQL database and schema

```bash
createdb syllabus_bot
```
```bash
psql syllabus_bot -v ON_ERROR_STOP=1 -f backend/db/schema.sql
```

`schema.sql` is idempotent (`CREATE TABLE IF NOT EXISTS` throughout), so it is
safe to re-run. It creates the current schema in full, including everything the
migrations in `backend/db/migrations/` add — those exist for upgrading a database
that already has data, not for a fresh one.

### 4. Backend configuration

```bash
cp backend/.env.example backend/.env
```

Then set at least `DATABASE_URL`, for example
`postgresql://localhost:5432/syllabus_bot`. Every other value has a working
default. See [backend/README.md](backend/README.md) for what each one does.

### 5. Frontend configuration

```bash
cp .env.example .env.local
```

The one value that matters locally:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8001/api
```

**The `/api` suffix is required.** The frontend clients write paths *below* it —
`dbApi.ts` composes `/db/courses`, `api.ts` composes `/courses/{id}/seeds` — and
only 6 of the backend's 58 routes have a root-level alias. Without the suffix,
every persistence request 404s.

### 6. Ollama

```bash
ollama serve
```
```bash
ollama pull llama3.2:3b && ollama pull nomic-embed-text
```

### 7. Run it

```bash
backend/.venv/bin/uvicorn app.main:app --reload --host 127.0.0.1 --port 8001 --app-dir backend
```
```bash
npm run dev
```

### 8. Check it

```bash
curl -s http://127.0.0.1:8001/api/health
```

Expect `{"status":"ok","service":"syllabus-model-lab-backend"}`. Then open the
Vite URL, create a course, and upload a syllabus.

---

## Tests

```bash
npm test
```
```bash
npm run lint
```
```bash
npm run build
```
```bash
backend/.venv/bin/python -m pytest backend/tests -q
```
```bash
backend/.venv/bin/python -m pytest training scripts -q
```

Run the backend suite and the training suite **separately**, or with
`backend/tests` first. `training/inference_service/app.py` shadows the backend
`app` package if the cluster paths are collected first.

No test needs a GPU, a database, a network, or Ollama. The backend suite fails
closed if a test tries to reach any of them — see
`backend/tests/test_test_isolation.py` for why that barrier exists.

---

## Project structure

```
.
├── backend/
│   ├── app/                  # FastAPI routes, repositories, RAG, seed generation
│   ├── db/schema.sql         # Full current schema; migrations/ upgrades old ones
│   ├── scripts/              # Operational and one-off maintenance scripts
│   └── tests/
├── docs/                     # Architecture, data model, operations, roadmap
├── scripts/
│   ├── lib/                  # Shared stdlib-only helpers for cluster scripts
│   ├── register_course_model.py       # manual registration (recovery only)
│   ├── report_model_published.py      # publication reporting
│   └── *_finetuned_tunnel.sh          # UWB VM side of the inference tunnel
├── src/                      # React application
│   ├── app/                  # Route tree and legacy redirects
│   ├── components/ui/        # Design-system primitives
│   ├── lib/                  # API clients, error mapping, route builders
│   ├── pages/{student,professor,admin}/
│   └── types/
└── training/                 # Everything that runs on Tillicum
    ├── inference_service/    # Per-course fine-tuned serving
    ├── run_training_queue.sh # The one command an operator runs
    └── *.slurm               # Job scripts
```

---

## Current limitations

- **No authentication or access control.** Anyone with a link can open any
  course, in any role.
- **No enrolment**, join codes, or class rosters. The invite page is a
  placeholder.
- **Fine-tuned inference needs a GPU session started by hand**, because opening
  the tunnel authenticates to UW and two-factor is not automated.
- **Syllabus artifacts and indexes are local disk only**, so the backend is not
  horizontally scalable as written.
- The archived Firebase snapshot is retained deliberately; nothing deletes it.

What is genuinely unfinished, and what is deliberately out of scope:
**[docs/remaining-work.md](docs/remaining-work.md)**.

---

## Privacy and ethics

- Contributed questions and evaluations are stored in PostgreSQL, scoped by
  `courseId`.
- **The application does not store student identity.** There is no
  authentication, so it never collects a name, email, or account id to store.
  This is a statement about the application database only — it is not a claim
  about Nginx access logs, systemd journals, or anything else the host records.
- Avoid entering sensitive personal information into contributed questions or
  evaluation notes. Nothing redacts them.
- Model outputs can be wrong. Use the syllabus and instructor judgment as the
  authority, not an answer from any of the four approaches.
- Evaluation ratings are subjective classroom observations, not ground-truth
  labels.

---

## Where to go next

| Document | For |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | How a request flows, and how courses stay isolated |
| [docs/data-model.md](docs/data-model.md) | Every table and API record, field by field |
| [docs/tillicum-operations.md](docs/tillicum-operations.md) | Operator runbook: training, serving, retries, secrets |
| [docs/deployment.md](docs/deployment.md) | Deploying to the UWB VM |
| [docs/remaining-work.md](docs/remaining-work.md) | What is actually unfinished |
| [docs/verification-history.md](docs/verification-history.md) | What has been proven against production |
| [backend/README.md](backend/README.md) | Backend setup and API groups |
| [training/README.md](training/README.md) | QLoRA training on Tillicum |
| [training/inference_service/README.md](training/inference_service/README.md) | The per-course inference service |

## History

This application previously stored course data in Firebase Realtime Database. It
no longer does: PostgreSQL is the only live database, and neither the frontend,
the backend, nor the training worker needs Firebase configuration to start. The
historical snapshot reader (`backend/app/firebase_snapshot.py`) and the one-time
importer (`backend/scripts/import_firebase_snapshot.py`) are kept so the
migration can be replayed or audited. Nothing in the running system reaches them.
