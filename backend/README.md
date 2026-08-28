# Backend

FastAPI application: course and syllabus management, retrieval, seed generation,
the model registry, and the training queue the Tillicum cluster claims work from.

PostgreSQL is the system of record. The browser never reaches it directly, and
neither does the cluster.

For the whole-system picture see [../docs/architecture.md](../docs/architecture.md);
for setting up a fresh clone see the [root README](../README.md).

---

## Setup

```bash
python3 -m venv .venv
```
```bash
.venv/bin/pip install -r requirements.txt
```
```bash
ollama pull llama3.2:3b && ollama pull nomic-embed-text
```

Create the database and apply the schema:

```bash
createdb syllabus_bot
```
```bash
psql syllabus_bot -v ON_ERROR_STOP=1 -f db/schema.sql
```

`db/schema.sql` is idempotent and creates the current schema in full, including
everything `db/migrations/` adds. Migrations exist to upgrade a database that
already has data; a fresh one does not need them.

---

## Configuration

```bash
cp .env.example .env
```

Loaded automatically on startup by `app/config.py`.

| Variable | Required | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | **yes** | The only database the application uses |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | yes | Base and RAG generation |
| `STARTER_EMBED_MODEL` | yes | Embeddings for retrieval |
| `CORS_ALLOWED_ORIGINS` | yes | Origins the browser may call from |
| `SEED_GENERATION_MODEL`, `STARTER_*` | for starter seeds | The job that drafts examples from a syllabus |
| `TRAINING_WORKER_TOKEN` | for training | Shared secret for `/api/training-queue`. **Unset ⇒ that router refuses every request with 503**, which is deliberate: an unconfigured deployment must not be an unauthenticated queue |
| `FINETUNED_SERVICE_URL` | for fine-tuned paths | Set by the tunnel script to `http://127.0.0.1:9001`. Unset ⇒ those two approaches report unavailable; Base and RAG are unaffected |
| `APP_ENV` | no | `production` on the VM. `test` disables env-file loading entirely |

Never set `TEST_DATABASE_URL` on a deployed host. It is the only DSN a test
process may connect to.

---

## Run

```bash
.venv/bin/uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```
```bash
curl -s http://127.0.0.1:8001/api/health
```

---

## API groups

58 routes. Every one is under `/api`, except six root-level aliases
(`/health`, `/base-model/generate`, `/rag/generate`, `/fine-tuned/generate`,
`/fine-tuned/health`, `/fine-tuned-rag/generate`) kept because Nginx forwards
only `location /api/` and those are useful directly on the VM.

| Group | Prefix | Purpose |
| --- | --- | --- |
| Health | `/api/health` | Reports that the API is responding. Does not probe Ollama or the database |
| Inference | `/api/base-model/…`, `/api/rag/…`, `/api/fine-tuned/…`, `/api/fine-tuned-rag/…` | The four comparison approaches. All require `courseId` |
| Syllabus | `/api/courses/{courseId}/syllabus…`, `/chunks` | Upload, extract, chunk, embed; read extracted text and chunk metadata |
| Seeds | `/api/courses/{courseId}/seeds…` | Generation, validation, review, quality checks, approved export, train/validation split |
| Persistence | `/api/db/…` | `db_routes.py`. Courses, seeds, evaluations, model registry, model requests, training runs, serving session. What the browser reads and writes |
| Training queue | `/api/training-queue/…` | `training_queue_routes.py`. **The cluster's API, not the browser's.** Claim a run, download its dataset, report submission/failure/completion, register a version, report a publication, record a serving session. Authenticated with `X-Training-Worker-Token` |

The two authenticated surfaces are deliberately separate. The worker token
reaches the queue endpoints and nothing else — it is not a database credential
and cannot be used as one — so adding browser authentication later does not
disturb the cluster's.

---

## Local artifacts

Written to disk, not to the database, and not reproducible from it:

| Path | Contents |
| --- | --- |
| `course_data/{courseId}/original.(pdf\|txt)` | The uploaded file |
| `course_data/{courseId}/syllabus.txt` | Extracted text |
| `data/indexes/{courseId}.json` | That course's embedding index |
| `data/indexes/{courseId}.facts.json` | Extracted-fact cache |
| `../data/exports/{courseId}/` | Prepared train/validation split, fetched by the cluster |

All gitignored. Because one process owns them, the backend cannot currently run
as more than one instance.

`../docs/syllabus.txt` is a fixture for chunking unit tests only (`app/rag.py`
reads it). No live route uses a fixed index.

---

## Tests

```bash
.venv/bin/python -m pytest tests -q
```

No test needs a database, a network, Ollama, or a GPU, and the suite fails closed
if one tries to reach them:

- **HTTP** to any non-local host raises, naming the URL.
- **PostgreSQL** — under pytest, `backend/.env` is not read at all and
  `DATABASE_URL` is ignored even when set. Only `TEST_DATABASE_URL` can supply a
  DSN, and nothing in a deployment sets it.
- **`TRAINING_WORKER_TOKEN`** is removed, so the queue router refuses with 503
  unless a test configures one deliberately.

That barrier exists because it was once absent: `env -u DATABASE_URL pytest`
reached the production database anyway, because `app.db` reloaded `backend/.env`
precisely when the variable was missing. See `tests/test_test_isolation.py`.

Run the backend suite separately from `training/` and `scripts/`, or first —
`training/inference_service/app.py` shadows the `app` package otherwise.

---

## Scripts

| Script | Purpose |
| --- | --- |
| `scripts/prepare_qlora_dataset.py` | Export approved seeds and prepare the train/validation split for one course |
| `scripts/export_approved_seeds.py` | Approved-only JSONL export |
| `scripts/reconcile_starter_generation.py` | Repair starter-seed job state after an interrupted run |
| `scripts/run_fact_inventory.py` | Diagnostic: what facts were extracted from a syllabus |
| `scripts/phase7_verify.py` | Diagnostic: multi-course inventory and allocation check |
| `scripts/benchmark_seed_models.py` | Compare candidate generation models |
| `scripts/import_firebase_snapshot.py` | **Historical.** One-time import from the pre-PostgreSQL Firebase snapshot. Nothing in the running system calls it |

The diagnostics write JSON and log files beside themselves; those outputs are
gitignored and have no readers other than a person.
