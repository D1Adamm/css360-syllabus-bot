# Architecture

What the system does today, and why it is shaped this way. For setup see the
root [README](../README.md); for operating the cluster side see
[tillicum-operations.md](tillicum-operations.md).

---

## The whole system

```
Browser (React 19 + TypeScript + Vite, built by Vite, served by Nginx)
  │
  │  every request path is under /api
  ▼
FastAPI (uvicorn, 127.0.0.1:8001)
  │   every browser request carries session cookies; every route names its guard
  │
  ├──► PostgreSQL              system of record for all application state,
  │                            sessions and identity included
  │
  ├──► Ollama (localhost)      Base generation, RAG generation, embeddings
  │
  ├──► local disk              uploaded syllabi, extracted text, embedding
  │                            indexes, prepared training datasets
  │
  └──► Tillicum
        ├─ fine-tuned inference   FastAPI → http://127.0.0.1:9001 → SSH tunnel
        │                         → compute node :8001 → base model + adapter
        └─ training queue         cluster → outbound HTTPS → /api/training-queue
```

Two boundaries do the structural work:

**The browser never reaches PostgreSQL.** Every read and write goes through
FastAPI, which is what makes course scoping enforceable in one place instead of
in every component.

**The cluster never holds a database connection.** Tillicum is not on the
database's network, and exposing PostgreSQL beyond the VM to make a queue work
would be a bad trade. So the cluster makes outbound HTTPS calls to
`/api/training-queue`, authenticated with a shared worker token that reaches
those endpoints and nothing else. It is not a database credential and cannot be
used as one.

---

## Course isolation

`courseId` is the partition key for the entire system. It matches
`^[a-z0-9]+(?:-[a-z0-9]+)*$` and is validated at every boundary it crosses —
`assert_valid_course_id` in the backend, `assertValidCourseId` in the frontend,
`validate_course_id` in the cluster scripts. The rule is restated rather than
shared because the three run in different languages on different machines, and
each one is the last line of defence where it sits.

Isolation is mostly *structural* rather than checked:

- Every course-scoped table has `course_id` in its primary key.
- `course_models.course_id` is a primary key, so a version registered for one
  course cannot land on another — there is no shape of the request that writes a
  different course's row.
- Artifact and index paths are built from the validated id, so a traversal
  attempt is a name that fails validation, not a path that escapes.
- Fine-tuned requests carry the course, the adapter is resolved from it, and the
  response echoes which course answered. The backend discards a response naming a
  different course.

---

## Identity and access

Three kinds of caller, two of them with sessions in the browser and one with a
header token from the cluster:

| Caller | Identity | Where it lives | Reaches |
| --- | --- | --- | --- |
| Student | An anonymous **participant**: a random UUID bound to one course, created when a classroom code is redeemed. No name, email or NetID exists to store | `sml_participant` cookie → `auth_sessions` → `participants` | The student pages and APIs of that one course |
| Professor | A `users` row with global role `professor`, plus `course_memberships` for each course they instruct | `sml_staff` cookie → `auth_sessions` → `users` | Course-scoped staff routes for member courses only |
| Administrator | A `users` row with global role `admin`. No memberships needed | same cookie | Every course, `/api/admin`, the training queue's browser side |
| Tillicum runner | `TRAINING_WORKER_TOKEN` header | `backend/.env`, `.env.local` | `/api/training-queue` and nothing else |

**Role and membership are separate.** `users.role` says what kind of staff
someone is; `course_memberships` says which courses a professor may touch. A
professor with no memberships sees no courses. A professor never gains a course
by editing a URL, a body or an id: the guard compares the course in the *path*
against the membership table on every request.

**Two cookies, on purpose.** A professor can open their own course's class code
in the same browser, walk the student flow as an anonymous participant, and
return to the professor pages without signing out. Their test ratings land under
a participant like any student's.

**Sessions are rows.** The cookie holds a 256-bit random token; `auth_sessions`
holds its SHA-256, the principal, an absolute deadline, and `last_seen_at` for
the idle rule. Sign-out, disabling an account, and changing a password revoke
rows, so they take effect on the next request. Cookies are `HttpOnly; Secure;
SameSite=Lax; Path=/api`.

**Invitations are rows too.** `invitations` holds four kinds: a reusable
`student` classroom code (six characters from an unambiguous alphabet, stored as
typed because it has to be shown again and grants only a seat in the class), and
single-use `professor`, `admin` and `reset` links whose 256-bit tokens are stored
only as hashes and shown exactly once. Redemption is one conditional `UPDATE …
RETURNING`, so two people racing a single-use link cannot both win. Every
create, revoke and staff acceptance writes `admin_actions`.

**Guards are declared, and checked for.** Every route in `main.py`,
`db_routes.py`, `admin_routes.py` and `student_invite_routes.py` takes its guard
as a handler parameter (`require_admin`, `require_course_staff`,
`require_course_access`, `require_participant`, …; `app/auth/dependencies.py`),
so the on-VM root aliases inherit it. `tests/route_classification.py` names every
route and its class; `test_route_auth_coverage.py` fails the suite for any route
mounted without a class or without the guard its class requires, and
`test_authorization_matrix.py` drives every route with six kinds of principal.
The four generation routes take the course from the body and authorize it
explicitly before any model is asked anything.

**What a student never sees.** A participant lists only their own ratings, and of
the examples only the instructor-approved ones plus their own contributions, with
review notes, validation and evidence stripped. Their home page shows counts
from a counts-only endpoint. Attribution comes from the session, never from a
request body; the evaluation id is allocated server-side.

**CSRF.** `SameSite=Lax` plus a required custom header on every
cookie-authenticated mutation, plus an `Origin` check against the site's
origins. Login, join and invitation acceptance require the header too.

**The frontend guards are navigation.** `src/components/auth/RouteGuards.tsx`
sends a browser to sign-in, the join page or an explanation based on the cached
session; the backend makes the same decision again for every request the page
issues, and the backend's answer is the one that counts.

---

## Request flows

### Base and RAG (local, no GPU)

```
POST /api/base-model/generate     → Ollama llama3.2:3b, no context
POST /api/rag/generate            → retrieve from backend/data/indexes/{courseId}.json
                                    → prompt Ollama with the passages
```

Both hit the same CPU-bound Ollama process, so the Compare page issues them
**sequentially**. Running them concurrently makes both slower and can time out;
the ordering is covered by tests.

Retrieval is course-local: the index is built from that course's uploaded
syllabus at upload time and lives beside it on disk.

### Fine-Tuned and Fine-Tuned + RAG (Tillicum)

```
POST /api/fine-tuned/generate     { courseId, question }
POST /api/fine-tuned-rag/generate { courseId, question }  ← retrieves first, then this
   │
   ├─ resolve which version answers for this course   (PostgreSQL)
   ├─ POST FINETUNED_SERVICE_URL/generate { courseId, modelVersion, question }
   │     → SSH tunnel → compute node
   │     → base model (loaded once) + that course's LoRA adapter
   └─ verify the response names the course that was asked for
```

`FINETUNED_SERVICE_URL` points at `http://127.0.0.1:9001` on the VM, which is the
local end of an SSH tunnel to whichever compute node currently holds the GPU
allocation. Compute hostnames change every job, so nothing hardcodes one — the
cluster records the session and the tunnel script looks it up.

The two fine-tuned paths use a separate service from Base and RAG, so they
overlap with them rather than queueing behind them.

---

## Registered, published, served

Three different facts about a model version. Conflating any two of them caused a
real outage, so they are kept apart deliberately.

| Fact | Where it lives | Means |
| --- | --- | --- |
| **Registered / ready** | `course_model_versions.status = 'ready'` | A usable adapter exists somewhere |
| **Published** | `course_model_versions.deployment = 'online'` | The adapter is in the cluster's serving tree — **this is what inference resolves** |
| **Actively served** | `serving_sessions` | A GPU session is running right now |
| **Newest** | `course_models.current_version` | The most recent registered version — what a professor is shown |

A successful training run registers a new version and moves `current_version` to
it. It does **not** change what answers questions, because the adapter is not on
the cluster until somebody publishes it. Publishing is a deliberate operator
action that reports itself back to the backend after the copy has landed and been
validated; only then does inference switch.

Resolution therefore prefers the published version, falling back to
`current_version` only for a course that has never had a publication reported —
which keeps courses from before publication reporting answering exactly as they
did.

A GPU session ending does not unpublish anything. The adapter is still on the
filesystem and the next session loads it.

---

## The training lifecycle

```
professor requests a model              model_requests: requested
admin prepares the dataset              → data/exports/{courseId}/ (+ manifest checksums)
admin queues a run                      training_runs: queued
operator runs one command on Tillicum   claimed → dataset downloaded and verified
                                        → submitted (Slurm job id recorded)
Slurm trains
the job reports its own completion      → succeeded, model version registered
                                        → model_requests: ready
operator publishes deliberately         → deployment: online; inference switches
```

Properties that matter:

- **The dataset travels over the same credential the worker already holds.** It
  is fetched from a protected backend endpoint, verified against manifest
  checksums, and written atomically. There is no `rsync` step in the normal path.
- **Every report is persisted to the shared filesystem before it is sent.** A
  process killed between deciding to report and finishing the attempt is
  indistinguishable from one that never tried; only the file on disk makes the
  difference recoverable. The next queue run delivers it.
- **Callbacks are idempotent, keyed by run.** A redelivered completion finds the
  version its first delivery created rather than allocating a new one — enforced
  by a unique index as well as by a read.
- **A superseded run cannot report.** If an admin retried a run, a late callback
  from the retired one is refused with 409 rather than moving state out from
  under its replacement.
- **A run that already has a Slurm job id is never submitted again**, so an
  ambiguous network failure cannot produce two GPU jobs for one run.

---

## Where state lives, and what owns it

| State | Owner | Notes |
| --- | --- | --- |
| Courses, seeds, evaluations, model registry, training queue | PostgreSQL | Reached only through FastAPI |
| Accounts, memberships, invitations, participants, sessions, audit | PostgreSQL | Tokens stored hashed; participants carry nothing identifying |
| Uploaded syllabi and extracted text | VM local disk | `backend/course_data/{courseId}/` |
| Embedding indexes | VM local disk | `backend/data/indexes/{courseId}.json` |
| Prepared training datasets | VM local disk | `data/exports/{courseId}/`, fetched by the cluster |
| Adapters and training runs | Tillicum GPFS | The registry records references, never absolute paths |
| Published adapters | Tillicum GPFS | `serving/{courseId}/{version}/adapter` |
| Run id ↔ job id ↔ output dir, undelivered reports | Tillicum GPFS | `training/state/`, machine-local |

Artifact references stored in the database are deliberately **relative**. An
absolute cluster path embeds a home directory and a username, and admin surfaces
display these strings.

---

## Frontend

React 19 + TypeScript, built by Vite. Routing is a single tree in `src/App.tsx`
with three role sections and a block of redirects from older URLs.

- `src/lib/api.ts`, `dbApi.ts`, `adminApi.ts` — the only places that call the
  backend. All paths are relative to `VITE_API_BASE_URL`, which **carries the
  `/api` prefix**; the clients write what comes after it.
- `src/components/ui/` — design-system primitives over CSS custom properties in
  `src/styles/tokens.css`. Self-hosted Inter and Source Serif 4.
- `src/context/` — course, session, and comparison-run providers.
  `SessionContext` caches `/api/auth/session` and refreshes on any 401.
- `src/components/auth/RouteGuards.tsx` — navigation guards per role area. The
  chrome and the guards follow the session the backend reported; nothing is
  stored locally that could grant anything.
- `src/lib/authApi.ts`, `inviteApi.ts`, `adminPeopleApi.ts` — the session,
  classroom-code and administration clients, on the same `httpClient.ts` as the
  rest, which sends credentials and the CSRF header.

Professor-facing surfaces never render infrastructure detail: no artifact
references, no Slurm job ids, no service addresses. Admin surfaces do.

---

## Static fixtures

| File | Still used for |
| --- | --- |
| `src/data/comparisonData.json` | Question suggestions on Compare, and backward compatibility for evaluations recorded before live comparison runs |
| `docs/syllabus.txt` | Fixture for backend chunking unit tests (`backend/app/rag.py` reads it). Not used by live course pages |
| `training/heldout_questions.json` | Input to the offline comparison job (`training/compare_inference.py`) |
