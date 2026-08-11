# Syllabus Model Lab

A teaching and research project at UW Bothell (School of STEM) for comparing how
different AI approaches answer questions about a course syllabus. Not an
official University of Washington service.

## Who uses it

The application is organised around three roles, each with its own navigation
and its own vocabulary.

**Students** read the syllabus, contribute example questions, ask a question and
compare four answers side by side, then rate what they saw. They see no
infrastructure: no service names, no storage paths, no dataset internals.

**Professors** create courses, upload a syllabus, review and approve the example
questions collected for their course, and see aggregate results. Course
management, not ML operations.

**Admins** get the technical surface: service health, per-course diagnostics,
the full dataset with validation detail, and dataset export and train/validation
split. This is the only place implementation detail appears.

Roles are selected with a **development-only switcher** in the header. It is not
authentication and grants nothing — every route is reachable by URL. It exists so
the prototype can be walked through as each audience before sign-in is built.

## The four approaches compared

Internally `base`, `rag`, `fineTuned`, and `fineTunedRag`. Students see:

| Shown to students | Description |
|---|---|
| Base Model | General model, no course context |
| Syllabus-Aware | Uses information from your syllabus |
| Course-Trained | Learned from approved course examples |
| Course-Trained + Syllabus | Combines course examples with syllabus context |

All four are live from the backend. The base and syllabus-aware paths share one
CPU-bound local model process and are therefore issued **sequentially**; the two
course-trained paths use a separate service and overlap with them. That ordering
is load-bearing and covered by tests.

## Current state

- Courses are separated by `courseId` under Firebase `courses/{courseId}/...`
- Each course has its own syllabus artifacts, example questions, evaluations, and
  retrieval index
- Syllabus artifacts and indexes are stored **locally** by the FastAPI backend
- **No authentication.** Anyone with a link can open any course
- **No enrolment**, join codes, or class rosters
- **No fine-tuning requests, job tracking, or model versioning.** Training runs
  through the Slurm scripts in `training/`, outside this application

The three gaps above ship as documented integration boundaries rather than fake
persistence — the UI is complete, states plainly what is missing, and writes
nothing. See **[docs/frontend-backend-gaps.md](docs/frontend-backend-gaps.md)**
for the exact endpoints each one needs.

Legacy root-level Firebase nodes (`seedExamples/`, `evaluations/`) may still
exist from earlier prototypes. The live UI does not read or write them. They can
be deleted manually in the Firebase console after confirming nothing external
depends on them; this repository never deletes Firebase data automatically.

## Architecture

| Layer | Technology |
|-------|------------|
| Frontend | React 19 + TypeScript + Vite, Firebase Hosting |
| Design system | Design tokens + UI primitives in `src/components/ui`, self-hosted Inter and Source Serif 4 |
| Course data | Firebase Realtime Database (`courses/{courseId}/metadata\|seedExamples\|evaluations`) |
| Backend | FastAPI |
| Generation | Ollama `llama3.2:3b` |
| Embeddings | Ollama `nomic-embed-text` |
| Fine-tuned paths | Separate inference service via `FINETUNED_SERVICE_URL` |
| Artifacts | Local course storage (`backend/course_data/...`, `backend/data/indexes/...`) |
| Future hosting | Move artifact storage / services toward GCP or a VM |

## Routes

Old URLs (`/course/:courseId/*`, `/compare`, `/architecture`, …) all redirect to
their new homes, preserving query strings.

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
/professor/course/:courseId/model        course model (boundary)
/professor/course/:courseId/results      aggregate results
/professor/course/:courseId/invite       invite students (boundary)
/professor/reviews · /professor/models   cross-course hubs

/admin                                   service health
/admin/courses                           technical course list
/admin/courses/:courseId                 course diagnostics
/admin/courses/:courseId/examples        full dataset + export
/admin/training                          export approved, prepare split
/admin/models                            deployed inference service
/admin/system                            architecture reference

/styleguide                              design system reference (dev)
```

## Local run guide

Use three terminals.

**Terminal 1 — Ollama**

```bash
ollama serve
```

Ensure models are available locally (for example `ollama pull llama3.2:3b` and `ollama pull nomic-embed-text`).

**Terminal 2 — FastAPI backend**

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

**Terminal 3 — Frontend**

```bash
npm run dev
```

Configure frontend env (`.env`) with Firebase settings and:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8001
```

Other useful commands:

```bash
npm install
npm test
npm run build
npm run lint
```

Backend tests:

```bash
cd backend
source .venv/bin/activate
pytest
```

## Routes

| Route | Description |
|-------|-------------|
| `/` | Course picker (lists Firebase courses) |
| `/create-course` | Create a course and optionally upload a syllabus |
| `/architecture` | Architecture overview |
| `/course/:courseId/home` | Course home |
| `/course/:courseId/syllabus` | Extracted syllabus text for that course |
| `/course/:courseId/seeds` | Seed Data Builder |
| `/course/:courseId/dataset` | Seed dataset browser/export |
| `/course/:courseId/compare` | Model comparison (live Base + course RAG) |
| `/course/:courseId/evaluate` | Evaluation form |
| `/course/:courseId/results` | Evaluation results |
| `/home`, `/syllabus`, `/compare`, … | Legacy redirects to `/course/css360-default/...` |

## Data locations

### Firebase Realtime Database

```
courses/
  {courseId}/
    metadata/
    seedExamples/
    evaluations/
```

### Local backend artifacts (not in Firebase)

| Path | Purpose |
|------|---------|
| `backend/course_data/{courseId}/original.(pdf\|txt)` | Uploaded original file |
| `backend/course_data/{courseId}/syllabus.txt` | Extracted syllabus text |
| `backend/data/indexes/{courseId}.json` | Course-specific embedding index |

### Prototype static files

| File | Purpose |
|------|---------|
| `src/data/seedData.json` | Offline fixture read by `scripts/export_seed_dataset.py` — not used by the application |
| `src/data/comparisonData.json` | Example questions offered as suggestions on Compare |
| `docs/syllabus.txt` | Legacy CSS 360 syllabus fixture for backend chunking unit tests only — **not** used by live course pages |

## Current limitations

- No authentication or access control
- No enrolment, join codes, or class rosters
- No fine-tuning requests, training job tracking, or model versioning
- Syllabus artifacts and indexes are local disk only (not GCP yet)
- Root-level legacy Firebase `seedExamples` / `evaluations` data is not auto-migrated or auto-deleted

## Privacy and ethics

- Contributed questions and evaluations are stored in the configured Firebase project under `courses/{courseId}/...`
- Student contributions are anonymous: no name or identifier is collected or stored
- Avoid entering sensitive personal information into contributed questions or evaluation notes
- Live reasoning models can still hallucinate; use syllabus sources and instructor judgment
- Evaluation ratings are subjective classroom observations, not ground-truth labels

## Project structure

```
.
├── backend/                  # FastAPI + Ollama RAG/upload pipeline
├── docs/                     # Architecture and notes
├── scripts/                  # Offline dataset helpers
├── src/
│   ├── app/                  # Route tree and legacy redirects
│   ├── assets/illustrations/ # Named illustration slots (optional assets)
│   ├── components/
│   │   ├── ui/               # Design-system primitives
│   │   └── …                 # Feature components by area
│   ├── context/              # Course, role, and comparison-run providers
│   ├── data/                 # Example questions fixture
│   ├── hooks/                # Course-scoped Firebase and API hooks
│   ├── lib/                  # API clients, error mapping, route builders
│   ├── pages/{student,professor,admin}/
│   ├── shell/                # App shell, navigation, dev role switcher
│   ├── styles/               # Tokens, base, per-area stylesheets
│   ├── types/
│   └── utils/
├── README.md
└── package.json
```
