# Syllabus Model Lab

A classroom prototype for comparing how base models, retrieval-augmented generation (RAG), fine-tuning, and fine-tuning combined with RAG answer questions about a course syllabus — across multiple courses.

## Purpose

Syllabus Model Lab supports multi-course syllabus-grounded AI assistant research. Instructors or facilitators create courses, upload syllabi, and students explore extracted syllabus text, create seed training examples, compare live Base Model and course-specific RAG answers alongside simulated fine-tuned outputs, evaluate responses, and review results.

## Phase 1 complete

Phase 1 delivers a working dynamic multi-course system:

- Courses are separated by `courseId` under Firebase `courses/{courseId}/...`
- Each course has its own syllabus artifacts, seed examples, evaluations, and RAG index
- Uploaded syllabus artifacts and embedding indexes are stored **locally** by the FastAPI backend for now
- A future storage backend will move those artifacts to **GCP** (or a VM)
- Instructor authentication is **not** implemented yet
- Fine-Tuned and Fine-Tuned + RAG comparison responses are **still simulated**

Legacy root-level Firebase nodes (`seedExamples/`, `evaluations/`) may still exist in the database from earlier prototypes. The live UI no longer reads or writes them. They can be **manually deleted in the Firebase console** after confirming nothing external still depends on them. This repository does not delete Firebase data automatically.

## Architecture

| Layer | Technology |
|-------|------------|
| Frontend | React + TypeScript + Vite, Firebase Hosting |
| Course data | Firebase Realtime Database (`courses/{courseId}/metadata\|seedExamples\|evaluations`) |
| Backend | FastAPI |
| Generation | Ollama `llama3.2:3b` |
| Embeddings | Ollama `nomic-embed-text` |
| Artifacts | Local course storage (`backend/course_data/...`, `backend/data/indexes/...`) |
| Syllabus pipeline | Dynamic multi-course upload → extract → chunk → embed |
| RAG | Course-specific indexes (`backend/data/indexes/{courseId}.json`) |
| Fine-tuning UI | Fine-Tuned and Fine-Tuned + RAG still simulated |
| Future hosting | Move artifact storage / services toward GCP or a VM |

## Activity flow

1. **Pick or create a course** — Open `/` or create a course at `/create-course`.
2. **Upload a syllabus** — PDF/TXT upload extracts text, chunks it, and builds a course RAG index.
3. **Read the syllabus** — `/course/{courseId}/syllabus` shows that course’s extracted text.
4. **Review / create seed examples** — Dataset and Seed Data Builder pages use `courses/{courseId}/seedExamples`.
5. **Compare model approaches** — Live Base Model and course-specific RAG; Fine-Tuned cards remain simulated.
6. **Evaluate responses** — Ratings save to `courses/{courseId}/evaluations`.
7. **Review results** — Aggregated metrics for the selected course.

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
| `src/data/seedData.json` | Read-only prototype seed examples shown with course seeds |
| `src/data/comparisonData.json` | Simulated Fine-Tuned comparison records |
| `docs/syllabus.txt` | Legacy CSS 360 syllabus fixture for backend chunking unit tests only — **not** used by live course pages |

## Current limitations

- No instructor authentication or access control
- Syllabus artifacts and indexes are local disk only (not GCP yet)
- Fine-Tuned and Fine-Tuned + RAG answers are simulated
- Grounding labels on simulated cards are prototype annotations
- Root-level legacy Firebase `seedExamples` / `evaluations` data is not auto-migrated or auto-deleted

## Privacy and ethics

- Course seeds and evaluations are stored in the configured Firebase project under `courses/{courseId}/...`
- Avoid entering sensitive personal information into seed examples or evaluation notes
- Simulated model outputs may contain incorrect or invented information by design
- Live reasoning models can still hallucinate; use syllabus sources and instructor judgment
- Evaluation ratings are subjective classroom observations, not ground-truth labels

## Project structure

```
.
├── backend/                  # FastAPI + Ollama RAG/upload pipeline
├── docs/                     # Architecture and notes
├── scripts/                  # Offline dataset helpers
├── src/
│   ├── components/
│   ├── context/              # CourseProvider / useCourseId
│   ├── data/                 # Prototype JSON (seeds, comparisons)
│   ├── hooks/                # Course-scoped Firebase hooks
│   ├── lib/                  # API client, Firebase helpers, course ids
│   ├── pages/
│   ├── styles/
│   ├── types/
│   └── utils/
├── README.md
└── package.json
```
