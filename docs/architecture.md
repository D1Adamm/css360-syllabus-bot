# Architecture

> **Phase 1 status:** See the root [README.md](../README.md) for the current multi-course architecture, local run guide, and completion notes. This document summarizes the same stack for contributors.

## Stack

- React + TypeScript + Vite frontend, served by Nginx on the UWB VM
- PostgreSQL for courses, seeds, evaluations, the model registry, model
  requests, and the training queue — every table keyed by `courseId`, reached
  only through FastAPI
- FastAPI backend with Ollama (`llama3.2:3b`, `nomic-embed-text`)
- Local course artifacts under `backend/course_data/` and `backend/data/indexes/`
- Fine-Tuned / Fine-Tuned + RAG UI cards remain simulated
- Future: move local artifact storage toward GCP or a VM; add instructor auth

## Routing

| Path | Component |
|------|-----------|
| `/` | `CoursePickerPage` |
| `/create-course` | `CreateCoursePage` |
| `/architecture` | `ArchitecturePage` |
| `/course/:courseId/*` | Course-scoped pages via `CourseRoute` + `CourseProvider` |
| `/home`, `/compare`, … | Legacy redirects → `/course/css360-default/...` |

Course pages must receive a real `courseId` from the URL. There is no live-page fallback that silently loads a fixed CSS 360 syllabus.

## Data flow

1. Create course → `courses` row via `POST /api/db/courses`
2. Upload syllabus → local extract/chunk/embed → per-course index
3. Syllabus page reads `GET /api/courses/{courseId}/syllabus/text`
4. Seed examples / evaluations are course-scoped rows; every statement binds `courseId`
5. Compare page calls Base + RAG with `courseId`

## Historical note: the Firebase migration

Course data used to live in Firebase Realtime Database under
`courses/{courseId}/...`, and the Tillicum training queue used to be a set of
Firebase REST calls guarded by ETag compare-and-set. Both moved to PostgreSQL.

Nothing in the running system depends on Firebase any more — not the frontend
bundle, not the backend, not the training worker, and none of them needs
Firebase environment variables to start. What remains is deliberately isolated
one-time tooling: `backend/app/firebase_snapshot.py` parses the archived
`courses.json` export and `backend/scripts/import_firebase_snapshot.py` writes
it into PostgreSQL. Both read a file from disk and reach no network service.

The archived snapshot is kept. Do not delete it, and do not wire the importer
into a live code path.

The training queue's concurrency story changed shape in the move. It used to be
a conditional write retried on a stale ETag; it is now a single short
transaction that selects one eligible row `FOR UPDATE SKIP LOCKED` and stamps a
lease. The lease still expires, so a worker that dies mid-job does not strand
its run.

## Prototype static data

- `src/data/seedData.json` — offline/export fixture only (not shown on live Dataset pages)
- `src/data/comparisonData.json` — simulated Fine-Tuned comparison records
- `docs/syllabus.txt` — fixture for backend chunking unit tests only
