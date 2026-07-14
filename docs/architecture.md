# Architecture

> **Phase 1 status:** See the root [README.md](../README.md) for the current multi-course architecture, local run guide, and completion notes. This document summarizes the same stack for contributors.

## Stack

- React + TypeScript + Vite frontend (Firebase Hosting)
- Firebase Realtime Database for `courses/{courseId}/metadata|seedExamples|evaluations`
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

1. Create course → Firebase metadata
2. Upload syllabus → local extract/chunk/embed → per-course index
3. Syllabus page reads `GET /api/courses/{courseId}/syllabus/text`
4. Seed examples / evaluations use course-scoped Firebase paths only
5. Compare page calls Base + RAG with `courseId`

## Legacy Firebase cleanup (manual)

Older prototypes wrote to root-level `seedExamples/` and `evaluations/`. The live app no longer uses those paths. Leave them in Firebase until you confirm nothing external needs them, then delete them manually in the Firebase console. The app never auto-deletes Firebase data.

## Prototype static data

- `src/data/seedData.json` — read-only prototype seeds shown beside course seeds
- `src/data/comparisonData.json` — simulated Fine-Tuned comparison records
- `docs/syllabus.txt` — fixture for backend chunking unit tests only
