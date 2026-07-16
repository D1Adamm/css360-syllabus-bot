# Syllabus Model Lab Backend

FastAPI backend for multi-course syllabus upload, extraction, embeddings, Base Model generation, and course-specific RAG.

## Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

## Run

Copy `.env.example` to `.env` and set values (including `FIREBASE_DATABASE_URL` for seed persistence). The backend loads `backend/.env` automatically on startup.

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

## Key endpoints

| Method | Path | Notes |
|--------|------|-------|
| GET | `/health` | Service health |
| POST | `/base-model/generate` | Requires `{ "courseId", "question" }` (courseId validated, unused for context) |
| POST | `/rag/generate` | Requires `{ "courseId", "question" }`; uses `backend/data/indexes/{courseId}.json` |
| POST | `/api/courses/{courseId}/syllabus` | Upload PDF/TXT, extract, chunk, embed |
| GET | `/api/courses/{courseId}/syllabus/text` | Extracted syllabus text |
| GET | `/api/courses/{courseId}/chunks` | Indexed chunk metadata |

Local artifacts:

- `course_data/{courseId}/original.(pdf|txt)`
- `course_data/{courseId}/syllabus.txt`
- `data/indexes/{courseId}.json`

`docs/syllabus.txt` remains only as a fixture for legacy chunking unit tests. Live routes do not use a fixed CSS 360 index.

## Tests

```bash
source .venv/bin/activate
pytest
```
