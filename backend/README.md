# CSS360 Syllabus Model Backend

Minimal FastAPI backend for the Syllabus Model Lab project.

## Setup

From the repository root:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pull the embedding model used by the RAG retrieval endpoint:

```bash
ollama pull nomic-embed-text
```

## Run

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Test health endpoint

With the server running:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok","service":"css360-syllabus-model-backend"}
```

## Test base model endpoint

With Ollama running locally and the `llama3.2:3b` model available:

```bash
curl -X POST http://127.0.0.1:8000/base-model/generate \
  -H "Content-Type: application/json" \
  -d '{"question":"What should a student do if they miss class?"}'
```

Expected response shape:

```json
{
  "answer": "...",
  "model": "llama3.2:3b",
  "responseType": "base"
}
```

If Ollama is not running, the endpoint returns HTTP 503.

## Test RAG retrieval endpoint

With Ollama running locally and the `nomic-embed-text` model available:

```bash
curl -X POST http://127.0.0.1:8000/rag/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question":"What should a student do if they miss class?","topK":3}'
```

Expected response shape:

```json
{
  "embeddingModel": "nomic-embed-text",
  "results": [
    {
      "chunkId": "course-absence-form-001",
      "section": "Course Websites",
      "text": "...",
      "score": 0.82
    }
  ]
}
```

The first request may take longer because the backend builds a local syllabus index at `backend/data/syllabus_index.json`. Later requests reuse that file unless the syllabus content or embedding model changes.

The generated index is local development data and should not be committed to git.

## Test RAG answer generation endpoint

With Ollama running locally, the `nomic-embed-text` embedding model available, and the `llama3.2:3b` generation model available:

```bash
curl -X POST http://127.0.0.1:8000/rag/generate \
  -H "Content-Type: application/json" \
  -d '{"question":"What should a student do if they miss class?","topK":3}'
```

Expected response shape:

```json
{
  "answer": "...",
  "model": "llama3.2:3b",
  "sources": [
    {
      "section": "Course Websites"
    }
  ],
  "retrievedChunks": [
    {
      "chunkId": "course-websites-001",
      "section": "Course Websites",
      "text": "...",
      "score": 0.82
    }
  ],
  "responseType": "rag"
}
```

The endpoint retrieves relevant syllabus chunks, builds a strict context-only prompt, and sends it to the generation model. The first request may take longer if the local syllabus index still needs to be built.
