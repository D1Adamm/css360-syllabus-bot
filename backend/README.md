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
