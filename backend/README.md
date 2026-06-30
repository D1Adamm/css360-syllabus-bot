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
