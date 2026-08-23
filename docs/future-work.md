# Future Work

Realistic future phases for Syllabus Model Lab, organized by workstream.

## Frontend prototype work

The current React application covers the classroom workflow through Phase 7. Remaining frontend improvements could include:

- Improved accessibility testing with automated audit tools
- Offline support via service workers
- Enhanced custom question matching beyond keyword overlap
- Side-by-side diff highlighting between model responses
- Instructor dashboard for importing exported evaluation JSON from multiple students

These items extend the prototype without requiring backend infrastructure.

## ML pipeline work

### 1. Add syllabus parsing and chunking

Parse `docs/syllabus.txt` into structured sections and smaller chunks suitable for retrieval. Define chunk boundaries that respect syllabus headings and policy paragraphs.

### 2. Add embeddings

Generate vector embeddings for syllabus chunks using an open embedding model. Store embeddings with metadata linking back to source sections.

### 3. Add vector retrieval

Implement similarity search over embedded chunks. Retrieve top-k passages for a given question to support RAG inference.

### 4. Connect a base model

Integrate an open-weight language model for base inference without retrieved context. Define prompt templates and safety constraints.

### 5. Build a reviewed fine-tuning dataset

Collect classroom-created seed examples, apply instructor review, and export a curated JSONL training set with quality labels.

### 6. Fine-tune a small open model with LoRA

**Largely done** for CSS 360 QLoRA on Tillicum (`training/README.md`). Remaining work is operational (dataset refresh, evaluation, explicit adapter promotion).

### 7. Add fine-tuned inference

**Done** for the workshop path: Tillicum inference service + UWB SSH tunnel (`training/inference_service/README.md`).

### 8. Combine fine-tuned inference with retrieved context

**Done** for the workshop path: Fine-Tuned + RAG uses course retrieval on the UWB backend plus the remote fine-tuned service.

### 9. Add model and prompt version tracking

Record which model version, adapter, and prompt template produced each response. Enable reproducible comparison runs.

### 10. Add reproducible evaluation runs

Replace static comparison JSON with generated responses tied to versioned model configurations. Allow re-running evaluations against new model versions.

## Infrastructure work

### 11. Add a backend API

Introduce a server layer between the frontend and model services. Endpoints for seeds, comparisons, evaluations, and exports.

### 12. Add authentication only if needed

Implement user accounts if shared classroom storage or instructor review workflows require identity. Skip if anonymous local use remains sufficient.

### 13. Add shared classroom submission storage

Course-scoped PostgreSQL storage for seeds and evaluations already exists. Next: instructor-facing aggregation across students/courses, with auth as needed.

### 14. Add server-side evaluation storage

Persist evaluation records with timestamps, comparison IDs, and optional student identifiers.

### 15. Add deployment and monitoring

Deploy API and model services with health checks, latency monitoring, error alerting, and cost tracking.

## Classroom-research considerations

### 16. Add human-review workflows

Allow instructors to approve, reject, or edit seed examples before they enter the fine-tuning dataset.

### 17. Add privacy controls

Define data retention policies, support deletion requests, and minimize collection of personally identifiable information.

### 18. Ethical use guidelines

Document that model outputs may be wrong, that students should verify answers against the official syllabus, and that simulated outputs must not be presented as authoritative course policy.

## Workstream summary

| Workstream | Phases | Depends on |
|------------|--------|------------|
| Frontend prototype | Accessibility, offline, instructor import | Current app (complete through Phase 7) |
| ML pipeline | Parsing, embeddings, retrieval, models, fine-tuning | Backend API, compute resources |
| Infrastructure | API, auth, database, deployment | Institutional hosting decisions |
| Classroom research | Review workflows, privacy, ethics | Shared storage, instructor tooling |

The recommended next technical step after the frontend prototype is **adding a backend API with shared evaluation storage**, because it unblocks classroom data collection without requiring the full ML pipeline to be operational first.
