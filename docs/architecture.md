# Architecture

Technical architecture documentation for the Syllabus Model Lab frontend prototype.

## Component structure

The application follows a flat component hierarchy organized by responsibility:

- **Layout** (`Layout.tsx`) — App shell with prototype banner, header, navigation, main content area, and footer.
- **Navigation** (`Navigation.tsx`) — Desktop and mobile navigation with React Router `NavLink` active states.
- **Page components** (`src/pages/`) — One component per route, composing shared components and calling utilities.
- **Feature components** (`src/components/`) — Reusable UI such as `SeedForm`, `ModelResponseCard`, `ComparisonQuestionSelector`, `ModelBarChart`, and filter controls.
- **Hooks** (`src/hooks/`) — `useLocalStorage` wraps read/write/reset with optional validation.
- **Utilities** (`src/utils/`) — Pure functions for seed filtering, comparison matching, evaluation aggregation, and file export.
- **Types** (`src/types/index.ts`) — Shared interfaces for syllabus topics, seeds, comparisons, and evaluations.

Pages are responsible for data loading (JSON imports and localStorage) and orchestration. Business logic lives in utilities to keep components readable and testable.

## Routing

React Router v7 configures client-side routes in `App.tsx`:

| Path | Component |
|------|-----------|
| `/` | `HomePage` |
| `/syllabus` | `SyllabusPage` |
| `/seed-builder` | `SeedBuilderPage` |
| `/dataset` | `SeedDatasetPage` |
| `/compare` | `ComparisonPage` |
| `/evaluate` | `EvaluationPage` |
| `/results` | `ResultsPage` |
| `/architecture` | `ArchitecturePage` |
| `*` | `NotFoundPage` |

`ScrollToTop` resets scroll position on navigation. The Evaluation page accepts an optional `?comparison=<id>` query parameter to pre-select a comparison record.

## Local data files

Static JSON files are imported at build time and bundled with the application:

| File | Type | Usage |
|------|------|-------|
| `src/data/syllabusTopics.json` | `SyllabusTopic[]` | Syllabus Explorer |
| `src/data/seedData.json` | `SeedExample[]` | Dataset, Seed Builder source sections |
| `src/data/comparisonData.json` | `ComparisonRecord[]` | Model Comparison, Evaluation, Results |

`docs/syllabus.txt` is the authoritative syllabus document. It is not parsed at runtime but informs the structured topic and seed data.

## localStorage

Two independent localStorage keys persist user-generated data:

### `syllabus-demo-user-seeds`

Stores an array of `SeedExample` objects with `origin: "user"`. Validated on read via `isSeedExampleArray` in `seedDataUtils.ts`. Reset from the Seed Data Builder page deletes only user seeds.

### `syllabus-demo-evaluations`

Stores an array of `EvaluationRecord` objects. Validated on read via `isEvaluationRecordArray` in `evaluationUtils.ts`. Reset from the Results page deletes only evaluations. Multiple evaluations per comparison question are appended, not overwritten.

The `useLocalStorage` hook handles JSON serialization, malformed data fallback to defaults, and quota errors gracefully.

## Utilities

| Module | Responsibility |
|--------|----------------|
| `seedDataUtils.ts` | Seed filtering, sorting, statistics, duplicate detection, ID generation |
| `comparisonUtils.ts` | Text normalization and custom question matching |
| `evaluationUtils.ts` | Model labels, aggregation, tie handling, per-question grouping, comment extraction |
| `exportData.ts` | Client-side file download for JSON and JSONL exports |

## Current data flow

```
docs/syllabus.txt
        ↓
src/data/syllabusTopics.json
        ↓
src/data/seedData.json  +  localStorage (user seeds)
        ↓
src/data/comparisonData.json
        ↓
localStorage (evaluations)
        ↓
Results dashboard (client-side aggregation)
```

1. Students browse syllabus topics to understand course policies.
2. Prototype and user-created seed examples represent potential fine-tuning data.
3. Comparison records provide four pre-written responses per question.
4. Evaluations capture subjective ratings and hallucination flags.
5. The Results page aggregates evaluations entirely in the browser.

## Future API boundary

A production system would introduce a backend between the React frontend and model services:

```
Frontend (React)
        ↓  HTTP / WebSocket
API layer (REST or GraphQL)
        ↓
Model services:
  - Base model inference
  - RAG retrieval + generation
  - Fine-tuned model inference
  - Fine-tuned + RAG combined pipeline
        ↓
Database (evaluations, seeds, audit logs)
```

The frontend would shift from static JSON imports to API calls while preserving similar page structure and user workflows.

## Future model services

| Service | Role |
|---------|------|
| Base model | General-purpose inference without syllabus context |
| RAG pipeline | Chunk syllabus, embed, retrieve relevant passages, augment prompts |
| Fine-tuning pipeline | Train LoRA adapters on reviewed seed datasets |
| Combined service | Fine-tuned model with retrieved context at inference time |

Each service would need version tracking, prompt templates, and reproducible evaluation runs for classroom research.

## Firebase multi-course foundation

Firebase Realtime Database helpers now include a course-scoped layout:

```
courses/{courseId}/metadata
courses/{courseId}/seedExamples/{seedExampleId}
courses/{courseId}/evaluations/{evaluationId}
```

See `CourseMetadata` in `src/types/index.ts` and helpers in `src/lib/coursesDb.ts`, `src/lib/seedExamplesDb.ts`, and `src/lib/evaluationsDb.ts`.

The live UI continues to use legacy global paths (`seedExamples`, `evaluations`) via backward-compatible overloads. Existing CSS 360 Firebase data is not migrated yet. `DEFAULT_COURSE_ID` (`css360-default`) is reserved for a later migration step.

Course routes, syllabus upload, PDF processing, and RAG course scoping are intentionally out of scope for this foundation step.

## Future database role

A database would store:

- Reviewed seed examples with author attribution and approval status
- Comparison question definitions and model response versions
- Evaluation records from multiple students
- Model and prompt version metadata
- Audit trails for human-review workflows

The current localStorage approach is sufficient for single-browser prototyping but cannot support shared classroom analytics.

## Deployment considerations

The current Vite build produces static assets deployable to any static host (GitHub Pages, Netlify, etc.). No server is required.

Future deployment would need:

- API hosting with authentication (if shared storage is required)
- GPU or inference API access for model services
- Database hosting with backup and privacy controls
- Environment-specific configuration for model endpoints
- Monitoring for inference latency, error rates, and cost

Until those services exist, the frontend prototype should remain clearly labeled as a simulation.
