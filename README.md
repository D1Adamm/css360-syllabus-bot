# Syllabus Model Lab

A classroom prototype for comparing how base models, retrieval-augmented generation (RAG), fine-tuning, and fine-tuning combined with RAG answer questions about a course syllabus.

## Purpose

Syllabus Model Lab supports CSS 360 classroom research on syllabus-grounded AI assistants. Students explore official syllabus content, review and create seed training examples, compare Base Model and RAG live responses alongside simulated fine-tuned outputs, evaluate those responses, and review aggregated results.

## Activity flow

1. **Explore syllabus topics** — Browse structured syllabus summaries on the Syllabus page.
2. **Review prototype seed data** — Inspect instruction–response pairs on the Dataset page.
3. **Create new seed examples** — Add classroom-created examples on the Seed Data Builder page.
4. **Compare four model approaches** — Review live Base Model and RAG responses alongside simulated fine-tuned outputs on the Model Comparison page.
5. **Evaluate responses** — Rate accuracy, helpfulness, conciseness, grounding, and preference.
6. **Review local results** — View aggregated metrics and export evaluation data.

## Current features

- Persistent prototype banner noting live Base Model and RAG responses
- Syllabus Explorer with search and category filtering
- Prototype seed dataset with statistics, filters, and sorting
- User seed creation with validation and duplicate detection
- Combined prototype and user seed dataset view
- JSON and JSONL export for seeds
- Hybrid four-model comparison interface with live Base Model and RAG responses
- Custom question matcher (keyword overlap against predefined questions)
- Evaluation workflow with form validation and localStorage persistence
- Results dashboard with summary metrics, bar charts, and per-question breakdown
- Evaluation JSON export and evaluation data reset
- Architecture documentation page
- Responsive navigation and mobile menu
- Accessible form controls with fieldsets, legends, and error messages

## Technology stack

- React
- TypeScript
- Vite
- React Router
- Plain CSS
- JSON
- JSONL
- localStorage

## Setup

```bash
npm install
npm run dev
npm run build
npm run lint
```

- `npm run dev` — Start the Vite development server.
- `npm run build` — Type-check and build for production.
- `npm run lint` — Run oxlint on the source files.
- `npm run preview` — Preview the production build locally.

## Routes

| Route | Page | Description |
|-------|------|-------------|
| `/` | Home | Overview and classroom workflow |
| `/syllabus` | Syllabus Explorer | Browse structured syllabus topics |
| `/seed-builder` | Seed Data Builder | Create user seed examples |
| `/dataset` | Seed Dataset | Browse and export seed data |
| `/compare` | Model Comparison | Compare live Base Model and RAG responses with simulated fine-tuned outputs |
| `/evaluate` | Evaluation | Rate model responses (`?comparison=<id>`) |
| `/results` | Results | View aggregated evaluation metrics |
| `/architecture` | Architecture | Technical architecture overview |
| `*` | Not Found | Invalid route handler |

## Data storage

### Prototype JSON files (read-only at runtime)

| File | Purpose |
|------|---------|
| `docs/syllabus.txt` | Authoritative syllabus source text |
| `src/data/syllabusTopics.json` | Structured syllabus topic summaries |
| `src/data/seedData.json` | Prototype seed examples |
| `src/data/comparisonData.json` | Simulated model comparison records |

### localStorage keys

| Key | Contents |
|-----|----------|
| `syllabus-demo-user-seeds` | User-created seed examples (JSON array) |
| `syllabus-demo-evaluations` | Evaluation records (JSON array) |

User seeds and evaluations are stored independently. Resetting one does not affect the other.

## Export formats

### JSON

Exports a formatted JSON array. Used for filtered seed exports and evaluation exports (`syllabus-evaluations.json`).

### JSONL

Exports one JSON object per line (newline-delimited JSON). Used for complete seed datasets and user seed exports. Suitable for fine-tuning pipeline input in future phases.

## Current limitations

- No backend server or API
- No authentication or user accounts
- No shared classroom database across browsers
- No real model inference or text generation
- No real RAG retrieval, embeddings, or vector databases
- No real fine-tuning or training scripts
- No live evaluation service or server-side storage
- Grounding labels on comparison responses are prototype annotations, not automated scores
- Results reflect only evaluations saved in the current browser

## Future phases

See `docs/future-work.md` for a detailed roadmap. High-level next steps include adding a backend API, shared evaluation storage, syllabus parsing and chunking, embeddings, vector retrieval, connecting real model services, building a reviewed fine-tuning dataset, and deployment with monitoring.

## Privacy and ethics

- All user-created seeds and evaluations are stored **only in the browser** via localStorage.
- **No data is submitted** to a server in the current prototype.
- Avoid entering sensitive personal information into seed examples or evaluation notes.
- Simulated model outputs may contain incorrect or invented information by design.
- Simulated outputs must not be mistaken for live AI results — the prototype banner and page notices reinforce this.
- Evaluation ratings are subjective classroom observations, not ground-truth labels.

## Manual compare-page checks

### Custom matcher reset

1. Open `/compare`.
2. Enter a custom question such as `What is the late policy for bot project tasks?` and click **Ask question**.
3. Confirm a custom match or no-match notice appears and Base/RAG reload for that question.
4. Select a different predefined question from the dropdown, such as `What is the difference between open lab and office hours?`
5. Confirm the custom input is cleared, the custom match notice disappears, and the category / relevant syllabus section / simulated cards all match the newly selected predefined question.
6. Confirm Base and RAG answers reload for the selected predefined question.

### Open lab scope check

1. With the backend running, ask the predefined open-lab vs office-hours question on `/compare`.
2. Confirm the RAG answer does not claim open lab sessions are limited to 120 minutes unless the retrieved syllabus context explicitly says so for open lab.

## Project structure

```
.
├── docs/
│   ├── architecture.md       # Technical architecture details
│   ├── data-format.md        # Data type and export documentation
│   ├── future-work.md        # Planned future phases
│   └── syllabus.txt          # Official syllabus source (do not modify)
├── src/
│   ├── components/           # Reusable UI components
│   ├── data/                 # Static JSON prototype data
│   ├── hooks/                # React hooks (e.g. useLocalStorage)
│   ├── pages/                # Route page components
│   ├── styles/
│   │   └── global.css        # Application styles
│   ├── types/
│   │   └── index.ts          # Shared TypeScript types
│   ├── utils/                # Pure utility functions
│   ├── App.tsx               # Router configuration
│   └── main.tsx              # Application entry point
├── index.html
├── package.json
├── tsconfig.json
└── vite.config.ts
```
