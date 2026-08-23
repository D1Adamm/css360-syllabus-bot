# Seed Dataset Export

> **Legacy / not used by current QLoRA training.**  
> Prefer the approved export via `backend/scripts/prepare_qlora_dataset.py`
> (see `training/README.md`).

This document describes the read-only local export script used to combine prototype seed examples with a course's stored examples for later review and fine-tuning preparation.

This step **only prepares data**. It does **not** train or fine-tune a model, and it does **not** modify or delete stored course data.

## What the export includes

The script combines:

1. Prototype examples from `src/data/seedData.json`
2. The course's stored examples, read through `GET /api/db/courses/{courseId}/seeds`

Each exported record preserves useful fields such as:

- `id`
- `question`
- `answer`
- `category`
- `sourceSection`
- `difficulty`
- `answerType`
- `source` (`prototype` or `student`)
- `createdAt` when available
- `notes` when available

The export also keeps the original fine-tuning-friendly fields (`instruction`, `response`, `directlyAnswered`, `origin`) for compatibility with the existing app schema.

## Duplicate handling

Exact duplicates are removed using normalized `question + answer` text:

- trim whitespace
- lowercase
- remove punctuation
- collapse repeated whitespace

The script prints how many duplicates were removed. Semantic deduplication is intentionally not performed yet.

## Output files

Generated files are written to `data/exports/`:

| File | Description |
|------|-------------|
| `seed-dataset-combined.json` | Complete deduplicated dataset as JSON |
| `seed-dataset-combined.jsonl` | Complete deduplicated dataset as JSONL |
| `seed-dataset-prototype.jsonl` | Prototype-only JSONL subset |
| `seed-dataset-student.jsonl` | Student-only JSONL subset |

Generated export files are ignored by git. The `data/exports/` directory itself is kept via `.gitkeep`.

## How to run

From the project root:

```bash
python3 scripts/export_seed_dataset.py
```

No extra Python packages are required. The script uses the Python standard library only.

If the backend is unavailable, the script still exports prototype examples and prints a warning for the skipped stored data.

## Required environment variables

The script reads environment variables from the shell and from a local `.env` file if present.

| Variable | Required | Purpose |
|----------|----------|---------|
| `EXPORT_COURSE_ID` | Yes | Which course's seeds to export — seeds are course-scoped, so there is no unscoped read |
| `API_BASE_URL` | Preferred for scripts | Backend origin, e.g. `http://127.0.0.1:8001` |
| `VITE_API_BASE_URL` | Fallback | The same value the frontend uses, if `API_BASE_URL` is not set |

Example `.env` values:

```bash
EXPORT_COURSE_ID=css-360-winter-2026-a7rp
API_BASE_URL=http://127.0.0.1:8001
```

## Backend access assumptions

The script makes one **read-only** request through FastAPI:

```text
GET {API_BASE_URL}/api/db/courses/{EXPORT_COURSE_ID}/seeds
```

Behavior:

- It talks to the backend, never to PostgreSQL, so it needs no database
  credentials and none can leak through it.
- The script never writes.
- Credentials are not hardcoded and should not be committed.

## Review workflow

After export:

1. Inspect `data/exports/seed-dataset-combined.json`
2. Review category, difficulty, and answer-type counts printed by the script
3. Clean or curate examples manually as needed
4. Use the JSONL files later for fine-tuning preparation

See `docs/prepare-dataset.md` for the next step:

```bash
python3 scripts/prepare_seed_dataset.py
```

Train/validation/test splits are intentionally **not** created by this script. Split the reviewed dataset in a later step.

## Related app pages

This export script does not change the student-facing website. The existing Seed Data Builder and Seed Dataset pages continue to work as before.
