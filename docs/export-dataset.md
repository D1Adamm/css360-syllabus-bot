# Seed Dataset Export

This document describes the read-only local export script used to combine prototype seed examples and student-created Firebase examples for later review and fine-tuning preparation.

This step **only prepares data**. It does **not** train or fine-tune a model, and it does **not** modify or delete Firebase data.

## What the export includes

The script combines:

1. Prototype examples from `src/data/seedData.json`
2. Student-created examples from Firebase Realtime Database path `seedExamples`

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

If Firebase is unavailable, the script still exports prototype examples and prints a warning for the skipped Firebase data.

## Required environment variables

The script reads environment variables from the shell and from a local `.env` file if present.

| Variable | Required | Purpose |
|----------|----------|---------|
| `FIREBASE_DATABASE_URL` | Preferred for scripts | Firebase Realtime Database URL |
| `VITE_FIREBASE_DATABASE_URL` | Fallback | Same value used by the frontend if `FIREBASE_DATABASE_URL` is not set |
| `FIREBASE_AUTH_TOKEN` | Only if rules require auth | Firebase auth token for read-only REST access |

Example `.env` values:

```bash
FIREBASE_DATABASE_URL=https://your-project-default-rtdb.firebaseio.com
# Optional, only if database rules require authentication:
# FIREBASE_AUTH_TOKEN=your-read-token
```

You can copy the database URL from the existing frontend Firebase configuration in `.env.example`.

## Firebase access assumptions

The script uses a **read-only** Firebase Realtime Database REST request:

```text
GET {FIREBASE_DATABASE_URL}/seedExamples.json
```

Behavior:

- If database rules allow public reads, no auth token is needed.
- If authentication is required, set `FIREBASE_AUTH_TOKEN`.
- The script never writes to Firebase.
- Credentials are not hardcoded and should not be committed.

If your project is temporarily using public read access for classroom prototyping, document that assumption in your deployment notes and tighten rules before production use.

## Review workflow

After export:

1. Inspect `data/exports/seed-dataset-combined.json`
2. Review category, difficulty, and answer-type counts printed by the script
3. Clean or curate examples manually as needed
4. Use the JSONL files later for fine-tuning preparation

Train/validation/test splits are intentionally **not** created by this script. Split the reviewed dataset in a later step.

## Related app pages

This export script does not change the student-facing website. The existing Seed Data Builder and Seed Dataset pages continue to work as before.
