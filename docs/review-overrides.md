# Manual Dataset Review Overrides

Use this workflow when you need to correct or reject examples after export and preparation.

Generated files under `data/prepared/` are overwritten every time you run `scripts/prepare_seed_dataset.py`. Do **not** edit them manually.

Instead, record review decisions in the tracked override file:

```text
data/reviews/seed-review-overrides.json
```

This file is committed to git so review corrections persist across export reruns.

## Override file format

Each key is an example `id` from the combined export:

```json
{
  "seed-025": {
    "status": "accepted",
    "answer": "Corrected answer text",
    "reviewNotes": "Reason for correction"
  }
}
```

### Supported statuses

| Status | Meaning | Included in fine-tuning JSONL |
|--------|---------|-------------------------------|
| `accepted` | Reviewed and approved | Yes |
| `rejected` | Do not use for fine-tuning | No |
| `needs_review` | Still under review | Yes |

### Supported override fields

You may override any of these fields:

- `question`
- `answer`
- `category`
- `sourceSection`
- `difficulty`
- `answerType`
- `reviewNotes`

Only include fields you want to change. Unspecified fields keep the exported values.

## Workflow

1. Export the latest combined dataset:

   ```bash
   python3 scripts/export_seed_dataset.py
   ```

2. Prepare the export:

   ```bash
   python3 scripts/prepare_seed_dataset.py
   ```

3. Inspect generated review output:

   ```text
   data/prepared/seed-dataset-prepared.json
   ```

4. Add or update entries in:

   ```text
   data/reviews/seed-review-overrides.json
   ```

5. Re-run preparation:

   ```bash
   python3 scripts/prepare_seed_dataset.py
   ```

6. Confirm overrides in:

   - `data/prepared/seed-dataset-prepared.json`
   - `data/prepared/preparation-summary.json`

## What the preparation script does with overrides

1. Loads and validates the combined export
2. Loads `data/reviews/seed-review-overrides.json` if it exists
3. Applies overrides by example ID after validation
4. Writes prepared outputs with review metadata:
   - `reviewStatus`
   - `reviewNotes`
   - `appliedOverrideFields`
5. Excludes `rejected` examples from `seed-dataset-finetuning.jsonl`
6. Records applied overrides and unmatched override IDs in `preparation-summary.json`

Original export files under `data/exports/` are never modified.

## Example override entries

```json
{
  "-OwMF9WJsT88k-VttbEu": {
    "status": "accepted",
    "answer": "No. There is no direct way to make up missed in-class activities. Filing the absence form avoids a no-call, no-show penalty but does not replace the missed activity.",
    "reviewNotes": "Corrected student attendance answer to match syllabus policy on missed in-class activities."
  },
  "seed-025": {
    "status": "accepted",
    "answer": "The syllabus does not provide one standard recovery plan for a delayed project. It warns that late work reduces time for later stages and may create code conflicts and coordination problems. Check the current task requirements and coordinate with your group.",
    "reviewNotes": "Corrected prototype project delay answer to reflect syllabus language on late work and coordination risk."
  }
}
```

## Unmatched override IDs

If an override ID is not present in the current export file, preparation continues and prints a warning.

This can happen when:

- you are preparing against a prototype-only export in one environment
- a student example has not been exported yet
- an ID was removed from Firebase

Re-run export and preparation locally once the example is available in `data/exports/seed-dataset-combined.json`.

## Related docs

- Export step: `docs/export-dataset.md`
- Preparation step: `docs/prepare-dataset.md`
