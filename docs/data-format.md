# Data Format Reference

> For the live multi-course architecture and storage locations, see the root [README.md](../README.md).

## CourseMetadata

Stored at `courses/{courseId}/metadata`.

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Course name/code |
| `title` | string | Course title |
| `term` | string | Term label |
| `instructorName` | string | Optional instructor display name |
| `createdAt` | string | ISO timestamp |
| `syllabusStatus` | string | e.g. `not_uploaded`, `uploaded`, `extracted`, `indexed`, failure statuses |
| `syllabusFileName` | string\|null | Original upload filename |
| `syllabusType` | string\|null | Upload type |
| `chunkCount` | number | Indexed chunk count |

## SeedExample

Instruction–response pair for fine-tuning demonstration and classroom seed building.

Stored under `courses/{courseId}/seedExamples/{id}` (plus read-only prototype rows from `src/data/seedData.json` in the Dataset UI).

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique example id |
| `instruction` | string | Student-style question |
| `response` | string | Target answer |
| `category` | string | Classroom category |
| `sourceSection` | string | Syllabus section label |
| `difficulty` | string | Easy / Medium / Hard |
| `directlyAnswered` | boolean | Whether syllabus answers it directly |
| `origin` | string | `prototype` or `user` |
| `createdAt` | string? | ISO timestamp |
| `notes` | string? | Optional notes |

## EvaluationRecord

Stored under `courses/{courseId}/evaluations/{id}`.

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique evaluation id |
| `comparisonId` | string | Linked comparison record id |
| Preferencing fields | `ModelKey` | mostAccurate, mostHelpful, etc. |
| `hallucinationFlags` | `ModelKey[]` | Models flagged for hallucination |
| `comment` | string? | Freeer feedback |
| `createdAt` | string | ISO timestamp |

## Legacy root Firebase paths

Historical prototypes also used:

- `seedExamples/`
- `evaluations/`

The live UI does **not** use these paths. They may remain in Firebase until manually removed after confirmation. See README Phase 1 notes.
