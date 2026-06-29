# Data Format Reference

Documentation for data types, export formats, and localStorage keys used in Syllabus Model Lab.

## SyllabusTopic

Structured summary of a syllabus section, used by the Syllabus Explorer.

```json
{
  "id": "topic-001",
  "title": "Course Overview",
  "category": "Course Basics",
  "summary": "CSS360 surveys software engineering processes, tools, and techniques used in development and quality assurance.",
  "sourceSection": "Course Catalog Description of Topics",
  "details": [
    "The course covers life-cycle models, requirements analysis, quality assurance, verification, validation, testing, and project planning.",
    "Prerequisites include completion of a two-course programming sequence."
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique topic identifier |
| `title` | string | Display title |
| `category` | string | Topic category for filtering |
| `summary` | string | Short summary paragraph |
| `sourceSection` | string | Syllabus section name |
| `details` | string[] | Bullet-point details |

**Source file:** `src/data/syllabusTopics.json`

## SeedExample

Instruction–response pair for fine-tuning demonstration.

```json
{
  "id": "seed-001",
  "instruction": "When and where does CSS360 meet?",
  "response": "Class meets Tuesday and Thursday from 11:00 a.m. to 1:00 p.m. in UW1 room 302.",
  "category": "Course Basics",
  "sourceSection": "Course Meetings",
  "difficulty": "Easy",
  "directlyAnswered": true,
  "origin": "prototype"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique seed identifier |
| `instruction` | string | Syllabus question |
| `response` | string | Expected answer |
| `category` | string | Topic category |
| `sourceSection` | string | Relevant syllabus section |
| `difficulty` | `"Easy"` \| `"Medium"` \| `"Hard"` | Difficulty label |
| `directlyAnswered` | boolean | Whether the syllabus directly answers the question |
| `origin` | `"prototype"` \| `"user"` | Data source |
| `notes` | string (optional) | Author notes |
| `createdAt` | string (optional) | ISO 8601 timestamp for user seeds |

**Prototype source:** `src/data/seedData.json`

## User-created seed additions

User seeds created on the Seed Data Builder page include the same fields as `SeedExample` with these conventions:

- `origin` is always `"user"`
- `id` follows the pattern `user-seed-<timestamp>-<random>`
- `createdAt` is set to an ISO 8601 string on creation
- `notes` is optional free text

**Storage key:** `syllabus-demo-user-seeds` (localStorage JSON array)

## ComparisonRecord

A syllabus question with four simulated model responses.

```json
{
  "id": "comparison-001",
  "question": "What should I do if I know I will miss class?",
  "category": "Attendance",
  "relevantSyllabusSection": "Course absence form and Impact of Missing Class",
  "baseResponse": {
    "text": "If you know you will miss class, email your instructor...",
    "grounding": "Low",
    "simulated": true
  },
  "ragResponse": {
    "text": "Submit the course absence form at least one hour before class begins...",
    "grounding": "High",
    "simulated": true
  },
  "fineTunedResponse": {
    "text": "Submit the course absence form at least one hour before class begins...",
    "grounding": "Medium",
    "simulated": true
  },
  "fineTunedRagResponse": {
    "text": "Submit the course absence form at least one hour before class begins...",
    "grounding": "High",
    "simulated": true
  },
  "notes": "The base model assumes a generic university policy..."
}
```

### ComparisonResponse

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Simulated model response text |
| `grounding` | `"Low"` \| `"Medium"` \| `"High"` | Prototype grounding annotation |
| `simulated` | boolean | Always `true` in current data |

**Source file:** `src/data/comparisonData.json`

## EvaluationRecord

A student's ratings for one comparison question.

```json
{
  "id": "evaluation-1719590400000-a3f2b1",
  "comparisonId": "comparison-001",
  "mostAccurate": "rag",
  "mostHelpful": "fineTunedRag",
  "mostConcise": "fineTuned",
  "bestGrounded": "rag",
  "preferredModel": "fineTunedRag",
  "hallucinationFlags": ["base"],
  "comment": "The base model invented a makeup policy.",
  "createdAt": "2025-06-28T18:30:00.000Z"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique evaluation identifier |
| `comparisonId` | string | Reference to a `ComparisonRecord.id` |
| `mostAccurate` | `ModelKey` | Most accurate model selection |
| `mostHelpful` | `ModelKey` | Most helpful model selection |
| `mostConcise` | `ModelKey` | Most concise model selection |
| `bestGrounded` | `ModelKey` | Best grounded model selection |
| `preferredModel` | `ModelKey` | Overall preferred model |
| `hallucinationFlags` | `ModelKey[]` | Models flagged for unsupported information (may be empty) |
| `comment` | string (optional) | Free-text notes, max 1000 characters |
| `createdAt` | string | ISO 8601 submission timestamp |

### ModelKey values

| Key | Display label |
|-----|---------------|
| `base` | Base Model |
| `rag` | RAG |
| `fineTuned` | Fine-Tuned Model |
| `fineTunedRag` | Fine-Tuned + RAG |

**Storage key:** `syllabus-demo-evaluations` (localStorage JSON array)

## JSON export

Exports a formatted JSON array with 2-space indentation. Each element is a complete object.

**Used for:**

- Filtered seed export (`syllabus-seed-data-filtered.json`)
- Evaluation export (`syllabus-evaluations.json`)

**Example (evaluations):**

```json
[
  {
    "id": "evaluation-1719590400000-a3f2b1",
    "comparisonId": "comparison-001",
    "mostAccurate": "rag",
    "mostHelpful": "fineTunedRag",
    "mostConcise": "fineTuned",
    "bestGrounded": "rag",
    "preferredModel": "fineTunedRag",
    "hallucinationFlags": ["base"],
    "comment": "The base model invented a makeup policy.",
    "createdAt": "2025-06-28T18:30:00.000Z"
  }
]
```

## JSONL export

Exports one JSON object per line with no trailing comma. Each line is a valid standalone JSON object.

**Used for:**

- Complete seed dataset (`syllabus-seed-data.jsonl`)
- Filtered seeds (`syllabus-seed-data-filtered.jsonl`)
- User seeds only (`syllabus-user-seed-data.jsonl`)

**Example:**

```jsonl
{"id":"seed-001","instruction":"When and where does CSS360 meet?","response":"Class meets Tuesday and Thursday...","category":"Course Basics","sourceSection":"Course Meetings","difficulty":"Easy","directlyAnswered":true,"origin":"prototype"}
{"id":"seed-002","instruction":"Is there a required textbook?","response":"No. There is no required textbook...","category":"Course Basics","sourceSection":"Textbook","difficulty":"Easy","directlyAnswered":true,"origin":"prototype"}
```

JSONL is the preferred format for future fine-tuning pipelines because each line can be streamed and processed independently.

## localStorage keys

| Key | Type | Written by | Reset by |
|-----|------|------------|----------|
| `syllabus-demo-user-seeds` | `SeedExample[]` | Seed Data Builder | Seed Data Builder (delete all) |
| `syllabus-demo-evaluations` | `EvaluationRecord[]` | Evaluation page | Results page (delete all) |

Both keys store JSON-serialized arrays. Malformed data falls back to an empty array without crashing the application. The two keys are independent — resetting one does not affect the other.
