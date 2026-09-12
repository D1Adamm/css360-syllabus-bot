# CSS 360 model-version benchmark (v2 vs v3)

A comparison of five configurations on the same CSS 360 questions through the
administrator-only route `POST /api/model-testing/generate`:

| Key | Configuration | Route mode | modelVersion |
| --- | --- | --- | --- |
| `rag` | RAG (base model + retrieval) | `rag` | none |
| `ft_v2` | Fine-Tuned v2 | `fineTuned` | `v2` |
| `ftrag_v2` | Fine-Tuned + RAG v2 | `fineTunedRag` | `v2` |
| `ft_v3` | Fine-Tuned v3 | `fineTuned` | `v3` |
| `ftrag_v3` | Fine-Tuned + RAG v3 | `fineTunedRag` | `v3` |

Nothing here changes registry, deployment, publication or classroom state.
The route reads the registry and writes nothing, and the runner never sets a
cookie or holds a password.

## Files

| File | Contents |
| --- | --- |
| `questions.json` | 22 questions (18 answerable, 4 unanswerable probes) with scorer-only reference answers and key facts. **Never put `referenceAnswer` or `keyFacts` in a prompt.** |
| `run_benchmark.py` | Sends every question to every configuration, config by config, and appends each raw answer to `results.json`. Resumable. |
| `results.json` | Raw answers: request, HTTP status, full response body, wall time. |
| `scores.json` | Per-answer rubric scores with rationale. |
| `summary.md` | The benchmark table and findings. |
| `run.log` | Timestamped progress log. |

## Authentication

The normal admin login, done by a person with curl. Keep the password out of
shell history:

```bash
read -rs SML_PASSWORD
```
```bash
curl -s -c /tmp/sml.jar https://aiswe.uwb.edu/api/auth/login -H 'Content-Type: application/json' -H 'X-Requested-With: SyllabusModelLab' -d "{\"email\":\"<admin-email>\",\"password\":\"$SML_PASSWORD\"}" -o /dev/null -w '%{http_code}\n'
```

Expect `204`. Then `unset SML_PASSWORD`. Delete `/tmp/sml.jar` when finished.

## Running

```bash
python3 evaluation/model_version_benchmark/run_benchmark.py --preflight
```
```bash
python3 evaluation/model_version_benchmark/run_benchmark.py
```

The preflight only reads `/api/auth/session` and `/api/fine-tuned/health`.

## Scoring rubric

Each answer is scored against the syllabus text (not against any model) on
four 1–5 scales plus two flags:

| Field | 5 | 3 | 1 |
| --- | --- | --- | --- |
| `correctness` | every key fact right, nothing wrong | some key facts right, some wrong or missing | wrong or fabricated |
| `groundedness` | every claim traceable to the syllabus | mix of supported and unsupported claims | mostly unsupported |
| `relevance` | answers exactly what was asked | partly on topic or evasive | off topic |
| `completeness` | all key facts covered | about half | almost nothing |
| `hallucination` | `true` if the answer asserts any specific fact, policy, number, date or name the syllabus does not support (contradiction or invention) | | |
| `abstained` | unanswerable questions only: `true` if the answer says the syllabus does not cover it | | |

For an unanswerable question, correctness 5 means the answer says the
information is not in the syllabus (and, ideally, says what the syllabus does
say nearby); a confident invented answer is correctness 1 and a hallucination.
