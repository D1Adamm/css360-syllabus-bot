# CSS 360 controlled benchmark (September 2026)

The controlled comparison the evolution record calls for: Base and RAG as
controls, then standalone Fine-Tuned and Fine-Tuned + RAG for v2, v3, v4
trained on the VM, and v4 trained on Tillicum, every grounded condition
answering the same prompt bytes over the same ordered chunks with the same
decoding and output cap, through the protected research route
(`docs/css360-benchmark-endpoint.md`). Nothing here retrains, registers,
publishes or promotes a model; CSS 360 v2 stays current throughout.

## What is compared

| Alias | Condition names | Ollama tag (from `evaluation/model_lineage.json`) | Lineage id |
| --- | --- | --- | --- |
| `base` | `base` (standalone), `rag` (pair) | `llama3.2:3b` | none, the base model |
| `v2` | `ft:v2`, `ft_rag:v2` | `css360-ft-v2:latest` | `css360-v2`, production |
| `v3` | `ft:v3`, `ft_rag:v3` | `css360-cpu-v3-test:latest` | `css360-v3` |
| `v4_vm` | `ft:v4_vm`, `ft_rag:v4_vm` | `css360-v4-test:latest` | `css360-v4-vm` |
| `v4_tillicum` | `ft:v4_tillicum`, `ft_rag:v4_tillicum` | `css360-v4-tillicum-test:latest` | `css360-v4-tillicum` |

The runner's preflight reads the benchmark service's `/health` and refuses to
start unless every alias is mapped, available, and served under the tag and
the digest the lineage record names. The digest of the model that answered is
saved with every response, and the route invalidates any condition whose tag
or digest disagrees with the record.

**v1 is not run.** Its test tag `css360-ft-v1-test:latest` exists on the VM,
but the route's alias allowlist is fixed in code (adding it means code, tests
and a redeploy, which would delay the v2–v4 run), and its GGUF is an F16
narrowing of an F32 adapter that the record marks `includeInControlledClaims:
false`. Its lineage is recorded in each run's `manifest.json` under `v1` and
in the report's training-history section.

## Two question sets, kept apart

- `questions_repeat22.json`: the 22 questions of the 2026-09-11 benchmark,
  text unchanged, asked again under the controlled settings. Their results
  are a **repeat**, reported separately from the 2026-09-11 scores (which were
  produced under confounded settings) and from the held-out set.
- `questions_heldout.json`: 24 new questions written from the syllabus for
  this run: 10 factual, 7 unanswerable, 7 false-premise. Each carries
  verbatim `supportingPassages` that `build_question_sets.py` verifies against
  the syllabus text, and the builder refuses any question too similar to a
  repeat-22 question.

Both files carry, per question, `overlapJulySplit`: the nearest question in
the July 2026 laptop export (54 approved, 48 train, 6 validation), which is
the only candidate for the v2/v3 training data since those bytes are not
retained, and which key facts appear in its responses. The runner adds
`overlap_vm_export.json` per run, the same check against the export files
present on the VM, which are the v4 training data (hashes in the manifest).
Overlap is recorded, never used to drop a question: the report labels
questions whose facts a model may have seen.

## Fairness guarantees and how they are checked

Per question, the pair route retrieves once per request and answers every
condition in that request from the same prompt. Because the proxy's 300-second
limit would apply to a public request, conditions are sent in groups of at
most three (`--group-size`, default 3: `rag, ft_rag:v2, ft_rag:v3` then
`ft_rag:v4_vm, ft_rag:v4_tillicum`). Across a question's groups the runner
compares `promptSha256`, `retrieval.setSha256`, `promptTemplate.sha256` and
`decoding`; a mismatch is written to `grounded_identity.jsonl` with
`valid: false` and that question's grounded comparison is excluded from every
RAG-vs-FT+RAG pairing. On the VM the runner talks to loopback, where the
proxy is not involved, but the grouping and the check are the same.

## Running it (on the VM)

```bash
./evaluation/css360_controlled_benchmark/vm.sh setup
./evaluation/css360_controlled_benchmark/vm.sh start
./evaluation/css360_controlled_benchmark/vm.sh status
./evaluation/css360_controlled_benchmark/vm.sh cleanup
```

`setup` installs and starts the `aiswe-benchmark` user unit
(`training/inference_service/aiswe-benchmark.service`) with all five aliases
mapped, enables the route with `backend/scripts/css360_benchmark_env.py`,
restarts the backend, and runs the preflight. `start` runs the runner as a
user unit (or under `nohup` if `systemd-run` is unavailable): preflight, a
three-question pilot with every condition, verification of what the pilot
saved, then the full run with no further prompting. `status` shows progress
and an ETA. `cleanup` turns the route off, restarts the backend, and stops the
service unit. `stop-run` stops the runner; `start` afterwards resumes.

The bearer token stays in `backend/.env`; the runner reads it there and never
prints, logs or saves it.

## What a run saves

`results/<run-id>/`:

| File | Contents |
| --- | --- |
| `manifest.json` | settings, groups, git commit, index and syllabus SHA-256, question-file SHA-256s, service tags and digests vs lineage, training-export file hashes, v1 note |
| `preflight.json` | the preflight report |
| `responses/*.json` | every request's raw response (or its failure), one file per request |
| `records.jsonl` | one line per attempted condition: outcome, answer, served tag and digest, flags, timing, hashes, request status and attempts |
| `grounded_identity.jsonl` | per question, whether its pair-route groups were the same grounded comparison |
| `overlap_vm_export.json` | overlap of every question with the training export on the VM |
| `status.json`, `run.log`, `FINISHED` | progress, log, completion marker |
| `service_health_after.json` | the service's tags and digests after the run |

Every attempted condition is recorded, including timeouts, empty answers,
invalid tags or digests, decoding or prompt-echo mismatches, and requests that
failed outright (`outcome: request_failed`). The report's denominators are
attempts.

Return the run to the analysis machine with `scp -r` (see the deployment
notes); the directory contains no token.

## Scoring and report

`RUBRIC.md` was written before any answer was read. Scoring is blind
(`blind.py` shuffles conditions behind letters per question) and every score
cites a syllabus passage. `summarize.py` produces the tables; `REPORT.md`
holds the write-up, with the 2026-09-11 scores kept in their own directory and
never merged.

## Tests

```bash
backend/.venv/bin/python -m pytest -q evaluation/css360_controlled_benchmark
```
