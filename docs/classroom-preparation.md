# Classroom preparation and the answer cache

Written 2026-11-10 on branch `feat/compare-answer-cache`. **Not deployed, and
off by default.** Local tests pass and the feature was exercised end to end on
a laptop; it has not yet been measured or verified on the VM. This page says
what the feature does, how to operate it, what it does not cover, and how to
deploy it.

## Why

The VM has one CPU-only model server, and one generation uses all of it
([classroom-capacity.md](classroom-capacity.md)). A class of forty asking four
approaches each cannot be answered live. The first generation of any answer is
the cost, so the answers a class is most likely to need are generated *before*
class, when nobody is waiting, and kept.

This does not raise the VM's capacity. A question nobody prepared still costs
four generations, and forty different unprepared questions will still be
turned away as busy. How much it helps depends on how often students ask a
prepared question.

## What staff do

Course staff (the course's professors, or an administrator) open **Classroom
answers** from the course overview (professor) or the course page's Record
section (administrator). Both lead to the same page.

1. **Choose questions.** The page lists the course's existing questions:
   staff-curated imports, AI-generated drafts, and behaviour examples (hidden
   until the Source filter asks for them). A student's contribution is listed
   only once an instructor has approved it. Rejected questions are not listed.
   Nothing is selected until someone ticks it, and a selection is saved as it
   is made.
2. **Prepare answers.** One button. The server asks each of the four
   approaches for its real answer to each selected question, one generation at
   a time, and stores every answer as soon as it exists.
3. **Stop** at any time. Everything prepared so far is kept and students can
   use it at once. **Prepare remaining answers** carries on later and
   generates only what is missing or out of date.

Selecting a question is separate from everything else about it:

- it does **not** approve the question for training, and approving a question
  does not select it;
- it does **not** suggest the question to students. The Compare page's
  suggestions are unchanged;
- only the question's wording is used. Its example answer is never copied
  anywhere; every stored answer comes from the model pipelines.

## What students see

Nothing new. They type any question, or use the existing suggestions. If the
question matches a prepared one, each approach that has a stored answer
returns it in milliseconds; the others generate as usual. The four cards fill
in order, about 0.4 s apart, rather than all at once. Nothing says an answer
was prepared. A rating records the stored answer's id.

Matching is deliberately narrow: case, spacing, typographic quotes and a
trailing `?`, `!` or `.` are ignored, and nothing else. "When is Homework 2
due?" and "When's HW2 due?" are different questions. There is no semantic
matching.

## What is stored, and for whom

- An answer is stored or served **only** for a question that is currently
  selected for that course. Anything else a student types is answered live and
  never stored. Identical requests that arrive together share one generation
  in memory, which is gone when it finishes.
- A stored row holds the answer exactly as the pipeline returned it (with its
  sources), what produced it, and when. It holds no participant, user or
  session.
- Never stored: a failure, an empty answer, an answer from classroom GPU mode,
  or an answer whose configuration changed while it was being generated.
- If the cache cannot be read or written, the request is answered live. The
  cache is never the reason a request fails.

## When a stored answer stops being used

Automatically, with nothing to clear. Each row is stored under a fingerprint
of everything that decides the answer, and a changed input is a different
fingerprint:

| Change | Affects |
|---|---|
| Base model rebuilt or re-pulled (Ollama digest) | Base, RAG |
| Course adapter rebuilt (Ollama digest) | Fine-Tuned, Fine-Tuned + RAG |
| Another course model version activated | Fine-Tuned, Fine-Tuned + RAG |
| Embedding model changed | RAG, Fine-Tuned + RAG |
| Ollama upgraded | Whatever that server answers |
| Syllabus replaced or re-indexed with different content | RAG, Fine-Tuned + RAG |
| Prompt wording, decoding options, classroom profile | The approaches using them |
| Retrieval code edited (`course_rag.py`, `rag.py`, `retrieval_diversity.py`, `retrieval_facets.py`, `grounded_rag.py`) | RAG, Fine-Tuned + RAG |
| The VM's CPU model | All |
| A question deselected, rejected, deleted or reworded | That question |

The page then shows those answers as out of date, and **Prepare remaining
answers** regenerates them. Old rows are kept, so an earlier rating can still
be traced to the answer it was given.

Two things to know:

- **A deploy that edits a retrieval module retires every stored RAG and
  Fine-Tuned + RAG answer**, even for a comment. Re-prepare after such a
  deploy.
- **Not covered:** the fine-tuned service's `FINETUNED_NUM_CTX` setting, which
  the backend cannot see. If it is ever changed, re-prepare.

If the backend cannot establish what would answer (the fine-tuned service is
down, a model is not installed), it neither reads nor writes the cache for
that approach, and the page says the approach cannot be prepared right now.
The course model's digest is read from the Ollama server the fine-tuned
service itself reports using, so the two servers are not assumed to share
anything.

## How a run behaves

- **One run at a time** on the server, and one generation at a time within it,
  through the existing generation queue at background priority. Every student
  request goes first.
- **Quiet period.** The next generation starts only when no student request,
  in any course, has been active or waiting for `CLASSROOM_PREP_QUIET_SECONDS`
  (default 120). A stored answer served to a student counts as activity. The
  page says "Waiting for students to finish" meanwhile. This is server-wide on
  purpose: the second Ollama holds one course's adapter at a time, so
  preparing one course during another's class would swap that class's model
  out.
- **Not in classroom GPU mode.** A run will not start, and stops if the mode
  switches. Nothing is read from or written to the cache in GPU mode.
- **Stop is graceful.** Work that has not reached a model is dropped at once.
  A generation that is running finishes and its answer is kept; every model
  call has a 120 s timeout. A run that has still not ended
  `CLASSROOM_PREP_STOP_GRACE_SECONDS` (default 180) after Stop is cancelled,
  so the page can never stay on "Stopping". That last resort drops the
  backend's connection only: Ollama may go on computing that one answer for a
  while. Preparation starts nothing after it, so the only possible overlap is
  with a student's request.
- **Failures.** A failed answer is skipped for the rest of that run. If the
  fine-tuned model does not respond, both fine-tuned approaches are skipped
  for the run and Base and RAG continue. Three failures in a row end the run
  as failed, with the reason. Retrying is the same button.
- **Restart.** Stored answers survive. At most the one answer being generated
  is lost. Nothing resumes on its own; the page reports that the run was
  interrupted and offers **Prepare remaining answers**.
- **Order.** Question by question, in the card order. Batching by approach
  (`CLASSROOM_PREP_BATCH_SIZE`) exists but is off: it should be turned on only
  if `scripts/answer_repeatability_check.py` shows a real saving on the VM.

## Settings

All in `backend/.env`; the backend reads them at start.

| Variable | Default | Meaning |
|---|---|---|
| `ANSWER_CACHE_ENABLED` | off | `true` turns on stored answers, sharing of identical concurrent requests, and preparation. Off, the application behaves exactly as before |
| `CLASSROOM_PREP_QUIET_SECONDS` | 120 | How long the site must be quiet before the next preparation generation |
| `CLASSROOM_PREP_STOP_GRACE_SECONDS` | 180 | How long after Stop before a run that has not ended is cancelled |
| `CLASSROOM_PREP_BATCH_SIZE` | 1 | Questions per batch; above 1, each approach is grouped within the batch |

The backend must run as one worker, as it already must
([deployment.md](deployment.md)): the queue, the shared generations and the
run all live in that process.

## Deploying it to production (not done; each step waits for approval)

Nothing below has been run. It follows the standing procedure in
[deployment.md](deployment.md#uwb-vm) and adds only what this
feature needs: one migration and, later, one setting. The branch changes no
Python or Node dependency and nothing under `training/inference_service/`, so
there is no `pip install` and the fine-tuned service is not restarted.

Do it outside class time, and only after the isolated test
([answer-cache-vm-test.md](answer-cache-vm-test.md)) has passed.

Everything up to the backend restart leaves production serving the old code,
so any step before it is a safe place to stop. The feature stays **off** until
step 8.

### 1. Merge (on the development machine, on approval)

`feat/compare-answer-cache` is merged into `main` with `--no-ff` and pushed.
Nothing on the VM changes.

### 2. Before pulling (VM, read-only apart from the backup directory)

The VM is idle and in VM mode (expect the current commit, only the known
untracked files, three `active`, `MODE: VM (normal)`, and `0`):

```bash
cd ~/css360-syllabus-bot && git rev-parse --short HEAD && git status --short && systemctl --user is-active aiswe-backend aiswe-finetuned aiswe-gpu-failback.timer; ./scripts/classroom_gpu_mode.sh status | head -4; journalctl --user-unit aiswe-backend --since '-15 min' --no-pager -o cat | grep -c 'generation condition='
```

Back up what a rollback would need, including the database (the fuller
backup in [deployment.md](deployment.md#before-pulling) also covers service
units and the Nginx configuration, which this deployment does not touch).
Keep the same shell so `$B` stays set:

```bash
B=~/backups/deploy-$(date -u +%Y%m%dT%H%M%SZ) && mkdir -p "$B" && chmod 700 "$B" && echo "$B"
```
```bash
cd ~/css360-syllabus-bot && cp -a $(git ls-files --others --exclude-standard) "$B/" 2>/dev/null; cp -p backend/.env "$B/backend.env" && chmod 600 "$B/backend.env" && git rev-parse HEAD > "$B/previous-commit.txt" && (set -a && source backend/.env && set +a && pg_dump "$DATABASE_URL" -Fc -f "$B/db-before.dump") && ls -la "$B"
```

What will arrive (expect `backend/db/migrations/004_classroom_answer_cache.sql`
and no `requirements.txt`, `package.json` or `training/inference_service/`):

```bash
cd ~/css360-syllabus-bot && git fetch origin && git log --oneline HEAD..origin/main && git diff --name-only HEAD origin/main | grep -E 'requirements\.txt|package(-lock)?\.json|training/inference_service/|backend/db/'
```

### 3. Pull (tracked files only; untracked and ignored files are not touched)

```bash
cd ~/css360-syllabus-bot && git pull --ff-only origin main && git rev-parse --short HEAD && git status --short
```

### 4. Tests under the VM's own Python (safe: no database, no `.env`)

```bash
cd ~/css360-syllabus-bot && backend/.venv/bin/python -m pytest backend/tests -q 2>&1 | tail -2 && backend/.venv/bin/python -m pytest training scripts -q 2>&1 | tail -2
```

### 5. Migration 004 (changes the production database: three new tables only)

It creates `classroom_questions`, `answer_cache` and `classroom_preparation`.
No existing table, column or row changes, and the old code ignores the new
tables, so it is safe to apply before the restart. Running it twice is a
no-op.

```bash
cd ~/css360-syllabus-bot && (set -a && source backend/.env && set +a && psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f backend/db/migrations/004_classroom_answer_cache.sql && psql "$DATABASE_URL" -Atc "select count(*) from information_schema.tables where table_name in ('classroom_questions', 'answer_cache', 'classroom_preparation')")
```

Expect `3`.

### 6. Frontend (publishes the new build; Nginx is reloaded, not restarted)

```bash
cd ~/css360-syllabus-bot && npm ci && npm run build
```
```bash
cd ~/css360-syllabus-bot && sudo cp -a dist/. /usr/share/nginx/html/ && sudo restorecon -R /usr/share/nginx/html && sudo nginx -t && sudo systemctl reload nginx
```

### 7. Backend restart (the one moment of interruption; an in-flight generation fails once)

```bash
systemctl --user restart aiswe-backend && sleep 3 && curl -s http://127.0.0.1:8001/api/health && echo && systemctl --user show aiswe-backend -p ActiveState -p NRestarts && journalctl --user-unit aiswe-backend --since '-2 min' --no-pager -o cat | grep -iE 'traceback|error|Application startup complete' | tail -5
```

Then the standing checks in
[deployment.md](deployment.md#after-a-deployment), and with the feature still
off:

- Admin → a course → Record → **Classroom answers** opens, says classroom
  answers are switched off, and lists the course's questions;
- a Compare in the student preview answers all four approaches as before.

At this point the deployment is complete and behaves as before, apart from the
new page and three extra fields on the generate responses.

### 8. Turn the feature on (a separate decision; needs a second backend restart)

Appends one line to `backend/.env` and prints nothing from the file:

```bash
cd ~/css360-syllabus-bot && grep -q '^ANSWER_CACHE_ENABLED=' backend/.env || printf '\nANSWER_CACHE_ENABLED=true\n' >> backend/.env; grep -c '^ANSWER_CACHE_ENABLED=true' backend/.env
```

Expect `1`, then restart the backend with the command in step 7. Prepare a
few questions for one course, ask one of them on Compare, and watch for a
`classroom preparation: … started` line followed by `generation` lines:

```bash
journalctl --user-unit aiswe-backend --since '-10 min' --no-pager -o cat | grep -E 'classroom preparation|generation condition=' | tail -12
```

Measure with `scripts/classroom_load_test.py` at low levels before relying on
it in a class.

### Turning it off, and rolling back

Off, without undoing anything (stored answers stay and are simply not used):

```bash
cd ~/css360-syllabus-bot && sed -i 's/^ANSWER_CACHE_ENABLED=.*/ANSWER_CACHE_ENABLED=false/' backend/.env && grep -c '^ANSWER_CACHE_ENABLED=false' backend/.env
```

then restart the backend (step 7).

Code, as in [deployment.md](deployment.md#rolling-back): `git reset --hard
"$(cat "$B/previous-commit.txt")"`, rebuild and republish the frontend,
restart the backend. The three new tables can stay: the old code never reads
them. The database backup from step 2 is there if they must go.

On Tillicum nothing changes for this feature; `git pull origin main` in the
cluster checkout keeps it in step whenever convenient.

## What has and has not been verified

Verified locally (2026-11-10):

- The automated suites: cache rules, fingerprint coverage, request sharing,
  preparation (stop, resume, restart, failures, priority), routes and
  permissions, the page, and the staggered reveal.
- The SQL, against a real PostgreSQL 16 (`tests/test_answer_cache_postgres.py`,
  opt-in with `TEST_DATABASE_URL`).
- The whole thing on a laptop against a real backend, PostgreSQL and Ollama,
  for Base and RAG: selecting, preparing, stopping, resuming, killing the
  backend mid-run and recovering, a prepared question served from the store
  (Base shown at 0.4 s, RAG at 0.8 s, no model run), and six identical
  uncached requests costing one generation.

Not verified:

- **Anything on the VM**: its timings, its two Ollama servers, the fine-tuned
  service's `/health` reporting `ollamaUrl` and `models` as the code expects.
- **The two fine-tuned approaches end to end.** The laptop has no fine-tuned
  service, so they were tested only against fakes.
- **How often VM answers repeat**, and whether batching saves time there
  (`scripts/answer_repeatability_check.py`).
- **A classroom.** No load test has been run with the cache on.

## Found on the way, not changed

`backend/db/schema.sql` does not apply to an empty database in one pass: it
creates `idx_evaluations_participant_run` before the `ALTER TABLE` near the
end of the file that adds `evaluations.participant_id`. Production is
unaffected (it was built by migrations). A second pass succeeds. It is
unrelated to this feature and was left as it is.
