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

## Deploying it (not done yet)

Nothing here has been run on the VM. Each step changes production and waits
for a decision.

1. Check the VM facts in [answer-cache-phase0.md](answer-cache-phase0.md),
   Part A.
2. Deploy the branch's code the usual way ([deployment.md](deployment.md)).
   With `ANSWER_CACHE_ENABLED` unset nothing behaves differently, except that
   the four generate responses carry three extra fields (`cached: false`,
   `generatedAt: null`, `cacheEntryId: null`).
3. Apply `backend/db/migrations/004_classroom_answer_cache.sql`. It creates
   three new tables (`classroom_questions`, `answer_cache`,
   `classroom_preparation`) and changes no existing table or row. If migration
   003 has not been applied yet, apply it first.
4. Restart the backend; build and publish the frontend.
5. With the flag still off, open Classroom answers for a course: the page
   loads, says classroom answers are switched off, and questions can be
   selected.
6. Outside class time, set `ANSWER_CACHE_ENABLED=true`, restart the backend,
   prepare a few questions for one course, and ask one of them on Compare.
7. Measure with `scripts/classroom_load_test.py` at low levels first.

To turn it off again: unset `ANSWER_CACHE_ENABLED` and restart the backend.
Stored rows stay and are simply not used.

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
