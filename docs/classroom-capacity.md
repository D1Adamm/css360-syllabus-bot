# Classroom capacity: serving four conditions to a class on one CPU VM

Written 2026-10-01 after Compare failed in class (Fine-Tuned and Fine-Tuned +
RAG returned 503 while the backend stayed up, with about 12 GiB of RAM still
free). This page explains the cause, what was changed, how to deploy it, and
how to measure the VM.

## What actually happened

### 1. Base and the course model can't both stay loaded in one Ollama

Every course model is built `FROM llama3.2:3b` plus a LoRA `ADAPTER`. In Ollama
it therefore has **the same weights file** as the base model. Ollama's
scheduler tracks loaded runners by weights file, so it keeps at most one of
the two loaded. A request for the other one is treated as a reload: Ollama
waits for the running request to finish, unloads that runner, and loads the
other. `OLLAMA_MAX_LOADED_MODELS` makes no difference, because it is the same
file.

Reproduced on Ollama 0.33.2 (the VM's version) with a zero-effect LoRA built
`FROM llama3.2:3b`:

| request | `load_duration` | resident afterwards |
|---|---|---|
| base | 1.8 s | base |
| course model | 1.2 s | course model only |
| base | 1.1 s | base only |
| course model | 1.1 s | course model only |

With the course model on a second Ollama server (`:11435`) instead, both stay
loaded, and every request after the first loads in 1–2 ms.

### 2. Two locks guaranteed a swap on nearly every request

The backend serialised Base/RAG behind its own lock. The fine-tuned service
serialised Fine-Tuned/Fine-Tuned + RAG behind a different one. So during a
class there was nearly always one base request and one course-model request in
flight against the same Ollama. That is the alternation above, at every
hand-off. The Compare page makes this worse, because each student sends Base
and both fine-tuned requests at the same moment. A swap also discards the
runner's prompt cache, so the shared rules prefix of the next RAG prompt is
evaluated again from scratch.

This explains the logs: runners repeatedly started for the base model and the
CSS360E model, `n_seq_max = 1` (that is the default `OLLAMA_NUM_PARALLEL=1`,
not a fault), and individual `/api/chat` calls taking 40–50 s.

### 3. Nothing bounded the wait, so failures arrived as late 503s

- A fine-tuned request's 120 s client timeout included its time queued behind
  the fine-tuned service's lock. Under load it spent that time waiting and
  came back as **503** having generated nothing. That is the 503s observed
  in class.
- Base/RAG waited for the backend lock with no limit at all. Only Nginx's
  300 s `proxy_read_timeout` ended the wait.
- An abandoned request (closed tab, refresh) still kept its place in the queue
  and its CPU time.

### 4. The underlying limit is CPU, not RAM

Prompt evaluation of a RAG prompt (about 1,600–2,100 tokens) is compute-bound
on 8 vCPUs, and one generation already uses all of them. The 2026-09-22
latency probe (FT ~2 s, FT+RAG ~4 s) asked the *same* question every time, so
Ollama's prompt cache answered most of the prompt. A class asks different
questions, which costs far more per request.

## What changed

| Area | Change |
|---|---|
| Fine-tuned serving | The course models are served by a **second Ollama server** on loopback `:11435` (`training/inference_service/ollama-finetuned.service`). It reads the same model store, so nothing is copied, and `ollama create` on the main server is visible there immediately. Base and embeddings stay on `:11434`. Both models stay loaded, and answers are unchanged: same weights, adapter, template and options. |
| One queue for all generation | `backend/app/generation_queue.py` replaces the backend lock. Every generation goes through it: Base, RAG, the fine-tuned service calls (made from the backend), and starter jobs. At most `GENERATION_MAX_CONCURRENCY` run at once (default **1**). |
| Bounded, honest waiting | A request waits at most `GENERATION_QUEUE_TIMEOUT_SECONDS` (150 s). It is refused **immediately** if `GENERATION_MAX_WAITING` (160) are already waiting, or if the estimated wait is clearly beyond the limit (1.5×; the estimate swings with the prompt cache, so a borderline request waits instead). A refusal is `503 {"detail": {"code": "generation_busy", "message": …, "retryAfterSeconds": n}}` with `Retry-After`. The Compare page shows "Busy right now … try again in a minute" instead of "temporarily unavailable". A refused request never reaches a model. |
| Order | The four requests of one comparison share the server-side time the comparison began (`X-Comparison-Id`, sent by the Compare page). A student who asked first finishes first, instead of every student getting three answers out of four. Starter jobs wait behind all classroom requests. |
| Abandoned requests | A waiting request checks once a second whether its client is still there. If it has gone, the request leaves the queue (logged `outcome=client_gone`). |
| Warm models | Base/RAG requests and question embeddings now send `keep_alive` (`BASE_MODEL_KEEP_ALIVE`, default `30m`), as the fine-tuned service already did. `scripts/warm_classroom_models.py` loads every classroom model before class with the exact `num_ctx` the requests use. A load with a different context size would make the first real request reload the model. |
| Logging | One line per generation in the backend journal: `generation condition=rag model=llama3.2:3b outcome=ok queue_wait_ms=… ahead=… elapsed_ms=… load_ms=… prompt_tokens=… prompt_eval_ms=… output_tokens=… eval_ms=…`. Refusals log `outcome=busy reason=queue_full\|estimated_wait\|queue_timeout`. Failures log `outcome=error reason=ollama_timeout\|service_timeout\|…`. The fine-tuned service logs its own line and returns Ollama's timings to the backend. No question or answer text is logged. `load_ms` above zero after warm-up means a reload. Previously no backend INFO line reached the journal at all, because nothing configured logging. |
| Visibility | `GET /api/admin/generation-queue` (admin only) returns limits, active, waiting and refusal counters. |
| Load test | `scripts/classroom_load_test.py` reproduces the Compare page's request pattern for N students at once. `scripts/classroom_load_harness.py` runs the real backend on a laptop without a database, for local comparisons. |

What did not change: prompts, the classroom profile (`classroom-concise-v1`),
decoding options, `num_ctx`, output caps, which version a course serves, and
the four-condition schedule in the browser.

## Measured on a laptop (before → after)

These numbers come from a MacBook (10 cores), not the VM. They used Ollama
0.33.2 forced onto the CPU with 8 threads (`num_gpu 0`, `num_thread 8`) and
the real backend code via `scripts/classroom_load_harness.py`. The course
model was a zero-effect LoRA built `FROM llama3.2:3b`, so it collides with
the base model exactly as a real adapter does. Every student started at once;
each level is one burst. The absolute times are faster than the VM will be.
The ratios and failure modes are the point.

"Before" is `main` (two locks, one Ollama). "After" is this change with one
generation at a time, the course model on a second Ollama, and models warmed
first. These local runs did not yet send `X-Comparison-Id`, so the queue
served them in plain arrival order. The load test now sends it, as the Compare
page does, so VM runs also exercise ordering by comparison.

**Standard run** (students cycle through 10 common questions, as a class does):

| students | run | requests OK | complete comparisons | wall | p50 Base / RAG / FT / FT+RAG | worst p95 | failures | model loads |
|---|---|---|---|---|---|---|---|---|
| 1 | before | 4/4 | 1/1 | 21 s | 3.5 / 10 / 6.1 / 21 s | 21 s | none | 4 |
| 1 | after | 4/4 | 1/1 | 3 s | 1.5 / 1.3 / 0.9 / 2.2 s | 2.2 s | none | 0 |
| 5 | before | 20/20 | 5/5 | 113 s | 21 / 62 / 17 / 74 s | 106 s | none | 19 |
| 5 | after | 20/20 | 5/5 | 27 s | 6.9 / 19 / 8.4 / 18 s | 23 s | none | 0 |
| 10 | before | 33/40 | 4/10 | 226 s | 42 / 118 / 33 / 92 s | 142 s | 7 × 503 | 40 |
| 10 | after | 40/40 | 10/10 | 52 s | 10 / 36 / 18 / 32 s | 40 s | none | 0 |
| 20 | before | 54/80 | 3/20 | 437 s | 71 / 231 / 43 / 102 s | 261 s | 26 × 503 | 78 |
| 20 | after | 80/80 | 20/20 | 128 s | 33 / 84 / 32 / 70 s | 92 s | none | 0 |
| 30 | before | 54/120 | 1/30 | 552 s | 114 / 281 / 54 / 116 s | 300 s | 44 × 503, 22 timeouts | 103 |
| 30 | after | 120/120 | 30/30 | 153 s | 42 / 92 / 54 / 89 s | 111 s | none | 0 |

**Worst case** (`--unique-questions`: every prompt is new, so Ollama's prompt
cache helps least):

| students | run | requests OK | complete comparisons | wall | worst p95 | failures | model loads |
|---|---|---|---|---|---|---|---|
| 1 | before | 4/4 | 1/1 | 22 s | 22 s | none | 4 |
| 1 | after | 4/4 | 1/1 | 13 s | 11 s | none | 0 |
| 10 | before | 33/40 | 3/10 | 214 s | 149 s | 7 × 503 | 37 |
| 10 | after | 40/40 | 10/10 | 90 s | 74 s | none | 0 |
| 10 | after, concurrency 2 | 40/40 | 10/10 | 146 s | 129 s | none | 0 |
| 30 | before | 51/120 | 4/30 | 564 s | 296 s | 45 × 503, 24 timeouts | 100 |
| 30 | after | 92/120 | 13/30 | 142 s | 107 s | 28 busy (refused at once) | 0 |
| 30 | after, concurrency 2 | 57/120 | 0/30 | 189 s | 157 s | 63 busy | 0 |

At 30 students, throughput rose from 5.9 to 47 generations per minute
(standard) and from 5.4 to 39 (worst case). Mean CPU went from 65% to 83%,
which is time previously spent loading models rather than answering.

How to read it:

- **Before**, "model loads" is roughly one per request, which is the swap.
  Fine-Tuned + RAG failed with 503 after the client's 120 s, matching the
  classroom logs. Each burst's abandoned requests kept running into the next
  level, because the old code never cancels them.
- **After**, there are no reloads, no 503s and no timeouts. In the worst case
  the queue refuses about a quarter of the 30-student burst *immediately*
  rather than letting it wait and fail. The comparisons that finish are
  complete ones, earliest students first.
- **Concurrency 2 is worse** on a CPU-bound host: two 8-thread runners compete
  for the same cores. At 30 students no comparison completed.
- Tuning made during these runs: refusing on an estimated wait only when it is
  1.5× the limit (refusing at 1.0× turned away work that then drained in
  time), and a waiting cap of 160 rather than 64 (64 refused a third of a
  burst that drained in two minutes).

## Safe settings for the VM (8 vCPU, 15 GiB)

- **RAM is not the constraint.** Each 3B Q4 runner is about 2 GiB of weights
  plus about 0.45 GiB of KV cache at `num_ctx` 4096 with one sequence. Base +
  embeddings + one course model on the second server, plus qwen3 when a
  starter job runs, comes to well under 10 GiB. If both servers memory-map the
  same blob, the weights pages are shared. Check with `free -g` before and
  after warm-up.
- **CPU is the constraint, so keep one generation at a time.** One prompt
  evaluation already saturates all 8 vCPUs. Two runners at once (Base on one
  server, the course model on the other) each start 8 threads on 8 vCPUs, so
  they compete rather than overlap. Keep `GENERATION_MAX_CONCURRENCY=1`,
  `OLLAMA_NUM_PARALLEL=1` on both servers, and `FINETUNED_MAX_CONCURRENCY=1`
  unless the VM load test shows otherwise.
- **Worth measuring on the VM, one at a time:** `num_thread` 4 vs 8 (if the 8
  vCPUs are 4 cores with hyperthreading, 4 is often faster), and
  `GENERATION_MAX_CONCURRENCY=2` with `OLLAMA_NUM_PARALLEL=2`. Neither changes
  answers.
- **Capacity, plainly:** a comparison is four generations, and two of them
  evaluate a ~1,000–2,000-token prompt. On the laptop above, 30 simultaneous
  students fit in about 2.5 minutes when questions repeat, and about three
  quarters fit when none do. The VM's CPU is slower, so expect fewer. The
  queue makes the limit visible and orderly, but it does not remove it. Earlier
  comparisons complete, and later students get a "busy, try again" within
  seconds instead of a spinner and a 503 two minutes later. Staggering the
  class (half asks, then the other half) or a larger VM are the real levers.
- **Starter jobs during class:** a starter call already running holds the
  single generation slot until it finishes (minutes on CPU). Classroom
  requests go first at every hand-off, but avoid uploading a syllabus, which
  starts a starter job, during a class.

## Deploying on the VM

1. Deploy the code as usual: pull, restart `aiswe-backend` and
   `aiswe-finetuned`, rebuild the frontend. No migration.
2. Install the second Ollama (needs sudo). First compare `systemctl cat ollama`
   with `training/inference_service/ollama-finetuned.service`, and make
   `ExecStart`, `User`, `Group` and `OLLAMA_MODELS` match the main server:
   ```bash
   sudo cp training/inference_service/ollama-finetuned.service /etc/systemd/system/
   ```
   ```bash
   sudo systemctl daemon-reload && sudo systemctl enable --now ollama-finetuned
   ```
   ```bash
   curl -s http://127.0.0.1:11435/api/tags | head -c 300
   ```
   The tags listed must include the course models (same store).
3. Point the fine-tuned service at it. Add `OLLAMA_BASE_URL=http://127.0.0.1:11435`
   to `~/.config/aiswe/finetuned.env`, then run, from a shell that has not
   sourced `backend/.env` (a shell-exported `OLLAMA_BASE_URL` wins over the file):
   ```bash
   ./scripts/aiswe_finetuned.sh restart && ./scripts/aiswe_finetuned.sh check
   ```
4. Before class, warm everything (names from the mapping and backend env):
   ```bash
   backend/.venv/bin/python scripts/warm_classroom_models.py --base-model llama3.2:3b --finetuned-model css360e-v1:latest --keep-alive 4h
   ```
   It exits 1 unless all three models are actually resident with the classroom
   context size. Only one course model can be resident on `:11435`, so warm the
   course that is about to be taught.
5. Watch a class:
   ```bash
   journalctl --user-unit aiswe-backend -f | grep ' generation '
   ```
   (`journalctl --user` finds no journal on the VM; `--user-unit` reads the
   system journal's view of the user unit, as `aiswe_finetuned.sh logs` does.)

Rollback: remove `OLLAMA_BASE_URL` from `finetuned.env` and restart
`aiswe-finetuned`. This restores the old shared-server behaviour. The queue
then still bounds waiting.

## Measuring the VM

Run on the VM so CPU and RAM are sampled. It is thirty students' worth of
load, so run it outside class time:

```bash
AISWE_VERIFY_PASSWORD='<admin password>' backend/.venv/bin/python scripts/classroom_load_test.py --admin-email <admin email> --course-id <course id> --levels 1,5,10,20,30 --ollama-url http://127.0.0.1:11434 --ollama-url http://127.0.0.1:11435 --out load-$(date -u +%Y%m%dT%H%M%SZ).json --yes
```

It prints one table row per level: requests OK, complete comparisons, wall
time, generations per minute, busy/timeout/other counts, p50/p95 per
condition, model loads observed on each Ollama, CPU mean/max and RAM peak. It
does not save any answer text. After warm-up, the "model loads" column should
be 0. Anything else means a runner is still being swapped.
