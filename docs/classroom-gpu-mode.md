# Classroom GPU mode

Two serving modes, switched with one command each way:

| Condition | VM mode (normal) | Classroom GPU mode |
|---|---|---|
| Base | VM Ollama `:11434`, `llama3.2:3b` | Tillicum GPU, base model with adapters off |
| RAG | VM Ollama `:11434`; retrieval on the VM | Tillicum GPU, base model with adapters off; retrieval on the VM |
| Fine-Tuned | VM fine-tuned service `:9001` → Ollama `:11435` | Tillicum GPU, the course's PEFT adapter |
| Fine-Tuned + RAG | the same; retrieval on the VM | Tillicum GPU, the course's adapter; retrieval on the VM |

In both modes the VM runs the frontend, FastAPI, auth, the database,
retrieval (embeddings on `:11434`), the generation queue and the logging.

```bash
./scripts/classroom_gpu_mode.sh start  --course css360e-autumn-2026-c08m --admin-email <you>
./scripts/classroom_gpu_mode.sh status [--verify --admin-email <you>]
./scripts/classroom_gpu_mode.sh stop   --course css360e-autumn-2026-c08m --admin-email <you> [--cancel-job]
```

## What stays identical, and what cannot

**The same in both modes:**
- every prompt: the backend builds it either way;
- the course version;
- the output cap (128 tokens in class);
- greedy decoding;
- the 1.05 repetition penalty over the whole context;
- the seed;
- the 4096-token context;
- the stop tokens;
- the prompt *format*.

The GPU service renders each prompt exactly as the VM's Ollama renders it. This
was checked against Ollama 0.33.2: identical text, and identical prompt token
counts (31/43/68 for the three probes). `start` repeats the token-count check
live against the VM's Ollama.

The backend refuses any GPU answer whose reported decoding differs from what
it would have sent the VM. It also refuses an answer from an older GPU service
build that reports nothing; that older build ignored the 128-token cap and
used Transformers' chat template, which adds a "Today Date" line. Those two
differences were why the earlier Tillicum test produced long, drifting
answers.

**Not identical:** the numbers inside the model. The GPU loads the
checkpoint with bitsandbytes NF4, the VM uses GGUF Q4_K_M, so wording can
differ slightly between modes. Every answer's `model` says which engine
answered:
- `… (Tillicum GPU)` on the GPU;
- the Ollama tag on the VM.

The backend log line carries `engine=gpu|vm`. Do not mix modes within one
study condition without recording which was used.

## How switching works

The backend reads `~/.config/aiswe/generation-mode.json`
(`GENERATION_MODE_FILE`) before every generation. The script writes that file
with an atomic rename, so:

- **No service restarts.** Nothing in flight is dropped.
- **The VM services keep running in GPU mode,** so `stop` is immediate.
- **A missing or invalid file means VM mode,** and the error is logged.
- **The GPU URL must be loopback.** That is the tunnel's local end, `127.0.0.1:9101`.
- **Admin view:** `GET /api/admin/generation-mode` shows the mode the next request will use, and where each condition goes.

**`start`** changes nothing until all of these pass:
1. The backend is up and in VM mode.
2. The course's activated version is read from the backend.
3. The Tillicum serving session is found.
4. The tunnel opens (UW Duo).
5. The GPU is healthy: CUDA available, at least `--min-minutes` (60) left, the course version published on Tillicum, and production decoding.
6. A direct probe of Base and course answers stays within the cap and matches the VM's prompt token count.
7. The mode file is written, and the backend confirms GPU mode.
8. All four conditions are answered through the backend, and the GPU's own counters prove they reached it.

If any step after the switch fails, it restores VM mode and closes the tunnel
it opened.

**`stop`**:
1. Switches back first.
2. Confirms the backend is in VM mode.
3. Warms the VM models.
4. Proves all four conditions on the VM.
5. Closes the tunnel.
6. With `--cancel-job`, runs `scancel` through the already-authenticated connection. Otherwise it reminds you the Tillicum job is still running.

**`status`** needs no password. It shows:
- the mode;
- the tunnel;
- GPU time left and the GPU's counters;
- where each condition is generated;
- whether the VM services are ready;
- the failback watchdog: whether its timer is running, its last check, and the last automatic failback (when, why, which job and node, whether the VM warm-up worked).

It warns, and exits 1, when GPU mode is on but the GPU is unreachable.

**If the GPU goes away mid-class** (the allocation ends, the tunnel drops),
generation requests fail with a clear 503 until the mode is VM again. With the
failback watchdog installed (next section) that happens on its own, within
about a minute, and is logged. Without it, requests fail until you run `stop`.
`start` refuses a session with less than an hour left, and `status` warns in
the last 15 minutes.

## Automatic failback (the watchdog)

A Tillicum job once expired while the mode file still said GPU, and every
question failed until `stop` was run by hand. The watchdog closes that gap. It does one thing: when GPU mode is on and the GPU is gone, it switches
generation back to the VM.

```bash
./scripts/gpu_failback_watchdog.sh install      # user units; a check every 15 seconds
./scripts/gpu_failback_watchdog.sh status
./scripts/gpu_failback_watchdog.sh run --dry-run   # one check now; says what it would do, writes nothing
./scripts/gpu_failback_watchdog.sh uninstall
```

**When it acts.** Only when the mode file says GPU mode. Then, either:
- **`expired`**: the mode file's `expiresAt` is past, or within 30 seconds (so the switch happens just before Slurm kills the job, not one check after). The same if the GPU service itself reports that little time left. Acted on at once.
- **`gpu_unreachable`** / **`gpu_unhealthy`**: `/health` through the tunnel gives no answer, or answers in a state the backend could not use (still loading, another build's decoding, the course no longer published), on **three checks in a row**. One failed check changes nothing, and a healthy check starts the count again. The count belongs to one activation of GPU mode and is never carried to the next.

**What it does, in this order.**
1. Replaces the mode file with VM mode: one atomic write. New requests generate on the VM from that moment.
2. Logs the failback.
3. Loads the VM's base, embedding and course models, exactly as `stop` does, and logs whether that worked. If it did not, the mode stays VM; the first answers are just slower.

**What it never does.** It never submits, restarts or cancels a Tillicum job,
never opens or closes a tunnel, never switches *to* GPU mode, and needs no
admin password and no Duo. After a failback the Tillicum job may still be
running and costing GPU time: run `stop` as usual. It repeats the full check
on the VM, closes the tunnel, and names the job to cancel (`--cancel-job`
works if the tunnel's connection is still alive).

**It cannot overwrite a session you just started.** `start`, `stop` and the
watchdog all write the mode file under one lock
(`~/.config/aiswe/generation-mode.json.lock`), held for the write only. Under
that lock the watchdog replaces the file only if it is still, byte for byte,
the one it judged. If you ran `stop` or `start` while it was checking, it
leaves your file alone. A switch back to the VM is never refused for want of
the lock.

**Where it records things** (`~/.local/state/css360-syllabus-bot/`):
- `gpu-failback.log`: one JSON line per automatic failback (`at`, `reason`, `jobId`, `node`, `courseId`, `modelVersion`, `expiresAt`), and one for its warm-up (`ok`, and the failing models if not).
- `gpu-failback-state.json`: the last check, the current count of failed checks, the last failback. `classroom_gpu_mode.sh status` reads it.
- The journal (`journalctl --user -u aiswe-gpu-failback.service`) gets failed checks, failbacks and errors only; the routine "nothing to do" line is filtered out.

The mode file the watchdog writes says `"switchedBy": "gpu-failback-watchdog"`
and keeps the reason and the old job in a `failback` object, which `status`
shows and `stop` uses for its job reminder.

**Settings.** None are needed. To change one, put it in
`~/.config/aiswe/gpu-failback.env`:

| Variable | Default | Meaning |
|---|---|---|
| `AISWE_GPU_FAILBACK_FAILURES` | 3 | failed checks in a row before failing back |
| `AISWE_GPU_FAILBACK_EXPIRY_MARGIN` | 30 | seconds before `expiresAt` to switch (0 = only once it is past) |
| `AISWE_GPU_FAILBACK_HEALTH_TIMEOUT` | 5 | seconds to wait for `/health` |
| `AISWE_GPU_FAILBACK_KEEP_ALIVE` | 4h | `keep_alive` for the warmed VM models |

**Things to know.**
- The watchdog trusts the recorded `expiresAt`. If a job's time limit is extended after `start`, run `stop` and `start` again so the new end is recorded.
- A failback changes the engine mid-class. Every answer's `model` and the backend's `engine=` log field still say which engine answered, and the failback log gives the moment.
- The VM serves about three simultaneous comparisons (see `docs/classroom-capacity.md`); after a failback a large class is slower, not broken.
- The timer is a user unit: it needs lingering (`loginctl enable-linger`) to run while nobody is logged in, as `aiswe-backend` does.

## Capacity

The GPU service generates one answer at a time (one GPU, adapters switched
under a lock), and the backend's queue still allows one generation at a time.
A concise answer takes a few seconds on the GPU, against about 10 s on the
VM's CPU. The 5-student acceptance run below measures the real number.

Thirty simultaneous four-condition comparisons would still need batched
serving: vLLM with multi-LoRA, or Ollama with `OLLAMA_NUM_PARALLEL` on the GPU.
That is a separate change.

## Before class

On Tillicum, update the checkout so the service has this change, and start a
job long enough for the class. The `debug` QOS allows only one hour; `start`
refuses less than `--min-minutes`.

```bash
cd /gpfs/projects/simswe/$USER/css360-syllabus-bot && git pull origin main
```
```bash
SERVICE_QOS=normal ./training/start_finetuned_service.sh --hours 3
```

On the VM:

```bash
cd ~/css360-syllabus-bot && ./scripts/classroom_gpu_mode.sh start --course css360e-autumn-2026-c08m --admin-email <you>
```

After class:

```bash
./scripts/classroom_gpu_mode.sh stop --course css360e-autumn-2026-c08m --admin-email <you> --cancel-job
```

## Tested locally (2026-10-01)

On a laptop:
- the real backend through `scripts/classroom_load_harness.py`;
- the VM fine-tuned service and both Ollama servers;
- a stand-in for the Tillicum node: the real `app.py` and `InferenceEngine`, with the real Llama 3 tokenizer and a real PEFT LoRA on a tiny random model, on CPU.

| Check | Result |
|---|---|
| `start` with a version not published on the GPU | refused, VM untouched |
| `start` when a probe fails | refused, VM untouched |
| `start` | every check PASS; four conditions on the GPU, counters base +2 / course +2 |
| 5 students in GPU mode | 20/20 OK, 5/5 complete, 0 model reloads; all 24 backend generations `engine=gpu` |
| GPU gone while in GPU mode | clear 503; `status` warns |
| `stop` | VM mode, models warmed, four conditions on the VM (`load_ms` 0–2), job reminder printed |
| Unit tests | a failure after the switch restores VM mode; refused decoding; loopback-only URL |

The stand-in's answers are meaningless (random weights); answer quality and
GPU speed are what the VM acceptance run checks.

## Watchdog: tested locally (2026-10-01)

`backend/tests/test_gpu_failback_watchdog.py`, plus two runs of the real
wrapper script on a laptop against a temporary mode file, a stub `/health`
server and the laptop's own Ollama.

| Check | Result |
|---|---|
| VM mode, no mode file, or a file the backend ignores | no-op; the GPU is not asked |
| GPU mode, healthy GPU | no-op, file untouched |
| `expiresAt` passed, or within the margin | VM at once; logged `expired` |
| one failed check, then healthy | no failback; the count resets |
| three failed checks in a row | VM; logged `gpu_unreachable` (or `gpu_unhealthy`) |
| stale mode file after the job expired (the incident) | VM on the first check |
| VM warm-up fails | mode stays VM; failure logged; exit 1 |
| manual `start` during or at the moment of a failback | the new session is never overwritten |
| manual `stop` during or after a failback | VM either way; `stop` still names the job |
| killed during the warm-up; run twice at once; torn state file | the next run finishes or ignores it |
| unit files, `install` / `uninstall` twice | checked against a recording `systemctl` |

Not exercised here: real systemd (the laptop has none). The deployment steps
check `enable`, `stop` and `restart` on the VM.
