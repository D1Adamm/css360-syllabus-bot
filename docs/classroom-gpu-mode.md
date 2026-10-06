# Classroom GPU mode

How AISWE serves a class from a Tillicum GPU, how it falls back to the VM on
its own, and how to operate both. Written for whoever runs the next class
without the chat history that produced it.

The short, command-only version is
**[classroom-gpu-runbook.md](classroom-gpu-runbook.md)**. This document
explains what those commands do and why.

Contents:

1. [Architecture](#1-architecture)
2. [The two modes](#2-the-two-modes)
3. [The mode file: `generation-mode.json`](#3-the-mode-file-generation-modejson)
4. [The tunnel](#4-the-tunnel)
5. [How a course's model is chosen](#5-how-a-courses-model-is-chosen)
6. [Starting a Tillicum serving job](#6-starting-a-tillicum-serving-job)
7. [Switching to GPU mode: `start`](#7-switching-to-gpu-mode-start)
8. [Checking: `status`](#8-checking-status)
9. [Switching back: `stop`](#9-switching-back-stop)
10. [When the Tillicum job ends](#10-when-the-tillicum-job-ends)
11. [Automatic failback: the watchdog](#11-automatic-failback-the-watchdog)
12. [Monitoring](#12-monitoring)
13. [Troubleshooting](#13-troubleshooting)
14. [Manual recovery](#14-manual-recovery)
15. [Deploying from GitHub to AISWE](#15-deploying-from-github-to-aiswe)
16. [Related changes in the same release](#16-related-changes-in-the-same-release)
17. [Known limitations](#17-known-limitations)
18. [What stays identical between modes](#18-what-stays-identical-between-modes)
19. [What has been tested](#19-what-has-been-tested)

---

## 1. Architecture

Two machines. Only one of them ever talks to students.

**AISWE VM** (`aiswe.uwb.edu`) does everything that is not GPU generation, in
both modes:
- the frontend (static files served by Nginx);
- the FastAPI backend (`aiswe-backend` user unit, `127.0.0.1:8001`);
- sign-in, sessions, classroom codes and PostgreSQL;
- retrieval for RAG and Fine-Tuned + RAG (embeddings on the VM's Ollama, `:11434`);
- building every prompt;
- the generation queue, which allows one generation at a time (`GENERATION_MAX_CONCURRENCY=1`);
- logging every generation, with `engine=vm` or `engine=gpu`.

In VM mode it also generates all four conditions. The engines are the main
Ollama (`:11434`, `llama3.2:3b`), the VM fine-tuned service (`aiswe-finetuned`,
`:9001`) and the second Ollama it uses (`:11435`). These keep running in GPU
mode, so switching back is immediate.

**Tillicum** (the Hyak GPU cluster) does one thing in classroom GPU mode:
generate. A Slurm job named `css360-ft-infer` runs
`training/inference_service/app.py` on one GPU node, port `8001`. It loads
`meta-llama/Llama-3.2-3B-Instruct` once (4-bit) and answers two kinds of request:
- `target: "base"`: the base model with every adapter switched off (Base and RAG);
- a course request: the base model with that course's LoRA adapter (Fine-Tuned and Fine-Tuned + RAG).

It never sees a student, a cookie or the database. It gets a finished prompt
and returns text, plus the decoding settings it used.

**Between them:** an SSH tunnel opened *from* the VM *to* Tillicum. Its local
end is `127.0.0.1:9101` on the VM.

### Where each condition is generated

| Condition | VM mode (normal) | Classroom GPU mode |
|---|---|---|
| Base | VM Ollama `:11434`, `llama3.2:3b` | Tillicum GPU, base model with adapters off |
| RAG | VM Ollama `:11434`; retrieval on the VM | Tillicum GPU, base model with adapters off; retrieval on the VM |
| Fine-Tuned | VM fine-tuned service `:9001` → Ollama `:11435` | Tillicum GPU, the course's PEFT adapter |
| Fine-Tuned + RAG | the same; retrieval on the VM | Tillicum GPU, the course's adapter; retrieval on the VM |

```text
browser ── Nginx ── aiswe-backend (VM) ──┬── VM mode ──► Ollama :11434  (Base, RAG, embeddings)
                                         │              aiswe-finetuned :9001 ► Ollama :11435  (FT, FT+RAG)
                                         │
                                         └── GPU mode ─► 127.0.0.1:9101 ══ SSH tunnel ══► <node>:8001 on Tillicum
                                                         (all four; embeddings stay on :11434)
```

---

## 2. The two modes

**VM mode** is the normal state and the default. It needs nothing from
Tillicum: no job, no tunnel, no Duo, nothing open. It survives logout and
reboots. Its limit is CPU: about three simultaneous four-condition comparisons
(see [classroom-capacity.md](classroom-capacity.md)).

**Classroom GPU mode** moves all four conditions' generation to a Tillicum GPU
for the length of a class. A concise answer takes a few seconds there, against
about 10 s on the VM's CPU. It needs:
- a running Slurm job (it costs GPU time for as long as it runs);
- an SSH tunnel opened with your UW password and Duo;
- the mode file switched by `classroom_gpu_mode.sh start`.

It ends when you run `stop`, or when the watchdog fails back to the VM.

One script, `scripts/classroom_gpu_mode.sh`, switches in each direction:

```bash
./scripts/classroom_gpu_mode.sh start  --course <courseId> --admin-email <you>
./scripts/classroom_gpu_mode.sh status [--verify --admin-email <you>]
./scripts/classroom_gpu_mode.sh stop   --course <courseId> --admin-email <you> [--cancel-job]
```

It is a wrapper that runs `scripts/classroom_gpu_mode.py` with the backend's
virtualenv. That way it uses the backend's own rules for what a mode file
means and what "production decoding" is.

---

## 3. The mode file: `generation-mode.json`

`~/.config/aiswe/generation-mode.json` on the VM, or wherever
`GENERATION_MODE_FILE` in `backend/.env` points. The rules are in
`backend/app/generation_mode.py`.

- **The backend reads it before every generation.** It calls `stat` each time and re-parses only when the file changed. Switching needs no restart and drops nothing in flight.
- **A missing file means VM mode.** So does a file that cannot be read or parsed, and the error is logged: a typo must not take a class offline.
- **The GPU URL must be loopback** (`http://127.0.0.1:<port>`, origin only). The file decides where students' questions go, so nothing else is accepted.
- **Writes are atomic** (temporary file + rename, mode `0600`). The backend reads either the old file or the new one, never half of one.
- **Writes are locked.** `start`, `stop` and the watchdog all hold `flock` on `~/.config/aiswe/generation-mode.json.lock` while they write (§11).

What `start` writes:

```json
{
  "mode": "gpu",
  "gpuUrl": "http://127.0.0.1:9101",
  "since": "2026-10-06T16:55:02+00:00",
  "node": "g014",
  "jobId": "296331",
  "expiresAt": "2026-10-06T19:54:40Z",
  "courseId": "css360d-fall-2026-q0ne",
  "modelVersion": "v1",
  "switchedBy": "madamk"
}
```

What `stop` writes:

```json
{"mode": "vm", "since": "…", "switchedBy": "madamk"}
```

What the watchdog writes when it fails back:

```json
{
  "mode": "vm",
  "since": "…",
  "switchedBy": "gpu-failback-watchdog",
  "failback": {
    "at": "…", "reason": "expired", "detail": "…",
    "jobId": "296331", "node": "g014",
    "courseId": "css360d-fall-2026-q0ne", "modelVersion": "v1",
    "expiresAt": "…", "gpuUrl": "http://127.0.0.1:9101",
    "gpuModeSince": "…", "consecutiveFailures": 0
  }
}
```

The backend acts only on `mode` and `gpuUrl`. The other fields are records:
- `status` prints them;
- the watchdog reads `expiresAt`, `courseId` and `modelVersion`;
- `stop` reads `jobId`, from `failback.jobId` after an automatic failback, for its "job still running" reminder.

To see what the backend itself believes, ask it:
- `./scripts/classroom_gpu_mode.sh status --verify --admin-email <you>`, which signs in;
- or `GET /api/admin/generation-mode` as an administrator.

---

## 4. The tunnel

`start` opens the tunnel itself. Do **not** use `scripts/start_finetuned_tunnel.sh`
for classroom GPU mode (see the note at the end of this section). The command
it runs:

```text
ssh -f -N -o ControlMaster=yes -o ControlPath=~/.local/state/css360-syllabus-bot/ssh-gpu-mode.sock \
    -o ControlPersist=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -L 127.0.0.1:9101:<node>:8001 $USER@tillicum.hyak.uw.edu
```

- **Duo.** The UW password and Duo prompt appear in your terminal. Nothing is stored. This is the one step that needs a person.
- **Login name.** The login is `$USER@tillicum.hyak.uw.edu` with the *VM's* `$USER`. There is no option to change it, so the VM account name must be your UW NetID.
- **Background master.** The SSH process is a control master in the background. It keeps running after `start` returns.
- **State files**, both in `~/.local/state/css360-syllabus-bot/` (or `$XDG_STATE_HOME/css360-syllabus-bot/`):
  - `ssh-gpu-mode.sock`: the control socket;
  - `gpu-mode-tunnel.json`: the node, ports, login and time opened.
- **Reuse.** `start` reuses an open tunnel when it points at the same node and `/health` answers through it. Otherwise it closes the old one and opens a new one.
- **Closing.** `stop` closes it (`ssh -O exit`) and deletes both state files.
- **Cancelling the job.** `stop --cancel-job` runs `scancel <jobId>` through this already-authenticated connection, so it needs no second Duo prompt. It works only while the control master is still alive.

`--no-tunnel` (on `start`) uses a forward you opened by hand on `--gpu-port`.
It is meant for testing.

**Not this tunnel:** `scripts/start_finetuned_tunnel.sh` /
`status_finetuned_tunnel.sh` / `stop_finetuned_tunnel.sh` run the older
*emergency fallback*. That fallback forwards `127.0.0.1:9001` to Tillicum
for the fine-tuned conditions only, and needs `aiswe-finetuned` stopped first
([tillicum-operations.md](tillicum-operations.md#inference-the-tillicum-fallback)).
It is a different thing. `start_finetuned_service.sh` still prints that script
as its "Next" step. For classroom GPU mode, ignore that line and run
`classroom_gpu_mode.sh start` instead.

---

## 5. How a course's model is chosen

**The version** is the same in both modes. Fine-Tuned and Fine-Tuned + RAG
always answer from the course's **activated** version: the highest version
that is `deployment = online` and `status = ready` in PostgreSQL. A version is
activated in Admin → Models. `start` and `stop` read that version from the
backend; `--version vN` overrides it for that run only.

**On the VM**, `~/.config/aiswe/finetuned.env` maps each course version to an
Ollama tag:

```text
FINETUNED_OLLAMA_MODELS=css360d-fall-2026-q0ne@v1=css360d-v1:latest,css360e-autumn-2026-c08m@v1=css360e-v1:latest
```

**On Tillicum**, the service builds the adapter path from the validated course
id and version in each request:

```text
/gpfs/projects/simswe/$USER/training_outputs/serving/<courseId>/<version>/adapter/
```

Up to `MAX_LOADED_ADAPTERS` (4) stay loaded. Switching adapters and generating
happen under one lock. The response echoes the course and version, and the
backend refuses a mismatch. Base and RAG requests use `target: "base"`, so no
adapter is involved.

**`start` refuses a course version the GPU cannot serve.** It reads `/health`
through the tunnel and requires the course and version under `courses`. To see
what Tillicum has published, run `./training/status_finetuned_service.sh` on
Tillicum: it prints "Published course adapters".

If the version is missing, publish it into the serving tree with
`./training/promote_qlora_adapter.sh --course <courseId> --version <vN> <adapter dir>`.
Add `--no-report` when the version is already activated on the VM: without it,
the script also reports the publication to the backend, which is the legacy
way of choosing a course's version.

`--course` is always explicit. `start` checks only the course you name, and
`stop` verifies only that course on the VM. Other courses' fine-tuned
requests in GPU mode also go to the GPU, which answers them if their adapter
is published there. `start` does not check that for you.

---

## 6. Starting a Tillicum serving job

On Tillicum (`ssh $USER@tillicum.hyak.uw.edu`, UW password + Duo):

```bash
cd /gpfs/projects/simswe/$USER/css360-syllabus-bot && git pull origin main
```

**3-hour job:**
```bash
SERVICE_QOS=normal ./training/start_finetuned_service.sh --hours 3
```

**2-hour job:**
```bash
SERVICE_QOS=normal ./training/start_finetuned_service.sh --hours 2
```

- **Use the `normal` QOS.** The default QOS is `debug`, which caps a job at 1 hour; `normal` allows up to 8 in the script's table. A request over the QOS ceiling is refused before `sbatch`, because Slurm would otherwise accept it and leave it pending forever (`QOSMaxWallDurationPerJobLimit`).
- **Size the job for the class.** `start` on the VM refuses a session with less than 60 minutes left (`--min-minutes`), and GPU mode ends when the job does. Start a job that covers the class plus the time before it. A 3-hour job started 30 minutes before a 2-hour class is right.
- **What the script does.** It submits `training/inference_service/serve.slurm` (1 GPU, 8 CPUs, 200 GB), waits up to 600 s for the allocation and up to 900 s for the model to load, and records the session with the backend: node, port, job, expiry. That record is how `start` on the VM finds the node without anyone typing a hostname.
- **Re-running reuses an active job.** It never submits a second one, and it does **not** extend one. For a longer session, wait for the job to end (or `scancel` it) and start a new one.
- **Logs:** `training/logs/infer-<jobId>.out` and `.err` in the Tillicum checkout.

Success ends with:

```text
Fine-tuned service READY
Job ID: 296331
Node: g014
GPU endpoint: http://g014:8001
Session ends in about 179 minutes.

Serving session recorded: serve-296331
…
Courses this session can answer for:
  css360d-fall-2026-q0ne  current=v1
  css360e-autumn-2026-c08m  current=v1
```

The printed "Next" step is the old fallback tunnel (§4). For classroom GPU
mode, go to the VM and run `start` instead.

While it waits for a node, it prints the Slurm pending reason in plain words,
for example `State: PD — waiting for a node with the requested resources`
(§13).

Tillicum-side status and stop:

```bash
./training/status_finetuned_service.sh
./training/stop_finetuned_service.sh
```

---

## 7. Switching to GPU mode: `start`

On the VM (`ssh <you>@aiswe.uwb.edu`), for CSS360D:

```bash
cd ~/css360-syllabus-bot
./scripts/classroom_gpu_mode.sh start \
  --course css360d-fall-2026-q0ne \
  --admin-email madamk@uw.edu
```

You are asked for the admin password (or it is read from
`AISWE_VERIFY_PASSWORD`), then for the UW password and Duo when the tunnel
opens. **Nothing changes until every check passes:**

1. the backend is up, runs code that knows about modes, and is in VM mode;
2. the course's activated version, from the backend;
3. the Tillicum serving session (the record `start_finetuned_service.sh` made); `--node <gNNN> --job-id <id>` skips the lookup;
4. the tunnel opens, or the open one is reused;
5. the GPU service is healthy: base model loaded, CUDA available, the build answers `base`, decoding identical to production, ≥ 60 min left, the course version published;
6. a direct probe: one Base and one course answer within the 128-token classroom cap, and the same prompt token count as the VM's Ollama for the same question;
7. **the switch**: one atomic write of the mode file, then the backend must report GPU mode;
8. all four conditions through the backend. Each must be answered by the GPU (its `model` ends in `(Tillicum GPU)`), and the GPU's own counters must rise by 2 base and 2 course.

If anything after step 7 fails, it restores the previous mode file and closes
the tunnel it opened. If anything before step 7 fails, nothing was switched.
Either way you are told: `The backend is in VM mode; nothing was left half-switched.`

**What success looks like** (abridged):

```text
Classroom GPU mode: start. Nothing changes until every check below passes.

PASS  backend up, in VM mode
PASS  css360d-fall-2026-q0ne: version v1 (activated in the backend)
PASS  Tillicum session: job 296331 on g014:8001, ends 2026-10-06T19:54:40Z
      Opening the tunnel 127.0.0.1:9101 -> g014:8001 via madamk@tillicum.hyak.uw.edu.
      Complete UW password / Duo if prompted.
PASS  tunnel open: http://127.0.0.1:9101 -> g014:8001
PASS  GPU service healthy on NVIDIA H200; decoding = production; 172 min left
INFO  GPU base: 40 tokens in 1.9s
PASS  prompt tokenised identically to the VM's Ollama (31 tokens)
INFO  GPU course: 38 tokens in 1.7s
PASS  direct GPU probe: Base and course answers within the classroom cap
PASS  switched: backend now generating on the GPU (/home/madamk/.config/aiswe/generation-mode.json)
INFO  base           2.3s  412 chars  meta-llama/Llama-3.2-3B-Instruct (Tillicum GPU)
INFO  rag            2.6s  389 chars  meta-llama/Llama-3.2-3B-Instruct (Tillicum GPU)
INFO  fineTuned      2.1s  305 chars  meta-llama/Llama-3.2-3B-Instruct + css360d-fall-2026-q0ne@v1 (Tillicum GPU)
INFO  fineTunedRag   2.4s  331 chars  meta-llama/Llama-3.2-3B-Instruct + css360d-fall-2026-q0ne@v1 (Tillicum GPU)
PASS  all four conditions answered on the GPU (GPU counters: base +2, course +2)

CLASSROOM GPU MODE ACTIVE: Base, RAG, Fine-Tuned and Fine-Tuned + RAG are generated on the Tillicum GPU.
  Job 296331 on g014 ends 2026-10-06T19:54:40Z; GPU mode stops working then.
```

(Numbers, the GPU name and model names will differ; the `PASS` lines and the
final `CLASSROOM GPU MODE ACTIVE` line are what to look for.)

Other options: `--version vN`, `--node`/`--remote-port`/`--job-id` (bypass
the session lookup), `--min-minutes N`, `--gpu-port N` (default `9101`, or
`GPU_MODE_LOCAL_PORT`), `--backend-url`, `--timeout`.

---

## 8. Checking: `status`

```bash
./scripts/classroom_gpu_mode.sh status
```

It needs no password and changes nothing. It prints:
- the mode file and **`MODE: CLASSROOM GPU`** or **`MODE: VM (normal)`**, with the file's fields; after an automatic failback, also an `AUTOMATIC FAILBACK at …` line;
- the tunnel (`open to <node>` or `closed`), and the GPU's own `/health`: status, GPU name, minutes left, loaded adapters and `served` counters;
- where each condition is generated right now;
- the VM services: backend, fine-tuned service, and what each Ollama has resident;
- the failback watchdog: timer state, last check, last automatic failback (§11).

**It exits 1 with a `WARNING`** when:
- GPU mode is on but the GPU service does not answer (`GPU MODE IS ON BUT THE GPU SERVICE IS UNREACHABLE: students get errors.`);
- the allocation ends within 15 minutes;
- VM mode is on but the GPU tunnel is still open;
- with `--verify`, the backend and the file disagree.

**It prints a `NOTE`** (exit code unchanged) when:
- GPU mode is on without the watchdog running;
- the watchdog's timer is active but has not checked for over 2 minutes.

`--verify --admin-email <you>` also signs in and asks the backend which mode
it is in.

Healthy GPU mode looks like:

```text
MODE: CLASSROOM GPU
  since: 2026-10-06T16:55:02+00:00
  node: g014
  jobId: 296331
  expiresAt: 2026-10-06T19:54:40Z
  courseId: css360d-fall-2026-q0ne
  modelVersion: v1
  switchedBy: madamk

Tunnel: open to g014 (http://127.0.0.1:9101)
  GPU service: ok on NVIDIA H200, 150 min left, adapters ['css360d-fall-2026-q0ne@v1'], served {'base': 40, 'course': 38}

Where each condition is generated:
  base          Tillicum GPU via http://127.0.0.1:9101 (base model, adapters off)
  …

Automatic failback watchdog (aiswe-gpu-failback.timer):
  timer          RUNNING (enabled, active)
  last check     2026-10-06T17:20:15+00:00 (4s ago): GPU mode, GPU healthy (job 296331 on g014, 150 min left)
  last failback  none
```

---

## 9. Switching back: `stop`

```bash
./scripts/classroom_gpu_mode.sh stop \
  --course css360d-fall-2026-q0ne \
  --admin-email madamk@uw.edu \
  --cancel-job
```

In order:
1. **Writes VM mode first**, before asking for the password. New requests generate on the VM from this moment, even if everything after it fails.
2. Signs in and confirms the backend reports VM mode.
3. Warms the VM models with `scripts/warm_classroom_models.py`: the base model and the embedding model on `:11434`, and the course's model on the fine-tuned Ollama, with `--keep-alive 4h` by default.
4. Checks the VM fine-tuned service's `/health`, then answers all four conditions through the backend and requires each to come from the VM.
5. With `--cancel-job`, if the tunnel's connection is still alive, runs `scancel <jobId>` on Tillicum.
6. Closes the tunnel.

Success:

```text
PASS  mode file: vm (new requests generate on the VM from now)
PASS  backend reports VM mode
…
PASS  VM models resident
PASS  all four conditions answered on the VM
PASS  cancelled Tillicum job 296331
PASS  tunnel closed

VM MODE ACTIVE: all four conditions are generated on this VM, and verified.
```

- **Without `--cancel-job`**, or if the cancel could not run, it ends with `REMINDER: Tillicum job <id> is still running and using GPU time. On Tillicum: scancel <id>`. Do that.
- **If a check fails**, it prints `VM MODE IS ACTIVE, BUT SOME CHECKS FAILED:` and the list, and exits 1. The mode is VM regardless.
- **It is safe to run twice, or in VM mode.** Run it after an automatic failback, too: it does the verification and cleanup the watchdog deliberately does not.

---

## 10. When the Tillicum job ends

A Slurm job ends at its wall clock whatever anyone is doing. The node stops
answering, and the tunnel's forward then leads nowhere.

**Without the watchdog**, this is what happened once: the mode file kept
pointing at the dead job. Every generation failed with a 503, such as "The GPU
service timed out", until someone ran `stop` by hand.

**With the watchdog** (§11), the mode switches to the VM on its own, within
about 30 s of the recorded `expiresAt`. In practice it switches up to 30 s
*before* it, so the switch beats Slurm. If no expiry was recorded, it switches
on the third failed health check, roughly 30–50 s after the GPU went away.
Students who asked in that window see an error once; the next question goes to
the VM.

Either way the GPU job's own time is over. Run `stop` afterwards to verify the
VM path and close the tunnel. Classroom GPU mode cannot be resumed on a dead
job: start a new Tillicum job (§6) and run `start` again.

A job you need to end early: `stop --cancel-job` (§9), or `scancel <jobId>` on
Tillicum, or `./training/stop_finetuned_service.sh` there.

---

## 11. Automatic failback: the watchdog

`scripts/gpu_failback_watchdog.py`, run by a systemd **user** timer on the VM.
Its one job: when GPU mode is on and the GPU is gone or about to be, put the
VM back.

### Units

| Unit | What it is |
|---|---|
| `aiswe-gpu-failback.timer` | Fires every 15 s (`OnCalendar=*:*:0/15`, `AccuracySec=1s`). Enabled at install; starts at boot with lingering. |
| `aiswe-gpu-failback.service` | `Type=oneshot`: one check per activation, running `scripts/gpu_failback_watchdog.sh run`. No `[Install]` section; only the timer starts it. `TimeoutStartSec=600`. |

The templates are in `scripts/systemd/`. `install` renders `@REPO_ROOT@` to
the checkout path and writes them to `~/.config/systemd/user/`. The service
runs code from the checkout, so a `git pull` updates what it runs; re-run
`install` only when the unit files themselves change.

### Commands

```bash
./scripts/gpu_failback_watchdog.sh run --dry-run   # one check now; says what it would do; writes nothing
./scripts/gpu_failback_watchdog.sh install         # write units, daemon-reload, enable + (re)start the timer
./scripts/gpu_failback_watchdog.sh status          # timer, state file, last log lines
./scripts/gpu_failback_watchdog.sh uninstall       # disable + stop + remove units (state and log kept)
```

- `install --check` shows what `install` would write, and writes and enables nothing.
- `run` (without `--dry-run`) is exactly what the timer runs.
- `install` warns if lingering is off: `loginctl enable-linger $USER`, once.

### What it checks, each run

1. **The mode file.** VM mode, no file, or a file the backend would ignore: nothing to do; it does not even contact the GPU. Only valid GPU mode goes further.
2. **Expiry (reason `expired`).** If the mode file's `expiresAt` is past or within the margin (30 s), it fails back **at once**. The same if the GPU service's `/health` reports `secondsRemaining` within the margin. This covers sessions started with `--node`, which have no recorded expiry.
3. **Health.** `GET <gpuUrl>/health` through the tunnel, 5 s timeout. A check fails if:
   - there is no answer (`gpu_unreachable`);
   - or the answer is something the backend could not use (`gpu_unhealthy`). That is the same test `start` applies, less the time-left margin: base model not loaded, no CUDA, a build without the `base` target, decoding different from production, or the recorded course version not published.
4. **Threshold.** It fails back on the **3rd consecutive** failed check. A healthy check resets the count to zero.
   - The count is tied to one GPU session: a fingerprint of the mode file's exact contents. A new `start` therefore never inherits the old session's failures.
   - Failures more than 120 s apart do not count as consecutive, for example after the timer was stopped.

### What it does when it fails back, in this order

1. **Switch first.** It replaces the mode file with VM mode (§3), atomically, under the mode-file lock. It does so **only if the file is still byte-for-byte the one it judged.** New requests go to the VM from this moment.
2. **Record.** It appends a `failback` line to the log and updates the state file *before* the slow part, so a run killed mid-warm-up still leaves a trace.
3. **Warm the VM models.** These are the same three models `stop` warms: base and embedding on `:11434`, and the course's tag on the fine-tuned Ollama from `finetuned.env`. It uses `keep_alive` 4h and a 120 s timeout per model. It logs `failback_warmup` with `ok: true/false` and the per-model lines.
   - **If the warm-up fails, the mode stays VM.** The first answers are just slower, and the run exits 1 so the journal shows a failure.
   - A warm-up interrupted by a restart is finished by the next run, at most twice, and otherwise recorded as failed.

### Race protection

- **Mode-file lock.** `start`, `stop` and the watchdog write under one `flock` (`~/.config/aiswe/generation-mode.json.lock`), held only for the write, never across a network call. The kernel releases it if the holder dies, so there is no stale lock to clean up.
- **Compare-before-write.** If you ran `start` (a new session) or `stop` while the watchdog was checking, the file no longer matches, and the watchdog leaves it alone (`outcome: superseded`).
- **A busy lock.** If `start`/`stop` hold the lock for more than 10 s, the watchdog defers to its next run (`outcome: deferred`).
- **No lock for VM.** `start` refuses to switch *to* GPU if it cannot get the lock in 30 s, but a switch *to* the VM (`stop`) is never refused for want of it.
- **One run at a time.** A run lock (`gpu-failback.run.lock`) stops two runs counting the same failure twice.

### State and logs

All in `~/.local/state/css360-syllabus-bot/` (override: `AISWE_GPU_FAILBACK_STATE_DIR`):

| File | Contents |
|---|---|
| `gpu-failback-state.json` | `lastCheck` (time, mode, outcome, detail); `streak` (consecutive failures for the current session); `lastFailback` (the full record plus `vmWarmup: ok/failed/pending`). Rewritten on every check. `classroom_gpu_mode.sh status` reads it. |
| `gpu-failback.log` | Append-only, one JSON object per line: `event: "failback"` (`at`, `reason`, `detail`, `jobId`, `node`, `courseId`, `modelVersion`, `expiresAt`, `gpuUrl`, `gpuModeSince`, `consecutiveFailures`) and `event: "failback_warmup"` (`at`, `ok`, `detail`, and the same job fields). Written only on failbacks. |
| journal | `journalctl --user -u aiswe-gpu-failback.service`. Failed checks, failbacks and errors only: routine "nothing to do" lines are logged at info and dropped by `LogLevelMax=notice`. |

Outcomes in `lastCheck.outcome`:

| Outcome | Meaning |
|---|---|
| `vm_mode` | Nothing to do. |
| `healthy` | GPU mode, GPU fine. |
| `check_failed` | GPU mode, `n/3` failed. |
| `failback` | Switched to VM. |
| `superseded` | A human changed the file mid-check. |
| `deferred` | Lock busy. |

### Settings

None are needed. To change one, create `~/.config/aiswe/gpu-failback.env`:

| Variable | Default | Meaning |
|---|---|---|
| `AISWE_GPU_FAILBACK_FAILURES` | 3 | consecutive failed checks before failing back |
| `AISWE_GPU_FAILBACK_EXPIRY_MARGIN` | 30 | seconds before `expiresAt` to switch (0 = only once past) |
| `AISWE_GPU_FAILBACK_HEALTH_TIMEOUT` | 5 | seconds to wait for `/health` |
| `AISWE_GPU_FAILBACK_KEEP_ALIVE` | 4h | `keep_alive` for the warmed VM models |

### What it intentionally does not do

- It never submits, restarts, extends or cancels a Tillicum job, and never incurs GPU cost.
- It never opens or closes the tunnel.
- It never switches *to* GPU mode.
- It never signs in: it needs no admin password and no Duo.
- It never runs the four-condition verification; that is `stop`'s job.

After a failback, the Tillicum job may still be running and costing GPU time.
That is the case when the GPU became *unreachable* rather than expired, for
example when the tunnel dropped. Run `stop --cancel-job`, or `scancel` on
Tillicum.

### Verifying it is running

```bash
./scripts/classroom_gpu_mode.sh status
```
```bash
systemctl --user list-timers aiswe-gpu-failback.timer --no-pager
```
```bash
journalctl --user -u aiswe-gpu-failback.service -n 20 --no-pager
```

Healthy means:
- `timer RUNNING (enabled, active)`;
- a `last check` a few seconds old;
- `list-timers` showing the next run within 15 s.

---

## 12. Monitoring

### Tillicum job

On Tillicum:

```bash
./training/status_finetuned_service.sh
```

It prints:
- the published adapters;
- the application's record of the session;
- the job's state, node, elapsed time and time left;
- the service's `/health`.

It exits non-zero when no job is active or a running job fails health. With
plain Slurm:

```bash
squeue -u $USER -n css360-ft-infer
```
```bash
sacct -j <jobId> -X -o JobID,State,ExitCode,Elapsed,NodeList
```

### GPU and node usage

From the VM, through the tunnel:

```bash
curl -s http://127.0.0.1:9101/health
```

The fields to read are:
- `gpuName`, `cudaAvailable`;
- `secondsRemaining`;
- `loadedAdapters`;
- `served` (`base`/`course` counters, which rise with every answer).

On the node itself (standard Slurm; not wrapped by a repository script), from
a Tillicum login:

```bash
srun --jobid=<jobId> --overlap nvidia-smi
```

### VM CPU, RAM and Ollama

On the VM (standard Linux tools, plus each Ollama's API):

```bash
uptime && free -h
```
```bash
top -o %CPU
```
```bash
curl -s http://127.0.0.1:11434/api/ps
```
```bash
curl -s http://127.0.0.1:11435/api/ps
```
```bash
systemctl --user status aiswe-backend aiswe-finetuned aiswe-gpu-failback.timer --no-pager
```
```bash
systemctl status ollama ollama-finetuned --no-pager
```

`status` (§8) already summarises the Ollama `/api/ps` answers. The admin
endpoint `GET /api/admin/generation-queue` shows the queue's active and
waiting counts. For a measured load test outside class time, see
[classroom-capacity.md](classroom-capacity.md#measuring-the-vm).

---

## 13. Troubleshooting

### `State: PD — waiting for a node with the requested resources` (on Tillicum)

The job is queued behind other work (`PD`, reason `Resources`). Nothing is
wrong with it.
- `start_finetuned_service.sh` keeps waiting up to `ALLOC_TIMEOUT_SECONDS` (600).
- Then it exits with `Timed out … waiting for Slurm job <id> to reach RUNNING`. The job **stays queued** and will start (and cost GPU time) when a node frees up.
- Re-running the script reuses that job and keeps waiting.
- If class time is near, teach in VM mode: nothing needs doing on the VM. Cancel the queued job with `scancel <jobId>` if you no longer want it.
- `QOSMaxWallDurationPerJobLimit` is different: that job can never start, and the script says so and exits. Use `SERVICE_QOS=normal`.

### Stale GPU mode pointing at an expired job

Signs:
- `status` shows `MODE: CLASSROOM GPU` with an `expiresAt` in the past and `WARNING: GPU MODE IS ON BUT THE GPU SERVICE IS UNREACHABLE`;
- students get 503s.

With the watchdog running this resolves itself on the next check. Confirm
with `status`: `MODE: VM (normal)`, `switchedBy: gpu-failback-watchdog`,
`AUTOMATIC FAILBACK at …: expired`.

Without it, or if it is not acting:

```bash
./scripts/classroom_gpu_mode.sh stop --course css360d-fall-2026-q0ne --admin-email madamk@uw.edu
```

`stop` writes VM mode before anything else, even if the password prompt then
fails.

### Tunnel unreachable

Signs: `status` shows `Tunnel: closed` while `MODE: CLASSROOM GPU`, or
`curl -s http://127.0.0.1:9101/health` returns nothing.

- **The watchdog** fails back after 3 failed checks (`gpu_unreachable`).
- **If the job is still running** and you want the GPU back: `stop` (no `--cancel-job`), then `start` again. That opens a new tunnel, with Duo.
- **If `start` reports an SSH failure** (`SSH tunnel failed (authentication cancelled, network, or the port forward was refused)`), check that you can `ssh $USER@tillicum.hyak.uw.edu` by hand, and that the job is `R` on Tillicum.

### `MODE: VM` when you expected GPU mode

Read the lines under it:
- **`switchedBy: gpu-failback-watchdog` and `AUTOMATIC FAILBACK at …: <reason>`**: the watchdog switched. The reason and job are in `status` and `gpu-failback.log`. Run `stop` to verify the VM and clean up the tunnel and job; `start` again on a healthy job if the class needs the GPU.
- **`switchedBy: <you>`**: a `stop`, or a `start` that failed and restored VM mode (it printed `FAIL …` and `restored the previous mode (VM)`).
- **`MODE: VM (the file is invalid …; the backend ignores it)`**: the file was edited by hand. Run `stop` to rewrite it properly.
- **`Mode file: … (absent: VM mode)`**: GPU mode was never started on this VM.

### GPU mode active but the service is unreachable

`WARNING: GPU MODE IS ON BUT THE GPU SERVICE IS UNREACHABLE: students get errors.`

- Within ~50 s the watchdog should fail back. Check its section in the same `status` output: `last check … GPU check FAILED n/3`.
- If the timer is `NOT RUNNING` or `NOT INSTALLED`, or you cannot wait, run `stop` now.
- Then find out why on Tillicum: `./training/status_finetuned_service.sh`; the job may have ended or failed. See `training/logs/infer-<jobId>.err`.

### No Ollama models resident

`status` shows `ollama http://127.0.0.1:11434: nothing resident`.

- **In GPU mode this is normal.** The VM's models unload after their keep-alive while the GPU does the work.
- **In VM mode**, the first questions will each pay a model load (seconds) until they are resident. `stop` and the watchdog warm them on the way back. To warm them by hand (tag from `finetuned.env`; `:11435` is the default fine-tuned Ollama):

  ```bash
  backend/.venv/bin/python scripts/warm_classroom_models.py --base-model llama3.2:3b --finetuned-model css360d-v1:latest --keep-alive 4h
  ```

- **`DOWN` instead of `nothing resident`** means that Ollama is not answering: `systemctl status ollama` (`:11434`) or `systemctl status ollama-finetuned` (`:11435`).

### Watchdog healthy

```text
Automatic failback watchdog (aiswe-gpu-failback.timer):
  timer          RUNNING (enabled, active)
  last check     2026-10-06T17:20:15+00:00 (4s ago): VM mode, nothing to do
  last failback  none
```

In GPU mode the last check reads `GPU mode, GPU healthy (job … on …, N min left)`.
Anything else in the `timer` line:

| Line | Meaning | Fix |
|---|---|---|
| `NOT INSTALLED` | The units were never installed. | `./scripts/gpu_failback_watchdog.sh install` |
| `NOT RUNNING (disabled, inactive)` | The timer was stopped or disabled. | `install`, which enables and restarts it |
| `systemctl is not available on this host` | Not the VM. | — |
| `NOTE: … last check was N min ago` | The timer is active but runs are failing. | `journalctl --user -u aiswe-gpu-failback.service -n 20` |

---

## 14. Manual recovery

In order of preference:

1. **`stop`** (§9). It switches to VM first, then verifies. Use it in any doubt; it is safe in any state.
2. **If `stop` cannot run** (backend venv broken, Python error), remove the mode file:

   ```bash
   rm ~/.config/aiswe/generation-mode.json
   ```

   A missing file is VM mode by definition, and the backend picks that up on the next request. Then fix whatever stopped `stop`, and run it for the verification.
3. **The watchdog can be forced to act now** if its criteria are met, instead of waiting for the timer:

   ```bash
   ./scripts/gpu_failback_watchdog.sh run
   ```

### Stale session and tunnel cleanup

| Leftover | How to clean it up |
|---|---|
| **Tunnel still open in VM mode** (`status`: `VM mode, but the GPU tunnel is still open`) | `stop` closes it. By hand: `ssh -O exit -o ControlPath=$HOME/.local/state/css360-syllabus-bot/ssh-gpu-mode.sock $USER@tillicum.hyak.uw.edu`, then `rm -f ~/.local/state/css360-syllabus-bot/ssh-gpu-mode.sock ~/.local/state/css360-syllabus-bot/gpu-mode-tunnel.json`. |
| **Something else holding `127.0.0.1:9101`** | `ss -ltnp \| grep 9101` names the process; stop it, or choose another `--gpu-port` for `start`. |
| **Tillicum job still running after GPU mode ended** | `scancel <jobId>` on Tillicum, or `./training/stop_finetuned_service.sh`, which also marks the recorded session stopped. |
| **The recorded serving session** | Expires on its own at `expiresAt`, so nothing needs clearing. `start` uses the active record: if a stale one names a dead node, its health check fails and nothing is switched. Then start a new job, or pass `--node`/`--job-id` explicitly. |
| **Watchdog state** | Safe to delete (`gpu-failback-state.json`). The next check starts fresh. Keep `gpu-failback.log` as the history. |

---

## 15. Deploying from GitHub to AISWE

On the VM. The full reference is [deployment.md](deployment.md#uwb-vm).

```bash
cd ~/css360-syllabus-bot && git pull origin main
```

**Frontend** (rebuild and redeploy whenever `src/`, `package.json` or
`package-lock.json` changed):

```bash
npm ci
```
```bash
npm run build
```
```bash
sudo cp -a dist/. /usr/share/nginx/html/
```
```bash
sudo restorecon -R /usr/share/nginx/html
```
```bash
sudo nginx -t
```
```bash
sudo systemctl reload nginx
```

**Backend** (whenever `backend/` changed):

```bash
systemctl --user restart aiswe-backend
```
```bash
curl -s http://127.0.0.1:8001/api/health
```

- **Fine-tuned service:** if `training/inference_service/` changed, run `./scripts/aiswe_finetuned.sh restart`.
- **Classroom GPU scripts** (`scripts/classroom_gpu_mode.*`, `scripts/gpu_failback_watchdog.*`, `scripts/lib/`) run from the checkout and need no restart.
- **Watchdog, first time only**, or when `scripts/systemd/aiswe-gpu-failback.*` changed:

  ```bash
  ./scripts/gpu_failback_watchdog.sh run --dry-run
  ```
  ```bash
  ./scripts/gpu_failback_watchdog.sh install
  ```

  The dry run first: if a GPU session is active and already past its `expiresAt`, the watchdog's first real check fails it back.
- **Tillicum:** if `training/inference_service/` changed, `git pull` there *before* starting the next serving job; `start` refuses an older build.

Do not deploy in the middle of a class in GPU mode unless you must. A backend
restart is safe (the mode file is re-read), but a failed frontend copy is
visible to everyone.

To remove the watchdog: `./scripts/gpu_failback_watchdog.sh uninstall`. That
stops and disables the timer and removes the units; state and log are kept.
Or pause it with `systemctl --user disable --now aiswe-gpu-failback.timer`.

---

## 16. Related changes in the same release

**Second-tab "Checking your session…" forever (fixed, `adfb9f9`).**
- **The bug.** A new AISWE tab could stay on the loading screen indefinitely. Every page first asks `GET /api/auth/session`, and that check had no timeout.
  - Production Nginx speaks HTTP/1.1, so a browser opens at most six connections to the host, shared by all tabs.
  - A Compare run in another tab holds several long generation requests, and can hold all six.
  - The new tab's check then waits inside the browser without being sent.
- **The fix.** The check now gives up after **10 s**: `fetchSession({timeoutMs})`, with the limits in `src/context/session.ts`. Giving up also removes it from the browser's queue.
  - It is retried after **1 s**, then **3 s**, but only when no answer arrived at all. A reply from the backend, even an error, is final.
  - Worst case, a tab lands on sign-in after about 34 s instead of never; normally it gets through once a connection frees up.
- **The test.** `src/context/SessionContext.freshTab.test.tsx` mounts the real app like a fresh tab. It models the six-connection pool, with a fake `fetch` that honours `AbortSignal` the way a browser's does. Five of its cases fail against the old client.
- **Also fixed (`97265d8`):** signing out when the logout request fails now still navigates to sign-in.
- **Not done:** `http2 on;` in Nginx, which removes the six-connection limit itself, is the remaining infrastructure fix.

**npm security updates** (`ca66f1f`):
- **Non-breaking updates.** `npm audit fix` updated 16 packages within their ranges, including `react-router`/`react-router-dom` 7.18.4.
- **Vitest 5.0.3.** Vitest was upgraded to 5.0.3, which removes the vulnerable `tinypool` and `@vitest/mocker`.
- **Test files.** Three test files that use Node built-ins now declare `/// <reference types="node" />`, which Vitest 3 used to supply implicitly.
- **Result.** `npm audit` reports **0 vulnerabilities**, and so does `npm audit --omit=dev`.

---

## 17. Known limitations

- **The GPU server is serial.** One GPU, one answer at a time: adapter switching and generation share one lock. The backend's queue also allows one generation at a time in both modes. GPU mode makes each answer faster; it does not run answers in parallel.
- **30 simultaneous students is improved, not proven.** A 5-student run passed locally against a stand-in. Thirty simultaneous four-condition comparisons would need batched serving (vLLM with multi-LoRA, or Ollama with `OLLAMA_NUM_PARALLEL` on the GPU), which is a separate change.
- **The VM fallback is slower.** About 10 s per answer on CPU, and about three simultaneous comparisons. After a failback a large class waits longer; requests beyond the queue's limits are refused with a "busy" message rather than hanging.
- **The watchdog never starts GPU work.** It does not submit, extend or restart jobs. Getting back to the GPU after a failback is a person running §6 and §7.
- **GPU cost continues until the job ends.** Leaving GPU mode does not cancel the job, whether by `stop` without `--cancel-job` or by the watchdog. Cancel it, or it runs to its wall clock.
- **Errors during the gap.** Between the GPU disappearing and the failback (up to ~50 s when there is no expiry to anticipate), students' requests fail once.
- **A changed job end is not seen.** The watchdog trusts the `expiresAt` recorded at `start`. If a job's time limit is changed, run `stop` and `start` again.
- **The engine can change mid-class.** A failback switches engines during a class. Every answer's `model` says which engine answered (`… (Tillicum GPU)` or the Ollama tag), the backend log line carries `engine=gpu|vm`, and the failback log gives the moment. Record it if it matters to a study condition.
- **systemd on the VM.** The watchdog's systemd behaviour was tested against a recording fake `systemctl`; confirm `list-timers` and the journal on the VM after the first install.

---

## 18. What stays identical between modes

**The same in both modes:**
- every prompt (the backend builds it either way);
- the course version;
- the output cap (128 tokens in class);
- greedy decoding;
- the 1.05 repetition penalty over the whole context;
- the seed;
- the 4096-token context;
- the stop tokens;
- the prompt *format*.

The GPU service renders each prompt exactly as the VM's Ollama renders it.
This was checked against Ollama 0.33.2: identical text, and identical prompt
token counts (31/43/68 for the three probes). `start` repeats the token-count
check live against the VM's Ollama.

The backend refuses any GPU answer whose reported decoding differs from what
it would have sent the VM. It also refuses an answer from an older GPU service
build that reports no decoding at all: that build ignored the 128-token cap
and used Transformers' chat template, which adds a "Today Date" line.

**Not identical:** the numbers inside the model. The GPU loads the checkpoint
with bitsandbytes NF4 and the VM uses GGUF Q4_K_M, so wording can differ
slightly between modes.

---

## 19. What has been tested

### GPU mode, locally (2026-10-01)

The test setup, on a laptop:
- the real backend, through `scripts/classroom_load_harness.py`;
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

The stand-in's answers are meaningless (random weights). Answer quality and
GPU speed are what a run on the real VM and Tillicum checks.

### Watchdog (2026-10-01)

`backend/tests/test_gpu_failback_watchdog.py` (55 tests), plus two runs of the
real wrapper script against a temporary mode file, a stub `/health` server and
a local Ollama.

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

Test commands:

```bash
backend/.venv/bin/python -m pytest backend/tests/test_gpu_failback_watchdog.py backend/tests/test_classroom_gpu_mode_script.py backend/tests/test_generation_mode.py -q
```
```bash
npx vitest run --config vitest.config.ts src/context/SessionContext.freshTab.test.tsx src/context/SessionContext.test.tsx
```
