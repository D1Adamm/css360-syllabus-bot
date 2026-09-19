# Per-course fine-tuned inference service

One HTTP contract — `GET /health`, `GET /courses`, `POST /generate` — with two
implementations. The backend (`backend/app/finetuned_client.py`) talks to
whichever one is listening at `FINETUNED_SERVICE_URL` and cannot tell them
apart.

| | `ollama_service.py` (UWB VM) | `app.py` (Tillicum) |
| --- | --- | --- |
| Runs on | the VM's CPU, through the VM's own Ollama | one Tillicum GPU, under Slurm |
| Model | `llama3.2:3b` with the course's adapter, one Ollama model per course and version | `meta-llama/Llama-3.2-3B-Instruct` in 4-bit, one PEFT adapter per course |
| Course → model | the `FINETUNED_OLLAMA_MODELS` mapping | `<SERVING_ROOT>/<courseId>/<version>/adapter` |
| Needs a person | no | yes: SSH + Duo to open the tunnel |
| Listens on | `127.0.0.1:9001`, nothing else | compute node `:8001`, tunnelled to `127.0.0.1:9001` |
| Python deps | FastAPI, uvicorn, pydantic, httpx (all in `backend/.venv`) | plus torch, Transformers, PEFT, bitsandbytes |
| Role | current path (CSS 360 v2) | fallback and reference implementation |

Both refuse a course they have nothing for, and both echo the course and
version they answered with; the backend discards a response naming a different
course. **Never run both at once**: they claim the same `127.0.0.1:9001`.

## Serving on the UWB VM through Ollama

### 1) Once per adapter: build the Ollama model

Training writes a PEFT adapter directory. Ollama loads a LoRA adapter from
GGUF, so convert it once (llama.cpp's `convert_lora_to_gguf.py`, against the
same base model) and build a model from it:

```text
# Modelfile
FROM llama3.2:3b
ADAPTER css360-v2-lora.gguf
```

```bash
ollama create css360-ft-v2 -f Modelfile
ollama list          # css360-ft-v2:latest
```

No `PARAMETER` lines are needed: the service sends the decoding settings with
every request, and request options override the Modelfile.

`css<number>-ft-v<N>` is a convention, not something the service enforces. One
Ollama model per course *and* version, so a promotion is a new model rather
than a silent overwrite of the one being served.

### 2) Map courses to models

`FINETUNED_OLLAMA_MODELS` holds one entry per course and version,
`courseId@vN=ollamaModel`, comma separated. Course ids and versions are
validated with the same rules the Tillicum service applies to a serving path.
The version is the one registered in PostgreSQL for that course — the backend
sends it with every request — and a version this host does not have is refused,
never answered by a different one.

```bash
FINETUNED_OLLAMA_MODELS="css-360-winter-2026-a7rp@v2=css360-ft-v2:latest"

# Later, CSS 350 is one more entry:
FINETUNED_OLLAMA_MODELS="css-360-winter-2026-a7rp@v2=css360-ft-v2:latest,css-350-spring-2026-n3h9@v1=css350-ft-v1:latest"
```

Nothing is mapped by default. With an empty mapping `/health` reports no
courses and every `/generate` is a 409. A malformed entry stops the service at
startup rather than silently dropping a course.

### 3) Run it

From the repository root on the VM, with the backend's virtualenv. It already
has everything this service imports; no torch, Transformers, PEFT or
bitsandbytes is installed or needed.

```bash
cd training/inference_service
FINETUNED_OLLAMA_MODELS="css-360-winter-2026-a7rp@v2=css360-ft-v2:latest" \
  ../../backend/.venv/bin/python ollama_service.py
```

It binds `127.0.0.1:9001` and nothing else; there is no host override. For an
always-on service, a user unit beside `aiswe-backend`:

```ini
# ~/.config/systemd/user/aiswe-finetuned.service
[Unit]
Description=Per-course fine-tuned inference (local Ollama)
After=network.target

[Service]
WorkingDirectory=%h/css360-syllabus-bot/training/inference_service
Environment=FINETUNED_OLLAMA_MODELS=css-360-winter-2026-a7rp@v2=css360-ft-v2:latest
Environment=FINETUNED_KEEP_ALIVE=30m
ExecStart=%h/css360-syllabus-bot/backend/.venv/bin/python ollama_service.py
Restart=on-failure

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now aiswe-finetuned
```

Environment:

| Variable | Default | Meaning |
| --- | --- | --- |
| `FINETUNED_OLLAMA_MODELS` | empty | The course → model mapping above |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | The VM's Ollama |
| `INFERENCE_PORT` | `9001` | Loopback port; must match the backend's `FINETUNED_SERVICE_URL` |
| `FINETUNED_OLLAMA_TIMEOUT_SECONDS` | `120` | Per-generation Ollama timeout |
| `FINETUNED_NUM_CTX` | `4096` | Context window. Ollama truncates a longer prompt from the front, which for Fine-Tuned + RAG would drop the grounding rules first |
| `FINETUNED_KEEP_ALIVE` | Ollama's default (5 min) | How long a model stays resident after a request. Set e.g. `30m` before a class so the first question does not pay the model load |
| `FINETUNED_BASE_MODEL` | `llama3.2:3b` | Reported as `model` on `/health` |

### 4) Wire the backend

`backend/.env` on the VM:

```bash
FINETUNED_SERVICE_URL=http://127.0.0.1:9001
```

That is the value the tunnel script used to write, so an existing `.env` may
already say it — but the tunnel has to be closed first
(`./scripts/stop_finetuned_tunnel.sh`), or this service cannot take the port.
Restart `aiswe-backend` if the value changed.

### 5) Check

```bash
curl -s http://127.0.0.1:9001/health
curl -s -X POST http://127.0.0.1:9001/generate \
  -H "Content-Type: application/json" \
  -d '{"courseId":"css-360-winter-2026-a7rp","modelVersion":"v2","question":"When does the course meet?"}'
curl -s http://127.0.0.1:8001/api/fine-tuned/health      # through the backend (admin)
```

`/health` answers 200 whether or not Ollama is up. `status` is `ok` or
`unavailable`; `adapterLoaded` is true only when Ollama answered and at least
one mapped model exists there; `courses` lists what can be answered right now;
`models` lists every mapping entry with `available` saying whether Ollama has
it. The readiness rule the operator scripts apply — `status == ok` and
`adapterLoaded` — is unchanged.

Responses have the GPU service's shape, with three differences:

- `model` on `/generate` is the Ollama model that answered
  (`css360-ft-v2:latest`), not the base model id.
- `generationSeconds` is wall clock for the Ollama call. After an idle period
  that includes loading the model.
- `secondsRemaining` is always null: nothing expires.

### What is kept the same as the GPU service

The GPU service wraps the question as one user turn with the tokenizer's chat
template and decodes greedily: 256 new tokens, repetition penalty 1.05 over the
whole sequence, seed 360. This service sends the same single user turn to
`/api/chat`, where the model's own Llama 3.2 template is applied, with
`temperature 0`, `num_predict 256`, `repeat_penalty 1.05` over the whole
context window, and `seed 360`. A Fine-Tuned + RAG prompt travels verbatim as
that user turn, exactly as it did to Tillicum.

Answers are comparable, not bit-identical: the two stacks quantise the base
differently (4-bit NF4 there, the Q4 GGUF Ollama ships here), and the two chat
templates differ in their fixed system header.

### Tests

```bash
cd training/inference_service
../../backend/.venv/bin/python -m unittest test_ollama_service.py -v

# The seam with the backend client, from the backend suite:
cd ../../backend && .venv/bin/python -m pytest -q tests/test_finetuned_local_service_contract.py
```

Neither needs Ollama.

---

## The benchmark-only service (`benchmark_service.py`)

A third process, for the CSS 360 controlled benchmark and nothing else. The
production service above maps *courses and registered versions* to models and
answers students; this one maps a fixed set of *experiment aliases* to Ollama
tags and answers only the backend's research route
(`POST /api/research/css360/benchmark/*`, off by default). The two v4
adapters become servable here without ever entering `FINETUNED_OLLAMA_MODELS`.

| | `benchmark_service.py` |
| --- | --- |
| Listens on | `127.0.0.1:9002`, no host override; runs beside `:9001` |
| Aliases | `base`, `v2`, `v3`, `v4_vm`, `v4_tillicum`, fixed in code |
| Alias → tag | `CSS360_BENCHMARK_OLLAMA_MODELS=base=llama3.2:3b,v2=css360-ft-v2:latest,…`; an unknown alias or malformed entry stops startup |
| Decoding | the production service's own `build_generation_options`, with `num_ctx` pinned to 4096 (`FINETUNED_NUM_CTX` is ignored), reported on every response |
| `POST /generate` | `{"alias": "v4_vm", "prompt": "…"}` and nothing else; answers `alias`, `model`, `modelDigest` (the tag's manifest digest from `/api/tags`, looked up per generation), `answer`, `promptSha256`, `options`, `generationSeconds`, `ollama` timings |
| `GET /health` | every alias with `mapped`, `available` and the `digest` Ollama holds for its tag |
| Failure codes | 409 `alias_not_mapped` / `model_missing`, 503 `ollama_*`, 502 `ollama_rejected` / `malformed_ollama_response` |

```bash
cd training/inference_service
CSS360_BENCHMARK_OLLAMA_MODELS="base=llama3.2:3b,v2=css360-ft-v2:latest,v3=css360-cpu-v3-test:latest,v4_vm=css360-v4-test:latest,v4_tillicum=css360-v4-tillicum-test:latest" \
  ../../backend/.venv/bin/python benchmark_service.py
```

Environment: `CSS360_BENCHMARK_OLLAMA_MODELS`, `OLLAMA_BASE_URL` (shared),
`BENCHMARK_INFERENCE_PORT` (9002), `CSS360_BENCHMARK_OLLAMA_TIMEOUT_SECONDS`
(120), `CSS360_BENCHMARK_KEEP_ALIVE` (unset). The production variables
`INFERENCE_PORT`, `FINETUNED_NUM_CTX`, `FINETUNED_KEEP_ALIVE` and
`FINETUNED_OLLAMA_TIMEOUT_SECONDS` are not read.

Tests: `../../backend/.venv/bin/python -m pytest -q test_benchmark_service.py`
here, and `tests/test_research_benchmark_service_contract.py` in the backend
suite for the seam with the backend client. Design, configuration and the
deployment steps: `docs/css360-benchmark-endpoint.md`.

## The Tillicum GPU service (`app.py`)

Kept as the fallback and the reference implementation. Everything below is
about it.

- Base model: `meta-llama/Llama-3.2-3B-Instruct`, loaded once
- Adapters: `<SERVING_ROOT>/<courseId>/<version>/adapter`, attached on top
- Does **not** merge LoRA adapters into the base model
- Slurm job name: `css360-ft-infer`, under the `debug` QOS by default
- A session is **bounded**, and its length is bounded by the QOS: `debug` caps a
  job at **1 hour**, which is both the default and the maximum under it.
  Workshop/research infrastructure, not always-on production hosting.

## Why per course

Training is per course. The service used to load exactly one adapter from one
path with no course identity anywhere in it, so publishing CSS 360 replaced
whatever CSS 350 was being served with — and no request carried enough
information for anything to notice.

Now every request names its course, the adapter is resolved from a validated
course id and version, and the response echoes back which course and version
actually answered. The backend discards a response whose course does not match
what it asked for.

Several adapters share one base model because that is both the simplest and the
cheapest arrangement: a LoRA adapter here is ~47 MB against a ~2.5 GB 4-bit
base, so a second course costs a rounding error of GPU memory. One process per
course would cost a whole GPU per course on a shared cluster; reloading a single
adapter per request would put a multi-second load in front of every question.

## Adapter format

Training writes `adapter_config.json` and `adapter_model.safetensors` through
PEFT's `save_pretrained`. That is exactly what `PeftModel.load_adapter` reads.
There is no conversion step and none is needed — no GGUF, no merged checkpoint.

## Architecture (Tillicum fallback)

```text
Browser  (asks as CSS 350)
  -> UWB VM FastAPI (aiswe.uwb.edu, :8001)
     -> Base / RAG          via local Ollama on the VM
     -> Fine-Tuned          resolve CSS 350's current version from PostgreSQL
     -> Fine-Tuned + RAG    retrieve CSS 350 chunks, then the same resolution
           -> FINETUNED_SERVICE_URL=http://127.0.0.1:9001
              -> SSH port forward through tillicum.hyak.uw.edu
                 -> Slurm compute node:8001  (hostname changes each job)
                    -> Llama 3.2 3B + the CSS 350 adapter, selected per request
                    -> response says courseId=css-350-…, modelVersion=v1
     <- refused if the response names a different course
```

On the current path the same `127.0.0.1:9001` is `ollama_service.py` on the VM
itself, and the arrow into Tillicum does not exist.

Fine-Tuned + RAG retrieves syllabus chunks on the UWB VM, then sends the grounded
prompt to the same remote fine-tuned service as Fine-Tuned.

The Slurm compute hostname is **not** stable. Helper scripts discover it from
`squeue` and pass it into the VM tunnel command. Do not hardcode nodes like `g001`.

## Quick start (admin)

### 1) Tillicum — start or reuse the GPU job

```bash
ssh $USER@tillicum.hyak.uw.edu
cd /gpfs/projects/simswe/$USER/css360-syllabus-bot
./training/start_finetuned_service.sh          # 1 hour, the most `debug` allows
```

This submits `training/inference_service/serve.slurm` only if no active
`css360-ft-infer` job exists, waits for allocation and for the base model to
load, records the session — node, port, expiry, published courses — with the
application, and prints what is being served.

Session length is capped by the configured QOS. A request over the ceiling is
refused here rather than left pending forever; `SERVICE_QOS=normal` submits under
a QOS that permits longer sessions.

The session ends when the Slurm allocation does. A dropped login session, a
closed laptop, or a forgotten stop command all resolve themselves at exactly the
moment the GPU is released.

Inspect without starting:

```bash
./training/status_finetuned_service.sh
```

### 2) UWB VM — open the tunnel and wire the backend

```bash
ssh <you>@aiswe.uwb.edu
cd ~/css360-syllabus-bot
./scripts/start_finetuned_tunnel.sh --from-backend
```

`--from-backend` looks the compute node up from the session Tillicum recorded,
rather than the operator reading a hostname off one machine and typing it into
another. Passing a hostname explicitly still works.

This will:

1. Open `localhost:9001 -> <NODE>:8001` via `tillicum.hyak.uw.edu`
2. Require interactive UW / Duo authentication (not bypassed; credentials are not stored)
3. Set `FINETUNED_SERVICE_URL=http://127.0.0.1:9001` in `backend/.env`
4. Restart the user systemd unit `aiswe-backend`
5. Verify `/api/health` and `/api/fine-tuned/health` on the backend

### 3) Inspect / stop

On the UWB VM:

```bash
./scripts/status_finetuned_tunnel.sh
./scripts/stop_finetuned_tunnel.sh
```

On Tillicum (stops GPU billing for this job):

```bash
./training/stop_finetuned_service.sh
```

Closing the tunnel disables Fine-Tuned and Fine-Tuned + RAG on the website, but
leaves Base and RAG unaffected. Cancelling the Slurm job stops the GPU allocation.
Use `hyakusage` on Tillicum to inspect GPU usage / cost / credits.

## Important limitations

- **Duo is still manual** for establishing the SSH tunnel. Helpers never store UW passwords or automate interactive auth. This is the one remaining manual step in the serving path, and it is manual because opening the tunnel authenticates to UW.
- A session has a **bounded wall time**, and the bound comes from the QOS rather than from preference. Under the default `debug` QOS that is 1 hour. Asking for longer is refused before submission, because Slurm would otherwise accept the job and leave it `PENDING` forever with `QOSMaxWallDurationPerJobLimit` — which looks like a busy cluster rather than a request that can never be satisfied. For a longer sitting, submit under a QOS that permits it: `SERVICE_QOS=normal ./training/start_finetuned_service.sh --hours 3`. This is intentional workshop/research infrastructure, not permanent production GPU hosting.
- A course with no published adapter gets a clear 409, not another course's answer.
- The website does **not** submit GPU jobs automatically.
- Helpers refuse to submit a second `css360-ft-infer` job when one is already PENDING/RUNNING.
- Compute node hostnames change between jobs.

## Setup (venv)

Prefer the shared QLoRA environment (name is `qlora`, not `css360-qlora`):

```bash
source /gpfs/projects/simswe/$USER/venvs/qlora/bin/activate
pip install -r training/inference_service/requirements.txt
```

`serve.slurm` will use `training/.venv` if present, otherwise the shared `qlora`
venv, and fails immediately if neither exists.

Hugging Face auth uses the same paths as training:

```bash
export HF_HOME=/gpfs/projects/simswe/$USER/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_TOKEN_PATH=$HF_HOME/token
```

Optional overrides:

```bash
export SERVING_ROOT=/gpfs/projects/simswe/$USER/training_outputs/serving
export MODEL_ID=meta-llama/Llama-3.2-3B-Instruct
export INFERENCE_PORT=8001
export MAX_LOADED_ADAPTERS=4
```

## Manual launch (without helpers)

From the repository root on Tillicum:

```bash
mkdir -p training/logs
sbatch training/inference_service/serve.slurm
squeue -u $USER
```

Prefer `squeue` for the node name. Then on the UWB VM:

```bash
./scripts/start_finetuned_tunnel.sh <NODE>
```

## Curl examples

Health on the compute node (from Tillicum login / allocated network):

```bash
curl http://NODE:8001/health
```

Generate one answer:

```bash
curl -sS -X POST "http://NODE:8001/generate" \
  -H "Content-Type: application/json" \
  -d '{"courseId":"css-350-spring-2026-n3h9","question":"When does the course meet?"}'
```

`courseId` is required. There is no course-agnostic fine-tuned model, so a
request without one would be asking the service to choose a course.

Example response shape:

```json
{
  "answer": "...",
  "model": "meta-llama/Llama-3.2-3B-Instruct",
  "courseId": "css-350-spring-2026-n3h9",
  "modelVersion": "v1",
  "adapterLoaded": true,
  "generationSeconds": 1.23
}
```

Which courses this session can answer for:

```bash
curl http://NODE:8001/courses
```

Via the UWB tunnel after helpers succeed:

```bash
# The tunnel to the Tillicum service — that service only has /health.
curl http://127.0.0.1:9001/health

# The UWB backend — Nginx proxies only /api/, so /api/... is the canonical form
# and is the one that works both directly and through the proxy.
curl http://127.0.0.1:8001/api/fine-tuned/health
```

## Local helper tests

```bash
# Deploy helper unit tests (sbatch/squeue/.env/hostname parsing)
python -m unittest training.test_finetuned_deploy_helpers -v

# Inference service helper tests
cd training/inference_service
python -m unittest test_app_helpers.py -v
```
