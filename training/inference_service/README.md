# Per-course fine-tuned inference service

Small FastAPI service that serves each course's trained QLoRA adapter on one
Tillicum GPU.

- Base model: `meta-llama/Llama-3.2-3B-Instruct`, loaded once
- Adapters: `<SERVING_ROOT>/<courseId>/<version>/adapter`, attached on top
- Does **not** merge LoRA adapters into the base model
- Slurm job name: `css360-ft-infer` (debug QOS; a bounded session, 2 hours by
  default — workshop/research infrastructure, not always-on production hosting)

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

## Architecture (current)

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

Fine-Tuned + RAG retrieves syllabus chunks on the UWB VM, then sends the grounded
prompt to the same remote fine-tuned service as Fine-Tuned.

The Slurm compute hostname is **not** stable. Helper scripts discover it from
`squeue` and pass it into the VM tunnel command. Do not hardcode nodes like `g001`.

## Quick start (admin)

### 1) Tillicum — start or reuse the GPU job

```bash
ssh $USER@tillicum.hyak.uw.edu
cd /gpfs/projects/simswe/$USER/css360-syllabus-bot
./training/start_finetuned_service.sh          # 2 hours; --hours 3 for longer
```

This submits `training/inference_service/serve.slurm` only if no active
`css360-ft-infer` job exists, waits for allocation and for the base model to
load, records the session — node, port, expiry, published courses — with the
application, and prints what is being served.

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
- A session has a **bounded wall time** (2 hours by default, `--hours` to extend, 8 hours maximum). This is intentional workshop/research infrastructure, not permanent production GPU hosting.
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
