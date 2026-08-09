# CSS 360 fine-tuned inference service

Small FastAPI service that serves the trained QLoRA adapter on one Tillicum GPU.

- Base model: `meta-llama/Llama-3.2-3B-Instruct`
- Adapter (default): `/gpfs/projects/simswe/$USER/training_outputs/css-360-qlora/adapter`
- Does **not** merge the LoRA adapter into the base model
- Slurm job name: `css360-ft-infer` (debug QOS, 1 hour — workshop/research prototype, not always-on production)

## Architecture (current)

```text
Browser
  -> UWB VM FastAPI (aiswe.uwb.edu, :8001)
     -> Base / RAG          via local Ollama on the VM
     -> Fine-Tuned
     -> Fine-Tuned + RAG    via FINETUNED_SERVICE_URL=http://127.0.0.1:9001
           -> SSH port forward through tillicum.hyak.uw.edu
              -> Slurm compute node:8001  (hostname changes each job, e.g. g001, g014)
                 -> Llama 3.2 3B + QLoRA adapter
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
./training/start_finetuned_service.sh
```

This submits `training/inference_service/serve.slurm` only if no active
`css360-ft-infer` job exists, waits for allocation + `/health`
(`status=ok`, `adapterLoaded=true`), then prints the compute node and the
exact VM command to run next.

Inspect without starting:

```bash
./training/status_finetuned_service.sh
```

### 2) UWB VM — open the tunnel and wire the backend

```bash
ssh <you>@aiswe.uwb.edu
cd ~/css360-syllabus-bot
./scripts/start_finetuned_tunnel.sh <NODE>
```

Replace `<NODE>` with the hostname printed by the Tillicum helper (from `squeue`,
not from guesswork).

This will:

1. Open `localhost:9001 -> <NODE>:8001` via `tillicum.hyak.uw.edu`
2. Require interactive UW / Duo authentication (not bypassed; credentials are not stored)
3. Set `FINETUNED_SERVICE_URL=http://127.0.0.1:9001` in `backend/.env`
4. Restart the user systemd unit `aiswe-backend`
5. Verify `/health` and `/fine-tuned/health`

### 3) Inspect / stop

On the UWB VM:

```bash
./scripts/status_finetuned_tunnel.sh
./scripts/stop_finetuned_tunnel.sh
```

On Tillicum (stops GPU billing for this job):

```bash
scancel <JOB_ID>
```

Closing the tunnel disables Fine-Tuned and Fine-Tuned + RAG on the website, but
leaves Base and RAG unaffected. Cancelling the Slurm job stops the GPU allocation.
Use `hyakusage` on Tillicum to inspect GPU usage / cost / credits.

## Important limitations

- **Duo is still manual** for establishing the SSH tunnel. Helpers never store UW passwords or automate interactive auth.
- The current debug job has a **one-hour** wall time. This is intentional workshop/research infrastructure, not permanent production GPU hosting.
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
export ADAPTER_PATH=/gpfs/projects/simswe/$USER/training_outputs/css-360-qlora/adapter
export MODEL_ID=meta-llama/Llama-3.2-3B-Instruct
export INFERENCE_PORT=8001
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
  -d '{"question":"When and where does CSS 360 meet?"}'
```

Example response shape:

```json
{
  "answer": "...",
  "model": "meta-llama/Llama-3.2-3B-Instruct",
  "adapterLoaded": true,
  "generationSeconds": 1.23
}
```

Via the UWB tunnel after helpers succeed:

```bash
curl http://127.0.0.1:9001/health
curl http://127.0.0.1:8001/fine-tuned/health
```

## Local helper tests

```bash
# Deploy helper unit tests (sbatch/squeue/.env/hostname parsing)
python -m unittest training.test_finetuned_deploy_helpers -v

# Inference service helper tests
cd training/inference_service
python -m unittest test_app_helpers.py -v
```
