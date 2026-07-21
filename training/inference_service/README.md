# CSS 360 fine-tuned inference service

Small FastAPI service that serves the trained QLoRA adapter on one GPU.

- Base model: `meta-llama/Llama-3.2-3B-Instruct`
- Adapter (default): `/gpfs/projects/simswe/$USER/training_outputs/css-360-qlora/adapter`
- Does **not** merge the LoRA adapter into the base model
- Does **not** use RAG
- Not wired into the main backend/frontend yet

## Setup

Use the existing QLoRA virtual environment, then install service extras:

```bash
source /gpfs/projects/simswe/$USER/venvs/css360-qlora/bin/activate
pip install -r training/inference_service/requirements.txt
```

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

## Launch on Tillicum

From the repository root:

```bash
mkdir -p training/logs
sbatch training/inference_service/serve.slurm
squeue -u $USER
```

Find the compute node hostname in `training/logs/infer-<JOB_ID>.out` (look for
`Node hostname:` / `Listening port:`).

## Curl examples

Health:

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

Replace `NODE` with the Slurm compute hostname from the job log.

## Local helper tests

```bash
cd training/inference_service
python -m unittest test_app_helpers.py -v
```
