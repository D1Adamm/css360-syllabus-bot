#!/bin/bash
# Launch the CSS 360 fine-tuned inference service on one Tillicum GPU.
# Submit from the repository root:
#   sbatch training/inference_service/serve.slurm
#
# After the job starts, find the compute node hostname in the log, then:
#   curl http://NODE:8001/health
#   curl -X POST http://NODE:8001/generate -H 'Content-Type: application/json' \
#     -d '{"question":"When and where does CSS 360 meet?"}'

#SBATCH --job-name=css360-ft-infer
#SBATCH --account=simswe
#SBATCH --qos=debug
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=training/logs/infer-%j.out
#SBATCH --error=training/logs/infer-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

mkdir -p training/logs
mkdir -p "/gpfs/projects/simswe/${USER}/hf_cache"

export HF_HOME="/gpfs/projects/simswe/${USER}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_TOKEN_PATH="${HF_HOME}/token"
export ADAPTER_PATH="${ADAPTER_PATH:-/gpfs/projects/simswe/${USER}/training_outputs/css-360-qlora/adapter}"
export MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-3B-Instruct}"
export INFERENCE_HOST="${INFERENCE_HOST:-0.0.0.0}"
export INFERENCE_PORT="${INFERENCE_PORT:-8001}"
export SLURM_JOB_ID="${SLURM_JOB_ID:-}"

test -s "$HF_TOKEN_PATH" || {
  echo "Missing Hugging Face token at $HF_TOKEN_PATH" >&2
  exit 1
}

test -d "$ADAPTER_PATH" || {
  echo "Adapter path does not exist: $ADAPTER_PATH" >&2
  exit 1
}

echo "=== Job info ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
echo "Listening port: ${INFERENCE_PORT}"
echo "Adapter: ${ADAPTER_PATH}"
echo "Model: ${MODEL_ID}"
echo "Repo: ${REPO_ROOT}"
date -u

echo "=== nvidia-smi ==="
nvidia-smi

if [[ -f "${REPO_ROOT}/training/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/training/.venv/bin/activate"
elif [[ -f "/gpfs/projects/simswe/${USER}/venvs/css360-qlora/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "/gpfs/projects/simswe/${USER}/venvs/css360-qlora/bin/activate"
fi

python -m pip install -q -r training/inference_service/requirements.txt

echo "=== Python / package versions ==="
python -V
python - <<'PY'
import importlib
mods = ["torch", "transformers", "peft", "bitsandbytes", "fastapi", "uvicorn"]
for name in mods:
    try:
        mod = importlib.import_module(name)
        print(f"{name}={getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:  # noqa: BLE001
        print(f"{name}=IMPORT_ERROR ({exc})")
PY

echo "Node hostname: $(hostname)"
echo "Listening port: ${INFERENCE_PORT}"
echo "Service URL hint: http://$(hostname):${INFERENCE_PORT}"

cd training/inference_service
exec python app.py
