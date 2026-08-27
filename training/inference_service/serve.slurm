#!/bin/bash
# Launch the per-course fine-tuned inference service on one Tillicum GPU.
# Preferred (from repository root on Tillicum):
#   ./training/start_finetuned_service.sh
#   ./training/start_finetuned_service.sh --hours 3
#
# Manual submit:
#   sbatch --time=02:00:00 training/inference_service/serve.slurm
# Then discover NODE via squeue (not hardcoded), and on aiswe.uwb.edu:
#   ./scripts/start_finetuned_tunnel.sh --from-backend
#
# The service loads the base model once and attaches one LoRA adapter per course
# from SERVING_ROOT, choosing between them per request. There is no single
# ADAPTER_PATH any more: that variable had no course in it, so publishing one
# course replaced whatever every other course was being served with.

#SBATCH --job-name=css360-ft-infer
#SBATCH --account=simswe
#SBATCH --qos=debug
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
# Default only. start_finetuned_service.sh overrides it with `sbatch --time` to
# give the session an explicit, bounded lifetime (2 hours by default). The
# allocation ending is what makes a session expire without anyone having to
# remember to stop it — including when the operator's login session drops.
#SBATCH --time=02:00:00
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
export SERVING_ROOT="${SERVING_ROOT:-/gpfs/projects/simswe/${USER}/training_outputs/serving}"
export MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-3B-Instruct}"
export MAX_LOADED_ADAPTERS="${MAX_LOADED_ADAPTERS:-4}"
export INFERENCE_HOST="${INFERENCE_HOST:-0.0.0.0}"
export INFERENCE_PORT="${INFERENCE_PORT:-8001}"
export SLURM_JOB_ID="${SLURM_JOB_ID:-}"

test -s "$HF_TOKEN_PATH" || {
  echo "Missing Hugging Face token at $HF_TOKEN_PATH" >&2
  exit 1
}

# The serving root is created rather than required. A service with no published
# course still answers /health and returns a clear 409 for every course, which
# is a better failure than a GPU allocation that refuses to start: the operator
# publishes an adapter and retries the request instead of re-queueing a job.
mkdir -p "${SERVING_ROOT}"

# The moment this allocation ends, from the wall clock Slurm actually gave it.
# Reported on /health so a caller can see a session expiring rather than
# discover it as a dropped connection.
if command -v squeue >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  REMAINING="$(squeue -j "${SLURM_JOB_ID}" -h -o '%L' 2>/dev/null | head -n 1 || true)"
  if [[ -n "${REMAINING}" ]]; then
    DEADLINE="$(python3 "${REPO_ROOT}/scripts/lib/slurm_time_left.py" "${REMAINING}" 2>/dev/null || true)"
    if [[ -n "${DEADLINE}" ]]; then
      export SERVICE_DEADLINE_EPOCH="${DEADLINE}"
    fi
  fi
fi

echo "=== Job info ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
echo "Listening port: ${INFERENCE_PORT}"
echo "Serving root: ${SERVING_ROOT}"
echo "Model: ${MODEL_ID}"
echo "Max loaded adapters: ${MAX_LOADED_ADAPTERS}"
echo "Session ends (unix): ${SERVICE_DEADLINE_EPOCH:-unknown}"
echo "Repo: ${REPO_ROOT}"
date -u

echo "=== nvidia-smi ==="
nvidia-smi

QLORA_VENV="/gpfs/projects/simswe/${USER}/venvs/qlora/bin/activate"
if [[ -f "${REPO_ROOT}/training/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/training/.venv/bin/activate"
  echo "Using venv: ${REPO_ROOT}/training/.venv"
elif [[ -f "${QLORA_VENV}" ]]; then
  # shellcheck disable=SC1091
  source "${QLORA_VENV}"
  echo "Using venv: /gpfs/projects/simswe/${USER}/venvs/qlora"
else
  echo "No qlora virtual environment found." >&2
  echo "Checked:" >&2
  echo "  - ${REPO_ROOT}/training/.venv/bin/activate" >&2
  echo "  - ${QLORA_VENV}" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python not found after activating the qlora virtual environment." >&2
  exit 1
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

echo "=== Published course adapters ==="
python3 "${REPO_ROOT}/scripts/lib/list_published_adapters.py" "${SERVING_ROOT}" || true

echo "Node hostname: $(hostname)"
echo "Listening port: ${INFERENCE_PORT}"
echo "Service URL hint: http://$(hostname):${INFERENCE_PORT}"

cd training/inference_service
exec python app.py
