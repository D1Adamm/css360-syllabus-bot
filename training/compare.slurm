#!/bin/bash
# Compare base Llama 3.2 3B Instruct vs a QLoRA adapter on held-out questions.
# Submit from the repository root:
#   sbatch training/compare.slurm
#
# Environment overrides:
#   TRAIN_FILE, VAL_FILE, ADAPTER_PATH, COMPARISON_OUTPUT_DIR

#SBATCH --job-name=css360-qlora-compare
#SBATCH --account=simswe
#SBATCH --qos=debug
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:45:00
#SBATCH --output=training/logs/compare-%j.out
#SBATCH --error=training/logs/compare-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

mkdir -p training/logs

export HF_HOME="/gpfs/projects/simswe/${USER}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_TOKEN_PATH="${HF_HOME}/token"
export TRAIN_GPU_COUNT="${SLURM_GPUS_ON_NODE:-1}"
export SLURM_JOB_ID="${SLURM_JOB_ID:-}"

mkdir -p "${HF_HOME}"
mkdir -p "${HF_HUB_CACHE}"

test -s "$HF_TOKEN_PATH" || {
  echo "Missing Hugging Face token at $HF_TOKEN_PATH" >&2
  exit 1
}

TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/data/exports/css-360-winter-2026-a7rp/train.jsonl}"
VAL_FILE="${VAL_FILE:-${REPO_ROOT}/data/exports/css-360-winter-2026-a7rp/validation.jsonl}"
HELDOUT_FILE="${HELDOUT_FILE:-${REPO_ROOT}/training/heldout_questions.json}"
ADAPTER_PATH="${ADAPTER_PATH:-/gpfs/projects/simswe/${USER}/training_outputs/css-360-qlora/adapter}"
OUT_DIR="${COMPARISON_OUTPUT_DIR:-/gpfs/projects/simswe/${USER}/training_outputs/css-360-comparison}"
mkdir -p "${OUT_DIR}"

echo "=== Job info ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "GPUs: ${TRAIN_GPU_COUNT}"
echo "Host: $(hostname)"
echo "Repo: ${REPO_ROOT}"
echo "Adapter: ${ADAPTER_PATH}"
echo "Output dir: ${OUT_DIR}"
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

echo "=== Python / package versions ==="
python -V
python - <<'PY'
import importlib
mods = ["torch", "transformers", "peft", "bitsandbytes", "accelerate"]
for name in mods:
    try:
        mod = importlib.import_module(name)
        print(f"{name}={getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:  # noqa: BLE001
        print(f"{name}=IMPORT_ERROR ({exc})")
PY

python training/compare_inference.py \
  --model-id "meta-llama/Llama-3.2-3B-Instruct" \
  --adapter-path "${ADAPTER_PATH}" \
  --heldout-file "${HELDOUT_FILE}" \
  --train-file "${TRAIN_FILE}" \
  --validation-file "${VAL_FILE}" \
  --output-dir "${OUT_DIR}"

echo "=== comparison_summary.json ==="
python - <<PY
from pathlib import Path
path = Path("${OUT_DIR}") / "comparison_summary.json"
print(path.read_text(encoding="utf-8") if path.is_file() else "missing")
PY

echo "Comparison job finished."
date -u
