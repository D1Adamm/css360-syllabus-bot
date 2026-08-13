#!/bin/bash
# Tillicum full QLoRA training.
# Use:
#   ./training/start_qlora_training.sh --course <courseId> --full
#
# TRAINING_OUTPUT_DIR is required (set by start_qlora_training.sh to a versioned
# qlora-runs/ path). Raw `sbatch training/train.slurm` fails fast if unset.
# The live inference adapter is never written here — only via promote_qlora_adapter.sh.
#
# Environment (set by start_qlora_training.sh):
#   TRAIN_FILE, VAL_FILE, TRAINING_OUTPUT_DIR

#SBATCH --job-name=css360-qlora-train
# Default name only; start_qlora_training.sh overrides with sbatch -J
# qlora-train-<courseId> for per-course isolation.
#SBATCH --account=simswe
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=training/logs/train-%j.out
#SBATCH --error=training/logs/train-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

mkdir -p training/logs

HELPERS="${REPO_ROOT}/scripts/lib/qlora_training_helpers.py"
if [[ ! -f "${HELPERS}" ]]; then
  echo "ERROR: Missing helpers module: ${HELPERS}" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required to validate TRAINING_OUTPUT_DIR." >&2
  exit 1
fi

# Fail before any model download / training if the output dir is missing or unsafe.
OUT_DIR="$(python3 "${HELPERS}" require-training-output-dir --user "${USER}")" || {
  echo "ERROR: TRAINING_OUTPUT_DIR is required." >&2
  echo "Use ./training/start_qlora_training.sh --course <courseId> --full" >&2
  echo "to create a safe versioned training run." >&2
  exit 1
}

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
mkdir -p "${OUT_DIR}"

echo "=== Job info ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "GPUs: ${TRAIN_GPU_COUNT}"
echo "Host: $(hostname)"
echo "Repo: ${REPO_ROOT}"
echo "Train file: ${TRAIN_FILE}"
echo "Validation file: ${VAL_FILE}"
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
mods = ["torch", "transformers", "datasets", "peft", "bitsandbytes", "trl", "accelerate"]
for name in mods:
    try:
        mod = importlib.import_module(name)
        print(f"{name}={getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:  # noqa: BLE001
        print(f"{name}=IMPORT_ERROR ({exc})")
PY

python training/train_qlora.py \
  --train-file "${TRAIN_FILE}" \
  --validation-file "${VAL_FILE}" \
  --output-dir "${OUT_DIR}" \
  --gpu-count "${TRAIN_GPU_COUNT}"

echo "=== runtime-report.json ==="
python - <<PY
from pathlib import Path
path = Path("${OUT_DIR}") / "runtime-report.json"
print(path.read_text(encoding="utf-8") if path.is_file() else "missing")
PY

echo "Full training job finished."
echo "Adapter (if successful): ${OUT_DIR}/adapter"
date -u
