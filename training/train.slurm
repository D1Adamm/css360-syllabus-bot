#!/bin/bash
# Tillicum full QLoRA training for CSS 360 (48 train / 6 validation).
# Submit only after a successful smoke job:
#   sbatch training/train.slurm

#SBATCH --job-name=css360-qlora-train
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
mkdir -p "/gpfs/projects/simswe/${USER}/hf_cache"
mkdir -p "/gpfs/projects/simswe/${USER}/training_outputs/css-360-qlora"

export HF_HOME="/gpfs/projects/simswe/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TRAIN_GPU_COUNT="${SLURM_GPUS_ON_NODE:-1}"
export SLURM_JOB_ID="${SLURM_JOB_ID:-}"

echo "=== Job info ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "GPUs: ${TRAIN_GPU_COUNT}"
echo "Host: $(hostname)"
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

TRAIN_FILE="${REPO_ROOT}/data/exports/css-360-winter-2026-a7rp/train.jsonl"
VAL_FILE="${REPO_ROOT}/data/exports/css-360-winter-2026-a7rp/validation.jsonl"
OUT_DIR="/gpfs/projects/simswe/${USER}/training_outputs/css-360-qlora"

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
date -u
