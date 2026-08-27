#!/bin/bash
# Tillicum smoke test for QLoRA (tiny run + full-run duration estimate).
# Use:
#   ./training/start_qlora_training.sh --course <courseId> --smoke
#
# TRAINING_OUTPUT_DIR is required (set by start_qlora_training.sh to a versioned
# qlora-runs/ path). Raw `sbatch training/smoke.slurm` fails fast if unset.
#
# Environment (set by start_qlora_training.sh):
#   TRAIN_FILE, VAL_FILE, TRAINING_OUTPUT_DIR

#SBATCH --job-name=css360-qlora-smoke
# Default name only; start_qlora_training.sh overrides with sbatch -J
# qlora-smoke-<courseId> for per-course isolation.
#SBATCH --account=simswe
#SBATCH --qos=debug
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:45:00
#SBATCH --output=training/logs/smoke-%j.out
#SBATCH --error=training/logs/smoke-%j.err

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
  echo "Use ./training/start_qlora_training.sh --course <courseId> --smoke" >&2
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

TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/data/exports/css-360-winter-2026-a7rp/train.jsonl}"
VAL_FILE="${VAL_FILE:-${REPO_ROOT}/data/exports/css-360-winter-2026-a7rp/validation.jsonl}"
mkdir -p "${OUT_DIR}"

# ---------------------------------------------------------------------------
# Completion reporting
#
# The job tells the application how it ended, instead of a person doing it
# hours later. Job 253552 finished successfully and the run sat at `submitted`
# for days because nothing here existed.
#
# The trap fires on any nonzero exit or signal, including a Slurm timeout, so a
# failure is reported as reliably as a success. `report_training_result.py`
# persists the payload to training/state/pending/ before it tries to send it, so
# a backend that is unreachable right now costs a delay and not the event: the
# next ./training/run_training_queue.sh --once delivers it.
#
# QUEUE_RUN_ID is empty when the job was launched by hand rather than by the
# queue. That is a supported way to run this script, and it simply means there
# is no run to report against.
# ---------------------------------------------------------------------------

STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
REPORTED=0

report_result() {
  local outcome="$1"
  local stage="$2"
  local message="$3"

  [[ -n "${QUEUE_RUN_ID:-}" ]] || return 0
  # Report once. Both the success path and the trap can reach this, and a
  # failure during reporting must not re-enter it.
  [[ "${REPORTED}" -eq 0 ]] || return 0
  REPORTED=1

  # An array, not an interpolated string: the error message is free text with
  # spaces in it, and an unquoted `${message:+--error "$message"}` would split
  # it into separate arguments.
  local -a args=(
    --course-id "${COURSE_ID:-}"
    --run-id "${QUEUE_RUN_ID}"
    --output-dir "${OUT_DIR}"
    --export-dir "$(dirname "${TRAIN_FILE}")"
    --job-id "${SLURM_JOB_ID:-}"
    --started-at "${STARTED_AT}"
    --outcome "${outcome}"
    --no-register
  )
  # `if` rather than `[[ ... ]] && ...`: under `set -e` a false test is a
  # nonzero statement, which would fire the ERR trap and re-enter this function
  # simply because there was no failure stage to add.
  if [[ -n "${stage}" ]]; then
    args+=(--failure-stage "${stage}")
  fi
  if [[ -n "${message}" ]]; then
    args+=(--error "${message}")
  fi

  python3 "${REPO_ROOT}/training/report_training_result.py" "${args[@]}" \
    || echo "WARNING: completion reporting failed; the payload is queued under training/state/pending/" >&2
}

on_failure() {
  local status=$?
  report_result failed "${FAILURE_STAGE:-training}" \
    "The Slurm job exited with status ${status}. See training/logs/smoke-${SLURM_JOB_ID}.err for details."
}

trap on_failure ERR
trap on_failure INT TERM

# Checked after the trap is installed, not before. A job that dies here
# is still a job that has to report its failure — leaving the run at
# `submitted` with nothing to say why is the whole problem the trap
# exists to fix.
FAILURE_STAGE="preflight"
if [[ ! -s "${HF_TOKEN_PATH}" ]]; then
  # Reported explicitly rather than left to the ERR trap: `exit` inside a
  # `||` list is not a failing command, so the trap would never see it and the
  # run would sit at `submitted` exactly as it used to.
  echo "Missing Hugging Face token at ${HF_TOKEN_PATH}" >&2
  report_result failed preflight \
    "No Hugging Face token at ${HF_TOKEN_PATH} on the compute node."
  exit 1
fi


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

FAILURE_STAGE="training"
python training/train_qlora.py \
  --smoke-test \
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

# A smoke run is a rehearsal, not a model. It reports its outcome so the queue
# does not strand the run, but asks the backend not to register anything:
# `--outcome succeeded` with no artifact registration, which the completion
# route supports through `register: false`.
FAILURE_STAGE="reporting"
report_result succeeded "" ""

echo "Smoke job finished."
echo "Adapter (if successful): ${OUT_DIR}/adapter"
date -u
