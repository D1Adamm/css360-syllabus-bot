#!/usr/bin/env bash
# Run a QLoRA smoke or full training job on THIS machine's CPU (the UWB VM).
# No Slurm, no GPU, no Tillicum. The GPU path (start_qlora_training.sh,
# train.slurm, smoke.slurm) is untouched and remains the cluster workflow.
#
# Usage (from the repository root on the VM):
#   ./training/start_cpu_qlora_training.sh --course css-360-winter-2026-a7rp --smoke
#   ./training/start_cpu_qlora_training.sh --course css-360-winter-2026-a7rp --full
#
# The trainer is invoked with --cpu, which is the only way CPU mode is ever
# selected: float32 compute, no CUDA, no fp16/bf16, batch size 1, and the same
# data, chat formatting, LoRA and NF4 configuration as the cluster recipe.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HELPERS="${REPO_ROOT}/scripts/lib/qlora_training_helpers.py"
TRAINER="${REPO_ROOT}/training/train_qlora.py"

COURSE_ID=""
MODE=""
ASSUME_YES=0
FOREGROUND=0
THREADS="${CPU_TRAINING_THREADS:-}"
VENV_DIR="${CPU_TRAINING_VENV:-${HOME}/cpu-training-venv}"
OUTPUT_ROOT="${CPU_TRAINING_OUTPUT_ROOT:-${HOME}}"
MODEL_ID="meta-llama/Llama-3.2-3B-Instruct"

usage() {
  cat <<'USAGE'
Run a QLoRA smoke or full training job on this machine's CPU (no Slurm, no GPU).

Usage:
  ./training/start_cpu_qlora_training.sh --course <courseId> --smoke
  ./training/start_cpu_qlora_training.sh --course <courseId> --full

Exactly one of --smoke or --full is required. Full is never chained after smoke.

Options:
  --threads N     torch threads for the run (default: every core, `nproc`).
                  Leave a couple free if the backend and Ollama must stay
                  responsive while training runs.
  --foreground    Run in this terminal instead of the background. The default
                  detaches with nohup so a dropped SSH session does not kill
                  an hour-long run; follow it with the printed `tail -f`.
  --yes           Skip the confirmation prompt.

Environment:
  CPU_TRAINING_VENV         venv with the CPU stack (default: ~/cpu-training-venv)
  CPU_TRAINING_OUTPUT_ROOT  where outputs go (default: $HOME); runs land under
                            <root>/training_outputs/qlora-runs/<courseId>/<runId>-<mode>/
  CPU_TRAINING_THREADS      default for --threads
  HF_HOME / HF_TOKEN        Hugging Face cache and access for the gated base model

Outputs (same layout as a cluster run): adapter/, tokenizer/, checkpoints/,
resolved_config.json, runtime-report.json, training_metrics.json,
evaluation_metrics.json, run-meta.env. Nothing is registered or published:
register a finished full run with scripts/register_course_model.py, then
convert adapter/ to GGUF and `ollama create` it (training/inference_service/README.md).
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

helpers() {
  python3 "${HELPERS}" "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --course)
      [[ $# -ge 2 ]] || die "--course requires a value"
      COURSE_ID="$2"
      shift 2
      ;;
    --smoke)
      [[ -z "${MODE}" ]] || die "Specify only one of --smoke or --full"
      MODE="smoke"
      shift
      ;;
    --full)
      [[ -z "${MODE}" ]] || die "Specify only one of --smoke or --full"
      MODE="full"
      shift
      ;;
    --threads)
      [[ $# -ge 2 ]] || die "--threads requires a value"
      THREADS="$2"
      shift 2
      ;;
    --foreground)
      FOREGROUND=1
      shift
      ;;
    --yes|-y)
      ASSUME_YES=1
      shift
      ;;
    *)
      die "Unknown argument: $1 (try --help)"
      ;;
  esac
done

[[ -n "${COURSE_ID}" ]] || die "--course <courseId> is required"
[[ -n "${MODE}" ]] || die "Exactly one of --smoke or --full is required"

cd "${REPO_ROOT}"
command -v python3 >/dev/null 2>&1 || die "python3 is required"
[[ -f "${HELPERS}" ]] || die "Missing helpers: ${HELPERS}"
[[ -f "${TRAINER}" ]] || die "Missing trainer: ${TRAINER}"

COURSE_ID="$(helpers validate-course-id "${COURSE_ID}")" || die "Invalid course ID."
EXPORT_DIR="${REPO_ROOT}/data/exports/${COURSE_ID}"
COUNTS_JSON="$(helpers validate-export-dir "${EXPORT_DIR}")" || die "Export validation failed for ${EXPORT_DIR}"
TRAIN_COUNT="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["train_count"])' "${COUNTS_JSON}")"
VAL_COUNT="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["validation_count"])' "${COUNTS_JSON}")"
TRAIN_FILE="${EXPORT_DIR}/train.jsonl"
VAL_FILE="${EXPORT_DIR}/validation.jsonl"

[[ -f "${VENV_DIR}/bin/activate" ]] \
  || die "No CPU training venv at ${VENV_DIR} (set CPU_TRAINING_VENV to the venv with torch, transformers, peft, bitsandbytes and trl)."

if [[ -z "${THREADS}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    THREADS="$(nproc)"
  else
    THREADS="$(python3 -c 'import os; print(os.cpu_count() or 1)')"
  fi
fi
[[ "${THREADS}" =~ ^[0-9]+$ && "${THREADS}" -ge 1 ]] || die "--threads must be a positive integer (got ${THREADS})"

# Llama 3.2 is gated. Either a token must be reachable or the base model must
# already be in the Hugging Face cache (a previous run leaves it there).
HF_ROOT="${HF_HOME:-${HOME}/.cache/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_PATH:-${HF_ROOT}/token}"
MODEL_CACHE="${HF_HUB_CACHE:-${HF_ROOT}/hub}/models--${MODEL_ID//\//--}"
if [[ -z "${HF_TOKEN:-}" && ! -s "${HF_TOKEN_FILE}" && ! -d "${MODEL_CACHE}" ]]; then
  die "No Hugging Face token (HF_TOKEN or ${HF_TOKEN_FILE}) and no cached base model at ${MODEL_CACHE}. Run 'huggingface-cli login' in ${VENV_DIR} first."
fi

RUN_ID="$(python3 -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))')"
TRAINING_OUTPUT_DIR="$(helpers local-versioned-outdir --root "${OUTPUT_ROOT}" --course-id "${COURSE_ID}" --run-id "${RUN_ID}" --mode "${MODE}")" \
  || die "Could not build a versioned output directory."
# Same gate the Slurm scripts apply: never the promoted adapter tree.
TRAINING_OUTPUT_DIR="$(helpers require-training-output-dir --user "${USER:-$(id -un)}" "${TRAINING_OUTPUT_DIR}")" \
  || die "Refusing unsafe TRAINING_OUTPUT_DIR."
OUTPUT_REF="$(helpers relative-output-ref "${TRAINING_OUTPUT_DIR}")"

LOG_PREFIX="cpu-smoke"
[[ "${MODE}" == "smoke" ]] || LOG_PREFIX="cpu-train"
LOG_FILE="${REPO_ROOT}/training/logs/${LOG_PREFIX}-${RUN_ID}.log"

echo "QLoRA CPU training (this machine, no Slurm)"
echo "Course: ${COURSE_ID}"
echo "Mode: ${MODE}"
echo "Train examples: ${TRAIN_COUNT}"
echo "Validation examples: ${VAL_COUNT}"
echo "Base model: ${MODEL_ID} (4-bit NF4, float32 compute)"
echo "Threads: ${THREADS} of $(python3 -c 'import os; print(os.cpu_count() or "?")') cores"
echo "Venv: ${VENV_DIR}"
echo "Output directory (versioned):"
echo "  ${TRAINING_OUTPUT_DIR}"
echo "Adapter will be written to:"
echo "  ${TRAINING_OUTPUT_DIR}/adapter"
echo "Log: ${LOG_FILE}"
echo
if [[ "${MODE}" == "smoke" ]]; then
  echo "A smoke run trains 3 optimizer steps on 4 examples and prints a full-run estimate."
else
  echo "A full run trains ${TRAIN_COUNT} examples for 3 epochs; expect roughly an hour"
  echo "for a course this size on 8 cores, during which the backend and Ollama"
  echo "share these cores. Nothing is registered or published automatically."
fi
echo

if [[ "${ASSUME_YES}" -ne 1 ]]; then
  printf "Start %s CPU training now? [y/N] " "${MODE}"
  read -r reply </dev/tty || die "Could not read confirmation from terminal."
  case "${reply}" in
    y|Y|yes|YES) ;;
    *)
      echo "Aborted."
      exit 1
      ;;
  esac
fi

mkdir -p "${REPO_ROOT}/training/logs"
mkdir -p "${TRAINING_OUTPUT_DIR}"

STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
META_FILE="${REPO_ROOT}/training/logs/last-cpu-${MODE}.env"
cat > "${META_FILE}" <<META
COURSE_ID=${COURSE_ID}
MODE=${MODE}
DEVICE=cpu
RUN_ID=${RUN_ID}
THREADS=${THREADS}
TRAINING_OUTPUT_DIR=${TRAINING_OUTPUT_DIR}
OUTPUT_REF=${OUTPUT_REF}
TRAIN_FILE=${TRAIN_FILE}
VAL_FILE=${VAL_FILE}
LOG_FILE=${LOG_FILE}
STARTED_AT=${STARTED_AT}
META
cp "${META_FILE}" "${TRAINING_OUTPUT_DIR}/run-meta.env"

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
command -v python >/dev/null 2>&1 || die "python not found after activating ${VENV_DIR}"

export OMP_NUM_THREADS="${THREADS}"
export TOKENIZERS_PARALLELISM=false

{
  echo "=== CPU training job ==="
  echo "Run ID: ${RUN_ID}"
  echo "Host: $(hostname)"
  echo "Started: ${STARTED_AT}"
  echo "Course: ${COURSE_ID}"
  echo "Mode: ${MODE}"
  echo "Threads: ${THREADS}"
  echo "Train file: ${TRAIN_FILE}"
  echo "Validation file: ${VAL_FILE}"
  echo "Output dir: ${TRAINING_OUTPUT_DIR}"
  echo "=== Python / package versions ==="
  python -V
  python - <<'PY'
import importlib
for name in ["torch", "transformers", "datasets", "peft", "bitsandbytes", "trl", "accelerate"]:
    try:
        mod = importlib.import_module(name)
        print(f"{name}={getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:  # noqa: BLE001
        print(f"{name}=IMPORT_ERROR ({exc})")
PY
} | tee "${LOG_FILE}"

TRAIN_ARGS=(
  --cpu
  --cpu-threads "${THREADS}"
  --train-file "${TRAIN_FILE}"
  --validation-file "${VAL_FILE}"
  --output-dir "${TRAINING_OUTPUT_DIR}"
)
if [[ "${MODE}" == "smoke" ]]; then
  TRAIN_ARGS+=(--smoke-test)
fi

print_next_steps() {
  echo
  echo "When it finishes, read:"
  echo "  ${TRAINING_OUTPUT_DIR}/runtime-report.json"
  if [[ "${MODE}" == "smoke" ]]; then
    echo "The smoke benchmark at the end of the log estimates the full run. Then, explicitly:"
    echo "  ./training/start_cpu_qlora_training.sh --course ${COURSE_ID} --full"
  else
    echo "Register the model version (nothing is registered automatically):"
    echo "  backend/.venv/bin/python scripts/register_course_model.py \\"
    echo "    --course-id ${COURSE_ID} --base-model ${MODEL_ID} \\"
    echo "    --training-examples ${TRAIN_COUNT} \\"
    echo "    --artifact-ref ${OUTPUT_REF}/adapter --status ready --deployment offline"
    echo "Then convert ${TRAINING_OUTPUT_DIR}/adapter to GGUF and 'ollama create' it"
    echo "(training/inference_service/README.md)."
  fi
}

if [[ "${FOREGROUND}" -eq 1 ]]; then
  echo "=== training (foreground) ===" | tee -a "${LOG_FILE}"
  set +e
  python "${TRAINER}" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
  STATUS=${PIPESTATUS[0]}
  set -e
  echo "EXIT_STATUS=${STATUS}" >> "${META_FILE}"
  echo "EXIT_STATUS=${STATUS}" >> "${TRAINING_OUTPUT_DIR}/run-meta.env"
  if [[ "${STATUS}" -ne 0 ]]; then
    die "Training exited with status ${STATUS}. See ${LOG_FILE}"
  fi
  echo "Training finished."
  print_next_steps
  exit 0
fi

echo "=== training (background) ===" >> "${LOG_FILE}"
nohup python "${TRAINER}" "${TRAIN_ARGS[@]}" >> "${LOG_FILE}" 2>&1 &
PID=$!
echo "PID=${PID}" >> "${META_FILE}"
echo "PID=${PID}" >> "${TRAINING_OUTPUT_DIR}/run-meta.env"

echo
echo "Started ${MODE} CPU training in the background."
echo "PID: ${PID}"
echo "Follow: tail -f ${LOG_FILE}"
echo "Stop:   kill ${PID}"
print_next_steps
