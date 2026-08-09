#!/usr/bin/env bash
# Submit a QLoRA smoke or full training job on Tillicum (never auto-chains smoke→full).
#
# Usage (from repository root on Tillicum):
#   ./training/start_qlora_training.sh --course css-360-winter-2026-a7rp --smoke
#   ./training/start_qlora_training.sh --course css-360-winter-2026-a7rp --full

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HELPERS="${REPO_ROOT}/scripts/lib/qlora_training_helpers.py"
DEPLOY_HELPERS="${REPO_ROOT}/scripts/lib/finetuned_deploy_helpers.py"

COURSE_ID=""
MODE=""
ASSUME_YES=0

usage() {
  cat <<'EOF'
Submit a QLoRA smoke or full training job on Tillicum.

Usage:
  ./training/start_qlora_training.sh --course <courseId> --smoke
  ./training/start_qlora_training.sh --course <courseId> --full

Exactly one of --smoke or --full is required. Full is never auto-submitted after smoke.

Full training writes to a VERSIONED directory under:
  /gpfs/projects/simswe/$USER/training_outputs/qlora-runs/<courseId>/<runId>-full/
and does NOT overwrite the live inference adapter at:
  .../training_outputs/css-360-qlora/adapter

Promote explicitly later with:
  ./training/promote_qlora_adapter.sh <versioned-adapter-path>
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

helpers() {
  python3 "${HELPERS}" "$@"
}

is_active_state() {
  case "$1" in
    PD|PENDING|R|RUNNING|CF|CONFIGURING) return 0 ;;
    *) return 1 ;;
  esac
}

find_active_job() {
  local job_name="$1"
  local line state
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    state="$(printf '%s\n' "${line}" | helpers parse-squeue-line --field state)"
    if is_active_state "${state}"; then
      printf '%s\n' "${line}"
      return 0
    fi
  done < <(squeue -u "${USER}" -n "${job_name}" -h -o "%i %j %t %N %M %L" 2>/dev/null || true)
  return 1
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
require_cmd sbatch
require_cmd squeue
require_cmd sacct
require_cmd python3
[[ -f "${HELPERS}" ]] || die "Missing helpers: ${HELPERS}"
[[ -f "${REPO_ROOT}/training/train.slurm" ]] || die "Does not look like the repo root."

COURSE_ID="$(helpers validate-course-id "${COURSE_ID}")" || die "Invalid course ID."
EXPORT_DIR="${REPO_ROOT}/data/exports/${COURSE_ID}"
COUNTS_JSON="$(helpers validate-export-dir "${EXPORT_DIR}")" || die "Export validation failed for ${EXPORT_DIR}"
TRAIN_COUNT="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["train_count"])' "${COUNTS_JSON}")"
VAL_COUNT="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["validation_count"])' "${COUNTS_JSON}")"

TRAIN_FILE="${EXPORT_DIR}/train.jsonl"
VAL_FILE="${EXPORT_DIR}/validation.jsonl"

QLORA_VENV="/gpfs/projects/simswe/${USER}/venvs/qlora/bin/activate"
if [[ ! -f "${REPO_ROOT}/training/.venv/bin/activate" && ! -f "${QLORA_VENV}" ]]; then
  die "No qlora virtual environment found (checked training/.venv and ${QLORA_VENV})."
fi

HF_TOKEN_PATH="/gpfs/projects/simswe/${USER}/huggingface/token"
[[ -s "${HF_TOKEN_PATH}" ]] || die "Missing Hugging Face token at ${HF_TOKEN_PATH}"

if [[ "${MODE}" == "smoke" ]]; then
  JOB_NAME="css360-qlora-smoke"
  SLURM_SCRIPT="${REPO_ROOT}/training/smoke.slurm"
else
  JOB_NAME="css360-qlora-train"
  SLURM_SCRIPT="${REPO_ROOT}/training/train.slurm"
fi

if EXISTING="$(find_active_job "${JOB_NAME}")"; then
  EXISTING_ID="$(printf '%s\n' "${EXISTING}" | helpers parse-squeue-line --field job_id)"
  EXISTING_STATE="$(printf '%s\n' "${EXISTING}" | helpers parse-squeue-line --field state)"
  echo "Existing active ${JOB_NAME} job found; will not submit another."
  echo "Job ID: ${EXISTING_ID}"
  echo "State: ${EXISTING_STATE}"
  echo "Inspect with: ./training/status_qlora_training.sh"
  exit 0
fi

RUN_ID="$(python3 -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))')"
TRAINING_OUTPUT_DIR="$(helpers versioned-outdir --user "${USER}" --course-id "${COURSE_ID}" --run-id "${RUN_ID}" --mode "${MODE}")" \
  || die "Could not build versioned output directory."

# Same gate train.slurm/smoke.slurm use: refuse missing/live-tree output dirs.
TRAINING_OUTPUT_DIR="$(helpers require-training-output-dir --user "${USER}" "${TRAINING_OUTPUT_DIR}")" \
  || die "Refusing unsafe TRAINING_OUTPUT_DIR."

echo "QLoRA training submission"
echo "Course: ${COURSE_ID}"
echo "Mode: ${MODE}"
echo "Train examples: ${TRAIN_COUNT}"
echo "Validation examples: ${VAL_COUNT}"
echo "Slurm script: ${SLURM_SCRIPT}"
echo "Job name: ${JOB_NAME}"
echo "Output directory (versioned):"
echo "  ${TRAINING_OUTPUT_DIR}"
echo "Adapter will be written to:"
echo "  ${TRAINING_OUTPUT_DIR}/adapter"
echo
echo "This action consumes GPU resources on Tillicum (account=simswe)."
echo "Live inference adapter will NOT be overwritten by this job."
echo

if [[ "${ASSUME_YES}" -ne 1 ]]; then
  printf "Submit %s job now? [y/N] " "${MODE}"
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

export TRAIN_FILE
export VAL_FILE
export TRAINING_OUTPUT_DIR
export COURSE_ID

# Persist planned run metadata for status helper (updated with JOB_ID after sbatch).
META_FILE="${REPO_ROOT}/training/logs/last-qlora-${MODE}.env"
cat > "${META_FILE}" <<EOF
COURSE_ID=${COURSE_ID}
MODE=${MODE}
RUN_ID=${RUN_ID}
TRAINING_OUTPUT_DIR=${TRAINING_OUTPUT_DIR}
TRAIN_FILE=${TRAIN_FILE}
VAL_FILE=${VAL_FILE}
JOB_NAME=${JOB_NAME}
SLURM_SCRIPT=${SLURM_SCRIPT}
EOF

SBATCH_OUT="$(sbatch "${SLURM_SCRIPT}")" || die "sbatch failed."
echo "${SBATCH_OUT}"
if [[ -f "${DEPLOY_HELPERS}" ]]; then
  JOB_ID="$(printf '%s\n' "${SBATCH_OUT}" | python3 "${DEPLOY_HELPERS}" parse-sbatch-job-id)" \
    || die "Could not parse job ID from sbatch output."
else
  JOB_ID="$(printf '%s\n' "${SBATCH_OUT}" | sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p')"
  [[ -n "${JOB_ID}" ]] || die "Could not parse job ID from sbatch output."
fi

{
  echo "JOB_ID=${JOB_ID}"
  echo "SUBMITTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${META_FILE}"
cp "${META_FILE}" "${REPO_ROOT}/training/logs/qlora-job-${JOB_ID}.env"
cp "${META_FILE}" "${TRAINING_OUTPUT_DIR}/run-meta.env"

echo
echo "Submitted ${MODE} training job"
echo "Job ID: ${JOB_ID}"
echo "Output dir: ${TRAINING_OUTPUT_DIR}"
echo "Logs:"
if [[ "${MODE}" == "smoke" ]]; then
  echo "  ${REPO_ROOT}/training/logs/smoke-${JOB_ID}.out"
  echo "  ${REPO_ROOT}/training/logs/smoke-${JOB_ID}.err"
else
  echo "  ${REPO_ROOT}/training/logs/train-${JOB_ID}.out"
  echo "  ${REPO_ROOT}/training/logs/train-${JOB_ID}.err"
fi
echo
echo "Status: ./training/status_qlora_training.sh"
if [[ "${MODE}" == "smoke" ]]; then
  echo "After smoke succeeds, explicitly run full training:"
  echo "  ./training/start_qlora_training.sh --course ${COURSE_ID} --full"
else
  echo "After full training + evaluation, promote ONLY if you intend to replace the live adapter:"
  echo "  ./training/promote_qlora_adapter.sh ${TRAINING_OUTPUT_DIR}/adapter"
fi
echo "Cancel if needed: scancel ${JOB_ID}"
