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

# Nodes to keep this submission off. Empty by default, and never persisted.
#
# Temporary infrastructure troubleshooting only. A node whose GPU preflight
# fails today — `Failed to get device handle for GPU 0` — is usually repaired
# within a day or two, and Slurm must stay free to schedule it the moment it is.
# So no node is ever named in this repository, and an exclusion lives exactly as
# long as the operator keeps typing it.
EXCLUDE_NODES="${TRAINING_EXCLUDE_NODES:-}"
QUEUE_RUN_ID=""
WALLTIME_OVERRIDE="${QLORA_WALLTIME:-}"
WALLTIME_CEILING="${QLORA_MAX_WALLTIME:-}"

usage() {
  cat <<'EOF'
Submit a QLoRA smoke or full training job on Tillicum.

Usage:
  ./training/start_qlora_training.sh --course <courseId> --smoke
  ./training/start_qlora_training.sh --course <courseId> --full

Exactly one of --smoke or --full is required. Full is never auto-submitted after smoke.

Options:
  --exclude-node NODE     Do not schedule on NODE. Repeatable, or comma-separated.
                          TEMPORARY troubleshooting for a node failing its GPU
                          preflight right now. Nothing is remembered between
                          runs; set TRAINING_EXCLUDE_NODES to default it for one
                          shell.
  --queue-run-id <runId>  PostgreSQL training run this job is executing. Passed
                          into the Slurm job so it can report its own completion
                          back to the application. Set automatically by
                          ./training/run_training_queue.sh.
  --time HH:MM:SS         Override the requested wall clock.
  --yes                   Skip the confirmation prompt.

Wall clock is chosen from the dataset size rather than fixed at 8 hours:
a full run asks for 30 minutes of overhead plus 20 seconds per optimizer step,
with a 1 hour floor and an 8 hour ceiling (raise with QLORA_MAX_WALLTIME).

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

# Return the first active squeue line for this course+mode (course-scoped name,
# plus legacy names only when qlora-job-<id>.env COURSE_ID matches).
find_active_job_for_course() {
  local course_id="$1"
  local mode="$2"
  local job_name legacy_name selected
  job_name="$(helpers slurm-job-name --course-id "${course_id}" --mode "${mode}")" \
    || return 1
  legacy_name="$(helpers legacy-slurm-job-name --mode "${mode}")" || return 1
  selected="$(
    {
      squeue -u "${USER}" -n "${job_name}" -h -o "%i %j %t %N %M %L" 2>/dev/null || true
      squeue -u "${USER}" -n "${legacy_name}" -h -o "%i %j %t %N %M %L" 2>/dev/null || true
    } | helpers select-active-training-job \
      --course-id "${course_id}" \
      --mode "${mode}" \
      --meta-dir "${REPO_ROOT}/training/logs"
  )" || return 1
  [[ -n "${selected}" ]] || return 1
  printf '%s\n' "${selected}"
  return 0
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
    --queue-run-id)
      [[ $# -ge 2 ]] || die "--queue-run-id requires a value"
      QUEUE_RUN_ID="$2"
      shift 2
      ;;
    --time)
      [[ $# -ge 2 ]] || die "--time requires a value (HH:MM:SS)"
      WALLTIME_OVERRIDE="$2"
      shift 2
      ;;
    --exclude-node)
      [[ $# -ge 2 ]] || die "--exclude-node requires a node name"
      if [[ -n "${EXCLUDE_NODES}" ]]; then
        EXCLUDE_NODES="${EXCLUDE_NODES},$2"
      else
        EXCLUDE_NODES="$2"
      fi
      shift 2
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
if [[ -n "${EXCLUDE_NODES}" ]]; then
  EXCLUDE_NODES="$(python3 "${DEPLOY_HELPERS}" validate-exclude-nodes "${EXCLUDE_NODES}")" \
    || die "Invalid --exclude-node value."
fi
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
  SLURM_SCRIPT="${REPO_ROOT}/training/smoke.slurm"
else
  SLURM_SCRIPT="${REPO_ROOT}/training/train.slurm"
fi

JOB_NAME="$(helpers slurm-job-name --course-id "${COURSE_ID}" --mode "${MODE}")" \
  || die "Could not build Slurm job name."

if EXISTING="$(find_active_job_for_course "${COURSE_ID}" "${MODE}")"; then
  EXISTING_ID="$(printf '%s\n' "${EXISTING}" | helpers parse-squeue-line --field job_id)"
  EXISTING_STATE="$(printf '%s\n' "${EXISTING}" | helpers parse-squeue-line --field state)"
  echo "Existing active ${JOB_NAME} job found; will not submit another."
  echo "Job ID: ${EXISTING_ID}"
  echo "State: ${EXISTING_STATE}"
  echo "Inspect with: ./training/status_qlora_training.sh"
  exit 0
fi

# The queue run id is what the finished job reports its completion against.
# Validated here rather than trusted: it becomes a file name in training/state/
# and a path segment in the callback URL.
if [[ -n "${QUEUE_RUN_ID}" ]]; then
  [[ "${QUEUE_RUN_ID}" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]] \
    || die "Invalid --queue-run-id: ${QUEUE_RUN_ID}"
fi

# Wall clock from dataset size, not a fixed 8 hours. See
# scripts/lib/qlora_training_helpers.py for the policy and the measurements
# behind it.
if [[ -n "${WALLTIME_OVERRIDE}" ]]; then
  WALLTIME="${WALLTIME_OVERRIDE}"
  WALLTIME_REASON="explicit override"
  WALLTIME_CLAMPED="null"
else
  WALLTIME_PLAN="$(
    helpers training-walltime \
      --mode "${MODE}" \
      --train-examples "${TRAIN_COUNT}" \
      ${WALLTIME_CEILING:+--ceiling "${WALLTIME_CEILING}"}
  )" || die "Could not choose a wall clock for this run."
  WALLTIME="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["walltime"])' "${WALLTIME_PLAN}")"
  WALLTIME_REASON="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["reason"])' "${WALLTIME_PLAN}")"
  WALLTIME_CLAMPED="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["clamped"])' "${WALLTIME_PLAN}")"
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
if [[ -n "${EXCLUDE_NODES}" ]]; then
  echo "Excluded nodes: ${EXCLUDE_NODES}  (temporary, this submission only)"
fi
echo "Wall clock: ${WALLTIME} (${WALLTIME_REASON})"
if [[ "${WALLTIME_CLAMPED}" == "ceiling" ]]; then
  echo "  Note: the estimate exceeded the maximum request and was capped."
  echo "  If this course times out, raise it: QLORA_MAX_WALLTIME=16:00:00 $0 ..."
fi
if [[ -n "${QUEUE_RUN_ID}" ]]; then
  echo "Queue run: ${QUEUE_RUN_ID} (the job will report its own completion)"
else
  echo "Queue run: (none — this job will not report a completion)"
fi
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
# Read by train.slurm / smoke.slurm to address the completion callback. Empty
# when the job was launched by hand, which simply means no callback is sent.
export QUEUE_RUN_ID

# Persist planned run metadata for status helper (updated with JOB_ID after sbatch).
META_FILE="${REPO_ROOT}/training/logs/last-qlora-${MODE}.env"
cat > "${META_FILE}" <<EOF
COURSE_ID=${COURSE_ID}
MODE=${MODE}
EXCLUDE_NODES=${EXCLUDE_NODES}
RUN_ID=${RUN_ID}
QUEUE_RUN_ID=${QUEUE_RUN_ID}
WALLTIME=${WALLTIME}
TRAINING_OUTPUT_DIR=${TRAINING_OUTPUT_DIR}
TRAIN_FILE=${TRAIN_FILE}
VAL_FILE=${VAL_FILE}
JOB_NAME=${JOB_NAME}
SLURM_SCRIPT=${SLURM_SCRIPT}
EOF

# Override #SBATCH --job-name with a course-scoped name so CSS360 never collides
# with CSS490/CSS350 active-job detection.
# `env -u TRAINING_WORKER_TOKEN`: the token would otherwise be inherited into
# the job environment on the compute node. The job does need it — it reports its
# own completion — but it reads it from .env.local on the shared filesystem,
# where it is already protected by file permissions, rather than travelling
# through the scheduler.
SBATCH_OUT="$(env -u TRAINING_WORKER_TOKEN sbatch \
  -J "${JOB_NAME}" \
  --time="${WALLTIME}" \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"} \
  "${SLURM_SCRIPT}")" \
  || die "sbatch failed."
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
