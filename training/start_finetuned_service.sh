#!/usr/bin/env bash
# Start (or reuse) the CSS 360 fine-tuned inference Slurm job on Tillicum.
#
# Usage (from repository root on Tillicum):
#   ./training/start_finetuned_service.sh
#   ./training/start_finetuned_service.sh --no-wait
#   ./training/start_finetuned_service.sh --help

set -euo pipefail

JOB_NAME="css360-ft-infer"
ALLOC_TIMEOUT_SECONDS="${ALLOC_TIMEOUT_SECONDS:-600}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-600}"
POLL_SECONDS="${POLL_SECONDS:-5}"
INFERENCE_PORT="${INFERENCE_PORT:-8001}"
NO_WAIT=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HELPERS="${REPO_ROOT}/scripts/lib/finetuned_deploy_helpers.py"
SLURM_SCRIPT="${REPO_ROOT}/training/inference_service/serve.slurm"

usage() {
  cat <<'EOF'
Start or reuse the fine-tuned inference Slurm job on Tillicum.

Usage:
  ./training/start_finetuned_service.sh
  ./training/start_finetuned_service.sh --no-wait

Options:
  --no-wait   Submit/reuse the job, print the job ID, and return immediately
  -h, --help  Show this help

Environment overrides:
  ALLOC_TIMEOUT_SECONDS   Max seconds to wait for RUNNING (default: 600)
  HEALTH_TIMEOUT_SECONDS  Max seconds to wait for /health (default: 600)
  POLL_SECONDS            Poll interval (default: 5)
  INFERENCE_PORT          Service port on the compute node (default: 8001)

After the service is ready, run on aiswe.uwb.edu:
  ./scripts/start_finetuned_tunnel.sh <NODE>
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_cmd() {
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1 || die "Required command not found: ${cmd}"
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

find_active_job_line() {
  # Primary source for job identity/state/node. Format: JOBID STATE NODE ELAPSED TIMELIMIT
  local line state
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    state="$(printf '%s\n' "${line}" | helpers parse-squeue-line --field state)"
    if is_active_state "${state}"; then
      printf '%s\n' "${line}"
      return 0
    fi
  done < <(squeue -u "${USER}" -n "${JOB_NAME}" -h -o "%i %t %N %M %L" 2>/dev/null || true)
  return 1
}

job_log_paths() {
  local job_id="$1"
  echo "  stdout: ${REPO_ROOT}/training/logs/infer-${job_id}.out"
  echo "  stderr: ${REPO_ROOT}/training/logs/infer-${job_id}.err"
}

print_sacct_failure() {
  local job_id="$1"
  echo "Slurm job exited before service became ready." >&2
  echo "Job ID: ${job_id}" >&2
  echo "sacct summary:" >&2
  sacct -j "${job_id}" -X --parsable2 --noheader \
    -o JobID,State,ExitCode,Elapsed,NodeList >&2 || true
  echo "Log paths:" >&2
  job_log_paths "${job_id}" >&2
}

wait_for_running_node() {
  # Contract: the ONLY stdout from this function is the final compute hostname.
  # All human progress/status messages must go to stderr so
  # NODE="$(wait_for_running_node ...)" cannot be contaminated.
  local job_id="$1"
  local deadline=$((SECONDS + ALLOC_TIMEOUT_SECONDS))
  local line state node

  while (( SECONDS < deadline )); do
    line="$(squeue -j "${job_id}" -h -o "%i %t %N %M %L" 2>/dev/null | head -n 1 || true)"
    if [[ -z "${line}" ]]; then
      print_sacct_failure "${job_id}"
      exit 1
    fi
    state="$(printf '%s\n' "${line}" | helpers parse-squeue-line --field state)"
    node="$(printf '%s\n' "${line}" | helpers parse-squeue-line --field node)"
    case "${state}" in
      PD|PENDING|CF|CONFIGURING)
        echo "State: ${state} (waiting for allocation...)" >&2
        ;;
      R|RUNNING)
        if [[ -n "${node}" ]]; then
          printf '%s\n' "${node}"
          return 0
        fi
        echo "State: RUNNING (node not reported yet...)" >&2
        ;;
      *)
        echo "Unexpected Slurm state for job ${job_id}: ${state}" >&2
        print_sacct_failure "${job_id}"
        exit 1
        ;;
    esac
    sleep "${POLL_SECONDS}"
  done

  die "Timed out after ${ALLOC_TIMEOUT_SECONDS}s waiting for Slurm job ${job_id} to reach RUNNING."
}

health_ready() {
  local url="$1"
  local body
  if ! body="$(curl -fsS --max-time 5 "${url}" 2>/dev/null)"; then
    return 1
  fi
  printf '%s' "${body}" | helpers health-ready >/dev/null 2>&1
}

wait_for_health() {
  local job_id="$1"
  local node="$2"
  local url="http://${node}:${INFERENCE_PORT}/health"
  local deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))

  echo "Waiting for model health at ${url} ..."
  while (( SECONDS < deadline )); do
    if ! squeue -j "${job_id}" -h -o "%i" 2>/dev/null | grep -q .; then
      print_sacct_failure "${job_id}"
      exit 1
    fi
    if health_ready "${url}"; then
      return 0
    fi
    echo "Health not ready yet; retrying in ${POLL_SECONDS}s..."
    sleep "${POLL_SECONDS}"
  done

  echo "ERROR: Slurm job is RUNNING on ${node}, but /health never became ready (status=ok, adapterLoaded=true) within ${HEALTH_TIMEOUT_SECONDS}s." >&2
  echo "Check logs:" >&2
  job_log_paths "${job_id}" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --no-wait)
      NO_WAIT=1
      shift
      ;;
    *)
      die "Unknown argument: $1 (try --help)"
      ;;
  esac
done

cd "${REPO_ROOT}"

require_cmd sbatch
require_cmd squeue
require_cmd sacct
require_cmd curl
require_cmd python3
[[ -f "${SLURM_SCRIPT}" ]] || die "Does not look like the repo root (missing ${SLURM_SCRIPT}). Run from the repository root."
[[ -f "${HELPERS}" ]] || die "Missing helpers module: ${HELPERS}"

mkdir -p "${REPO_ROOT}/training/logs"

EXISTING_LINE=""
REUSED=0
JOB_ID=""

if EXISTING_LINE="$(find_active_job_line)"; then
  JOB_ID="$(printf '%s\n' "${EXISTING_LINE}" | helpers parse-squeue-line --field job_id)"
  REUSED=1
  echo "Existing active Slurm job found for ${JOB_NAME}; will not submit another."
  echo "Job ID: ${JOB_ID}"
else
  echo "Fine-tuned inference startup"
  echo "Submitting ${SLURM_SCRIPT} ..."
  SBATCH_OUT="$(sbatch "${SLURM_SCRIPT}")" || die "sbatch failed."
  echo "${SBATCH_OUT}"
  JOB_ID="$(printf '%s\n' "${SBATCH_OUT}" | helpers parse-sbatch-job-id)" || die "Could not parse JOB_ID from sbatch output."
fi

echo
echo "Fine-tuned inference startup"
echo "Job ID: ${JOB_ID}"
if [[ "${REUSED}" -eq 1 ]]; then
  echo "Mode: reuse existing job"
else
  echo "Mode: newly submitted"
fi

if [[ "${NO_WAIT}" -eq 1 ]]; then
  echo
  echo "(--no-wait) Returning without waiting for allocation/health."
  echo "Inspect later with: ./training/status_finetuned_service.sh"
  echo "Stop GPU job with: scancel ${JOB_ID}"
  exit 0
fi

echo "Waiting for Slurm allocation..."
NODE="$(wait_for_running_node "${JOB_ID}")"
# Defense in depth: reject contaminated multi-line stdout (progress must be on stderr).
NODE="$(printf '%s\n' "${NODE}" | helpers parse-wait-node-stdout)" \
  || die "Captured compute node from wait_for_running_node was invalid (stdout contamination?)."
echo "Allocated node: ${NODE}"

wait_for_health "${JOB_ID}" "${NODE}"

echo
echo "Fine-tuned service READY"
echo "Job ID: ${JOB_ID}"
echo "Node: ${NODE}"
echo "GPU endpoint: http://${NODE}:${INFERENCE_PORT}"
echo "Health: OK"
echo "Adapter loaded: true"
echo
echo "On aiswe.uwb.edu run:"
echo "  ./scripts/start_finetuned_tunnel.sh ${NODE}"
echo
echo "Stop GPU job when finished:"
echo "  scancel ${JOB_ID}"
