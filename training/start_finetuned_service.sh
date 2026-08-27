#!/usr/bin/env bash
# Start (or reuse) the per-course fine-tuned inference session on Tillicum.
#
# The one command to run after SSH + Duo, before a class or a demo:
#   ./training/start_finetuned_service.sh
#   ./training/start_finetuned_service.sh --hours 3
#   ./training/start_finetuned_service.sh --no-wait
#
# What it does, in order:
#   1. refuse to start a second session when one is already active
#   2. submit the serving job with a bounded wall clock (2 hours by default)
#   3. wait for the allocation and for the model to finish loading
#   4. record the session — node, port, expiry, published courses — with the
#      application, so the UWB VM can find the service without anyone typing a
#      hostname from one machine into another
#   5. print what is being served and what to run next
#
# The session ends when the Slurm allocation does. That is deliberate: a
# dropped login session, a closed laptop, or a forgotten stop command all
# resolve themselves at exactly the moment the GPU is released.

set -euo pipefail

JOB_NAME="css360-ft-infer"
ALLOC_TIMEOUT_SECONDS="${ALLOC_TIMEOUT_SECONDS:-600}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-900}"
POLL_SECONDS="${POLL_SECONDS:-5}"
INFERENCE_PORT="${INFERENCE_PORT:-8001}"
NO_WAIT=0
HOURS="${SERVICE_HOURS:-2}"
MAX_HOURS="${SERVICE_MAX_HOURS:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HELPERS="${REPO_ROOT}/scripts/lib/finetuned_deploy_helpers.py"
SLURM_SCRIPT="${REPO_ROOT}/training/inference_service/serve.slurm"
SERVING_ROOT="${SERVING_ROOT:-/gpfs/projects/simswe/${USER}/training_outputs/serving}"

usage() {
  cat <<'EOF'
Start or reuse the per-course fine-tuned inference session on Tillicum.

Usage:
  ./training/start_finetuned_service.sh
  ./training/start_finetuned_service.sh --hours 3
  ./training/start_finetuned_service.sh --no-wait

Options:
  --hours N   Session length in hours (default 2, maximum 8)
  --no-wait   Submit/reuse the job, print the job ID, and return immediately
  -h, --help  Show this help

Environment overrides:
  ALLOC_TIMEOUT_SECONDS   Max seconds to wait for RUNNING (default: 600)
  HEALTH_TIMEOUT_SECONDS  Max seconds to wait for /health (default: 900)
  POLL_SECONDS            Poll interval (default: 5)
  INFERENCE_PORT          Service port on the compute node (default: 8001)
  SERVING_ROOT            Where published per-course adapters live
  SERVICE_MAX_HOURS       Ceiling for --hours (default: 8)

Safe to re-run. An active session is reused and its record refreshed; a second
GPU job is never submitted.

Status:  ./training/status_finetuned_service.sh
Stop:    ./training/stop_finetuned_service.sh
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
  # Primary source for job identity/state/node. Format: JOBID STATE NODE ELAPSED TIMELEFT
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
  echo "Slurm job exited before the service became ready." >&2
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

  echo "Waiting for the model to load at ${url} ..."
  while (( SECONDS < deadline )); do
    if ! squeue -j "${job_id}" -h -o "%i" 2>/dev/null | grep -q .; then
      print_sacct_failure "${job_id}"
      exit 1
    fi
    if health_ready "${url}"; then
      return 0
    fi
    echo "Not ready yet; retrying in ${POLL_SECONDS}s..."
    sleep "${POLL_SECONDS}"
  done

  echo "ERROR: Slurm job is RUNNING on ${node}, but /health never became ready within ${HEALTH_TIMEOUT_SECONDS}s." >&2
  echo "Check logs:" >&2
  job_log_paths "${job_id}" >&2
  exit 1
}

seconds_left_for_job() {
  local job_id="$1"
  local remaining
  remaining="$(squeue -j "${job_id}" -h -o '%L' 2>/dev/null | head -n 1 || true)"
  [[ -n "${remaining}" ]] || return 1
  python3 "${REPO_ROOT}/scripts/lib/slurm_time_left.py" "${remaining}" 0 2>/dev/null
}

# Record the session with the application. Never fatal: the GPU job is the
# service, and this is how other machines find it. A backend that is briefly
# unreachable should not make a working session look like a failed start.
record_session() {
  local job_id="$1"
  local node="$2"
  local seconds="$3"

  if python3 "${REPO_ROOT}/training/serving_session.py" register \
    --job-id "${job_id}" \
    --node "${node}" \
    --port "${INFERENCE_PORT}" \
    --state ready \
    --seconds-remaining "${seconds}" \
    --serving-root "${SERVING_ROOT}"; then
    return 0
  fi
  echo
  echo "WARNING: the session is running but could not be recorded with the" >&2
  echo "application. The service works; the UWB VM cannot look up its node." >&2
  echo "Re-run this command later, or pass the node by hand on the VM:" >&2
  echo "  ./scripts/start_finetuned_tunnel.sh ${node}" >&2
  return 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --hours)
      [[ $# -ge 2 ]] || die "--hours requires a value"
      HOURS="$2"
      shift 2
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

[[ "${HOURS}" =~ ^[0-9]+$ ]] || die "--hours must be a whole number of hours."
(( HOURS >= 1 )) || die "--hours must be at least 1."
(( HOURS <= MAX_HOURS )) || die "--hours must not exceed ${MAX_HOURS} (raise SERVICE_MAX_HOURS deliberately)."
WALLTIME="$(printf '%02d:00:00' "${HOURS}")"

cd "${REPO_ROOT}"

require_cmd sbatch
require_cmd squeue
require_cmd sacct
require_cmd curl
require_cmd python3
[[ -f "${SLURM_SCRIPT}" ]] || die "Does not look like the repo root (missing ${SLURM_SCRIPT}). Run from the repository root."
[[ -f "${HELPERS}" ]] || die "Missing helpers module: ${HELPERS}"

mkdir -p "${REPO_ROOT}/training/logs"
mkdir -p "${SERVING_ROOT}"

echo "Fine-tuned serving session"
echo "Serving root: ${SERVING_ROOT}"
python3 "${REPO_ROOT}/scripts/lib/list_published_adapters.py" "${SERVING_ROOT}" || true
echo

EXISTING_LINE=""
REUSED=0
JOB_ID=""

if EXISTING_LINE="$(find_active_job_line)"; then
  JOB_ID="$(printf '%s\n' "${EXISTING_LINE}" | helpers parse-squeue-line --field job_id)"
  REUSED=1
  echo "An active ${JOB_NAME} job already exists; reusing it."
  echo "Job ID: ${JOB_ID}"
  echo "A second GPU allocation is never started for the same service."
else
  echo "Submitting ${SLURM_SCRIPT} for ${WALLTIME} ..."
  SBATCH_OUT="$(SERVING_ROOT="${SERVING_ROOT}" INFERENCE_PORT="${INFERENCE_PORT}" \
    env -u TRAINING_WORKER_TOKEN sbatch --time="${WALLTIME}" "${SLURM_SCRIPT}")" \
    || die "sbatch failed."
  echo "${SBATCH_OUT}"
  JOB_ID="$(printf '%s\n' "${SBATCH_OUT}" | helpers parse-sbatch-job-id)" \
    || die "Could not parse JOB_ID from sbatch output."
fi

echo
echo "Job ID: ${JOB_ID}"
if [[ "${REUSED}" -eq 1 ]]; then
  echo "Mode: reusing an existing session"
else
  echo "Mode: newly submitted (${WALLTIME})"
fi

if [[ "${NO_WAIT}" -eq 1 ]]; then
  echo
  echo "(--no-wait) Returning without waiting for allocation or model load."
  echo "Inspect later with: ./training/status_finetuned_service.sh"
  echo "Stop with:          ./training/stop_finetuned_service.sh"
  exit 0
fi

echo "Waiting for the Slurm allocation..."
NODE="$(wait_for_running_node "${JOB_ID}")"
# Defense in depth: reject contaminated multi-line stdout (progress must be on stderr).
NODE="$(printf '%s\n' "${NODE}" | helpers parse-wait-node-stdout)" \
  || die "Captured compute node from wait_for_running_node was invalid (stdout contamination?)."
echo "Allocated node: ${NODE}"

wait_for_health "${JOB_ID}" "${NODE}"

SECONDS_LEFT="$(seconds_left_for_job "${JOB_ID}" || true)"
SECONDS_LEFT="${SECONDS_LEFT%%.*}"
[[ -n "${SECONDS_LEFT}" ]] || SECONDS_LEFT=$((HOURS * 3600))

echo
echo "Fine-tuned service READY"
echo "Job ID: ${JOB_ID}"
echo "Node: ${NODE}"
echo "GPU endpoint: http://${NODE}:${INFERENCE_PORT}"
echo "Session ends in about $((SECONDS_LEFT / 60)) minutes."
echo

record_session "${JOB_ID}" "${NODE}" "${SECONDS_LEFT}"

echo
echo "Courses this session can answer for:"
curl -fsS --max-time 5 "http://${NODE}:${INFERENCE_PORT}/courses" 2>/dev/null \
  | python3 -c 'import json,sys
payload = json.load(sys.stdin)
courses = payload.get("courses") or []
if not courses:
    print("  (none published yet)")
for course in courses:
    print("  {0}  current={1}".format(course["courseId"], course["currentVersion"]))' \
  || echo "  (could not read /courses)"

echo
echo "Next, on aiswe.uwb.edu — this is the one step that still needs a person,"
echo "because opening the tunnel authenticates to UW and Duo is not automated:"
echo "  ./scripts/start_finetuned_tunnel.sh --from-backend"
echo
echo "Status: ./training/status_finetuned_service.sh"
echo "Stop:   ./training/stop_finetuned_service.sh"
