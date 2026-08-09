#!/usr/bin/env bash
# Show status for the CSS 360 fine-tuned inference Slurm job on Tillicum.
#
# Usage:
#   ./training/status_finetuned_service.sh
#   ./training/status_finetuned_service.sh --help

set -euo pipefail

JOB_NAME="css360-ft-infer"
INFERENCE_PORT="${INFERENCE_PORT:-8001}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HELPERS="${REPO_ROOT}/scripts/lib/finetuned_deploy_helpers.py"

usage() {
  cat <<'EOF'
Show status for the active css360-ft-infer Slurm job.

Usage:
  ./training/status_finetuned_service.sh

Exits nonzero when no active job is found, or when a RUNNING job fails health.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
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

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -gt 0 ]]; then
  die "Unknown argument: $1 (try --help)"
fi

command -v squeue >/dev/null 2>&1 || die "Required command not found: squeue"
command -v python3 >/dev/null 2>&1 || die "Required command not found: python3"
[[ -f "${HELPERS}" ]] || die "Missing helpers module: ${HELPERS}"

LINE=""
while IFS= read -r candidate; do
  [[ -z "${candidate}" ]] && continue
  state="$(printf '%s\n' "${candidate}" | helpers parse-squeue-line --field state)"
  if is_active_state "${state}"; then
    LINE="${candidate}"
    break
  fi
done < <(squeue -u "${USER}" -n "${JOB_NAME}" -h -o "%i %t %N %M %L" 2>/dev/null || true)

if [[ -z "${LINE}" ]]; then
  echo "No active ${JOB_NAME} job found for user ${USER}."
  exit 1
fi

JOB_ID="$(printf '%s\n' "${LINE}" | helpers parse-squeue-line --field job_id)"
STATE="$(printf '%s\n' "${LINE}" | helpers parse-squeue-line --field state)"
NODE="$(printf '%s\n' "${LINE}" | helpers parse-squeue-line --field node)"
ELAPSED="$(printf '%s\n' "${LINE}" | helpers parse-squeue-line --field elapsed)"
TIME_LEFT="$(printf '%s\n' "${LINE}" | helpers parse-squeue-line --field time_left)"

echo "Fine-tuned Slurm job"
echo "Job ID: ${JOB_ID}"
echo "State: ${STATE}"
echo "Elapsed: ${ELAPSED:-unknown}"
echo "Time left: ${TIME_LEFT:-unknown}"
echo "Node: ${NODE:-"(none yet)"}"

case "${STATE}" in
  R|RUNNING)
    if [[ -z "${NODE}" ]]; then
      echo "Health: unavailable (RUNNING but node unknown)"
      exit 1
    fi
    URL="http://${NODE}:${INFERENCE_PORT}/health"
    echo "GPU endpoint: http://${NODE}:${INFERENCE_PORT}"
    if body="$(curl -fsS --max-time 5 "${URL}" 2>/dev/null)" \
      && printf '%s' "${body}" | helpers health-ready >/dev/null 2>&1; then
      echo "Health: OK (adapterLoaded=true)"
      exit 0
    fi
    echo "Health: not ready at ${URL}"
    exit 1
    ;;
  *)
    echo "Health: n/a (job not RUNNING yet)"
    exit 0
    ;;
esac
