#!/usr/bin/env bash
# Show the state of the fine-tuned serving session on Tillicum.
#
# Usage:
#   ./training/status_finetuned_service.sh
#   ./training/status_finetuned_service.sh --help
#
# Reports the Slurm allocation, how much wall clock is left, whether the model
# has finished loading, which courses have a published adapter, and what the
# application currently believes about the session.

set -euo pipefail

JOB_NAME="css360-ft-infer"
INFERENCE_PORT="${INFERENCE_PORT:-8001}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HELPERS="${REPO_ROOT}/scripts/lib/finetuned_deploy_helpers.py"
SERVING_ROOT="${SERVING_ROOT:-/gpfs/projects/simswe/${USER}/training_outputs/serving}"

usage() {
  cat <<'EOF'
Show status for the active css360-ft-infer serving session.

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

echo "Published course adapters (${SERVING_ROOT})"
python3 "${REPO_ROOT}/scripts/lib/list_published_adapters.py" "${SERVING_ROOT}" \
  | sed 's/^/  /' || true
echo

echo "Application's view of the session"
python3 "${REPO_ROOT}/training/serving_session.py" show 2>&1 | sed 's/^/  /' || true
echo

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
  echo "Start one with: ./training/start_finetuned_service.sh"
  exit 1
fi

JOB_ID="$(printf '%s\n' "${LINE}" | helpers parse-squeue-line --field job_id)"
STATE="$(printf '%s\n' "${LINE}" | helpers parse-squeue-line --field state)"
NODE="$(printf '%s\n' "${LINE}" | helpers parse-squeue-line --field node)"
ELAPSED="$(printf '%s\n' "${LINE}" | helpers parse-squeue-line --field elapsed)"
TIME_LEFT="$(printf '%s\n' "${LINE}" | helpers parse-squeue-line --field time_left)"

echo "Slurm job"
echo "  Job ID: ${JOB_ID}"
echo "  State: ${STATE}"
echo "  Elapsed: ${ELAPSED:-unknown}"
echo "  Time left: ${TIME_LEFT:-unknown}"
echo "  Node: ${NODE:-"(none yet)"}"

case "${STATE}" in
  R|RUNNING)
    if [[ -z "${NODE}" ]]; then
      echo "  Health: unavailable (RUNNING but node unknown)"
      exit 1
    fi
    URL="http://${NODE}:${INFERENCE_PORT}/health"
    echo "  GPU endpoint: http://${NODE}:${INFERENCE_PORT}"
    if body="$(curl -fsS --max-time 5 "${URL}" 2>/dev/null)" \
      && printf '%s' "${body}" | helpers health-ready >/dev/null 2>&1; then
      echo "  Health: OK (model loaded)"
      printf '%s' "${body}" | python3 -c 'import json,sys
payload = json.load(sys.stdin)
loaded = payload.get("loadedAdapters") or []
courses = payload.get("courses") or []
print("  Adapters resident: {0}".format(", ".join(loaded) or "none yet"))
print("  Answering for: {0}".format(
    ", ".join("{0} ({1})".format(item["courseId"], item["currentVersion"]) for item in courses)
    or "no published course adapters"))
remaining = payload.get("secondsRemaining")
if isinstance(remaining, (int, float)):
    print("  Session ends in: {0} minutes".format(int(remaining // 60)))' \
        || true
      exit 0
    fi
    echo "  Health: not ready at ${URL} (the model may still be loading)"
    echo "  Logs: ${REPO_ROOT}/training/logs/infer-${JOB_ID}.err"
    exit 1
    ;;
  *)
    echo "  Health: n/a (job not RUNNING yet)"
    exit 0
    ;;
esac
