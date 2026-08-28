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
  # CG/COMPLETING counts: the job is still in the queue and still holds the
  # allocation, and an operator watching a session end should see that rather
  # than be told nothing is running.
  case "$1" in
    PD|PENDING|R|RUNNING|CF|CONFIGURING|CG|COMPLETING) return 0 ;;
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
command -v sacct >/dev/null 2>&1 || die "Required command not found: sacct"
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
done < <(squeue -u "${USER}" -n "${JOB_NAME}" -h -o "%i|%t|%N|%M|%L|%R" 2>/dev/null || true)

if [[ -z "${LINE}" ]]; then
  echo "No active ${JOB_NAME} job found for user ${USER}."
  # A job that has left the queue is only visible through sacct. Showing how the
  # last one ended is the difference between "nothing is running" and "the last
  # session failed and nobody noticed".
  LAST="$(sacct -u "${USER}" --name "${JOB_NAME}" -X --parsable2 --noheader \
    -o JobID,State,ExitCode,Elapsed,End 2>/dev/null | tail -n 1 || true)"
  if [[ -n "${LAST}" ]]; then
    IFS='|' read -r LAST_ID LAST_STATE LAST_EXIT LAST_ELAPSED LAST_END <<< "${LAST}"
    echo "Most recent session"
    echo "  Job ID: ${LAST_ID}"
    echo "  State: $(helpers describe-state "${LAST_STATE}") (${LAST_STATE})"
    echo "  Exit code: ${LAST_EXIT:-unknown}"
    echo "  Ran for: ${LAST_ELAPSED:-unknown}"
    echo "  Ended: ${LAST_END:-unknown}"
    echo "  Logs: ${REPO_ROOT}/training/logs/infer-${LAST_ID%%.*}.err"
  fi
  echo "Start one with: ./training/start_finetuned_service.sh"
  exit 1
fi

JOB_ID="$(printf '%s\n' "${LINE}" | helpers parse-squeue-line --field job_id)"
STATE="$(printf '%s\n' "${LINE}" | helpers parse-squeue-line --field state)"
NODE="$(printf '%s\n' "${LINE}" | helpers parse-squeue-line --field node)"
ELAPSED="$(printf '%s\n' "${LINE}" | helpers parse-squeue-line --field elapsed)"
TIME_LEFT="$(printf '%s\n' "${LINE}" | helpers parse-squeue-line --field time_left)"
REASON="$(printf '%s\n' "${LINE}" | helpers parse-squeue-line --field reason)"

echo "Slurm job"
echo "  Job ID: ${JOB_ID}"
echo "  State: $(helpers describe-state "${STATE}") (${STATE})"

# A pending job has no elapsed time and no node, and saying so is more useful
# than printing whichever field happened to land in that column. The previous
# format could not tell the difference: `%N` is empty for a pending job, so
# every field after it shifted left and a time *limit* was displayed as elapsed.
case "${STATE}" in
  PD|PENDING|CF|CONFIGURING)
    echo "  Waiting since: submitted (not started, so no elapsed run time)"
    echo "  Requested wall clock: ${TIME_LEFT:-unknown}"
    if [[ -n "${REASON}" ]]; then
      echo "  Pending because: $(helpers describe-pending-reason "${REASON}")"
    fi
    echo "  Node: (not allocated yet)"
    ;;
  *)
    echo "  Elapsed: ${ELAPSED:-unknown}"
    echo "  Time left: ${TIME_LEFT:-unknown}"
    echo "  Node: ${NODE:-"(none yet)"}"
    ;;
esac

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
  CG|COMPLETING)
    echo "  Health: n/a (the job is finishing; the session is ending)"
    exit 0
    ;;
  PD|PENDING|CF|CONFIGURING)
    echo "  Health: n/a (no allocation yet)"
    case "${REASON}" in
      QOSMaxWallDurationPerJobLimit|PartitionTimeLimit)
        echo "  This job will never start. Cancel it and request less time:"
        echo "    scancel ${JOB_ID}"
        echo "    ./training/start_finetuned_service.sh"
        exit 1
        ;;
    esac
    exit 0
    ;;
  *)
    echo "  Health: n/a (state ${STATE})"
    exit 0
    ;;
esac
