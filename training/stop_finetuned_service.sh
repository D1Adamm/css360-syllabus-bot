#!/usr/bin/env bash
# Stop the fine-tuned serving session on Tillicum and release the GPU.
#
# Usage (from repository root on Tillicum):
#   ./training/stop_finetuned_service.sh
#   ./training/stop_finetuned_service.sh --job-id 264787
#
# Cancels the serving allocation and marks the recorded session stopped. Safe to
# run when nothing is up: a session that already reached its wall clock has
# ended on its own, and this reports that rather than failing.
#
# It does not close the SSH tunnel on the UWB VM — that is a different machine
# and a different process. Run ./scripts/stop_finetuned_tunnel.sh there.

set -euo pipefail

JOB_NAME="css360-ft-infer"
JOB_ID=""
ASSUME_YES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HELPERS="${REPO_ROOT}/scripts/lib/finetuned_deploy_helpers.py"

usage() {
  cat <<'EOF'
Stop the fine-tuned serving session on Tillicum.

Usage:
  ./training/stop_finetuned_service.sh
  ./training/stop_finetuned_service.sh --job-id 264787
  ./training/stop_finetuned_service.sh --yes

Cancels the css360-ft-infer allocation and marks the recorded session stopped.

This does NOT close the SSH tunnel on aiswe.uwb.edu. Run there:
  ./scripts/stop_finetuned_tunnel.sh
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --job-id)
      [[ $# -ge 2 ]] || die "--job-id requires a value"
      JOB_ID="$2"
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

cd "${REPO_ROOT}"
command -v squeue >/dev/null 2>&1 || die "Required command not found: squeue"
command -v python3 >/dev/null 2>&1 || die "Required command not found: python3"
[[ -f "${HELPERS}" ]] || die "Missing helpers module: ${HELPERS}"

if [[ -z "${JOB_ID}" ]]; then
  while IFS= read -r candidate; do
    [[ -z "${candidate}" ]] && continue
    state="$(printf '%s\n' "${candidate}" | helpers parse-squeue-line --field state)"
    if is_active_state "${state}"; then
      JOB_ID="$(printf '%s\n' "${candidate}" | helpers parse-squeue-line --field job_id)"
      break
    fi
  done < <(squeue -u "${USER}" -n "${JOB_NAME}" -h -o "%i %t %N %M %L" 2>/dev/null || true)
fi

if [[ -z "${JOB_ID}" ]]; then
  echo "No active ${JOB_NAME} job found for ${USER}."
  echo "If a session record is still open, it will expire on its own at the end"
  echo "of the allocation it described."
  exit 0
fi

echo "Stopping fine-tuned serving session"
echo "Job ID: ${JOB_ID}"
echo
echo "Students and professors lose Fine-Tuned and Fine-Tuned + RAG immediately."
echo "Base and RAG on the UWB VM are unaffected."
echo

if [[ "${ASSUME_YES}" -ne 1 ]]; then
  printf "Cancel job %s now? [y/N] " "${JOB_ID}"
  read -r reply </dev/tty || die "Could not read confirmation from terminal."
  case "${reply}" in
    y|Y|yes|YES) ;;
    *)
      echo "Aborted."
      exit 1
      ;;
  esac
fi

# Cancel first, then record. If the record fails the GPU is still released,
# which is the part that costs money; the session row expires on its own at the
# wall clock it was given.
if command -v scancel >/dev/null 2>&1; then
  scancel "${JOB_ID}" || die "scancel failed for job ${JOB_ID}."
  echo "Cancelled Slurm job ${JOB_ID}."
else
  die "Required command not found: scancel"
fi

python3 "${REPO_ROOT}/training/serving_session.py" stop --job-id "${JOB_ID}" || {
  echo "WARNING: the job was cancelled but the session record could not be" >&2
  echo "updated. It expires on its own at the end of its allocation." >&2
}

echo
echo "Fine-tuned serving stopped."
echo "On aiswe.uwb.edu, close the tunnel when you are done:"
echo "  ./scripts/stop_finetuned_tunnel.sh"
