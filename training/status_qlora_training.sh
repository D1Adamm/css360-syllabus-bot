#!/usr/bin/env bash
# Show active (and recent) QLoRA smoke/full training jobs for the current user.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HELPERS="${REPO_ROOT}/scripts/lib/qlora_training_helpers.py"

usage() {
  cat <<'EOF'
Show QLoRA smoke/full training job status for the current user.

Usage:
  ./training/status_qlora_training.sh
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

show_job_line() {
  local line="$1"
  local job_id name state node elapsed time_left meta out_dir
  job_id="$(printf '%s\n' "${line}" | helpers parse-squeue-line --field job_id)"
  name="$(printf '%s\n' "${line}" | helpers parse-squeue-line --field name)"
  state="$(printf '%s\n' "${line}" | helpers parse-squeue-line --field state)"
  node="$(printf '%s\n' "${line}" | helpers parse-squeue-line --field node)"
  elapsed="$(printf '%s\n' "${line}" | helpers parse-squeue-line --field elapsed)"
  time_left="$(printf '%s\n' "${line}" | helpers parse-squeue-line --field time_left)"
  out_dir=""
  meta="${REPO_ROOT}/training/logs/qlora-job-${job_id}.env"
  if [[ -f "${meta}" ]]; then
    # shellcheck disable=SC1090
    TRAINING_OUTPUT_DIR=""
    source "${meta}"
    out_dir="${TRAINING_OUTPUT_DIR:-}"
  fi

  echo "Job ID: ${job_id}"
  echo "  Name: ${name}"
  echo "  State: ${state}"
  echo "  Elapsed: ${elapsed:-unknown}"
  echo "  Time left: ${time_left:-unknown}"
  echo "  Node: ${node:-"(none yet)"}"
  case "${name}" in
    css360-qlora-smoke)
      echo "  Logs:"
      echo "    ${REPO_ROOT}/training/logs/smoke-${job_id}.out"
      echo "    ${REPO_ROOT}/training/logs/smoke-${job_id}.err"
      ;;
    css360-qlora-train)
      echo "  Logs:"
      echo "    ${REPO_ROOT}/training/logs/train-${job_id}.out"
      echo "    ${REPO_ROOT}/training/logs/train-${job_id}.err"
      ;;
  esac
  if [[ -n "${out_dir}" ]]; then
    echo "  Output dir: ${out_dir}"
    echo "  Adapter: ${out_dir}/adapter"
  else
    echo "  Output dir: (unknown — no local run metadata for this job id)"
  fi
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
[[ -f "${HELPERS}" ]] || die "Missing helpers: ${HELPERS}"

ACTIVE=0
echo "QLoRA training status (user=${USER})"
echo

for JOB_NAME in css360-qlora-smoke css360-qlora-train; do
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    state="$(printf '%s\n' "${line}" | helpers parse-squeue-line --field state)"
    if is_active_state "${state}"; then
      show_job_line "${line}"
      echo
      ACTIVE=1
    fi
  done < <(squeue -u "${USER}" -n "${JOB_NAME}" -h -o "%i %j %t %N %M %L" 2>/dev/null || true)
done

if [[ "${ACTIVE}" -eq 0 ]]; then
  echo "No active css360-qlora-smoke or css360-qlora-train jobs."
  echo
fi

# Recent finished jobs from local metadata, if any.
if command -v sacct >/dev/null 2>&1; then
  RECENT_COUNT=0
  echo "Recent tracked jobs (sacct):"
  # shellcheck disable=SC2012
  while IFS= read -r meta; do
    [[ -z "${meta}" ]] && continue
    # shellcheck disable=SC1090
    JOB_ID=""
    MODE=""
    TRAINING_OUTPUT_DIR=""
    # shellcheck source=/dev/null
    source "${meta}"
    [[ -n "${JOB_ID:-}" ]] || continue
    if squeue -j "${JOB_ID}" -h -o "%i" 2>/dev/null | grep -q .; then
      continue
    fi
    echo "  Job ${JOB_ID} (${MODE:-unknown})"
    sacct -j "${JOB_ID}" -X --parsable2 --noheader \
      -o JobID,State,ExitCode,Elapsed,NodeList 2>/dev/null | sed 's/^/    /' || true
    if [[ -n "${TRAINING_OUTPUT_DIR:-}" ]]; then
      echo "    Output: ${TRAINING_OUTPUT_DIR}"
    fi
    RECENT_COUNT=$((RECENT_COUNT + 1))
    [[ "${RECENT_COUNT}" -ge 3 ]] && break
  done < <(ls -t "${REPO_ROOT}/training/logs"/qlora-job-*.env 2>/dev/null || true)
  if [[ "${RECENT_COUNT}" -eq 0 ]]; then
    echo "  (none)"
  fi
fi

exit 0
