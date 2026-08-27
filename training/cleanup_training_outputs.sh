#!/usr/bin/env bash
# Show — and, only when told to, reclaim — disk under training_outputs/.
#
# Usage (from repository root on Tillicum):
#   ./training/cleanup_training_outputs.sh              # print the plan only
#   ./training/cleanup_training_outputs.sh --apply      # act on it, with confirmation
#   ./training/cleanup_training_outputs.sh --no-smoke   # keep smoke runs too
#
# Dry run by default, and deliberately so: this is the only script in the
# repository that removes anything, and the cost of a wrong deletion here is a
# model artifact that cannot be recreated without another GPU allocation.
#
# What it will ever propose:
#   - <run>/checkpoints/ from completed full runs — intermediate Trainer state,
#     which is where nearly all the disk goes. The adapter is saved separately
#     at <run>/adapter/ and is not touched.
#   - whole smoke run directories — a smoke run trains four examples for three
#     optimizer steps; it is a rehearsal and is never registered as a model.
#
# What it will never propose, whatever else is true:
#   - anything under serving/ — the published per-course adapters inference loads
#   - any adapter/ directory, or adapter-backups/
#   - a run a published current.json says a served adapter came from
#   - a run this cluster still owes the application a report for
#   - a run with no runtime report, which is either still going or died

set -euo pipefail

APPLY=0
NO_SMOKE=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RETENTION="${REPO_ROOT}/scripts/lib/retention.py"
OUTPUTS_ROOT="${TRAINING_OUTPUTS_ROOT:-/gpfs/projects/simswe/${USER}/training_outputs}"

usage() {
  cat <<'EOF'
Show, and optionally reclaim, disk under training_outputs/.

Usage:
  ./training/cleanup_training_outputs.sh
  ./training/cleanup_training_outputs.sh --apply
  ./training/cleanup_training_outputs.sh --no-smoke --apply

Options:
  --apply     Delete what the plan proposes, after an explicit confirmation
  --no-smoke  Leave smoke run directories alone as well as everything protected
  -h, --help  Show this help

Environment:
  TRAINING_OUTPUTS_ROOT  default: /gpfs/projects/simswe/$USER/training_outputs

Registered model artifacts, published course adapters, and any run still
outstanding with the application are never proposed.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --apply)
      APPLY=1
      shift
      ;;
    --no-smoke)
      NO_SMOKE=1
      shift
      ;;
    *)
      die "Unknown argument: $1 (try --help)"
      ;;
  esac
done

command -v python3 >/dev/null 2>&1 || die "Required command not found: python3"
[[ -f "${RETENTION}" ]] || die "Missing helper: ${RETENTION}"
[[ -d "${OUTPUTS_ROOT}" ]] || die "No training outputs directory at ${OUTPUTS_ROOT}"

PLAN_ARGS=("${OUTPUTS_ROOT}" --repo-root "${REPO_ROOT}")
if [[ "${NO_SMOKE}" -eq 1 ]]; then
  PLAN_ARGS+=(--no-smoke)
fi

python3 "${RETENTION}" "${PLAN_ARGS[@]}"

if [[ "${APPLY}" -ne 1 ]]; then
  echo
  echo "(dry run — nothing was deleted). Re-run with --apply to act on this plan."
  exit 0
fi

PLAN_JSON="$(python3 "${RETENTION}" "${PLAN_ARGS[@]}" --json)"
COUNT="$(printf '%s' "${PLAN_JSON}" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["candidates"]))')"

if [[ "${COUNT}" -eq 0 ]]; then
  echo
  echo "Nothing to delete."
  exit 0
fi

echo
printf "Type DELETE to remove the %s item(s) above: " "${COUNT}"
read -r reply </dev/tty || die "Could not read confirmation from terminal."
[[ "${reply}" == "DELETE" ]] || {
  echo "Aborted (confirmation phrase not matched)."
  exit 1
}

# Each path is re-checked against the safety rule immediately before it is
# removed, rather than trusted from the plan that was printed. A plan can be
# read and acted on minutes later, and in between an adapter may have been
# published from a run that was a candidate when the plan was made.
PLAN_FILE="$(mktemp)"
trap 'rm -f "${PLAN_FILE}"' EXIT
printf '%s' "${PLAN_JSON}" > "${PLAN_FILE}"

python3 "${REPO_ROOT}/scripts/lib/apply_retention_plan.py" \
  --plan "${PLAN_FILE}" \
  --outputs-root "${OUTPUTS_ROOT}"
