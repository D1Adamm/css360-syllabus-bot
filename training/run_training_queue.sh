#!/usr/bin/env bash
# Claim one queued training run and submit it through start_qlora_training.sh.
#
# Run this on Tillicum inside a session you logged into normally. It reads the
# training queue over HTTPS, claims at most one run, checks it against the
# prepared dataset already pushed to this machine, and invokes
# training/start_qlora_training.sh --yes. --dry-run never claims, writes, or
# spawns the launcher.
#
# Usage (from the repository root on Tillicum):
#   ./training/run_training_queue.sh --once
#   ./training/run_training_queue.sh --once --dry-run
#   ./training/run_training_queue.sh --once --course css-360-winter-2026-a7rp

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_training_queue.py"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

command -v python3 >/dev/null 2>&1 || die "Required command not found: python3"
[[ -f "${RUNNER}" ]] || die "Missing runner: ${RUNNER}"
[[ -f "${REPO_ROOT}/scripts/lib/qlora_training_helpers.py" ]] \
  || die "Does not look like the repo root: ${REPO_ROOT}"

cd "${REPO_ROOT}"
exec python3 "${RUNNER}" "$@"
