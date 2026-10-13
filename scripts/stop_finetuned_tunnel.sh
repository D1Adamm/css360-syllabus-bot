#!/usr/bin/env bash
# Close the CSS 360 fine-tuned SSH tunnel managed by start_finetuned_tunnel.sh,
# then return Fine-Tuned to the VM-local service.
#
# Usage:
#   ./scripts/stop_finetuned_tunnel.sh [--keep-local-stopped]
#   ./scripts/stop_finetuned_tunnel.sh --help

set -euo pipefail

TILLICUM_LOGIN="${TILLICUM_LOGIN:-${USER}@tillicum.hyak.uw.edu}"
STATE_ROOT="${XDG_STATE_HOME:-${HOME}/.local/state}/css360-syllabus-bot"
CONTROL_PATH="${STATE_ROOT}/ssh-ft-tunnel.sock"
STATE_FILE="${STATE_ROOT}/ft-tunnel.env"
FALLBACK_MARKER="${STATE_ROOT}/ft-fallback-local.env"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_SCRIPT="${SCRIPT_DIR}/aiswe_finetuned.sh"
LOCAL_UNIT="${LOCAL_UNIT:-aiswe-finetuned}"

usage() {
  cat <<'EOF'
Stop the fine-tuned SSH tunnel created by start_finetuned_tunnel.sh.

Usage:
  ./scripts/stop_finetuned_tunnel.sh [--keep-local-stopped]

Closes only this project's ControlMaster session. Does not use pkill.
Does not cancel the Tillicum GPU job (use scancel <JOB_ID> on Tillicum).

Then restores the VM-local service (aiswe-finetuned) and checks that it serves
the courses and versions it served before the fallback (aiswe_finetuned.sh
restore). That happens when start_finetuned_tunnel.sh --stop-local recorded
the stop, and also when the unit is enabled but stopped (the older manual
workflow). --keep-local-stopped skips it and keeps the record for later.
Safe to run again: with nothing to close and nothing to restore it does nothing.
EOF
}

KEEP_LOCAL_STOPPED=0
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  --keep-local-stopped) KEEP_LOCAL_STOPPED=1; shift ;;
  "") ;;
  *) echo "ERROR: Unknown argument: $1 (try --help)" >&2; exit 1 ;;
esac
if [[ $# -gt 0 ]]; then
  echo "ERROR: Unknown argument: $1 (try --help)" >&2
  exit 1
fi

# Prefer login/control path recorded when the tunnel was created.
if [[ -f "${STATE_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${STATE_FILE}"
  TILLICUM_LOGIN="${TILLICUM_LOGIN:-${USER}@tillicum.hyak.uw.edu}"
  CONTROL_PATH="${CONTROL_PATH:-${STATE_ROOT}/ssh-ft-tunnel.sock}"
fi

ACTIVE=0
if [[ -S "${CONTROL_PATH}" ]]; then
  if ssh -O check -o ControlPath="${CONTROL_PATH}" "${TILLICUM_LOGIN}" >/dev/null 2>&1; then
    ACTIVE=1
    echo "Closing fine-tuned SSH tunnel via ControlMaster..."
    if ssh -O exit -o ControlPath="${CONTROL_PATH}" "${TILLICUM_LOGIN}"; then
      echo "Tunnel closed."
    elif ssh -O check -o ControlPath="${CONTROL_PATH}" "${TILLICUM_LOGIN}" >/dev/null 2>&1; then
      # Still forwarding the port. Deleting the socket now would orphan it, and
      # starting the local service beside it would compete for the port.
      echo "ERROR: the tunnel did not close and still holds 127.0.0.1:${LOCAL_PORT:-9001}." >&2
      echo "  Nothing else was changed (state kept, ${LOCAL_UNIT} not started). Retry this script." >&2
      exit 1
    else
      echo "WARNING: ssh -O exit reported failure, but the master is gone; cleaning local state." >&2
    fi
  else
    echo "Control socket exists but is not an active SSH master; cleaning stale state."
  fi
else
  echo "No fine-tuned tunnel appears active (no control socket at ${CONTROL_PATH})."
fi

rm -f "${CONTROL_PATH}"
rm -f "${STATE_FILE}"

if [[ "${ACTIVE}" -eq 1 ]]; then
  echo "Remember to scancel the Tillicum GPU job if you are done using the GPU."
fi

# ---- back to the VM-local service -------------------------------------------
if ! command -v systemctl >/dev/null 2>&1; then
  echo "No systemctl on this host; ${LOCAL_UNIT} not checked."
  exit 0
fi
state="$(systemctl --user is-active "${LOCAL_UNIT}" 2>/dev/null || true)"
enabled="$(systemctl --user is-enabled "${LOCAL_UNIT}" 2>/dev/null || true)"

if [[ "${KEEP_LOCAL_STOPPED}" -eq 1 ]]; then
  if [[ "${state}" != "active" ]]; then
    echo "--keep-local-stopped: ${LOCAL_UNIT} left ${state:-stopped}. Fine-Tuned and Fine-Tuned + RAG FAIL for every course"
    echo "  until: ${LOCAL_SCRIPT} restore"
  fi
  exit 0
fi

if [[ -f "${FALLBACK_MARKER}" ]]; then
  echo "Restoring ${LOCAL_UNIT}, stopped for the fallback..."
elif [[ "${state}" != "active" && "${enabled}" == "enabled" ]]; then
  echo "${LOCAL_UNIT} is enabled but ${state:-not running} (stopped by hand before the fallback?); restoring it..."
else
  if [[ "${state}" == "active" ]]; then
    echo "${LOCAL_UNIT} is active; nothing to restore."
  fi
  exit 0
fi
if ! "${LOCAL_SCRIPT}" restore; then
  echo "Fine-Tuned / Fine-Tuned + RAG are still unavailable on the VM (Base and RAG are unaffected)." >&2
  exit 1
fi
echo "Fine-Tuned and Fine-Tuned + RAG are served by the VM-local service again."
