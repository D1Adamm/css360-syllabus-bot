#!/usr/bin/env bash
# Close only the CSS 360 fine-tuned SSH tunnel managed by start_finetuned_tunnel.sh.
#
# Usage:
#   ./scripts/stop_finetuned_tunnel.sh
#   ./scripts/stop_finetuned_tunnel.sh --help

set -euo pipefail

TILLICUM_LOGIN="${TILLICUM_LOGIN:-${USER}@tillicum.hyak.uw.edu}"
STATE_ROOT="${XDG_STATE_HOME:-${HOME}/.local/state}/css360-syllabus-bot"
CONTROL_PATH="${STATE_ROOT}/ssh-ft-tunnel.sock"
STATE_FILE="${STATE_ROOT}/ft-tunnel.env"

usage() {
  cat <<'EOF'
Stop the fine-tuned SSH tunnel created by start_finetuned_tunnel.sh.

Usage:
  ./scripts/stop_finetuned_tunnel.sh

Closes only this project's ControlMaster session. Does not use pkill.
Does not cancel the Tillicum GPU job (use scancel <JOB_ID> on Tillicum).
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
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
    else
      echo "WARNING: ssh -O exit reported failure; cleaning local state anyway." >&2
    fi
  else
    echo "Control socket exists but is not an active SSH master; cleaning stale state."
  fi
else
  echo "No fine-tuned tunnel appears active (no control socket at ${CONTROL_PATH})."
fi

rm -f "${CONTROL_PATH}"
rm -f "${STATE_FILE}"

if [[ "${ACTIVE}" -eq 0 ]]; then
  exit 0
fi

echo "Local Fine-Tuned / Fine-Tuned + RAG paths will be unavailable until the tunnel is started again."
echo "Base and RAG on the UWB VM are unaffected."
echo "Remember to scancel the Tillicum GPU job if you are done using the GPU."
