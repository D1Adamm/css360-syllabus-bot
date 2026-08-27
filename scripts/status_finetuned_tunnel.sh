#!/usr/bin/env bash
# Report local tunnel + backend fine-tuned readiness on the UWB VM.
#
# Usage:
#   ./scripts/status_finetuned_tunnel.sh
#   ./scripts/status_finetuned_tunnel.sh --help

set -euo pipefail

TILLICUM_LOGIN="${TILLICUM_LOGIN:-${USER}@tillicum.hyak.uw.edu}"
LOCAL_PORT="${LOCAL_PORT:-9001}"
BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8001}"
FINETUNED_LOCAL_URL="http://127.0.0.1:${LOCAL_PORT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HELPERS="${REPO_ROOT}/scripts/lib/finetuned_deploy_helpers.py"

STATE_ROOT="${XDG_STATE_HOME:-${HOME}/.local/state}/css360-syllabus-bot"
CONTROL_PATH="${STATE_ROOT}/ssh-ft-tunnel.sock"
STATE_FILE="${STATE_ROOT}/ft-tunnel.env"

usage() {
  cat <<'EOF'
Show fine-tuned tunnel and backend health on the UWB VM.

Usage:
  ./scripts/status_finetuned_tunnel.sh

Checks:
  - saved tunnel state / control socket
  - localhost:$LOCAL_PORT/health
  - $BACKEND_URL/api/health
  - $BACKEND_URL/api/fine-tuned/health

Exits nonzero if Fine-Tuned is unavailable.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

helpers() {
  python3 "${HELPERS}" "$@"
}

health_ready() {
  local url="$1"
  local body
  if ! body="$(curl -fsS --max-time 5 "${url}" 2>/dev/null)"; then
    return 1
  fi
  printf '%s' "${body}" | helpers health-ready >/dev/null 2>&1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -gt 0 ]]; then
  die "Unknown argument: $1 (try --help)"
fi

command -v curl >/dev/null 2>&1 || die "Required command not found: curl"
command -v python3 >/dev/null 2>&1 || die "Required command not found: python3"
[[ -f "${HELPERS}" ]] || die "Missing helpers module: ${HELPERS}"

SAVED_NODE=""
if [[ -f "${STATE_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${STATE_FILE}"
  SAVED_NODE="${NODE:-}"
  LOCAL_PORT="${LOCAL_PORT:-9001}"
  FINETUNED_LOCAL_URL="http://127.0.0.1:${LOCAL_PORT}"
  CONTROL_PATH="${CONTROL_PATH:-${STATE_ROOT}/ssh-ft-tunnel.sock}"
  TILLICUM_LOGIN="${TILLICUM_LOGIN:-${USER}@tillicum.hyak.uw.edu}"
fi

echo "Fine-tuned tunnel status"
if [[ -n "${SAVED_NODE}" ]]; then
  echo "Saved node: ${SAVED_NODE}"
else
  echo "Saved node: (none)"
fi
echo "Local port: ${LOCAL_PORT}"

if [[ -S "${CONTROL_PATH}" ]] && ssh -O check -o ControlPath="${CONTROL_PATH}" "${TILLICUM_LOGIN}" >/dev/null 2>&1; then
  echo "SSH control master: active"
else
  echo "SSH control master: inactive"
fi

FT_OK=1

if health_ready "${FINETUNED_LOCAL_URL}/health"; then
  echo "Tunnel health (${FINETUNED_LOCAL_URL}/health): OK (adapterLoaded=true)"
else
  echo "Tunnel health (${FINETUNED_LOCAL_URL}/health): UNAVAILABLE"
  FT_OK=0
fi

if curl -fsS --max-time 5 "${BACKEND_URL}/api/health" >/dev/null 2>&1; then
  echo "FastAPI (${BACKEND_URL}/api/health): OK"
else
  echo "FastAPI (${BACKEND_URL}/api/health): UNAVAILABLE"
  FT_OK=0
fi

if health_ready "${BACKEND_URL}/api/fine-tuned/health"; then
  echo "Fine-Tuned endpoint (${BACKEND_URL}/api/fine-tuned/health): OK (adapterLoaded=true)"
else
  echo "Fine-Tuned endpoint (${BACKEND_URL}/api/fine-tuned/health): UNAVAILABLE"
  FT_OK=0
fi

if [[ "${FT_OK}" -eq 1 ]]; then
  echo "Overall: Fine-Tuned / Fine-Tuned + RAG available"
  exit 0
fi

echo "Overall: Fine-Tuned unavailable"
exit 1
