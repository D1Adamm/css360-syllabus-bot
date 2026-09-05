#!/usr/bin/env bash
# Open an SSH tunnel from the UWB VM to a Tillicum fine-tuned inference node,
# point backend/.env at the local forward, and restart aiswe-backend.
#
# Usage (on aiswe.uwb.edu, from repository root):
#   ./scripts/start_finetuned_tunnel.sh --from-backend
#   ./scripts/start_finetuned_tunnel.sh g014
#   ./scripts/start_finetuned_tunnel.sh --help
#
# `--from-backend` asks the application which node the Tillicum start script
# registered, instead of the operator reading a hostname off one machine and
# typing it into another. The compute node changes with every Slurm job, so that
# copy was both required and the easiest thing to get wrong.

set -euo pipefail

TILLICUM_LOGIN="${TILLICUM_LOGIN:-${USER}@tillicum.hyak.uw.edu}"
LOCAL_PORT="${LOCAL_PORT:-9001}"
REMOTE_PORT="${REMOTE_PORT:-8001}"
BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8001}"
FINETUNED_LOCAL_URL="http://127.0.0.1:${LOCAL_PORT}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-120}"
BACKEND_HEALTH_TIMEOUT_SECONDS="${BACKEND_HEALTH_TIMEOUT_SECONDS:-60}"
POLL_SECONDS="${POLL_SECONDS:-3}"
BACKEND_SERVICE="${BACKEND_SERVICE:-aiswe-backend}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HELPERS="${REPO_ROOT}/scripts/lib/finetuned_deploy_helpers.py"
ENV_FILE="${REPO_ROOT}/backend/.env"

STATE_ROOT="${XDG_STATE_HOME:-${HOME}/.local/state}/css360-syllabus-bot"
CONTROL_PATH="${STATE_ROOT}/ssh-ft-tunnel.sock"
STATE_FILE="${STATE_ROOT}/ft-tunnel.env"

usage() {
  cat <<'EOF'
Open the UWB VM -> Tillicum -> compute-node tunnel for fine-tuned inference.

Usage:
  ./scripts/start_finetuned_tunnel.sh --from-backend
  ./scripts/start_finetuned_tunnel.sh <compute-node>
  ./scripts/start_finetuned_tunnel.sh --help

Examples:
  ./scripts/start_finetuned_tunnel.sh --from-backend   # look the node up
  ./scripts/start_finetuned_tunnel.sh g014             # name it explicitly

--from-backend reads the session ./training/start_finetuned_service.sh recorded
on Tillicum. It needs TRAINING_API_BASE_URL and TRAINING_WORKER_TOKEN — the same
pair the training worker uses — and reads them from backend/.env, the file the
backend service itself loads, so no shell export is needed. A variable already
exported in the environment (or set in .env.local) takes precedence.

Environment overrides:
  TILLICUM_LOGIN     SSH target (default: $USER@tillicum.hyak.uw.edu)
  LOCAL_PORT         Local forward port (default: 9001)
  REMOTE_PORT        Remote inference port (default: 8001)
  BACKEND_URL        Local FastAPI base URL (default: http://127.0.0.1:8001)
  BACKEND_SERVICE    systemd --user unit name (default: aiswe-backend)

UW Duo / password prompts are intentional and interactive. Credentials are not stored.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_cmd() {
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1 || die "Required command not found: ${cmd}"
}

helpers() {
  python3 "${HELPERS}" "$@"
}

local_tcp_open() {
  local port="$1"
  if command -v nc >/dev/null 2>&1; then
    nc -z 127.0.0.1 "${port}" >/dev/null 2>&1
    return $?
  fi
  # Bash /dev/tcp fallback
  (echo >/dev/tcp/127.0.0.1/"${port}") >/dev/null 2>&1
}

health_ready() {
  local url="$1"
  local body
  if ! body="$(curl -fsS --max-time 5 "${url}" 2>/dev/null)"; then
    return 1
  fi
  printf '%s' "${body}" | helpers health-ready >/dev/null 2>&1
}

wait_for_url_health() {
  local url="$1"
  local timeout_seconds="$2"
  local label="$3"
  local deadline=$((SECONDS + timeout_seconds))
  while (( SECONDS < deadline )); do
    if health_ready "${url}"; then
      return 0
    fi
    sleep "${POLL_SECONDS}"
  done
  die "${label}"
}

control_master_alive() {
  [[ -S "${CONTROL_PATH}" ]] || return 1
  ssh -O check -o ControlPath="${CONTROL_PATH}" "${TILLICUM_LOGIN}" >/dev/null 2>&1
}

write_state() {
  local node="$1"
  mkdir -p "${STATE_ROOT}"
  cat > "${STATE_FILE}" <<EOF
NODE=${node}
LOCAL_PORT=${LOCAL_PORT}
REMOTE_PORT=${REMOTE_PORT}
TILLICUM_LOGIN=${TILLICUM_LOGIN}
CONTROL_PATH=${CONTROL_PATH}
CREATED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
}

read_state_field() {
  local key="$1"
  local line
  [[ -f "${STATE_FILE}" ]] || return 1
  line="$(grep -E "^${key}=" "${STATE_FILE}" | head -n 1 || true)"
  [[ -n "${line}" ]] || return 1
  printf '%s\n' "${line#*=}"
}

managed_tunnel_is_trustworthy() {
  [[ -f "${STATE_FILE}" ]] || return 1
  control_master_alive
}

NODE=""
FROM_BACKEND=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --from-backend)
      FROM_BACKEND=1
      shift
      ;;
    -*)
      die "Unknown option: $1 (try --help)"
      ;;
    *)
      if [[ -n "${NODE}" ]]; then
        die "Expected exactly one compute node hostname (got extra: $1)"
      fi
      NODE="$1"
      shift
      ;;
  esac
done

if [[ "${FROM_BACKEND}" -eq 1 ]]; then
  [[ -z "${NODE}" ]] || die "Pass either --from-backend or a hostname, not both."
  echo "Looking up the current serving session..."
  # serving_session.py reads the backend URL and worker token from the
  # environment, then .env.local/.env, then backend/.env. Exit 2 means it could
  # not ask the backend at all; exit 1 means it asked and there is no session.
  lookup_status=0
  SESSION_JSON="$(python3 "${REPO_ROOT}/training/serving_session.py" show --json)" \
    || lookup_status=$?
  if [[ "${lookup_status}" -eq 2 ]]; then
    die "Could not look up the serving session (see the message above).
  --from-backend needs TRAINING_API_BASE_URL and TRAINING_WORKER_TOKEN, read from
  ${ENV_FILE} or the environment, and a reachable backend."
  elif [[ "${lookup_status}" -ne 0 ]]; then
    die "No serving session is recorded. Start one on Tillicum first:
  ./training/start_finetuned_service.sh"
  fi
  NODE="$(printf '%s' "${SESSION_JSON}" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("node") or "")')"
  REMOTE_PORT="$(printf '%s' "${SESSION_JSON}" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("port") or 8001)')"
  SESSION_EXPIRES="$(printf '%s' "${SESSION_JSON}" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("expiresAt") or "")')"
  [[ -n "${NODE}" ]] || die "The recorded serving session names no compute node."
  echo "Session node: ${NODE}:${REMOTE_PORT} (session ends ${SESSION_EXPIRES:-unknown})"
fi

[[ -n "${NODE}" ]] || die "Compute node hostname required. Either:
  ./scripts/start_finetuned_tunnel.sh --from-backend
  ./scripts/start_finetuned_tunnel.sh g014"
NODE="$(helpers validate-hostname "${NODE}")" || die "Invalid compute node hostname."

require_cmd ssh
require_cmd curl
require_cmd python3
require_cmd systemctl
[[ -f "${HELPERS}" ]] || die "Missing helpers module: ${HELPERS}"
[[ -d "${REPO_ROOT}/backend" ]] || die "Does not look like the repo root (missing backend/). Run from the repository root."

mkdir -p "${STATE_ROOT}"

echo "Fine-tuned tunnel startup"
echo "Node: ${NODE}"
echo "Local forward: ${FINETUNED_LOCAL_URL} -> ${NODE}:${REMOTE_PORT}"
echo "SSH target: ${TILLICUM_LOGIN}"
echo

# Reuse only when this project's managed tunnel is healthy AND matches NODE.
if health_ready "${FINETUNED_LOCAL_URL}/health"; then
  if managed_tunnel_is_trustworthy; then
    SAVED_NODE="$(read_state_field NODE || true)"
    if [[ -z "${SAVED_NODE}" ]]; then
      die "localhost:${LOCAL_PORT}/health is healthy and a control socket exists, but the state file is missing NODE=. Refusing to rewrite state for ${NODE}. Run: ./scripts/stop_finetuned_tunnel.sh then retry."
    fi
    if [[ "${SAVED_NODE}" == "${NODE}" ]]; then
      echo "Existing healthy managed tunnel for ${NODE} on localhost:${LOCAL_PORT}; reusing (not opening another)."
    else
      die "Healthy managed tunnel on localhost:${LOCAL_PORT} is for node '${SAVED_NODE}', not requested '${NODE}'. Run: ./scripts/stop_finetuned_tunnel.sh then retry with ${NODE}."
    fi
  else
    die "localhost:${LOCAL_PORT}/health is healthy, but there is no trustworthy project tunnel state/control socket proving it belongs to ${NODE}. Refusing to overwrite state or kill arbitrary processes. Free the port or identify the forwarder, then retry. If this was our tunnel, run: ./scripts/stop_finetuned_tunnel.sh"
  fi
else
  if local_tcp_open "${LOCAL_PORT}"; then
    die "Local port ${LOCAL_PORT} is occupied by another process, but /health is not a healthy fine-tuned endpoint. Free the port (or set LOCAL_PORT) and retry. Refusing to kill arbitrary processes."
  fi

  if control_master_alive; then
    echo "Found stale SSH control socket; closing it before opening a new tunnel..."
    ssh -O exit -o ControlPath="${CONTROL_PATH}" "${TILLICUM_LOGIN}" >/dev/null 2>&1 || true
    rm -f "${CONTROL_PATH}"
  fi
  # Drop stale state that pointed at a previous node before creating a new tunnel.
  rm -f "${STATE_FILE}"

  if command -v nc >/dev/null 2>&1; then
    if ! nc -z -w 5 tillicum.hyak.uw.edu 22 >/dev/null 2>&1; then
      die "Cannot reach tillicum.hyak.uw.edu:22 from this VM (nc check failed)."
    fi
    echo "Reachability check: tillicum.hyak.uw.edu:22 OK"
  else
    echo "nc not available; skipping tillicum.hyak.uw.edu:22 reachability probe."
  fi

  echo
  echo "Opening SSH tunnel (complete UW/Duo authentication if prompted)..."
  # -f backgrounds only after authentication succeeds.
  ssh -f -N \
    -o ControlMaster=yes \
    -o "ControlPath=${CONTROL_PATH}" \
    -o ControlPersist=yes \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -L "${LOCAL_PORT}:${NODE}:${REMOTE_PORT}" \
    "${TILLICUM_LOGIN}" || die "SSH tunnel setup failed (auth cancelled, network error, or port forward refused)."

  write_state "${NODE}"

  wait_for_url_health \
    "${FINETUNED_LOCAL_URL}/health" \
    "${HEALTH_TIMEOUT_SECONDS}" \
    "Tillicum tunnel established but /health never became ready at ${FINETUNED_LOCAL_URL}/health (need status=ok and adapterLoaded=true)."
fi

echo "Tillicum health via tunnel: OK"

CHANGED="$(helpers update-env-key "${ENV_FILE}" "FINETUNED_SERVICE_URL" "${FINETUNED_LOCAL_URL}")" \
  || die "Failed to update ${ENV_FILE} (FINETUNED_SERVICE_URL)."
if [[ "${CHANGED}" == "updated" ]]; then
  echo "Updated FINETUNED_SERVICE_URL in backend/.env"
else
  echo "FINETUNED_SERVICE_URL already pointed at ${FINETUNED_LOCAL_URL}"
fi

echo "Restarting ${BACKEND_SERVICE} ..."
systemctl --user restart "${BACKEND_SERVICE}" \
  || die "Failed to restart ${BACKEND_SERVICE}. Is the user systemd unit installed?"

echo "Waiting for FastAPI ${BACKEND_URL}/api/health ..."
deadline=$((SECONDS + BACKEND_HEALTH_TIMEOUT_SECONDS))
backend_ok=0
while (( SECONDS < deadline )); do
  if curl -fsS --max-time 5 "${BACKEND_URL}/api/health" >/dev/null 2>&1; then
    backend_ok=1
    break
  fi
  sleep "${POLL_SECONDS}"
done
[[ "${backend_ok}" -eq 1 ]] || die "Backend restarted but ${BACKEND_URL}/api/health never became ready."

if ! health_ready "${BACKEND_URL}/api/fine-tuned/health"; then
  # Give the backend a moment after /health before declaring FT failure.
  sleep 2
  if ! health_ready "${BACKEND_URL}/api/fine-tuned/health"; then
    die "Backend restarted but /api/fine-tuned/health failed (need status=ok and adapterLoaded=true). Check FINETUNED_SERVICE_URL and the Tillicum job."
  fi
fi

echo
echo "Fine-tuned backend READY"
echo "Tunnel:"
echo "  localhost:${LOCAL_PORT} -> ${NODE}:${REMOTE_PORT} via tillicum.hyak.uw.edu"
echo "Tillicum health: OK"
echo "Adapter loaded: true"
echo "FastAPI: OK"
echo "Fine-Tuned endpoint: OK"
echo "Fine-Tuned + RAG uses the same service and is now available."
echo
echo "Status: ./scripts/status_finetuned_tunnel.sh"
echo "Stop tunnel: ./scripts/stop_finetuned_tunnel.sh"
echo
echo "The Tillicum session has a bounded wall clock. When it ends, the tunnel"
echo "goes dead and Fine-Tuned becomes unavailable until a new session is"
echo "started; Base and RAG are unaffected throughout."
