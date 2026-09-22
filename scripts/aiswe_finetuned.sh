#!/usr/bin/env bash
# Operate the VM-local fine-tuned inference service (`aiswe-finetuned`), the
# normal production path for Fine-Tuned and Fine-Tuned + RAG on aiswe.uwb.edu.
#
#   ./scripts/aiswe_finetuned.sh install [--check]     # render + install the user unit, env file, backend URL
#   ./scripts/aiswe_finetuned.sh check                 # read-only: unit, mapping, port owner, tunnel, Ollama, health
#   ./scripts/aiswe_finetuned.sh set-mapping <courseId> <vN> <ollamaTag> [--replace] [--dry-run]
#   ./scripts/aiswe_finetuned.sh start | stop | restart | status | logs
#   ./scripts/aiswe_finetuned.sh preflight             # what the unit runs before ExecStart
#
# The local service and the Tillicum SSH tunnel both listen on 127.0.0.1:9001.
# `start` and `preflight` refuse while the tunnel owns the port, and
# `scripts/start_finetuned_tunnel.sh` refuses while this unit is active. Nothing
# here kills a process it did not start: when the port is taken, the owner is
# named and the operator decides.
#
# Nothing here reads or prints a secret. backend/.env is touched for exactly one
# non-secret key, FINETUNED_SERVICE_URL, through the same helper the tunnel
# script uses.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

UNIT="${AISWE_FINETUNED_UNIT:-aiswe-finetuned}"
PORT="${INFERENCE_PORT:-9001}"
UNIT_SRC="${REPO_ROOT}/training/inference_service/aiswe-finetuned.service"
UNIT_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
UNIT_DST="${UNIT_DIR}/${UNIT}.service"
ENV_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/aiswe"
ENV_FILE="${AISWE_FINETUNED_ENV:-${ENV_DIR}/finetuned.env}"
ENV_EXAMPLE="${REPO_ROOT}/scripts/finetuned.env.example"
BACKEND_ENV="${REPO_ROOT}/backend/.env"
BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8001}"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
SERVICE_URL="http://127.0.0.1:${PORT}"
OLLAMA_WAIT_SECONDS="${OLLAMA_WAIT_SECONDS:-90}"
HEALTH_WAIT_SECONDS="${HEALTH_WAIT_SECONDS:-30}"

STATE_ROOT="${XDG_STATE_HOME:-${HOME}/.local/state}/css360-syllabus-bot"
TUNNEL_STATE_FILE="${STATE_ROOT}/ft-tunnel.env"
TUNNEL_CONTROL_PATH="${STATE_ROOT}/ssh-ft-tunnel.sock"

LOCAL_HELPERS="${REPO_ROOT}/scripts/lib/finetuned_local_helpers.py"
DEPLOY_HELPERS="${REPO_ROOT}/scripts/lib/finetuned_deploy_helpers.py"
VENV_PY="${REPO_ROOT}/backend/.venv/bin/python"

usage() {
  sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

py() {
  # The helpers are standard library only; prefer the interpreter the service
  # itself runs under so a check exercises the same Python.
  if [[ -x "${VENV_PY}" ]]; then
    "${VENV_PY}" "$@"
  else
    python3 "$@"
  fi
}

local_helpers() { py "${LOCAL_HELPERS}" "$@"; }
deploy_helpers() { py "${DEPLOY_HELPERS}" "$@"; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

# --------------------------------------------------------------------------- #
# Facts
# --------------------------------------------------------------------------- #

listeners_text() {
  # `ss` on the VM (Linux). `lsof` keeps `check` useful on a developer machine.
  if command -v ss >/dev/null 2>&1; then
    ss -H -ltnp 2>/dev/null || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null | awk 'NR>1 {printf "LISTEN 0 0 %s *:* users:((\"%s\",pid=%s,fd=0))\n", $9, $1, $2}' || true
  fi
}

port_owner_json() { listeners_text | local_helpers port-owner --port "${PORT}"; }
port_owner_field() { listeners_text | local_helpers port-owner --port "${PORT}" --field "$1"; }

tunnel_control_alive() {
  [[ -S "${TUNNEL_CONTROL_PATH}" ]] || return 1
  local login
  login="$(grep -E '^TILLICUM_LOGIN=' "${TUNNEL_STATE_FILE}" 2>/dev/null | head -n 1 | cut -d= -f2- || true)"
  login="${login:-${USER}@tillicum.hyak.uw.edu}"
  ssh -O check -o ControlPath="${TUNNEL_CONTROL_PATH}" "${login}" >/dev/null 2>&1
}

unit_state() {
  # active | inactive | failed | activating | unknown
  systemctl --user is-active "${UNIT}" 2>/dev/null || true
}

unit_enabled() {
  systemctl --user is-enabled "${UNIT}" 2>/dev/null || true
}

health_json() {
  curl -fsS --max-time 5 "${SERVICE_URL}/health" 2>/dev/null || true
}

health_ready() {
  local body
  body="$(health_json)"
  [[ -n "${body}" ]] || return 1
  printf '%s' "${body}" | deploy_helpers health-ready >/dev/null 2>&1
}

ollama_up() {
  curl -fsS --max-time 5 "${OLLAMA_URL}/api/tags" >/dev/null 2>&1
}

refuse_if_tunnel_owns_port() {
  # Two independent signals, either of which is decisive: the tunnel script's
  # own control socket, and an ssh process listening on the port.
  if tunnel_control_alive; then
    die "The Tillicum fallback tunnel is active (control socket ${TUNNEL_CONTROL_PATH}).
  It owns 127.0.0.1:${PORT}. Close it first, then retry:
    ./scripts/stop_finetuned_tunnel.sh"
  fi
  local kind
  kind="$(port_owner_field kind)"
  case "${kind}" in
    tunnel)
      die "An ssh process owns 127.0.0.1:${PORT} (pid $(port_owner_field pid)) — the Tillicum tunnel.
  Close it first, then retry:
    ./scripts/stop_finetuned_tunnel.sh"
      ;;
  esac
}

# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

render_unit() {
  # A literal substitution; the checkout path is the only rendered value.
  sed "s|@REPO_ROOT@|${REPO_ROOT}|g" "${UNIT_SRC}"
}

cmd_install() {
  local check_only=0
  [[ "${1:-}" == "--check" ]] && check_only=1
  [[ "${check_only}" -eq 1 ]] || require_cmd systemctl
  [[ -f "${UNIT_SRC}" ]] || die "Missing unit template: ${UNIT_SRC}"
  [[ -x "${VENV_PY}" ]] || die "Missing backend virtualenv: ${VENV_PY} (create it first; the service runs from it)."
  [[ -d "${REPO_ROOT}/backend" ]] || die "Not a repository checkout: ${REPO_ROOT}"

  local rendered
  rendered="$(render_unit)"

  echo "Unit:        ${UNIT_DST}"
  if [[ -f "${UNIT_DST}" ]] && [[ "$(cat "${UNIT_DST}")" == "${rendered}" ]]; then
    echo "  unchanged"
  else
    if [[ "${check_only}" -eq 1 ]]; then
      echo "  would write (rendered from ${UNIT_SRC#"${REPO_ROOT}"/})"
    else
      mkdir -p "${UNIT_DIR}"
      printf '%s\n' "${rendered}" > "${UNIT_DST}"
      echo "  written"
    fi
  fi

  echo "Environment: ${ENV_FILE}"
  if [[ -f "${ENV_FILE}" ]]; then
    echo "  exists (kept; edit with: $0 set-mapping <courseId> <vN> <ollamaTag>)"
  elif [[ "${check_only}" -eq 1 ]]; then
    echo "  would create from scripts/finetuned.env.example (empty mapping, mode 600)"
  else
    mkdir -p "${ENV_DIR}"
    (umask 077 && cp "${ENV_EXAMPLE}" "${ENV_FILE}")
    chmod 600 "${ENV_FILE}"
    echo "  created from scripts/finetuned.env.example (empty mapping, mode 600)"
  fi
  if [[ -f "${ENV_FILE}" ]]; then
    chmod 600 "${ENV_FILE}" 2>/dev/null || true
    local_helpers validate-env "${ENV_FILE}" | sed 's/^/  mapping: /' || die "The mapping in ${ENV_FILE} is malformed; fix it before starting the unit."
  fi

  echo "Backend:     ${BACKEND_ENV} FINETUNED_SERVICE_URL"
  if [[ "${check_only}" -eq 1 ]]; then
    if [[ -f "${BACKEND_ENV}" ]] && grep -qE "^FINETUNED_SERVICE_URL=${SERVICE_URL}/?$" "${BACKEND_ENV}"; then
      echo "  already ${SERVICE_URL}"
    else
      echo "  would set ${SERVICE_URL} (then: systemctl --user restart aiswe-backend)"
    fi
  else
    [[ -f "${BACKEND_ENV}" ]] || die "Missing ${BACKEND_ENV}; the backend is not configured on this host."
    local changed
    changed="$(deploy_helpers update-env-key "${BACKEND_ENV}" FINETUNED_SERVICE_URL "${SERVICE_URL}")"
    if [[ "${changed}" == "updated" ]]; then
      echo "  set to ${SERVICE_URL} — restart the backend to pick it up: systemctl --user restart aiswe-backend"
    else
      echo "  already ${SERVICE_URL}"
    fi
  fi

  if [[ "${check_only}" -eq 1 ]]; then
    echo "Check only; nothing written. Lingering: $(loginctl show-user "${USER}" -p Linger --value 2>/dev/null || echo unknown)"
    return 0
  fi

  systemctl --user daemon-reload
  systemctl --user enable "${UNIT}" >/dev/null 2>&1 || die "Could not enable ${UNIT}."
  echo "Enabled:     ${UNIT} (starts at boot when lingering is on)"
  local linger
  linger="$(loginctl show-user "${USER}" -p Linger --value 2>/dev/null || echo unknown)"
  if [[ "${linger}" != "yes" ]]; then
    echo "WARNING: lingering is '${linger}' for ${USER}; user units stop at logout and do not start at boot." >&2
    echo "         Enable it once with: loginctl enable-linger ${USER}" >&2
  fi
  echo "Next: $0 set-mapping <courseId> <vN> <ollamaTag>   (if the mapping is empty)"
  echo "      $0 start"
}

cmd_set_mapping() {
  [[ $# -ge 3 ]] || die "Usage: $0 set-mapping <courseId> <vN> <ollamaTag> [--replace] [--dry-run]"
  local course="$1" version="$2" tag="$3"
  shift 3
  local args=()
  local dry=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --replace) args+=(--replace) ;;
      --dry-run) args+=(--dry-run); dry=1 ;;
      *) die "Unknown option: $1" ;;
    esac
    shift
  done
  [[ -f "${ENV_FILE}" ]] || die "Missing ${ENV_FILE}; run: $0 install"
  local out
  out="$(local_helpers set-mapping "${ENV_FILE}" "${course}" "${version}" "${tag}" "${args[@]}")" \
    || die "Mapping not changed (see above)."
  echo "${out}"
  if [[ "${dry}" -eq 0 && "${out}" != "unchanged" ]]; then
    if ollama_up; then
      if ! curl -fsS --max-time 5 "${OLLAMA_URL}/api/tags" | py -c '
import json, sys
wanted = sys.argv[1]
names = {m.get("name") or m.get("model") for m in json.load(sys.stdin).get("models", [])}
sys.exit(0 if wanted in names or wanted + ":latest" in names else 1)' "${tag}"; then
        echo "WARNING: Ollama does not have '${tag}' yet; the entry is saved but the course stays unservable until the model exists (scripts/install_finetuned_adapter.py)." >&2
      fi
    fi
    echo "Apply with: $0 restart"
  fi
}

cmd_preflight() {
  # ExecStartPre. Exit non-zero and the unit does not start; systemd shows the
  # message in `status`. Ollama being slow to come up is not a failure here —
  # the service reports `unavailable` until it arrives — but waiting avoids a
  # burst of 503s at boot.
  [[ -f "${ENV_FILE}" ]] || die "Missing ${ENV_FILE}; run: ${REPO_ROOT}/scripts/aiswe_finetuned.sh install"
  local_helpers validate-env "${ENV_FILE}" >/dev/null \
    || die "Malformed ${MODEL_MAP_ENV:-FINETUNED_OLLAMA_MODELS} in ${ENV_FILE}; fix it with: ${REPO_ROOT}/scripts/aiswe_finetuned.sh set-mapping ..."
  refuse_if_tunnel_owns_port
  local kind
  kind="$(port_owner_field kind)"
  case "${kind}" in
    free) ;;
    service|other)
      die "127.0.0.1:${PORT} is already owned by '$(port_owner_field process)' (pid $(port_owner_field pid)); refusing to start a second listener. Stop it, or if it is a stale ${UNIT}: systemctl --user stop ${UNIT}"
      ;;
  esac
  local waited=0
  until ollama_up; do
    if (( waited >= OLLAMA_WAIT_SECONDS )); then
      echo "WARNING: Ollama at ${OLLAMA_URL} not reachable after ${OLLAMA_WAIT_SECONDS}s; starting anyway (health will say 'unavailable' until it is)." >&2
      break
    fi
    sleep 3
    waited=$((waited + 3))
  done
  echo "preflight ok: mapping valid, port ${PORT} free, ollama $(ollama_up && echo up || echo down)"
}

cmd_start() {
  require_cmd systemctl
  [[ -f "${UNIT_DST}" ]] || die "Unit not installed; run: $0 install"
  if [[ "$(unit_state)" == "active" ]]; then
    echo "${UNIT} is already active."
  else
    refuse_if_tunnel_owns_port
    systemctl --user reset-failed "${UNIT}" >/dev/null 2>&1 || true
    systemctl --user start "${UNIT}" || die "systemctl --user start ${UNIT} failed; see: systemctl --user status ${UNIT}"
  fi
  local deadline=$((SECONDS + HEALTH_WAIT_SECONDS))
  while (( SECONDS < deadline )); do
    if health_ready; then
      echo "${UNIT}: active, ${SERVICE_URL}/health ready (status=ok, adapterLoaded=true)"
      cmd_status_courses
      return 0
    fi
    sleep 2
  done
  if [[ -n "$(health_json)" ]]; then
    echo "${UNIT}: active but not ready — $(health_json | py -c 'import json,sys; h=json.load(sys.stdin); print("status", h.get("status"), "detail", h.get("detail"), "servable", [c["courseId"] for c in h.get("courses", [])])')"
    echo "Usually: no mapping yet, the mapped Ollama model does not exist, or Ollama is down."
  else
    echo "${UNIT}: did not answer ${SERVICE_URL}/health within ${HEALTH_WAIT_SECONDS}s; see: systemctl --user status ${UNIT}"
  fi
  return 1
}

cmd_stop() {
  require_cmd systemctl
  systemctl --user stop "${UNIT}" || die "systemctl --user stop ${UNIT} failed"
  echo "${UNIT} stopped. Fine-Tuned and Fine-Tuned + RAG are unavailable until it starts again ($0 start); Base and RAG are unaffected."
}

cmd_restart() {
  require_cmd systemctl
  if [[ "$(unit_state)" == "active" ]]; then
    systemctl --user stop "${UNIT}" || die "stop failed"
  fi
  cmd_start
}

cmd_status_courses() {
  local body
  body="$(health_json)"
  [[ -n "${body}" ]] || return 0
  printf '%s' "${body}" | py -c '
import json, sys
h = json.load(sys.stdin)
for m in h.get("models", []):
    print("  {} {} -> {} [{}]".format(m["courseId"], m["version"], m["ollamaModel"], "available" if m["available"] else "MISSING in Ollama"))
if not h.get("models"):
    print("  (no mapping)")'
}

cmd_status() {
  echo "unit:      ${UNIT} $(unit_state) ($(unit_enabled))"
  echo "port ${PORT}: $(port_owner_json)"
  if tunnel_control_alive; then echo "tunnel:    ACTIVE (Tillicum fallback control socket)"; else echo "tunnel:    inactive"; fi
  echo "ollama:    $(ollama_up && echo up || echo DOWN) (${OLLAMA_URL})"
  if health_ready; then echo "health:    ready"; elif [[ -n "$(health_json)" ]]; then echo "health:    answering but NOT ready"; else echo "health:    no answer"; fi
  cmd_status_courses
}

cmd_check() {
  # Read-only. Prints one line per fact; exits 1 if the service could not
  # start from this state.
  local ok=1
  say() { echo "$1 $2"; [[ "$1" == "FAIL" ]] && ok=0; return 0; }

  if [[ -f "${UNIT_DST}" ]]; then
    if [[ "$(cat "${UNIT_DST}")" == "$(render_unit)" ]]; then say PASS "unit installed and current: ${UNIT_DST}"; else say WARN "unit installed but differs from the tracked template (run: $0 install)"; fi
  else
    say FAIL "unit not installed (run: $0 install)"
  fi
  [[ -x "${VENV_PY}" ]] && say PASS "backend venv python: ${VENV_PY}" || say FAIL "missing ${VENV_PY}"
  if [[ -f "${ENV_FILE}" ]]; then
    local mode
    mode="$(stat -c '%a' "${ENV_FILE}" 2>/dev/null || stat -f '%Lp' "${ENV_FILE}" 2>/dev/null || echo '?')"
    [[ "${mode}" == "600" ]] && say PASS "env file ${ENV_FILE} (mode 600)" || say WARN "env file ${ENV_FILE} has mode ${mode}, expected 600"
    if local_helpers validate-env "${ENV_FILE}" > /tmp/aiswe-finetuned-check.$$ 2>&1; then
      if [[ -s /tmp/aiswe-finetuned-check.$$ ]]; then say PASS "mapping valid:"; sed 's/^/       /' /tmp/aiswe-finetuned-check.$$; else say WARN "mapping is empty (no course is servable)"; fi
    else
      say FAIL "mapping malformed: $(cat /tmp/aiswe-finetuned-check.$$)"
    fi
    rm -f /tmp/aiswe-finetuned-check.$$
  else
    say FAIL "env file missing: ${ENV_FILE} (run: $0 install)"
  fi
  if command -v systemctl >/dev/null 2>&1; then
    say INFO "unit state: $(unit_state) ($(unit_enabled))"
    local linger
    linger="$(loginctl show-user "${USER}" -p Linger --value 2>/dev/null || echo unknown)"
    [[ "${linger}" == "yes" ]] && say PASS "lingering enabled for ${USER}" || say WARN "lingering is '${linger}' (loginctl enable-linger ${USER})"
  else
    say INFO "no systemctl on this host; unit state not checked"
  fi
  local owner kind
  owner="$(port_owner_json)"; kind="$(port_owner_field kind)"
  case "${kind}" in
    free) say INFO "port ${PORT} free" ;;
    tunnel) say FAIL "port ${PORT} owned by the Tillicum tunnel: ${owner}" ;;
    service) if [[ "$(unit_state)" == "active" ]]; then say PASS "port ${PORT} owned by ${UNIT}: ${owner}"; else say WARN "port ${PORT} owned by a python process that is not ${UNIT}: ${owner}"; fi ;;
    *) say FAIL "port ${PORT} owned by something else: ${owner}" ;;
  esac
  if tunnel_control_alive; then say FAIL "Tillicum tunnel control socket is active (stop it: ./scripts/stop_finetuned_tunnel.sh)"; else say PASS "no Tillicum tunnel active"; fi
  ollama_up && say PASS "ollama reachable at ${OLLAMA_URL}" || say FAIL "ollama not reachable at ${OLLAMA_URL}"
  if health_ready; then say PASS "${SERVICE_URL}/health ready"; cmd_status_courses; elif [[ -n "$(health_json)" ]]; then say WARN "${SERVICE_URL}/health answers but is not ready"; cmd_status_courses; else say INFO "${SERVICE_URL}/health no answer (unit not running)"; fi
  if [[ -f "${BACKEND_ENV}" ]] && grep -qE "^FINETUNED_SERVICE_URL=${SERVICE_URL}/?$" "${BACKEND_ENV}"; then
    say PASS "backend/.env FINETUNED_SERVICE_URL=${SERVICE_URL}"
  else
    say WARN "backend/.env FINETUNED_SERVICE_URL is not ${SERVICE_URL} (install sets it)"
  fi
  [[ "${ok}" -eq 1 ]]
}

cmd_logs() {
  require_cmd journalctl
  journalctl --user -u "${UNIT}" -n "${1:-50}" --no-pager
}

case "${1:-}" in
  install) shift; cmd_install "$@" ;;
  check) shift; cmd_check "$@" ;;
  set-mapping) shift; cmd_set_mapping "$@" ;;
  preflight) shift; cmd_preflight "$@" ;;
  start) shift; cmd_start "$@" ;;
  stop) shift; cmd_stop "$@" ;;
  restart) shift; cmd_restart "$@" ;;
  status) shift; cmd_status "$@" ;;
  logs) shift; cmd_logs "$@" ;;
  -h|--help|help|"") usage; [[ "${1:-}" == "" ]] && exit 2 || exit 0 ;;
  *) die "Unknown command: $1 (try --help)" ;;
esac
