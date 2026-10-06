#!/usr/bin/env bash
# Automatic GPU -> VM failback for classroom GPU mode (docs/classroom-gpu-mode.md).
#
#   ./scripts/gpu_failback_watchdog.sh install [--check]   # render + install the user units, enable and start the timer
#   ./scripts/gpu_failback_watchdog.sh uninstall           # stop, disable and remove the units (state and log are kept)
#   ./scripts/gpu_failback_watchdog.sh status              # the timer, the last check, the last failbacks
#   ./scripts/gpu_failback_watchdog.sh run [--dry-run]     # one check now; this is what the timer runs
#
# `run` is scripts/gpu_failback_watchdog.py under the backend's virtualenv. It
# needs no password and no Duo, and never starts or cancels a Tillicum job.
# `run --dry-run` checks and says what it would do, and writes nothing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PY="${REPO_ROOT}/backend/.venv/bin/python"
WATCHDOG="${REPO_ROOT}/scripts/gpu_failback_watchdog.py"

SERVICE="aiswe-gpu-failback.service"
TIMER="aiswe-gpu-failback.timer"
UNIT_SRC_DIR="${REPO_ROOT}/scripts/systemd"
UNIT_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
STATE_DIR="${AISWE_GPU_FAILBACK_STATE_DIR:-${XDG_STATE_HOME:-${HOME}/.local/state}/css360-syllabus-bot}"

usage() {
  sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

render_unit() {
  # A literal substitution; the checkout path is the only rendered value.
  sed "s|@REPO_ROOT@|${REPO_ROOT}|g" "${UNIT_SRC_DIR}/$1"
}

install_unit() {
  # Prints "written" or "unchanged"; with $2 = 1, only says what it would do.
  local unit="$1" check_only="$2" rendered
  rendered="$(render_unit "${unit}")"
  if [[ -f "${UNIT_DIR}/${unit}" ]] && [[ "$(cat "${UNIT_DIR}/${unit}")" == "${rendered}" ]]; then
    echo "  ${unit}: unchanged"
  elif [[ "${check_only}" -eq 1 ]]; then
    echo "  ${unit}: would write (rendered from scripts/systemd/${unit})"
  else
    mkdir -p "${UNIT_DIR}"
    printf '%s\n' "${rendered}" > "${UNIT_DIR}/${unit}"
    echo "  ${unit}: written"
  fi
}

cmd_install() {
  local check_only=0
  [[ "${1:-}" == "--check" ]] && check_only=1
  [[ "${check_only}" -eq 1 ]] || require_cmd systemctl
  [[ -f "${UNIT_SRC_DIR}/${SERVICE}" && -f "${UNIT_SRC_DIR}/${TIMER}" ]] || die "Missing unit templates in ${UNIT_SRC_DIR}"
  [[ -x "${VENV_PY}" ]] || die "Missing backend virtualenv: ${VENV_PY} (the watchdog runs from it)."

  echo "Units: ${UNIT_DIR}"
  install_unit "${SERVICE}" "${check_only}"
  install_unit "${TIMER}" "${check_only}"
  if [[ "${check_only}" -eq 1 ]]; then
    echo "Check only; nothing written, nothing enabled."
    return 0
  fi

  systemctl --user daemon-reload
  systemctl --user enable "${TIMER}" >/dev/null 2>&1 || die "Could not enable ${TIMER}."
  # restart, not start: a timer that was already running picks up a changed schedule.
  systemctl --user restart "${TIMER}" || die "Could not start ${TIMER}; see: systemctl --user status ${TIMER}"
  echo "Enabled and started: ${TIMER} (a check every 15 seconds; starts at boot when lingering is on)"
  local linger
  linger="$(loginctl show-user "${USER}" -p Linger --value 2>/dev/null || echo unknown)"
  if [[ "${linger}" != "yes" ]]; then
    echo "WARNING: lingering is '${linger}' for ${USER}; user units stop at logout and do not start at boot." >&2
    echo "         Enable it once with: loginctl enable-linger ${USER}" >&2
  fi
  echo "See it:  ${REPO_ROOT}/scripts/classroom_gpu_mode.sh status"
}

cmd_uninstall() {
  require_cmd systemctl
  systemctl --user disable --now "${TIMER}" >/dev/null 2>&1 || true
  systemctl --user stop "${SERVICE}" >/dev/null 2>&1 || true
  rm -f "${UNIT_DIR}/${TIMER}" "${UNIT_DIR}/${SERVICE}"
  systemctl --user daemon-reload
  systemctl --user reset-failed "${SERVICE}" >/dev/null 2>&1 || true
  echo "Removed ${TIMER} and ${SERVICE}. Classroom GPU mode no longer fails back on its own."
  echo "Kept: ${STATE_DIR}/gpu-failback-state.json and gpu-failback.log"
}

cmd_status() {
  if command -v systemctl >/dev/null 2>&1; then
    echo "Timer:   $(systemctl --user is-enabled "${TIMER}" 2>/dev/null || true) / $(systemctl --user is-active "${TIMER}" 2>/dev/null || true)"
    systemctl --user list-timers "${TIMER}" --no-pager 2>/dev/null | sed 's/^/  /' || true
  else
    echo "Timer:   systemctl is not available on this host"
  fi
  echo "State:   ${STATE_DIR}/gpu-failback-state.json"
  if [[ -f "${STATE_DIR}/gpu-failback-state.json" ]]; then
    sed 's/^/  /' "${STATE_DIR}/gpu-failback-state.json"
  else
    echo "  (no check has run yet)"
  fi
  echo "Log:     ${STATE_DIR}/gpu-failback.log (automatic failbacks, newest last)"
  if [[ -f "${STATE_DIR}/gpu-failback.log" ]]; then
    tail -n 6 "${STATE_DIR}/gpu-failback.log" | sed 's/^/  /'
  else
    echo "  (no automatic failback yet)"
  fi
}

cmd_run() {
  [[ -x "${VENV_PY}" ]] || die "${VENV_PY} not found (the backend virtualenv)."
  exec "${VENV_PY}" "${WATCHDOG}" "$@"
}

command="${1:-}"
[[ $# -gt 0 ]] && shift
case "${command}" in
  install) cmd_install "$@" ;;
  uninstall) cmd_uninstall ;;
  status) cmd_status ;;
  run) cmd_run "$@" ;;
  -h|--help|help|"") usage ;;
  *) usage >&2; die "Unknown command: ${command}" ;;
esac
