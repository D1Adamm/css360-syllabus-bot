#!/usr/bin/env bash
# Switch classroom generation between the VM (normal) and a Tillicum GPU.
#
#   ./scripts/classroom_gpu_mode.sh start  --course <courseId> --admin-email <you>
#   ./scripts/classroom_gpu_mode.sh status [--verify --admin-email <you>]
#   ./scripts/classroom_gpu_mode.sh stop   --course <courseId> --admin-email <you> [--cancel-job]
#
# The logic lives in classroom_gpu_mode.py, run with the backend's virtualenv
# so it uses the backend's own rules for what a mode file means and what
# production decoding is. See docs/classroom-gpu-mode.md.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO_ROOT}/backend/.venv/bin/python"
[[ -x "${PY}" ]] || { echo "ERROR: ${PY} not found (the backend virtualenv)." >&2; exit 2; }
exec "${PY}" "${REPO_ROOT}/scripts/classroom_gpu_mode.py" "$@"
