#!/usr/bin/env bash
# The CSS 360 controlled benchmark on the UWB VM, in four commands:
#
#   ./evaluation/css360_controlled_benchmark/vm.sh setup     # service unit up, route on, preflight
#   ./evaluation/css360_controlled_benchmark/vm.sh start     # pilot 3 questions, verify, then run everything (background)
#   ./evaluation/css360_controlled_benchmark/vm.sh status    # progress, last log lines, unit state
#   ./evaluation/css360_controlled_benchmark/vm.sh cleanup   # route off, backend restarted, service unit stopped
#
# Also: `stop-run` stops the background runner (finished requests are kept;
# `start` resumes), `logs` follows the runner's log. The bearer token stays in
# backend/.env; nothing here prints it. Nothing here touches PostgreSQL, the
# model registry, or the production fine-tuned service.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO/backend/.venv/bin/python"
RUNNER="$REPO/evaluation/css360_controlled_benchmark/run_benchmark.py"
ENV_SCRIPT="$REPO/backend/scripts/css360_benchmark_env.py"
UNIT_SRC="$REPO/training/inference_service/aiswe-benchmark.service"
UNIT=aiswe-benchmark
RUN_UNIT=css360-benchmark
RESULTS="$REPO/evaluation/css360_controlled_benchmark/results"
NOHUP_LOG="$RESULTS/runner-nohup.log"

health_backend() { curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8001/api/health; }
route_code() {
  curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:8001/api/research/css360/benchmark/pair \
    -H 'Content-Type: application/json' -d '{"question":"x"}'
}
wait_for() { # url, expected-code, seconds
  local url="$1" want="$2" tries="${3:-30}" code=""
  for _ in $(seq 1 "$tries"); do
    code="$(curl -s -o /dev/null -w '%{http_code}' "$url" || true)"
    [ "$code" = "$want" ] && return 0
    sleep 1
  done
  echo "expected $want from $url, got '$code'" >&2
  return 1
}

cmd_setup() {
  echo "== 1/4 benchmark service unit"
  mkdir -p ~/.config/systemd/user
  cp "$UNIT_SRC" ~/.config/systemd/user/"$UNIT".service
  systemctl --user daemon-reload
  systemctl --user enable --now "$UNIT"
  wait_for http://127.0.0.1:9002/health 200 30
  curl -s http://127.0.0.1:9002/health | "$PY" -c 'import json,sys; h=json.load(sys.stdin); print("service:", h["status"], "servable:", h["servable"]); [print("  ", r["alias"], r["ollamaModel"], "available" if r["available"] else "MISSING", (r.get("digest") or "")[:12]) for r in h["aliases"]]'
  echo "== 2/4 route on (token stays in backend/.env)"
  (cd "$REPO/backend" && "$PY" "$ENV_SCRIPT" --enable)
  systemctl --user restart aiswe-backend
  wait_for http://127.0.0.1:8001/api/health 200 30
  echo "== 3/4 route answers $(route_code) without a token (401 expected)"
  echo "== 4/4 preflight"
  (cd "$REPO" && "$PY" "$RUNNER" preflight)
}

cmd_start() {
  mkdir -p "$RESULTS"
  if systemctl --user is-active --quiet "$RUN_UNIT" 2>/dev/null; then
    echo "runner unit $RUN_UNIT is already active; use status or stop-run" >&2
    exit 1
  fi
  if command -v systemd-run >/dev/null 2>&1 && systemd-run --user --unit "$RUN_UNIT" --collect \
      -p WorkingDirectory="$REPO" "$PY" "$RUNNER" run "$@" >/dev/null 2>&1; then
    echo "started as user unit $RUN_UNIT (survives logout); follow with: $0 logs"
  else
    nohup "$PY" "$RUNNER" run "$@" >"$NOHUP_LOG" 2>&1 &
    echo "started with nohup (pid $!), output in $NOHUP_LOG; follow with: $0 logs"
  fi
}

cmd_status() {
  (cd "$REPO" && "$PY" "$RUNNER" status) || true
  echo "--- units ---"
  systemctl --user is-active "$UNIT" 2>/dev/null | sed "s/^/$UNIT: /" || true
  systemctl --user is-active "$RUN_UNIT" 2>/dev/null | sed "s/^/$RUN_UNIT: /" || true
  pgrep -af "run_benchmark.py run" | sed 's/^/process: /' || true
}

cmd_logs() {
  local run; run="$(cat "$RESULTS/CURRENT" 2>/dev/null || true)"
  if [ -n "$run" ] && [ -f "$RESULTS/$run/run.log" ]; then tail -n 20 -f "$RESULTS/$run/run.log"; else tail -n 20 -f "$NOHUP_LOG"; fi
}

cmd_stop_run() {
  systemctl --user stop "$RUN_UNIT" 2>/dev/null || true
  pkill -f "run_benchmark.py run" 2>/dev/null || true
  echo "runner stopped; finished requests are kept and 'start' resumes"
}

cmd_cleanup() {
  cmd_stop_run
  (cd "$REPO/backend" && "$PY" "$ENV_SCRIPT" --disable)
  systemctl --user restart aiswe-backend
  wait_for http://127.0.0.1:8001/api/health 200 30
  echo "route answers $(route_code) (404 expected)"
  systemctl --user disable --now "$UNIT" 2>/dev/null || true
  echo "benchmark service unit $UNIT stopped and disabled"
}

case "${1:-}" in
  setup) shift; cmd_setup "$@" ;;
  start) shift; cmd_start "$@" ;;
  status) shift; cmd_status "$@" ;;
  logs) shift; cmd_logs "$@" ;;
  stop-run) shift; cmd_stop_run "$@" ;;
  cleanup) shift; cmd_cleanup "$@" ;;
  *) sed -n '2,14p' "$0"; exit 2 ;;
esac
