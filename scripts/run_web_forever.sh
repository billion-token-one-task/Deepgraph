#!/usr/bin/env bash
set -euo pipefail

ROOT="${DEEPGRAPH_ROOT:-/root/hf_models/hk_backup/Deepgraph}"
PYTHON="${DEEPGRAPH_PYTHON:-/root/anaconda3/bin/python3}"
LOG_DIR="$ROOT/logs"
WEB_LOG="$LOG_DIR/web_8081.log"

mkdir -p "$LOG_DIR"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if [[ -f "$ROOT/scripts/deepgraph_proxy_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/deepgraph_proxy_env.sh"
fi

export DEEPGRAPH_WEB_PORT="${DEEPGRAPH_WEB_PORT:-8081}"

while true; do
  cd "$ROOT"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting deepgraph web on port $DEEPGRAPH_WEB_PORT" >> "$WEB_LOG"
  "$PYTHON" -u main.py >> "$WEB_LOG" 2>&1 || true
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] deepgraph web exited; restarting in 5s" >> "$WEB_LOG"
  sleep 5
done
