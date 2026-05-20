#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/billion-token/Deepgraph"
PYTHON="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/logs"
WEB_LOG="$LOG_DIR/web.log"

mkdir -p "$LOG_DIR"

while true; do
  cd "$ROOT"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting deepgraph web" >> "$WEB_LOG"
  "$PYTHON" -u main.py >> "$WEB_LOG" 2>&1 || true
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] deepgraph web exited; restarting in 5s" >> "$WEB_LOG"
  sleep 5
done
