#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/billion-token/Deepgraph"
LOG_DIR="$ROOT/logs"
KEEPER_LOG="$LOG_DIR/keeper.log"
STARTER="$ROOT/start_background.sh"

mkdir -p "$LOG_DIR"

while true; do
  web_count="$(pgrep -fc '/home/billion-token/Deepgraph/.venv/bin/python -u main.py' || true)"
  tunnel_count="$(pgrep -fc '/home/billion-token/bin/cloudflared tunnel --url http://127.0.0.1:8080' || true)"
  processor_count="$(pgrep -fc '/home/billion-token/Deepgraph/scripts/run_pipeline_forever.sh' || true)"

  if [[ "${web_count:-0}" -lt 1 || "${tunnel_count:-0}" -lt 1 || "${processor_count:-0}" -lt 1 ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] keeper restart triggered web=$web_count tunnel=$tunnel_count processor=$processor_count" >> "$KEEPER_LOG"
    "$STARTER" >> "$KEEPER_LOG" 2>&1 || true
  fi

  sleep 10
done
