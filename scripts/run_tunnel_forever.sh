#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/billion-token/Deepgraph"
CF_BIN="/home/billion-token/bin/cloudflared"
LOG_DIR="$ROOT/logs"
CF_LOG="$LOG_DIR/cloudflared.log"
URL_FILE="$LOG_DIR/tunnel-url.txt"

mkdir -p "$LOG_DIR"

while true; do
  rm -f "$URL_FILE"
  cd "$ROOT"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting cloudflared tunnel" >> "$CF_LOG"
  "$CF_BIN" tunnel --url http://127.0.0.1:8080 >> "$CF_LOG" 2>&1 || true
  latest_url="$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "$CF_LOG" | tail -n 1 || true)"
  if [[ -n "$latest_url" ]]; then
    printf '%s\n' "$latest_url" > "$URL_FILE"
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] cloudflared exited; restarting in 5s" >> "$CF_LOG"
  sleep 5
done
