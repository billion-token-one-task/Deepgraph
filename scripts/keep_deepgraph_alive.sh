#!/usr/bin/env bash
set -euo pipefail

ROOT="${DEEPGRAPH_ROOT:-/root/hf_models/hk_backup/Deepgraph}"
LOG_DIR="$ROOT/logs"
KEEPER_LOG="$LOG_DIR/keeper.log"
STARTER="$ROOT/start_background.sh"
SCREEN_BIN="${SCREEN_BIN:-/usr/bin/screen}"
START_PROCESSOR="${DEEPGRAPH_START_PROCESSOR:-0}"
WEB_HEALTH_URL="${DEEPGRAPH_WEB_HEALTH_URL:-http://127.0.0.1:8081/api/meta}"
WEB_HEALTH_TIMEOUT="${DEEPGRAPH_WEB_HEALTH_TIMEOUT:-5}"
WEB_STARTUP_GRACE_SECONDS="${DEEPGRAPH_WEB_STARTUP_GRACE_SECONDS:-90}"
web_unhealthy_since=0

mkdir -p "$LOG_DIR"

while true; do
  web_ok=0
  web_health_ok=0
  frpc_ok=0
  processor_ok=1
  "$SCREEN_BIN" -ls | grep -qE '[[:space:]][0-9]+\.deepgraph-web[[:space:]]' && web_ok=1 || true
  if [[ "$web_ok" -eq 1 ]]; then
    curl -fsS --max-time "$WEB_HEALTH_TIMEOUT" "$WEB_HEALTH_URL" >/dev/null 2>&1 && web_health_ok=1 || true
  fi
  "$SCREEN_BIN" -ls | grep -qE '[[:space:]][0-9]+\.deepgraph-frpc[[:space:]]' && frpc_ok=1 || true
  if [[ "$START_PROCESSOR" == "1" ]]; then
    processor_ok=0
    "$SCREEN_BIN" -ls | grep -qE '[[:space:]][0-9]+\.deepgraph-processor[[:space:]]' && processor_ok=1 || true
  fi

  if [[ "$web_health_ok" -eq 1 ]]; then
    web_unhealthy_since=0
  fi

  if [[ "$web_ok" -eq 1 && "$web_health_ok" -ne 1 ]]; then
    now_ts=$(date +%s)
    if [[ "$web_unhealthy_since" -eq 0 ]]; then
      web_unhealthy_since="$now_ts"
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] keeper waiting for web startup grace url=$WEB_HEALTH_URL" >> "$KEEPER_LOG"
    elif (( now_ts - web_unhealthy_since >= WEB_STARTUP_GRACE_SECONDS )); then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] keeper stopping unhealthy web screen url=$WEB_HEALTH_URL after ${WEB_STARTUP_GRACE_SECONDS}s grace" >> "$KEEPER_LOG"
      "$SCREEN_BIN" -S deepgraph-web -X quit >/dev/null 2>&1 || true
      web_ok=0
      web_unhealthy_since=0
    fi
  fi

  if [[ "$web_ok" -ne 1 || "$frpc_ok" -ne 1 || "$processor_ok" -ne 1 ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] keeper restart triggered web=$web_ok web_health=$web_health_ok frpc=$frpc_ok processor=$processor_ok" >> "$KEEPER_LOG"
    "$STARTER" >> "$KEEPER_LOG" 2>&1 || true
  fi

  sleep 10
done
