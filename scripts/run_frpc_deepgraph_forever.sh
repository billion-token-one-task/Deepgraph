#!/usr/bin/env bash
set -euo pipefail

ROOT="${DEEPGRAPH_ROOT:-/root/hf_models/hk_backup/Deepgraph}"
FRPC_BIN="${DEEPGRAPH_FRPC_BIN:-/usr/local/bin/frpc}"
FRPC_CONFIG="${DEEPGRAPH_FRPC_CONFIG:-/etc/frp/frpc-deepgraph.toml}"
LOG_DIR="$ROOT/logs"
FRPC_LOG="$LOG_DIR/frpc_deepgraph.log"
PUBLIC_URL="http://1.13.190.7:25281"

mkdir -p "$LOG_DIR"

if [[ ! -x "$FRPC_BIN" ]]; then
  echo "frpc not found or not executable: $FRPC_BIN" >&2
  exit 1
fi

if [[ ! -r "$FRPC_CONFIG" ]]; then
  echo "frpc config not readable: $FRPC_CONFIG" >&2
  exit 1
fi

while true; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting frpc for DeepGraph tunnel ($PUBLIC_URL)" >> "$FRPC_LOG"
  "$FRPC_BIN" -c "$FRPC_CONFIG" >> "$FRPC_LOG" 2>&1 || true
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] frpc exited; restarting in 5s" >> "$FRPC_LOG"
  sleep 5
done
