#!/usr/bin/env bash
set -euo pipefail

ROOT="${DEEPGRAPH_ROOT:-/root/hf_models/hk_backup/Deepgraph}"
PYTHON="${DEEPGRAPH_PYTHON:-/root/anaconda3/bin/python3}"
SCREEN_BIN="${SCREEN_BIN:-/usr/bin/screen}"
FRPC_BIN="${DEEPGRAPH_FRPC_BIN:-/usr/local/bin/frpc}"
FRPC_CONFIG="${DEEPGRAPH_FRPC_CONFIG:-/etc/frp/frpc-deepgraph.toml}"
START_PROCESSOR="${DEEPGRAPH_START_PROCESSOR:-0}"
LOG_DIR="$ROOT/logs"
WEB_RUNNER="$ROOT/scripts/run_web_forever.sh"
FRPC_RUNNER="$ROOT/scripts/run_frpc_deepgraph_forever.sh"
KEEPER_RUNNER="$ROOT/scripts/keep_deepgraph_alive.sh"
PROCESSOR_RUNNER="$ROOT/scripts/run_pipeline_forever.sh"
PUBLIC_URL="http://1.13.190.7:25281"

mkdir -p "$LOG_DIR"

if [[ ! -d "$ROOT" ]]; then
  echo "DeepGraph root not found: $ROOT" >&2
  exit 1
fi

if [[ ! -x "$SCREEN_BIN" ]]; then
  echo "screen is not installed: $SCREEN_BIN" >&2
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "python not found or not executable: $PYTHON" >&2
  exit 1
fi

if [[ ! -x "$FRPC_BIN" ]]; then
  echo "frpc not found or not executable: $FRPC_BIN" >&2
  exit 1
fi

if [[ ! -r "$FRPC_CONFIG" ]]; then
  echo "frpc config not readable: $FRPC_CONFIG" >&2
  exit 1
fi

chmod +x "$WEB_RUNNER" "$FRPC_RUNNER" "$KEEPER_RUNNER" "$PROCESSOR_RUNNER"

"$SCREEN_BIN" -S deepgraph-web -Q select . >/dev/null 2>&1 || \
  "$SCREEN_BIN" -dmS deepgraph-web bash -lc "exec \"$WEB_RUNNER\""

"$SCREEN_BIN" -S deepgraph-frpc -Q select . >/dev/null 2>&1 || \
  "$SCREEN_BIN" -dmS deepgraph-frpc bash -lc "exec \"$FRPC_RUNNER\""

"$SCREEN_BIN" -S deepgraph-keeper -Q select . >/dev/null 2>&1 || \
  "$SCREEN_BIN" -dmS deepgraph-keeper bash -lc "exec \"$KEEPER_RUNNER\""

if [[ "$START_PROCESSOR" == "1" ]]; then
  "$SCREEN_BIN" -S deepgraph-processor -Q select . >/dev/null 2>&1 || \
    "$SCREEN_BIN" -dmS deepgraph-processor bash -lc "exec \"$PROCESSOR_RUNNER\""
fi

echo "screen sessions:"
"$SCREEN_BIN" -ls | sed -n "/deepgraph-/p" || true
echo "public_url: $PUBLIC_URL"
echo "pdf_url: $PUBLIC_URL/papers/3/pdf"
echo "frpc_log: $LOG_DIR/frpc_deepgraph.log"
echo "web_log: $LOG_DIR/web_8081.log"
