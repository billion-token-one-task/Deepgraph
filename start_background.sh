#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/billion-token/Deepgraph"
PYTHON="$ROOT/.venv/bin/python"
SCREEN_BIN="/usr/bin/screen"
CF_BIN="/home/billion-token/bin/cloudflared"
LOG_DIR="$ROOT/logs"
WEB_LOG="$LOG_DIR/web.log"
CF_LOG="$LOG_DIR/cloudflared.log"
URL_FILE="$LOG_DIR/tunnel-url.txt"
WEB_RUNNER="$ROOT/scripts/run_web_forever.sh"
TUNNEL_RUNNER="$ROOT/scripts/run_tunnel_forever.sh"
PROCESSOR_RUNNER="$ROOT/scripts/run_pipeline_forever.sh"

mkdir -p "$LOG_DIR"

if ! command -v "$SCREEN_BIN" >/dev/null 2>&1; then
  echo "screen is not installed" >&2
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "python venv not found: $PYTHON" >&2
  exit 1
fi

if [[ ! -x "$CF_BIN" ]]; then
  echo "cloudflared not found: $CF_BIN" >&2
  exit 1
fi

chmod +x "$WEB_RUNNER" "$TUNNEL_RUNNER" "$PROCESSOR_RUNNER"

"$SCREEN_BIN" -S deepgraph-web -Q select . >/dev/null 2>&1 || \
  "$SCREEN_BIN" -dmS deepgraph-web bash -lc "exec '$WEB_RUNNER'"

"$SCREEN_BIN" -S deepgraph-tunnel -Q select . >/dev/null 2>&1 || \
  "$SCREEN_BIN" -dmS deepgraph-tunnel bash -lc "exec '$TUNNEL_RUNNER'"

"$SCREEN_BIN" -S deepgraph-processor -Q select . >/dev/null 2>&1 || \
  "$SCREEN_BIN" -dmS deepgraph-processor bash -lc "exec '$PROCESSOR_RUNNER'"

rm -f "$URL_FILE"
for _ in $(seq 1 20); do
  if grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "$CF_LOG" | tail -n 1 > "$URL_FILE" 2>/dev/null; then
    if [[ -s "$URL_FILE" ]]; then
      break
    fi
  fi
  sleep 1
done

echo "screen sessions:"
"$SCREEN_BIN" -ls | sed -n '/deepgraph-/p'
if [[ -s "$URL_FILE" ]]; then
  echo "public_url: $(cat "$URL_FILE")"
else
  echo "public_url: pending; check $CF_LOG"
fi
