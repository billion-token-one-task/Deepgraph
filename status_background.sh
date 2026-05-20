#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/billion-token/Deepgraph"
SCREEN_BIN="/usr/bin/screen"
URL_FILE="$ROOT/logs/tunnel-url.txt"
CF_LOG="$ROOT/logs/cloudflared.log"

echo "screen sessions:"
"$SCREEN_BIN" -ls | sed -n '/deepgraph-/p' || true

if [[ -s "$URL_FILE" ]]; then
  echo "public_url: $(cat "$URL_FILE")"
else
  url="$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "$CF_LOG" | tail -n 1 || true)"
  if [[ -n "$url" ]]; then
    echo "public_url: $url"
  else
    echo "public_url: unavailable"
  fi
fi
