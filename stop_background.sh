#!/usr/bin/env bash
set -euo pipefail

SCREEN_BIN="/usr/bin/screen"

while read -r session; do
  [[ -z "$session" ]] && continue
  "$SCREEN_BIN" -S "$session" -X quit || true
  echo "stopped: $session"
done < <("$SCREEN_BIN" -ls | awk '/deepgraph-(web|tunnel)/ {print $1}')

pkill -f 'python main.py' || true
pkill -f 'cloudflared tunnel --url http://127.0.0.1:8080' || true
