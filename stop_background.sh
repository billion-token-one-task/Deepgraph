#!/usr/bin/env bash
set -euo pipefail

SCREEN_BIN="${SCREEN_BIN:-/usr/bin/screen}"

while read -r session; do
  [[ -z "$session" ]] && continue
  "$SCREEN_BIN" -S "$session" -X quit || true
  echo "stopped: $session"
done < <("$SCREEN_BIN" -ls | awk "/deepgraph-(web|frpc|keeper|processor|tunnel)/ {print \$1}")

pkill -f "/root/anaconda3/bin/python3 -u main.py" || true
pkill -f "/usr/local/bin/frpc -c /etc/frp/frpc-deepgraph.toml" || true
pkill -f "cloudflared tunnel --url http://127.0.0.1:8081" || true
