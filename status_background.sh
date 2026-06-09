#!/usr/bin/env bash
set -euo pipefail

ROOT="${DEEPGRAPH_ROOT:-/root/hf_models/hk_backup/Deepgraph}"
SCREEN_BIN="${SCREEN_BIN:-/usr/bin/screen}"
PUBLIC_URL="http://1.13.190.7:25281"
API_BASE="http://127.0.0.1:8081"
FRPC_LOG="$ROOT/logs/frpc_deepgraph.log"
WEB_LOG="$ROOT/logs/web_8081.log"

echo "screen sessions:"
"$SCREEN_BIN" -ls | sed -n "/deepgraph-/p" || true

echo "local_meta: $(curl -sS --max-time 3 -o /dev/null -w "%{http_code}" "$API_BASE/api/meta" || true)"
echo "public_home: $(curl -sS --max-time 5 -o /dev/null -w "%{http_code}" "$PUBLIC_URL/" || true)"
echo "public_meta: $(curl -sS --max-time 5 -o /dev/null -w "%{http_code}" "$PUBLIC_URL/api/meta" || true)"
echo "public_url: $PUBLIC_URL"
echo "pdf_url: $PUBLIC_URL/papers/3/pdf"
echo "frpc_log: $FRPC_LOG"
echo "web_log: $WEB_LOG"
