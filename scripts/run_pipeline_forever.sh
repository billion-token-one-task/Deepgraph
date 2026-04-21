#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/billion-token/Deepgraph"
LOG_DIR="$ROOT/logs"
PROCESSOR_LOG="$LOG_DIR/processor.log"
API_BASE="http://127.0.0.1:8080"
MAX_PAPERS_PER_RUN="${DEEPGRAPH_FOREVER_BATCH_SIZE:-1000}"

mkdir -p "$LOG_DIR"

while true; do
  processing_json="$(curl -s --max-time 10 "$API_BASE/api/processing" || true)"
  if [[ -z "$processing_json" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] web unavailable; retrying in 15s" >> "$PROCESSOR_LOG"
    sleep 15
    continue
  fi

  if [[ "$processing_json" == *'"pipeline_running":true'* ]]; then
    sleep 30
    continue
  fi

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting pipeline batch max_papers=$MAX_PAPERS_PER_RUN" >> "$PROCESSOR_LOG"
  curl -s --max-time 30 -X POST "$API_BASE/api/start" \
    -H 'Content-Type: application/json' \
    -d "{\"max_papers\":$MAX_PAPERS_PER_RUN}" >> "$PROCESSOR_LOG" 2>&1 || true
  printf '\n' >> "$PROCESSOR_LOG"
  sleep 120
done
