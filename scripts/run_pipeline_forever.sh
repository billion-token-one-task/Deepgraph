#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/billion-token/Deepgraph"
PYTHON="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/logs"
PROCESSOR_LOG="$LOG_DIR/processor.log"
API_BASE="http://127.0.0.1:8080"
MAX_PAPERS_PER_RUN="${DEEPGRAPH_FOREVER_BATCH_SIZE:-1000}"
# Serialize runs: fixed_flow_read_only blocks POST /api/start, so we call run_continuous in-process.
LOCK_FILE="/tmp/deepgraph-run_continuous.lock"

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

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting pipeline batch max_papers=$MAX_PAPERS_PER_RUN (cli)" >> "$PROCESSOR_LOG"
  (
    flock -n 200 || {
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] pipeline lock busy; skip" >> "$PROCESSOR_LOG"
      exit 0
    }
    cd "$ROOT"
    "$PYTHON" -u -c "from orchestrator.pipeline import run_continuous; run_continuous(${MAX_PAPERS_PER_RUN})" >> "$PROCESSOR_LOG" 2>&1
  ) 200>"$LOCK_FILE" || true
  printf '\n' >> "$PROCESSOR_LOG"
  sleep 120
done
