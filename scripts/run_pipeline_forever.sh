#!/usr/bin/env bash
set -euo pipefail

ROOT="${DEEPGRAPH_ROOT:-/root/hf_models/hk_backup/Deepgraph}"
PYTHON="${DEEPGRAPH_PYTHON:-/root/anaconda3/bin/python3}"
LOG_DIR="$ROOT/logs"
PROCESSOR_LOG="$LOG_DIR/processor.log"
API_BASE="${DEEPGRAPH_API_BASE:-http://127.0.0.1:8081}"
MAX_PAPERS_PER_RUN="${DEEPGRAPH_FOREVER_BATCH_SIZE:-1000}"
LOCK_FILE="/tmp/deepgraph-run_continuous.lock"

mkdir -p "$LOG_DIR"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if [[ -f "$ROOT/scripts/deepgraph_proxy_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/deepgraph_proxy_env.sh"
fi

while true; do
  processing_json="$(curl -s --max-time 10 "$API_BASE/api/processing" || true)"
  if [[ -z "$processing_json" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] web unavailable at $API_BASE; retrying in 15s" >> "$PROCESSOR_LOG"
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
  printf "\n" >> "$PROCESSOR_LOG"
  sleep 120
done
