#!/usr/bin/env bash
set -euo pipefail

curl -s -X POST http://127.0.0.1:8080/api/start \
  -H 'Content-Type: application/json' \
  -d '{"max_papers":1000}'
