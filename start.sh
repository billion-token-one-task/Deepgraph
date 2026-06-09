#!/bin/bash
cd "/root/hk/Deepgraph"
set -a; source .env; set +a
exec python3 main.py
