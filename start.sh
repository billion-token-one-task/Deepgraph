#!/bin/bash
cd /home/ec2-user/deepgraph
set -a; source .env; set +a
exec python3.12 main.py
