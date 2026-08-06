#!/usr/bin/env bash
# Install the V1 observation + advance units. Run as root AFTER the V1 code
# deploy (safe-update procedure) has landed in /home/billion-token/Deepgraph.
set -euo pipefail

UNIT_SRC="/home/billion-token/Deepgraph/ops/v1"
REPORTS_DIR="/home/ec2-user/deepgraph-reports"

# billion-token (the service user) must be able to write traces/reports onto
# the persistent ec2-user volume.
setfacl -m u:billion-token:rwx "$REPORTS_DIR"

install -m 0644 "$UNIT_SRC/deepgraph-observe@.service" /etc/systemd/system/
install -m 0644 "$UNIT_SRC/deepgraph-render-report.service" /etc/systemd/system/
install -m 0644 "$UNIT_SRC/deepgraph-render-report.timer" /etc/systemd/system/
install -m 0644 "$UNIT_SRC/deepgraph-auto-advance.service" /etc/systemd/system/
install -m 0644 "$UNIT_SRC/deepgraph-auto-advance.timer" /etc/systemd/system/

systemctl daemon-reload
systemctl enable --now deepgraph-observe@10.service deepgraph-observe@11.service
systemctl enable --now deepgraph-render-report.timer
systemctl enable --now deepgraph-auto-advance.timer

systemctl --no-pager --plain list-timers | grep -E 'deepgraph-(render|auto)' || true
echo "V1 units installed."
