#!/usr/bin/env bash
# Copy workspace PDF cache to another machine without needing extra disk locally.
#
# Usage:
#   ./scripts/rsync_workspace_pdfs_to_remote.sh user@newhost:/path/to/Deepgraph/workspace/pdfs/
#
# Remote directory should exist or rsync create missing parents depending on rsync version;
# safest: ssh user@host 'mkdir -p /path/to/Deepgraph/workspace/pdfs'
#
# Same idea over tar pipe (writes ONE tar file only on remote):
#   cd "$(dirname "$0")/../workspace" && tar cf - pdfs | ssh user@host 'cat > ~/deepgraph-pdfs.tar'

set -euo pipefail

REMOTE="${1:?用法: $0 user@host:/绝对路径/workspace/pdfs/}"
SRC="$(cd "$(dirname "$0")/../workspace/pdfs" && pwd)"

exec rsync -avh --partial --progress "${SRC}/" "${REMOTE}"
