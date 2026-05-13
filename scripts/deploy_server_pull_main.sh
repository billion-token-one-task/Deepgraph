#!/usr/bin/env bash
# Update a remote DeepGraph checkout by pulling a git branch on the server.
#
# Typical usage:
#   scripts/deploy_server_pull_main.sh \
#     --host your.server.com \
#     --user root \
#     --repo-dir /root/Deepgraph
#
# With restart:
#   scripts/deploy_server_pull_main.sh \
#     --host your.server.com \
#     --user root \
#     --repo-dir /root/Deepgraph \
#     --restart-cmd "systemctl restart deepgraph"
#
# You can also source values from env:
#   DEPLOY_HOST, DEPLOY_USER, DEPLOY_PORT, DEPLOY_REPO_DIR,
#   DEPLOY_BRANCH, DEPLOY_REMOTE, DEPLOY_RESTART_CMD, DEPLOY_PASSWORD

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

HOST="${DEPLOY_HOST:-}"
USER_NAME="${DEPLOY_USER:-}"
PORT="${DEPLOY_PORT:-22}"
REPO_DIR="${DEPLOY_REPO_DIR:-}"
BRANCH="${DEPLOY_BRANCH:-main}"
REMOTE_NAME="${DEPLOY_REMOTE:-origin}"
RESTART_CMD="${DEPLOY_RESTART_CMD:-}"
PASSWORD="${DEPLOY_PASSWORD:-}"
SKIP_LOCAL_CHECKS=0

usage() {
  cat <<'EOF'
Usage:
  scripts/deploy_server_pull_main.sh --host HOST --user USER --repo-dir REMOTE_DIR [options]

Required:
  --host HOST              Remote SSH host
  --user USER              Remote SSH user
  --repo-dir REMOTE_DIR    Remote repository directory

Optional:
  --port PORT              SSH port (default: 22)
  --branch BRANCH          Git branch to deploy (default: main)
  --remote NAME            Git remote name (default: origin)
  --restart-cmd CMD        Command to run remotely after pull
  --password PASSWORD      SSH password for sshpass-based login
  --skip-local-checks      Skip local git status/branch checks
  --help                   Show this help

Examples:
  scripts/deploy_server_pull_main.sh \
    --host gpu.example.com \
    --user root \
    --repo-dir /root/Deepgraph

  scripts/deploy_server_pull_main.sh \
    --host gpu.example.com \
    --user root \
    --repo-dir /root/Deepgraph \
    --restart-cmd "bash scripts/run_web_forever.sh"

  scripts/deploy_server_pull_main.sh \
    --host gpu.example.com \
    --user root \
    --port 55860 \
    --password 'your-password' \
    --repo-dir /root/Deepgraph
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="${2:?missing value for --host}"
      shift 2
      ;;
    --user)
      USER_NAME="${2:?missing value for --user}"
      shift 2
      ;;
    --port)
      PORT="${2:?missing value for --port}"
      shift 2
      ;;
    --repo-dir)
      REPO_DIR="${2:?missing value for --repo-dir}"
      shift 2
      ;;
    --branch)
      BRANCH="${2:?missing value for --branch}"
      shift 2
      ;;
    --remote)
      REMOTE_NAME="${2:?missing value for --remote}"
      shift 2
      ;;
    --restart-cmd)
      RESTART_CMD="${2:?missing value for --restart-cmd}"
      shift 2
      ;;
    --password)
      PASSWORD="${2:?missing value for --password}"
      shift 2
      ;;
    --skip-local-checks)
      SKIP_LOCAL_CHECKS=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$HOST" || -z "$USER_NAME" || -z "$REPO_DIR" ]]; then
  echo "Missing required arguments." >&2
  usage >&2
  exit 1
fi

if [[ "$SKIP_LOCAL_CHECKS" -ne 1 ]]; then
  echo "[deploy] local repo: $ROOT"
  CURRENT_BRANCH="$(git -C "$ROOT" branch --show-current)"
  if [[ -n "$CURRENT_BRANCH" && "$CURRENT_BRANCH" != "$BRANCH" ]]; then
    echo "[deploy] warning: local branch is '$CURRENT_BRANCH' but deploy branch is '$BRANCH'" >&2
  fi

  if [[ -n "$(git -C "$ROOT" status --porcelain)" ]]; then
    echo "[deploy] warning: local worktree has uncommitted changes." >&2
    echo "[deploy] commit/push first if you expect the server to receive your latest edits." >&2
  fi

  echo "[deploy] local remotes:"
  git -C "$ROOT" remote -v | sed 's/^/[deploy]   /'
fi

SSH_TARGET="${USER_NAME}@${HOST}"
SSH_OPTS=(-p "$PORT" -o BatchMode=yes -o StrictHostKeyChecking=accept-new)

read -r -d '' REMOTE_SCRIPT <<EOF || true
set -euo pipefail
cd "$REPO_DIR"
echo "[remote] pwd: \$(pwd)"
if [[ ! -d .git ]]; then
  echo "[remote] error: $REPO_DIR is not a git repository" >&2
  exit 1
fi

echo "[remote] current branch: \$(git branch --show-current || true)"
echo "[remote] fetching $REMOTE_NAME ..."
git fetch "$REMOTE_NAME"

echo "[remote] checking out $BRANCH ..."
git checkout "$BRANCH"

echo "[remote] pulling $REMOTE_NAME/$BRANCH ..."
git pull "$REMOTE_NAME" "$BRANCH"

echo "[remote] HEAD: \$(git rev-parse HEAD)"
echo "[remote] latest commit:"
git --no-pager log -1 --oneline

if [[ -n "$RESTART_CMD" ]]; then
  echo "[remote] restart: $RESTART_CMD"
  eval "$RESTART_CMD"
fi
EOF

echo "[deploy] ssh target: $SSH_TARGET:$PORT"
echo "[deploy] remote repo dir: $REPO_DIR"
echo "[deploy] branch: $BRANCH"
echo "[deploy] remote: $REMOTE_NAME"
if [[ -n "$RESTART_CMD" ]]; then
  echo "[deploy] restart command enabled"
fi

if [[ -n "$PASSWORD" ]]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "[deploy] error: DEPLOY_PASSWORD/--password was provided but sshpass is not installed." >&2
    exit 1
  fi
  sshpass -p "$PASSWORD" ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "bash -s" <<<"$REMOTE_SCRIPT"
else
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "bash -s" <<<"$REMOTE_SCRIPT"
fi

echo "[deploy] done"
