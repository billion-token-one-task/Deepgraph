#!/usr/bin/env bash
# Portable DeepGraph tarball: PostgreSQL dump + full source + optional local artifacts.
# Covers ingestion → discovery → EvoScientist → SciForge/GPU → PaperOrchestra manuscript:
#   - PaperOrchestra code & prompts ship inside Deepgraph/
#   - EvoScientist expects ~/EvoScientist/.venv/bin/EvoSci (install separately)
#   - PDF cache under workspace/pdfs is never copied (empty placeholder only)
#
# Requirements: DEEPGRAPH_DATABASE_URL points at a reachable Postgres; pg_dump or Docker.
#
# Usage:
#   export DEEPGRAPH_DATABASE_URL='postgresql://user:pass@host:port/dbname'
#   # Optional: skip ~/deepgraph_ideas and ~/research (~GB scale):
#   # DEEPGRAPH_BUNDLE_INCLUDE_ARTIFACTS=0 ./scripts/build_deploy_bundle_postgres.sh out.tar.gz
#   ./scripts/build_deploy_bundle_postgres.sh [output.tar.gz]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_TAR="${1:-$HOME/deepgraph-deploy-postgres-$(date +%Y%m%d).tar.gz}"
# Avoid /tmp tmpfs (often a few GB): artifacts + DB dump need disk backing store.
STAGING_PARENT="${DEEPGRAPH_BUNDLE_STAGING_PARENT:-$(dirname "$ROOT")}"
STAGING="$(mktemp -d "${STAGING_PARENT}/deepgraph-bundle-staging.XXXXXX")"
trap 'rm -rf "$STAGING"' EXIT

DUMP_DIR="$STAGING/deploy_bundle/postgres"
mkdir -p "$DUMP_DIR"

url="${DEEPGRAPH_DATABASE_URL:-}"
if [[ -z "$url" ]]; then
  echo "Set DEEPGRAPH_DATABASE_URL (postgresql://...)" >&2
  exit 1
fi

eval "$(DEEPGRAPH_DATABASE_URL="$url" python3 - << 'PY'
import os
import shlex
from urllib.parse import urlparse, unquote

raw = os.environ.get("DEEPGRAPH_DATABASE_URL", "").strip()
u = urlparse(raw)
host = u.hostname or "127.0.0.1"
port = u.port or 5432
user = unquote(u.username or "")
password = unquote(u.password or "")
dbname = (u.path or "").lstrip("/")
if not dbname:
    raise SystemExit("DATABASE_URL missing database name in path")
for k, v in (
    ("PGHOST", host),
    ("PGPORT", str(port)),
    ("PGUSER", user),
    ("PGPASSWORD", password),
    ("PGDATABASE", dbname),
):
    print(f"export {k}={shlex.quote(v)}")
PY
)"

echo "[bundle] staging source from $ROOT ..."
mkdir -p "$STAGING/Deepgraph"
rsync -a "$ROOT/" "$STAGING/Deepgraph/" \
  --exclude '.venv/' \
  --exclude '.git/' \
  --exclude 'workspace/pdfs/' \
  --exclude '__pycache__/' \
  --exclude '*.py[cod]' \
  --exclude '.env' \
  --exclude 'deepgraph.db' \
  --exclude 'deepgraph.db-*' \
  --exclude 'deepgraph.db.backup.*' \
  --exclude 'logs/' \
  --exclude '.cursor/' \
  --exclude 'deploy_bundle/'

mkdir -p "$STAGING/Deepgraph/workspace/pdfs"
touch "$STAGING/Deepgraph/workspace/pdfs/.gitkeep"
mkdir -p "$STAGING/Deepgraph/logs"

DEEPGRAPH_BUNDLE_INCLUDE_ARTIFACTS="${DEEPGRAPH_BUNDLE_INCLUDE_ARTIFACTS:-1}"
ARTIFACT_ROOT="$STAGING/deploy_bundle/artifacts"

_bundle_copy_tree() {
  local src="$1"
  local dest_name="$2"
  if [[ -d "$src" ]]; then
    echo "[bundle] artifacts: $src -> deploy_bundle/artifacts/$dest_name"
    mkdir -p "$ARTIFACT_ROOT/$dest_name"
    rsync -a "$src/" "$ARTIFACT_ROOT/$dest_name/"
  else
    echo "[bundle] skip missing artifact dir: $src"
  fi
}

if [[ "${DEEPGRAPH_BUNDLE_INCLUDE_ARTIFACTS}" == "1" ]]; then
  mkdir -p "$ARTIFACT_ROOT"
  _bundle_copy_tree "${HOME}/deepgraph_ideas" "deepgraph_ideas"
  _bundle_copy_tree "${HOME}/research" "research"
fi

DUMP_FILE="$DUMP_DIR/deepgraph_pg.dump"

run_pg_dump_host() {
  PGPASSWORD="$PGPASSWORD" pg_dump \
    -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" \
    -Fc --no-owner --no-acl \
    -f "$DUMP_FILE"
}

run_pg_dump_docker() {
  docker run --rm --network host \
    -v "$DUMP_DIR:/out" \
    -e PGPASSWORD="$PGPASSWORD" \
    pgvector/pgvector:pg16 \
    pg_dump \
    -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" \
    -Fc --no-owner --no-acl \
    -f "/out/deepgraph_pg.dump"
}

echo "[bundle] dumping PostgreSQL -> $DUMP_FILE ..."
if command -v pg_dump >/dev/null 2>&1; then
  run_pg_dump_host
elif command -v docker >/dev/null 2>&1; then
  run_pg_dump_docker
else
  echo "Need pg_dump or docker (image pgvector/pgvector:pg16)." >&2
  exit 1
fi

ls -lh "$DUMP_FILE"

cat > "$STAGING/DEPLOY_README.txt" << 'EOF'
DeepGraph 全链路部署包（PostgreSQL + 源码 + EvoScientist/PaperOrchestra）
========================================================================

打包内容
--------
- Deepgraph/
    完整源码（含 agents/paperorchestra、prompts/paper_orchestra、third_party 参考材料）。
    默认手稿后端：MANUSCRIPT_BACKEND=paper_orchestra（见 config.py）。
- workspace/pdfs/
    仅占位目录；PDF 缓存默认不打入包（如需请在目标机上重新抓取或使用 rsync）。
- deploy_bundle/postgres/deepgraph_pg.dump
    pg_dump -Fc；含库里已入库论文全文/结构化字段、图谱、discovery、experiment、manuscript 等。
- deploy_bundle/artifacts/（若打包机存在对应目录）
    deepgraph_ideas/   → 对应原机 ~/deepgraph_ideas（实验分区、手稿工作区等）
    research/          → 对应原机 ~/research（含 EvoScientist 会话产出路径习惯）

不含
----
- .env（密钥）；请用 Deepgraph/.env.example 复制后填写。
- 本地 SQLite deepgraph.db（生产以 Postgres 为准）。
- EvoScientist / PaperBanana 本体仓库（体积与许可证单独维护），见下文安装路径约定。

--------------------------------------------------------------------
一、恢复 Postgres（示例：docker-compose.postgres.yml）
--------------------------------------------------------------------
  TOP=$(pwd)                     # DEPLOY_README.txt 所在目录
  cd Deepgraph
  docker compose -f docker-compose.postgres.yml up -d
  # Compose 默认映射宿主 5433 → 容器 5432，按需调整：

  docker run --rm --network host \
    -v "$TOP/deploy_bundle/postgres:/dump:ro" \
    -e PGPASSWORD=deepgraph \
    pgvector/pgvector:pg16 \
    pg_restore -h 127.0.0.1 -p 5433 -U deepgraph -d deepgraph \
      --clean --if-exists --no-owner --no-acl \
      /dump/deepgraph_pg.dump

--------------------------------------------------------------------
二、恢复 artifacts（若有 deploy_bundle/artifacts）
--------------------------------------------------------------------
  mkdir -p ~/deepgraph_ideas ~/research
  TOP=$(pwd)
  rsync -a "$TOP/deploy_bundle/artifacts/deepgraph_ideas/" ~/deepgraph_ideas/
  rsync -a "$TOP/deploy_bundle/artifacts/research/" ~/research/

--------------------------------------------------------------------
三、Python 与应用
--------------------------------------------------------------------
  cd Deepgraph
  python3.12 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  cp .env.example .env
  # 必填示例：
  #   DEEPGRAPH_DATABASE_URL=postgresql://...
  #   DEEPGRAPH_LLM_*（主模型网关）
  # PaperOrchestra 文献检索（Semantic Scholar API，降低配额压力）：
  #   DEEPGRAPH_SEMANTIC_SCHOLAR_API_KEY=...
  # 可选配图（PaperOrchestra Step 2）：PaperBanana + DEEPGRAPH_PAPERBANANA_* 或 DEEPGRAPH_PAPERBANANA_CMD
  python3.12 main.py

可选：GROBID（PDF 正文质量） docker compose -f docker-compose.grobid.yml up -d

--------------------------------------------------------------------
四、EvoScientist（新颖性验证 / 深度科研会话）
--------------------------------------------------------------------
代码假定 EvoSci 可执行文件位于：
  ~/EvoScientist/.venv/bin/EvoSci
（见 agents/research_bridge.py、agents/novelty_verifier.py）

请在目标机上自行克隆/安装 EvoScientist，创建该 venv，并确保 EvoSci 在 PATH 约定路径。
也可用 PyPI `pip install EvoScientist` 等方式安装，但若二进制路径不同需自行适配或 symlink。

novelty / verify 会使用 DeepGraph 配置里的 OpenAI-compatible 网关（见 novelty_verifier 内环境变量）。

--------------------------------------------------------------------
五、PaperOrchestra
--------------------------------------------------------------------
已在仓库 agents/paperorchestra + prompts/paper_orchestra；无需单独安装 Python 包。
依赖主 LLM +（可选）Semantic Scholar +（可选）PaperBanana。

EOF

(
  cd "$STAGING"
  tar czvf "$OUT_TAR" DEPLOY_README.txt Deepgraph/ deploy_bundle/
)
echo "[bundle] wrote $OUT_TAR"
ls -lh "$OUT_TAR"
