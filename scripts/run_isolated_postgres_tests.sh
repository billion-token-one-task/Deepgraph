#!/usr/bin/env bash
# Run tests/integration/ against a throwaway PostgreSQL cluster.
#
# The isolated integration modules are the only automated coverage of the V1
# data layer, and they refuse to run without an explicitly disposable database.
# This script creates one, tears it down again, and never touches a database a
# service is using: its own data directory, its own port, its own socket
# directory, and a database name the guarded migration recognises as disposable.
#
# Environment:
#   PYTHON                      interpreter to use (default: python3)
#   ISOLATED_PG_BINDIR          directory holding initdb/pg_ctl/createdb
#   ISOLATED_PG_PORT            port for the throwaway cluster (default 5455)
#   ISOLATED_PG_DBNAME          database name, must contain an isolation marker
#   ISOLATED_PG_ROOT            cluster directory (default: a fresh mktemp -d)
#   DEEPGRAPH_LIVE_PG_PORT      port this script must refuse to reuse
#   KEEP_CLUSTER=1              leave the cluster running for inspection
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
PORT="${ISOLATED_PG_PORT:-5455}"
DBNAME="${ISOLATED_PG_DBNAME:-deepgraph_test}"
# Same value as LIVE_LOCAL_PORT in scripts/meta_harness_migration.py; this is
# the port the fixture refuses to reuse, not one it connects to.
LIVE_PORT="${DEEPGRAPH_LIVE_PG_PORT:-5433}"

if [ -n "${ISOLATED_PG_BINDIR:-}" ]; then
  PG_INITDB="$ISOLATED_PG_BINDIR/initdb"
  PG_CTL="$ISOLATED_PG_BINDIR/pg_ctl"
  PG_CREATEDB="$ISOLATED_PG_BINDIR/createdb"
else
  PG_INITDB="$(command -v initdb)"
  PG_CTL="$(command -v pg_ctl)"
  PG_CREATEDB="$(command -v createdb)"
fi

# Fail closed rather than discover the mistake by writing to a live database.
if [ "$PORT" = "$LIVE_PORT" ]; then
  echo "refusing: isolated port $PORT equals the live service port" >&2
  exit 2
fi
case "$DBNAME" in
  *test*|*ci*|*canary*|*sandbox*|*staging*|*restore*|*shadow*) ;;
  *) echo "refusing: database name '$DBNAME' does not identify a disposable database" >&2; exit 2 ;;
esac

# A cluster left behind by KEEP_CLUSTER=1 fails later as "could not start
# server", which reads like a broken fixture rather than a busy port.
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q "127.0.0.1:$PORT "; then
  echo "refusing: something is already listening on 127.0.0.1:$PORT" >&2
  echo "(a cluster kept with KEEP_CLUSTER=1? stop it, or set ISOLATED_PG_PORT)" >&2
  exit 2
fi

CLUSTER="${ISOLATED_PG_ROOT:-$(mktemp -d "${TMPDIR:-/var/tmp}/deepgraph-isolated-pg.XXXXXX")}"
DATA="$CLUSTER/data"
SOCKETS="$CLUSTER/run"
OWNER="$(id -un)"
URL="postgresql://$OWNER@127.0.0.1:$PORT/$DBNAME"
VIRGIN_URL="postgresql://$OWNER@127.0.0.1:$PORT/${DBNAME}_virgin"
EMPTY_URL="postgresql://$OWNER@127.0.0.1:$PORT/${DBNAME}_empty"

cleanup() {
  if [ "${KEEP_CLUSTER:-0}" = "1" ]; then
    echo "cluster kept at $CLUSTER ($URL)"
    return
  fi
  "$PG_CTL" -D "$DATA" -m immediate stop >/dev/null 2>&1 || true
  rm -rf "$CLUSTER"
}
trap cleanup EXIT

mkdir -p "$SOCKETS"
"$PG_INITDB" -D "$DATA" -U "$OWNER" --auth=trust -E UTF8 >"$CLUSTER/initdb.log" 2>&1
"$PG_CTL" -D "$DATA" -l "$CLUSTER/postgres.log" \
  -o "-p $PORT -k $SOCKETS -h 127.0.0.1" -w start >/dev/null
"$PG_CREATEDB" -h 127.0.0.1 -p "$PORT" "$DBNAME"
# test_meta_harness_postgres.py asserts that its first apply reports
# "applied", so it needs a database where the migration has not run yet.
"$PG_CREATEDB" -h 127.0.0.1 -p "$PORT" "${DBNAME}_virgin"
# test_fresh_schema_bootstrap_postgres.py asserts that init_db() can create the
# schema from nothing, so its database must have had nothing applied to it.
"$PG_CREATEDB" -h 127.0.0.1 -p "$PORT" "${DBNAME}_empty"

cd "$ROOT_DIR"

# db/pg_init is the documented once-per-database bootstrap. init_db() repairs an
# existing schema; it is not the path that creates one.
DEEPGRAPH_DATABASE_URL="$URL" "$PYTHON" -m db.pg_init
DEEPGRAPH_DATABASE_URL="$VIRGIN_URL" "$PYTHON" -m db.pg_init

# The integration modules each apply the migrations they need, but only the
# ones they name. Applying the whole ordered set here keeps the fixture from
# depending on which module happens to run first.
DEEPGRAPH_DATABASE_URL="" "$PYTHON" - "$URL" <<'PY'
import pathlib
import sys

from scripts.meta_harness_migration import MIGRATION_KEYS, apply_to_isolated_restore

url = sys.argv[1]
# The fixture is not a candidate release; stamp the tree it was built from.
import subprocess
commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
).stdout.strip()
for key in MIGRATION_KEYS:
    if not (pathlib.Path("db/migrations") / f"{key}.sql").is_file():
        raise SystemExit(f"migration file is missing:{key}")
    result = apply_to_isolated_restore(url, source_commit=commit, migration_key=key)
    print(f"{key} {result['status']}", flush=True)
PY

COMMIT="$(git rev-parse HEAD)"
status=0
for module in "$ROOT_DIR"/tests/integration/test_*.py; do
  name="$(basename "$module")"
  echo "=== $name"
  case "$name" in
    test_meta_harness_postgres.py) module_url="$VIRGIN_URL" ;;
    test_fresh_schema_bootstrap_postgres.py) module_url="$EMPTY_URL" ;;
    *) module_url="$URL" ;;
  esac
  # Each module asserts that db.database was not imported under another URL, so
  # every module needs its own interpreter.
  DEEPGRAPH_ISOLATED_POSTGRES_URL="$module_url" \
  DEEPGRAPH_ALLOW_ISOLATED_INTEGRATION_TESTS=1 \
  META_HARNESS_CANDIDATE_COMMIT="$COMMIT" \
  DEEPGRAPH_DATABASE_URL="" \
    "$PYTHON" -m pytest -q -p no:cacheprovider "$@" "$module" || status=1
done
exit $status
