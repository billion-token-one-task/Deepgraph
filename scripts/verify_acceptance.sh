#!/usr/bin/env bash
# verify_acceptance.sh — one-shot regenerator + smoke check for the 6
# acceptance bundles produced by issues #9, #11(#12, #13, #14, #15).
#
# Reviewer ask (PR #10 review, 2026-05-16, item 5):
#
#   "请交付 scripts/verify_acceptance.sh（或 Makefile target）：
#    从干净 venv 一行命令跑通全部 6 个 acceptance bundle 的端到端
#    regenerate，结尾打印明确 PASS。让任何贡献者拿到 main 都能一行
#    验证系统状态。"
#
# What this script does, end-to-end, from a clean checkout:
#
#   1. Validates a Python interpreter is available.
#   2. Wipes the per-bundle scratch SQLite DBs under /tmp so each builder
#      starts from a deterministic blank slate.
#   3. Runs each builder via `python -m scripts.<builder>` so REPO_ROOT
#      resolves correctly regardless of CWD; the agenda-loop builder
#      additionally takes DEEPGRAPH_DB_PATH per its module docstring.
#   4. Runs the umbrella aggregator last so its commit field + sub-bundle
#      pointers always match what was just rewritten (this is what fixed
#      review item 2 — the stale `commit: ef15797` in main).
#   5. Verifies all 6 artifacts/<name>.json exist and that their `commit`
#      field (where present) matches the current `git rev-parse HEAD`.
#   6. Prints `PASS` (and exits 0) only if every step above succeeded.
#
# Usage:
#
#   bash scripts/verify_acceptance.sh
#
# Optional environment variables:
#
#   PYTHON      Python interpreter to use (default: python3).
#   SKIP_DEMO   Set to 1 to skip the optional scripts/demo_full_paper_compile.py
#               PDF generation step (default: skipped — tectonic isn't usually
#               installed on contributor laptops; the umbrella tolerates the
#               "deferred" state and the verifier does not require PDFs).
#
# Exit codes:
#
#   0   all 6 bundles regenerated + commit fields fresh + JSON parses OK
#   1   any builder failed, missing artifact, or stale commit field

set -Eeuo pipefail

PYTHON="${PYTHON:-python3}"
SKIP_DEMO="${SKIP_DEMO:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ARTIFACTS="$REPO_ROOT/artifacts"
mkdir -p "$ARTIFACTS"

# ---------------------------------------------------------------- helpers ---

log()  { printf '\033[1;34m[verify]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[verify]\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[1;31m[verify FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || fail "required command not on PATH: $1"
}

run_builder() {
    # $1 = human label, $2 = python module path, $3 = optional db path
    local label="$1" module="$2" db_path="${3:-}"
    log "regenerating $label  (python -m $module)"
    if [[ -n "$db_path" ]]; then
        rm -f "$db_path"
        DEEPGRAPH_DATABASE_URL="" DEEPGRAPH_DB_PATH="$db_path" \
            "$PYTHON" -m "$module" \
            || fail "builder failed: $module"
    else
        "$PYTHON" -m "$module" \
            || fail "builder failed: $module"
    fi
}

# ----------------------------------------------------------- preflight ---

require_cmd "$PYTHON"
require_cmd git

HEAD_SHA="$(git rev-parse HEAD)"
log "repo HEAD = $HEAD_SHA"
log "python    = $($PYTHON --version 2>&1)"

# Wipe any leftover scratch DBs from a previous run so SQLite auto-increment
# ids start at 1 and `result_packet_structural_sha256` stays stable.
rm -f /tmp/agenda_loop_acceptance.db \
      /tmp/d1_acceptance.db \
      /tmp/d2_acceptance.db \
      /tmp/d3_acceptance.db \
      /tmp/d4_acceptance.db

# --------------------------------------------------- builders (ordered) ---

# d1-d4 first so the umbrella sees fresh sub-bundles; agenda-loop is
# independent and can run alongside but we serialise for clarity.
run_builder "d1 template router"     scripts.build_d1_acceptance         /tmp/d1_acceptance.db
run_builder "d2 top-venue adapters"  scripts.build_d2_acceptance         /tmp/d2_acceptance.db
run_builder "d3 format linter"       scripts.build_d3_acceptance         /tmp/d3_acceptance.db
run_builder "d4 manuscript routing"  scripts.build_d4_acceptance         /tmp/d4_acceptance.db
run_builder "agenda loop (#9)"       scripts.build_agenda_loop_acceptance /tmp/agenda_loop_acceptance.db

# Optional: produce real PDFs for the umbrella to pick up. Skipped by
# default because tectonic isn't usually installed on contributor
# laptops; the umbrella records "deferred" per venue in that case.
if [[ "$SKIP_DEMO" != "1" ]]; then
    log "compiling per-venue PDFs (scripts/demo_full_paper_compile.py)"
    "$PYTHON" scripts/demo_full_paper_compile.py \
        || warn "demo_full_paper_compile.py failed; umbrella will record 'deferred' bundles"
fi

# Umbrella last — aggregates d1-d4 + git HEAD into the issue-#11 bundle.
run_builder "umbrella (#11)"         scripts.build_manuscript_venue_routing_umbrella

# ------------------------------------------------------- verification ---

EXPECTED=(
    "agenda_loop_acceptance.json"
    "d1_template_router_acceptance.json"
    "d2_top_venue_adapters_acceptance.json"
    "d3_format_linter_acceptance.json"
    "d4_manuscript_routing_api_acceptance.json"
    "manuscript_venue_routing_acceptance.json"
)

log "verifying 6 acceptance bundles exist + parse + match HEAD"

for name in "${EXPECTED[@]}"; do
    path="$ARTIFACTS/$name"
    [[ -f "$path" ]] || fail "missing artifact: $path"
    # Parse + (when the field exists) cross-check commit against HEAD.
    "$PYTHON" - "$path" "$HEAD_SHA" <<'PY' || fail "stale or invalid: $path"
import json, sys
path, head = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as fh:
    bundle = json.load(fh)
commit = bundle.get("commit")
if commit is None:
    # Some sub-bundles use "head_ref" instead; either is acceptable as
    # long as something points back at HEAD.
    commit = bundle.get("head_ref")
if commit and commit != head:
    print(f"STALE: {path} -> commit={commit!r} expected={head!r}", file=sys.stderr)
    sys.exit(1)
print(f"  ok  {path}  (commit={(commit or 'n/a')[:12]})")
PY
done

# ------------------------------------------------------------------- done ---

echo
echo "================================================================"
echo "PASS — 6/6 acceptance bundles regenerated at commit ${HEAD_SHA:0:12}"
echo "================================================================"
