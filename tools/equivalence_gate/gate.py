"""Record what one Deepgraph tree does, so two trees can be compared for equivalence.

usage: python -I gate.py <tree> <out_dir> [--pg <server_url>] [--tag <name>]

Writes <out_dir>/gate.json. compare.py then decides whether a change kept
behaviour. Run the same copy of this script on both trees (CI runs the base
branch's copy), so a change cannot loosen the checks it is judged by.

  tests     pytest: ids that pass, ids that fail
  imports   every production module imported in its own process
  routes    the Flask url map (rule, methods, endpoint)
  names     ruff F821 (undefined names) outside tests
  entry     the scripts systemd and ops jobs start still compile
  symbols   module:name pairs imported from outside the repository still resolve
With --pg <server_url> (a PostgreSQL server you can create databases on):
  schema    columns and indexes after init_db() and the meta-harness migrations
  pg_tests  tests/integration, each file on its own fresh database
  api       every GET route, on a bootstrapped database seeded by seed_db.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENTRY = [
    "main.py", "scripts/auto_advance.py", "scripts/auto_execute.py",
    "scripts/harvest_papers.py", "scripts/observe_agenda_run.py",
    "scripts/render_observation_report.py", "scripts/repair_orphaned_grants.py",
    "scripts/deepgraph_selfheal.py", "orchestrator/selfheal_policy.py",
]
SKIP_DIRS = {".git", ".venv", "node_modules", "tests", "__pycache__", "third_party", "tools"}
SCHEMA_SQL = """
SELECT json_build_object(
  'columns', (SELECT json_agg(t ORDER BY t) FROM (
      SELECT table_name || '.' || column_name || ' ' || data_type || ' ' || is_nullable
             || ' ' || coalesce(column_default, '') AS t
      FROM information_schema.columns WHERE table_schema = 'public') s),
  'indexes', (SELECT json_agg(indexdef ORDER BY indexdef) FROM pg_indexes WHERE schemaname = 'public'))
"""

tree = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)
args = sys.argv[3:]
pg = args[args.index("--pg") + 1].rstrip("/") if "--pg" in args else ""
tag = args[args.index("--tag") + 1] if "--tag" in args else "t"
PY = sys.executable
base_env = {k: v for k, v in os.environ.items() if not k.startswith("DEEPGRAPH_")}
base_env.update(PYTHONPATH=str(tree), PYTHONDONTWRITEBYTECODE="1")
res: dict = {"commit": subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"],
                                      capture_output=True, text=True, check=False).stdout.strip()}


def run(cmd, timeout=600, env=None):
    try:
        p = subprocess.run(cmd, cwd=tree, env=env or base_env, capture_output=True, text=True, timeout=timeout,
                           check=False)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as exc:
        return "timeout", str(exc)


def psql(sql, db="postgres"):
    import psycopg
    with psycopg.connect(f"{pg}/{db}", autocommit=True) as conn:
        cur = conn.execute(sql)
        return cur.fetchall() if cur.description else []


def fresh_db(name):
    psql(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    psql(f'CREATE DATABASE "{name}"')
    return f"{pg}/{name}"


def bootstrap(url, migrations=True):
    """Bring an empty database to production's shape: init_db(), then the ordered migrations."""
    env = dict(base_env, DEEPGRAPH_DATABASE_URL=url)
    rc, log = run([PY, "-c", "from db import database; database.init_db()"], env=env)
    if rc != 0:
        return f"init_db failed: {log[-500:]}"
    if not migrations:
        return ""
    rc, log = run([PY, "-c", (
        "from scripts.meta_harness_migration import apply_to_isolated_restore, MIGRATION_KEYS\n"
        f"for k in MIGRATION_KEYS: apply_to_isolated_restore({url!r}, source_commit={res['commit']!r}, migration_key=k)")],
        env=base_env)
    return "" if rc == 0 else f"migrations failed: {log[-500:]}"


# tests
rc, log = run([PY, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", "tests",
               "--ignore=tests/integration"], 3600)
(out / "pytest.log").write_text(log)
res["tests"] = {
    "passed": sorted(set(re.findall(r"^PASSED (\S+)", log, re.MULTILINE))),
    "failing": sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", log, re.MULTILINE))),
    "summary": (log.strip().splitlines() or [""])[-1],
}

# imports
mods = []
for dp, dn, fn in os.walk(tree):
    dn[:] = [d for d in dn if d not in SKIP_DIRS]
    for f in fn:
        if f.endswith(".py"):
            rel = Path(dp, f).relative_to(tree)
            m = ".".join(rel.with_suffix("").parts).removesuffix(".__init__")
            if "-" in m or ("." not in m and rel.parent != Path(".")):
                continue
            mods.append(m)
failing = {}
for m in sorted(mods):
    rc, log = run([PY, "-c", f"import importlib; importlib.import_module({m!r})"], 120)
    if rc != 0:
        failing[m] = ([line for line in log.strip().splitlines() if line.strip()][-1:] or [str(rc)])[0][:300]
res["imports"] = {"modules": sorted(mods), "failing": failing}

# routes
rc, log = run([PY, "-c", ("import json; from web.app import app; print(json.dumps(sorted("
               "f'{r.rule} {sorted(r.methods - {\"HEAD\", \"OPTIONS\"})} {r.endpoint}' for r in app.url_map.iter_rules())))")])
try:
    res["routes"] = json.loads(log.strip().splitlines()[-1])
except (ValueError, IndexError):
    res["routes"] = {"error": log[-2000:]}

# names
rc, log = run(["ruff", "check", "--no-cache", "--isolated", "--select", "F821", "--output-format", "concise",
               "--exclude", "tests,third_party,tools", "."])
res["names"] = sorted({re.sub(r":\d+:\d+:", ":", line) for line in log.splitlines() if " F821 " in line})

# entry
res["entry"] = {}
for script in ENTRY:
    if not (tree / script).exists():
        res["entry"][script] = "missing"
        continue
    rc, log = run([PY, "-m", "py_compile", script])
    res["entry"][script] = "ok" if rc == 0 else log.strip()[-300:]

# symbols
res["symbols"] = {}
for line in (HERE / "external_symbols.txt").read_text().split():
    mod, _, name = line.partition(":")
    code = (f"import importlib\nm = importlib.import_module({mod!r})\n"
            + (f"if not hasattr(m, {name!r}): importlib.import_module({mod + '.' + name!r})" if name else ""))
    rc, log = run([PY, "-c", code], 120)
    res["symbols"][line] = "ok" if rc == 0 else (log.strip().splitlines() or ["?"])[-1][:200]

if pg:
    url = fresh_db(f"deepgraph_test_schema_{tag}")
    err = bootstrap(url)
    res["schema"] = {"error": err} if err else psql(SCHEMA_SQL, url.rsplit("/", 1)[1])[0][0]

    res["pg_tests"] = {}
    for f in sorted((tree / "tests" / "integration").glob("test_*.py")):
        url = fresh_db(f"deepgraph_test_it_{tag}")
        # The bootstrap test needs an empty database, and the migration test
        # applies the migrations itself.
        if "fresh_schema_bootstrap" not in f.name:
            err = bootstrap(url, migrations="test_meta_harness_postgres" not in f.name)
            if err:
                res["pg_tests"][f.name] = {"error": err}
                continue
        env = dict(base_env, DEEPGRAPH_ISOLATED_POSTGRES_URL=url, DEEPGRAPH_ALLOW_ISOLATED_INTEGRATION_TESTS="1",
                   META_HARNESS_CANDIDATE_COMMIT=res["commit"])
        rc, log = run([PY, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", str(f)], 900, env=env)
        res["pg_tests"][f.name] = {
            "passed": sorted(set(re.findall(r"^PASSED (\S+)", log, re.MULTILINE))),
            "failing": sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", log, re.MULTILINE))),
        }

    # Two snapshots on two identically seeded databases: what differs between
    # them is volatile (clocks, cache ages) and is masked when trees are compared.
    for n in (1, 2):
        url = fresh_db(f"deepgraph_test_api_{tag}{n}")
        err = bootstrap(url)
        if err:
            res["api"] = {"error": err}
            break
        rc, log = run([PY, "-I", str(HERE / "seed_db.py"), url])
        rc, log = run([PY, "-I", str(HERE / "api_snapshot.py"), "snap", str(tree), url, str(out / f"api{n}")], 1800)
        if rc != 0:
            res["api"] = {"error": log[-1000:]}
            break
    else:
        res["api"] = {"dir": str(out / "api1"), "mask_dir": str(out / "api2")}

(out / "gate.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
t = res["tests"]
print(f"[gate] {res['commit'][:7]} tests {len(t['passed'])} passed / {len(t['failing'])} failing; "
      f"imports {len(mods) - len(failing)}/{len(mods)}; routes "
      f"{len(res['routes']) if isinstance(res['routes'], list) else 'ERROR'}; undefined names {len(res['names'])}")
