"""Snapshot every GET route of web.app against a database.

usage:
  python -I api_snapshot.py snap <repo_root> <db_url> <out_dir>
  python -I api_snapshot.py mask <snapA_dir> <snapB_dir> <mask.json>     # same code twice -> volatile paths
  python -I api_snapshot.py diff <base_dir> <cand_dir> <mask.json>       # exit 1 on any unmasked difference

snap: imports web.app from <repo_root> with DEEPGRAPH_DATABASE_URL=<db_url>, samples ids for
parameterised routes from the database (deterministic: lowest/highest ids), requests each URL with
the Flask test client and writes <out_dir>/routes.json {url: {"status", "mimetype", "body"}} where
body is parsed JSON when possible, else the sha256 of the bytes. SSE routes record headers only.
The caller must give each snap a fresh copy of the database (CREATE DATABASE .. TEMPLATE ..).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

IDX = re.compile(r"\[\d+\]")


def _flatten(value, path="$", out=None):
    out = {} if out is None else out
    if isinstance(value, dict):
        if not value:
            out[path] = {}
        for k, v in value.items():
            _flatten(v, f"{path}.{k}", out)
    elif isinstance(value, list):
        out[path + ".__len__"] = len(value)
        for i, v in enumerate(value):
            _flatten(v, f"{path}[{i}]", out)
    else:
        out[path] = value
    return out


def _generalise(path: str) -> str:
    return IDX.sub("[*]", path)


def _masked(path: str, masks: set[str]) -> bool:
    g = _generalise(path)
    for m in masks:
        if g == m or g.startswith((m + ".", m + "[")):
            return True
    return False


def _sample_ids(db) -> dict[str, list]:
    def ids(sql):
        try:
            return [r[next(iter(dict(r)))] for r in db.fetchall(sql)]
        except Exception as exc:  # noqa: BLE001 - a table may not exist on this backend
            print(f"[snap] id sample failed: {sql[:60]}: {exc}", file=sys.stderr)
            return []

    def lo_hi(table, col="id", where=""):
        w = f" WHERE {where}" if where else ""
        return ids(f"SELECT {col} FROM {table}{w} ORDER BY {col} ASC LIMIT 2") + ids(
            f"SELECT {col} FROM {table}{w} ORDER BY {col} DESC LIMIT 2"
        )

    return {
        "insight_id": lo_hi("deep_insights")
        + ids("SELECT deep_insight_id FROM experiment_runs WHERE deep_insight_id IS NOT NULL ORDER BY id DESC LIMIT 3"),
        "run_id": lo_hi("experiment_runs"),
        "paper_id": lo_hi("papers"),
        "node_id": ids("SELECT id FROM taxonomy_nodes ORDER BY id ASC LIMIT 3"),
        "agenda_id": lo_hi("research_agendas"),
        "bundle_id": lo_hi("submission_bundles"),
        "candidate_id": lo_hi("entity_merge_candidates"),
        "authority_id": lo_hi("frontier_evaluation_authorities"),
        "artifact_id": lo_hi("experiment_artifacts"),
    }


EXTRA_URLS = [
    "/api/search?q=transformer",
    "/api/search?q=",
    "/api/papers?page=2",
    "/api/papers?limit=5&offset=10",
    "/api/deep_insights?limit=5",
    "/api/experiments?limit=5",
    "/api/claims?limit=5",
    "/static/js/app.js",
    "/static/js/i18n.js",
    "/static/css/style.css",
]


def snap(repo_root: str, db_url: str, out_dir: str) -> None:
    out_dir = str(Path(out_dir).resolve())  # before chdir below
    os.environ["DEEPGRAPH_DATABASE_URL"] = db_url
    os.environ.pop("DEEPGRAPH_DB_PATH", None)
    sys.path.insert(0, repo_root)
    os.chdir(repo_root)
    os.environ["DEEPGRAPH_META_HARNESS_OPERATOR_TOKEN"] = "snapshot-operator-token"
    import web.app as web_app
    from db import database as db

    app = web_app.app
    # main.py warms the dashboard cache before serving; do it synchronously so
    # /api/stats answers the way production does, without a refresher thread.
    web_app._stats_cache.prewarm()
    samples = _sample_ids(db)
    busiest = db.fetchall(
        "SELECT agenda_id FROM experiment_runs WHERE agenda_id IS NOT NULL "
        "GROUP BY agenda_id ORDER BY COUNT(*) DESC, agenda_id LIMIT 1"
    )
    busy_agenda = dict(busiest[0])["agenda_id"] if busiest else None
    urls: list[str] = []
    for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
        if "GET" not in rule.methods or rule.endpoint == "static":
            continue
        args = sorted(rule.arguments)
        if not args:
            urls.append(rule.rule)
            if rule.rule.startswith("/api/"):
                urls.append(rule.rule + "?limit=3")
                if busy_agenda is not None:
                    urls.append(f"{rule.rule}?agenda_id={busy_agenda}")
            continue
        if args == ["asset", "insight_id"]:
            continue  # served files; covered by the pdf/tex routes
        key = args[0]
        for value in dict.fromkeys(samples.get(key) or []):
            urls.append(_build(rule, key, value))
    urls += EXTRA_URLS
    client = app.test_client()
    result = {}
    for url in urls:
        headers = {"X-DeepGraph-Operator-Token": "snapshot-operator-token"} if "/meta-harness/" in url else {}
        resp = client.get(url, buffered=False, headers=headers)
        entry = {"status": resp.status_code, "mimetype": resp.mimetype}
        if resp.mimetype == "text/event-stream":
            resp.close()
            entry["body"] = "<stream>"
        else:
            data = resp.get_data()
            resp.close()
            try:
                entry["body"] = json.loads(data)
            except ValueError:
                entry["body"] = "sha256:" + hashlib.sha256(data).hexdigest()
        result[url] = entry
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, "routes.json").write_text(json.dumps(result, sort_keys=True, default=str, ensure_ascii=False))
    Path(out_dir, "samples.json").write_text(json.dumps(samples, default=str))
    print(f"[snap] {len(result)} urls -> {out_dir}")


def _build(rule, key, value) -> str:
    return re.sub(r"<(?:[^:<>]+:)?" + key + r">", str(value), rule.rule)


def _load(d: str) -> dict:
    return json.loads(Path(d, "routes.json").read_text())


def _differences(a: dict, b: dict, masks: set[str]) -> dict[str, list[str]]:
    diffs: dict[str, list[str]] = {}
    for url in sorted(set(a) | set(b)):
        if url not in a or url not in b:
            diffs[url] = ["<url present on one side only>"]
            continue
        fa, fb = _flatten(a[url]), _flatten(b[url])
        bad = []
        for p in sorted(set(fa) | set(fb)):
            if fa.get(p, "<missing>") != fb.get(p, "<missing>") and not _masked(f"{url}|{p}", masks):
                bad.append(f"{p}: {str(fa.get(p, '<missing>'))[:120]!r} -> {str(fb.get(p, '<missing>'))[:120]!r}")
        if bad:
            diffs[url] = bad
    return diffs


def mask(a_dir: str, b_dir: str, out: str) -> None:
    a, b = _load(a_dir), _load(b_dir)
    masks: set[str] = set()
    for url, lines in _differences(a, b, set()).items():
        for line in lines:
            p = line.split(": ", 1)[0]
            p = p.removesuffix(".__len__")
            masks.add(_generalise(f"{url}|{p}"))
    Path(out).write_text(json.dumps(sorted(masks), indent=1))
    print(f"[mask] {len(masks)} volatile paths over {len({m.split('|')[0] for m in masks})} urls -> {out}")


def diff(base: str, cand: str, mask_file: str) -> int:
    masks = set(json.loads(Path(mask_file).read_text()))
    diffs = _differences(_load(base), _load(cand), masks)
    a = _load(base)
    print(f"[diff] {len(a)} urls compared, {len(diffs)} differ")
    for url, lines in diffs.items():
        print(f"  {url}")
        for line in lines[:8]:
            print(f"      {line}")
        if len(lines) > 8:
            print(f"      ... {len(lines) - 8} more")
    return 1 if diffs else 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "snap":
        snap(*sys.argv[2:5])
    elif cmd == "mask":
        mask(*sys.argv[2:5])
    elif cmd == "diff":
        sys.exit(diff(*sys.argv[2:5]))
    else:
        sys.exit(__doc__)
