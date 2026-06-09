#!/usr/bin/env python
"""Seed a temp SQLite DB with a production-shaped dataset and serve the
dashboard, for the Playwright responsiveness E2E (Acceptance A).

Seeds every one of the 9 overview metrics to a distinct non-zero value, a
~3300-node taxonomy (the size that froze the page), and papers in mixed
statuses so ``papers_processed`` (the "文献" card) is deliberately *less* than
the total ingested — exercising known trap ①.

Usage:  python seed_and_serve.py <port>
Prints "READY <port>" to stdout once the server is accepting connections.
"""
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

# Repo root (two levels up from tests/js/) must be importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# A throwaway DB so we never touch the real one.
_TMPDIR = tempfile.mkdtemp(prefix="dg_e2e_")
os.environ["DEEPGRAPH_DB_PATH"] = str(Path(_TMPDIR) / "e2e.db")
os.environ.pop("DEEPGRAPH_DATABASE_URL", None)
# Match a healthy prod deployment: an LLM provider is configured (so
# /api/providers reports stats instead of 500-ing). config.py reads this at
# import, so it must be set before any deepgraph module is imported. No real
# API call is made — /api/providers only reports configured-provider stats.
os.environ.setdefault("MINIMAX_API_KEY", "e2e-dummy-key")

from db import database  # noqa: E402

database.DATABASE_URL = ""
database.DB_PATH = Path(_TMPDIR) / "e2e.db"
database.init_db()

TAXONOMY_NODES = 3300
PAPERS_PROCESSED = 1500   # status in extracted/abstracted/reasoned
PAPERS_UNPROCESSED = 700  # status ingested  -> total 2200, processed 1500
RESULTS = 800
CONTRADICTIONS = 120
INSIGHTS = 240
# Large enough to exercise chunked rendering of the discoveries / experiments
# tabs (each renders one card per row) — these lists are unbounded in prod.
EXPERIMENT_RUNS = 250
DEEP_INSIGHTS = 250
SUBMISSION_BUNDLES = 12
TOKENS_PER_PAPER = 1234


def seed():
    db = database
    # Seeding only needs the COUNT(*) metrics to be right; relax FK enforcement
    # so we can populate tables without standing up every referenced parent row
    # (e.g. submission_bundles -> manuscript_runs).
    db.execute("PRAGMA foreign_keys=OFF")
    # Build a realistic taxonomy TREE (fan-out ~12), NOT 3299 direct children of
    # the root. Prod taxonomies are trees; the radial graph renders a node's
    # direct children, so a flat root would be an artificial stress that exists
    # only in a naive fixture. The Evidence datalist still lists all ~3300 nodes
    # (it flattens the whole tree), which is the real picker-size we care about.
    from collections import deque
    db.execute("INSERT INTO taxonomy_nodes (id, name, parent_id, depth, sort_order) VALUES (?,?,?,?,?)",
               ("ml", "Machine Learning", None, 0, 0))
    created, counter, fanout = 1, 0, 12
    q = deque([("ml", 0)])
    while created < TAXONOMY_NODES and q:
        parent, depth = q.popleft()
        for k in range(fanout):
            if created >= TAXONOMY_NODES:
                break
            nid = f"{parent}.n{k}"
            db.execute("INSERT INTO taxonomy_nodes (id, name, parent_id, depth, sort_order) VALUES (?,?,?,?,?)",
                       (nid, f"Research Area {counter}", parent, depth + 1, k))
            q.append((nid, depth + 1))
            created += 1
            counter += 1

    # Papers: processed (counted by the 文献 card) + unprocessed (NOT counted).
    for i in range(PAPERS_PROCESSED):
        st = ("extracted", "abstracted", "reasoned")[i % 3]
        db.execute("INSERT INTO papers (id, title, status, token_cost) VALUES (?,?,?,?)",
                   (f"proc-{i}", f"Processed paper {i}", st, TOKENS_PER_PAPER))
    for i in range(PAPERS_UNPROCESSED):
        db.execute("INSERT INTO papers (id, title, status, token_cost) VALUES (?,?,?,?)",
                   (f"raw-{i}", f"Ingested-only paper {i}", "ingested", 0))

    # Results (基准结果).
    for i in range(RESULTS):
        db.execute(
            "INSERT INTO results (paper_id, node_id, method_name, dataset_name, metric_name, metric_value) "
            "VALUES (?,?,?,?,?,?)",
            (f"proc-{i % PAPERS_PROCESSED}", "ml", f"method{i}", f"dataset{i % 50}", "accuracy", 0.9))

    # Claims + contradictions (矛盾).
    for i in range(CONTRADICTIONS * 2):
        db.execute("INSERT INTO claims (paper_id, claim_text, claim_type) VALUES (?,?,?)",
                   (f"proc-{i % PAPERS_PROCESSED}", f"claim {i}", "result"))
    for i in range(CONTRADICTIONS):
        db.execute("INSERT INTO contradictions (claim_a_id, claim_b_id, description) VALUES (?,?,?)",
                   (2 * i + 1, 2 * i + 2, f"conflict {i}"))

    # Insights (研究洞见  -> insights table).
    for i in range(INSIGHTS):
        db.execute(
            "INSERT INTO insights (node_id, insight_type, title, hypothesis) VALUES (?,?,?,?)",
            ("ml", "cross_domain_bridge", f"Insight {i}", f"hypothesis {i}"))

    # Deep insights (深度发现 -> deep_insights table).
    for i in range(DEEP_INSIGHTS):
        db.execute("INSERT INTO deep_insights (tier, status, title) VALUES (?,?,?)",
                   (1 + (i % 2), "discovered", f"Discovery {i}"))

    # Experiment runs (实验运行).
    for i in range(EXPERIMENT_RUNS):
        db.execute("INSERT INTO experiment_runs (deep_insight_id, status) VALUES (?,?)",
                   (1 + (i % DEEP_INSIGHTS), "completed"))

    # Submission bundles (投稿包).
    for i in range(SUBMISSION_BUNDLES):
        db.execute(
            "INSERT INTO submission_bundles (manuscript_run_id, bundle_format, status, bundle_path) "
            "VALUES (?,?,?,?)",
            (i + 1, "arxiv", "ready", f"/tmp/bundle_{i}.zip"))

    # An active research agenda so /api/research_agenda/current returns 200
    # (as a healthy prod deployment with an uploaded agenda does) rather than
    # the "no agenda configured" 404.
    db.execute(
        "INSERT INTO research_agendas (name, description, focus_json, is_active) VALUES (?,?,?,?)",
        ("Default Research Agenda", "Seeded for the responsiveness E2E.",
         json.dumps(["machine learning"]), 1))
    # A latest selection so /api/research_agenda/selection/latest returns 200
    # (the agenda tab fetches it after loading the current agenda).
    db.execute(
        "INSERT INTO agenda_selections (agenda_id, selected_insight_id, score, rationale, status) "
        "VALUES (?,?,?,?,?)",
        (1, 1, 0.91, "Seeded selection for the responsiveness E2E.", "completed"))

    db.commit()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5055
    seed()

    from web import app as web_app
    web_app.prewarm_stats_cache()  # warm the /api/stats cache against seeded data

    # Sanity: every overview metric must be non-zero before we serve.
    s = web_app._stats_cache.get()
    expect = {
        "papers_processed": PAPERS_PROCESSED,
        "results_total": RESULTS,
        "taxonomy_nodes_total": TAXONOMY_NODES,
        "contradictions_total": CONTRADICTIONS,
        "insights_total": INSIGHTS,
        "tokens_consumed": PAPERS_PROCESSED * TOKENS_PER_PAPER,
        "experiment_runs_total": EXPERIMENT_RUNS,
        "deep_insights_total": DEEP_INSIGHTS,
        "submission_bundles_total": SUBMISSION_BUNDLES,
    }
    bad = {k: (s.get(k), v) for k, v in expect.items() if s.get(k) != v}
    if bad:
        print("SEED_MISMATCH " + repr(bad), file=sys.stderr)
        sys.exit(2)

    from werkzeug.serving import make_server
    srv = make_server("127.0.0.1", port, web_app.app, threaded=True)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    print(f"READY {port}", flush=True)
    try:
        while True:
            threading.Event().wait(3600)
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
