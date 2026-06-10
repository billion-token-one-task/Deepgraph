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
BENCH_NODE = "ml.bench"   # deterministic leaf with a real method x dataset matrix
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
    # Reserve one slot for a deterministic *leaf* benchmark node ("ml.bench")
    # so the Evidence E2E can navigate to a node that actually has a
    # method x dataset matrix to render (and exercise the heatmap).
    created, counter, fanout = 1, 0, 12
    q = deque([("ml", 0)])
    while created < TAXONOMY_NODES - 1 and q:
        parent, depth = q.popleft()
        for k in range(fanout):
            if created >= TAXONOMY_NODES - 1:
                break
            nid = f"{parent}.n{k}"
            db.execute("INSERT INTO taxonomy_nodes (id, name, parent_id, depth, sort_order) VALUES (?,?,?,?,?)",
                       (nid, f"Research Area {counter}", parent, depth + 1, k))
            q.append((nid, depth + 1))
            created += 1
            counter += 1
    db.execute("INSERT INTO taxonomy_nodes (id, name, parent_id, depth, sort_order) VALUES (?,?,?,?,?)",
               (BENCH_NODE, "Benchmark Leaf", "ml", 1, 99))

    # Papers: processed (counted by the 文献 card) + unprocessed (NOT counted).
    for i in range(PAPERS_PROCESSED):
        st = ("extracted", "abstracted", "reasoned")[i % 3]
        db.execute("INSERT INTO papers (id, title, status, token_cost) VALUES (?,?,?,?)",
                   (f"proc-{i}", f"Processed paper {i}", st, TOKENS_PER_PAPER))
    for i in range(PAPERS_UNPROCESSED):
        db.execute("INSERT INTO papers (id, title, status, token_cost) VALUES (?,?,?,?)",
                   (f"raw-{i}", f"Ingested-only paper {i}", "ingested", 0))

    # Results (基准结果). Lay them out as a dense method x dataset grid on the
    # BENCH_NODE leaf, with *varied* metric values across two metrics so the
    # Evidence matrix renders a real heatmap (cell shade follows the value).
    # Every result is linked to BENCH_NODE via result_taxonomy (the table the
    # matrix builder actually joins on).
    METHODS, DATASETS, METRICS = 30, 14, ("accuracy", "f1")
    rid = 0
    for mi in range(METHODS):
        for di in range(DATASETS):
            for metric in METRICS:
                if rid >= RESULTS:
                    break
                # Spread values widely (40–99) and deterministically so the
                # heatmap has many distinct shades, not one flat fill.
                val = 40 + ((mi * 7 + di * 13 + (0 if metric == "accuracy" else 5)) % 60)
                is_sota = 1 if (mi == di % METHODS and metric == "accuracy") else 0
                db.execute(
                    "INSERT INTO results (paper_id, node_id, method_name, dataset_name, "
                    "metric_name, metric_value, is_sota) VALUES (?,?,?,?,?,?,?)",
                    (f"proc-{rid % PAPERS_PROCESSED}", BENCH_NODE,
                     f"Method {mi:02d}", f"Dataset {di:02d}", metric, float(val), is_sota))
                db.execute("INSERT INTO result_taxonomy (result_id, node_id) VALUES (?,?)",
                           (rid + 1, BENCH_NODE))
                rid += 1
            if rid >= RESULTS:
                break
        if rid >= RESULTS:
            break
    # Top up to the exact RESULTS count (keeps the overview metric stable) with
    # a few extra rows that are not part of the displayed grid.
    while rid < RESULTS:
        db.execute(
            "INSERT INTO results (paper_id, node_id, method_name, dataset_name, metric_name, metric_value) "
            "VALUES (?,?,?,?,?,?)",
            (f"proc-{rid % PAPERS_PROCESSED}", "ml", f"extra_method{rid}", "extra_ds", "accuracy", 0.9))
        rid += 1

    # Claims + contradictions (矛盾).
    for i in range(CONTRADICTIONS * 2):
        db.execute("INSERT INTO claims (paper_id, claim_text, claim_type) VALUES (?,?,?)",
                   (f"proc-{i % PAPERS_PROCESSED}", f"claim {i}", "result"))
    for i in range(CONTRADICTIONS):
        db.execute("INSERT INTO contradictions (claim_a_id, claim_b_id, description) VALUES (?,?,?)",
                   (2 * i + 1, 2 * i + 2, f"conflict {i}"))

    # Insights (研究洞见  -> insights table). Heavy, realistic rows so the
    # Insights tab renders production-weight cards (long hypotheses/experiments
    # plus a JSON-encoded supporting_papers list the frontend parses per card).
    _LOREM = ("We hypothesize that the observed gains stem from a previously "
              "unmodeled interaction between the gating mechanism and the "
              "normalization schedule, which compounds across depth. ") * 4
    _types = ("contradiction_analysis", "method_transfer", "assumption_challenge",
              "ignored_limitation", "paradigm_exhaustion", "cross_domain_bridge")
    for i in range(INSIGHTS):
        papers = json.dumps([f"2401.{1000 + ((i * 7 + k) % 9000):05d}" for k in range(8)])
        db.execute(
            "INSERT INTO insights (node_id, insight_type, title, hypothesis, evidence, "
            "experiment, impact, novelty_score, feasibility_score, supporting_papers) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("ml", _types[i % len(_types)], f"Research Insight {i}: {_LOREM[:60]}",
             _LOREM, _LOREM, _LOREM, _LOREM, 1 + (i % 5), 1 + ((i + 2) % 5), papers))

    # Deep insights (深度发现 -> deep_insights table). Heavy + *displayable*
    # (problem_statement / proposed_method / experimental_plan present) so the
    # Discoveries tab renders real, production-weight cards. Each row carries
    # several JSON fields the frontend parses per card — this is the ~746KB
    # /api/deep_insights?limit=50 payload that drove the main-thread long tasks.
    method_json = json.dumps({
        "name": "Adaptive Gated Mixture", "type": "architecture",
        "one_line": _LOREM[:120],
        "definition": _LOREM * 2,
    })
    plan_json = json.dumps({
        "baselines": [{"name": f"Baseline {b}"} for b in range(6)],
        "datasets": [{"name": f"Dataset {d:02d}"} for d in range(8)],
        "compute_budget": {"total_gpu_hours": 512},
        "ablations": [{"name": f"ablation_{a}", "detail": _LOREM[:200]} for a in range(5)],
    })
    preds_json = json.dumps([{"statement": _LOREM[:140]} for _ in range(4)])
    crit_json = json.dumps({"strongest_attack": _LOREM, "rebuttals": [_LOREM] * 3})
    fielda_json = json.dumps({"node_id": "ml", "name": "Source Field"})
    fieldb_json = json.dumps({"node_id": BENCH_NODE, "name": "Target Field"})
    for i in range(DEEP_INSIGHTS):
        tier = 1 + (i % 2)
        db.execute(
            "INSERT INTO deep_insights (tier, status, title, novelty_status, "
            "adversarial_score, formal_structure, transformation, field_a, field_b, "
            "predictions, adversarial_critique, problem_statement, existing_weakness, "
            "proposed_method, experimental_plan, evidence_summary) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (tier, "discovered", f"Discovery {i}: {_LOREM[:50]}",
             ("novel", "partially_exists", "exists")[i % 3], 6.0 + (i % 4),
             _LOREM, _LOREM, fielda_json, fieldb_json, preds_json, crit_json,
             _LOREM, _LOREM, method_json, plan_json, _LOREM * 2))

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
