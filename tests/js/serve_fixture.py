#!/usr/bin/env python3
"""Seed a throwaway SQLite fixture and serve the dashboard for the Playwright gate.

This NEVER touches the real deepgraph.db: it points DEEPGRAPH_DB_PATH at a temp
file (default /tmp/dg_fixture_e2e.db), builds the schema via the app's own
init_db, seeds the taxonomy, then inserts a small but realistic dataset so the
9 stat cards show real numbers, the taxonomy node has 10+ children, and one node
has an entity-relation graph + gaps + insights (incl. a contradiction) to drive
the "domain -> gap -> discovery" story panel.

Usage:  DEEPGRAPH_DB_PATH=... PORT=8765 python serve_fixture.py
"""
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_port = os.environ.get("PORT", "8766")
DB_PATH = os.environ.setdefault("DEEPGRAPH_DB_PATH", f"/tmp/dg_fixture_e2e_{_port}.db")
os.environ.setdefault("DEEPGRAPH_ROOT_NODE_ID", "ml")
# keep background pipelines/agents off for a deterministic, lightweight server
os.environ.setdefault("DEEPGRAPH_AUTO_PIPELINE_ENABLED", "0")
os.environ.setdefault("DEEPGRAPH_AUTO_RESEARCH_ENABLED", "0")
# A dummy provider key so /api/providers returns stats instead of 500 (no network
# call is made — get_provider_stats only reads the configured provider table).
os.environ.setdefault("MINIMAX_API_KEY", "fixture-dummy-key")

# fresh db each run
p = Path(DB_PATH)
if p.exists():
    p.unlink()

from db import database as db  # noqa: E402
from db.database import init_db  # noqa: E402
from db.taxonomy import seed_taxonomy, get_children  # noqa: E402
from db.evidence_graph import upsert_node_graph_summary  # noqa: E402

init_db()
seed_taxonomy()

PARENT = "ml.dl"

# Add synthetic children so the parent has 10+ subnodes (readability case A).
existing = get_children(PARENT)
base_sort = len(existing) + 1
extra = [
    ("ml.dl.neuro_sym", "Neuro-Symbolic Reasoning"),
    ("ml.dl.world_models", "World Models & Simulation"),
    ("ml.dl.continual", "Continual / Lifelong Learning"),
    ("ml.dl.interpret", "Mechanistic Interpretability"),
]
for i, (nid, name) in enumerate(extra):
    db.execute(
        "INSERT INTO taxonomy_nodes (id, name, parent_id, depth, description, sort_order) "
        "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO NOTHING",
        (nid, name, PARENT, 2, f"Fixture subarea: {name}.", base_sort + i),
    )
db.commit()

children = get_children(PARENT)

# Papers + classification -> gives each child a paper_count and feeds stats.
pid_seq = 0
for ci, c in enumerate(children):
    n_papers = 3 + (ci * 2) % 11  # 3..13
    for k in range(n_papers):
        pid_seq += 1
        pid = f"2401.{pid_seq:05d}"
        db.execute(
            "INSERT INTO papers (id, title, abstract, status, token_cost, published_date) "
            "VALUES (?, ?, ?, 'reasoned', ?, '2024-01-15')",
            (pid, f"{c['name']} advances, part {k + 1}",
             f"A study on {c['name']} with new methods and benchmarks.", 1200 + k * 10),
        )
        db.execute(
            "INSERT INTO paper_taxonomy (paper_id, node_id, confidence) VALUES (?, ?, 1.0) "
            "ON CONFLICT (paper_id, node_id) DO NOTHING",
            (pid, c["id"]),
        )
        # a couple of results -> method_count (needs a result_taxonomy mapping)
        if k < 2:
            rkey = f"{pid}-r{k}"
            db.execute(
                "INSERT INTO results (paper_id, node_id, method_name, dataset_name, metric_name, "
                "metric_value, result_key) VALUES (?, ?, ?, ?, 'accuracy', ?, ?)",
                (pid, c["id"], f"Method-{c['id'].split('.')[-1]}-{k}", "BenchX", 0.8 + k * 0.05, rkey),
            )
            row = db.fetchone("SELECT id FROM results WHERE result_key=?", (rkey,))
            if row:
                db.execute(
                    "INSERT INTO result_taxonomy (result_id, node_id) VALUES (?, ?) "
                    "ON CONFLICT (result_id, node_id) DO NOTHING",
                    (row["id"], c["id"]),
                )
db.commit()

# Matrix gaps -> gap_count on a few children + leaf gaps for the story panel.
gap_children = children[:4]
for gi, c in enumerate(gap_children):
    for j in range(1 + gi % 3):
        db.execute(
            "INSERT INTO matrix_gaps (node_id, method_name, dataset_name, metric_name, "
            "gap_description, research_proposal, value_score) VALUES (?, ?, ?, 'accuracy', ?, ?, ?)",
            (c["id"], f"Method-{j}", "BenchX",
             f"No strong baseline for {c['name']} on BenchX under distribution shift.",
             f"Evaluate {c['name']} methods on BenchX with shifted splits.", 3.5 + j),
        )
db.commit()

# Insights (incl. a contradiction) on the parent + children -> drives the story
# panel and the insights/contradictions stat cards.
insight_rows = [
    (PARENT, "contradiction_analysis",
     "Scaling claims conflict across two BenchX evaluations",
     "If both hold, the scaling law and the saturation result cannot both be correct.",
     "Paper A reports monotone gains; Paper B reports saturation at the same scale.",
     "Re-run both protocols on a shared split and reconcile.", 4, 3),
    (children[0]["id"], "method_transfer",
     f"Transfer a {children[0]['name']} trick to a sibling area",
     "A regularizer from area X should reduce overfitting in area Y.",
     "Both areas share the same failure mode on small data.",
     "Port the regularizer and measure validation gap.", 5, 4),
    (children[0]["id"], "ignored_limitation",
     "An under-reported limitation blocks deployment",
     "Latency under load is the true bottleneck, not accuracy.",
     "Three papers footnote latency but none measure it.",
     "Profile end-to-end latency at production batch sizes.", 4, 4),
]
for node_id, itype, title, hyp, ev, exp, nov, feas in insight_rows:
    db.execute(
        "INSERT INTO insights (node_id, insight_type, title, hypothesis, evidence, experiment, "
        "novelty_score, feasibility_score, supporting_papers) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (node_id, itype, title, hyp, ev, exp, nov, feas, json.dumps(["2401.00001", "2401.00002"])),
    )
db.commit()

# Entity-relation graph summary for the parent (12 entities + 12 relations) so
# the Explore/Evidence network graphs render a real top-N subgraph.
entities = [
    {"name": "Transformer", "entity_type": "model", "paper_count": 9, "mention_count": 60},
    {"name": "BERT", "entity_type": "model", "paper_count": 7, "mention_count": 41},
    {"name": "GLUE", "entity_type": "dataset", "paper_count": 6, "mention_count": 33},
    {"name": "SQuAD", "entity_type": "dataset", "paper_count": 5, "mention_count": 28},
    {"name": "Adam", "entity_type": "method", "paper_count": 8, "mention_count": 39},
    {"name": "Dropout", "entity_type": "method", "paper_count": 6, "mention_count": 25},
    {"name": "Accuracy", "entity_type": "metric", "paper_count": 9, "mention_count": 52},
    {"name": "F1", "entity_type": "metric", "paper_count": 5, "mention_count": 21},
    {"name": "Pretraining", "entity_type": "task", "paper_count": 7, "mention_count": 30},
    {"name": "Fine-tuning", "entity_type": "task", "paper_count": 8, "mention_count": 34},
    {"name": "Attention", "entity_type": "concept", "paper_count": 9, "mention_count": 58},
    {"name": "Tokenizer", "entity_type": "artifact", "paper_count": 4, "mention_count": 17},
]
relations = [
    {"subject": "BERT", "predicate": "based_on", "object": "Transformer", "paper_count": 7, "relation_count": 12},
    {"subject": "BERT", "predicate": "evaluated_on", "object": "GLUE", "paper_count": 6, "relation_count": 10},
    {"subject": "BERT", "predicate": "evaluated_on", "object": "SQuAD", "paper_count": 5, "relation_count": 8},
    {"subject": "Transformer", "predicate": "uses", "object": "Attention", "paper_count": 9, "relation_count": 18},
    {"subject": "Transformer", "predicate": "trained_with", "object": "Adam", "paper_count": 7, "relation_count": 9},
    {"subject": "BERT", "predicate": "trained_with", "object": "Adam", "paper_count": 6, "relation_count": 7},
    {"subject": "Pretraining", "predicate": "followed_by", "object": "Fine-tuning", "paper_count": 7, "relation_count": 11},
    {"subject": "Fine-tuning", "predicate": "measured_by", "object": "Accuracy", "paper_count": 8, "relation_count": 13},
    {"subject": "GLUE", "predicate": "reports", "object": "F1", "paper_count": 4, "relation_count": 5},
    {"subject": "Transformer", "predicate": "regularized_by", "object": "Dropout", "paper_count": 6, "relation_count": 8},
    {"subject": "BERT", "predicate": "requires", "object": "Tokenizer", "paper_count": 4, "relation_count": 6},
    {"subject": "Attention", "predicate": "improves", "object": "Accuracy", "paper_count": 6, "relation_count": 7},
]
summary = {
    "top_entities": entities,
    "top_relations": relations,
    "top_entity_types": [{"entity_type": "model", "mention_count": 101}],
    "generated_from_papers": ["2401.00001"],
    "paper_count": 24,
    "entity_count": len(entities),
    "relation_count": len(relations),
}
upsert_node_graph_summary(PARENT, summary)
# also give the root a summary so the overview preview is populated
upsert_node_graph_summary("ml", summary)

from web.app import app, prewarm_stats_cache  # noqa: E402

prewarm_stats_cache()

port = int(os.environ.get("PORT", "8765"))
print(f"[fixture] db={DB_PATH} port={port} parent={PARENT} children={len(children)}", flush=True)
app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)
