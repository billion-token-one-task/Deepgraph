"""Seed a demo dataset for the agenda-driven research loop (issue #9 step 7).

Inserts:
- 8 fake deep_insights spanning topics that match / partially match / reject
  against the token_scale_v1 agenda
- 1 experiment_run with verdict='confirmed' linked to the best candidate
- 2 experimental_claims (one confirmed, one inconclusive)
- 1 manuscript_run + submission_bundle so the reviewer has a complete artifact chain
- The agenda itself (loaded from research_agendas/token_scale_v1.yaml) set active

Usage:
    DEEPGRAPH_DATABASE_URL="" DEEPGRAPH_DB_PATH=/tmp/agenda_demo.db \\
        python3 -m scripts.seed_agenda_demo

Idempotent: re-running on the same DB simply re-uses existing rows when possible.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db import database as db  # noqa: E402


DEMO_INSIGHTS = [
    # ---- strong matches (high score) ----
    {
        "tier": 2,
        "status": "verified",
        "title": "Linear attention with chunked recurrence for long context",
        "problem_statement": (
            "Quadratic attention dominates the cost of training on long context "
            "windows; existing linear approximations lose accuracy."
        ),
        "formal_structure": "y_t = sum_{s<=t} phi(q_t) phi(k_s)^T v_s",
        "adversarial_score": 7.8,
        "novelty_status": "novel",
        "resource_class": "gpu_small",
        "experimentability": "easy",
        "supporting_papers": ["2024.01234", "2024.05678"],
        "evidence_plan": {
            "main_table": {"enabled": True, "priority": "required"},
            "ablation": {"enabled": True, "priority": "required"},
            "long_context_scaling": {"enabled": True, "priority": "required"},
        },
        "predictions": [
            {
                "statement": "Throughput scales linearly in sequence length up to 32k tokens.",
                "required": True,
            }
        ],
        "signal_mix": ["scaling_laws", "long_context", "linear_attention"],
        "_pick_for_run": True,  # this one gets the experiment_run + bundle
    },
    {
        "tier": 2,
        "status": "verified",
        "title": "Structured sparsity via block-diagonal MoE routing",
        "problem_statement": (
            "Mixture-of-experts gating is sparse but routing overhead dominates "
            "at small expert counts; block-diagonal routing eliminates this."
        ),
        "adversarial_score": 7.2,
        "novelty_status": "novel",
        "resource_class": "gpu_small",
        "experimentability": "medium",
        "evidence_plan": {"main_table": {"enabled": True, "priority": "required"}},
        "signal_mix": ["mixture_of_experts", "structured_sparsity"],
    },
    {
        "tier": 1,
        "status": "verified",
        "title": "Flash attention v3 kernel fusion for sub-quadratic prefill",
        "adversarial_score": 6.9,
        "novelty_status": "novel",
        "resource_class": "gpu_small",
        "experimentability": "medium",
        "signal_mix": ["flash_attention", "sub-quadratic attention"],
    },
    # ---- weaker matches ----
    {
        "tier": 2,
        "status": "candidate",
        "title": "Scaling laws for retrieval-augmented language models",
        "problem_statement": (
            "Scaling laws for RAG remain undercharacterized at >7B parameters."
        ),
        "adversarial_score": 6.1,
        "novelty_status": "novel",
        "resource_class": "gpu_large",  # outside agenda's preferred resource class
        "experimentability": "hard",
        "signal_mix": ["scaling_laws", "retrieval"],
    },
    {
        "tier": 2,
        "status": "candidate",
        "title": "State space model with learned non-stationary kernels",
        "adversarial_score": 5.8,
        "novelty_status": "partially_exists",
        "resource_class": "gpu_small",
        "experimentability": "medium",
        "signal_mix": ["state space model"],
    },
    # ---- rejected (matches agenda.reject) ----
    {
        "tier": 2,
        "status": "candidate",
        "title": "Closed-source dataset distillation for instruction tuning",
        "problem_statement": (
            "Distillation from a proprietary corpus only — relies on closed-source "
            "dataset; included to test that the agenda's reject.keywords filter "
            "actually blocks it."
        ),
        "adversarial_score": 7.0,
        "novelty_status": "novel",
        "resource_class": "gpu_small",
        "experimentability": "easy",
        "signal_mix": ["closed-source dataset"],
    },
    {
        "tier": 2,
        "status": "rejected",  # status-based reject
        "title": "Hand-crafted prompt search for arithmetic reasoning",
        "adversarial_score": 4.5,
        "novelty_status": "exists",
        "resource_class": "cpu",
        "experimentability": "easy",
    },
    # ---- off-topic but novel (medium score) ----
    {
        "tier": 1,
        "status": "candidate",
        "title": "Mechanistic interpretability of induction heads in vision transformers",
        "adversarial_score": 6.0,
        "novelty_status": "novel",
        "resource_class": "gpu_small",
        "experimentability": "hard",
        "signal_mix": ["interpretability"],
    },
]


def _ensure_active_agenda():
    """Load the sample YAML and persist it as the active agenda."""
    from agents import agenda_loader

    existing = agenda_loader.get_active_agenda()
    if existing and existing.name == "token_scale_efficiency_v1":
        return existing
    path = REPO_ROOT / "research_agendas" / "token_scale_v1.yaml"
    agenda = agenda_loader.load_agenda_from_file(path)
    aid = agenda_loader.save_agenda(agenda)
    agenda.agenda_id = aid
    agenda_loader.set_active_agenda(aid)
    return agenda


def _insert_insight(row: dict) -> int:
    """Insert a fake deep_insight; return its id."""
    payload = {
        "tier": row["tier"],
        "status": row.get("status", "candidate"),
        "title": row["title"],
        "problem_statement": row.get("problem_statement"),
        "formal_structure": row.get("formal_structure"),
        "adversarial_score": row.get("adversarial_score"),
        "novelty_status": row.get("novelty_status"),
        "resource_class": row.get("resource_class", "cpu"),
        "experimentability": row.get("experimentability", "medium"),
        "supporting_papers": json.dumps(row.get("supporting_papers", [])),
        "evidence_plan": json.dumps(row.get("evidence_plan", {})),
        "predictions": json.dumps(row.get("predictions", [])),
        "signal_mix": json.dumps(row.get("signal_mix", [])),
    }
    existing = db.fetchone(
        "SELECT id FROM deep_insights WHERE title=?", (payload["title"],)
    )
    if existing:
        return int(existing["id"])
    new_id = db.insert_returning_id(
        """
        INSERT INTO deep_insights
            (tier, status, title, problem_statement, formal_structure,
             adversarial_score, novelty_status, resource_class,
             experimentability, supporting_papers, evidence_plan,
             predictions, signal_mix)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING id
        """,
        (
            payload["tier"], payload["status"], payload["title"],
            payload["problem_statement"], payload["formal_structure"],
            payload["adversarial_score"], payload["novelty_status"],
            payload["resource_class"], payload["experimentability"],
            payload["supporting_papers"], payload["evidence_plan"],
            payload["predictions"], payload["signal_mix"],
        ),
    )
    return int(new_id)


def _ensure_full_run_for(insight_id: int) -> dict:
    """Make sure there's an experiment_run + claims + manuscript + bundle for insight_id."""
    exp = db.fetchone(
        "SELECT id, submission_bundle_id FROM experiment_runs WHERE deep_insight_id=? ORDER BY id DESC LIMIT 1",
        (insight_id,),
    )
    if exp:
        exp_id = int(exp["id"])
    else:
        exp_id = int(db.insert_returning_id(
            """
            INSERT INTO experiment_runs
                (deep_insight_id, status, phase, hypothesis_verdict,
                 baseline_metric_name, baseline_metric_value,
                 best_metric_value, effect_size, effect_pct, workdir)
            VALUES (?, 'completed', 'hypothesis_testing', 'confirmed',
                    'accuracy', 0.652, 0.781, 0.129, 19.78, '/tmp/agenda_demo/exp')
            RETURNING id
            """,
            (insight_id,),
        ))

    # claims
    existing_claims = db.fetchall(
        "SELECT id FROM experimental_claims WHERE run_id=?", (exp_id,)
    )
    if not existing_claims:
        db.execute(
            """
            INSERT INTO experimental_claims
                (run_id, deep_insight_id, claim_text, claim_type,
                 verdict, effect_size, confidence, p_value)
            VALUES (?, ?, ?, 'experimental', 'confirmed', 0.129, 0.94, 0.008)
            """,
            (
                exp_id, insight_id,
                "Linear attention with chunked recurrence improves long-context "
                "accuracy by 19.78% (effect_size=0.129) vs flash-attention baseline.",
            ),
        )
        db.execute(
            """
            INSERT INTO experimental_claims
                (run_id, deep_insight_id, claim_text, claim_type,
                 verdict, effect_size, confidence, p_value)
            VALUES (?, ?, ?, 'experimental', 'inconclusive', 0.022, 0.55, 0.31)
            """,
            (
                exp_id, insight_id,
                "Long-context scaling beyond 32k tokens: effect attenuates and "
                "becomes statistically inconclusive.",
            ),
        )

    # manuscript_run + bundle
    manu = db.fetchone(
        "SELECT id FROM manuscript_runs WHERE deep_insight_id=? ORDER BY id DESC LIMIT 1",
        (insight_id,),
    )
    if manu:
        manu_id = int(manu["id"])
    else:
        manu_id = int(db.insert_returning_id(
            """
            INSERT INTO manuscript_runs
                (experiment_run_id, deep_insight_id, status, workdir)
            VALUES (?, ?, 'bundle_ready', '/tmp/agenda_demo/manu')
            RETURNING id
            """,
            (exp_id, insight_id),
        ))

    bundle = db.fetchone(
        "SELECT id FROM submission_bundles WHERE manuscript_run_id=? ORDER BY id DESC LIMIT 1",
        (manu_id,),
    )
    if bundle:
        bundle_id = int(bundle["id"])
    else:
        bundle_id = int(db.insert_returning_id(
            """
            INSERT INTO submission_bundles
                (manuscript_run_id, bundle_format, status, bundle_path, manifest_path)
            VALUES (?, 'conference', 'ready',
                    '/tmp/agenda_demo/bundle.zip',
                    '/tmp/agenda_demo/manifest.json')
            RETURNING id
            """,
            (manu_id,),
        ))
        db.execute(
            "UPDATE experiment_runs SET submission_bundle_id=? WHERE id=?",
            (bundle_id, exp_id),
        )

    return {"experiment_run_id": exp_id, "manuscript_run_id": manu_id, "submission_bundle_id": bundle_id}


def main():
    db.init_db()
    agenda = _ensure_active_agenda()
    print(f"[agenda] active: #{agenda.agenda_id} {agenda.name}")

    pick_for_run_id = None
    for row in DEMO_INSIGHTS:
        iid = _insert_insight(row)
        marker = " *" if row.get("_pick_for_run") else ""
        print(f"[insight #{iid}]{marker} {row['title']}")
        if row.get("_pick_for_run"):
            pick_for_run_id = iid

    if pick_for_run_id is not None:
        artifacts = _ensure_full_run_for(pick_for_run_id)
        print(f"[run] for insight #{pick_for_run_id}: {artifacts}")

    db.commit()
    print("\n[done] Now run the selector + dispatch from the UI or via:")
    print(
        "   curl -X POST http://localhost:5000/api/research_agenda/select "
        "-H 'Content-Type: application/json' -d '{\"dispatch_mode\":\"auto\"}'"
    )


if __name__ == "__main__":
    main()
