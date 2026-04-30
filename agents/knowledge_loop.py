"""Knowledge Loop: feed experimental results back into the knowledge graph.

Three functions:
  4a. Cascade Reasoning — when a hypothesis is confirmed/refuted, update related claims
  4b. Track Record — record outcomes per signal type for meta-learning
  4c. Trigger New Hypotheses — refutations and confirmations seed new discoveries
"""
import json
from db import database as db


def cascade_from_claim(claim_id: int):
    """Process an experimental claim and cascade its effects through the KG.

    When confirmed: boost related hypotheses, create new claims
    When refuted: weaken related claims, generate refutation insights
    """
    claim = db.fetchone("SELECT * FROM experimental_claims WHERE id=?", (claim_id,))
    if not claim or claim.get("cascaded"):
        return

    verdict = claim["verdict"]
    insight_id = claim["deep_insight_id"]
    run_id = claim["run_id"]
    effect_size = claim.get("effect_size", 0) or 0

    insight = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (insight_id,))
    if not insight:
        return

    source_nodes = []
    try:
        source_nodes = json.loads(insight.get("source_node_ids") or "[]")
    except (json.JSONDecodeError, TypeError):
        pass
    source_nodes = _existing_source_nodes(source_nodes)

    supporting = []
    try:
        supporting = json.loads(insight.get("supporting_papers") or "[]")
    except (json.JSONDecodeError, TypeError):
        pass

    if verdict == "confirmed":
        _cascade_confirmation(insight, source_nodes, supporting, effect_size)
    elif verdict == "refuted":
        _cascade_refutation(insight, source_nodes, supporting)

    db.execute("UPDATE experimental_claims SET cascaded=1 WHERE id=?", (claim_id,))
    db.commit()


def _existing_source_nodes(source_nodes: list) -> list[str]:
    """Keep only taxonomy nodes that exist, preserving order."""
    valid = []
    seen = set()
    for node_id in source_nodes:
        node_id = str(node_id)
        if node_id in seen:
            continue
        if db.fetchone("SELECT id FROM taxonomy_nodes WHERE id=?", (node_id,)):
            valid.append(node_id)
            seen.add(node_id)
    return valid


def _cascade_confirmation(insight: dict, source_nodes: list, supporting: list, effect_size: float):
    """When a hypothesis is confirmed, strengthen related knowledge."""
    insight_id = insight["id"]

    related = db.fetchall("""
        SELECT id, title, source_node_ids, status, adversarial_score
        FROM deep_insights
        WHERE id != ? AND status IN ('candidate', 'verified', 'forged')
    """, (insight_id,))

    for rel in related:
        try:
            rel_nodes = json.loads(rel.get("source_node_ids") or "[]")
        except (json.JSONDecodeError, TypeError):
            rel_nodes = []

        shared = set(source_nodes) & set(rel_nodes)
        if shared:
            current_score = rel.get("adversarial_score") or 5.0
            boost = min(1.0, effect_size * 0.5)
            new_score = min(10.0, current_score + boost)
            db.execute(
                "UPDATE deep_insights SET adversarial_score=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (new_score, rel["id"]))

    for node_id in source_nodes:
        existing = db.fetchall(
            "SELECT id FROM node_opportunities WHERE node_id=? AND title LIKE '%confirmed%' LIMIT 1",
            (node_id,))
        if not existing:
            db.execute(
                """INSERT INTO node_opportunities
                   (node_id, opportunity_type, title, description, why_now, value_score, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (node_id, "experimental_confirmation",
                 f"Experimentally confirmed: {insight['title'][:100]}",
                 f"SciForge validated this hypothesis with effect size {effect_size:.4f}.",
                 "Automated experimental validation passed",
                 min(5.0, 3.0 + abs(effect_size) * 2), 0.9))

    db.commit()
    print(f"[CASCADE] Confirmation cascaded for insight {insight_id}: "
          f"{len(source_nodes)} nodes updated", flush=True)


def _cascade_refutation(insight: dict, source_nodes: list, supporting: list):
    """When a hypothesis is refuted, weaken related knowledge and seed new questions."""
    insight_id = insight["id"]

    related = db.fetchall("""
        SELECT id, title, source_node_ids, adversarial_score
        FROM deep_insights
        WHERE id != ? AND status IN ('candidate', 'verified', 'forged')
    """, (insight_id,))

    for rel in related:
        try:
            rel_nodes = json.loads(rel.get("source_node_ids") or "[]")
        except (json.JSONDecodeError, TypeError):
            rel_nodes = []

        shared = set(source_nodes) & set(rel_nodes)
        if shared:
            current_score = rel.get("adversarial_score") or 5.0
            penalty = min(1.0, 0.3 * len(shared))
            new_score = max(0.0, current_score - penalty)
            db.execute(
                "UPDATE deep_insights SET adversarial_score=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (new_score, rel["id"]))

    db.commit()
    print(f"[CASCADE] Refutation cascaded for insight {insight_id}: "
          f"related hypotheses penalized", flush=True)


def update_track_record(run_id: int):
    """Update the discovery_track_record table based on a completed run."""
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not run:
        return

    verdict = run.get("hypothesis_verdict")
    if not verdict or verdict not in ("confirmed", "refuted", "inconclusive"):
        return

    insight_id = run["deep_insight_id"]
    insight = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (insight_id,))
    if not insight:
        return

    signal_type = _infer_signal_type(insight)
    effect = abs(run.get("effect_size") or 0)

    existing = db.fetchone(
        "SELECT * FROM discovery_track_record WHERE signal_type=?", (signal_type,))

    if existing:
        new_total = existing["hypothesis_count"] + 1
        new_confirmed = existing["confirmed_count"] + (1 if verdict == "confirmed" else 0)
        new_refuted = existing["refuted_count"] + (1 if verdict == "refuted" else 0)
        new_inconclusive = existing["inconclusive_count"] + (1 if verdict == "inconclusive" else 0)

        prev_effect_sum = (existing["avg_effect_size"] or 0) * existing["hypothesis_count"]
        new_avg_effect = (prev_effect_sum + effect) / new_total

        decided = new_confirmed + new_refuted
        hit_rate = new_confirmed / decided if decided > 0 else 0

        db.execute(
            """UPDATE discovery_track_record
               SET hypothesis_count=?, confirmed_count=?, refuted_count=?,
                   inconclusive_count=?, avg_effect_size=?, hit_rate=?,
                   last_updated=CURRENT_TIMESTAMP
               WHERE signal_type=?""",
            (new_total, new_confirmed, new_refuted, new_inconclusive,
             new_avg_effect, hit_rate, signal_type))
    else:
        hit_rate = 1.0 if verdict == "confirmed" else 0.0
        db.execute(
            """INSERT INTO discovery_track_record
               (signal_type, hypothesis_count, confirmed_count, refuted_count,
                inconclusive_count, avg_effect_size, hit_rate)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (signal_type, 1,
             1 if verdict == "confirmed" else 0,
             1 if verdict == "refuted" else 0,
             1 if verdict == "inconclusive" else 0,
             effect, hit_rate))

    db.commit()
    print(f"[TRACK] Updated track record: {signal_type} -> {verdict} "
          f"(effect={effect:.6f})", flush=True)


def _infer_signal_type(insight: dict) -> str:
    """Infer the signal type that generated this insight.

    Uses tier, existing_weakness, evidence_summary, and problem_statement
    to classify what signal originally triggered this hypothesis.
    """
    evidence = " ".join(filter(None, [
        (insight.get("evidence_summary") or ""),
        (insight.get("existing_weakness") or ""),
        (insight.get("problem_statement") or ""),
    ])).lower()

    if insight.get("tier") == 1:
        if "pattern" in evidence or "convergent" in evidence:
            return "pattern_match"
        return "entity_overlap"

    contradiction_kw = ["contradiction", "conflict", "disagree", "inconsistent"]
    if any(kw in evidence for kw in contradiction_kw):
        return "contradiction_cluster"

    plateau_kw = ["plateau", "converge", "diminish", "saturat", "within 1%", "spread"]
    if any(kw in evidence for kw in plateau_kw):
        return "plateau"

    overlap_kw = ["overlap", "shared entit", "cross-node", "cross-domain"]
    if any(kw in evidence for kw in overlap_kw):
        return "entity_overlap"

    pattern_kw = ["pattern", "convergent", "recur"]
    if any(kw in evidence for kw in pattern_kw):
        return "pattern_match"

    return "insight_derived"


def process_completed_run(run_id: int):
    """Full post-experiment processing: interpret, cascade, update track record.

    This is the main entry point called after a validation loop completes.
    """
    from agents.result_interpreter import interpret_run

    print(f"[KLOOP] Processing completed run {run_id}...", flush=True)

    result = interpret_run(run_id)
    if result.get("status") == "error":
        print(f"[KLOOP] Run {run_id} interpretation failed: {result.get('reason')}", flush=True)
        return result

    verdict = result.get("verdict", "inconclusive")

    claims = db.fetchall(
        "SELECT id FROM experimental_claims WHERE run_id=? AND cascaded=0",
        (run_id,))
    for claim in claims:
        cascade_from_claim(claim["id"])

    update_track_record(run_id)

    print(f"[KLOOP] Run {run_id} fully processed: verdict={verdict}", flush=True)
    return result
