"""Meta-Learner: self-improvement through accumulated experimental history.

After 20+ experimental outcomes, the system learns:
- Which signal types (contradiction clusters, entity overlaps, etc.) yield confirmed discoveries
- Which taxonomy nodes have the most "low-hanging fruit"
- Whether adversarial scores predict confirmation
- How to re-weight the Signal Harvester for better hit rates
"""
import json
from db import database as db


def get_track_record_summary(agenda_id: int) -> dict:
    """Get an agenda-local track record from authority-gated signal outcomes."""
    records = db.fetchall(
        """
        SELECT signal_table AS signal_type,
               COUNT(*) AS hypothesis_count,
               SUM(CASE WHEN verdict='supported' THEN 1 ELSE 0 END)
                   AS confirmed_count,
               SUM(CASE WHEN verdict='refuted' THEN 1 ELSE 0 END)
                   AS refuted_count,
               SUM(CASE WHEN verdict='inconclusive' THEN 1 ELSE 0 END)
                   AS inconclusive_count
        FROM agenda_signal_outcomes
        WHERE agenda_id=?
        GROUP BY signal_table
        ORDER BY confirmed_count DESC, signal_table
        """,
        (agenda_id,),
    )

    normalized = []
    for row in records:
        item = dict(row)
        decided = int(item.get("confirmed_count") or 0) + int(
            item.get("refuted_count") or 0
        )
        item["hit_rate"] = round(
            int(item.get("confirmed_count") or 0) / max(decided, 1),
            4,
        )
        normalized.append(item)
    total_hypotheses = sum(int(r.get("hypothesis_count") or 0) for r in normalized)
    total_confirmed = sum(int(r.get("confirmed_count") or 0) for r in normalized)
    total_refuted = sum(int(r.get("refuted_count") or 0) for r in normalized)
    total_inconclusive = sum(
        int(r.get("inconclusive_count") or 0) for r in normalized
    )

    overall_hit_rate = total_confirmed / max(total_confirmed + total_refuted, 1)

    return {
        "agenda_id": agenda_id,
        "signal_types": normalized,
        "total_hypotheses": total_hypotheses,
        "total_confirmed": total_confirmed,
        "total_refuted": total_refuted,
        "total_inconclusive": total_inconclusive,
        "overall_hit_rate": round(overall_hit_rate, 4),
    }


def get_node_hit_rates(agenda_id: int) -> list[dict]:
    """Analyze which taxonomy nodes produce the most confirmed hypotheses."""
    if db.use_postgres():
        sql = """
            SELECT
                node_items.node_id AS node_id,
                COUNT(*) as total,
                SUM(CASE WHEN sdr.verdict='supported' THEN 1 ELSE 0 END) as confirmed,
                SUM(CASE WHEN sdr.verdict='refuted' THEN 1 ELSE 0 END) as refuted,
                AVG(ABS(COALESCE(er.effect_size, 0))) as avg_effect
            FROM experiment_runs er
            JOIN deep_insights di ON er.deep_insight_id = di.id
            JOIN scientific_decision_records sdr
              ON sdr.experiment_run_id=er.id AND sdr.agenda_id=er.agenda_id
            CROSS JOIN LATERAL jsonb_array_elements_text(COALESCE(di.source_node_ids, '[]')::jsonb) AS node_items(node_id)
            WHERE er.agenda_id=?
            GROUP BY node_items.node_id
            HAVING COUNT(*) >= 2
            ORDER BY confirmed DESC, avg_effect DESC
        """
    else:
        sql = """
            SELECT
                json_each.value as node_id,
                COUNT(*) as total,
                SUM(CASE WHEN sdr.verdict='supported' THEN 1 ELSE 0 END) as confirmed,
                SUM(CASE WHEN sdr.verdict='refuted' THEN 1 ELSE 0 END) as refuted,
                AVG(ABS(COALESCE(er.effect_size, 0))) as avg_effect
            FROM experiment_runs er
            JOIN deep_insights di ON er.deep_insight_id = di.id
            JOIN scientific_decision_records sdr
              ON sdr.experiment_run_id=er.id AND sdr.agenda_id=er.agenda_id
            CROSS JOIN json_each(di.source_node_ids)
            WHERE er.agenda_id=?
            GROUP BY json_each.value
            HAVING COUNT(*) >= 2
            ORDER BY confirmed DESC, avg_effect DESC
        """
    rows = db.fetchall(sql, (agenda_id,))
    result = []
    for r in rows:
        decided = (r["confirmed"] or 0) + (r["refuted"] or 0)
        hit_rate = (r["confirmed"] or 0) / max(decided, 1)
        result.append({
            "node_id": r["node_id"],
            "total": r["total"],
            "confirmed": r["confirmed"] or 0,
            "refuted": r["refuted"] or 0,
            "avg_effect": round(r["avg_effect"] or 0, 6),
            "hit_rate": round(hit_rate, 4),
        })
    return result


def get_adversarial_calibration(agenda_id: int) -> dict:
    """Check if adversarial scores actually predict experimental confirmation."""
    rows = db.fetchall("""
        SELECT
            CASE
                WHEN di.adversarial_score >= 7 THEN 'high_7+'
                WHEN di.adversarial_score >= 5 THEN 'mid_5-6'
                WHEN di.adversarial_score > 0 THEN 'low_1-4'
                ELSE 'unscored'
            END as bucket,
            COUNT(*) as total,
            SUM(CASE WHEN sdr.verdict='supported' THEN 1 ELSE 0 END) as confirmed,
            SUM(CASE WHEN sdr.verdict='refuted' THEN 1 ELSE 0 END) as refuted
        FROM experiment_runs er
        JOIN deep_insights di ON er.deep_insight_id = di.id
        JOIN scientific_decision_records sdr
          ON sdr.experiment_run_id=er.id AND sdr.agenda_id=er.agenda_id
        WHERE er.agenda_id=?
        GROUP BY bucket
        ORDER BY bucket
    """, (agenda_id,))

    buckets = {}
    for r in rows:
        decided = (r["confirmed"] or 0) + (r["refuted"] or 0)
        hit_rate = (r["confirmed"] or 0) / max(decided, 1)
        buckets[r["bucket"]] = {
            "total": r["total"],
            "confirmed": r["confirmed"] or 0,
            "refuted": r["refuted"] or 0,
            "hit_rate": round(hit_rate, 4),
        }
    return buckets


def get_method_type_analysis(agenda_id: int) -> list[dict]:
    """Analyze which method types (loss_function, architecture, etc.) work best."""
    rows = db.fetchall("""
        SELECT
            di.proposed_method,
            sdr.verdict AS scientific_verdict,
            er.effect_size
        FROM experiment_runs er
        JOIN deep_insights di ON er.deep_insight_id = di.id
        JOIN scientific_decision_records sdr
          ON sdr.experiment_run_id=er.id AND sdr.agenda_id=er.agenda_id
        WHERE er.agenda_id=? AND di.tier = 2
    """, (agenda_id,))

    type_stats = {}
    for r in rows:
        try:
            method = json.loads(r.get("proposed_method") or "{}")
        except (json.JSONDecodeError, TypeError):
            method = {}
        mtype = method.get("type", "unknown")
        if mtype not in type_stats:
            type_stats[mtype] = {"total": 0, "confirmed": 0, "refuted": 0, "effects": []}
        type_stats[mtype]["total"] += 1
        verdict = r["scientific_verdict"]
        if verdict == "supported":
            type_stats[mtype]["confirmed"] += 1
        elif verdict == "refuted":
            type_stats[mtype]["refuted"] += 1
        if r.get("effect_size"):
            type_stats[mtype]["effects"].append(abs(r["effect_size"]))

    result = []
    for mtype, stats in type_stats.items():
        decided = stats["confirmed"] + stats["refuted"]
        result.append({
            "method_type": mtype,
            "total": stats["total"],
            "confirmed": stats["confirmed"],
            "refuted": stats["refuted"],
            "hit_rate": round(stats["confirmed"] / max(decided, 1), 4),
            "avg_effect": round(sum(stats["effects"]) / max(len(stats["effects"]), 1), 6),
        })
    result.sort(key=lambda x: x["hit_rate"], reverse=True)
    return result


def compute_signal_weights(agenda_id: int | None = None) -> dict[str, float]:
    """Compute transparent agenda-local weights from audited outcomes only.

    A missing agenda deliberately returns no learned weights. The legacy
    global ``discovery_track_record`` must not leak one agenda's outcomes into
    another agenda's ranking policy.
    """
    try:
        agenda_id = int(agenda_id or 0)
    except (TypeError, ValueError):
        agenda_id = 0
    if agenda_id <= 0:
        return {}
    rows = db.fetchall(
        """
        SELECT signal_table,
               SUM(CASE WHEN verdict='supported' THEN 1 ELSE 0 END)
                   AS supported,
               SUM(CASE WHEN verdict='refuted' THEN 1 ELSE 0 END)
                   AS refuted
        FROM agenda_signal_outcomes
        WHERE agenda_id=?
        GROUP BY signal_table
        """,
        (agenda_id,),
    )
    weights: dict[str, float] = {}
    for row in rows:
        supported = int(row.get("supported") or 0)
        refuted = int(row.get("refuted") or 0)
        total = supported + refuted
        # Beta(1,1) smoothing and gradual trust prevent tiny samples from
        # dominating general search.
        posterior = (supported + 1.0) / (total + 2.0)
        trust = min(1.0, total / 10.0)
        weight = max(0.5, min(1.5, 1.0 + (posterior - 0.5) * 2.0 * trust))
        table = str(row.get("signal_table") or "")
        if table:
            weights[table] = round(weight, 3)
            weights[table.rstrip("s")] = round(weight, 3)
    return weights


def get_full_meta_report(agenda_id: int | None = None) -> dict:
    """Generate an agenda-local report from canonical scientific decisions."""
    try:
        agenda_id = int(agenda_id or 0)
    except (TypeError, ValueError):
        agenda_id = 0
    if agenda_id <= 0:
        return {
            "status": "agenda_required",
            "message": "An explicit agenda_id is required for meta-learning.",
            "total_experiments": 0,
        }
    total_runs = db.fetchone(
        """
        SELECT COUNT(*) as c
        FROM scientific_decision_records
        WHERE agenda_id=?
        """,
        (agenda_id,),
    )
    total = total_runs["c"] if total_runs else 0

    if total < 1:
        return {
            "status": "insufficient_data",
            "message": f"Need at least 1 completed experiment (have {total})",
            "total_experiments": total,
        }

    return {
        "status": "ready",
        "agenda_id": agenda_id,
        "total_experiments": total,
        "track_record": get_track_record_summary(agenda_id),
        "node_hit_rates": get_node_hit_rates(agenda_id),
        "adversarial_calibration": get_adversarial_calibration(agenda_id),
        "method_type_analysis": get_method_type_analysis(agenda_id),
        "signal_weights": compute_signal_weights(agenda_id),
    }
