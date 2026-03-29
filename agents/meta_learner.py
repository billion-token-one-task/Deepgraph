"""Meta-Learner: self-improvement through accumulated experimental history.

After 20+ experimental outcomes, the system learns:
- Which signal types (contradiction clusters, entity overlaps, etc.) yield confirmed discoveries
- Which taxonomy nodes have the most "low-hanging fruit"
- Whether adversarial scores predict confirmation
- How to re-weight the Signal Harvester for better hit rates
"""
import json
from db import database as db


def get_track_record_summary() -> dict:
    """Get the current track record across all signal types."""
    records = db.fetchall(
        "SELECT * FROM discovery_track_record ORDER BY hit_rate DESC")

    total_hypotheses = sum(r["hypothesis_count"] for r in records)
    total_confirmed = sum(r["confirmed_count"] for r in records)
    total_refuted = sum(r["refuted_count"] for r in records)
    total_inconclusive = sum(r["inconclusive_count"] for r in records)

    overall_hit_rate = total_confirmed / max(total_confirmed + total_refuted, 1)

    return {
        "signal_types": [dict(r) for r in records],
        "total_hypotheses": total_hypotheses,
        "total_confirmed": total_confirmed,
        "total_refuted": total_refuted,
        "total_inconclusive": total_inconclusive,
        "overall_hit_rate": round(overall_hit_rate, 4),
    }


def get_node_hit_rates() -> list[dict]:
    """Analyze which taxonomy nodes produce the most confirmed hypotheses."""
    rows = db.fetchall("""
        SELECT
            json_each.value as node_id,
            COUNT(*) as total,
            SUM(CASE WHEN er.hypothesis_verdict='confirmed' THEN 1 ELSE 0 END) as confirmed,
            SUM(CASE WHEN er.hypothesis_verdict='refuted' THEN 1 ELSE 0 END) as refuted,
            AVG(ABS(COALESCE(er.effect_size, 0))) as avg_effect
        FROM experiment_runs er
        JOIN deep_insights di ON er.deep_insight_id = di.id
        CROSS JOIN json_each(di.source_node_ids)
        WHERE er.hypothesis_verdict IS NOT NULL
        GROUP BY json_each.value
        HAVING total >= 2
        ORDER BY confirmed DESC, avg_effect DESC
    """)
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


def get_adversarial_calibration() -> dict:
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
            SUM(CASE WHEN er.hypothesis_verdict='confirmed' THEN 1 ELSE 0 END) as confirmed,
            SUM(CASE WHEN er.hypothesis_verdict='refuted' THEN 1 ELSE 0 END) as refuted
        FROM experiment_runs er
        JOIN deep_insights di ON er.deep_insight_id = di.id
        WHERE er.hypothesis_verdict IS NOT NULL
        GROUP BY bucket
        ORDER BY bucket
    """)

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


def get_method_type_analysis() -> list[dict]:
    """Analyze which method types (loss_function, architecture, etc.) work best."""
    rows = db.fetchall("""
        SELECT
            di.proposed_method,
            er.hypothesis_verdict,
            er.effect_size
        FROM experiment_runs er
        JOIN deep_insights di ON er.deep_insight_id = di.id
        WHERE er.hypothesis_verdict IS NOT NULL AND di.tier = 2
    """)

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
        verdict = r["hypothesis_verdict"]
        if verdict == "confirmed":
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


def compute_signal_weights() -> dict[str, float]:
    """Compute priority weights for each signal type based on track record.

    Returns a dict of signal_type -> weight (higher = should be prioritized).
    Default weight is 1.0 for types with no track record.
    """
    records = db.fetchall("SELECT * FROM discovery_track_record")
    if not records:
        return {}

    weights = {}
    overall_hit = sum(r["confirmed_count"] for r in records) / max(
        sum(r["confirmed_count"] + r["refuted_count"] for r in records), 1)

    for r in records:
        decided = r["confirmed_count"] + r["refuted_count"]
        if decided < 3:
            weights[r["signal_type"]] = 1.0
            continue

        hit_rate = r["hit_rate"] or 0
        avg_effect = r["avg_effect_size"] or 0

        weight = (hit_rate / max(overall_hit, 0.01)) * (1.0 + avg_effect)
        weight = max(0.1, min(5.0, weight))
        weights[r["signal_type"]] = round(weight, 3)

    return weights


def get_full_meta_report() -> dict:
    """Generate a comprehensive meta-learning report."""
    total_runs = db.fetchone("SELECT COUNT(*) as c FROM experiment_runs WHERE hypothesis_verdict IS NOT NULL")
    total = total_runs["c"] if total_runs else 0

    if total < 1:
        return {
            "status": "insufficient_data",
            "message": f"Need at least 1 completed experiment (have {total})",
            "total_experiments": total,
        }

    return {
        "status": "ready",
        "total_experiments": total,
        "track_record": get_track_record_summary(),
        "node_hit_rates": get_node_hit_rates(),
        "adversarial_calibration": get_adversarial_calibration(),
        "method_type_analysis": get_method_type_analysis(),
        "signal_weights": compute_signal_weights(),
    }
