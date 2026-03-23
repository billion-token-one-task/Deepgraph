"""Deterministic opportunity scoring using results, graph relations, and evidence signals."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from db import database as db
from db import evidence_graph as graph


TRIAGE_BANDS = (
    (4.4, "high"),
    (3.1, "medium"),
    (0.0, "watchlist"),
)


def _clamp_score(value: float) -> float:
    return round(max(1.0, min(5.0, value)), 1)


def _short_title(prefix: str, text: str, max_words: int = 6) -> str:
    words = re.findall(r"[A-Za-z0-9][^,.;:!?]*", text.strip())
    if not words:
        return prefix
    head = " ".join(text.strip().split()[:max_words]).rstrip(",.;:!?")
    return f"{prefix}: {head}"


def _normalize_band(score: float) -> str:
    for threshold, band in TRIAGE_BANDS:
        if score >= threshold:
            return band
    return "watchlist"


def score_coverage_imbalance(method_count: int, dataset_count: int) -> float:
    """Score how badly benchmark coverage lags behind method variety."""
    if method_count <= 1:
        return 1.0
    effective_datasets = max(dataset_count, 1)
    ratio = method_count / effective_datasets
    return _clamp_score(1.4 + min(3.4, ratio * 0.6))


def score_metric_diversity(method_count: int, metric_count: int) -> float:
    """Score how narrow metric diversity is for an active area."""
    if method_count <= 2:
        return 1.0
    missing_diversity = max(0, method_count - max(metric_count, 1))
    return _clamp_score(1.8 + min(3.0, missing_diversity * 0.22))


def _fetch_paper_signals(node_id: str) -> list[dict]:
    return db.fetchall(
        """SELECT p.id AS paper_id, pi.limitations, pi.open_questions
           FROM papers p
           JOIN paper_taxonomy pt ON p.id = pt.paper_id
           JOIN taxonomy_nodes t ON pt.node_id = t.id
           JOIN paper_insights pi ON pi.paper_id = p.id
           WHERE t.id = ? OR t.id LIKE ? || '.%'
           GROUP BY p.id
           ORDER BY p.published_date DESC""",
        (node_id, node_id),
    )


def _count_text_clusters(rows: list[dict], field: str) -> dict[str, dict]:
    buckets: dict[str, dict] = {}
    for row in rows:
        for text in db._load_json(row.get(field), []):
            cleaned = (text or "").strip()
            if not cleaned:
                continue
            bucket = buckets.setdefault(cleaned, {"count": 0, "paper_ids": []})
            bucket["count"] += 1
            if row["paper_id"] not in bucket["paper_ids"]:
                bucket["paper_ids"].append(row["paper_id"])
    return buckets


def _fetch_contradiction_signals(node_id: str) -> list[dict]:
    return db.fetchall(
        """SELECT c.id, c.description, c.hypothesis, ca.paper_id AS paper_a, cb.paper_id AS paper_b,
                  ca.method_name, ca.dataset_name, ca.metric_name
           FROM contradictions c
           LEFT JOIN claims ca ON ca.id = c.claim_a_id
           LEFT JOIN claims cb ON cb.id = c.claim_b_id
           LEFT JOIN paper_taxonomy pta ON pta.paper_id = ca.paper_id
           LEFT JOIN paper_taxonomy ptb ON ptb.paper_id = cb.paper_id
           WHERE (pta.node_id = ? OR pta.node_id LIKE ? || '.%')
              OR (ptb.node_id = ? OR ptb.node_id LIKE ? || '.%')
           ORDER BY c.id DESC""",
        (node_id, node_id, node_id, node_id),
    )


def _fetch_entity_type_counts(node_id: str) -> Counter:
    rows = db.fetchall(
        """SELECT ge.entity_type, COUNT(DISTINCT er.canonical_entity_id) AS c
           FROM paper_entity_mentions pem
           JOIN entity_resolutions er ON er.entity_id = pem.entity_id
           JOIN graph_entities ge ON ge.id = er.canonical_entity_id
           JOIN taxonomy_nodes t ON pem.node_id = t.id
           WHERE t.id = ? OR t.id LIKE ? || '.%'
           GROUP BY ge.entity_type""",
        (node_id, node_id),
    )
    counter = Counter()
    for row in rows:
        counter[row["entity_type"]] = row["c"]
    return counter


def _fetch_relation_counts(node_id: str) -> Counter:
    rows = db.fetchall(
        """SELECT gr.predicate, COUNT(DISTINCT gr.id) AS c
           FROM graph_relations gr
           JOIN taxonomy_nodes t ON gr.node_id = t.id
           WHERE t.id = ? OR t.id LIKE ? || '.%'
           GROUP BY gr.predicate""",
        (node_id, node_id),
    )
    counter = Counter()
    for row in rows:
        counter[row["predicate"]] = row["c"]
    return counter


def _dedupe_opportunities(opportunities: list[dict]) -> list[dict]:
    seen = set()
    ordered = []
    for opportunity in sorted(opportunities, key=lambda item: (-item["value_score"], -item["confidence"], item["title"])):
        key = (opportunity["opportunity_type"], opportunity["title"])
        if key in seen:
            continue
        seen.add(key)
        ordered.append(opportunity)
    return ordered


def get_node_opportunities(node_id: str) -> list[dict]:
    rows = db.fetchall(
        """SELECT * FROM node_opportunities
           WHERE node_id=?
           ORDER BY value_score DESC, confidence DESC, id DESC""",
        (node_id,),
    )
    for row in rows:
        row["signal_counts"] = db._load_json(row.get("signal_counts"), {})
        row["evidence_paper_ids"] = db._load_json(row.get("evidence_paper_ids"), [])
    return rows


def replace_node_opportunities(node_id: str, opportunities: list[dict]) -> list[dict]:
    db.execute("DELETE FROM opportunity_triage WHERE node_id=?", (node_id,))
    db.execute("DELETE FROM node_opportunities WHERE node_id=?", (node_id,))
    for opportunity in opportunities:
        db.execute(
            """INSERT INTO node_opportunities
               (node_id, opportunity_type, title, description, why_now, value_score, confidence, signal_counts, evidence_paper_ids)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                node_id,
                opportunity["opportunity_type"],
                opportunity["title"],
                opportunity["description"],
                opportunity.get("why_now"),
                opportunity.get("value_score", 0),
                opportunity.get("confidence", 0),
                db._dump_json(opportunity.get("signal_counts", {})),
                db._dump_json(opportunity.get("evidence_paper_ids", [])),
            ),
        )
    # Force the plain-language summary to rebuild on next access with the updated opportunities.
    db.execute("DELETE FROM node_summaries WHERE node_id=?", (node_id,))
    db.commit()
    return get_node_opportunities(node_id)


def build_node_opportunities(node_id: str) -> list[dict]:
    """Generate richer opportunity themes from matrix gaps, graph structure, and paper signals."""
    from db import taxonomy as tax

    opportunities: list[dict] = []
    graph_summary = graph.ensure_node_graph_summary(node_id) or {}
    matrix_gaps = tax.get_node_gaps(node_id)
    for gap in matrix_gaps[:3]:
        base_value = float(gap.get("value_score") or 3.0)
        opportunities.append({
            "opportunity_type": "evaluation_gap",
            "title": f"Test {gap['method_name']} on {gap['dataset_name']}",
            "description": gap["gap_description"],
            "why_now": gap.get("research_proposal") or (
                f"{gap['method_name']} and {gap['dataset_name']} both appear in this area, "
                "but that combination still lacks evidence."
            ),
            "value_score": _clamp_score(2.0 + base_value * 0.6),
            "confidence": 0.82,
            "signal_counts": {
                "matrix_value_score": base_value,
                "method_count": graph_summary.get("entity_count", 0),
            },
            "evidence_paper_ids": db._load_json(gap.get("evidence_paper_ids"), []),
        })

    paper_signals = _fetch_paper_signals(node_id)
    limitation_clusters = _count_text_clusters(paper_signals, "limitations")
    for text, bucket in sorted(limitation_clusters.items(), key=lambda item: (-item[1]["count"], item[0]))[:2]:
        opportunities.append({
            "opportunity_type": "limitation_cluster",
            "title": _short_title("Fix a recurring bottleneck", text),
            "description": text,
            "why_now": f"This limitation appears in {bucket['count']} papers in this area.",
            "value_score": _clamp_score(2.2 + min(bucket["count"], 4) * 0.65),
            "confidence": 0.78,
            "signal_counts": {"paper_mentions": bucket["count"]},
            "evidence_paper_ids": bucket["paper_ids"][:8],
        })

    open_question_clusters = _count_text_clusters(paper_signals, "open_questions")
    for text, bucket in sorted(open_question_clusters.items(), key=lambda item: (-item[1]["count"], item[0]))[:2]:
        opportunities.append({
            "opportunity_type": "open_question",
            "title": _short_title("Answer an open question", text),
            "description": text,
            "why_now": f"Authors explicitly name this as unresolved in {bucket['count']} papers.",
            "value_score": _clamp_score(1.9 + min(bucket["count"], 4) * 0.55),
            "confidence": 0.74,
            "signal_counts": {"paper_mentions": bucket["count"]},
            "evidence_paper_ids": bucket["paper_ids"][:8],
        })

    contradictions = _fetch_contradiction_signals(node_id)
    if contradictions:
        sample = contradictions[0]
        anchor = sample.get("method_name") or sample.get("dataset_name") or sample.get("metric_name") or "key claims"
        opportunities.append({
            "opportunity_type": "contradiction_resolution",
            "title": f"Resolve conflicting evidence around {anchor}",
            "description": sample["description"],
            "why_now": f"There are {len(contradictions)} contradictory claim pairs in this area.",
            "value_score": _clamp_score(2.6 + min(len(contradictions), 4) * 0.5),
            "confidence": 0.84,
            "signal_counts": {"contradiction_pairs": len(contradictions)},
            "evidence_paper_ids": [sample["paper_a"], sample["paper_b"]],
        })

    entity_type_counts = _fetch_entity_type_counts(node_id)
    relation_counts = _fetch_relation_counts(node_id)
    method_count = entity_type_counts.get("method", 0)
    dataset_count = entity_type_counts.get("dataset", 0)
    metric_count = entity_type_counts.get("metric", 0)
    concept_count = entity_type_counts.get("concept", 0)

    if method_count >= 5 and dataset_count > 0 and method_count >= dataset_count * 3:
        opportunities.append({
            "opportunity_type": "benchmark_diversification",
            "title": "Broaden benchmark coverage",
            "description": (
                f"This area has {method_count} method entities but only {dataset_count} dataset entities "
                "showing up in the graph, so progress may be overfitted to narrow evaluation settings."
            ),
            "why_now": (
                f"The graph currently shows only {relation_counts.get('evaluated_on', 0)} "
                "`evaluated_on` links relative to the number of methods."
            ),
            "value_score": score_coverage_imbalance(method_count, dataset_count),
            "confidence": 0.79,
            "signal_counts": {
                "method_entities": method_count,
                "dataset_entities": dataset_count,
                "evaluated_on_relations": relation_counts.get("evaluated_on", 0),
            },
            "evidence_paper_ids": graph_summary.get("generated_from_papers", [])[:8],
        })

    if method_count >= 5 and metric_count <= 2:
        opportunities.append({
            "opportunity_type": "metric_diversification",
            "title": "Expand how the field measures progress",
            "description": (
                f"There are {method_count} method entities but only {metric_count} metric entities in the graph, "
                "which suggests the field may be using a narrow definition of success."
            ),
            "why_now": (
                f"Only {relation_counts.get('measured_by', 0)} `measured_by` relations are currently captured "
                "for this area."
            ),
            "value_score": score_metric_diversity(method_count, metric_count),
            "confidence": 0.76,
            "signal_counts": {
                "method_entities": method_count,
                "metric_entities": metric_count,
                "measured_by_relations": relation_counts.get("measured_by", 0),
            },
            "evidence_paper_ids": graph_summary.get("generated_from_papers", [])[:8],
        })

    if concept_count >= 10 and relation_counts.get("evaluated_on", 0) <= max(2, method_count // 2):
        opportunities.append({
            "opportunity_type": "problem_operationalization",
            "title": "Turn important concepts into testable benchmarks",
            "description": (
                f"The graph contains {concept_count} concept entities but relatively few evaluation links, "
                "which suggests the area may describe important problems faster than it standardizes how to test them."
            ),
            "why_now": "This is often the step that unlocks reproducible progress and comparable evidence.",
            "value_score": _clamp_score(2.1 + min(concept_count / 5.0, 2.6)),
            "confidence": 0.7,
            "signal_counts": {
                "concept_entities": concept_count,
                "evaluated_on_relations": relation_counts.get("evaluated_on", 0),
            },
            "evidence_paper_ids": graph_summary.get("generated_from_papers", [])[:8],
        })

    return _dedupe_opportunities(opportunities)[:8]


def ensure_node_opportunities(node_id: str, force: bool = False) -> list[dict]:
    existing = get_node_opportunities(node_id)
    if existing and not force:
        return existing
    built = build_node_opportunities(node_id)
    return replace_node_opportunities(node_id, built)


def upsert_opportunity_triage(triage: dict) -> dict:
    db.execute(
        """INSERT INTO opportunity_triage
           (opportunity_id, node_id, scientific_value, innovation, verifiability, cost,
            success_probability, evidence_strength, dependency_risk, priority_score,
            priority_band, rationale, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(opportunity_id) DO UPDATE SET
             node_id=excluded.node_id,
             scientific_value=excluded.scientific_value,
             innovation=excluded.innovation,
             verifiability=excluded.verifiability,
             cost=excluded.cost,
             success_probability=excluded.success_probability,
             evidence_strength=excluded.evidence_strength,
             dependency_risk=excluded.dependency_risk,
             priority_score=excluded.priority_score,
             priority_band=excluded.priority_band,
             rationale=excluded.rationale,
             status=excluded.status,
             updated_at=CURRENT_TIMESTAMP,
             scored_at=CURRENT_TIMESTAMP""",
        (
            triage["opportunity_id"],
            triage["node_id"],
            triage["scientific_value"],
            triage["innovation"],
            triage["verifiability"],
            triage["cost"],
            triage["success_probability"],
            triage["evidence_strength"],
            triage["dependency_risk"],
            triage["priority_score"],
            triage["priority_band"],
            triage.get("rationale"),
            triage.get("status", "ready"),
        ),
    )
    db.commit()
    row = db.fetchone(
        "SELECT * FROM opportunity_triage WHERE opportunity_id=?",
        (triage["opportunity_id"],),
    )
    if row:
        return row
    return triage


def rebuild_opportunity_triage(node_id: str | None = None, force: bool = False) -> list[dict]:
    if node_id:
        opportunities = get_node_opportunities(node_id)
        if not opportunities:
            opportunities = ensure_node_opportunities(node_id, force=force)
    else:
        opportunities = db.fetchall("SELECT * FROM node_opportunities ORDER BY node_id, value_score DESC, id DESC")

    valid_ids = {opportunity["id"] for opportunity in opportunities}
    if node_id:
        if valid_ids:
            placeholders = ",".join("?" for _ in valid_ids)
            db.execute(
                f"DELETE FROM opportunity_triage WHERE node_id=? AND opportunity_id NOT IN ({placeholders})",
                (node_id, *sorted(valid_ids)),
            )
        else:
            db.execute("DELETE FROM opportunity_triage WHERE node_id=?", (node_id,))
    elif valid_ids:
        placeholders = ",".join("?" for _ in valid_ids)
        db.execute(
            f"DELETE FROM opportunity_triage WHERE opportunity_id NOT IN ({placeholders})",
            tuple(sorted(valid_ids)),
        )
    else:
        db.execute("DELETE FROM opportunity_triage")

    triaged: list[dict] = []
    for opportunity in opportunities:
        scored = triage_opportunity(opportunity)
        triaged.append(upsert_opportunity_triage(scored))
    return sorted(triaged, key=lambda item: (-item["priority_score"], item["node_id"], item["opportunity_id"]))


def get_opportunity_triage(node_id: str | None = None, band: str | None = None, limit: int = 200) -> list[dict]:
    sql = """SELECT ot.*, no.title, no.description, no.opportunity_type, no.why_now,
                    no.value_score AS opportunity_value_score,
                    no.confidence AS opportunity_confidence,
                    no.evidence_paper_ids,
                    tn.name AS node_name
             FROM opportunity_triage ot
             JOIN node_opportunities no ON no.id = ot.opportunity_id
             JOIN taxonomy_nodes tn ON tn.id = ot.node_id
             WHERE 1=1"""
    params: list[object] = []
    if node_id:
        sql += " AND (ot.node_id=? OR ot.node_id LIKE ? || '.%')"
        params.extend([node_id, node_id])
    if band:
        sql += " AND ot.priority_band=?"
        params.append(band)
    sql += " ORDER BY ot.priority_score DESC, ot.scored_at DESC, ot.id DESC LIMIT ?"
    params.append(limit)
    rows = db.fetchall(sql, tuple(params))
    for row in rows:
        row["evidence_paper_ids"] = db._load_json(row.get("evidence_paper_ids"), [])
    return rows


def get_opportunity_triage_stats() -> dict:
    total = db.fetchone("SELECT COUNT(*) AS c FROM opportunity_triage")
    bands = db.fetchall(
        "SELECT priority_band, COUNT(*) AS c FROM opportunity_triage GROUP BY priority_band"
    )
    by_band = {row["priority_band"]: row["c"] for row in bands}
    avg_row = db.fetchone(
        """SELECT
               AVG(priority_score) AS avg_priority,
               AVG(scientific_value) AS avg_scientific_value,
               AVG(innovation) AS avg_innovation,
               AVG(verifiability) AS avg_verifiability,
               AVG(cost) AS avg_cost,
               AVG(success_probability) AS avg_success_probability
           FROM opportunity_triage"""
    ) or {}
    return {
        "total": total["c"] if total else 0,
        "bands": by_band,
        "averages": {
            "priority_score": round(avg_row.get("avg_priority") or 0, 2),
            "scientific_value": round(avg_row.get("avg_scientific_value") or 0, 2),
            "innovation": round(avg_row.get("avg_innovation") or 0, 2),
            "verifiability": round(avg_row.get("avg_verifiability") or 0, 2),
            "cost": round(avg_row.get("avg_cost") or 0, 2),
            "success_probability": round(avg_row.get("avg_success_probability") or 0, 2),
        },
    }


def triage_opportunity(opportunity: dict) -> dict:
    """Score a node opportunity for execution priority."""
    node_id = opportunity["node_id"]
    title = (opportunity.get("title") or "").lower()
    description = (opportunity.get("description") or "").lower()
    combined = f"{title} {description}"
    papers = opportunity.get("evidence_paper_ids") or []
    if isinstance(papers, str):
        papers = db._load_json(papers, [])
    signal_counts = opportunity.get("signal_counts") or {}
    if isinstance(signal_counts, str):
        signal_counts = db._load_json(signal_counts, {})

    scientific_value = _clamp_score(float(opportunity.get("value_score") or 3.0))

    novelty = 2.5
    if opportunity.get("opportunity_type") in {"contradiction_resolution", "problem_operationalization"}:
        novelty += 1.0
    if "broaden" in combined or "expand" in combined:
        novelty += 0.6
    if "recurring" in combined or "unresolved" in combined:
        novelty += 0.4

    verifiability = 2.5
    if opportunity.get("opportunity_type") in {"evaluation_gap", "problem_operationalization"}:
        verifiability += 1.1
    if papers:
        verifiability += 0.5
    if signal_counts:
        verifiability += min(0.6, len(signal_counts) * 0.15)

    cost = 2.0
    if opportunity.get("opportunity_type") == "contradiction_resolution":
        cost += 0.4
    if opportunity.get("opportunity_type") == "benchmark_diversification":
        cost += 0.3
    if len(papers) >= 5:
        cost += 0.4

    success_probability = 2.8
    if papers:
        success_probability += 0.6
    if opportunity.get("confidence"):
        success_probability += min(0.7, float(opportunity.get("confidence")) - 0.5)
    if opportunity.get("opportunity_type") == "open_question":
        success_probability -= 0.2

    evidence_strength = 2.4
    if len(papers) >= 2:
        evidence_strength += 0.9
    if signal_counts:
        evidence_strength += min(0.7, len(signal_counts) * 0.2)
    if opportunity.get("why_now"):
        evidence_strength += 0.2

    dependency_risk = 1.8
    if opportunity.get("opportunity_type") in {"benchmark_diversification", "problem_operationalization"}:
        dependency_risk += 0.2
    if len(papers) == 0:
        dependency_risk += 0.5

    priority_score = (
        0.30 * scientific_value
        + 0.20 * novelty
        + 0.20 * verifiability
        + 0.20 * success_probability
        + 0.10 * evidence_strength
        - 0.15 * cost
        - 0.10 * dependency_risk
    )
    priority_score = round(max(1.0, min(5.0, priority_score)), 2)
    band = _normalize_band(priority_score)

    rationale_parts = [
        f"{opportunity['opportunity_type']} on {node_id}",
        f"evidence={len(papers)} papers",
        f"signal={len(signal_counts)} signals",
        f"value={scientific_value:.1f}",
    ]
    if opportunity.get("why_now"):
        rationale_parts.append(opportunity["why_now"])

    return {
        "opportunity_id": opportunity["id"],
        "node_id": node_id,
        "scientific_value": scientific_value,
        "innovation": _clamp_score(novelty),
        "verifiability": _clamp_score(verifiability),
        "cost": _clamp_score(cost),
        "success_probability": _clamp_score(success_probability),
        "evidence_strength": _clamp_score(evidence_strength),
        "dependency_risk": _clamp_score(dependency_risk),
        "priority_score": priority_score,
        "priority_band": band,
        "rationale": " | ".join(rationale_parts),
        "status": "ready" if band != "watchlist" else "needs_review",
    }



def triage_node_opportunities(node_id: str, force: bool = False) -> list[dict]:
    opportunities = get_node_opportunities(node_id)
    if not opportunities:
        opportunities = ensure_node_opportunities(node_id, force=force)
    triaged = [triage_opportunity(opportunity) for opportunity in opportunities]
    return sorted(triaged, key=lambda item: (-item["priority_score"], -item["scientific_value"], item["opportunity_id"]))


def triage_all_opportunities(force: bool = False) -> list[dict]:
    nodes = db.fetchall(
        """SELECT DISTINCT node_id FROM node_opportunities ORDER BY node_id"""
    )
    rows: list[dict] = []
    for node in nodes:
        rows.extend(triage_node_opportunities(node["node_id"], force=force))
    return sorted(rows, key=lambda item: (-item["priority_score"], item["node_id"], item["opportunity_id"]))
