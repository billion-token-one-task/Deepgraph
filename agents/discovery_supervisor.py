"""Mechanism-first discovery candidate assembly and pairwise ranking."""

from __future__ import annotations

import json
from itertools import combinations

from agents.idea_taste import compute_taste_score, enrich_candidate_with_graph_taste
from db import database as db


def _json_load(value, default):
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _candidate_from_signal(row: dict, source: str) -> dict:
    title = row.get("title") or row.get("summary") or row.get("theme") or row.get("shared_factor") or source
    node_ids = []
    for key in ("node_id", "node_a_id", "node_b_id"):
        if row.get(key):
            node_ids.append(row[key])
    packet = {
        "signal_mix": [source],
        "non_numeric_evidence": [row.get("summary") or row.get("theme") or title],
        "structural_evidence": [title],
        "falsification": {"summary": "Promote to deep insight and validate experimentally."},
    }
    return {
        "id": f"{source}:{row.get('id', title)}",
        "source": source,
        "title": title,
        "mechanism_type": source,
        "signal_mix": [source],
        "source_node_ids": node_ids,
        "evidence_summary": row.get("summary") or row.get("theme") or title,
        "evidence_packet": packet,
        "resource_class": "cpu",
        "stored": False,
        "support_score": float(row.get("support_count") or row.get("score") or row.get("cluster_size") or 1),
    }


def collect_candidate_pool(limit: int = 50) -> list[dict]:
    candidates = []

    stored = db.fetchall(
        """
        SELECT id, title, mechanism_type, signal_mix, evidence_packet, evidence_summary,
               source_node_ids, resource_class, adversarial_score,
               proposed_method, problem_statement
        FROM deep_insights
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    for row in stored:
        packet = _json_load(row.get("evidence_packet"), {})
        candidate = enrich_candidate_with_graph_taste(
            {
                "id": row["id"],
                "source": "deep_insight",
                "title": row["title"],
                "mechanism_type": row.get("mechanism_type") or "deep_insight",
                "signal_mix": _json_load(row.get("signal_mix"), []),
                "source_node_ids": _json_load(row.get("source_node_ids"), []),
                "evidence_summary": row.get("evidence_summary"),
                "evidence_packet": packet,
                "resource_class": row.get("resource_class") or "cpu",
                "stored": True,
                "support_score": float(row.get("adversarial_score") or row.get("taste_score") or 0),
                "proposed_method": row.get("proposed_method"),
                "problem_statement": row.get("problem_statement"),
            },
            insight=dict(row),
        )
        candidates.append(candidate)

    signal_sources = [
        ("protocol_artifact", "SELECT * FROM protocol_artifacts ORDER BY support_count DESC LIMIT 10"),
        ("negative_space_gap", "SELECT * FROM negative_space_gaps ORDER BY support_count DESC LIMIT 10"),
        ("mechanism_mismatch", "SELECT * FROM mechanism_mismatches ORDER BY support_count DESC LIMIT 10"),
        ("hidden_variable_bridge", "SELECT * FROM hidden_variable_bridges ORDER BY score DESC LIMIT 10"),
        ("claim_method_gap", "SELECT * FROM claim_method_gaps ORDER BY support_count DESC LIMIT 10"),
    ]
    for source, sql in signal_sources:
        for row in db.fetchall(sql):
            signal_candidate = _candidate_from_signal(row, source)
            signal_candidate["taste_score"] = compute_taste_score(signal_candidate)["taste_score"]
            candidates.append(signal_candidate)

    deduped = {}
    for candidate in candidates:
        key = (candidate["title"], tuple(candidate["signal_mix"]))
        deduped[key] = candidate
    return list(deduped.values())[:limit]


def _quality(candidate: dict) -> float:
    if candidate.get("taste_score") is not None:
        return float(candidate["taste_score"])
    return compute_taste_score(candidate)["taste_score"]


def rank_candidates(limit: int = 20) -> list[dict]:
    pool_limit = min(max(limit * 3, 20), 30)
    candidates = collect_candidate_pool(limit=pool_limit)
    if not candidates:
        return []

    quality_by_id = {str(candidate["id"]): _quality(candidate) for candidate in candidates}
    ratings = {str(candidate["id"]): 1000.0 for candidate in candidates}
    for left, right in combinations(candidates, 2):
        left_id = str(left["id"])
        right_id = str(right["id"])
        q_left = quality_by_id[str(left["id"])]
        q_right = quality_by_id[str(right["id"])]
        if q_left == q_right:
            score_left = 0.5
        else:
            score_left = 1.0 if q_left > q_right else 0.0
        expected_left = 1.0 / (1.0 + 10 ** ((ratings[right_id] - ratings[left_id]) / 400.0))
        expected_right = 1.0 - expected_left
        k = 24.0
        ratings[left_id] += k * (score_left - expected_left)
        ratings[right_id] += k * ((1.0 - score_left) - expected_right)

    ranked = sorted(candidates, key=lambda candidate: ratings[str(candidate["id"])], reverse=True)[:limit]
    output = []
    for idx, candidate in enumerate(ranked, start=1):
        output.append(
            {
                **candidate,
                "rank": idx,
                "elo": round(ratings[str(candidate["id"])], 2),
                "quality_score": round(quality_by_id[str(candidate["id"])], 2),
            }
        )
    return output
