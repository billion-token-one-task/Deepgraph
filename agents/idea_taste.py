"""Graph-grounded idea taste: novelty, counterevidence, frontier, and ranking."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from db import database as db
from db import evidence_graph as graph

_CACHE_TTL_SECONDS = 300.0
_frontier_cache: dict[str, tuple[float, dict]] = {}
_stakes_cache: dict[str, tuple[float, float]] = {}
_signal_weights_cache: tuple[float, dict[str, float]] | None = None


def _json_load(value: Any, default):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _normalize_tokens(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]{3,}", (text or "").lower())}


def _node_ids_from_insight(insight: dict) -> list[str]:
    node_ids = _json_load(insight.get("source_node_ids"), [])
    if isinstance(node_ids, str) and node_ids.strip():
        return [node_ids.strip()]
    if not isinstance(node_ids, list):
        return []
    return [str(node_id).strip() for node_id in node_ids if str(node_id).strip()]


def _method_payload(insight: dict) -> dict:
    return _json_load(insight.get("proposed_method"), {})


def _compact_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _cache_key(node_ids: list[str]) -> str:
    return "|".join(sorted(node_ids[:6]))


def _cached_value(cache: dict, key: str):
    entry = cache.get(key)
    if not entry:
        return None
    expires_at, value = entry
    if time.time() > expires_at:
        cache.pop(key, None)
        return None
    return value


def _store_cache(cache: dict, key: str, value) -> None:
    cache[key] = (time.time() + _CACHE_TTL_SECONDS, value)


def signal_type_weight(signal_type: str) -> float:
    global _signal_weights_cache
    now = time.time()
    if _signal_weights_cache is None or now > _signal_weights_cache[0]:
        try:
            from agents.meta_learner import compute_signal_weights

            weights = compute_signal_weights()
        except Exception:
            weights = {}
        _signal_weights_cache = (now + _CACHE_TTL_SECONDS, weights)
    return float(_signal_weights_cache[1].get(signal_type, 1.0))


def _relation_count_for_node(node_id: str) -> int:
    row = db.fetchone(
        """
        SELECT COUNT(*) AS c
        FROM graph_relations
        WHERE node_id = ? OR node_id LIKE ? || '.%'
        """,
        (node_id, node_id),
    )
    return int((row or {}).get("c") or 0)


def score_graph_novelty(insight: dict) -> dict:
    """Score novelty against the evidence graph (0-10, higher = more novel)."""
    method = _method_payload(insight)
    method_name = str(method.get("name") or "").strip()
    node_ids = _node_ids_from_insight(insight)

    score = 10.0
    reasons: list[str] = []
    nearest: list[dict] = []

    if method_name:
        norm = graph.normalize_entity_name(method_name)
        rows = db.fetchall(
            """
            SELECT ge.id, ge.canonical_name, ge.entity_type,
                   COUNT(DISTINCT pem.paper_id) AS paper_count
            FROM graph_entities ge
            LEFT JOIN paper_entity_mentions pem ON pem.entity_id = ge.id
            WHERE ge.normalized_name = ?
               OR ge.normalized_name LIKE ? || '%'
            GROUP BY ge.id, ge.canonical_name, ge.entity_type
            ORDER BY paper_count DESC
            LIMIT 3
            """,
            (norm, norm[:12]),
        )
        if rows:
            top = rows[0]
            paper_count = int(top.get("paper_count") or 0)
            if paper_count >= 2:
                penalty = min(6.5, 1.5 + paper_count * 0.55)
                score -= penalty
                reasons.append(
                    f"Method-like entity '{top['canonical_name']}' already appears in {paper_count} papers."
                )
                nearest.append(
                    {
                        "entity_id": top["id"],
                        "name": top["canonical_name"],
                        "entity_type": top.get("entity_type"),
                        "paper_count": paper_count,
                    }
                )

    for node_id in node_ids[:3]:
        relation_count = _relation_count_for_node(node_id)
        if relation_count >= 80:
            score -= 0.4
            reasons.append(
                f"Node {node_id} already has dense relation coverage ({relation_count} edges)."
            )

    score = round(max(0.0, min(10.0, score)), 2)
    if score >= 6.5:
        status = "novel"
    elif score >= 3.5:
        status = "partial"
    else:
        status = "likely_exists"

    return {
        "score": score,
        "status": status,
        "reasons": reasons,
        "nearest_neighbors": nearest,
    }


def retrieve_graph_counterevidence(insight: dict, *, limit: int = 6) -> list[dict]:
    """Pull graph-backed contradictions and prior claims for an idea."""
    node_ids = _node_ids_from_insight(insight)[:3]
    method = _method_payload(insight)
    method_name = str(method.get("name") or "").strip()
    focus_tokens = _normalize_tokens(method_name) if method_name else set()

    hits: list[dict] = []
    seen: set[str] = set()

    for node_id in node_ids:
        rows = db.fetchall(
            """
            SELECT c.id, c.description,
                   ca.paper_id AS paper_a, cb.paper_id AS paper_b,
                   ca.method_name AS method_a, cb.method_name AS method_b,
                   ca.metric_value AS value_a, cb.metric_value AS value_b
            FROM contradictions c
            LEFT JOIN claims ca ON ca.id = c.claim_a_id
            LEFT JOIN claims cb ON cb.id = c.claim_b_id
            WHERE EXISTS (
                SELECT 1 FROM paper_taxonomy pta
                WHERE pta.paper_id = ca.paper_id
                  AND (pta.node_id = ? OR pta.node_id LIKE ? || '.%')
            )
            OR EXISTS (
                SELECT 1 FROM paper_taxonomy ptb
                WHERE ptb.paper_id = cb.paper_id
                  AND (ptb.node_id = ? OR ptb.node_id LIKE ? || '.%')
            )
            ORDER BY c.id DESC
            LIMIT ?
            """,
            (node_id, node_id, node_id, node_id, max(1, limit // max(len(node_ids), 1))),
        )
        for row in rows:
            key = f"contradiction:{row['id']}"
            if key in seen:
                continue
            seen.add(key)
            hits.append(
                {
                    "type": "contradiction",
                    "node_id": node_id,
                    "description": (row.get("description") or "")[:280],
                    "paper_a": row.get("paper_a"),
                    "paper_b": row.get("paper_b"),
                    "method_a": row.get("method_a"),
                    "method_b": row.get("method_b"),
                    "value_a": row.get("value_a"),
                    "value_b": row.get("value_b"),
                }
            )

    if focus_tokens and len(hits) < limit:
        token = sorted(focus_tokens, key=len, reverse=True)[0]
        claim_rows = db.fetchall(
            """
            SELECT c.id, c.paper_id, c.method_name, c.dataset_name, c.metric_name,
                   c.metric_value, c.claim_text
            FROM claims c
            WHERE LOWER(COALESCE(c.method_name, '')) LIKE ?
            ORDER BY c.id DESC
            LIMIT ?
            """,
            (f"%{token}%", limit - len(hits)),
        )
        for row in claim_rows:
            key = f"claim:{row['id']}"
            if key in seen:
                continue
            seen.add(key)
            hits.append(
                {
                    "type": "prior_claim",
                    "paper_id": row.get("paper_id"),
                    "method_name": row.get("method_name"),
                    "dataset_name": row.get("dataset_name"),
                    "metric_name": row.get("metric_name"),
                    "metric_value": row.get("metric_value"),
                    "claim_text": (row.get("claim_text") or "")[:220],
                }
            )

    return hits[:limit]


def compute_frontier_profile(node_ids: list[str]) -> dict:
    """Estimate activity / saturation / headroom for idea-bearing nodes."""
    if not node_ids:
        return {}

    key = _cache_key(node_ids)
    cached = _cached_value(_frontier_cache, key)
    if cached is not None:
        return cached

    activity = 0
    spreads: list[float] = []
    method_counts: list[int] = []

    for node_id in node_ids[:3]:
        paper_row = db.fetchone(
            """
            SELECT COUNT(DISTINCT paper_id) AS paper_count
            FROM paper_taxonomy
            WHERE node_id = ? OR node_id LIKE ? || '.%'
            """,
            (node_id, node_id),
        )
        activity += int((paper_row or {}).get("paper_count") or 0)

        result_rows = db.fetchall(
            """
            SELECT method_name, dataset_name, metric_name, metric_value
            FROM results
            WHERE node_id = ? OR node_id LIKE ? || '.%'
              AND metric_value IS NOT NULL
            LIMIT 120
            """,
            (node_id, node_id),
        )
        if not result_rows:
            continue

        by_bucket: dict[tuple[str, str, str], list[float]] = {}
        methods = set()
        for row in result_rows:
            method = str(row.get("method_name") or "").strip()
            dataset = str(row.get("dataset_name") or "").strip() or "unknown"
            metric = str(row.get("metric_name") or "").strip() or "metric"
            try:
                value = float(row.get("metric_value"))
            except (TypeError, ValueError):
                continue
            methods.add(method)
            by_bucket.setdefault((dataset, metric, method), []).append(value)

        method_counts.append(len(methods))
        for values in by_bucket.values():
            if len(values) < 2:
                continue
            lo, hi = min(values), max(values)
            if hi <= 0:
                continue
            spreads.append((hi - lo) / max(abs(hi), 1e-9))

    avg_spread = sum(spreads) / len(spreads) if spreads else 0.5
    saturation = max(0.0, min(1.0, 1.0 - avg_spread * 4.0))
    headroom = max(0.0, min(1.0, 1.0 - saturation))
    profile = {
        "activity": activity,
        "saturation": round(saturation, 3),
        "headroom": round(headroom, 3),
        "method_diversity": max(method_counts) if method_counts else 0,
    }
    _store_cache(_frontier_cache, key, profile)
    return profile


def compute_stakes(node_ids: list[str]) -> float:
    """Higher when the idea touches highly connected graph regions."""
    if not node_ids:
        return 0.0

    key = _cache_key(node_ids)
    cached = _cached_value(_stakes_cache, key)
    if cached is not None:
        return cached

    total = 0.0
    counted = 0
    for node_id in node_ids[:3]:
        relation_count = _relation_count_for_node(node_id)
        entity_row = db.fetchone(
            """
            SELECT COUNT(DISTINCT entity_id) AS entity_count
            FROM paper_entity_mentions
            WHERE node_id = ? OR node_id LIKE ? || '.%'
            """,
            (node_id, node_id),
        )
        entity_count = int((entity_row or {}).get("entity_count") or 0)
        total += min(10.0, relation_count / 250.0 + entity_count / 120.0)
        counted += 1

    value = round(total / max(counted, 1), 3)
    _store_cache(_stakes_cache, key, value)
    return value


def score_excitement(candidate: dict) -> dict:
    """Score field leverage / invention potential (0-10, higher is more exciting).

    This is intentionally orthogonal to novelty/feasibility: it rewards reusable
    mechanisms, formal objects, objectives, and cross-task leverage, while
    penalizing ideas that are only benchmark/diagnostic/evaluation papers.
    """
    method = _method_payload(candidate)
    exp = _json_load(candidate.get("experimental_plan"), {})
    related = _json_load(candidate.get("related_work_positioning"), {})
    packet = candidate.get("evidence_packet") or {}
    if not isinstance(packet, dict):
        packet = {}

    text = " ".join(
        [
            _compact_text(candidate.get("title")),
            _compact_text(candidate.get("problem_statement")),
            _compact_text(candidate.get("existing_weakness")),
            _compact_text(method),
            _compact_text(exp),
            _compact_text(related),
            _compact_text(packet.get("structural_evidence")),
            _compact_text(packet.get("non_numeric_evidence")),
        ]
    ).lower()
    method_type = str(method.get("type") or "").lower()

    score = 5.0
    reasons: list[str] = []

    mechanism_terms = [
        "loss", "objective", "optimizer", "optimization", "gradient", "architecture",
        "module", "training procedure", "training objective", "inference-time",
        "algorithm", "policy", "latent variable", "hidden variable", "formal object",
        "representation", "state space", "invariance", "equivariance", "causal",
        "theorem", "guarantee", "bound", "certificate",
    ]
    mechanism_hits = [term for term in mechanism_terms if term in text or term in method_type]
    if mechanism_hits:
        bonus = min(2.2, 0.35 * len(set(mechanism_hits)))
        score += bonus
        reasons.append(f"mechanism-level terms present: {', '.join(sorted(set(mechanism_hits))[:6])}")

    reusable_terms = [
        "across tasks", "cross-task", "multi-domain", "generalizes", "generalization",
        "transfer", "plug-in", "drop-in", "reusable", "family of", "framework",
        "operator", "functional", "estimator", "protocol-independent",
    ]
    reusable_hits = [term for term in reusable_terms if term in text]
    if reusable_hits:
        bonus = min(1.6, 0.32 * len(set(reusable_hits)))
        score += bonus
        reasons.append(f"field-leverage terms present: {', '.join(sorted(set(reusable_hits))[:5])}")

    formal_density = 0.0
    definition = _compact_text(method.get("definition"))
    if any(sym in definition for sym in ("\\", "min", "max", "arg", "E[", "Pr", "∑", "forall", "subject to")):
        formal_density += 0.8
    if method.get("pseudocode"):
        formal_density += 0.4
    if method.get("key_properties"):
        formal_density += 0.3
    if formal_density:
        score += formal_density
        reasons.append("method has explicit formal/pseudocode structure")

    signal_mix = candidate.get("signal_mix") or packet.get("signal_mix") or []
    if isinstance(signal_mix, str):
        signal_mix = _json_load(signal_mix, [])
    if any(s in {"hidden_variable_bridge", "mechanism_mismatch", "claim_method_gap"} for s in signal_mix):
        score += 0.8
        reasons.append("source signal favors mechanism invention")

    eval_terms = [
        "benchmark", "leaderboard", "evaluation", "diagnostic", "diagnose",
        "audit", "audited", "metric", "metrics", "stress test", "robustness suite",
        "test suite", "protocol", "harness",
    ]
    eval_hits = [term for term in eval_terms if term in text]
    method_terms = [
        "loss", "objective", "optimizer", "architecture", "training procedure",
        "algorithm", "policy", "representation", "latent", "formal object",
    ]
    method_hits = [term for term in method_terms if term in text or term in method_type]
    if eval_hits:
        penalty = min(2.2, 0.22 * len(set(eval_hits)))
        if len(method_hits) < 3:
            penalty += 1.0
            reasons.append("evaluation/benchmark framing dominates without enough mechanism")
        else:
            penalty *= 0.45
            reasons.append("evaluation framing present but offset by mechanism content")
        score -= penalty

    narrow_terms = ["case study", "dataset-specific", "single benchmark", "taxonomy", "survey"]
    narrow_hits = [term for term in narrow_terms if term in text]
    if narrow_hits:
        penalty = min(1.2, 0.4 * len(set(narrow_hits)))
        score -= penalty
        reasons.append(f"narrow-scope terms present: {', '.join(sorted(set(narrow_hits))[:4])}")

    datasets = exp.get("datasets") if isinstance(exp, dict) else []
    if isinstance(datasets, list) and len(datasets) >= 2:
        score += 0.4
        reasons.append("experiment spans multiple datasets/tasks")

    status = (
        "high_leverage" if score >= 7.5
        else "solid" if score >= 5.5
        else "low_excitement"
    )
    return {
        "score": round(max(0.0, min(10.0, score)), 3),
        "status": status,
        "reasons": reasons[:8],
        "penalizes_pure_benchmark": bool(eval_hits),
    }


def compute_taste_score(
    candidate: dict,
    *,
    frontier: dict | None = None,
    stakes: float | None = None,
) -> dict:
    """Unified graph-aware taste score used for candidate ranking."""
    packet = candidate.get("evidence_packet") or {}
    non_numeric = len(packet.get("non_numeric_evidence", []))
    structural = len(packet.get("structural_evidence", []))
    support = float(candidate.get("support_score") or 0)

    node_ids = candidate.get("source_node_ids") or []
    if frontier is None:
        frontier = compute_frontier_profile(node_ids) if node_ids else {}
    if stakes is None:
        stakes = compute_stakes(node_ids) if node_ids else 0.0

    signal_mix = candidate.get("signal_mix") or []
    if signal_mix:
        signal_bonus = sum(signal_type_weight(signal) for signal in signal_mix) / len(signal_mix)
    else:
        signal_bonus = 1.0

    mechanism_type = candidate.get("mechanism_type") or "deep_insight"
    mechanism_bonus = 1.2 if mechanism_type not in {"plateau", "deep_insight", "paper_idea"} else 0.35
    gpu_penalty = 0.25 if candidate.get("resource_class") == "gpu_large" else 0.0

    graph_novelty = candidate.get("graph_novelty") or packet.get("graph_novelty") or {}
    novelty_bonus = (float(graph_novelty.get("score") or 5.0) - 5.0) * 0.45
    counterevidence = candidate.get("graph_counterevidence") or packet.get("graph_counterevidence") or []
    counter_penalty = min(2.0, 0.35 * len(counterevidence))
    excitement = candidate.get("excitement") or packet.get("excitement") or score_excitement(candidate)
    excitement_score = float(excitement.get("score") or 5.0)
    excitement_bonus = (excitement_score - 5.0) * 0.65

    frontier_bonus = 0.0
    if frontier:
        activity = float(frontier.get("activity") or 0)
        saturation = float(frontier.get("saturation") or 0)
        headroom = float(frontier.get("headroom") or 0)
        if activity >= 8 and saturation >= 0.55 and headroom >= 0.25:
            frontier_bonus = 1.4
        elif saturation >= 0.9 and headroom <= 0.1:
            frontier_bonus = -1.0

    taste_score = (
        non_numeric * 1.4
        + structural * 1.0
        + support * 0.75
        + mechanism_bonus
        + stakes * 0.9
        + signal_bonus * 0.55
        + novelty_bonus
        + excitement_bonus
        + frontier_bonus
        - counter_penalty
        - gpu_penalty
    )

    return {
        "taste_score": round(taste_score, 3),
        "components": {
            "non_numeric": non_numeric,
            "structural": structural,
            "support": support,
            "mechanism_bonus": mechanism_bonus,
            "stakes": stakes,
            "signal_bonus": round(signal_bonus, 3),
            "novelty_bonus": round(novelty_bonus, 3),
            "excitement_bonus": round(excitement_bonus, 3),
            "excitement_score": round(excitement_score, 3),
            "excitement_status": excitement.get("status"),
            "frontier_bonus": frontier_bonus,
            "counter_penalty": round(counter_penalty, 3),
            "gpu_penalty": gpu_penalty,
        },
        "excitement": excitement,
        "frontier": frontier,
        "stakes": stakes,
    }


def enrich_candidate_with_graph_taste(candidate: dict, insight: dict | None = None) -> dict:
    payload = dict(candidate)
    packet = payload.get("evidence_packet") or {}
    if not isinstance(packet, dict):
        packet = {}

    if packet.get("graph_novelty") and payload.get("taste_score") is not None:
        payload["graph_novelty"] = packet["graph_novelty"]
        payload["graph_counterevidence"] = packet.get("graph_counterevidence") or []
        payload["frontier_profile"] = packet.get("frontier_profile") or {}
        return payload

    source = insight or candidate
    node_ids = payload.get("source_node_ids") or _node_ids_from_insight(source)
    novelty = packet.get("graph_novelty") or score_graph_novelty(source)
    counterevidence = packet.get("graph_counterevidence")
    if counterevidence is None:
        counterevidence = retrieve_graph_counterevidence(source)
    frontier = packet.get("frontier_profile") or compute_frontier_profile(node_ids)
    stakes = compute_stakes(node_ids) if node_ids else 0.0

    payload["graph_novelty"] = novelty
    payload["graph_counterevidence"] = counterevidence
    payload["frontier_profile"] = frontier
    payload["source_node_ids"] = node_ids

    taste = compute_taste_score(payload, frontier=frontier, stakes=stakes)
    payload["taste_score"] = taste["taste_score"]
    payload["taste_components"] = taste["components"]
    payload["excitement"] = taste.get("excitement")
    payload["excitement_score"] = (taste.get("excitement") or {}).get("score")
    payload["excitement_status"] = (taste.get("excitement") or {}).get("status")
    return payload


def graph_novelty_gate(insight: dict, *, min_score: float = 3.5) -> dict | None:
    """Reject ideas that the graph already treats as largely existing."""
    result = score_graph_novelty(insight)
    if result["status"] == "likely_exists" and float(result["score"]) < min_score:
        return {
            "error": "Graph novelty gate rejected this idea as likely already present in the literature graph.",
            "graph_novelty": result,
        }
    return None


def format_counterevidence_block(counterevidence: list[dict]) -> str:
    if not counterevidence:
        return ""
    lines = ["## GRAPH COUNTEREVIDENCE (from DeepGraph evidence graph)", ""]
    for item in counterevidence[:6]:
        if item.get("type") == "contradiction":
            lines.append(
                f"- Contradiction in {item.get('node_id')}: {item.get('description')} "
                f"({item.get('paper_a')} vs {item.get('paper_b')})"
            )
        else:
            lines.append(
                f"- Prior claim in {item.get('paper_id')}: {item.get('method_name')} on "
                f"{item.get('dataset_name')} -> {item.get('metric_name')}={item.get('metric_value')}"
            )
    lines.append("")
    return "\n".join(lines)


def format_frontier_block(node_ids: list[str]) -> str:
    profile = compute_frontier_profile(node_ids)
    if not profile:
        return ""
    return (
        "## FRONTIER PROFILE\n"
        f"- activity (papers in scope): {profile.get('activity')}\n"
        f"- saturation (0=open, 1=plateau): {profile.get('saturation')}\n"
        f"- headroom (0=closed, 1=room to improve): {profile.get('headroom')}\n"
        f"- method diversity: {profile.get('method_diversity')}\n"
    )


def attach_graph_taste_to_insight(insight: dict) -> dict:
    """Attach graph taste metadata before persistence."""
    enriched = dict(insight)
    node_ids = _node_ids_from_insight(enriched)

    novelty = score_graph_novelty(enriched)
    counterevidence = retrieve_graph_counterevidence(enriched)
    frontier = compute_frontier_profile(node_ids)
    stakes = compute_stakes(node_ids) if node_ids else 0.0

    packet = _json_load(enriched.get("evidence_packet"), {})
    if not isinstance(packet, dict):
        packet = {}
    packet["graph_novelty"] = novelty
    packet["graph_counterevidence"] = counterevidence[:6]
    taste = compute_taste_score(
        {
            **enriched,
            "evidence_packet": packet,
            "graph_novelty": novelty,
            "graph_counterevidence": counterevidence,
            "source_node_ids": node_ids,
            "signal_mix": _json_load(enriched.get("signal_mix"), []),
            "support_score": float(enriched.get("adversarial_score") or 0),
            "mechanism_type": enriched.get("mechanism_type"),
            "resource_class": enriched.get("resource_class"),
        },
        frontier=frontier,
        stakes=stakes,
    )
    packet["frontier_profile"] = frontier
    packet["excitement"] = taste.get("excitement")
    packet["taste_components"] = taste.get("components")
    enriched["evidence_packet"] = packet
    enriched["graph_novelty_score"] = novelty.get("score")
    enriched["graph_novelty_status"] = novelty.get("status")
    enriched["taste_score"] = taste["taste_score"]
    enriched["excitement_score"] = (taste.get("excitement") or {}).get("score")
    enriched["excitement_status"] = (taste.get("excitement") or {}).get("status")
    return enriched
