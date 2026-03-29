"""Signal Harvester: pure SQL-based cross-field signal detection.

Zero LLM cost. Finds structural signals that seed Tier 1 and Tier 2 discovery:
- Cross-node entity overlap (shared methods/datasets/concepts between distant fields)
- Convergent pattern matching (different domains discovering the same solution)
- Contradiction clustering (groups of related conflicts)
- Performance plateau detection (diminishing returns in a subfield)
"""
import json
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from db import database as db


def _taxonomic_distance(node_a: str, node_b: str) -> int:
    """Compute hops between two taxonomy node IDs via their LCA."""
    parts_a = node_a.split(".")
    parts_b = node_b.split(".")
    common = 0
    for pa, pb in zip(parts_a, parts_b):
        if pa == pb:
            common += 1
        else:
            break
    return (len(parts_a) - common) + (len(parts_b) - common)


def harvest_entity_overlap(min_shared: int = 3, top_k: int = 100):
    """Find taxonomy node pairs sharing entities (methods, concepts, datasets).

    Focuses on meaningful entity types and filters out generic entities.
    """
    print("[SIGNAL] Computing cross-node entity overlap...", flush=True)

    GENERIC_ENTITIES = {
        "model", "system", "accuracy", "training method", "analysis",
        "performance", "evaluation", "baseline", "dataset", "method",
        "framework", "approach", "results", "experiment", "task",
    }

    rows = db.fetchall("""
        SELECT pem.node_id, pem.entity_id, ge.canonical_name, ge.entity_type
        FROM paper_entity_mentions pem
        JOIN graph_entities ge ON pem.entity_id = ge.id
        WHERE pem.node_id IS NOT NULL
          AND ge.entity_type IN ('method', 'dataset', 'concept', 'task', 'theory')
          AND ge.canonical_name NOT IN ({})
        GROUP BY pem.node_id, pem.entity_id
    """.format(",".join("?" * len(GENERIC_ENTITIES))),
        tuple(GENERIC_ENTITIES)
    )

    node_entities = defaultdict(set)
    entity_info = {}
    for r in rows:
        node_entities[r["node_id"]].add(r["entity_id"])
        entity_info[r["entity_id"]] = {
            "name": r["canonical_name"],
            "type": r["entity_type"],
        }

    nodes = list(node_entities.keys())
    overlaps = []

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            na, nb = nodes[i], nodes[j]
            shared = node_entities[na] & node_entities[nb]
            if len(shared) < min_shared:
                continue

            dist = _taxonomic_distance(na, nb)
            if dist < 2:
                continue

            type_counts = Counter(entity_info[eid]["type"] for eid in shared)
            total_union = len(node_entities[na] | node_entities[nb])
            overlap_score = len(shared) / max(total_union, 1) * math.log2(dist + 1)

            shared_list = sorted(shared, key=lambda eid: entity_info[eid]["name"])[:20]

            overlaps.append({
                "node_a_id": na,
                "node_b_id": nb,
                "shared_entity_count": len(shared),
                "shared_entity_ids": json.dumps([
                    {"id": eid, "name": entity_info[eid]["name"], "type": entity_info[eid]["type"]}
                    for eid in shared_list
                ]),
                "shared_entity_types": json.dumps(dict(type_counts)),
                "taxonomic_distance": dist,
                "overlap_score": round(overlap_score, 4),
            })

    overlaps.sort(key=lambda x: x["overlap_score"], reverse=True)
    overlaps = overlaps[:top_k]

    db.execute("DELETE FROM node_entity_overlap")
    for ov in overlaps:
        db.execute(
            """INSERT OR REPLACE INTO node_entity_overlap
               (node_a_id, node_b_id, shared_entity_count, shared_entity_ids,
                shared_entity_types, taxonomic_distance, overlap_score)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (ov["node_a_id"], ov["node_b_id"], ov["shared_entity_count"],
             ov["shared_entity_ids"], ov["shared_entity_types"],
             ov["taxonomic_distance"], ov["overlap_score"])
        )
    db.commit()
    print(f"[SIGNAL] Entity overlap: {len(overlaps)} cross-node links stored", flush=True)
    return len(overlaps)


def _tokenize(text: str) -> set[str]:
    """Extract meaningful tokens from pattern text."""
    stops = {"the", "a", "an", "in", "on", "of", "to", "for", "and", "or",
             "is", "are", "was", "were", "be", "been", "being", "with",
             "from", "at", "by", "this", "that", "it", "its", "as", "not",
             "but", "if", "than", "more", "across", "between", "when"}
    words = set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
    return words - stops


def harvest_pattern_matches(min_similarity: float = 0.45, top_k: int = 80):
    """Find convergent patterns across different taxonomy nodes.

    Two patterns from different domains describing similar phenomena signals
    a deeper structural regularity.
    """
    print("[SIGNAL] Computing convergent pattern matches...", flush=True)

    patterns = db.fetchall("""
        SELECT id, pattern_text, pattern_type, node_id, domains
        FROM patterns
        WHERE pattern_text IS NOT NULL AND LENGTH(pattern_text) > 20
    """)

    matches = []
    for i in range(len(patterns)):
        for j in range(i + 1, len(patterns)):
            pa, pb = patterns[i], patterns[j]
            if pa["node_id"] == pb["node_id"] and pa["node_id"] is not None:
                continue

            tokens_a = _tokenize(pa["pattern_text"])
            tokens_b = _tokenize(pb["pattern_text"])
            if not tokens_a or not tokens_b:
                continue

            shared = tokens_a & tokens_b
            jaccard = len(shared) / len(tokens_a | tokens_b)
            seq_sim = SequenceMatcher(None, pa["pattern_text"].lower(),
                                      pb["pattern_text"].lower()).ratio()
            score = 0.4 * jaccard + 0.6 * seq_sim

            if score < min_similarity:
                continue

            matches.append({
                "pattern_a_id": pa["id"],
                "pattern_b_id": pb["id"],
                "similarity_score": round(score, 4),
                "node_a_id": pa.get("node_id"),
                "node_b_id": pb.get("node_id"),
                "shared_tokens": json.dumps(sorted(shared)[:15]),
            })

    matches.sort(key=lambda x: x["similarity_score"], reverse=True)
    matches = matches[:top_k]

    db.execute("DELETE FROM pattern_matches")
    for m in matches:
        db.execute(
            """INSERT OR REPLACE INTO pattern_matches
               (pattern_a_id, pattern_b_id, similarity_score, node_a_id, node_b_id, shared_tokens)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (m["pattern_a_id"], m["pattern_b_id"], m["similarity_score"],
             m["node_a_id"], m["node_b_id"], m["shared_tokens"])
        )
    db.commit()
    print(f"[SIGNAL] Pattern matches: {len(matches)} convergent pairs stored", flush=True)
    return len(matches)


def harvest_contradiction_clusters(min_cluster_size: int = 2):
    """Group contradictions by shared entities to find systemic conflicts.

    A cluster of contradictions around the same method/dataset suggests
    a deeper methodological problem, not just noise.
    """
    print("[SIGNAL] Clustering contradictions by shared entities...", flush=True)

    contras = db.fetchall("""
        SELECT c.id, c.description, c.hypothesis,
               ca.method_name as method_a, ca.dataset_name as dataset_a,
               ca.metric_name as metric_a, ca.paper_id as paper_a,
               cb.method_name as method_b, cb.dataset_name as dataset_b,
               cb.metric_name as metric_b, cb.paper_id as paper_b
        FROM contradictions c
        JOIN claims ca ON c.claim_a_id = ca.id
        JOIN claims cb ON c.claim_b_id = cb.id
    """)

    entity_to_contras = defaultdict(set)
    contra_entities = {}

    for c in contras:
        entities = set()
        for field in ["method_a", "method_b", "dataset_a", "dataset_b", "metric_a", "metric_b"]:
            val = c.get(field)
            if val and val.strip():
                entities.add(val.strip().lower())
        contra_entities[c["id"]] = entities
        for ent in entities:
            entity_to_contras[ent].add(c["id"])

    visited = set()
    clusters = []

    for contra in contras:
        if contra["id"] in visited:
            continue

        cluster_ids = set()
        queue = [contra["id"]]
        while queue:
            cid = queue.pop()
            if cid in cluster_ids:
                continue
            cluster_ids.add(cid)
            visited.add(cid)
            for ent in contra_entities.get(cid, set()):
                for linked_cid in entity_to_contras.get(ent, set()):
                    if linked_cid not in cluster_ids:
                        queue.append(linked_cid)

        if len(cluster_ids) < min_cluster_size:
            continue

        all_ents = set()
        all_nodes = set()
        for cid in cluster_ids:
            all_ents.update(contra_entities.get(cid, set()))
            for c in contras:
                if c["id"] == cid:
                    for pid in [c["paper_a"], c["paper_b"]]:
                        nodes = db.fetchall(
                            "SELECT node_id FROM paper_taxonomy WHERE paper_id=?", (pid,))
                        for n in nodes:
                            all_nodes.add(n["node_id"])

        theme_parts = sorted(all_ents, key=lambda e: sum(
            1 for cid in cluster_ids if e in contra_entities.get(cid, set())
        ), reverse=True)[:3]
        theme = " / ".join(theme_parts)

        clusters.append({
            "theme": theme,
            "contradiction_ids": json.dumps(sorted(cluster_ids)),
            "shared_entities": json.dumps(sorted(all_ents)[:20]),
            "cluster_size": len(cluster_ids),
            "node_ids": json.dumps(sorted(all_nodes)[:10]),
        })

    db.execute("DELETE FROM contradiction_clusters")
    for cl in clusters:
        db.execute(
            """INSERT INTO contradiction_clusters
               (theme, contradiction_ids, shared_entities, cluster_size, node_ids)
               VALUES (?, ?, ?, ?, ?)""",
            (cl["theme"], cl["contradiction_ids"], cl["shared_entities"],
             cl["cluster_size"], cl["node_ids"])
        )
    db.commit()
    print(f"[SIGNAL] Contradiction clusters: {len(clusters)} clusters stored", flush=True)
    return len(clusters)


def harvest_performance_plateaus(max_spread_pct: float = 3.0, min_methods: int = 4):
    """Detect taxonomy nodes where top methods have converged (diminishing returns).

    A plateau = top N methods on the same dataset/metric are within X% of each other.
    """
    print("[SIGNAL] Detecting performance plateaus...", flush=True)

    groups = db.fetchall("""
        SELECT r.node_id, r.dataset_name, r.metric_name,
               GROUP_CONCAT(DISTINCT r.method_name) as methods,
               COUNT(DISTINCT r.method_name) as method_count,
               COUNT(DISTINCT r.paper_id) as paper_count,
               MAX(r.metric_value) as max_val,
               MIN(r.metric_value) as min_val
        FROM results r
        WHERE r.node_id IS NOT NULL
          AND r.metric_value IS NOT NULL
          AND r.metric_value > 0
        GROUP BY r.node_id, r.dataset_name, r.metric_name
        HAVING method_count >= ?
    """, (min_methods,))

    plateaus = []
    for g in groups:
        if g["max_val"] is None or g["max_val"] == 0:
            continue

        top_results = db.fetchall("""
            SELECT DISTINCT method_name, metric_value
            FROM results
            WHERE node_id = ? AND dataset_name = ? AND metric_name = ?
              AND metric_value IS NOT NULL
            ORDER BY CAST(metric_value AS REAL) DESC
            LIMIT 5
        """, (g["node_id"], g["dataset_name"], g["metric_name"]))

        if len(top_results) < min_methods:
            continue

        values = []
        for r in top_results:
            try:
                values.append(float(r["metric_value"]))
            except (ValueError, TypeError):
                pass
        if not values:
            continue

        spread = max(values) - min(values)
        spread_pct = (spread / max(abs(max(values)), 1e-9)) * 100

        if spread_pct > max_spread_pct:
            continue

        top_methods = []
        for r in top_results:
            try:
                top_methods.append({"method": r["method_name"], "value": float(r["metric_value"])})
            except (ValueError, TypeError):
                pass

        plateaus.append({
            "node_id": g["node_id"],
            "dataset_name": g["dataset_name"],
            "metric_name": g["metric_name"],
            "top_methods": json.dumps(top_methods),
            "spread": round(spread, 6),
            "spread_pct": round(spread_pct, 4),
            "method_count": g["method_count"],
            "paper_count": g["paper_count"],
        })

    db.execute("DELETE FROM performance_plateaus")
    for p in plateaus:
        db.execute(
            """INSERT OR REPLACE INTO performance_plateaus
               (node_id, dataset_name, metric_name, top_methods,
                spread, spread_pct, method_count, paper_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (p["node_id"], p["dataset_name"], p["metric_name"], p["top_methods"],
             p["spread"], p["spread_pct"], p["method_count"], p["paper_count"])
        )
    db.commit()
    print(f"[SIGNAL] Performance plateaus: {len(plateaus)} detected", flush=True)
    return len(plateaus)


def harvest_all() -> dict:
    """Run all signal harvesting stages. Returns counts."""
    stats = {}
    stats["entity_overlaps"] = harvest_entity_overlap()
    stats["pattern_matches"] = harvest_pattern_matches()
    stats["contradiction_clusters"] = harvest_contradiction_clusters()
    stats["performance_plateaus"] = harvest_performance_plateaus()
    print(f"[SIGNAL] Harvest complete: {stats}", flush=True)
    return stats


def get_tier1_signals(top_overlaps: int = 20, top_patterns: int = 15) -> dict:
    """Assemble signals for Tier 1 (Paradigm Discovery) agent."""
    overlaps = db.fetchall(
        """SELECT * FROM node_entity_overlap
           ORDER BY overlap_score DESC LIMIT ?""", (top_overlaps,))

    pattern_ms = db.fetchall("""
        SELECT pm.*, pa.pattern_text as text_a, pa.pattern_type as type_a,
               pb.pattern_text as text_b, pb.pattern_type as type_b
        FROM pattern_matches pm
        JOIN patterns pa ON pm.pattern_a_id = pa.id
        JOIN patterns pb ON pm.pattern_b_id = pb.id
        ORDER BY pm.similarity_score DESC LIMIT ?
    """, (top_patterns,))

    clusters = db.fetchall(
        "SELECT * FROM contradiction_clusters WHERE cluster_size >= 2 ORDER BY cluster_size DESC")

    taxonomy = db.fetchall("""
        SELECT t.id, t.name, t.parent_id, t.depth,
               COUNT(DISTINCT pt.paper_id) as paper_count
        FROM taxonomy_nodes t
        LEFT JOIN paper_taxonomy pt ON pt.node_id = t.id
        GROUP BY t.id
        ORDER BY t.depth, t.sort_order
    """)

    return {
        "entity_overlaps": overlaps,
        "pattern_matches": pattern_ms,
        "contradiction_clusters": clusters,
        "taxonomy_map": taxonomy,
    }


def get_tier2_signals(
    *,
    plateau_limit: int = 20,
    limitation_node_limit: int = 15,
) -> dict:
    """Assemble signals for Tier 2 (Paper-Ready Ideas) agent."""
    clusters = db.fetchall(
        "SELECT * FROM contradiction_clusters ORDER BY cluster_size DESC")

    plateaus = db.fetchall(
        "SELECT * FROM performance_plateaus ORDER BY method_count DESC LIMIT ?",
        (plateau_limit,),
    )

    limitation_clusters = db.fetchall(
        """
        SELECT node_id, COUNT(*) as lim_count,
               GROUP_CONCAT(paper_id) as paper_ids
        FROM (
            SELECT pt.node_id, pi.paper_id
            FROM paper_insights pi
            JOIN paper_taxonomy pt ON pt.paper_id = pi.paper_id
            WHERE pi.limitations IS NOT NULL AND pi.limitations != '[]'
        )
        GROUP BY node_id
        HAVING lim_count >= 3
        ORDER BY lim_count DESC
        LIMIT ?
    """,
        (limitation_node_limit,),
    )

    high_insights = db.fetchall("""
        SELECT * FROM insights
        WHERE novelty_score >= 4 AND paradigm_score >= 6
        ORDER BY paradigm_score DESC
        LIMIT 10
    """)

    return {
        "contradiction_clusters": clusters,
        "performance_plateaus": plateaus,
        "limitation_clusters": limitation_clusters,
        "high_potential_insights": high_insights,
    }
