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
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from contracts import DiscoverySignalBundle
from db import database as db
from db.insight_outcomes import record_harvester_run


PROTOCOL_ARTIFACT_KEYWORDS = {
    "annotation_scheme": {"annotation", "label", "labeled", "labeling", "adjudication"},
    "metric_choice": {"metric", "metrics", "f1", "exact match", "accuracy", "judge", "scoring"},
    "benchmark_protocol": {"benchmark", "protocol", "prompt format", "evaluation setup", "single-turn", "multi-turn"},
    "temporal_window": {"temporal", "window", "24h", "7 days", "30 days", "verification window"},
}

NEGATIVE_SPACE_KEYWORDS = {
    "missing_ablation": {"ablation", "ablate"},
    "reproducibility_gap": {"reproduc", "replicate", "seed", "implementation detail"},
    "failure_boundary": {"failure", "fails", "break", "boundary", "edge case", "stress test"},
    "robustness_gap": {"robust", "distribution shift", "ood", "generalize", "generalization"},
}

MECHANISM_GAP_KEYWORDS = {
    "mechanism", "causal", "why", "unclear", "under-specified", "not explained", "interpret", "analysis"
}


def _json_list(value: str | None) -> list:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _norm_text_list(value: str | None) -> list[str]:
    items = []
    for item in _json_list(value):
        if isinstance(item, str):
            text = item.strip()
            if text:
                items.append(text)
    return items


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
        GROUP BY pem.node_id, pem.entity_id, ge.canonical_name, ge.entity_type
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
            """INSERT INTO node_entity_overlap
               (node_a_id, node_b_id, shared_entity_count, shared_entity_ids,
                shared_entity_types, taxonomic_distance, overlap_score)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (node_a_id, node_b_id) DO UPDATE SET
                 shared_entity_count = excluded.shared_entity_count,
                 shared_entity_ids = excluded.shared_entity_ids,
                 shared_entity_types = excluded.shared_entity_types,
                 taxonomic_distance = excluded.taxonomic_distance,
                 overlap_score = excluded.overlap_score""",
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
            """INSERT INTO pattern_matches
               (pattern_a_id, pattern_b_id, similarity_score, node_a_id, node_b_id, shared_tokens)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT (pattern_a_id, pattern_b_id) DO UPDATE SET
                 similarity_score = excluded.similarity_score,
                 node_a_id = excluded.node_a_id,
                 node_b_id = excluded.node_b_id,
                 shared_tokens = excluded.shared_tokens""",
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
        HAVING COUNT(DISTINCT r.method_name) >= ?
    """, (min_methods,))

    plateaus = []
    for g in groups:
        if g["max_val"] is None or g["max_val"] == 0:
            continue

        top_results = db.fetchall("""
            SELECT method_name, MAX(CAST(metric_value AS REAL)) AS metric_value
            FROM results
            WHERE node_id = ? AND dataset_name = ? AND metric_name = ?
              AND metric_value IS NOT NULL
            GROUP BY method_name
            ORDER BY metric_value DESC
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
            """INSERT INTO performance_plateaus
               (node_id, dataset_name, metric_name, top_methods,
                spread, spread_pct, method_count, paper_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (node_id, dataset_name, metric_name) DO UPDATE SET
                 top_methods = excluded.top_methods,
                 spread = excluded.spread,
                 spread_pct = excluded.spread_pct,
                 method_count = excluded.method_count,
                 paper_count = excluded.paper_count""",
            (p["node_id"], p["dataset_name"], p["metric_name"], p["top_methods"],
             p["spread"], p["spread_pct"], p["method_count"], p["paper_count"])
        )
    db.commit()
    print(f"[SIGNAL] Performance plateaus: {len(plateaus)} detected", flush=True)
    return len(plateaus)


def harvest_protocol_artifacts(min_support: int = 2):
    """Mine protocol-related limitations and open questions."""
    rows = db.fetchall(
        """
        SELECT pt.node_id, pi.paper_id, pi.limitations, pi.open_questions
        FROM paper_insights pi
        JOIN paper_taxonomy pt ON pt.paper_id = pi.paper_id
        """
    )

    buckets: dict[tuple[str, str], dict] = {}
    for row in rows:
        texts = _norm_text_list(row.get("limitations")) + _norm_text_list(row.get("open_questions"))
        joined = " ".join(texts).lower()
        if not joined:
            continue
        for artifact_type, keywords in PROTOCOL_ARTIFACT_KEYWORDS.items():
            if not any(keyword in joined for keyword in keywords):
                continue
            key = (row["node_id"], artifact_type)
            bucket = buckets.setdefault(
                key,
                {"node_id": row["node_id"], "artifact_type": artifact_type, "paper_ids": set(), "snippets": []},
            )
            bucket["paper_ids"].add(row["paper_id"])
            bucket["snippets"].extend(texts[:2])

    db.execute("DELETE FROM protocol_artifacts")
    count = 0
    for bucket in buckets.values():
        support_count = len(bucket["paper_ids"])
        if support_count < min_support:
            continue
        summary = "; ".join(bucket["snippets"][:3])[:500]
        db.execute(
            """INSERT INTO protocol_artifacts
               (node_id, artifact_type, summary, paper_ids, support_count)
               VALUES (?, ?, ?, ?, ?)""",
            (
                bucket["node_id"],
                bucket["artifact_type"],
                summary,
                json.dumps(sorted(bucket["paper_ids"])),
                support_count,
            ),
        )
        count += 1
    db.commit()
    return count


def harvest_negative_space_gaps(min_support: int = 2):
    """Find repeated mentions of missing controls, robustness, or reproducibility."""
    rows = db.fetchall(
        """
        SELECT pt.node_id, pi.paper_id, pi.limitations, pi.open_questions
        FROM paper_insights pi
        JOIN paper_taxonomy pt ON pt.paper_id = pi.paper_id
        """
    )

    buckets: dict[tuple[str, str], dict] = {}
    for row in rows:
        texts = _norm_text_list(row.get("limitations")) + _norm_text_list(row.get("open_questions"))
        joined = " ".join(texts).lower()
        if not joined:
            continue
        for gap_type, keywords in NEGATIVE_SPACE_KEYWORDS.items():
            if not any(keyword in joined for keyword in keywords):
                continue
            key = (row["node_id"], gap_type)
            bucket = buckets.setdefault(
                key,
                {"node_id": row["node_id"], "gap_type": gap_type, "paper_ids": set(), "snippets": []},
            )
            bucket["paper_ids"].add(row["paper_id"])
            bucket["snippets"].extend(texts[:2])

    db.execute("DELETE FROM negative_space_gaps")
    count = 0
    for bucket in buckets.values():
        support_count = len(bucket["paper_ids"])
        if support_count < min_support:
            continue
        summary = "; ".join(bucket["snippets"][:3])[:500]
        db.execute(
            """INSERT INTO negative_space_gaps
               (node_id, gap_type, summary, paper_ids, support_count)
               VALUES (?, ?, ?, ?, ?)""",
            (
                bucket["node_id"],
                bucket["gap_type"],
                summary,
                json.dumps(sorted(bucket["paper_ids"])),
                support_count,
            ),
        )
        count += 1
    db.commit()
    return count


def harvest_claim_method_gaps(min_support: int = 2):
    """Identify nodes with strong claims but weak mechanistic explanation."""
    rows = db.fetchall(
        """
        SELECT pt.node_id,
               pt.paper_id,
               MAX(pi.limitations) AS limitations,
               MAX(pi.problem_statement) AS problem_statement,
               COUNT(DISTINCT c.id) AS claim_count,
               COUNT(DISTINCT r.id) AS result_count
        FROM paper_taxonomy pt
        LEFT JOIN paper_insights pi ON pi.paper_id = pt.paper_id
        LEFT JOIN claims c ON c.paper_id = pt.paper_id
        LEFT JOIN results r ON r.paper_id = pt.paper_id
        GROUP BY pt.node_id, pt.paper_id
        """
    )

    buckets: dict[str, dict] = {}
    for row in rows:
        if (row.get("claim_count") or 0) < 2 and (row.get("result_count") or 0) < 2:
            continue
        texts = _norm_text_list(row.get("limitations"))
        joined = " ".join(texts + [row.get("problem_statement") or ""]).lower()
        if not joined or not any(keyword in joined for keyword in MECHANISM_GAP_KEYWORDS):
            continue
        bucket = buckets.setdefault(
            row["node_id"],
            {"node_id": row["node_id"], "paper_ids": set(), "snippets": [], "support_count": 0},
        )
        bucket["paper_ids"].add(row["paper_id"])
        bucket["snippets"].extend(texts[:2])
        bucket["support_count"] += 1

    db.execute("DELETE FROM claim_method_gaps")
    count = 0
    for bucket in buckets.values():
        if bucket["support_count"] < min_support:
            continue
        summary = "; ".join(bucket["snippets"][:3])[:500] or "Strong claims/results but missing mechanism-oriented evidence."
        db.execute(
            """INSERT INTO claim_method_gaps
               (node_id, summary, paper_ids, support_count)
               VALUES (?, ?, ?, ?)""",
            (bucket["node_id"], summary, json.dumps(sorted(bucket["paper_ids"])), bucket["support_count"]),
        )
        count += 1
    db.commit()
    return count


def harvest_mechanism_mismatches(min_variants: int = 2):
    """Cluster contradictory claims that offer distinct explanations."""
    clusters = db.fetchall("SELECT * FROM contradiction_clusters ORDER BY cluster_size DESC")
    db.execute("DELETE FROM mechanism_mismatches")
    count = 0
    for cluster in clusters:
        contradiction_ids = _json_list(cluster.get("contradiction_ids"))
        node_ids = _json_list(cluster.get("node_ids"))
        variants = set()
        paper_ids = set()
        for cid in contradiction_ids:
            row = db.fetchone(
                """
                SELECT c.hypothesis, ca.paper_id AS paper_a, cb.paper_id AS paper_b
                FROM contradictions c
                JOIN claims ca ON c.claim_a_id = ca.id
                JOIN claims cb ON c.claim_b_id = cb.id
                WHERE c.id=?
                """,
                (cid,),
            )
            if not row:
                continue
            if row.get("hypothesis"):
                variants.add(row["hypothesis"].strip())
            if row.get("paper_a"):
                paper_ids.add(row["paper_a"])
            if row.get("paper_b"):
                paper_ids.add(row["paper_b"])
        if len(variants) < min_variants:
            continue
        for node_id in node_ids or [None]:
            db.execute(
                """INSERT INTO mechanism_mismatches
                   (node_id, theme, explanation_variants, paper_ids, support_count)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    node_id,
                    cluster["theme"],
                    json.dumps(sorted(variants)[:6]),
                    json.dumps(sorted(paper_ids)),
                    len(variants),
                ),
            )
            count += 1
    db.commit()
    return count


def harvest_hidden_variable_bridges(min_score: float = 1.5):
    """Find distant nodes sharing protocol/gap factors more than entities."""
    factor_rows = db.fetchall(
        """
        SELECT node_id, artifact_type AS factor
        FROM protocol_artifacts
        UNION ALL
        SELECT node_id, gap_type AS factor
        FROM negative_space_gaps
        """
    )
    node_factors: dict[str, set[str]] = defaultdict(set)
    for row in factor_rows:
        if row["node_id"] and row["factor"]:
            node_factors[row["node_id"]].add(row["factor"])

    overlap_lookup = {
        tuple(sorted((row["node_a_id"], row["node_b_id"]))): row.get("overlap_score", 0)
        for row in db.fetchall("SELECT node_a_id, node_b_id, overlap_score FROM node_entity_overlap")
    }

    db.execute("DELETE FROM hidden_variable_bridges")
    count = 0
    nodes = sorted(node_factors)
    for i, node_a in enumerate(nodes):
        for node_b in nodes[i + 1:]:
            shared = node_factors[node_a] & node_factors[node_b]
            if not shared:
                continue
            overlap_score = overlap_lookup.get(tuple(sorted((node_a, node_b))), 0)
            score = len(shared) * (1.0 if overlap_score < 0.12 else 0.4)
            if score < min_score:
                continue
            for factor in sorted(shared):
                db.execute(
                    """INSERT INTO hidden_variable_bridges
                       (node_a_id, node_b_id, shared_factor, paper_ids, score)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT (node_a_id, node_b_id, shared_factor) DO UPDATE SET
                         paper_ids = excluded.paper_ids,
                         score = excluded.score""",
                    (node_a, node_b, factor, json.dumps([]), round(score, 4)),
                )
                count += 1
    db.commit()
    return count


def _timed_harvest(name: str, fn) -> int:
    t0 = time.perf_counter()
    try:
        count = int(fn())
    except Exception as e:
        ms = int((time.perf_counter() - t0) * 1000)
        try:
            db.rollback()
        except Exception:
            pass
        record_harvester_run(name, 0, ms, meta={"error": str(e)})
        raise
    ms = int((time.perf_counter() - t0) * 1000)
    record_harvester_run(name, max(0, count), ms)
    return count


def harvest_all() -> dict:
    """Run all signal harvesting stages. Returns counts."""
    stats = {}
    stats["entity_overlaps"] = _timed_harvest("entity_overlap", harvest_entity_overlap)
    stats["pattern_matches"] = _timed_harvest("pattern_match", harvest_pattern_matches)
    stats["contradiction_clusters"] = _timed_harvest("contradiction_cluster", harvest_contradiction_clusters)
    stats["performance_plateaus"] = _timed_harvest("performance_plateau", harvest_performance_plateaus)
    stats["mechanism_mismatches"] = _timed_harvest("mechanism_mismatch", harvest_mechanism_mismatches)
    stats["protocol_artifacts"] = _timed_harvest("protocol_artifact", harvest_protocol_artifacts)
    stats["negative_space_gaps"] = _timed_harvest("negative_space_gap", harvest_negative_space_gaps)
    stats["hidden_variable_bridges"] = _timed_harvest("hidden_variable_bridge", harvest_hidden_variable_bridges)
    stats["claim_method_gaps"] = _timed_harvest("claim_method_gap", harvest_claim_method_gaps)
    print(f"[SIGNAL] Harvest complete: {stats}", flush=True)
    return stats


def get_tier1_signals(top_overlaps: int = 20, top_patterns: int = 15) -> DiscoverySignalBundle:
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
        GROUP BY t.id, t.name, t.parent_id, t.depth, t.sort_order
        ORDER BY t.depth, t.sort_order
    """)

    payload = {
        "entity_overlaps": overlaps,
        "pattern_matches": pattern_ms,
        "contradiction_clusters": clusters,
        "protocol_artifacts": db.fetchall("SELECT * FROM protocol_artifacts ORDER BY support_count DESC LIMIT 10"),
        "hidden_variable_bridges": db.fetchall("SELECT * FROM hidden_variable_bridges ORDER BY score DESC LIMIT 10"),
        "claim_method_gaps": db.fetchall("SELECT * FROM claim_method_gaps ORDER BY support_count DESC LIMIT 10"),
        "taxonomy_map": taxonomy,
    }
    return DiscoverySignalBundle.from_payload(
        tier=1,
        stage="signal_harvest",
        payload=payload,
        metadata={
            "entity_overlap_count": len(overlaps),
            "pattern_match_count": len(pattern_ms),
            "contradiction_cluster_count": len(clusters),
        },
    )


def get_tier2_signals(
    *,
    plateau_limit: int = 20,
    limitation_node_limit: int = 15,
) -> DiscoverySignalBundle:
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
        HAVING COUNT(*) >= 3
        ORDER BY lim_count DESC
        LIMIT ?
    """,
        (limitation_node_limit,),
    )

    try:
        high_insights = db.fetchall(
            """
            SELECT id, title, mechanism_type, evidence_packet, adversarial_score,
                   evidence_summary, experimental_plan, signal_mix, resource_class
            FROM deep_insights
            WHERE tier=1
            ORDER BY COALESCE(adversarial_score, 0) DESC, created_at DESC
            LIMIT 10
            """
        )
    except Exception:
        high_insights = []

    if not high_insights:
        insight_cols = db.column_names("insights")
        if "paradigm_score" in insight_cols:
            high_insights = db.fetchall(
                """
                SELECT * FROM insights
                WHERE novelty_score >= 4 AND paradigm_score >= 6
                ORDER BY paradigm_score DESC
                LIMIT 10
                """
            )
        else:
            high_insights = db.fetchall(
                """
                SELECT id, title, novelty_score, feasibility_score, evidence
                FROM insights
                WHERE novelty_score >= 4
                ORDER BY novelty_score DESC, feasibility_score DESC
                LIMIT 10
                """
            )

    payload = {
        "contradiction_clusters": clusters,
        "performance_plateaus": plateaus,
        "limitation_clusters": limitation_clusters,
        "high_potential_insights": high_insights,
        "mechanism_mismatches": db.fetchall("SELECT * FROM mechanism_mismatches ORDER BY support_count DESC LIMIT 15"),
        "protocol_artifacts": db.fetchall("SELECT * FROM protocol_artifacts ORDER BY support_count DESC LIMIT 15"),
        "negative_space_gaps": db.fetchall("SELECT * FROM negative_space_gaps ORDER BY support_count DESC LIMIT 15"),
        "hidden_variable_bridges": db.fetchall("SELECT * FROM hidden_variable_bridges ORDER BY score DESC LIMIT 15"),
        "claim_method_gaps": db.fetchall("SELECT * FROM claim_method_gaps ORDER BY support_count DESC LIMIT 15"),
    }
    return DiscoverySignalBundle.from_payload(
        tier=2,
        stage="signal_harvest",
        payload=payload,
        metadata={
            "contradiction_cluster_count": len(clusters),
            "plateau_count": len(plateaus),
            "limitation_cluster_count": len(limitation_clusters),
            "high_potential_insight_count": len(high_insights),
        },
    )
