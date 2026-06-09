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

GENERIC_BRIDGE_FACTORS = {
    "protocol:annotation_scheme",
    "protocol:metric_choice",
    "protocol:benchmark_protocol",
    "protocol:temporal_window",
    "gap:missing_ablation",
    "gap:reproducibility_gap",
    "gap:failure_boundary",
    "gap:robustness_gap",
}

FINE_PROTOCOL_RULES = [
    {
        "factor": "method_generation_to_perception_interface",
        "groups": [
            {"perception", "visual understanding", "vision task", "vision tasks"},
            {"image generation", "generated image", "generated images", "generative model"},
        ],
    },
    {
        "factor": "method_rgb_output_decoding",
        "groups": [
            {"rgb image", "rgb images", "color-coded", "colored", "visualization", "heatmap"},
            {"decoded", "decoding", "output", "outputs", "generated image", "generated images"},
        ],
    },
    {
        "factor": "method_prompt_conditioned_task_interface",
        "groups": [
            {"prompt", "prompts", "prompt-conditioned", "prompt conditioning"},
            {"category", "categories", "class", "classes", "color mapping", "task changes", "specified"},
        ],
    },
    {
        "factor": "method_low_ratio_mixed_instruction_tuning",
        "groups": [
            {"low ratio", "small amount", "mixture", "mixed", "mixing"},
            {"instruction tuning", "instruction-tuning", "training mixture"},
        ],
    },
    {
        "factor": "method_generative_capability_preservation",
        "groups": [
            {"preserve", "preserves", "maintain", "maintains", "retains", "without sacrificing", "catastrophic"},
            {"generation", "generative", "text-to-image", "image editing"},
        ],
    },
    {
        "factor": "method_benchmark_exclusion_protocol",
        "groups": [
            {"no evaluation benchmark", "not include", "excluded", "without evaluation"},
            {"training data", "benchmark", "benchmarks"},
        ],
    },
    {
        "factor": "method_unified_prompt_task_switching",
        "groups": [
            {"single unified model", "shared weights", "same model", "unified model"},
            {"prompt", "prompts", "task changes", "controlled"},
        ],
    },
    {
        "factor": "method_generated_output_metric_adapter",
        "groups": [
            {"metric", "metrics", "quantitative", "evaluation"},
            {"decoded", "decoding", "generated image", "generated images", "rgb output", "rgb outputs"},
        ],
    },
]

NEGATIVE_SPACE_KEYWORDS = {
    "missing_ablation": {"ablation", "ablate"},
    "reproducibility_gap": {"reproduc", "replicate", "seed", "implementation detail"},
    "failure_boundary": {"failure", "fails", "break", "boundary", "edge case", "stress test"},
    "robustness_gap": {"robust", "distribution shift", "ood", "generalize", "generalization"},
}

MECHANISM_GAP_KEYWORDS = {
    "mechanism", "causal", "why", "unclear", "under-specified", "not explained", "interpret", "analysis"
}

LOWER_IS_BETTER_METRICS = {
    "error", "loss", "wer", "cer", "perplexity", "latency", "time", "cost",
    "mae", "mse", "rmse", "distance", "epe", "fréchet", "frechet", "fid",
    "memory", "flops", "parameters",
}

POSITIVE_CLAIM_TERMS = {
    "improve", "improves", "improved", "outperform", "outperforms", "achieve",
    "achieves", "achieved", "state-of-the-art", "sota", "robust", "generalize",
    "generalizes", "competitive", "reduces", "reduced", "better", "effective",
    "successful", "significant improvement",
}

NEGATIVE_CLAIM_TERMS = {
    "fail", "fails", "failure", "worse", "degrade", "degrades", "collapse",
    "no significant", "does not", "do not", "limited", "limitation", "cannot",
    "constrained", "underperform", "not evaluate", "not evaluated", "lack",
    "lacks", "missing", "brittle", "only modest", "insufficient",
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


def _ensure_column(table: str, column: str, column_type: str) -> None:
    cols = {row["name"] for row in db.fetchall(f"PRAGMA table_info({table})")}
    if column not in cols:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
        db.commit()


def _metric_lower_is_better(metric_name: str | None) -> bool:
    metric = (metric_name or "").lower()
    return any(term in metric for term in LOWER_IS_BETTER_METRICS)


def _relative_spread(high: float, low: float) -> float:
    return abs(high - low) / max(abs(high), abs(low), 1e-9)


def _norm_key(value: str | None) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _claim_polarity(text: str | None, claim_type: str | None = None) -> str | None:
    haystack = f"{claim_type or ''} {text or ''}".lower()
    pos = sum(1 for term in POSITIVE_CLAIM_TERMS if term in haystack)
    neg = sum(1 for term in NEGATIVE_CLAIM_TERMS if term in haystack)
    if neg and not pos:
        return "negative"
    if pos and not neg:
        return "positive"
    if neg > pos:
        return "negative"
    if pos > neg:
        return "positive"
    return None


def _matched_fine_protocol_factors(text: str | None) -> list[str]:
    haystack = (text or "").lower()
    if not haystack:
        return []
    factors = []
    for rule in FINE_PROTOCOL_RULES:
        if all(any(term in haystack for term in group) for group in rule["groups"]):
            factors.append(rule["factor"])
    return factors


def _slug_factor(text: str | None, prefix: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower()).strip("_")
    slug = slug[:80].strip("_")
    return f"{prefix}_{slug}" if slug else prefix


def _primary_paper_nodes() -> dict[str, str]:
    rows = db.fetchall(
        """
        SELECT paper_id, node_id, confidence
        FROM paper_taxonomy
        ORDER BY paper_id, confidence DESC, node_id
        """
    )
    primary = {}
    for row in rows:
        primary.setdefault(row["paper_id"], row["node_id"])
    return primary


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


def harvest_pattern_matches(min_similarity: float = 0.45, top_k: int = 80, max_candidates: int = 50000):
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

    tokenized = []
    token_df = Counter()
    for idx, pattern in enumerate(patterns):
        tokens = _tokenize(pattern["pattern_text"])
        if not tokens:
            continue
        tokenized.append((idx, pattern, tokens))
        token_df.update(tokens)

    max_df = max(8, int(len(tokenized) * 0.12))
    token_index: dict[str, list[int]] = defaultdict(list)
    for idx, _pattern, tokens in tokenized:
        for token in tokens:
            if 2 <= token_df[token] <= max_df:
                token_index[token].append(idx)

    candidate_pairs: Counter[tuple[int, int]] = Counter()
    for ids in token_index.values():
        ids = sorted(set(ids))
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                candidate_pairs[(a, b)] += 1

    ranked_pairs = sorted(candidate_pairs.items(), key=lambda item: item[1], reverse=True)[:max_candidates]
    pattern_by_idx = {idx: (pattern, tokens) for idx, pattern, tokens in tokenized}
    matches = []
    for (idx_a, idx_b), shared_seed_count in ranked_pairs:
        pa, tokens_a = pattern_by_idx[idx_a]
        pb, tokens_b = pattern_by_idx[idx_b]
        if pa["node_id"] == pb["node_id"] and pa["node_id"] is not None:
            continue

        shared = tokens_a & tokens_b
        if shared_seed_count < 2 and len(shared) < 2:
            continue
        jaccard = len(shared) / len(tokens_a | tokens_b)
        if jaccard < 0.04:
            continue
        seq_sim = SequenceMatcher(
            None, pa["pattern_text"].lower(), pb["pattern_text"].lower()
        ).ratio()
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
    generic_cluster_entities = {
        "accuracy", "performance", "result", "results", "score", "quality",
        "benchmark", "benchmarks", "dataset", "method", "model", "metric",
        "not applicable", "overall", "f1", "macro f1", "precision", "recall",
    }

    for c in contras:
        entities = set()
        for field in ["method_a", "method_b", "dataset_a", "dataset_b", "metric_a", "metric_b"]:
            val = c.get(field)
            if val and val.strip():
                ent = val.strip().lower()
                if len(ent) >= 3 and ent not in generic_cluster_entities:
                    entities.add(ent)
        contra_entities[c["id"]] = entities
        for ent in entities:
            entity_to_contras[ent].add(c["id"])

    max_entity_df = max(4, int(len(contras) * 0.12))
    overly_common = {ent for ent, ids in entity_to_contras.items() if len(ids) > max_entity_df}
    if overly_common:
        entity_to_contras = defaultdict(
            set,
            {ent: ids for ent, ids in entity_to_contras.items() if ent not in overly_common},
        )
        for cid, entities in list(contra_entities.items()):
            contra_entities[cid] = entities - overly_common

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


def harvest_result_contradictions(top_k: int = 200):
    """Mine implicit contradictions/tensions from comparable numeric claims.

    This is a high-recall first pass. It does not claim semantic contradiction by
    itself; it creates comparable-result tensions that downstream clustering and
    review can inspect. The key purpose is to revive the contradiction signal
    chain when explicit "A contradicts B" extraction has low recall.
    """
    print("[SIGNAL] Mining comparable-result contradictions...", flush=True)
    db.execute("DELETE FROM contradictions WHERE description LIKE '[auto_result_tension]%'")
    db.execute("DELETE FROM contradictions WHERE description LIKE '[auto_claim_polarity]%'")

    groups = db.fetchall(
        """
        SELECT dataset_name, metric_name,
               COUNT(*) AS claim_count,
               COUNT(DISTINCT paper_id) AS paper_count,
               COUNT(DISTINCT method_name) AS method_count,
               MIN(metric_value) AS min_value,
               MAX(metric_value) AS max_value
        FROM claims
        WHERE metric_value IS NOT NULL
          AND COALESCE(dataset_name, '') != ''
          AND COALESCE(metric_name, '') != ''
        GROUP BY dataset_name, metric_name
        HAVING COUNT(*) >= 2
           AND COUNT(DISTINCT paper_id) >= 2
        """
    )

    candidates = []
    for group in groups:
        try:
            min_value = float(group["min_value"])
            max_value = float(group["max_value"])
        except (TypeError, ValueError):
            continue
        spread = abs(max_value - min_value)
        rel = _relative_spread(max_value, min_value)

        # Ignore pure all-zero/non-comparable groups and tiny numeric noise.
        scale = max(abs(max_value), abs(min_value))
        min_abs = 0.03 if scale <= 1.5 else 2.0
        min_rel = 0.08 if scale > 1.5 else 0.12
        if spread < min_abs and rel < min_rel:
            continue

        rows = db.fetchall(
            """
            SELECT id, paper_id, claim_text, method_name, dataset_name, metric_name,
                   metric_value, conditions, source_node_ids
            FROM claims
            WHERE dataset_name = ?
              AND metric_name = ?
              AND metric_value IS NOT NULL
              AND COALESCE(method_name, '') != ''
            ORDER BY metric_value ASC
            """,
            (group["dataset_name"], group["metric_name"]),
        )
        if len(rows) < 2:
            continue

        lower_better = _metric_lower_is_better(group["metric_name"])
        low = rows[0]
        high = rows[-1]
        if low["paper_id"] == high["paper_id"]:
            # Prefer cross-paper tension; fall back only if no alternative exists.
            alt_low = next((r for r in rows if r["paper_id"] != high["paper_id"]), None)
            alt_high = next((r for r in reversed(rows) if r["paper_id"] != low["paper_id"]), None)
            if alt_low and abs(float(high["metric_value"]) - float(alt_low["metric_value"])) >= spread * 0.5:
                low = alt_low
            elif alt_high and abs(float(alt_high["metric_value"]) - float(low["metric_value"])) >= spread * 0.5:
                high = alt_high
            else:
                continue

        best = low if lower_better else high
        worst = high if lower_better else low
        candidates.append(
            {
                "claim_a_id": int(best["id"]),
                "claim_b_id": int(worst["id"]),
                "dataset_name": group["dataset_name"],
                "metric_name": group["metric_name"],
                "best_method": best.get("method_name") or "",
                "worst_method": worst.get("method_name") or "",
                "best_value": float(best["metric_value"]),
                "worst_value": float(worst["metric_value"]),
                "spread": spread,
                "relative_spread": rel,
                "paper_count": int(group["paper_count"] or 0),
                "method_count": int(group["method_count"] or 0),
                "lower_better": lower_better,
            }
        )

    candidates.sort(
        key=lambda c: (
            c["relative_spread"],
            c["paper_count"],
            c["method_count"],
            c["spread"],
        ),
        reverse=True,
    )
    inserted = 0
    seen_pairs: set[tuple[int, int]] = set()
    for c in candidates[:top_k]:
        pair = tuple(sorted((c["claim_a_id"], c["claim_b_id"])))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        direction = "lower-is-better" if c["lower_better"] else "higher-is-better"
        description = (
            "[auto_result_tension] Comparable numeric claims disagree on "
            f"{c['dataset_name']} / {c['metric_name']} ({direction}): "
            f"{c['best_method']}={c['best_value']} vs {c['worst_method']}={c['worst_value']} "
            f"(spread={c['spread']:.4g}, rel={c['relative_spread']:.2%})."
        )
        condition_diff = json.dumps(
            {
                "dataset_name": c["dataset_name"],
                "metric_name": c["metric_name"],
                "direction": direction,
                "spread": round(c["spread"], 6),
                "relative_spread": round(c["relative_spread"], 4),
                "paper_count": c["paper_count"],
                "method_count": c["method_count"],
                "source": "claims.metric_value weak comparable-result miner",
            },
            ensure_ascii=False,
        )
        hypothesis = (
            f"Result tension on {c['metric_name']} suggests hidden setup, protocol, "
            "data split, implementation, or method-family variables affect the claim."
        )
        db.execute(
            """
            INSERT INTO contradictions
              (claim_a_id, claim_b_id, description, condition_diff, hypothesis, severity)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                c["claim_a_id"],
                c["claim_b_id"],
                description,
                condition_diff,
                hypothesis,
                "medium" if c["relative_spread"] >= 0.2 else "low",
            ),
        )
        inserted += 1

    rows = db.fetchall(
        """
        SELECT id, paper_id, claim_text, claim_type, method_name, dataset_name,
               metric_name, metric_value, conditions, source_node_ids
        FROM claims
        WHERE claim_text IS NOT NULL
          AND LENGTH(claim_text) > 20
          AND (
            COALESCE(method_name, '') != ''
            OR COALESCE(dataset_name, '') != ''
            OR COALESCE(metric_name, '') != ''
          )
        """
    )
    buckets: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: {"positive": [], "negative": []})
    for row in rows:
        polarity = _claim_polarity(row.get("claim_text"), row.get("claim_type"))
        if polarity not in {"positive", "negative"}:
            continue
        keys = []
        method_key = _norm_key(row.get("method_name"))
        dataset_key = _norm_key(row.get("dataset_name"))
        metric_key = _norm_key(row.get("metric_name"))
        if len(method_key) >= 4:
            keys.append(("method", method_key))
        if dataset_key and metric_key:
            keys.append(("dataset_metric", f"{dataset_key} / {metric_key}"))
        elif dataset_key:
            keys.append(("dataset", dataset_key))
        for key in keys:
            buckets[key][polarity].append(row)

    qualitative_candidates = []
    for (key_type, key_text), bucket in buckets.items():
        positives = bucket["positive"]
        negatives = bucket["negative"]
        if not positives or not negatives:
            continue
        pair_count = 0
        for pos in positives[:8]:
            for neg in negatives[:8]:
                if pos["id"] == neg["id"]:
                    continue
                cross_paper = pos["paper_id"] != neg["paper_id"]
                if not cross_paper and key_type != "method":
                    continue
                qualitative_candidates.append(
                    {
                        "claim_a_id": int(pos["id"]),
                        "claim_b_id": int(neg["id"]),
                        "key_type": key_type,
                        "key_text": key_text,
                        "paper_count": 2 if cross_paper else 1,
                        "positive_text": (pos.get("claim_text") or "")[:220],
                        "negative_text": (neg.get("claim_text") or "")[:220],
                        "severity": "medium" if cross_paper else "low",
                    }
                )
                pair_count += 1
                if pair_count >= 3:
                    break
            if pair_count >= 3:
                break

    qualitative_candidates.sort(
        key=lambda c: (c["paper_count"], 1 if c["key_type"] == "dataset_metric" else 0),
        reverse=True,
    )
    for c in qualitative_candidates[:max(0, top_k - inserted)]:
        pair = tuple(sorted((c["claim_a_id"], c["claim_b_id"])))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        description = (
            "[auto_claim_polarity] Opposing claim polarity around "
            f"{c['key_type']}={c['key_text']}: positive claim vs limitation/failure claim."
        )
        condition_diff = json.dumps(
            {
                "key_type": c["key_type"],
                "key_text": c["key_text"],
                "positive_claim": c["positive_text"],
                "negative_claim": c["negative_text"],
                "source": "claims.claim_text weak polarity miner",
            },
            ensure_ascii=False,
        )
        hypothesis = (
            f"Opposing claim polarity around {c['key_text']} suggests an unmodeled "
            "boundary condition, protocol dependency, or method-family failure mode."
        )
        db.execute(
            """
            INSERT INTO contradictions
              (claim_a_id, claim_b_id, description, condition_diff, hypothesis, severity)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                c["claim_a_id"],
                c["claim_b_id"],
                description,
                condition_diff,
                hypothesis,
                c["severity"],
            ),
        )
        inserted += 1
    db.commit()
    print(f"[SIGNAL] Comparable-result contradictions: {inserted} mined", flush=True)
    return inserted


def harvest_performance_plateaus(max_spread_pct: float = 3.0, min_methods: int = 4):
    """Detect taxonomy nodes where top methods have converged (diminishing returns).

    A plateau = top N methods on the same dataset/metric are within X% of each other.
    """
    print("[SIGNAL] Detecting performance plateaus...", flush=True)
    _ensure_column("performance_plateaus", "plateau_kind", "TEXT")
    _ensure_column("performance_plateaus", "plateau_level", "TEXT")
    _ensure_column("performance_plateaus", "max_value", "REAL")
    _ensure_column("performance_plateaus", "min_value", "REAL")

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

        max_value = max(values)
        min_value = min(values)
        high_score_scale = max_value >= 95.0 or (0.90 <= max_value <= 1.05)
        low_score_scale = max_value <= 70.0 if max_value > 1.5 else max_value <= 0.70
        unique_rounded = len({round(v, 4) for v in values})
        if high_score_scale and spread_pct <= 1.0:
            plateau_kind = "benchmark_saturation"
            plateau_level = "high"
        elif low_score_scale:
            plateau_kind = "method_bottleneck"
            plateau_level = "low"
        elif unique_rounded <= 2 or spread_pct <= 0.25:
            plateau_kind = "metric_insensitive"
            plateau_level = "mid"
        else:
            plateau_kind = "method_plateau"
            plateau_level = "mid"

        plateaus.append({
            "node_id": g["node_id"],
            "dataset_name": g["dataset_name"],
            "metric_name": g["metric_name"],
            "top_methods": json.dumps(top_methods),
            "spread": round(spread, 6),
            "spread_pct": round(spread_pct, 4),
            "method_count": g["method_count"],
            "paper_count": g["paper_count"],
            "plateau_kind": plateau_kind,
            "plateau_level": plateau_level,
            "max_value": round(max_value, 6),
            "min_value": round(min_value, 6),
        })

    db.execute("DELETE FROM performance_plateaus")
    for p in plateaus:
        db.execute(
            """INSERT INTO performance_plateaus
               (node_id, dataset_name, metric_name, top_methods,
                spread, spread_pct, method_count, paper_count,
                plateau_kind, plateau_level, max_value, min_value)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (node_id, dataset_name, metric_name) DO UPDATE SET
                 top_methods = excluded.top_methods,
                 spread = excluded.spread,
                 spread_pct = excluded.spread_pct,
                 method_count = excluded.method_count,
                 paper_count = excluded.paper_count,
                 plateau_kind = excluded.plateau_kind,
                 plateau_level = excluded.plateau_level,
                 max_value = excluded.max_value,
                 min_value = excluded.min_value""",
            (p["node_id"], p["dataset_name"], p["metric_name"], p["top_methods"],
             p["spread"], p["spread_pct"], p["method_count"], p["paper_count"],
             p["plateau_kind"], p["plateau_level"], p["max_value"], p["min_value"])
        )
    db.commit()
    print(f"[SIGNAL] Performance plateaus: {len(plateaus)} detected", flush=True)
    return len(plateaus)


def harvest_protocol_artifacts(min_support: int = 2):
    """Mine protocol-related limitations, with open questions only as context."""
    primary_nodes = _primary_paper_nodes()
    rows = db.fetchall(
        """
        SELECT pt.node_id, pi.paper_id, pi.limitations, pi.open_questions
        FROM paper_insights pi
        JOIN paper_taxonomy pt ON pt.paper_id = pi.paper_id
        """
    )

    buckets: dict[tuple[str, str], dict] = {}
    for row in rows:
        if primary_nodes.get(row["paper_id"]) != row["node_id"]:
            continue
        limitation_texts = _norm_text_list(row.get("limitations"))
        open_question_texts = _norm_text_list(row.get("open_questions"))
        joined = " ".join(limitation_texts).lower()
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
            bucket["snippets"].extend((limitation_texts + open_question_texts[:1])[:2])

    method_rows = db.fetchall(
        """
        SELECT pt.node_id,
               m.first_paper_id AS paper_id,
               m.name,
               m.description,
               m.key_innovation,
               m.builds_on
        FROM methods m
        JOIN paper_taxonomy pt ON pt.paper_id = m.first_paper_id
        WHERE m.first_paper_id IS NOT NULL
        """
    )
    for row in method_rows:
        if primary_nodes.get(row["paper_id"]) != row["node_id"]:
            continue
        text = " ".join(
            str(part or "")
            for part in [
                row.get("name"),
                row.get("description"),
                row.get("key_innovation"),
                row.get("builds_on"),
            ]
        )
        for factor in _matched_fine_protocol_factors(text):
            key = (row["node_id"], factor)
            bucket = buckets.setdefault(
                key,
                {"node_id": row["node_id"], "artifact_type": factor, "paper_ids": set(), "snippets": []},
            )
            bucket["paper_ids"].add(row["paper_id"])
            bucket["snippets"].append(text[:260])

    claim_rows = db.fetchall(
        """
        SELECT pt.node_id,
               c.paper_id,
               c.claim_text,
               c.method_name,
               c.dataset_name,
               c.metric_name
        FROM claims c
        JOIN paper_taxonomy pt ON pt.paper_id = c.paper_id
        WHERE c.claim_text IS NOT NULL
        """
    )
    for row in claim_rows:
        if primary_nodes.get(row["paper_id"]) != row["node_id"]:
            continue
        text = " ".join(
            str(part or "")
            for part in [
                row.get("claim_text"),
                row.get("method_name"),
                row.get("dataset_name"),
                row.get("metric_name"),
            ]
        )
        for factor in _matched_fine_protocol_factors(text):
            key = (row["node_id"], factor)
            bucket = buckets.setdefault(
                key,
                {"node_id": row["node_id"], "artifact_type": factor, "paper_ids": set(), "snippets": []},
            )
            bucket["paper_ids"].add(row["paper_id"])
            bucket["snippets"].append(text[:260])

    facet_rows = db.fetchall(
        """
        SELECT pt.node_id,
               f.paper_id,
               f.facet_type,
               f.facet_name,
               f.summary,
               f.source_quote,
               f.metadata
        FROM paper_research_facets f
        JOIN paper_taxonomy pt ON pt.paper_id = f.paper_id
        WHERE f.facet_type IN (
          'motivation_rationale',
          'methodology_unit',
          'design_decision',
          'protocol_mechanism'
        )
        """
    )
    for row in facet_rows:
        if primary_nodes.get(row["paper_id"]) != row["node_id"]:
            continue
        text = " ".join(
            str(part or "")
            for part in [
                row.get("facet_name"),
                row.get("summary"),
                row.get("source_quote"),
                row.get("metadata"),
            ]
        )
        factors = _matched_fine_protocol_factors(text)
        factors.append(_slug_factor(row.get("facet_name"), f"facet_{row['facet_type']}"))
        for factor in sorted(set(factors)):
            key = (row["node_id"], factor)
            bucket = buckets.setdefault(
                key,
                {"node_id": row["node_id"], "artifact_type": factor, "paper_ids": set(), "snippets": []},
            )
            bucket["paper_ids"].add(row["paper_id"])
            bucket["snippets"].append(text[:260])

    db.execute("DELETE FROM protocol_artifacts")
    count = 0
    for bucket in buckets.values():
        support_count = len(bucket["paper_ids"])
        artifact_type = str(bucket["artifact_type"])
        is_fine_factor = artifact_type.startswith("method_") or artifact_type.startswith("facet_")
        if support_count < min_support and not is_fine_factor:
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
    """Find repeated stated limitations about missing controls or robustness."""
    primary_nodes = _primary_paper_nodes()
    rows = db.fetchall(
        """
        SELECT pt.node_id, pi.paper_id, pi.limitations, pi.open_questions
        FROM paper_insights pi
        JOIN paper_taxonomy pt ON pt.paper_id = pi.paper_id
        """
    )

    buckets: dict[tuple[str, str], dict] = {}
    for row in rows:
        if primary_nodes.get(row["paper_id"]) != row["node_id"]:
            continue
        texts = _norm_text_list(row.get("limitations"))
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

    facet_rows = db.fetchall(
        """
        SELECT pt.node_id, f.paper_id, f.facet_name, f.summary, f.source_quote
        FROM paper_research_facets f
        JOIN paper_taxonomy pt ON pt.paper_id = f.paper_id
        WHERE f.facet_type = 'boundary_condition'
        """
    )
    for row in facet_rows:
        if primary_nodes.get(row["paper_id"]) != row["node_id"]:
            continue
        gap_type = _slug_factor(row.get("facet_name"), "boundary")
        key = (row["node_id"], gap_type)
        bucket = buckets.setdefault(
            key,
            {"node_id": row["node_id"], "gap_type": gap_type, "paper_ids": set(), "snippets": []},
        )
        bucket["paper_ids"].add(row["paper_id"])
        bucket["snippets"].append(
            " ".join(str(part or "") for part in [row.get("summary"), row.get("source_quote")])[:260]
        )

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
    primary_nodes = _primary_paper_nodes()
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
        if primary_nodes.get(row["paper_id"]) != row["node_id"]:
            continue
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


def harvest_hidden_variable_bridges(min_score: float = 0.55, top_k: int = 500):
    """Find distant nodes sharing rare protocol/gap factor combinations.

    Older scoring used len(shared_factors), which made generic factors such as
    reproducibility_gap behave like strong bridges. This version down-weights
    common factors by node-level IDF and rewards rare shared combinations.
    """
    _ensure_column("hidden_variable_bridges", "factor_idf", "REAL")
    _ensure_column("hidden_variable_bridges", "pair_rarity", "REAL")
    _ensure_column("hidden_variable_bridges", "shared_factor_count", "INTEGER")
    factor_rows = db.fetchall(
        """
        SELECT node_id, 'protocol:' || artifact_type AS factor, paper_ids
        FROM protocol_artifacts
        UNION ALL
        SELECT node_id, 'gap:' || gap_type AS factor, paper_ids
        FROM negative_space_gaps
        """
    )
    node_factors: dict[str, set[str]] = defaultdict(set)
    factor_nodes: dict[str, set[str]] = defaultdict(set)
    node_factor_papers: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in factor_rows:
        if row["node_id"] and row["factor"]:
            node_factors[row["node_id"]].add(row["factor"])
            factor_nodes[row["factor"]].add(row["node_id"])
            node_factor_papers[(row["node_id"], row["factor"])].update(_json_list(row.get("paper_ids")))

    node_count = max(len(node_factors), 1)
    max_idf = math.log((node_count + 1) / 1) + 1.0
    factor_idf = {
        factor: (math.log((node_count + 1) / (len(nodes) + 1)) + 1.0) / max_idf
        for factor, nodes in factor_nodes.items()
    }

    overlap_lookup = {
        tuple(sorted((row["node_a_id"], row["node_b_id"]))): row.get("overlap_score", 0)
        for row in db.fetchall("SELECT node_a_id, node_b_id, overlap_score FROM node_entity_overlap")
    }

    nodes = sorted(node_factors)
    raw_candidates = []
    combo_counts: Counter[tuple[str, ...]] = Counter()
    for i, node_a in enumerate(nodes):
        for node_b in nodes[i + 1:]:
            shared = node_factors[node_a] & node_factors[node_b]
            if not shared:
                continue
            combo = tuple(sorted(shared))
            combo_counts[combo] += 1
            overlap_score = overlap_lookup.get(tuple(sorted((node_a, node_b))), 0)
            raw_candidates.append((node_a, node_b, combo, overlap_score))

    pair_total = max(len(raw_candidates), 1)
    candidates = []
    for node_a, node_b, shared_combo, overlap_score in raw_candidates:
        shared = list(shared_combo)
        has_specific_factor = any(factor not in GENERIC_BRIDGE_FACTORS for factor in shared)
        idf_sum = sum(factor_idf.get(f, 0.0) for f in shared)
        if idf_sum <= 0:
            continue
        pair_rarity = math.log((pair_total + 1) / (combo_counts[shared_combo] + 1)) / max(math.log(pair_total + 1), 1e-9)
        overlap_multiplier = 1.0 if overlap_score < 0.12 else 0.35
        specificity_multiplier = 1.25 if has_specific_factor else 0.55
        score = idf_sum * (0.5 + pair_rarity) * overlap_multiplier * specificity_multiplier
        if score < min_score:
            continue
        if not has_specific_factor and score < 1.2:
            continue
        paper_ids = set()
        for factor in shared:
            paper_ids.update(node_factor_papers.get((node_a, factor), set()))
            paper_ids.update(node_factor_papers.get((node_b, factor), set()))
        candidates.append(
            {
                "node_a_id": node_a,
                "node_b_id": node_b,
                "shared_factors": shared,
                "paper_ids": sorted(paper_ids),
                "score": round(score, 4),
                "pair_rarity": round(pair_rarity, 4),
                "shared_factor_count": len(shared),
                "factor_idf": round(idf_sum / max(len(shared), 1), 4),
            }
        )

    candidates.sort(key=lambda c: (c["score"], c["pair_rarity"], c["factor_idf"]), reverse=True)
    candidates = candidates[:top_k]

    db.execute("DELETE FROM hidden_variable_bridges")
    count = 0
    for c in candidates:
        db.execute(
            """INSERT INTO hidden_variable_bridges
               (node_a_id, node_b_id, shared_factor, paper_ids, score,
                factor_idf, pair_rarity, shared_factor_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (node_a_id, node_b_id, shared_factor) DO UPDATE SET
                 paper_ids = excluded.paper_ids,
                 score = excluded.score,
                 factor_idf = excluded.factor_idf,
                 pair_rarity = excluded.pair_rarity,
                 shared_factor_count = excluded.shared_factor_count""",
            (
                c["node_a_id"],
                c["node_b_id"],
                json.dumps(c["shared_factors"], ensure_ascii=False),
                json.dumps(c["paper_ids"][:20], ensure_ascii=False),
                c["score"],
                c["factor_idf"],
                c["pair_rarity"],
                c["shared_factor_count"],
            ),
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
    stats["result_contradictions"] = _timed_harvest("result_contradiction", harvest_result_contradictions)
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

    literature_seed_insights: list[dict] = []
    if not high_insights:
        literature_seed_rows = db.fetchall(
            """
            SELECT pi.paper_id AS id,
                   p.title AS title,
                   'literature_seed' AS mechanism_type,
                   pi.problem_statement AS evidence_summary,
                   pi.plain_summary AS plain_summary,
                   pi.approach_summary AS approach_summary,
                   pi.key_findings AS key_findings,
                   pi.limitations AS limitations,
                   pi.open_questions AS open_questions,
                   GROUP_CONCAT(pt.node_id) AS related_node_ids
            FROM paper_insights pi
            JOIN papers p ON p.id = pi.paper_id
            LEFT JOIN paper_taxonomy pt ON pt.paper_id = pi.paper_id
            WHERE COALESCE(pi.problem_statement, '') != ''
               OR COALESCE(pi.plain_summary, '') != ''
               OR COALESCE(pi.approach_summary, '') != ''
               OR COALESCE(pi.key_findings, '') NOT IN ('', '[]')
               OR COALESCE(pi.limitations, '') NOT IN ('', '[]')
               OR COALESCE(pi.open_questions, '') NOT IN ('', '[]')
            GROUP BY pi.paper_id, p.title, pi.problem_statement, pi.plain_summary,
                     pi.approach_summary, pi.key_findings, pi.limitations, pi.open_questions
            ORDER BY p.updated_at DESC
            LIMIT 10
            """
        )
        literature_seed_insights = [dict(row) for row in literature_seed_rows]
        high_insights = [
            {
                "id": row["id"],
                "title": row["title"],
                "mechanism_type": row["mechanism_type"],
                "evidence_summary": row.get("evidence_summary") or row.get("plain_summary") or "",
                "evidence_packet": json.dumps(
                    {
                        "non_numeric_evidence": (
                            _norm_text_list(row.get("key_findings"))
                            + _norm_text_list(row.get("limitations"))
                            + _norm_text_list(row.get("open_questions"))
                        ),
                        "structural_evidence": [
                            text
                            for text in [
                                row.get("evidence_summary"),
                                row.get("plain_summary"),
                                row.get("approach_summary"),
                            ]
                            if text
                        ],
                    },
                    ensure_ascii=False,
                ),
                "experimental_plan": "{}",
                "signal_mix": json.dumps(["literature_seed"], ensure_ascii=False),
                "resource_class": "cpu",
                "related_node_ids": row.get("related_node_ids") or "",
            }
            for row in literature_seed_insights
        ]

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
            "literature_seed_count": len(literature_seed_insights),
        },
    )
