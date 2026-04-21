"""Abstraction Agent: Extract universal patterns from domain-specific claims,
then match patterns across domains to find cross-disciplinary transfer opportunities.

Two phases:
1. ABSTRACT: Claims in a node → domain-independent patterns
2. BRIDGE: Match patterns across distant nodes → cross-domain insights
"""
import json
from agents.llm_client import call_llm_json
from db import database as db
from db import taxonomy as tax


# ── Phase 1: Abstracting claims to patterns ──────────────────────────

ABSTRACT_PROMPT = """You are a pattern abstraction agent. Your job is to take specific scientific claims from one research area and abstract them into UNIVERSAL PATTERNS that could apply across completely different disciplines.

The goal: enable cross-disciplinary discovery. If "students lose critical thinking when using ChatGPT" and "pilots lose manual skills when using autopilot" share the pattern "automation → human skill atrophy", then a solution from aviation might transfer to education.

You will receive claims from one research area. For each meaningful group of claims, produce ONE abstract pattern.

Return JSON:
{
  "patterns": [
    {
      "pattern_text": "CAUSE → EFFECT format, domain-free language",
      "pattern_type": "problem|solution|phenomenon|tradeoff|scaling_law",
      "abstraction_level": "domain|cross-domain|universal",
      "domains_applicable": ["other domains where this pattern likely holds"],
      "source_claim_indices": [0, 3, 7],
      "confidence": 0.8,
      "example_in_other_domain": "One concrete example of how this pattern manifests in a different field"
    }
  ]
}

Abstraction rules:
- Strip ALL domain-specific jargon — keep only the structural relationship
- "LoRA is worse than full fine-tuning on large models" → "parameter-efficient approximation degrades with scale"
- "Ensemble of weak classifiers beats single strong one" → "aggregation of diverse weak signals outperforms single strong signal"
- "Curriculum learning helps on hard tasks" → "progressive difficulty ordering accelerates complex skill acquisition"
- A good pattern: someone in a DIFFERENT field says "yes, this happens to us too"
- SKIP trivial patterns like "more data helps" or "bigger model is better"
- Only return cross-domain or universal patterns (skip domain-only ones)
- Return 0-5 patterns. Quality over quantity. ZERO is acceptable.
- Return ONLY valid JSON."""


def abstract_node_claims(node_id: str, max_claims: int = 40) -> tuple[list[dict], int]:
    """Extract universal patterns from claims in a taxonomy node."""
    node = tax.get_node(node_id)
    if not node:
        return [], 0

    # Get claims with context
    claims = db.fetchall("""
        SELECT c.id, c.claim_text, c.claim_type, c.method_name, c.dataset_name,
               c.metric_name, c.metric_value, p.title as paper_title
        FROM claims c
        JOIN papers p ON c.paper_id = p.id
        JOIN paper_taxonomy pt ON pt.paper_id = p.id
        WHERE pt.node_id = ?
        AND c.claim_type IN ('performance', 'finding', 'method', 'analysis',
                             'generalization', 'limitation', 'robustness', 'tradeoff')
        ORDER BY RANDOM()
        LIMIT ?
    """, (node_id, max_claims))

    if len(claims) < 5:
        return [], 0

    claims_text = "\n".join(
        f"[{i}] ({c['claim_type']}): {c['claim_text']}"
        + (f" [{c['method_name']} on {c['dataset_name']} = {c['metric_value']}]"
           if c.get('metric_value') else "")
        for i, c in enumerate(claims)
    )

    prompt = f"""Research area: {node['name']} ({node_id})
Number of claims: {len(claims)}

Claims:
{claims_text}

Find universal patterns that transcend this specific research area. Focus on structural relationships that appear in multiple claims and could apply to completely different fields."""

    try:
        result, tokens = call_llm_json(ABSTRACT_PROMPT, prompt)
        patterns = result.get("patterns", [])

        # Attach metadata
        for pat in patterns:
            pat["node_id"] = node_id
            pat["node_name"] = node["name"]
            # Map source indices to claim IDs
            indices = pat.get("source_claim_indices", [])
            pat["source_claim_ids"] = [
                claims[i]["id"] for i in indices
                if isinstance(i, int) and 0 <= i < len(claims)
            ]

        return patterns, tokens
    except Exception as e:
        print(f"Abstraction error for {node_id}: {e}", flush=True)
        return [], 0


# ── Phase 2: Cross-domain pattern bridging ───────────────────────────

BRIDGE_PROMPT = """You are a cross-disciplinary research strategist. You will see abstract patterns extracted from different research areas. Your job is to find BRIDGES — cases where two patterns from distant areas are structurally similar, suggesting a method/solution from one area could transfer to the other.

This is NOT about finding vague analogies. A bridge must:
1. Identify a SPECIFIC structural match between patterns from different domains
2. Explain WHY the match is not superficial (what shared mechanism drives both)
3. Propose a CONCRETE experiment that tests the transfer
4. Have a clear supporting_papers trail

Return JSON:
{
  "bridges": [
    {
      "pattern_a_id": 0,
      "pattern_b_id": 3,
      "bridge_title": "One-line title: what transfers from where to where",
      "structural_match": "WHY these two patterns are the same phenomenon",
      "mechanism": "The shared underlying mechanism driving both patterns",
      "transfer_proposal": "Concrete experiment: take method X from area A, adapt it for area B, test on dataset D with metric M",
      "risk": "What could go wrong / why the analogy might break",
      "novelty_score": 1-5,
      "feasibility_score": 1-5
    }
  ]
}

Rules:
- Only return bridges between patterns from DIFFERENT research areas
- The areas must be meaningfully distant (not just "NLP generation" and "NLP summarization")
- Return 0-3 bridges. ZERO is fine if nothing genuine exists.
- A bridge must be specific enough to write a paper about
- Return ONLY valid JSON."""


def find_cross_domain_bridges(max_patterns: int = 60) -> tuple[list[dict], int]:
    """Find structural bridges between patterns from different research areas."""

    # Get diverse patterns from different top-level areas
    patterns = db.fetchall("""
        SELECT p.id, p.pattern_text, p.pattern_type, p.abstraction_level,
               p.domains, p.node_id, p.source_claims
        FROM patterns p
        WHERE p.abstraction_level IN ('cross-domain', 'universal')
        AND p.node_id IS NOT NULL
        ORDER BY p.created_at DESC
        LIMIT ?
    """, (max_patterns,))

    if len(patterns) < 6:
        return [], 0

    # Group by top-level area to ensure we have cross-domain diversity
    area_groups = {}
    for p in patterns:
        top_area = (p["node_id"] or "").split(".")[1] if "." in (p["node_id"] or "") else "unknown"
        area_groups.setdefault(top_area, []).append(p)

    if len(area_groups) < 2:
        return [], 0

    # Build prompt with patterns indexed
    pattern_text = "\n".join(
        f"[{i}] Area: {p['node_id']} | Type: {p['pattern_type']} | Level: {p['abstraction_level']}\n"
        f"    Pattern: {p['pattern_text']}\n"
        f"    Applicable domains: {p.get('domains', '[]')}"
        for i, p in enumerate(patterns)
    )

    prompt = f"""You have {len(patterns)} abstract patterns from {len(area_groups)} different research areas:
{', '.join(area_groups.keys())}

Patterns:
{pattern_text}

Find bridges — cases where a pattern in one area structurally matches a pattern in a distant area, suggesting concrete method transfer opportunities. The bridge should be specific enough to design an experiment around."""

    try:
        result, tokens = call_llm_json(BRIDGE_PROMPT, prompt)
        bridges = result.get("bridges", [])

        # Enrich with pattern details
        for bridge in bridges:
            idx_a = bridge.get("pattern_a_id", 0)
            idx_b = bridge.get("pattern_b_id", 0)
            if idx_a < len(patterns):
                bridge["area_a"] = patterns[idx_a]["node_id"]
                bridge["pattern_a_text"] = patterns[idx_a]["pattern_text"]
            if idx_b < len(patterns):
                bridge["area_b"] = patterns[idx_b]["node_id"]
                bridge["pattern_b_text"] = patterns[idx_b]["pattern_text"]

        return bridges, tokens
    except Exception as e:
        print(f"Bridge discovery error: {e}", flush=True)
        return [], 0


# ── Storage ──────────────────────────────────────────────────────────

def store_pattern(pattern: dict) -> int:
    """Store an abstracted pattern."""
    domains = pattern.get("domains_applicable", [])
    if isinstance(domains, list):
        domains = json.dumps(domains)
    source_claims = pattern.get("source_claim_ids", [])
    if isinstance(source_claims, list):
        source_claims = json.dumps(source_claims)

    pid = db.insert_returning_id(
        """INSERT INTO patterns
           (pattern_text, pattern_type, abstraction_level, domains, domain_count,
            node_id, source_claims)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           RETURNING id""",
        (pattern["pattern_text"], pattern.get("pattern_type", "phenomenon"),
         pattern.get("abstraction_level", "cross-domain"),
         domains, len(pattern.get("domains_applicable", [])),
         pattern.get("node_id"), source_claims),
    )
    db.commit()
    return pid


def store_bridge(bridge: dict) -> int:
    """Store a cross-domain bridge as a special insight."""
    from agents.insight_agent import store_insight
    insight = {
        "node_id": bridge.get("area_a", "ml"),
        "type": "cross_domain_bridge",
        "title": bridge.get("bridge_title", "Cross-domain bridge"),
        "hypothesis": bridge.get("structural_match", "") + " Mechanism: " + bridge.get("mechanism", ""),
        "supporting_papers": [],
        "evidence": f"Pattern A ({bridge.get('area_a', '?')}): {bridge.get('pattern_a_text', '?')}\n"
                    f"Pattern B ({bridge.get('area_b', '?')}): {bridge.get('pattern_b_text', '?')}",
        "experiment": bridge.get("transfer_proposal", ""),
        "impact": f"Cross-domain transfer from {bridge.get('area_a', '?')} to {bridge.get('area_b', '?')}. "
                  f"Risk: {bridge.get('risk', 'unknown')}",
        "novelty_score": bridge.get("novelty_score", 4),
        "feasibility_score": bridge.get("feasibility_score", 3),
    }
    return store_insight(insight)


# ── Pipeline entry points ────────────────────────────────────────────

def run_abstraction_for_nodes(min_claims: int = 15) -> tuple[int, int]:
    """Run abstraction on nodes that have enough claims but no patterns yet."""
    nodes = db.fetchall("""
        SELECT t.id, COUNT(DISTINCT c.id) as claim_count
        FROM taxonomy_nodes t
        JOIN paper_taxonomy pt ON pt.node_id = t.id
        JOIN claims c ON c.paper_id = pt.paper_id
        LEFT JOIN patterns p ON p.node_id = t.id
        WHERE p.id IS NULL
        GROUP BY t.id
        HAVING claim_count >= ?
        ORDER BY claim_count DESC
        LIMIT 20
    """, (min_claims,))

    total_patterns = 0
    total_tokens = 0
    for node in nodes:
        print(f"  Abstracting {node['id']} ({node['claim_count']} claims)...", flush=True)
        patterns, tokens = abstract_node_claims(node["id"])
        total_tokens += tokens
        for pat in patterns:
            store_pattern(pat)
            total_patterns += 1
            print(f"    [{pat.get('abstraction_level', '?')}] {pat['pattern_text'][:80]}", flush=True)

    return total_patterns, total_tokens


def run_bridge_discovery() -> tuple[int, int]:
    """Find cross-domain bridges between existing patterns."""
    bridges, tokens = find_cross_domain_bridges()
    for bridge in bridges:
        store_bridge(bridge)
        print(f"  BRIDGE: {bridge.get('bridge_title', '?')[:80]}", flush=True)
    return len(bridges), tokens
