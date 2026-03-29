"""Reasoning Agent: Matrix gap finding + contradiction detection."""
import json
from db import database as db
from db import taxonomy as tax
from agents.llm_client import call_llm_json

CONTRADICTION_SYSTEM = """You are a scientific contradiction detector. Given a NEW claim and a set of EXISTING related claims from other papers, identify any contradictions.

A contradiction exists when:
- Two claims make opposing statements about similar things under similar conditions
- Two claims report significantly different numbers for the same method+dataset+metric

Return JSON:
{
  "contradictions": [
    {
      "existing_claim_id": <id of the contradicting existing claim>,
      "description": "clear explanation of the contradiction",
      "condition_diff": "key differences in experimental conditions that might explain it",
      "hypothesis": "a testable hypothesis that could resolve this contradiction",
      "severity": "low|medium|high"
    }
  ]
}

If no contradictions found, return {"contradictions": []}
Return ONLY valid JSON."""

GAP_SYSTEM = """You are a senior ML researcher analyzing a research area for GENUINE research opportunities — not trivial missing experiments.

You are given a method x dataset matrix. Most empty cells are WORTHLESS ("nobody tried X on Y" is not a research gap). Your job is to find the rare empty cells that represent REAL scientific questions.

IGNORE these types of "gaps" (they are worthless):
- Method A wasn't tested on Dataset B, but there's no reason it would behave differently than on similar datasets
- A trivial combination that nobody tested because it's obviously not interesting
- Incremental "we ran X on Y" experiments with no hypothesis

ONLY report gaps that fall into these categories:

1. TECHNICAL BARRIER: "Nobody tested X on Y because Y requires Z which X can't handle — but if someone solved Z, the result would be significant"
   Example: "Transformers on full-slide pathology images — nobody did it because 100K+ token sequences cause OOM. FlashAttention might finally make this feasible."

2. COGNITIVE BLIND SPOT: "Two communities use different terms for the same problem. Domain A has a solution that Domain B doesn't know about."
   Example: "Curriculum learning (ML term) is the same idea as scaffolded instruction (education term). Education researchers haven't adopted this."

3. CONTRADICTORY EVIDENCE: "Paper A says method X beats Y, Paper B says the opposite. The conditions differ in a specific way that nobody has isolated."
   Example: "LoRA vs full fine-tuning: results flip between 7B and 70B models. Nobody has mapped the exact crossover point."

4. ASSUMPTION CHALLENGE: "Everyone in this area assumes Z, but recent evidence suggests Z might be wrong. Nobody has tested this directly."
   Example: "Data augmentation is assumed to always help small datasets, but 3 recent papers show it hurts under distribution shift. No systematic study exists."

Return JSON:
{
  "gaps": [
    {
      "method_name": "exact method name from matrix (or 'N/A' for non-matrix gaps)",
      "dataset_name": "exact dataset name from matrix (or 'N/A')",
      "metric_name": "what metric matters",
      "gap_type": "technical_barrier|cognitive_blind_spot|contradictory_evidence|assumption_challenge",
      "why_gap_exists": "WHY has nobody done this? What barrier or blind spot kept this unexplored?",
      "gap_description": "What is the actual scientific question?",
      "what_we_would_learn": "If someone did this experiment, what would the field learn regardless of the outcome?",
      "research_proposal": "Concrete 3-sentence experiment design",
      "value_score": 1-5
    }
  ]
}

Be VERY selective. Most matrices have 0-2 genuine gaps. Return empty list rather than low-value filler.
Return ONLY valid JSON."""


def detect_contradictions(new_claim: dict, new_claim_id: int) -> tuple[list[dict], int]:
    """Check a new claim against existing related claims for contradictions."""
    related = []
    if new_claim.get("method_name"):
        related += db.fetchall(
            "SELECT id, claim_text, claim_type, method_name, dataset_name, metric_name, metric_value, paper_id "
            "FROM claims WHERE method_name=? AND id!=? LIMIT 20",
            (new_claim["method_name"], new_claim_id)
        )
    if new_claim.get("dataset_name"):
        related += db.fetchall(
            "SELECT id, claim_text, claim_type, method_name, dataset_name, metric_name, metric_value, paper_id "
            "FROM claims WHERE dataset_name=? AND id!=? LIMIT 20",
            (new_claim["dataset_name"], new_claim_id)
        )

    if not related:
        return [], 0

    # Deduplicate
    seen = set()
    unique_related = []
    for r in related:
        if r["id"] not in seen:
            seen.add(r["id"])
            unique_related.append(r)

    existing_text = "\n".join(
        f"[ID={r['id']}] (paper {r['paper_id']}): {r['claim_text']} "
        f"[method={r.get('method_name')}, dataset={r.get('dataset_name')}, "
        f"metric={r.get('metric_name')}, value={r.get('metric_value')}]"
        for r in unique_related[:20]
    )

    user_prompt = f"""NEW CLAIM:
{new_claim['claim_text']}
[method={new_claim.get('method_name')}, dataset={new_claim.get('dataset_name')},
 metric={new_claim.get('metric_name')}, value={new_claim.get('metric_value')}]

EXISTING RELATED CLAIMS:
{existing_text}"""

    result, tokens = call_llm_json(CONTRADICTION_SYSTEM, user_prompt)
    return result.get("contradictions", []), tokens


BATCH_CONTRADICTION_SYSTEM = """You are a scientific contradiction detector. Given a set of NEW claims from one paper and EXISTING claims from other papers, identify contradictions.

A contradiction exists when:
- Two claims make opposing statements about similar things under similar conditions
- Two claims report significantly different numbers for the same method+dataset+metric

Return JSON:
{
  "contradictions": [
    {
      "new_claim_idx": <0-based index of the new claim>,
      "existing_claim_id": <id of the contradicting existing claim>,
      "description": "clear explanation of the contradiction with specific numbers",
      "condition_diff": "key differences in experimental conditions",
      "hypothesis": "a testable hypothesis to resolve this",
      "severity": "low|medium|high"
    }
  ]
}

If no contradictions, return {"contradictions": []}. Be selective — only flag REAL contradictions.
Return ONLY valid JSON."""


def detect_contradictions_batch(claims: list[dict]) -> tuple[list[dict], int]:
    """Check all performance claims from a paper in one LLM call."""
    perf_claims = [c for c in claims if c.get("claim_type") == "performance" and c.get("method_name")]
    if not perf_claims:
        return [], 0

    # Gather all related existing claims
    all_related = {}
    seen_ids = set()
    for c in perf_claims:
        claim_id = c.get("_id", 0)
        if c.get("method_name"):
            rows = db.fetchall(
                "SELECT id, claim_text, claim_type, method_name, dataset_name, metric_name, metric_value, paper_id "
                "FROM claims WHERE method_name=? AND id!=? LIMIT 10",
                (c["method_name"], claim_id)
            )
            for r in rows:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    all_related[r["id"]] = r

    if not all_related:
        return [], 0

    # Build prompt
    new_claims_text = "\n".join(
        f"[{i}] {c['claim_text'][:200]} [method={c.get('method_name')}, dataset={c.get('dataset_name')}, "
        f"metric={c.get('metric_name')}, value={c.get('metric_value')}]"
        for i, c in enumerate(perf_claims)
    )

    existing_text = "\n".join(
        f"[ID={r['id']}] (paper {r['paper_id']}): {r['claim_text'][:200]} "
        f"[method={r.get('method_name')}, dataset={r.get('dataset_name')}, "
        f"metric={r.get('metric_name')}, value={r.get('metric_value')}]"
        for r in list(all_related.values())[:30]
    )

    user_prompt = f"""NEW CLAIMS FROM THIS PAPER:
{new_claims_text}

EXISTING CLAIMS FROM OTHER PAPERS:
{existing_text}"""

    result, tokens = call_llm_json(BATCH_CONTRADICTION_SYSTEM, user_prompt)
    contras = result.get("contradictions", [])

    # Map back to claim IDs
    for contra in contras:
        idx = contra.get("new_claim_idx", 0)
        if 0 <= idx < len(perf_claims):
            contra["_new_claim"] = perf_claims[idx]

    return contras, tokens


def discover_matrix_gaps(node_id: str) -> tuple[list[dict], int]:
    """Find valuable empty cells in the method x dataset matrix for a taxonomy node."""
    matrix = tax.get_method_dataset_matrix(node_id)

    if len(matrix["methods"]) < 2 or len(matrix["datasets"]) < 2:
        return [], 0

    node = tax.get_node(node_id)
    node_name = node["name"] if node else node_id

    # Build a readable matrix representation
    filled_cells = []
    empty_combos = []
    for m in matrix["methods"]:
        for d in matrix["datasets"]:
            # Check if any metric exists for this method+dataset
            has_data = False
            for metric in (matrix["metrics"] or [""]):
                key = f"{m}|||{d}|||{metric}"
                if key in matrix["cells"]:
                    cell = matrix["cells"][key]
                    filled_cells.append(
                        f"  {m} + {d} [{metric}] = {cell['value']} (paper {cell['paper_id']})"
                    )
                    has_data = True
            if not has_data:
                empty_combos.append(f"  {m} + {d}")

    if not empty_combos:
        return [], 0

    user_prompt = f"""Research area: {node_name} (taxonomy: {node_id})

FILLED CELLS (existing results):
{chr(10).join(filled_cells[:100])}

EMPTY CELLS (untested combinations):
{chr(10).join(empty_combos[:80])}

Methods in the matrix: {', '.join(matrix['methods'])}
Datasets in the matrix: {', '.join(matrix['datasets'])}

Find the most valuable empty cells. Why would testing these combinations advance the field?"""

    try:
        result, tokens = call_llm_json(GAP_SYSTEM, user_prompt)
        gaps = result.get("gaps", [])
        # Attach node_id to each gap
        for g in gaps:
            g["node_id"] = node_id
        return gaps, tokens
    except Exception as e:
        print(f"Gap discovery error for {node_id}: {e}", flush=True)
        return [], 0


def discover_all_gaps() -> tuple[list[dict], int]:
    """Run gap discovery on all taxonomy nodes that have enough data."""
    all_gaps = []
    total_tokens = 0

    # Find nodes with at least 2 methods and 2 datasets
    nodes = db.fetchall(
        """SELECT rt.node_id, COUNT(DISTINCT r.method_name) as mc,
                  COUNT(DISTINCT r.dataset_name) as dc
           FROM results r
           JOIN result_taxonomy rt ON rt.result_id = r.id
           GROUP BY rt.node_id
           HAVING mc >= 2 AND dc >= 2"""
    )

    for node in nodes:
        gaps, tokens = discover_matrix_gaps(node["node_id"])
        all_gaps.extend(gaps)
        total_tokens += tokens

    return all_gaps, total_tokens
