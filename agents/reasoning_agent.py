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

GAP_SYSTEM = """You are a research gap detector for ML. You are given a method x dataset matrix for a specific research area. The matrix shows which (method, dataset) combinations have been evaluated.

Your job: find valuable EMPTY CELLS -- method+dataset combinations that nobody has tried but would be interesting.

A valuable gap is one where:
1. The method has been tested on other datasets in this area (so it's relevant)
2. The dataset has been used with other methods (so it's a real benchmark)
3. Testing this combination would yield useful scientific knowledge
4. There's a non-obvious reason why this would be interesting (not just "nobody tried it")

Return JSON:
{
  "gaps": [
    {
      "method_name": "exact method name from the matrix",
      "dataset_name": "exact dataset name from the matrix",
      "metric_name": "what metric should be used",
      "gap_description": "why this empty cell is interesting",
      "research_proposal": "2-3 sentence proposal for filling this gap",
      "value_score": 1-5
    }
  ]
}

Focus on HIGH-VALUE gaps only (score >= 3). Skip trivial or obvious ones.
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
