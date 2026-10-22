"""Reasoning Agent: Matrix gap finding + contradiction detection."""
from collections.abc import Mapping
from typing import Any

from db import database as db
from config import CONTRADICTION_REQUIRES_GROUNDED
from meta_harness.scoped_llm import proposer_json


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


def detect_contradictions_batch(
    claims: list[dict],
    *,
    llm_scope: Mapping[str, Any] | None = None,
    paper_id: str | None = None,
) -> tuple[list[dict], int]:
    """Check all performance claims from a paper in one LLM call.

    Comparison claims must come from OTHER papers. The lookup used to exclude
    only the claim's own row, so it mostly found the same paper's sibling
    claims: on 2026-09-24, 705 of 756 "related" rows sampled were from the
    paper under test, every one of the twelve newest stored contradictions set
    a paper against itself, and 57 of 258 in the table were of that kind. A
    paper disagreeing with itself across ablation settings is not a finding.
    """
    perf_claims = [c for c in claims if c.get("claim_type") == "performance" and c.get("method_name")]
    if CONTRADICTION_REQUIRES_GROUNDED:
        perf_claims = [c for c in perf_claims if c.get("grounding_status") == "verified"]
    if not perf_claims:
        return [], 0

    # Gather all related existing claims
    all_related = {}
    seen_ids = set()
    for c in perf_claims:
        claim_id = c.get("_id", 0)
        if c.get("method_name"):
            own_paper = paper_id or c.get("paper_id") or ""
            rows = db.fetchall(
                "SELECT id, claim_text, claim_type, method_name, dataset_name, metric_name, metric_value, paper_id "
                "FROM claims WHERE method_name=? AND id!=? AND paper_id<>? LIMIT 10",
                (c["method_name"], claim_id, own_paper)
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

    claim_scope = ",".join(
        str(claim.get("_id") or index)
        for index, claim in enumerate(perf_claims)
    )
    result, tokens, _route = proposer_json(
        BATCH_CONTRADICTION_SYSTEM,
        user_prompt,
        llm_scope=llm_scope,
        operation=f"contradiction_check:batch:{claim_scope}",
    )
    contras = result.get("contradictions", [])

    # Map back to claim IDs
    for contra in contras:
        idx = contra.get("new_claim_idx", 0)
        if 0 <= idx < len(perf_claims):
            contra["_new_claim"] = perf_claims[idx]

    return contras, tokens
