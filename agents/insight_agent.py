"""Insight Agent: Deep cross-paper reasoning to find genuine research opportunities.

NOT "method X wasn't tested on dataset Y" — that's trivial.
Instead: contradictions, ignored limitations, method transfers, paradigm shifts.
"""
import json
from agents.llm_client import call_llm_json
from db import database as db
from db import taxonomy as tax


SYSTEM_PROMPT = """You are a senior research strategist analyzing a body of ML papers. Your job is to find GENUINE research insights that could lead to publishable work — NOT trivial "test X on Y" suggestions.

You will receive:
1. Paper summaries with their claims, specific numbers, and limitations
2. Contradictions between papers (with exact conflicting values)
3. Common limitations that multiple papers share
4. Method descriptions and where they've been applied

Find insights in these categories ONLY:

## CONTRADICTION ANALYSIS
When two papers disagree, don't just flag it — propose a HYPOTHESIS that explains the disagreement and an experiment to test it. CITE the specific papers and numbers that conflict.
Example: "2401.12345 reports LoRA matches full fine-tuning at 7B (MMLU 63.2 vs 63.5), while 2402.67890 shows a 4.8pt gap at 70B (MMLU 78.1 vs 82.9). Hypothesis: LoRA's low-rank bottleneck becomes capacity-limiting above ~30B parameters. Test: systematic sweep at 1B/3B/7B/13B/30B/70B with rank-scaling ablation."

## IGNORED LIMITATION
When 3+ papers share the same limitation but nobody addresses it. LIST the papers by ID.
Example: "2401.11111, 2401.22222, 2402.33333, 2402.44444 (4 of 7 code-gen papers) note 'only tested on Python'. This isn't a missing benchmark — it's a fundamental question: do code generation techniques transfer across language paradigms with different type systems?"

## METHOD TRANSFER
When a technique from area A could solve a known problem in area B. Explain the STRUCTURAL match with specifics.
Example: "Hard negative mining in contrastive vision (2401.55555 shows +3.2% on ImageNet) addresses representation collapse to easy patterns. Dialogue systems (2402.66666 reports 38% generic-response rate) suffer the same collapse. The structural match: both optimize a similarity metric that can be gamed by outputting the average."

## PARADIGM EXHAUSTION
When a research direction shows diminishing returns. CITE the actual numbers showing diminishing gains.
Example: "Attention variants on ImageNet: 2401.77777 +0.3%, 2402.88888 +0.2%, 2403.99999 +0.1%. The curve is asymptoting at 87.4% top-1. Meanwhile, 2403.00001 achieves +2.1% via data curation alone. The field is over-investing in architecture vs data quality."

## ASSUMPTION CHALLENGE
When papers universally assume something that evidence suggests is wrong. NAME the assumption and the counter-evidence with numbers.
Example: "All 8 RLHF papers (2401.11111-2401.88888) assume preference consistency. But 2402.99999 measures inter-annotator kappa=0.58 and 2403.11111 shows 23% self-contradiction rate. If the training signal has 23% noise, standard RLHF likely overfits to annotator bias."

Return JSON:
{
  "insights": [
    {
      "type": "contradiction_analysis|ignored_limitation|method_transfer|paradigm_exhaustion|assumption_challenge",
      "title": "One-line title (compelling, specific, includes key numbers)",
      "hypothesis": "The testable scientific hypothesis — specific and falsifiable",
      "supporting_papers": ["2401.12345", "2402.67890"],
      "evidence": "Cite specific papers by ID, quote their exact numbers, explain why they jointly support this insight",
      "experiment": "Concrete experiment design: what models, what datasets, what metrics, what baselines, what would confirm vs falsify the hypothesis",
      "impact": "What the field would learn. Be specific: which subfield, which practice would change",
      "novelty_score": 1-5,
      "feasibility_score": 1-5
    }
  ]
}

RULES:
- Return 0-3 insights. ZERO is acceptable if nothing genuine exists.
- Every insight MUST have a testable hypothesis, not just an observation.
- Every insight MUST cite specific paper IDs and specific numbers from those papers.
- "Test X on Y" is NEVER an insight. Delete it if you catch yourself writing one.
- supporting_papers MUST list arxiv IDs of papers that directly contribute to the insight.
- evidence MUST quote specific metrics/numbers from specific papers, not vague summaries.
- experiment MUST specify: models (with sizes), datasets, metrics, baselines, success criteria.
- Novelty 5 = nobody has thought about this. Novelty 1 = obvious next step.
- Feasibility 5 = can do in a week with existing tools. Feasibility 1 = needs years of work.
- Return ONLY valid JSON."""


def gather_node_evidence(node_id: str, max_papers: int = 30) -> dict:
    """Gather all evidence for a taxonomy node for deep reasoning."""

    # Papers with claims and limitations
    papers = tax.get_node_papers(node_id, limit=max_papers)

    # Get claims grouped by paper
    paper_claims = {}
    for p in papers:
        claims = db.fetchall(
            "SELECT claim_text, claim_type, method_name, dataset_name, metric_name, metric_value "
            "FROM claims WHERE paper_id=? ORDER BY claim_type",
            (p["id"],)
        )
        paper_claims[p["id"]] = {
            "title": p["title"],
            "claims": claims,
            "limitations": p.get("limitations") or [],
            "key_findings": p.get("key_findings") or [],
        }

    # Contradictions in this area
    contradictions = db.fetchall("""
        SELECT c.description, c.condition_diff, c.hypothesis,
               ca.claim_text as claim_a, ca.paper_id as paper_a,
               cb.claim_text as claim_b, cb.paper_id as paper_b
        FROM contradictions c
        JOIN claims ca ON c.claim_a_id = ca.id
        JOIN claims cb ON c.claim_b_id = cb.id
        JOIN paper_taxonomy pt ON ca.paper_id = pt.paper_id
        JOIN taxonomy_nodes t ON pt.node_id = t.id
        WHERE t.id = ? OR t.id LIKE ? || '.%'
        ORDER BY c.id DESC LIMIT 20
    """, (node_id, node_id))

    # Common limitations (from paper_insights if available)
    all_limitations = []
    for pid, pdata in paper_claims.items():
        for lim in pdata.get("limitations", []):
            if isinstance(lim, str) and len(lim) > 10:
                all_limitations.append(lim)

    # Methods in this area
    methods = db.fetchall("""
        SELECT r.method_name, COUNT(DISTINCT r.paper_id) as paper_count,
               COUNT(DISTINCT r.dataset_name) as dataset_count
        FROM results r
        JOIN result_taxonomy rt ON rt.result_id = r.id
        JOIN taxonomy_nodes t ON rt.node_id = t.id
        WHERE t.id = ? OR t.id LIKE ? || '.%'
        GROUP BY r.method_name
        ORDER BY paper_count DESC LIMIT 20
    """, (node_id, node_id))

    # Node summary if available
    summary = tax.get_node_summary(node_id)

    return {
        "node_id": node_id,
        "paper_count": len(papers),
        "papers": paper_claims,
        "contradictions": contradictions,
        "limitations": all_limitations,
        "methods": methods,
        "summary": summary,
    }


def build_evidence_prompt(evidence: dict) -> str:
    """Format evidence into a prompt for the LLM."""
    node_id = evidence["node_id"]
    node = tax.get_node(node_id)
    node_name = node["name"] if node else node_id

    sections = [f"Research area: {node_name} ({node_id})\nPapers analyzed: {evidence['paper_count']}\n"]

    # Paper summaries — include arxiv IDs and exact numbers for citation
    sections.append("## PAPERS AND THEIR CLAIMS")
    sections.append("(Use paper IDs like '2401.12345' when citing evidence in your insights)")
    for pid, pdata in list(evidence["papers"].items())[:25]:
        title = pdata["title"]
        sections.append(f"\n### {pid}: {title}")

        perf_claims = [c for c in pdata["claims"] if c["claim_type"] == "performance"]
        finding_claims = [c for c in pdata["claims"] if c["claim_type"] == "finding"]

        if perf_claims:
            sections.append("Performance (cite these exact numbers):")
            for c in perf_claims[:8]:
                val = f" = {c['metric_value']}" if c['metric_value'] else ""
                sections.append(f"  - {c['method_name'] or '?'} on {c['dataset_name'] or '?'} [{c['metric_name'] or '?'}]{val}")

        if finding_claims:
            sections.append("Key findings:")
            for c in finding_claims[:5]:
                sections.append(f"  - {c['claim_text'][:200]}")

        if pdata.get("key_findings"):
            findings = pdata["key_findings"]
            if isinstance(findings, list) and findings:
                for f in findings[:3]:
                    if isinstance(f, str) and len(f) > 10:
                        sections.append(f"  - {f[:200]}")

        if pdata.get("limitations"):
            lims = pdata["limitations"]
            if isinstance(lims, list) and lims:
                sections.append(f"Limitations: {'; '.join(str(l)[:120] for l in lims[:5])}")

    # Contradictions
    if evidence["contradictions"]:
        sections.append("\n## CONTRADICTIONS BETWEEN PAPERS")
        for c in evidence["contradictions"][:10]:
            sections.append(f"- {c['description'][:200]}")
            if c.get("condition_diff"):
                sections.append(f"  Condition diff: {c['condition_diff'][:100]}")

    # Common limitations
    if evidence["limitations"]:
        from collections import Counter
        lim_counts = Counter(evidence["limitations"])
        repeated = [(l, c) for l, c in lim_counts.most_common(10) if c >= 2]
        if repeated:
            sections.append("\n## RECURRING LIMITATIONS (shared by multiple papers)")
            for lim, count in repeated:
                sections.append(f"- [{count} papers] {lim[:120]}")

    # Methods landscape
    if evidence["methods"]:
        sections.append("\n## METHODS LANDSCAPE")
        for m in evidence["methods"][:15]:
            sections.append(f"- {m['method_name']}: {m['paper_count']} papers, {m['dataset_count']} datasets")

    return "\n".join(sections)


def discover_insights(node_id: str) -> tuple[list[dict], int]:
    """Run deep cross-paper reasoning on a taxonomy node."""
    evidence = gather_node_evidence(node_id)

    if evidence["paper_count"] < 5:
        return [], 0

    prompt = build_evidence_prompt(evidence)

    try:
        result, tokens = call_llm_json(SYSTEM_PROMPT, prompt)
        insights = result.get("insights", [])

        # Attach node_id
        for ins in insights:
            ins["node_id"] = node_id

        return insights, tokens
    except Exception as e:
        print(f"Insight discovery error for {node_id}: {e}", flush=True)
        return [], 0


def discover_all_insights(min_papers: int = 10) -> tuple[list[dict], int]:
    """Run insight discovery on all nodes with enough data."""
    # Find nodes with enough papers
    nodes = db.fetchall("""
        SELECT t.id, t.name, COUNT(DISTINCT pt.paper_id) as paper_count
        FROM taxonomy_nodes t
        JOIN paper_taxonomy pt ON pt.node_id = t.id
        GROUP BY t.id
        HAVING paper_count >= ?
        ORDER BY paper_count DESC
    """, (min_papers,))

    all_insights = []
    total_tokens = 0

    for node in nodes:
        print(f"  Analyzing {node['id']} ({node['paper_count']}p)...", flush=True)
        insights, tokens = discover_insights(node["id"])
        total_tokens += tokens

        for ins in insights:
            all_insights.append(ins)
            print(f"    [{ins['type']}] {ins['title'][:80]}", flush=True)

    return all_insights, total_tokens


def store_insight(insight: dict) -> int:
    """Store an insight in the database."""
    supporting = insight.get("supporting_papers", [])
    if isinstance(supporting, list):
        supporting = json.dumps(supporting)

    cur = db.execute(
        """INSERT INTO insights
           (node_id, insight_type, title, hypothesis, evidence, experiment, impact,
            novelty_score, feasibility_score, supporting_papers)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (insight["node_id"], insight["type"], insight["title"],
         insight["hypothesis"], insight.get("evidence", ""),
         insight["experiment"], insight.get("impact", ""),
         insight.get("novelty_score", 0), insight.get("feasibility_score", 0),
         supporting)
    )
    db.commit()
    return cur.lastrowid
