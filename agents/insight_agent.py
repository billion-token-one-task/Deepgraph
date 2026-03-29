"""Insight Agent: Deep cross-paper reasoning to find genuine research opportunities.

NOT "method X wasn't tested on dataset Y" — that's trivial.
Instead: contradictions, ignored limitations, method transfers, paradigm shifts.
"""
import json
from difflib import SequenceMatcher
from agents.llm_client import call_llm_json
from db import database as db
from db import taxonomy as tax


SYSTEM_PROMPT = """You are a principal investigator at a top ML lab reviewing a concentrated body of recent papers. Your job: find the SINGLE most publishable research direction hiding in this evidence — the kind of finding that could anchor a top-venue paper (NeurIPS / ICML / ICLR spotlight level).

You will receive:
1. Papers with their EXACT reported numbers (method, dataset, metric, value)
2. Plain-language summaries including key findings, limitations, and open questions the authors themselves raised
3. Contradictions between papers (with conflicting values)
4. Structured results: method × dataset × metric × value tables
5. EXISTING INSIGHTS already discovered — you must NOT repeat these

Your standard: Would a senior reviewer say "this is a real contribution" or "this is obvious / incremental"?

## WHAT MAKES A TOP-VENUE INSIGHT

A publishable insight has ALL of these:
- A SPECIFIC numerical tension or gap (not "X hasn't been tested on Y")
- A MECHANISTIC hypothesis explaining WHY (not just WHAT)
- An experiment that could be done in 2-4 weeks with available models/data
- A result that would change how practitioners work

## CATEGORIES

### CONTRADICTION ANALYSIS
Two papers report conflicting numbers on comparable setups. You explain the hidden variable.
GOOD: "Paper A gets 63.2 on MMLU with LoRA at 7B, Paper B gets 78.1 vs 82.9 gap at 70B. The hidden variable: rank bottleneck scales with model capacity. A rank-sweep at 1B→70B would prove this."
BAD: "There's a disagreement between papers." (no numbers, no mechanism, no experiment)

### IGNORED LIMITATION
3+ papers share a limitation that is actually a deeper structural problem nobody frames correctly.
GOOD: "4/7 code-gen papers note 'Python only'. The real question isn't language coverage — it's whether token-level generation can handle languages where indentation isn't meaningful and types are explicit."
BAD: "Many papers haven't tested on other languages." (obvious, no depth)

### METHOD TRANSFER
A technique from area A solves a known, quantified problem in area B. The STRUCTURAL match must be precise.
GOOD: "Hard negative mining gives +3.2% in contrastive vision by fighting collapse to easy patterns. Dialogue (38% generic response rate) has the SAME collapse — both optimize similarity metrics gameable by outputting the mean."
BAD: "Technique X from vision could help NLP." (no structural match, no numbers)

### PARADIGM EXHAUSTION
A research direction's returns are measurably diminishing. You identify what should replace it.
GOOD: "Attention variants: +0.3%, +0.2%, +0.1% over 3 papers, asymptoting at 87.4%. Meanwhile data curation alone gives +2.1%. The field should redirect from architecture to data."
BAD: "This area is getting diminishing returns." (no numbers, no alternative)

### ASSUMPTION CHALLENGE
Papers universally assume X, but evidence in the data suggests X is wrong. You name the assumption and the counter-evidence.
GOOD: "8 RLHF papers assume preference consistency. But kappa=0.58 inter-annotator and 23% self-contradiction. With 23% label noise, standard RLHF overfits to annotator bias, not human preference."
BAD: "The assumption might not hold." (vague, no numbers)

Return JSON:
{
  "insights": [
    {
      "type": "contradiction_analysis|ignored_limitation|method_transfer|paradigm_exhaustion|assumption_challenge",
      "title": "One-line title with KEY NUMBERS from the evidence (e.g. '63.2 vs 78.1 gap reveals...')",
      "hypothesis": "A specific, falsifiable scientific claim. State the mechanism, not just the observation.",
      "supporting_papers": ["2401.12345", "2402.67890"],
      "evidence": "Quote EXACT numbers from SPECIFIC papers. Format: 'Paper 2401.12345 reports [method] achieves [value] on [dataset], while 2402.67890 reports [value] under [condition]. This [X]% gap suggests...'",
      "experiment": "Week 1: [setup]. Week 2: [run]. Models: [specific names+sizes]. Datasets: [names]. Metrics: [names]. Baselines: [what to compare]. Success = [falsification criterion].",
      "impact": "If confirmed, practitioners in [subfield] would stop doing [X] and start doing [Y] because [reason].",
      "novelty_score": 1-5,
      "feasibility_score": 1-5
    }
  ]
}

HARD RULES:
- Return 1-3 insights. Return ZERO if nothing reaches top-venue quality.
- EVERY insight MUST cite ≥2 paper IDs AND ≥2 specific numbers from those papers.
- An insight without specific numbers is AUTOMATICALLY rejected. Delete it.
- supporting_papers MUST be non-empty. If you can't cite papers, the insight isn't real.
- The hypothesis must be MECHANISTIC: explain WHY, not just WHAT.
- "Test X on Y" is NEVER an insight. "X's mechanism breaks under condition Y because Z" IS an insight.
- DO NOT repeat any insight listed in the EXISTING INSIGHTS section.
- Novelty 5 = genuine surprise, nobody framed it this way. Novelty 1 = obvious next step anyone would think of.
- Feasibility 5 = 2 weeks with open-source models. Feasibility 1 = needs new infrastructure.
- Return ONLY valid JSON."""


def gather_node_evidence(node_id: str, max_papers: int = 30) -> dict:
    """Gather rich evidence for a taxonomy node for deep reasoning."""

    papers = tax.get_node_papers(node_id, limit=max_papers)

    paper_claims = {}
    for p in papers:
        claims = db.fetchall(
            "SELECT claim_text, claim_type, method_name, dataset_name, metric_name, metric_value "
            "FROM claims WHERE paper_id=? ORDER BY claim_type",
            (p["id"],)
        )

        # Get rich paper insight if available
        pi = db.fetchone(
            "SELECT plain_summary, problem_statement, approach_summary, work_type, "
            "key_findings, limitations, open_questions FROM paper_insights WHERE paper_id=?",
            (p["id"],)
        )

        key_findings = []
        limitations = []
        open_questions = []
        if pi:
            try:
                key_findings = json.loads(pi["key_findings"]) if pi["key_findings"] else []
            except (json.JSONDecodeError, TypeError):
                key_findings = []
            try:
                limitations = json.loads(pi["limitations"]) if pi["limitations"] else []
            except (json.JSONDecodeError, TypeError):
                limitations = []
            try:
                open_questions = json.loads(pi["open_questions"]) if pi["open_questions"] else []
            except (json.JSONDecodeError, TypeError):
                open_questions = []

        paper_claims[p["id"]] = {
            "title": p["title"],
            "claims": claims,
            "plain_summary": pi["plain_summary"] if pi else "",
            "problem_statement": pi["problem_statement"] if pi else "",
            "approach_summary": pi["approach_summary"] if pi else "",
            "work_type": pi["work_type"] if pi else "",
            "key_findings": key_findings,
            "limitations": limitations,
            "open_questions": open_questions,
        }

    # Contradictions
    contradictions = db.fetchall("""
        SELECT c.description, c.condition_diff, c.hypothesis,
               ca.claim_text as claim_a, ca.paper_id as paper_a,
               ca.method_name as method_a, ca.metric_name as metric_a, ca.metric_value as value_a,
               cb.claim_text as claim_b, cb.paper_id as paper_b,
               cb.method_name as method_b, cb.metric_name as metric_b, cb.metric_value as value_b
        FROM contradictions c
        JOIN claims ca ON c.claim_a_id = ca.id
        JOIN claims cb ON c.claim_b_id = cb.id
        JOIN paper_taxonomy pt ON ca.paper_id = pt.paper_id
        JOIN taxonomy_nodes t ON pt.node_id = t.id
        WHERE t.id = ? OR t.id LIKE ? || '.%'
        ORDER BY c.id DESC LIMIT 20
    """, (node_id, node_id))

    # Structured results with actual numbers
    top_results = db.fetchall("""
        SELECT r.paper_id, r.method_name, r.dataset_name, r.metric_name, r.metric_value
        FROM results r
        JOIN result_taxonomy rt ON rt.result_id = r.id
        WHERE (rt.node_id = ? OR rt.node_id LIKE ? || '.%')
          AND r.metric_value IS NOT NULL AND r.metric_value != ''
        ORDER BY r.id DESC LIMIT 80
    """, (node_id, node_id))

    # Methods in this area
    methods = db.fetchall("""
        SELECT r.method_name, COUNT(DISTINCT r.paper_id) as paper_count,
               COUNT(DISTINCT r.dataset_name) as dataset_count,
               GROUP_CONCAT(DISTINCT r.dataset_name) as datasets
        FROM results r
        JOIN result_taxonomy rt ON rt.result_id = r.id
        JOIN taxonomy_nodes t ON rt.node_id = t.id
        WHERE t.id = ? OR t.id LIKE ? || '.%'
        GROUP BY r.method_name
        ORDER BY paper_count DESC LIMIT 20
    """, (node_id, node_id))

    # Existing insights for this node (for dedup)
    existing_insights = db.fetchall(
        "SELECT title, insight_type, hypothesis FROM insights WHERE node_id=? ORDER BY id DESC LIMIT 20",
        (node_id,)
    )

    return {
        "node_id": node_id,
        "paper_count": len(papers),
        "papers": paper_claims,
        "contradictions": contradictions,
        "top_results": top_results,
        "methods": methods,
        "existing_insights": existing_insights,
    }


def build_evidence_prompt(evidence: dict) -> str:
    """Format evidence into a rich prompt for the LLM."""
    node_id = evidence["node_id"]
    node = tax.get_node(node_id)
    node_name = node["name"] if node else node_id

    sections = [f"Research area: {node_name} ({node_id})\nPapers analyzed: {evidence['paper_count']}\n"]

    # Papers with full context
    sections.append("## PAPERS (cite by ID, e.g. '2401.12345')")
    for pid, pdata in list(evidence["papers"].items())[:25]:
        title = pdata["title"]
        sections.append(f"\n### {pid}: {title}")

        if pdata.get("problem_statement"):
            sections.append(f"Problem: {pdata['problem_statement'][:200]}")
        if pdata.get("approach_summary"):
            sections.append(f"Approach: {pdata['approach_summary'][:200]}")

        perf_claims = [c for c in pdata["claims"] if c["claim_type"] == "performance"]
        finding_claims = [c for c in pdata["claims"] if c["claim_type"] == "finding"]

        if perf_claims:
            sections.append("Reported numbers:")
            for c in perf_claims[:10]:
                val = f" = {c['metric_value']}" if c['metric_value'] else ""
                sections.append(f"  - {c['method_name'] or '?'} on {c['dataset_name'] or '?'} [{c['metric_name'] or '?'}]{val}")

        if finding_claims:
            sections.append("Key findings:")
            for c in finding_claims[:5]:
                sections.append(f"  - {c['claim_text'][:250]}")

        if pdata.get("key_findings"):
            for f in pdata["key_findings"][:3]:
                if isinstance(f, str) and len(f) > 10:
                    sections.append(f"  Finding: {f[:250]}")

        if pdata.get("limitations"):
            lim_strs = [str(l)[:150] for l in pdata["limitations"][:4] if isinstance(l, str) and len(l) > 10]
            if lim_strs:
                sections.append(f"Limitations: {'; '.join(lim_strs)}")

        if pdata.get("open_questions"):
            for q in pdata["open_questions"][:2]:
                if isinstance(q, str) and len(q) > 10:
                    sections.append(f"  Open question: {q[:200]}")

    # Structured results table (method × dataset × metric = value)
    if evidence.get("top_results"):
        sections.append("\n## STRUCTURED RESULTS (method × dataset × metric = value)")
        seen = set()
        for r in evidence["top_results"]:
            key = f"{r['method_name']}|{r['dataset_name']}|{r['metric_name']}"
            if key in seen:
                continue
            seen.add(key)
            sections.append(f"  {r['paper_id']}: {r['method_name']} on {r['dataset_name']} [{r['metric_name']}] = {r['metric_value']}")
            if len(seen) >= 40:
                break

    # Contradictions with actual values
    if evidence["contradictions"]:
        sections.append("\n## CONTRADICTIONS (papers disagree on comparable setups)")
        for c in evidence["contradictions"][:10]:
            parts = [f"Paper {c['paper_a']}"]
            if c.get("method_a") and c.get("value_a"):
                parts.append(f"reports {c['method_a']} [{c['metric_a']}] = {c['value_a']}")
            parts.append(f"vs Paper {c['paper_b']}")
            if c.get("method_b") and c.get("value_b"):
                parts.append(f"reports {c['method_b']} [{c['metric_b']}] = {c['value_b']}")
            sections.append(f"- {' '.join(parts)}")
            sections.append(f"  Description: {c['description'][:200]}")
            if c.get("condition_diff"):
                sections.append(f"  Condition diff: {c['condition_diff'][:150]}")

    # Recurring limitations
    all_limitations = []
    for pid, pdata in evidence["papers"].items():
        for lim in pdata.get("limitations", []):
            if isinstance(lim, str) and len(lim) > 10:
                all_limitations.append(f"[{pid}] {lim}")
    if all_limitations:
        sections.append("\n## LIMITATIONS REPORTED BY AUTHORS")
        for lim in all_limitations[:20]:
            sections.append(f"- {lim[:200]}")

    # Open questions from authors
    all_questions = []
    for pid, pdata in evidence["papers"].items():
        for q in pdata.get("open_questions", []):
            if isinstance(q, str) and len(q) > 10:
                all_questions.append(f"[{pid}] {q}")
    if all_questions:
        sections.append("\n## OPEN QUESTIONS RAISED BY AUTHORS")
        for q in all_questions[:15]:
            sections.append(f"- {q[:200]}")

    # Methods landscape
    if evidence["methods"]:
        sections.append("\n## METHODS LANDSCAPE")
        for m in evidence["methods"][:15]:
            datasets = m.get("datasets", "")
            if datasets and len(datasets) > 80:
                datasets = datasets[:80] + "..."
            sections.append(f"- {m['method_name']}: {m['paper_count']} papers, {m['dataset_count']} datasets ({datasets})")

    # Existing insights — LLM must avoid repeating these
    if evidence.get("existing_insights"):
        sections.append("\n## EXISTING INSIGHTS (DO NOT REPEAT THESE — find something NEW)")
        for ins in evidence["existing_insights"]:
            sections.append(f"- [{ins['insight_type']}] {ins['title']}")

    return "\n".join(sections)


def _is_similar(title_a: str, title_b: str, threshold: float = 0.55) -> bool:
    """Check if two insight titles are semantically similar enough to be duplicates."""
    a = title_a.lower().strip()
    b = title_b.lower().strip()
    ratio = SequenceMatcher(None, a, b).ratio()
    if ratio > threshold:
        return True
    words_a = set(a.split())
    words_b = set(b.split())
    if len(words_a) > 3 and len(words_b) > 3:
        overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
        if overlap > 0.65:
            return True
    return False


def _dedup_insight(insight: dict, node_id: str) -> bool:
    """Return True if this insight is too similar to an existing one (should be skipped)."""
    existing = db.fetchall(
        "SELECT title, hypothesis FROM insights WHERE node_id=?",
        (node_id,)
    )
    for ex in existing:
        if _is_similar(insight.get("title", ""), ex["title"], threshold=0.85):
            return True
        if _is_similar(insight.get("hypothesis", ""), ex["hypothesis"], threshold=0.80):
            return True
    return False


def _validate_insight(insight: dict) -> bool:
    """Return True if insight meets minimum quality bar."""
    evidence = insight.get("evidence", "")
    if len(evidence) < 30:
        return False
    hypothesis = insight.get("hypothesis", "")
    if len(hypothesis) < 20:
        return False
    return True


def discover_insights(node_id: str) -> tuple[list[dict], int]:
    """Run deep cross-paper reasoning on a taxonomy node."""
    evidence = gather_node_evidence(node_id)

    if evidence["paper_count"] < 5:
        return [], 0

    prompt = build_evidence_prompt(evidence)

    try:
        result, tokens = call_llm_json(SYSTEM_PROMPT, prompt)
        raw_insights = result.get("insights", [])
        if not raw_insights:
            print(f"[INSIGHT] {node_id}: LLM returned no insights. Keys: {list(result.keys()) if isinstance(result, dict) else type(result).__name__}. Preview: {str(result)[:200]}", flush=True)

        # Filter: validate + dedup
        insights = []
        for ins in raw_insights:
            ins["node_id"] = node_id
            if not _validate_insight(ins):
                print(f"[INSIGHT] Filtered (validation): {ins.get('title', '?')[:60]}", flush=True)
                continue
            if _dedup_insight(ins, node_id):
                print(f"[INSIGHT] Filtered (dedup): {ins.get('title', '?')[:60]}", flush=True)
                continue
            insights.append(ins)
        print(f"[INSIGHT] {node_id}: {len(raw_insights)} raw -> {len(insights)} passed filters", flush=True)

        return insights, tokens
    except Exception as e:
        print(f"Insight discovery error for {node_id}: {e}", flush=True)
        return [], 0


def discover_all_insights(min_papers: int = 10) -> tuple[list[dict], int]:
    """Run insight discovery on all nodes with enough data."""
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
    """Store an insight in the database, with dedup check."""
    if _dedup_insight(insight, insight.get("node_id", "")):
        return -1

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
