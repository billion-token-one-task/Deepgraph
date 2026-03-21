"""Bridge: DeepGraph Insights → EvoScientist Research Proposals.

Takes a DeepGraph insight (with its supporting evidence from the knowledge graph)
and produces an enriched research proposal that EvoScientist can execute as a
full research plan with experiment design, code, and analysis.
"""

import json
import os
import subprocess
import sqlite3
import threading
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "deepgraph.db"


def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def gather_context(insight_id: int) -> dict:
    """Gather all supporting evidence for an insight from the knowledge graph."""
    conn = get_conn()

    insight = dict(conn.execute("SELECT * FROM insights WHERE id = ?", (insight_id,)).fetchone())

    node_id = insight["node_id"]

    # Get papers in this node AND child nodes
    papers = [dict(r) for r in conn.execute("""
        SELECT DISTINCT p.id, p.title, p.abstract, p.authors, p.pdf_url
        FROM papers p
        JOIN paper_taxonomy pt ON pt.paper_id = p.id
        WHERE (pt.node_id = ? OR pt.node_id LIKE ? || '.%')
          AND p.status NOT IN ('error', 'ingested')
        ORDER BY p.created_at DESC
        LIMIT 30
    """, (node_id, node_id)).fetchall()]

    paper_ids = [p["id"] for p in papers]
    if not paper_ids:
        conn.close()
        return {"insight": insight, "papers": [], "claims": [], "methods": [],
                "contradictions": [], "paper_insights": [], "results_sample": []}

    placeholders = ",".join("?" * len(paper_ids))

    # Key claims from these papers
    claims = [dict(r) for r in conn.execute(f"""
        SELECT paper_id, claim_text, claim_type, method_name, dataset_name,
               metric_name, metric_value, conditions
        FROM claims
        WHERE paper_id IN ({placeholders})
        ORDER BY metric_value DESC NULLS LAST
        LIMIT 40
    """, paper_ids).fetchall()]

    # Methods
    methods = [dict(r) for r in conn.execute(f"""
        SELECT name, category, description, key_innovation, first_paper_id, builds_on
        FROM methods
        WHERE first_paper_id IN ({placeholders})
        LIMIT 20
    """, paper_ids).fetchall()]

    # Contradictions involving these papers
    contradictions = [dict(r) for r in conn.execute(f"""
        SELECT c.description, c.condition_diff, c.hypothesis, c.severity,
               ca.paper_id as paper_a, ca.claim_text as claim_a,
               cb.paper_id as paper_b, cb.claim_text as claim_b
        FROM contradictions c
        JOIN claims ca ON ca.id = c.claim_a_id
        JOIN claims cb ON cb.id = c.claim_b_id
        WHERE ca.paper_id IN ({placeholders}) OR cb.paper_id IN ({placeholders})
    """, paper_ids + paper_ids).fetchall()]

    # Paper insights (plain summaries, limitations, open questions)
    paper_insights = [dict(r) for r in conn.execute(f"""
        SELECT paper_id, problem_statement, approach_summary, key_findings,
               limitations, open_questions, work_type
        FROM paper_insights
        WHERE paper_id IN ({placeholders})
        LIMIT 20
    """, paper_ids).fetchall()]

    # Sample results with numbers
    results_sample = [dict(r) for r in conn.execute(f"""
        SELECT paper_id, method_name, dataset_name, metric_name, metric_value, conditions
        FROM results
        WHERE paper_id IN ({placeholders})
        AND metric_value IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 30
    """, paper_ids).fetchall()]

    conn.close()
    return {
        "insight": insight,
        "papers": papers,
        "claims": claims,
        "methods": methods,
        "contradictions": contradictions,
        "paper_insights": paper_insights,
        "results_sample": results_sample,
    }


def format_proposal(ctx: dict) -> str:
    """Format gathered context into a rich research proposal for EvoScientist."""
    ins = ctx["insight"]

    # Build paper reference list
    paper_refs = ""
    for p in ctx["papers"][:20]:
        paper_refs += f"- [{p['id']}] {p['title']}\n"

    # Build claims summary
    claims_text = ""
    for c in ctx["claims"][:25]:
        val = f" = {c['metric_value']}" if c['metric_value'] is not None else ""
        claims_text += f"- [{c['paper_id']}] {c['claim_text'][:200]}"
        if c['metric_name']:
            claims_text += f" ({c['metric_name']}{val})"
        claims_text += "\n"

    # Build contradictions
    contra_text = ""
    for c in ctx["contradictions"]:
        contra_text += f"- CONFLICT: {c['description'][:300]}\n"
        contra_text += f"  Paper A [{c['paper_a']}]: {c['claim_a'][:150]}\n"
        contra_text += f"  Paper B [{c['paper_b']}]: {c['claim_b'][:150]}\n"
        if c['hypothesis']:
            contra_text += f"  Hypothesis: {c['hypothesis'][:200]}\n"
        contra_text += "\n"

    # Build methods
    methods_text = ""
    for m in ctx["methods"][:10]:
        methods_text += f"- **{m['name']}** ({m['category']}): {m['description'][:200]}\n"
        if m['key_innovation']:
            methods_text += f"  Innovation: {m['key_innovation'][:200]}\n"

    # Build limitations & open questions from paper insights
    limitations = []
    open_questions = []
    for pi in ctx["paper_insights"]:
        if pi["limitations"]:
            try:
                lims = json.loads(pi["limitations"]) if pi["limitations"].startswith("[") else [pi["limitations"]]
                for l in lims[:2]:
                    limitations.append(f"[{pi['paper_id']}] {l}")
            except:
                limitations.append(f"[{pi['paper_id']}] {pi['limitations'][:200]}")
        if pi["open_questions"]:
            try:
                qs = json.loads(pi["open_questions"]) if pi["open_questions"].startswith("[") else [pi["open_questions"]]
                for q in qs[:2]:
                    open_questions.append(f"[{pi['paper_id']}] {q}")
            except:
                open_questions.append(f"[{pi['paper_id']}] {pi['open_questions'][:200]}")

    lim_text = "\n".join(f"- {l}" for l in limitations[:15])
    oq_text = "\n".join(f"- {q}" for q in open_questions[:10])

    # Build results table
    results_text = ""
    for r in ctx["results_sample"][:20]:
        results_text += f"- [{r['paper_id']}] {r['method_name']} on {r['dataset_name']}: {r['metric_name']} = {r['metric_value']}\n"

    # Compose the full proposal
    proposal = f"""# Research Proposal: {ins['title']}

## Research Area
{ins['node_id']}

## Insight Type
{ins['insight_type'].replace('_', ' ').title()} (Novelty: {ins['novelty_score']}/5, Feasibility: {ins['feasibility_score']}/5)

## Core Hypothesis
{ins['hypothesis']}

## Evidence Base
{ins['evidence']}

## Proposed Experiment Design
{ins['experiment']}

## Expected Impact
{ins['impact']}

---

# Supporting Evidence from {len(ctx['papers'])} Papers

## Key Papers
{paper_refs}

## Quantitative Claims
{claims_text}

## Methods in This Area
{methods_text}

## Known Contradictions
{contra_text if contra_text else 'None detected in this area.'}

## Shared Limitations Across Papers
{lim_text if lim_text else 'None aggregated yet.'}

## Open Questions from the Literature
{oq_text if oq_text else 'None aggregated yet.'}

## Benchmark Results (Sample)
{results_text}

---

# Instructions for EvoScientist

Based on the above research insight and supporting evidence, please:

1. **Validate the hypothesis** — Search the web for any recent work (2024-2026) that may have already addressed this. If someone has already done this, identify what's still missing.

2. **Design a concrete experiment plan** — Break it into stages:
   - Stage 1: Reproduce key baselines from the cited papers
   - Stage 2: Implement the proposed method/analysis
   - Stage 3: Run controlled comparisons
   - Stage 4: Ablation studies to isolate the effect

3. **Specify exact details**:
   - Which models/architectures to use (with sizes)
   - Which datasets and evaluation metrics
   - What compute is needed (estimate GPU hours)
   - What the success criteria are (specific numbers that would confirm/reject the hypothesis)

4. **Identify risks and fallback plans** — What could go wrong? What's plan B?

5. **Draft a paper outline** — Title, abstract sketch, expected contributions, and how this advances the field.

The goal is a paper-ready research plan that a PhD student could pick up and execute.
"""
    return proposal


def launch_evoscientist(insight_id: int, workdir: str = None) -> dict:
    """Launch EvoScientist with an enriched research proposal from a DeepGraph insight.

    Returns {"status": "started", "workdir": ..., "pid": ...} immediately.
    EvoScientist runs in the background.
    """
    ctx = gather_context(insight_id)
    proposal = format_proposal(ctx)

    # Create workspace for this research
    if workdir is None:
        safe_title = ctx["insight"]["title"][:50].replace(" ", "_").replace("/", "-")
        workdir = str(Path.home() / "research" / f"insight_{insight_id}_{safe_title}")

    os.makedirs(workdir, exist_ok=True)

    # Write the proposal
    proposal_path = Path(workdir) / "research_proposal.md"
    proposal_path.write_text(proposal, encoding="utf-8")

    # Write context as JSON for reference
    ctx_path = Path(workdir) / "deepgraph_context.json"
    # Make serializable
    ctx_ser = {k: v for k, v in ctx.items()}
    ctx_ser["insight"] = {k: (str(v) if not isinstance(v, (str, int, float, type(None))) else v)
                          for k, v in ctx["insight"].items()}
    ctx_path.write_text(json.dumps(ctx_ser, indent=2, default=str), encoding="utf-8")

    # Launch EvoScientist in background
    # Write a short prompt that points to the full proposal file
    evosci_bin = str(Path.home() / "EvoScientist" / ".venv" / "bin" / "EvoSci")
    short_prompt = (
        f"Read the file research_proposal.md in this workspace. "
        f"It contains a detailed research proposal titled: {ctx['insight']['title']}. "
        f"Follow ALL the instructions in that file to produce a complete research plan "
        f"with experiment design, success criteria, risk analysis, and paper outline."
    )
    env = os.environ.copy()
    env["CUSTOM_OPENAI_USE_RESPONSES_API"] = "true"
    proc = subprocess.Popen(
        [
            evosci_bin,
            "--workdir", workdir,
            "--auto-approve",
            "--ui", "cli",
            "-p", short_prompt,
        ],
        stdout=open(Path(workdir) / "evoscientist.log", "w"),
        stderr=subprocess.STDOUT,
        cwd=workdir,
        env=env,
    )

    return {
        "status": "started",
        "insight_id": insight_id,
        "workdir": workdir,
        "pid": proc.pid,
        "proposal_path": str(proposal_path),
    }


def get_research_status(workdir: str) -> dict:
    """Check the status of an EvoScientist research session."""
    workdir = Path(workdir)
    status = {"workdir": str(workdir), "exists": workdir.exists()}

    if not workdir.exists():
        return status

    # Check for output files
    for fname in ["todos.md", "plan.md", "final_report.md", "research_proposal.md"]:
        fpath = workdir / fname
        if fpath.exists():
            status[fname] = {
                "exists": True,
                "size": fpath.stat().st_size,
                "preview": fpath.read_text(encoding="utf-8")[:500],
            }

    # Check log
    log_path = workdir / "evoscientist.log"
    if log_path.exists():
        content = log_path.read_text(encoding="utf-8")
        status["log_lines"] = len(content.split("\n"))
        status["log_tail"] = content[-1000:]

    return status
