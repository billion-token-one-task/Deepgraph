"""Bridge: DeepGraph Insights → EvoScientist Research Proposals.

Takes a DeepGraph insight (with its supporting evidence from the knowledge graph)
and produces an enriched research proposal that EvoScientist can execute as a
full research plan with experiment design, code, and analysis.
"""

import json
import os
import subprocess
import time
from pathlib import Path

from db import database as db


ACTIVE_SESSION_LOG_WINDOW_SECONDS = 15 * 60


def _pid_file_path(workdir: Path) -> Path:
    return workdir / "evoscientist.pid"


def _read_session_pid(workdir: Path) -> int | None:
    pid_path = _pid_file_path(workdir)
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _pid_is_running(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def write_session_pid(workdir: Path, pid: int) -> None:
    _pid_file_path(workdir).write_text(f"{pid}\n", encoding="utf-8")


def active_research_session(workdir: str | Path, *, log_window_seconds: int = ACTIVE_SESSION_LOG_WINDOW_SECONDS) -> dict | None:
    workdir = Path(workdir)
    if not workdir.exists():
        return None
    final_report = workdir / "final_report.md"
    if final_report.exists():
        return None
    pid = _read_session_pid(workdir)
    if _pid_is_running(pid):
        return {"workdir": str(workdir), "pid": pid, "active": True, "reason": "pid_running"}
    log_path = workdir / "evoscientist.log"
    if log_path.exists():
        age_seconds = max(0.0, time.time() - log_path.stat().st_mtime)
        if age_seconds <= max(1, log_window_seconds):
            return {
                "workdir": str(workdir),
                "pid": pid,
                "active": True,
                "reason": "recent_log_activity",
                "log_age_seconds": age_seconds,
            }
    return None


def _json_load(value, default):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _json_list(value) -> list:
    loaded = _json_load(value, [])
    return loaded if isinstance(loaded, list) else []


def _json_dict(value) -> dict:
    loaded = _json_load(value, {})
    return loaded if isinstance(loaded, dict) else {}


def _dedupe(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _paper_id_params(paper_ids: list[str]) -> tuple[str, tuple]:
    placeholders = ",".join("?" for _ in paper_ids)
    return placeholders, tuple(paper_ids)


def _derive_focus_areas(insight: dict) -> tuple[list[str], list[str]]:
    paper_ids = _dedupe(
        [str(x) for x in _json_list(insight.get("supporting_papers"))]
        + [str(x) for x in _json_list(insight.get("source_paper_ids"))]
    )
    node_ids = _dedupe([str(x) for x in _json_list(insight.get("source_node_ids"))])

    for side_key in ("field_a", "field_b"):
        side = _json_dict(insight.get(side_key))
        node_id = str(side.get("node_id") or "").strip()
        if node_id:
            node_ids.append(node_id)
        for paper in side.get("papers") or []:
            if isinstance(paper, dict):
                paper_id = paper.get("paper_id") or paper.get("id")
            else:
                paper_id = paper
            if paper_id:
                paper_ids.append(str(paper_id))

    return _dedupe(node_ids), _dedupe(paper_ids)


def _normalized_insight(insight: dict, node_ids: list[str], paper_ids: list[str]) -> dict:
    proposed_method = _json_dict(insight.get("proposed_method"))
    experimental_plan = _json_dict(insight.get("experimental_plan"))
    hypotheses = [
        insight.get("problem_statement"),
        proposed_method.get("definition"),
        proposed_method.get("one_line"),
        insight.get("transformation"),
        insight.get("formal_structure"),
        insight.get("title"),
    ]
    evidence_bits = [
        insight.get("evidence_summary"),
        insight.get("adversarial_critique"),
        insight.get("related_work_positioning"),
    ]
    impact_bits = [
        insight.get("related_work_positioning"),
        insight.get("falsification"),
        json.dumps(_json_list(insight.get("predictions")), ensure_ascii=False),
    ]
    insight_type = insight.get("mechanism_type") or ("tier_1_paradigm" if int(insight.get("tier") or 0) == 1 else "tier_2_paper_ready")
    return {
        **insight,
        "focus_node_ids": node_ids,
        "focus_paper_ids": paper_ids,
        "insight_type": insight_type,
        "hypothesis": next((x for x in hypotheses if x), ""),
        "evidence": "\n".join(str(x).strip() for x in evidence_bits if x),
        "experiment": json.dumps(experimental_plan or proposed_method, ensure_ascii=False, indent=2) if (experimental_plan or proposed_method) else "",
        "impact": "\n".join(str(x).strip() for x in impact_bits if x),
        "method_name": proposed_method.get("name") or "",
    }


def gather_context(insight_id: int) -> dict:
    """Gather all supporting evidence for a deep_insight from the knowledge graph."""
    row = db.fetchone("SELECT * FROM deep_insights WHERE id = ?", (insight_id,))
    if not row:
        return {
            "insight": {},
            "papers": [],
            "claims": [],
            "methods": [],
            "contradictions": [],
            "paper_insights": [],
            "results_sample": [],
        }
    insight = dict(row)
    node_ids, paper_ids = _derive_focus_areas(insight)

    papers = []
    if paper_ids:
        placeholders, params = _paper_id_params(paper_ids)
        papers = db.fetchall(
            f"""
            SELECT p.id, p.title, p.abstract, p.authors, p.pdf_url
            FROM papers p
            WHERE p.id IN ({placeholders})
            ORDER BY p.created_at DESC
            LIMIT 30
            """,
            params,
        )
    elif node_ids:
        clauses = " OR ".join("(pt.node_id = ? OR pt.node_id LIKE ? || '.%')" for _ in node_ids)
        params: list[str] = []
        for node_id in node_ids:
            params.extend([node_id, node_id])
        papers = db.fetchall(
            f"""
            SELECT DISTINCT p.id, p.title, p.abstract, p.authors, p.pdf_url
            FROM papers p
            JOIN paper_taxonomy pt ON pt.paper_id = p.id
            WHERE ({clauses})
              AND p.status NOT IN ('error', 'ingested')
            ORDER BY p.created_at DESC
            LIMIT 30
            """,
            tuple(params),
        )
        paper_ids = [paper["id"] for paper in papers]

    if not paper_ids:
        return {
            "insight": _normalized_insight(insight, node_ids, paper_ids),
            "papers": [],
            "claims": [],
            "methods": [],
            "contradictions": [],
            "paper_insights": [],
            "results_sample": [],
        }

    placeholders, params = _paper_id_params(paper_ids)
    claims = db.fetchall(
        f"""
        SELECT paper_id, claim_text, claim_type, method_name, dataset_name,
               metric_name, metric_value, conditions
        FROM claims
        WHERE paper_id IN ({placeholders})
        ORDER BY metric_value DESC NULLS LAST
        LIMIT 40
        """,
        params,
    )
    methods = db.fetchall(
        f"""
        SELECT name, category, description, key_innovation, first_paper_id, builds_on
        FROM methods
        WHERE first_paper_id IN ({placeholders})
        LIMIT 20
        """,
        params,
    )
    contradictions = db.fetchall(
        f"""
        SELECT c.description, c.condition_diff, c.hypothesis, c.severity,
               ca.paper_id as paper_a, ca.claim_text as claim_a,
               cb.paper_id as paper_b, cb.claim_text as claim_b
        FROM contradictions c
        JOIN claims ca ON ca.id = c.claim_a_id
        JOIN claims cb ON cb.id = c.claim_b_id
        WHERE ca.paper_id IN ({placeholders}) OR cb.paper_id IN ({placeholders})
        """,
        tuple(paper_ids + paper_ids),
    )
    paper_insights = db.fetchall(
        f"""
        SELECT paper_id, problem_statement, approach_summary, key_findings,
               limitations, open_questions, work_type
        FROM paper_insights
        WHERE paper_id IN ({placeholders})
        LIMIT 20
        """,
        params,
    )
    results_sample = db.fetchall(
        f"""
        SELECT paper_id, method_name, dataset_name, metric_name, metric_value, conditions
        FROM results
        WHERE paper_id IN ({placeholders})
          AND NULLIF(TRIM(CAST(metric_value AS TEXT)), '') IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 30
        """,
        params,
    )

    return {
        "insight": _normalized_insight(insight, node_ids, paper_ids),
        "papers": papers,
        "claims": claims,
        "methods": methods,
        "contradictions": contradictions,
        "paper_insights": paper_insights,
        "results_sample": results_sample,
    }


def format_proposal(ctx: dict) -> str:
    """Format gathered context into a deep_insights-first research proposal."""
    ins = ctx["insight"]
    proposed_method = _json_dict(ins.get("proposed_method"))
    experimental_plan = _json_dict(ins.get("experimental_plan"))
    focus_nodes = ins.get("focus_node_ids") or []
    signal_mix = _json_list(ins.get("signal_mix"))

    paper_refs = "\n".join(f"- [{p['id']}] {p['title']}" for p in ctx["papers"][:20]) or "- No linked papers found yet."

    claims_text = ""
    for c in ctx["claims"][:25]:
        val = f" = {c['metric_value']}" if c.get("metric_value") is not None else ""
        claims_text += f"- [{c['paper_id']}] {c['claim_text'][:200]}"
        if c.get("metric_name"):
            claims_text += f" ({c['metric_name']}{val})"
        claims_text += "\n"

    contra_text = ""
    for c in ctx["contradictions"][:10]:
        contra_text += f"- CONFLICT: {c['description'][:300]}\n"
        contra_text += f"  Paper A [{c['paper_a']}]: {c['claim_a'][:150]}\n"
        contra_text += f"  Paper B [{c['paper_b']}]: {c['claim_b'][:150]}\n"
        if c.get("hypothesis"):
            contra_text += f"  Hypothesis: {c['hypothesis'][:200]}\n"
        contra_text += "\n"

    methods_text = ""
    for m in ctx["methods"][:10]:
        methods_text += f"- **{m['name']}** ({m.get('category') or 'uncategorized'}): {str(m.get('description') or '')[:200]}\n"
        if m.get("key_innovation"):
            methods_text += f"  Innovation: {m['key_innovation'][:200]}\n"

    limitations = []
    open_questions = []
    for pi in ctx["paper_insights"]:
        for item in _json_list(pi.get("limitations")) or ([pi.get("limitations")] if pi.get("limitations") else []):
            limitations.append(f"[{pi['paper_id']}] {str(item)[:200]}")
        for item in _json_list(pi.get("open_questions")) or ([pi.get("open_questions")] if pi.get("open_questions") else []):
            open_questions.append(f"[{pi['paper_id']}] {str(item)[:200]}")

    lim_text = "\n".join(f"- {l}" for l in limitations[:15])
    oq_text = "\n".join(f"- {q}" for q in open_questions[:10])
    results_text = "\n".join(
        f"- [{r['paper_id']}] {r.get('method_name') or 'method'} on {r.get('dataset_name') or 'dataset'}: {r.get('metric_name') or 'metric'} = {r.get('metric_value')}"
        for r in ctx["results_sample"][:20]
    )

    proposal = f"""# Research Proposal: {ins['title']}

## Insight Profile
- ID: {ins.get('id')}
- Tier: {ins.get('tier')}
- Type: {str(ins.get('insight_type') or 'unknown').replace('_', ' ').title()}
- Status: {ins.get('status') or 'candidate'}
- Mechanism: {ins.get('mechanism_type') or 'unspecified'}
- Signal mix: {", ".join(signal_mix) if signal_mix else 'not recorded'}
- Focus nodes: {", ".join(focus_nodes) if focus_nodes else 'not recorded'}

## Core Hypothesis
{ins.get('hypothesis') or 'Not specified.'}

## Evidence Base
{ins.get('evidence') or 'No evidence summary stored yet.'}

## Proposed Method
{json.dumps(proposed_method, ensure_ascii=False, indent=2) if proposed_method else 'No structured proposed_method stored.'}

## Experimental Plan
{json.dumps(experimental_plan, ensure_ascii=False, indent=2) if experimental_plan else (ins.get('experiment') or 'No experimental_plan stored.')}

## Expected Impact
{ins.get('impact') or 'No expected-impact note stored yet.'}

---

# Supporting Evidence from {len(ctx['papers'])} Papers

## Key Papers
{paper_refs}

## Quantitative Claims
{claims_text or 'None linked yet.'}

## Methods in This Area
{methods_text or 'No method records linked yet.'}

## Known Contradictions
{contra_text if contra_text else 'None detected in this area.'}

## Shared Limitations Across Papers
{lim_text if lim_text else 'None aggregated yet.'}

## Open Questions from the Literature
{oq_text if oq_text else 'None aggregated yet.'}

## Benchmark Results (Sample)
{results_text if results_text else 'No structured benchmark results linked yet.'}

---

# Instructions for EvoScientist

Based on the above deep insight and supporting evidence, please:

1. Validate whether the idea is already solved by recent work (2024-2026), and identify the remaining gap if it is only partially covered.
2. Convert the hypothesis into a concrete staged experiment plan with baselines, datasets, metrics, ablations, and success thresholds.
3. State exact implementation and compute requirements, including model sizes, runtimes, GPU-hour estimates, and failure modes.
4. Identify the main risks, confounders, and fallback plans if the current hypothesis does not hold.
5. Draft a paper outline with title, abstract sketch, contributions, and evidence needed for each section.

The goal is a paper-ready research plan that an engineer or PhD student could execute directly.
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
    log_file = open(Path(workdir) / "evoscientist.log", "w")
    proc = subprocess.Popen(
        [
            evosci_bin,
            "--workdir", workdir,
            "--auto-approve",
            "--ui", "cli",
            "-p", short_prompt,
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=workdir,
        env=env,
    )
    write_session_pid(Path(workdir), proc.pid)

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

    pid = _read_session_pid(workdir)
    status["pid"] = pid
    status["pid_running"] = _pid_is_running(pid)
    active = active_research_session(workdir)
    status["active"] = bool(active)
    if active:
        status["active_reason"] = active.get("reason")
        if active.get("log_age_seconds") is not None:
            status["log_age_seconds"] = round(float(active["log_age_seconds"]), 2)

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
