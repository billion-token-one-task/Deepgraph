"""Official PaperOrchestra agent prompts (verbatim .tex from arXiv:2604.05018 source).

Files live under ``prompts/paper_orchestra/*.tex`` (copied from the paper's TeX submission).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from agents.evidence_planner import summarize_evidence_plan
from agents.stage_prompts import prompt_block
from config import MANUSCRIPT_LATEX_TEMPLATE, PROJECT_ROOT
from agents.manuscript_templates import get_adapter
from agents.manuscript_templates.style_guides import build_venue_style_guidelines_block

PROMPT_DIR = PROJECT_ROOT / "prompts" / "paper_orchestra"
VENUE_STYLES_DIR = PROJECT_ROOT / "prompts" / "venue_styles"

# Timeline rule in Song et al.; override via env if needed.
CUTOFF_DATE = os.getenv("DEEPGRAPH_PAPERORCHESTRA_CUTOFF_DATE", "2026-04-01").strip()


def load_manuscript_quality_gates() -> str:
    """Shared manuscript quality rules (evidence alignment, stats, anti-patterns)."""
    path = VENUE_STYLES_DIR / "_MANUSCRIPT_QUALITY_GATES.md"
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return ""


def load_experiment_table_requirements() -> str:
    """Top-venue experiment table layout rules (booktabs, concise headers, rowcolor)."""
    path = PROJECT_ROOT / "prompts" / "experiment_table_requirements.md"
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return ""


def load_prompt_tex(name: str) -> str:
    """Load ``{name}.tex`` (e.g. ``outline_agent``)."""
    path = PROMPT_DIR / f"{name}.tex"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing PaperOrchestra prompt {path} (see arXiv:2604.05018 source under content/prompts/agents/)"
        )
    return path.read_text(encoding="utf-8")


def apply_cutoff_to_outline_tex(tex: str, cutoff: str) -> str:
    """Replace ``\\{cutoff\\_date\\}`` placeholder from the outline agent .tex."""
    return tex.replace(r"\{cutoff\_date\}", cutoff)


def apply_literature_placeholders(tex: str, *, paper_count: int, min_cite: int, cutoff: str) -> str:
    """Fill ``\\{paper\\_count\\}``, ``\\{min\\_cite\\_paper\\_count\\}``, ``\\{cutoff\\_date\\}``."""
    out = tex
    out = out.replace(r"\{paper\_count\}", str(paper_count))
    out = out.replace(r"\{min\_cite\_paper\_count\}", str(min_cite))
    out = out.replace(r"\{cutoff\_date\}", cutoff)
    return out


def apply_plotting_placeholders(
    tex: str,
    *,
    task_name: str,
    raw_content: str,
    description: str,
    figure_desc: str,
) -> str:
    """Fill plotting agent placeholders."""
    out = tex
    out = out.replace(r"\{task\_name\}", task_name)
    out = out.replace(r"\{raw\_content\}", raw_content)
    out = out.replace(r"\{description\}", description)
    out = out.replace(r"\{figure\_desc\}", figure_desc)
    return out


def build_idea_md(state: dict, *, evidence_block: str) -> str:
    """Synthetic ``idea.md`` from DeepGraph canonical state."""
    evidence_plan = state.get("evidence_plan") if isinstance(state.get("evidence_plan"), dict) else {}
    paper_intent = state.get("paper_intent") if isinstance(state.get("paper_intent"), dict) else {}
    problem_awareness = state.get("problem_awareness") if isinstance(state.get("problem_awareness"), dict) else {}
    publication_contract = (
        state.get("publication_evidence_contract")
        if isinstance(state.get("publication_evidence_contract"), dict)
        else {}
    )
    claim_route = (
        state.get("claim_route")
        if isinstance(state.get("claim_route"), dict)
        else publication_contract.get("claim_route") if isinstance(publication_contract.get("claim_route"), dict) else {}
    )
    reviewer_objections = state.get("reviewer_objections") if isinstance(state.get("reviewer_objections"), list) else []
    lines = [
        f"# {state.get('title', 'Untitled')}",
        "",
        "## Claim Route",
        json.dumps(claim_route, indent=2, ensure_ascii=False)[:3000],
        "",
        "## Paper Intent",
        prompt_block("problem_framing_agent", "result_synthesis_agent", "manuscript_writer", "evidence_auditor"),
        "",
        json.dumps(paper_intent, indent=2, ensure_ascii=False)[:6000],
        "",
        "## Problem Awareness Contract",
        "Every paper draft must answer, in order: what problem, what motivation, what method, what result, and what limitation.",
        json.dumps(problem_awareness, indent=2, ensure_ascii=False)[:6000],
        "",
        "## Problem",
        str(state.get("problem_statement") or ""),
        "",
        "## Method",
        f"{state.get('method_name', '')}: {state.get('method_summary', '')}",
        "",
        "## Contributions",
        "\n".join(f"- {c}" for c in (state.get("contributions") or [])),
        "",
        "## Adaptive Evidence Plan",
        summarize_evidence_plan(evidence_plan),
        "",
        "## Publication Evidence Contract",
        json.dumps(publication_contract, indent=2, ensure_ascii=False)[:8000],
        "",
        "## Benchmark Evidence Boundary",
        "Sanity/proxy/bootstrap results are preliminary. Full benchmark claims require the completed benchmark manifest job matrix and required artifacts.",
        "",
        "## Reviewer Objections To Address",
        "\n".join(f"- {x}" for x in reviewer_objections[:8]),
        "",
        "## Evidence context (from graph)",
        evidence_block[:12000],
    ]
    return "\n".join(lines)


def build_experimental_log_md(state: dict, iterations: list[dict]) -> str:
    """Synthetic ``experimental_log.md`` from iterations + run metrics."""
    rows = []
    for it in iterations:
        rows.append(
            {
                "iteration": it.get("iteration_number"),
                "phase": it.get("phase"),
                "metric_value": it.get("metric_value"),
                "status": it.get("status"),
                "description": (it.get("description") or "")[:500],
            }
        )
    body = {
        "baseline_metric_name": state.get("baseline_metric_name"),
        "baseline_metric_value": state.get("baseline_metric_value"),
        "best_metric_value": state.get("best_metric_value"),
        "effect_pct": state.get("effect_pct"),
        "verdict": state.get("verdict"),
        "paper_intent": state.get("paper_intent") or {},
        "publication_evidence_contract": state.get("publication_evidence_contract") or {},
        "quality_gates": state.get("quality_gates") or {},
        "required_evidence": state.get("required_evidence") or {},
        "problem_awareness": state.get("problem_awareness") or {},
        "result_packet": state.get("result_packet") or {},
        "iterations": rows,
    }
    return "# Experimental log\n\n```json\n" + json.dumps(body, indent=2, ensure_ascii=False)[:24000] + "\n```\n"


def build_minimal_template_tex(state: dict, *, template_id: str | None = None) -> str:
    """Skeleton listing section commands; ICLR assets only when ``template_id`` is iclr2026."""
    title = (state.get("title") or "Title").replace("&", r"\&")
    tid = (template_id or MANUSCRIPT_LATEX_TEMPLATE or "iclr2026").strip()
    intro_hint = (
        r"\paragraph{Contributions.}" "\n"
        r"This paper makes the following contributions:" "\n"
        r"\begin{itemize}" "\n"
        r"  \item % Outline agent: fill 3--5 contribution bullets here." "\n"
        r"\end{itemize}"
    )
    if tid == "iclr2026":
        return rf"""\documentclass{{article}}
\usepackage{{iclr2026_conference,times}}
\input{{math_commands.tex}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{amsmath,amssymb}}
\usepackage{{hyperref}}
\usepackage{{url}}
\title{{{title}}}
\author{{Anonymous authors\\Paper under double-blind review}}
\begin{{document}}
\maketitle
\begin{{abstract}}
\end{{abstract}}
\section{{Introduction}}
{intro_hint}
\section{{Related Work}}
\section{{Method}}
\section{{Experiments}}
\section{{Discussion}}
\section{{Conclusion}}
\bibliographystyle{{iclr2026_conference}}
\bibliography{{references}}
\end{{document}}
"""
    return rf"""\documentclass{{article}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{amsmath,amssymb}}
\usepackage{{hyperref}}
\usepackage{{url}}
\title{{{title}}}
\author{{Anonymous authors\\Paper under double-blind review}}
\begin{{document}}
\maketitle
\begin{{abstract}}
\end{{abstract}}
\section{{Introduction}}
{intro_hint}
\section{{Related Work}}
\section{{Method}}
\section{{Experiments}}
\section{{Discussion}}
\section{{Conclusion}}
\bibliographystyle{{plain}}
\bibliography{{references}}
\end{{document}}
"""


def build_conference_guidelines(template_id: str | None = None) -> str:
    """Venue-specific writing rules + adapter metadata for PaperOrchestra agents."""
    tid = (template_id or MANUSCRIPT_LATEX_TEMPLATE or "iclr2026").strip()
    try:
        adapter = get_adapter(tid)
        venue_meta = (
            f"Target venue: {adapter.venue_label} (template_id={tid}).\n"
            f"Column layout: {adapter.column_layout}.\n"
            f"Bibliography style: {adapter.bibstyle_name}.\n"
            f"Soft page budget (main text): {adapter.max_pages} pages.\n"
            "Submission mode: double blind unless arxiv_plain; do not reveal DeepGraph operators.\n"
        )
    except KeyError:
        venue_meta = f"Target venue: {tid} (adapter not registered; use generic rules).\n"

    quality_gates = load_manuscript_quality_gates()
    table_requirements = load_experiment_table_requirements()
    evidence_rules = """
Evidence and claims (all venues) — see prompts/venue_styles/_SECTION_WRITING_FRAMEWORK.md:
- Reference corpus medians (workspace/pdfs, n~200): Abstract ~183 words; Introduction ~659 words;
  Related Work ~540 words; Method ~862 words; Experiments ~494 words; Conclusion ~154 words.
- HARD page budget: compiled main text (Abstract through Conclusion) MUST fill exactly the venue page
  limit—Conclusion ends at the bottom of the last allowed page, not one line over or under.
- HARD tables: use booktabs publication tables (tab:main_results + tab:ablations), not bare tabular dumps.
  See prompts/experiment_table_requirements.md: data-driven columns (CI, Delta, Rel.), compact headers,
  no vertical rules, subtle rowcolor (not blanket bold-best), tabularx fill-width, demote config tables to prose.
- HARD ablations: Experiments MUST include \\subsection{Ablation Study} plus tab:ablations with one row per
  required ablation variant; discuss which components matter and report deltas vs full model.
- HARD experiments: multi-dataset breakdown, seed variance / CI, statistical test, compute budget—not three
  scalar scores only.
- Introduction MUST include \\paragraph{Contributions.} with itemize (3--4 bullets).
- Related Work: 2--4 thematic \\subsections; Method: overview figure + mechanistic subsections.
- Intro cites macro literature (~10--20); Related Work cites micro/SOTA (~30--50).
- Do not present bootstrap probes or smoke tests as full benchmark validation.
""".strip()

    style_block = build_venue_style_guidelines_block(tid)
    parts = [venue_meta, style_block, evidence_rules]
    if table_requirements:
        parts.append(f"\n## Experiment table requirements (binding)\n{table_requirements}")
    if quality_gates:
        parts.append(f"\n## Manuscript quality gates (binding)\n{quality_gates}")
    return "\n".join(parts)
