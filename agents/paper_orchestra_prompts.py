"""Official PaperOrchestra agent prompts (verbatim .tex from arXiv:2604.05018 source).

Files live under ``prompts/paper_orchestra/*.tex`` (copied from the paper's TeX submission).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from agents.evidence_planner import summarize_evidence_plan
from agents.paperorchestra.writing_standard import MANUSCRIPT_WRITING_STANDARD_TEXT
from agents.paper_title_policy import TITLE_NAMING_STANDARD_TEXT
from agents.paperorchestra.venue_policy import generic_template_tex, infer_submission_target
from agents.stage_prompts import prompt_block
from config import MANUSCRIPT_LATEX_TEMPLATE, PROJECT_ROOT

PROMPT_DIR = PROJECT_ROOT / "prompts" / "paper_orchestra"

# Timeline rule in Song et al.; override via env if needed.
CUTOFF_DATE = os.getenv("DEEPGRAPH_PAPERORCHESTRA_CUTOFF_DATE", "2026-04-01").strip()


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
        MANUSCRIPT_WRITING_STANDARD_TEXT,
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


def build_minimal_template_tex(state: dict, venue_target=None) -> str:
    """Tiny venue-aware skeleton listing section commands."""
    target = venue_target or infer_submission_target(state, configured_template=MANUSCRIPT_LATEX_TEMPLATE)
    title = (state.get("title") or "Title").replace("&", r"\&")
    if target.template == "iclr2026":
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
\section{{Related Work}}
\section{{Method}}
\section{{Experiments}}
\section{{Discussion}}
\section{{Conclusion}}
\bibliographystyle{{iclr2026_conference}}
\bibliography{{references}}
\end{{document}}
"""
    return generic_template_tex(state, target)


def build_conference_guidelines(state: dict | None = None, venue_target=None) -> str:
    target = venue_target or infer_submission_target(state or {}, configured_template=MANUSCRIPT_LATEX_TEMPLATE)
    anonymity = "double blind; do not reveal DeepGraph operators or author identities" if target.double_blind else "follow the selected journal identity policy"
    guideline_sources = ", ".join(target.guideline_files) if target.guideline_files else "none configured"
    return f"""Target: {target.label}.
Routing key: {target.key}.
Routing reason: {target.route_reason}.
Template policy: {target.template}; bibliography style: {target.bibliography_style}.
Page policy: {target.page_limit}.
Guideline source files: {guideline_sources}.
Submission mode: {anonymity}.
Venue-specific rules: {target.guidelines}
Use PDFLaTeX; embed vector figures when possible.
Problem-awareness spine: Abstract and Introduction must make clear what problem, what motivation, what method, and what result.
Every major claim must be tied to completed evidence in the provided result packet.
Do not present bootstrap probes, proxy-only runs, or synthetic smoke tests as full benchmark validation.
Explicitly address reviewer objections, baseline fairness, ablation coverage, seed variance, and statistical tests.
Figures should be generated from benchmark artifacts first; API-generated method diagrams may be requested only after experiment results and a manuscript draft exist.
Paper Contract is binding: paper_contract.json fixes target, evidence scope, claims, metrics, terminology, and banned expressions.
Title policy is binding: do not use raw hypothesis sentences as titles; use a symbolic/acronym title plus a descriptive subtitle.
{TITLE_NAMING_STANDARD_TEXT}

""" + MANUSCRIPT_WRITING_STANDARD_TEXT
