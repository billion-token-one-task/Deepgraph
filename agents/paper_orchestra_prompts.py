"""Official PaperOrchestra agent prompts (verbatim .tex from arXiv:2604.05018 source).

Files live under ``prompts/paper_orchestra/*.tex`` (copied from the paper's TeX submission).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from agents.evidence_planner import summarize_evidence_plan
from config import PROJECT_ROOT

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
    lines = [
        f"# {state.get('title', 'Untitled')}",
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
        "iterations": rows,
    }
    return "# Experimental log\n\n```json\n" + json.dumps(body, indent=2, ensure_ascii=False)[:24000] + "\n```\n"


def build_minimal_template_tex(state: dict) -> str:
    """Tiny article skeleton listing section commands (per outline agent requirement)."""
    title = (state.get("title") or "Title").replace("&", r"\&")
    return rf"""\documentclass{{article}}
\usepackage{{graphicx,booktabs,hyperref}}
\title{{{title}}}
\begin{{document}}
\maketitle
\section{{Abstract}}
\section{{Introduction}}
\section{{Related Work}}
\section{{Method}}
\section{{Experiments}}
\section{{Discussion}}
\section{{Conclusion}}
\bibliography{{references}}
\end{{document}}
"""


def build_conference_guidelines() -> str:
    return """Target: top-tier ML venue (NeurIPS / ICML / ICLR / CVPR).
Main paper: up to 9 pages content + unlimited references for many tracks.
Use PDFLaTeX; embed vector figures when possible."""
