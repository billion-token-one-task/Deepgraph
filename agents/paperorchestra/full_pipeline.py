"""PaperOrchestra §4 full pipeline: Step1 → parallel(Step2,Step3) → Step4 → Step5 (AgentReview loop)."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from agents.llm_client import call_llm, call_llm_json
from agents.paper_orchestra_prompts import (
    CUTOFF_DATE,
    apply_cutoff_to_outline_tex,
    apply_literature_placeholders,
    apply_plotting_placeholders,
    build_conference_guidelines,
    build_experimental_log_md,
    build_idea_md,
    build_minimal_template_tex,
    load_prompt_tex,
)
from agents.paperorchestra.literature_discovery import run_literature_discovery
from agents.paperorchestra.plotting_orchestra import default_paperbanana_cmd, run_plotting_stage
from agents.paperorchestra.refinement_loop import iterative_refine_with_agentreview
from config import (
    PAPERBANANA_CMD,
    PAPERORCHESTRA_REFINEMENT_ITERS,
    SEMANTIC_SCHOLAR_API_KEY,
)

CITE_PATTERN = re.compile(r"\\cite[a-zA-Z*]*\{([^}]*)\}")


def _cutoff_year() -> int:
    c = (CUTOFF_DATE or "2026-04-01")[:4]
    return int(c) if c.isdigit() else 2026


def _default_cite_keys(claim_citation_map: dict[str, Any], bib_keys: list[str], limit: int = 2) -> list[str]:
    ordered: list[str] = []
    for row in claim_citation_map.values():
        for key in row.get("cite_keys") or []:
            if key not in ordered:
                ordered.append(key)
    for key in bib_keys:
        if key not in ordered:
            ordered.append(key)
    return ordered[:limit]


def _sanitize_latex_citations(tex: str, allowed_keys: set[str], fallback_keys: list[str]) -> str:
    if not tex:
        return tex

    def _replace(match: re.Match[str]) -> str:
        keys = [part.strip() for part in match.group(1).split(",") if part.strip()]
        valid = [key for key in keys if key in allowed_keys]
        if valid:
            return match.group(0).replace(match.group(1), ", ".join(valid))
        if fallback_keys:
            return match.group(0).replace(match.group(1), ", ".join(fallback_keys[:2]))
        return ""

    return CITE_PATTERN.sub(_replace, tex)


def run_paperorchestra_full(
    state: dict,
    literature_block: str,
    paper_ids: list[str],
    iterations: list,
    *,
    figures_dir: Path,
    baseline: float | None,
    metric_name: str,
) -> dict[str, Any]:
    cutoff = CUTOFF_DATE
    cutoff_y = _cutoff_year()
    idea_md = build_idea_md(state, evidence_block=literature_block)
    exp_log_md = build_experimental_log_md(state, [dict(x) for x in iterations])
    template_tex = build_minimal_template_tex(state)
    guidelines = build_conference_guidelines()

    # ── Step 1: Outline Agent ─────────────────────────────────────────────
    outline_sys = apply_cutoff_to_outline_tex(load_prompt_tex("outline_agent"), cutoff)
    outline_user = (
        "--- idea.md ---\n"
        + idea_md
        + "\n--- experimental_log.md ---\n"
        + exp_log_md
        + "\n--- template.tex ---\n"
        + template_tex
        + "\n--- conference_guidelines.md ---\n"
        + guidelines
    )
    o, _ = call_llm_json(outline_sys, outline_user)
    if not isinstance(o, dict):
        o = {}

    pb_cmd = (PAPERBANANA_CMD or "").strip() or default_paperbanana_cmd()

    def _job_plot():
        return run_plotting_stage(
            o,
            state,
            [dict(x) for x in iterations],
            figures_dir,
            baseline=baseline,
            metric_name=metric_name,
            paperbanana_cmd=pb_cmd,
        )

    def _job_lit():
        return run_literature_discovery(
            o,
            [str(x) for x in paper_ids],
            claim_evidence=state.get("claims") or [],
            cutoff_year=cutoff_y,
            api_key=SEMANTIC_SCHOLAR_API_KEY or None,
        )

    # ── Step 2 & 3 in parallel ────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_p = ex.submit(_job_plot)
        fut_l = ex.submit(_job_lit)
        plot_out = fut_p.result()
        lit_out = fut_l.result()

    collected = lit_out["collected_papers"]
    bibtex = lit_out["bibtex"]
    bib_keys = lit_out["bib_keys"]
    claim_citation_map = lit_out.get("claim_citation_map") or {}
    allowed_keys = set(bib_keys)
    fallback_cites = _default_cite_keys(claim_citation_map, bib_keys)
    citation_registry_prompt = [
        {
            "cite_key": row.get("cite_key"),
            "title": row.get("title"),
            "abstract": row.get("abstract"),
            "year": row.get("year"),
            "source": row.get("source"),
            "sources": row.get("sources") or [],
            "source_claim_ids": row.get("source_claim_ids") or [],
            "source_node_ids": row.get("source_node_ids") or [],
            "matched_queries": row.get("matched_queries") or [],
        }
        for row in collected[:120]
    ]

    # Captions via official Plotting Agent prompt (per planned figure)
    plot_prompt_tex = load_prompt_tex("plotting_agent")
    captions: list[dict[str, str]] = []
    pplan = o.get("plotting_plan") if isinstance(o, dict) else None
    plotting_assets = plot_out.get("assets") or []
    if isinstance(pplan, list) and pplan:
        for fig in pplan[:12]:
            if not isinstance(fig, dict):
                continue
            pu = apply_plotting_placeholders(
                plot_prompt_tex,
                task_name=str(state.get("method_name") or "experiment"),
                raw_content=json.dumps(fig, ensure_ascii=False)[:6000],
                description=str(fig.get("title") or fig.get("objective") or ""),
                figure_desc=str(fig.get("objective") or fig.get("title") or ""),
            )
            cap_text, _ = call_llm(pu, "Respond with the plain caption only.")
            captions.append({"figure_id": str(fig.get("figure_id") or ""), "caption": (cap_text or "").strip()})
    elif plotting_assets:
        pu = apply_plotting_placeholders(
            plot_prompt_tex,
            task_name=str(state.get("method_name") or "experiment"),
            raw_content=exp_log_md[:4000],
            description="Main metric trajectory vs iterations / baselines.",
            figure_desc=f"baseline={state.get('baseline_metric_value')}, best={state.get('best_metric_value')}, effect%={state.get('effect_pct')}",
        )
        cap_text, _ = call_llm(pu, "Respond with the plain caption only.")
        captions.append({"figure_id": "fig_metric", "caption": (cap_text or "").strip()})

    p_meta = {"figure_captions": captions, "plotting_executor": plot_out, "plotting_plan": pplan}

    # ── Step 4: Literature Review Agent (Intro + Related in LaTeX) ─────────
    n_papers = len(collected)
    min_cite = min(max(1, n_papers), max(3, min(8, n_papers)))
    lit_sys = apply_literature_placeholders(
        load_prompt_tex("literature_review_agent"),
        paper_count=max(1, n_papers),
        min_cite=min_cite,
        cutoff=cutoff,
    )
    intro_rw = o.get("intro_related_work_plan") if isinstance(o, dict) else {}
    lit_user = (
        "--- template.tex ---\n"
        + template_tex
        + "\n--- intro_related_work_plan ---\n"
        + json.dumps(intro_rw, ensure_ascii=False, default=str)[:16000]
        + "\n--- project_idea ---\n"
        + idea_md[:12000]
        + "\n--- project_experimental_log ---\n"
        + exp_log_md[:12000]
        + "\n--- citation_registry.json ---\n"
        + json.dumps(citation_registry_prompt, ensure_ascii=False)[:200000]
        + "\n--- claim_citation_map.json ---\n"
        + json.dumps(claim_citation_map, ensure_ascii=False)[:120000]
        + "\n--- citation_checklist ---\n"
        + json.dumps(
            {
                "allowed_cite_keys": bib_keys[:120],
                "rule": "Only cite keys listed here. Do not invent any new citation key.",
            },
            ensure_ascii=False,
        )
        + "\n--- collected_papers ---\n"
        + json.dumps(collected, ensure_ascii=False)[:200000]
    )
    lit_tex, _ = call_llm(lit_sys, lit_user)
    lit_tex = _sanitize_latex_citations(lit_tex or "", allowed_keys, fallback_cites)

    # ── Section Writing Agent ─────────────────────────────────────────────
    citation_map: dict[str, dict] = {}
    for row in collected[:80]:
        k = row.get("cite_key")
        if k:
            citation_map[k] = {
                "title": row.get("title"),
                "abstract": (row.get("abstract") or "")[:2000],
                "source_claim_ids": row.get("source_claim_ids") or [],
                "source_node_ids": row.get("source_node_ids") or [],
            }
    sec_sys = load_prompt_tex("section_writing_agent")
    fig_list = [a.get("figure_id") for a in plot_out.get("assets") or []] or [c.get("figure_id") for c in captions]
    sec_user = (
        "--- outline.json ---\n"
        + json.dumps(o, ensure_ascii=False, default=str)[:24000]
        + "\n--- idea.md ---\n"
        + idea_md[:12000]
        + "\n--- experimental_log.md ---\n"
        + exp_log_md[:12000]
        + "\n--- citation_map.json ---\n"
        + json.dumps({k: citation_map.get(k, {}) for k in bib_keys[:80]}, ensure_ascii=False)[:120000]
        + "\n--- claim_citation_map.json ---\n"
        + json.dumps(claim_citation_map, ensure_ascii=False)[:120000]
        + "\n--- citation_registry.json ---\n"
        + json.dumps(citation_registry_prompt, ensure_ascii=False)[:200000]
        + "\n--- conference_guidelines.md ---\n"
        + guidelines
        + "\n--- figures_list ---\n"
        + json.dumps(fig_list, ensure_ascii=False)
        + "\n--- partial_template_after_lit ---\n"
        + (lit_tex or "")[:48000]
    )
    sec_out, _ = call_llm(sec_sys, sec_user)
    sec_out = _sanitize_latex_citations(sec_out or "", allowed_keys, fallback_cites)

    # ── Step 5: Content Refinement + AgentReview accept/revert ─────────────
    ref_sys = load_prompt_tex("content_refinement_agent")

    def _ref_user(prev_tex: str, review_scores: dict[str, Any]) -> str:
        return (
            "--- paper.tex ---\n"
            + prev_tex[:120000]
            + "\n--- experimental_log.md ---\n"
            + exp_log_md[:20000]
            + "\n--- citation_map.json ---\n"
            + json.dumps({k: citation_map.get(k, {}) for k in bib_keys[:80]}, ensure_ascii=False)
            + "\n--- claim_citation_map.json ---\n"
            + json.dumps(claim_citation_map, ensure_ascii=False)
            + "\n--- citation_registry.json ---\n"
            + json.dumps(citation_registry_prompt, ensure_ascii=False)[:200000]
            + "\n--- reviewer_feedback ---\n"
            + json.dumps(
                {
                    "simulated_scores": review_scores,
                    "instruction": "Revise LaTeX to address weaknesses while preserving verified citations. Only use cite keys listed in citation_registry.json.",
                },
                ensure_ascii=False,
            )
            + "\n--- worklog.json ---\n"
            + "[]\n"
        )

    refined_tex, ar_log = iterative_refine_with_agentreview(
        content_refinement_system_tex=ref_sys,
        build_refinement_user=_ref_user,
        initial_tex=sec_out or "",
        max_iters=max(1, PAPERORCHESTRA_REFINEMENT_ITERS),
    )
    refined_tex = _sanitize_latex_citations(refined_tex or "", allowed_keys, fallback_cites)

    # Map to section fragments for assemble_main_tex (fallback split — optional)
    r_frag, _ = call_llm_json(
        "You output JSON only. Keys: introduction, method, experiments, discussion, abstract — LaTeX fragments, no preamble.",
        "Split the following LaTeX body into those sections (best effort).\n\n```latex\n" + refined_tex[:28000] + "\n```",
    )
    if not isinstance(r_frag, dict):
        r_frag = {}

    return {
        "outline": o,
        "plotting": p_meta,
        "literature_discovery": lit_out,
        "literature_text": lit_tex,
        "sections_raw": sec_out,
        "refined": r_frag,
        "refinement_full_text": refined_tex,
        "agentreview_worklog": ar_log,
        "bibtex": bibtex,
        "bib_keys": bib_keys,
        "citation_registry": citation_registry_prompt,
        "claim_citation_map": claim_citation_map,
    }
