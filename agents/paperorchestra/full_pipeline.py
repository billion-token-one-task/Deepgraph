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
from agents.paperorchestra.figure_orchestra import run_postwriting_api_figure_stage
from agents.paperorchestra.plotting_orchestra import default_paperbanana_cmd, run_plotting_stage
from agents.paperorchestra.refinement_loop import iterative_refine_with_agentreview
from config import (
    PAPERBANANA_CMD,
    PAPERORCHESTRA_REFINEMENT_ITERS,
    SEMANTIC_SCHOLAR_API_KEY,
)

CITE_PATTERN = re.compile(r"\\cite[a-zA-Z*]*\{([^}]*)\}")

DEEPGRAPH_WRITING_GUARD = """DeepGraph writing constraints:
- Treat paper_intent.json as the thesis and narrative spine.
- Treat problem_awareness.json as a binding problem-motivation-method-result contract.
- The abstract and first two Introduction paragraphs must answer: what problem, why now, what method, what result, and what limitation.
- Treat publication_evidence_contract.json as binding. Do not claim evidence that is not present.
- Treat evidence_manifest.json and claim_evidence_matrix.json as hard gates, not suggestions.
- Every empirical claim must be grounded in result_packet, iterations, tables/figures, or claim_citation_map.
- A claim may appear in Abstract, Introduction, or Conclusion only when claim_evidence_matrix marks it as allowed there.
- Unverified design intentions belong only in motivation, limitations, or future work.
- Method sections must be implementation-level: training data construction, gain estimator, uncertainty estimation, threshold/budget tuning, deployment pseudocode, complexity, and additional inference cost.
- For routing/gating/selective reasoning methods, include route rate, cost saving, easy/medium/hard breakdown, always/never/confidence/disagreement/random/oracle baselines, simple-case degradation, calibration/reliability, and multi-seed mean/std when present in evidence_manifest.
- Prefer data figures and tables over conceptual diagrams. Never include prompt text, TODOs, placeholders, or artifact-audit wording in the paper body or captions.
- API-generated conceptual figures must be requested only after a manuscript draft exists; early plotting is for artifact-backed data figures.
- Explicitly discuss baseline fairness, required ablations, seed variance, statistical testing, and limitations once, without repeatedly self-disqualifying the contribution.
- Bootstrap/proxy evidence may be reported as engineering validation only, never as full benchmark proof."""


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
    paper_intent_json = json.dumps(state.get("paper_intent") or {}, ensure_ascii=False, default=str)[:12000]
    problem_awareness_json = json.dumps(state.get("problem_awareness") or {}, ensure_ascii=False, default=str)[:12000]
    evidence_contract_json = json.dumps(
        state.get("publication_evidence_contract") or {},
        ensure_ascii=False,
        default=str,
    )[:16000]
    evidence_manifest_json = json.dumps(
        state.get("evidence_manifest") or {},
        ensure_ascii=False,
        default=str,
    )[:24000]
    claim_matrix_json = json.dumps(
        state.get("claim_evidence_matrix") or [],
        ensure_ascii=False,
        default=str,
    )[:18000]
    reviewer_report_json = json.dumps(
        state.get("reviewer_report") or {},
        ensure_ascii=False,
        default=str,
    )[:18000]
    method_requirements_json = json.dumps(
        state.get("method_reproducibility_requirements") or {},
        ensure_ascii=False,
        default=str,
    )[:12000]
    quality_gates_json = json.dumps(
        {
            "quality_gates": state.get("quality_gates") or {},
            "required_evidence": state.get("required_evidence") or {},
            "reviewer_objections": state.get("reviewer_objections") or [],
            "result_packet": state.get("result_packet") or {},
        },
        ensure_ascii=False,
        default=str,
    )[:20000]

    # ── Step 1: Outline Agent ─────────────────────────────────────────────
    outline_sys = DEEPGRAPH_WRITING_GUARD + "\n\n" + apply_cutoff_to_outline_tex(load_prompt_tex("outline_agent"), cutoff)
    outline_user = (
        "--- idea.md ---\n"
        + idea_md
        + "\n--- experimental_log.md ---\n"
        + exp_log_md
        + "\n--- paper_intent.json ---\n"
        + paper_intent_json
        + "\n--- problem_awareness.json ---\n"
        + problem_awareness_json
        + "\n--- publication_evidence_contract.json ---\n"
        + evidence_contract_json
        + "\n--- evidence_manifest.json ---\n"
        + evidence_manifest_json
        + "\n--- claim_evidence_matrix.json ---\n"
        + claim_matrix_json
        + "\n--- reviewer_report.json ---\n"
        + reviewer_report_json
        + "\n--- method_reproducibility_requirements.json ---\n"
        + method_requirements_json
        + "\n--- quality_gates.json ---\n"
        + quality_gates_json
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
    pplan = plot_out.get("plotting_plan_used") or (o.get("plotting_plan") if isinstance(o, dict) else None)
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
    lit_sys = DEEPGRAPH_WRITING_GUARD + "\n\n" + apply_literature_placeholders(
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
        + "\n--- paper_intent.json ---\n"
        + paper_intent_json
        + "\n--- problem_awareness.json ---\n"
        + problem_awareness_json
        + "\n--- publication_evidence_contract.json ---\n"
        + evidence_contract_json
        + "\n--- evidence_manifest.json ---\n"
        + evidence_manifest_json
        + "\n--- claim_evidence_matrix.json ---\n"
        + claim_matrix_json
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
    sec_sys = DEEPGRAPH_WRITING_GUARD + "\n\n" + load_prompt_tex("section_writing_agent")
    caption_by_id = {str(c.get("figure_id") or ""): str(c.get("caption") or "") for c in captions if isinstance(c, dict)}
    fig_list = []
    for asset in plot_out.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        raw_path = asset.get("pdf_path") or asset.get("path") or asset.get("svg_path") or ""
        if not raw_path:
            continue
        fid = str(asset.get("figure_id") or Path(raw_path).stem)
        fig_list.append(
            {
                "figure_id": fid,
                "file": f"figures/{Path(raw_path).name}",
                "caption": caption_by_id.get(fid) or asset.get("objective") or asset.get("title") or fid,
            }
        )
    if not fig_list:
        fig_list = [{"figure_id": str(c.get("figure_id") or ""), "file": "", "caption": str(c.get("caption") or "")} for c in captions]
    sec_user = (
        "--- outline.json ---\n"
        + json.dumps(o, ensure_ascii=False, default=str)[:24000]
        + "\n--- idea.md ---\n"
        + idea_md[:12000]
        + "\n--- experimental_log.md ---\n"
        + exp_log_md[:12000]
        + "\n--- paper_intent.json ---\n"
        + paper_intent_json
        + "\n--- problem_awareness.json ---\n"
        + problem_awareness_json
        + "\n--- publication_evidence_contract.json ---\n"
        + evidence_contract_json
        + "\n--- evidence_manifest.json ---\n"
        + evidence_manifest_json
        + "\n--- claim_evidence_matrix.json ---\n"
        + claim_matrix_json
        + "\n--- method_reproducibility_requirements.json ---\n"
        + method_requirements_json
        + "\n--- quality_gates.json ---\n"
        + quality_gates_json
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

    postwrite_figures = run_postwriting_api_figure_stage(
        o,
        state,
        sec_out,
        figures_dir,
        paperbanana_cmd=pb_cmd,
    )
    if isinstance(postwrite_figures, dict) and postwrite_figures.get("assets"):
        plot_out.setdefault("assets", []).extend(postwrite_figures.get("assets") or [])
        p_meta["postwriting_api_figure_stage"] = postwrite_figures
        for asset in postwrite_figures.get("assets") or []:
            if not isinstance(asset, dict):
                continue
            raw_path = asset.get("path") or asset.get("svg_path") or asset.get("pdf_path") or ""
            fid = str(asset.get("figure_id") or (Path(raw_path).stem if raw_path else "postwriting_figure"))
            caption = str(asset.get("objective") or asset.get("title") or fid)
            captions.append({"figure_id": fid, "caption": caption})
            if raw_path:
                fig_list.append(
                    {
                        "figure_id": fid,
                        "file": f"figures/{Path(raw_path).name}",
                        "caption": caption,
                    }
                )
    else:
        p_meta["postwriting_api_figure_stage"] = postwrite_figures

    # ── Step 5: Content Refinement + AgentReview accept/revert ─────────────
    ref_sys = DEEPGRAPH_WRITING_GUARD + "\n\n" + load_prompt_tex("content_refinement_agent")

    def _ref_user(prev_tex: str, review_scores: dict[str, Any]) -> str:
        return (
            "--- paper.tex ---\n"
            + prev_tex[:120000]
            + "\n--- experimental_log.md ---\n"
            + exp_log_md[:20000]
            + "\n--- paper_intent.json ---\n"
            + paper_intent_json
            + "\n--- problem_awareness.json ---\n"
            + problem_awareness_json
            + "\n--- publication_evidence_contract.json ---\n"
            + evidence_contract_json
            + "\n--- evidence_manifest.json ---\n"
            + evidence_manifest_json
            + "\n--- claim_evidence_matrix.json ---\n"
            + claim_matrix_json
            + "\n--- reviewer_report.json ---\n"
            + reviewer_report_json
            + "\n--- method_reproducibility_requirements.json ---\n"
            + method_requirements_json
            + "\n--- quality_gates.json ---\n"
            + quality_gates_json
            + "\n--- citation_map.json ---\n"
            + json.dumps({k: citation_map.get(k, {}) for k in bib_keys[:80]}, ensure_ascii=False)
            + "\n--- claim_citation_map.json ---\n"
            + json.dumps(claim_citation_map, ensure_ascii=False)
            + "\n--- citation_registry.json ---\n"
            + json.dumps(citation_registry_prompt, ensure_ascii=False)[:200000]
            + "\n--- figures_list ---\n"
            + json.dumps(fig_list, ensure_ascii=False, default=str)[:60000]
            + "\n--- reviewer_feedback ---\n"
            + json.dumps(
                {
                    "simulated_scores": review_scores,
                    "instruction": "Revise LaTeX to address weaknesses while preserving verified citations. Only use cite keys listed in citation_registry.json. If postwriting API figures are listed, use them only when they clarify the problem-method-result spine.",
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
