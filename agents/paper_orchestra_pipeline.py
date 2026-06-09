"""PaperOrchestra multi-stage manuscript generation (Song et al., arXiv:2604.05018 §4).

Full pipeline: Outline → parallel(Plot generation, Literature discovery+review) → Section writing →
AgentReview-style refinement. Official agent ``.tex`` prompts under ``prompts/paper_orchestra/``.

Bibliography: Semantic Scholar–verified registry merged with evidence-graph papers (real metadata).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from contracts import ContractValidationError
from agents.paper_completeness import (
    audit_citation_registry,
    audit_evidence_completeness,
    latex_sanity_check,
)
from agents.benchmark_audit import full_benchmark_evidence_blockers
from agents.plain_manuscript_reviewer import review_manuscript_plain
from agents.tex_code_agent import repair_latex_bundle
from agents.reference_corpus_audit import audit_against_reference_corpus
from agents.manuscript_pipeline import (
    _bundle_manifest,
    _ensure_dirs,
    _store_assets,
    _write,
    build_manuscript_input_state,
)
from agents.workspace_layout import get_idea_workspace, paper_bundle_root, write_latest_status, write_plan_files
from config import ICLR2026_TEMPLATE_DIR, ICLR2026_TEMPLATE_FILES, REFERENCE_PDF_CORPUS_DIR, SUBMISSION_BUNDLE_FORMATS
from db import database as db
from db.insight_outcomes import OUTCOME_BECAME_MANUSCRIPT, set_outcome
from orchestrator.tracking import log_artifact


def _run_full_pipeline(*args, **kwargs) -> dict:
    from agents.paperorchestra.full_pipeline import run_paperorchestra_full

    return run_paperorchestra_full(*args, **kwargs)


def _json_list(raw) -> list:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    try:
        v = json.loads(raw)
        return v if isinstance(v, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def build_references_bib_from_papers(paper_ids: list[str]) -> tuple[str, list[str]]:
    """Return (bibtex string, list of cite keys actually present in DB)."""
    keys_used: list[str] = []
    chunks: list[str] = []
    for pid in paper_ids:
        row = db.fetchone(
            """
            SELECT id, arxiv_base_id, title, authors, published_date, categories
            FROM papers
            WHERE id=? OR arxiv_base_id=?
            ORDER BY CASE WHEN id=? THEN 0 ELSE 1 END
            LIMIT 1
            """,
            (pid, pid, pid),
        )
        if not row:
            continue
        cite_id = row.get("arxiv_base_id") or row.get("id") or pid
        key = str(cite_id).replace(".", "_").replace("/", "_")
        keys_used.append(key)
        try:
            authors = json.loads(row["authors"]) if row.get("authors") else []
        except (json.JSONDecodeError, TypeError):
            authors = []
        au = " and ".join(authors[:40]) if authors else "Unknown"
        year = "2024"
        pd = row.get("published_date") or ""
        if len(pd) >= 4 and pd[:4].isdigit():
            year = pd[:4]
        title = (row.get("title") or "Untitled").replace("{", "\\{").replace("}", "\\}")
        chunks.append(
            f"@misc{{{key},\n  title = {{{title}}},\n  author = {{{au}}},\n  year = {{{year}}},\n  note = {{arXiv:{cite_id}}}\n}}\n"
        )
    return "\n".join(chunks), keys_used


def _latex_escape(text: str) -> str:
    return (
        str(text or "")
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def _figure_assets(orchestrated: dict) -> list[dict]:
    plotting = orchestrated.get("plotting") or {}
    assets: list[dict] = []
    executor = plotting.get("plotting_executor") if isinstance(plotting, dict) else {}
    if isinstance(executor, dict) and isinstance(executor.get("assets"), list):
        assets.extend(row for row in executor["assets"] if isinstance(row, dict))
    if isinstance(plotting, dict) and isinstance(plotting.get("assets"), list):
        assets.extend(row for row in plotting["assets"] if isinstance(row, dict))
    post = plotting.get("postwriting_api_figure_stage") if isinstance(plotting, dict) else {}
    if isinstance(post, dict) and isinstance(post.get("assets"), list):
        assets.extend(row for row in post["assets"] if isinstance(row, dict))

    deduped: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for asset in assets:
        key = (str(asset.get("figure_id") or ""), str(asset.get("path") or asset.get("pdf_path") or asset.get("svg_path") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(asset)
    return deduped


def _default_figure_caption(figure_id: str, fallback: str) -> str:
    if figure_id == "fig_motivation_symbolic":
        return (
            "Motivation for the proposed reasoning policy: fixed aggregation can discard useful "
            "minority evidence, while retaining every candidate spends unnecessary tokens; the "
            "method targets evidence-aware selection under a compute budget."
        )
    if figure_id == "fig_overview_symbolic":
        return (
            "Overview of the proposed method. Candidate traces are organized with confidence, "
            "cost, and disagreement evidence, and a budget-aware selector decides which reasoning "
            "paths to retain before producing the final answer."
        )
    return fallback


def _figure_caption_map(orchestrated: dict) -> dict[str, str]:
    plotting = orchestrated.get("plotting") or {}
    out: dict[str, str] = {}
    for row in plotting.get("figure_captions") or []:
        if not isinstance(row, dict):
            continue
        fid = str(row.get("figure_id") or "")
        if fid:
            out[fid] = str(row.get("caption") or "")
    return out


def _figure_latex_blocks(orchestrated: dict) -> str:
    captions = _figure_caption_map(orchestrated)
    blocks: list[str] = []
    blocklisted = {
        "fig_metric_trajectory",
        "fig_search_dynamics_keep_discard",
        "fig_benchmark_method_panel",
        "fig_selective_deliberation_utility_comparison",
    }
    for asset in _figure_assets(orchestrated):
        if not isinstance(asset, dict):
            continue
        path = asset.get("path") or asset.get("svg_path") or ""
        if not path:
            continue
        name = Path(path).name
        figure_id = str(asset.get("figure_id") or Path(path).stem)
        if figure_id in blocklisted:
            continue
        if asset.get("kind") == "fallback":
            continue
        if asset.get("kind") == "diagram" and asset.get("stage") != "postwriting_api_figures":
            continue
        caption = _default_figure_caption(
            figure_id,
            captions.get(figure_id) or asset.get("objective") or asset.get("title") or figure_id,
        )
        is_wide = (
            str(asset.get("placement") or "").lower() in {"double", "double_column", "figure*"}
            or str(asset.get("layout") or "") in {"1x3", "1x4"}
            or figure_id == "fig_backend_rank_lines_1x4"
            or str(asset.get("aspect_ratio") or "") == "4:1"
        )
        env = "figure*" if is_wide else "figure"
        width = r"\textwidth" if is_wide else r"\linewidth"
        is_experiment_plot = asset.get("kind") not in {"diagram", "fallback", "blocked"}
        include_lines = [rf"\includegraphics[width={width}]{{figures/{name}}}"]
        if is_experiment_plot:
            include_lines.append(r"\vspace{-0.2em}")
        blocks.append(
            "\n".join(
                [
                    rf"\begin{{{env}}}[t]",
                    r"\centering",
                    *include_lines,
                    rf"\caption{{{caption}}}",
                    rf"\label{{fig:{figure_id}}}",
                    rf"\end{{{env}}}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _concept_figure_blocks(orchestrated: dict, wanted: set[str]) -> str:
    """Return required post-writing concept figures for deterministic placement."""
    captions = _figure_caption_map(orchestrated)
    blocks: list[str] = []
    for asset in _figure_assets(orchestrated):
        figure_id = str(asset.get("figure_id") or "")
        if figure_id not in wanted:
            continue
        path = asset.get("path") or asset.get("svg_path") or ""
        if not path:
            continue
        name = Path(path).name
        caption = _default_figure_caption(
            figure_id,
            captions.get(figure_id) or asset.get("objective") or asset.get("title") or figure_id,
        )
        blocks.append(
            "\n".join(
                [
                    r"\begin{figure*}[t]",
                    r"\centering",
                    rf"\includegraphics[width=\textwidth]{{figures/{name}}}",
                    rf"\caption{{{caption}}}",
                    rf"\label{{fig:{figure_id}}}",
                    r"\end{figure*}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _fallback_related_work(state: dict, orchestrated: dict) -> str:
    lit_tex = (orchestrated.get("literature_text") or "").strip()
    if lit_tex:
        return lit_tex
    registry = orchestrated.get("citation_registry") or []
    snippets = []
    for row in registry[:4]:
        if not isinstance(row, dict):
            continue
        key = row.get("cite_key")
        title = row.get("title")
        year = row.get("year")
        if key and title:
            snippets.append(f"{title} ({year}) is included in the verified registry via \\cite{{{key}}}.")
    return "\n\n".join(snippets) or _latex_escape(str(state.get("evidence_summary") or "Verified prior work is listed in references.bib."))


def _strip_latex_document_shell(fragment: str) -> str:
    """Remove accidental full-document wrappers from section fragments."""
    text = str(fragment or "").strip()
    if not text:
        return ""
    if "\\begin{document}" in text:
        text = text.split("\\begin{document}", 1)[1]
    text = re.sub(r"\\documentclass(?:\[[^\]]*\])?\{[^}]*\}", "", text)
    text = re.sub(r"\\usepackage(?:\[[^\]]*\])?\{[^}]*\}", "", text)
    text = re.sub(r"\\input\{[^}]*\}", "", text)
    text = re.sub(r"\\title\{[^}]*\}", "", text)
    text = re.sub(r"\\author\{[^}]*\}", "", text)
    text = text.replace(r"\maketitle", "")
    text = re.sub(r"\\bibliographystyle\{[^}]*\}", "", text)
    text = re.sub(r"\\bibliography\{[^}]*\}", "", text)
    text = text.replace(r"\end{document}", "")
    text = re.sub(r"\\begin\{abstract\}\s*\\end\{abstract\}", "", text, flags=re.DOTALL)
    return text.strip()


def _strip_abstract_citations(source: str) -> str:
    """Remove citation commands inside the abstract environment.

    Most venues expect the abstract to be self-contained and citation-free.
    This is a hard post-processing guard because refinement models sometimes
    introduce citations even when prompted not to.
    """
    cite_cmd = re.compile(r"\\cite[a-zA-Z*]*(?:\[[^\]]*\]){0,2}\{[^}]*\}")

    def _clean(match: re.Match[str]) -> str:
        body = cite_cmd.sub("", match.group(1))
        body = re.sub(r"\s+([,.;:])", r"\1", body)
        body = re.sub(r"[ \t]{2,}", " ", body)
        return "\\begin{abstract}\n" + body.strip() + "\n\\end{abstract}"

    return re.sub(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        _clean,
        source,
        count=1,
        flags=re.DOTALL,
    )


def _trim_intro_related_sections(fragment: str) -> str:
    """Keep only Introduction/Related Work material from a literature fragment."""
    text = _strip_latex_document_shell(fragment)
    if not text:
        return ""
    allowed = {"introduction", "related work", "related works", "background"}
    sections = list(re.finditer(r"\\section\*?\{([^}]+)\}", text))
    for match in sections:
        title = re.sub(r"\s+", " ", match.group(1).strip().lower())
        if title not in allowed:
            text = text[: match.start()].rstrip()
            break
    return text.strip()


def _inject_problem_spine(intro_related: str, problem_spine: str) -> str:
    """Add a natural problem/motivation/method/result paragraph after the Introduction heading."""
    if not problem_spine.strip():
        return intro_related
    if "This paper studies" in intro_related or "We study" in intro_related[:1200]:
        return intro_related
    related_match = re.search(r"\\section\*?\{Related Work[s]?\}", intro_related, flags=re.IGNORECASE)
    if related_match:
        return intro_related[: related_match.start()].rstrip() + "\n\n" + problem_spine + "\n\n" + intro_related[related_match.start():].lstrip()
    return intro_related.rstrip() + "\n\n" + problem_spine


def assemble_main_tex(state: dict, orchestrated: dict, bundle_format: str) -> str:
    venue = "Conference submission draft" if bundle_format == "conference" else "Journal draft"
    refined = orchestrated.get("refined") if isinstance(orchestrated.get("refined"), dict) else {}
    abs_tex = _strip_latex_document_shell(refined.get("abstract") or "See experiments section for quantitative results.")
    intro = _strip_latex_document_shell(refined.get("introduction") or state.get("problem_statement", ""))
    meth = _strip_latex_document_shell(refined.get("method") or state.get("method_summary", ""))
    exp = _strip_latex_document_shell(refined.get("experiments") or "")
    dis = _strip_latex_document_shell(refined.get("discussion") or "")
    related = _trim_intro_related_sections(_fallback_related_work(state, orchestrated))
    all_figures = _figure_latex_blocks(orchestrated)
    motivation_figures = _concept_figure_blocks(orchestrated, {"fig_motivation_symbolic"})
    overview_figures = _concept_figure_blocks(orchestrated, {"fig_overview_symbolic"})
    experiment_figures = "\n\n".join(
        block
        for block in all_figures.split("\n\n")
        if "fig:fig_motivation_symbolic" not in block and "fig:fig_overview_symbolic" not in block
    )
    results_line = (
        f"Baseline {state['baseline_metric_name']}: {state.get('baseline_metric_value')}; "
        f"best: {state.get('best_metric_value')}; effect \\%: {state.get('effect_pct')}; "
        f"verdict: {state.get('verdict')}."
    )
    problem_awareness = state.get("problem_awareness") if isinstance(state.get("problem_awareness"), dict) else {}
    central_problem = str(problem_awareness.get("central_question") or state.get("problem_statement") or "").strip().replace("?", ".")
    central_problem = re.sub(r"^(?:can|could|does|do|should|is|are)\b", "whether", central_problem, flags=re.IGNORECASE)
    motivation = str(problem_awareness.get("motivation") or state.get("existing_weakness") or "").strip().replace("?", ".")
    method_answer = str(problem_awareness.get("method_answer") or state.get("method_summary") or "").strip().replace("?", ".")
    result_claim = str(problem_awareness.get("result_claim") or results_line).strip().replace("?", ".")
    spine_sentences = []
    if central_problem:
        spine_sentences.append("This paper studies " + central_problem[0].lower() + central_problem[1:] if central_problem[:1].isupper() else "This paper studies " + central_problem)
    if motivation:
        spine_sentences.append("The motivation is that " + motivation[0].lower() + motivation[1:] if motivation[:1].isupper() else "The motivation is that " + motivation)
    if method_answer:
        spine_sentences.append("We address this setting with " + method_answer[0].lower() + method_answer[1:] if method_answer[:1].isupper() else "We address this setting with " + method_answer)
    if result_claim:
        spine_sentences.append("The completed evidence shows " + result_claim[0].lower() + result_claim[1:] if result_claim[:1].isupper() else "The completed evidence shows " + result_claim)
    problem_spine = " ".join(_latex_escape(x.rstrip(".")) + "." for x in spine_sentences if x.strip())
    if "\\section{" in related:
        intro_related = _inject_problem_spine(related, problem_spine)
    else:
        intro_related = rf"""\section{{Introduction}}
{intro}
{problem_spine}
\section{{Related Work}}
{related}"""
    if bundle_format == "conference":
        return rf"""\documentclass{{article}}
\usepackage{{iclr2026_conference,times}}
\input{{math_commands.tex}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{tabularx}}
\usepackage{{amsmath,amssymb}}
\usepackage{{hyperref}}
\usepackage{{url}}
\usepackage[font=normalsize,labelfont=bf]{{caption}}
\captionsetup[figure]{{font=normalsize,labelfont=bf,skip=4pt}}
\setlength{{\abovecaptionskip}}{{4pt}}
\setlength{{\belowcaptionskip}}{{2pt}}
\title{{{state['title']}}}
\author{{Anonymous authors\\Paper under double-blind review}}
\begin{{document}}
\maketitle
\begin{{abstract}}
{abs_tex}
\end{{abstract}}
{intro_related}
{motivation_figures}
\section{{Method}}
{meth}
{overview_figures}
\section{{Experiments}}
{exp}
{experiment_figures}
\section{{Results}}
{results_line}
\section{{Discussion}}
{dis}
\bibliographystyle{{iclr2026_conference}}
\bibliography{{references}}
\end{{document}}
"""
    return rf"""\documentclass{{article}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{tabularx}}
\usepackage[font=normalsize,labelfont=bf]{{caption}}
\captionsetup[figure]{{font=normalsize,labelfont=bf,skip=4pt}}
\setlength{{\abovecaptionskip}}{{4pt}}
\setlength{{\belowcaptionskip}}{{2pt}}
\title{{{state['title']}}}
\author{{DeepGraph Auto Research (PaperOrchestra pipeline)}}
\date{{{venue}}}
\begin{{document}}
\maketitle
\begin{{abstract}}
{abs_tex}
\end{{abstract}}
{intro_related}
{motivation_figures}
\section{{Method}}
{meth}
{overview_figures}
\section{{Experiments}}
{exp}
{experiment_figures}
\section{{Results}}
{results_line}
\section{{Discussion}}
{dis}
\bibliographystyle{{plain}}
\bibliography{{references}}
\end{{document}}
"""


def _ensure_iclr2026_preamble(source: str) -> str:
    """Force an ICLR 2026 submission preamble without touching the paper body."""
    if "\\begin{document}" not in source:
        return source
    preamble, marker, body = source.partition(r"\begin{document}")
    if r"\documentclass" not in preamble:
        preamble = r"\documentclass{article}" + "\n" + preamble
    if "iclr2026_conference" not in preamble:
        preamble = re.sub(
            r"(\\documentclass(?:\[[^\]]*\])?\{[^}]+\}\s*)",
            r"\1\\usepackage{iclr2026_conference,times}" + "\n",
            preamble,
            count=1,
        )
    if "math_commands.tex" not in preamble:
        preamble = preamble.rstrip() + "\n" + r"\input{math_commands.tex}" + "\n"
    if "iclr2026_conference" in preamble and re.search(r"\\usepackage(?:\[[^\]]*\])?\{xcolor\}", preamble):
        preamble = re.sub(r"\\usepackage(?:\[[^\]]*\])?\{xcolor\}\s*", "", preamble)
        if r"\PassOptionsToPackage{table}{xcolor}" not in preamble:
            preamble = re.sub(
                r"(\\documentclass(?:\[[^\]]*\])?\{[^}]+\}\s*)",
                r"\1\PassOptionsToPackage{table}{xcolor}" + "\n",
                preamble,
                count=1,
            )
    packages = ["graphicx", "booktabs", "array", "tabularx", "amsmath,amssymb", "hyperref", "url"]
    if "iclr2026_conference" not in preamble:
        packages.insert(2, "xcolor")
    for package in packages:
        first_pkg = package.split(",", 1)[0]
        if first_pkg not in preamble:
            if package == "xcolor":
                preamble = preamble.rstrip() + "\n" + r"\usepackage[table]{xcolor}" + "\n"
            else:
                preamble = preamble.rstrip() + "\n" + rf"\usepackage{{{package}}}" + "\n"
    if r"\usepackage[font=normalsize,labelfont=bf]{caption}" not in preamble and r"\usepackage{caption}" not in preamble:
        preamble = preamble.rstrip() + "\n" + r"\usepackage[font=normalsize,labelfont=bf]{caption}" + "\n"
    if r"\captionsetup[figure]" not in preamble:
        preamble = preamble.rstrip() + "\n" + r"\captionsetup[figure]{font=normalsize,labelfont=bf,skip=4pt}" + "\n"
    if r"\abovecaptionskip" not in preamble:
        preamble = preamble.rstrip() + "\n" + r"\setlength{\abovecaptionskip}{4pt}" + "\n"
    if r"\belowcaptionskip" not in preamble:
        preamble = preamble.rstrip() + "\n" + r"\setlength{\belowcaptionskip}{2pt}" + "\n"
    if r"\author" not in preamble:
        preamble = preamble.rstrip() + "\n" + r"\author{Anonymous authors\\Paper under double-blind review}" + "\n"
    preamble = re.sub(r"\\usepackage(?:\[[^\]]*\])?\{geometry\}\s*", "", preamble)
    return preamble + marker + body


def normalize_latex_source(text: str, *, force_iclr2026: bool = False) -> str:
    """Strip markdown fences that LLMs sometimes wrap around LaTeX documents."""
    source = (text or "").strip()
    if source.startswith("```"):
        lines = source.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        source = "\n".join(lines).strip()
    if "```" in source:
        source = source.replace("```latex", "").replace("```tex", "").replace("```", "").strip()
    if force_iclr2026:
        source = _ensure_iclr2026_preamble(source)
    uses_iclr = "iclr2026_conference" in source
    if not uses_iclr:
        source = re.sub(r"\\documentclass\{article\}", r"\\documentclass[10pt]{article}", source, count=1)
    if "\\begin{document}" in source and "microtype" not in source and not uses_iclr:
        source = re.sub(
            r"(\\documentclass(?:\[[^\]]*\])?\{[^}]+\}\s*)",
            r"\1\n\\usepackage{microtype}\n",
            source,
            count=1,
        )
    if "\\begin{document}" in source and "geometry" not in source and not uses_iclr:
        source = re.sub(
            r"(\\documentclass(?:\[[^\]]*\])?\{[^}]+\}\s*)",
            r"\1\n\\usepackage[margin=1in]{geometry}\n",
            source,
            count=1,
        )
    preamble_probe, marker_probe, _body_probe = source.partition(r"\begin{document}")
    if marker_probe and r"\date" not in preamble_probe and not uses_iclr:
        source = preamble_probe.rstrip() + "\n\\date{}\n" + marker_probe + _body_probe
    source = re.sub(
        r"(\\maketitle\s*)\\section\{Abstract\}\s*(.*?)(?=\\section\{Introduction\})",
        r"\1\\begin{abstract}\n\2\n\\end{abstract}\n",
        source,
        count=1,
        flags=re.DOTALL,
    )
    if "\\bibliography{" in source and "\\bibliographystyle{" not in source:
        style = "iclr2026_conference" if uses_iclr else "plain"
        source = re.sub(
            r"(\s*)\\bibliography\{",
            rf"\1\\bibliographystyle{{{style}}}\1\\bibliography{{",
            source,
            count=1,
        )
    if uses_iclr:
        source = re.sub(r"\\bibliographystyle\{[^}]+\}", r"\\bibliographystyle{iclr2026_conference}", source)
    source = _strip_abstract_citations(source)
    preamble, marker, body = source.partition(r"\begin{document}")
    if marker:
        needs_algorithm = (
            "\\begin{algorithm}" in body
            and "algorithm}" not in preamble
            and "algorithm2e" not in preamble
        )
        needs_algpseudocode = (
            ("\\begin{algorithmic}" in body or "\\State" in body)
            and "algpseudocode" not in preamble
            and "algorithmic" not in preamble
        )
        needs_cleveref = ("\\Cref" in body or "\\cref" in body) and "cleveref" not in preamble
        needs_ams = (
            any(cmd in body for cmd in ("\\mathbb", "\\operatorname", "\\text", "\\eqref"))
            and "amsmath" not in preamble
        )
        if needs_ams or needs_cleveref or needs_algorithm or needs_algpseudocode:
            if needs_ams and "cleveref" in preamble:
                preamble = preamble.replace(
                    r"\usepackage{cleveref}",
                    r"\usepackage{amsmath,amssymb}" + "\n" + r"\usepackage{cleveref}",
                    1,
                )
                needs_ams = False
            additions = []
            if needs_ams:
                additions.append(r"\usepackage{amsmath,amssymb}")
            if needs_algorithm:
                additions.append(r"\usepackage{algorithm}")
            if needs_algpseudocode:
                additions.append(r"\usepackage{algpseudocode}")
            if needs_cleveref:
                additions.append(r"\usepackage{cleveref}")
            if additions:
                preamble = preamble.rstrip() + "\n" + "\n".join(additions) + "\n"
            source = preamble + marker + body
        elif "cleveref" in preamble and "amsmath" in preamble:
            clever_idx = preamble.find("cleveref")
            ams_idx = preamble.find("amsmath")
            if clever_idx >= 0 and ams_idx >= 0 and clever_idx < ams_idx:
                preamble = preamble.replace(r"\usepackage{cleveref}", "")
                preamble = preamble.replace(
                    r"\usepackage{amsmath,amssymb}",
                    r"\usepackage{amsmath,amssymb}" + "\n" + r"\usepackage{cleveref}",
                )
                source = preamble + marker + body
    return source + ("\n" if source and not source.endswith("\n") else "")


def _inject_after_first_section(source: str, section_name: str, block: str) -> str:
    if not block.strip():
        return source
    pattern = rf"(\\section\{{{re.escape(section_name)}\}}\s*)"
    match = re.search(pattern, source)
    if not match:
        return source
    insert_at = match.end()
    return source[:insert_at] + "\n" + block.strip() + "\n" + source[insert_at:]


def _ensure_required_concept_figures(source: str, orchestrated: dict) -> str:
    """Inject required post-writing concept figures even when refinement returns a full document."""
    motivation = _concept_figure_blocks(orchestrated, {"fig_motivation_symbolic"})
    overview = _concept_figure_blocks(orchestrated, {"fig_overview_symbolic"})
    if motivation and "fig:fig_motivation_symbolic" not in source:
        source = _inject_after_first_section(source, "Introduction", motivation)
    if overview and "fig:fig_overview_symbolic" not in source:
        source = _inject_after_first_section(source, "Method", overview)
    return source


def pick_main_tex(orchestrated: dict, state: dict, bundle_format: str) -> str:
    """Prefer full refined LaTeX if the model returned a complete ``\\documentclass`` document."""
    full = (orchestrated.get("refinement_full_text") or "").strip()
    force_iclr2026 = bundle_format == "conference"
    if full and re.match(r"^\s*(?:%[^\n]*\n\s*)*\\documentclass(?:\[[^\]]*\])?\{", full):
        tex = normalize_latex_source(full, force_iclr2026=force_iclr2026)
    else:
        tex = normalize_latex_source(assemble_main_tex(state, orchestrated, bundle_format), force_iclr2026=force_iclr2026)
    tex = _ensure_required_concept_figures(tex, orchestrated)
    return normalize_latex_source(tex, force_iclr2026=force_iclr2026)


def _compile_main_pdf(bundle_dir: Path) -> dict:
    main_tex = bundle_dir / "main.tex"
    if not main_tex.exists():
        return {"ok": False, "error": "main.tex missing"}
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    source = main_tex.read_text(encoding="utf-8", errors="replace")
    commands = []
    if latexmk:
        commands.append(("latexmk", [[latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]]))
    if pdflatex:
        # MiKTeX latexmk requires Perl; direct pdflatex is a useful fallback on Windows.
        sequence = [[pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"]]
        if bibtex and "\\bibliography{" in source:
            sequence.extend(
                [
                    [bibtex, "main"],
                    [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                    [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                ]
            )
        commands.append(("pdflatex", sequence))
    if not commands:
        return {"ok": False, "error": "No LaTeX compiler found"}
    attempts = []
    proc = None
    final_log = ""
    for name, sequence in commands:
        step_attempts: list[dict] = []
        ok = True
        for cmd in sequence:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(bundle_dir),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=180,
                )
            except Exception as exc:
                step_attempts.append({"cmd": Path(cmd[0]).name, "ok": False, "error": str(exc)})
                ok = False
                break
            final_log = proc.stdout or ""
            step_ok = proc.returncode == 0
            step_attempts.append({"cmd": Path(cmd[0]).name, "ok": step_ok, "returncode": proc.returncode})
            if not step_ok:
                ok = False
                break
        ok = ok and (bundle_dir / "main.pdf").exists()
        attempts.append({"cmd": name, "ok": ok, "steps": step_attempts})
        if ok:
            break
    log_path = bundle_dir / "latex_compile.log"
    log_body = "===== compile steps =====\n" + json.dumps(attempts, indent=2, ensure_ascii=False)
    if final_log:
        log_body += "\n\n===== final pdflatex =====\n" + final_log
    _write(log_path, log_body[-120_000:])
    ok = bool(attempts and attempts[-1].get("ok") and (bundle_dir / "main.pdf").exists())
    return {
        "ok": ok,
        "returncode": proc.returncode if proc else None,
        "log": str(log_path),
        "attempts": attempts,
    }


def _bundle_dir_for_format(root: Path, bundle_format: str) -> Path:
    return root / bundle_format


def _copy_iclr2026_template_files(bundle_dir: Path) -> list[str]:
    copied: list[str] = []
    if not ICLR2026_TEMPLATE_DIR.exists():
        return copied
    for name in ICLR2026_TEMPLATE_FILES:
        src = ICLR2026_TEMPLATE_DIR / name
        if not src.exists():
            continue
        dst = bundle_dir / name
        shutil.copy2(src, dst)
        copied.append(name)
    return copied


INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
CITE_RE = re.compile(r"\\cite[a-zA-Z*]*\{([^}]*)\}")
BIB_ENTRY_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,.*?(?=\n@\w+\s*\{|\Z)", re.DOTALL)


TOPIC_CITATION_TERMS = {
    "reasoning",
    "chain",
    "thought",
    "self-consistency",
    "tree of thoughts",
    "adaptive",
    "test-time",
    "compute",
    "compute allocation",
    "question answering",
    "qa",
    "uncertainty",
    "selective",
    "selective reasoning",
    "classification",
    "abstention",
    "early exit",
    "halting",
    "calibration",
    "confidence",
    "llm",
    "agent",
    "faithful",
    "verification",
    "budget",
    "overthink",
}
OFF_TOPIC_CITATION_TERMS = {
    "copd",
    "computed tomography",
    "memrist",
    "spiking neural",
    "portfolio",
    "finance",
    "stocks",
    "medical imaging",
    "chronic obstructive",
    "convolutional neural networks for chronic",
    "circuits and systems",
    "graph-organized intelligence",
    "pattern recognition : joint",
}


def _write_placeholder_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(7.2, 3.6))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.5,
            0.55,
            "Missing generated figure",
            ha="center",
            va="center",
            fontsize=18,
            weight="bold",
        )
        ax.text(0.5, 0.38, path.name, ha="center", va="center", fontsize=11)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    except Exception:
        # Last-resort 1x1 transparent PNG. For PDF/SVG paths matplotlib should normally work.
        if path.suffix.lower() == ".png":
            path.write_bytes(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
                b"\x00\x00\x00\x0bIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
                b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
            )


def _bib_entries_by_key(bibtex: str) -> dict[str, str]:
    return {match.group(1).strip(): match.group(0).strip() for match in BIB_ENTRY_RE.finditer(bibtex or "")}


def _bib_entry_relevance(entry: str, state: dict) -> float:
    text = " ".join(
        [
            entry or "",
            str(state.get("title") or ""),
            str(state.get("method_name") or ""),
            str(state.get("problem_statement") or ""),
        ]
    ).lower()
    score = 0.0
    for term in TOPIC_CITATION_TERMS:
        if term in text:
            score += 1.0
    for term in OFF_TOPIC_CITATION_TERMS:
        if term in text:
            score -= 5.0
    return score


def _clean_topic_citations(main_tex: str, bibtex: str, state: dict) -> tuple[str, str, list[str]]:
    entries = _bib_entries_by_key(bibtex)
    if not entries:
        return main_tex, bibtex, []
    kept = {key for key, entry in entries.items() if _bib_entry_relevance(entry, state) >= 1.0}
    if not kept:
        return main_tex, bibtex, []
    removed = sorted(set(entries) - kept)
    if not removed:
        return main_tex, bibtex, []

    paragraphs = re.split(r"(\n\s*\n)", main_tex or "")
    cleaned_parts: list[str] = []
    removed_set = set(removed)
    for part in paragraphs:
        if CITE_RE.search(part):
            cited = {
                key.strip()
                for match in CITE_RE.finditer(part)
                for key in match.group(1).split(",")
                if key.strip()
            }
            if cited and cited <= removed_set:
                continue
        cleaned_parts.append(part)
    tex = "".join(cleaned_parts)

    fallback_keys = [key for key in entries if key in kept][:2]

    def _replace_cite(match: re.Match[str]) -> str:
        keys = [key.strip() for key in match.group(1).split(",") if key.strip() in kept]
        if keys:
            return match.group(0).replace(match.group(1), ", ".join(keys))
        if fallback_keys:
            return match.group(0).replace(match.group(1), ", ".join(fallback_keys))
        return ""

    tex = CITE_RE.sub(_replace_cite, tex)
    new_bib = "\n\n".join(entries[key] for key in entries if key in kept)
    return tex, new_bib, removed


def _cited_keys(main_tex: str) -> list[str]:
    keys: list[str] = []
    for match in CITE_RE.finditer(main_tex or ""):
        keys.extend([key.strip() for key in match.group(1).split(",") if key.strip()])
    return sorted(set(keys))


def _page_count_from_log(bundle_dir: Path) -> int | None:
    for log_name in ("main.log", "latex_compile.log"):
        log_path = bundle_dir / log_name
        if not log_path.exists():
            continue
        raw = log_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"Output written on .*?\((\d+)\s+pages?", raw)
        if match:
            return int(match.group(1))
    return None



def _plain_tex_word_count(tex: str) -> int:
    body = (tex or "").split(r"\bibliographystyle")[0].split(r"\bibliography")[0]
    body = re.sub(r"%.*", " ", body)
    body = re.sub(r"\\(?:section|subsection|subsubsection|paragraph)\*?\{([^}]*)\}", r" \1 ", body)
    body = re.sub(r"\\cite[a-zA-Z*]*\{[^}]*\}", " ", body)
    body = re.sub(r"\\[a-zA-Z]+(?:\[[^]]*\])?(?:\{[^}]*\})?", " ", body)
    body = re.sub(r"[{}$&_#^~]", " ", body)
    return len(re.findall(r"\b[A-Za-z][A-Za-z0-9-]*\b", body))


def _extract_abstract(tex: str) -> str:
    match = re.search(r"\\begin\{abstract\}([\s\S]*?)\\end\{abstract\}", tex or "", re.I)
    return match.group(1) if match else ""


def _guide_issue(severity: str, standard: str, issue: str, evidence: str = "", fix: str = "") -> dict[str, str]:
    out = {"severity": severity, "standard": standard, "issue": issue}
    if evidence:
        out["evidence"] = evidence
    if fix:
        out["fix"] = fix
    return out



def _section_body(main_tex: str, section_name: str) -> str:
    pattern = rf"\\section\*?\{{{re.escape(section_name)}\}}"
    match = re.search(pattern, main_tex or "", flags=re.IGNORECASE)
    if not match:
        return ""
    rest = (main_tex or "")[match.end():]
    next_match = re.search(r"\\section\*?\{[^}]+\}", rest)
    return rest[: next_match.start()] if next_match else rest


def _normalise_caption_text(text: str) -> str:
    text = re.sub(r"\\[a-zA-Z]+(?:\[[^]]*\])?(?:\{[^}]*\})?", " ", text or "")
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _figure_captions(main_tex: str) -> list[str]:
    return re.findall(r"\\caption\{([\s\S]*?)\}", main_tex or "")


def _experiment_figure_roles(main_tex: str, includes: list[str]) -> dict[str, list[str]]:
    captions = _figure_captions(main_tex)
    joined = "\n".join(captions)
    roles: dict[str, list[str]] = {
        "main_benchmark": [],
        "ablation": [],
        "cost_latency_frontier": [],
        "subset_or_difficulty": [],
        "calibration_or_uncertainty": [],
    }
    for idx, raw in enumerate(includes):
        stem = Path(raw).stem.lower()
        if stem in {"fig_motivation_symbolic", "fig_overview_symbolic"}:
            continue
        caption = captions[idx] if idx < len(captions) else ""
        text = f"{stem} {_normalise_caption_text(caption)}"
        if any(term in text for term in ("benchmark", "main", "comparison", "score", "accuracy", "acc", "method panel")):
            roles["main_benchmark"].append(raw)
        if any(term in text for term in ("ablation", "component", "remove", "without", "field", "packet field")):
            roles["ablation"].append(raw)
        if any(term in text for term in ("cost", "token", "latency", "budget", "frontier", "pareto", "tradeoff", "trade off")):
            roles["cost_latency_frontier"].append(raw)
        if any(term in text for term in ("subset", "difficulty", "bucket", "disagreement", "margin", "route", "routing", "gate")):
            roles["subset_or_difficulty"].append(raw)
        if any(term in text for term in ("calibration", "uncertainty", "confidence", "entropy", "distortion")):
            roles["calibration_or_uncertainty"].append(raw)
    return roles


def _raw_float_examples(main_tex: str, limit: int = 8) -> list[str]:
    examples: list[str] = []
    for match in re.finditer(r"(?<![A-Za-z0-9_])[-−]?\d+\.\d{5,}(?![A-Za-z0-9_])", main_tex or ""):
        value = match.group(0)
        if value not in examples:
            examples.append(value)
        if len(examples) >= limit:
            break
    return examples

def _manuscript_guideline_audit(*, main_tex: str, sections: list[str], includes: list[str], page_count: int | None, manuscript_state: dict | None, compile_ok: bool) -> dict:
    state = manuscript_state or {}
    packet = state.get("result_packet") if isinstance(state.get("result_packet"), dict) else {}
    summary = packet.get("benchmark_summary") if isinstance(packet.get("benchmark_summary"), dict) else {}
    contract = state.get("publication_evidence_contract") if isinstance(state.get("publication_evidence_contract"), dict) else {}
    paper_intent = state.get("paper_intent") if isinstance(state.get("paper_intent"), dict) else {}
    problem_awareness = state.get("problem_awareness") if isinstance(state.get("problem_awareness"), dict) else {}
    quality_gates = state.get("quality_gates") if isinstance(state.get("quality_gates"), dict) else {}
    required = state.get("required_evidence") if isinstance(state.get("required_evidence"), dict) else {}
    method_name = str(state.get("method_name") or summary.get("candidate_method") or "")
    method_lower = method_name.lower()
    tex_lower = (main_tex or "").lower()
    abstract = _extract_abstract(main_tex)
    abstract_lower = abstract.lower()
    abstract_env_count = len(re.findall(r"\\begin\{abstract\}", main_tex or "", flags=re.IGNORECASE))
    abstract_section_count = len([x for x in sections if str(x).strip().lower() == "abstract"])
    raw_float_values = _raw_float_examples(main_tex)
    captions = _figure_captions(main_tex)
    normalised_captions = [_normalise_caption_text(x) for x in captions if _normalise_caption_text(x)]
    duplicate_captions = sorted({x for x in normalised_captions if normalised_captions.count(x) > 1})
    non_concept_includes = [x for x in includes if Path(x).stem not in {"fig_motivation_symbolic", "fig_overview_symbolic"}]
    duplicate_includes = sorted({Path(x).name for x in includes if includes.count(x) > 1})
    figure_roles = _experiment_figure_roles(main_tex, includes)
    covered_figure_roles = [name for name, rows in figure_roles.items() if rows]
    intro_body = _section_body(main_tex, "Introduction")
    results_body = _section_body(main_tex, "Results")
    discussion_body = _section_body(main_tex, "Discussion")
    issues = []
    word_count = _plain_tex_word_count(main_tex)
    duplicate_sections = sorted({x for x in sections if sections.count(x) > 1})
    expected_sections = ["Introduction", "Related Work", "Method", "Experiments", "Results", "Discussion"]
    missing_sections = [name for name in expected_sections if name not in sections]
    section_order = [x for x in sections if x in expected_sections]
    expected_order = [x for x in expected_sections if x in section_order]
    if not compile_ok:
        issues.append(_guide_issue("high", "Submission/format", "PDF compilation must pass before a bundle can be ready.", fix="Repair LaTeX/PDF compilation."))
    if duplicate_sections:
        issues.append(_guide_issue("high", "Writing structure", "The manuscript contains duplicate top-level sections.", ", ".join(duplicate_sections), "Deduplicate and merge repeated sections."))
    if missing_sections:
        issues.append(_guide_issue("high", "Writing structure", "The manuscript is missing required conference-paper sections.", ", ".join(missing_sections), "Add missing sections with evidence-grounded content."))
    if abstract_env_count + abstract_section_count != 1:
        issues.append(_guide_issue("high", "Abstract/title cleanup", "The manuscript must contain exactly one abstract and must not repeat an Abstract section.", f"abstract_env_count={abstract_env_count} abstract_section_count={abstract_section_count}", "Keep one abstract environment after maketitle and remove duplicate generated abstracts."))
    if section_order != expected_order:
        issues.append(_guide_issue("medium", "Problem-motivation-method-result spine", "Section order does not follow the required paper narrative.", " -> ".join(sections[:10]), "Use Introduction -> Related Work -> Method -> Experiments -> Results -> Discussion/Limitations."))
    if raw_float_values:
        issues.append(_guide_issue("high", "Table standard / numeric formatting", "Raw unrounded floating-point values remain in the manuscript.", ", ".join(raw_float_values), "Round metrics and costs for paper tables, typically 3 decimals for rates/accuracy and 1--2 decimals for token or latency costs."))
    if re.search(r"training[- ]free", main_tex or "", flags=re.IGNORECASE):
        issues.append(_guide_issue("high", "Writing guide / banned wording", "The manuscript uses the banned phrase 'training-free'.", "training-free", "Describe the actual scope as inference-time evaluation or no model-weight updates only when necessary."))
    if re.search(r"\\paragraph\{(?:Question|Motivation|Answer|Result)\.\}", main_tex or "") or re.search(r"(?m)^\s*(?:Question|Motivation|Answer|Result)\.", main_tex or ""):
        issues.append(_guide_issue("high", "Introduction style", "The manuscript contains forced Question/Motivation/Answer/Result mini-headings.", "Question/Motivation/Answer/Result", "Integrate the problem, motivation, method answer, and result as normal Introduction prose."))
    if re.search(r"\\item\s+\\(?:textbf|textit)\{[^}]{1,48}\}", main_tex or ""):
        issues.append(_guide_issue("medium", "Contribution standard", "Contribution bullets use small bold/italic labels instead of direct contribution statements.", fix="Write plain bullets beginning with We identify/formulate/propose/evaluate, without mini labels."))
    if re.search(r"\b(?:can|could|does|do|should|is|are)\b[^.?]{10,180}\?", (abstract + "\n" + intro_body[:1800]), flags=re.IGNORECASE):
        issues.append(_guide_issue("medium", "Introduction style", "The abstract or introduction frames the contribution as a rhetorical question.", fix="State the research problem and contribution directly rather than asking the reader a question."))
    if results_body and _plain_tex_word_count(results_body) < 120:
        issues.append(_guide_issue("high", "Results section", "Results section is too thin to support a full-paper claim.", f"results_words={_plain_tex_word_count(results_body)}", "Discuss the main table, strongest baseline comparison, statistical uncertainty, cost/latency, and failure cases."))
    if discussion_body and _plain_tex_word_count(discussion_body) < 180:
        issues.append(_guide_issue("medium", "Discussion section", "Discussion section is too short for a complete manuscript.", f"discussion_words={_plain_tex_word_count(discussion_body)}", "Add evidence-bounded interpretation, limitations, deployment scope, and threats to validity."))
    if "certified" in tex_lower and not re.search(r"\b(theorem|proof|formal guarantee|violation rate|coverage|calibration error|distortion bound)\b", tex_lower):
        issues.append(_guide_issue("medium", "Claim calibration", "The title/method uses 'Certified' without enough formal or empirical certification evidence.", "Certified", "Rename or weaken the claim unless the paper proves or measures what is certified."))
    if duplicate_includes:
        issues.append(_guide_issue("high", "Figure/experiment presentation", "The same figure asset is included more than once.", ", ".join(duplicate_includes), "Remove duplicate figure blocks or replace them with distinct evidence."))
    if duplicate_captions:
        issues.append(_guide_issue("high", "Figure/experiment presentation", "Figure captions are duplicated or near-identical.", duplicate_captions[0][:180], "Rewrite captions and replace repeated plots with distinct analyses."))
    if word_count < 4500:
        issues.append(_guide_issue("high", "篇幅 / full-paper length", "The main manuscript is too short for the binding full-paper writing guide.", f"word_count={word_count}", "Expand method, experiment setup, baseline analysis, ablations, limitations, and related work."))
    if page_count is not None and page_count < 8:
        issues.append(_guide_issue("medium", "篇幅 / ICLR-style length", "Compiled paper is short relative to full conference papers.", f"page_count={page_count}", "Target a complete main paper rather than a thin technical note."))
    required_story = problem_awareness.get("required_story_order") or paper_intent.get("required_story_order") or ["problem", "motivation", "method", "result", "limitations"]
    for key in required_story:
        key_l = str(key).lower()
        if key_l in {"problem", "motivation", "method", "result", "limitations"} and key_l not in tex_lower:
            issues.append(_guide_issue("medium", "Problem-awareness contract", f"Required story component is not explicit in the manuscript: {key}.", fix="Make the problem, motivation, method answer, result, and limitations explicit."))
    if not abstract:
        issues.append(_guide_issue("high", "Abstract standard", "The manuscript is missing an abstract."))
    elif not any(token in abstract_lower for token in ("problem", "challenge", "bottleneck")):
        issues.append(_guide_issue("medium", "Abstract standard", "Abstract does not clearly state the concrete problem."))
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    candidate_name = str(summary.get("candidate_method") or method_name)
    candidate_metric = per_method.get(candidate_name, {}).get("metric_value") if isinstance(per_method.get(candidate_name), dict) else None
    oracle_metric = None
    for name, row in per_method.items():
        if "oracle" in str(name).lower() and isinstance(row, dict):
            oracle_metric = row.get("metric_value")
            break
    if abstract and oracle_metric is not None and candidate_metric is not None:
        try:
            if str(round(float(oracle_metric), 6)) in abstract and str(round(float(candidate_metric), 6)) not in abstract:
                issues.append(_guide_issue("high", "Result-grounding", "Abstract appears to report an oracle/upper-bound metric instead of the candidate method metric.", f"oracle={oracle_metric}, candidate={candidate_metric}", "Rewrite abstract with candidate, strongest baseline, and limitation metrics."))
        except (TypeError, ValueError):
            pass
    if len(includes) < 3:
        issues.append(_guide_issue("high", "Figure/experiment presentation", "The manuscript has too few figures for full-paper evidence presentation.", f"figure_count={len(includes)}", "Include at least main benchmark comparison, ablation, and cost/quality or subset-analysis figures/tables."))
    if len(non_concept_includes) < 3:
        issues.append(_guide_issue("high", "Figure/experiment presentation", "Concept/overview diagrams cannot substitute for experiment evidence figures.", f"experiment_figure_count={len(non_concept_includes)}", "Add at least three distinct experiment figures beyond motivation/overview diagrams."))
    if non_concept_includes and len(covered_figure_roles) < 3:
        issues.append(_guide_issue("high", "Figure/experiment diversity", "Experiment figures do not cover enough distinct evidence types.", f"covered_roles={covered_figure_roles}", "Use different evidence displays: main benchmark, ablation, cost/latency frontier, subset/difficulty analysis, or calibration/uncertainty."))
    table_count = len(re.findall(r"\\begin\{table\*?\}", main_tex or "")) + len(re.findall(r"\\begin\{tabular", main_tex or ""))
    if table_count < 2:
        issues.append(_guide_issue("high", "Table standard / experiments", "The manuscript lacks enough numeric experiment tables.", f"table_count={table_count}", "Add main results and ablation/cost tables using the table standard."))
    required_ablations = required.get("ablations") or contract.get("required_ablations") or []
    has_ablation_artifact = bool(summary.get("ablation_table") or summary.get("ablation_results") or summary.get("ablations"))
    if required_ablations and (not has_ablation_artifact or "ablation" not in tex_lower):
        issues.append(_guide_issue("high", "Ablation requirement", "Required ablations are not presented in the manuscript.", f"required_ablations={len(required_ablations)} artifact_present={has_ablation_artifact}", "Add an ablation subsection/table grounded in ablation_table.json."))
    required_baselines = contract.get("required_baselines") or []
    if len(per_method) < max(4, len(required_baselines)):
        issues.append(_guide_issue("high", "Benchmark/baseline requirement", "Benchmark comparison does not cover the required baseline set.", f"per_method_count={len(per_method)} required_baselines={len(required_baselines)}", "Run or present all required baselines, or block full-paper claims."))
    if any(marker in method_lower or marker in tex_lower for marker in ("routing", "gate", "packet", "consensus")):
        for artifact_key, prose_key, label in (("latency_tokens_table", "latency", "latency/token-cost analysis"), ("cost_utility_tradeoff_table", "cost", "quality-cost tradeoff"), ("routing_analysis", "route", "routing analysis"), ("difficulty_breakdown_table", "difficulty", "difficulty/subset breakdown")):
            if not summary.get(artifact_key) and prose_key not in tex_lower:
                issues.append(_guide_issue("medium", "Routing/gating evidence", f"Missing {label} required by the writing guide."))
    if "disagreement" not in tex_lower and not summary.get("subset_analysis"):
        issues.append(_guide_issue("medium", "Disagreement subset analysis", "Motivation depends on disagreement/dissent, but the manuscript lacks a disagreement subset analysis."))
    stale_terms = []
    if "diversity-preserving" in tex_lower and "diversity-preserving" not in method_lower:
        stale_terms.append("Diversity-Preserving")
    if re.search(r"\bDPC\b", main_tex or "") and "dpc" not in method_lower:
        stale_terms.append("DPC")
    if stale_terms:
        issues.append(_guide_issue("high", "Terminology consistency", "Manuscript contains stale/cross-paper terminology.", ", ".join(stale_terms), "Remove cross-paper leftovers and regenerate captions/figures for the current method."))
    if packet.get("full_benchmark_completed") is not True and not summary.get("full_benchmark_completed"):
        issues.append(_guide_issue("high", "Evidence gate", "Full benchmark evidence is not complete; manuscript cannot be bundle_ready."))
    if quality_gates.get("requires_full_benchmark_package") and not summary.get("full_benchmark_completed"):
        issues.append(_guide_issue("high", "Evidence gate", "Quality gate requires full benchmark package, but summary is not marked complete."))
    decision = "bundle_ready"
    if any(x.get("severity") == "high" for x in issues):
        decision = "manuscript_blocked"
    elif any(x.get("severity") == "medium" for x in issues):
        decision = "needs_revision"
    return {"schema_version": "deepgraph_writing_guideline_audit_v1", "status": "pass" if not issues else "fail", "decision": decision, "word_count": word_count, "page_count": page_count, "sections": sections, "duplicate_sections": duplicate_sections, "figure_count": len(includes), "table_count": table_count, "standard_sources": ["agents/paperorchestra/writing_standard.py", "agents/paperorchestra/table_standard.py", "agents/paperorchestra/figure_standard.py", "docs/top_venue_manuscript_chain.md", "paper_intent.json/problem_awareness/publication_evidence_contract"], "issues": issues, "next_actions": [x.get("fix") or x.get("issue") for x in issues[:12]]}

def _paper_quality_report(
    *,
    bundle_dir: Path,
    main_tex: str,
    bibtex: str,
    figure_assets: list[dict],
    placeholder_figures: list[str],
    compile_result: dict,
    removed_cite_keys: list[str],
    template_files: list[str] | None = None,
    manuscript_state: dict | None = None,
) -> dict:
    entries = _bib_entries_by_key(bibtex)
    cited = _cited_keys(main_tex)
    includes = [raw.strip() for raw in INCLUDEGRAPHICS_RE.findall(main_tex or "") if raw.strip()]
    missing_figures: list[str] = []
    vector_figures: list[str] = []
    for raw in includes:
        path = bundle_dir / raw
        if not path.exists() and not path.suffix:
            path = path.with_suffix(".png")
        if not path.exists():
            missing_figures.append(raw)
        elif path.suffix.lower() == ".pdf":
            vector_figures.append(raw)
    off_topic_bib = [
        key
        for key, entry in entries.items()
        if any(term in entry.lower() for term in OFF_TOPIC_CITATION_TERMS)
    ]
    internal_audit_terms = [
        "available log",
        "supplied artifact",
        "provided material",
        "experiment artifacts",
        "faithful report of the recorded evidence",
    ]
    internal_audit_hits = [
        term for term in internal_audit_terms if term in (main_tex or "").lower()
    ]
    sections = re.findall(r"\\section\*?\{([^}]+)\}", main_tex or "")
    subsection_count = len(re.findall(r"\\subsection\*?\{", main_tex or ""))
    page_count = _page_count_from_log(bundle_dir)
    issues: list[dict[str, str]] = []
    if not compile_result.get("ok"):
        issues.append({"severity": "high", "issue": "PDF compile did not pass."})
    if missing_figures or placeholder_figures:
        issues.append({"severity": "high", "issue": "Referenced figures are missing or placeholder-rendered."})
    if set(cited) - set(entries):
        issues.append({"severity": "high", "issue": "The manuscript has citations that are absent from references.bib."})
    if off_topic_bib:
        issues.append({"severity": "medium", "issue": "Off-topic bibliography entries remain after cleanup."})
    if len(cited) < 10:
        issues.append({"severity": "medium", "issue": "Citation density is below a conference-paper target."})
    if len(includes) < 1:
        issues.append({"severity": "medium", "issue": "The paper has no native experiment figure."})
    if "iclr2026_conference" not in main_tex:
        issues.append({"severity": "high", "issue": "Conference bundle is not using the ICLR 2026 template."})
    if internal_audit_hits:
        issues.append({"severity": "medium", "issue": "Internal-audit wording remains in the main body."})
    if page_count is not None and page_count < 8:
        issues.append({"severity": "low", "issue": "The compiled paper is short relative to full conference papers."})
    scientific_review = _scientific_review_gate(main_tex, manuscript_state or {})
    for issue in scientific_review.get("issues") or []:
        if isinstance(issue, dict):
            enriched = dict(issue)
            enriched.setdefault("standard", "Scientific review gate")
            if enriched not in issues:
                issues.append(enriched)
    reference_corpus_audit = audit_against_reference_corpus(
        main_tex=main_tex,
        page_count=page_count,
        figure_count=len(includes),
        bibliography_entry_count=len(entries),
        corpus_dir=REFERENCE_PDF_CORPUS_DIR,
    )
    for issue in reference_corpus_audit.get("issues") or []:
        if isinstance(issue, dict) and issue not in issues:
            issues.append(issue)
    writing_guideline_audit = _manuscript_guideline_audit(
        main_tex=main_tex,
        sections=sections,
        includes=includes,
        page_count=page_count,
        manuscript_state=manuscript_state or {},
        compile_ok=bool(compile_result.get("ok")),
    )
    for issue in writing_guideline_audit.get("issues") or []:
        if isinstance(issue, dict) and issue not in issues:
            issues.append(issue)

    plain_reviewer = review_manuscript_plain(
        bundle_dir=bundle_dir,
        main_tex=main_tex,
        quality_context={
            "compile_ok": bool(compile_result.get("ok")),
            "page_count": page_count,
            "section_count": len(sections),
            "figure_reference_count": len(includes),
            "citation_count": len(cited),
            "scientific_review_gate": scientific_review,
            "writing_guideline_audit": writing_guideline_audit,
        },
    )
    for issue in plain_reviewer.get("issues") or []:
        if isinstance(issue, dict):
            merged = {
                "severity": issue.get("severity") or "medium",
                "issue": issue.get("issue") or "Plain final reviewer raised a manuscript-quality concern.",
            }
            if issue.get("area"):
                merged["standard"] = "Plain final reviewer / " + str(issue.get("area"))
            else:
                merged["standard"] = "Plain final reviewer"
            if issue.get("evidence"):
                merged["evidence"] = issue.get("evidence")
            if issue.get("fix"):
                merged["fix"] = issue.get("fix")
            if merged not in issues:
                issues.append(merged)
    if plain_reviewer.get("can_deliver") is False:
        blocker = {
            "severity": "high",
            "standard": "Plain final reviewer",
            "issue": "Plain final reviewer says the manuscript is not deliverable yet.",
            "evidence": str(plain_reviewer.get("summary") or plain_reviewer.get("recommendation") or ""),
        }
        if blocker not in issues:
            issues.append(blocker)

    return {
        "reference_corpus_dir": str(REFERENCE_PDF_CORPUS_DIR),
        "reference_exemplar": str(REFERENCE_PDF_CORPUS_DIR / "2604.14206.pdf"),
        "reference_corpus_audit": reference_corpus_audit,
        "venue_template": "iclr2026_conference",
        "template_files": template_files or [],
        "compile_ok": bool(compile_result.get("ok")),
        "page_count": page_count,
        "section_count": len(sections),
        "subsection_count": subsection_count,
        "citation_count": len(cited),
        "bibliography_entry_count": len(entries),
        "undefined_citations": sorted(set(cited) - set(entries)),
        "unused_bibliography_entries": sorted(set(entries) - set(cited)),
        "removed_offtopic_cite_keys": removed_cite_keys,
        "remaining_offtopic_bib_keys": off_topic_bib,
        "figure_reference_count": len(includes),
        "vector_figure_count": len(vector_figures),
        "missing_figures": missing_figures,
        "placeholder_figures": placeholder_figures,
        "generated_figure_assets": [
            {
                "figure_id": asset.get("figure_id"),
                "path": asset.get("path"),
                "pdf_path": asset.get("pdf_path"),
                "renderer": asset.get("notes"),
            }
            for asset in figure_assets
            if isinstance(asset, dict)
        ],
        "scientific_review_gate": scientific_review,
        "writing_guideline_audit": writing_guideline_audit,
        "plain_manuscript_reviewer": plain_reviewer,
        "internal_audit_wording_hits": internal_audit_hits,
        "issues": issues,
        "recommendations": [
            "Prefer real benchmark panels, seed/error bars, ablations, and budget-allocation plots over decorative conceptual diagrams.",
            "Keep related work restricted to QA reasoning, adaptive test-time compute, selective prediction, and uncertainty-based routing.",
            "Move missing implementation details into a concise reproducibility/scope subsection rather than repeating them throughout the paper.",
        ],
    }


def _scientific_review_gate(main_tex: str, state: dict) -> dict:
    """Deterministic venue-aware scientific-risk audit.

    This is deliberately harsher than formatting/sanity checks: a manuscript can
    compile and still be a weak scientific submission.
    """
    packet = state.get("result_packet") if isinstance(state.get("result_packet"), dict) else {}
    summary = packet.get("benchmark_summary") if isinstance(packet.get("benchmark_summary"), dict) else {}
    if not summary and isinstance(state.get("benchmark_summary"), dict):
        summary = state["benchmark_summary"]
    datasets = summary.get("datasets") if isinstance(summary.get("datasets"), list) else []
    total_examples = 0
    for row in datasets:
        if not isinstance(row, dict):
            continue
        for key in ("num_test", "num_materialized_examples", "count", "n"):
            try:
                value = int(row.get(key) or 0)
            except (TypeError, ValueError):
                value = 0
            if value:
                total_examples += value
                break
    seed_results = summary.get("seed_results") if isinstance(summary.get("seed_results"), list) else []
    try:
        num_seeds = int(summary.get("num_seeds") or len(seed_results) or 0)
    except (TypeError, ValueError):
        num_seeds = 0
    def _as_float(value):
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _metric(row: dict) -> float | None:
        if not isinstance(row, dict):
            return _as_float(row)
        for key in ("metric_value", "accuracy", "exact_match", "em", "score", "utility"):
            parsed = _as_float(row.get(key))
            if parsed is not None:
                return parsed
        return None

    def _tokens(row: dict) -> float | None:
        if not isinstance(row, dict):
            return None
        for key in ("avg_new_tokens", "tokens", "avg_tokens", "token_cost"):
            parsed = _as_float(row.get(key))
            if parsed is not None:
                return parsed
        return None

    p_value = None
    tests = summary.get("bootstrap_ci") if isinstance(summary.get("bootstrap_ci"), dict) else {}
    for source in (packet, tests):
        for key in ("p_value", "paired_permutation_p"):
            try:
                if source.get(key) is not None:
                    p_value = float(source.get(key))
                    break
            except (TypeError, ValueError):
                    pass
        if p_value is not None:
            break
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    method_names = " ".join(per_method.keys()).lower()
    candidate_name = str(summary.get("candidate_method") or packet.get("candidate_method") or state.get("method_name") or "")
    if not candidate_name or candidate_name not in per_method:
        candidate_name = next((name for name in per_method if any(token in name.lower() for token in ("ours", "dpc", "candidate", "proposed"))), "")
    candidate_row = per_method.get(candidate_name) if candidate_name in per_method else {}
    candidate_metric = _metric(candidate_row if isinstance(candidate_row, dict) else {})
    candidate_tokens = _tokens(candidate_row if isinstance(candidate_row, dict) else {})
    deployable: list[tuple[str, float, float | None]] = []
    for name, row in per_method.items():
        lower_name = name.lower()
        if name == candidate_name or "oracle" in lower_name:
            continue
        value = _metric(row if isinstance(row, dict) else {})
        if value is None:
            continue
        deployable.append((name, value, _tokens(row if isinstance(row, dict) else {})))
    strongest_baseline = max(deployable, key=lambda item: item[1], default=("", None, None))
    strongest_name, strongest_metric, strongest_tokens = strongest_baseline
    strongest_gap = None
    strongest_token_delta = None
    if candidate_metric is not None and strongest_metric is not None:
        strongest_gap = round(float(candidate_metric) - float(strongest_metric), 6)
    if candidate_tokens is not None and strongest_tokens is not None:
        strongest_token_delta = round(float(candidate_tokens) - float(strongest_tokens), 6)

    pairwise_tests = summary.get("pairwise_tests") if isinstance(summary.get("pairwise_tests"), dict) else {}
    significance = summary.get("significance") if isinstance(summary.get("significance"), dict) else {}
    pairwise_text = json.dumps({"pairwise_tests": pairwise_tests, "significance": significance, "bootstrap_ci": tests}, ensure_ascii=False).lower()
    strongest_key_terms = [term for term in re.split(r"[^a-z0-9]+", strongest_name.lower()) if term]
    has_strongest_pairwise = bool(
        strongest_name
        and (
            strongest_name.lower() in pairwise_text
            or all(term in pairwise_text for term in strongest_key_terms[:2])
        )
        and any(token in pairwise_text for token in ("p", "p_value", "paired", "permutation", "bootstrap"))
    )
    routing = summary.get("routing_analysis") if isinstance(summary.get("routing_analysis"), dict) else {}

    def _route_rate(row: dict) -> float | None:
        if not isinstance(row, dict):
            return None
        for key in ("route_rate", "routing_rate", "escalation_rate", "routed_fraction", "trigger_rate"):
            parsed = _as_float(row.get(key))
            if parsed is not None:
                return parsed
        return None

    candidate_route_rate = _route_rate(candidate_row if isinstance(candidate_row, dict) else {})
    if candidate_route_rate is None:
        candidate_route_rate = _route_rate(summary)
    if candidate_route_rate is None and isinstance(routing, dict):
        candidate_route_rate = _route_rate(routing.get(candidate_name) if isinstance(routing.get(candidate_name), dict) else routing)

    def _payload_text(*keys: str) -> str:
        payload = {key: summary.get(key) for key in keys if summary.get(key)}
        return json.dumps(payload, ensure_ascii=False).lower() if payload else ""

    disagreement_text = _payload_text("subset_analysis", "disagreement_subset", "margin_bucket_analysis")
    frontier_text = _payload_text("quality_cost_frontier", "frontier_analysis", "pareto_frontier")
    live_text = _payload_text("live_sanity_check", "live_evaluation", "fresh_sampling_check")
    analysis_text = " ".join(
        text
        for text in (
            json.dumps(routing, ensure_ascii=False).lower() if routing else "",
            disagreement_text,
            frontier_text,
            live_text,
        )
        if text
    )
    has_disagreement_subset = bool(disagreement_text) and any(token in disagreement_text for token in ("disagreement", "margin", "stable", "severe", "bucket"))
    has_frontier = bool(frontier_text) and any(token in frontier_text for token in ("frontier", "pareto", "quality-cost", "quality_cost"))
    has_live_sanity = bool(live_text) and any(token in live_text for token in ("live", "api", "fresh", "held-out", "sampling"))
    evidence_tier = str(packet.get("evidence_tier") or summary.get("evidence_tier") or "").lower()
    controlled_only = any(token in evidence_tier + " " + analysis_text + " " + (main_tex or "").lower() for token in ("materialized", "controlled", "offline trace"))
    required_baselines = {
        "self_consistency": ("self", "consistency"),
        "confidence_weighted": ("confidence",),
        "best_of_n_or_verifier": ("best", "verifier"),
        "debate_or_vote": ("debate", "vote"),
        "adaptive_compute": ("adaptive", "early", "routing"),
    }
    missing_baselines = [
        label
        for label, terms in required_baselines.items()
        if not any(term in method_names for term in terms)
    ]
    unresolved = bool(re.search(r"\?\?\??|Figure\s*~?\\ref\{[^}]*\}\?\?|Table\s*~?\\ref\{[^}]*\}\?\?", main_tex or ""))
    heuristic_terms = ("bonus", "threshold", "hand-crafted", "heuristic")
    looks_heuristic = any(term in (main_tex or "").lower() for term in heuristic_terms)
    issues: list[dict[str, str]] = []
    if total_examples and total_examples < 100:
        issues.append({"severity": "high", "issue": f"Scientific evidence is too small for a top-tier claim: only {total_examples} evaluation examples."})
    elif total_examples and total_examples < 1000:
        issues.append({"severity": "medium", "issue": f"Evaluation scale is thin for a top-tier empirical paper: only {total_examples} examples."})
    if p_value is not None and p_value >= 0.05:
        issues.append({"severity": "high", "issue": f"Core empirical result is not statistically significant at 0.05: p={p_value:.4g}."})
    if num_seeds and num_seeds < 5:
        issues.append({"severity": "medium", "issue": f"Seed coverage is thin for a top-tier empirical paper: {num_seeds} seed(s)."})
    if len(missing_baselines) >= 2:
        issues.append({"severity": "high", "issue": "Baseline coverage is weak for a selector/routing paper: missing " + ", ".join(missing_baselines[:5]) + "."})
    if strongest_name and strongest_gap is not None and strongest_gap <= 0:
        issues.append({"severity": "high", "issue": f"Candidate does not beat the strongest deployable baseline {strongest_name}: metric gap {strongest_gap:+.4f}."})
    if strongest_name and "confidence" in strongest_name.lower() and not has_strongest_pairwise:
        issues.append({"severity": "high", "issue": f"Strongest practical baseline is {strongest_name}; missing pairwise significance/trade-off test against it."})
    elif strongest_name and strongest_gap is not None and strongest_gap < 0.02 and not has_strongest_pairwise:
        issues.append({"severity": "medium", "issue": f"Gain over strongest practical baseline {strongest_name} is small ({strongest_gap:+.4f}) and lacks pairwise uncertainty."})
    if candidate_route_rate is not None and candidate_route_rate < 0.02 and any(token in (candidate_name + " " + main_tex).lower() for token in ("route", "routing", "gate", "packet", "residual", "selector")):
        issues.append({"severity": "high", "issue": f"Candidate route/gate trigger rate is nearly zero ({candidate_route_rate:.4f}), so the mechanism appears almost inactive."})
    if strongest_name and strongest_token_delta is not None and strongest_token_delta > 0 and not has_frontier:
        issues.append({"severity": "medium", "issue": f"Candidate spends {strongest_token_delta:.2f} more tokens than strongest practical baseline {strongest_name}; missing quality-cost frontier evidence."})
    if any(token in (main_tex or "").lower() for token in ("dissent", "disagreement", "consensus")) and not has_disagreement_subset:
        issues.append({"severity": "medium", "issue": "Motivation depends on disagreement/dissent, but disagreement-bucket or margin-subset analysis is missing."})
    if controlled_only and not has_live_sanity:
        issues.append({"severity": "medium", "issue": "Evidence is controlled/materialized only; no live-sampling sanity check is present for broad top-venue claims."})
    if looks_heuristic:
        issues.append({"severity": "medium", "issue": "Method appears to be a hand-crafted heuristic; novelty/technical depth should be justified or downgraded."})
    if unresolved:
        issues.append({"severity": "high", "issue": "Unresolved citation or cross-reference markers remain in the manuscript."})
    high_count = sum(1 for row in issues if row.get("severity") == "high")
    medium_count = sum(1 for row in issues if row.get("severity") == "medium")
    if high_count:
        score = 3 if high_count >= 2 else 4
    elif medium_count >= 3:
        score = 5
    elif medium_count:
        score = 6
    else:
        score = 7
    target_assessments = {
        "iclr_main": {
            "verdict": "reject" if high_count or controlled_only or not has_live_sanity else ("borderline" if medium_count else "weak_accept"),
            "reason": "Needs live/generalization evidence and strong-baseline analysis for a main-conference claim.",
        },
        "acl_emnlp_main": {
            "verdict": "reject" if high_count or controlled_only else ("borderline" if medium_count else "weak_accept"),
            "reason": "NLP main-track claims need live or clearly external benchmark evidence, not only controlled traces.",
        },
        "workshop": {
            "verdict": "promising_with_revisions" if total_examples >= 100 and num_seeds >= 5 and p_value is not None and p_value < 0.05 else "borderline",
            "reason": "Controlled selector evidence can support a workshop paper if scope is downgraded and missing analyses are addressed.",
        },
        "small_conference": {
            "verdict": "promising_with_revisions" if total_examples >= 100 and num_seeds >= 5 else "borderline",
            "reason": "May be viable as a controlled materialized-trace selector study with honest claims.",
        },
        "technical_report": {
            "verdict": "suitable" if total_examples >= 1 else "incomplete",
            "reason": "Useful as a technical report when evidence scope is explicit.",
        },
    }
    return {
        "schema_version": "scientific_review_gate_v2",
        "venue": "venue-aware",
        "estimated_score": score,
        "recommendation": "reject" if score <= 4 else "borderline" if score <= 6 else "weak_accept",
        "target_assessments": target_assessments,
        "total_examples": total_examples,
        "num_seeds": num_seeds,
        "p_value": p_value,
        "candidate_method": candidate_name,
        "candidate_metric": candidate_metric,
        "candidate_tokens": candidate_tokens,
        "candidate_route_rate": candidate_route_rate,
        "strongest_practical_baseline": {
            "name": strongest_name,
            "metric": strongest_metric,
            "tokens": strongest_tokens,
            "metric_gap": strongest_gap,
            "token_delta": strongest_token_delta,
            "has_pairwise_test": has_strongest_pairwise,
        },
        "evidence_scope": "controlled_materialized" if controlled_only else "live_or_external",
        "missing_analyses": {
            "pairwise_vs_strongest_baseline": bool(strongest_name and not has_strongest_pairwise),
            "disagreement_subset": not has_disagreement_subset,
            "quality_cost_frontier": not has_frontier,
            "live_sanity_check": not has_live_sanity,
        },
        "missing_baselines": missing_baselines,
        "issues": issues,
    }


def _submission_blockers_from_state(state: dict, error: str = "") -> list[str]:
    blockers: list[str] = []
    if error:
        blockers.append(error)
    if not state:
        return blockers
    if not state.get("formal_experiment") or state.get("smoke_test_only"):
        blockers.append("Run is not a formal non-smoke experiment.")
    packet = state.get("result_packet") if isinstance(state.get("result_packet"), dict) else {}
    if not packet:
        blockers.append("ExperimentResultPacket is missing.")
        return _dedupe_strings(blockers)
    if packet.get("blocks_manuscript"):
        blockers.append("Result packet currently blocks manuscript generation.")
    evidence_tier = str(packet.get("evidence_tier") or "").strip().lower()
    if evidence_tier in {"bootstrap_probe", "sanity_real_benchmark"}:
        blockers.append(f"Evidence tier is {evidence_tier}, not a full benchmark tier.")
    benchmark_summary = packet.get("benchmark_summary") if isinstance(packet.get("benchmark_summary"), dict) else {}
    publication_contract = (
        packet.get("publication_evidence_contract")
        if isinstance(packet.get("publication_evidence_contract"), dict)
        else state.get("publication_evidence_contract")
        if isinstance(state.get("publication_evidence_contract"), dict)
        else {}
    )
    quality_gates = packet.get("quality_gates") if isinstance(packet.get("quality_gates"), dict) else {}
    artifact_manifest = (
        packet.get("benchmark_artifact_manifest")
        if isinstance(packet.get("benchmark_artifact_manifest"), dict)
        else {}
    )
    artifact_paths = packet.get("artifact_paths") if isinstance(packet.get("artifact_paths"), dict) else {}
    if artifact_manifest.get("readiness_blockers"):
        blockers.extend(str(x) for x in artifact_manifest.get("readiness_blockers") or [])
    if not (packet.get("full_benchmark_completed") or artifact_manifest.get("full_benchmark_completed")):
        blockers.append("full_benchmark_completed is false.")
    if not (artifact_paths.get("artifact_manifest") or artifact_manifest.get("artifacts") or artifact_manifest.get("path")):
        blockers.append("benchmark_artifact_manifest.json is missing or not linked.")
    per_method = benchmark_summary.get("per_method") if isinstance(benchmark_summary.get("per_method"), dict) else {}
    if len(per_method) < 2:
        blockers.append("Benchmark summary must include at least two methods/baselines.")
    seed_results = benchmark_summary.get("seed_results") if isinstance(benchmark_summary.get("seed_results"), list) else []
    try:
        num_seeds = int(benchmark_summary.get("num_seeds") or len(seed_results) or 0)
    except (TypeError, ValueError):
        num_seeds = 0
    try:
        minimum_seeds = int(
            packet.get("minimum_seeds")
            or publication_contract.get("minimum_seeds")
            or quality_gates.get("minimum_seeds")
            or 3
        )
    except (TypeError, ValueError):
        minimum_seeds = 3
    if num_seeds < minimum_seeds:
        blockers.append(f"Only {num_seeds} seed(s) found; required minimum is {minimum_seeds}.")
    if publication_contract.get("required_ablations") and not (
        benchmark_summary.get("ablations")
        or benchmark_summary.get("ablation_results")
        or benchmark_summary.get("ablation_table")
    ):
        blockers.append("Required ablation table/results are missing.")
    benchmark_policy_criteria = {
        "publication_evidence_contract": publication_contract,
        "quality_gates": quality_gates,
        "metric_name": benchmark_summary.get("primary_metric") or publication_contract.get("primary_metric"),
        "metric_direction": publication_contract.get("metric_direction"),
    }
    blockers.extend(
        f"full benchmark policy: {item}"
        for item in full_benchmark_evidence_blockers(benchmark_summary, benchmark_policy_criteria)
    )
    # Weak statistical evidence should shape the manuscript claim strength, not
    # prevent a complete benchmark-backed technical report from being generated.
    if packet.get("crash_count"):
        blockers.append(f"Experiment loop had {packet.get('crash_count')} crash(es); repair/stability report should be included.")
    return _dedupe_strings(blockers)


def _dedupe_strings(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _ensure_referenced_figures(bundle_dir: Path, main_tex: str) -> list[str]:
    created: list[str] = []
    for raw in INCLUDEGRAPHICS_RE.findall(main_tex or ""):
        rel = raw.strip()
        if not rel:
            continue
        path = (bundle_dir / rel).resolve()
        try:
            if bundle_dir.resolve() not in path.parents and path != bundle_dir.resolve():
                continue
        except OSError:
            continue
        if path.suffix.lower() not in {".png", ".pdf", ".jpg", ".jpeg", ".svg"}:
            path = path.with_suffix(".png")
        if path.exists():
            continue
        _write_placeholder_figure(path)
        if path.exists():
            created.append(str(path.relative_to(bundle_dir.resolve())))
    return created


def _latex_caption_map(main_tex: str) -> dict[str, str]:
    out: dict[str, str] = {}
    figure_re = re.compile(r"\\begin\{figure\*?\}(.+?)\\end\{figure\*?\}", re.DOTALL)
    caption_re = re.compile(r"\\caption\{(.+?)\}", re.DOTALL)
    for match in figure_re.finditer(main_tex or ""):
        block = match.group(1)
        includes = INCLUDEGRAPHICS_RE.findall(block)
        if not includes:
            continue
        cap_match = caption_re.search(block)
        caption = ""
        if cap_match:
            caption = re.sub(r"\s+", " ", cap_match.group(1)).strip()
        for raw in includes:
            stem = Path(raw.strip()).stem
            if stem and caption:
                out[stem] = caption
    return out


def _is_placeholder_like_figure(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        if path.suffix.lower() == ".svg":
            raw = path.read_text(encoding="utf-8", errors="replace")[:2000].lower()
            return "missing generated figure" in raw or "placeholder" in raw
        if path.suffix.lower() == ".png":
            return path.stat().st_size < 20_000
    except OSError:
        return True
    return False


def _placeholder_like_asset_figures(bundle_dir: Path, figure_assets: list[dict]) -> list[str]:
    """Return figure asset names that resolve to placeholder or failed-render files."""
    flagged: list[str] = []
    seen: set[str] = set()
    for asset in figure_assets:
        if not isinstance(asset, dict):
            continue
        figure_id = str(asset.get("figure_id") or "").strip()
        for key in ("path", "svg_path", "pdf_path"):
            raw = str(asset.get(key) or "").strip()
            if not raw:
                continue
            raw_path = Path(raw)
            candidates: list[Path] = []
            if raw_path.is_absolute():
                candidates.append(raw_path)
                candidates.append(bundle_dir / "figures" / raw_path.name)
            else:
                candidates.append(bundle_dir / raw_path)
            for path in candidates:
                try:
                    if not path.exists():
                        continue
                    if _is_placeholder_like_figure(path):
                        label = figure_id or path.name
                        if label not in seen:
                            seen.add(label)
                            flagged.append(label)
                        break
                except OSError:
                    continue
    return flagged


def _dedupe_assets(assets: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        key = str(asset.get("figure_id") or Path(str(asset.get("path") or "")).stem)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(asset)
    return out


def _prefer_vector_figure_references(bundle_dir: Path, main_tex: str) -> str:
    root = bundle_dir.resolve()

    def _replace(match: re.Match[str]) -> str:
        raw = match.group(1).strip()
        if not raw:
            return match.group(0)
        path = (bundle_dir / raw).resolve()
        try:
            if root not in path.parents and path != root:
                return match.group(0)
        except OSError:
            return match.group(0)
        pdf_path = path.with_suffix(".pdf")
        if not pdf_path.exists():
            return match.group(0)
        rel = str(pdf_path.relative_to(root)).replace("\\", "/")
        return match.group(0).replace(match.group(1), rel)

    return INCLUDEGRAPHICS_RE.sub(_replace, main_tex or "")


def _materialize_referenced_figures(
    bundle_dir: Path,
    main_tex: str,
    *,
    state: dict,
    iterations: list[dict],
    baseline: float | None,
    metric_name: str,
) -> list[dict]:
    from agents.paperorchestra.figure_orchestra import infer_figure_spec_from_reference, render_native_figure

    captions = _latex_caption_map(main_tex)
    created: list[dict] = []
    root = bundle_dir.resolve()
    for raw in INCLUDEGRAPHICS_RE.findall(main_tex or ""):
        rel = raw.strip()
        if not rel:
            continue
        path = (bundle_dir / rel).resolve()
        try:
            if root not in path.parents and path != root:
                continue
        except OSError:
            continue
        if path.suffix.lower() not in {".png", ".pdf", ".jpg", ".jpeg", ".svg"}:
            path = path.with_suffix(".png")
        if not _is_placeholder_like_figure(path):
            continue
        caption = captions.get(Path(rel).stem) or captions.get(path.stem) or ""
        spec = infer_figure_spec_from_reference(str(path), caption)
        asset = render_native_figure(
            spec,
            figures_dir=path.parent,
            state=state,
            iterations=iterations,
            baseline=baseline,
            metric_name=metric_name,
            output_name=path.name,
        )
        created.append(asset)
    return created


def _mirror_legacy_paper_current(layout: dict, manuscript_root: Path) -> None:
    """Keep legacy idea_N/paper/current consumers working during layout migration."""
    legacy_current = Path(layout["workspace_root"]) / "paper" / "current"
    try:
        if legacy_current.resolve() == manuscript_root.resolve():
            return
    except OSError:
        pass
    if legacy_current.exists():
        if legacy_current.is_dir():
            shutil.rmtree(legacy_current)
        else:
            legacy_current.unlink()
    legacy_current.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(manuscript_root, legacy_current)


def _write_blocked_current_marker(layout: dict, report: dict) -> None:
    """Make stale/current manuscript directories visibly non-submittable."""
    current_root = Path(layout["paper_current_root"])
    current_root.mkdir(parents=True, exist_ok=True)
    marker = {
        "status": report.get("status") or "manuscript_blocked",
        "run_id": report.get("run_id"),
        "deep_insight_id": report.get("deep_insight_id"),
        "error": report.get("error"),
        "blockers": report.get("blockers") or [],
        "next_actions": report.get("next_actions") or [],
    }
    _write(current_root / "MANUSCRIPT_BLOCKED.json", json.dumps(marker, indent=2, ensure_ascii=False, default=str))
    _write(
        current_root / "DO_NOT_SUBMIT.md",
        "\n".join(
            [
                "# Do Not Submit",
                "",
                "This current manuscript directory is stale, blocked, or requires revision before submission.",
                f"Run: {marker.get('run_id')}",
                f"Error: {marker.get('error') or 'manuscript blocked'}",
                "",
                "Blockers:",
                *[f"- {item}" for item in marker.get("blockers") or []],
            ]
        ),
    )


def generate_bundle_paper_orchestra(
    run_id: int,
    bundle_formats: list[str] | None = None,
) -> dict:
    """PaperOrchestra-based bundle generation with verified citations and figure manifests."""
    db.init_db()
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not run:
        return {"error": f"Run {run_id} not found"}
    insight = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (run["deep_insight_id"],))
    iterations = db.fetchall(
        "SELECT * FROM experiment_iterations WHERE run_id=? ORDER BY iteration_number",
        (run_id,),
    )
    claims = db.fetchall("SELECT * FROM experimental_claims WHERE run_id=?", (run_id,))
    if not insight:
        return {"error": f"Insight for run {run_id} not found"}

    state_contract = None
    completeness_audit: dict = {}
    try:
        state_contract = build_manuscript_input_state(run, insight, iterations, claims)
        state_contract.require_submission_ready()
        completeness_audit = audit_evidence_completeness(state_contract.to_dict())
        if not completeness_audit.get("paper_generation_allowed"):
            state_contract.evidence_manifest = completeness_audit.get("evidence_manifest") or {}
            state_contract.claim_evidence_matrix = completeness_audit.get("claim_evidence_matrix") or []
            state_contract.reviewer_report = completeness_audit.get("reviewer_report") or {}
            state_contract.method_reproducibility_requirements = (
                completeness_audit.get("method_reproducibility_requirements") or {}
            )
            state_contract.missing_evidence_report = completeness_audit.get("missing_evidence_report") or {}
            raise ContractValidationError("Paper completeness gate blocked full-paper generation")
    except ContractValidationError as exc:
        state_for_report = state_contract.to_dict() if state_contract is not None else {}
        if state_for_report and not completeness_audit:
            completeness_audit = audit_evidence_completeness(state_for_report)
        blockers = _dedupe_strings(
            _submission_blockers_from_state(state_for_report, str(exc))
            + [str(x) for x in (completeness_audit.get("blockers") or [])]
        )
        layout = get_idea_workspace(int(run["deep_insight_id"]), insight=insight, create=True, sync_db=True)
        report = {
            "run_id": run_id,
            "deep_insight_id": run["deep_insight_id"],
            "status": "manuscript_blocked",
            "error": str(exc),
            "blockers": blockers,
            "gate": completeness_audit.get("schema_version") or "paper_completeness_gate_v1",
            "next_actions": [
                "complete evidence_manifest.json with dataset, split, model, prompt, decoding, seeds, hardware, latency, token cost, and statistical tests",
                "run the required datasets/baselines/ablations in the publication evidence contract",
                "regenerate the bundle only after the reviewer simulator passes",
            ],
        }
        write_plan_files(
            int(run["deep_insight_id"]),
            run_id=run_id,
            insight=insight,
            files={
                "manuscript_blockers.json": report,
                "missing_evidence_report.json": completeness_audit.get("missing_evidence_report") or report,
                "problem_awareness.json": state_for_report.get("problem_awareness") or {},
                "evidence_manifest.json": completeness_audit.get("evidence_manifest") or {},
                "claim_evidence_matrix.json": completeness_audit.get("claim_evidence_matrix") or [],
                "reviewer_report.json": completeness_audit.get("reviewer_report") or {},
                "method_reproducibility_requirements.json": completeness_audit.get("method_reproducibility_requirements") or {},
            },
            mirror_to_run_spec=False,
        )
        _write_blocked_current_marker(layout, report)
        write_latest_status(
            int(run["deep_insight_id"]),
            {
                "stage": "writing_submission",
                "status": "manuscript_blocked",
                "error": str(exc),
                "submission_blockers": blockers,
                "missing_evidence_report": completeness_audit.get("missing_evidence_report") or {},
                "paper_current_root": str(layout.get("paper_current_root") or ""),
            },
            run_id=run_id,
            insight=insight,
        )
        return {
            "error": str(exc),
            "submission_blockers": blockers,
            "missing_evidence_report": completeness_audit.get("missing_evidence_report") or {},
            "backend": "paper_orchestra",
        }
    state = state_contract.to_dict()
    paper_ids = [str(x) for x in _json_list(insight.get("supporting_papers")) if x]
    literature_block = insight.get("evidence_summary") or insight.get("related_work_positioning") or ""
    layout = get_idea_workspace(int(run["deep_insight_id"]), insight=insight, create=True, sync_db=True)
    manuscript_root = Path(layout["paper_current_root"])
    _ensure_dirs(manuscript_root)
    shared_fig = manuscript_root / "paperorchestra_figures"
    _ensure_dirs(shared_fig)
    write_plan_files(
        int(run["deep_insight_id"]),
        run_id=run_id,
        insight=insight,
        files={
            "manuscript_input_state.json": state,
            "paper_intent.json": state.get("paper_intent") or {},
            "problem_awareness.json": state.get("problem_awareness") or {},
            "publication_evidence_contract.json": state.get("publication_evidence_contract") or {},
            "evidence_manifest.json": state.get("evidence_manifest") or {},
            "claim_evidence_matrix.json": state.get("claim_evidence_matrix") or [],
            "reviewer_report.json": state.get("reviewer_report") or {},
            "method_reproducibility_requirements.json": state.get("method_reproducibility_requirements") or {},
        },
        mirror_to_run_spec=False,
    )
    write_latest_status(
        int(run["deep_insight_id"]),
        {"stage": "writing_submission", "status": "drafting", "paper_root": str(layout["paper_root"])},
        run_id=run_id,
        insight=insight,
    )

    manuscript_run_id: int
    initial_state_json = json.dumps(state, default=str)
    existing = db.fetchone("SELECT * FROM manuscript_runs WHERE experiment_run_id=?", (run_id,))
    if existing:
        manuscript_run_id = existing["id"]
        db.execute(
            """
            UPDATE manuscript_runs
            SET status='drafting', canonical_state=?, workdir=?, updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (initial_state_json, str(manuscript_root), manuscript_run_id),
        )
    else:
        manuscript_run_id = db.insert_returning_id(
            """
            INSERT INTO manuscript_runs (experiment_run_id, deep_insight_id, status, canonical_state, workdir)
            VALUES (?, ?, 'drafting', ?, ?)
            RETURNING id
            """,
            (run_id, run["deep_insight_id"], initial_state_json, str(manuscript_root)),
        )
    db.commit()

    try:
        orchestrated = _run_full_pipeline(
            state,
            literature_block,
            state.get("citation_seed_paper_ids") or paper_ids,
            iterations,
            figures_dir=shared_fig,
            baseline=run.get("baseline_metric_value"),
            metric_name=run.get("baseline_metric_name") or "metric",
        )
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        error = str(exc)
        db.execute(
            """
            UPDATE manuscript_runs
            SET status='failed', updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (manuscript_run_id,),
        )
        db.execute(
            "UPDATE deep_insights SET submission_status='failed', updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (run["deep_insight_id"],),
        )
        db.commit()
        write_latest_status(
            int(run["deep_insight_id"]),
            {
                "stage": "writing_submission",
                "status": "failed",
                "manuscript_run_id": manuscript_run_id,
                "error": error,
                "paper_current_root": str(manuscript_root),
            },
            run_id=run_id,
            insight=insight,
        )
        return {
            "error": error,
            "manuscript_run_id": manuscript_run_id,
            "workdir": str(manuscript_root),
            "backend": "paper_orchestra",
        }
    bibtex = (orchestrated.get("bibtex") or "").strip()
    if not bibtex:
        bibtex, _bk = build_references_bib_from_papers(state.get("citation_seed_paper_ids") or paper_ids)
        orchestrated["bibtex_fallback"] = True

    canonical_state_json = json.dumps({**state, "paper_orchestra": orchestrated}, default=str)
    _write(Path(layout["paper_manifests_root"]) / "canonical_state.json", canonical_state_json)
    write_plan_files(
        int(run["deep_insight_id"]),
        run_id=run_id,
        insight=insight,
        files={"canonical_state.json": json.loads(canonical_state_json)},
        mirror_to_run_spec=False,
    )
    db.execute(
        """
        UPDATE manuscript_runs
        SET status='drafting', canonical_state=?, workdir=?, updated_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (canonical_state_json, str(manuscript_root), manuscript_run_id),
    )
    db.commit()

    bundle_formats = bundle_formats or list(SUBMISSION_BUNDLE_FORMATS)
    bundle_ids: list[int] = []
    db.execute("DELETE FROM manuscript_assets WHERE manuscript_run_id=?", (manuscript_run_id,))
    db.execute("DELETE FROM submission_bundles WHERE manuscript_run_id=?", (manuscript_run_id,))

    preferred_bundle_dir: Path | None = None
    for bundle_format in bundle_formats:
        bundle_dir = paper_bundle_root(int(run["deep_insight_id"]), bundle_format, insight=insight)
        if bundle_dir.exists():
            for child in sorted(bundle_dir.iterdir()):
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        _ensure_dirs(bundle_dir)
        figures_dir = bundle_dir / "figures"
        _ensure_dirs(figures_dir)
        copied_template_files = _copy_iclr2026_template_files(bundle_dir) if bundle_format == "conference" else []
        if shared_fig.exists():
            for p in sorted(shared_fig.glob("*")):
                if p.is_file():
                    shutil.copy2(p, figures_dir / p.name)
        _write(
            figures_dir / "paperorchestra_plotting_meta.json",
            json.dumps(orchestrated.get("plotting") or {}, indent=2, default=str)[:100_000],
        )

        main_tex = pick_main_tex(orchestrated, state, bundle_format)
        bundle_bibtex = bibtex
        main_tex, bundle_bibtex, removed_cite_keys = _clean_topic_citations(main_tex, bundle_bibtex, state)
        orchestrated.setdefault("citation_cleanup", {})[bundle_format] = {
            "removed_offtopic_cite_keys": removed_cite_keys,
            "iclr2026_template_files": copied_template_files,
        }
        _write(bundle_dir / "main.tex", main_tex)
        materialized_assets = _materialize_referenced_figures(
            bundle_dir,
            main_tex,
            state=state,
            iterations=[dict(x) for x in iterations],
            baseline=run.get("baseline_metric_value"),
            metric_name=run.get("baseline_metric_name") or "metric",
        )
        main_tex = _prefer_vector_figure_references(bundle_dir, main_tex)
        _write(bundle_dir / "main.tex", main_tex)
        tex_code_pre_report = repair_latex_bundle(bundle_dir, stage="pre_compile_structure")
        if tex_code_pre_report.get("changed"):
            main_tex = (bundle_dir / "main.tex").read_text(encoding="utf-8", errors="replace")
        latex_sanity_report = latex_sanity_check(main_tex)
        _write(bundle_dir / "main.tex", main_tex)
        _write(
            bundle_dir / "latex_sanity_report.json",
            json.dumps(latex_sanity_report, indent=2, ensure_ascii=False, default=str),
        )
        if not latex_sanity_report.get("ok"):
            db.execute(
                """
                UPDATE manuscript_runs
                SET status='failed', updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (manuscript_run_id,),
            )
            db.commit()
            latex_block_report = {
                "run_id": run_id,
                "deep_insight_id": run["deep_insight_id"],
                "status": "manuscript_blocked",
                "error": "LaTeX sanity checker blocked prompt leakage or placeholder text",
                "blockers": latex_sanity_report.get("blockers") or [],
            }
            _write_blocked_current_marker(layout, latex_block_report)
            write_latest_status(
                int(run["deep_insight_id"]),
                {
                    "stage": "writing_submission",
                    "status": "manuscript_blocked",
                    "error": "LaTeX sanity checker blocked prompt leakage or placeholder text",
                    "latex_sanity_report": latex_sanity_report,
                    "paper_current_root": str(manuscript_root),
                },
                run_id=run_id,
                insight=insight,
            )
            return {
                "error": "LaTeX sanity checker blocked prompt leakage or placeholder text",
                "submission_blockers": latex_sanity_report.get("blockers") or [],
                "workdir": str(bundle_dir),
                "backend": "paper_orchestra",
            }
        figure_assets = _dedupe_assets(_figure_assets(orchestrated) + materialized_assets)
        placeholder_figures = _ensure_referenced_figures(bundle_dir, main_tex)
        placeholder_asset_figures = _placeholder_like_asset_figures(bundle_dir, figure_assets)
        all_placeholder_figures = _dedupe_strings(placeholder_figures + placeholder_asset_figures)
        if all_placeholder_figures:
            placeholder_report = {
                "ok": False,
                "blockers": [
                    "Referenced figures are missing or placeholder-rendered: "
                    + ", ".join(all_placeholder_figures)
                ],
                "placeholder_figures": all_placeholder_figures,
            }
            _write(
                bundle_dir / "latex_sanity_report.json",
                json.dumps(
                    {**latex_sanity_report, "ok": False, "placeholder_report": placeholder_report},
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
            )
            db.execute(
                """
                UPDATE manuscript_runs
                SET status='failed', updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (manuscript_run_id,),
            )
            db.commit()
            figure_block_report = {
                "run_id": run_id,
                "deep_insight_id": run["deep_insight_id"],
                "status": "manuscript_blocked",
                "error": "Figure sanity checker blocked placeholder figures",
                "blockers": placeholder_report["blockers"],
            }
            _write_blocked_current_marker(layout, figure_block_report)
            write_latest_status(
                int(run["deep_insight_id"]),
                {
                    "stage": "writing_submission",
                    "status": "manuscript_blocked",
                    "error": "Figure sanity checker blocked placeholder figures",
                    "placeholder_figures": all_placeholder_figures,
                    "paper_current_root": str(manuscript_root),
                },
                run_id=run_id,
                insight=insight,
            )
            return {
                "error": "Figure sanity checker blocked placeholder figures",
                "submission_blockers": placeholder_report["blockers"],
                "workdir": str(bundle_dir),
                "backend": "paper_orchestra",
            }
        _write(
            figures_dir / "figure_manifest.json",
            json.dumps(
                {
                    "assets": figure_assets,
                    "materialized_references": materialized_assets,
                    "placeholder_figures": all_placeholder_figures,
                },
                indent=2,
                default=str,
            )[:100_000],
        )
        _write(bundle_dir / "references.bib", bundle_bibtex)
        _write(
            bundle_dir / "citation_registry.json",
            json.dumps(orchestrated.get("citation_registry") or [], indent=2, default=str)[:200_000],
        )
        citation_audit = audit_citation_registry(
            orchestrated.get("citation_registry") or [],
            bundle_bibtex,
            main_tex,
            state,
        )
        _write(
            bundle_dir / "citation_audit.json",
            json.dumps(citation_audit, indent=2, ensure_ascii=False, default=str)[:200_000],
        )
        if not citation_audit.get("ok"):
            db.execute(
                """
                UPDATE manuscript_runs
                SET status='failed', updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (manuscript_run_id,),
            )
            db.commit()
            citation_block_report = {
                "run_id": run_id,
                "deep_insight_id": run["deep_insight_id"],
                "status": "manuscript_blocked",
                "error": "Citation verifier blocked unrelated or insufficient direct citations",
                "blockers": citation_audit.get("blockers") or [],
            }
            _write_blocked_current_marker(layout, citation_block_report)
            write_latest_status(
                int(run["deep_insight_id"]),
                {
                    "stage": "writing_submission",
                    "status": "manuscript_blocked",
                    "error": "Citation verifier blocked unrelated or insufficient direct citations",
                    "citation_audit": citation_audit,
                    "paper_current_root": str(manuscript_root),
                },
                run_id=run_id,
                insight=insight,
            )
            return {
                "error": "Citation verifier blocked unrelated or insufficient direct citations",
                "submission_blockers": citation_audit.get("blockers") or [],
                "workdir": str(bundle_dir),
                "backend": "paper_orchestra",
            }
        _write(
            bundle_dir / "claim_citation_map.json",
            json.dumps(orchestrated.get("claim_citation_map") or {}, indent=2, default=str)[:120_000],
        )
        _write(
            bundle_dir / "paper_orchestra_trace.json",
            json.dumps(orchestrated, indent=2, default=str)[:200_000],
        )
        _write(bundle_dir / "paper_intent.json", json.dumps(state.get("paper_intent") or {}, indent=2, default=str))
        _write(bundle_dir / "problem_awareness.json", json.dumps(state.get("problem_awareness") or {}, indent=2, default=str))
        _write(
            bundle_dir / "publication_evidence_contract.json",
            json.dumps(state.get("publication_evidence_contract") or {}, indent=2, default=str)[:100_000],
        )
        _write(
            bundle_dir / "evidence_manifest.json",
            json.dumps(state.get("evidence_manifest") or {}, indent=2, ensure_ascii=False, default=str)[:200_000],
        )
        _write(
            bundle_dir / "claim_evidence_matrix.json",
            json.dumps(state.get("claim_evidence_matrix") or [], indent=2, ensure_ascii=False, default=str)[:120_000],
        )
        _write(
            bundle_dir / "reviewer_report.json",
            json.dumps(state.get("reviewer_report") or {}, indent=2, ensure_ascii=False, default=str)[:120_000],
        )
        _write(
            bundle_dir / "method_reproducibility_requirements.json",
            json.dumps(state.get("method_reproducibility_requirements") or {}, indent=2, ensure_ascii=False, default=str)[:120_000],
        )
        _write(bundle_dir / "highlights.md", "\n".join(f"- {c}" for c in state.get("contributions", [])))
        _write(bundle_dir / "cover_letter.md", f"# Cover letter\n\nPaperOrchestra-style draft for: {state['title']}\n")
        _write(bundle_dir / "keywords.json", json.dumps(state.get("submission_keywords") or [], indent=2))
        _write(
            bundle_dir / "submission_checklist.md",
            "\n".join(
                [
                    "# Submission Checklist",
                    "- [x] Main LaTeX source",
                    "- [x] Figures and manifest",
                    "- [x] Verified references",
                    "- [x] Evidence manifest",
                    "- [x] Claim-evidence matrix",
                    "- [x] Problem-awareness contract",
                    "- [x] ICLR 2026 LaTeX template files",
                    "- [x] Reviewer simulator report",
                    "- [x] LaTeX sanity report",
                    "- [x] Citation verifier report",
                    "- [x] Claim citation map",
                    "- [x] Paper quality report",
                    "- [x] Cover letter",
                    "- [x] Highlights",
                ]
            ),
        )
        compile_result = _compile_main_pdf(bundle_dir)
        if not compile_result.get("ok"):
            tex_code_compile_report = repair_latex_bundle(
                bundle_dir,
                stage="post_compile_error",
                compile_result=compile_result,
            )
            if tex_code_compile_report.get("changed"):
                main_tex = (bundle_dir / "main.tex").read_text(encoding="utf-8", errors="replace")
                latex_sanity_report = latex_sanity_check(main_tex)
                _write(
                    bundle_dir / "latex_sanity_report.json",
                    json.dumps(latex_sanity_report, indent=2, ensure_ascii=False, default=str),
                )
                compile_result = _compile_main_pdf(bundle_dir)
        if not compile_result.get("ok"):
            _write(
                bundle_dir / "pdf_compile_status.json",
                json.dumps(compile_result, indent=2, ensure_ascii=False, default=str),
            )
        elif all_placeholder_figures:
            _write(
                bundle_dir / "pdf_compile_status.json",
                json.dumps(
                    {
                        **compile_result,
                        "placeholder_figures": all_placeholder_figures,
                    },
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
            )
        quality_report = _paper_quality_report(
            bundle_dir=bundle_dir,
            main_tex=main_tex,
            bibtex=bundle_bibtex,
            figure_assets=figure_assets,
            placeholder_figures=all_placeholder_figures,
            compile_result=compile_result,
            removed_cite_keys=removed_cite_keys,
            template_files=copied_template_files,
            manuscript_state=state,
        )
        _write(
            bundle_dir / "paper_quality_report.json",
            json.dumps(quality_report, indent=2, ensure_ascii=False, default=str),
        )
        writing_guideline_audit = quality_report.get("writing_guideline_audit") or {}
        guide_decision = str(writing_guideline_audit.get("decision") or "")
        quality_issues = [issue for issue in (quality_report.get("issues") or []) if isinstance(issue, dict)]
        if guide_decision not in {"manuscript_blocked", "needs_revision"}:
            if any(issue.get("severity") == "high" for issue in quality_issues):
                guide_decision = "manuscript_blocked"
            else:
                gate_medium = False
                for issue in quality_issues:
                    source = str(issue.get("standard") or "")
                    if issue.get("severity") == "medium" and source.startswith(("Scientific review gate", "Plain final reviewer")):
                        gate_medium = True
                        break
                if gate_medium:
                    guide_decision = "needs_revision"
        if guide_decision in {"manuscript_blocked", "needs_revision"}:
            blockers = [
                f"{issue.get('standard') or issue.get('severity')}: {issue.get('issue')}"
                for issue in quality_issues
            ]
            next_actions = [
                issue.get("fix") or issue.get("issue")
                for issue in quality_issues
                if issue.get("fix") or issue.get("issue")
            ][:16]
            block_report = {
                "run_id": run_id,
                "deep_insight_id": run["deep_insight_id"],
                "status": guide_decision,
                "error": "Manuscript quality gate failed",
                "blockers": blockers,
                "next_actions": next_actions or writing_guideline_audit.get("next_actions") or [],
                "paper_current_root": str(manuscript_root),
                "bundle_dir": str(bundle_dir),
                "quality_report": str(bundle_dir / "paper_quality_report.json"),
                "writing_standard_sources": writing_guideline_audit.get("standard_sources") or [],
            }
            _write(
                bundle_dir / "MANUSCRIPT_BLOCKED.json",
                json.dumps(block_report, indent=2, ensure_ascii=False, default=str),
            )
            _write(
                bundle_dir / "DO_NOT_SUBMIT.md",
                "\n".join(
                    [
                        "# Do Not Submit",
                        "",
                        "This manuscript does not satisfy the final quality gates, so it was not marked bundle_ready.",
                        f"Status: {guide_decision}",
                        f"Run: {run_id}",
                        "",
                        "Blockers:",
                        *[f"- {item}" for item in blockers],
                        "",
                        "Next actions:",
                        *[f"- {item}" for item in block_report.get("next_actions") or []],
                    ]
                ),
            )
            for child in sorted(manuscript_root.iterdir()):
                if child == shared_fig:
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            for path in sorted(bundle_dir.rglob("*")):
                if not path.is_file():
                    continue
                target = manuscript_root / path.relative_to(bundle_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            _write_blocked_current_marker(layout, block_report)
            db.execute(
                "UPDATE manuscript_runs SET status=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (guide_decision, manuscript_run_id),
            )
            db.execute(
                "UPDATE deep_insights SET submission_status=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (guide_decision, run["deep_insight_id"]),
            )
            db.execute(
                "UPDATE experiment_runs SET status=?, error_message=? WHERE id=?",
                (guide_decision, "Manuscript quality gate failed", run_id),
            )
            db.commit()
            write_latest_status(
                int(run["deep_insight_id"]),
                {
                    "stage": guide_decision,
                    "status": guide_decision,
                    "manuscript_run_id": manuscript_run_id,
                    "paper_current_root": str(manuscript_root),
                    "bundle_dir": str(bundle_dir),
                    "quality_report": str(bundle_dir / "paper_quality_report.json"),
                    "blockers": blockers[:20],
                },
                run_id=run_id,
            )
            return {
                "error": "Manuscript quality gate failed",
                "status": guide_decision,
                "submission_blockers": blockers,
                "writing_guideline_audit": writing_guideline_audit,
                "manuscript_run_id": manuscript_run_id,
                "workdir": str(manuscript_root),
                "backend": "paper_orchestra",
            }
        manifest = _bundle_manifest(bundle_dir)
        _write(bundle_dir / "artifact_manifest.json", json.dumps(manifest, indent=2))
        if preferred_bundle_dir is None or bundle_format == "conference":
            preferred_bundle_dir = bundle_dir
        bundle_ids.append(_store_assets(manuscript_run_id, bundle_dir, bundle_format))
        log_artifact(str(bundle_dir / "artifact_manifest.json"))

    if preferred_bundle_dir is not None:
        for child in sorted(manuscript_root.iterdir()):
            if child == shared_fig:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        for path in sorted(preferred_bundle_dir.rglob("*")):
            if not path.is_file():
                continue
            target = manuscript_root / path.relative_to(preferred_bundle_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        _mirror_legacy_paper_current(layout, manuscript_root)

    db.execute(
        """
        UPDATE manuscript_runs
        SET status='bundle_ready', updated_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (manuscript_run_id,),
    )
    latest_bundle_id = bundle_ids[-1] if bundle_ids else None
    if latest_bundle_id is not None:
        db.execute(
            "UPDATE experiment_runs SET submission_bundle_id=?, status='bundle_ready' WHERE id=?",
            (latest_bundle_id, run_id),
        )
        db.execute(
            "UPDATE deep_insights SET submission_status='bundle_ready', updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (run["deep_insight_id"],),
        )
    db.commit()
    write_latest_status(
        int(run["deep_insight_id"]),
        {
            "stage": "bundle_ready",
            "status": "bundle_ready",
            "manuscript_run_id": manuscript_run_id,
            "bundle_ids": bundle_ids,
            "paper_current_root": str(manuscript_root),
        },
        run_id=run_id,
        insight=insight,
    )

    if bundle_ids:
        if hasattr(db, "emit_pipeline_event"):
            db.emit_pipeline_event(
                "submission_bundle_ready",
                {
                    "run_id": run_id,
                    "deep_insight_id": run["deep_insight_id"],
                    "manuscript_run_id": manuscript_run_id,
                    "bundle_ids": bundle_ids,
                },
            )
        set_outcome(
            "deep_insights",
            run["deep_insight_id"],
            OUTCOME_BECAME_MANUSCRIPT,
            reason="PaperOrchestra bundle generated",
            triggered_by="pipeline",
        )

    return {
        "manuscript_run_id": manuscript_run_id,
        "bundle_ids": bundle_ids,
        "workdir": str(manuscript_root),
        "backend": "paper_orchestra",
    }
