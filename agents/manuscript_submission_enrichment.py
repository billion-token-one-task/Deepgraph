"""Post-assembly enrichment: figures, main-results table, venue word-count gates."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from agents.llm_client import call_llm
from agents.paper_orchestra_prompts import build_conference_guidelines

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-']+")

# Corpus medians from prompts/venue_styles/_EMPIRICAL_STATS.md (workspace/pdfs n=200)
VENUE_SECTION_MIN_WORDS: dict[str, int] = {
    "abstract": 183,
    "introduction": 659,
    "related work": 540,
    "method": 862,
    "experiments": 494,
    "conclusion": 154,
}


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


_UNICODE_LATEX_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("\u2208", r"$\in$"),
    ("\u2209", r"$\notin$"),
    ("\u2264", r"$\leq$"),
    ("\u2265", r"$\geq$"),
    ("\u2192", r"$\rightarrow$"),
    ("\u2014", "---"),
    ("\u2013", "--"),
    ("\u00b1", r"$\pm$"),
)


def strip_llm_wrapper_markup(text: str) -> str:
    """Remove thinking tags and preamble junk before ``\\documentclass``."""
    raw = (text or "").strip()
    # Drop reasoning/scratchpad blocks the writer model leaks (<think>...</think>,
    # <thinking>, <reasoning>, etc).
    raw = re.sub(
        r"<\s*(think|thinking|reasoning|scratchpad)[^>]*>.*?<\s*/\s*\1\s*>",
        "",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Drop a hanging opening think tag with no close (LLM truncation safety).
    raw = re.sub(
        r"<\s*(?:think|thinking|reasoning|scratchpad)[^>]*>",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    idx = raw.find(r"\documentclass")
    if idx > 0:
        return raw[idx:].strip()
    return raw


def sanitize_latex_body_unicode(main_tex: str) -> str:
    """Replace common Unicode math/punctuation that breaks pdfLaTeX."""
    out = main_tex or ""
    for src, dst in _UNICODE_LATEX_REPLACEMENTS:
        out = out.replace(src, dst)
    return out


_DOCCLASS_RE = re.compile(r"\\documentclass\b")
_BEGIN_DOC_RE = re.compile(r"\\begin\{document\}")
_END_DOC_RE = re.compile(r"\\end\{document\}")
_BEGIN_ABS_RE = re.compile(r"\\begin\{abstract\}\s*")
_END_ABS_RE = re.compile(r"\\end\{abstract\}\s*")
_USEPACK_CLEVEREF_RE = re.compile(r"\\usepackage(?:\[[^\]]*\])?\{cleveref\}\s*\n?")
_USEPACK_DOC_RE = re.compile(r"\\usepackage(?:\[[^\]]*\])?\{[^}]+\}")
_PLACEHOLDER_BRACKET_RE = re.compile(r"\\textbf\{\[[^\]]+\]\}")
_BOLD_XX_PLACEHOLDER_RE = re.compile(r"\\textbf\{XX[\\%]+\}")
_MD_BOLD_RE = re.compile(r"\*\*([^*\n]{1,200})\*\*")


def repair_main_tex_structure(main_tex: str) -> tuple[str, dict[str, Any]]:
    """Repair LLM-leaked LaTeX so the file becomes pdfLaTeX-clean.

    Handles, in order:

    * leading thinking/reasoning blocks and markdown chatter (via ``strip_llm_wrapper_markup``);
    * a second ``\\documentclass``/``\\begin{document}`` block that the writer model emitted
      inside the body (keep outer preamble, splice the inner body);
    * consecutive duplicate ``\\begin{abstract}``/``\\end{abstract}`` pairs;
    * duplicate ``\\usepackage{cleveref}`` declarations (option-clash safe);
    * markdown ``**bold**`` survivors and ``\\textbf{[placeholder]}`` template scars;
    * anything written after the first ``\\end{document}`` (relocated before it).
    """

    notes: dict[str, Any] = {}
    raw = strip_llm_wrapper_markup(main_tex)
    docclass_positions = [m.start() for m in _DOCCLASS_RE.finditer(raw)]
    end_positions = [m.start() for m in _END_DOC_RE.finditer(raw)]
    begin_doc_positions = [m.start() for m in _BEGIN_DOC_RE.finditer(raw)]

    if len(docclass_positions) >= 2 and begin_doc_positions:
        outer_begin = begin_doc_positions[0]
        outer_preamble = raw[: outer_begin]
        second_doc = docclass_positions[1]
        inner_begin_candidates = [p for p in begin_doc_positions if p > second_doc]
        inner_end_candidates = [p for p in end_positions if p > (inner_begin_candidates[0] if inner_begin_candidates else second_doc)]
        if inner_begin_candidates and inner_end_candidates:
            inner_begin = inner_begin_candidates[0]
            inner_end = inner_end_candidates[0]
            inner_body = raw[inner_begin + len(r"\begin{document}") : inner_end]
            raw = outer_preamble + r"\begin{document}" + inner_body + "\n" + r"\end{document}" + "\n"
            notes["spliced_inner_document"] = True

    cleveref_matches = list(_USEPACK_CLEVEREF_RE.finditer(raw))
    if len(cleveref_matches) > 1:
        with_opts = [m for m in cleveref_matches if "[" in m.group(0)]
        kept = with_opts[0] if with_opts else cleveref_matches[0]
        new_raw = []
        last = 0
        for m in cleveref_matches:
            new_raw.append(raw[last : m.start()])
            if m is kept:
                new_raw.append(m.group(0))
            last = m.end()
        new_raw.append(raw[last:])
        raw = "".join(new_raw)
        notes["deduped_cleveref"] = len(cleveref_matches) - 1

    raw = re.sub(r"(\\begin\{abstract\}\s*){2,}", r"\\begin{abstract}\n", raw)
    raw = re.sub(r"(\\end\{abstract\}\s*){2,}", r"\\end{abstract}\n", raw)

    raw = _PLACEHOLDER_BRACKET_RE.sub("", raw)
    raw = _BOLD_XX_PLACEHOLDER_RE.sub("", raw)
    raw = _MD_BOLD_RE.sub(r"\\textbf{\1}", raw)

    cite_alias_count = 0
    for alias in (r"\citeA", r"\citeNP", r"\citeyearNP"):
        before = raw
        raw = raw.replace(alias + "{", r"\cite{")
        if raw != before:
            cite_alias_count += 1
    if cite_alias_count:
        notes["rewrote_citation_aliases"] = cite_alias_count

    end_match = _END_DOC_RE.search(raw)
    if end_match:
        tail = raw[end_match.end() :]
        head = raw[: end_match.start()]
        tail_stripped = tail.strip()
        if tail_stripped:
            label_re = re.compile(r"\\label\{mainbody:end\}")
            label_match = label_re.search(head)
            insertion = "\n\n" + tail_stripped + "\n\n"
            if label_match:
                head = head[: label_match.start()] + insertion + head[label_match.start() :]
            else:
                head = head + insertion
            raw = head + r"\end{document}" + "\n"
            notes["relocated_post_end_document"] = True

    used_packages = {m.group(0).split("{")[-1].rstrip("}") for m in _USEPACK_DOC_RE.finditer(raw)}
    if r"\\cref" in raw or r"\\Cref" in raw or "\\cref{" in raw or "\\Cref{" in raw:
        if "cleveref" not in used_packages:
            preamble_end = raw.find(r"\begin{document}")
            if preamble_end > 0:
                raw = raw[:preamble_end] + r"\usepackage[capitalize]{cleveref}" + "\n" + raw[preamble_end:]
                notes["added_cleveref_for_cref_use"] = True

    notes["sanitized_length"] = len(raw)
    return raw, notes


def sanitize_main_tex_for_compile(main_tex: str) -> tuple[str, dict[str, Any]]:
    """Public entry point: structural repair + unicode replacement."""
    repaired, notes = repair_main_tex_structure(main_tex)
    repaired = sanitize_latex_body_unicode(repaired)
    return repaired, notes


def _latex_escape(text: str) -> str:
    return (
        str(text or "")
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def attach_main_results_table_to_state(state: dict, *, run_id: int | None = None) -> dict:
    """Merge ``main_results_table.json`` into state for ``build_main_results_table_tex``."""
    out = dict(state)
    if out.get("benchmark_summary") or out.get("main_results_table"):
        return out
    candidates: list[Path] = []
    if run_id is not None:
        candidates.append(
            Path(f"/root/deepgraph_ideas/idea_{out.get('deep_insight_id')}/experiments/main/runs/run_{run_id}/results/main_results_table.json")
        )
    workdir = out.get("experiment_workdir") or out.get("workdir")
    if workdir:
        candidates.append(Path(str(workdir)) / "results" / "main_results_table.json")
    for path in candidates:
        if path.is_file():
            try:
                out["main_results_table"] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
            break
    return out


def _table_rows_from_state(state: dict) -> list[tuple[str, float]]:
    from agents.paper_orchestra_pipeline import build_main_results_table_tex

    if build_main_results_table_tex(state):
        summary = state.get("benchmark_summary") if isinstance(state.get("benchmark_summary"), dict) else {}
        if not summary:
            packet = state.get("result_packet") if isinstance(state.get("result_packet"), dict) else {}
            summary = packet.get("benchmark_summary") if isinstance(packet.get("benchmark_summary"), dict) else {}
        per_method = summary.get("per_method") if isinstance(summary, dict) else {}
        metric = str(
            summary.get("metric_name") or summary.get("primary_metric") or state.get("baseline_metric_name") or "primary_score"
        )
        rows: list[tuple[str, float]] = []
        if isinstance(per_method, dict):
            for name, row in sorted(per_method.items(), key=lambda x: str(x[0])):
                val = None
                if isinstance(row, dict):
                    for key in (metric, "metric_value", "primary_score"):
                        if key in row:
                            val = row[key]
                            break
                    if val is None and len(row) == 1:
                        val = next(iter(row.values()))
                else:
                    val = row
                try:
                    rows.append((str(name), float(val)))
                except (TypeError, ValueError):
                    continue
        if rows:
            return rows
    raw = state.get("main_results_table")
    if isinstance(raw, dict) and raw:
        rows = []
        for name, row in sorted(raw.items(), key=lambda x: str(x[0])):
            if isinstance(row, dict):
                for key, val in row.items():
                    try:
                        rows.append((str(name), float(val)))
                        break
                    except (TypeError, ValueError):
                        continue
            else:
                try:
                    rows.append((str(name), float(row)))
                except (TypeError, ValueError):
                    continue
        return rows
    return []


def build_main_results_table_tex_from_state(state: dict) -> str:
    """Build publication-style main results table."""
    from agents.manuscript_publication_tables import build_publication_main_results_table_tex

    return build_publication_main_results_table_tex(state)


def _figure_caption_map(orchestrated: dict) -> dict[str, str]:
    plotting = orchestrated.get("plotting") or {}
    out: dict[str, str] = {}
    for row in plotting.get("figure_captions") or []:
        if isinstance(row, dict) and row.get("figure_id"):
            out[str(row["figure_id"])] = str(row.get("caption") or "")
    return out


def _figure_block(asset: dict, captions: dict[str, str], *, width: str = r"0.88\linewidth") -> str:
    path = asset.get("path") or asset.get("pdf_path") or asset.get("svg_path") or ""
    if not path:
        return ""
    path_obj = Path(str(path))
    figure_id = str(asset.get("figure_id") or path_obj.stem)
    if "gemini" in figure_id.lower() or "gemini" in path_obj.name.lower():
        name = path_obj.with_suffix(".png").name
    else:
        name = path_obj.name
    from agents.manuscript_submission_style import polish_figure_caption

    raw_cap = captions.get(figure_id) or asset.get("objective") or asset.get("title") or figure_id
    caption = _latex_escape(polish_figure_caption(str(raw_cap), asset))[:500]
    return "\n".join(
        [
            r"\begin{figure}[t]",
            r"\centering",
            rf"\includegraphics[width={width}]{{figures/{name}}}",
            rf"\caption{{{caption}}}",
            rf"\label{{fig:{figure_id}}}",
            r"\end{figure}",
        ]
    )


def _inject_after_section_header(tex: str, section_name: str, block: str) -> str:
    if not block or not block.strip():
        return tex
    label_stub = block.split(r"\label{", 1)[-1].split("}", 1)[0] if r"\label{" in block else ""
    if label_stub and f"fig:{label_stub}" in tex:
        return tex
    pat = rf"(\\section\*?\{{{re.escape(section_name)}\}}[^\n]*\n)"
    m = re.search(pat, tex, flags=re.IGNORECASE)
    if not m:
        return tex
    return tex[: m.end()] + block + "\n\n" + tex[m.end() :]


def _inject_before_marker(tex: str, marker_pat: str, block: str) -> str:
    if not block or not block.strip():
        return tex
    m = re.search(marker_pat, tex, flags=re.IGNORECASE)
    if not m:
        return tex + "\n\n" + block
    return tex[: m.start()] + block + "\n\n" + tex[m.start() :]


def copy_submission_figure_assets_to_bundle(bundle_dir: Path, orchestrated: dict) -> list[str]:
    """Copy selected submission figure files into ``bundle_dir/figures``."""
    from agents.paper_orchestra_pipeline import _select_submission_figure_assets

    figures_dir = bundle_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for asset in _select_submission_figure_assets(orchestrated):
        for key in ("path", "pdf_path", "svg_path"):
            src = asset.get(key)
            if not src:
                continue
            path = Path(str(src))
            if not path.is_file():
                continue
            dest = figures_dir / path.name
            if not dest.exists() or dest.stat().st_mtime < path.stat().st_mtime:
                shutil.copy2(path, dest)
            if path.name not in copied:
                copied.append(path.name)
    return copied


def enrich_submission_main_tex(
    main_tex: str,
    orchestrated: dict,
    state: dict,
) -> tuple[str, dict[str, Any]]:
    """Insert submission figures + main-results table into LLM-produced LaTeX."""
    from agents.paper_orchestra_pipeline import (
        _is_main_result_submission_figure,
        _is_motivation_overview_diagram,
        _outline_main_result_figure_ids,
        _select_submission_figure_assets,
    )

    meta: dict[str, Any] = {"injected_figures": [], "injected_table": False}
    captions = _figure_caption_map(orchestrated)
    main_ids = _outline_main_result_figure_ids(orchestrated)
    motivation_block = ""
    overview_block = ""
    main_plot_block = ""
    for asset in _select_submission_figure_assets(orchestrated):
        if _is_motivation_overview_diagram(asset):
            fid = str(asset.get("figure_id") or "").lower()
            block = _figure_block(asset, captions)
            if "motivation" in fid and not motivation_block:
                motivation_block = block
            elif "overview" in fid and not overview_block:
                overview_block = block
            elif not motivation_block:
                motivation_block = block
            elif not overview_block:
                overview_block = block
        elif _is_main_result_submission_figure(asset, main_ids):
            main_plot_block = _figure_block(asset, captions, width=r"\textwidth")

    out = main_tex
    if motivation_block and "fig_motivation" not in out:
        if re.search(r"\\paragraph\{Contributions", out, re.I):
            out = _inject_before_marker(out, r"\\paragraph\{Contributions", motivation_block)
        else:
            out = _inject_after_section_header(out, "Introduction", motivation_block)
        meta["injected_figures"].append("motivation")

    if overview_block and "fig_overview" not in out:
        out = _inject_after_section_header(out, "Method", overview_block)
        meta["injected_figures"].append("overview")

    table_tex = build_main_results_table_tex_from_state(state)
    if table_tex and r"\begin{table}" not in out:
        out = _inject_after_section_header(out, "Experiments", table_tex)
        meta["injected_table"] = True

    if main_plot_block and r"\label{tab:main_results}" in out:
        out = _inject_before_marker(out, r"\\begin\{table\}", main_plot_block)
        meta["injected_figures"].append("main_results_plot")
    elif main_plot_block:
        out = _inject_after_section_header(out, "Experiments", main_plot_block)
        meta["injected_figures"].append("main_results_plot")

    meta["figure_includegraphics_count"] = len(re.findall(r"\\includegraphics", out))
    return out, meta


def _extract_section_texts(tex: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.DOTALL | re.IGNORECASE)
    if m:
        sections["abstract"] = m.group(1)
    for sm in re.finditer(r"\\section\*?\{([^}]+)\}(.*?)(?=\\section\*?\{|$)", tex, flags=re.DOTALL | re.IGNORECASE):
        title = sm.group(1).strip().lower()
        sections[title] = sm.group(2)
    return sections


def audit_venue_section_lengths(
    main_tex: str,
    *,
    template_id: str,
) -> dict[str, Any]:
    sections = _extract_section_texts(main_tex)
    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    section_aliases = {
        "method": ("method", "methods", "approach"),
        "related work": ("related work", "background", "prior work"),
        "experiments": ("experiments", "experimental setup", "results", "evaluation"),
        "conclusion": ("conclusion", "conclusions"),
    }
    for key, minimum in VENUE_SECTION_MIN_WORDS.items():
        body = sections.get(key, "")
        if not body:
            for alias in section_aliases.get(key, ()):
                body = sections.get(alias, "")
                if body:
                    break
        wc = _word_count(body)
        ok = wc >= minimum
        rows.append({"section": key, "words": wc, "minimum": minimum, "ok": ok})
        if not ok:
            failures.append(
                {
                    "section": key,
                    "words": wc,
                    "minimum": minimum,
                    "deficit": minimum - wc,
                }
            )
    has_contrib = bool(re.search(r"\\paragraph\{Contributions", main_tex, re.I)) and bool(
        re.search(r"\\begin\{itemize\}", main_tex, re.I)
    )
    if not has_contrib:
        failures.append({"section": "introduction", "issue": "missing_contributions_itemize"})
    has_main_table = bool(re.search(r"\\label\{tab:main_results\}", main_tex, re.I))
    if not has_main_table:
        failures.append({"section": "experiments", "issue": "missing_main_results_table"})
    has_ablation_table = bool(re.search(r"\\label\{tab:ablations\}", main_tex, re.I))
    if not has_ablation_table:
        failures.append({"section": "experiments", "issue": "missing_ablation_table"})
    has_ablation_subsection = bool(re.search(r"\\subsection\*?\{[^}]*[Aa]blation", main_tex))
    if not has_ablation_subsection:
        failures.append({"section": "experiments", "issue": "missing_ablation_subsection"})
    fig_count = len(re.findall(r"\\includegraphics", main_tex, re.I))
    if fig_count < 2:
        failures.append({"section": "figures", "issue": "too_few_includegraphics", "count": fig_count})
    return {
        "template_id": template_id,
        "sections": rows,
        "failures": failures,
        "pass": not failures,
    }


def expand_submission_tex_for_venue(
    main_tex: str,
    audit: dict[str, Any],
    *,
    template_id: str,
    state: dict,
) -> str:
    """One LLM pass to expand under-length sections while preserving structure."""
    failures = audit.get("failures") or []
    if not failures:
        return main_tex
    guidelines = build_conference_guidelines(template_id)
    targets = [
        f"- {f.get('section')}: need >= {f.get('minimum')} words (currently {f.get('words')})"
        for f in failures
        if f.get("minimum")
    ]
    issues = [f"- {f.get('issue')}" for f in failures if f.get("issue")]
    system = (
        "You are a senior ML conference editor. Expand ONLY the under-length sections in the LaTeX. "
        "Match reference-corpus depth: multi-paragraph Related Work with thematic \\subsections; "
        "Experiments with setup, main results, per-dataset discussion, seed variance, and a dedicated "
        "\\subsection{Ablation Study} with interpretation of tab:ablations. "
        "Keep all \\cite keys, labels, publication tables (tab:main_results, tab:ablations), and figures. "
        "Return the full document starting with \\documentclass."
    )
    user = (
        f"--- conference_guidelines ---\n{guidelines[:12000]}\n"
        f"--- expansion_targets ---\n"
        + "\n".join(targets + issues)
        + "\n--- title ---\n"
        + str(state.get("title") or "")
        + "\n--- paper.tex ---\n"
        + main_tex[:100000]
    )
    expanded, _ = call_llm(system, user, temperature=0.2)
    text = strip_llm_wrapper_markup(expanded or "")
    if "\\documentclass" in text[:3000]:
        return sanitize_latex_body_unicode(text)
    return main_tex


def apply_venue_gates_with_retry(
    main_tex: str,
    *,
    template_id: str,
    state: dict,
    orchestrated: dict,
) -> tuple[str, dict[str, Any]]:
    """Audit venue lengths; up to two expansion passes; re-apply figure/table injection."""
    max_passes = 2
    current = sanitize_latex_body_unicode(main_tex)
    report: dict[str, Any] = {"passes": []}
    for pass_idx in range(max_passes):
        audit = audit_venue_section_lengths(current, template_id=template_id)
        report["passes"].append({"pass": pass_idx + 1, "audit": audit})
        if audit.get("pass"):
            report["initial"] = report["passes"][0]["audit"]
            report["final"] = audit
            report["expanded"] = pass_idx > 0
            report["pass"] = True
            return current, report
        expanded = expand_submission_tex_for_venue(current, audit, template_id=template_id, state=state)
        if expanded == current:
            break
        current = sanitize_latex_body_unicode(expanded)
        current, enrich_meta = enrich_submission_main_tex(current, orchestrated, state)
        report.setdefault("enrich_after_expand", []).append(enrich_meta)
        report["expanded"] = True
    report["initial"] = report["passes"][0]["audit"] if report["passes"] else {}
    report["final"] = audit_venue_section_lengths(current, template_id=template_id)
    report["pass"] = bool(report["final"].get("pass"))
    return current, report
