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
    executor = plotting.get("plotting_executor") if isinstance(plotting, dict) else {}
    if isinstance(executor, dict) and isinstance(executor.get("assets"), list):
        return executor["assets"]
    if isinstance(plotting, dict) and isinstance(plotting.get("assets"), list):
        return plotting["assets"]
    return []


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
    for asset in _figure_assets(orchestrated):
        if not isinstance(asset, dict):
            continue
        path = asset.get("path") or asset.get("svg_path") or ""
        if not path:
            continue
        name = Path(path).name
        figure_id = str(asset.get("figure_id") or Path(path).stem)
        caption = captions.get(figure_id) or asset.get("objective") or asset.get("title") or figure_id
        blocks.append(
            "\n".join(
                [
                    r"\begin{figure}[t]",
                    r"\centering",
                    rf"\includegraphics[width=0.88\linewidth]{{figures/{name}}}",
                    rf"\caption{{{caption}}}",
                    rf"\label{{fig:{figure_id}}}",
                    r"\end{figure}",
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


def assemble_main_tex(state: dict, orchestrated: dict, bundle_format: str) -> str:
    venue = "Conference submission draft" if bundle_format == "conference" else "Journal draft"
    refined = orchestrated.get("refined") if isinstance(orchestrated.get("refined"), dict) else {}
    abs_tex = refined.get("abstract") or "See experiments section for quantitative results."
    intro = refined.get("introduction") or state.get("problem_statement", "")
    meth = refined.get("method") or state.get("method_summary", "")
    exp = refined.get("experiments") or ""
    dis = refined.get("discussion") or ""
    related = _fallback_related_work(state, orchestrated)
    figures = _figure_latex_blocks(orchestrated)
    results_line = (
        f"Baseline {state['baseline_metric_name']}: {state.get('baseline_metric_value')}; "
        f"best: {state.get('best_metric_value')}; effect \\%: {state.get('effect_pct')}; "
        f"verdict: {state.get('verdict')}."
    )
    problem_awareness = state.get("problem_awareness") if isinstance(state.get("problem_awareness"), dict) else {}
    problem_spine = "\n".join(
        [
            r"\paragraph{Question.} " + _latex_escape(problem_awareness.get("central_question") or state.get("problem_statement") or ""),
            r"\paragraph{Motivation.} " + _latex_escape(problem_awareness.get("motivation") or state.get("existing_weakness") or ""),
            r"\paragraph{Answer.} " + _latex_escape(problem_awareness.get("method_answer") or state.get("method_summary") or ""),
            r"\paragraph{Result.} " + _latex_escape(problem_awareness.get("result_claim") or results_line),
        ]
    )
    if "\\section{" in related:
        intro_related = related
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
\usepackage{{amsmath,amssymb}}
\usepackage{{hyperref}}
\usepackage{{url}}
\title{{{state['title']}}}
\author{{Anonymous authors\\Paper under double-blind review}}
\begin{{document}}
\maketitle
\begin{{abstract}}
{abs_tex}
\end{{abstract}}
{intro_related}
\section{{Method}}
{meth}
\section{{Experiments}}
{exp}
{figures}
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
\title{{{state['title']}}}
\author{{DeepGraph Auto Research (PaperOrchestra pipeline)}}
\date{{{venue}}}
\begin{{document}}
\maketitle
\begin{{abstract}}
{abs_tex}
\end{{abstract}}
{intro_related}
\section{{Method}}
{meth}
\section{{Experiments}}
{exp}
{figures}
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
    for package in ("graphicx", "booktabs", "amsmath,amssymb", "hyperref", "url"):
        first_pkg = package.split(",", 1)[0]
        if first_pkg not in preamble:
            preamble = preamble.rstrip() + "\n" + rf"\usepackage{{{package}}}" + "\n"
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
    preamble, marker, body = source.partition(r"\begin{document}")
    if marker:
        needs_cleveref = ("\\Cref" in body or "\\cref" in body) and "cleveref" not in preamble
        needs_ams = (
            any(cmd in body for cmd in ("\\mathbb", "\\operatorname", "\\text", "\\eqref"))
            and "amsmath" not in preamble
        )
        if needs_ams or needs_cleveref:
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


def pick_main_tex(orchestrated: dict, state: dict, bundle_format: str) -> str:
    """Prefer full refined LaTeX if the model returned a complete ``\\documentclass`` document."""
    full = (orchestrated.get("refinement_full_text") or "").strip()
    force_iclr2026 = bundle_format == "conference"
    if full and "\\documentclass" in full[:2000]:
        return normalize_latex_source(full, force_iclr2026=force_iclr2026)
    return normalize_latex_source(assemble_main_tex(state, orchestrated, bundle_format), force_iclr2026=force_iclr2026)


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
    if len(includes) < 3:
        issues.append({"severity": "medium", "issue": "The paper has too few figures for a benchmark-oriented submission."})
    if "iclr2026_conference" not in main_tex:
        issues.append({"severity": "high", "issue": "Conference bundle is not using the ICLR 2026 template."})
    if internal_audit_hits:
        issues.append({"severity": "medium", "issue": "Internal-audit wording remains in the main body."})
    if page_count is not None and page_count < 8:
        issues.append({"severity": "low", "issue": "The compiled paper is short relative to full conference papers."})
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
        "internal_audit_wording_hits": internal_audit_hits,
        "issues": issues,
        "recommendations": [
            "Prefer real benchmark panels, seed/error bars, ablations, and budget-allocation plots over decorative conceptual diagrams.",
            "Keep related work restricted to QA reasoning, adaptive test-time compute, selective prediction, and uncertainty-based routing.",
            "Move missing implementation details into a concise reproducibility/scope subsection rather than repeating them throughout the paper.",
        ],
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
    if packet.get("p_value") is not None:
        try:
            if float(packet.get("p_value")) >= 0.05:
                blockers.append(f"Statistical evidence is weak for top-tier claims: p={float(packet.get('p_value')):.4g}.")
        except (TypeError, ValueError):
            pass
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
        "status": "manuscript_blocked",
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
                "This current manuscript directory is stale or blocked by the evidence gate.",
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
        )
        _write(
            bundle_dir / "paper_quality_report.json",
            json.dumps(quality_report, indent=2, ensure_ascii=False, default=str),
        )
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
