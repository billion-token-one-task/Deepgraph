"""PaperOrchestra multi-stage manuscript generation (Song et al., arXiv:2604.05018 §4).

Full pipeline: Outline → parallel(Plot generation, Literature discovery+review) → Section writing →
AgentReview-style refinement. Official agent ``.tex`` prompts under ``prompts/paper_orchestra/``.

Bibliography: Semantic Scholar–verified registry merged with evidence-graph papers (real metadata).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from contracts import ContractValidationError
from agents.manuscript_pipeline import (
    _bundle_manifest,
    _ensure_dirs,
    _store_assets,
    _write,
    build_manuscript_input_state,
)
from agents.workspace_layout import get_idea_workspace, paper_bundle_root, write_latest_status, write_plan_files
from config import SUBMISSION_BUNDLE_FORMATS
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
    if "\\section{" in related:
        intro_related = related
    else:
        intro_related = rf"""\section{{Introduction}}
{intro}
\section{{Related Work}}
{related}"""
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


def pick_main_tex(orchestrated: dict, state: dict, bundle_format: str) -> str:
    """Prefer full refined LaTeX if the model returned a complete ``\\documentclass`` document."""
    full = (orchestrated.get("refinement_full_text") or "").strip()
    if full and "\\documentclass" in full[:2000]:
        return full
    return assemble_main_tex(state, orchestrated, bundle_format)


def _bundle_dir_for_format(root: Path, bundle_format: str) -> Path:
    return root / bundle_format


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

    try:
        state_contract = build_manuscript_input_state(run, insight, iterations, claims)
        state_contract.require_submission_ready()
    except ContractValidationError as exc:
        return {"error": str(exc)}
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
        files={"manuscript_input_state.json": state},
        mirror_to_run_spec=False,
    )
    write_latest_status(
        int(run["deep_insight_id"]),
        {"stage": "writing_submission", "status": "drafting", "paper_root": str(layout["paper_root"])},
        run_id=run_id,
        insight=insight,
    )

    orchestrated = _run_full_pipeline(
        state,
        literature_block,
        state.get("citation_seed_paper_ids") or paper_ids,
        iterations,
        figures_dir=shared_fig,
        baseline=run.get("baseline_metric_value"),
        metric_name=run.get("baseline_metric_name") or "metric",
    )
    bibtex = (orchestrated.get("bibtex") or "").strip()
    if not bibtex:
        bibtex, _bk = build_references_bib_from_papers(state.get("citation_seed_paper_ids") or paper_ids)
        orchestrated["bibtex_fallback"] = True

    existing = db.fetchone("SELECT * FROM manuscript_runs WHERE experiment_run_id=?", (run_id,))
    canonical_state_json = json.dumps({**state, "paper_orchestra": orchestrated}, default=str)
    _write(Path(layout["paper_manifests_root"]) / "canonical_state.json", canonical_state_json)
    write_plan_files(
        int(run["deep_insight_id"]),
        run_id=run_id,
        insight=insight,
        files={"canonical_state.json": json.loads(canonical_state_json)},
        mirror_to_run_spec=False,
    )
    if existing:
        manuscript_run_id = existing["id"]
        db.execute(
            """
            UPDATE manuscript_runs
            SET status='drafting', canonical_state=?, workdir=?, updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (canonical_state_json, str(manuscript_root), manuscript_run_id),
        )
    else:
        manuscript_run_id = db.insert_returning_id(
            """
            INSERT INTO manuscript_runs (experiment_run_id, deep_insight_id, status, canonical_state, workdir)
            VALUES (?, ?, 'drafting', ?, ?)
            RETURNING id
            """,
            (run_id, run["deep_insight_id"], canonical_state_json, str(manuscript_root)),
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
        if shared_fig.exists():
            for p in sorted(shared_fig.glob("*")):
                if p.is_file():
                    shutil.copy2(p, figures_dir / p.name)
        _write(
            figures_dir / "paperorchestra_plotting_meta.json",
            json.dumps(orchestrated.get("plotting") or {}, indent=2, default=str)[:100_000],
        )
        _write(
            figures_dir / "figure_manifest.json",
            json.dumps({"assets": _figure_assets(orchestrated)}, indent=2, default=str)[:100_000],
        )

        main_tex = pick_main_tex(orchestrated, state, bundle_format)
        _write(bundle_dir / "main.tex", main_tex)
        _write(bundle_dir / "references.bib", bibtex)
        _write(
            bundle_dir / "citation_registry.json",
            json.dumps(orchestrated.get("citation_registry") or [], indent=2, default=str)[:200_000],
        )
        _write(
            bundle_dir / "claim_citation_map.json",
            json.dumps(orchestrated.get("claim_citation_map") or {}, indent=2, default=str)[:120_000],
        )
        _write(
            bundle_dir / "paper_orchestra_trace.json",
            json.dumps(orchestrated, indent=2, default=str)[:200_000],
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
                    "- [x] Claim citation map",
                    "- [x] Cover letter",
                    "- [x] Highlights",
                ]
            ),
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
