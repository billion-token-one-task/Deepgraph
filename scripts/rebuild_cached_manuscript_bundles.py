#!/usr/bin/env python3
"""Rebuild conference bundles from cached ``paper_orchestra`` state (no LLM writer)."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DEEPGRAPH_PAGE_BUDGET_MAX_LLM_ATTEMPTS", "0")

from agents.manuscript_page_budget import (
    apply_exact_page_budget,
    audit_page_budget,
    page_budget_blockers,
    tune_mainbody_linespread,
)
from agents.manuscript_pipeline import build_manuscript_input_state
from agents.manuscript_publication_tables import attach_benchmark_artifacts_to_state, replace_tables_in_tex
from agents.manuscript_submission_enrichment import (
    audit_venue_section_lengths,
    copy_submission_figure_assets_to_bundle,
    enrich_submission_main_tex,
    sanitize_main_tex_for_compile,
)
from agents.manuscript_submission_style import apply_submission_style_fixes
from agents.manuscript_compile_repair import repair_compile_loop
from agents.paper_orchestra_pipeline import (
    _bundle_manifest,
    _compile_main_pdf,
    _ensure_dirs,
    _ensure_referenced_figures,
    _materialize_referenced_figures,
    _prefer_vector_figure_references,
    _write,
    _write_blocked_current_marker,
    pick_main_tex,
)
from agents.manuscript_templates import get_adapter
from agents.workspace_layout import get_idea_workspace, paper_bundle_root, write_latest_status
from db import database as db


def normalize_for_iclr_compile(tex: str) -> str:
    """Strip venue-specific preamble debris that breaks ICLR pdflatex."""
    text = tex or ""
    text = re.sub(
        r"\\documentclass(?:\[[^\]]*\])?\{(?:acmart|neurips|icml2024|IEEEtran)\}",
        r"\\documentclass{article}",
        text,
        count=1,
    )
    text = re.sub(r"\\usepackage(?:\[[^\]]*\])?\{(?:acmart|neurips[^}]*|icml2024)\}\s*", "", text)
    text = re.sub(r"\\usepackage(?:\[[^\]]*\])?\{(?:algorithm|algpseudocode|algorithmicx)\}\s*", "", text)
    text = re.sub(r"\\settopmatter\{[^}]*\}\s*", "", text)
    text = re.sub(r"\\begin\{CCSXML\}.*?\\end\{CCSXML\}\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"\\keywords\{[^}]*\}\s*", "", text)
    if "\\newcommand{\\E}" in text and "math_commands.tex" in text:
        text = re.sub(r"\\newcommand\{\\E\}\{[^}]*\}\s*", "", text)
    text = re.sub(r"\\ccsdesc(?:\[[^\]]*\])?\{[^}]*\}\s*", "", text)
    text = re.sub(r"\\citestyle\{[^}]*\}\s*", "", text)
    text = re.sub(r"\\copyrightyear\{[^}]*\}\s*", "", text)
    text = re.sub(r"\\acm(?:Year|DOI|ISBN|Price|SubmissionID|Volume|Number|Article|Conference|Booktitle)\{[^}]*\}\s*", "", text)
    text = re.sub(r"\\DeclareMathOperator\*?\{\\mathbb\{E\}\}\{[^}]*\}", r"\\newcommand{\\E}{\\mathbb{E}}", text)
    text = text.replace("\\hdashline", "\\midrule")
    text = re.sub(r"\\label\{mainbody:end\}\s*(?:\\clearpage\s*)?", "", text)
    if "usepackage{geometry}" not in text and "usepackage[margin" not in text:
        text = re.sub(r"\\geometry\{[^}]*\}\s*", "", text)
    preamble, marker, body = text.partition(r"\begin{document}")
    if marker:
        preamble = re.sub(r"\\label\{mainbody:end\}\s*", "", preamble)
        preamble = re.sub(r"\\clearpage\s*", "", preamble)
        preamble = re.sub(r"\\bibliographystyle\{[^}]+\}\s*", "", preamble)
        preamble = re.sub(r"\\bibliography\{[^}]+\}\s*", "", preamble)
        text = preamble + marker + body
    return text


def rebuild_insight(insight_id: int) -> dict:
    db.init_db()
    mr = db.fetchone("SELECT * FROM manuscript_runs WHERE deep_insight_id=?", (insight_id,))
    if not mr:
        return {"insight_id": insight_id, "error": "no manuscript_run"}
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (mr["experiment_run_id"],))
    insight = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (insight_id,))
    if not run or not insight:
        return {"insight_id": insight_id, "error": "missing run/insight"}

    layout = get_idea_workspace(insight_id, insight=insight, create=True, sync_db=True)
    cs_path = Path(layout["plan_root"]) / "canonical_state.json"
    if not cs_path.is_file():
        cs_path = Path(layout["paper_manifests_root"]) / "canonical_state.json"
    if not cs_path.is_file():
        return {"insight_id": insight_id, "error": "canonical_state.json missing"}

    cached = json.loads(cs_path.read_text(encoding="utf-8"))
    orchestrated = cached.get("paper_orchestra") or {}
    if not orchestrated.get("refined") and not orchestrated.get("refinement_full_text"):
        return {"insight_id": insight_id, "error": "cached paper_orchestra empty"}

    iterations = db.fetchall(
        "SELECT * FROM experiment_iterations WHERE run_id=? ORDER BY iteration_number",
        (run["id"],),
    )
    claims = db.fetchall("SELECT * FROM experimental_claims WHERE run_id=?", (run["id"],))
    state = build_manuscript_input_state(run, insight, iterations, claims).to_dict()
    for key in (
        "paper_orchestra",
        "citation_seed_paper_ids",
        "evidence_manifest",
        "claim_evidence_matrix",
        "reviewer_report",
        "method_reproducibility_requirements",
        "publication_evidence_contract",
        "paper_intent",
        "problem_awareness",
    ):
        if cached.get(key) is not None:
            state[key] = cached[key]

    state = attach_benchmark_artifacts_to_state(state, run_id=int(run["id"]))
    template_id = "iclr2026"
    bundle_dir = paper_bundle_root(insight_id, "conference", insight=insight)
    figures_dir = bundle_dir / "figures"
    _ensure_dirs(figures_dir)
    for stale in ("main.aux", "main.bbl", "main.blg", "main.out", "main.log", "main.fls", "main.fdb_latexmk"):
        (bundle_dir / stale).unlink(missing_ok=True)

    get_adapter(template_id).copy_files(bundle_dir)
    shared_fig = Path(layout["paper_current_root"]) / "paperorchestra_figures"
    if shared_fig.is_dir():
        for p in sorted(shared_fig.glob("*")):
            if p.is_file():
                shutil.copy2(p, figures_dir / p.name)

    main_tex = pick_main_tex(orchestrated, state, "conference", template_id=template_id)
    main_tex = normalize_for_iclr_compile(main_tex)
    from agents.manuscript_templates import get_adapter

    main_tex = get_adapter(template_id).normalize_source(main_tex, submission_mode=True)
    copy_submission_figure_assets_to_bundle(bundle_dir, orchestrated)
    main_tex, _enrich = enrich_submission_main_tex(main_tex, orchestrated, state)
    main_tex, _style = apply_submission_style_fixes(
        main_tex, state=state, bundle_dir=bundle_dir, orchestrated=orchestrated
    )
    main_tex, _tables = replace_tables_in_tex(main_tex, state)
    venue_audit = audit_venue_section_lengths(main_tex, template_id=template_id)
    _write(
        bundle_dir / "venue_section_audit.json",
        json.dumps({"passes": [{"pass": 1, "audit": venue_audit}], "final": venue_audit, "pass": venue_audit.get("pass")}, indent=2, ensure_ascii=False),
    )
    main_tex, _tables2 = replace_tables_in_tex(main_tex, state)
    main_tex, _repair = sanitize_main_tex_for_compile(main_tex)
    _materialize_referenced_figures(
        bundle_dir,
        main_tex,
        state=state,
        iterations=[dict(x) for x in iterations],
        baseline=run.get("baseline_metric_value"),
        metric_name=run.get("baseline_metric_name") or "metric",
    )
    main_tex = _prefer_vector_figure_references(bundle_dir, main_tex)
    main_tex, _repair2 = sanitize_main_tex_for_compile(main_tex)
    main_tex, _tables = replace_tables_in_tex(main_tex, state)
    if r"\end{document}" not in main_tex:
        main_tex = pick_main_tex(orchestrated, state, "conference", template_id=template_id)
        main_tex = get_adapter(template_id).normalize_source(
            normalize_for_iclr_compile(main_tex), submission_mode=True
        )
        main_tex, _ = apply_submission_style_fixes(
            main_tex, state=state, bundle_dir=bundle_dir, orchestrated=orchestrated
        )
    main_tex, _tables3 = replace_tables_in_tex(main_tex, state)
    if r"\end{document}" not in main_tex:
        # Table injection truncated the body; keep the pre-table version.
        main_tex = pick_main_tex(orchestrated, state, "conference", template_id=template_id)
        main_tex = get_adapter(template_id).normalize_source(
            normalize_for_iclr_compile(main_tex), submission_mode=True
        )
    main_tex, page_report = apply_exact_page_budget(
        main_tex, bundle_dir, template_id=template_id, state=state, max_attempts=6
    )
    if not page_report.get("pass"):
        final_probe = page_report.get("final") or page_report.get("best_effort") or {}
        pages = final_probe.get("page_count")
        target = int(page_report.get("target_pages") or 9)
        if pages is not None and int(pages) < target:
            from agents.manuscript_deterministic_fill import apply_deterministic_page_fill

            main_tex, det_report = apply_deterministic_page_fill(
                main_tex, bundle_dir, template_id=template_id, state=state, target_pages=target
            )
            page_report["extra_deterministic_fill"] = det_report
            if det_report.get("pass"):
                page_report["pass"] = True
                page_report["final"] = det_report.get("final") or {}
        if not page_report.get("pass"):
            final_probe = page_report.get("final") or page_report.get("best_effort") or {}
            pages = final_probe.get("page_count")
            if pages is not None and int(pages) == target - 1:
                main_tex, spread_report = tune_mainbody_linespread(
                    main_tex, bundle_dir, template_id=template_id, target_pages=target
                )
                if spread_report.get("pass"):
                    page_report["pass"] = True
                    page_report["final"] = spread_report.get("final") or {}
                    page_report["linespread_tune"] = spread_report
    main_tex, _style2 = apply_submission_style_fixes(
        main_tex, state=state, bundle_dir=bundle_dir, orchestrated=orchestrated
    )
    main_tex, _repair3 = sanitize_main_tex_for_compile(main_tex)
    _write(bundle_dir / "main.tex", main_tex)
    bibtex = (orchestrated.get("bibtex") or "").strip()
    if bibtex:
        _write(bundle_dir / "references.bib", bibtex)
    _ensure_referenced_figures(bundle_dir, main_tex)

    compile_result = repair_compile_loop(bundle_dir, compile_fn=_compile_main_pdf, max_rounds=5)
    if not compile_result.get("ok"):
        compile_result = _compile_main_pdf(bundle_dir)

    final_audit = audit_page_budget(bundle_dir, template_id=template_id)
    page_report["compile_ok"] = bool(compile_result.get("ok"))
    page_report["final"] = final_audit
    page_report["pass"] = bool(final_audit.get("pass") and compile_result.get("ok"))
    _write(
        bundle_dir / "page_budget_audit.json",
        json.dumps(page_report, indent=2, ensure_ascii=False, default=str),
    )
    _write(
        bundle_dir / "artifact_manifest.json",
        json.dumps(_bundle_manifest(bundle_dir), indent=2, ensure_ascii=False, default=str),
    )

    current_root = Path(layout["paper_current_root"])
    current_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "main.tex",
        "main.pdf",
        "references.bib",
        "page_budget_audit.json",
        "artifact_manifest.json",
        "compile_repair_log.json",
        "pdf_compile_status.json",
    ):
        src = bundle_dir / name
        if src.is_file():
            shutil.copy2(src, current_root / name)

    result = {
        "insight_id": insight_id,
        "manuscript_run_id": mr["id"],
        "compile_ok": bool(compile_result.get("ok")),
        "page_budget_pass": page_report["pass"],
        "page_count": final_audit.get("page_count"),
        "target_pages": page_report.get("target_pages"),
    }

    if page_report["pass"]:
        for marker in ("MANUSCRIPT_BLOCKED.json", "DO_NOT_SUBMIT.md"):
            p = current_root / marker
            if p.is_file():
                p.unlink()
        db.execute(
            "UPDATE manuscript_runs SET status='drafting', workdir=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (str(current_root), mr["id"]),
        )
        db.execute(
            "UPDATE deep_insights SET submission_status='drafting', updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (insight_id,),
        )
        existing = db.fetchone(
            "SELECT id FROM submission_bundles WHERE manuscript_run_id=? AND bundle_format=?",
            (mr["id"], "conference"),
        )
        manifest = str(bundle_dir / "artifact_manifest.json")
        if existing:
            db.execute(
                "UPDATE submission_bundles SET status='ready', bundle_path=?, manifest_path=? WHERE id=?",
                (str(bundle_dir), manifest, existing["id"]),
            )
        else:
            db.insert_returning_id(
                "INSERT INTO submission_bundles (manuscript_run_id, bundle_format, status, bundle_path, manifest_path) VALUES (?, 'conference', 'ready', ?, ?) RETURNING id",
                (mr["id"], str(bundle_dir), manifest),
            )
        write_latest_status(
            insight_id,
            {
                "stage": "bundle_ready",
                "status": "bundle_ready",
                "manuscript_run_id": mr["id"],
                "paper_current_root": str(current_root),
            },
            run_id=int(run["id"]),
            insight=insight,
        )
        result["ok"] = True
    else:
        blockers = page_budget_blockers(page_report, template_id=template_id)
        if not compile_result.get("ok"):
            blockers.insert(0, "LaTeX compile failed after cached rebuild")
        block_report = {
            "status": "manuscript_blocked",
            "run_id": run["id"],
            "deep_insight_id": insight_id,
            "error": blockers[0] if blockers else "rebuild incomplete",
            "blockers": blockers,
            "next_actions": [],
        }
        _write_blocked_current_marker(layout, block_report)
        db.execute(
            "UPDATE manuscript_runs SET status='failed', updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (mr["id"],),
        )
        result["ok"] = False
        result["blockers"] = blockers[:3]

    db.commit()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ideas", nargs="*", type=int, default=[21, 42, 44, 57, 72, 87, 98, 104, 148, 149, 158])
    args = parser.parse_args()
    results = [rebuild_insight(i) for i in args.ideas]
    print(json.dumps(results, indent=2, ensure_ascii=False))
    ready = sum(1 for r in results if r.get("ok"))
    print(f"\nSUMMARY: {ready}/{len(results)} ready", file=sys.stderr)
    return 0 if ready == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
