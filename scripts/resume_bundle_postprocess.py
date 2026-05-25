#!/usr/bin/env python3
"""Resume bundle post-processing (tables, venue expand, page budget) without full PaperOrchestra."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.manuscript_page_budget import apply_exact_page_budget
from agents.manuscript_pipeline import build_manuscript_input_state
from agents.manuscript_publication_tables import attach_benchmark_artifacts_to_state, replace_tables_in_tex
from agents.manuscript_submission_enrichment import (
    apply_venue_gates_with_retry,
    copy_submission_figure_assets_to_bundle,
    enrich_submission_main_tex,
    sanitize_latex_body_unicode,
)
from agents.manuscript_submission_style import (
    apply_submission_style_fixes,
    dedupe_marked_paragraphs,
    strip_trailing_after_end_document,
)
from agents.paper_orchestra_pipeline import (
    _compile_main_pdf,
    _prefer_vector_figure_references,
    _purge_stale_gemini_vector_companions,
    strip_manual_section_numbers,
)
from agents.workspace_layout import get_idea_workspace
from db import database as db


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume conference bundle postprocess")
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--bundle-dir", type=str, default="")
    args = parser.parse_args()

    db.init_db()
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (args.run_id,))
    if not run:
        print(f"Run {args.run_id} not found", file=sys.stderr)
        return 1
    insight = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (run["deep_insight_id"],))
    layout = get_idea_workspace(int(run["deep_insight_id"]), insight=insight, create=True)
    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else Path(layout["paper_bundles_root"]) / "conference"
    main_tex_path = bundle_dir / "main.tex"
    if not main_tex_path.is_file():
        print(f"Missing {main_tex_path}", file=sys.stderr)
        return 1

    trace_path = Path(layout["paper_current_root"]) / "paper_orchestra_trace.json"
    orchestrated = {}
    if trace_path.is_file():
        orchestrated = json.loads(trace_path.read_text(encoding="utf-8"))

    iterations = db.fetchall(
        "SELECT * FROM experiment_iterations WHERE run_id=? ORDER BY iteration_number",
        (args.run_id,),
    )
    claims = db.fetchall("SELECT * FROM experimental_claims WHERE run_id=?", (args.run_id,))
    state = build_manuscript_input_state(run, insight, iterations, claims).to_dict()
    state = attach_benchmark_artifacts_to_state(state, run_id=int(args.run_id))
    template_id = "iclr2026"

    figures_dir = bundle_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    copy_submission_figure_assets_to_bundle(bundle_dir, orchestrated)
    _purge_stale_gemini_vector_companions(figures_dir)

    import os

    os.environ.setdefault("DEEPGRAPH_USE_GEMINI_FIGURES", "0")
    os.environ.setdefault("DEEPGRAPH_STRICT_PAGE_BUDGET", "1")
    os.environ.setdefault("DEEPGRAPH_PAGE_BUDGET_MAX_LLM_ATTEMPTS", "3")

    main_tex = strip_trailing_after_end_document(
        dedupe_marked_paragraphs(strip_manual_section_numbers(main_tex_path.read_text(encoding="utf-8")))
    )
    main_tex, enrich_meta = enrich_submission_main_tex(main_tex, orchestrated, state)
    main_tex, style_meta = apply_submission_style_fixes(
        main_tex, state=state, bundle_dir=bundle_dir, orchestrated=orchestrated
    )
    main_tex, table_meta_pre = replace_tables_in_tex(main_tex, state)
    main_tex = strip_trailing_after_end_document(main_tex)
    main_tex, venue_report = apply_venue_gates_with_retry(
        main_tex, template_id=template_id, state=state, orchestrated=orchestrated
    )
    main_tex = sanitize_latex_body_unicode(main_tex)
    main_tex, page_report = apply_exact_page_budget(
        main_tex, bundle_dir, template_id=template_id, state=state
    )
    main_tex, table_meta_final = replace_tables_in_tex(main_tex, state)
    main_tex = _prefer_vector_figure_references(bundle_dir, main_tex)
    main_tex, style_meta_final = apply_submission_style_fixes(
        main_tex, state=state, bundle_dir=bundle_dir, orchestrated=orchestrated
    )
    main_tex = strip_trailing_after_end_document(main_tex)
    style_meta = {**style_meta, **style_meta_final, "tables_pre": table_meta_pre, "tables_final": table_meta_final}
    main_tex_path.write_text(main_tex, encoding="utf-8")
    compile_result = _compile_main_pdf(bundle_dir)

    (bundle_dir / "venue_section_audit.json").write_text(
        json.dumps(venue_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (bundle_dir / "page_budget_audit.json").write_text(
        json.dumps(page_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    current = Path(layout["paper_current_root"])
    for path in bundle_dir.rglob("*"):
        if path.is_file():
            rel = path.relative_to(bundle_dir)
            dest = current / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    legacy = Path(layout["workspace_root"]) / "paper" / "current"
    if legacy.resolve() != current.resolve():
        if legacy.exists():
            shutil.rmtree(legacy) if legacy.is_dir() else legacy.unlink()
        shutil.copytree(current, legacy)

    print(
        json.dumps(
            {
                "bundle_dir": str(bundle_dir),
                "compile_ok": compile_result.get("ok"),
                "page_budget_pass": page_report.get("pass"),
                "page_count": (page_report.get("final") or page_report.get("best_effort") or {}).get("page_count"),
                "style": style_meta,
                "enrich": enrich_meta,
                "main_body_pages": (page_report.get("final") or page_report.get("best_effort") or {}).get(
                    "main_body_pages"
                )
                or (page_report.get("final") or page_report.get("best_effort") or {}).get("page_count"),
            },
            indent=2,
        )
    )
    if not page_report.get("pass"):
        from agents.manuscript_page_budget import page_budget_blockers

        for msg in page_budget_blockers(page_report, template_id=template_id):
            print(msg, file=sys.stderr)
        return 3
    return 0 if compile_result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
