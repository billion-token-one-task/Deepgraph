#!/usr/bin/env python3
"""Fast idea_2 bundle fix: publication tables + deterministic page fill (no venue LLM)."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DEEPGRAPH_PAGE_BUDGET_MAX_LLM_ATTEMPTS", "0")

from agents.manuscript_page_budget import apply_exact_page_budget, page_budget_blockers
from agents.manuscript_pipeline import build_manuscript_input_state
from agents.manuscript_publication_tables import attach_benchmark_artifacts_to_state, replace_tables_in_tex
from agents.manuscript_submission_style import (
    apply_submission_style_fixes,
    dedupe_marked_paragraphs,
    strip_trailing_after_end_document,
)
from agents.paper_orchestra_pipeline import _compile_main_pdf
from agents.workspace_layout import get_idea_workspace
from db import database as db


def main() -> int:
    db.init_db()
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (2,))
    insight = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (run["deep_insight_id"],))
    layout = get_idea_workspace(int(run["deep_insight_id"]), insight=insight, create=True)
    bundle_dir = Path(layout["paper_bundles_root"]) / "conference"
    main_tex_path = bundle_dir / "main.tex"
    main_tex = main_tex_path.read_text(encoding="utf-8")
    main_tex = strip_trailing_after_end_document(main_tex)
    main_tex = dedupe_marked_paragraphs(main_tex)

    iterations = db.fetchall(
        "SELECT * FROM experiment_iterations WHERE run_id=? ORDER BY iteration_number",
        (2,),
    )
    claims = db.fetchall("SELECT * FROM experimental_claims WHERE run_id=?", (2,))
    state = build_manuscript_input_state(run, insight, iterations, claims).to_dict()
    state = attach_benchmark_artifacts_to_state(state, run_id=2)

    main_tex, _ = apply_submission_style_fixes(
        main_tex, state=state, bundle_dir=bundle_dir, orchestrated={}
    )
    main_tex = strip_trailing_after_end_document(dedupe_marked_paragraphs(main_tex))
    main_tex, page_report = apply_exact_page_budget(
        main_tex, bundle_dir, template_id="iclr2026", state=state, max_attempts=1
    )
    main_tex, table_meta = replace_tables_in_tex(main_tex, state)
    main_tex = strip_trailing_after_end_document(main_tex)
    main_tex_path.write_text(main_tex, encoding="utf-8")
    compile_result = _compile_main_pdf(bundle_dir)
    current = Path(layout["paper_current_root"])
    current.mkdir(parents=True, exist_ok=True)
    (current / "main.tex").write_text(main_tex, encoding="utf-8")
    import shutil

    for name in ("main.pdf", "page_budget_audit.json"):
        src = bundle_dir / name
        if src.is_file():
            shutil.copy2(src, current / name)
    (bundle_dir / "page_budget_audit.json").write_text(
        json.dumps(page_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    out = {
        "table_meta": table_meta,
        "compile_ok": compile_result.get("ok"),
        "page_budget_pass": page_report.get("pass"),
        "page_count": (page_report.get("final") or {}).get("page_count"),
        "has_per_dataset": r"\label{tab:per_dataset}" in main_tex,
    }
    print(json.dumps(out, indent=2))
    if not page_report.get("pass"):
        for msg in page_budget_blockers(page_report):
            print(msg, file=sys.stderr)
        return 3
    return 0 if compile_result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
