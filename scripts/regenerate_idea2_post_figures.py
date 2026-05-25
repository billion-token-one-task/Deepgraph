#!/usr/bin/env python3
"""Force-regenerate idea_2 motivation/overview via POST (PaperBanana + Gemini image API)."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Locked defaults for this regeneration pass (override via env if needed).
os.environ.setdefault("DEEPGRAPH_PAPERBANANA_ENABLE_POSTWRITE", "1")
# Gateway model id is gemini-3-pro-image-preview (not gemini-3.0-pro-image-preview).
os.environ.setdefault("DEEPGRAPH_PAPERBANANA_IMAGE_MODEL", "gemini-3-pro-image-preview")
os.environ.setdefault("DEEPGRAPH_PAPERBANANA_IMAGE_PROTOCOL", "gemini_native")
os.environ.setdefault(
    "DEEPGRAPH_PAPERBANANA_CMD",
    f"python3 {ROOT / 'scripts' / 'paperbanana_wrapper.py'} --out {{output}} --spec {{spec}}",
)

from agents.paper_orchestra_pipeline import _compile_main_pdf
from agents.paperorchestra.figure_orchestra import run_postwriting_api_figure_stage
from agents.paperorchestra.plotting_orchestra import default_paperbanana_cmd
from agents.workspace_layout import get_idea_workspace
from db import database as db


def _manuscript_state(run: dict, insight: dict) -> dict:
    return {
        "title": insight.get("title") or run.get("title"),
        "method_name": insight.get("method_name") or "Contrastive Perceptual Grounding (CPG)",
        "method_summary": insight.get("method_summary") or "",
        "problem_statement": insight.get("problem_statement") or "",
        "existing_weakness": insight.get("existing_weakness") or "",
        "contributions": insight.get("contributions") or [],
        "baseline_metric_name": run.get("baseline_metric_name") or "primary_score",
        "baseline_metric_value": run.get("baseline_metric_value"),
        "best_metric_value": run.get("best_metric_value"),
        "effect_pct": run.get("effect_pct"),
        "verdict": run.get("verdict"),
    }

IDEA_ID = 2
RUN_ID = 2
IMAGE_MODEL = os.environ.get("DEEPGRAPH_PAPERBANANA_IMAGE_MODEL", "gemini-3-pro-image-preview")


def _fix_overview_caption(tex: str) -> str:
    return tex.replace(
        r"grounding discrepancy $\textbackslash{}Delta$",
        r"grounding discrepancy $\Delta$",
    )


def main() -> int:
    db.init_db()
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (RUN_ID,))
    if not run or int(run["deep_insight_id"]) != IDEA_ID:
        print(f"Run {RUN_ID} for idea {IDEA_ID} not found", file=sys.stderr)
        return 1

    insight = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (IDEA_ID,))
    layout = get_idea_workspace(IDEA_ID, insight=insight, create=True)
    bundle_dir = Path(layout["paper_bundles_root"]) / "conference"
    figures_dir = bundle_dir / "figures"
    paperorchestra_figures = Path(layout["paper_current_root"]) / "paperorchestra_figures"
    paperorchestra_figures.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    main_tex_path = bundle_dir / "main.tex"
    if not main_tex_path.is_file():
        print(f"Missing {main_tex_path}", file=sys.stderr)
        return 1

    pb_cmd = default_paperbanana_cmd()
    if not pb_cmd:
        print("DEEPGRAPH_PAPERBANANA_CMD is not configured.", file=sys.stderr)
        return 2

    state = _manuscript_state(dict(run), dict(insight))
    paper_tex = main_tex_path.read_text(encoding="utf-8")

    print(f"POST figure regen: model={IMAGE_MODEL} protocol={os.environ.get('DEEPGRAPH_PAPERBANANA_IMAGE_PROTOCOL')}")
    manifest = run_postwriting_api_figure_stage(
        {},
        state,
        paper_tex,
        paperorchestra_figures,
        paperbanana_cmd=pb_cmd,
    )

    assets = manifest.get("assets") or []
    api_ok = [
        a for a in assets
        if isinstance(a, dict) and str(a.get("notes") or "") == "paperbanana_ok"
    ]
    api_fail = [
        a for a in assets
        if isinstance(a, dict)
        and ("gemini_unavailable" in str(a.get("notes") or "") or "paperbanana_failed" in str(a.get("notes") or ""))
    ]
    if not manifest.get("generated_count"):
        print(json.dumps(manifest, indent=2, ensure_ascii=False), file=sys.stderr)
        print(
            "Gemini image API was NOT used (fallback diagrams only). "
            "Create /root/Deepgraph-main/.env with:\n"
            "  DEEPGRAPH_PAPERBANANA_IMAGE_PROTOCOL=openai_compatible\n"
            "  DEEPGRAPH_PAPERBANANA_IMAGE_BASE_URL=https://<your-gateway>/v1\n"
            "  DEEPGRAPH_PAPERBANANA_IMAGE_API_KEY=<your-key>\n"
            "  DEEPGRAPH_PAPERBANANA_IMAGE_MODEL=gemini-3-pro-image-preview\n"
            "  DEEPGRAPH_PAPERBANANA_IMAGE_PROTOCOL=gemini_native\n"
            "  DEEPGRAPH_PAPERBANANA_ENABLE_POSTWRITE=true",
            file=sys.stderr,
        )
        return 3

    if api_fail and not api_ok:
        print(json.dumps(manifest, indent=2, ensure_ascii=False), file=sys.stderr)
        print(
            "All POST figures fell back to native placeholders (Gemini API not used). "
            "Check gateway quota / 403 on generateContent.",
            file=sys.stderr,
        )
        return 3

    manifest["image_model"] = IMAGE_MODEL
    manifest["regenerated_at"] = datetime.now(timezone.utc).isoformat()
    copied: list[str] = []
    for asset in manifest.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        src = Path(str(asset.get("path") or ""))
        if not src.is_file():
            continue
        dest = figures_dir / src.name
        shutil.copy2(src, dest)
        bundle_asset = dict(asset)
        bundle_asset["path"] = str(dest)
        bundle_asset["stage"] = "postwriting_api_figures"
        bundle_asset["renderer"] = "gemini_image_api"
        bundle_asset["image_model"] = IMAGE_MODEL
        copied.append(dest.name)
        (paperorchestra_figures / src.name).write_bytes(dest.read_bytes())

    (figures_dir / "postwriting_api_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    tex = main_tex_path.read_text(encoding="utf-8")
    for stem in ("fig_motivation_gemini", "fig_overview_gemini"):
        tex = re.sub(
            rf"(\\includegraphics(?:\[[^\]]*\])?\{{[^}}]*?){re.escape(stem)}([^}}]*\}})",
            rf"\1{stem}\2",
            tex,
            count=1,
            flags=re.I,
        )
    tex = _fix_overview_caption(tex)
    main_tex_path.write_text(tex, encoding="utf-8")

    current = Path(layout["paper_current_root"])
    for name in copied:
        shutil.copy2(figures_dir / name, current / "figures" / name)
    shutil.copy2(main_tex_path, current / "main.tex")
    shutil.copy2(figures_dir / "postwriting_api_figure_manifest.json", current / "figures" / "postwriting_api_figure_manifest.json")

    compile_result = _compile_main_pdf(bundle_dir)
    shutil.copy2(bundle_dir / "main.pdf", current / "main.pdf")

    out = {
        "image_model": IMAGE_MODEL,
        "generated_count": manifest.get("generated_count"),
        "copied": copied,
        "compile_ok": compile_result.get("ok"),
        "bundle_dir": str(bundle_dir),
        "pdf": str(bundle_dir / "main.pdf"),
        "manifest": manifest,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    min_ok_bytes = 500_000
    real_gemini = [
        n for n in copied
        if (figures_dir / n).is_file() and (figures_dir / n).stat().st_size >= min_ok_bytes
    ]
    return 0 if compile_result.get("ok") and real_gemini else 4


if __name__ == "__main__":
    raise SystemExit(main())
