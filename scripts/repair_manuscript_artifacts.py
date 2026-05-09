#!/usr/bin/env python3.12
"""Repair manuscript artifacts after restoring a deploy bundle on a new host.

The PostgreSQL dump may contain absolute paths from the source machine, while
the deploy bundle carries the actual files under deploy_bundle/artifacts. This
script copies those files into the configured idea workspace, normalizes LaTeX
sources, compiles PDFs when a local LaTeX toolchain exists, and rewrites DB rows
to the local paths.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.paper_orchestra_pipeline import (
    _bundle_manifest,
    _compile_main_pdf,
    _ensure_referenced_figures,
    normalize_latex_source,
)
from agents.workspace_layout import get_idea_workspace
from config import IDEA_WORKSPACE_DIR, PROJECT_ROOT
from db import database as db


def _source_roots() -> list[Path]:
    roots = [
        IDEA_WORKSPACE_DIR,
        PROJECT_ROOT.parent / "deploy_bundle" / "artifacts" / "deepgraph_ideas",
    ]
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen and root.exists():
            out.append(root)
            seen.add(key)
    return out


def _copy_paper_artifacts(insight_id: int, paper_root: Path) -> list[str]:
    copied: list[str] = []
    for root in _source_roots():
        idea_root = root / f"idea_{insight_id}"
        for name in ("papers", "paper"):
            src = idea_root / name
            if not src.exists():
                continue
            try:
                if src.resolve() == paper_root.resolve():
                    continue
            except OSError:
                pass
            shutil.copytree(src, paper_root, dirs_exist_ok=True)
            copied.append(str(src))
    return copied


def _normalize_and_compile(root: Path) -> list[dict]:
    results: list[dict] = []
    for main_tex in sorted(root.rglob("main.tex")):
        try:
            raw = main_tex.read_text(encoding="utf-8", errors="replace")
            normalized = normalize_latex_source(raw)
            if normalized != raw:
                main_tex.write_text(normalized, encoding="utf-8")
            placeholders = _ensure_referenced_figures(main_tex.parent, normalized)
        except OSError as exc:
            results.append({"path": str(main_tex), "ok": False, "error": str(exc)})
            continue
        compile_result = _compile_main_pdf(main_tex.parent)
        manifest_path = main_tex.parent / "artifact_manifest.json"
        manifest_path.write_text(
            json.dumps(_bundle_manifest(main_tex.parent), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        results.append({"path": str(main_tex), "placeholder_figures": placeholders, **compile_result})
    return results


def _update_db_paths(insight: dict, layout: dict) -> None:
    insight_id = int(insight["id"])
    paper_root = Path(layout["paper_root"])
    current_root = Path(layout["paper_current_root"])
    bundles_root = Path(layout["paper_bundles_root"])
    db.execute(
        """
        UPDATE deep_insights
        SET workspace_root=?, experiment_root=?, plan_root=?, paper_root=?, updated_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (
            str(layout["workspace_root"]),
            str(layout["experiment_root"]),
            str(layout["plan_root"]),
            str(paper_root),
            insight_id,
        ),
    )
    db.execute(
        """
        UPDATE manuscript_runs
        SET workdir=?, updated_at=CURRENT_TIMESTAMP
        WHERE deep_insight_id=?
        """,
        (str(current_root), insight_id),
    )
    bundles = db.fetchall(
        """
        SELECT sb.id, sb.bundle_format
        FROM submission_bundles sb
        JOIN manuscript_runs mr ON mr.id = sb.manuscript_run_id
        WHERE mr.deep_insight_id=?
        """,
        (insight_id,),
    )
    for bundle in bundles:
        bundle_dir = bundles_root / str(bundle["bundle_format"])
        manifest = bundle_dir / "artifact_manifest.json"
        db.execute(
            """
            UPDATE submission_bundles
            SET bundle_path=?, manifest_path=?
            WHERE id=?
            """,
            (str(bundle_dir), str(manifest), bundle["id"]),
        )
    db.commit()


def repair_all(*, insight_id: int | None = None) -> list[dict]:
    db.init_db()
    where = "WHERE id=?" if insight_id is not None else ""
    params: tuple = (int(insight_id),) if insight_id is not None else ()
    insights = db.fetchall(f"SELECT * FROM deep_insights {where} ORDER BY id", params)
    repaired: list[dict] = []
    for insight in insights:
        manuscripts = db.fetchall(
            "SELECT id FROM manuscript_runs WHERE deep_insight_id=?",
            (insight["id"],),
        )
        if not manuscripts:
            continue
        layout = get_idea_workspace(int(insight["id"]), insight=dict(insight), create=True, sync_db=True)
        paper_root = Path(layout["paper_root"])
        copied = _copy_paper_artifacts(int(insight["id"]), paper_root)
        compile_results = _normalize_and_compile(paper_root)
        _update_db_paths(dict(insight), layout)
        repaired.append(
            {
                "insight_id": int(insight["id"]),
                "paper_root": str(paper_root),
                "copied_from": copied,
                "compiled": compile_results,
            }
        )
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair local manuscript paths and PDFs.")
    parser.add_argument("--insight-id", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(repair_all(insight_id=args.insight_id), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
