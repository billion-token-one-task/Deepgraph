from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from config import IDEA_WORKSPACE_DIR
from db import database as db


def _workspace_root_for_insight(insight_id: int) -> Path:
    return IDEA_WORKSPACE_DIR / f"idea_{int(insight_id)}"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _serialize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, indent=2, ensure_ascii=False, default=str)


def get_idea_workspace(insight_id: int, insight: dict | None = None, *, create: bool = True, sync_db: bool = True) -> dict[str, Any]:
    insight = insight or db.fetchone("SELECT * FROM deep_insights WHERE id=?", (insight_id,)) or {"id": insight_id}
    workspace_root = Path(str(insight.get("workspace_root") or _workspace_root_for_insight(insight_id)))
    experiment_root = Path(str(insight.get("experiment_root") or (workspace_root / "experiment")))
    plan_root = Path(str(insight.get("plan_root") or (workspace_root / "plan")))
    paper_root = Path(str(insight.get("paper_root") or (workspace_root / "paper")))
    layout = {
        "insight_id": int(insight_id),
        "workspace_root": workspace_root,
        "experiment_root": experiment_root,
        "experiment_current_root": experiment_root / "current",
        "experiment_runs_root": experiment_root / "runs",
        "plan_root": plan_root,
        "paper_root": paper_root,
        "paper_current_root": paper_root / "current",
        "paper_bundles_root": paper_root / "bundles",
        "paper_manifests_root": paper_root / "manifests",
        "canonical_run_id": insight.get("canonical_run_id"),
    }
    if create:
        for key in (
            "workspace_root",
            "experiment_root",
            "experiment_current_root",
            "experiment_runs_root",
            "plan_root",
            "paper_root",
            "paper_current_root",
            "paper_bundles_root",
            "paper_manifests_root",
        ):
            Path(layout[key]).mkdir(parents=True, exist_ok=True)
    if sync_db and insight.get("id") is not None:
        desired = {
            "workspace_root": str(workspace_root),
            "experiment_root": str(experiment_root),
            "plan_root": str(plan_root),
            "paper_root": str(paper_root),
        }
        current = {key: str(insight.get(key) or "") for key in desired}
        if any(current[key] != value for key, value in desired.items()):
            db.execute(
                """
                UPDATE deep_insights
                SET workspace_root=?, experiment_root=?, plan_root=?, paper_root=?, updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (
                    desired["workspace_root"],
                    desired["experiment_root"],
                    desired["plan_root"],
                    desired["paper_root"],
                    insight_id,
                ),
            )
            db.commit()
    return layout


def ensure_run_workspace(insight_id: int, run_id: int, insight: dict | None = None) -> dict[str, Any]:
    layout = get_idea_workspace(insight_id, insight=insight, create=True, sync_db=True)
    run_root = Path(layout["experiment_runs_root"]) / f"run_{int(run_id)}"
    info = {
        **layout,
        "run_root": run_root,
        "code_root": run_root / "code",
        "results_root": run_root / "results",
        "spec_root": run_root / "spec",
        "codex_root": run_root / "codex",
    }
    for key in ("run_root", "code_root", "results_root", "spec_root", "codex_root"):
        Path(info[key]).mkdir(parents=True, exist_ok=True)
    return info


def paper_bundle_root(insight_id: int, bundle_format: str, insight: dict | None = None) -> Path:
    layout = get_idea_workspace(insight_id, insight=insight, create=True, sync_db=True)
    return Path(layout["paper_bundles_root"]) / str(bundle_format)


def plan_file_path(insight_id: int, name: str, insight: dict | None = None) -> Path:
    layout = get_idea_workspace(insight_id, insight=insight, create=True, sync_db=True)
    return Path(layout["plan_root"]) / name


def write_plan_files(
    insight_id: int,
    *,
    files: dict[str, Any],
    run_id: int | None = None,
    insight: dict | None = None,
    mirror_to_run_spec: bool = True,
) -> dict[str, str]:
    layout = get_idea_workspace(insight_id, insight=insight, create=True, sync_db=True)
    run_info = ensure_run_workspace(insight_id, run_id, insight=insight) if run_id is not None else None
    written: dict[str, str] = {}
    for filename, content in files.items():
        text = _serialize_content(content)
        plan_path = Path(layout["plan_root"]) / filename
        _write_text(plan_path, text)
        written[filename] = str(plan_path)
        if run_info is not None and mirror_to_run_spec:
            _write_text(Path(run_info["spec_root"]) / filename, text)
    return written


def write_latest_status(insight_id: int, payload: dict[str, Any], *, run_id: int | None = None, insight: dict | None = None) -> str:
    payload = {
        **payload,
        "insight_id": int(insight_id),
        "run_id": int(run_id) if run_id is not None else payload.get("run_id"),
    }
    return write_plan_files(
        insight_id,
        files={"latest_status.json": payload},
        run_id=run_id,
        insight=insight,
        mirror_to_run_spec=False,
    )["latest_status.json"]


def promote_canonical_run(insight_id: int, run_id: int, insight: dict | None = None) -> dict[str, Any]:
    info = ensure_run_workspace(insight_id, run_id, insight=insight)
    db.execute(
        "UPDATE deep_insights SET canonical_run_id=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (int(run_id), int(insight_id)),
    )
    db.commit()
    _refresh_current_link(Path(info["experiment_current_root"]), Path(info["run_root"]))
    write_latest_status(
        insight_id,
        {
            "canonical_run_id": int(run_id),
            "experiment_current_root": str(info["experiment_current_root"]),
            "run_root": str(info["run_root"]),
            "status": "canonical_promoted",
        },
        run_id=run_id,
        insight=insight,
    )
    return info


def _refresh_current_link(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink() or link_path.is_file():
        link_path.unlink()
    elif link_path.is_dir():
        shutil.rmtree(link_path)
    try:
        link_path.symlink_to(target_path, target_is_directory=True)
    except OSError:
        link_path.mkdir(parents=True, exist_ok=True)
        _write_text(link_path / "CURRENT_RUN.txt", str(target_path))


def list_paper_assets(insight_id: int, insight: dict | None = None) -> list[dict[str, Any]]:
    layout = get_idea_workspace(insight_id, insight=insight, create=True, sync_db=True)
    assets: list[dict[str, Any]] = []
    for root in (
        Path(layout["paper_current_root"]),
        Path(layout["paper_bundles_root"]),
        Path(layout["paper_manifests_root"]),
    ):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(layout["paper_root"]).as_posix()
            assets.append(
                {
                    "path": rel,
                    "size": path.stat().st_size,
                    "suffix": path.suffix.lower(),
                }
            )
    return assets


def resolve_paper_asset(insight_id: int, asset: str, insight: dict | None = None) -> Path:
    layout = get_idea_workspace(insight_id, insight=insight, create=True, sync_db=True)
    paper_root = Path(layout["paper_root"]).resolve()
    candidate = (paper_root / asset).resolve()
    if candidate != paper_root and paper_root not in candidate.parents:
        raise ValueError("asset path escapes paper root")
    return candidate


def backfill_legacy_layout(
    *,
    insight: dict,
    run_rows: list[dict],
    manuscript_rows: list[dict],
    copy_files: bool = True,
) -> dict[str, Any]:
    layout = get_idea_workspace(int(insight["id"]), insight=insight, create=True, sync_db=True)
    migrated_runs: list[int] = []
    for run in run_rows:
        info = ensure_run_workspace(int(insight["id"]), int(run["id"]), insight=insight)
        source = Path(str(run.get("workdir") or "")).expanduser()
        target = Path(info["run_root"])
        if copy_files and source.exists() and source.resolve() != target.resolve():
            shutil.copytree(source, target, dirs_exist_ok=True)
        migrated_runs.append(int(run["id"]))
    for manuscript in manuscript_rows:
        source = Path(str(manuscript.get("workdir") or "")).expanduser()
        target = Path(layout["paper_current_root"])
        if copy_files and source.exists() and source.resolve() != target.resolve():
            shutil.copytree(source, target, dirs_exist_ok=True)
    return {
        "workspace_root": str(layout["workspace_root"]),
        "experiment_root": str(layout["experiment_root"]),
        "plan_root": str(layout["plan_root"]),
        "paper_root": str(layout["paper_root"]),
        "migrated_runs": migrated_runs,
    }
