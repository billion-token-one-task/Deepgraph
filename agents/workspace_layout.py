from __future__ import annotations

"""Idea workspace layout (one repo per deep_insight).

Layout on disk::

    idea_<id>/
      papers/                 # 论文与投稿产物 (current/, bundles/, manifests/)
      plan/                   # 计划与状态
      experiments/            # 实验大类根目录（不是单条「实验记录」）
        main/                 # 主表 / 主实验线
          current/            -> 指向当前 canonical run（符号链接或占位）
          runs/
            run_<db_id>/      # 一次可复现执行（才是以前的「run」语义）
        ablation/
          current/
          runs/
        visualization/
          current/
          runs/

Database:
  - ``deep_insights.experiment_root`` → ``.../idea_N/experiments``（所有 suite 的父目录）
  - ``experiment_runs.experiment_suite`` → ``main`` | ``ablation`` | ``visualization`` | 自定义

Legacy ``idea_*/paper`` / ``idea_*/experiment`` 在首次 ``get_idea_workspace(create=True)`` 时自动迁移为
``papers`` / ``experiments/main``。
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any

from config import IDEA_WORKSPACE_DIR
from db import database as db

PAPERS_DIR = "papers"
EXPERIMENTS_DIR = "experiments"
SUITE_MAIN = "main"
SUITE_ABLATION = "ablation"
SUITE_VISUALIZATION = "visualization"
DEFAULT_EXPERIMENT_SUITE = SUITE_MAIN
KNOWN_EXPERIMENT_SUITES = (SUITE_MAIN, SUITE_ABLATION, SUITE_VISUALIZATION)


def _workspace_root_for_insight(insight_id: int) -> Path:
    return IDEA_WORKSPACE_DIR / f"idea_{int(insight_id)}"


def _looks_foreign_restored_path(raw: str) -> bool:
    """Detect Unix absolute paths restored into a Windows runtime."""
    text = str(raw or "").strip().replace("\\", "/")
    if os.name != "nt" or not text:
        return False
    return text.startswith("/home/") or text.startswith("/root/") or text.startswith("/data/")


def _path_within(path: Path, root: Path) -> bool:
    try:
        resolved = path.resolve()
        resolved_root = root.resolve()
    except OSError:
        resolved = path.absolute()
        resolved_root = root.absolute()
    return resolved == resolved_root or resolved_root in resolved.parents


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _serialize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, indent=2, ensure_ascii=False, default=str)


def _migrate_legacy_idea_workspace(workspace_root: Path) -> None:
    """Rename legacy ``paper`` / ``experiment`` trees to ``papers`` / ``experiments/main``."""
    legacy_paper = workspace_root / "paper"
    modern_paper = workspace_root / PAPERS_DIR
    if legacy_paper.is_dir() and not modern_paper.exists():
        modern_paper.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_paper), str(modern_paper))

    legacy_exp = workspace_root / "experiment"
    suite_main = workspace_root / EXPERIMENTS_DIR / DEFAULT_EXPERIMENT_SUITE
    if legacy_exp.is_dir() and not suite_main.exists():
        suite_main.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_exp), str(suite_main))


def _ensure_dir_or_current_link(path: Path) -> None:
    """Create a directory unless the path is already a symlink/current pointer.

    ``current`` paths are allowed to be:
    - a real directory
    - a symlink to the canonical run / paper dir
    - a placeholder directory later populated with ``CURRENT_RUN.txt``

    ``Path.mkdir(exist_ok=True)`` still raises ``FileExistsError`` on symlinks
    on some runtimes, so we special-case them here.
    """
    if path.is_symlink():
        return
    if path.exists():
        if path.is_dir():
            return
        raise FileExistsError(f"Workspace path exists and is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _resolve_suite_for_run(run_id: int) -> str:
    row = db.fetchone("SELECT experiment_suite FROM experiment_runs WHERE id=?", (int(run_id),))
    if row:
        s = str(row.get("experiment_suite") or "").strip()
        if s:
            return s
    return DEFAULT_EXPERIMENT_SUITE


def get_idea_workspace(insight_id: int, insight: dict | None = None, *, create: bool = True, sync_db: bool = True) -> dict[str, Any]:
    insight = insight or db.fetchone("SELECT * FROM deep_insights WHERE id=?", (insight_id,)) or {"id": insight_id}
    stored_workspace = str(insight.get("workspace_root") or "").strip()
    if stored_workspace and not _looks_foreign_restored_path(stored_workspace):
        workspace_root = Path(stored_workspace)
    else:
        workspace_root = _workspace_root_for_insight(insight_id)
    if create:
        _migrate_legacy_idea_workspace(workspace_root)

    experiments_root = workspace_root / EXPERIMENTS_DIR
    stored_plan = str(insight.get("plan_root") or "").strip()
    if stored_plan and not _looks_foreign_restored_path(stored_plan):
        candidate_plan = Path(stored_plan)
        plan_root = candidate_plan if _path_within(candidate_plan, workspace_root) else workspace_root / "plan"
    else:
        plan_root = workspace_root / "plan"
    paper_root = workspace_root / PAPERS_DIR

    layout = {
        "insight_id": int(insight_id),
        "workspace_root": workspace_root,
        "experiments_root": experiments_root,
        # DB column experiment_root: parent of all suites (…/experiments), not a single run bucket
        "experiment_root": experiments_root,
        "experiment_current_root": experiments_root / DEFAULT_EXPERIMENT_SUITE / "current",
        "experiment_runs_root": experiments_root / DEFAULT_EXPERIMENT_SUITE / "runs",
        "plan_root": plan_root,
        "paper_root": paper_root,
        "paper_current_root": paper_root / "current",
        "paper_bundles_root": paper_root / "bundles",
        "paper_manifests_root": paper_root / "manifests",
        "canonical_run_id": insight.get("canonical_run_id"),
    }
    if create:
        _ensure_dir_or_current_link(workspace_root)
        _ensure_dir_or_current_link(plan_root)
        for key in ("paper_root", "paper_current_root", "paper_bundles_root", "paper_manifests_root"):
            _ensure_dir_or_current_link(Path(layout[key]))
        _ensure_dir_or_current_link(experiments_root)
        for suite in KNOWN_EXPERIMENT_SUITES:
            _ensure_dir_or_current_link(experiments_root / suite / "runs")
            _ensure_dir_or_current_link(experiments_root / suite / "current")

    if sync_db and insight.get("id") is not None:
        desired = {
            "workspace_root": str(workspace_root),
            "experiment_root": str(experiments_root),
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


def ensure_manuscript_run_workspace(
    insight_id: int,
    run_id: int | None = None,
    insight: dict | None = None,
) -> dict[str, Any]:
    """Ensure paper/manuscript directories exist for an insight (legacy import alias)."""
    layout = get_idea_workspace(insight_id, insight=insight, create=True, sync_db=True)
    for key in ("paper_root", "paper_current_root", "paper_bundles_root", "paper_manifests_root"):
        _ensure_dir_or_current_link(Path(layout[key]))
    if run_id is not None:
        ensure_run_workspace(insight_id, int(run_id), insight=insight)
    return layout


def ensure_run_workspace(
    insight_id: int,
    run_id: int,
    insight: dict | None = None,
    *,
    suite: str | None = None,
) -> dict[str, Any]:
    layout = get_idea_workspace(insight_id, insight=insight, create=True, sync_db=True)
    experiments_root = Path(layout["experiments_root"])
    resolved_suite = (suite or _resolve_suite_for_run(run_id)).strip() or DEFAULT_EXPERIMENT_SUITE
    suite_runs_root = experiments_root / resolved_suite / "runs"
    suite_current_root = experiments_root / resolved_suite / "current"
    run_root = suite_runs_root / f"run_{int(run_id)}"
    info = {
        **layout,
        "experiment_suite": resolved_suite,
        "suite_runs_root": suite_runs_root,
        "suite_current_root": suite_current_root,
        "run_root": run_root,
        "code_root": run_root / "code",
        "results_root": run_root / "results",
        "spec_root": run_root / "spec",
        "codex_root": run_root / "codex",
    }
    _ensure_dir_or_current_link(suite_runs_root)
    _ensure_dir_or_current_link(suite_current_root)
    for key in ("run_root", "code_root", "results_root", "spec_root", "codex_root"):
        _ensure_dir_or_current_link(Path(info[key]))
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
    _refresh_current_link(Path(info["suite_current_root"]), Path(info["run_root"]))
    write_latest_status(
        insight_id,
        {
            "canonical_run_id": int(run_id),
            "experiment_suite": info["experiment_suite"],
            "suite_current_root": str(info["suite_current_root"]),
            "run_root": str(info["run_root"]),
            "status": "canonical_promoted",
        },
        run_id=run_id,
        insight=insight,
    )
    return info


def _refresh_current_link(link_path: Path, target_path: Path) -> None:
    def _write_marker() -> None:
        link_path.mkdir(parents=True, exist_ok=True)
        _write_text(link_path / "CURRENT_RUN.txt", str(target_path))

    try:
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        elif link_path.is_dir():
            shutil.rmtree(link_path)
    except OSError:
        # Windows can transiently lock the placeholder current/ directory while
        # another scheduler thread is collecting artifacts. Keep the pipeline
        # moving by updating the marker instead of failing the whole GPU job.
        _write_marker()
        return
    try:
        link_path.symlink_to(target_path, target_is_directory=True)
    except OSError:
        _write_marker()


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
