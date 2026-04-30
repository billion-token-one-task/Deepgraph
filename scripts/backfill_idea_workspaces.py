#!/usr/bin/env python3.12
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.workspace_layout import backfill_legacy_layout, ensure_run_workspace, get_idea_workspace
from db import database as db

_ACTIVE_STATUSES = {"pending", "scaffolding", "reproducing", "testing", "running_gpu", "running_cpu"}


def _choose_canonical_run(insight: dict, runs: list[dict]) -> int | None:
    current = insight.get("canonical_run_id")
    if current:
        return int(current)
    for run in runs:
        if (run.get("status") or "") in _ACTIVE_STATUSES:
            return int(run["id"])
    completed = [run for run in runs if run.get("status") in {"bundle_ready", "completed"}]
    if completed:
        return int(sorted(completed, key=lambda row: (str(row.get("completed_at") or row.get("created_at") or ""), int(row["id"])), reverse=True)[0]["id"])
    if runs:
        return int(sorted(runs, key=lambda row: (str(row.get("created_at") or ""), int(row["id"])), reverse=True)[0]["id"])
    return None


def backfill_all(*, insight_id: int | None = None, dry_run: bool = False, copy_files: bool = True) -> list[dict]:
    db.init_db()
    params: tuple[object, ...] = ()
    where = ""
    if insight_id is not None:
        where = "WHERE di.id=?"
        params = (int(insight_id),)
    insights = db.fetchall(
        f"""
        SELECT di.*
        FROM deep_insights di
        {where}
        ORDER BY di.id
        """,
        params,
    )
    results: list[dict] = []
    for insight in insights:
        runs = db.fetchall(
            "SELECT * FROM experiment_runs WHERE deep_insight_id=? ORDER BY id",
            (insight["id"],),
        )
        manuscripts = db.fetchall(
            "SELECT * FROM manuscript_runs WHERE deep_insight_id=? ORDER BY id",
            (insight["id"],),
        )
        if not runs and not manuscripts:
            continue
        migration = backfill_legacy_layout(
            insight=dict(insight),
            run_rows=[dict(row) for row in runs],
            manuscript_rows=[dict(row) for row in manuscripts],
            copy_files=copy_files and not dry_run,
        )
        layout = get_idea_workspace(int(insight["id"]), insight=dict(insight), create=True, sync_db=not dry_run)
        canonical_run_id = _choose_canonical_run(dict(insight), [dict(row) for row in runs])
        if not dry_run:
            db.execute(
                """
                UPDATE deep_insights
                SET workspace_root=?, experiment_root=?, plan_root=?, paper_root=?, canonical_run_id=?, updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (
                    migration["workspace_root"],
                    migration["experiment_root"],
                    migration["plan_root"],
                    migration["paper_root"],
                    canonical_run_id,
                    insight["id"],
                ),
            )
            for run in runs:
                run_root = Path(ensure_run_workspace(int(insight["id"]), int(run["id"]), insight=dict(insight))["run_root"])
                db.execute(
                    "UPDATE experiment_runs SET workdir=? WHERE id=?",
                    (str(run_root), run["id"]),
                )
            for manuscript in manuscripts:
                db.execute(
                    "UPDATE manuscript_runs SET workdir=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (str(layout["paper_current_root"]), manuscript["id"]),
                )
            db.commit()
        results.append(
            {
                "insight_id": int(insight["id"]),
                "workspace_root": migration["workspace_root"],
                "canonical_run_id": canonical_run_id,
                "run_ids": migration["migrated_runs"],
                "manuscript_count": len(manuscripts),
                "dry_run": dry_run,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill legacy experiment/manuscript directories into idea workspaces.")
    parser.add_argument("--insight-id", type=int, default=None, help="Only migrate a single idea.")
    parser.add_argument("--dry-run", action="store_true", help="Plan the migration without updating files or database rows.")
    parser.add_argument("--skip-copy", action="store_true", help="Update database paths without copying legacy files.")
    args = parser.parse_args()
    result = backfill_all(insight_id=args.insight_id, dry_run=args.dry_run, copy_files=not args.skip_copy)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
