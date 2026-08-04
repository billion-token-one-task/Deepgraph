"""Bring pre-agenda historical work into agenda scope, audited and by lineage.

The meta-harness UI and APIs are agenda-scoped; rows created before agenda
scoping have agenda_id NULL and are therefore invisible. This script makes
them visible without inventing anything:

1. Orphan deep_insights (agenda_id NULL) are imported into a named target
   agenda through AgendaRepository.import_legacy_record - one audited,
   idempotent legacy_scope_imports row per insight.
2. Orphan auto_research_jobs whose insight is scoped follow their insight the
   same audited way.
3. Companion tables inherit scope purely by lineage with guarded UPDATEs
   (agenda_id IS NULL only): experiment_runs from their deep_insight,
   experiment_iterations / experimental_claims / experiment_artifacts from
   their run, manuscript_runs from their insight, submission_bundles from
   their manuscript run.

Nothing is deleted, no verdict or evidence state is created or changed, and
rows whose lineage is broken are reported and left untouched. Scope changes
only ever go NULL -> agenda, never agenda -> agenda.

Usage:
  python3 scripts/legacy_scope_backfill.py                # dry run (default)
  python3 scripts/legacy_scope_backfill.py --execute \
      --orphan-agenda-name legacy-archive-2026h1
Execution additionally requires DEEPGRAPH_ALLOW_LEGACY_BACKFILL=1 in the
environment and a fresh database backup taken by the operator.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.agenda_loader import parse_agenda
from agents.agenda_repository import AgendaRepository
from db import database as db

ACTOR = "legacy_scope_backfill"
REASON = "pre-agenda historical record made visible in agenda scope; lineage preserved, content untouched"

# Companion tables that inherit agenda scope from a parent row. Each entry is
# (table, parent_table, join_column_on_child). The correlated-subquery form
# works on both PostgreSQL and SQLite.
_LINEAGE = (
    ("auto_research_jobs", "deep_insights", "deep_insight_id"),
    ("experiment_runs", "deep_insights", "deep_insight_id"),
    ("experiment_iterations", "experiment_runs", "run_id"),
    ("experimental_claims", "experiment_runs", "run_id"),
    ("experiment_artifacts", "experiment_runs", "run_id"),
    ("manuscript_runs", "deep_insights", "deep_insight_id"),
    ("submission_bundles", "manuscript_runs", "manuscript_run_id"),
)


def _count(sql: str, params: tuple = ()) -> int:
    row = db.fetchone(sql, params)
    return int((row or {}).get("c") or 0)


def _orphans(table: str) -> int:
    return _count(f"SELECT COUNT(*) as c FROM {table} WHERE agenda_id IS NULL")


def ensure_orphan_agenda(name: str, execute: bool) -> int | None:
    row = db.fetchone("SELECT id FROM research_agendas WHERE name=?", (name,))
    if row:
        return int(row["id"])
    if not execute:
        return None
    agenda = parse_agenda(
        {
            "version": "v1",
            "name": name,
            "description": (
                "Archive scope for historical work that predates agenda scoping. "
                "Imported records keep their original content and operational "
                "status; none carries a scientific decision unless it later "
                "passes the evidence ladder."
            ),
            "focus": ["legacy", "archive"],
            "required_output": {"goal": "idea_only"},
            "submitter": "operator",
            "token_budget": 1,
            "gpu_hours_budget": 0,
            "backend_allowlist": ["cpu"],
            "max_concurrency": 1,
            "backlog_policy": "explicit_import_only",
            "source": "legacy_scope_backfill",
        }
    )
    agenda_id = AgendaRepository().create(agenda)
    print(f"created orphan-target agenda '{name}' as id {agenda_id}")
    return agenda_id


def import_orphan_insights(agenda_id: int, execute: bool) -> int:
    rows = db.fetchall(
        "SELECT id FROM deep_insights WHERE agenda_id IS NULL ORDER BY id"
    )
    if not execute:
        print(f"would import {len(rows)} orphan deep_insights into agenda {agenda_id}")
        return 0
    repo = AgendaRepository()
    done = 0
    for row in rows:
        repo.import_legacy_record(
            agenda_id=agenda_id,
            entity_type="deep_insight",
            entity_id=int(row["id"]),
            actor=ACTOR,
            reason=REASON,
            idempotency_key=f"backfill-deep_insight-{row['id']}",
        )
        done += 1
    print(f"imported {done} orphan deep_insights into agenda {agenda_id}")
    return done


def backfill_lineage(execute: bool) -> None:
    for table, parent, join_col in _LINEAGE:
        eligible = _count(
            f"""
            SELECT COUNT(*) as c FROM {table} t
            WHERE t.agenda_id IS NULL
              AND (SELECT p.agenda_id FROM {parent} p WHERE p.id = t.{join_col})
                  IS NOT NULL
            """
        )
        if not execute:
            # Counts cascade at execute time: children become eligible only
            # after their parent table is backfilled earlier in this loop, so
            # a dry run under-reports downstream tables.
            print(f"would backfill {eligible:5d} rows in {table} from {parent} (more after cascade)")
            continue
        db.execute(
            f"""
            UPDATE {table}
            SET agenda_id = (
                SELECT p.agenda_id FROM {parent} p
                WHERE p.id = {table}.{join_col}
            )
            WHERE agenda_id IS NULL
              AND (SELECT p.agenda_id FROM {parent} p WHERE p.id = {table}.{join_col})
                  IS NOT NULL
            """
        )
        db.commit()
        print(f"backfilled {eligible:5d} rows in {table} from {parent}")


def report(title: str) -> None:
    print(f"--- {title} ---")
    for table in (
        "deep_insights",
        "auto_research_jobs",
        "experiment_runs",
        "experiment_iterations",
        "experimental_claims",
        "experiment_artifacts",
        "manuscript_runs",
        "submission_bundles",
    ):
        try:
            print(f"{table:24s} orphans: {_orphans(table)}")
        except Exception as exc:
            db.rollback()
            print(f"{table:24s} unavailable: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="apply changes (default is dry run)")
    parser.add_argument("--orphan-agenda-name", default="legacy-archive-2026h1",
                        help="agenda that receives orphan deep_insights")
    args = parser.parse_args()

    if args.execute and os.environ.get("DEEPGRAPH_ALLOW_LEGACY_BACKFILL") != "1":
        print("refusing: --execute requires DEEPGRAPH_ALLOW_LEGACY_BACKFILL=1 "
              "and a fresh database backup", file=sys.stderr)
        return 2

    report("before")
    agenda_id = ensure_orphan_agenda(args.orphan_agenda_name, args.execute)
    if agenda_id is None:
        print(f"(dry run) orphan-target agenda '{args.orphan_agenda_name}' "
              "would be created")
        agenda_id = -1
    import_orphan_insights(agenda_id, args.execute)
    # Lineage backfill runs after insight import so freshly scoped insights
    # propagate to their jobs/runs in the same pass. auto_research_jobs go
    # through lineage rather than the per-entity importer because
    # import_legacy_record validates the target agenda contract on read, and
    # historical agendas may not validate (e.g. NULL token_budget).
    backfill_lineage(args.execute)
    report("after")
    if not args.execute:
        print("dry run complete; re-run with --execute "
              "DEEPGRAPH_ALLOW_LEGACY_BACKFILL=1 after a database backup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
