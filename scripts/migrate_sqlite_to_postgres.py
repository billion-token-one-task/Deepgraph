#!/usr/bin/env python3
"""Copy data from DeepGraph SQLite (DEEPGRAPH_DB_PATH) into PostgreSQL (DEEPGRAPH_DATABASE_URL).

Run after: python -m db.pg_init

Order respects foreign keys loosely — disable triggers if needed for your data.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

try:
    import psycopg
except ImportError:
    print("pip install 'psycopg[binary]>=3.1'", file=sys.stderr)
    raise

# Tables in safe-ish order (parents first). Adjust if migrations add FKs.
TABLES = [
    "taxonomy_nodes",
    "papers",
    "paper_taxonomy",
    "claims",
    "methods",
    "patterns",
    "contradictions",
    "gaps",
    "stats",
    "results",
    "result_taxonomy",
    "matrix_gaps",
    "paper_insights",
    "insights",
    "node_summaries",
    "graph_entities",
    "paper_entity_mentions",
    "graph_relations",
    "node_graph_summaries",
    "node_opportunities",
    "entity_resolutions",
    "entity_merge_candidates",
    "deep_insights",
    "node_entity_overlap",
    "pattern_matches",
    "contradiction_clusters",
    "performance_plateaus",
    "experiment_runs",
    "experiment_iterations",
    "experimental_claims",
    "discovery_track_record",
    "auto_research_jobs",
    "mechanism_mismatches",
    "protocol_artifacts",
    "negative_space_gaps",
    "hidden_variable_bridges",
    "claim_method_gaps",
    "gpu_workers",
    "gpu_jobs",
    "experiment_artifacts",
    "manuscript_runs",
    "manuscript_assets",
    "submission_bundles",
    "insight_events",
    "harvester_runs",
    "pipeline_events",
    "pipeline_event_consumers",
    "paper_stage_checkpoints",
]


def _dedupe_papers(rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
    chosen: dict[str, sqlite3.Row] = {}
    for row in rows:
        cols = row.keys()
        base_id = row["arxiv_base_id"] if "arxiv_base_id" in cols else None
        key = base_id or row["id"]
        current = chosen.get(key)
        if current is None:
            chosen[key] = row
            continue
        current_cols = current.keys()
        current_date = current["published_date"] if "published_date" in current_cols else ""
        row_date = row["published_date"] if "published_date" in cols else ""
        current_date = current_date or ""
        row_date = row_date or ""
        if row_date >= current_date:
            chosen[key] = row
    return list(chosen.values())


def _postgres_column_info(pg: psycopg.Connection, table: str) -> list[tuple[str, str]]:
    with pg.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position
            """,
            (table,),
        )
        return [(row[0], row[1]) for row in cur.fetchall()]


def _clean_value(value, pg_type: str | None = None):
    if isinstance(value, str):
        if value == "" and pg_type in {
            "smallint",
            "integer",
            "bigint",
            "real",
            "double precision",
            "numeric",
        }:
            return None
        return value.replace("\x00", "")
    if isinstance(value, bytes):
        return value.replace(b"\x00", b"")
    if isinstance(value, list):
        return [_clean_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clean_value(v) for v in value)
    if isinstance(value, dict):
        return {k: _clean_value(v) for k, v in value.items()}
    return value


def main() -> None:
    sqlite_path = os.getenv("DEEPGRAPH_DB_PATH", str(Path(__file__).resolve().parents[1] / "deepgraph.db"))
    pg_url = os.getenv("DEEPGRAPH_DATABASE_URL", "").strip()
    if not pg_url:
        print("Set DEEPGRAPH_DATABASE_URL", file=sys.stderr)
        sys.exit(1)
    sl = sqlite3.connect(sqlite_path)
    sl.row_factory = sqlite3.Row
    with psycopg.connect(pg_url) as pg:
        for table in TABLES:
            try:
                rows = sl.execute(f"SELECT * FROM {table}").fetchall()
            except sqlite3.OperationalError:
                continue
            if not rows:
                continue
            if table == "papers":
                rows = _dedupe_papers(rows)
            target_col_info = _postgres_column_info(pg, table)
            if not target_col_info:
                continue
            target_cols = [name for name, _ in target_col_info]
            target_types = {name: data_type for name, data_type in target_col_info}
            cols = [c for c in rows[0].keys() if c in target_cols]
            if not cols:
                continue
            placeholders = ",".join(["%s"] * len(cols))
            collist = ",".join(cols)
            sql = f"INSERT INTO {table} ({collist}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
            inserted = 0
            skipped = 0
            with pg.cursor() as cur:
                for r in rows:
                    vals = tuple(_clean_value(r[c], target_types.get(c)) for c in cols)
                    try:
                        cur.execute(sql, vals)
                        inserted += 1
                    except Exception as e:
                        pg.rollback()
                        skipped += 1
                        print(f"Skip row in {table}: {e}", flush=True)
            pg.commit()
            print(f"{table}: total={len(rows)} inserted={inserted} skipped={skipped}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
