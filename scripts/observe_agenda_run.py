#!/usr/bin/env python3
"""Sample the live research loop at a fixed interval, one JSON line per tick.

Written for supervised observation windows: turn autonomy on for one agenda,
let this record what the system actually does, then compare the trace against
what the agenda was supposed to do. It is strictly read-only -- it takes no
lock, claims no job and writes nothing to any business table.

The fields are chosen to separate "the loop is moving" from "the loop is
producing evidence", because the two have come apart before: runs reach
``completed`` while every metric artifact carries a null value, so nothing
ever climbs the evidence ladder.

    python3 scripts/observe_agenda_run.py --agenda 8 --interval 1800 \
        --hours 6 --out /tmp/agenda8_observation.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import database as db  # noqa: E402


def _one(sql: str, params: tuple = ()) -> dict:
    return dict(db.fetchone(sql, params) or {})


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    return [dict(row) for row in db.fetchall(sql, params)]


def snapshot(agenda_id: int) -> dict:
    """One tick. Every number is scoped to the agenda unless noted."""
    agenda = _one(
        "SELECT token_budget, token_spent, token_reserved, gpu_hours_budget,"
        " gpu_hours_spent, gpu_hours_reserved FROM research_agendas WHERE id=?",
        (agenda_id,),
    )
    return {
        "at": datetime.now(timezone.utc).isoformat(),
        "agenda_id": agenda_id,
        "budget": agenda,
        "jobs": _rows(
            "SELECT status, stage, COUNT(*) AS n FROM auto_research_jobs"
            " WHERE agenda_id=? GROUP BY 1,2 ORDER BY 3 DESC",
            (agenda_id,),
        ),
        "runs": _rows(
            "SELECT status, scientific_evidence_state, COUNT(*) AS n"
            " FROM experiment_runs WHERE agenda_id=? GROUP BY 1,2 ORDER BY 3 DESC",
            (agenda_id,),
        ),
        "ideas": _rows(
            "SELECT status, COUNT(*) AS n FROM deep_insights"
            " WHERE agenda_id=? GROUP BY 1 ORDER BY 2 DESC",
            (agenda_id,),
        ),
        "grants": _rows(
            "SELECT stage, status, COUNT(*) AS n, COALESCE(SUM(token_cap),0) AS token_cap,"
            " COALESCE(SUM(max_gpu_hours),0) AS gpu_hours"
            " FROM resource_grants WHERE agenda_id=? GROUP BY 1,2",
            (agenda_id,),
        ),
        "outcomes": _one(
            "SELECT COUNT(*) AS n FROM outcome_records WHERE agenda_id=?",
            (agenda_id,),
        ).get("n", 0),
        # The honesty check: artifacts that exist versus artifacts that carry a
        # number. A high count with zero values is the silent-failure signature.
        "artifacts": _rows(
            "SELECT artifact_type, COUNT(*) AS n, COUNT(metric_value) AS with_value"
            " FROM experiment_artifacts WHERE agenda_id=? GROUP BY 1 ORDER BY 2 DESC",
            (agenda_id,),
        ),
        "gpu_jobs": _rows(
            "SELECT status, COUNT(*) AS n FROM gpu_jobs WHERE agenda_id=? GROUP BY 1",
            (agenda_id,),
        ),
        # Unscoped on purpose: worker health is a property of the fleet, and a
        # stale heartbeat is the thing most likely to strand this agenda.
        "gpu_workers": _rows(
            "SELECT status, gpu_model, COUNT(*) AS n, MAX(heartbeat_at) AS last_heartbeat"
            " FROM gpu_workers WHERE id LIKE 'ssh:%' GROUP BY 1,2",
        ),
        "llm_routes": _rows(
            "SELECT role, provider, status, COUNT(*) AS n"
            " FROM llm_route_observations GROUP BY 1,2,3 ORDER BY 4 DESC",
        ),
        "recent_errors": _rows(
            "SELECT id, stage, LEFT(COALESCE(last_error,''), 180) AS last_error"
            " FROM auto_research_jobs WHERE agenda_id=? AND last_error IS NOT NULL"
            " ORDER BY updated_at DESC LIMIT 5",
            (agenda_id,),
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agenda", type=int, required=True)
    parser.add_argument("--interval", type=int, default=1800, help="seconds between ticks")
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--out", required=True, help="JSONL output path")
    args = parser.parse_args()

    deadline = time.time() + args.hours * 3600
    out = Path(args.out)
    tick = 0
    while True:
        tick += 1
        try:
            record = snapshot(args.agenda)
            record["tick"] = tick
        except Exception as exc:  # a failed tick must not end the window
            db.rollback()
            record = {
                "at": datetime.now(timezone.utc).isoformat(),
                "tick": tick,
                "agenda_id": args.agenda,
                "error": f"{type(exc).__name__}: {exc}",
            }
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        print(f"[observe] tick {tick} written at {record['at']}", flush=True)
        if time.time() >= deadline:
            return 0
        time.sleep(max(60, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
