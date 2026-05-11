"""Force a DeepGraph insight through a fresh auto_research forge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db import database as db
from orchestrator import auto_research


def _as_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _summary(insight_id: int) -> list[dict]:
    return [
        dict(row)
        for row in db.fetchall(
            """
            SELECT di.id AS insight_id, di.title, di.resource_class, di.experimentability,
                   arj.status AS auto_status, arj.stage AS auto_stage,
                   arj.experiment_run_id, arj.last_note, arj.last_error,
                   er.status AS run_status, er.phase AS run_phase,
                   er.workdir, er.best_metric_value, er.hypothesis_verdict,
                   gj.id AS gpu_job_id, gj.status AS gpu_job_status, gj.assigned_worker
            FROM deep_insights di
            LEFT JOIN auto_research_jobs arj ON arj.deep_insight_id = di.id
            LEFT JOIN experiment_runs er ON er.id = arj.experiment_run_id
            LEFT JOIN gpu_jobs gj ON gj.experiment_run_id = er.id
            WHERE di.id=?
            ORDER BY gj.id DESC
            """,
            (insight_id,),
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--insight-id", type=int, required=True)
    parser.add_argument(
        "--stage",
        default="manual_rerun_completed",
        choices=[
            "manual_rerun_completed",
            "manual_reforge_unfinished",
            "manual_requeue_unfinished",
            "retry_failed_run",
        ],
    )
    parser.add_argument("--note", default="Manual non-CGGR DeepGraph idea fallback.")
    args = parser.parse_args()

    insight = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (args.insight_id,))
    if not insight:
        print(_as_json({"event": "missing_insight", "insight_id": args.insight_id}), flush=True)
        return 2

    print(
        _as_json(
            {
                "event": "reforge_start",
                "insight_id": args.insight_id,
                "title": insight.get("title"),
                "stage": args.stage,
            }
        ),
        flush=True,
    )
    auto_research._upsert_job(
        args.insight_id,
        status="queued",
        stage=args.stage,
        experiment_run_id=None,
        last_note=args.note,
        last_error=None,
    )
    candidate = db.fetchone(
        """
        SELECT di.*, arj.status AS auto_status, arj.stage AS auto_stage,
               arj.cpu_eligible AS auto_cpu_eligible
        FROM deep_insights di
        LEFT JOIN auto_research_jobs arj ON arj.deep_insight_id = di.id
        WHERE di.id=?
        """,
        (args.insight_id,),
    )
    auto_research._process_candidate(dict(candidate or insight))
    print(_as_json({"event": "reforge_done", "summary": _summary(args.insight_id)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
