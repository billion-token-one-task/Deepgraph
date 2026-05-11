"""Generate and launch one non-CGGR DeepGraph paper idea.

This is a focused operator script for recovering momentum when the current
paper line is exhausted or should be paused. It uses the DeepGraph discovery
stack rather than mutating an existing method.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.paper_idea_agent import discover_paper_ideas
from agents.paradigm_agent import store_deep_insight
from db import database as db
from orchestrator import auto_research
from orchestrator.discovery_scheduler import harvest_signals


CGGR_TERMS = (
    "cggr",
    "counterfactual gain gated",
    "gain gated reasoning",
    "gated reasoning",
    "selective deliberation",
)


def _as_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _contains_cggr(text: str | None) -> bool:
    lowered = str(text or "").lower()
    return any(term in lowered for term in CGGR_TERMS)


def _is_cggr_related(insight: dict) -> bool:
    fields = [
        insight.get("title"),
        insight.get("problem_statement"),
        insight.get("existing_weakness"),
        insight.get("proposed_method"),
        insight.get("experimental_plan"),
        insight.get("evidence_summary"),
    ]
    return any(_contains_cggr(str(field)) for field in fields if field)


def _job_summary(insight_ids: list[int]) -> list[dict]:
    if not insight_ids:
        return []
    placeholders = ", ".join("?" for _ in insight_ids)
    return [
        dict(row)
        for row in db.fetchall(
            f"""
            SELECT di.id AS insight_id, di.title, di.resource_class, di.experimentability,
                   arj.status AS auto_status, arj.stage AS auto_stage,
                   arj.experiment_run_id, arj.last_note, arj.last_error,
                   er.status AS run_status, er.phase AS run_phase,
                   gj.id AS gpu_job_id, gj.status AS gpu_job_status, gj.assigned_worker
            FROM deep_insights di
            LEFT JOIN auto_research_jobs arj ON arj.deep_insight_id = di.id
            LEFT JOIN experiment_runs er ON er.id = arj.experiment_run_id
            LEFT JOIN gpu_jobs gj ON gj.experiment_run_id = er.id
            WHERE di.id IN ({placeholders})
            ORDER BY di.id DESC, gj.id DESC
            """,
            tuple(insight_ids),
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-problems", type=int, default=8)
    parser.add_argument("--max-generated", type=int, default=5)
    parser.add_argument("--store-limit", type=int, default=1)
    parser.add_argument("--skip-harvest", action="store_true")
    args = parser.parse_args()

    print(
        _as_json(
            {
                "event": "deepgraph_new_idea_start",
                "max_problems": args.max_problems,
                "max_generated": args.max_generated,
                "store_limit": args.store_limit,
                "cggr_filter_terms": CGGR_TERMS,
            }
        ),
        flush=True,
    )

    if not args.skip_harvest:
        print(_as_json({"event": "harvest_signals_start"}), flush=True)
        print(_as_json({"event": "harvest_signals_done", "stats": harvest_signals()}), flush=True)

    ideas = discover_paper_ideas(
        max_problems=args.max_problems,
        max_papers=args.max_generated,
    )
    print(_as_json({"event": "ideas_generated", "count": len(ideas)}), flush=True)

    stored: list[int] = []
    skipped: list[dict] = []
    for idea in ideas:
        if _is_cggr_related(idea):
            skipped.append({"title": idea.get("title"), "reason": "cggr_related"})
            continue
        insight_id = store_deep_insight(idea)
        if insight_id:
            stored.append(int(insight_id))
            print(
                _as_json(
                    {
                        "event": "idea_stored",
                        "insight_id": insight_id,
                        "title": idea.get("title"),
                    }
                ),
                flush=True,
            )
        if len(stored) >= args.store_limit:
            break

    print(_as_json({"event": "ideas_skipped", "items": skipped[:10]}), flush=True)
    if not stored:
        print(_as_json({"event": "no_non_cggr_idea_stored"}), flush=True)
        return 2

    for insight_id in stored:
        row = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (insight_id,))
        if not row:
            continue
        print(_as_json({"event": "auto_research_process_start", "insight_id": insight_id}), flush=True)
        auto_research._process_candidate(dict(row))
        print(_as_json({"event": "auto_research_process_done", "insight_id": insight_id}), flush=True)

    print(_as_json({"event": "job_summary", "items": _job_summary(stored)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
