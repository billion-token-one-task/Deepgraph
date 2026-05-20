"""Coordinate experiment progress, feasibility, and tuning agents."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agents.experiment_feasibility_agent import assess_feasibility
from agents.experiment_progress_monitor import ProgressReport, inspect_run
from agents.experiment_tuning_agent import recommend_tuning
from agents.metric_parser import (
    benchmark_scores,
    build_benchmark_summary_from_predictions,
)
from config import EXPERIMENT_WATCHDOG_ENABLED
from db import database as db
from orchestrator.compute_routing import apply_route_gpu

_LAST_LINE_CACHE: dict[int, tuple[int, float]] = {}


def _write_report(workdir: Path, payload: dict[str, Any]) -> Path:
    report_dir = workdir / "spec" / "agent_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "latest_watchdog.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _apply_proxy_updates(run_id: int, updates: dict[str, Any]) -> None:
    if not updates:
        return
    row = db.fetchone("SELECT proxy_config, workdir FROM experiment_runs WHERE id=?", (run_id,))
    if not row:
        return
    try:
        proxy = json.loads(row["proxy_config"]) if isinstance(row["proxy_config"], str) else dict(row["proxy_config"] or {})
    except (TypeError, json.JSONDecodeError):
        proxy = {}
    proxy.update(updates)
    db.execute("UPDATE experiment_runs SET proxy_config=? WHERE id=?", (json.dumps(proxy), run_id))
    db.commit()
    workdir = Path(row["workdir"] or "")
    spec_path = workdir / "spec" / "proxy_config.json"
    if workdir.is_dir():
        try:
            spec_path.write_text(json.dumps(proxy, indent=2), encoding="utf-8")
        except OSError:
            pass


def _finalize_partial_metrics(run_id: int, workdir: Path, summary: dict[str, Any]) -> float | None:
    results_dir = workdir / "results"
    summary_path = results_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metric_name, _candidate_method, candidate_value, _baseline, _n = benchmark_scores(summary)
    if candidate_value is not None:
        db.execute(
            """UPDATE experiment_runs
               SET baseline_metric_value=COALESCE(baseline_metric_value, ?),
                   best_metric_value=?,
                   error_message=NULL
               WHERE id=?""",
            (candidate_value, candidate_value, run_id),
        )
        db.commit()
    return candidate_value


def run_watchdog_for_run(run_id: int) -> dict[str, Any]:
    """Run progress + feasibility + tuning agents for one experiment run."""
    if not EXPERIMENT_WATCHDOG_ENABLED:
        return {"enabled": False, "run_id": run_id}

    row = db.fetchone(
        """SELECT er.*, di.title AS insight_title, di.resource_class AS insight_resource_class
           FROM experiment_runs er
           JOIN deep_insights di ON di.id = er.deep_insight_id
           WHERE er.id=?""",
        (run_id,),
    )
    if not row:
        return {"error": "run not found", "run_id": run_id}

    workdir = Path(row["workdir"] or "")
    insight = {
        "id": row["deep_insight_id"],
        "title": row.get("insight_title"),
        "resource_class": row.get("insight_resource_class") or row.get("resource_class"),
    }
    run = dict(row)

    prev = _LAST_LINE_CACHE.get(run_id)
    prev_lines, prev_at = (prev if prev else (None, None))
    progress: ProgressReport = inspect_run(run_id, previous_lines=prev_lines, previous_at=prev_at)
    _LAST_LINE_CACHE[run_id] = (progress.predictions_lines, time.time())

    run_config_path = workdir / "results" / "run_config.json"
    candidate_method = None
    metric_name = "primary_score"
    if run_config_path.exists():
        try:
            cfg = json.loads(run_config_path.read_text(encoding="utf-8"))
            candidate_method = cfg.get("method") or cfg.get("method_name")
            metric_name = cfg.get("metric_name") or metric_name
        except (json.JSONDecodeError, OSError):
            pass

    partial_summary = build_benchmark_summary_from_predictions(
        workdir / "results",
        candidate_method=candidate_method,
        metric_name=metric_name,
        min_lines=50,
    )

    feasibility = assess_feasibility(
        insight=insight,
        run=run,
        progress=progress,
        partial_summary=partial_summary or None,
    )

    proxy = {}
    if row.get("proxy_config"):
        try:
            proxy = json.loads(row["proxy_config"]) if isinstance(row["proxy_config"], str) else dict(row["proxy_config"])
        except (TypeError, json.JSONDecodeError):
            proxy = {}

    tuning = recommend_tuning(
        insight=insight,
        run=run,
        progress=progress,
        feasibility=feasibility,
        partial_summary=partial_summary or None,
        proxy=proxy,
    )

    applied: dict[str, Any] = {}
    if tuning.action == "finalize_partial" and partial_summary:
        metric = _finalize_partial_metrics(run_id, workdir, partial_summary)
        applied["partial_metric"] = metric
        applied["benchmark_summary_written"] = True
    elif tuning.action in {"scale_down", "scale_up_time"} and tuning.parameter_updates:
        _apply_proxy_updates(run_id, tuning.parameter_updates)
        applied["proxy_updates"] = tuning.parameter_updates
    elif tuning.action == "route_gpu":
        routed = apply_route_gpu(
            insight_id=int(row["deep_insight_id"]),
            run_id=run_id,
            resource_class=row.get("resource_class") or insight.get("resource_class"),
        )
        applied["route_gpu"] = routed

    payload = {
        "schema_version": "experiment_watchdog_v1",
        "run_id": run_id,
        "deep_insight_id": row["deep_insight_id"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "progress": progress.to_dict(),
        "feasibility": feasibility.to_dict(),
        "tuning": tuning.to_dict(),
        "partial_summary_lines": partial_summary.get("prediction_lines") if partial_summary else 0,
        "applied": applied,
    }
    if workdir.is_dir():
        _write_report(workdir, payload)

    note = (
        f"[Watchdog] progress={progress.status} feasibility={feasibility.verdict} "
        f"action={tuning.action} preds={progress.predictions_lines}"
    )
    db.execute(
        """UPDATE auto_research_jobs
           SET last_note=?, last_checked_at=CURRENT_TIMESTAMP
           WHERE experiment_run_id=?""",
        (note[:500], run_id),
    )
    db.commit()

    return payload


def run_watchdog_for_active_runs(limit: int = 10) -> list[dict[str, Any]]:
    """Scan all in-flight experiment runs."""
    if not EXPERIMENT_WATCHDOG_ENABLED:
        return []
    rows = db.fetchall(
        """SELECT id FROM experiment_runs
           WHERE status IN ('reproducing', 'testing', 'running', 'scaffolding')
           ORDER BY id DESC
           LIMIT ?""",
        (limit,),
    )
    return [run_watchdog_for_run(int(row["id"])) for row in rows]


def promote_pending_jobs(limit: int = 20) -> int:
    """Move stale pending auto_research jobs into the schedulable queue."""
    rows = db.fetchall(
        """SELECT deep_insight_id FROM auto_research_jobs
           WHERE status='pending'
           ORDER BY deep_insight_id
           LIMIT ?""",
        (limit,),
    )
    promoted = 0
    for row in rows:
        iid = int(row["deep_insight_id"])
        db.execute(
            """UPDATE auto_research_jobs
               SET status='queued', stage='watchdog_promoted', last_note='Promoted from pending by experiment watchdog'
               WHERE deep_insight_id=? AND status='pending'""",
            (iid,),
        )
        promoted += 1
    if promoted:
        db.commit()
    return promoted
