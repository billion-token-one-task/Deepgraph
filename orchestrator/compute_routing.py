"""Resolve CPU vs GPU scheduler lanes for experiment execution."""

from __future__ import annotations

import json
from typing import Any

from db import database as db
from orchestrator import gpu_scheduler


def _load_proxy(run: dict | None) -> dict[str, Any]:
    if not run:
        return {}
    raw = run.get("proxy_config")
    if not raw:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}


def _needs_gpu_lane(proxy: dict[str, Any], resource_class: str) -> bool:
    if resource_class in {"gpu_small", "gpu_large"}:
        return True
    if proxy.get("real_benchmark_required") or proxy.get("benchmark_model"):
        return True
    if proxy.get("required_real_benchmarks") or proxy.get("benchmark_targets"):
        return True
    return False


def resolve_execution_lane(
    *,
    resource_class: str,
    run: dict | None = None,
) -> tuple[str, str, str]:
    """Return (resource_class, lane, note) where lane is ``gpu`` or ``cpu_async``."""
    proxy = _load_proxy(run)
    note = ""
    rc = (resource_class or "cpu").strip() or "cpu"
    if rc == "cpu" and _needs_gpu_lane(proxy, rc):
        rc = "gpu_small"
        note = "Real-benchmark workload upgraded from cpu to gpu_small for gpu_scheduler."
    lane = "gpu" if rc in {"gpu_small", "gpu_large"} else "cpu_async"
    return rc, lane, note


def apply_route_gpu(
    *,
    insight_id: int,
    run_id: int,
    resource_class: str | None = None,
) -> dict[str, Any]:
    """Enqueue ``run_id`` on gpu_scheduler and sync ARJ / experiment_runs rows."""
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not run:
        return {"ok": False, "error": "run not found"}

    rc, _lane, note = resolve_execution_lane(
        resource_class=(resource_class or run.get("resource_class") or "gpu_small"),
        run=run,
    )
    gpu_scheduler.start()
    queued_job = db.fetchone(
        """
        SELECT * FROM gpu_jobs
        WHERE experiment_run_id=? AND status IN ('queued', 'running')
        ORDER BY id DESC LIMIT 1
        """,
        (run_id,),
    )
    if queued_job:
        job_id = int(queued_job["id"])
        job_note = f"GPU job {job_id} already {queued_job['status']}."
    else:
        job_id = gpu_scheduler.queue_run(
            insight_id=insight_id,
            run_id=run_id,
            resource_class=rc,
            priority=2 if rc == "gpu_large" else 1,
            vram_required_gb=40 if rc == "gpu_large" else 16,
        )
        job_note = f"Watchdog routed to GPU scheduler as job {job_id}."

    db.execute(
        "UPDATE experiment_runs SET resource_class=?, status='running_gpu' WHERE id=?",
        (rc, run_id),
    )
    db.execute(
        """
        UPDATE deep_insights
        SET resource_class=?, updated_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (rc, insight_id),
    )
    db.execute(
        """
        UPDATE auto_research_jobs
        SET status='queued_gpu', stage='gpu_scheduler', resource_class=?,
            last_note=?, last_error=NULL, updated_at=CURRENT_TIMESTAMP
        WHERE deep_insight_id=?
        """,
        (rc, job_note if not note else f"{job_note} {note}", insight_id),
    )
    db.commit()
    return {"ok": True, "gpu_job_id": job_id, "resource_class": rc, "note": job_note}
