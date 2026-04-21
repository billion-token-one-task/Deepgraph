"""Single-host GPU scheduler and artifact collector for DeepGraph."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from agents.knowledge_loop import process_completed_run
from agents.manuscript_pipeline import generate_submission_bundle
from agents.validation_loop import run_validation_loop
from config import (
    GPU_MODE,
    GPU_DEFAULT_MODEL,
    GPU_DEFAULT_VRAM_GB,
    GPU_JOB_TIMEOUT_SECONDS,
    GPU_POLL_SECONDS,
    GPU_REMOTE_BASE_DIR,
    GPU_REMOTE_PYTHON,
    GPU_REMOTE_SSH_HOST,
    GPU_REMOTE_SSH_PASSWORD,
    GPU_REMOTE_SSH_PORT,
    GPU_REMOTE_SSH_USER,
    GPU_VISIBLE_DEVICES,
)
from db import database as db
from orchestrator.tracking import log_artifact, log_metrics, tracked_run

_scheduler_thread: threading.Thread | None = None
_scheduler_lock = threading.Lock()
_stop_event = threading.Event()
GPU_SCHEDULER_CONSUMER = "gpu_scheduler"


def register_default_workers() -> list[dict]:
    db.init_db()
    if GPU_MODE == "ssh":
        if not GPU_REMOTE_SSH_HOST or not GPU_REMOTE_SSH_USER:
            raise RuntimeError(
                "DEEPGRAPH_GPU_MODE=ssh requires DEEPGRAPH_GPU_REMOTE_SSH_HOST and DEEPGRAPH_GPU_REMOTE_SSH_USER."
            )
        workers = []
        for idx, gpu_id in enumerate(GPU_VISIBLE_DEVICES):
            worker_id = f"ssh:{GPU_REMOTE_SSH_HOST}:gpu{gpu_id}"
            metadata = {
                "backend": "ssh",
                "visible_device": gpu_id,
                "ssh_host": GPU_REMOTE_SSH_HOST,
                "ssh_port": GPU_REMOTE_SSH_PORT,
                "ssh_user": GPU_REMOTE_SSH_USER,
                "ssh_password": GPU_REMOTE_SSH_PASSWORD,
                "remote_base_dir": GPU_REMOTE_BASE_DIR,
                "python_bin": GPU_REMOTE_PYTHON,
            }
            existing = db.fetchone("SELECT id FROM gpu_workers WHERE id=?", (worker_id,))
            payload = (
                worker_id,
                GPU_REMOTE_SSH_HOST,
                idx,
                GPU_DEFAULT_MODEL,
                float(GPU_DEFAULT_VRAM_GB),
                "idle",
                json.dumps(metadata),
            )
            if existing:
                db.execute(
                    """UPDATE gpu_workers
                       SET hostname=?, gpu_index=?, gpu_model=?, total_mem_gb=?,
                           status=?, heartbeat_at=CURRENT_TIMESTAMP, metadata=?
                       WHERE id=?""",
                    (
                        GPU_REMOTE_SSH_HOST,
                        idx,
                        GPU_DEFAULT_MODEL,
                        float(GPU_DEFAULT_VRAM_GB),
                        "idle",
                        json.dumps(metadata),
                        worker_id,
                    ),
                )
            else:
                db.execute(
                    """INSERT INTO gpu_workers
                       (id, hostname, gpu_index, gpu_model, total_mem_gb, status, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    payload,
                )
            workers.append(
                {
                    "id": worker_id,
                    "hostname": GPU_REMOTE_SSH_HOST,
                    "gpu_index": idx,
                    "gpu_model": GPU_DEFAULT_MODEL,
                    "total_mem_gb": float(GPU_DEFAULT_VRAM_GB),
                    "status": "idle",
                    **metadata,
                }
            )
        db.commit()
        return workers

    hostname = os.uname().nodename
    workers = []
    for idx, gpu_id in enumerate(GPU_VISIBLE_DEVICES):
        worker_id = f"{hostname}:gpu{gpu_id}"
        existing = db.fetchone("SELECT id FROM gpu_workers WHERE id=?", (worker_id,))
        payload = (
            worker_id,
            hostname,
            idx,
            GPU_DEFAULT_MODEL,
            float(GPU_DEFAULT_VRAM_GB),
            "idle",
            json.dumps({"visible_device": gpu_id}),
        )
        if existing:
            db.execute(
                """UPDATE gpu_workers
                   SET hostname=?, gpu_index=?, gpu_model=?, total_mem_gb=?,
                       status=?, heartbeat_at=CURRENT_TIMESTAMP, metadata=?
                   WHERE id=?""",
                (hostname, idx, GPU_DEFAULT_MODEL, float(GPU_DEFAULT_VRAM_GB), "idle", json.dumps({"visible_device": gpu_id}), worker_id),
            )
        else:
            db.execute(
                """INSERT INTO gpu_workers
                   (id, hostname, gpu_index, gpu_model, total_mem_gb, status, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                payload,
            )
        workers.append(
            {
                "id": worker_id,
                "hostname": hostname,
                "gpu_index": idx,
                "gpu_model": GPU_DEFAULT_MODEL,
                "total_mem_gb": float(GPU_DEFAULT_VRAM_GB),
                "status": "idle",
                "visible_device": gpu_id,
            }
        )
    db.commit()
    return workers


def list_workers() -> list[dict]:
    db.init_db()
    return db.fetchall("SELECT * FROM gpu_workers ORDER BY gpu_index, id")


def list_jobs(limit: int = 100) -> list[dict]:
    db.init_db()
    return db.fetchall(
        """
        SELECT gj.*, di.title AS insight_title
        FROM gpu_jobs gj
        LEFT JOIN deep_insights di ON di.id = gj.deep_insight_id
        ORDER BY gj.created_at DESC
        LIMIT ?
        """,
        (limit,),
    )


def get_status() -> dict:
    db.init_db()
    with _scheduler_lock:
        running = bool(_scheduler_thread and _scheduler_thread.is_alive())
    counts = db.fetchone(
        """
        SELECT
          COUNT(*) AS total_jobs,
          SUM(CASE WHEN status='queued' THEN 1 ELSE 0 END) AS queued_jobs,
          SUM(CASE WHEN status='running' THEN 1 ELSE 0 END) AS running_jobs,
          SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) AS completed_jobs,
          SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed_jobs
        FROM gpu_jobs
        """
    ) or {}
    return {
        "running": running,
        "workers": list_workers(),
        **counts,
    }


def queue_run(
    *,
    insight_id: int,
    run_id: int,
    resource_class: str,
    priority: int = 0,
    gpu_count: int = 1,
    vram_required_gb: float = 0,
    timeout_s: int | None = None,
) -> int:
    db.init_db()
    jid = db.insert_returning_id(
        """
        INSERT INTO gpu_jobs
        (deep_insight_id, experiment_run_id, resource_class, gpu_count, vram_required_gb, timeout_s, priority, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'queued')
        RETURNING id
        """,
        (
            insight_id,
            run_id,
            resource_class,
            gpu_count,
            vram_required_gb,
            timeout_s or GPU_JOB_TIMEOUT_SECONDS,
            priority,
        ),
    )
    db.commit()
    db.emit_pipeline_event(
        "gpu_job_queued",
        {"gpu_job_id": jid, "experiment_run_id": run_id, "deep_insight_id": insight_id, "resource_class": resource_class},
        entity_type="gpu_job",
        entity_id=str(jid),
        dedupe_key=f"gpu_job_queued:{jid}",
    )
    return jid


def collect_run_artifacts(run_id: int) -> list[dict]:
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not run or not run.get("workdir"):
        return []
    workdir = Path(run["workdir"])
    if not workdir.exists():
        return []
    artifacts = []
    spec_dir = workdir / "spec"
    candidate_files = [
        (workdir / "run.log", "log"),
        (spec_dir / "evaluate.py", "source_data"),
        (spec_dir / "program.md", "source_data"),
        (spec_dir / "success_criteria.json", "metric"),
        (spec_dir / "proxy_config.json", "source_data"),
        (spec_dir / "experiment_spec.json", "source_data"),
        (spec_dir / "experiment_judgement.json", "source_data"),
        (spec_dir / "evidence_plan.json", "source_data"),
        (workdir / "evaluate.py", "source_data"),
        (workdir / "program.md", "source_data"),
        (workdir / "success_criteria.json", "metric"),
        (workdir / "proxy_config.json", "source_data"),
    ]
    codex_dir = workdir / "codex" / "runs"
    if codex_dir.exists():
        for path in sorted(codex_dir.glob("*")):
            candidate_files.append((path, "source_data"))
    plot_dir = workdir / "submission_bundle" / "figures"
    if plot_dir.exists():
        for path in sorted(plot_dir.glob("*")):
            candidate_files.append((path, "plot"))

    for path, artifact_type in candidate_files:
        if not path.exists():
            continue
        db.execute(
            """
            INSERT INTO experiment_artifacts (run_id, artifact_type, path)
            VALUES (?, ?, ?)
            """,
            (run_id, artifact_type, str(path)),
        )
        artifacts.append({"artifact_type": artifact_type, "path": str(path)})
    db.commit()
    return artifacts


def _worker_filter_sql() -> tuple[str, tuple]:
    if GPU_MODE == "ssh":
        return " AND metadata LIKE ?", ('%"backend": "ssh"%',)
    return " AND (metadata IS NULL OR metadata NOT LIKE ?)", ('%"backend": "ssh"%',)


def _claim_idle_worker() -> dict | None:
    register_default_workers()
    filter_sql, params = _worker_filter_sql()
    workers = db.fetchall(
        f"SELECT * FROM gpu_workers WHERE status='idle'{filter_sql} ORDER BY gpu_index, id LIMIT 1",
        params,
    )
    return workers[0] if workers else None


def _next_job() -> dict | None:
    rows = db.fetchall(
        """
        SELECT * FROM gpu_jobs
        WHERE status='queued'
        ORDER BY priority DESC, created_at ASC
        LIMIT 1
        """
    )
    return rows[0] if rows else None


def _run_job(job: dict, worker: dict) -> None:
    job_id = job["id"]
    run_id = job["experiment_run_id"]
    insight_id = job["deep_insight_id"]
    worker_id = worker["id"]

    db.execute(
        "UPDATE gpu_workers SET status='busy', heartbeat_at=CURRENT_TIMESTAMP WHERE id=?",
        (worker_id,),
    )
    db.execute(
        """
        UPDATE gpu_jobs
        SET status='running', assigned_worker=?, started_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (worker_id, job_id),
    )
    db.execute(
        """
        UPDATE experiment_runs
        SET status='running_gpu', resource_class=?
        WHERE id=?
        """,
        (job.get("resource_class", "gpu_small"), run_id),
    )
    db.execute(
        """
        UPDATE auto_research_jobs
        SET status='running_gpu', stage='gpu_scheduler', assigned_worker=?
        WHERE deep_insight_id=?
        """,
        (worker_id, insight_id),
    )
    db.commit()

    try:
        with tracked_run(
            f"deepgraph-gpu-run-{run_id}",
            tags={"insight_id": insight_id, "resource_class": job.get("resource_class", "gpu_small")},
        ):
            result = run_validation_loop(run_id, execution_context={"worker": worker, "job": job})
            process_completed_run(run_id)
            collect_run_artifacts(run_id)
            bundle = generate_submission_bundle(run_id)
            log_metrics(
                {
                    "effect_pct": db.fetchone("SELECT effect_pct FROM experiment_runs WHERE id=?", (run_id,)).get("effect_pct"),
                }
            )
            for artifact in db.fetchall("SELECT path FROM experiment_artifacts WHERE run_id=?", (run_id,)):
                log_artifact(artifact["path"])
        db.execute(
            """
            UPDATE gpu_jobs
            SET status='completed', completed_at=CURRENT_TIMESTAMP, artifact_uri=?
            WHERE id=?
            """,
            (db.fetchone("SELECT workdir FROM experiment_runs WHERE id=?", (run_id,)).get("workdir"), job_id),
        )
        db.execute(
            """
            UPDATE auto_research_jobs
            SET status=?, stage=?, artifact_bundle_id=?, last_note=?
            WHERE deep_insight_id=?
            """,
            (
                "bundle_ready" if "error" not in bundle else "completed",
                "writing_submission" if "error" not in bundle else "closed_loop_complete",
                (bundle.get("bundle_ids") or [None])[-1],
                f"GPU run completed with verdict={result.get('verdict', 'unknown')}. Submission bundle status={'ok' if 'error' not in bundle else 'failed'}.",
                insight_id,
            ),
        )
        db.commit()
        db.emit_pipeline_event(
            "gpu_job_completed",
            {"gpu_job_id": job_id, "experiment_run_id": run_id, "deep_insight_id": insight_id},
            entity_type="gpu_job",
            entity_id=str(job_id),
            dedupe_key=f"gpu_job_completed:{job_id}",
        )
        db.emit_pipeline_event(
            "experiment_run_completed",
            {"experiment_run_id": run_id, "deep_insight_id": insight_id},
            entity_type="experiment_run",
            entity_id=str(run_id),
            dedupe_key=f"experiment_run_completed:{run_id}",
        )
    except Exception as exc:  # pragma: no cover - background guard
        db.execute(
            "UPDATE gpu_jobs SET status='failed', error_message=?, completed_at=CURRENT_TIMESTAMP WHERE id=?",
            (str(exc), job_id),
        )
        db.execute(
            "UPDATE experiment_runs SET status='failed', error_message=? WHERE id=?",
            (str(exc), run_id),
        )
        db.execute(
            "UPDATE auto_research_jobs SET status='failed', stage='gpu_failed', last_error=? WHERE deep_insight_id=?",
            (str(exc), insight_id),
        )
        db.commit()
        db.emit_pipeline_event(
            "gpu_job_failed",
            {"gpu_job_id": job_id, "experiment_run_id": run_id, "deep_insight_id": insight_id, "error": str(exc)},
            entity_type="gpu_job",
            entity_id=str(job_id),
            dedupe_key=f"gpu_job_failed:{job_id}",
        )
    finally:
        db.execute(
            "UPDATE gpu_workers SET status='idle', heartbeat_at=CURRENT_TIMESTAMP WHERE id=?",
            (worker_id,),
        )
        db.commit()


def consume_pipeline_events_once(limit: int = 50) -> dict:
    db.init_db()
    events = db.fetch_pipeline_events(
        GPU_SCHEDULER_CONSUMER,
        limit=limit,
        event_types=["gpu_job_queued"],
    )
    processed = 0
    last_event_id = 0
    for event in events:
        last_event_id = int(event["id"])
        worker = _claim_idle_worker()
        job = _next_job()
        if worker and job:
            thread = threading.Thread(target=_run_job, args=(job, worker), daemon=True)
            thread.start()
            processed += 1
    if last_event_id:
        db.ack_pipeline_events(GPU_SCHEDULER_CONSUMER, last_event_id)
    return {"events": len(events), "started_jobs": processed}


def _loop() -> None:
    while not _stop_event.is_set():
        stats = consume_pipeline_events_once(limit=50)
        if not stats.get("events"):
            worker = _claim_idle_worker()
            job = _next_job()
            if worker and job:
                thread = threading.Thread(target=_run_job, args=(job, worker), daemon=True)
                thread.start()
            _stop_event.wait(max(1, GPU_POLL_SECONDS))


def start() -> dict:
    global _scheduler_thread
    db.init_db()
    register_default_workers()
    with _scheduler_lock:
        if _scheduler_thread and _scheduler_thread.is_alive():
            return {"status": "already_running"}
        _stop_event.clear()
        _scheduler_thread = threading.Thread(target=_loop, daemon=True, name="deepgraph-gpu-scheduler")
        _scheduler_thread.start()
    return {"status": "started", "workers": list_workers()}


def stop() -> dict:
    _stop_event.set()
    return {"status": "stopping"}
