"""Single-host GPU scheduler and artifact collector for DeepGraph."""

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import threading
import time
from pathlib import Path

from agents.knowledge_loop import process_completed_run
from agents.manuscript_pipeline import generate_submission_bundle
from agents.validation_loop import run_full_benchmark_completion, run_validation_loop
from compat.filelock import FileLock
from config import (
    GPU_MODE,
    GPU_DEFAULT_MODEL,
    GPU_DEFAULT_VRAM_GB,
    GPU_JOB_TIMEOUT_SECONDS,
    GPU_POLL_SECONDS,
    GPU_STALE_RECOVERY_POLL_SECONDS,
    GPU_REMOTE_BASE_DIR,
    GPU_REMOTE_PYTHON,
    GPU_REMOTE_SSH_HOST,
    GPU_REMOTE_SSH_PASSWORD,
    GPU_REMOTE_SSH_PORT,
    GPU_REMOTE_SSH_USER,
    GPU_VISIBLE_DEVICES,
)
from db import database as db
from orchestrator import ssh_gpu_backend
from orchestrator.benchmark_completion import BENCHMARK_COMPLETION_STAGE, schedule_benchmark_completion
from orchestrator.tracking import log_artifact, log_metrics, tracked_run

_scheduler_thread: threading.Thread | None = None
_scheduler_lock = threading.Lock()
_stop_event = threading.Event()
_process_lock: FileLock | None = None
_last_recovery_check = 0.0
GPU_SCHEDULER_CONSUMER = "gpu_scheduler"
# Serialize claim-worker + pick-job + thread.start to avoid two jobs racing the same idle worker.
_job_dispatch_lock = threading.Lock()
_active_job_lock = threading.Lock()
_active_job_ids: set[int] = set()


def _local_hostname() -> str:
    if hasattr(os, "uname"):
        try:
            return os.uname().nodename  # type: ignore[attr-defined]
        except Exception:
            pass
    return socket.gethostname()


def _mark_job_active(job_id: int) -> None:
    with _active_job_lock:
        _active_job_ids.add(int(job_id))


def _mark_job_inactive(job_id: int) -> None:
    with _active_job_lock:
        _active_job_ids.discard(int(job_id))


def _job_is_active_in_this_process(job_id: int) -> bool:
    with _active_job_lock:
        return int(job_id) in _active_job_ids


def _try_start_next_gpu_job() -> bool:
    with _job_dispatch_lock:
        job = _next_job()
        worker = _claim_idle_worker(job)
        if not worker or not job:
            return False
        thread = threading.Thread(target=_run_job, args=(job, worker), daemon=True)
        thread.start()
        return True


def _local_worker_ids(workers: list[dict] | None = None) -> list[str]:
    source = workers if workers is not None else list_workers()
    ids: list[str] = []
    for worker in source:
        worker_id = str(worker.get("id") or "")
        if not worker_id or worker_id.startswith("ssh:"):
            continue
        metadata = {}
        raw = worker.get("metadata")
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                parsed = {}
            if isinstance(parsed, dict):
                metadata = parsed
        if metadata.get("backend", "local") == "ssh":
            continue
        ids.append(worker_id)
    return ids


def recover_stale_local_running_jobs(workers: list[dict] | None = None) -> int:
    """Requeue local GPU jobs left running by a controller restart.

    Background worker threads are in-process, so after a fresh scheduler start
    any local ``gpu_jobs.status='running'`` row owned by this host has no live
    Python thread behind it. Requeue it and let ``run_validation_loop`` resume
    from saved iteration state.
    """
    local_ids = _local_worker_ids(workers)
    if not local_ids:
        return 0
    placeholders = ", ".join("?" for _ in local_ids)
    stale_jobs = db.fetchall(
        f"""
        SELECT gj.*, er.status AS run_status, er.hypothesis_verdict
        FROM gpu_jobs gj
        LEFT JOIN experiment_runs er ON er.id = gj.experiment_run_id
        WHERE gj.status='running'
          AND gj.assigned_worker IN ({placeholders})
        """,
        tuple(local_ids),
    )
    recovered = 0
    for job in stale_jobs:
        job_id = job["id"]
        run_id = job["experiment_run_id"]
        insight_id = job["deep_insight_id"]
        if _current_run_is_successful(run_id):
            db.execute(
                """
                UPDATE gpu_jobs
                SET status='completed', completed_at=CURRENT_TIMESTAMP,
                    error_message=COALESCE(error_message, ?)
                WHERE id=?
                """,
                ("Recovered completed run after scheduler restart.", job_id),
            )
            db.execute(
                """
                UPDATE auto_research_jobs
                SET status='completed', stage='closed_loop_complete',
                    assigned_worker=NULL,
                    last_note=?, updated_at=CURRENT_TIMESTAMP, last_checked_at=CURRENT_TIMESTAMP
                WHERE deep_insight_id=?
                """,
                (f"Recovered completed GPU run {run_id} after scheduler restart.", insight_id),
            )
        else:
            db.execute(
                """
                UPDATE gpu_jobs
                SET status='queued', assigned_worker=NULL, started_at=NULL,
                    completed_at=NULL, error_message=?
                WHERE id=?
                """,
                (
                    "Recovered stale local running job after scheduler restart; "
                    "validation will resume from saved run state.",
                    job_id,
                ),
            )
            db.execute(
                """
                UPDATE auto_research_jobs
                SET status='queued_gpu', stage='gpu_scheduler',
                    assigned_worker=NULL, experiment_run_id=?,
                    last_note=?, last_error=NULL,
                    updated_at=CURRENT_TIMESTAMP, last_checked_at=CURRENT_TIMESTAMP
                WHERE deep_insight_id=?
                """,
                (
                    run_id,
                    f"Recovered stale GPU job {job_id}; queued it for automatic resume.",
                    insight_id,
                ),
            )
        if job.get("assigned_worker"):
            db.execute(
                "UPDATE gpu_workers SET status='idle', heartbeat_at=CURRENT_TIMESTAMP WHERE id=?",
                (job["assigned_worker"],),
            )
        recovered += 1
    if recovered:
        db.commit()
    return recovered


def _ssh_run_has_live_process(worker: dict, run_id: int) -> bool:
    metadata = {}
    raw_metadata = worker.get("metadata")
    if raw_metadata:
        try:
            parsed = json.loads(raw_metadata)
        except (json.JSONDecodeError, TypeError):
            parsed = {}
        if isinstance(parsed, dict):
            metadata = parsed
    remote_base = str(metadata.get("remote_base_dir") or GPU_REMOTE_BASE_DIR).rstrip("/")
    remote_run = f"{remote_base}/runs/run_{run_id}"
    cmd = "\n".join(
        [
            f"remote_run={shlex.quote(remote_run)}",
            "for pid in $(pgrep -f 'deepgraph_exec_run_|train.py|eval_cggr.py' || true); do",
            "  cwd=$(readlink /proc/$pid/cwd 2>/dev/null || true)",
            "  args=$(tr '\\0' ' ' < /proc/$pid/cmdline 2>/dev/null || true)",
            "  if printf '%s\\n%s\\n' \"$cwd\" \"$args\" | grep -F \"$remote_run\" >/dev/null 2>&1; then",
            "    echo $pid",
            "    exit 0",
            "  fi",
            "done",
        ]
    )
    try:
        proc = ssh_gpu_backend._run_ssh(worker, cmd, timeout=20)
    except Exception:
        return True
    return bool((proc.stdout or "").strip())


def recover_stale_ssh_running_jobs() -> int:
    """Recover SSH jobs whose controller died after the remote process exited."""
    stale_jobs = db.fetchall(
        """
        SELECT gj.*, gw.metadata AS worker_metadata, gw.hostname, gw.gpu_index,
               gw.gpu_model, gw.total_mem_gb
        FROM gpu_jobs gj
        JOIN gpu_workers gw ON gw.id = gj.assigned_worker
        WHERE gj.status='running'
          AND gw.metadata LIKE ?
        """,
        ('%"backend": "ssh"%',),
    )
    recovered = 0
    for job in stale_jobs:
        if _job_is_active_in_this_process(int(job["id"])):
            continue
        worker = {
            "id": job["assigned_worker"],
            "hostname": job.get("hostname"),
            "gpu_index": job.get("gpu_index"),
            "gpu_model": job.get("gpu_model"),
            "total_mem_gb": job.get("total_mem_gb"),
            "metadata": job.get("worker_metadata"),
        }
        run_id = int(job["experiment_run_id"])
        if _ssh_run_has_live_process(worker, run_id):
            continue
        job_id = int(job["id"])
        insight_id = int(job["deep_insight_id"])
        message = (
            "Recovered stale SSH GPU job: no remote process was found for "
            f"run_{run_id}; queued a fresh automatic retry."
        )
        if _current_run_is_successful(run_id):
            db.execute(
                """
                UPDATE gpu_jobs
                SET status='completed', completed_at=CURRENT_TIMESTAMP,
                    error_message=COALESCE(error_message, ?)
                WHERE id=?
                """,
                (message, job_id),
            )
            db.execute(
                """
                UPDATE auto_research_jobs
                SET status='completed', stage='closed_loop_complete',
                    assigned_worker=NULL,
                    last_note=?, updated_at=CURRENT_TIMESTAMP, last_checked_at=CURRENT_TIMESTAMP
                WHERE deep_insight_id=?
                """,
                (message, insight_id),
            )
        else:
            db.execute(
                """
                UPDATE gpu_jobs
                SET status='failed', completed_at=CURRENT_TIMESTAMP,
                    error_message=?
                WHERE id=?
                """,
                (message, job_id),
            )
            db.execute(
                """
                UPDATE experiment_runs
                SET status='failed', error_message=?, completed_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (message, run_id),
            )
            db.execute(
                """
                UPDATE auto_research_jobs
                SET status='queued', stage='retry_failed_run',
                    assigned_worker=NULL, experiment_run_id=?,
                    last_note=?, last_error=NULL,
                    updated_at=CURRENT_TIMESTAMP, last_checked_at=CURRENT_TIMESTAMP
                WHERE deep_insight_id=?
                """,
                (run_id, message, insight_id),
            )
        if job.get("assigned_worker"):
            db.execute(
                "UPDATE gpu_workers SET status='idle', heartbeat_at=CURRENT_TIMESTAMP WHERE id=?",
                (job["assigned_worker"],),
            )
        recovered += 1
    if recovered:
        db.commit()
    return recovered


def _local_gpu_inventory() -> dict[str, dict]:
    """Return nvidia-smi GPU inventory keyed by visible device index."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if proc.returncode != 0:
        return {}
    inventory: dict[str, dict] = {}
    for raw in proc.stdout.splitlines():
        parts = [part.strip() for part in raw.split(",", 2)]
        if len(parts) != 3 or not parts[0]:
            continue
        try:
            total_mem_gb = round(float(parts[2]) / 1024, 2)
        except ValueError:
            total_mem_gb = float(GPU_DEFAULT_VRAM_GB)
        inventory[parts[0]] = {
            "gpu_model": parts[1] or GPU_DEFAULT_MODEL,
            "total_mem_gb": total_mem_gb,
        }
    return inventory


def _configured_local_devices(inventory: dict[str, dict]) -> list[str]:
    if os.getenv("DEEPGRAPH_GPU_VISIBLE_DEVICES"):
        return list(GPU_VISIBLE_DEVICES)
    if inventory:
        return sorted(inventory.keys(), key=lambda value: (0, int(value)) if value.isdigit() else (1, value))
    return list(GPU_VISIBLE_DEVICES)


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
                           status=CASE WHEN status='busy' THEN status ELSE ? END,
                           heartbeat_at=CURRENT_TIMESTAMP, metadata=?
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

    hostname = _local_hostname()
    inventory = _local_gpu_inventory()
    visible_devices = _configured_local_devices(inventory)
    workers = []
    active_worker_ids = []
    for idx, gpu_id in enumerate(visible_devices):
        worker_id = f"{hostname}:gpu{gpu_id}"
        active_worker_ids.append(worker_id)
        gpu_info = inventory.get(str(gpu_id), {})
        gpu_model = gpu_info.get("gpu_model", GPU_DEFAULT_MODEL)
        total_mem_gb = float(gpu_info.get("total_mem_gb", GPU_DEFAULT_VRAM_GB))
        existing = db.fetchone("SELECT id FROM gpu_workers WHERE id=?", (worker_id,))
        metadata = json.dumps({"visible_device": gpu_id, "backend": "local"})
        payload = (
            worker_id,
            hostname,
            idx,
            gpu_model,
            total_mem_gb,
            "idle",
            metadata,
        )
        if existing:
            db.execute(
                """UPDATE gpu_workers
                   SET hostname=?, gpu_index=?, gpu_model=?, total_mem_gb=?,
                       status=CASE WHEN status='busy' THEN status ELSE ? END,
                       heartbeat_at=CURRENT_TIMESTAMP, metadata=?
                   WHERE id=?""",
                (hostname, idx, gpu_model, total_mem_gb, "idle", metadata, worker_id),
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
                "gpu_model": gpu_model,
                "total_mem_gb": total_mem_gb,
                "status": "idle",
                "visible_device": gpu_id,
            }
        )
    if active_worker_ids:
        placeholders = ", ".join("?" for _ in active_worker_ids)
        db.execute(
            f"""UPDATE gpu_workers
                SET status='offline', heartbeat_at=CURRENT_TIMESTAMP
                WHERE hostname=?
                  AND id NOT IN ({placeholders})
                  AND (metadata IS NULL OR metadata NOT LIKE ?)""",
            (hostname, *active_worker_ids, '%"backend": "ssh"%'),
        )
    db.commit()
    return workers


def _try_acquire_process_lock() -> bool:
    global _process_lock
    if _process_lock is not None:
        return True
    lock_path = (
        Path(os.environ.get("TEMP", str(Path.home() / ".cache"))) / "deepgraph-gpu-scheduler.lock"
        if os.name == "nt"
        else Path("/tmp/deepgraph-gpu-scheduler.lock")
    )
    lock = FileLock(str(lock_path))
    if not lock.try_acquire():
        return False
    _process_lock = lock
    return True


def _release_process_lock() -> None:
    global _process_lock
    if _process_lock is None:
        return
    try:
        _process_lock.release()
    finally:
        _process_lock = None


_SENSITIVE_METADATA_KEYS = {"ssh_password", "password", "token", "api_key", "secret"}


def _sanitize_worker(row: dict) -> dict:
    worker = dict(row)
    raw = worker.get("metadata")
    if raw:
        try:
            metadata = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            metadata = None
        if isinstance(metadata, dict):
            for key in list(metadata.keys()):
                if key.lower() in _SENSITIVE_METADATA_KEYS:
                    metadata[key] = "***"
            worker["metadata"] = json.dumps(metadata)
    return worker


def list_workers() -> list[dict]:
    db.init_db()
    return [_sanitize_worker(row) for row in db.fetchall("SELECT * FROM gpu_workers ORDER BY gpu_index, id")]


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


def recover_busy_workers_without_running_jobs() -> int:
    cur = db.execute(
        """
        UPDATE gpu_workers
        SET status='idle', heartbeat_at=CURRENT_TIMESTAMP
        WHERE status='busy'
          AND NOT EXISTS (
              SELECT 1 FROM gpu_jobs gj
              WHERE gj.assigned_worker=gpu_workers.id
                AND gj.status='running'
          )
        """
    )
    db.commit()
    return max(0, int(getattr(cur, "rowcount", 0) or 0))


def _claim_idle_worker(job: dict | None = None) -> dict | None:
    recover_busy_workers_without_running_jobs()
    register_default_workers()
    filter_sql, params = _worker_filter_sql()
    if GPU_MODE != "ssh":
        filter_sql += " AND hostname=?"
        params = (*params, _local_hostname())
    required_vram = 0.0
    if job is not None:
        try:
            required_vram = float(job.get("vram_required_gb") or 0)
        except (TypeError, ValueError):
            required_vram = 0.0
    if required_vram > 0:
        filter_sql += " AND COALESCE(total_mem_gb, 0) >= ?"
        params = (*params, required_vram)
    workers = db.fetchall(
        f"""
        SELECT * FROM gpu_workers
        WHERE status='idle'{filter_sql}
          AND NOT EXISTS (
              SELECT 1 FROM gpu_jobs gj
              WHERE gj.assigned_worker=gpu_workers.id
                AND gj.status='running'
          )
        ORDER BY gpu_index, id
        LIMIT 1
        """,
        params,
    )
    return workers[0] if workers else None


def _next_job() -> dict | None:
    rows = db.fetchall(
        """
        SELECT * FROM gpu_jobs
        WHERE status='queued'
        ORDER BY priority DESC, created_at ASC
        LIMIT 20
        """
    )
    for job in rows:
        run = db.fetchone("SELECT status, phase, error_message FROM experiment_runs WHERE id=?", (job["experiment_run_id"],))
        blocker = _launch_blocker_for_run(run)
        if blocker:
            _fail_blocked_queued_job(job, blocker)
            continue
        return job
    return None


def _launch_blocker_for_run(run: dict | None) -> str | None:
    if not run:
        return "experiment_run is missing; refusing to launch queued GPU job"
    phase = str(run.get("phase") or "").strip().lower()
    status = str(run.get("status") or "").strip().lower()
    error = str(run.get("error_message") or "")
    if status == "canceled":
        return "experiment_run is canceled; refusing to launch queued GPU job"
    if phase == "recipe_blocked" or phase.startswith("invalid"):
        return f"experiment_run phase={phase!r} is blocked; refusing to launch queued GPU job"
    invalid_tokens = (
        "must remain blocked",
        "do not relaunch",
        "do not cite",
        "do not merge",
        "invalid benchmark",
    )
    if any(token in error.lower() for token in invalid_tokens):
        return "experiment_run error_message marks it invalid or blocked; refusing to launch queued GPU job"
    return None


def _fail_blocked_queued_job(job: dict, reason: str) -> None:
    job_id = int(job["id"])
    run_id = int(job["experiment_run_id"])
    insight_id = int(job["deep_insight_id"]) if job.get("deep_insight_id") is not None else None
    db.execute(
        """
        UPDATE gpu_jobs
        SET status='failed', assigned_worker=NULL, completed_at=CURRENT_TIMESTAMP,
            error_message=?
        WHERE id=?
        """,
        (reason, job_id),
    )
    db.execute(
        """
        UPDATE experiment_runs
        SET status='failed', error_message=COALESCE(error_message, ?),
            completed_at=COALESCE(completed_at, CURRENT_TIMESTAMP)
        WHERE id=?
        """,
        (reason, run_id),
    )
    if insight_id is not None:
        db.execute(
            """
            UPDATE auto_research_jobs
            SET status='failed', stage='gpu_blocked',
                assigned_worker=NULL, last_error=?, updated_at=CURRENT_TIMESTAMP,
                last_checked_at=CURRENT_TIMESTAMP
            WHERE deep_insight_id=? AND experiment_run_id=?
            """,
            (reason, insight_id, run_id),
        )
    db.commit()


def _append_error(prefix: str, exc: Exception) -> str:
    return f"{prefix}: {exc}"


def _current_run_is_successful(run_id: int) -> bool:
    run = db.fetchone("SELECT status, hypothesis_verdict FROM experiment_runs WHERE id=?", (run_id,))
    if not run:
        return False
    if run.get("status") in {"completed", "bundle_ready"}:
        return True
    return bool(run.get("hypothesis_verdict"))


def _release_worker_if_no_running_jobs(worker_id: str, *, finished_job_id: int | None = None) -> None:
    params: tuple = (worker_id,)
    exclude_sql = ""
    if finished_job_id is not None:
        exclude_sql = " AND id<>?"
        params = (worker_id, finished_job_id)
    active = db.fetchone(
        f"""
        SELECT COUNT(*) AS count
        FROM gpu_jobs
        WHERE assigned_worker=?
          AND status='running'
          {exclude_sql}
        """,
        params,
    )
    next_status = "busy" if active and int(active.get("count") or 0) > 0 else "idle"
    db.execute(
        "UPDATE gpu_workers SET status=?, heartbeat_at=CURRENT_TIMESTAMP WHERE id=?",
        (next_status, worker_id),
    )


def _run_job(job: dict, worker: dict) -> None:
    job_id = job["id"]
    run_id = job["experiment_run_id"]
    insight_id = job["deep_insight_id"]
    worker_id = worker["id"]
    _mark_job_active(int(job_id))
    auto_job = db.fetchone(
        "SELECT stage FROM auto_research_jobs WHERE deep_insight_id=?",
        (insight_id,),
    )
    benchmark_completion_mode = bool(auto_job and auto_job.get("stage") == BENCHMARK_COMPLETION_STAGE)

    db.execute(
        "UPDATE gpu_workers SET status='busy', heartbeat_at=CURRENT_TIMESTAMP WHERE id=?",
        (worker_id,),
    )
    db.execute(
        """
        UPDATE gpu_jobs
        SET status='running', assigned_worker=?, started_at=CURRENT_TIMESTAMP,
            completed_at=NULL, error_message=NULL
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
        SET status='running_gpu',
            stage=CASE WHEN stage=? THEN stage ELSE 'gpu_scheduler' END,
            assigned_worker=?
        WHERE deep_insight_id=?
        """,
        (BENCHMARK_COMPLETION_STAGE, worker_id, insight_id),
    )
    db.commit()

    try:
        post_run_errors: list[str] = []
        bundle: dict = {}
        with tracked_run(
            f"deepgraph-gpu-run-{run_id}",
            tags={"insight_id": insight_id, "resource_class": job.get("resource_class", "gpu_small")},
        ):
            execution_context = {
                "worker": worker,
                "job": job,
                "full_benchmark": benchmark_completion_mode,
            }
            if benchmark_completion_mode:
                result = run_full_benchmark_completion(run_id, execution_context=execution_context)
            else:
                result = run_validation_loop(run_id, execution_context=execution_context)
            if not isinstance(result, dict):
                result = {"verdict": "failed", "error": f"validation loop returned {type(result).__name__}"}
            try:
                process_completed_run(run_id)
            except Exception as exc:
                post_run_errors.append(_append_error("knowledge_loop_failed", exc))
            try:
                collect_run_artifacts(run_id)
            except Exception as exc:
                post_run_errors.append(_append_error("artifact_collection_failed", exc))
            try:
                bundle = generate_submission_bundle(run_id)
            except Exception as exc:
                bundle = {"error": str(exc)}
                post_run_errors.append(_append_error("submission_bundle_failed", exc))
            if "error" in bundle:
                post_run_errors.append(f"submission_bundle_result_error: {bundle['error']}")
            log_metrics(
                {
                    "effect_pct": db.fetchone("SELECT effect_pct FROM experiment_runs WHERE id=?", (run_id,)).get("effect_pct"),
                }
            )
            try:
                for artifact in db.fetchall("SELECT path FROM experiment_artifacts WHERE run_id=?", (run_id,)):
                    log_artifact(artifact["path"])
            except Exception as exc:
                post_run_errors.append(_append_error("artifact_logging_failed", exc))
        completion_queued = schedule_benchmark_completion(
            insight_id,
            run_id,
            bundle,
            source="gpu_scheduler",
            resource_class=job.get("resource_class", "gpu_large"),
        )
        gpu_error = "\n".join(post_run_errors) if post_run_errors else None
        db.execute(
            """
            UPDATE gpu_jobs
            SET status='completed', completed_at=CURRENT_TIMESTAMP, artifact_uri=?, error_message=?
            WHERE id=?
            """,
            (
                db.fetchone("SELECT workdir FROM experiment_runs WHERE id=?", (run_id,)).get("workdir"),
                gpu_error,
                job_id,
            ),
        )
        if not completion_queued:
            db.execute(
                """
                UPDATE auto_research_jobs
                SET status=?, stage=?, artifact_bundle_id=?, last_note=?, last_error=?
                WHERE deep_insight_id=?
                """,
                (
                    "bundle_ready" if "error" not in bundle else "completed",
                    "writing_submission" if "error" not in bundle else "closed_loop_complete",
                    (bundle.get("bundle_ids") or [None])[-1],
                    f"GPU run completed with verdict={result.get('verdict', 'unknown')}. Submission bundle status={'ok' if 'error' not in bundle else 'failed'}.",
                    gpu_error,
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
        if _current_run_is_successful(run_id):
            db.execute(
                "UPDATE experiment_runs SET error_message=? WHERE id=?",
                (str(exc), run_id),
            )
            auto_research_status = "completed"
            auto_research_stage = "post_run_failed"
        else:
            db.execute(
                "UPDATE experiment_runs SET status='failed', error_message=? WHERE id=?",
                (str(exc), run_id),
            )
            auto_research_status = "failed"
            auto_research_stage = "gpu_failed"
        db.execute(
            "UPDATE auto_research_jobs SET status=?, stage=?, last_error=? WHERE deep_insight_id=?",
            (auto_research_status, auto_research_stage, str(exc), insight_id),
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
        _mark_job_inactive(int(job_id))
        _release_worker_if_no_running_jobs(worker_id, finished_job_id=job_id)
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
        if _try_start_next_gpu_job():
            processed += 1
    if last_event_id:
        db.ack_pipeline_events(GPU_SCHEDULER_CONSUMER, last_event_id)
    return {"events": len(events), "started_jobs": processed}


def _maybe_recover_stale_jobs() -> int:
    global _last_recovery_check
    now = time.time()
    if now - _last_recovery_check < max(30, GPU_STALE_RECOVERY_POLL_SECONDS):
        return 0
    _last_recovery_check = now
    return recover_stale_ssh_running_jobs() + recover_busy_workers_without_running_jobs()


def _loop() -> None:
    while not _stop_event.is_set():
        try:
            _maybe_recover_stale_jobs()
            stats = consume_pipeline_events_once(limit=50)
            if not stats.get("events"):
                _try_start_next_gpu_job()
                _stop_event.wait(max(1, GPU_POLL_SECONDS))
        except Exception as exc:  # pragma: no cover - defensive background guard
            try:
                db.rollback()
            except Exception:
                pass
            from orchestrator.pipeline import log_event

            log_event("error", {"step": "gpu_scheduler_loop", "error": str(exc)})
            _stop_event.wait(max(1, GPU_POLL_SECONDS))


def start() -> dict:
    global _scheduler_thread
    db.init_db()
    workers = register_default_workers()
    with _scheduler_lock:
        if _scheduler_thread and _scheduler_thread.is_alive():
            return {"status": "already_running"}
        if not _try_acquire_process_lock():
            return {"status": "already_running_elsewhere", "workers": list_workers()}
        recovered = (
            recover_stale_local_running_jobs(workers)
            + recover_stale_ssh_running_jobs()
            + recover_busy_workers_without_running_jobs()
        )
        _stop_event.clear()
        _scheduler_thread = threading.Thread(target=_loop, daemon=True, name="deepgraph-gpu-scheduler")
        _scheduler_thread.start()
    return {"status": "started", "workers": list_workers(), "recovered_stale_jobs": recovered}


def stop() -> dict:
    _stop_event.set()
    _release_process_lock()
    return {"status": "stopping"}
