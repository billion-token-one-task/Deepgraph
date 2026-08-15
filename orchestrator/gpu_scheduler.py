"""Single-host GPU scheduler and artifact collector for DeepGraph."""

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from agents.knowledge_loop import process_completed_run
from agents.manuscript_pipeline import generate_submission_bundle
from agents.validation_loop import run_full_benchmark_completion, run_validation_loop
from agents.compute_profile import detect_compute_profile
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
    GPU_REMOTE_SSH_ENDPOINTS,
    GPU_REMOTE_SSH_HOST,
    GPU_REMOTE_SSH_PORT,
    GPU_REMOTE_SSH_USER,
    GPU_REMOTE_SSH_PASSWORD,
    COMPUTE_SSH_CREDENTIAL_REF,
    GPU_VISIBLE_DEVICES,
)
from db import database as db
from meta_harness.scientific_authority import positive_decision_authorized
from orchestrator import ssh_gpu_backend
from orchestrator.benchmark_completion import (
    BENCHMARK_COMPLETION_STAGE,
    benchmark_completion_bundle_from_run,
    schedule_benchmark_completion,
)
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
_active_run_ids: set[int] = set()


def _bundle_failure_retry_fields(bundle: dict | None) -> dict | None:
    if not isinstance(bundle, dict) or "error" not in bundle:
        return None
    status = str(bundle.get("status") or "").strip()
    blockers = bundle.get("submission_blockers") if isinstance(bundle.get("submission_blockers"), list) else []
    default_error = "Manuscript quality gate failed" if status in {"manuscript_blocked", "needs_revision"} else "Submission bundle generation failed"
    blocker_text = "; ".join(str(item) for item in blockers[:8]) or str(bundle.get("error") or default_error)
    note = (
        "Manuscript quality gate failed; queued targeted manuscript revision instead of closing the loop."
        if status in {"manuscript_blocked", "needs_revision"}
        else "Submission bundle failed; queued targeted manuscript revision instead of closing the loop."
    )
    return {
        "status": "queued",
        "stage": "manuscript_retry_after_quality_gate",
        "last_note": note,
        "last_error": blocker_text[:4000],
    }


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


def _try_mark_run_active(run_id: int) -> bool:
    with _active_job_lock:
        run_id = int(run_id)
        if run_id in _active_run_ids:
            return False
        _active_run_ids.add(run_id)
        return True


def _mark_run_inactive(run_id: int) -> None:
    with _active_job_lock:
        _active_run_ids.discard(int(run_id))


def _job_is_active_in_this_process(job_id: int) -> bool:
    with _active_job_lock:
        return int(job_id) in _active_job_ids


def _run_is_active_in_this_process(run_id: int | None) -> bool:
    if run_id is None:
        return False
    with _active_job_lock:
        return int(run_id) in _active_run_ids


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



def _pmon_gpu_processes() -> dict[str, list[int]]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if proc.returncode != 0:
        return {}
    out: dict[str, list[int]] = {}
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2 or parts[1] == "-":
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        out.setdefault(parts[0], []).append(pid)
    return out


def _worker_visible_device(worker_id: str, worker: dict | None = None) -> str:
    if worker:
        raw = worker.get("metadata")
        if isinstance(raw, str) and raw.strip():
            try:
                metadata = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            if isinstance(metadata, dict) and metadata.get("visible_device") is not None:
                return str(metadata.get("visible_device"))
    if ":gpu" in worker_id:
        return worker_id.rsplit(":gpu", 1)[-1]
    return ""


def _pid_matches_run(pid: int, run_workdir: str) -> bool:
    try:
        cwd = Path(f"/proc/{pid}/cwd").resolve(strict=False)
    except Exception:
        cwd = Path("")
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except Exception:
        cmdline = ""
    needle = str(run_workdir or "").strip()
    if not needle:
        return False
    return needle in str(cwd) or needle in cmdline


def _local_run_has_live_process(job: dict) -> bool:
    worker_id = str(job.get("assigned_worker") or "")
    if not worker_id or worker_id.startswith("ssh:"):
        return True
    visible = _worker_visible_device(worker_id, job)
    if not visible:
        return True
    pids = _pmon_gpu_processes().get(visible, [])
    if not pids:
        return False
    run = db.fetchone("SELECT workdir FROM experiment_runs WHERE id=?", (job.get("experiment_run_id"),)) or {}
    workdir = str(run.get("workdir") or "")
    if not workdir:
        return bool(pids)
    return any(_pid_matches_run(pid, workdir) for pid in pids)

def recover_stale_local_running_jobs(workers: list[dict] | None = None) -> int:
    """Settle local GPU attempts whose in-process controller was lost."""
    local_ids = _local_worker_ids(workers)
    if not local_ids:
        return 0
    placeholders = ", ".join("?" for _ in local_ids)
    stale_jobs = db.fetchall(
        f"""
        SELECT gj.*, er.status AS run_status, er.hypothesis_verdict
        FROM gpu_jobs gj
        LEFT JOIN experiment_runs er
          ON er.agenda_id=gj.agenda_id
         AND er.id = gj.experiment_run_id
        WHERE gj.status='running'
          AND gj.assigned_worker IN ({placeholders})
        """,
        tuple(local_ids),
    )
    recovered = 0
    terminal_attempts: list[tuple[int, int, datetime, str]] = []
    for job in stale_jobs:
        agenda_id = int(job.get("agenda_id") or 0)
        if agenda_id <= 0:
            raise ValueError("stale local GPU recovery requires agenda scope")
        job_id = job["id"]
        run_id = job["experiment_run_id"]
        insight_id = job["deep_insight_id"]
        if _job_is_active_in_this_process(int(job_id)):
            continue
        if _local_run_has_live_process(job):
            continue
        if _current_run_closed_loop_complete(run_id):
            db.execute(
                """
                UPDATE gpu_jobs
                SET status='completed', completed_at=CURRENT_TIMESTAMP,
                    error_message=COALESCE(error_message, ?)
                WHERE id=? AND agenda_id=?
                """,
                (
                    "Recovered completed run after scheduler restart.",
                    job_id,
                    agenda_id,
                ),
            )
            db.execute(
                """
                UPDATE auto_research_jobs
                SET status='completed', stage='closed_loop_complete',
                    assigned_worker=NULL,
                    last_note=?, updated_at=CURRENT_TIMESTAMP, last_checked_at=CURRENT_TIMESTAMP
                WHERE deep_insight_id=? AND agenda_id=?
                """,
                (
                    f"Recovered completed GPU run {run_id} after scheduler restart.",
                    insight_id,
                    agenda_id,
                ),
            )
            terminal_attempts.append(
                (
                    int(job.get("gpu_attempt_reservation_id") or 0),
                    int(job_id),
                    datetime.now(timezone.utc),
                    "controller_lost_after_completion",
                )
            )
        else:
            db.execute(
                """
                UPDATE gpu_jobs
                SET status='failed', assigned_worker=NULL,
                    completed_at=CURRENT_TIMESTAMP, error_message=?
                WHERE id=? AND agenda_id=?
                """,
                (
                    "Recovered stale local running job after controller restart; "
                    "the lost attempt was settled before any retry.",
                    job_id,
                    agenda_id,
                ),
            )
            db.execute(
                """
                UPDATE auto_research_jobs
                SET status='queued', stage='retry_failed_run',
                    assigned_worker=NULL, experiment_run_id=?,
                    last_note=?, last_error=NULL,
                    updated_at=CURRENT_TIMESTAMP, last_checked_at=CURRENT_TIMESTAMP
                WHERE deep_insight_id=? AND agenda_id=?
                """,
                (
                    run_id,
                    f"Recovered and settled stale GPU attempt {job_id}; a retry requires a fresh attempt reservation.",
                    insight_id,
                    agenda_id,
                ),
            )
            terminal_attempts.append(
                (
                    int(job.get("gpu_attempt_reservation_id") or 0),
                    int(job_id),
                    datetime.now(timezone.utc),
                    "controller_lost",
                )
            )
        if job.get("assigned_worker"):
            db.execute(
                "UPDATE gpu_workers SET status='idle', heartbeat_at=CURRENT_TIMESTAMP WHERE id=?",
                (job["assigned_worker"],),
            )
        recovered += 1
    if recovered:
        db.commit()
    for reservation_id, job_id, completed_at, reason_code in terminal_attempts:
        if reservation_id <= 0:
            continue
        try:
            from meta_harness.attempt_gpu_usage import GrantGPUUsageControl
            from orchestrator.meta_compute_runtime import settle_legacy_job

            GrantGPUUsageControl().settle_attempt(
                reservation_id,
                completed_at=completed_at,
                reason_code=reason_code,
            )
            settle_legacy_job(job_id)
        except Exception:
            db.rollback()
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
            "for pid in $(pgrep -f 'deepgraph_exec_run_|train.py|benchmark_runner.py|evaluation.py' || true); do",
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


_FALSE_UNBOUND_BLOCKER = "experiment run is not bound to a ResourceGrant"


def recover_falsely_unbound_gpu_jobs(limit: int = 25) -> int:
    """Requeue runs a fixed projection defect reported as grant-less.

    _next_job selected the run without resource_grant_id, so the capability
    blocker resolved grant 0 and failed correctly bound jobs with a verdict
    that contradicts the database. Recovery is restricted to that provable
    contradiction: the recorded blocker claims the run has no grant while the
    run holds an active grant whose capability preflight passed. The scaffold
    and its admitted adapter are kept -- the experiment never ran, only the
    controller refused to launch it -- so no regeneration is charged. Once no
    row contradicts itself this is a no-op.
    """
    from orchestrator import meta_compute_runtime

    rows = db.fetchall(
        """
        SELECT j.id AS job_id, j.agenda_id, j.deep_insight_id,
               r.id AS run_id, r.phase
          FROM gpu_jobs j
          JOIN experiment_runs r ON r.id = j.experiment_run_id
          JOIN resource_grants g ON g.id = r.resource_grant_id
          JOIN candidate_preflight_results_v1 p
            ON p.id = g.preflight_result_id AND p.status='passed'
         WHERE j.status='failed'
           AND j.error_message = ?
           AND g.status='active'
           AND g.expires_at > CURRENT_TIMESTAMP
           AND r.phase='scaffold_ready'
         ORDER BY j.id
         LIMIT ?
        """,
        (_FALSE_UNBOUND_BLOCKER, limit),
    )
    recovered = 0
    for row in rows:
        job_id = int(row["job_id"])
        run_id = int(row["run_id"])
        agenda_id = int(row["agenda_id"] or 0)
        insight_id = int(row["deep_insight_id"] or 0)
        if agenda_id <= 0 or insight_id <= 0:
            continue
        try:
            meta_compute_runtime.settle_legacy_job(job_id)
        except Exception:
            db.rollback()
            continue
        note = (
            f"Recovered GPU job {job_id}: the launch blocker reported an unbound "
            "run while the run held an active grant with a passed preflight. The "
            "scaffold is intact and is requeued without regeneration."
        )
        db.execute(
            """
            UPDATE experiment_runs
            SET status='scaffolding', error_message=NULL,
                completed_at=NULL
            WHERE id=? AND agenda_id=? AND status='failed'
              AND phase='scaffold_ready'
            """,
            (run_id, agenda_id),
        )
        db.execute(
            """
            UPDATE auto_research_jobs
            SET status='queued', stage='retry_failed_run',
                assigned_worker=NULL, experiment_run_id=?,
                last_note=?, last_error=NULL,
                updated_at=CURRENT_TIMESTAMP, last_checked_at=CURRENT_TIMESTAMP
            WHERE deep_insight_id=? AND agenda_id=?
            """,
            (run_id, note, insight_id, agenda_id),
        )
        db.commit()
        recovered += 1
    return recovered


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
    terminal_attempts: list[tuple[int, int, object, str]] = []
    for job in stale_jobs:
        agenda_id = int(job.get("agenda_id") or 0)
        if agenda_id <= 0:
            raise ValueError("stale SSH GPU recovery requires agenda scope")
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
        if _current_run_closed_loop_complete(run_id):
            db.execute(
                """
                UPDATE gpu_jobs
                SET status='completed', completed_at=CURRENT_TIMESTAMP,
                    error_message=COALESCE(error_message, ?)
                WHERE id=? AND agenda_id=?
                """,
                (message, job_id, agenda_id),
            )
            db.execute(
                """
                UPDATE auto_research_jobs
                SET status='completed', stage='closed_loop_complete',
                    assigned_worker=NULL,
                    last_note=?, updated_at=CURRENT_TIMESTAMP, last_checked_at=CURRENT_TIMESTAMP
                WHERE deep_insight_id=? AND agenda_id=?
                """,
                (message, insight_id, agenda_id),
            )
            terminal_attempts.append(
                (
                    int(job.get("gpu_attempt_reservation_id") or 0),
                    job_id,
                    datetime.now(timezone.utc),
                    "controller_lost_after_completion",
                )
            )
        elif _current_run_is_successful(run_id):
            db.execute(
                """
                UPDATE gpu_jobs
                SET status='completed', assigned_worker=NULL,
                    completed_at=CURRENT_TIMESTAMP, error_message=?
                WHERE id=? AND agenda_id=?
                """,
                (
                    "Recovered successful SSH execution after controller loss; GPU usage was "
                    "settled and only post-run scientific work remains.",
                    job_id,
                    agenda_id,
                ),
            )
            db.execute(
                """
                UPDATE auto_research_jobs
                SET status='review_pending', stage='scientific_decision_required',
                    assigned_worker=NULL, experiment_run_id=?,
                    last_note=?, last_error=NULL,
                    updated_at=CURRENT_TIMESTAMP, last_checked_at=CURRENT_TIMESTAMP
                WHERE deep_insight_id=? AND agenda_id=?
                """,
                (
                    run_id,
                    f"Recovered successful SSH attempt {job_id}; GPU use is settled and post-run review will resume without rerunning compute.",
                    insight_id,
                    agenda_id,
                ),
            )
            terminal_attempts.append(
                (
                    int(job.get("gpu_attempt_reservation_id") or 0),
                    job_id,
                    datetime.now(timezone.utc),
                    "controller_lost_after_success",
                )
            )
        else:
            db.execute(
                """
                UPDATE gpu_jobs
                SET status='failed', completed_at=CURRENT_TIMESTAMP,
                    error_message=?
                WHERE id=? AND agenda_id=?
                """,
                (message, job_id, agenda_id),
            )
            db.execute(
                """
                UPDATE experiment_runs
                SET status='failed', error_message=?, completed_at=CURRENT_TIMESTAMP
                WHERE id=? AND agenda_id=?
                """,
                (message, run_id, agenda_id),
            )
            db.execute(
                """
                UPDATE auto_research_jobs
                SET status='queued', stage='retry_failed_run',
                    assigned_worker=NULL, experiment_run_id=?,
                    last_note=?, last_error=NULL,
                    updated_at=CURRENT_TIMESTAMP, last_checked_at=CURRENT_TIMESTAMP
                WHERE deep_insight_id=? AND agenda_id=?
                """,
                (run_id, message, insight_id, agenda_id),
            )
            terminal_attempts.append(
                (
                    int(job.get("gpu_attempt_reservation_id") or 0),
                    job_id,
                    datetime.now(timezone.utc),
                    "controller_lost",
                )
            )
        if job.get("assigned_worker"):
            db.execute(
                "UPDATE gpu_workers SET status='idle', heartbeat_at=CURRENT_TIMESTAMP WHERE id=?",
                (job["assigned_worker"],),
            )
        recovered += 1
    if recovered:
        db.commit()
    for reservation_id, job_id, completed_at, reason_code in terminal_attempts:
        if reservation_id <= 0:
            continue
        try:
            from meta_harness.attempt_gpu_usage import GrantGPUUsageControl
            from orchestrator.meta_compute_runtime import settle_legacy_job

            GrantGPUUsageControl().settle_attempt(
                reservation_id,
                completed_at=completed_at,
                reason_code=reason_code,
            )
            settle_legacy_job(job_id)
        except Exception:
            db.rollback()
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
    return []


def ssh_worker_id(host: str, port: int, device: str) -> str:
    """Stable id for one SSH GPU device.

    The port is part of the id so that several nodes behind the same public IP
    on different ports -- a common setup for rented GPU boxes -- register as
    distinct workers instead of colliding on ``ssh:{host}:gpu{n}``. Nothing
    parses this id back into host/port; connection details live in metadata, so
    this is purely a uniqueness fix.
    """
    return f"ssh:{host}:{int(port)}:gpu{device}"


def upsert_ssh_worker(
    *,
    host: str,
    port: int,
    user: str,
    device: str,
    gpu_index: int,
    credential_ref: str,
    gpu_model: str | None = None,
    total_mem_gb: float | None = None,
    remote_base_dir: str | None = None,
    python_bin: str | None = None,
) -> dict:
    """Insert or refresh one SSH GPU worker row. Returns the worker summary."""
    worker_id = ssh_worker_id(host, port, device)
    metadata = {
        "backend": "ssh",
        "visible_device": device,
        "ssh_host": host,
        "ssh_port": int(port),
        "ssh_user": user,
        "credential_ref": credential_ref,
        "remote_base_dir": remote_base_dir or GPU_REMOTE_BASE_DIR,
        "python_bin": python_bin or GPU_REMOTE_PYTHON,
    }
    model = gpu_model or GPU_DEFAULT_MODEL
    mem = float(total_mem_gb if total_mem_gb is not None else GPU_DEFAULT_VRAM_GB)
    existing = db.fetchone("SELECT id FROM gpu_workers WHERE id=?", (worker_id,))
    if existing:
        db.execute(
            """UPDATE gpu_workers
               SET hostname=?, gpu_index=?, gpu_model=?, total_mem_gb=?,
                   status=CASE WHEN status IN ('busy','retired') THEN status ELSE ? END,
                   heartbeat_at=CURRENT_TIMESTAMP, metadata=?
               WHERE id=?""",
            (host, gpu_index, model, mem, "idle", json.dumps(metadata), worker_id),
        )
    else:
        db.execute(
            """INSERT INTO gpu_workers
               (id, hostname, gpu_index, gpu_model, total_mem_gb, status, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (worker_id, host, gpu_index, model, mem, "idle", json.dumps(metadata)),
        )
    return {
        "id": worker_id,
        "hostname": host,
        "gpu_index": gpu_index,
        "gpu_model": model,
        "total_mem_gb": mem,
        "status": "idle",
        **metadata,
    }


def upsert_colab_worker(
    *,
    account_ref: str,
    gpu_model: str,
    total_mem_gb: float,
    measured_at: str,
    detail: dict | None = None,
) -> dict:
    """Record a Colab accelerator that a canary actually reached.

    Preflight refuses to infer Colab hardware from an account manifest and
    holds its VRAM at zero until something measures it, which is the right
    default for a backend whose allocation is not guaranteed. There was no way
    to record that measurement, so the zero was permanent and Colab could never
    be selected. This writes only what a canary observed on the device.
    """
    if not str(account_ref).strip():
        raise ValueError("a Colab worker row requires an account reference")
    if float(total_mem_gb) <= 0:
        raise ValueError("a Colab worker row requires measured VRAM")
    hostname = _local_hostname()
    worker_id = f"{hostname}:colab-{account_ref}"
    metadata = {
        "backend": "colab",
        "visible_device": None,
        "account_ref": str(account_ref),
        "measured_at": str(measured_at),
        **{str(k): v for k, v in (detail or {}).items()},
    }
    existing = db.fetchone("SELECT id FROM gpu_workers WHERE id=?", (worker_id,))
    if existing:
        db.execute(
            """UPDATE gpu_workers
               SET hostname=?, gpu_model=?, total_mem_gb=?, status='idle',
                   heartbeat_at=CURRENT_TIMESTAMP, metadata=?
               WHERE id=?""",
            (hostname, gpu_model, float(total_mem_gb), json.dumps(metadata), worker_id),
        )
    else:
        db.execute(
            """INSERT INTO gpu_workers
               (id, hostname, gpu_index, gpu_model, total_mem_gb, status, metadata)
               VALUES (?, ?, ?, ?, ?, 'idle', ?)""",
            (worker_id, hostname, 0, gpu_model, float(total_mem_gb), json.dumps(metadata)),
        )
    return {
        "id": worker_id,
        "gpu_model": gpu_model,
        "total_mem_gb": float(total_mem_gb),
        "status": "idle",
        **metadata,
    }


def register_default_workers() -> list[dict]:
    db.init_db()
    if GPU_MODE == "ssh":
        if not (GPU_REMOTE_SSH_ENDPOINTS or GPU_REMOTE_SSH_HOST) or not GPU_REMOTE_SSH_USER:
            raise RuntimeError(
                "DEEPGRAPH_GPU_MODE=ssh requires DEEPGRAPH_GPU_REMOTE_SSH_HOST (or"
                " DEEPGRAPH_GPU_REMOTE_SSH_ENDPOINTS) and DEEPGRAPH_GPU_REMOTE_SSH_USER."
            )
        endpoints: list[tuple[str, int]] = []
        for entry in GPU_REMOTE_SSH_ENDPOINTS:
            host, sep, port = entry.rpartition(":")
            if not sep or not host or not port.isdigit():
                raise RuntimeError(
                    f"DEEPGRAPH_GPU_REMOTE_SSH_ENDPOINTS entry {entry!r} is not host:port"
                )
            endpoints.append((host, int(port)))
        if not endpoints:
            endpoints = [(GPU_REMOTE_SSH_HOST, GPU_REMOTE_SSH_PORT)]
        workers = []
        for host, port in endpoints:
            for idx, gpu_id in enumerate(GPU_VISIBLE_DEVICES):
                summary = upsert_ssh_worker(
                    host=host,
                    port=port,
                    user=GPU_REMOTE_SSH_USER,
                    device=gpu_id,
                    gpu_index=idx,
                    credential_ref=COMPUTE_SSH_CREDENTIAL_REF,
                )
                workers.append(summary)
        db.commit()
        return workers

    hostname = _local_hostname()
    inventory = _local_gpu_inventory()
    if not inventory and not os.getenv("DEEPGRAPH_GPU_VISIBLE_DEVICES"):
        profile = detect_compute_profile()
        db.execute(
            """UPDATE gpu_workers
               SET status='offline', heartbeat_at=CURRENT_TIMESTAMP
               WHERE hostname=?
                 AND (metadata IS NULL OR metadata NOT LIKE ?)""",
            (hostname, '%"backend": "ssh"%'),
        )
        db.commit()
        if not profile.local_gpu_available:
            return []
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



def _max_schedulable_worker_vram_gb() -> float:
    try:
        workers = register_default_workers()
    except Exception:
        workers = []
    values: list[float] = []
    for worker in workers:
        try:
            value = float(worker.get("total_mem_gb") or 0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            values.append(value)
    if not values:
        filter_sql, params = _worker_filter_sql()
        if GPU_MODE != "ssh":
            filter_sql += " AND hostname=?"
            params = (*params, _local_hostname())
        rows = db.fetchall(
            f"""
            SELECT total_mem_gb FROM gpu_workers
            WHERE COALESCE(status, '') <> 'offline'{filter_sql}
            """,
            params,
        )
        for row in rows:
            try:
                value = float(row.get("total_mem_gb") or 0)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                values.append(value)
    return max(values) if values else 0.0


def _effective_vram_required_gb(resource_class: str, requested_vram_gb: float) -> tuple[float, str | None]:
    try:
        requested = float(requested_vram_gb or 0)
    except (TypeError, ValueError):
        requested = 0.0
    if requested <= 0:
        return 0.0, None
    max_worker_vram = _max_schedulable_worker_vram_gb()
    if max_worker_vram <= 0 or requested <= max_worker_vram:
        return requested, None
    if max_worker_vram >= 16:
        effective = max_worker_vram
        note = (
            f"Adjusted vram_required_gb from {requested:g} to {effective:g} for {resource_class}; "
            "runner will receive single/micro-batch VRAM environment hints."
        )
        return effective, note
    return requested, None


def queue_run(
    *,
    insight_id: int,
    run_id: int,
    resource_grant_id: int,
    resource_class: str,
    priority: int = 0,
    gpu_count: int = 1,
    vram_required_gb: float = 0,
    timeout_s: int | None = None,
    meta_harness_idempotency_key: str | None = None,
) -> int:
    db.init_db()
    run = db.fetchone(
        "SELECT agenda_id, resource_grant_id FROM experiment_runs WHERE id=? AND deep_insight_id=?",
        (run_id, insight_id),
    )
    if not run or int(run.get("agenda_id") or 0) <= 0:
        raise RuntimeError("GPU queue requires an agenda-scoped experiment run")
    if int(run.get("resource_grant_id") or 0) != int(resource_grant_id or 0):
        raise RuntimeError("GPU queue ResourceGrant does not match the experiment run")
    grant = db.fetchone(
        """
        SELECT id, stage FROM resource_grants
        WHERE id=? AND agenda_id=? AND idea_id=?
          AND status='active' AND expires_at > CURRENT_TIMESTAMP
        """,
        (resource_grant_id, int(run["agenda_id"]), insight_id),
    )
    if not grant or str(grant.get("stage") or "") not in {
        "pilot",
        "full_benchmark",
    }:
        raise RuntimeError("GPU queue requires an active pilot/full_benchmark ResourceGrant")
    durable_key = str(meta_harness_idempotency_key or "").strip()
    claim = None
    if db._use_pg():  # noqa: SLF001
        if not durable_key:
            raise RuntimeError(
                "PostgreSQL GPU queue requires durable ComputeScheduler admission"
            )
        claim = db.fetchone(
            """
            SELECT id, gpu_attempt_reservation_id FROM compute_jobs_v1
            WHERE agenda_id=? AND idea_id=? AND resource_grant_id=?
              AND idempotency_key=? AND command_ref=?
              AND backend_kind IN ('local_gpu', 'ssh_gpu')
              AND status='submitting'
            """,
            (
                int(run["agenda_id"]),
                int(insight_id),
                int(resource_grant_id),
                durable_key,
                f"experiment-run:{int(run_id)}",
            ),
        )
        if not claim:
            raise RuntimeError(
                "GPU queue durable compute claim is missing or out of scope"
            )
        if int(claim.get("gpu_attempt_reservation_id") or 0) <= 0:
            raise RuntimeError(
                "GPU queue canonical attempt reservation is missing"
            )
    if timeout_s is not None and int(timeout_s) <= 0:
        raise ValueError("GPU job timeout must be a positive hard limit")
    effective_vram_required_gb, scheduling_note = _effective_vram_required_gb(resource_class, vram_required_gb)
    jid = db.insert_returning_id(
        """
        INSERT INTO gpu_jobs
        (agenda_id, resource_grant_id, deep_insight_id, experiment_run_id,
         resource_class, gpu_count, vram_required_gb, timeout_s, priority,
         status, error_message, meta_harness_idempotency_key)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
        RETURNING id
        """,
        (
            int(run["agenda_id"]),
            resource_grant_id,
            insight_id,
            run_id,
            resource_class,
            gpu_count,
            effective_vram_required_gb,
            GPU_JOB_TIMEOUT_SECONDS if timeout_s is None else timeout_s,
            priority,
            scheduling_note,
            durable_key or None,
        ),
    )
    if durable_key and claim:
        from meta_harness.attempt_gpu_usage import GrantGPUUsageControl

        GrantGPUUsageControl().bind_gpu_job(
            int(claim["gpu_attempt_reservation_id"]),
            int(jid),
            commit=False,
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
            INSERT INTO experiment_artifacts
                (agenda_id, run_id, artifact_type, path)
            VALUES (?, ?, ?, ?)
            """,
            (int(run["agenda_id"]), run_id, artifact_type, str(path)),
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
        if _run_is_active_in_this_process(job.get("experiment_run_id")):
            if not _job_is_active_in_this_process(int(job["id"])):
                _mark_run_inactive(int(job["experiment_run_id"]))
            else:
                db.execute(
                    """
                    UPDATE gpu_jobs
                    SET error_message=?
                    WHERE id=? AND agenda_id=? AND status='queued'
                    """,
                    (
                        "Deferred because this experiment run is already active in this scheduler process; "
                        "skipping queue head so other GPU jobs can launch.",
                        job["id"],
                        int(job["agenda_id"]),
                    ),
                )
                db.commit()
                continue
        # _capability_preflight_blocker reads resource_grant_id and agenda_id
        # to resolve the grant's passed preflight. They were absent from this
        # projection, so every queued job reported "experiment run is not bound
        # to a ResourceGrant" however well bound it was, and the blocker could
        # never pass.
        run = db.fetchone(
            "SELECT id, deep_insight_id, agenda_id, resource_grant_id, status,"
            " phase, error_message, workdir FROM experiment_runs WHERE id=?",
            (job["experiment_run_id"],),
        )
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
    if status == "adapter_repairing" or phase == "adapter_repairing":
        return "experiment_run adapter repair is in progress; refusing to launch queued GPU job"
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
    preflight_blocker = _capability_preflight_blocker(run)
    if preflight_blocker:
        return preflight_blocker
    return None


def _capability_preflight_blocker(run: dict) -> str | None:
    """Fail closed for production compute not bound to a passed capability."""
    if not db._use_pg():  # noqa: SLF001 - SQLite unit fixtures predate V1 grants.
        return None
    # A caller that never selected these columns is a defect, not an unbound
    # run. Reporting both the same way hid a projection that omitted
    # resource_grant_id behind a plausible policy refusal, and every queued GPU
    # job failed as "not bound to a ResourceGrant" while being correctly bound.
    absent = [
        column
        for column in ("resource_grant_id", "agenda_id", "deep_insight_id")
        if column not in run
    ]
    if absent:
        return (
            "experiment_run row is missing columns required for the capability "
            "check: " + ", ".join(absent)
        )
    grant_id = int(run.get("resource_grant_id") or 0)
    if grant_id <= 0:
        return "experiment run is not bound to a ResourceGrant"
    row = db.fetchone(
        """
        SELECT rg.preflight_result_id, p.status, p.adapter_id,
               p.dataset_revision, p.model_revision
        FROM resource_grants rg
        LEFT JOIN candidate_preflight_results_v1 p
          ON p.id=rg.preflight_result_id
         AND p.agenda_id=rg.agenda_id AND p.idea_id=rg.idea_id
        WHERE rg.id=? AND rg.agenda_id=? AND rg.idea_id=?
        """,
        (
            grant_id,
            int(run.get("agenda_id") or 0),
            int(run.get("deep_insight_id") or 0),
        ),
    )
    if not row or int(row.get("preflight_result_id") or 0) <= 0:
        return "legacy compute run lacks a capability preflight binding"
    if (
        str(row.get("status") or "") != "passed"
        or not str(row.get("adapter_id") or "").strip()
        or not str(row.get("dataset_revision") or "").strip()
        or not str(row.get("model_revision") or "").strip()
    ):
        return "compute grant capability preflight is incomplete or not passed"
    return None


def _fail_blocked_queued_job(job: dict, reason: str) -> None:
    job_id = int(job["id"])
    run_id = int(job["experiment_run_id"])
    agenda_id = int(job.get("agenda_id") or 0)
    if agenda_id <= 0:
        raise ValueError("blocking a GPU job requires agenda scope")
    insight_id = int(job["deep_insight_id"]) if job.get("deep_insight_id") is not None else None
    db.execute(
        """
        UPDATE gpu_jobs
        SET status='failed', assigned_worker=NULL, completed_at=CURRENT_TIMESTAMP,
            error_message=?
        WHERE id=? AND agenda_id=?
        """,
        (reason, job_id, agenda_id),
    )
    db.execute(
        """
        UPDATE experiment_runs
        SET status='failed', error_message=COALESCE(error_message, ?),
            completed_at=COALESCE(completed_at, CURRENT_TIMESTAMP)
        WHERE id=? AND agenda_id=?
        """,
        (reason, run_id, agenda_id),
    )
    if insight_id is not None:
        db.execute(
            """
            UPDATE auto_research_jobs
            SET status='failed', stage='gpu_blocked',
                assigned_worker=NULL, last_error=?, updated_at=CURRENT_TIMESTAMP,
                last_checked_at=CURRENT_TIMESTAMP
            WHERE deep_insight_id=? AND experiment_run_id=? AND agenda_id=?
            """,
            (reason, insight_id, run_id, agenda_id),
        )
    db.commit()


def _append_error(prefix: str, exc: Exception) -> str:
    return f"{prefix}: {exc}"


def _current_run_is_successful(run_id: int) -> bool:
    run = db.fetchone("SELECT status, hypothesis_verdict FROM experiment_runs WHERE id=?", (run_id,))
    if not run:
        return False
    verdict = str(run.get("hypothesis_verdict") or "").strip().lower()
    return (
        run.get("status") in {"completed", "bundle_ready"}
        and verdict
        in {"supported", "confirmed", "refuted", "inconclusive", "reproduced"}
    )


def _current_run_closed_loop_complete(run_id: int) -> bool:
    run = db.fetchone(
        "SELECT status, submission_bundle_id FROM experiment_runs WHERE id=?",
        (run_id,),
    )
    if not run:
        return False
    if run.get("status") == "bundle_ready" or run.get("submission_bundle_id"):
        return True
    rows = db.fetchall(
        """
        SELECT mr.status AS manuscript_status, sb.status AS bundle_status
        FROM manuscript_runs mr
        LEFT JOIN submission_bundles sb ON sb.manuscript_run_id=mr.id
        WHERE mr.experiment_run_id=?
        """,
        (run_id,),
    )
    return any(
        row.get("manuscript_status") == "bundle_ready" or row.get("bundle_status") == "ready"
        for row in rows
    )


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
    agenda_id = int(job.get("agenda_id") or 0)
    if agenda_id <= 0:
        raise ValueError("GPU execution requires agenda scope")
    attempt_reservation_id = int(job.get("gpu_attempt_reservation_id") or 0)
    if attempt_reservation_id <= 0:
        reason = "canonical GPU attempt reservation is required before launch"
        _fail_blocked_queued_job(job, reason)
        _release_worker_if_no_running_jobs(worker_id, finished_job_id=int(job_id))
        db.commit()
        return
    # _next_job may have selected this row before a concurrent capability
    # adapter repair claimed the run. Re-read under the database lock at the
    # actual launch boundary rather than trusting that stale queue snapshot.
    run_lock = " FOR UPDATE" if db._use_pg() else ""  # noqa: SLF001
    current_run = db.fetchone(
        "SELECT id, agenda_id, deep_insight_id, status, phase, error_message, "
        "resource_grant_id FROM experiment_runs WHERE id=? AND agenda_id=?" + run_lock,
        (run_id, agenda_id),
    )
    launch_blocker = _launch_blocker_for_run(current_run)
    if launch_blocker:
        _fail_blocked_queued_job(job, launch_blocker)
        _release_worker_if_no_running_jobs(worker_id, finished_job_id=int(job_id))
        db.commit()
        return
    _mark_job_active(int(job_id))
    if not _try_mark_run_active(int(run_id)):
        _mark_run_inactive(int(run_id))
        if not _try_mark_run_active(int(run_id)):
            db.execute(
                """
                UPDATE gpu_jobs
                SET status='queued', assigned_worker=NULL, started_at=NULL,
                    error_message='Deferred because this experiment run is already active in this scheduler process.'
                WHERE id=? AND agenda_id=?
                """,
                (job_id, agenda_id),
            )
            db.execute(
                "UPDATE gpu_workers SET status='idle', heartbeat_at=CURRENT_TIMESTAMP WHERE id=?",
                (worker_id,),
            )
            db.commit()
            _mark_job_inactive(int(job_id))
            return
    try:
        from meta_harness.attempt_gpu_usage import GrantGPUUsageControl

        attempt = GrantGPUUsageControl().start_attempt(
            attempt_reservation_id,
            commit=False,
        )
    except Exception as exc:
        reason = f"GPU attempt admission failed:{exc}"
        try:
            GrantGPUUsageControl().release_unstarted(
                attempt_reservation_id,
                reason_code="attempt_start_refused",
            )
        except Exception:
            db.rollback()
        _fail_blocked_queued_job(job, reason)
        _mark_job_inactive(int(job_id))
        _mark_run_inactive(int(run_id))
        _release_worker_if_no_running_jobs(worker_id, finished_job_id=int(job_id))
        db.commit()
        return
    job["timeout_s"] = min(
        int(job.get("timeout_s") or attempt.timeout_seconds),
        int(attempt.timeout_seconds),
    )
    db.execute(
        "UPDATE gpu_jobs SET timeout_s=? WHERE id=? AND agenda_id=? AND status='queued'",
        (int(job["timeout_s"]), int(job_id), agenda_id),
    )
    auto_job = db.fetchone(
        "SELECT stage FROM auto_research_jobs WHERE deep_insight_id=? AND agenda_id=?",
        (insight_id, agenda_id),
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
        WHERE id=? AND agenda_id=?
        """,
        (worker_id, job_id, agenda_id),
    )
    db.execute(
        """
        UPDATE experiment_runs
        SET status='running_gpu', resource_class=?
        WHERE id=? AND agenda_id=?
        """,
        (job.get("resource_class", "gpu_small"), run_id, agenda_id),
    )
    db.execute(
        """
        UPDATE auto_research_jobs
        SET status='running_gpu',
            stage=CASE WHEN stage=? THEN stage ELSE 'gpu_scheduler' END,
            assigned_worker=?
        WHERE deep_insight_id=? AND agenda_id=?
        """,
        (BENCHMARK_COMPLETION_STAGE, worker_id, insight_id, agenda_id),
    )
    db.commit()

    try:
        post_run_errors: list[str] = []
        bundle: dict = {}
        completion_queued = False
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
            execution_verdict = str(result.get("verdict") or "").strip().lower()
            if result.get("error") or execution_verdict in {
                "failed",
                "blocked",
                "invalid",
                "cancelled",
                "timed_out",
            }:
                raise RuntimeError(
                    "validation execution failed: "
                    + str(
                        result.get("error")
                        or result.get("reason")
                        or execution_verdict
                        or "unknown"
                    )
                )
            try:
                process_completed_run(run_id)
            except Exception as exc:
                post_run_errors.append(_append_error("knowledge_loop_failed", exc))
            try:
                collect_run_artifacts(run_id)
            except Exception as exc:
                post_run_errors.append(_append_error("artifact_collection_failed", exc))
            benchmark_bundle = benchmark_completion_bundle_from_run(run_id, result=result)
            completion_queued = schedule_benchmark_completion(
                insight_id,
                run_id,
                benchmark_bundle,
                source="gpu_scheduler_pre_manuscript",
                resource_class=job.get("resource_class", "gpu_large"),
            )
            scientific_decision_ready = positive_decision_authorized(
                agenda_id=int(job.get("agenda_id") or 0),
                run_id=run_id,
            )
            if not completion_queued and scientific_decision_ready:
                try:
                    bundle = generate_submission_bundle(run_id)
                except Exception as exc:
                    bundle = {"error": str(exc)}
                    post_run_errors.append(_append_error("submission_bundle_failed", exc))
                if "error" in bundle:
                    post_run_errors.append("submission_bundle_result_error: " + str(bundle.get("error")))
            elif not completion_queued:
                bundle = {
                    "error": "supported scientific decision required before manuscript",
                    "status": "scientific_decision_required",
                }
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
        if not completion_queued and scientific_decision_ready:
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
            WHERE id=? AND agenda_id=?
            """,
            (
                db.fetchone("SELECT workdir FROM experiment_runs WHERE id=?", (run_id,)).get("workdir"),
                gpu_error,
                job_id,
                agenda_id,
            ),
        )
        if not completion_queued:
            if not scientific_decision_ready:
                db.execute(
                    """
                    UPDATE auto_research_jobs
                    SET status='review_pending',
                        stage='scientific_decision_required',
                        artifact_bundle_id=NULL,
                        last_note=?,
                        last_error=NULL,
                        assigned_worker=NULL, updated_at=CURRENT_TIMESTAMP
                    WHERE deep_insight_id=? AND agenda_id=?
                    """,
                    (
                        "Execution completed; waiting for evidence audit and "
                        "an independent supported scientific decision.",
                        insight_id,
                        agenda_id,
                    ),
                )
            else:
                retry_fields = _bundle_failure_retry_fields(bundle if isinstance(bundle, dict) else None)
                if retry_fields:
                    db.execute(
                        """
                        UPDATE auto_research_jobs
                        SET status=?, stage=?, artifact_bundle_id=?, last_note=?, last_error=?,
                            assigned_worker=NULL, updated_at=CURRENT_TIMESTAMP
                        WHERE deep_insight_id=? AND agenda_id=?
                        """,
                        (
                            retry_fields["status"],
                            retry_fields["stage"],
                            (bundle.get("bundle_ids") or [None])[-1] if isinstance(bundle, dict) else None,
                            retry_fields["last_note"],
                            retry_fields["last_error"] or gpu_error,
                            insight_id,
                            agenda_id,
                        ),
                    )
                else:
                    bundle_ok = isinstance(bundle, dict) and "error" not in bundle
                    db.execute(
                        """
                        UPDATE auto_research_jobs
                        SET status=?, stage=?, artifact_bundle_id=?, last_note=?, last_error=?
                        WHERE deep_insight_id=? AND agenda_id=?
                        """,
                        (
                            "bundle_ready" if bundle_ok else "completed",
                            "writing_submission" if bundle_ok else "closed_loop_complete",
                            (bundle.get("bundle_ids") or [None])[-1] if isinstance(bundle, dict) else None,
                            f"GPU run completed with verdict={result.get('verdict', 'unknown')}. Submission bundle status={'ok' if bundle_ok else 'failed'}.",
                            gpu_error,
                            insight_id,
                            agenda_id,
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
            "UPDATE gpu_jobs SET status='failed', error_message=?, completed_at=CURRENT_TIMESTAMP WHERE id=? AND agenda_id=?",
            (str(exc), job_id, agenda_id),
        )
        if _current_run_is_successful(run_id):
            db.execute(
                "UPDATE experiment_runs SET error_message=? WHERE id=? AND agenda_id=?",
                (str(exc), run_id, agenda_id),
            )
            auto_research_status = "completed"
            auto_research_stage = "post_run_failed"
        else:
            db.execute(
                "UPDATE experiment_runs SET status='failed', error_message=? WHERE id=? AND agenda_id=?",
                (str(exc), run_id, agenda_id),
            )
            auto_research_status = "failed"
            auto_research_stage = "gpu_failed"
        db.execute(
            "UPDATE auto_research_jobs SET status=?, stage=?, last_error=? WHERE deep_insight_id=? AND agenda_id=?",
            (
                auto_research_status,
                auto_research_stage,
                str(exc),
                insight_id,
                agenda_id,
            ),
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
        try:
            from meta_harness.attempt_gpu_usage import GrantGPUUsageControl

            terminal = db.fetchone(
                """
                SELECT status, completed_at, error_message
                FROM gpu_jobs WHERE id=? AND agenda_id=?
                """,
                (int(job_id), agenda_id),
            ) or {}
            terminal_status = str(terminal.get("status") or "failed")
            reason_code = {
                "completed": "attempt_completed",
                "timed_out": "attempt_timed_out",
                "cancelled": "attempt_cancelled",
            }.get(terminal_status, "attempt_failed")
            GrantGPUUsageControl().settle_attempt(
                attempt_reservation_id,
                completed_at=terminal.get("completed_at"),
                reason_code=reason_code,
            )
        except Exception as exc:
            try:
                db.rollback()
            except Exception:
                pass
            try:
                from orchestrator.pipeline import log_event

                log_event(
                    "error",
                    {
                        "step": "attempt_gpu_settlement",
                        "gpu_job_id": int(job_id),
                        "attempt_reservation_id": attempt_reservation_id,
                        "error": str(exc),
                    },
                )
            except Exception:
                pass
        try:
            from orchestrator.meta_compute_runtime import settle_legacy_job

            settle_legacy_job(int(job_id))
        except Exception as exc:  # durable v1 remains fail-closed/reconcilable
            try:
                from orchestrator.pipeline import log_event

                log_event(
                    "error",
                    {
                        "step": "meta_compute_settlement",
                        "gpu_job_id": int(job_id),
                        "error": str(exc),
                    },
                )
            except Exception:
                pass
        _mark_job_inactive(int(job_id))
        _mark_run_inactive(int(run_id))
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
    workers = register_default_workers()
    recovered = (
        recover_stale_local_running_jobs(workers)
        + recover_stale_ssh_running_jobs()
        + recover_busy_workers_without_running_jobs()
    )
    if db._use_pg():  # noqa: SLF001
        from meta_harness.attempt_gpu_usage import GrantGPUUsageControl
        from orchestrator.meta_compute_runtime import settle_legacy_job

        durable_job_ids = GrantGPUUsageControl().reconcile_terminal_attempts()
        for gpu_job_id in durable_job_ids:
            settle_legacy_job(gpu_job_id)
        recovered += len(durable_job_ids)
    return recovered


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
    with _scheduler_lock:
        if _scheduler_thread and _scheduler_thread.is_alive():
            return {"status": "already_running"}
        if not _try_acquire_process_lock():
            return {"status": "already_running_elsewhere", "workers": list_workers()}
        durable_recovery: dict[str, int] = {}
        try:
            try:
                workers = register_default_workers()
            except Exception as exc:  # pragma: no cover - SQLite compatibility
                try:
                    db.rollback()
                except Exception:
                    pass
                from orchestrator.pipeline import log_event

                log_event(
                    "warning",
                    {
                        "step": "gpu_scheduler_register_workers",
                        "error": str(exc),
                    },
                )
                try:
                    workers = list_workers()
                except Exception:
                    workers = []
        except Exception:
            _release_process_lock()
            raise
        recovered = (
            recover_stale_local_running_jobs(workers)
            + recover_stale_ssh_running_jobs()
            + recover_busy_workers_without_running_jobs()
        )
        # Backend-aware stale recovery must run before generic durable timeout
        # quarantine.  It can prove that an SSH process ended and settle the
        # canonical attempt; doing this in the opposite order loses usage as
        # ``usage_unknown`` during a controller restart.
        if db._use_pg():  # noqa: SLF001
            from orchestrator.meta_compute_runtime import reconcile_on_startup

            durable_recovery = reconcile_on_startup()
        _stop_event.clear()
        _scheduler_thread = threading.Thread(target=_loop, daemon=True, name="deepgraph-gpu-scheduler")
        _scheduler_thread.start()
        colab_worker_status = {"status": "disabled"}
        try:
            from orchestrator.meta_compute_runtime import _enabled_backend_kinds

            if "colab_gpu" in _enabled_backend_kinds():
                from orchestrator.colab_worker import start as start_colab_worker

                colab_worker_status = start_colab_worker()
        except Exception:
            _stop_event.set()
            _release_process_lock()
            raise
    return {
        "status": "started",
        "workers": list_workers(),
        "recovered_stale_jobs": recovered,
        "durable_recovery": durable_recovery,
        "colab_worker": colab_worker_status,
    }


def stop() -> dict:
    _stop_event.set()
    try:
        from orchestrator.colab_worker import stop as stop_colab_worker

        stop_colab_worker()
    except Exception:
        pass
    _release_process_lock()
    return {"status": "stopping"}
