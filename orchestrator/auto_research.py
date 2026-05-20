"""Auto Research: closed-loop background orchestration for deep insights.

Flow:
1. Pick promising Tier-2 deep insights
2. Run EvoScientist verification / deep research (optional unless
   DEEPGRAPH_REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS=true)
3. Route experiments into CPU / GPU lanes
4. Forge and execute SciForge experiments
5. Feed results back into the graph and expose status to the dashboard
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from agents.discovery_metadata import infer_experimentability, infer_resource_class
from agents.experiment_forge import forge_experiment
from agents.insight_validation import (
    INSIGHT_INPUT_MISSING_ERROR_CODE,
    get_evosci_input_issue,
)
from agents.novelty_verifier import (
    check_verification_result,
    launch_full_research,
    launch_verification,
)
from agents.research_bridge import active_research_session, get_research_status
from orchestrator import experiment_runner
from orchestrator.compute_routing import resolve_execution_lane
from compat.filelock import FileLock
from agents.evosci_requirements import (
    evosci_binary_path,
    evosci_installed,
    final_report_ready,
)
from config import (
    AUTO_RESEARCH_INTERVAL_SECONDS,
    AUTO_RESEARCH_MAX_ACTIVE,
    REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS,
)
from db import database as db
from db.insight_outcomes import (
    OUTCOME_EXPERIMENT_FAILED_RUN,
    OUTCOME_EXPERIMENT_FAILED_SETUP,
    apply_experiment_finished_deep,
    set_outcome,
)
from orchestrator.benchmark_completion import (
    BENCHMARK_COMPLETION_STAGE,
    schedule_benchmark_completion,
)
from orchestrator import gpu_scheduler
from orchestrator import manuscript_watchdog
from orchestrator.pipeline import log_event

_worker_thread: threading.Thread | None = None
_worker_lock = threading.Lock()
_stop_event = threading.Event()
_process_lock: FileLock | None = None
AUTO_RESEARCH_CONSUMER = "auto_research"
VERIFY_STALE_SECONDS = 60 * 60
RESEARCH_STALE_SECONDS = 6 * 60 * 60
REVIEW_PENDING_STALE_SECONDS = 15 * 60
MAX_PARALLEL_VERIFICATIONS = 2
MANUAL_REFORGE_STAGES = {
    "manual_reforge_unfinished",
    "manual_requeue_unfinished",
    "retry_failed_run",
    "review_retry",
}
MANUAL_RERUN_COMPLETED_STAGES = {
    BENCHMARK_COMPLETION_STAGE,
    "manual_rerun_completed",
    "paper_blocked_benchmark_completion",
    "manuscript_blocked",
    "reset_completed_experiments",
}
IGNORED_EXISTING_RUN_STATUSES = {"superseded", "reset", "archived", "cancelled"}

HEAVY_KEYWORDS = {
    "llm", "gpt", "llama", "mistral", "diffusion", "stable diffusion",
    "video", "multimodal", "vision-language", "vlm", "7b", "13b", "70b",
    "gpu", "pretrain", "pre-training", "billion", "transformer-xl",
}


def evosci_available() -> bool:
    return evosci_installed()


def _load_json(value: str | None, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _run_is_formal(run: dict | None) -> bool:
    if not run:
        return False
    proxy = _load_json(run.get("proxy_config"), {})
    return bool(proxy.get("formal_experiment")) and not bool(proxy.get("smoke_test_only"))


def _run_review_decision_ready(run: dict | None) -> bool:
    if not run:
        return False
    proxy = _load_json(run.get("proxy_config"), {})
    return "formal_experiment" in proxy or "smoke_test_only" in proxy


def _run_scaffold_ready(run: dict | None) -> bool:
    if not run:
        return False
    if not _run_review_decision_ready(run):
        return False
    return bool(
        (run.get("workdir") or "").strip()
        and (run.get("program_md") or "").strip()
        and (run.get("success_criteria") or "").strip()
    )


def _manual_reforge_requested(insight: dict, run: dict | None) -> bool:
    if not run:
        return False
    stage = str(insight.get("auto_stage") or "").strip()
    if stage == BENCHMARK_COMPLETION_STAGE:
        return False
    status = str(run.get("status") or "").strip()
    if stage in MANUAL_RERUN_COMPLETED_STAGES:
        return status in {"completed", "failed", "bundle_ready", "superseded"}
    if stage not in MANUAL_REFORGE_STAGES:
        return False
    if status == "failed":
        return True
    if status == "scaffolding" and not _run_scaffold_ready(run):
        return True
    return False


def _auto_job_stage(insight_id: int) -> str:
    row = db.fetchone("SELECT stage FROM auto_research_jobs WHERE deep_insight_id=?", (insight_id,))
    return str((row or {}).get("stage") or "").strip()


def _queue_benchmark_completion_run(insight_id: int, run: dict, resource_class: str) -> bool:
    queued_job = db.fetchone(
        """
        SELECT * FROM gpu_jobs
        WHERE experiment_run_id=? AND status IN ('queued', 'running')
        ORDER BY id DESC LIMIT 1
        """,
        (run["id"],),
    )
    if queued_job:
        note = f"Full benchmark completion GPU job {queued_job['id']} already {queued_job['status']}."
    else:
        gpu_scheduler.start()
        gpu_job_id = gpu_scheduler.queue_run(
            insight_id=insight_id,
            run_id=run["id"],
            resource_class=resource_class,
            priority=3,
            vram_required_gb=40 if resource_class == "gpu_large" else 16,
            timeout_s=None,
        )
        note = f"Queued full benchmark completion on GPU scheduler as job {gpu_job_id}."
    _upsert_job(
        insight_id,
        status="queued_gpu",
        stage=BENCHMARK_COMPLETION_STAGE,
        experiment_run_id=run["id"],
        resource_class=resource_class,
        assigned_worker=None,
        last_note=note,
        last_error=None,
    )
    return True


def _run_reusable_for_auto_research(run: dict | None) -> bool:
    if not run:
        return False
    return str(run.get("status") or "").strip() not in IGNORED_EXISTING_RUN_STATUSES


def _existing_run_for_candidate(insight: dict) -> dict | None:
    insight_id = int(insight["id"])
    canonical_run_id = insight.get("canonical_run_id")
    if canonical_run_id:
        run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (canonical_run_id,))
        if _run_reusable_for_auto_research(run):
            return run
    return db.fetchone(
        """
        SELECT * FROM experiment_runs
        WHERE deep_insight_id=?
          AND COALESCE(status, '') NOT IN ('superseded', 'reset', 'archived', 'cancelled')
        ORDER BY id DESC LIMIT 1
        """,
        (insight_id,),
    )


def _coerce_datetime(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _job_age_seconds(job: dict) -> float:
    ts = _coerce_datetime(
        job.get("updated_at") or job.get("last_checked_at") or job.get("created_at")
    )
    if ts is None:
        return 0.0
    now = datetime.now(ts.tzinfo) if ts.tzinfo else datetime.now()
    return max(0.0, (now - ts).total_seconds())


def _supersede_stale_scaffold_run(run_id: int, reason: str) -> None:
    db.execute(
        """
        UPDATE experiment_runs
        SET status='superseded',
            phase='superseded',
            error_message=?,
            completed_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (reason, run_id),
    )
    db.commit()


def _upsert_job(insight_id: int, **fields) -> None:
    existing = db.fetchone(
        "SELECT id FROM auto_research_jobs WHERE deep_insight_id=?",
        (insight_id,),
    )
    fields["updated_at"] = "CURRENT_TIMESTAMP"
    fields["last_checked_at"] = "CURRENT_TIMESTAMP"

    if existing:
        assigns = []
        params = []
        for key, value in fields.items():
            if value == "CURRENT_TIMESTAMP":
                assigns.append(f"{key}=CURRENT_TIMESTAMP")
            else:
                assigns.append(f"{key}=?")
                params.append(value)
        params.append(insight_id)
        db.execute(
            f"UPDATE auto_research_jobs SET {', '.join(assigns)} WHERE deep_insight_id=?",
            tuple(params),
        )
    else:
        cols = ["deep_insight_id"]
        placeholders = ["?"]
        params = [insight_id]
        for key, value in fields.items():
            cols.append(key)
            if value == "CURRENT_TIMESTAMP":
                placeholders.append("CURRENT_TIMESTAMP")
            else:
                placeholders.append("?")
                params.append(value)
        db.execute(
            f"INSERT INTO auto_research_jobs ({', '.join(cols)}) VALUES ({', '.join(placeholders)})",
            tuple(params),
        )
    db.commit()


def _parse_gpu_hours(plan: dict) -> float | None:
    compute = plan.get("compute_budget", {}) if isinstance(plan, dict) else {}
    raw = (
        compute.get("total_gpu_hours")
        or compute.get("gpu_hours")
        or compute.get("gpu")
    )
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip().lower()
    num = []
    for ch in text:
        if ch.isdigit() or ch == ".":
            num.append(ch)
        elif num:
            break
    if not num:
        return None
    try:
        return float("".join(num))
    except ValueError:
        return None


def assess_experiment_route(insight: dict) -> tuple[str, str]:
    """Route insights into cpu / gpu_small / gpu_large lanes."""
    resource_class = infer_resource_class(insight)
    experimentability = infer_experimentability(insight)
    return resource_class, f"Experimentability={experimentability}; routed to {resource_class}."


def _dispatch_experiment_execution(
    *,
    insight_id: int,
    existing_run: dict,
    resource_class: str,
) -> None:
    """Queue GPU or async CPU validation; returns immediately (no validation_loop join)."""
    run_id = int(existing_run["id"])
    if experiment_runner.is_run_active(run_id):
        _upsert_job(
            insight_id,
            status="running_cpu",
            stage="validation_loop",
            experiment_run_id=run_id,
            resource_class=resource_class,
            last_note="Async CPU validation already in progress.",
            last_error=None,
        )
        return

    resource_class, lane, lane_note = resolve_execution_lane(resource_class=resource_class, run=existing_run)
    if lane_note:
        db.execute(
            "UPDATE experiment_runs SET resource_class=? WHERE id=?",
            (resource_class, run_id),
        )
        db.execute(
            """UPDATE deep_insights
               SET resource_class=?, updated_at=CURRENT_TIMESTAMP
               WHERE id=?""",
            (resource_class, insight_id),
        )
        db.commit()

    if lane == "gpu":
        gpu_scheduler.start()
        queued_job = db.fetchone(
            """
            SELECT * FROM gpu_jobs
            WHERE experiment_run_id=? AND status IN ('queued', 'running')
            ORDER BY id DESC LIMIT 1
            """,
            (run_id,),
        )
        if not queued_job:
            gpu_job_id = gpu_scheduler.queue_run(
                insight_id=insight_id,
                run_id=run_id,
                resource_class=resource_class,
                priority=2 if resource_class == "gpu_large" else 1,
                vram_required_gb=40 if resource_class == "gpu_large" else 16,
            )
            note = f"Queued on GPU scheduler as job {gpu_job_id}."
        else:
            note = f"GPU job {queued_job['id']} already {queued_job['status']}."
        if lane_note:
            note = f"{note} {lane_note}"
        _upsert_job(
            insight_id,
            status="queued_gpu",
            stage="gpu_scheduler",
            experiment_run_id=run_id,
            resource_class=resource_class,
            last_note=note,
            last_error=None,
        )
        log_event("auto_research", {"step": "gpu_job_queued", "insight_id": insight_id, "run_id": run_id})
        return

    _upsert_job(
        insight_id,
        status="running_cpu",
        stage="validation_loop",
        experiment_run_id=run_id,
        resource_class=resource_class,
        last_note="Dispatched async SciForge validation loop (non-blocking).",
        last_error=None,
    )
    log_event("auto_research", {"step": "experiment_started_async", "insight_id": insight_id, "run_id": run_id})
    experiment_runner.start_validation_loop_async(
        insight_id=insight_id,
        run_id=run_id,
        resource_class=resource_class,
    )


def _research_report_ready(workdir: str | None) -> bool:
    if not workdir:
        return False
    return (Path(workdir) / "final_report.md").exists()


def _try_acquire_process_lock() -> bool:
    global _process_lock
    if _process_lock is not None:
        return True
    lock_path = (
        Path(os.environ.get("TEMP", str(Path.home() / ".cache"))) / "deepgraph-auto-research.lock"
        if os.name == "nt"
        else Path("/tmp/deepgraph-auto-research.lock")
    )
    lock = FileLock(str(lock_path))
    if not lock.try_acquire():
        return False
    try:
        handle = getattr(lock, "_handle")
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
    except OSError:
        lock.release()
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


def list_jobs(limit: int = 50) -> list[dict]:
    db.init_db()
    rows = db.fetchall(
        """SELECT arj.*, di.title, di.tier, di.status AS insight_status,
                  di.novelty_status, di.created_at AS insight_created_at,
                  er.status AS experiment_status, er.hypothesis_verdict,
                  er.effect_pct
           FROM auto_research_jobs arj
           JOIN deep_insights di ON di.id = arj.deep_insight_id
           LEFT JOIN experiment_runs er ON er.id = arj.experiment_run_id
           ORDER BY arj.updated_at DESC
           LIMIT ?""",
        (limit,),
    )
    return rows


def get_status() -> dict:
    with _worker_lock:
        running = bool(_worker_thread and _worker_thread.is_alive())
    counts = db.fetchone(
        """SELECT
             COUNT(*) AS total,
             SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) AS completed,
             SUM(CASE WHEN status='running_experiment' THEN 1 ELSE 0 END) AS running_experiment,
             SUM(CASE WHEN status='queued_gpu' THEN 1 ELSE 0 END) AS queued_gpu,
             SUM(CASE WHEN status='running_gpu' THEN 1 ELSE 0 END) AS running_gpu,
             SUM(CASE WHEN status='verifying' THEN 1 ELSE 0 END) AS verifying,
             SUM(CASE WHEN status='researching' THEN 1 ELSE 0 END) AS researching,
             SUM(CASE WHEN status='review_pending' THEN 1 ELSE 0 END) AS review_pending,
             SUM(CASE WHEN status='smoke_only' THEN 1 ELSE 0 END) AS smoke_only,
             SUM(CASE WHEN status='blocked' THEN 1 ELSE 0 END) AS blocked,
             SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed
           FROM auto_research_jobs"""
    ) or {}
    return {
        "running": running,
        "interval_seconds": AUTO_RESEARCH_INTERVAL_SECONDS,
        "max_active": AUTO_RESEARCH_MAX_ACTIVE,
        "evoscientist_available": evosci_available(),
        **counts,
    }


def _execution_active_job_count() -> int:
    row = db.fetchone(
        """SELECT COUNT(*) AS c
           FROM auto_research_jobs
           WHERE status IN ('running_experiment', 'running_gpu', 'running_cpu', 'queued_gpu')"""
    )
    db_count = row["c"] if row else 0
    # Include in-flight async lanes not yet reflected in ARJ status.
    thread_active = experiment_runner.active_execution_count()
    return max(db_count, thread_active)


def _research_job_count() -> int:
    row = db.fetchone(
        """SELECT COUNT(*) AS c
           FROM auto_research_jobs
           WHERE status IN ('researching')"""
    )
    return row["c"] if row else 0


def _verification_job_count() -> int:
    row = db.fetchone(
        """SELECT COUNT(*) AS c
           FROM auto_research_jobs
           WHERE status IN ('verifying')"""
    )
    return row["c"] if row else 0


def _active_job_count() -> int:
    return _execution_active_job_count() + _verification_job_count() + _research_job_count()


def _candidate_pool() -> list[dict]:
    rows = db.fetchall(
        """SELECT di.*,
                  arj.status AS auto_status,
                  arj.stage AS auto_stage,
                  arj.cpu_eligible AS auto_cpu_eligible
           FROM deep_insights di
           LEFT JOIN auto_research_jobs arj ON arj.deep_insight_id = di.id
           WHERE COALESCE(di.status, 'candidate') NOT IN ('exists')
             AND (
               arj.status IS NULL
               OR arj.status IN ('queued', 'eligible', 'queued_cpu', 'queued_gpu', 'pending')
               OR (
                    arj.status='failed'
                    AND arj.stage IN (
                        'manual_reforge_unfinished',
                        'manual_requeue_unfinished',
                        'retry_failed_run',
                        'manual_rerun_completed',
                        'reset_completed_experiments'
                    )
                  )
               OR (
                    arj.status='completed'
                    AND arj.stage='tier1_research_complete'
                    AND NOT EXISTS (
                        SELECT 1 FROM experiment_runs er
                        WHERE er.deep_insight_id = di.id
                    )
                  )
               OR (
                    arj.status='blocked'
                    AND (
                      arj.stage='cpu_ineligible'
                      OR arj.stage IN ('verification_input_missing', 'research_input_missing')
                    )
                  )
             )
           ORDER BY di.tier DESC, di.created_at DESC
           LIMIT 20"""
    )
    return rows


def _resource_experimentability(resource_class: str) -> str:
    if resource_class == "cpu":
        return "easy"
    if resource_class == "gpu_small":
        return "medium"
    return "hard"


def _route_recovered_legacy_job(insight: dict) -> tuple[str, str]:
    resource_class, reason = assess_experiment_route(insight)
    note = str(insight.get("auto_last_note") or "").lower()
    if resource_class == "cpu" and ("gpu-heavy" in note or "looks gpu" in note):
        resource_class = "gpu_small"
        reason = f"{reason} Legacy cpu_ineligible note indicates GPU-heavy; routed to gpu_small."
    return resource_class, reason


def recover_legacy_cpu_ineligible_jobs(limit: int = 50) -> int:
    """Requeue jobs blocked by the pre-GPU-era CPU-only filter."""
    rows = db.fetchall(
        """SELECT di.*, arj.last_note AS auto_last_note
           FROM auto_research_jobs arj
           JOIN deep_insights di ON di.id = arj.deep_insight_id
           WHERE arj.status='blocked'
             AND arj.stage='cpu_ineligible'
             AND COALESCE(di.status, 'candidate') NOT IN ('exists')
           ORDER BY arj.updated_at ASC
           LIMIT ?""",
        (limit,),
    )
    recovered = 0
    for insight in rows:
        insight_id = int(insight["id"])
        resource_class, reason = _route_recovered_legacy_job(dict(insight))
        experimentability = _resource_experimentability(resource_class)
        db.execute(
            """UPDATE deep_insights
               SET resource_class=?, experimentability=?, updated_at=CURRENT_TIMESTAMP
               WHERE id=?""",
            (resource_class, experimentability, insight_id),
        )
        _upsert_job(
            insight_id,
            status="queued",
            stage="legacy_gpu_requeue",
            cpu_eligible=1,
            cpu_reason=reason,
            resource_class=resource_class,
            scheduler_priority=2 if resource_class == "gpu_large" else 1,
            last_error=None,
            last_note="Recovered from legacy cpu_ineligible block; waiting for Auto Research scheduling.",
        )
        log_event(
            "auto_research",
            {"step": "legacy_cpu_ineligible_recovered", "insight_id": insight_id, "resource_class": resource_class},
        )
        recovered += 1
    if recovered:
        db.commit()
    return recovered


def _candidate_needs_verification(candidate: dict) -> bool:
    novelty = (candidate.get("novelty_status") or "unchecked").strip()
    return evosci_available() and novelty in {"", "unchecked"}


def _candidate_still_missing_required_inputs(candidate: dict) -> bool:
    if candidate.get("auto_status") != "blocked":
        return False
    stage = (candidate.get("auto_stage") or "").strip()
    if stage not in {"verification_input_missing", "research_input_missing"}:
        return False
    mode = "verification" if stage == "verification_input_missing" else "research"
    return get_evosci_input_issue(candidate, mode=mode) is not None


def _next_candidate() -> dict | None:
    execution_active = _execution_active_job_count()
    verifying_active = _verification_job_count()
    max_active = max(1, AUTO_RESEARCH_MAX_ACTIVE)
    for candidate in _candidate_pool():
        if _candidate_still_missing_required_inputs(candidate):
            continue
        if _candidate_needs_verification(candidate):
            if verifying_active < MAX_PARALLEL_VERIFICATIONS:
                return candidate
            continue
        if execution_active < max_active:
            return candidate
    return None


def _refresh_running_jobs() -> None:
    try:
        from agents.experiment_watchdog import promote_pending_jobs, run_watchdog_for_active_runs

        promote_pending_jobs()
        for report in run_watchdog_for_active_runs(limit=5):
            run_id = report.get("run_id")
            tuning = report.get("tuning") or {}
            if run_id and tuning.get("action") == "abort":
                db.execute(
                    "UPDATE experiment_runs SET status='failed', error_message=? WHERE id=?",
                    ("Aborted by experiment watchdog.", run_id),
                )
                db.commit()
    except Exception as exc:  # pragma: no cover - watchdog must not break scheduler
        log_event("warning", {"step": "experiment_watchdog_failed", "error": str(exc)})

    jobs = db.fetchall(
        """SELECT arj.*, di.novelty_status
           FROM auto_research_jobs arj
           JOIN deep_insights di ON di.id = arj.deep_insight_id
           WHERE arj.status IN ('verifying', 'researching', 'review_pending', 'running_experiment', 'queued_gpu', 'running_gpu', 'running_cpu')"""
    )
    for job in jobs:
        insight_id = job["deep_insight_id"]
        if job["status"] == "verifying":
            result = check_verification_result(insight_id)
            if result.get("status") == "complete":
                note = f"Novelty verdict: {result.get('verdict', 'unknown')}"
                new_status = "blocked" if result.get("verdict") == "exists" else "queued"
                _upsert_job(insight_id, status=new_status, stage="verification_complete", last_note=note)
            elif result.get("status") == "running" and not REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS:
                _upsert_job(
                    insight_id,
                    status="queued",
                    stage="novelty_verification_background",
                    last_note="Novelty verification is running in background; optional mode proceeds to experiment pipeline.",
                    last_error=None,
                )
            elif result.get("status") == "failed":
                _upsert_job(
                    insight_id,
                    status="failed",
                    stage="verification_failed",
                    last_error=result.get("error") or "Novelty verification exited without a report.",
                    last_note="Novelty verification failed; released slot for retry.",
                )
                log_event(
                    "warning",
                    {"step": "auto_research_verification_failed", "insight_id": insight_id, "error": result.get("error")},
                )
            elif _job_age_seconds(job) >= VERIFY_STALE_SECONDS:
                db.execute(
                    "UPDATE deep_insights SET novelty_status='unchecked', updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (insight_id,),
                )
                db.commit()
                _upsert_job(
                    insight_id,
                    status="failed",
                    stage="verification_stale",
                    last_error="Novelty verification stalled; released slot for retry.",
                )
                log_event("warning", {"step": "auto_research_verification_stale", "insight_id": insight_id})
        elif job["status"] == "researching":
            workdir = job.get("research_workdir")
            if _research_report_ready(workdir):
                _upsert_job(
                    insight_id,
                    status="queued",
                    stage="research_complete",
                    last_note="EvoScientist final_report.md available.",
                )
            elif workdir:
                status = get_research_status(workdir)
                note = f"log lines: {status.get('log_lines', 0)}"
                if note != job.get("last_note"):
                    _upsert_job(
                        insight_id,
                        stage="researching",
                        last_note=note,
                    )
                if _job_age_seconds(job) >= RESEARCH_STALE_SECONDS:
                    _upsert_job(
                        insight_id,
                        status="failed",
                        stage="research_stale",
                        last_error="Deep research stalled; released slot for retry.",
                    )
                    log_event("warning", {"step": "auto_research_research_stale", "insight_id": insight_id})
        elif job["status"] == "review_pending":
            run = None
            if job.get("experiment_run_id"):
                run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (job["experiment_run_id"],))
            if run and _run_scaffold_ready(run):
                _upsert_job(
                    insight_id,
                    status="eligible" if _run_is_formal(run) else "smoke_only",
                    stage="formal_ready" if _run_is_formal(run) else "experiment_review_smoke_only",
                    experiment_run_id=run["id"],
                    resource_class=run.get("resource_class") or job.get("resource_class"),
                    last_error=None,
                    last_note="Recovered scaffold-ready review job and resumed scheduling.",
                )
                log_event(
                    "auto_research",
                    {"step": "auto_research_review_ready_recovered", "insight_id": insight_id, "run_id": run["id"]},
                )
            elif run and str(run.get("status") or "") == "failed":
                _upsert_job(
                    insight_id,
                    status="failed",
                    stage="experiment_review_failed",
                    experiment_run_id=run["id"],
                    last_error=run.get("error_message") or "Experiment forge run failed during review/scaffold.",
                    last_note="Recovered failed review/scaffold run and released scheduler slot.",
                )
                log_event(
                    "warning",
                    {"step": "auto_research_review_failed_recovered", "insight_id": insight_id, "run_id": run["id"]},
                )
            elif run and _job_age_seconds(job) >= REVIEW_PENDING_STALE_SECONDS:
                reason = (
                    "Recovered stale review/scaffold run: no complete review decision, "
                    "program, or success criteria appeared before the stale timeout."
                )
                _supersede_stale_scaffold_run(int(run["id"]), reason)
                _upsert_job(
                    insight_id,
                    status="queued",
                    stage="review_retry",
                    experiment_run_id=None,
                    last_error=None,
                    last_note=reason,
                )
                log_event(
                    "warning",
                    {"step": "auto_research_review_scaffold_stale", "insight_id": insight_id, "run_id": run["id"]},
                )
            elif job.get("last_error") and not job.get("experiment_run_id"):
                note = job.get("last_note") or job["last_error"]
                _upsert_job(
                    insight_id,
                    status="blocked",
                    stage="experiment_review_blocked",
                    last_error=job["last_error"],
                    last_note=note,
                )
            elif _job_age_seconds(job) >= REVIEW_PENDING_STALE_SECONDS and not job.get("experiment_run_id"):
                _upsert_job(
                    insight_id,
                    status="queued",
                    stage="review_retry",
                    last_error=None,
                    last_note="Structured experiment review stalled; requeued for retry.",
                )
                log_event("warning", {"step": "auto_research_review_stale", "insight_id": insight_id})
        elif job["status"] in {"running_experiment", "running_gpu", "running_cpu", "queued_gpu"} and job.get("experiment_run_id"):
            run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (job["experiment_run_id"],))
            if not run:
                _upsert_job(insight_id, status="failed", stage="missing_run", last_error="Experiment run missing.")
            elif run["status"] == "completed":
                note = f"Verdict={run.get('hypothesis_verdict')}, effect_pct={run.get('effect_pct')}"
                _upsert_job(insight_id, status="completed", stage="closed_loop_complete", last_note=note)
                v = (run.get("hypothesis_verdict") or "").lower()
                apply_experiment_finished_deep(
                    insight_id,
                    verdict=run.get("hypothesis_verdict"),
                    success=v == "confirmed",
                    inconclusive=v == "inconclusive",
                )
            elif run["status"] == "failed":
                _upsert_job(insight_id, status="failed", stage="experiment_failed", last_error=run.get("error_message"))
                set_outcome(
                    "deep_insights",
                    insight_id,
                    OUTCOME_EXPERIMENT_FAILED_RUN,
                    reason=run.get("error_message"),
                    triggered_by="experiment",
                )
            elif run["status"] == "running_gpu":
                _upsert_job(insight_id, status="running_gpu", stage="gpu_scheduler", last_note="GPU job running.")
            elif run["status"] == "running_cpu":
                _upsert_job(insight_id, status="running_cpu", stage="validation_loop", last_note="CPU validation loop running.")
            elif run["status"] in {"reproducing", "testing"}:
                lane_status = "running_gpu" if job["status"] in {"running_gpu", "queued_gpu"} else "running_cpu"
                phase = run.get("phase") or "validation_loop"
                note = f"SciForge {phase}: best={run.get('best_metric_value')}, baseline={run.get('baseline_metric_value')}."
                _upsert_job(
                    insight_id,
                    status=lane_status,
                    stage=phase,
                    last_note=note,
                    last_error=None,
                )


def _launch_candidates_to_capacity() -> dict:
    _refresh_running_jobs()
    scheduled: list[int] = []
    seen_candidates: set[int] = set()

    while True:
        candidate = _next_candidate()
        if not candidate:
            break
        candidate_id = int(candidate["id"])
        if candidate_id in seen_candidates:
            break
        seen_candidates.add(candidate_id)
        try:
            _process_candidate(candidate)
            scheduled.append(candidate_id)
        except Exception as exc:  # pragma: no cover - defensive background guard
            _upsert_job(candidate_id, status="failed", stage="exception", last_error=str(exc))
            log_event("error", {"step": "auto_research", "insight_id": candidate_id, "error": str(exc)})
            continue

    return {
        "scheduled": scheduled,
        "active": _active_job_count(),
        "execution_active": _execution_active_job_count(),
        "verifying_active": _verification_job_count(),
    }


def _process_candidate(insight: dict) -> None:
    insight_id = insight["id"]
    tier = insight.get("tier")

    resource_class, reason = assess_experiment_route(insight)
    _upsert_job(
        insight_id,
        cpu_eligible=1,
        cpu_reason=reason,
        resource_class=resource_class,
        scheduler_priority=2 if resource_class == "gpu_large" else 1,
    )

    if REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS:
        fresh = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (insight_id,))
        if fresh:
            insight = dict(fresh)

    if REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS and not evosci_available():
        _upsert_job(
            insight_id,
            status="blocked",
            stage="evosci_binary_missing",
            cpu_eligible=0,
            last_error="EvoScientist is required but EvoSci executable was not found.",
            last_note=(
                f"Install EvoScientist and ensure EvoSci exists at {evosci_binary_path()}, "
                "or set DEEPGRAPH_REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS=false."
            ),
        )
        return

    novelty = (insight.get("novelty_status") or "unchecked").strip()
    if novelty in {"", "unchecked"}:
        if evosci_available():
            verification = launch_verification(insight_id)
            if "error" in verification:
                if verification.get("error_code") == INSIGHT_INPUT_MISSING_ERROR_CODE:
                    missing = ", ".join(verification.get("missing_fields") or [])
                    note = "Waiting for required insight fields before novelty verification can run."
                    if missing:
                        note = f"{note} Missing: {missing}."
                    if not REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS:
                        _upsert_job(
                            insight_id,
                            stage="verification_skipped_input_missing",
                            last_error=verification["error"],
                            last_note=f"{note} Optional mode proceeds to experiment pipeline.",
                        )
                        log_event(
                            "warning",
                            {
                                "step": "verification_input_missing_optional",
                                "insight_id": insight_id,
                                "error": verification["error"],
                                "missing_fields": verification.get("missing_fields", []),
                            },
                        )
                    else:
                        _upsert_job(
                            insight_id,
                            status="blocked",
                            stage="verification_input_missing",
                            cpu_eligible=0,
                            last_error=verification["error"],
                            last_note=note,
                        )
                        log_event(
                            "warning",
                            {
                                "step": "verification_input_missing",
                                "insight_id": insight_id,
                                "error": verification["error"],
                                "missing_fields": verification.get("missing_fields", []),
                            },
                        )
                        return
                elif REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS:
                    _upsert_job(
                        insight_id,
                        status="failed",
                        stage="verification_launch_failed",
                        last_error=verification["error"],
                    )
                    log_event(
                        "error",
                        {
                            "step": "verification_launch_failed",
                            "insight_id": insight_id,
                            "error": verification["error"],
                        },
                    )
                    return
                else:
                    _upsert_job(
                        insight_id,
                        stage="verification_launch_failed_optional",
                        last_error=verification["error"],
                        last_note="Novelty verification failed to launch; optional mode proceeds to experiment pipeline.",
                    )
                    log_event(
                        "warning",
                        {
                            "step": "verification_launch_failed_optional",
                            "insight_id": insight_id,
                            "error": verification["error"],
                        },
                    )
            else:
                _upsert_job(
                    insight_id,
                    status="verifying" if REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS else "queued",
                    stage="novelty_verification" if REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS else "novelty_verification_background",
                    last_note=(
                        "Launched EvoScientist novelty check."
                        if REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS
                        else "Launched EvoScientist novelty check in background; proceeding to experiment pipeline."
                    ),
                    last_error=None,
                )
                log_event("auto_research", {"step": "verification_started", "insight_id": insight_id})
                if REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS:
                    return
        if REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS:
            _upsert_job(
                insight_id,
                status="blocked",
                stage="novelty_verification_required",
                cpu_eligible=0,
                last_error="Novelty verification requires EvoScientist but EvoSci was not found.",
                last_note="Install EvoScientist or disable DEEPGRAPH_REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS.",
            )
            return

    if novelty == "verifying":
        if REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS:
            _upsert_job(
                insight_id,
                status="verifying",
                stage="novelty_verification",
                last_note="Novelty verification still running.",
                last_error=None,
            )
            return
        _upsert_job(
            insight_id,
            stage="novelty_verification_background",
            last_note="Novelty verification still running in background; proceeding to experiment pipeline.",
            last_error=None,
        )
    if novelty == "exists":
        _upsert_job(insight_id, status="blocked", stage="prior_work_exists", last_note="Insight already exists in prior work.")
        return

    if REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS:
        if novelty == "partially_exists":
            _upsert_job(
                insight_id,
                status="blocked",
                stage="novelty_partially_exists",
                cpu_eligible=0,
                last_note="novelty_status=partially_exists is not sufficient for SciForge in strict EvoScientist mode.",
            )
            return
        if novelty != "novel":
            _upsert_job(
                insight_id,
                status="blocked",
                stage="novelty_not_novel",
                cpu_eligible=0,
                last_note=f"Novelty status {novelty!r}; strict mode requires 'novel' after EvoScientist verification.",
            )
            return

    background_research_note = None
    if REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS:
        workdir = str(insight.get("evoscientist_workdir") or "").strip()
        if not _research_report_ready(workdir):
            sess = active_research_session(workdir) if workdir else None
            if sess:
                _upsert_job(
                    insight_id,
                    status="researching",
                    stage="evosci_deep_research_running",
                    last_note="Waiting for EvoScientist final_report.md before SciForge.",
                    last_error=None,
                )
                return
            result = launch_full_research(insight_id)
            if "error" in result:
                if result.get("error_code") == INSIGHT_INPUT_MISSING_ERROR_CODE:
                    missing = ", ".join(result.get("missing_fields") or [])
                    note = "Waiting for required insight fields before EvoScientist deep research can run."
                    if missing:
                        note = f"{note} Missing: {missing}."
                    _upsert_job(
                        insight_id,
                        status="blocked",
                        stage="deep_research_input_missing",
                        cpu_eligible=0,
                        last_error=result["error"],
                        last_note=note,
                    )
                    log_event(
                        "warning",
                        {
                            "step": "deep_research_input_missing",
                            "insight_id": insight_id,
                            "error": result["error"],
                            "missing_fields": result.get("missing_fields", []),
                        },
                    )
                    return
                _upsert_job(
                    insight_id,
                    status="failed",
                    stage="deep_research_launch_failed",
                    last_error=result["error"],
                    last_note="EvoScientist deep research failed to launch (strict mode stops here).",
                )
                log_event(
                    "error",
                    {
                        "step": "deep_research_launch_failed",
                        "insight_id": insight_id,
                        "error": result["error"],
                    },
                )
                return
            reused = bool(result.get("reused"))
            _upsert_job(
                insight_id,
                status="researching",
                stage="evosci_deep_research_running" if reused else "evosci_deep_research_started",
                research_workdir=result.get("workdir"),
                last_note=(
                    "Reusing active EvoScientist session; waiting for final_report.md before SciForge."
                    if reused
                    else "Launched EvoScientist deep research; waiting for final_report.md before SciForge."
                ),
                last_error=None,
            )
            log_event("auto_research", {"step": "deep_research_started", "insight_id": insight_id})
            return
        background_research_note = "EvoScientist final_report.md ready; proceeding to experiment forge."
    else:
        if evosci_available():
            workdir = insight.get("evoscientist_workdir")
            if not _research_report_ready(workdir):
                result = launch_full_research(insight_id)
                if "error" in result:
                    if result.get("error_code") == INSIGHT_INPUT_MISSING_ERROR_CODE:
                        missing = ", ".join(result.get("missing_fields") or [])
                        note = "Waiting for required insight fields before deep research can run."
                        if missing:
                            note = f"{note} Missing: {missing}."
                        background_research_note = f"{note} Continuing to experiment pipeline without deep research report."
                        _upsert_job(
                            insight_id,
                            stage="research_skipped_input_missing",
                            last_error=result["error"],
                            last_note=background_research_note,
                        )
                        log_event(
                            "warning",
                            {
                                "step": "deep_research_input_missing",
                                "insight_id": insight_id,
                                "error": result["error"],
                                "missing_fields": result.get("missing_fields", []),
                            },
                        )
                    else:
                        _upsert_job(
                            insight_id,
                            stage="research_launch_failed",
                            last_error=result["error"],
                            last_note="Deep research launch failed; continuing to experiment pipeline.",
                        )
                        log_event(
                            "error",
                            {
                                "step": "deep_research_launch_failed",
                                "insight_id": insight_id,
                                "error": result["error"],
                            },
                        )
                        background_research_note = "Deep research launch failed; continuing to experiment pipeline."
                else:
                    reused_research = bool(result.get("reused"))
                    background_research_note = (
                        "Reusing active EvoScientist deep research while continuing experiment pipeline."
                        if reused_research
                        else "Launched EvoScientist deep research in background while continuing experiment pipeline."
                    )
                    _upsert_job(
                        insight_id,
                        stage="deep_research_background",
                        research_workdir=result.get("workdir"),
                        last_note=background_research_note,
                        last_error=None,
                    )
                    log_event("auto_research", {"step": "deep_research_started", "insight_id": insight_id})
        else:
            _upsert_job(
                insight_id,
                stage="research_unavailable",
                last_note="EvoScientist binary not found; continuing with experiment-only path.",
            )

    existing_run = _existing_run_for_candidate(insight)
    if _manual_reforge_requested(insight, existing_run):
        _upsert_job(
            insight_id,
            stage="reforge_from_unfinished_run",
            experiment_run_id=None,
            last_note=f"Ignoring unfinished run {existing_run['id']} and forging a fresh experiment run.",
            last_error=None,
        )
        existing_run = None
    if not existing_run:
        _upsert_job(
            insight_id,
            status="review_pending",
            stage="experiment_review",
            last_note=background_research_note or "Running structured experiment review before forge.",
        )
        forged = forge_experiment(insight_id)
        if "error" in forged:
            route = forged.get("route")
            if route == "blocked":
                _upsert_job(
                    insight_id,
                    status="blocked",
                    stage="experiment_review_blocked",
                    last_error=forged["error"],
                    last_note=(forged.get("judgement") or {}).get("summary") if isinstance(forged.get("judgement"), dict) else forged["error"],
                )
                return
            _upsert_job(insight_id, status="failed", stage="forge_failed", last_error=forged["error"])
            set_outcome(
                "deep_insights",
                insight_id,
                OUTCOME_EXPERIMENT_FAILED_SETUP,
                reason=str(forged.get("error", "")),
                triggered_by="experiment",
            )
            return
        existing_run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (forged["run_id"],))
        if forged.get("smoke_test_only") or not forged.get("formal_experiment"):
            _upsert_job(
                insight_id,
                status="smoke_only",
                stage="experiment_review_smoke_only",
                experiment_run_id=forged["run_id"],
                resource_class=resource_class,
                last_note=(forged.get("judgement") or {}).get(
                    "summary",
                    "Experiment is smoke-test only; continuing with compute validation (formal manuscript path remains blocked).",
                ),
            )
        else:
            _upsert_job(
                insight_id,
                status="eligible",
                stage="formal_ready",
                experiment_run_id=forged["run_id"],
                resource_class=resource_class,
                last_note="Structured review passed and experiment was forged.",
            )
        db.execute(
            "UPDATE experiment_runs SET resource_class=? WHERE id=?",
            (resource_class, forged["run_id"]),
        )
        db.commit()
    elif not _run_scaffold_ready(existing_run) and existing_run.get("status") in {"scaffolding"}:
        _upsert_job(
            insight_id,
            status="review_pending",
            stage="experiment_review",
            experiment_run_id=existing_run["id"],
            resource_class=resource_class,
            last_note="Experiment forge is still preparing workspace, review, or scaffold metadata.",
            last_error=None,
        )
        return
    elif not _run_is_formal(existing_run):
        _upsert_job(
            insight_id,
            status="smoke_only",
            stage="experiment_review_smoke_only",
            experiment_run_id=existing_run["id"],
            resource_class=resource_class,
            last_note="Existing experiment run is marked non-formal; continuing with compute validation (formal manuscript path remains blocked).",
        )
        db.execute(
            "UPDATE experiment_runs SET resource_class=? WHERE id=?",
            (resource_class, existing_run["id"]),
        )
        db.commit()

    if existing_run["status"] in {"completed"} and _auto_job_stage(insight_id) == BENCHMARK_COMPLETION_STAGE:
        _queue_benchmark_completion_run(insight_id, existing_run, resource_class)
        log_event(
            "auto_research",
            {
                "step": "benchmark_completion_gpu_queued",
                "insight_id": insight_id,
                "run_id": existing_run["id"],
            },
        )
        return

    if existing_run["status"] in {"completed"}:
        note = f"Verdict={existing_run.get('hypothesis_verdict')}, effect_pct={existing_run.get('effect_pct')}"
        _upsert_job(insight_id, status="completed", stage="closed_loop_complete", experiment_run_id=existing_run["id"], last_note=note)
        return
    if existing_run["status"] in {"failed"}:
        _upsert_job(insight_id, status="failed", stage="experiment_failed", experiment_run_id=existing_run["id"], last_error=existing_run.get("error_message"))
        return

    if REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS:
        fresh_exec = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (insight_id,))
        if fresh_exec:
            insight = dict(fresh_exec)

    if REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS and not final_report_ready(insight):
        _upsert_job(
            insight_id,
            status="blocked",
            stage="evosci_report_required_before_compute",
            experiment_run_id=existing_run["id"],
            resource_class=resource_class,
            last_note="final_report.md required before GPU/CPU SciForge execution (strict EvoScientist mode).",
            last_error="EvoScientist deep research report not ready.",
        )
        return

    _dispatch_experiment_execution(
        insight_id=insight_id,
        existing_run=existing_run,
        resource_class=resource_class,
    )


def consume_pipeline_events_once(limit: int = 50) -> dict:
    db.init_db()
    events = db.fetch_pipeline_events(
        AUTO_RESEARCH_CONSUMER,
        limit=limit,
        event_types=[
            "deep_insight_created",
            "experiment_run_completed",
            "submission_bundle_ready",
            "gpu_job_completed",
            "gpu_job_failed",
            "benchmark_completion_required",
        ],
    )
    if not events:
        return {"events": 0}

    processed = 0
    last_event_id = 0
    for event in events:
        last_event_id = int(event["id"])
        payload = db._load_json(event.get("payload"), {})
        event_type = event.get("event_type")
        if event_type == "deep_insight_created" and _active_job_count() < max(1, AUTO_RESEARCH_MAX_ACTIVE):
            insight_id = payload.get("insight_id")
            insight = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (insight_id,))
            if insight:
                _process_candidate(insight)
                processed += 1
        else:
            _refresh_running_jobs()
            processed += 1
    db.ack_pipeline_events(AUTO_RESEARCH_CONSUMER, last_event_id)
    return {"events": len(events), "processed": processed}


def run_cycle() -> dict:
    db.init_db()
    recovered = recover_legacy_cpu_ineligible_jobs()
    try:
        manuscript_audit = manuscript_watchdog.audit_ready_submission_bundles(limit=50, mark_stale=True)
    except Exception as exc:  # pragma: no cover - defensive background guard
        manuscript_audit = {"error": str(exc)}
        log_event("warning", {"step": "manuscript_watchdog_failed", "error": str(exc)})
    launch_stats = _launch_candidates_to_capacity()
    if launch_stats["scheduled"]:
        return {
            "status": "processed",
            "insight_ids": launch_stats["scheduled"],
            "recovered_legacy": recovered,
            "manuscript_audit": manuscript_audit,
        }
    if launch_stats["execution_active"] >= max(1, AUTO_RESEARCH_MAX_ACTIVE) or launch_stats["verifying_active"] >= MAX_PARALLEL_VERIFICATIONS:
        return {"status": "busy", "recovered_legacy": recovered, "manuscript_audit": manuscript_audit}
    if not _next_candidate():
        return {"status": "idle", "recovered_legacy": recovered, "manuscript_audit": manuscript_audit}
    return {"status": "pending", "recovered_legacy": recovered, "manuscript_audit": manuscript_audit}


def _run_once() -> dict:
    db.init_db()
    event_stats = consume_pipeline_events_once(limit=50)
    cycle_stats = run_cycle()
    active = _active_job_count()
    return {
        "events": event_stats.get("events", 0),
        "cycle_status": cycle_stats.get("status"),
        "manuscript_audit": cycle_stats.get("manuscript_audit"),
        "active_jobs": active,
    }


def _run_loop() -> None:
    while not _stop_event.is_set():
        try:
            stats = _run_once()
            sleep_s = (
                1
                if stats.get("events")
                or stats.get("active_jobs")
                or stats.get("cycle_status") in {"processed", "busy", "pending"}
                else max(5, AUTO_RESEARCH_INTERVAL_SECONDS)
            )
        except Exception as exc:  # pragma: no cover - defensive background guard
            try:
                db.rollback()
            except Exception:
                pass
            log_event("error", {"step": "auto_research_loop", "error": str(exc)})
            sleep_s = max(5, AUTO_RESEARCH_INTERVAL_SECONDS)
        _stop_event.wait(sleep_s)


def start() -> dict:
    global _worker_thread
    db.init_db()
    with _worker_lock:
        if _worker_thread and _worker_thread.is_alive():
            return {"status": "already_running"}
        if not _try_acquire_process_lock():
            return {"status": "already_running_elsewhere"}
        _stop_event.clear()
        _worker_thread = threading.Thread(target=_run_loop, daemon=True, name="deepgraph-auto-research")
        _worker_thread.start()
    log_event("auto_research", {"step": "started"})
    return {"status": "started"}


def stop() -> dict:
    _stop_event.set()
    _release_process_lock()
    log_event("auto_research", {"step": "stopped"})
    return {"status": "stopping"}
