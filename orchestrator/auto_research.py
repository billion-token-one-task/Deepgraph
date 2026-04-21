"""Auto Research: closed-loop background orchestration for deep insights.

Flow:
1. Pick promising Tier-2 deep insights
2. Optionally run EvoScientist verification / deep research if available
3. Filter for CPU/no-GPU experimentability
4. Forge and execute SciForge experiments
5. Feed results back into the graph and expose status to the dashboard
"""
from __future__ import annotations

import json
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
from agents.knowledge_loop import process_completed_run
from agents.manuscript_pipeline import generate_submission_bundle
from agents.novelty_verifier import (
    check_verification_result,
    launch_full_research,
    launch_verification,
)
from agents.research_bridge import get_research_status
from agents.validation_loop import run_validation_loop
from config import AUTO_RESEARCH_INTERVAL_SECONDS, AUTO_RESEARCH_MAX_ACTIVE
from db import database as db
from db.insight_outcomes import (
    OUTCOME_EXPERIMENT_FAILED_RUN,
    OUTCOME_EXPERIMENT_FAILED_SETUP,
    apply_experiment_finished_deep,
    set_outcome,
)
from orchestrator import gpu_scheduler
from orchestrator.pipeline import log_event

_worker_thread: threading.Thread | None = None
_worker_lock = threading.Lock()
_stop_event = threading.Event()
AUTO_RESEARCH_CONSUMER = "auto_research"
VERIFY_STALE_SECONDS = 60 * 60
RESEARCH_STALE_SECONDS = 6 * 60 * 60
MAX_PARALLEL_VERIFICATIONS = 2

HEAVY_KEYWORDS = {
    "llm", "gpt", "llama", "mistral", "diffusion", "stable diffusion",
    "video", "multimodal", "vision-language", "vlm", "7b", "13b", "70b",
    "gpu", "pretrain", "pre-training", "billion", "transformer-xl",
}


def _evosci_bin() -> Path:
    return Path.home() / "EvoScientist" / ".venv" / "bin" / "EvoSci"


def evosci_available() -> bool:
    return _evosci_bin().exists()


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
    now = datetime.now(ts.tzinfo) if ts.tzinfo else datetime.utcnow()
    return max(0.0, (now - ts).total_seconds())


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
    tier = insight.get("tier")
    resource_class = infer_resource_class(insight)
    experimentability = infer_experimentability(insight)
    if tier == 1:
        predictions = _load_json(insight.get("predictions"), [])
        if predictions:
            return "cpu", "Tier-1 insight has testable predictions; verify/research on CPU lane."
        return "cpu", "Tier-1 insight is research-only; experiment lane optional."
    return resource_class, f"Experimentability={experimentability}; routed to {resource_class}."


def _research_report_ready(workdir: str | None) -> bool:
    if not workdir:
        return False
    return (Path(workdir) / "final_report.md").exists()


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
           WHERE status IN ('researching', 'running_experiment', 'running_gpu', 'running_cpu', 'queued_gpu')"""
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
    return _execution_active_job_count() + _verification_job_count()


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
               OR arj.status IN ('queued', 'eligible', 'failed', 'queued_cpu', 'queued_gpu')
               OR (
                    arj.status='blocked'
                    AND (
                      arj.cpu_eligible=1
                      OR arj.stage IN ('verification_input_missing', 'research_input_missing')
                    )
                  )
             )
           ORDER BY di.tier DESC, di.created_at DESC
           LIMIT 20"""
    )
    return rows


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
    jobs = db.fetchall(
        """SELECT arj.*, di.novelty_status
           FROM auto_research_jobs arj
           JOIN deep_insights di ON di.id = arj.deep_insight_id
           WHERE arj.status IN ('verifying', 'researching', 'running_experiment', 'queued_gpu', 'running_gpu', 'running_cpu')"""
    )
    for job in jobs:
        insight_id = job["deep_insight_id"]
        if job["status"] == "verifying":
            result = check_verification_result(insight_id)
            if result.get("status") == "complete":
                note = f"Novelty verdict: {result.get('verdict', 'unknown')}"
                new_status = "blocked" if result.get("verdict") == "exists" else "queued"
                _upsert_job(insight_id, status=new_status, stage="verification_complete", last_note=note)
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
            break

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

    novelty = (insight.get("novelty_status") or "unchecked").strip()
    if novelty in {"", "unchecked"} and evosci_available():
        verification = launch_verification(insight_id)
        if "error" in verification:
            if verification.get("error_code") == INSIGHT_INPUT_MISSING_ERROR_CODE:
                missing = ", ".join(verification.get("missing_fields") or [])
                note = "Waiting for required insight fields before novelty verification can run."
                if missing:
                    note = f"{note} Missing: {missing}."
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
        _upsert_job(
            insight_id,
            status="verifying",
            stage="novelty_verification",
            last_note="Launched EvoScientist novelty check.",
            last_error=None,
        )
        log_event("auto_research", {"step": "verification_started", "insight_id": insight_id})
        return
    if novelty == "verifying":
        _upsert_job(
            insight_id,
            status="verifying",
            stage="novelty_verification",
            last_note="Novelty verification still running.",
            last_error=None,
        )
        return
    if novelty == "exists":
        _upsert_job(insight_id, status="blocked", stage="prior_work_exists", last_note="Insight already exists in prior work.")
        return

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
                    _upsert_job(
                        insight_id,
                        status="blocked",
                        stage="research_input_missing",
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
                    stage="research_launch_failed",
                    last_error=result["error"],
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
            _upsert_job(
                insight_id,
                status="researching",
                stage="deep_research",
                research_workdir=result.get("workdir"),
                last_note="Launched EvoScientist deep research.",
                last_error=None,
            )
            log_event("auto_research", {"step": "deep_research_started", "insight_id": insight_id})
            return
    else:
        _upsert_job(
            insight_id,
            stage="research_unavailable",
            last_note="EvoScientist binary not found; continuing with experiment-only path.",
        )

    if tier == 1:
        _upsert_job(
            insight_id,
            status="completed",
            stage="tier1_research_complete",
            last_note="Tier-1 auto research complete. Deep research finished; no paper-idea scaffold required.",
        )
        return

    existing_run = db.fetchone(
        "SELECT * FROM experiment_runs WHERE deep_insight_id=? ORDER BY id DESC LIMIT 1",
        (insight_id,),
    )
    if not existing_run:
        _upsert_job(
            insight_id,
            status="review_pending",
            stage="experiment_review",
            last_note="Running structured experiment review before forge.",
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
                last_note=(forged.get("judgement") or {}).get("summary", "Experiment is smoke-test only and blocked from formal pipeline."),
            )
            return
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
    elif not _run_is_formal(existing_run):
        _upsert_job(
            insight_id,
            status="smoke_only",
            stage="experiment_review_smoke_only",
            experiment_run_id=existing_run["id"],
            resource_class=resource_class,
            last_note="Existing experiment run is marked non-formal; manuscript path remains blocked.",
        )
        return

    if existing_run["status"] in {"completed"}:
        note = f"Verdict={existing_run.get('hypothesis_verdict')}, effect_pct={existing_run.get('effect_pct')}"
        _upsert_job(insight_id, status="completed", stage="closed_loop_complete", experiment_run_id=existing_run["id"], last_note=note)
        return
    if existing_run["status"] in {"failed"}:
        _upsert_job(insight_id, status="failed", stage="experiment_failed", experiment_run_id=existing_run["id"], last_error=existing_run.get("error_message"))
        return

    if resource_class != "cpu":
        gpu_scheduler.start()
        queued_job = db.fetchone(
            """
            SELECT * FROM gpu_jobs
            WHERE experiment_run_id=? AND status IN ('queued', 'running')
            ORDER BY id DESC LIMIT 1
            """,
            (existing_run["id"],),
        )
        if not queued_job:
            gpu_job_id = gpu_scheduler.queue_run(
                insight_id=insight_id,
                run_id=existing_run["id"],
                resource_class=resource_class,
                priority=2 if resource_class == "gpu_large" else 1,
                vram_required_gb=40 if resource_class == "gpu_large" else 16,
            )
            note = f"Queued on GPU scheduler as job {gpu_job_id}."
        else:
            note = f"GPU job {queued_job['id']} already {queued_job['status']}."
        _upsert_job(
            insight_id,
            status="queued_gpu",
            stage="gpu_scheduler",
            experiment_run_id=existing_run["id"],
            resource_class=resource_class,
            last_note=note,
            last_error=None,
        )
        log_event("auto_research", {"step": "gpu_job_queued", "insight_id": insight_id, "run_id": existing_run["id"]})
        return

    _upsert_job(
        insight_id,
        status="running_cpu",
        stage="validation_loop",
        experiment_run_id=existing_run["id"],
        resource_class=resource_class,
        last_note="Starting SciForge validation loop.",
        last_error=None,
    )
    log_event("auto_research", {"step": "experiment_started", "insight_id": insight_id, "run_id": existing_run["id"]})
    result = run_validation_loop(existing_run["id"])
    process_completed_run(existing_run["id"])
    bundle = generate_submission_bundle(existing_run["id"])
    _upsert_job(
        insight_id,
        status="bundle_ready" if "error" not in bundle else "completed",
        stage="writing_submission" if "error" not in bundle else "closed_loop_complete",
        experiment_run_id=existing_run["id"],
        artifact_bundle_id=(bundle.get("bundle_ids") or [None])[-1],
        last_note=f"Completed with verdict={result.get('verdict', 'unknown')}. Submission bundle status={'ok' if 'error' not in bundle else 'failed'}.",
    )
    log_event("auto_research", {"step": "experiment_completed", "insight_id": insight_id, "run_id": existing_run["id"], "verdict": result.get("verdict")})


def consume_pipeline_events_once(limit: int = 50) -> dict:
    db.init_db()
    events = db.fetch_pipeline_events(
        AUTO_RESEARCH_CONSUMER,
        limit=limit,
        event_types=["deep_insight_created", "experiment_run_completed", "submission_bundle_ready", "gpu_job_completed", "gpu_job_failed"],
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
    launch_stats = _launch_candidates_to_capacity()
    if launch_stats["scheduled"]:
        return {"status": "processed", "insight_ids": launch_stats["scheduled"]}
    if launch_stats["execution_active"] >= max(1, AUTO_RESEARCH_MAX_ACTIVE) or launch_stats["verifying_active"] >= MAX_PARALLEL_VERIFICATIONS:
        return {"status": "busy"}
    if not _next_candidate():
        return {"status": "idle"}
    return {"status": "pending"}


def _run_once() -> dict:
    db.init_db()
    event_stats = consume_pipeline_events_once(limit=50)
    cycle_stats = run_cycle()
    active = _active_job_count()
    return {
        "events": event_stats.get("events", 0),
        "cycle_status": cycle_stats.get("status"),
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
            log_event("error", {"step": "auto_research_loop", "error": str(exc)})
            sleep_s = max(5, AUTO_RESEARCH_INTERVAL_SECONDS)
        _stop_event.wait(sleep_s)


def start() -> dict:
    global _worker_thread
    db.init_db()
    with _worker_lock:
        if _worker_thread and _worker_thread.is_alive():
            return {"status": "already_running"}
        _stop_event.clear()
        _worker_thread = threading.Thread(target=_run_loop, daemon=True, name="deepgraph-auto-research")
        _worker_thread.start()
    log_event("auto_research", {"step": "started"})
    return {"status": "started"}


def stop() -> dict:
    _stop_event.set()
    log_event("auto_research", {"step": "stopped"})
    return {"status": "stopping"}
