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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from agents.discovery_metadata import infer_experimentability, infer_resource_class
from agents.compute_profile import detect_compute_profile, gpu_resource_allowed
from agents.experiment_forge import forge_experiment, repair_experiment_plan_from_review, _ensure_real_benchmark_plan
from agents.benchmark_manager import (
    HARNESS_REQUIRED_STAGE,
    HARNESS_REQUIRED_STATUS,
    judgement_requires_benchmark_harness,
    record_harness_required,
)
from agents.loop_router import compact_loop_note, route_blockers
from agents.benchmark_harness_loop import prepare_harness_loop_task
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
from agents.research_bridge import active_research_session, get_research_status
from agents.validation_loop import run_validation_loop
from agents.workspace_layout import write_plan_files
from compat.filelock import FileLock
from agents.evosci_requirements import (
    evosci_binary_path,
    evosci_installed,
    final_report_ready,
)
from config import (
    AUTO_RESEARCH_INTERVAL_SECONDS,
    AUTO_RESEARCH_MAX_ACTIVE,
    EXPERIMENT_TIME_BUDGET,
    GPU_JOB_TIMEOUT_SECONDS,
    REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS,
)
from db import database as db
from meta_harness.scientific_authority import positive_decision_authorized
from meta_harness.failure_repository import FailureRecoveryRepository
from meta_harness.job_states import claim_predicate_sql
from db.insight_outcomes import (
    OUTCOME_EXPERIMENT_FAILED_RUN,
    OUTCOME_EXPERIMENT_FAILED_SETUP,
    apply_experiment_finished_deep,
    set_outcome,
)
from orchestrator.benchmark_completion import (
    BENCHMARK_COMPLETION_STAGE,
    benchmark_completion_blockers,
    benchmark_completion_bundle_from_run,
    schedule_benchmark_completion,
)
from orchestrator import gpu_scheduler
from orchestrator import meta_compute_runtime
from orchestrator import manuscript_watchdog
from orchestrator.pipeline import log_event

_worker_thread: threading.Thread | None = None
_worker_lock = threading.Lock()
_stop_event = threading.Event()
_process_lock: FileLock | None = None
_active_execution_lock = threading.Lock()
_active_execution: dict | None = None
_active_queue_worker_lock = threading.Lock()
_active_queue_workers: dict[int, str] = {}
AUTO_RESEARCH_CONSUMER = "auto_research"
VERIFY_STALE_SECONDS = 60 * 60
RESEARCH_STALE_SECONDS = 6 * 60 * 60
REVIEW_PENDING_STALE_SECONDS = int(os.environ.get("DEEPGRAPH_REVIEW_PENDING_STALE_SECONDS", str(90 * 60)))
REVIEW_WORKER_STALE_SECONDS = int(os.environ.get("DEEPGRAPH_REVIEW_WORKER_STALE_SECONDS", "1800"))
SCAFFOLD_STALE_SECONDS = int(os.environ.get("DEEPGRAPH_SCAFFOLD_STALE_SECONDS", str(45 * 60)))
REVIEW_SCAFFOLD_STALE_SECONDS = int(os.environ.get("DEEPGRAPH_REVIEW_SCAFFOLD_STALE_SECONDS", str(90 * 60)))
EXECUTION_STALE_SECONDS = int(os.environ.get("DEEPGRAPH_EXECUTION_STALE_SECONDS", "600"))
MAX_EXPERIMENT_REVIEW_REPAIR_ATTEMPTS = int(os.environ.get("DEEPGRAPH_EXPERIMENT_REVIEW_REPAIR_ATTEMPTS", "2"))
MAX_BENCHMARK_HARNESS_REPAIR_ATTEMPTS = int(os.environ.get("DEEPGRAPH_BENCHMARK_HARNESS_REPAIR_ATTEMPTS", "3"))
BENCHMARK_HARNESS_DESIGN_REPAIR_STAGE = "benchmark_harness_design_repair"
BENCHMARK_HARNESS_DESIGN_REPAIR_QUEUED_STATUS = "harness_design_repair_queued"
BENCHMARK_HARNESS_DESIGN_REPAIRED_STATUS = "harness_design_repair_requeued"
MAX_REVIEW_STALE_RETRIES = int(os.environ.get("DEEPGRAPH_REVIEW_STALE_RETRIES", "2"))
MAX_FAILED_RUN_REPAIR_ATTEMPTS = int(os.environ.get("DEEPGRAPH_FAILED_RUN_REPAIR_ATTEMPTS", "1"))
AUTO_RESEARCH_CANDIDATE_POOL_LIMIT = int(os.environ.get("DEEPGRAPH_AUTO_RESEARCH_CANDIDATE_POOL_LIMIT", "100"))
MAX_PARALLEL_VERIFICATIONS = 2
MAX_PARALLEL_REVIEWS = int(os.environ.get("DEEPGRAPH_MAX_PARALLEL_REVIEWS", "2"))
MAX_PARALLEL_REPAIRS = int(os.environ.get("DEEPGRAPH_MAX_PARALLEL_REPAIRS", "1"))
QUEUE_VERIFICATION = "verification"
QUEUE_RESEARCH = "research"
QUEUE_REVIEW = "experiment_review"
QUEUE_REPAIR = "repair"
QUEUE_EXECUTION = "execution"
QUEUE_HARNESS = "harness_required"
QUEUE_WAITING = "waiting"
QUEUE_DONE = "done"
QUEUE_BLOCKED = "blocked"
QUEUE_ORDER = (QUEUE_REPAIR, QUEUE_VERIFICATION, QUEUE_EXECUTION, QUEUE_REVIEW)
REPAIR_REVIEW_PENDING_STAGES = {
    BENCHMARK_HARNESS_DESIGN_REPAIR_STAGE,
    "repair_worker",
}


def _insight_is_archived_or_cleaned(row: dict | None) -> bool:
    if not row:
        return False
    status = str(row.get("status") or "").strip().lower()
    novelty = str(row.get("novelty_status") or "").strip().lower()
    outcome = str(row.get("outcome") or "").strip().lower()
    submission = str(row.get("submission_status") or "").strip().lower()
    return (
        status in {"exists"}
        or novelty in {"cleaned_similar_duplicate", "exists"}
        or outcome in {"cleaned", "archived"}
        or submission in {"stale"}
    )


TERMINAL_RUN_STATUSES = {
    "completed",
    "failed",
    "superseded",
    "reset",
    "archived",
    "cancelled",
    "manuscript_blocked",
    "bundle_ready",
}
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
MANUSCRIPT_RETRY_STAGES = {
    "manuscript_retry_after_quality_gate",
    "manuscript_retry_after_soft_benchmark_gate",
    "manuscript_blocked",
}
OPTIONAL_RESEARCH_NONBLOCKING_STAGES = {
    "deep_research_background",
    "research_launch_failed",
    "research_unavailable",
    "research_skipped_input_missing",
}
IGNORED_EXISTING_RUN_STATUSES = {"superseded", "reset", "archived", "cancelled"}

HEAVY_KEYWORDS = {
    "llm", "gpt", "llama", "mistral", "diffusion", "stable diffusion",
    "video", "multimodal", "vision-language", "vlm", "7b", "13b", "70b",
    "gpu", "pretrain", "pre-training", "billion", "transformer-xl",
}


@dataclass(frozen=True)
class QueueDecision:
    queue: str
    runnable: bool
    reason: str = ""


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
    if str(run.get("status") or "") != "scaffolding":
        return False
    if not _run_review_decision_ready(run):
        return False
    return bool(
        (run.get("workdir") or "").strip()
        and (run.get("program_md") or "").strip()
        and (run.get("success_criteria") or "").strip()
    )


def _run_has_incomplete_review_scaffold(run: dict | None) -> bool:
    if not run:
        return False
    if str(run.get("status") or "") != "scaffolding":
        return False
    if str(run.get("phase") or "") != "review_decision_ready":
        return False
    if _run_age_seconds(run) < SCAFFOLD_STALE_SECONDS:
        return False
    return not (run.get("program_md") or "").strip() or not (run.get("success_criteria") or "").strip()


def _json_mapping(value) -> dict:
    data = _load_json(value, {}) if isinstance(value, str) or value is None else value
    return dict(data) if isinstance(data, dict) else {}



def _json_list(value) -> list:
    data = _load_json(value, []) if isinstance(value, str) or value is None else value
    return list(data) if isinstance(data, list) else []


def _dedupe_texts(items: list[object]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _named_text_rows(value) -> list[str]:
    rows = value if isinstance(value, list) else [value] if value not in (None, "") else []
    out: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            text = str(row.get("issue") or row.get("summary") or row.get("evidence") or row.get("name") or row.get("message") or row.get("error") or "").strip()
        else:
            text = str(row or "").strip()
        if text:
            out.append(text)
    return _dedupe_texts(out)


def _harness_loop_report_from_row(row: dict) -> dict:
    task = _json_mapping(row.get("task_plan"))
    existing_report = task.get("loop_router") if isinstance(task.get("loop_router"), dict) else {}
    if existing_report.get("schema_version") == "loop_router_v1" and existing_report.get("blocked"):
        return existing_report
    blockers: list[object] = [row.get("last_error"), row.get("last_note"), row.get("benchmark_name")]
    if task:
        blockers.extend(task.get("recipe_blockers") if isinstance(task.get("recipe_blockers"), list) else [])
        blockers.extend(_named_text_rows(task.get("dataset_refs")))
        judgement = _json_mapping(task.get("review_judgement"))
        blockers.append(judgement.get("summary"))
        blockers.extend(_named_text_rows(judgement.get("blockers")))
        blockers.extend(_named_text_rows(judgement.get("warnings")))
    blockers = _dedupe_texts(blockers)
    if not blockers:
        blockers = ["Generated runner cannot execute the benchmark contract."]
    return route_blockers(
        blockers,
        context={
            "source": "benchmark_harness_consumer",
            "stage": HARNESS_REQUIRED_STAGE,
            "insight_id": row.get("deep_insight_id"),
        },
    )


def _annotate_unrecovered_harness_job(row: dict) -> bool:
    task = _json_mapping(row.get("task_plan"))
    loop_report = _harness_loop_report_from_row(row)
    if task:
        task = prepare_harness_loop_task(
            task,
            benchmark_name=str(row.get("benchmark_name") or ""),
            loop_report=loop_report,
        )
    loop_note = compact_loop_note(loop_report)
    loop_state = task.get("loop_state") if isinstance(task, dict) else {}
    primary_owner = str(loop_state.get("owner") or loop_report.get("primary_owner") or "Benchmark Harness Code Agent").strip()
    primary_stage = str(loop_state.get("stage") or loop_report.get("primary_stage") or HARNESS_REQUIRED_STAGE).strip()
    substatus = str(loop_state.get("status") or "waiting_on_loop_owner").strip()
    new_note = (
        "Benchmark harness job is waiting on automatic loop repair; "
        f"owner={primary_owner}; stage={primary_stage}; substatus={substatus}."
    )
    if loop_note:
        new_note = f"{new_note} {loop_note}"
    new_error = str(row.get("last_error") or "Generated runner cannot execute the benchmark contract.").strip()
    payload = json.dumps(task, ensure_ascii=False, default=str) if task else row.get("task_plan")
    dataset_refs_payload = (
        json.dumps(task.get("dataset_refs") or [], ensure_ascii=False, default=str)
        if task
        else row.get("dataset_refs")
    )
    if (
        str(row.get("last_note") or "") == new_note
        and (not task or str(row.get("task_plan") or "") == str(payload or ""))
        and (not task or str(row.get("dataset_refs") or "") == str(dataset_refs_payload or ""))
    ):
        return False
    db.execute(
        """
        UPDATE benchmark_harness_jobs
        SET task_plan=?,
            dataset_refs=?,
            last_error=?,
            last_note=?,
            updated_at=CURRENT_TIMESTAMP
        WHERE id=? AND agenda_id=?
        """,
        (
            payload,
            dataset_refs_payload,
            new_error,
            new_note,
            int(row["id"]),
            int(row["agenda_id"]),
        ),
    )
    db.commit()
    if task:
        try:
            materialization_plan = task.get("dataset_materialization_plan") or {}
            materialization_status = task.get("dataset_materialization_status") or {}
            write_plan_files(
                int(row["deep_insight_id"]),
                files={
                    "benchmark_harness_task.json": task,
                    "benchmark_harness_status.json": {
                        "status": HARNESS_REQUIRED_STATUS,
                        "stage": HARNESS_REQUIRED_STAGE,
                        "harness_job_id": int(row["id"]),
                        "benchmark_name": row.get("benchmark_name"),
                        "loop_router": loop_report,
                        "loop_state": loop_state or {},
                        "dataset_materialization_plan": materialization_plan,
                        "dataset_materialization_status": materialization_status,
                        "last_error": new_error,
                        "last_note": new_note,
                    },
                    "dataset_materialization_plan.json": materialization_plan,
                    "dataset_materialization_status.json": materialization_status,
                },
                mirror_to_run_spec=False,
            )
        except Exception:
            pass
    return True


_GENERATED_RUNNER_RECOVERY_TASK_TYPES = {
    "",
    "qa",
    "math_qa",
    "multihop_qa",
    "boolean_qa",
    "multiple_choice",
    "code_generation",
    "derived_stress_split",
}


def _target_name(target: dict) -> str:
    return str(target.get("name") or target.get("hf_dataset") or target.get("dataset") or "").strip()


def _target_has_concrete_source(target: dict) -> bool:
    hf_dataset = str(target.get("hf_dataset") or "").strip()
    return bool(target.get("direct_files") or target.get("derive_from_loaded_benchmarks") or (hf_dataset and "/" in hf_dataset))


def _target_recoverable_by_generated_runner(target: dict) -> bool:
    if target.get("requires_harness"):
        return False
    task_type = str(target.get("task_type") or "").strip().lower()
    if task_type == "benchmark":
        task_type = ""
    if task_type not in _GENERATED_RUNNER_RECOVERY_TASK_TYPES:
        return False
    if not _target_has_concrete_source(target):
        return False
    if target.get("generated_runner_supported") is False:
        return False
    return True


def _supported_harness_targets_from_task(task: dict) -> tuple[list[dict], list[dict]]:
    refs = task.get("dataset_refs") if isinstance(task.get("dataset_refs"), list) else []
    supported: list[dict] = []
    deferred: list[dict] = []
    for raw in refs:
        if not isinstance(raw, dict):
            continue
        target = dict(raw)
        name = _target_name(target)
        if _target_recoverable_by_generated_runner(target):
            target["generated_runner_supported"] = True
            supported.append(target)
        elif name:
            deferred.append(target)
    return supported, deferred


def _reset_review_repair_history_after_harness_recovery(plan: dict) -> dict:
    """Give a harness-recovered runnable subset one fresh review/forge attempt."""
    repaired = dict(plan or {})
    history = repaired.get("review_repair_history")
    if isinstance(history, list) and history:
        archived = repaired.get("harness_recovery_archived_review_repair_history")
        archived_rows = list(archived) if isinstance(archived, list) else []
        archived_rows.extend(item for item in history if isinstance(item, dict))
        repaired["harness_recovery_archived_review_repair_history"] = archived_rows[-20:]
    repaired["review_repair_history"] = []
    repaired["harness_recovery_fresh_forge"] = True
    repaired["harness_recovery_fresh_forge_at"] = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    return repaired


def _repair_harness_job_from_task_plan(row: dict) -> dict | None:
    task = _json_mapping(row.get("task_plan"))
    if not task:
        return None
    supported, deferred = _supported_harness_targets_from_task(task)
    if not supported:
        return None
    insight = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (int(row["deep_insight_id"]),))
    if not insight:
        return None
    plan = _json_mapping(insight.get("experimental_plan"))
    method = _json_mapping(insight.get("proposed_method"))
    plan["benchmark_targets"] = supported
    plan["datasets"] = [{"name": t.get("name") or t.get("hf_dataset") or t.get("dataset")} for t in supported]
    plan["generated_runner_supported"] = True
    plan["real_benchmark_required"] = True
    plan["benchmark_harness_deferred"] = bool(deferred)
    if deferred:
        plan["deferred_benchmark_targets"] = [
            t.get("name") or t.get("hf_dataset") or t.get("dataset")
            for t in deferred
            if t.get("name") or t.get("hf_dataset") or t.get("dataset")
        ]
        plan["deferred_benchmark_target_details"] = deferred
    repaired = _ensure_real_benchmark_plan(
        {**dict(insight), "experimental_plan": plan, "proposed_method": method},
        method,
        plan,
        row.get("resource_class"),
        resolve_datasets=False,
    )
    if repaired.get("generated_runner_supported") is not True or not repaired.get("benchmark_targets"):
        return None
    return _reset_review_repair_history_after_harness_recovery(repaired)


_HARNESS_DESIGN_REPAIR_STATUSES = {
    "source_resolution_required",
}
_CLOSED_AUTO_STATUSES = ("completed", "blocked", "failed")


def _harness_loop_state(task: dict) -> dict:
    state = task.get("loop_state") if isinstance(task.get("loop_state"), dict) else {}
    if state:
        return state
    status = task.get("dataset_materialization_status")
    if isinstance(status, dict):
        return {"status": status.get("status")}
    return {}


def _harness_design_repair_judgement(row: dict, task: dict) -> dict:
    loop_state = _harness_loop_state(task)
    refs = task.get("dataset_refs") if isinstance(task.get("dataset_refs"), list) else []
    blockers: list[str] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        name = str(ref.get("name") or ref.get("requested_name") or ref.get("hf_dataset") or "dataset").strip()
        source_state = str(ref.get("source_state") or "").strip()
        hf_dataset = str(ref.get("hf_dataset") or "").strip()
        official_url = str(ref.get("official_url") or ref.get("url") or "").strip()
        requires_harness = bool(ref.get("requires_harness"))
        if not hf_dataset and not official_url:
            blockers.append(f"{name} has no pinned official dataset id or source URL.")
        elif source_state == "unresolved":
            blockers.append(f"{name} source is unresolved; benchmark selection must pin official files/splits.")
        elif requires_harness and not hf_dataset:
            blockers.append(f"{name} requires harness/materialization but lacks a concrete dataset artifact reference.")
    if not blockers:
        blockers.append(str(row.get("last_error") or "Benchmark harness requires pre-execution design repair."))
    summary = (
        "Pre-execution benchmark harness gate found unresolved benchmark/data-source design; "
        "repair the experimental plan before any GPU experiment is allowed."
    )
    return {
        "summary": summary,
        "blockers": blockers[:12],
        "warnings": [str(row.get("last_note") or "")[:1000]] if row.get("last_note") else [],
        "environment_review": {"benchmark_harness_required": False},
        "harness_loop_state": loop_state,
    }


def _harness_design_repair_attempt(row: dict) -> int:
    return max(
        _repair_attempt_from_note(row.get("last_note"), "benchmark_harness"),
        _repair_attempt_from_note(row.get("auto_last_note"), "benchmark_harness"),
        _repair_attempt_from_note(row.get("last_error"), "benchmark_harness"),
        _repair_attempt_from_note(row.get("auto_last_error"), "benchmark_harness"),
    )


def repair_benchmark_harness_design_jobs(limit: int = 5) -> int:
    """Queue harness design blockers for asynchronous experiment-plan repair.

    Harness checks are pre-execution gates. The scheduler must flag unresolved
    benchmark/data-source design before any GPU/CPU experiment is allowed, but
    LLM repair can be slow, so the main scheduler only enqueues a repair worker.
    """

    rows = db.fetchall(
        """
        SELECT bhj.*, arj.status AS auto_status, arj.stage AS auto_stage,
               arj.last_note AS auto_last_note, arj.last_error AS auto_last_error,
               arj.resource_grant_id AS auto_resource_grant_id
        FROM benchmark_harness_jobs bhj
        LEFT JOIN auto_research_jobs arj
          ON arj.agenda_id=bhj.agenda_id
         AND arj.deep_insight_id=bhj.deep_insight_id
        WHERE bhj.status='harness_required'
          AND COALESCE(arj.status, '')='harness_required'
        ORDER BY bhj.updated_at ASC, bhj.id ASC
        LIMIT ?
        """,
        (int(limit),),
    )
    queued_count = 0
    max_attempts = max(0, MAX_BENCHMARK_HARNESS_REPAIR_ATTEMPTS)
    if max_attempts <= 0:
        return 0
    for row in rows:
        insight_id = int(row["deep_insight_id"])
        task = _json_mapping(row.get("task_plan"))
        if task:
            task = prepare_harness_loop_task(task, benchmark_name=row.get("benchmark_name"))
        loop_state = _harness_loop_state(task)
        loop_status = str(loop_state.get("status") or "").strip()
        if loop_status not in _HARNESS_DESIGN_REPAIR_STATUSES:
            continue
        previous_attempt = _harness_design_repair_attempt(row)
        if previous_attempt >= max_attempts:
            continue
        attempt = previous_attempt + 1
        tag = _repair_tag("benchmark_harness", attempt, max_attempts)
        db.execute(
            """
            UPDATE benchmark_harness_jobs
            SET status=?,
                last_error=NULL,
                last_note=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=? AND agenda_id=? AND status='harness_required'
            """,
            (
                BENCHMARK_HARNESS_DESIGN_REPAIR_QUEUED_STATUS,
                (
                    f"{tag} queued asynchronous pre-execution experiment-design repair. "
                    f"Owner={loop_state.get('owner') or 'Benchmark Manager'}; status={loop_status}."
                ),
                int(row["id"]),
                int(row["agenda_id"]),
            ),
        )
        db.commit()
        _upsert_job(
            insight_id,
            status="queued",
            stage=BENCHMARK_HARNESS_DESIGN_REPAIR_STAGE,
            experiment_run_id=None,
            assigned_worker=None,
            last_error=None,
            last_note=(
                f"{tag} benchmark/data-source design blocker caught before experiment execution; "
                "queued asynchronous plan repair and then structured review."
            ),
        )
        log_event(
            "auto_research",
            {
                "step": "benchmark_harness_design_repair_queued",
                "insight_id": insight_id,
                "harness_job_id": int(row["id"]),
                "attempt": attempt,
                "loop_status": loop_status,
            },
        )
        queued_count += 1
    return queued_count


def archive_inactive_benchmark_harness_jobs(limit: int = 50) -> int:
    """Archive stale harness rows whose owning auto job is already closed.

    ``benchmark_harness_jobs`` is an auxiliary queue. Once the owning
    ``auto_research_jobs`` row has reached a terminal state, stale harness rows
    should not keep being diagnosed as if they were active pre-execution work.
    """

    rows = db.fetchall(
        """
        SELECT bhj.id, bhj.agenda_id, bhj.deep_insight_id,
               arj.status AS auto_status, arj.stage AS auto_stage
        FROM benchmark_harness_jobs bhj
        JOIN auto_research_jobs arj
          ON arj.agenda_id=bhj.agenda_id
         AND arj.deep_insight_id=bhj.deep_insight_id
        WHERE bhj.status='harness_required'
          AND arj.status IN (?, ?, ?)
        ORDER BY bhj.updated_at ASC, bhj.id ASC
        LIMIT ?
        """,
        (*_CLOSED_AUTO_STATUSES, int(limit)),
    )
    archived = 0
    for row in rows:
        note = (
            "Archived inactive harness row because the owning auto-research job "
            f"is {row.get('auto_status')}/{row.get('auto_stage')}; no active "
            "pre-execution harness loop remains scheduled."
        )
        db.execute(
            """
            UPDATE benchmark_harness_jobs
            SET status='auto_job_closed_archived',
                last_error=NULL,
                last_note=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=? AND agenda_id=? AND status='harness_required'
            """,
            (note, int(row["id"]), int(row["agenda_id"])),
        )
        log_event(
            "auto_research",
            {
                "step": "benchmark_harness_inactive_archived",
                "insight_id": int(row["deep_insight_id"]),
                "harness_job_id": int(row["id"]),
                "auto_status": row.get("auto_status"),
                "auto_stage": row.get("auto_stage"),
            },
        )
        archived += 1
    if archived:
        db.commit()
    return archived


def _benchmark_harness_design_repair_row(insight_id: int) -> dict | None:
    row = db.fetchone(
        """
        SELECT bhj.*, arj.status AS auto_status, arj.stage AS auto_stage,
               arj.last_note AS auto_last_note, arj.last_error AS auto_last_error,
               arj.resource_grant_id AS auto_resource_grant_id
        FROM benchmark_harness_jobs bhj
        LEFT JOIN auto_research_jobs arj ON arj.deep_insight_id=bhj.deep_insight_id
        WHERE bhj.deep_insight_id=?
          AND bhj.status IN (?, 'harness_required', ?)
        ORDER BY CASE
                   WHEN bhj.status=? THEN 0
                   WHEN bhj.status='harness_required' THEN 1
                   ELSE 2
                 END,
                 bhj.updated_at ASC,
                 bhj.id ASC
        LIMIT 1
        """,
        (
            int(insight_id),
            BENCHMARK_HARNESS_DESIGN_REPAIR_QUEUED_STATUS,
            BENCHMARK_HARNESS_DESIGN_REPAIRED_STATUS,
            BENCHMARK_HARNESS_DESIGN_REPAIR_QUEUED_STATUS,
        ),
    )
    return dict(row) if row else None


def _run_benchmark_harness_design_repair_job(insight_id: int) -> bool:
    """Run one queued pre-execution benchmark/data-source design repair."""

    row = _benchmark_harness_design_repair_row(insight_id)
    if not row:
        _upsert_job(
            insight_id,
            status=HARNESS_REQUIRED_STATUS,
            stage=HARNESS_REQUIRED_STAGE,
            assigned_worker=None,
            last_error="No benchmark harness design repair job was found for this insight.",
            last_note="Pre-execution design repair worker found no harness job; staying in harness queue.",
        )
        return False

    if str(row.get("status") or "") == BENCHMARK_HARNESS_DESIGN_REPAIRED_STATUS:
        _upsert_job(
            insight_id,
            status="queued",
            stage="experiment_review_repair",
            experiment_run_id=None,
            assigned_worker=None,
            last_error=None,
            last_note="Benchmark/data-source design repair was already completed; requeued structured review.",
        )
        return True

    task = _json_mapping(row.get("task_plan"))
    if task:
        task = prepare_harness_loop_task(task, benchmark_name=row.get("benchmark_name"))
    loop_state = _harness_loop_state(task)
    loop_status = str(loop_state.get("status") or "").strip()
    if loop_status not in _HARNESS_DESIGN_REPAIR_STATUSES:
        db.execute(
            """
            UPDATE benchmark_harness_jobs
            SET status='harness_required', updated_at=CURRENT_TIMESTAMP
            WHERE id=? AND agenda_id=?
            """,
            (int(row["id"]), int(row["agenda_id"])),
        )
        db.commit()
        _upsert_job(
            insight_id,
            status=HARNESS_REQUIRED_STATUS,
            stage=HARNESS_REQUIRED_STAGE,
            experiment_run_id=None,
            assigned_worker=None,
            last_error="Benchmark harness requires custom harness/materialization rather than experiment-plan repair.",
            last_note=(
                "Pre-execution harness gate stayed in harness queue because the loop status "
                f"is {loop_status or 'unknown'}."
            ),
        )
        return False

    max_attempts = max(0, MAX_BENCHMARK_HARNESS_REPAIR_ATTEMPTS)
    attempt = max(1, _harness_design_repair_attempt(row))
    if max_attempts <= 0 or attempt > max_attempts:
        tag = _repair_tag("benchmark_harness", min(attempt, max_attempts), max_attempts)
        db.execute(
            """
            UPDATE benchmark_harness_jobs
            SET status='harness_required', updated_at=CURRENT_TIMESTAMP
            WHERE id=? AND agenda_id=?
            """,
            (int(row["id"]), int(row["agenda_id"])),
        )
        db.commit()
        _upsert_job(
            insight_id,
            status=HARNESS_REQUIRED_STATUS,
            stage=HARNESS_REQUIRED_STAGE,
            experiment_run_id=None,
            assigned_worker=None,
            last_error=f"{tag} benchmark harness design repair attempts exhausted.",
            last_note="Pre-execution benchmark/data-source design still requires harness-owner intervention.",
        )
        return False

    tag = _repair_tag("benchmark_harness", attempt, max_attempts)
    judgement = _harness_design_repair_judgement(row, task)
    _upsert_job(
        insight_id,
        status="review_pending",
        stage=BENCHMARK_HARNESS_DESIGN_REPAIR_STAGE,
        experiment_run_id=None,
        last_error=None,
        last_note=f"{tag} running asynchronous pre-execution benchmark/data-source design repair.",
    )
    repair = repair_experiment_plan_from_review(
        insight_id,
        judgement=judgement,
        attempt=attempt,
        resource_grant_id=row.get("auto_resource_grant_id"),
    )
    if repair.get("error"):
        db.execute(
            """
            UPDATE benchmark_harness_jobs
            SET status='harness_required',
                last_error=?,
                last_note=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=? AND agenda_id=?
            """,
            (
                f"{tag} {repair['error']}",
                (
                    f"{tag} benchmark harness design repair failed; staying in harness loop. "
                    f"Owner={loop_state.get('owner') or 'Benchmark Manager'}; status={loop_status}."
                ),
                int(row["id"]),
                int(row["agenda_id"]),
            ),
        )
        db.commit()
        _upsert_job(
            insight_id,
            status=HARNESS_REQUIRED_STATUS,
            stage=HARNESS_REQUIRED_STAGE,
            experiment_run_id=None,
            assigned_worker=None,
            last_error=f"{tag} {repair['error']}",
            last_note=(
                f"{tag} benchmark harness design repair failed; staying in harness loop. "
                f"Owner={loop_state.get('owner') or 'Benchmark Manager'}; status={loop_status}."
            ),
        )
        log_event(
            "warning",
            {
                "step": "benchmark_harness_design_repair_failed",
                "insight_id": insight_id,
                "harness_job_id": int(row["id"]),
                "attempt": attempt,
                "error": repair.get("error"),
            },
        )
        return False

    db.execute(
        """
        UPDATE benchmark_harness_jobs
        SET status=?,
            last_error=NULL,
            last_note=?,
            updated_at=CURRENT_TIMESTAMP
        WHERE id=? AND agenda_id=?
        """,
        (
            BENCHMARK_HARNESS_DESIGN_REPAIRED_STATUS,
            f"{tag} {repair.get('repair_summary') or 'Experiment plan repaired from harness blockers.'} Requeued formal review.",
            int(row["id"]),
            int(row["agenda_id"]),
        ),
    )
    db.commit()
    _upsert_job(
        insight_id,
        status="queued",
        stage="experiment_review_repair",
        experiment_run_id=None,
        assigned_worker=None,
        last_error=None,
        last_note=f"{tag} {repair.get('repair_summary') or 'Benchmark/data-source design repaired from harness loop.'} Requeued pre-execution review.",
    )
    log_event(
        "auto_research",
        {
            "step": "benchmark_harness_design_repair_requeued",
            "insight_id": insight_id,
            "harness_job_id": int(row["id"]),
            "attempt": attempt,
            "loop_status": loop_status,
            "llm_repair_used": bool(repair.get("llm_repair_used")),
        },
    )
    return True


def process_benchmark_harness_jobs(limit: int = 10) -> int:
    """Consume harness_required rows that already contain a runnable subset.

    The dedicated custom-harness path remains explicit for unsupported targets,
    but capability-supported probes should not sit forever in harness work just
    because a broader formal target needs a custom adapter later.
    """
    rows = db.fetchall(
        """
        SELECT bhj.*, arj.resource_class, arj.status AS auto_status, arj.stage AS auto_stage
        FROM benchmark_harness_jobs bhj
        LEFT JOIN auto_research_jobs arj
          ON arj.agenda_id=bhj.agenda_id
         AND arj.deep_insight_id=bhj.deep_insight_id
        WHERE bhj.status='harness_required'
          AND COALESCE(arj.status, '')='harness_required'
        ORDER BY bhj.updated_at ASC, bhj.id ASC
        LIMIT ?
        """,
        (int(limit),),
    )
    recovered = 0
    for row in rows:
        insight_id = int(row["deep_insight_id"])
        try:
            repaired = _repair_harness_job_from_task_plan(dict(row))
        except Exception as exc:  # pragma: no cover - defensive guard
            log_event("warning", {"step": "benchmark_harness_consumer_failed", "insight_id": insight_id, "error": str(exc)})
            continue
        if not repaired:
            try:
                annotated = _annotate_unrecovered_harness_job(dict(row))
            except Exception as exc:  # pragma: no cover - defensive diagnosis guard
                log_event("warning", {"step": "benchmark_harness_loop_annotation_failed", "insight_id": insight_id, "error": str(exc)})
                continue
            if annotated:
                loop_report = _harness_loop_report_from_row(dict(row))
                log_event(
                    "auto_research",
                    {
                        "step": "benchmark_harness_loop_diagnosed",
                        "insight_id": insight_id,
                        "primary_owner": loop_report.get("primary_owner"),
                        "primary_stage": loop_report.get("primary_stage"),
                        "next_actions": loop_report.get("next_actions") or [],
                    },
                )
            continue
        try:
            db.execute(
                "UPDATE deep_insights SET experimental_plan=?, updated_at=CURRENT_TIMESTAMP WHERE id=? AND agenda_id=?",
                (
                    json.dumps(repaired, ensure_ascii=False, default=str),
                    insight_id,
                    int(row["agenda_id"]),
                ),
            )
            db.execute(
                """
                UPDATE benchmark_harness_jobs
                SET status='deferred_supported_subset_recovered',
                    last_error=NULL,
                    last_note='Benchmark harness consumer recovered a generated-runner-supported subset; custom formal targets remain deferred.',
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND agenda_id=?
                """,
                (int(row["id"]), int(row["agenda_id"])),
            )
            db.commit()
            _upsert_job(
                insight_id,
                status="queued",
                stage="harness_supported_subset_recovered",
                experiment_run_id=None,
                assigned_worker=None,
                last_error=None,
                last_note="Benchmark harness consumer recovered a supported runnable subset and requeued fresh forge.",
            )
        except Exception as exc:  # pragma: no cover - lock/contention guard
            try:
                db.rollback()
            except Exception:
                pass
            log_event(
                "warning",
                {
                    "step": "benchmark_harness_consumer_write_failed",
                    "insight_id": insight_id,
                    "error": str(exc),
                },
            )
            continue
        log_event(
            "auto_research",
            {
                "step": "benchmark_harness_consumer_requeued",
                "insight_id": insight_id,
                "benchmark_targets": [
                    t.get("name") or t.get("hf_dataset") or t.get("dataset")
                    for t in repaired.get("benchmark_targets", [])
                    if isinstance(t, dict)
                ],
            },
        )
        recovered += 1
    return recovered

def _repair_harness_plan_for_supported_subset(row: dict) -> dict | None:
    """Recover only already-concrete generated-runner targets, without LLM calls."""

    plan = _json_mapping(row.get("experimental_plan"))
    if not plan or plan.get("generated_runner_supported") is not False:
        return None
    task = {"dataset_refs": [item for item in _json_list(plan.get("benchmark_targets")) if isinstance(item, dict)]}
    if not task["dataset_refs"]:
        task = {"dataset_refs": [item for item in _json_list(plan.get("datasets")) if isinstance(item, dict)]}
    supported, deferred = _supported_harness_targets_from_task(task)
    if not supported or deferred:
        return None
    repaired = dict(plan)
    repaired["benchmark_targets"] = supported
    repaired["datasets"] = [{"name": _target_name(target)} for target in supported if _target_name(target)]
    repaired["generated_runner_supported"] = True
    repaired["real_benchmark_required"] = True
    repaired["benchmark_harness_deferred"] = bool(deferred)
    if deferred:
        repaired["deferred_benchmark_targets"] = [_target_name(target) for target in deferred if _target_name(target)]
        repaired["deferred_benchmark_target_details"] = deferred
    else:
        repaired.pop("benchmark_harness_deferred", None)
        repaired.pop("deferred_benchmark_targets", None)
        repaired.pop("deferred_benchmark_target_details", None)
    return _reset_review_repair_history_after_harness_recovery(repaired)


def recover_partially_supported_harness_jobs(limit: int = 25) -> int:
    """Requeue harness jobs when a supported benchmark subset can run now.

    Benchmark Manager still owns the deferred unsupported targets; this only
    prevents one unsupported benchmark from freezing otherwise runnable work.
    """

    rows = db.fetchall(
        """
        SELECT di.*, arj.resource_class, arj.experiment_run_id
        FROM auto_research_jobs arj
        JOIN deep_insights di
          ON di.agenda_id=arj.agenda_id
         AND di.id = arj.deep_insight_id
        WHERE arj.status=? AND arj.stage=?
        ORDER BY arj.updated_at ASC
        LIMIT ?
        """,
        (HARNESS_REQUIRED_STATUS, HARNESS_REQUIRED_STAGE, int(limit)),
    )
    recovered = 0
    for row in rows:
        insight_id = int(row["id"])
        try:
            repaired = _repair_harness_plan_for_supported_subset(dict(row))
        except Exception as exc:  # pragma: no cover - defensive recovery guard
            log_event("warning", {"step": "harness_partial_recovery_failed", "insight_id": insight_id, "error": str(exc)})
            continue
        if not repaired:
            continue
        db.execute(
            "UPDATE deep_insights SET experimental_plan=?, updated_at=CURRENT_TIMESTAMP WHERE id=? AND agenda_id=?",
            (
                json.dumps(repaired, ensure_ascii=False, default=str),
                insight_id,
                int(row["agenda_id"]),
            ),
        )
        db.execute(
            """
            UPDATE benchmark_harness_jobs
            SET status='deferred_supported_subset_recovered',
                last_error=NULL,
                last_note='Supported benchmark subset recovered for automatic execution; unsupported targets remain deferred for Benchmark Manager.',
                updated_at=CURRENT_TIMESTAMP
            WHERE deep_insight_id=? AND agenda_id=?
            """,
            (insight_id, int(row["agenda_id"])),
        )
        db.commit()
        _upsert_job(
            insight_id,
            status="queued",
            stage="harness_supported_subset_recovered",
            experiment_run_id=None,
            last_error=None,
            last_note=(
                "Recovered from benchmark_harness_required: supported benchmark subset "
                "will run automatically; unsupported targets remain deferred for Benchmark Manager."
            ),
        )
        log_event(
            "auto_research",
            {
                "step": "harness_supported_subset_recovered",
                "insight_id": insight_id,
                "active_targets": [
                    target.get("name") or target.get("hf_dataset")
                    for target in repaired.get("benchmark_targets", [])
                    if isinstance(target, dict)
                ],
                "deferred_targets": repaired.get("deferred_benchmark_targets") or [],
            },
        )
        recovered += 1
    return recovered


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
        compute_job = meta_compute_runtime.submit_experiment_run(
            agenda_id=int(run.get("agenda_id") or 0),
            idea_id=insight_id,
            experiment_run_id=int(run["id"]),
            resource_grant_id=int(run.get("resource_grant_id") or 0),
            timeout_seconds=GPU_JOB_TIMEOUT_SECONDS,
        )
        note = (
            "Queued full benchmark completion through ComputeScheduler as "
            f"{compute_job.backend_job_id}."
        )
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
    auto_run_id = insight.get("auto_experiment_run_id") or insight.get("experiment_run_id")
    if auto_run_id:
        run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (auto_run_id,))
        if _run_reusable_for_auto_research(run):
            return run
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


def _run_age_seconds(run: dict | None) -> float:
    if not run:
        return 0.0
    ts = _coerce_datetime(run.get("created_at") or run.get("started_at"))
    if ts is None:
        return 0.0
    now = datetime.now(ts.tzinfo) if ts.tzinfo else datetime.now()
    return max(0.0, (now - ts).total_seconds())


def _repair_attempt_from_note(note: str | None, kind: str) -> int:
    text = str(note or "")
    token = f"[auto_repair:{kind} attempt="
    idx = text.rfind(token)
    if idx < 0:
        return 0
    rest = text[idx + len(token):]
    digits = []
    for ch in rest:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    try:
        return int("".join(digits)) if digits else 0
    except ValueError:
        return 0


def _experiment_review_repair_attempt_from_plan_data(raw_plan) -> int:
    if not raw_plan:
        return 0
    try:
        plan = json.loads(raw_plan) if isinstance(raw_plan, str) else raw_plan
    except (TypeError, json.JSONDecodeError):
        return 0
    if not isinstance(plan, dict):
        return 0
    history = plan.get("review_repair_history")
    if not isinstance(history, list):
        return 0
    attempts: list[int] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        try:
            attempts.append(int(item.get("attempt") or 0))
        except (TypeError, ValueError):
            continue
    # Older buggy runs repeatedly wrote attempt=1 after the job note was
    # overwritten. Count history length as a floor so those runs still exhaust.
    return max([len(history), *attempts] or [0])


def _experiment_review_repair_attempt_from_plan(insight_id: int) -> int:
    """Return persisted review-repair attempt count from the insight plan.

    ``last_note`` is intentionally overwritten while a fresh forge is running,
    so relying on only the auto job note can reset the bounded retry counter and
    create an infinite review/repair loop. The experiment plan repair history is
    the durable source of truth.
    """

    try:
        row = db.fetchone("SELECT experimental_plan FROM deep_insights WHERE id=?", (insight_id,))
    except Exception:
        return 0
    raw = (row or {}).get("experimental_plan") if isinstance(row, dict) else None
    return _experiment_review_repair_attempt_from_plan_data(raw)


def _repair_tag(kind: str, attempt: int, max_attempts: int) -> str:
    return f"[auto_repair:{kind} attempt={attempt}/{max_attempts}]"


def _coerce_review_judgement(payload: dict | None) -> dict:
    payload = payload or {}
    judgement = payload.get("judgement") if isinstance(payload.get("judgement"), dict) else {}
    if judgement:
        return judgement
    error = str(payload.get("error") or "Experiment review blocked formalization.").strip()
    return {
        "summary": error,
        "blockers": [error] if error else [],
        "warnings": [],
    }


def _blocked_review_payload_from_run(run: dict) -> dict:
    proxy = _load_json(run.get("proxy_config"), {})
    judgement = proxy.get("experiment_judgement") if isinstance(proxy.get("experiment_judgement"), dict) else {}
    error = str(run.get("error_message") or judgement.get("summary") or "Experiment review blocked formalization.").strip()
    if not judgement:
        judgement = {"summary": error, "blockers": [error] if error else [], "warnings": []}
    environment_review = judgement.get("environment_review") if isinstance(judgement.get("environment_review"), dict) else {}
    return {
        "error": error,
        "judgement": judgement,
        "route": "blocked",
        "harness_required": bool(environment_review.get("benchmark_harness_required")),
        "harness_queue": environment_review.get("harness_queue") or "",
    }


def _queue_benchmark_harness_required(
    insight_id: int,
    forged: dict | None,
    *,
    judgement: dict,
    source: str,
    summary: str,
) -> bool:
    result = record_harness_required(
        insight_id,
        judgement_payload=forged or {"judgement": judgement},
        source=source,
    )
    if result.get("error"):
        return False
    benchmark_name = str(result.get("benchmark_name") or "custom benchmark").strip()
    harness_job_id = result.get("harness_job_id")
    paths = result.get("paths") if isinstance(result.get("paths"), dict) else {}
    path_note = f" Task: {paths.get('benchmark_harness_task.json')}" if paths.get("benchmark_harness_task.json") else ""
    task = result.get("task") if isinstance(result.get("task"), dict) else {}
    loop_route = task.get("loop_router") if isinstance(task.get("loop_router"), dict) else route_blockers(
        [(forged or {}).get("error"), summary],
        context={"source": source, "stage": HARNESS_REQUIRED_STAGE, "insight_id": insight_id},
    )
    loop_note = compact_loop_note(loop_route)
    job_note = (
        f"Benchmark harness job {harness_job_id} queued for {benchmark_name}. "
        "Main experiment scheduling is released; Benchmark Manager/Dataset/Baseline/Harness agents must complete this before GPU execution."
        f"{path_note}"
    )
    if loop_note:
        job_note = f"{job_note} {loop_note}"
    _upsert_job(
        insight_id,
        status=HARNESS_REQUIRED_STATUS,
        stage=HARNESS_REQUIRED_STAGE,
        experiment_run_id=None,
        last_error=(forged or {}).get("error") or summary,
        last_note=job_note,
    )
    log_event(
        "auto_research",
        {
            "step": "benchmark_harness_required",
            "insight_id": insight_id,
            "benchmark_name": benchmark_name,
            "harness_job_id": harness_job_id,
            "source": source,
            "loop_router": loop_route,
        },
    )
    return True


def _handle_experiment_review_blocked(insight_id: int, forged: dict | None, *, source: str = "review") -> None:
    """Feed structured review blockers back into experiment design, then requeue.

    The previous implementation treated review blockers as terminal. These are
    usually design/benchmark-contract defects, so they should flow back to the
    experiment-design agent with a bounded retry count.
    """

    job = db.fetchone(
        """
        SELECT last_note, last_error, resource_grant_id
        FROM auto_research_jobs
        WHERE deep_insight_id=?
        """,
        (insight_id,),
    ) or {}
    previous_attempt = max(
        _repair_attempt_from_note(job.get("last_note"), "experiment_review"),
        _repair_attempt_from_note(job.get("last_error"), "experiment_review"),
        _experiment_review_repair_attempt_from_plan(insight_id),
    )
    next_attempt = previous_attempt + 1
    judgement = _coerce_review_judgement(forged)
    summary = str(judgement.get("summary") or (forged or {}).get("error") or "Experiment review blocked formalization.").strip()
    if judgement_requires_benchmark_harness(forged or {"judgement": judgement}):
        if _queue_benchmark_harness_required(
            insight_id,
            forged,
            judgement=judgement,
            source=source,
            summary=summary,
        ):
            return

    max_attempts = max(0, MAX_EXPERIMENT_REVIEW_REPAIR_ATTEMPTS)
    if next_attempt > max_attempts:
        tag = _repair_tag("experiment_review", previous_attempt, max_attempts)
        exhausted_summary = (
            f"{tag} automatic review repair exhausted; routing to benchmark/code harness agents. {summary}"
        )
        if _queue_benchmark_harness_required(
            insight_id,
            forged or {"judgement": judgement, "error": summary},
            judgement=judgement,
            source=f"{source}_repair_exhausted",
            summary=exhausted_summary,
        ):
            return
        _upsert_job(
            insight_id,
            status="blocked",
            stage="experiment_review_blocked_final",
            experiment_run_id=None,
            last_error=(forged or {}).get("error") or summary,
            last_note=exhausted_summary,
        )
        log_event(
            "warning",
            {
                "step": "experiment_review_repair_exhausted",
                "insight_id": insight_id,
                "source": source,
                "attempts": previous_attempt,
            },
        )
        return

    tag = _repair_tag("experiment_review", next_attempt, max_attempts)
    repair = repair_experiment_plan_from_review(
        insight_id,
        judgement=judgement,
        attempt=next_attempt,
        resource_grant_id=job.get("resource_grant_id"),
    )
    if repair.get("error"):
        _upsert_job(
            insight_id,
            status="blocked",
            stage="experiment_review_repair_failed",
            experiment_run_id=None,
            last_error=f"{tag} {repair['error']}",
            last_note=f"{tag} experiment design repair failed before retry. Review: {summary}",
        )
        log_event(
            "warning",
            {
                "step": "experiment_review_repair_failed",
                "insight_id": insight_id,
                "source": source,
                "error": repair.get("error"),
            },
        )
        return

    _upsert_job(
        insight_id,
        status="queued",
        stage="experiment_review_repair",
        experiment_run_id=None,
        last_error=None,
        last_note=f"{tag} {repair.get('repair_summary') or 'Experiment design repaired from review blockers.'} Requeued structured review.",
    )
    log_event(
        "auto_research",
        {
            "step": "experiment_review_repair_requeued",
            "insight_id": insight_id,
            "source": source,
            "attempt": next_attempt,
            "llm_repair_used": bool(repair.get("llm_repair_used")),
        },
    )


def _maybe_repair_preexisting_review_block(insight: dict) -> bool:
    if str(insight.get("auto_status") or "") != "blocked":
        return False
    stage = str(insight.get("auto_stage") or "")
    if stage not in {"experiment_review_blocked", "experiment_review_repair_failed", "experiment_review_blocked_final"}:
        return False
    error = str(insight.get("auto_last_error") or insight.get("auto_last_note") or "Experiment review blocked formalization.")
    _handle_experiment_review_blocked(
        int(insight["id"]),
        {"error": error, "judgement": {"summary": error, "blockers": [error], "warnings": []}},
        source="blocked_recovery",
    )
    return True


def _run_has_automation_failure(run: dict) -> bool:
    error = str(run.get("error_message") or "").lower()
    verdict = str(run.get("hypothesis_verdict") or "").lower()
    if verdict != "inconclusive":
        return False
    markers = (
        "automation failed:",
        "no benchmarked candidate method change",
        "no benchmarked candidate method changes",
        "code_repair_required",
        "experiment_reforge_required",
    )
    return any(marker in error for marker in markers)


def _retry_failed_run_with_repair(insight_id: int, run: dict, resource_class: str) -> bool:
    job = db.fetchone(
        "SELECT last_note, last_error, resource_grant_id FROM auto_research_jobs"
        " WHERE deep_insight_id=?",
        (insight_id,),
    ) or {}
    previous_attempt = max(
        _repair_attempt_from_note(job.get("last_note"), "failed_run"),
        _repair_attempt_from_note(job.get("last_error"), "failed_run"),
    )
    next_attempt = previous_attempt + 1
    max_attempts = max(0, MAX_FAILED_RUN_REPAIR_ATTEMPTS)
    if next_attempt > max_attempts:
        return False
    error = str(run.get("error_message") or "Experiment run failed without an error message.").strip()
    try:
        recovery, fingerprint, _record_id = FailureRecoveryRepository().decide_for_run(
            experiment_run_id=int(run["id"]),
            execution_result={
                "error": error,
                "failure_type": error,
                "final_results_present": False,
            },
            retry_count=previous_attempt,
        )
    except Exception:
        db.rollback()
        recovery = None
        fingerprint = "unavailable"
    if recovery is not None and not recovery.invoke_llm_repair:
        _upsert_job(
            insight_id,
            status="blocked",
            stage=f"execution_{recovery.action}",
            experiment_run_id=run.get("id"),
            resource_class=resource_class,
            last_error=(
                f"reason_code={recovery.reason_code}; action={recovery.action}; "
                f"fingerprint={fingerprint}; {error}"
            )[:4000],
            last_note=(
                "Generic failure policy suppressed LLM repair because this "
                "failure does not require a code change."
            ),
        )
        return True
    tag = _repair_tag("failed_run", next_attempt, max_attempts)
    repair = repair_experiment_plan_from_review(
        insight_id,
        judgement={
            "summary": f"Execution failed after forge/review: {error}",
            "blockers": [error],
            "warnings": ["Repair the experiment design or runnable benchmark contract before reforge."],
        },
        attempt=next_attempt,
        # The run carries the grant it was forged under, which is usually
        # expired by the time a repair is attempted; the job carries the
        # current one. Prefer the job's, fall back to the run's.
        resource_grant_id=job.get("resource_grant_id") or run.get("resource_grant_id"),
    )
    if repair.get("error"):
        _upsert_job(
            insight_id,
            status="failed",
            stage="experiment_failed_repair_failed",
            experiment_run_id=run.get("id"),
            resource_class=resource_class,
            last_error=f"{tag} {repair['error']}",
            last_note=f"{tag} failed-run repair could not update the experiment design.",
        )
        return True
    _supersede_stale_scaffold_run(
        int(run["id"]),
        f"{tag} superseded failed run for automatic repaired reforge: {error[:500]}",
    )
    _upsert_job(
        insight_id,
        status="queued",
        stage="retry_failed_run",
        experiment_run_id=None,
        resource_class=resource_class,
        assigned_worker=None,
        last_error=None,
        last_note=f"{tag} {repair.get('repair_summary') or 'Experiment design repaired after failed run.'} Requeued fresh forge.",
    )
    log_event(
        "auto_research",
        {
            "step": "failed_run_repair_requeued",
            "insight_id": insight_id,
            "run_id": run.get("id"),
            "attempt": next_attempt,
        },
    )
    return True


def _supersede_stale_scaffold_run(run_id: int, reason: str) -> None:
    scope = db.fetchone(
        "SELECT agenda_id FROM experiment_runs WHERE id=?",
        (int(run_id),),
    )
    agenda_id = int((scope or {}).get("agenda_id") or 0)
    if agenda_id <= 0:
        raise ValueError("superseding a run requires agenda scope")
    db.execute(
        """
        UPDATE experiment_runs
        SET status='superseded',
            phase='superseded',
            error_message=?,
            completed_at=CURRENT_TIMESTAMP
        WHERE id=? AND agenda_id=?
        """,
        (reason, run_id, agenda_id),
    )
    db.commit()


def _review_stale_recovery_count(insight_id: int) -> int:
    row = db.fetchone(
        """
        SELECT COUNT(*) AS count
        FROM experiment_runs
        WHERE deep_insight_id=?
          AND status='superseded'
          AND phase='superseded'
          AND COALESCE(error_message, '') LIKE '%Recovered stale review/scaffold run%'
        """,
        (int(insight_id),),
    )
    return int(row.get("count") or 0) if row else 0


def _route_review_stale_to_harness(insight_id: int, run_id: int, reason: str) -> None:
    judgement = {
        "summary": reason,
        "blockers": [
            reason,
            "benchmark harness required after repeated stale review/scaffold retries",
        ],
        "warnings": [],
        "environment_review": {"benchmark_harness_required": True},
    }
    queued = _queue_benchmark_harness_required(
        int(insight_id),
        {
            "error": reason,
            "judgement": judgement,
            "harness_required": True,
        },
        judgement=judgement,
        source="review_scaffold_stale_repair_exhausted",
        summary=reason,
    )
    if not queued:
        _upsert_job(
            int(insight_id),
            status="blocked",
            stage="review_scaffold_stale_repair_exhausted",
            experiment_run_id=None,
            assigned_worker=None,
            last_error=reason,
            last_note=(
                "Review/scaffold stale recovery exhausted and harness queueing failed; "
                "manual benchmark/code intervention is required."
            ),
        )
    log_event(
        "warning",
        {
            "step": "auto_research_review_stale_routed_to_harness",
            "insight_id": int(insight_id),
            "run_id": int(run_id),
            "reason": reason,
        },
    )


def _recover_stale_review_scaffold_run(insight_id: int, run: dict, *, source: str) -> bool:
    run_id = int(run["id"])
    run_age = _run_age_seconds(run)
    if _review_worker_live_in_process(int(insight_id)) and run_age < REVIEW_SCAFFOLD_STALE_SECONDS:
        _upsert_job(
            int(insight_id),
            touch_updated_at=False,
            status="review_pending",
            stage="experiment_review",
            experiment_run_id=run_id,
            resource_class=run.get("resource_class"),
            last_error=None,
            last_note=(
                "Experiment forge worker is still active; extended stale recovery is deferred "
                f"until {REVIEW_SCAFFOLD_STALE_SECONDS}s."
            ),
        )
        return True

    stale_count = _review_stale_recovery_count(int(insight_id))
    max_retries = max(0, MAX_REVIEW_STALE_RETRIES)
    base_reason = (
        "Recovered stale review/scaffold run: no complete review decision, "
        "program, or success criteria appeared before the stale timeout."
    )
    if stale_count >= max_retries:
        reason = (
            f"Review/scaffold stale recovery exceeded {max_retries} automatic retry(s); "
            "routing to benchmark/code harness instead of repeating the forge loop."
        )
        _supersede_stale_scaffold_run(run_id, reason)
        _route_review_stale_to_harness(int(insight_id), run_id, reason)
        return True

    attempt = stale_count + 1
    reason = f"[auto_repair:review_stale attempt={attempt}/{max_retries}] {base_reason}"
    _supersede_stale_scaffold_run(run_id, reason)
    _upsert_job(
        int(insight_id),
        status="queued",
        stage="review_retry",
        experiment_run_id=None,
        assigned_worker=None,
        last_error=None,
        last_note=reason,
    )
    log_event(
        "warning",
        {
            "step": "auto_research_review_scaffold_stale",
            "insight_id": int(insight_id),
            "run_id": run_id,
            "attempt": attempt,
            "source": source,
        },
    )
    return True


def recover_orphaned_review_pending_jobs(limit: int = 50) -> int:
    rows = db.fetchall(
        """
        SELECT arj.deep_insight_id,
               arj.experiment_run_id,
               arj.assigned_worker,
               arj.last_note,
               er.status AS run_status,
               er.phase AS run_phase,
               er.error_message AS run_error
        FROM auto_research_jobs arj
        LEFT JOIN experiment_runs er ON er.id = arj.experiment_run_id
        WHERE arj.status='review_pending'
        ORDER BY arj.updated_at ASC
        LIMIT ?
        """,
        (limit,),
    )
    recovered = 0
    for row in rows:
        insight_id = int(row["deep_insight_id"])
        if _review_job_has_live_worker(row):
            continue
        run = None
        run_id = row.get("experiment_run_id")
        if run_id:
            run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (int(run_id),))
        if not run:
            run = db.fetchone(
                """SELECT * FROM experiment_runs
                   WHERE deep_insight_id=? AND status='scaffolding'
                   ORDER BY id DESC LIMIT 1""",
                (insight_id,),
            )
        reason = (
            "Recovered orphaned experiment-review worker after process restart; "
            "releasing the stale claim for a fresh forge."
        )
        if run and _run_scaffold_ready(run):
            _upsert_job(
                insight_id,
                status="eligible" if _run_is_formal(run) else "smoke_only",
                stage="formal_ready" if _run_is_formal(run) else "experiment_review_smoke_only",
                experiment_run_id=run["id"],
                resource_class=run.get("resource_class"),
                assigned_worker=None,
                last_error=None,
                last_note="Recovered scaffold-ready orphaned review job and resumed scheduling.",
            )
        elif run and str(run.get("status") or "") == "failed" and str(run.get("phase") or "") == "experiment_review_blocked":
            _handle_experiment_review_blocked(insight_id, _blocked_review_payload_from_run(run), source="orphaned_review_recovery")
        else:
            if run and str(run.get("status") or "") == "scaffolding":
                _supersede_stale_scaffold_run(int(run["id"]), reason)
            _upsert_job(
                insight_id,
                status="queued",
                stage="review_orphan_recovered",
                experiment_run_id=None,
                assigned_worker=None,
                last_error=None,
                last_note=reason,
            )
        log_event(
            "warning",
            {
                "step": "auto_research_orphaned_review_recovered",
                "insight_id": insight_id,
                "run_id": run.get("id") if run else None,
                "assigned_worker": row.get("assigned_worker"),
            },
        )
        recovered += 1
    return recovered


def recover_runaway_review_scaffold_jobs(limit: int = 25) -> int:
    rows = db.fetchall(
        """
        SELECT arj.deep_insight_id, arj.experiment_run_id
        FROM auto_research_jobs arj
        JOIN experiment_runs er ON er.id = arj.experiment_run_id
        WHERE arj.status='review_pending'
          AND er.status='scaffolding'
        ORDER BY arj.updated_at ASC
        LIMIT ?
        """,
        (limit,),
    )
    recovered = 0
    for row in rows:
        insight_id = int(row["deep_insight_id"])
        run_id = row.get("experiment_run_id")
        if not run_id:
            continue
        run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (int(run_id),))
        if not run or str(run.get("status") or "") != "scaffolding":
            continue
        stale_count = _review_stale_recovery_count(insight_id)
        if stale_count < max(0, MAX_REVIEW_STALE_RETRIES) and _run_age_seconds(run) < REVIEW_PENDING_STALE_SECONDS:
            continue
        if _recover_stale_review_scaffold_run(insight_id, run, source="runaway_review_recovery"):
            recovered += 1
    return recovered


def recover_invalid_ready_jobs(limit: int = 50) -> int:
    rows = db.fetchall(
        """
        SELECT arj.deep_insight_id, arj.experiment_run_id, er.status AS run_status, er.error_message
        FROM auto_research_jobs arj
        LEFT JOIN experiment_runs er ON er.id = arj.experiment_run_id
        WHERE arj.status IN ('eligible', 'smoke_only')
          AND arj.experiment_run_id IS NOT NULL
          AND (er.id IS NULL OR er.status!='scaffolding')
        ORDER BY arj.updated_at ASC
        LIMIT ?
        """,
        (limit,),
    )
    recovered = 0
    for row in rows:
        insight_id = int(row["deep_insight_id"])
        run_id = row.get("experiment_run_id")
        reason = (
            "Ready job pointed at a non-scaffolding experiment run; requeueing before execution. "
            f"run_id={run_id}; run_status={row.get('run_status') or 'missing'}; "
            f"run_error={str(row.get('error_message') or '')[:300]}"
        )
        if _review_stale_recovery_count(insight_id) >= max(0, MAX_REVIEW_STALE_RETRIES):
            _route_review_stale_to_harness(insight_id, int(run_id or 0), reason)
        else:
            _upsert_job(
                insight_id,
                status="queued",
                stage="invalid_ready_run_reforge",
                experiment_run_id=None,
                assigned_worker=None,
                last_error=None,
                last_note=reason,
            )
        recovered += 1
    return recovered


def recover_foreign_key_review_failures(limit: int = 25) -> int:
    rows = db.fetchall(
        """
        SELECT arj.deep_insight_id,
               arj.experiment_run_id,
               arj.last_error,
               er.status AS run_status,
               er.phase AS run_phase,
               er.proxy_config,
               er.error_message
        FROM auto_research_jobs arj
        LEFT JOIN experiment_runs er ON er.id = arj.experiment_run_id
        WHERE arj.status='failed'
          AND arj.stage='exception'
          AND COALESCE(arj.last_error, '') LIKE '%FOREIGN KEY constraint failed%'
        ORDER BY arj.updated_at ASC
        LIMIT ?
        """,
        (limit,),
    )
    recovered = 0
    for row in rows:
        insight_id = int(row["deep_insight_id"])
        run_id = row.get("experiment_run_id")
        reason = (
            "Recovered experiment-review database consistency failure; requeueing a fresh "
            "forge under durable blocked-review handling."
        )
        if run_id and str(row.get("run_status") or "") == "failed" and str(row.get("run_phase") or "") == "experiment_review_blocked":
            _handle_experiment_review_blocked(insight_id, _blocked_review_payload_from_run(row), source="review_fk_recovery")
            recovered += 1
            continue
        if run_id and str(row.get("run_status") or "") == "scaffolding":
            _supersede_stale_scaffold_run(int(run_id), reason)
        _upsert_job(
            insight_id,
            status="queued",
            stage="review_fk_recovered",
            experiment_run_id=None,
            assigned_worker=None,
            last_error=None,
            last_note=reason,
        )
        log_event(
            "warning",
            {
                "step": "auto_research_review_fk_recovered",
                "insight_id": insight_id,
                "run_id": run_id,
                "previous_error": str(row.get("last_error") or "")[:300],
            },
        )
        recovered += 1
    return recovered


def _active_execution_run_id() -> int | None:
    with _active_execution_lock:
        active = _active_execution
    if not active:
        return None
    run_id = active.get("run_id")
    return int(run_id) if run_id is not None else None


def _is_execution_live_in_process(run_id: int | None) -> bool:
    active_run_id = _active_execution_run_id()
    if active_run_id is None or run_id is None:
        return False
    return active_run_id == int(run_id)


def _gpu_execution_live_for_run(run_id: int | None) -> bool:
    if run_id is None:
        return False
    row = db.fetchone(
        """
        SELECT COUNT(*) AS count
        FROM gpu_jobs
        WHERE experiment_run_id=?
          AND status='running'
        """,
        (int(run_id),),
    )
    return bool(row and int(row.get("count") or 0) > 0)


def _requeue_stale_execution_job(job: dict, reason: str) -> None:
    insight_id = int(job["deep_insight_id"])
    run_id = job.get("experiment_run_id")
    _upsert_job(
        insight_id,
        status="queued",
        stage="execution_retry",
        experiment_run_id=run_id,
        assigned_worker=None,
        last_error=None,
        last_note=reason,
    )
    log_event(
        "warning",
        {
            "step": "auto_research_execution_stale",
            "insight_id": insight_id,
            "run_id": run_id,
            "reason": reason,
        },
    )


def recover_stale_execution_jobs() -> int:
    """Requeue CPU validation jobs left running after a controller restart.

    ``run_validation_loop`` executes synchronously inside the auto-research
    worker thread. If ``main.py`` restarts, the DB can still show
    ``running_cpu`` even though no loop is alive. Those rows block scheduling
    because they count against ``max_active``.
    """
    active_run_id = _active_execution_run_id()
    jobs = db.fetchall(
        """
        SELECT arj.*, er.status AS run_status
        FROM auto_research_jobs arj
        LEFT JOIN experiment_runs er ON er.id = arj.experiment_run_id
        WHERE arj.status IN ('running_experiment', 'running_cpu')
          AND arj.experiment_run_id IS NOT NULL
        """
    )
    recovered = 0
    for job in jobs:
        run_id = job.get("experiment_run_id")
        if run_id is not None and active_run_id is not None and int(run_id) == active_run_id:
            continue
        run_status = str(job.get("run_status") or "").strip()
        if run_status in TERMINAL_RUN_STATUSES:
            continue
        if _is_execution_live_in_process(run_id):
            continue
        reason = (
            "Recovered stale CPU execution after scheduler restart; "
            "validation will resume from saved run state."
        )
        _requeue_stale_execution_job(job, reason)
        recovered += 1
    return recovered


def _execute_cpu_validation_loop(insight_id: int, run_id: int) -> dict:
    global _active_execution
    with _active_execution_lock:
        if _active_execution is not None:
            raise RuntimeError(
                f"CPU validation already active for run {_active_execution.get('run_id')}"
            )
        _active_execution = {
            "run_id": int(run_id),
            "insight_id": int(insight_id),
            "started_at": time.time(),
        }
    try:
        return run_validation_loop(run_id)
    finally:
        with _active_execution_lock:
            _active_execution = None


def _upsert_job(insight_id: int, *, touch_updated_at: bool = True, **fields) -> None:
    scope = db.fetchone(
        "SELECT agenda_id FROM deep_insights WHERE id=?",
        (insight_id,),
    )
    agenda_id = int((scope or {}).get("agenda_id") or 0)
    if agenda_id <= 0:
        raise RuntimeError(
            "auto-research refuses an unscoped insight; explicit agenda import "
            "is required"
        )
    existing = db.fetchone(
        "SELECT id, agenda_id FROM auto_research_jobs WHERE deep_insight_id=?",
        (insight_id,),
    )
    if existing and int(existing.get("agenda_id") or 0) != agenda_id:
        raise RuntimeError(
            "auto-research job is unscoped or cross-agenda; explicit import is required"
        )
    if touch_updated_at:
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
        params.extend((insight_id, agenda_id))
        db.execute(
            f"UPDATE auto_research_jobs SET {', '.join(assigns)} "
            "WHERE deep_insight_id=? AND agenda_id=?",
            tuple(params),
        )
    else:
        cols = ["agenda_id", "deep_insight_id"]
        placeholders = ["?", "?"]
        params = [agenda_id, insight_id]
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


def _run_closed_loop_complete(run_id: int) -> bool:
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


def _manuscript_retry_blocker(run: dict | None) -> str | None:
    if not run:
        return "Experiment run is missing."
    status = str(run.get("status") or "").strip().lower()
    if status in {"superseded", "failed", "cancelled", "canceled", "stale"}:
        return f"Experiment run status={status} is not valid for manuscript retry."
    verdict = str(run.get("hypothesis_verdict") or "").strip().lower()
    if verdict not in {"confirmed", "supported"}:
        return f"Experiment verdict={verdict or 'missing'} is not submission-grade."
    if not positive_decision_authorized(
        agenda_id=int(run.get("agenda_id") or 0),
        run_id=int(run.get("id") or 0),
    ):
        return (
            "A persisted supported scientific decision is required before "
            "manuscript retry."
        )
    preflight_blocker = gpu_scheduler._capability_preflight_blocker(run)
    if preflight_blocker:
        return preflight_blocker
    if _run_has_automation_failure(run):
        return run.get("error_message") or "Automation failure blocks manuscript retry."
    return None


def _submission_grade_run_for_insight(insight_id: int, *, exclude_run_id: int | None = None) -> dict | None:
    """Return the best available run that can safely enter manuscript retry."""
    rows = db.fetchall(
        """
        SELECT *
        FROM experiment_runs
        WHERE deep_insight_id=?
        ORDER BY
          CASE
            WHEN status='bundle_ready' THEN 0
            WHEN status='completed' THEN 1
            ELSE 2
          END,
          CASE WHEN submission_bundle_id IS NOT NULL THEN 0 ELSE 1 END,
          COALESCE(effect_pct, -1000000000.0) DESC,
          id DESC
        """,
        (int(insight_id),),
    )
    for row in rows:
        if exclude_run_id is not None and int(row.get("id") or -1) == int(exclude_run_id):
            continue
        if _manuscript_retry_blocker(row) is None:
            return row
    return None


def _block_invalid_manuscript_retry_run(insight_id: int, run_id: int | None, reason: str) -> None:
    scope = db.fetchone(
        "SELECT agenda_id FROM deep_insights WHERE id=?",
        (int(insight_id),),
    )
    agenda_id = int((scope or {}).get("agenda_id") or 0)
    if agenda_id <= 0:
        raise ValueError("blocking manuscript retry requires agenda scope")
    if run_id is not None:
        db.execute(
            """
            UPDATE manuscript_runs
            SET status='stale',
                updated_at=CURRENT_TIMESTAMP
            WHERE experiment_run_id=?
              AND agenda_id=?
              AND status IN ('manuscript_blocked', 'needs_revision', 'failed', 'drafting')
            """,
            (int(run_id), agenda_id),
        )
    replacement = _submission_grade_run_for_insight(insight_id, exclude_run_id=run_id)
    if replacement:
        replacement_run_id = int(replacement["id"])
        _upsert_job(
            int(insight_id),
            status="queued",
            stage="manuscript_retry_after_quality_gate",
            experiment_run_id=replacement_run_id,
            resource_class=replacement.get("resource_class") or "cpu",
            assigned_worker=None,
            last_error=reason,
            last_note=(
                "Recovered manuscript retry by switching from invalid run "
                f"{run_id} to submission-grade run {replacement_run_id}."
            ),
        )
        log_event(
            "auto_research",
            {
                "step": "invalid_manuscript_retry_recovered_with_replacement_run",
                "insight_id": int(insight_id),
                "invalid_run_id": run_id,
                "replacement_run_id": replacement_run_id,
                "reason": reason,
            },
        )
        return
    _upsert_job(
        int(insight_id),
        status="completed",
        stage="closed_loop_complete",
        experiment_run_id=run_id,
        assigned_worker=None,
        last_error=reason,
        last_note="Manuscript retry skipped because the underlying run is not submission-grade; closing this loop.",
    )
    log_event(
        "warning",
        {
            "step": "invalid_manuscript_retry_blocked",
            "insight_id": int(insight_id),
            "run_id": run_id,
            "reason": reason,
        },
    )


def recover_invalid_manuscript_retry_jobs(limit: int = 50) -> int:
    rows = db.fetchall(
        """
        SELECT arj.deep_insight_id, arj.experiment_run_id
        FROM auto_research_jobs arj
        WHERE arj.status='queued'
          AND arj.stage IN ('manuscript_retry_after_quality_gate', 'manuscript_retry_after_soft_benchmark_gate', 'manuscript_blocked')
          AND arj.experiment_run_id IS NOT NULL
        ORDER BY arj.updated_at ASC
        LIMIT ?
        """,
        (limit,),
    )
    recovered = 0
    for row in rows:
        insight_id = int(row["deep_insight_id"])
        run_id = int(row["experiment_run_id"])
        run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
        blocker = _manuscript_retry_blocker(run)
        if not blocker:
            continue
        _block_invalid_manuscript_retry_run(insight_id, run_id, blocker)
        recovered += 1
    return recovered


def _run_manuscript_retry_job(insight_id: int, run_id: int, resource_class: str | None) -> None:
    _upsert_job(
        insight_id,
        status="running_cpu",
        stage="manuscript_revision",
        experiment_run_id=run_id,
        resource_class=resource_class,
        assigned_worker=None,
        last_note="Running targeted manuscript revision from previous quality-gate blockers.",
        last_error=None,
    )
    benchmark_bundle = benchmark_completion_bundle_from_run(run_id)
    if schedule_benchmark_completion(
        insight_id,
        run_id,
        benchmark_bundle,
        source="auto_research_manuscript_retry_pre_manuscript",
        resource_class=resource_class,
    ):
        log_event("auto_research", {"step": "benchmark_completion_queued_before_manuscript_retry", "insight_id": insight_id, "run_id": run_id})
        return
    try:
        bundle = generate_submission_bundle(run_id)
    except Exception as exc:
        _upsert_job(
            insight_id,
            status="queued",
            stage="manuscript_retry_after_quality_gate",
            experiment_run_id=run_id,
            resource_class=resource_class,
            assigned_worker=None,
            last_note="Manuscript revision raised an exception; queued another targeted writing repair.",
            last_error=str(exc)[:4000],
        )
        log_event("error", {"step": "manuscript_retry_failed", "insight_id": insight_id, "run_id": run_id, "error": str(exc)})
        return
    if schedule_benchmark_completion(
        insight_id,
        run_id,
        bundle,
        source="auto_research_manuscript_retry",
        resource_class=resource_class,
    ):
        log_event("auto_research", {"step": "benchmark_completion_queued_from_manuscript_retry", "insight_id": insight_id, "run_id": run_id})
        return
    retry_fields = _bundle_failure_retry_fields(bundle if isinstance(bundle, dict) else None)
    if retry_fields:
        _upsert_job(
            insight_id,
            experiment_run_id=run_id,
            resource_class=resource_class,
            assigned_worker=None,
            **retry_fields,
        )
        log_event("auto_research", {"step": "manuscript_retry_requeued", "insight_id": insight_id, "run_id": run_id})
        return
    bundle_ok = isinstance(bundle, dict) and "error" not in bundle
    status_text = "ok" if bundle_ok else "failed"
    _upsert_job(
        insight_id,
        status="bundle_ready" if bundle_ok else "completed",
        stage="writing_submission" if bundle_ok else "closed_loop_complete",
        experiment_run_id=run_id,
        resource_class=resource_class,
        artifact_bundle_id=(bundle.get("bundle_ids") or [None])[-1] if isinstance(bundle, dict) else None,
        assigned_worker=None,
        last_note=f"Manuscript retry completed. Submission bundle status={status_text}.",
        last_error=None if bundle_ok else str(bundle.get("error") if isinstance(bundle, dict) else bundle),
    )
    log_event("auto_research", {"step": "manuscript_retry_completed", "insight_id": insight_id, "run_id": run_id, "bundle_ok": bundle_ok})


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


def _json_object(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _requires_accelerated_runner(insight: dict) -> bool:
    plan = _json_object(insight.get("experimental_plan"))
    requirements = plan.get("execution_requirements")
    if not isinstance(requirements, dict):
        return False
    model = requirements.get("model")
    model = model if isinstance(model, dict) else {}
    preferred = {
        str(value).strip()
        for value in requirements.get("preferred_backends") or []
    }
    return bool(model.get("requires_cuda")) or bool(
        preferred.intersection({"local_gpu", "ssh_gpu", "colab_gpu"})
    )


def assess_experiment_route(insight: dict) -> tuple[str, str]:
    """Route insights into cpu / gpu_small / gpu_large lanes."""
    inferred_resource = str(insight.get("resource_class") or "").strip() or infer_resource_class(insight)
    resource_class = inferred_resource
    route_note = ""
    if resource_class == "cpu" and _requires_accelerated_runner(insight):
        resource_class = "gpu_large"
        route_note = " structured runner requirements request an accelerated backend."
    experimentability = infer_experimentability({**insight, "resource_class": resource_class})
    allowed, block_reason = gpu_resource_allowed(resource_class)
    if not allowed:
        profile = detect_compute_profile()
        return (
            "gpu_unavailable",
            (
                f"Experimentability={experimentability}; inferred {resource_class}, "
                f"but GPU lane is unavailable ({block_reason}). "
                f"accelerator={profile.accelerator}; local_gpu={profile.local_gpu_available}; "
                f"remote_gpu={profile.remote_gpu_configured}."
            ),
        )
    return resource_class, f"Experimentability={experimentability}; routed to {resource_class}.{route_note}"


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
           JOIN deep_insights di
             ON di.agenda_id=arj.agenda_id
            AND di.id = arj.deep_insight_id
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
             SUM(CASE WHEN status='harness_required' THEN 1 ELSE 0 END) AS harness_required,
             SUM(CASE WHEN status='smoke_only' THEN 1 ELSE 0 END) AS smoke_only,
             SUM(CASE WHEN status='blocked' THEN 1 ELSE 0 END) AS blocked,
             SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed
           FROM auto_research_jobs"""
    ) or {}
    return {
        "running": running,
        "interval_seconds": AUTO_RESEARCH_INTERVAL_SECONDS,
        "max_active": AUTO_RESEARCH_MAX_ACTIVE,
        "max_parallel_reviews": MAX_PARALLEL_REVIEWS,
        "max_parallel_repairs": MAX_PARALLEL_REPAIRS,
        "review_active": _review_pending_job_count(),
        "repair_active": _repair_pending_job_count(),
        "evoscientist_available": evosci_available(),
        **counts,
    }


def _execution_active_job_count() -> int:
    row = db.fetchone(
        """SELECT COUNT(*) AS c
           FROM auto_research_jobs
           WHERE status IN ('running_experiment', 'running_gpu', 'running_cpu', 'queued_gpu')"""
    )
    return row["c"] if row else 0


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


def _review_pending_job_count() -> int:
    placeholders = ", ".join("?" for _ in REPAIR_REVIEW_PENDING_STAGES)
    row = db.fetchone(
        f"""SELECT COUNT(*) AS c
            FROM auto_research_jobs
            WHERE status='review_pending'
              AND COALESCE(stage, '') NOT IN ({placeholders})""",
        tuple(REPAIR_REVIEW_PENDING_STAGES),
    )
    return row["c"] if row else 0


def _repair_pending_job_count() -> int:
    placeholders = ", ".join("?" for _ in REPAIR_REVIEW_PENDING_STAGES)
    row = db.fetchone(
        f"""SELECT COUNT(*) AS c
            FROM auto_research_jobs
            WHERE status='review_pending'
              AND COALESCE(stage, '') IN ({placeholders})""",
        tuple(REPAIR_REVIEW_PENDING_STAGES),
    )
    return row["c"] if row else 0


def _active_job_count() -> int:
    return (
        _execution_active_job_count()
        + _verification_job_count()
        + _research_job_count()
        + _review_pending_job_count()
        + _repair_pending_job_count()
    )


def _queue_active_counts() -> dict[str, int]:
    return {
        QUEUE_EXECUTION: _execution_active_job_count() + _active_queue_worker_count(QUEUE_EXECUTION),
        QUEUE_VERIFICATION: _verification_job_count(),
        QUEUE_RESEARCH: _research_job_count(),
        QUEUE_REVIEW: max(_review_pending_job_count(), _active_queue_worker_count(QUEUE_REVIEW)),
        QUEUE_REPAIR: max(_repair_pending_job_count(), _active_queue_worker_count(QUEUE_REPAIR)),
    }


def _queue_capacity(queue: str) -> int:
    if queue == QUEUE_VERIFICATION:
        return MAX_PARALLEL_VERIFICATIONS
    if queue == QUEUE_EXECUTION:
        return max(1, AUTO_RESEARCH_MAX_ACTIVE)
    if queue == QUEUE_REVIEW:
        return max(1, MAX_PARALLEL_REVIEWS)
    if queue == QUEUE_REPAIR:
        return max(1, MAX_PARALLEL_REPAIRS)
    return 0


def _queue_has_capacity(queue: str, counts: dict[str, int]) -> bool:
    return counts.get(queue, 0) < _queue_capacity(queue)


def _active_queue_worker_count(queue: str | None = None) -> int:
    with _active_queue_worker_lock:
        if queue is None:
            return len(_active_queue_workers)
        return sum(1 for active_queue in _active_queue_workers.values() if active_queue == queue)


def _review_worker_live_in_process(insight_id: int) -> bool:
    with _active_queue_worker_lock:
        return int(insight_id) in _active_queue_workers


def _candidate_active_worker_queue(insight_id: int | None) -> str | None:
    if insight_id is None:
        return None
    try:
        key = int(insight_id)
    except (TypeError, ValueError):
        return None
    with _active_queue_worker_lock:
        return _active_queue_workers.get(key)


def _assigned_worker_pid(value: str | None) -> int | None:
    if not value:
        return None
    parts = str(value).split(":")
    if len(parts) < 2 or parts[0] != "pid":
        return None
    try:
        return int(parts[1])
    except (TypeError, ValueError):
        return None


def _pid_is_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _review_job_has_live_worker(job: dict) -> bool:
    insight_id = int(job.get("deep_insight_id") or 0)
    pid = _assigned_worker_pid(job.get("assigned_worker"))
    if pid == os.getpid():
        return _review_worker_live_in_process(insight_id)
    if pid:
        return _pid_is_alive(pid)
    return _review_worker_live_in_process(insight_id)


def _claim_review_candidate(candidate: dict, queue: str) -> bool:
    """Atomically claim a review/repair candidate before spawning forge.

    The in-process worker map prevents duplicate threads in one interpreter, but
    web/API/process restarts can race through SQLite. Claiming in the DB keeps a
    single insight from being forged multiple times at once.
    """

    insight_id = int(candidate["id"])
    agenda_id = int(candidate.get("agenda_id") or 0)
    if agenda_id <= 0:
        return False
    expected_status = str(candidate.get("auto_status") or "").strip()
    expected_stage = str(candidate.get("auto_stage") or "").strip()
    worker_stage = f"{queue}_worker"
    assigned_worker = f"pid:{os.getpid()}:thread:{threading.get_ident()}:{queue}"
    note = f"Dispatched to {queue} worker under multi-queue scheduler."

    if not expected_status:
        try:
            db.execute(
                """
                INSERT INTO auto_research_jobs
                    (agenda_id, deep_insight_id, status, stage, assigned_worker,
                     last_error, last_note, last_checked_at, updated_at)
                VALUES (?, ?, 'review_pending', ?, ?, NULL, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (agenda_id, insight_id, worker_stage, assigned_worker, note),
            )
            db.commit()
            return True
        except Exception as exc:
            try:
                db.rollback()
            except Exception:
                pass
            if "constraint" in str(exc).lower() or "unique" in str(exc).lower():
                return False
            raise

    cur = db.execute(
        """
        UPDATE auto_research_jobs
        SET status='review_pending',
            stage=?,
            assigned_worker=?,
            last_error=NULL,
            last_note=?,
            last_checked_at=CURRENT_TIMESTAMP,
            updated_at=CURRENT_TIMESTAMP
        WHERE deep_insight_id=?
          AND agenda_id=?
          AND status=?
          AND COALESCE(stage, '')=?
          AND status NOT IN (
              'review_pending', 'researching', 'verifying',
              'running_experiment', 'running_gpu', 'running_cpu', 'queued_gpu'
          )
        """,
        (
            worker_stage,
            assigned_worker,
            note,
            insight_id,
            agenda_id,
            expected_status,
            expected_stage,
        ),
    )
    db.commit()
    return int(getattr(cur, "rowcount", 0) or 0) == 1


def _start_candidate_worker(candidate: dict, queue: str) -> bool:
    candidate_id = int(candidate["id"])
    worker_queue = queue
    if queue in {QUEUE_REVIEW, QUEUE_REPAIR} and not _claim_review_candidate(candidate, queue):
        return False
    with _active_queue_worker_lock:
        if candidate_id in _active_queue_workers:
            return False
        _active_queue_workers[candidate_id] = worker_queue

    def _run() -> None:
        try:
            _process_candidate(candidate)
        except Exception as exc:  # pragma: no cover - defensive worker guard
            _upsert_job(candidate_id, status="failed", stage="exception", last_error=str(exc))
            log_event("error", {"step": "auto_research_worker", "queue": queue, "insight_id": candidate_id, "error": str(exc)})
        finally:
            with _active_queue_worker_lock:
                _active_queue_workers.pop(candidate_id, None)

    thread = threading.Thread(target=_run, name=f"deepgraph-auto-{queue}-{candidate_id}", daemon=True)
    thread.start()
    return True


def _candidate_repair_stage(status: str, stage: str) -> bool:
    if status == "blocked" and (
        stage == "cpu_ineligible"
        or stage in {
            "verification_input_missing",
            "research_input_missing",
            "experiment_review_blocked",
            "experiment_review_repair_failed",
            "experiment_review_blocked_final",
        }
    ):
        return True
    return status == "failed" and stage in {
        "manual_reforge_unfinished",
        "manual_requeue_unfinished",
        "retry_failed_run",
        "manual_rerun_completed",
        "reset_completed_experiments",
    }


def _is_benchmark_completion_candidate(candidate: dict) -> bool:
    return (
        str(candidate.get("auto_status") or "").strip() == "queued"
        and str(candidate.get("auto_stage") or "").strip() == BENCHMARK_COMPLETION_STAGE
        and bool(candidate.get("auto_experiment_run_id"))
    )


def _candidate_queue_decision(candidate: dict) -> QueueDecision:
    active_worker_queue = _candidate_active_worker_queue(candidate.get("id"))
    if active_worker_queue:
        return QueueDecision(active_worker_queue, False, "worker already active for this insight")
    status = str(candidate.get("auto_status") or "").strip()
    stage = str(candidate.get("auto_stage") or "").strip()
    if status == HARNESS_REQUIRED_STATUS or stage == HARNESS_REQUIRED_STAGE:
        return QueueDecision(QUEUE_HARNESS, False, "waiting for benchmark harness agents")
    if status in {"completed", "bundle_ready", "smoke_only"} and stage != "tier1_research_complete":
        return QueueDecision(QUEUE_DONE, False, "terminal or non-formal job")
    if status in {"running_experiment", "running_gpu", "running_cpu", "queued_gpu"}:
        return QueueDecision(QUEUE_EXECUTION, False, "execution already active")
    if status == "queued" and stage == BENCHMARK_HARNESS_DESIGN_REPAIR_STAGE:
        return QueueDecision(QUEUE_REPAIR, True, "pre-execution benchmark/data-source design repair")
    if status == "queued" and stage in MANUSCRIPT_RETRY_STAGES and candidate.get("auto_experiment_run_id"):
        return QueueDecision(QUEUE_REPAIR, True, "manuscript quality-gate retry")
    if status == "review_pending":
        return QueueDecision(QUEUE_REVIEW, False, "review already pending")
    if status == "researching":
        return QueueDecision(QUEUE_RESEARCH, False, "deep research already active")
    if status == "verifying":
        return QueueDecision(QUEUE_VERIFICATION, False, "verification already active")
    if status == "blocked" and _candidate_still_missing_required_inputs(candidate):
        return QueueDecision(QUEUE_BLOCKED, False, "required inputs still missing")
    if _candidate_repair_stage(status, stage):
        return QueueDecision(QUEUE_REPAIR, True, "repairable blocked/failed state")
    if _candidate_needs_verification(candidate):
        return QueueDecision(QUEUE_VERIFICATION, True, "novelty verification required")
    if status in {"eligible", "queued_cpu", "queued_gpu"} or candidate.get("auto_experiment_run_id"):
        return QueueDecision(QUEUE_EXECUTION, True, "existing formal run can execute or resume")
    return QueueDecision(QUEUE_REVIEW, True, "ready for experiment review/forge")


def _candidate_queues(candidates: list[dict]) -> dict[str, list[tuple[dict, QueueDecision]]]:
    queues: dict[str, list[tuple[dict, QueueDecision]]] = {
        QUEUE_REPAIR: [],
        QUEUE_VERIFICATION: [],
        QUEUE_EXECUTION: [],
        QUEUE_REVIEW: [],
        QUEUE_HARNESS: [],
        QUEUE_BLOCKED: [],
        QUEUE_WAITING: [],
        QUEUE_DONE: [],
    }
    for candidate in candidates:
        decision = _candidate_queue_decision(candidate)
        queues.setdefault(decision.queue, []).append((candidate, decision))
    return queues


def _select_candidate_from_queues(candidates: list[dict] | None = None) -> tuple[dict | None, dict]:
    candidates = _candidate_pool() if candidates is None else candidates
    queues = _candidate_queues(candidates)
    counts = _queue_active_counts()
    summary = {queue: len(rows) for queue, rows in queues.items() if rows}
    if _queue_has_capacity(QUEUE_EXECUTION, counts):
        for candidate, decision in queues.get(QUEUE_EXECUTION, []):
            if decision.runnable and _is_benchmark_completion_candidate(candidate):
                return candidate, {
                    "selected_queue": QUEUE_EXECUTION,
                    "queue_counts": summary,
                    "active_counts": counts,
                    "decision": "benchmark completion is the pre-manuscript evidence gate",
                }
    for queue in QUEUE_ORDER:
        if not _queue_has_capacity(queue, counts):
            continue
        for candidate, decision in queues.get(queue, []):
            if decision.runnable:
                return candidate, {
                    "selected_queue": queue,
                    "queue_counts": summary,
                    "active_counts": counts,
                    "decision": decision.reason,
                }
    return None, {
        "selected_queue": None,
        "queue_counts": summary,
        "active_counts": counts,
    }


def _candidate_pool() -> list[dict]:
    rows = db.fetchall(
        """SELECT di.*,
                  arj.status AS auto_status,
                  arj.stage AS auto_stage,
                  arj.cpu_eligible AS auto_cpu_eligible,
                  arj.resource_class AS auto_resource_class,
                  arj.experiment_run_id AS auto_experiment_run_id,
                  arj.resource_grant_id AS auto_resource_grant_id,
                  arj.last_note AS auto_last_note,
                  arj.last_error AS auto_last_error
           FROM deep_insights di
           LEFT JOIN auto_research_jobs arj
             ON arj.deep_insight_id = di.id AND arj.agenda_id = di.agenda_id
           LEFT JOIN resource_grants rg
             ON rg.id = arj.resource_grant_id
            AND rg.agenda_id = di.agenda_id
            AND rg.idea_id = di.id
            AND rg.status = 'active'
            AND rg.expires_at > CURRENT_TIMESTAMP
           WHERE di.agenda_id IS NOT NULL
             AND arj.resource_grant_id IS NOT NULL
             AND rg.id IS NOT NULL
             AND COALESCE(di.status, 'candidate') NOT IN ('exists')
             AND COALESCE(di.outcome, 'pending') NOT IN ('cleaned', 'archived')
             AND COALESCE(di.novelty_status, '') NOT IN ('cleaned_similar_duplicate', 'exists')
             AND COALESCE(di.submission_status, 'not_started') NOT IN ('stale')
             AND """ + claim_predicate_sql() + """
           ORDER BY
             CASE
               WHEN arj.status='queued' AND arj.stage IN (
                 'manuscript_retry_after_quality_gate',
                 'manuscript_retry_after_soft_benchmark_gate',
                 'manuscript_blocked'
               ) THEN 0
               WHEN arj.status='queued' AND arj.stage='benchmark_harness_design_repair' THEN 0
               WHEN arj.status='queued' AND arj.stage IN (
                 'review_incomplete_reforge',
                 'schema_exception_repair',
                 'retry_failed_run',
                 'execution_retry'
               ) THEN 1
               WHEN arj.status='queued' AND arj.stage='review_retry' THEN 2
               WHEN arj.status='queued' AND arj.stage='harness_supported_subset_recovered' THEN 4
               ELSE 3
             END,
             arj.scheduler_priority DESC,
             arj.updated_at ASC,
             di.tier DESC,
             di.created_at DESC
           LIMIT ?""",
        (AUTO_RESEARCH_CANDIDATE_POOL_LIMIT,),
    )
    return rows


def _resource_experimentability(resource_class: str) -> str:
    if resource_class == "cpu":
        return "easy"
    if resource_class == "gpu_small":
        return "medium"
    if resource_class == "gpu_unavailable":
        return "blocked"
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
             AND COALESCE(di.outcome, 'pending') NOT IN ('cleaned', 'archived')
             AND COALESCE(di.novelty_status, '') NOT IN ('cleaned_similar_duplicate', 'exists')
             AND COALESCE(di.submission_status, 'not_started') NOT IN ('stale')
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
               WHERE id=? AND agenda_id=?""",
            (
                resource_class,
                experimentability,
                insight_id,
                int(insight["agenda_id"]),
            ),
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
    candidate, _ = _select_candidate_from_queues()
    return candidate


def _refresh_running_jobs() -> None:
    jobs = db.fetchall(
        """SELECT arj.*, di.novelty_status
           FROM auto_research_jobs arj
           JOIN deep_insights di
             ON di.agenda_id=arj.agenda_id
            AND di.id = arj.deep_insight_id
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
                    "UPDATE deep_insights SET novelty_status='unchecked', updated_at=CURRENT_TIMESTAMP WHERE id=? AND agenda_id=?",
                    (insight_id, int(job["agenda_id"])),
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
            if not run:
                run = db.fetchone(
                    """SELECT * FROM experiment_runs
                       WHERE deep_insight_id=? AND status='scaffolding'
                       ORDER BY id DESC LIMIT 1""",
                    (insight_id,),
                )
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
                if str(run.get("phase") or "") == "experiment_review_blocked":
                    _handle_experiment_review_blocked(
                        insight_id,
                        _blocked_review_payload_from_run(run),
                        source="review_failed_recovered",
                    )
                    log_event(
                        "auto_research",
                        {
                            "step": "auto_research_review_blocked_recovered",
                            "insight_id": insight_id,
                            "run_id": run["id"],
                        },
                    )
                else:
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
            elif run and _run_has_incomplete_review_scaffold(run):
                reason = (
                    "Recovered incomplete review scaffold: review decision exists but program_md or "
                    "success_criteria is empty; reforge required."
                )
                _supersede_stale_scaffold_run(int(run["id"]), reason)
                _upsert_job(
                    insight_id,
                    status="queued",
                    stage="review_incomplete_reforge",
                    experiment_run_id=None,
                    last_error=None,
                    last_note=reason,
                )
                log_event(
                    "warning",
                    {"step": "auto_research_review_incomplete_reforge", "insight_id": insight_id, "run_id": run["id"]},
                )
            elif run and _run_age_seconds(run) >= REVIEW_PENDING_STALE_SECONDS:
                _recover_stale_review_scaffold_run(insight_id, run, source="refresh_running_jobs")
            elif run and str(run.get("status") or "") == "scaffolding":
                _upsert_job(
                    insight_id,
                    touch_updated_at=False,
                    status="review_pending",
                    stage="experiment_review",
                    experiment_run_id=run["id"],
                    resource_class=run.get("resource_class") or job.get("resource_class"),
                    last_error=None,
                    last_note="Experiment forge is still generating scaffold metadata; waiting before stale recovery.",
                )
            elif job.get("last_error") and not job.get("experiment_run_id"):
                note = job.get("last_note") or job["last_error"]
                _handle_experiment_review_blocked(
                    insight_id,
                    {
                        "error": job["last_error"],
                        "judgement": {"summary": note, "blockers": [job["last_error"]], "warnings": []},
                    },
                    source="review_pending_error",
                )
            elif _job_age_seconds(job) >= REVIEW_WORKER_STALE_SECONDS and not job.get("experiment_run_id"):
                _upsert_job(
                    insight_id,
                    status="queued",
                    stage="review_retry",
                    last_error=None,
                    last_note=(
                        "Structured experiment review worker did not produce a run or blocker "
                        "before the worker stale timeout; requeued for retry."
                    ),
                )
                log_event("warning", {"step": "auto_research_review_worker_stale", "insight_id": insight_id})
        elif job["status"] in {"running_experiment", "running_gpu", "running_cpu", "queued_gpu"} and job.get("experiment_run_id"):
            run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (job["experiment_run_id"],))
            if not run:
                _upsert_job(insight_id, status="failed", stage="missing_run", last_error="Experiment run missing.")
            elif run["status"] == "completed":
                if job.get("stage") == BENCHMARK_COMPLETION_STAGE:
                    _queue_benchmark_completion_run(
                        insight_id,
                        run,
                        str(job.get("resource_class") or run.get("resource_class") or "gpu_large"),
                    )
                    continue
                if _run_closed_loop_complete(int(run["id"])):
                    note = f"Verdict={run.get('hypothesis_verdict')}, effect_pct={run.get('effect_pct')}"
                    _upsert_job(insight_id, status="completed", stage="closed_loop_complete", last_note=note)
                    v = (run.get("hypothesis_verdict") or "").lower()
                    apply_experiment_finished_deep(
                        insight_id,
                        verdict=run.get("hypothesis_verdict"),
                        success=(
                            v in {"confirmed", "supported"}
                            and positive_decision_authorized(
                                agenda_id=int(run.get("agenda_id") or 0),
                                run_id=int(run.get("id") or 0),
                            )
                        ),
                        inconclusive=v == "inconclusive",
                    )
                else:
                    _upsert_job(
                        insight_id,
                        touch_updated_at=False,
                        status=job.get("status") or "running_gpu",
                        stage=job.get("stage") or "gpu_scheduler",
                        experiment_run_id=run["id"],
                        resource_class=job.get("resource_class") or run.get("resource_class"),
                        assigned_worker=job.get("assigned_worker"),
                        last_error=None,
                        last_note="Experiment completed; waiting for manuscript/submission bundle before closing the loop.",
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
                running_stage = BENCHMARK_COMPLETION_STAGE if job.get("stage") == BENCHMARK_COMPLETION_STAGE else "gpu_scheduler"
                running_note = (
                    "Full benchmark completion GPU job running."
                    if running_stage == BENCHMARK_COMPLETION_STAGE
                    else "GPU job running."
                )
                _upsert_job(insight_id, status="running_gpu", stage=running_stage, last_note=running_note)
            elif run["status"] == "running_cpu":
                _upsert_job(
                    insight_id,
                    status="running_cpu",
                    stage="validation_loop",
                    last_note="CPU validation loop running.",
                    touch_updated_at=False,
                )
            elif run["status"] in {"reproducing", "testing"}:
                gpu_lane = job["status"] in {"running_gpu", "queued_gpu"}
                if not _is_execution_live_in_process(run.get("id")) and not (gpu_lane and _gpu_execution_live_for_run(run.get("id"))):
                    _requeue_stale_execution_job(
                        job,
                        (
                            "Recovered interrupted SciForge run; "
                            f"resuming from phase {run.get('phase') or 'validation_loop'}."
                        ),
                    )
                    continue
                lane_status = "running_gpu" if gpu_lane else "running_cpu"
                phase = run.get("phase") or "validation_loop"
                note = f"SciForge {phase}: best={run.get('best_metric_value')}, baseline={run.get('baseline_metric_value')}."
                _upsert_job(
                    insight_id,
                    status=lane_status,
                    stage=phase,
                    last_note=note,
                    last_error=None,
                    touch_updated_at=False,
                )


def recover_blocked_manuscript_jobs(limit: int = 50) -> int:
    """Queue latest blocked manuscript runs for targeted writing repair."""
    rows = db.fetchall(
        """
        SELECT mr.deep_insight_id,
               mr.experiment_run_id,
               mr.status AS manuscript_status,
               mr.workdir AS manuscript_workdir,
               arj.id AS auto_job_id,
               arj.status AS auto_status,
               arj.stage AS auto_stage,
               arj.resource_class AS auto_resource_class,
               er.resource_class AS run_resource_class,
               er.error_message AS run_error_message
        FROM manuscript_runs mr
        JOIN (
            SELECT deep_insight_id, MAX(id) AS latest_id
            FROM manuscript_runs
            WHERE status IN ('manuscript_blocked', 'needs_revision')
            GROUP BY deep_insight_id
        ) latest ON latest.latest_id = mr.id
        JOIN deep_insights di ON di.id = mr.deep_insight_id
        LEFT JOIN auto_research_jobs arj ON arj.deep_insight_id = mr.deep_insight_id
        LEFT JOIN experiment_runs er ON er.id = mr.experiment_run_id
        WHERE COALESCE(di.status, 'candidate') NOT IN ('exists')
          AND COALESCE(di.outcome, 'pending') NOT IN ('cleaned', 'archived')
          AND COALESCE(di.novelty_status, '') NOT IN ('cleaned_similar_duplicate', 'exists')
          AND COALESCE(di.submission_status, 'not_started') NOT IN ('stale')
          AND mr.experiment_run_id IS NOT NULL
        ORDER BY mr.updated_at ASC
        LIMIT ?
        """,
        (limit,),
    )
    recovered = 0
    active_statuses = {
        "queued",
        "queued_cpu",
        "queued_gpu",
        "running_cpu",
        "running_gpu",
        "running_experiment",
        "review_pending",
        "researching",
        "verifying",
    }
    active_stages = set(MANUSCRIPT_RETRY_STAGES) | {
        BENCHMARK_COMPLETION_STAGE,
        "manuscript_revision",
        "gpu_scheduler",
        "validation_loop",
        "experiment_review",
    }
    for row in rows:
        auto_status = str(row.get("auto_status") or "").strip()
        auto_stage = str(row.get("auto_stage") or "").strip()
        if auto_status in active_statuses and (not auto_stage or auto_stage in active_stages):
            continue
        insight_id = int(row["deep_insight_id"])
        run_id = int(row["experiment_run_id"])
        run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
        retry_blocker = _manuscript_retry_blocker(run)
        if retry_blocker:
            _block_invalid_manuscript_retry_run(insight_id, run_id, retry_blocker)
            recovered += 1
            continue
        blocker = str(row.get("run_error_message") or row.get("manuscript_status") or "Manuscript quality gate failed.").strip()
        _upsert_job(
            insight_id,
            status="queued",
            stage="manuscript_retry_after_quality_gate",
            experiment_run_id=run_id,
            resource_class=row.get("auto_resource_class") or row.get("run_resource_class") or "cpu",
            assigned_worker=None,
            last_note="Recovered blocked manuscript and queued targeted writing/TeX/figure revision.",
            last_error=blocker[:4000],
        )
        log_event(
            "auto_research",
            {
                "step": "blocked_manuscript_recovered",
                "insight_id": insight_id,
                "run_id": run_id,
                "manuscript_status": row.get("manuscript_status"),
                "workdir": row.get("manuscript_workdir"),
            },
        )
        recovered += 1
    return recovered


def recover_soft_benchmark_completion_jobs(limit: int = 50) -> int:
    """Release confirmed runs that were blocked only by extended benchmark gaps."""
    rows = db.fetchall(
        """
        SELECT arj.*, er.status AS run_status, er.hypothesis_verdict
        FROM auto_research_jobs arj
        JOIN experiment_runs er
          ON er.agenda_id=arj.agenda_id
         AND er.id = arj.experiment_run_id
        WHERE arj.status='queued'
          AND arj.stage=?
          AND er.status='completed'
          AND LOWER(COALESCE(er.hypothesis_verdict, '')) IN ('confirmed', 'supported')
        ORDER BY arj.updated_at ASC
        LIMIT ?
        """,
        (BENCHMARK_COMPLETION_STAGE, limit),
    )
    recovered = 0
    for row in rows:
        if not positive_decision_authorized(
            agenda_id=int(row.get("agenda_id") or 0),
            run_id=int(row.get("experiment_run_id") or 0),
        ):
            continue
        raw_error = str(row.get("last_error") or "").strip()
        blocker_items = [item.strip() for item in raw_error.split(";") if item.strip()]
        bundle = {"error": raw_error, "submission_blockers": blocker_items}
        if benchmark_completion_blockers(bundle):
            continue
        db.execute(
            """
            UPDATE auto_research_jobs
            SET status='queued',
                stage='manuscript_retry_after_soft_benchmark_gate',
                assigned_worker=NULL,
                last_error=NULL,
                last_note=?,
                last_checked_at=CURRENT_TIMESTAMP,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=? AND agenda_id=?
            """,
            (
                "Recovered from soft benchmark-completion gate; confirmed run can draft manuscript while extra baselines/seeds/ablations remain follow-up work.",
                row["id"],
                int(row["agenda_id"]),
            ),
        )
        log_event(
            "auto_research",
            {
                "step": "soft_benchmark_completion_recovered",
                "insight_id": row.get("deep_insight_id"),
                "run_id": row.get("experiment_run_id"),
            },
        )
        recovered += 1
    if recovered:
        db.commit()
    return recovered


def _launch_candidates_to_capacity() -> dict:
    recovered_execution = recover_stale_execution_jobs()
    archived_inactive_harness = archive_inactive_benchmark_harness_jobs()
    recovered_harness = recover_partially_supported_harness_jobs()
    repaired_harness_design = repair_benchmark_harness_design_jobs()
    consumed_harness = process_benchmark_harness_jobs()
    _refresh_running_jobs()
    scheduled: list[int] = []
    seen_candidates: set[int] = set()
    last_selection: dict = {}

    while True:
        candidate, selection = _select_candidate_from_queues()
        last_selection = selection
        if not candidate:
            break
        candidate_id = int(candidate["id"])
        if candidate_id in seen_candidates:
            break
        seen_candidates.add(candidate_id)
        queue = selection.get("selected_queue") if isinstance(selection, dict) else None
        if queue in {QUEUE_REVIEW, QUEUE_REPAIR, QUEUE_EXECUTION}:
            if _start_candidate_worker(candidate, str(queue)):
                scheduled.append(candidate_id)
                continue
            continue
        try:
            _process_candidate(candidate)
            scheduled.append(candidate_id)
        except Exception as exc:  # pragma: no cover - defensive background guard
            error = str(exc)
            if "object has no attribute 'get'" in error:
                _upsert_job(
                    candidate_id,
                    status="queued",
                    stage="schema_exception_repair",
                    experiment_run_id=None,
                    assigned_worker=None,
                    last_error=None,
                    last_note=(
                        "Recovered schema exception from a list/dict mismatch; "
                        "requeued for structured review/forge with normalized inputs. "
                        f"Original error: {error[:300]}"
                    ),
                )
                log_event("warning", {"step": "auto_research_schema_exception_requeued", "insight_id": candidate_id, "error": error})
                continue
            _upsert_job(candidate_id, status="failed", stage="exception", last_error=error)
            log_event("error", {"step": "auto_research", "insight_id": candidate_id, "error": error})
            break

    active_counts = _queue_active_counts()
    queue_counts = last_selection.get("queue_counts", {}) if isinstance(last_selection, dict) else {}
    return {
        "scheduled": scheduled,
        "active": _active_job_count(),
        "execution_active": active_counts.get(QUEUE_EXECUTION, 0),
        "verifying_active": active_counts.get(QUEUE_VERIFICATION, 0),
        "review_active": active_counts.get(QUEUE_REVIEW, 0),
        "repair_active": active_counts.get(QUEUE_REPAIR, 0),
        "research_active": active_counts.get(QUEUE_RESEARCH, 0),
        "queue_counts": queue_counts,
        "selected_queue": last_selection.get("selected_queue") if isinstance(last_selection, dict) else None,
        "recovered_execution": recovered_execution,
        "archived_inactive_harness": archived_inactive_harness,
        "recovered_harness": recovered_harness,
        "repaired_harness_design": repaired_harness_design,
        "consumed_harness": consumed_harness,
    }


def _process_candidate(insight: dict) -> None:
    insight_id = insight["id"]
    tier = insight.get("tier")
    initial_auto_stage = str(insight.get("auto_stage") or "").strip()
    preserve_queue_stage = (
        initial_auto_stage == BENCHMARK_COMPLETION_STAGE
        or initial_auto_stage in MANUAL_REFORGE_STAGES
        or initial_auto_stage in MANUSCRIPT_RETRY_STAGES
    )

    if initial_auto_stage == BENCHMARK_HARNESS_DESIGN_REPAIR_STAGE:
        _run_benchmark_harness_design_repair_job(int(insight_id))
        return

    if initial_auto_stage in MANUSCRIPT_RETRY_STAGES and insight.get("auto_experiment_run_id"):
        run_id = int(insight["auto_experiment_run_id"])
        run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
        blocker = _manuscript_retry_blocker(run)
        if blocker:
            _block_invalid_manuscript_retry_run(int(insight_id), run_id, blocker)
            return
        _run_manuscript_retry_job(
            int(insight_id),
            run_id,
            insight.get("auto_resource_class") or insight.get("resource_class"),
        )
        return

    resource_class, reason = assess_experiment_route(insight)
    if resource_class == "gpu_unavailable":
        _upsert_job(
            insight_id,
            status="blocked",
            stage="gpu_unavailable",
            cpu_eligible=0,
            cpu_reason=reason,
            resource_class=resource_class,
            scheduler_priority=0,
            last_error=reason,
            last_note=(
                "Idea requires GPU resources, but this runtime has no usable local GPU "
                "and no configured SSH GPU worker. Set [compute].local_gpu_policy='force_cpu' "
                "to prefer CPU-only ideas, configure [gpu.remote], or set the policy to 'ignore' "
                "if you intentionally want to queue GPU work."
            ),
        )
        log_event("auto_research", {"step": "gpu_unavailable_block", "insight_id": insight_id, "reason": reason})
        return
    _upsert_job(
        insight_id,
        cpu_eligible=1,
        cpu_reason=reason,
        resource_class=resource_class,
        scheduler_priority=2 if resource_class == "gpu_large" else 1,
    )

    if _maybe_repair_preexisting_review_block(insight):
        return

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
        if not preserve_queue_stage:
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
        optional_research_already_tried = initial_auto_stage in OPTIONAL_RESEARCH_NONBLOCKING_STAGES
        if evosci_available() and not preserve_queue_stage and not optional_research_already_tried:
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
        elif not preserve_queue_stage and not optional_research_already_tried:
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
        prior_review_repairs = _experiment_review_repair_attempt_from_plan_data(insight.get("experimental_plan"))
        max_review_repairs = max(0, MAX_EXPERIMENT_REVIEW_REPAIR_ATTEMPTS)
        if max_review_repairs and prior_review_repairs >= max_review_repairs:
            tag = _repair_tag("experiment_review", max_review_repairs, max_review_repairs)
            summary = (
                f"{tag} automatic review repair exhausted before another forge attempt; "
                "routing to benchmark/code harness agents."
            )
            judgement = {
                "summary": summary,
                "blockers": ["Experiment review repair exhausted before forge; benchmark/code harness intervention required."],
                "warnings": [],
            }
            if _queue_benchmark_harness_required(
                insight_id,
                {"error": judgement["blockers"][0], "judgement": judgement},
                judgement=judgement,
                source="experiment_review_repair_exhausted_pre_forge",
                summary=summary,
            ):
                return
            _upsert_job(
                insight_id,
                status="blocked",
                stage="experiment_review_blocked_final",
                experiment_run_id=None,
                last_error=judgement["blockers"][0],
                last_note=summary,
            )
            log_event(
                "warning",
                {
                    "step": "experiment_review_repair_exhausted_pre_forge",
                    "insight_id": insight_id,
                    "attempts": prior_review_repairs,
                },
            )
            return
        _upsert_job(
            insight_id,
            status="review_pending",
            stage="experiment_review",
            last_note=background_research_note or "Running structured experiment review before forge.",
        )
        forged = forge_experiment(
            insight_id,
            resource_grant_id=insight.get("auto_resource_grant_id"),
        )
        if "error" in forged:
            route = forged.get("route")
            if route == "blocked":
                _handle_experiment_review_blocked(insight_id, forged, source="forge_review")
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
            "UPDATE experiment_runs SET resource_class=? WHERE id=? AND agenda_id=?",
            (
                resource_class,
                forged["run_id"],
                int(insight["agenda_id"]),
            ),
        )
        db.commit()
    elif not _run_scaffold_ready(existing_run) and existing_run.get("status") in {"scaffolding"}:
        if _run_has_incomplete_review_scaffold(existing_run):
            reason = (
                "Existing review scaffold is incomplete: review decision exists but program_md or "
                "success_criteria is empty; superseding and reforge is required."
            )
            _supersede_stale_scaffold_run(int(existing_run["id"]), reason)
            _upsert_job(
                insight_id,
                status="queued",
                stage="review_incomplete_reforge",
                experiment_run_id=None,
                resource_class=resource_class,
                last_note=reason,
                last_error=None,
            )
            return
        else:
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
            "UPDATE experiment_runs SET resource_class=? WHERE id=? AND agenda_id=?",
            (
                resource_class,
                existing_run["id"],
                int(insight["agenda_id"]),
            ),
        )
        db.commit()

    if existing_run["status"] in {"completed"} and _auto_job_stage(insight_id) == BENCHMARK_COMPLETION_STAGE:
        completion_resource_class = str(
            insight.get("auto_resource_class")
            or existing_run.get("resource_class")
            or resource_class
            or "gpu_large"
        )
        if not completion_resource_class.startswith("gpu"):
            completion_resource_class = "gpu_large"
        _queue_benchmark_completion_run(insight_id, existing_run, completion_resource_class)
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
        if _run_has_automation_failure(existing_run):
            if _retry_failed_run_with_repair(insight_id, existing_run, resource_class):
                return
            _upsert_job(
                insight_id,
                status="blocked",
                stage="experiment_automation_failed_final",
                experiment_run_id=existing_run["id"],
                resource_class=resource_class,
                last_error=existing_run.get("error_message"),
                last_note="Automation produced no benchmarked candidate method change and automatic repair attempts are exhausted.",
            )
            return
        verdict = str(existing_run.get("hypothesis_verdict") or "").strip().lower()
        if verdict in {"confirmed", "supported"}:
            if not positive_decision_authorized(
                agenda_id=int(existing_run.get("agenda_id") or 0),
                run_id=int(existing_run.get("id") or 0),
            ):
                _upsert_job(
                    insight_id,
                    status="review_pending",
                    stage="scientific_decision_required",
                    experiment_run_id=existing_run["id"],
                    resource_class=resource_class,
                    last_note=(
                        "Execution reported support; waiting for evidence audit "
                        "and an independent scientific decision."
                    ),
                    last_error=None,
                )
                return
            benchmark_bundle = benchmark_completion_bundle_from_run(existing_run["id"])
            if schedule_benchmark_completion(
                insight_id,
                existing_run["id"],
                benchmark_bundle,
                source="auto_research_completed_run_pre_manuscript",
                resource_class=resource_class,
            ):
                log_event(
                    "auto_research",
                    {
                        "step": "benchmark_completion_queued_before_completed_run_manuscript",
                        "insight_id": insight_id,
                        "run_id": existing_run["id"],
                    },
                )
                return
            try:
                bundle = generate_submission_bundle(existing_run["id"])
            except Exception as exc:
                bundle = {"error": str(exc)}
            if schedule_benchmark_completion(
                insight_id,
                existing_run["id"],
                bundle,
                source="auto_research_completed_run",
                resource_class=resource_class,
            ):
                log_event(
                    "auto_research",
                    {
                        "step": "benchmark_completion_queued_from_completed_run",
                        "insight_id": insight_id,
                        "run_id": existing_run["id"],
                    },
                )
                return
            bundle_ok = "error" not in bundle
            retry_fields = _bundle_failure_retry_fields(bundle if isinstance(bundle, dict) else None)
            if retry_fields:
                _upsert_job(
                    insight_id,
                    experiment_run_id=existing_run["id"],
                    resource_class=resource_class,
                    **retry_fields,
                )
            else:
                _upsert_job(
                    insight_id,
                    status="bundle_ready" if bundle_ok else "completed",
                    stage="writing_submission" if bundle_ok else "closed_loop_complete",
                    experiment_run_id=existing_run["id"],
                    resource_class=resource_class,
                    artifact_bundle_id=(bundle.get("bundle_ids") or [None])[-1] if isinstance(bundle, dict) else None,
                    last_note=f"Completed confirmed run reused. Submission bundle status={'ok' if bundle_ok else 'failed'}.",
                    last_error=None if bundle_ok else str(bundle.get("error") if isinstance(bundle, dict) else bundle),
                )
            return
        note = f"Verdict={existing_run.get('hypothesis_verdict')}, effect_pct={existing_run.get('effect_pct')}"
        _upsert_job(insight_id, status="completed", stage="closed_loop_complete", experiment_run_id=existing_run["id"], last_note=note)
        return
    if existing_run["status"] in {"failed"}:
        if _retry_failed_run_with_repair(insight_id, existing_run, resource_class):
            return
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

    preflight_blocker = gpu_scheduler._capability_preflight_blocker(existing_run)
    if preflight_blocker:
        db.execute(
            """
            UPDATE experiment_runs
            SET status='failed', phase='invalid_benchmark_design', error_message=?,
                completed_at=COALESCE(completed_at, CURRENT_TIMESTAMP)
            WHERE id=? AND agenda_id=?
            """,
            (
                preflight_blocker,
                existing_run["id"],
                int(insight["agenda_id"]),
            ),
        )
        db.commit()
        _upsert_job(
            insight_id,
            status="failed",
            stage="capability_preflight_blocked",
            experiment_run_id=existing_run["id"],
            resource_class=resource_class,
            last_error=preflight_blocker,
            last_note="Compute blocked because its grant is not bound to a passed structured capability preflight.",
        )
        set_outcome(
            "deep_insights",
            insight_id,
            OUTCOME_EXPERIMENT_FAILED_RUN,
            reason=preflight_blocker,
            triggered_by="experiment",
        )
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
            compute_job = meta_compute_runtime.submit_experiment_run(
                agenda_id=int(existing_run.get("agenda_id") or 0),
                idea_id=insight_id,
                experiment_run_id=int(existing_run["id"]),
                resource_grant_id=int(
                    existing_run.get("resource_grant_id") or 0
                ),
                timeout_seconds=GPU_JOB_TIMEOUT_SECONDS,
            )
            note = (
                "Queued through ComputeScheduler as "
                f"{compute_job.backend_job_id}."
            )
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

    compute_job = meta_compute_runtime.submit_experiment_run(
        agenda_id=int(existing_run.get("agenda_id") or 0),
        idea_id=insight_id,
        experiment_run_id=int(existing_run["id"]),
        resource_grant_id=int(existing_run.get("resource_grant_id") or 0),
        timeout_seconds=max(EXPERIMENT_TIME_BUDGET, GPU_JOB_TIMEOUT_SECONDS),
        backend_kind="cpu",
    )
    if _active_execution_run_id() is not None:
        _upsert_job(
            insight_id,
            status="queued_cpu",
            stage="cpu_execution_wait",
            experiment_run_id=existing_run["id"],
            resource_class=resource_class,
            last_note=(
                "CPU validation lane is busy; durable compute job "
                f"{compute_job.backend_job_id} remains queued."
            ),
            last_error=None,
        )
        return
    meta_compute_runtime.mark_cpu_running(compute_job)
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
    try:
        result = _execute_cpu_validation_loop(insight_id, existing_run["id"])
    except Exception as exc:
        db.execute(
            """
            UPDATE experiment_runs
            SET status='failed', error_message=?,
                completed_at=COALESCE(completed_at, CURRENT_TIMESTAMP)
            WHERE id=? AND agenda_id=?
              AND status NOT IN ('completed', 'failed', 'cancelled')
            """,
            (
                f"cpu_validation_exception:{type(exc).__name__}",
                int(existing_run["id"]),
                int(existing_run["agenda_id"]),
            ),
        )
        db.commit()
        meta_compute_runtime.settle_cpu_run(int(existing_run["id"]))
        raise
    run_after_cpu = db.fetchone(
        """
        SELECT status, error_message
        FROM experiment_runs
        WHERE id=? AND agenda_id=?
        """,
        (int(existing_run["id"]), int(existing_run["agenda_id"])),
    ) or {}
    if str(run_after_cpu.get("status") or "") != "completed":
        failure = (
            str(run_after_cpu.get("error_message") or "").strip()
            or str((result or {}).get("reason") or (result or {}).get("error") or "")
            or "cpu_validation_did_not_complete"
        )
        db.execute(
            """
            UPDATE experiment_runs
            SET status='failed', error_message=?,
                completed_at=COALESCE(completed_at, CURRENT_TIMESTAMP)
            WHERE id=? AND agenda_id=?
              AND status NOT IN ('failed', 'cancelled')
            """,
            (
                failure[:4000],
                int(existing_run["id"]),
                int(existing_run["agenda_id"]),
            ),
        )
        db.commit()
    try:
        compute_status = meta_compute_runtime.settle_cpu_run(
            int(existing_run["id"])
        )
    except Exception as exc:
        db.execute(
            """
            UPDATE experiment_runs
            SET status='failed', error_message=?,
                completed_at=COALESCE(completed_at, CURRENT_TIMESTAMP)
            WHERE id=? AND agenda_id=? AND status='completed'
            """,
            (
                f"compute_certification_failed:{type(exc).__name__}",
                int(existing_run["id"]),
                int(existing_run["agenda_id"]),
            ),
        )
        db.commit()
        raise
    if compute_status != "succeeded":
        raise RuntimeError(
            f"CPU validation did not settle successfully: {compute_status}"
        )
    process_completed_run(existing_run["id"])
    benchmark_bundle = benchmark_completion_bundle_from_run(existing_run["id"], result=result if isinstance(result, dict) else None)
    if schedule_benchmark_completion(
        insight_id,
        existing_run["id"],
        benchmark_bundle,
        source="auto_research_cpu_pre_manuscript",
        resource_class=resource_class,
    ):
        log_event(
            "auto_research",
            {
                "step": "benchmark_completion_queued_before_cpu_manuscript",
                "insight_id": insight_id,
                "run_id": existing_run["id"],
            },
        )
        return
    bundle = generate_submission_bundle(existing_run["id"])
    if schedule_benchmark_completion(
        insight_id,
        existing_run["id"],
        bundle,
        source="auto_research_cpu",
        resource_class=resource_class,
    ):
        log_event(
            "auto_research",
            {
                "step": "benchmark_completion_queued",
                "insight_id": insight_id,
                "run_id": existing_run["id"],
            },
        )
        return
    retry_fields = _bundle_failure_retry_fields(bundle if isinstance(bundle, dict) else None)
    if retry_fields:
        _upsert_job(
            insight_id,
            experiment_run_id=existing_run["id"],
            resource_class=resource_class,
            **retry_fields,
        )
    else:
        _upsert_job(
            insight_id,
            status="bundle_ready" if "error" not in bundle else "completed",
            stage="writing_submission" if "error" not in bundle else "closed_loop_complete",
            experiment_run_id=existing_run["id"],
            artifact_bundle_id=(bundle.get("bundle_ids") or [None])[-1],
            last_note=f"Completed with verdict={result.get('verdict', 'unknown')}. Submission bundle status={'ok' if 'error' not in bundle else 'failed'}.",
            last_error=None if "error" not in bundle else str(bundle.get("error")),
        )
    log_event("auto_research", {"step": "experiment_completed", "insight_id": insight_id, "run_id": existing_run["id"], "verdict": result.get("verdict")})


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
        if event_type == "deep_insight_created":
            insight_id = payload.get("insight_id")
            insight = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (insight_id,))
            if insight and _insight_is_archived_or_cleaned(insight):
                processed += 1
                continue
            if insight:
                _upsert_job(
                    int(insight_id),
                    status="queued",
                    stage="idea_ready",
                    last_error=None,
                    last_note="Queued by deep_insight_created event for multi-queue scheduling.",
                )
                processed += 1
        else:
            _refresh_running_jobs()
            processed += 1
    db.ack_pipeline_events(AUTO_RESEARCH_CONSUMER, last_event_id)
    return {"events": len(events), "processed": processed}


def run_cycle() -> dict:
    """Queue one explicitly agenda-scoped candidate for portfolio review.

    The legacy global recovery/consumer path is intentionally not invoked:
    pre-migration backlog has no agenda_id and remains excluded until an
    audited explicit import.
    """
    db.init_db()
    from agents.agenda_orchestrator import run_scoped_cycle

    return run_scoped_cycle()

def _run_once() -> dict:
    db.init_db()
    cycle_stats = run_cycle()
    active = _active_job_count()
    return {
        # Legacy pipeline events are not consumed by meta-harness-v1 because
        # they do not carry a mandatory agenda_id.
        "events": 0,
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
