"""Recommend experiment tuning and pipeline decisions from progress + feasibility."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from agents.experiment_feasibility_agent import FeasibilityReport
from agents.experiment_progress_monitor import ProgressReport
from config import (
    EXPERIMENT_REAL_BENCHMARK_MAX_EXAMPLES,
    EXPERIMENT_REAL_BENCHMARK_SEEDS,
    EXPERIMENT_REAL_BENCHMARK_TIME_BUDGET,
    EXPERIMENT_WATCHDOG_USE_LLM,
)


@dataclass
class TuningDecision:
    role: str = "ExperimentTuningAgent"
    action: str = "continue"  # continue | retry | finalize_partial | abort | scale_down | scale_up_time | route_gpu
    reason: str = ""
    parameter_updates: dict[str, Any] = field(default_factory=dict)
    next_checks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def recommend_tuning(
    *,
    insight: dict[str, Any],
    run: dict[str, Any],
    progress: ProgressReport,
    feasibility: FeasibilityReport,
    partial_summary: dict[str, Any] | None = None,
    proxy: dict[str, Any] | None = None,
) -> TuningDecision:
    """Deterministic tuning/decision policy for the experiment watchdog."""
    proxy = dict(proxy or {})
    decision = TuningDecision()

    if feasibility.verdict == "not_viable" and progress.crash_streak >= 2:
        if partial_summary and partial_summary.get("per_method"):
            decision.action = "finalize_partial"
            decision.reason = "Repeated timeouts but sufficient predictions exist to extract partial metrics."
            decision.next_checks.append("write benchmark_summary.json from predictions")
            return decision
        decision.action = "scale_down"
        decision.reason = "Repeated failures; reduce benchmark cost before next attempt."
        decision.parameter_updates = {
            "time_budget_seconds": max(1800, int(EXPERIMENT_REAL_BENCHMARK_TIME_BUDGET)),
            "benchmark_max_examples": min(16, int(EXPERIMENT_REAL_BENCHMARK_MAX_EXAMPLES)),
            "benchmark_seeds": 1,
            "reproduction_iterations": 1,
        }
        decision.next_checks.append("re-run reproduction with reduced contract")
        return decision

    if progress.status == "stale" and progress.predictions_lines > 100:
        if partial_summary:
            decision.action = "finalize_partial"
            decision.reason = "Process stopped but artifacts support partial metric finalization."
            return decision
        decision.action = "retry"
        decision.reason = "Stale run without summarizable artifacts; retry reproduction."
        return decision

    if progress.subprocess_running:
        if progress.time_budget_seconds and progress.elapsed_seconds:
            remaining = progress.time_budget_seconds - progress.elapsed_seconds
            if remaining < 600:
                decision.action = "scale_up_time"
                decision.reason = "Active run close to budget; extend time_budget for completion."
                decision.parameter_updates["time_budget_seconds"] = int(
                    max(progress.time_budget_seconds * 1.5, EXPERIMENT_REAL_BENCHMARK_TIME_BUDGET)
                )
                return decision
        decision.action = "continue"
        decision.reason = "Experiment actively producing predictions."
        decision.next_checks.append("wait for subprocess completion or benchmark_summary")
        return decision

    if partial_summary and not partial_summary.get("full_benchmark_completed"):
        coverage = progress.coverage_pct or 0
        if coverage >= 50 or progress.predictions_lines >= 500:
            decision.action = "finalize_partial"
            decision.reason = "Partial benchmark sufficient to unblock validation loop."
            return decision

    resource_class = str(insight.get("resource_class") or "cpu")
    run_status = str(run.get("status") or "")
    real_benchmark = bool(proxy.get("real_benchmark_required") or proxy.get("benchmark_model"))
    if run_status in {"reproducing", "testing", "running_cpu"} and (
        resource_class.startswith("gpu") or real_benchmark
    ):
        decision.action = "route_gpu"
        decision.reason = (
            "GPU-class or real-benchmark workload should use gpu_scheduler "
            "instead of blocking the CPU lane."
        )
        decision.next_checks.append("enqueue gpu_jobs row for experiment_run")
        return decision

    if feasibility.verdict == "viable":
        decision.action = "continue"
        decision.reason = "Feasibility checks passed; keep current experiment contract."
        return decision

    decision.action = "continue"
    decision.reason = "No strong signal to change parameters; monitor next cycle."
    decision.next_checks.append("re-check progress in 2-5 minutes")

    if EXPERIMENT_WATCHDOG_USE_LLM:
        _maybe_enrich_with_llm(decision, insight=insight, progress=progress, feasibility=feasibility)

    return decision


def _maybe_enrich_with_llm(
    decision: TuningDecision,
    *,
    insight: dict[str, Any],
    progress: ProgressReport,
    feasibility: FeasibilityReport,
) -> None:
    try:
        from agents.llm_client import call_llm_json
    except ImportError:
        return

    prompt = {
        "task": "Choose the next experiment orchestration action.",
        "insight_title": insight.get("title"),
        "current_action": decision.action,
        "progress_status": progress.status,
        "feasibility": feasibility.to_dict(),
        "allowed_actions": [
            "continue",
            "retry",
            "finalize_partial",
            "abort",
            "scale_down",
            "scale_up_time",
            "route_gpu",
        ],
        "output_schema": {"action": "string", "reason": "string"},
    }
    try:
        payload, _tokens = call_llm_json(
            "You are the Experiment Tuning Agent. Prefer finalize_partial when predictions exist after timeout.",
            json.dumps(prompt, ensure_ascii=False),
            temperature=0.1,
        )
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    action = str(payload.get("action") or "").strip().lower()
    if action in {
        "continue",
        "retry",
        "finalize_partial",
        "abort",
        "scale_down",
        "scale_up_time",
        "route_gpu",
    }:
        decision.action = action
    reason = str(payload.get("reason") or "").strip()
    if reason:
        decision.reason = reason
