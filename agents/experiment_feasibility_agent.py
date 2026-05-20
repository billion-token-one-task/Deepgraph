"""Assess whether an experiment plan is viable on this machine and codebase."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agents.experiment_progress_monitor import ProgressReport
from config import EXPERIMENT_WATCHDOG_USE_LLM


@dataclass
class FeasibilityReport:
    role: str = "ExperimentFeasibilityAgent"
    verdict: str = "unknown"  # viable | risky | not_viable | needs_contract_change
    confidence: float = 0.5
    runnable: bool = False
    publishable_path: bool = False
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    recommended_route: str = "full_paper"  # full_paper | smoke_test | deferred | blocked

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_judgement(workdir: Path) -> dict[str, Any]:
    path = workdir / "spec" / "experiment_judgement.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def assess_feasibility(
    *,
    insight: dict[str, Any],
    run: dict[str, Any],
    progress: ProgressReport,
    partial_summary: dict[str, Any] | None = None,
) -> FeasibilityReport:
    """Rule-based feasibility assessment; optional LLM narrative when enabled."""
    report = FeasibilityReport()
    workdir = Path(run.get("workdir") or "")
    judgement = _load_judgement(workdir)

    if judgement.get("recommended_route") == "blocked":
        report.verdict = "not_viable"
        report.runnable = False
        report.recommended_route = "deferred"
        report.blockers.extend(judgement.get("blockers") or ["experiment review blocked"])
        report.confidence = 0.85
        return report

    if not progress.alive:
        report.verdict = "not_viable"
        report.blockers.append("experiment workdir missing")
        report.confidence = 0.9
        return report

    if progress.crash_streak >= 3:
        report.verdict = "not_viable"
        report.blockers.append(f"{progress.crash_streak} consecutive crashed reproduction attempts")
        report.recommended_route = "smoke_test"
        report.confidence = 0.8
        return report

    if progress.status == "stale" and not progress.subprocess_running:
        report.verdict = "risky"
        report.warnings.append("experiment appears stale (no process, old artifacts)")
        report.confidence = 0.65

    if progress.subprocess_running or progress.predictions_lines >= 100:
        report.runnable = True
        report.evidence.append("active subprocess or substantial prediction artifacts")

    if partial_summary and partial_summary.get("per_method"):
        report.runnable = True
        report.evidence.append(
            f"partial metrics from {partial_summary.get('prediction_lines', 0)} predictions"
        )
        candidate = partial_summary.get("candidate_method")
        per_method = partial_summary.get("per_method") or {}
        if candidate and candidate in per_method:
            score = per_method[candidate].get(
                partial_summary.get("primary_metric") or "primary_score"
            )
            if score is not None:
                report.evidence.append(f"candidate {candidate} partial mean={float(score):.4f}")

    if progress.coverage_pct is not None and progress.coverage_pct >= 80:
        report.publishable_path = True
        report.evidence.append(f"prediction coverage {progress.coverage_pct:.0f}%")
    elif progress.predictions_lines >= 500:
        report.warnings.append("large partial run but coverage unknown; finalize may need summary file")

    if progress.time_budget_seconds and progress.elapsed_seconds:
        if progress.elapsed_seconds > progress.time_budget_seconds * 0.95:
            report.warnings.append("run near or past configured time budget")

    resource_class = str(insight.get("resource_class") or run.get("resource_class") or "cpu")
    if resource_class in {"gpu_large", "gpu_small"} and not progress.subprocess_running:
        if progress.predictions_lines == 0:
            report.warnings.append(f"resource_class={resource_class} but no GPU workload observed yet")

    if report.blockers:
        report.verdict = "not_viable"
        report.runnable = False
    elif report.runnable and (report.publishable_path or partial_summary):
        report.verdict = "viable"
        report.confidence = 0.75
        report.recommended_route = "full_paper"
    elif report.runnable:
        report.verdict = "risky"
        report.confidence = 0.6
        report.recommended_route = "smoke_test"
    else:
        report.verdict = "risky"
        report.warnings.append("insufficient runtime evidence yet")
        report.confidence = 0.5
        report.recommended_route = "smoke_test"

    if EXPERIMENT_WATCHDOG_USE_LLM:
        _maybe_enrich_with_llm(report, insight=insight, run=run, progress=progress)

    return report


def _maybe_enrich_with_llm(
    report: FeasibilityReport,
    *,
    insight: dict[str, Any],
    run: dict[str, Any],
    progress: ProgressReport,
) -> None:
    try:
        from agents.llm_client import call_llm_json
    except ImportError:
        return

    prompt = {
        "task": "Assess experiment feasibility for an automated research pipeline.",
        "insight_title": insight.get("title"),
        "run_status": run.get("status"),
        "phase": run.get("phase"),
        "progress": progress.to_dict(),
        "current_verdict": report.verdict,
        "blockers": report.blockers,
        "warnings": report.warnings,
        "output_schema": {
            "verdict": "viable|risky|not_viable",
            "confidence": "0-1 float",
            "one_line_rationale": "string",
        },
    }
    try:
        payload, _tokens = call_llm_json(
            "You are the Experiment Feasibility Agent. Be conservative and evidence-first.",
            json.dumps(prompt, ensure_ascii=False),
            temperature=0.1,
        )
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    verdict = str(payload.get("verdict") or "").strip().lower()
    if verdict in {"viable", "risky", "not_viable"}:
        report.verdict = verdict
    try:
        report.confidence = float(payload.get("confidence", report.confidence))
    except (TypeError, ValueError):
        pass
    rationale = str(payload.get("one_line_rationale") or "").strip()
    if rationale:
        report.evidence.append(f"llm: {rationale}")
