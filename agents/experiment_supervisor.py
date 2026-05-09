"""Supervisor planning layer for experiment iterations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from contracts import ExperimentSpec
from agents.stage_prompts import prompt_block


def _as_float(value: Any) -> float | None:
    if value in (None, "", []):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_comparison(
    value: float | None,
    reference: float | None,
    direction: str,
) -> dict[str, Any] | None:
    current = _as_float(value)
    anchor = _as_float(reference)
    if current is None or anchor is None:
        return None
    if str(direction or "higher").lower() == "lower":
        effect = anchor - current
    else:
        effect = current - anchor
    return {
        "value": current,
        "reference": anchor,
        "direction": direction or "higher",
        "effect": effect,
        "effect_pct": (effect / abs(anchor) * 100.0) if abs(anchor) > 1e-12 else None,
        "beats_reference": effect > 1e-12,
        "ties_reference": abs(effect) <= 1e-12,
    }


def _format_gap(comparison: dict[str, Any]) -> str:
    effect = abs(float(comparison.get("effect") or 0.0))
    pct = comparison.get("effect_pct")
    if pct is None:
        return f"{effect:.6g}"
    return f"{effect:.6g} ({abs(float(pct)):.2f}%)"


def _history_summary(history: list[dict], limit: int = 12) -> list[dict]:
    out: list[dict] = []
    for row in history[-limit:]:
        out.append(
            {
                "iteration": row.get("iteration"),
                "status": row.get("status"),
                "metric": row.get("metric"),
                "description": str(row.get("description") or "")[:200],
            }
        )
    return out


def _recent_cost_microtuning_streak(history: list[dict], limit: int = 12) -> int:
    streak = 0
    for row in reversed(history[-limit:]):
        if row.get("status") != "keep":
            continue
        text = json.dumps(row, ensure_ascii=False).lower()
        budget_signal = any(marker in text for marker in ("token", "budget", "cap", "cost"))
        narrow_signal = any(marker in text for marker in ("yes/no", "zero-budget", "one-word", "1-token", "2-token", "3-token"))
        if not (budget_signal and narrow_signal):
            break
        streak += 1
    return streak


def _recent_failed_prompt_shortening(history: list[dict], limit: int = 12) -> bool:
    prompt_markers = (
        "shortest exact answer",
        "shortest final answer",
        "shortest answer",
        "answer span",
        "answer phrase only",
        "short phrase",
        "do not explain",
        "extra text",
        "non-boolean zero-budget",
        "open zero-budget",
    )
    for row in reversed(history[-limit:]):
        if row.get("status") != "discard":
            continue
        text = json.dumps(row, ensure_ascii=False).lower()
        if any(marker in text for marker in prompt_markers):
            return True
    return False


def _recent_failed_context_propagation(history: list[dict], limit: int = 12) -> bool:
    markers = (
        "context-preserving prompting",
        "benchmark context",
        "context/passages/facts",
        "benchmark-provided context",
        "context fields",
    )
    for row in reversed(history[-limit:]):
        if row.get("status") != "discard":
            continue
        text = json.dumps(row, ensure_ascii=False).lower()
        if any(marker in text for marker in markers):
            return True
    return False


def _budget_summary(spec: ExperimentSpec, success_criteria: dict[str, Any]) -> dict[str, Any]:
    proxy = spec.proxy_config if isinstance(spec.proxy_config, dict) else {}
    plan = spec.experimental_plan if isinstance(spec.experimental_plan, dict) else {}
    publication_contract = success_criteria.get("publication_evidence_contract") or {}
    if not isinstance(publication_contract, dict):
        publication_contract = {}
    benchmark_manifest = (
        proxy.get("benchmark_manifest")
        or success_criteria.get("benchmark_manifest")
        or publication_contract.get("benchmark_manifest")
        or {}
    )
    if not isinstance(benchmark_manifest, dict):
        benchmark_manifest = {}

    return {
        "per_iteration_time_budget_seconds": proxy.get("time_budget_seconds"),
        "max_hypothesis_iterations": proxy.get("max_iterations"),
        "reproduction_iterations": proxy.get("reproduction_iterations"),
        "refute_min_iterations": proxy.get("refute_min_iterations"),
        "estimated_gpu_hours": proxy.get("estimated_gpu_hours"),
        "real_benchmark_required": proxy.get("real_benchmark_required"),
        "allowed_scope": "method/dependency/compatibility edits only; benchmark contract changes require explicit contract revision",
        "benchmark": {
            "model": proxy.get("benchmark_model") or benchmark_manifest.get("model") or plan.get("model_targets"),
            "dataset": proxy.get("benchmark_dataset") or benchmark_manifest.get("dataset") or plan.get("datasets"),
            "seeds": proxy.get("benchmark_seeds") or benchmark_manifest.get("seeds"),
            "max_examples_per_seed": proxy.get("benchmark_max_examples_per_seed") or benchmark_manifest.get("max_examples_per_seed"),
            "methods": benchmark_manifest.get("methods") or benchmark_manifest.get("baselines"),
            "metric": success_criteria.get("metric_name"),
        },
    }


def build_supervisor_plan(
    *,
    spec: ExperimentSpec,
    environment_report: dict[str, Any],
    baseline: float | None,
    best_so_far: float | None,
    history: list[dict],
    iteration: int,
    success_criteria: dict[str, Any],
) -> dict[str, Any]:
    """Build a deterministic supervisor plan for the next worker turn."""
    last = history[-1] if history else {}
    crash_streak = 0
    for row in reversed(history):
        if row.get("status") == "crash":
            crash_streak += 1
        else:
            break

    if crash_streak >= 1:
        mode = "repair"
        diagnosis = "Recent iterations crashed; prioritize restoring a runnable baseline-compatible path."
    elif history and last.get("status") == "discard":
        mode = "redirect"
        diagnosis = "Last hypothesis did not improve metrics; try a materially different intervention."
    elif history and last.get("status") == "keep":
        mode = "refine"
        diagnosis = "Previous change improved the metric; refine carefully without destabilizing the run."
    else:
        mode = "bootstrap"
        diagnosis = "No successful hypothesis iteration yet; start with the smallest plausible implementation of the method."

    main_train_file = spec.proxy_config.get("main_train_file") or environment_report.get("resolved_train_file") or "auto-detect"
    metric_name = success_criteria.get("metric_name", "metric")
    metric_direction = success_criteria.get("metric_direction", "higher")
    baseline_comparison = _metric_comparison(best_so_far, baseline, metric_direction)
    if history and last.get("status") == "keep" and baseline_comparison:
        if baseline_comparison["beats_reference"]:
            diagnosis = "Previous kept change beats the baseline; refine carefully without destabilizing the run."
        elif baseline_comparison["ties_reference"]:
            diagnosis = "Previous kept change only ties the baseline; prioritize a measurable improvement before any paper claim."
            mode = "recover"
        else:
            diagnosis = (
                "Previous kept change improved over earlier attempts but remains below baseline by "
                f"{_format_gap(baseline_comparison)}; prioritize closing this negative effect before any paper claim."
            )
            mode = "recover"
    cost_microtuning_streak = _recent_cost_microtuning_streak(history)
    failed_prompt_shortening = _recent_failed_prompt_shortening(history)
    failed_context_propagation = _recent_failed_context_propagation(history)
    if mode == "refine" and cost_microtuning_streak >= 2:
        mode = "redirect"
        diagnosis = (
            "Recent kept changes are narrow token-budget micro-tuning on the sanity slice. "
            "Do not keep shrinking answer caps; switch to a substantive routing or accuracy hypothesis, "
            "or prepare full benchmark evidence before any paper claim."
        )
    elif mode == "redirect" and cost_microtuning_streak >= 2:
        diagnosis = (
            "Last hypothesis did not improve metrics, and recent kept changes are narrow token-budget "
            "micro-tuning on the sanity slice. Do not keep shrinking answer caps; switch to a substantive "
            "routing or accuracy hypothesis, or prepare full benchmark evidence before any paper claim."
        )
    publication_contract = success_criteria.get("publication_evidence_contract") or {}
    if not isinstance(publication_contract, dict):
        publication_contract = {}

    if mode == "repair":
        next_actions = [
            "Locate the exact failing command, import, or file path that broke execution.",
            "Make the minimum repo change needed to restore a clean runnable experiment.",
            "Avoid broad method changes until the experiment executes again.",
        ]
    elif mode == "redirect":
        next_actions = [
            "Avoid repeating the last discarded intervention.",
            "Choose a different mechanism consistent with the proposed method definition.",
            "Keep the change narrow enough that its effect can be isolated in one run.",
        ]
        if cost_microtuning_streak >= 2:
            next_actions[0] = "Stop repeating token-cap or budget-only micro-tuning on the same sanity slice."
            next_actions.append("Prefer a substantive routing, calibration, or accuracy-improving mechanism over another answer-cap shrink.")
        if failed_prompt_shortening:
            next_actions.append("Do not change the zero-budget open-answer prompt into a shortest-answer or answer-span prompt; recent discarded attempts in that family lost utility.")
            next_actions.append("Do not edit `_build_cggr_zero_budget_prompt`, `_cggr_zero_budget_max_tokens`, or add question-type regexes that alter zero-budget answer shape.")
        if failed_context_propagation:
            next_actions.append("Do not repeat broad benchmark-context propagation or prompt-context rewrites for all methods; a recent fair-context attempt lost utility.")
    elif mode == "refine":
        next_actions = [
            "Build on the best-performing idea without changing unrelated components.",
            "Protect baseline fairness and evaluation compatibility.",
            "Prefer parameterization, scheduling, or integration refinements over wholesale rewrites.",
        ]
    elif mode == "recover":
        next_actions = [
            "Diagnose why the candidate still fails to beat the locked baseline.",
            "Change only the proposed method path, routing policy, or parameters needed to close that deficit.",
            "Preserve baseline fairness and report any negative effect plainly in the next iteration packet.",
        ]
    else:
        next_actions = [
            "Implement the smallest end-to-end version of the proposed method.",
            "Preserve the current training/evaluation pipeline shape where possible.",
            "Keep the code change local and easy to reason about.",
        ]

    guardrails = [
        "Honor the evidence plan and do not invent disabled analyses.",
        "Respect the budget block; do not reduce models, datasets, seeds, examples, or baselines just to make a run pass.",
        "Keep changes hypothesis-directed and minimal.",
        "Do not break the existing baseline execution path.",
        "Do not silently modify locked benchmark contract fields; request a contract revision instead.",
    ]
    if failed_prompt_shortening:
        guardrails.append(
            "Do not alter the candidate zero-budget open-answer prompt toward shortest-answer, answer-span, or phrase-only output; that discarded intervention family already lost utility."
        )
        guardrails.append(
            "Do not edit `_build_cggr_zero_budget_prompt`, `_cggr_zero_budget_max_tokens`, zero-budget reasoning-budget metadata, or answer-shape regex helpers in this iteration."
        )
    if failed_context_propagation:
        guardrails.append(
            "Do not make broad context-propagation or all-method prompt rewrites; recent benchmark-context prompting lowered the sanity metric."
        )

    plan = {
        "role": "ExperimentSupervisor",
        "iteration": iteration,
        "mode": mode,
        "stage_prompt": prompt_block("method_worker"),
        "objective": f"Improve {metric_name} ({metric_direction}) over baseline {baseline} and best-so-far {best_so_far}.",
        "diagnosis": diagnosis,
        "baseline_comparison": baseline_comparison,
        "main_train_file": main_train_file,
        "baseline_command": spec.proxy_config.get("baseline_command"),
        "environment": {
            "formal_ready": environment_report.get("formal_ready"),
            "entrypoint_exists": environment_report.get("entrypoint_exists"),
            "resource_class": environment_report.get("resource_class"),
            "resolved_train_file": environment_report.get("resolved_train_file"),
        },
        "repo_focus": [main_train_file],
        "budget": _budget_summary(spec, success_criteria),
        "guardrails": guardrails,
        "benchmark_manifest": (
            spec.proxy_config.get("benchmark_manifest")
            or spec.success_criteria.get("benchmark_manifest")
            or publication_contract.get("benchmark_manifest", {})
        ),
        "next_actions": next_actions,
        "history": _history_summary(history),
        "success_criteria": success_criteria,
    }
    return plan


def write_supervisor_artifacts(workdir: Path, plan: dict[str, Any]) -> dict[str, str]:
    """Persist supervisor artifacts for the current iteration."""
    iter_num = int(plan.get("iteration") or 0)
    sup_dir = workdir / "codex" / "supervisor"
    sup_dir.mkdir(parents=True, exist_ok=True)
    json_path = sup_dir / f"iter_{iter_num:03d}_plan.json"
    md_path = sup_dir / f"iter_{iter_num:03d}_plan.md"

    json_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    md_text = "\n".join(
        [
            "# Experiment Supervisor Plan",
            f"- Iteration: {plan.get('iteration')}",
            f"- Mode: {plan.get('mode')}",
            f"- Objective: {plan.get('objective')}",
            f"- Diagnosis: {plan.get('diagnosis')}",
            "",
            "## Next Actions",
            *[f"- {row}" for row in plan.get("next_actions", [])],
            "",
            "## Guardrails",
            *[f"- {row}" for row in plan.get("guardrails", [])],
        ]
    )
    md_path.write_text(md_text, encoding="utf-8")
    return {
        "supervisor_plan_json": str(json_path),
        "supervisor_plan_md": str(md_path),
    }
