"""Structured experiment judgement before forge/execution."""

from __future__ import annotations

from typing import Any, Mapping

from contracts import DeepInsightSpec, ExperimentJudgement


def _non_empty_text(value: Any) -> str:
    return str(value or "").strip()


def _baseline_names(plan: dict[str, Any]) -> list[str]:
    rows = plan.get("baselines") or []
    names: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            name = _non_empty_text(row.get("name") or row.get("model"))
        else:
            name = _non_empty_text(row)
        if name:
            names.append(name)
    return names


def _dataset_names(plan: dict[str, Any]) -> list[str]:
    rows = plan.get("datasets") or []
    names: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            name = _non_empty_text(row.get("name"))
        else:
            name = _non_empty_text(row)
        if name:
            names.append(name)
    return names


def _primary_metric(plan: dict[str, Any]) -> str:
    metrics = plan.get("metrics") or {}
    if isinstance(metrics, dict):
        return _non_empty_text(metrics.get("primary"))
    return _non_empty_text(metrics)


def review_experiment_candidate(
    insight: Mapping[str, Any] | DeepInsightSpec,
    *,
    codebase: Mapping[str, Any] | None = None,
    entrypoint_available: bool | None = None,
) -> ExperimentJudgement:
    """Judge whether an insight is ready for formal experiments or smoke-only."""

    spec = insight if isinstance(insight, DeepInsightSpec) else DeepInsightSpec.from_raw(insight)
    plan = spec.experimental_plan
    method = spec.proposed_method
    codebase = dict(codebase or {})

    blockers: list[str] = []
    warnings: list[str] = []

    baselines = _baseline_names(plan)
    datasets = _dataset_names(plan)
    primary_metric = _primary_metric(plan)

    baseline_review = {
        "baseline_count": len(baselines),
        "baselines": baselines,
        "strong_enough": len(baselines) >= 2,
    }
    if len(baselines) < 2:
        blockers.append("Experimental plan lacks at least two explicit baselines.")

    scale_target = (
        plan.get("compute_budget", {}).get("total_gpu_hours")
        if isinstance(plan.get("compute_budget"), dict)
        else None
    )
    scale_review = {
        "resource_class": spec.resource_class or "unknown",
        "requested_scale": scale_target or "unspecified",
        "sufficient_signal": bool(scale_target) or (spec.resource_class == "cpu"),
    }
    if spec.resource_class and spec.resource_class != "cpu" and not scale_review["sufficient_signal"]:
        blockers.append("Compute budget is underspecified for a non-CPU formal experiment.")

    alignment_review = {
        "method_name": _non_empty_text(method.get("name")),
        "method_definition_present": bool(_non_empty_text(method.get("definition"))),
        "dataset_count": len(datasets),
        "primary_metric": primary_metric,
        "aligned": bool(datasets and primary_metric and _non_empty_text(method.get("definition"))),
    }
    if not datasets:
        blockers.append("Experimental plan is missing explicit datasets.")
    if not primary_metric:
        blockers.append("Experimental plan is missing a primary metric.")
    if not _non_empty_text(method.get("definition")):
        blockers.append("Proposed method lacks a formal definition for experiment design.")

    repo_url = _non_empty_text(codebase.get("url"))
    baseline_command = _non_empty_text(codebase.get("main_eval_command"))
    main_train_file = _non_empty_text(codebase.get("main_train_file"))
    codebase_review = {
        "url": repo_url or "scratch",
        "name": _non_empty_text(codebase.get("name")),
        "entrypoint_available": bool(entrypoint_available) if entrypoint_available is not None else None,
        "main_train_file": main_train_file,
        "baseline_command": baseline_command,
    }
    if repo_url == "scratch" or not repo_url:
        warnings.append("Repository scout fell back to scratch; formal experiment path is not allowed.")
    if repo_url != "scratch" and entrypoint_available is False:
        warnings.append("Selected repository is missing the expected train entrypoint.")
    if not baseline_command:
        warnings.append("Codebase scout did not provide a baseline command; validation will rely on heuristic entrypoint search.")

    environment_review = {
        "formal_repo_available": bool(repo_url and repo_url != "scratch"),
        "entrypoint_available": entrypoint_available if entrypoint_available is not None else bool(main_train_file),
        "cpu_compatible": spec.resource_class in {"", "cpu"},
    }

    smoke_only = False
    formal_experiment = False
    route = "blocked"

    codebase_is_formal = environment_review["formal_repo_available"] and (
        entrypoint_available is not False
    )
    if blockers:
        route = "blocked"
    elif codebase_is_formal:
        route = "formal"
        formal_experiment = True
    else:
        route = "smoke_test"
        smoke_only = True

    summary_bits = []
    if blockers:
        summary_bits.append(f"blocked: {len(blockers)} blocking review issues")
    elif formal_experiment:
        summary_bits.append("formal-ready: baseline, dataset, and method contracts are present")
    else:
        summary_bits.append("smoke-only: design is coherent but repository support is insufficient for a formal run")
    if warnings:
        summary_bits.append("; ".join(warnings[:2]))

    return ExperimentJudgement(
        deep_insight_id=spec.insight_id,
        recommended_route=route,
        formal_experiment=formal_experiment,
        smoke_test_only=smoke_only,
        summary=". ".join(summary_bits),
        blockers=blockers,
        warnings=warnings,
        baseline_review=baseline_review,
        scale_review=scale_review,
        alignment_review=alignment_review,
        environment_review=environment_review,
        codebase_review=codebase_review,
    )
