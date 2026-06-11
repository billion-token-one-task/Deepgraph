"""Structured experiment judgement before forge/execution."""

from __future__ import annotations

from typing import Any, Mapping

from agents.benchmark_protocol import resolve_benchmark_protocol
from contracts import DeepInsightSpec, ExperimentJudgement
from config import (
    EXPERIMENT_ALLOW_SYNTHETIC_FALLBACK,
    EXPERIMENT_REQUIRE_REAL_BENCHMARK,
)


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


def _looks_synthetic(name: str) -> bool:
    lowered = _non_empty_text(name).lower()
    return not lowered or any(token in lowered for token in ("synthetic", "simulated", "toy", "smoke", "probe", "dummy"))


def _model_names(plan: dict[str, Any]) -> list[str]:
    rows = []
    for key in ("model_targets", "models"):
        value = plan.get(key)
        if isinstance(value, list):
            rows.extend(value)
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if isinstance(row, dict):
            name = _non_empty_text(row.get("name") or row.get("hf_model") or row.get("model"))
        else:
            name = _non_empty_text(row)
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
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
    real_datasets = [name for name in datasets if not _looks_synthetic(name)]
    model_targets = _model_names(plan)
    primary_metric = _primary_metric(plan)
    publication_contract = plan.get("publication_evidence_contract") if isinstance(plan.get("publication_evidence_contract"), dict) else {}
    benchmark_protocol = (
        publication_contract.get("benchmark_protocol")
        if isinstance(publication_contract.get("benchmark_protocol"), dict)
        else plan.get("benchmark_protocol") if isinstance(plan.get("benchmark_protocol"), dict) else None
    )
    if not isinstance(benchmark_protocol, dict):
        benchmark_protocol = resolve_benchmark_protocol(
            plan,
            method=method,
            claim=spec.title or spec.problem_statement,
        )
    protocol_requirements = benchmark_protocol.get("full_benchmark_requirements") if isinstance(benchmark_protocol.get("full_benchmark_requirements"), dict) else {}
    protocol_baselines = protocol_requirements.get("required_baseline_names") if isinstance(protocol_requirements.get("required_baseline_names"), list) else []
    protocol_datasets = protocol_requirements.get("required_dataset_names") if isinstance(protocol_requirements.get("required_dataset_names"), list) else []
    protocol_models = protocol_requirements.get("required_model_names") if isinstance(protocol_requirements.get("required_model_names"), list) else []
    protocol_seed_policy = benchmark_protocol.get("seed_policy") if isinstance(benchmark_protocol.get("seed_policy"), dict) else {}
    try:
        protocol_minimum_seeds = max(1, int(protocol_seed_policy.get("minimum_repeats") or 1))
    except (TypeError, ValueError):
        protocol_minimum_seeds = 1

    minimum_baselines = max(2, len(protocol_baselines) if protocol_baselines else 2)
    baseline_review = {
        "baseline_count": len(baselines),
        "baselines": baselines,
        "strong_enough": len(baselines) >= minimum_baselines,
        "minimum_required_for_paper": minimum_baselines,
        "source": "benchmark_protocol",
    }
    if len(baselines) < 2:
        blockers.append("Experimental plan lacks at least two explicit baselines.")
    elif len(baselines) < minimum_baselines:
        blockers.append(
            f"Experimental plan has only {len(baselines)} baseline(s); "
            f"the benchmark protocol requires {minimum_baselines}."
        )

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
        "real_dataset_count": len(real_datasets),
        "minimum_real_datasets_for_paper": max(1, len(protocol_datasets) if protocol_datasets else 1),
        "model_targets": model_targets,
        "minimum_models_for_paper": max(1, len(protocol_models) if protocol_models else 1),
        "primary_metric": primary_metric,
        "benchmark_protocol": {
            "status": benchmark_protocol.get("status"),
            "datasets": protocol_datasets,
            "models": protocol_models,
            "minimum_seeds": protocol_minimum_seeds,
            "examples_policy": protocol_requirements.get("examples_policy"),
            "global_numeric_thresholds_allowed": protocol_requirements.get("global_numeric_thresholds_allowed"),
            "warnings": benchmark_protocol.get("warnings") or [],
        },
        "aligned": bool(datasets and primary_metric and _non_empty_text(method.get("definition"))),
    }
    if not datasets:
        blockers.append("Experimental plan is missing explicit datasets.")
    if EXPERIMENT_REQUIRE_REAL_BENCHMARK and not real_datasets:
        blockers.append("Experimental plan must name at least one real public benchmark dataset; synthetic/proxy datasets are not allowed.")
    if EXPERIMENT_REQUIRE_REAL_BENCHMARK and not model_targets:
        blockers.append("Experimental plan must name at least one real model target.")
    seed_raw = plan.get("minimum_seeds") or plan.get("seeds") or protocol_minimum_seeds
    try:
        planned_seed_count = len(seed_raw) if isinstance(seed_raw, list) else int(seed_raw)
    except (TypeError, ValueError):
        planned_seed_count = 0
    if EXPERIMENT_REQUIRE_REAL_BENCHMARK and planned_seed_count < protocol_minimum_seeds:
        blockers.append(
            f"Experimental plan has only {planned_seed_count} seed(s); "
            f"the benchmark protocol requires at least {protocol_minimum_seeds}."
        )
    protocol_blockers = benchmark_protocol.get("blockers") if isinstance(benchmark_protocol.get("blockers"), list) else []
    blockers.extend(str(item) for item in protocol_blockers if str(item).strip())
    protocol_warnings = benchmark_protocol.get("warnings") if isinstance(benchmark_protocol.get("warnings"), list) else []
    warnings.extend(str(item) for item in protocol_warnings if str(item).strip())
    if EXPERIMENT_REQUIRE_REAL_BENCHMARK and plan.get("proxy_allowed") and not EXPERIMENT_ALLOW_SYNTHETIC_FALLBACK:
        blockers.append("Synthetic/proxy fallback is disabled for formal experiments.")
    recipe_blockers = plan.get("benchmark_recipe_blockers")
    benchmark_harness_required = bool(
        EXPERIMENT_REQUIRE_REAL_BENCHMARK and plan.get("generated_runner_supported") is False
    )
    unsupported_benchmark_targets: list[str] = []
    if benchmark_harness_required:
        if isinstance(recipe_blockers, list) and recipe_blockers:
            names = [
                _non_empty_text(item.get("name") if isinstance(item, dict) else item)
                for item in recipe_blockers
            ]
            names = [name for name in names if name]
            unsupported_benchmark_targets = names
            detail = ", ".join(names[:3]) if names else "the requested benchmark targets"
            blockers.append(
                "Generated real-benchmark runner does not support "
                f"{detail}; a dedicated benchmark harness/recipe is required before GPU execution."
            )
        else:
            blockers.append(
                "Generated real-benchmark runner is not supported for this benchmark contract; "
                "a dedicated benchmark harness/recipe is required before GPU execution."
            )
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
    generated_real_runner = bool(
        EXPERIMENT_REQUIRE_REAL_BENCHMARK
        and real_datasets
        and model_targets
        and plan.get("generated_runner_supported") is not False
    )
    if (repo_url == "scratch" or not repo_url) and not generated_real_runner:
        warnings.append("Repository scout fell back to scratch; formal experiment path is not allowed.")
    if repo_url != "scratch" and entrypoint_available is False:
        warnings.append("Selected repository is missing the expected train entrypoint.")
    if not baseline_command:
        warnings.append("Codebase scout did not provide a baseline command; validation will rely on heuristic entrypoint search.")

    environment_review = {
        "formal_repo_available": bool(repo_url and repo_url != "scratch"),
        "generated_real_benchmark_runner_allowed": generated_real_runner,
        "entrypoint_available": entrypoint_available if entrypoint_available is not None else bool(main_train_file),
        "cpu_compatible": spec.resource_class in {"", "cpu"},
        "benchmark_harness_required": benchmark_harness_required,
        "unsupported_benchmark_targets": unsupported_benchmark_targets,
        "harness_queue": "benchmark_harness_jobs" if benchmark_harness_required else "",
        "required_harness_agents": [
            "Benchmark Manager",
            "Dataset Fetch Agent",
            "Baseline Fetch Agent",
            "Benchmark Harness Code Agent",
            "Harness Review Agent",
        ] if benchmark_harness_required else [],
    }

    smoke_only = False
    formal_experiment = False
    route = "blocked"

    codebase_is_formal = (
        environment_review["formal_repo_available"] or environment_review["generated_real_benchmark_runner_allowed"]
    ) and (
        entrypoint_available is not False or environment_review["generated_real_benchmark_runner_allowed"]
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
