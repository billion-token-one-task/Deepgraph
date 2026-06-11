"""Benchmark-specific experiment protocol resolution.

This module keeps paper-evidence requirements tied to the benchmark contract
instead of process-wide example/model/baseline constants. It is intentionally
deterministic: later agents may inspect datasets and refine the protocol, but
forge/review/audit always get a stable first contract to preserve.
"""

from __future__ import annotations

from typing import Any, Mapping


def _text(value: Any) -> str:
    return str(value or "").strip()


def _canonical(value: Any) -> str:
    return "".join(ch for ch in _text(value).lower() if ch.isalnum())


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _text(value)
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _named_values(rows: Any, keys: tuple[str, ...]) -> list[str]:
    if not isinstance(rows, list):
        rows = [rows] if rows not in (None, "", "unknown") else []
    out: list[str] = []
    for row in rows:
        if isinstance(row, Mapping):
            for key in keys:
                text = _text(row.get(key))
                if text:
                    out.append(text)
                    break
        else:
            text = _text(row)
            if text:
                out.append(text)
    return _unique(out)


KNOWN_BENCHMARK_PROTOCOLS: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "canonical_name": "GSM8K",
        "aliases": ("gsm8k", "openai/gsm8k", "grade school math"),
        "hf_dataset": "openai/gsm8k",
        "config": "main",
        "official_split": "test",
        "task_family": "math_word_problem_qa",
        "primary_metric": "exact_match",
        "allowed_secondary_metrics": ("numeric_accuracy", "cost_adjusted_accuracy", "latency", "tokens"),
        "answer_format": "final numeric answer",
        "minimum_repeats": 1,
    },
    "strategyqa": {
        "canonical_name": "StrategyQA",
        "aliases": ("strategyqa", "strategy qa"),
        "hf_dataset": "ChilleD/StrategyQA",
        "config": "",
        "official_split": "test",
        "task_family": "boolean_question_answering",
        "primary_metric": "accuracy",
        "allowed_secondary_metrics": ("exact_match", "cost_adjusted_accuracy", "latency", "tokens"),
        "answer_format": "yes/no",
        "minimum_repeats": 1,
    },
    "musique": {
        "canonical_name": "MuSiQue-Ans",
        "aliases": ("musique", "musique-ans", "musique ans"),
        "hf_dataset": "dgslibisey/MuSiQue",
        "config": "",
        "official_split": "validation",
        "task_family": "multi_hop_question_answering",
        "primary_metric": "exact_match",
        "allowed_secondary_metrics": ("f1", "cost_adjusted_accuracy", "latency", "tokens"),
        "answer_format": "short answer string",
        "minimum_repeats": 1,
    },
    "2wikimultihopqa": {
        "canonical_name": "2WikiMultihopQA",
        "aliases": ("2wikimultihopqa", "2wiki", "two wiki multihop qa"),
        "hf_dataset": "voidful/2WikiMultihopQA",
        "config": "",
        "official_split": "validation",
        "task_family": "multi_hop_question_answering",
        "primary_metric": "exact_match",
        "allowed_secondary_metrics": ("f1", "cost_adjusted_accuracy", "latency", "tokens"),
        "answer_format": "short answer string",
        "minimum_repeats": 1,
    },
    "mbpp": {
        "canonical_name": "MBPP",
        "aliases": ("mbpp", "google-research-datasets/mbpp"),
        "hf_dataset": "google-research-datasets/mbpp",
        "config": "",
        "official_split": "test",
        "task_family": "code_generation",
        "primary_metric": "pass_at_1",
        "allowed_secondary_metrics": ("pass_at_k", "runtime", "tokens"),
        "answer_format": "python program",
        "minimum_repeats": 1,
    },
    "cifar10": {
        "canonical_name": "CIFAR-10",
        "aliases": ("cifar10", "cifar-10"),
        "hf_dataset": "cifar10",
        "config": "",
        "official_split": "test",
        "task_family": "image_classification",
        "primary_metric": "accuracy",
        "allowed_secondary_metrics": ("calibration_error", "latency"),
        "answer_format": "class label",
        "minimum_repeats": 3,
    },
}


def _known_protocol_for(name: str) -> dict[str, Any] | None:
    canon = _canonical(name)
    for protocol in KNOWN_BENCHMARK_PROTOCOLS.values():
        aliases = [protocol["canonical_name"], protocol.get("hf_dataset"), *(protocol.get("aliases") or [])]
        if any(canon == _canonical(alias) or canon in _canonical(alias) or _canonical(alias) in canon for alias in aliases):
            return dict(protocol)
    return None


def _dataset_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("benchmark_targets", "datasets"):
        value = plan.get(key)
        if not isinstance(value, list):
            continue
        for row in value:
            if isinstance(row, Mapping):
                candidate = dict(row)
            else:
                candidate = {"name": row}
            name = _text(candidate.get("name") or candidate.get("hf_dataset") or candidate.get("dataset"))
            if name and not any(_canonical(name) == _canonical(existing.get("name") or existing.get("hf_dataset") or existing.get("dataset")) for existing in rows):
                rows.append(candidate)
    return rows


def _model_names(plan: Mapping[str, Any]) -> list[str]:
    rows: list[Any] = []
    for key in ("model_targets", "models"):
        value = plan.get(key)
        if isinstance(value, list):
            rows.extend(value)
    return _named_values(rows, ("hf_model", "model", "name"))


def _baseline_names(plan: Mapping[str, Any]) -> list[str]:
    return _named_values(plan.get("baselines"), ("name", "model", "method"))


def _ablation_names(plan: Mapping[str, Any]) -> list[str]:
    return _named_values(plan.get("ablations"), ("name", "component", "factor"))


def _metric_name(plan: Mapping[str, Any]) -> str:
    metrics = plan.get("metrics")
    if isinstance(metrics, Mapping):
        return _text(metrics.get("primary") or metrics.get("name"))
    if isinstance(metrics, list):
        names = _named_values(metrics, ("name",))
        return names[0] if names else ""
    return _text(metrics)


def _seed_values(plan: Mapping[str, Any], minimum_repeats: int) -> list[int]:
    raw = plan.get("seeds")
    if isinstance(raw, list):
        out: list[int] = []
        for value in raw:
            try:
                out.append(int(value))
            except (TypeError, ValueError):
                continue
        if out:
            return out
    try:
        count = int(plan.get("minimum_seeds") or minimum_repeats or 1)
    except (TypeError, ValueError):
        count = minimum_repeats or 1
    return list(range(max(1, count)))


def resolve_benchmark_protocol(
    plan: Mapping[str, Any] | None,
    *,
    method: Mapping[str, Any] | None = None,
    claim: str | None = None,
) -> dict[str, Any]:
    """Resolve benchmark-specific protocol requirements for a plan."""

    plan = plan or {}
    method = method or {}
    dataset_protocols: list[dict[str, Any]] = []
    warnings: list[str] = []
    blockers: list[str] = []
    primary_metric = _metric_name(plan)

    for row in _dataset_rows(plan):
        raw_name = _text(row.get("name") or row.get("hf_dataset") or row.get("dataset"))
        known = _known_protocol_for(raw_name) or {}
        protocol_source = "known_public_benchmark_protocol" if known else "dataset_card_inspection_required"
        metric = primary_metric or _text(known.get("primary_metric")) or "metric_value"
        official_split = _text(row.get("split") or known.get("official_split"))
        if not official_split:
            official_split = "inspect_dataset_card"
            warnings.append(f"{raw_name or 'dataset'} has no official split in the local protocol registry; inspect the dataset card before paper claims.")
        hf_dataset = _text(row.get("hf_dataset") or row.get("dataset_id") or known.get("hf_dataset"))
        if not hf_dataset and "/" in raw_name:
            hf_dataset = raw_name
        if not hf_dataset:
            warnings.append(f"{raw_name or 'dataset'} has no concrete dataset id; execution must materialize and record the source.")

        expected_examples = None
        for key in ("num_examples", "num_test", "expected_examples", "size"):
            try:
                value = int(row.get(key) or 0)
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                expected_examples = value
                break

        dataset_protocols.append(
            {
                "name": raw_name or _text(known.get("canonical_name")) or "dataset",
                "canonical_name": _text(known.get("canonical_name")) or raw_name,
                "protocol_source": protocol_source,
                "hf_dataset": hf_dataset,
                "config": _text(row.get("config") if row.get("config") is not None else known.get("config")),
                "official_split": official_split,
                "evaluation_split": official_split,
                "task_family": _text(row.get("task_type") or known.get("task_family") or "benchmark"),
                "primary_metric": metric,
                "metric_policy": {
                    "official_primary": _text(known.get("primary_metric")) or metric,
                    "selected_primary": metric,
                    "allowed_secondary": list(known.get("allowed_secondary_metrics") or []),
                },
                "sample_policy": {
                    "paper_evaluation": "use_official_or_materialized_full_split",
                    "sanity_evaluation": "bounded_real_examples_only",
                    "expected_examples": expected_examples,
                    "requires_materialized_count": True,
                    "global_numeric_thresholds_allowed": False,
                },
                "answer_policy": {
                    "format": _text(row.get("answer_format") or known.get("answer_format") or "benchmark_defined"),
                    "prediction_logging_required": True,
                },
            }
        )

    if not dataset_protocols:
        blockers.append("No benchmark dataset is resolved; experiment cannot produce paper evidence.")

    baselines = _baseline_names(plan)
    models = _model_names(plan)
    ablations = _ablation_names(plan)
    if not baselines:
        blockers.append("No explicit baseline methods are resolved.")
    if not models:
        blockers.append("No concrete model target is resolved.")

    known_min_repeats = max([int(row.get("minimum_repeats") or 1) for row in dataset_protocols] or [1])
    try:
        planned_repeats = int(plan.get("minimum_seeds") or 0)
    except (TypeError, ValueError):
        planned_repeats = 0
    minimum_repeats = max(known_min_repeats, planned_repeats or known_min_repeats or 1)
    if any(token in " ".join([_text(method.get("name")), _text(method.get("type")), _text(claim)]).lower() for token in ("training", "finetun", "fine-tun", "learned", "stochastic")):
        minimum_repeats = max(minimum_repeats, 3)

    required_artifacts = [
        "run_config.json",
        "benchmark_summary.json",
        "raw_predictions.jsonl",
        "per_seed_results.json",
        "per_dataset_results.json",
        "main_results_table.json",
        "latency_tokens_table.json",
        "artifact_manifest.json",
    ]
    if ablations:
        required_artifacts.append("ablation_table.json")
    claim_text = " ".join([_text(method.get("name")), _text(method.get("type")), _text(method.get("definition")), _text(claim)]).lower()
    routing_like = any(token in claim_text for token in ("route", "routing", "gate", "gating", "selective", "budget"))
    if routing_like:
        required_artifacts.extend(
            [
                "routing_decisions.jsonl",
                "route_rate_sweep_table.json",
                "cost_utility_tradeoff_table.json",
                "quality_cost_frontier.json",
                "difficulty_breakdown_table.json",
                "routing_analysis.json",
                "simple_case_degradation.json",
                "calibration_reliability.json",
            ]
        )

    required_artifacts.extend(["bootstrap_ci.json", "failure_cases.jsonl"])
    required_dataset_names = [row["name"] for row in dataset_protocols]
    metric = primary_metric or next((row["primary_metric"] for row in dataset_protocols if row.get("primary_metric")), "metric_value")

    return {
        "schema_version": "benchmark_protocol_v1",
        "resolver": "deterministic_local_protocol_resolver",
        "status": "blocked" if blockers else "resolved_with_warnings" if warnings else "resolved",
        "summary": (
            "Benchmark requirements are benchmark-specific: use official/materialized splits "
            "and do not apply global example/model/baseline thresholds."
        ),
        "claim_to_validate": _text(claim),
        "dataset_protocols": dataset_protocols,
        "metric_policy": {
            "primary_metric": metric,
            "per_dataset_metrics": {row["name"]: row["primary_metric"] for row in dataset_protocols},
        },
        "seed_policy": {
            "minimum_repeats": minimum_repeats,
            "seed_values": _seed_values(plan, minimum_repeats),
            "rationale": "Use the benchmark official protocol first; repeat stochastic training or sampling only when the method/model path is stochastic.",
        },
        "baseline_policy": {
            "required_baselines": baselines,
            "minimum_deployable_baselines": len(baselines),
            "fairness_requirements": [
                "same evaluator and answer normalization for candidate and baselines",
                "same dataset split and prompt template disclosure for all methods",
                "compute/budget matching when the claim is cost or routing related",
            ],
        },
        "model_policy": {
            "required_models": models,
            "minimum_models": len(models),
        },
        "ablation_policy": {
            "required_ablations": ablations,
            "required_when_claiming_mechanism": bool(ablations),
        },
        "full_benchmark_requirements": {
            "required_dataset_names": required_dataset_names,
            "required_model_names": models,
            "required_baseline_names": baselines,
            "required_ablations": ablations,
            "required_artifacts": _unique(required_artifacts),
            "examples_policy": "benchmark_specific_official_or_materialized_full_split",
            "global_numeric_thresholds_allowed": False,
        },
        "blockers": blockers,
        "warnings": warnings,
    }


def protocol_minimum_seeds(protocol: Mapping[str, Any] | None) -> int:
    if not isinstance(protocol, Mapping):
        return 1
    seed_policy = protocol.get("seed_policy") if isinstance(protocol.get("seed_policy"), Mapping) else {}
    try:
        return max(1, int(seed_policy.get("minimum_repeats") or 1))
    except (TypeError, ValueError):
        return 1


def protocol_required_names(protocol: Mapping[str, Any] | None, key: str) -> list[str]:
    if not isinstance(protocol, Mapping):
        return []
    requirements = protocol.get("full_benchmark_requirements")
    if isinstance(requirements, Mapping):
        value = requirements.get(key)
        if isinstance(value, list):
            return _named_values(value, ("name", "id", "dataset", "model", "method"))
    return []

