#!/usr/bin/env python3
"""Merge full-scale CGGR method shards into one auditable benchmark artifact."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


REQUIRED_METHODS = {
    "Vanilla Direct Answering",
    "Always-Reason Chain-of-Thought",
    "Self-Consistency Reasoning",
    "Least-to-Most Prompting",
    "Confidence Gate",
    "Disagreement Routing",
    "Random Budget-Matched Routing",
    "CGGR",
    "CGGR/no_counterfactual_delta",
    "CGGR/no_lcb",
    "CGGR/no_self_divergence_penalty",
    "CGGR/no_qstruct_term",
}

REQUIRED_ABLATIONS = {
    "no_counterfactual_delta",
    "no_lcb",
    "no_self_divergence_penalty",
    "no_qstruct_term",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return float(math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1)))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


def _score(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or 0.0)
    except (TypeError, ValueError):
        return default


def _route_lookup(routing_rows: list[dict]) -> dict[tuple[str, Any, str, str], bool]:
    routes: dict[tuple[str, Any, str, str], bool] = {}
    for row in routing_rows:
        method = str(row.get("method") or "")
        dataset = str(row.get("dataset") or "")
        example_id = str(row.get("example_id") or "")
        if method and dataset and example_id:
            routes[(method, row.get("seed"), dataset, example_id)] = bool(row.get("routed_to_deliberation"))
    return routes


def _difficulty_lookup(routing_rows: list[dict]) -> dict[tuple[Any, str, str], float]:
    difficulties: dict[tuple[Any, str, str], float] = {}
    for row in routing_rows:
        dataset = str(row.get("dataset") or "")
        example_id = str(row.get("example_id") or "")
        if not dataset or not example_id or row.get("difficulty") is None:
            continue
        try:
            difficulties[(row.get("seed"), dataset, example_id)] = float(row.get("difficulty"))
        except (TypeError, ValueError):
            continue
    return difficulties


def _stress_thresholds(difficulties: dict[tuple[Any, str, str], float]) -> dict[tuple[Any, str], float]:
    grouped: dict[tuple[Any, str], list[float]] = defaultdict(list)
    for (seed, dataset, _example_id), value in difficulties.items():
        if "stress test split" in dataset.lower():
            grouped[(seed, dataset)].append(value)
    return {key: _percentile(values, 0.5) for key, values in grouped.items() if values}


def _difficulty_bucket(row: dict, difficulties: dict[tuple[Any, str, str], float], stress: dict[tuple[Any, str], float]) -> str:
    seed = row.get("seed")
    dataset = str(row.get("dataset") or "")
    example_id = str(row.get("example_id") or "")
    difficulty = difficulties.get((seed, dataset, example_id))
    if difficulty is None:
        return "unknown"
    if "stress test split" in dataset.lower():
        return "simple" if difficulty <= stress.get((seed, dataset), difficulty) else "hard"
    if difficulty < 0.33:
        return "easy"
    if difficulty < 0.66:
        return "medium"
    return "hard"


def _route_rate_for_rows(rows: list[dict], method: str, routes: dict[tuple[str, Any, str, str], bool]) -> float:
    decisions = []
    for row in rows:
        dataset = str(row.get("dataset") or "")
        example_id = str(row.get("example_id") or "")
        key = (method, row.get("seed"), dataset, example_id)
        if key in routes:
            decisions.append(routes[key])
    if not decisions:
        return 0.0
    return sum(1.0 for value in decisions if value) / len(decisions)


def _aggregate_rows(
    raw_rows: list[dict],
    routing_rows: list[dict],
    *,
    lambda_cost: float,
) -> tuple[dict, dict, list, list, list, list, dict, dict, list]:
    per_method_rows: dict[str, list[dict]] = defaultdict(list)
    per_dataset_method_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    per_seed_rows: dict[tuple[Any, str], list[dict]] = defaultdict(list)
    difficulty_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    routes = _route_lookup(routing_rows)
    difficulties = _difficulty_lookup(routing_rows)
    stress = _stress_thresholds(difficulties)

    for row in raw_rows:
        method = str(row.get("method") or "")
        dataset = str(row.get("dataset") or "")
        seed = row.get("seed")
        if not method:
            continue
        per_method_rows[method].append(row)
        per_dataset_method_rows[(dataset, method)].append(row)
        per_seed_rows[(seed, method)].append(row)
        difficulty_rows[(method, _difficulty_bucket(row, difficulties, stress))].append(row)

    def aggregate(rows: list[dict], *, method: str, dataset: str | None = None, seed: Any | None = None) -> dict:
        count = len(rows)
        avg_score = _mean([_score(row, "primary_score") for row in rows])
        avg_tokens = _mean([_score(row, "new_tokens") for row in rows])
        metric_value = avg_score - lambda_cost * (avg_tokens / 192.0)
        route_rate = _route_rate_for_rows(rows, method, routes)
        return {
            "score": avg_score,
            "exact": _mean([_score(row, "exact") for row in rows]),
            "f1": _mean([_score(row, "f1") for row in rows]),
            "avg_new_tokens": avg_tokens,
            "avg_latency_seconds": _mean([_score(row, "latency_seconds") for row in rows]),
            "route_rate": float(route_rate),
            "cost_adjusted_accuracy": metric_value,
            "metric_value": metric_value,
            "count": count,
        }

    per_method = {method: aggregate(rows, method=method) for method, rows in sorted(per_method_rows.items())}
    per_dataset_results: dict[str, dict[str, Any]] = defaultdict(dict)
    for (dataset, method), rows in sorted(per_dataset_method_rows.items()):
        per_dataset_results[dataset][method] = aggregate(rows, method=method, dataset=dataset)

    per_seed_results: list[dict] = []
    per_seed_metric_values: dict[str, list[float]] = defaultdict(list)
    seed_values = sorted({seed for seed, _method in per_seed_rows.keys()})
    for seed in seed_values:
        methods: dict[str, dict] = {}
        datasets: dict[str, dict] = defaultdict(lambda: {"methods": {}})
        for (row_seed, method), rows in sorted(per_seed_rows.items()):
            if row_seed != seed:
                continue
            methods[method] = aggregate(rows, method=method, seed=seed)
            by_dataset: dict[str, list[dict]] = defaultdict(list)
            for row in rows:
                by_dataset[str(row.get("dataset") or "")].append(row)
            for dataset, dataset_rows in by_dataset.items():
                datasets[dataset]["num_examples"] = len(dataset_rows)
                datasets[dataset]["methods"][method] = aggregate(
                    dataset_rows,
                    method=method,
                    dataset=dataset,
                    seed=seed,
                )
        for method, row in methods.items():
            per_seed_metric_values[method].append(float(row.get("metric_value") or 0.0))
        per_seed_results.append({"seed": seed, "methods": methods, "datasets": dict(datasets)})
    per_method_std = {method: _std(values) for method, values in sorted(per_seed_metric_values.items())}

    difficulty_breakdown_table = []
    for (method, difficulty), rows in sorted(difficulty_rows.items()):
        if difficulty == "unknown":
            continue
        row = aggregate(rows, method=method)
        difficulty_breakdown_table.append(
            {
                "method": method,
                "difficulty": difficulty,
                "accuracy": row["score"],
                "avg_new_tokens": row["avg_new_tokens"],
                "avg_latency_seconds": row["avg_latency_seconds"],
                "route_rate": row["route_rate"],
                "count": row["count"],
            }
        )

    always_tokens = float(per_method.get("Always-Reason Chain-of-Thought", {}).get("avg_new_tokens") or 0.0)
    always_latency = float(per_method.get("Always-Reason Chain-of-Thought", {}).get("avg_latency_seconds") or 0.0)
    cost_table = []
    for method, row in sorted(per_method.items()):
        avg_tokens = float(row.get("avg_new_tokens") or 0.0)
        avg_latency = float(row.get("avg_latency_seconds") or 0.0)
        cost_table.append(
            {
                "method": method,
                "metric_value": row["metric_value"],
                "accuracy": row["score"],
                "avg_new_tokens": avg_tokens,
                "avg_latency_seconds": avg_latency,
                "route_rate": row["route_rate"],
                "token_saving_vs_always_reason": 0.0 if always_tokens <= 0 else (always_tokens - avg_tokens) / always_tokens,
                "latency_saving_vs_always_reason": 0.0
                if always_latency <= 0
                else (always_latency - avg_latency) / always_latency,
            }
        )

    simple_case_degradation = _simple_case_degradation(difficulty_breakdown_table)
    calibration_reliability = _calibration_reliability(difficulty_breakdown_table)
    routing_analysis = _routing_analysis(
        cost_table,
        difficulty_breakdown_table,
        simple_case_degradation=simple_case_degradation,
        calibration_reliability=calibration_reliability,
    )

    return (
        per_method,
        per_method_std,
        per_seed_results,
        dict(per_dataset_results),
        cost_table,
        difficulty_breakdown_table,
        routing_analysis,
        simple_case_degradation,
        calibration_reliability,
    )


def _routing_analysis(
    cost_table: list[dict],
    difficulty_breakdown_table: list[dict],
    *,
    simple_case_degradation: dict,
    calibration_reliability: list[dict],
) -> dict:
    return {
        "methods": [
            {
                "method": row["method"],
                "route_rate": row["route_rate"],
                "cost_saving": row["token_saving_vs_always_reason"],
                "latency_saving": row["latency_saving_vs_always_reason"],
                "avg_new_tokens": row["avg_new_tokens"],
                "avg_latency_seconds": row["avg_latency_seconds"],
                "utility": row["metric_value"],
            }
            for row in cost_table
            if any(token in row["method"].lower() for token in ("cggr", "gate", "routing"))
        ],
        "easy_medium_hard_breakdown": difficulty_breakdown_table,
        "simple_case_degradation": simple_case_degradation,
        "calibration_reliability": calibration_reliability,
    }


def _simple_case_degradation(difficulty_breakdown_table: list[dict]) -> dict:
    simple_labels = ("simple", "easy", "medium")
    direct = next(
        (
            row
            for label in simple_labels
            for row in difficulty_breakdown_table
            if row.get("method") == "Vanilla Direct Answering" and row.get("difficulty") == label
        ),
        {},
    )
    cggr = next(
        (
            row
            for label in simple_labels
            for row in difficulty_breakdown_table
            if row.get("method") == "CGGR" and row.get("difficulty") == label
        ),
        {},
    )
    baseline_accuracy = direct.get("accuracy")
    candidate_accuracy = cggr.get("accuracy")
    return {
        "subset": direct.get("difficulty") or cggr.get("difficulty") or "simple",
        "baseline_method": "Vanilla Direct Answering",
        "candidate_method": "CGGR",
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "degradation": (
            float(candidate_accuracy) - float(baseline_accuracy)
            if baseline_accuracy is not None and candidate_accuracy is not None
            else None
        ),
        "candidate_route_rate": cggr.get("route_rate"),
    }


def _calibration_reliability(difficulty_breakdown_table: list[dict]) -> list[dict]:
    labels = [("simple", 0.17), ("easy", 0.17), ("medium", 0.50), ("hard", 0.83)]
    rows = []
    seen = set()
    for bucket_name, proxy_value in labels:
        if bucket_name in seen:
            continue
        direct = next(
            (
                row
                for row in difficulty_breakdown_table
                if row.get("method") == "Vanilla Direct Answering" and row.get("difficulty") == bucket_name
            ),
            {},
        )
        cggr = next(
            (
                row
                for row in difficulty_breakdown_table
                if row.get("method") == "CGGR" and row.get("difficulty") == bucket_name
            ),
            {},
        )
        if direct and cggr:
            rows.append(
                {
                    "difficulty_bucket": bucket_name,
                    "difficulty_proxy": proxy_value,
                    "observed_gain_vs_direct": float(cggr.get("accuracy", 0.0) - direct.get("accuracy", 0.0)),
                    "route_rate": cggr.get("route_rate"),
                    "count": cggr.get("count"),
                }
            )
            seen.add(bucket_name)
    return rows


def _bootstrap_summary(candidate: list[float], baseline: list[float]) -> dict:
    pairs = [(float(c), float(b)) for c, b in zip(candidate, baseline)]
    if not pairs:
        return {
            "candidate_ci95": [0.0, 0.0],
            "baseline_ci95": [0.0, 0.0],
            "delta_ci95": [0.0, 0.0],
            "observed_delta": 0.0,
        }
    n = len(pairs)
    if n <= 7:
        samples = itertools.product(range(n), repeat=n)
    else:
        rng = random.Random(12345)
        samples = ([rng.randrange(n) for _ in range(n)] for _ in range(10000))
    candidate_means = []
    baseline_means = []
    delta_means = []
    for sample in samples:
        selected = [pairs[idx] for idx in sample]
        cand = _mean([row[0] for row in selected])
        base = _mean([row[1] for row in selected])
        candidate_means.append(cand)
        baseline_means.append(base)
        delta_means.append(cand - base)
    return {
        "candidate_ci95": [_percentile(candidate_means, 0.025), _percentile(candidate_means, 0.975)],
        "baseline_ci95": [_percentile(baseline_means, 0.025), _percentile(baseline_means, 0.975)],
        "delta_ci95": [_percentile(delta_means, 0.025), _percentile(delta_means, 0.975)],
        "observed_delta": _mean([c - b for c, b in pairs]),
    }


def _paired_permutation_pvalue(candidate: list[float], baseline: list[float]) -> float:
    pairs = [(float(c), float(b)) for c, b in zip(candidate, baseline)]
    if not pairs:
        return 1.0
    diffs = [c - b for c, b in pairs]
    observed = abs(_mean(diffs))
    if observed <= 1e-12:
        return 1.0
    total = 0
    extreme = 0
    for signs in itertools.product([-1, 1], repeat=len(diffs)):
        diff = abs(_mean([sign * value for sign, value in zip(signs, diffs)]))
        total += 1
        if diff >= observed - 1e-12:
            extreme += 1
    return float(extreme / max(1, total))


def _bootstrap_ci(per_seed_results: list[dict]) -> dict:
    candidate = []
    baseline = []
    for row in per_seed_results:
        methods = row.get("methods") if isinstance(row.get("methods"), dict) else {}
        if "CGGR" in methods and "Always-Reason Chain-of-Thought" in methods:
            candidate.append(float(methods["CGGR"].get("metric_value") or 0.0))
            baseline.append(float(methods["Always-Reason Chain-of-Thought"].get("metric_value") or 0.0))
    summary = _bootstrap_summary(candidate, baseline)
    return {
        "candidate_method": "CGGR",
        "baseline_method": "Always-Reason Chain-of-Thought",
        **summary,
        "paired_permutation_p": _paired_permutation_pvalue(candidate, baseline),
    }


def _expected_coverage(run_configs: list[dict]) -> tuple[list[str], list[Any], int]:
    datasets: list[str] = []
    seen_datasets: set[str] = set()
    declared_seed_counts: list[int] = []
    observed_seed_values: list[Any] = []
    seen_seeds: set[str] = set()
    examples = 0
    for config in run_configs:
        for row in config.get("targets") or []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or row.get("hf_dataset") or "").strip()
            if name and name not in seen_datasets:
                seen_datasets.add(name)
                datasets.append(name)
        try:
            declared = int(config.get("seeds") or 0)
        except (TypeError, ValueError):
            declared = 0
        if declared > 0:
            declared_seed_counts.append(declared)
        seed_values = config.get("seed_values")
        if isinstance(seed_values, list):
            for seed in seed_values:
                key = str(seed)
                if key not in seen_seeds:
                    seen_seeds.add(key)
                    observed_seed_values.append(seed)
        try:
            examples = max(examples, int(config.get("max_examples_per_dataset_seed") or 0))
        except (TypeError, ValueError):
            pass
    if declared_seed_counts:
        seed_values = list(range(max(declared_seed_counts)))
    else:
        seed_values = observed_seed_values
    return datasets, seed_values, examples


def _coverage_blockers(raw_rows: list[dict], run_configs: list[dict]) -> list[str]:
    datasets, seed_values, examples = _expected_coverage(run_configs)
    if not datasets or not seed_values or examples <= 0:
        return ["cannot verify full cell coverage from run_config targets/seed_values/max_examples_per_dataset_seed"]
    counts: dict[tuple[str, str, Any], int] = defaultdict(int)
    example_counts: dict[tuple[str, str, Any, str], int] = defaultdict(int)
    for row in raw_rows:
        counts[(str(row.get("method") or ""), str(row.get("dataset") or ""), row.get("seed"))] += 1
        example_id = str(row.get("example_id") or "")
        if example_id:
            example_counts[
                (
                    str(row.get("method") or ""),
                    str(row.get("dataset") or ""),
                    row.get("seed"),
                    example_id,
                )
            ] += 1
    blockers = []
    duplicates = [key for key, count in example_counts.items() if count > 1]
    if duplicates:
        method, dataset, seed, example_id = duplicates[0]
        blockers.append(
            "duplicate raw prediction rows after merge: "
            f"method={method} dataset={dataset} seed={seed} example_id={example_id}"
        )
    for method in sorted(REQUIRED_METHODS):
        for dataset in datasets:
            for seed in seed_values:
                count = counts.get((method, dataset, seed), 0)
                if count < examples:
                    blockers.append(
                        f"method/dataset/seed cell below paper gate: method={method} dataset={dataset} seed={seed} count={count}/{examples}"
                    )
                    if len(blockers) >= 20:
                        blockers.append("additional missing or incomplete cells omitted")
                        return blockers
                elif count > examples:
                    blockers.append(
                        f"method/dataset/seed cell above paper gate, indicating overlapping shards: "
                        f"method={method} dataset={dataset} seed={seed} count={count}/{examples}"
                    )
                    if len(blockers) >= 20:
                        blockers.append("additional duplicate or overlapping cells omitted")
                        return blockers
    return blockers


def merge(shards: list[Path], out_workdir: Path) -> dict:
    if len(shards) < 2:
        return {"ok": False, "blockers": ["at least two shard workdirs are required"]}
    blockers: list[str] = []
    out_results = out_workdir / "results"
    if out_results.exists() and any(out_results.iterdir()):
        return {"ok": False, "blockers": [f"output results directory is not empty: {out_results}"]}
    out_results.mkdir(parents=True, exist_ok=True)

    run_configs = [_load_json(path / "results" / "run_config.json") for path in shards]
    summaries = [_load_json(path / "results" / "benchmark_summary.json") for path in shards]
    raw_rows: list[dict] = []
    routing_rows: list[dict] = []
    failure_rows: list[dict] = []
    for shard in shards:
        results = shard / "results"
        raw_rows.extend(_iter_jsonl(results / "raw_predictions.jsonl"))
        routing_rows.extend(_iter_jsonl(results / "routing_decisions.jsonl"))
        failure_rows.extend(_iter_jsonl(results / "failure_cases.jsonl"))

    generation_failures = [row for row in failure_rows if row.get("stage") == "generation_or_scoring"]
    if generation_failures:
        blockers.append("generation_or_scoring failures present in shard failure_cases.jsonl")
    methods_seen = {str(row.get("method") or "") for row in raw_rows if row.get("method")}
    missing_methods = sorted(REQUIRED_METHODS - methods_seen)
    if missing_methods:
        blockers.append("missing methods after merge: " + ", ".join(missing_methods))
    for method in sorted(methods_seen):
        if method not in REQUIRED_METHODS:
            continue
        count = sum(1 for row in raw_rows if row.get("method") == method)
        if count < 5 * 4 * 128:
            blockers.append(f"method {method} has too few rows for paper gate: {count}")
    blockers.extend(_coverage_blockers(raw_rows, run_configs))
    if blockers:
        return {"ok": False, "blockers": blockers}

    try:
        lambda_cost = float(run_configs[0].get("cost_lambda") or 0.03)
    except (TypeError, ValueError):
        lambda_cost = 0.03
    (
        per_method,
        per_method_std,
        per_seed_results,
        per_dataset_results,
        cost_table,
        difficulty_table,
        routing_analysis,
        simple_case_degradation,
        calibration_reliability,
    ) = _aggregate_rows(
        raw_rows,
        routing_rows,
        lambda_cost=lambda_cost,
    )
    ablation_table = []
    cggr_value = float(per_method.get("CGGR", {}).get("metric_value") or 0.0)
    for ablation in sorted(REQUIRED_ABLATIONS):
        method = "CGGR/" + ablation
        value = float(per_method.get(method, {}).get("metric_value") or 0.0)
        ablation_table.append(
            {
                "ablation": ablation,
                "method": method,
                "metric_value": value,
                "delta_vs_cggr": value - cggr_value,
            }
        )

    base_config = dict(run_configs[0])
    base_config["methods"] = sorted(REQUIRED_METHODS)
    base_config["sharded_run"] = False
    base_config["shard_axes"] = {"method": False, "target": False, "seed": False}
    base_config["merged_from_method_shards"] = [str(path) for path in shards]
    base_config["full_benchmark_completed"] = True
    _write_json(out_results / "run_config.json", base_config)

    _write_jsonl(out_results / "raw_predictions.jsonl", raw_rows)
    _write_jsonl(out_results / "routing_decisions.jsonl", routing_rows)
    _write_jsonl(out_results / "failure_cases.jsonl", failure_rows)
    _write_json(out_results / "per_seed_results.json", per_seed_results)
    _write_json(out_results / "per_dataset_results.json", per_dataset_results)
    _write_json(out_results / "main_results_table.json", cost_table)
    _write_json(out_results / "cost_utility_tradeoff_table.json", cost_table)
    _write_json(out_results / "latency_tokens_table.json", cost_table)
    _write_json(out_results / "difficulty_breakdown_table.json", difficulty_table)
    _write_json(out_results / "routing_analysis.json", routing_analysis)
    _write_json(out_results / "ablation_table.json", ablation_table)
    _write_json(out_results / "simple_case_degradation.json", simple_case_degradation)
    _write_json(out_results / "calibration_reliability.json", calibration_reliability)
    bootstrap = _bootstrap_ci(per_seed_results)
    _write_json(out_results / "bootstrap_ci.json", bootstrap)

    datasets = []
    for row in base_config.get("targets") or []:
        if isinstance(row, dict):
            datasets.append(
                {
                    "name": row.get("name"),
                    "id": row.get("hf_dataset") or row.get("name"),
                    "config": row.get("config") or "",
                    "split": row.get("split") or "",
                    "num_materialized_examples": 5 * 128,
                    "license_or_source": row.get("hf_dataset") or row.get("name"),
                    "preprocessing": "Answer normalization with exact/F1 scoring and task-specific numeric/boolean extraction.",
                }
            )
    environment_report = {
        "schema_version": "merged_benchmark_environment_report_v1",
        "merged_from_method_shards": [str(path) for path in shards],
        "shard_environment_reports": [
            _load_json(path / "results" / "environment_report.json")
            for path in shards
            if (path / "results" / "environment_report.json").exists()
        ],
    }
    _write_json(out_results / "environment_report.json", environment_report)

    summary = {
        "primary_metric": "cost_adjusted_accuracy",
        "metric_name": "cost_adjusted_accuracy",
        "candidate_method": "CGGR",
        "best_method": max(per_method.items(), key=lambda item: float(item[1].get("metric_value") or 0.0))[0],
        "per_method": per_method,
        "per_method_std": per_method_std,
        "seed_results": per_seed_results,
        "num_seeds": 5,
        "datasets": datasets,
        "dataset": datasets[0] if datasets else {},
        "model": {
            "id": base_config.get("model_id"),
            "backend": "transformers",
            "cuda": True,
            "hardware": "NVIDIA L40S",
        },
        "ablations": sorted(REQUIRED_ABLATIONS),
        "ablation_results": ablation_table,
        "ablation_table": ablation_table,
        "cost_utility_tradeoff_table": cost_table,
        "difficulty_breakdown_table": difficulty_table,
        "routing_analysis": routing_analysis,
        "latency_tokens_table": cost_table,
        "simple_case_degradation": simple_case_degradation,
        "calibration_reliability": calibration_reliability,
        "bootstrap_ci": bootstrap,
        "load_failures": [],
        "budget": {
            "seeds": 5,
            "max_examples_per_dataset_seed": 128,
            "methods": sorted(REQUIRED_METHODS),
            "target_count": 4,
        },
        "method": "Counterfactual Gain Gated Reasoning (CGGR)",
        "duration_seconds": sum(float(summary.get("duration_seconds") or 0.0) for summary in summaries),
        "peak_vram_mb": max(float(summary.get("peak_vram_mb") or 0.0) for summary in summaries),
        "hardware": "NVIDIA L40S",
        "full_benchmark_completed": True,
        "merged_from_method_shards": [str(path) for path in shards],
    }
    _write_json(out_results / "benchmark_summary.json", summary)
    manifest = {
        "full_benchmark_completed": True,
        "merged_from_method_shards": [str(path) for path in shards],
        "datasets": datasets,
        "methods": sorted(REQUIRED_METHODS),
        "model": summary["model"],
        "artifact_paths": {path.name: str(path) for path in out_results.iterdir() if path.is_file()},
    }
    _write_json(out_results / "artifact_manifest.json", manifest)
    _write_json(out_results / "benchmark_artifact_manifest.json", manifest)
    return {
        "ok": True,
        "output_workdir": str(out_workdir),
        "raw_predictions_lines": len(raw_rows),
        "methods": sorted(methods_seen),
        "written": sorted(str(path) for path in out_results.iterdir() if path.is_file()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-workdir", type=Path, required=True)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    result = merge(args.shards, args.out_workdir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
