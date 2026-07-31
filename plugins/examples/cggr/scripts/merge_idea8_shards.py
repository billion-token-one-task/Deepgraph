#!/usr/bin/env python3
"""Merge idea8 benchmark shards into a paper-auditable result package."""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any


CANDIDATE = "Certified Residual Policy Packets"
DIRECT = "Vanilla Direct Answering"
CONFIDENCE = "Confidence Gate"
DEFAULT_METHODS = [
    DIRECT,
    CONFIDENCE,
    CANDIDATE,
    "Always-Reason Chain-of-Thought",
    "Self-Consistency Reasoning",
    "Least-to-Most Prompting",
    "Disagreement Routing",
    "Random Budget-Matched Routing",
    "CAR-Style Certainty Adaptive Routing",
    "Self-Route-Style Mode Routing",
    "Rational-Metareasoning VOC Routing",
]
DEFAULT_DATASETS = ["OpenBookQA", "QASC"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DATASET_META = {
    "OpenBookQA": {
        "name": "OpenBookQA",
        "hf_dataset": "allenai/openbookqa",
        "config": "main",
        "split": "validation",
        "num_materialized_examples": 500,
        "sample_policy": "official validation split; all 500 examples",
    },
    "QASC": {
        "name": "QASC",
        "hf_dataset": "allenai/qasc",
        "config": "default",
        "split": "validation",
        "num_materialized_examples": 926,
        "sample_policy": "official validation split; all 926 examples",
    },
}


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _source_priority(shard: Path, row: dict[str, Any]) -> int:
    name = shard.name.lower()
    method = str(row.get("method") or "")
    if method == CANDIDATE:
        if "crpp_v8_voc028_cap38" in name:
            return 0
        if "crpp_v7_cap38" in name or "direct_conf_crpp_v7" in name:
            return 5
        return 50
    if method in {DIRECT, CONFIDENCE}:
        if "direct_conf_crpp_v7" in name or "direct_conf_v7_baselines" in name:
            return 0
        if "direct_conf_crpp_v4" in name:
            return 5
        return 40
    if "missing_baselines_v" in name:
        return 0
    return 30


def _candidate_source_allowed(shard: Path, row: dict[str, Any]) -> bool:
    name = shard.name.lower()
    method = str(row.get("method") or "")
    if method == CANDIDATE:
        return "crpp_v8_voc028_cap38" in name or "crpp_v7_cap38" in name or "direct_conf_crpp_v7" in name
    if method in {DIRECT, CONFIDENCE}:
        return (
            "direct_conf_crpp_v7" in name
            or "direct_conf_v7_baselines" in name
            or "direct_conf_crpp_v4" in name
        )
    return "missing_baselines_v" in name


def _discover_shards(root: Path) -> list[Path]:
    rows = []
    for status_path in root.glob("*/shard_status.json"):
        status = _read_json(status_path, {})
        results = Path(status.get("results_dir") or status_path.parent / "results")
        if status.get("status") == "ok" and (results / "raw_predictions.jsonl").exists():
            rows.append(status_path.parent)
    return sorted(rows, key=lambda p: p.name)


def _as_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(parsed) or math.isinf(parsed):
        return 0.0
    return parsed


def _metric_from_rows(rows: list[dict[str, Any]], cost_lambda: float) -> dict[str, Any]:
    count = len(rows)
    if count <= 0:
        return {
            "score": 0.0,
            "exact": 0.0,
            "f1": 0.0,
            "avg_new_tokens": 0.0,
            "avg_latency_seconds": 0.0,
            "route_rate": 0.0,
            "cost_adjusted_accuracy": 0.0,
            "metric_value": 0.0,
            "count": 0,
        }
    score = sum(_as_float(r.get("primary_score", r.get("exact"))) for r in rows)
    exact = sum(_as_float(r.get("exact")) for r in rows)
    f1 = sum(_as_float(r.get("f1")) for r in rows)
    tokens = sum(_as_float(r.get("new_tokens")) for r in rows)
    latency = sum(_as_float(r.get("latency_seconds")) for r in rows)
    routed = sum(1.0 for r in rows if r.get("routed_to_deliberation"))
    metric = (score / count) - cost_lambda * ((tokens / count) / 192.0)
    return {
        "score": float(score / count),
        "exact": float(exact / count),
        "f1": float(f1 / count),
        "avg_new_tokens": float(tokens / count),
        "avg_latency_seconds": float(latency / count),
        "route_rate": float(routed / count),
        "cost_adjusted_accuracy": float(metric),
        "metric_value": float(metric),
        "count": count,
    }


def _std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _bootstrap_ci(values: list[float], rounds: int = 2000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(12345)
    means = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return [float(means[int(0.025 * (len(means) - 1))]), float(means[int(0.975 * (len(means) - 1))])]


def _paired_permutation_pvalue(candidate: list[float], baseline: list[float]) -> float:
    pairs = [(float(c), float(b)) for c, b in zip(candidate, baseline)]
    if not pairs:
        return 1.0
    observed = abs(sum(c - b for c, b in pairs) / len(pairs))
    if len(pairs) <= 16:
        signs_iter = itertools.product([-1, 1], repeat=len(pairs))
    else:
        rng = random.Random(20260611)
        signs_iter = ([rng.choice([-1, 1]) for _ in pairs] for _ in range(20000))
    count = 0
    extreme = 0
    for signs in signs_iter:
        diff = abs(sum(sign * (c - b) for sign, (c, b) in zip(signs, pairs)) / len(pairs))
        count += 1
        if diff >= observed - 1e-12:
            extreme += 1
    return float(extreme / max(1, count))


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-root", type=Path, default=Path("/root/deepgraph_ideas/idea_8/experiments/main/shards"))
    parser.add_argument("--out-dir", type=Path, default=Path("/root/deepgraph_ideas/idea_8/experiments/main/merged/idea8_qwen_v7"))
    parser.add_argument("--require-full-matrix", action="store_true")
    parser.add_argument("--cost-lambda", type=float, default=0.03)
    args = parser.parse_args()

    out_results = args.out_dir / "results"
    out_results.mkdir(parents=True, exist_ok=True)
    required_methods = list(DEFAULT_METHODS)
    required_datasets = list(DEFAULT_DATASETS)
    required_seeds = list(DEFAULT_SEEDS)

    selected: dict[tuple[str, int, str, str], tuple[int, dict[str, Any], str]] = {}
    route_by_example: dict[tuple[str, int, str], float] = {}
    route_rows: list[dict[str, Any]] = []
    source_statuses: list[dict[str, Any]] = []
    for shard in _discover_shards(args.shards_root):
        status = _read_json(shard / "shard_status.json", {})
        source_statuses.append(status)
        results = Path(status.get("results_dir") or shard / "results")
        for row in _iter_jsonl(results / "routing_decisions.jsonl") or []:
            route_rows.append(row)
            try:
                route_by_example[(str(row.get("dataset")), int(row.get("seed")), str(row.get("example_id")))] = float(row.get("difficulty"))
            except (TypeError, ValueError):
                pass
        for row in _iter_jsonl(results / "raw_predictions.jsonl") or []:
            if not _candidate_source_allowed(shard, row):
                continue
            dataset = str(row.get("dataset") or "")
            method = str(row.get("method") or "")
            try:
                seed = int(row.get("seed"))
            except (TypeError, ValueError):
                continue
            if dataset not in required_datasets or seed not in required_seeds or method not in required_methods:
                continue
            key = (dataset, seed, method, str(row.get("example_id") or ""))
            priority = _source_priority(shard, row)
            current = selected.get(key)
            if current is None or priority < current[0]:
                selected[key] = (priority, row, str(shard))

    raw_rows = [item[1] for item in selected.values()]
    raw_rows.sort(key=lambda r: (str(r.get("dataset")), int(r.get("seed", -1)), str(r.get("method")), str(r.get("example_id"))))
    by_method: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    by_dataset_method: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    by_seed_method: dict[tuple[int, str], list[dict[str, Any]]] = collections.defaultdict(list)
    by_dataset_seed_method: dict[tuple[str, int, str], list[dict[str, Any]]] = collections.defaultdict(list)
    by_example_method: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for row in raw_rows:
        dataset = str(row.get("dataset"))
        method = str(row.get("method"))
        seed = int(row.get("seed"))
        by_method[method].append(row)
        by_dataset_method[(dataset, method)].append(row)
        by_seed_method[(seed, method)].append(row)
        by_dataset_seed_method[(dataset, seed, method)].append(row)
        by_example_method[(dataset, seed, str(row.get("example_id")), method)] = row

    per_method = {method: _metric_from_rows(by_method.get(method, []), args.cost_lambda) for method in required_methods if by_method.get(method)}
    per_dataset_results: dict[str, dict[str, Any]] = {}
    for dataset in required_datasets:
        per_dataset_results[dataset] = {
            method: _metric_from_rows(by_dataset_method.get((dataset, method), []), args.cost_lambda)
            for method in required_methods
            if by_dataset_method.get((dataset, method))
        }

    seed_results = []
    per_seed_values: dict[str, list[float]] = collections.defaultdict(list)
    for seed in required_seeds:
        seed_row = {"seed": seed, "datasets": {}, "methods": {}}
        for dataset in required_datasets:
            dataset_row = {"num_examples": DATASET_META[dataset]["num_materialized_examples"], "methods": {}}
            for method in required_methods:
                rows = by_dataset_seed_method.get((dataset, seed, method), [])
                if rows:
                    dataset_row["methods"][method] = _metric_from_rows(rows, args.cost_lambda)
            seed_row["datasets"][dataset] = dataset_row
        for method in required_methods:
            rows = by_seed_method.get((seed, method), [])
            if rows:
                metric = _metric_from_rows(rows, args.cost_lambda)
                seed_row["methods"][method] = metric
                per_seed_values[method].append(float(metric["metric_value"]))
        seed_results.append(seed_row)

    per_method_std = {method: _std(values) for method, values in per_seed_values.items()}
    strongest_name = ""
    strongest_value = None
    for method, row in per_method.items():
        if method == CANDIDATE:
            continue
        value = float(row.get("metric_value") or 0.0)
        if strongest_value is None or value > strongest_value:
            strongest_name = method
            strongest_value = value
    candidate_values = per_seed_values.get(CANDIDATE, [])
    baseline_values = per_seed_values.get(strongest_name, [])
    bootstrap = {
        "candidate_method": CANDIDATE,
        "baseline_method": strongest_name,
        "candidate_ci95": _bootstrap_ci(candidate_values),
        "baseline_ci95": _bootstrap_ci(baseline_values),
        "paired_permutation_p": _paired_permutation_pvalue(candidate_values, baseline_values),
        "p_value": _paired_permutation_pvalue(candidate_values, baseline_values),
    }

    missing_matrix = []
    for dataset in required_datasets:
        expected = int(DATASET_META[dataset]["num_materialized_examples"])
        for seed in required_seeds:
            for method in required_methods:
                got = len(by_dataset_seed_method.get((dataset, seed, method), []))
                if got != expected:
                    missing_matrix.append({"dataset": dataset, "seed": seed, "method": method, "expected_rows": expected, "observed_rows": got})

    difficulty_acc: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in raw_rows:
        key = (str(row.get("dataset")), int(row.get("seed")), str(row.get("example_id")))
        difficulty = route_by_example.get(key)
        if difficulty is None:
            continue
        bucket = "easy" if difficulty < 0.33 else "medium" if difficulty < 0.66 else "hard"
        difficulty_acc[(str(row.get("method")), bucket)].append(row)
    difficulty_breakdown = []
    for (method, bucket), rows in sorted(difficulty_acc.items()):
        metric = _metric_from_rows(rows, args.cost_lambda)
        difficulty_breakdown.append({
            "method": method,
            "difficulty": bucket,
            "accuracy": metric["score"],
            "avg_new_tokens": metric["avg_new_tokens"],
            "avg_latency_seconds": metric["avg_latency_seconds"],
            "route_rate": metric["route_rate"],
            "count": metric["count"],
        })

    latency_tokens_table = []
    always_tokens = float(per_method.get("Always-Reason Chain-of-Thought", {}).get("avg_new_tokens") or 0.0)
    always_latency = float(per_method.get("Always-Reason Chain-of-Thought", {}).get("avg_latency_seconds") or 0.0)
    for method, row in per_method.items():
        avg_tokens = float(row.get("avg_new_tokens") or 0.0)
        avg_latency = float(row.get("avg_latency_seconds") or 0.0)
        latency_tokens_table.append({
            "method": method,
            "metric_value": row["metric_value"],
            "accuracy": row["score"],
            "avg_new_tokens": avg_tokens,
            "avg_latency_seconds": avg_latency,
            "route_rate": row["route_rate"],
            "token_saving_vs_always_reason": 0.0 if always_tokens <= 0 else 1.0 - (avg_tokens / always_tokens),
            "latency_saving_vs_always_reason": 0.0 if always_latency <= 0 else 1.0 - (avg_latency / always_latency),
        })
    latency_tokens_table.sort(key=lambda row: str(row["method"]))
    quality_cost_frontier = sorted(
        [
            {"method": method, "metric_value": row["metric_value"], "avg_new_tokens": row["avg_new_tokens"], "avg_latency_seconds": row["avg_latency_seconds"]}
            for method, row in per_method.items()
        ],
        key=lambda row: (row["avg_new_tokens"], -row["metric_value"]),
    )
    route_rate_sweep = [
        {"method": method, "route_rate": row["route_rate"], "metric_value": row["metric_value"], "avg_new_tokens": row["avg_new_tokens"]}
        for method, row in sorted(per_method.items())
        if any(token in method.lower() for token in ("gate", "routing", "route", "packet", "metareasoning"))
    ]
    routing_analysis = {
        "candidate_method": CANDIDATE,
        "per_method_route_rate": {method: row["route_rate"] for method, row in per_method.items()},
        "routing_decision_count": len(route_rows),
        "note": "Route rates are read from instrumented routing_decisions.jsonl where available; non-routing baselines have zero route rate.",
    }

    easy_direct = [row for row in by_method.get(DIRECT, []) if route_by_example.get((str(row.get("dataset")), int(row.get("seed")), str(row.get("example_id"))), 1.0) < 0.33]
    easy_candidate = [row for row in by_method.get(CANDIDATE, []) if route_by_example.get((str(row.get("dataset")), int(row.get("seed")), str(row.get("example_id"))), 1.0) < 0.33]
    direct_easy_metric = _metric_from_rows(easy_direct, args.cost_lambda)
    candidate_easy_metric = _metric_from_rows(easy_candidate, args.cost_lambda)
    simple_case_degradation = {
        "subset": "easy",
        "baseline_method": DIRECT,
        "candidate_method": CANDIDATE,
        "baseline_accuracy": direct_easy_metric["score"],
        "candidate_accuracy": candidate_easy_metric["score"],
        "degradation": direct_easy_metric["score"] - candidate_easy_metric["score"],
        "candidate_route_rate": candidate_easy_metric["route_rate"],
        "count": candidate_easy_metric["count"],
    }

    calibration = []
    for bucket, lo, hi, proxy in (("easy", 0.0, 0.33, 0.17), ("medium", 0.33, 0.66, 0.50), ("hard", 0.66, 1.01, 0.83)):
        gains = []
        routes = []
        for dataset, seed, example_id, method in list(by_example_method.keys()):
            if method != CANDIDATE:
                continue
            diff = route_by_example.get((dataset, seed, example_id))
            if diff is None or not (lo <= diff < hi):
                continue
            cand = by_example_method.get((dataset, seed, example_id, CANDIDATE))
            direct = by_example_method.get((dataset, seed, example_id, DIRECT))
            if not cand or not direct:
                continue
            gains.append(_as_float(cand.get("primary_score", cand.get("exact"))) - _as_float(direct.get("primary_score", direct.get("exact"))))
            routes.append(1.0 if cand.get("routed_to_deliberation") else 0.0)
        calibration.append({
            "difficulty_bucket": bucket,
            "difficulty_proxy": proxy,
            "observed_gain_vs_direct": float(sum(gains) / max(1, len(gains))),
            "route_rate": float(sum(routes) / max(1, len(routes))),
            "count": len(gains),
        })

    ablation_table = []
    if "Rational-Metareasoning VOC Routing" in per_method and CANDIDATE in per_method:
        ablation_table.append({
            "ablation": "remove_v7_direct_budget_cap",
            "method": "Rational-Metareasoning VOC Routing",
            "metric_value": per_method["Rational-Metareasoning VOC Routing"]["metric_value"],
            "delta_vs_candidate": per_method["Rational-Metareasoning VOC Routing"]["metric_value"] - per_method[CANDIDATE]["metric_value"],
            "note": "Baseline uses the same VOC routing family without the v7 CRPP direct-token budget patch.",
        })
    for method in ("Confidence Gate", "Disagreement Routing", "Random Budget-Matched Routing"):
        if method in per_method and CANDIDATE in per_method:
            ablation_table.append({
                "ablation": "selector_family_" + method.lower().replace(" ", "_").replace("-", "_"),
                "method": method,
                "metric_value": per_method[method]["metric_value"],
                "delta_vs_candidate": per_method[method]["metric_value"] - per_method[CANDIDATE]["metric_value"],
            })

    full_matrix_complete = not missing_matrix
    candidate_metric = float(per_method.get(CANDIDATE, {}).get("metric_value") or 0.0)
    strongest_gap = candidate_metric - float(strongest_value or 0.0)
    gate_blockers = []
    if missing_matrix:
        gate_blockers.append(f"missing {len(missing_matrix)} dataset/seed/method cells from required Qwen matrix")
    if strongest_name and strongest_gap <= 0:
        gate_blockers.append(f"candidate does not beat strongest observed deployable baseline {strongest_name}: gap {strongest_gap:+.6g}")
    if CANDIDATE not in per_method:
        gate_blockers.append("candidate method rows are missing")
    if DIRECT not in per_method or CONFIDENCE not in per_method:
        gate_blockers.append("core direct/confidence baselines are missing")

    summary = {
        "schema_version": "idea8_merged_benchmark_v1",
        "primary_metric": "cost_adjusted_accuracy",
        "metric_name": "cost_adjusted_accuracy",
        "metric_direction": "higher",
        "candidate_method": CANDIDATE,
        "best_method": max(per_method, key=lambda key: per_method[key]["metric_value"]) if per_method else "",
        "strongest_deployable_baseline": strongest_name,
        "strongest_baseline_gap": strongest_gap,
        "per_method": per_method,
        "per_method_std": per_method_std,
        "seed_results": seed_results,
        "num_seeds": len(required_seeds),
        "seeds": required_seeds,
        "datasets": [DATASET_META[name] for name in required_datasets],
        "dataset": {"names": required_datasets, "split_policy": "official/materialized validation splits"},
        "models": [{"id": "/root/hf_models/Qwen/Qwen2-7B-Instruct", "backend": "transformers", "load_in_4bit": True, "requires_cuda": True}],
        "model": {"id": "/root/hf_models/Qwen/Qwen2-7B-Instruct", "backend": "transformers", "load_in_4bit": True, "requires_cuda": True},
        "methods": required_methods,
        "per_dataset_results": per_dataset_results,
        "main_results_table": per_method,
        "latency_tokens_table": latency_tokens_table,
        "cost_utility_tradeoff_table": latency_tokens_table,
        "quality_cost_frontier": quality_cost_frontier,
        "route_rate_sweep_table": route_rate_sweep,
        "routing_analysis": routing_analysis,
        "ablation_table": ablation_table,
        "difficulty_breakdown_table": difficulty_breakdown,
        "simple_case_degradation": simple_case_degradation,
        "calibration_reliability": calibration,
        "bootstrap_ci": bootstrap,
        "statistical_tests": bootstrap,
        "load_failures": [],
        "raw_prediction_count": len(raw_rows),
        "routing_decision_count": len(route_rows),
        "failure_case_count": sum(1 for row in raw_rows if row.get("method") == CANDIDATE and _as_float(row.get("primary_score", row.get("exact"))) < 0.5),
        "policy_env": {
            "candidate_v7": {
                "DEEPGRAPH_VOC_THRESHOLD": "0.99",
                "DEEPGRAPH_VOC_REASONING_COST": "0.16",
                "DEEPGRAPH_CRPP_DIRECT_MAX_NEW_TOKENS": "38",
                "DEEPGRAPH_VOC_DELIBERATE_MAX_NEW_TOKENS": "48",
            }
        },
        "benchmark_protocol": {
            "schema_version": "benchmark_protocol_v2",
            "official_or_materialized_split": True,
            "dataset_protocols": [
                {
                    "name": meta["name"],
                    "hf_dataset": meta["hf_dataset"],
                    "config": meta["config"],
                    "split": meta["split"],
                    "sample_policy": {"kind": "official_validation_full_or_cap", "expected_examples": meta["num_materialized_examples"]},
                }
                for meta in (DATASET_META[name] for name in required_datasets)
            ],
            "seed_policy": {"minimum_repeats": 5, "seeds": required_seeds},
            "full_benchmark_requirements": {
                "required_dataset_names": required_datasets,
                "required_model_names": ["/root/hf_models/Qwen/Qwen2-7B-Instruct"],
                "required_baseline_names": [method for method in required_methods if method != CANDIDATE],
                "required_artifacts": [
                    "run_config.json",
                    "raw_predictions.jsonl",
                    "routing_decisions.jsonl",
                    "per_seed_results.json",
                    "per_dataset_results.json",
                    "main_results_table.json",
                    "cost_utility_tradeoff_table.json",
                    "quality_cost_frontier.json",
                    "route_rate_sweep_table.json",
                    "ablation_table.json",
                    "difficulty_breakdown_table.json",
                    "routing_analysis.json",
                    "latency_tokens_table.json",
                    "simple_case_degradation.json",
                    "calibration_reliability.json",
                    "bootstrap_ci.json",
                    "failure_cases.jsonl",
                    "artifact_manifest.json",
                ],
            },
        },
        "full_benchmark_completed": bool(full_matrix_complete and not gate_blockers),
        "readiness_blockers": gate_blockers,
        "missing_matrix": missing_matrix[:200],
        "hardware": "NVIDIA GeForce RTX 3090",
        "merged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    artifact_paths = {
        "run_config": out_results / "run_config.json",
        "benchmark_summary": out_results / "benchmark_summary.json",
        "raw_predictions": out_results / "raw_predictions.jsonl",
        "routing_decisions": out_results / "routing_decisions.jsonl",
        "failure_cases": out_results / "failure_cases.jsonl",
        "per_seed_results": out_results / "per_seed_results.json",
        "per_dataset_results": out_results / "per_dataset_results.json",
        "main_results_table": out_results / "main_results_table.json",
        "cost_utility_tradeoff_table": out_results / "cost_utility_tradeoff_table.json",
        "quality_cost_frontier": out_results / "quality_cost_frontier.json",
        "route_rate_sweep_table": out_results / "route_rate_sweep_table.json",
        "ablation_table": out_results / "ablation_table.json",
        "difficulty_breakdown_table": out_results / "difficulty_breakdown_table.json",
        "routing_analysis": out_results / "routing_analysis.json",
        "latency_tokens_table": out_results / "latency_tokens_table.json",
        "simple_case_degradation": out_results / "simple_case_degradation.json",
        "calibration_reliability": out_results / "calibration_reliability.json",
        "bootstrap_ci": out_results / "bootstrap_ci.json",
        "source_shard_statuses": out_results / "source_shard_statuses.json",
        "gate_report": args.out_dir / "idea8_v7_gate_report.json",
    }
    _append_jsonl(artifact_paths["raw_predictions"], raw_rows)
    _append_jsonl(artifact_paths["routing_decisions"], route_rows)
    _append_jsonl(artifact_paths["failure_cases"], [row for row in raw_rows if row.get("method") == CANDIDATE and _as_float(row.get("primary_score", row.get("exact"))) < 0.5])
    _write_json(artifact_paths["run_config"], {
        "source": "merged idea8 shards",
        "shards_root": str(args.shards_root),
        "required_methods": required_methods,
        "required_datasets": required_datasets,
        "required_seeds": required_seeds,
        "cost_lambda": args.cost_lambda,
    })
    _write_json(artifact_paths["per_seed_results"], seed_results)
    _write_json(artifact_paths["per_dataset_results"], per_dataset_results)
    _write_json(artifact_paths["main_results_table"], per_method)
    _write_json(artifact_paths["cost_utility_tradeoff_table"], latency_tokens_table)
    _write_json(artifact_paths["quality_cost_frontier"], quality_cost_frontier)
    _write_json(artifact_paths["route_rate_sweep_table"], route_rate_sweep)
    _write_json(artifact_paths["ablation_table"], ablation_table)
    _write_json(artifact_paths["difficulty_breakdown_table"], difficulty_breakdown)
    _write_json(artifact_paths["routing_analysis"], routing_analysis)
    _write_json(artifact_paths["latency_tokens_table"], latency_tokens_table)
    _write_json(artifact_paths["simple_case_degradation"], simple_case_degradation)
    _write_json(artifact_paths["calibration_reliability"], calibration)
    _write_json(artifact_paths["bootstrap_ci"], bootstrap)
    _write_json(artifact_paths["source_shard_statuses"], source_statuses)
    _write_json(artifact_paths["benchmark_summary"], summary)

    manifest_artifacts = {key: str(path) for key, path in artifact_paths.items()}
    manifest = {
        "contract_type": "BenchmarkArtifactManifest",
        "schema_version": "idea8_manifest_v1",
        "run_id": 13,
        "deep_insight_id": 8,
        "full_benchmark_completed": summary["full_benchmark_completed"],
        "verdict": "confirmed" if summary["full_benchmark_completed"] else "needs_more_shards",
        "primary_metric": summary["primary_metric"],
        "metric_name": summary["metric_name"],
        "num_seeds": summary["num_seeds"],
        "method_count": len(per_method),
        "datasets": summary["datasets"],
        "model": summary["model"],
        "hardware": summary["hardware"],
        "readiness_blockers": gate_blockers,
        "missing_matrix_count": len(missing_matrix),
        "artifacts": manifest_artifacts,
        "artifact_hashes": {key: _hash_file(path) for key, path in artifact_paths.items() if path.exists() and path.is_file()},
    }
    summary["artifact_manifest"] = manifest
    _write_json(artifact_paths["benchmark_summary"], summary)
    _write_json(out_results / "artifact_manifest.json", manifest)
    _write_json(out_results / "benchmark_artifact_manifest.json", manifest)
    gate_report = {
        "status": "passed" if not gate_blockers else "needs_more_shards",
        "candidate_method": CANDIDATE,
        "candidate_metric": candidate_metric,
        "strongest_deployable_baseline": strongest_name,
        "strongest_baseline_metric": strongest_value,
        "strongest_baseline_gap": strongest_gap,
        "full_matrix_complete": full_matrix_complete,
        "readiness_blockers": gate_blockers,
        "missing_matrix_count": len(missing_matrix),
        "missing_matrix_preview": missing_matrix[:20],
        "out_dir": str(args.out_dir),
    }
    _write_json(artifact_paths["gate_report"], gate_report)
    print(json.dumps(gate_report, indent=2, ensure_ascii=False))
    if args.require_full_matrix and gate_blockers:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
