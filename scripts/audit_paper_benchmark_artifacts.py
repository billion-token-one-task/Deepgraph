#!/usr/bin/env python3
"""Audit DeepGraph benchmark artifacts before manuscript claims are allowed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_DATASETS = {
    "musique-ans",
    "strategyqa",
    "2wikimultihopqa",
    "stress test split: simple-vs-hard counterfactual partition",
}

REQUIRED_METHODS = {
    "Vanilla Direct Answering",
    "Always-Reason Chain-of-Thought",
    "Self-Consistency Reasoning",
    "Least-to-Most Prompting",
    "Confidence Gate",
    "Disagreement Routing",
    "Random Budget-Matched Routing",
    "CGGR",
}

TOP_VENUE_BASELINE_METHODS = {
    "CAR-Style Certainty Adaptive Routing",
    "Self-Route-Style Mode Routing",
    "Rational-Metareasoning VOC Routing",
}

REQUIRED_ABLATIONS = {
    "no_counterfactual_delta",
    "no_lcb",
    "no_self_divergence_penalty",
    "no_qstruct_term",
}

REQUIRED_RESULT_FILES = {
    "run_config.json",
    "raw_predictions.jsonl",
    "routing_decisions.jsonl",
    "per_seed_results.json",
    "per_dataset_results.json",
    "main_results_table.json",
    "cost_utility_tradeoff_table.json",
    "ablation_table.json",
    "latency_tokens_table.json",
    "difficulty_breakdown_table.json",
    "routing_analysis.json",
    "simple_case_degradation.json",
    "calibration_reliability.json",
    "bootstrap_ci.json",
    "failure_cases.jsonl",
    "environment_report.json",
}

FORBIDDEN_DATASET_TOKENS = {"spider", "samsum", "gsm8k", "openai/gsm8k"}
ALLOW_EMPTY_RESULT_FILES = {"failure_cases.jsonl"}


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _nonnull_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _nonnull_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _file_nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _file_exists(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _dataset_names(summary: dict, run_config: dict, manifest: dict) -> set[str]:
    names: set[str] = set()
    for row in _nonnull_list(summary.get("datasets")):
        if isinstance(row, dict):
            names.add(_norm(row.get("name")))
            names.add(_norm(row.get("id")))
    for row in _nonnull_list(run_config.get("targets")):
        if isinstance(row, dict):
            names.add(_norm(row.get("name")))
            names.add(_norm(row.get("hf_dataset")))
    for row in _nonnull_list(manifest.get("datasets")):
        if isinstance(row, dict):
            names.add(_norm(row.get("name")))
            names.add(_norm(row.get("id")))
    return {name for name in names if name}


def _method_names(summary: dict, run_config: dict, manifest: dict) -> set[str]:
    names = set(_nonnull_dict(summary.get("per_method")).keys())
    names.update(str(item) for item in _nonnull_list(run_config.get("methods")))
    names.update(str(item) for item in _nonnull_list(manifest.get("methods")))
    return {name for name in names if str(name).strip()}


def _ablation_names(summary: dict, run_config: dict) -> set[str]:
    names = set(str(item) for item in _nonnull_list(run_config.get("ablations")))
    for row in _nonnull_list(summary.get("ablation_table") or summary.get("ablation_results")):
        if isinstance(row, dict):
            names.add(str(row.get("ablation") or ""))
            names.add(str(row.get("method") or "").replace("CGGR/", ""))
    return {name for name in names if name.strip()}


def _iter_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return []
    return rows


def _row_text(row: dict, key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value)


def _routing_diagnostics(results: Path) -> tuple[list[str], list[str], dict]:
    blockers: list[str] = []
    warnings: list[str] = []
    decisions = _iter_jsonl(results / "routing_decisions.jsonl")
    raw_rows = _iter_jsonl(results / "raw_predictions.jsonl")
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    caps: dict[tuple[str, str, str, str], int] = {}
    for row in decisions:
        method = _row_text(row, "method")
        dataset = _row_text(row, "dataset")
        seed = _row_text(row, "seed")
        key = (method, dataset, seed)
        group = groups.setdefault(key, {"total": 0, "routed": 0})
        group["total"] += 1
        if bool(row.get("routed_to_deliberation")):
            group["routed"] += 1
        example_id = _row_text(row, "example_id")
        try:
            cap = int(row.get("max_new_tokens"))
        except (TypeError, ValueError):
            cap = 0
        if example_id and cap > 0:
            caps[(method, dataset, seed, example_id)] = cap

    cap_groups: dict[tuple[str, str, str], dict[str, int]] = {}
    for row in raw_rows:
        method = _row_text(row, "method")
        dataset = _row_text(row, "dataset")
        seed = _row_text(row, "seed")
        example_id = _row_text(row, "example_id")
        cap = caps.get((method, dataset, seed, example_id))
        if not cap:
            continue
        group = cap_groups.setdefault((method, dataset, seed), {"total": 0, "hits": 0})
        group["total"] += 1
        try:
            new_tokens = int(row.get("new_tokens"))
        except (TypeError, ValueError):
            new_tokens = 0
        if new_tokens >= cap:
            group["hits"] += 1

    routing_summary: dict[str, dict[str, float]] = {}
    cggr_total = 0
    cggr_routed = 0
    for (method, dataset, seed), group in sorted(groups.items()):
        method_l = method.lower()
        total = int(group.get("total") or 0)
        routed = int(group.get("routed") or 0)
        if total <= 0:
            continue
        label = f"{method}|{dataset}|seed={seed}"
        route_rate = routed / total
        routing_summary[label] = {"total": total, "routed": routed, "route_rate": route_rate}
        if method == "CGGR":
            cggr_total += total
            cggr_routed += routed
        if any(token in method_l for token in ("cggr", "gate", "routing")) and total >= 20 and routed in {0, total}:
            warnings.append(f"routing rate is {route_rate:.3f} for {label}")

    if cggr_total >= 20 and cggr_routed == 0:
        blockers.append("CGGR routing collapsed to zero deliberation across all audited cells")

    for (method, dataset, seed), group in sorted(cap_groups.items()):
        total = int(group.get("total") or 0)
        hits = int(group.get("hits") or 0)
        if total < 20:
            continue
        hit_rate = hits / total
        if hit_rate >= 0.50:
            warnings.append(
                f"token cap hit rate is {hit_rate:.3f} for {method}|{dataset}|seed={seed}; inspect truncation"
            )

    return blockers, warnings, routing_summary


def _generation_failure_diagnostics(results: Path) -> tuple[list[str], dict]:
    rows = _iter_jsonl(results / "failure_cases.jsonl")
    generation_failures = [
        row for row in rows if str(row.get("stage") or "").strip().lower() == "generation_or_scoring"
    ]
    diagnostics = {
        "failure_cases_rows": len(rows),
        "generation_or_scoring_failures": len(generation_failures),
    }
    if not generation_failures:
        return [], diagnostics
    sample = generation_failures[0]
    return [
        "generation_or_scoring failures present in failure_cases.jsonl; "
        f"sample={sample.get('error_type') or ''}:{sample.get('error_repr') or sample.get('error') or ''}"
    ], diagnostics


def _prediction_diagnostics(raw_rows: list[dict]) -> tuple[list[str], list[str], dict]:
    blockers: list[str] = []
    warnings: list[str] = []
    total = len(raw_rows)
    empty_predictions = 0
    zero_token_rows = 0
    method_groups: dict[str, dict[str, int]] = {}
    for row in raw_rows:
        method = _row_text(row, "method")
        group = method_groups.setdefault(method, {"total": 0, "empty_predictions": 0, "zero_token_rows": 0})
        group["total"] += 1
        prediction = _row_text(row, "prediction").strip()
        if not prediction:
            empty_predictions += 1
            group["empty_predictions"] += 1
        try:
            new_tokens = int(row.get("new_tokens"))
        except (TypeError, ValueError):
            new_tokens = 0
        if new_tokens <= 0:
            zero_token_rows += 1
            group["zero_token_rows"] += 1

    diagnostics = {
        "raw_prediction_rows": total,
        "empty_predictions": empty_predictions,
        "zero_token_rows": zero_token_rows,
        "by_method": method_groups,
    }
    if total <= 0:
        return blockers, warnings, diagnostics
    empty_rate = empty_predictions / total
    zero_token_rate = zero_token_rows / total
    if empty_predictions:
        blockers.append(f"empty decoded predictions present: {empty_predictions}/{total} ({empty_rate:.3f})")
    if zero_token_rows:
        blockers.append(f"zero-token generations present: {zero_token_rows}/{total} ({zero_token_rate:.3f})")
    for method, group in sorted(method_groups.items()):
        count = int(group.get("total") or 0)
        if count < 20:
            continue
        method_empty = int(group.get("empty_predictions") or 0)
        method_zero = int(group.get("zero_token_rows") or 0)
        if method_empty == count:
            blockers.append(f"all predictions are empty for method: {method}")
        elif method_empty / count >= 0.05:
            warnings.append(f"empty prediction rate is {method_empty / count:.3f} for method: {method}")
        if method_zero == count:
            blockers.append(f"all generations have zero new tokens for method: {method}")
        elif method_zero / count >= 0.05:
            warnings.append(f"zero-token generation rate is {method_zero / count:.3f} for method: {method}")
    return blockers, warnings, diagnostics


def _coverage_diagnostics(
    raw_rows: list[dict],
    run_config: dict,
    *,
    require_full: bool,
    required_methods: set[str],
) -> tuple[list[str], dict]:
    if not require_full:
        return [], {}
    seed_raw = run_config.get("seed_values")
    if isinstance(seed_raw, list) and seed_raw:
        seed_values = seed_raw
    else:
        try:
            seed_values = list(range(int(run_config.get("seeds") or 0)))
        except (TypeError, ValueError):
            seed_values = []
    datasets = [
        str(row.get("name") or row.get("hf_dataset") or "").strip()
        for row in _nonnull_list(run_config.get("targets"))
        if isinstance(row, dict) and str(row.get("name") or row.get("hf_dataset") or "").strip()
    ]
    try:
        examples = int(run_config.get("max_examples_per_dataset_seed") or 0)
    except (TypeError, ValueError):
        examples = 0
    expected_main_rows = len(required_methods) * len(datasets) * len(seed_values) * examples
    counts: dict[tuple[str, str, str], int] = {}
    example_counts: dict[tuple[str, str, str, str], int] = {}
    for row in raw_rows:
        key = (_row_text(row, "method"), _row_text(row, "dataset"), _row_text(row, "seed"))
        counts[key] = counts.get(key, 0) + 1
        example_id = _row_text(row, "example_id")
        if example_id:
            example_key = key + (example_id,)
            example_counts[example_key] = example_counts.get(example_key, 0) + 1
    diagnostics = {
        "datasets": datasets,
        "seed_values": seed_values,
        "examples_per_dataset_seed": examples,
        "expected_main_method_rows": expected_main_rows,
        "observed_raw_rows": len(raw_rows),
    }
    blockers = []
    if expected_main_rows <= 0:
        blockers.append("cannot verify full raw coverage from run_config targets/seed_values/max_examples_per_dataset_seed")
        return blockers, diagnostics
    duplicates = [key for key, count in example_counts.items() if count > 1]
    if duplicates:
        method, dataset, seed, example_id = duplicates[0]
        blockers.append(
            "duplicate raw prediction rows present: "
            f"method={method} dataset={dataset} seed={seed} example_id={example_id}"
        )
    if len(raw_rows) < expected_main_rows:
        blockers.append(f"raw_predictions rows below full main-method gate: {len(raw_rows)}/{expected_main_rows}")
    for method in sorted(required_methods):
        for dataset in datasets:
            for seed in seed_values:
                count = counts.get((method, dataset, str(seed)), 0)
                if count < examples:
                    blockers.append(
                        f"method/dataset/seed cell below paper gate: method={method} dataset={dataset} seed={seed} count={count}/{examples}"
                    )
                    if len(blockers) >= 20:
                        blockers.append("additional missing or incomplete cells omitted")
                        return blockers, diagnostics
                elif count > examples:
                    blockers.append(
                        f"method/dataset/seed cell above paper gate, indicating duplicate or overlapping rows: "
                        f"method={method} dataset={dataset} seed={seed} count={count}/{examples}"
                    )
                    if len(blockers) >= 20:
                        blockers.append("additional duplicate or overlapping cells omitted")
                        return blockers, diagnostics
    return blockers, diagnostics


def _stats_diagnostics(summary: dict, results: Path, *, required_methods: set[str]) -> tuple[list[str], dict]:
    blockers = []
    bootstrap = _nonnull_dict(_load_json(results / "bootstrap_ci.json"))
    per_method_std = _nonnull_dict(summary.get("per_method_std"))
    simple_case = _nonnull_dict(_load_json(results / "simple_case_degradation.json"))
    calibration = _nonnull_list(_load_json(results / "calibration_reliability.json"))
    diagnostics = {
        "bootstrap_ci": bootstrap,
        "per_method_std_methods": sorted(per_method_std),
        "simple_case_degradation": simple_case,
        "calibration_reliability_rows": len(calibration),
    }
    for key in ("candidate_method", "baseline_method", "candidate_ci95", "baseline_ci95", "paired_permutation_p"):
        if key not in bootstrap:
            blockers.append(f"bootstrap_ci missing required field: {key}")
    for key in ("candidate_ci95", "baseline_ci95"):
        value = bootstrap.get(key)
        if not isinstance(value, list) or len(value) != 2:
            blockers.append(f"bootstrap_ci field must be length-2 list: {key}")
    try:
        p_value = float(bootstrap.get("paired_permutation_p"))
        if p_value < 0.0 or p_value > 1.0:
            blockers.append("paired_permutation_p outside [0, 1]")
    except (TypeError, ValueError):
        blockers.append("paired_permutation_p is not numeric")
    for method in sorted(required_methods):
        if method not in per_method_std:
            blockers.append(f"per_method_std missing required method: {method}")
            break
    for key in ("baseline_accuracy", "candidate_accuracy", "degradation", "candidate_route_rate"):
        if simple_case.get(key) is None:
            blockers.append(f"simple_case_degradation missing numeric field: {key}")
    if not calibration:
        blockers.append("calibration_reliability.json has no rows")
    return blockers, diagnostics


def _is_sharded_artifact(summary: dict, run_config: dict, manifest: dict, runner_manifest: dict) -> bool:
    shard_axes = run_config.get("shard_axes") if isinstance(run_config.get("shard_axes"), dict) else {}
    return bool(
        summary.get("sharded_run")
        or run_config.get("sharded_run")
        or any(bool(value) for value in shard_axes.values())
        or manifest.get("sharded_run")
        or runner_manifest.get("sharded_run")
    )


def audit(workdir: Path, *, require_full: bool, require_top_venue_baselines: bool = False) -> dict:
    results = workdir / "results"
    blockers: list[str] = []
    warnings: list[str] = []
    required_methods = set(REQUIRED_METHODS)
    if require_top_venue_baselines:
        required_methods.update(TOP_VENUE_BASELINE_METHODS)

    if not workdir.exists():
        return {"ok": False, "blockers": [f"workdir does not exist: {workdir}"], "warnings": []}
    if not results.exists():
        return {"ok": False, "blockers": [f"results dir does not exist: {results}"], "warnings": []}

    summary = _nonnull_dict(_load_json(results / "benchmark_summary.json"))
    run_config = _nonnull_dict(_load_json(results / "run_config.json"))
    manifest = _nonnull_dict(_load_json(results / "benchmark_artifact_manifest.json"))
    runner_manifest = _nonnull_dict(_load_json(results / "artifact_manifest.json"))
    if not manifest and runner_manifest:
        manifest = runner_manifest

    if not summary:
        blockers.append("missing or invalid results/benchmark_summary.json")
    if not run_config:
        blockers.append("missing or invalid results/run_config.json")
    if not manifest:
        blockers.append("missing benchmark_artifact_manifest.json or artifact_manifest.json")

    for filename in sorted(REQUIRED_RESULT_FILES):
        path = results / filename
        if filename in ALLOW_EMPTY_RESULT_FILES:
            if not _file_exists(path):
                blockers.append(f"missing results/{filename}")
            continue
        if not _file_nonempty(path):
            blockers.append(f"missing or empty results/{filename}")

    full_completed = bool(summary.get("full_benchmark_completed") or manifest.get("full_benchmark_completed"))
    if require_full and not full_completed:
        blockers.append("full_benchmark_completed is not true")

    dataset_names = _dataset_names(summary, run_config, manifest)
    for required in sorted(REQUIRED_DATASETS):
        if not any(required == name or required in name for name in dataset_names):
            blockers.append(f"required dataset missing: {required}")
    for name in sorted(dataset_names):
        if any(token in name for token in FORBIDDEN_DATASET_TOKENS):
            blockers.append(f"forbidden or unsupported dataset alias present: {name}")

    methods = _method_names(summary, run_config, manifest)
    for method in sorted(required_methods):
        if method not in methods:
            blockers.append(f"required method missing: {method}")

    ablations = _ablation_names(summary, run_config)
    for ablation in sorted(REQUIRED_ABLATIONS):
        if ablation not in ablations:
            blockers.append(f"required ablation missing: {ablation}")

    seeds = summary.get("num_seeds") or _nonnull_dict(summary.get("budget")).get("seeds") or run_config.get("seeds")
    try:
        seed_count = int(seeds)
    except (TypeError, ValueError):
        seed_count = 0
    if seed_count < 5:
        blockers.append(f"seed count below paper gate: {seed_count}")

    max_examples = (
        _nonnull_dict(summary.get("budget")).get("max_examples_per_dataset_seed")
        or run_config.get("max_examples_per_dataset_seed")
    )
    try:
        example_count = int(max_examples)
    except (TypeError, ValueError):
        example_count = 0
    if example_count < 128:
        blockers.append(f"examples per dataset/seed below paper gate: {example_count}")

    raw_path = results / "raw_predictions.jsonl"
    raw_rows: list[dict] = []
    if raw_path.exists():
        with raw_path.open("rb") as handle:
            raw_lines = sum(1 for _ in handle)
        if raw_lines <= 0:
            blockers.append("raw_predictions.jsonl has no rows")
        raw_rows = _iter_jsonl(raw_path)
    else:
        raw_lines = 0

    if _is_sharded_artifact(summary, run_config, manifest, runner_manifest):
        warnings.append("artifact is marked sharded_run; it must be merged and re-audited before manuscript claims")
        if require_full:
            blockers.append("sharded_run artifact cannot satisfy require-full gate by itself")

    routing_blockers, routing_warnings, routing_summary = _routing_diagnostics(results)
    blockers.extend(routing_blockers)
    warnings.extend(routing_warnings)
    generation_blockers, failure_diagnostics = _generation_failure_diagnostics(results)
    blockers.extend(generation_blockers)
    prediction_blockers, prediction_warnings, prediction_summary = _prediction_diagnostics(raw_rows)
    blockers.extend(prediction_blockers)
    warnings.extend(prediction_warnings)
    coverage_blockers, coverage_summary = _coverage_diagnostics(
        raw_rows,
        run_config,
        require_full=require_full,
        required_methods=required_methods,
    )
    blockers.extend(coverage_blockers)
    stats_blockers, stats_summary = _stats_diagnostics(summary, results, required_methods=required_methods)
    blockers.extend(stats_blockers)
    blockers = list(dict.fromkeys(blockers))
    warnings = list(dict.fromkeys(warnings))

    return {
        "ok": not blockers,
        "workdir": str(workdir),
        "results_dir": str(results),
        "require_full": require_full,
        "require_top_venue_baselines": require_top_venue_baselines,
        "full_benchmark_completed": full_completed,
        "raw_predictions_lines": raw_lines,
        "datasets_seen": sorted(dataset_names),
        "methods_seen": sorted(methods),
        "ablations_seen": sorted(ablations),
        "routing_diagnostics": routing_summary,
        "failure_diagnostics": failure_diagnostics,
        "prediction_diagnostics": prediction_summary,
        "coverage_diagnostics": coverage_summary,
        "stats_diagnostics": stats_summary,
        "blockers": blockers,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument("--require-top-venue-baselines", action="store_true")
    args = parser.parse_args()
    result = audit(
        args.workdir,
        require_full=args.require_full,
        require_top_venue_baselines=args.require_top_venue_baselines,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
