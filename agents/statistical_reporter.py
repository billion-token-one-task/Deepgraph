"""Build statistical reports from benchmark result artifacts."""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

from agents.artifact_manager import artifact_path, ensure_artifact_dirs, record_artifact
from db import database as db


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def _bootstrap_ci(values: list[float], n_resamples: int = 1000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(13)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(_mean(sample))
    means.sort()
    low_idx = int(0.025 * (len(means) - 1))
    high_idx = int(0.975 * (len(means) - 1))
    return means[low_idx], means[high_idx]


def _sign_test_p_value(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _metric_rows(rows: list[dict], primary_metric: str) -> list[dict]:
    filtered = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        metrics = row.get("metrics") or {}
        if primary_metric not in metrics:
            continue
        filtered.append(row)
    return filtered


def _available_metric_names(rows: list[dict]) -> list[str]:
    names = set()
    for row in rows:
        if row.get("status") != "ok":
            continue
        for name, value in (row.get("metrics") or {}).items():
            if name in {"fairness_penalty", "cost_limit", "safety_penalty"}:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                names.add(name)
    priority = [
        "safe_return",
        "reward",
        "cost",
        "constraint_violation",
        "lp_flow_residual",
        "lp_cost_residual",
        "lp_objective_gap",
        "policy_entropy",
        "runtime_seconds",
    ]
    return sorted(names, key=lambda name: (priority.index(name) if name in priority else len(priority), name))


def _analysis_rows(rows: list[dict], analysis_type: str) -> list[dict]:
    return [
        row for row in rows
        if (row.get("analysis_type") or "main") == analysis_type
    ]


def _summary_rows(rows: list[dict], primary_metric: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = (row.get("ablation") or "", row.get("dataset"), row.get("method"))
        groups[key].append(float(row["metrics"][primary_metric]))

    summary = []
    for (ablation, dataset, method), values in sorted(groups.items()):
        ci_low, ci_high = _bootstrap_ci(values)
        item = {
            "dataset": dataset,
            "method": method,
            "metric": primary_metric,
            "mean": _mean(values),
            "std": _sample_std(values),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n": len(values),
        }
        if ablation:
            item["ablation"] = ablation
        summary.append(item)
    return summary


def _best_method(summary: list[dict], direction: str) -> str | None:
    totals = defaultdict(list)
    for item in summary:
        totals[item["method"]].append(float(item["mean"]))
    if not totals:
        return None
    method_means = {
        method: _mean(values)
        for method, values in totals.items()
    }
    reverse = direction != "lower"
    return sorted(method_means, key=method_means.get, reverse=reverse)[0]


def _summary_methods(summary: list[dict]) -> set[str]:
    return {str(item["method"]) for item in summary if item.get("method")}


def _comparisons(rows: list[dict], primary_metric: str, baseline_method: str,
                 best_method: str | None, direction: str) -> list[dict]:
    if not best_method or baseline_method == best_method:
        return []

    by_key = {}
    for row in rows:
        key = (row.get("dataset"), row.get("seed"), row.get("method"))
        by_key[key] = float(row["metrics"][primary_metric])

    comparisons = []
    datasets = sorted({row.get("dataset") for row in rows})
    for dataset in datasets:
        wins = losses = ties = 0
        deltas = []
        seeds = sorted({row.get("seed") for row in rows if row.get("dataset") == dataset})
        for seed in seeds:
            base_key = (dataset, seed, baseline_method)
            cand_key = (dataset, seed, best_method)
            if base_key not in by_key or cand_key not in by_key:
                continue
            raw_delta = by_key[cand_key] - by_key[base_key]
            delta = raw_delta if direction != "lower" else -raw_delta
            deltas.append(delta)
            if delta > 0:
                wins += 1
            elif delta < 0:
                losses += 1
            else:
                ties += 1
        if deltas:
            comparisons.append({
                "dataset": dataset,
                "baseline": baseline_method,
                "candidate": best_method,
                "metric": primary_metric,
                "mean_delta": _mean(deltas),
                "paired_sign_test_p": _sign_test_p_value(wins, losses),
                "wins": wins,
                "losses": losses,
                "ties": ties,
            })
    return comparisons


def _pairwise_comparisons(rows: list[dict], primary_metric: str,
                          best_method: str | None, direction: str) -> list[dict]:
    if not best_method:
        return []
    methods = sorted({row.get("method") for row in rows if row.get("method")})
    comparisons = []
    for method in methods:
        if method == best_method:
            continue
        comparisons.extend(_comparisons(rows, primary_metric, method, best_method, direction))
    return comparisons


def _metric_summaries(rows: list[dict], primary_metric: str) -> list[dict]:
    metric_names = _available_metric_names(rows)
    if primary_metric in metric_names:
        metric_names.remove(primary_metric)
        metric_names.insert(0, primary_metric)
    summaries = []
    for metric_name in metric_names:
        summaries.extend(_summary_rows(_metric_rows(rows, metric_name), metric_name))
    return summaries


def _aggregate_metric_summaries(rows: list[dict], primary_metric: str) -> list[dict]:
    metric_names = _available_metric_names(rows)
    if primary_metric in metric_names:
        metric_names.remove(primary_metric)
        metric_names.insert(0, primary_metric)

    summaries = []
    for metric_name in metric_names:
        groups = defaultdict(list)
        for row in _metric_rows(rows, metric_name):
            groups[row.get("method")].append(float(row["metrics"][metric_name]))
        for method, values in sorted(groups.items()):
            ci_low, ci_high = _bootstrap_ci(values)
            summaries.append({
                "method": method,
                "metric": metric_name,
                "mean": _mean(values),
                "std": _sample_std(values),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": len(values),
            })
    return summaries


def build_statistical_report(rows: list[dict], primary_metric: str,
                             direction: str, baseline_method: str,
                             candidate_method: str | None = None) -> dict:
    """Create a statistical report from benchmark result rows."""
    main_rows = _analysis_rows(rows, "main")
    filtered = _metric_rows(main_rows, primary_metric)
    ablation_filtered = _metric_rows(_analysis_rows(rows, "ablation"), primary_metric)
    summary = _summary_rows(filtered, primary_metric)
    ablation_summary = _summary_rows(ablation_filtered, primary_metric)
    for item in ablation_summary:
        if item.get("ablation"):
            continue
        matching = [
            row for row in ablation_filtered
            if row.get("dataset") == item.get("dataset") and row.get("method") == item.get("method")
        ]
        names = sorted({str(row.get("ablation") or "") for row in matching if row.get("ablation")})
        item["ablation"] = names[0] if len(names) == 1 else ", ".join(names)
    absolute_best_method = _best_method(summary, direction)
    available_methods = _summary_methods(summary)
    best_method = candidate_method if candidate_method in available_methods else absolute_best_method
    comparisons = _comparisons(filtered, primary_metric, baseline_method, best_method, direction)
    pairwise_comparisons = _pairwise_comparisons(filtered, primary_metric, best_method, direction)
    return {
        "schema_version": 1,
        "primary_metric": primary_metric,
        "metric_direction": direction,
        "baseline_method": baseline_method,
        "best_method": best_method,
        "candidate_method": best_method,
        "absolute_best_method": absolute_best_method,
        "summary": summary,
        "metric_summaries": _metric_summaries(main_rows, primary_metric),
        "aggregate_metric_summaries": _aggregate_metric_summaries(main_rows, primary_metric),
        "ablation_summary": ablation_summary,
        "comparisons": comparisons,
        "pairwise_comparisons": pairwise_comparisons,
    }


def _main_results_table(report: dict) -> str:
    lines = [
        "| Dataset | Method | Metric | Mean | Std | 95% CI | N |",
        "| --- | --- | --- | ---: | ---: | --- | ---: |",
    ]
    for row in report.get("summary", []):
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['metric']} | "
            f"{row['mean']:.6f} | {row['std']:.6f} | "
            f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}] | {row['n']} |"
        )
    if report.get("ablation_summary"):
        lines.extend([
            "",
            "## Ablation And Sensitivity",
            "",
            "| Ablation | Dataset | Method | Metric | Mean | Std | 95% CI | N |",
            "| --- | --- | --- | --- | ---: | ---: | --- | ---: |",
        ])
        for row in report.get("ablation_summary", []):
            lines.append(
                f"| {row.get('ablation', '')} | {row['dataset']} | {row['method']} | {row['metric']} | "
                f"{row['mean']:.6f} | {row['std']:.6f} | "
                f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}] | {row['n']} |"
            )
    secondary = [
        row for row in report.get("metric_summaries", [])
        if row.get("metric") != report.get("primary_metric")
    ]
    if secondary:
        lines.extend([
            "",
            "## Secondary Metrics",
            "",
            "| Dataset | Method | Metric | Mean | Std | 95% CI | N |",
            "| --- | --- | --- | ---: | ---: | --- | ---: |",
        ])
        for row in secondary:
            lines.append(
                f"| {row['dataset']} | {row['method']} | {row['metric']} | "
                f"{row['mean']:.6f} | {row['std']:.6f} | "
                f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}] | {row['n']} |"
            )
    if report.get("pairwise_comparisons"):
        lines.extend([
            "",
            "## Pairwise Baseline Comparisons",
            "",
            "| Dataset | Baseline | Candidate | Metric | Mean Delta | p | Wins/Losses/Ties |",
            "| --- | --- | --- | --- | ---: | ---: | --- |",
        ])
        for row in report.get("pairwise_comparisons", []):
            lines.append(
                f"| {row['dataset']} | {row['baseline']} | {row['candidate']} | {row['metric']} | "
                f"{row['mean_delta']:.6f} | {row['paired_sign_test_p']:.6f} | "
                f"{row['wins']}/{row['losses']}/{row['ties']} |"
            )
    return "\n".join(lines) + "\n"


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def write_statistical_report(run_id: int) -> dict:
    """Read benchmark results for a run and write statistical report artifacts."""
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not run:
        return {"status": "error", "reason": "run_not_found", "run_id": run_id}

    workdir = Path(run.get("workdir") or "")
    benchmark_results = _load_json(
        workdir / "artifacts" / "results" / "benchmark_results.json",
        {},
    )
    rows = benchmark_results.get("rows") or []
    if not rows:
        return {"status": "error", "reason": "benchmark_results_not_found", "run_id": run_id}

    config = _load_json(workdir / "benchmark_config.json", {})
    primary_metric = config.get("primary_metric") or run.get("baseline_metric_name") or "metric"
    direction = config.get("metric_direction") or "higher"
    methods = config.get("methods") or []
    baseline_method = config.get("baseline_method") or (methods[0] if methods else "baseline")
    candidate_method = config.get("candidate_method")
    report = build_statistical_report(rows, primary_metric, direction, baseline_method, candidate_method)
    report["run_id"] = run_id
    if "fairness_penalty" in config:
        report["fairness_penalty"] = config.get("fairness_penalty")

    ensure_artifact_dirs(workdir)
    report_path = artifact_path(workdir, "artifacts/results/statistical_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    record_artifact(workdir, run_id, "statistical_report", report_path, {
        "primary_metric": primary_metric,
        "best_method": report.get("best_method"),
    })

    table_path = artifact_path(workdir, "artifacts/tables/main_results.md")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(_main_results_table(report), encoding="utf-8")
    record_artifact(workdir, run_id, "result_table", table_path, {
        "primary_metric": primary_metric,
    })

    return {
        "status": "complete",
        "run_id": run_id,
        "best_method": report.get("best_method"),
        "summary_rows": len(report.get("summary", [])),
        "comparison_rows": len(report.get("comparisons", [])),
    }
