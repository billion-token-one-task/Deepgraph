"""Materialize paper-grade benchmark tables from run artifacts (predictions, manifest)."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from agents.metric_parser import persist_main_results_table


def _parse_score(row: dict, metric_name: str) -> float | None:
    for key in (metric_name, "primary_score", "metric_value", "exact", "score"):
        if key in row:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return None


def _load_predictions(results_dir: Path) -> list[dict]:
    path = results_dir / "raw_predictions.jsonl"
    if not path.is_file():
        return []
    rows: list[dict] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows


def _short_dataset(name: str) -> str:
    mapping = {
        "GSM8K": "GSM8K",
        "MuSiQue-Ans": "MuSiQue",
        "StrategyQA": "StrategyQA",
        "2WikiMultihopQA": "2Wiki",
        "Stress Test Split: Simple-vs-Hard Counterfactual Partition": "Stress",
    }
    return mapping.get(name, name[:12])


def materialize_deep_benchmark_artifacts(
    results_dir: Path,
    *,
    publication_contract: dict | None = None,
    metric_name: str = "primary_score",
    min_lines: int = 100,
) -> dict[str, Any]:
    """Build main/per-dataset/seed/ablation JSON tables from ``raw_predictions.jsonl``."""
    results_dir = Path(results_dir)
    rows = _load_predictions(results_dir)
    report: dict[str, Any] = {
        "results_dir": str(results_dir),
        "prediction_lines": len(rows),
        "artifacts_written": [],
    }
    if len(rows) < min_lines:
        report["error"] = "insufficient_predictions"
        return report

    contract = publication_contract if isinstance(publication_contract, dict) else {}
    required_ablations = [
        str(x) for x in (contract.get("required_ablations") or []) if str(x).strip()
    ]

    by_method: dict[str, list[float]] = defaultdict(list)
    by_method_dataset: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_method_seed: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    ablation_methods: dict[str, list[float]] = defaultdict(list)

    datasets: set[str] = set()
    seeds: set[int] = set()
    for row in rows:
        method = str(row.get("method") or "unknown")
        dataset = str(row.get("dataset") or "unknown")
        score = _parse_score(row, metric_name)
        if score is None:
            continue
        datasets.add(dataset)
        by_method[method].append(score)
        by_method_dataset[method][dataset].append(score)
        seed_raw = row.get("seed")
        if seed_raw is not None:
            try:
                seed = int(seed_raw)
                seeds.add(seed)
                by_method_seed[method][seed].append(score)
            except (TypeError, ValueError):
                pass
        low = method.lower()
        if any(
            token in low
            for token in ("ablation", "disable_", "remove_", "no_", "compute_matched")
        ):
            ablation_methods[method].append(score)

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")

    per_method = {
        m: {metric_name: _mean(v), "n": len(v)} for m, v in sorted(by_method.items()) if v
    }
    per_dataset: dict[str, dict[str, float]] = {}
    for method, ds_map in sorted(by_method_dataset.items()):
        per_dataset[method] = {
            _short_dataset(ds): _mean(vals) for ds, vals in sorted(ds_map.items()) if vals
        }

    seed_variance: dict[str, dict[str, Any]] = {}
    for method, seed_map in sorted(by_method_seed.items()):
        seed_means = [_mean(vals) for vals in seed_map.values() if vals]
        if not seed_means:
            continue
        seed_variance[method] = {
            "mean": statistics.mean(seed_means),
            "std": statistics.pstdev(seed_means) if len(seed_means) > 1 else 0.0,
            "n_seeds": len(seed_means),
            "per_seed": {str(s): _mean(vals) for s, vals in sorted(seed_map.items()) if vals},
        }

    full_method = None
    for name in per_method:
        if "cpg" in name.lower() or "contrastive perceptual" in name.lower():
            full_method = name
            break
    full_score = per_method.get(full_method or "", {}).get(metric_name) if full_method else None

    ablation_table: dict[str, dict[str, Any]] = {}
    executed_ablation_names: list[str] = []
    for name, vals in sorted(ablation_methods.items()):
        if not vals:
            continue
        score = _mean(vals)
        delta = None if full_score is None else score - float(full_score)
        ablation_table[name] = {
            metric_name: score,
            "delta_vs_full": delta,
            "n": len(vals),
            "executed": True,
        }
        executed_ablation_names.append(name)

    for variant in required_ablations:
        if variant in ablation_table:
            continue
        ablation_table[variant] = {
            metric_name: None,
            "delta_vs_full": None,
            "n": 0,
            "executed": False,
            "status": "not_executed_in_run",
        }

    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "benchmark_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}

    candidate = full_method or summary.get("candidate_method")
    summary.update(
        {
            "primary_metric": metric_name,
            "metric_name": metric_name,
            "candidate_method": candidate,
            "per_method": per_method,
            "per_dataset": per_dataset,
            "datasets": sorted(datasets),
            "num_seeds": len(seeds) or int(summary.get("num_seeds") or 0),
            "seed_variance": seed_variance,
            "ablations": ablation_table,
            "ablation_executed_count": len(executed_ablation_names),
            "ablation_required_count": len(required_ablations),
            "deep_artifacts_materialized": True,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report["artifacts_written"].append(str(summary_path))

    main_path = persist_main_results_table(results_dir, summary)
    if main_path:
        report["artifacts_written"].append(str(main_path))

    per_ds_path = results_dir / "per_dataset_results.json"
    per_ds_path.write_text(json.dumps(per_dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    report["artifacts_written"].append(str(per_ds_path))

    seed_path = results_dir / "seed_variance_table.json"
    seed_path.write_text(json.dumps(seed_variance, indent=2, ensure_ascii=False), encoding="utf-8")
    report["artifacts_written"].append(str(seed_path))

    ablation_path = results_dir / "ablation_table.json"
    ablation_path.write_text(json.dumps(ablation_table, indent=2, ensure_ascii=False), encoding="utf-8")
    report["artifacts_written"].append(str(ablation_path))

    report["per_method_count"] = len(per_method)
    report["dataset_count"] = len(datasets)
    report["seed_count"] = len(seeds)
    report["ablation_executed_count"] = len(executed_ablation_names)
    report["ablation_required_count"] = len(required_ablations)
    report["ok"] = bool(per_method) and len(datasets) >= 2
    return report
