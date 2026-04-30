"""Harness for the fairness classification benchmark capability."""
from __future__ import annotations

import time

from benchmarks.fairness_classification.datasets import make_dataset
from benchmarks.fairness_classification.methods import fit_predict
from benchmarks.fairness_classification.metrics import compute_metrics


def run_fairness_benchmark(config: dict) -> dict:
    """Run configured fairness classification datasets, methods, and seeds."""
    datasets = config.get("datasets") or ["synthetic_grouped"]
    methods = config.get("methods") or ["logistic_regression"]
    seeds = config.get("seeds") or [0]
    fairness_penalty = float(config.get("fairness_penalty", 0.45))
    rows = []

    for dataset_name in datasets:
        for seed in seeds:
            try:
                dataset = make_dataset(dataset_name, int(seed))
            except Exception as exc:
                for method in methods:
                    rows.append({
                        "dataset": dataset_name,
                        "seed": int(seed),
                        "method": method,
                        "status": "error",
                        "metrics": {},
                        "error": f"dataset error: {exc}",
                    })
                continue

            for method in methods:
                start = time.time()
                try:
                    y_pred = fit_predict(method, dataset)
                    metrics = compute_metrics(
                        dataset.y_test,
                        y_pred,
                        dataset.sensitive_test,
                        fairness_penalty=fairness_penalty,
                    )
                    metrics["runtime_seconds"] = time.time() - start
                    rows.append({
                        "dataset": dataset_name,
                        "seed": int(seed),
                        "method": method,
                        "status": "ok",
                        "metrics": metrics,
                        "error": None,
                    })
                except Exception as exc:
                    rows.append({
                        "dataset": dataset_name,
                        "seed": int(seed),
                        "method": method,
                        "status": "error",
                        "metrics": {"runtime_seconds": time.time() - start},
                        "error": str(exc),
                    })

    return {
        "schema_version": 1,
        "capability": "fairness_classification",
        "rows": rows,
    }
