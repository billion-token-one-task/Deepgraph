"""Harness for finite CMDP safe RL benchmark capability."""
from __future__ import annotations

import time

from benchmarks.safe_rl_cmdp.envs import make_cmdp
from benchmarks.safe_rl_cmdp.solvers import (
    occupancy_lp_optimal,
    deterministic_feasible_best,
    lagrangian_grid_best,
    lagrangian_policy,
    occupancy_enumeration,
    preference_cone_policy,
    reward_only_policy,
)


def _run_method(method: str, env, config: dict) -> dict:
    safety_penalty = float(config.get("safety_penalty", 6.0))
    if method == "reward_only":
        return reward_only_policy(env, safety_penalty=safety_penalty)
    if method == "occupancy_lp_optimal":
        return occupancy_lp_optimal(env, safety_penalty=safety_penalty)
    if method == "occupancy_enumeration":
        return occupancy_enumeration(env, safety_penalty=safety_penalty)
    if method == "deterministic_feasible_best":
        return deterministic_feasible_best(env, safety_penalty=safety_penalty)
    if method == "preference_cone_policy":
        return preference_cone_policy(env, config.get("preference_penalties"), safety_penalty=safety_penalty)
    if method == "lagrangian_grid_best":
        return lagrangian_grid_best(env, config.get("preference_penalties"), safety_penalty=safety_penalty)
    prefix = "lagrangian_penalty_"
    if method.startswith(prefix):
        try:
            penalty = float(method[len(prefix):])
        except ValueError as exc:
            raise ValueError(f"invalid Lagrangian penalty method: {method}") from exc
        return lagrangian_policy(env, penalty, safety_penalty=safety_penalty)
    raise ValueError(f"unknown safe RL method: {method}")


def run_safe_rl_benchmark(config: dict) -> dict:
    """Run finite CMDP datasets, methods, and seeds."""
    datasets = config.get("datasets") or ["risky_shortcut"]
    methods = config.get("methods") or ["reward_only"]
    seeds = config.get("seeds") or [0]
    rows = []

    for dataset_name in datasets:
        for seed in seeds:
            try:
                env = make_cmdp(dataset_name, int(seed))
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
                    result = _run_method(method, env, config)
                    metrics = dict(result["metrics"])
                    metrics["runtime_seconds"] = time.time() - start
                    rows.append({
                        "dataset": dataset_name,
                        "seed": int(seed),
                        "method": method,
                        "status": "ok",
                        "metrics": metrics,
                        "policy": result.get("policy"),
                        "policy_type": result.get("policy_type"),
                        "selected_penalty": result.get("selected_penalty"),
                        "selection_protocol": result.get("selection_protocol"),
                        "baseline_role": result.get("baseline_role"),
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
        "capability": "safe_rl_cmdp",
        "rows": rows,
    }
