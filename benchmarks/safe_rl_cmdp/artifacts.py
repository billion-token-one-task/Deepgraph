"""Reproducibility artifacts for finite CMDP benchmark runs."""
from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np

from agents.artifact_manager import artifact_path, record_artifact
from benchmarks.safe_rl_cmdp.envs import FiniteCMDP, make_cmdp
from benchmarks.safe_rl_cmdp.solvers import (
    deterministic_feasible_best,
    lagrangian_policy,
    occupancy_lp_optimal,
)


def _round_nested(value):
    array = np.asarray(value)
    return json.loads(json.dumps(array.tolist()))


def _package_versions() -> dict:
    packages = {}
    for name in ("numpy", "scipy", "flask", "httpx", "pydantic"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return packages


def build_environment_appendix(config: dict) -> dict:
    """Return exact finite-CMDP definitions for configured datasets and seeds."""
    environments = []
    for dataset in config.get("datasets") or []:
        for seed in config.get("seeds") or []:
            env = make_cmdp(str(dataset), int(seed))
            environments.append({
                "dataset": env.name,
                "seed": int(seed),
                "n_states": env.n_states,
                "n_actions": env.n_actions,
                "gamma": float(env.gamma),
                "cost_limit": float(env.cost_limit),
                "start": _round_nested(env.start),
                "transitions": _round_nested(env.transitions),
                "rewards": _round_nested(env.rewards),
                "costs": _round_nested(env.costs),
                "seed_rule": "Transitions are fixed by dataset; rewards and costs use deterministic Gaussian jitter and nonnegative cost clipping in envs.py.",
            })
    return {
        "schema_version": 1,
        "capability": "safe_rl_cmdp",
        "environments": environments,
    }


def _analytic_randomized_one_state_check(safety_penalty: float) -> dict:
    env = FiniteCMDP(
        name="analytic_randomized_one_state",
        seed=0,
        transitions=np.asarray([[[1.0], [1.0]]], dtype=float),
        rewards=np.asarray([[0.0, 1.0]], dtype=float),
        costs=np.asarray([[0.0, 1.0]], dtype=float),
        start=np.asarray([1.0], dtype=float),
        gamma=0.5,
        cost_limit=1.0,
    )
    lp = occupancy_lp_optimal(env, safety_penalty=safety_penalty)
    expected_action_1_probability = 0.5
    expected_reward = 1.0
    expected_cost = 1.0
    observed_action_1_probability = float(lp["policy"][0][1])
    reward_gap = abs(float(lp["metrics"]["reward"]) - expected_reward)
    cost_gap = abs(float(lp["metrics"]["cost"]) - expected_cost)
    probability_gap = abs(observed_action_1_probability - expected_action_1_probability)
    status = "ok" if max(reward_gap, cost_gap, probability_gap) <= 1e-8 else "warning"
    return {
        "name": env.name,
        "status": status,
        "description": "One-state two-action CMDP with gamma=0.5 and cost_limit=1.0; the analytic LP optimum mixes actions 50/50, giving reward=1.0 and cost=1.0.",
        "expected_action_1_probability": expected_action_1_probability,
        "observed_action_1_probability": observed_action_1_probability,
        "expected_reward": expected_reward,
        "observed_reward": lp["metrics"]["reward"],
        "reward_gap": reward_gap,
        "expected_cost": expected_cost,
        "observed_cost": lp["metrics"]["cost"],
        "cost_gap": cost_gap,
        "probability_gap": probability_gap,
    }


def _candidate_reference(env: FiniteCMDP, config: dict, safety_penalty: float) -> dict | None:
    method = str(config.get("candidate_method") or "")
    prefix = "lagrangian_penalty_"
    if not method.startswith(prefix):
        return None
    try:
        penalty = float(method.removeprefix(prefix))
    except ValueError:
        return None
    result = lagrangian_policy(env, penalty, safety_penalty=safety_penalty)
    return {
        "candidate_method": method,
        "candidate_reward": result["metrics"]["reward"],
        "candidate_cost": result["metrics"]["cost"],
        "candidate_safe_return": result["metrics"]["safe_return"],
        "candidate_constraint_violation": result["metrics"]["constraint_violation"],
        "candidate_policy_entropy": result["metrics"]["policy_entropy"],
    }


def build_lp_validation(config: dict) -> dict:
    """Cross-check occupancy LP solutions and deterministic feasible references."""
    rows = []
    cross_checks = []
    status = "ok"
    methods = ["highs", "highs-ds", "highs-ipm"]
    safety_penalty = float(config.get("safety_penalty", 6.0))
    for dataset in config.get("datasets") or []:
        for seed in config.get("seeds") or []:
            env = make_cmdp(str(dataset), int(seed))
            lp = occupancy_lp_optimal(env, safety_penalty=safety_penalty)
            det = deterministic_feasible_best(env, safety_penalty=safety_penalty)
            row = {
                "dataset": env.name,
                "seed": int(seed),
                "lp_reward": lp["metrics"]["reward"],
                "deterministic_feasible_reward": det["metrics"]["reward"],
                "randomization_reward_gap": lp["metrics"]["reward"] - det["metrics"]["reward"],
                "lp_vs_deterministic_feasible_reward_gap": lp["metrics"]["reward"] - det["metrics"]["reward"],
                "lp_constraint_violation": lp["metrics"]["constraint_violation"],
                "lp_policy_entropy": lp["metrics"]["policy_entropy"],
                "lp_flow_residual": lp["metrics"].get("lp_flow_residual"),
                "lp_cost_residual": lp["metrics"].get("lp_cost_residual"),
                "lp_objective_gap": lp["metrics"].get("lp_objective_gap"),
            }
            candidate = _candidate_reference(env, config, safety_penalty)
            if candidate:
                row.update(candidate)
                row["lp_vs_candidate_reward_gap"] = lp["metrics"]["reward"] - candidate["candidate_reward"]
                row["lp_vs_candidate_safe_return_gap"] = lp["metrics"]["safe_return"] - candidate["candidate_safe_return"]
            rows.append(row)
            objectives = []
            backend_status = {}
            for method in methods:
                try:
                    checked = occupancy_lp_optimal(env, safety_penalty=safety_penalty, solver_method=method)
                    objectives.append(float(checked["occupancy_objective"]))
                    backend_status[method] = "ok"
                except Exception as exc:
                    backend_status[method] = f"error: {exc}"
                    status = "warning"
            max_gap = max(objectives) - min(objectives) if objectives else None
            if max_gap is not None and max_gap > 1e-7:
                status = "warning"
            cross_checks.append({
                "dataset": env.name,
                "seed": int(seed),
                "backend_status": backend_status,
                "max_objective_gap": max_gap,
            })
    analytic_checks = [_analytic_randomized_one_state_check(safety_penalty)]
    if any(item["status"] != "ok" for item in analytic_checks):
        status = "warning"
    return {
        "schema_version": 1,
        "status": status,
        "solver_backend_cross_checks": cross_checks,
        "analytic_randomized_checks": analytic_checks,
        "deterministic_reference_comparisons": rows,
        "tolerances": {
            "flow_residual": 1e-8,
            "cost_residual": 1e-8,
            "objective_gap": 1e-8,
            "backend_objective_gap": 1e-7,
        },
    }


def build_reproduction_manifest(config: dict, run_id: int) -> dict:
    """Return commands and environment metadata for reproducing benchmark artifacts."""
    return {
        "schema_version": 1,
        "run_id": run_id,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(),
        "commands": [
            ".\\.venv\\Scripts\\python.exe -m unittest tests.test_safe_rl_cmdp_benchmark tests.test_benchmark_suite tests.test_statistical_reporter",
            ".\\.venv\\Scripts\\python.exe -c \"from agents.validation_loop import run_validation_loop; run_validation_loop(RUN_ID)\"",
            ".\\.venv\\Scripts\\python.exe -c \"from agents.manuscript_writer import generate_manuscript; generate_manuscript(RUN_ID)\"",
        ],
        "benchmark_config": {
            key: config.get(key)
            for key in (
                "capability",
                "datasets",
                "methods",
                "seeds",
                "primary_metric",
                "metric_direction",
                "safety_penalty",
                "preference_penalties",
            )
        },
    }


def _write_json_artifact(workdir: Path, run_id: int, relative_path: str, artifact_type: str, payload: dict) -> str:
    path = artifact_path(workdir, relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    record_artifact(workdir, run_id, artifact_type, path, {
        "capability": "safe_rl_cmdp",
        "schema_version": payload.get("schema_version"),
    })
    return str(path)


def write_safe_rl_reproducibility_artifacts(workdir: Path, run_id: int, config: dict) -> dict:
    """Write CMDP appendix, LP validation, and reproduction manifest artifacts."""
    outputs = {
        "environment_appendix": _write_json_artifact(
            workdir,
            run_id,
            "artifacts/results/cmdp_environment_appendix.json",
            "cmdp_environment_appendix",
            build_environment_appendix(config),
        ),
        "lp_validation": _write_json_artifact(
            workdir,
            run_id,
            "artifacts/results/lp_validation.json",
            "lp_validation",
            build_lp_validation(config),
        ),
        "reproduction_manifest": _write_json_artifact(
            workdir,
            run_id,
            "artifacts/results/reproduction_manifest.json",
            "reproduction_manifest",
            build_reproduction_manifest(config, run_id),
        ),
    }
    return {"status": "complete", "outputs": outputs}
