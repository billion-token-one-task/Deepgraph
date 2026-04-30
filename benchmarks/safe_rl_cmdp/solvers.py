"""Exact tabular solvers and baselines for finite CMDP benchmarks."""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog

from benchmarks.safe_rl_cmdp.envs import FiniteCMDP
from benchmarks.safe_rl_cmdp.metrics import cmdp_metrics


def enumerate_deterministic_policies(env: FiniteCMDP) -> list[np.ndarray]:
    """Enumerate all deterministic stationary policies for a small finite CMDP."""
    return [
        np.asarray(actions, dtype=int)
        for actions in itertools.product(range(env.n_actions), repeat=env.n_states)
    ]


def _policy_model(env: FiniteCMDP, policy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.arange(env.n_states)
    actions = np.asarray(policy, dtype=int)
    transition = env.transitions[states, actions, :]
    reward = env.rewards[states, actions]
    cost = env.costs[states, actions]
    return transition, reward, cost


def _stationary_policy_model(env: FiniteCMDP, policy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    policy = np.asarray(policy, dtype=float)
    if policy.shape != (env.n_states, env.n_actions):
        raise ValueError(
            f"stationary policy must have shape {(env.n_states, env.n_actions)}, got {policy.shape}"
        )
    row_sums = policy.sum(axis=1)
    if np.any(policy < -1e-12) or not np.allclose(row_sums, 1.0, atol=1e-8):
        raise ValueError("stationary policy rows must be probability distributions")
    transition = np.einsum("sa,san->sn", policy, env.transitions)
    reward = np.sum(policy * env.rewards, axis=1)
    cost = np.sum(policy * env.costs, axis=1)
    return transition, reward, cost


def evaluate_policy(env: FiniteCMDP, policy: np.ndarray, safety_penalty: float = 6.0) -> dict:
    """Evaluate discounted reward and cost for a deterministic stationary policy."""
    transition, reward_vector, cost_vector = _policy_model(env, policy)
    system = np.eye(env.n_states) - env.gamma * transition
    reward_values = np.linalg.solve(system, reward_vector)
    cost_values = np.linalg.solve(system, cost_vector)
    reward = float(env.start @ reward_values)
    cost = float(env.start @ cost_values)
    metrics = cmdp_metrics(env, policy, reward, cost, safety_penalty=safety_penalty)
    return {
        "status": "ok",
        "policy": [int(action) for action in policy.tolist()],
        "policy_type": "stationary_deterministic",
        "reward": reward,
        "cost": cost,
        "metrics": metrics,
    }


def evaluate_stationary_policy(env: FiniteCMDP, policy: np.ndarray, safety_penalty: float = 6.0) -> dict:
    """Evaluate discounted reward and cost for a stationary randomized policy."""
    transition, reward_vector, cost_vector = _stationary_policy_model(env, policy)
    system = np.eye(env.n_states) - env.gamma * transition
    reward_values = np.linalg.solve(system, reward_vector)
    cost_values = np.linalg.solve(system, cost_vector)
    reward = float(env.start @ reward_values)
    cost = float(env.start @ cost_values)
    metrics = cmdp_metrics(env, policy, reward, cost, safety_penalty=safety_penalty)
    return {
        "status": "ok",
        "policy": policy.tolist(),
        "policy_type": "stationary_randomized",
        "reward": reward,
        "cost": cost,
        "metrics": metrics,
    }


def _best_by_score(env: FiniteCMDP, score_fn, safety_penalty: float = 6.0) -> dict:
    best = None
    best_score = -np.inf
    for policy in enumerate_deterministic_policies(env):
        result = evaluate_policy(env, policy, safety_penalty=safety_penalty)
        score = float(score_fn(result))
        if score > best_score + 1e-12:
            best = result
            best_score = score
    if best is None:
        raise ValueError("no deterministic policies were enumerated")
    return best


def reward_only_policy(env: FiniteCMDP, safety_penalty: float = 6.0) -> dict:
    """Select the deterministic policy with maximum discounted reward."""
    result = _best_by_score(env, lambda item: item["metrics"]["reward"], safety_penalty=safety_penalty)
    result["method"] = "reward_only"
    result["baseline_role"] = "unconstrained_reward_baseline"
    result["selection_protocol"] = "max_discounted_reward"
    return result


def lagrangian_policy(env: FiniteCMDP, penalty: float, safety_penalty: float = 6.0) -> dict:
    """Select policy maximizing reward minus fixed cost penalty."""
    result = _best_by_score(
        env,
        lambda item: item["metrics"]["reward"] - float(penalty) * item["metrics"]["cost"],
        safety_penalty=safety_penalty,
    )
    result["method"] = f"lagrangian_penalty_{float(penalty):.2f}"
    result["penalty"] = float(penalty)
    result["baseline_role"] = "fixed_penalty_baseline"
    result["selection_protocol"] = "fixed_penalty_scalarization"
    return result


def occupancy_enumeration(env: FiniteCMDP, safety_penalty: float = 6.0) -> dict:
    """Exact finite-CMDP frontier by deterministic policy enumeration.

    For these small tabular benchmarks, enumeration is the exact baseline used instead
    of a separate LP dependency. It picks the highest-reward feasible policy and falls
    back to the lowest-violation policy if no feasible deterministic policy exists.
    """
    feasible = []
    all_results = []
    for policy in enumerate_deterministic_policies(env):
        result = evaluate_policy(env, policy, safety_penalty=safety_penalty)
        all_results.append(result)
        if result["metrics"]["feasible"]:
            feasible.append(result)

    if feasible:
        best = max(feasible, key=lambda item: item["metrics"]["reward"])
    else:
        best = min(
            all_results,
            key=lambda item: (item["metrics"]["constraint_violation"], -item["metrics"]["reward"]),
        )
    best = dict(best)
    best["method"] = "occupancy_enumeration"
    best["baseline_role"] = "deterministic_enumeration_reference"
    best["selection_protocol"] = "best_feasible_deterministic_policy"
    return best


def deterministic_feasible_best(env: FiniteCMDP, safety_penalty: float = 6.0) -> dict:
    """Best feasible deterministic stationary policy by exhaustive enumeration."""
    result = occupancy_enumeration(env, safety_penalty=safety_penalty)
    result = dict(result)
    result["method"] = "deterministic_feasible_best"
    result["baseline_role"] = "deterministic_feasible_reference"
    result["selection_protocol"] = "best_feasible_deterministic_policy"
    return result


def occupancy_lp_optimal(
    env: FiniteCMDP,
    safety_penalty: float = 6.0,
    solver_method: str = "highs",
) -> dict:
    """Solve the discounted CMDP occupancy-measure LP.

    Variables are unnormalized discounted occupancies x(s,a). The flow constraints
    are sum_a x(s,a) - gamma * sum_{s',a'} P(s | s',a') x(s',a') = rho_0(s).
    The cost constraint is sum_{s,a} x(s,a)c(s,a) <= cost_limit.
    """
    n_variables = env.n_states * env.n_actions
    rewards = env.rewards.reshape(n_variables)
    costs = env.costs.reshape(n_variables)

    flow = np.zeros((env.n_states, n_variables), dtype=float)
    for state in range(env.n_states):
        for action in range(env.n_actions):
            idx = state * env.n_actions + action
            flow[state, idx] += 1.0
            flow[:, idx] -= env.gamma * env.transitions[state, action, :]

    result = linprog(
        c=-rewards,
        A_ub=np.asarray([costs]),
        b_ub=np.asarray([env.cost_limit]),
        A_eq=flow,
        b_eq=env.start,
        bounds=[(0.0, None)] * n_variables,
        method=solver_method,
    )
    if not result.success:
        raise ValueError(f"occupancy LP failed: {result.message}")

    occupancy = np.maximum(result.x.reshape(env.n_states, env.n_actions), 0.0)
    state_mass = occupancy.sum(axis=1, keepdims=True)
    policy = np.zeros_like(occupancy)
    active = state_mass[:, 0] > 1e-12
    policy[active] = occupancy[active] / state_mass[active]
    if np.any(~active):
        policy[~active, 0] = 1.0

    evaluated = evaluate_stationary_policy(env, policy, safety_penalty=safety_penalty)
    evaluated["method"] = "occupancy_lp_optimal"
    evaluated["solver_method"] = solver_method
    evaluated["baseline_role"] = "exact_lp_reference"
    evaluated["selection_protocol"] = "hard_constrained_occupancy_lp"
    occupancy_objective = float(-result.fun)
    occupancy_cost = float(costs @ result.x)
    flow_residual = float(np.max(np.abs(flow @ result.x - env.start)))
    cost_residual = float(max(0.0, occupancy_cost - env.cost_limit))
    objective_gap = float(abs(occupancy_objective - evaluated["reward"]))
    evaluated["occupancy_objective"] = occupancy_objective
    evaluated["occupancy_cost"] = occupancy_cost
    evaluated["metrics"]["lp_flow_residual"] = flow_residual
    evaluated["metrics"]["lp_cost_residual"] = cost_residual
    evaluated["metrics"]["lp_objective_gap"] = objective_gap
    return evaluated


def lagrangian_grid_best(
    env: FiniteCMDP,
    penalties: list[float] | None = None,
    safety_penalty: float = 6.0,
) -> dict:
    """Tuned fixed-penalty Lagrangian baseline over the same configured grid."""
    penalties = penalties or [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    candidates = [lagrangian_policy(env, penalty, safety_penalty=safety_penalty) for penalty in penalties]
    best = max(
        candidates,
        key=lambda item: (
            item["metrics"]["safe_return"],
            -item["metrics"]["constraint_violation"],
            item["metrics"]["reward"],
        ),
    )
    best = dict(best)
    best["method"] = "lagrangian_grid_best"
    best["selected_penalty"] = float(best.get("penalty", 0.0))
    best["baseline_role"] = "oracle_grid_selected_baseline"
    best["selection_protocol"] = "oracle_grid_selected_safe_return"
    return best


def preference_cone_policy(
    env: FiniteCMDP,
    penalties: list[float] | None = None,
    safety_penalty: float = 6.0,
) -> dict:
    """Validation-style selection over Lagrangian policies using safe-return metric."""
    best = lagrangian_grid_best(env, penalties, safety_penalty=safety_penalty)
    best = dict(best)
    best["method"] = "preference_cone_policy"
    best["selected_penalty"] = float(best.get("penalty", 0.0))
    best["baseline_role"] = "oracle_grid_selected_baseline"
    best["selection_protocol"] = "oracle_grid_selected_safe_return"
    return best
