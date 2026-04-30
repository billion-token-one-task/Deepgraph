"""Metrics for finite CMDP safe RL benchmark rows."""
from __future__ import annotations

import numpy as np

from benchmarks.safe_rl_cmdp.envs import FiniteCMDP


def policy_support_size(policy: np.ndarray) -> int:
    """Return the number of state-action choices with nonzero probability."""
    policy = np.asarray(policy)
    if policy.ndim == 1:
        return int(policy.shape[0])
    return int(np.sum(policy > 1e-12))


def policy_entropy(policy: np.ndarray) -> float:
    """Return mean per-state action entropy for a stationary policy."""
    policy = np.asarray(policy, dtype=float)
    if policy.ndim == 1:
        return 0.0
    entropies = []
    for row in policy:
        probs = row[row > 1e-12]
        if probs.size == 0:
            entropies.append(0.0)
        else:
            entropies.append(float(-np.sum(probs * np.log(probs))))
    return float(np.mean(entropies)) if entropies else 0.0


def cmdp_metrics(
    env: FiniteCMDP,
    policy: np.ndarray,
    reward: float,
    cost: float,
    safety_penalty: float = 6.0,
    tolerance: float = 1e-9,
) -> dict:
    """Build metric dict from discounted reward/cost values."""
    violation = max(0.0, float(cost) - float(env.cost_limit))
    support = policy_support_size(policy)
    entropy = policy_entropy(policy)
    return {
        "reward": float(reward),
        "cost": float(cost),
        "cost_limit": float(env.cost_limit),
        "constraint_violation": violation,
        "safe_return": float(reward) - float(safety_penalty) * violation,
        "feasible": bool(violation <= tolerance),
        "policy_entropy": float(entropy),
        "support_size": support,
        "safety_penalty": float(safety_penalty),
    }
