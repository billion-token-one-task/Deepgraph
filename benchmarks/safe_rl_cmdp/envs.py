"""Small finite CMDP environments for safe RL benchmark runs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FiniteCMDP:
    name: str
    seed: int
    transitions: np.ndarray
    rewards: np.ndarray
    costs: np.ndarray
    start: np.ndarray
    gamma: float
    cost_limit: float

    @property
    def n_states(self) -> int:
        return int(self.transitions.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.transitions.shape[1])


def _normalize(transitions: np.ndarray) -> np.ndarray:
    totals = transitions.sum(axis=2, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("every state-action transition row must have positive mass")
    return transitions / totals


def _jitter(values: np.ndarray, seed: int, scale: float = 0.015) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return values + rng.normal(0.0, scale, size=values.shape)


def _risky_shortcut(seed: int) -> FiniteCMDP:
    transitions = np.zeros((3, 2, 3), dtype=float)
    transitions[0, 0, 1] = 1.0
    transitions[0, 1, 2] = 1.0
    transitions[1, :, 1] = 1.0
    transitions[2, :, 2] = 1.0
    rewards = np.array([
        [1.00, 1.95],
        [0.25, 0.25],
        [0.45, 0.45],
    ])
    costs = np.array([
        [0.04, 0.62],
        [0.04, 0.04],
        [0.16, 0.16],
    ])
    return FiniteCMDP(
        name="risky_shortcut",
        seed=seed,
        transitions=_normalize(transitions),
        rewards=_jitter(rewards, seed, scale=0.01),
        costs=np.maximum(0.0, _jitter(costs, seed + 101, scale=0.004)),
        start=np.array([1.0, 0.0, 0.0]),
        gamma=0.82,
        cost_limit=0.50,
    )


def _delayed_safety(seed: int) -> FiniteCMDP:
    transitions = np.zeros((4, 2, 4), dtype=float)
    transitions[0, 0, 1] = 1.0
    transitions[0, 1, 2] = 1.0
    transitions[1, 0, 3] = 1.0
    transitions[1, 1, 2] = 1.0
    transitions[2, 0, 3] = 1.0
    transitions[2, 1, 2] = 1.0
    transitions[3, :, 3] = 1.0
    rewards = np.array([
        [0.55, 1.05],
        [0.70, 1.20],
        [0.45, 0.95],
        [0.20, 0.20],
    ])
    costs = np.array([
        [0.03, 0.28],
        [0.04, 0.48],
        [0.02, 0.32],
        [0.01, 0.01],
    ])
    return FiniteCMDP(
        name="delayed_safety",
        seed=seed,
        transitions=_normalize(transitions),
        rewards=_jitter(rewards, seed + 7, scale=0.012),
        costs=np.maximum(0.0, _jitter(costs, seed + 131, scale=0.004)),
        start=np.array([1.0, 0.0, 0.0, 0.0]),
        gamma=0.86,
        cost_limit=0.42,
    )


def _resource_gathering(seed: int) -> FiniteCMDP:
    transitions = np.zeros((4, 2, 4), dtype=float)
    transitions[0, 0, 1] = 1.0
    transitions[0, 1, 2] = 1.0
    transitions[1, 0, 1] = 0.55
    transitions[1, 0, 3] = 0.45
    transitions[1, 1, 2] = 1.0
    transitions[2, 0, 3] = 1.0
    transitions[2, 1, 2] = 0.65
    transitions[2, 1, 3] = 0.35
    transitions[3, :, 3] = 1.0
    rewards = np.array([
        [0.45, 0.92],
        [0.50, 0.88],
        [0.38, 1.08],
        [0.18, 0.18],
    ])
    costs = np.array([
        [0.03, 0.20],
        [0.05, 0.22],
        [0.04, 0.36],
        [0.01, 0.01],
    ])
    return FiniteCMDP(
        name="resource_gathering",
        seed=seed,
        transitions=_normalize(transitions),
        rewards=_jitter(rewards, seed + 17, scale=0.012),
        costs=np.maximum(0.0, _jitter(costs, seed + 151, scale=0.004)),
        start=np.array([1.0, 0.0, 0.0, 0.0]),
        gamma=0.84,
        cost_limit=0.38,
    )


def _randomized_bandit(seed: int) -> FiniteCMDP:
    transitions = np.array([[[1.0], [1.0]]], dtype=float)
    rewards = np.array([[0.05, 1.00]])
    costs = np.array([[0.02, 1.00]])
    return FiniteCMDP(
        name="randomized_bandit",
        seed=seed,
        transitions=_normalize(transitions),
        rewards=_jitter(rewards, seed + 23, scale=0.008),
        costs=np.maximum(0.0, _jitter(costs, seed + 181, scale=0.003)),
        start=np.array([1.0]),
        gamma=0.50,
        cost_limit=1.00,
    )


def _stochastic_bridge(seed: int) -> FiniteCMDP:
    transitions = np.zeros((5, 2, 5), dtype=float)
    transitions[0, 0, 1] = 0.82
    transitions[0, 0, 2] = 0.18
    transitions[0, 1, 2] = 0.75
    transitions[0, 1, 3] = 0.25
    transitions[1, 0, 4] = 1.0
    transitions[1, 1, 3] = 1.0
    transitions[2, 0, 4] = 0.65
    transitions[2, 0, 3] = 0.35
    transitions[2, 1, 3] = 1.0
    transitions[3, 0, 4] = 1.0
    transitions[3, 1, 3] = 1.0
    transitions[4, :, 4] = 1.0
    rewards = np.array([
        [0.45, 0.95],
        [0.40, 1.10],
        [0.52, 1.05],
        [0.25, 0.85],
        [0.12, 0.12],
    ])
    costs = np.array([
        [0.03, 0.26],
        [0.04, 0.44],
        [0.05, 0.34],
        [0.03, 0.30],
        [0.01, 0.01],
    ])
    return FiniteCMDP(
        name="stochastic_bridge",
        seed=seed,
        transitions=_normalize(transitions),
        rewards=_jitter(rewards, seed + 29, scale=0.012),
        costs=np.maximum(0.0, _jitter(costs, seed + 191, scale=0.004)),
        start=np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        gamma=0.88,
        cost_limit=0.44,
    )


def _tight_budget_chain(seed: int) -> FiniteCMDP:
    transitions = np.zeros((5, 2, 5), dtype=float)
    transitions[0, 0, 1] = 1.0
    transitions[0, 1, 2] = 1.0
    transitions[1, 0, 2] = 1.0
    transitions[1, 1, 3] = 1.0
    transitions[2, 0, 3] = 1.0
    transitions[2, 1, 4] = 1.0
    transitions[3, 0, 4] = 1.0
    transitions[3, 1, 3] = 1.0
    transitions[4, :, 4] = 1.0
    rewards = np.array([
        [0.38, 0.78],
        [0.45, 0.88],
        [0.42, 0.92],
        [0.28, 0.82],
        [0.10, 0.10],
    ])
    costs = np.array([
        [0.02, 0.18],
        [0.03, 0.25],
        [0.02, 0.30],
        [0.02, 0.24],
        [0.01, 0.01],
    ])
    return FiniteCMDP(
        name="tight_budget_chain",
        seed=seed,
        transitions=_normalize(transitions),
        rewards=_jitter(rewards, seed + 37, scale=0.010),
        costs=np.maximum(0.0, _jitter(costs, seed + 211, scale=0.003)),
        start=np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        gamma=0.90,
        cost_limit=0.30,
    )


SYSTEMATIC_SPECS = {
    "systematic_chain_s3_g70_c30": {
        "kind": "chain",
        "states": 3,
        "actions": 2,
        "gamma": 0.70,
        "cost_limit": 0.30,
        "stochasticity": 0.00,
    },
    "systematic_chain_s4_g82_c35": {
        "kind": "chain",
        "states": 4,
        "actions": 2,
        "gamma": 0.82,
        "cost_limit": 0.35,
        "stochasticity": 0.08,
    },
    "systematic_random_s4_a3_g78_c40": {
        "kind": "random",
        "states": 4,
        "actions": 3,
        "gamma": 0.78,
        "cost_limit": 0.40,
        "stochasticity": 0.35,
    },
    "systematic_random_s5_a2_g90_c45": {
        "kind": "random",
        "states": 5,
        "actions": 2,
        "gamma": 0.90,
        "cost_limit": 0.45,
        "stochasticity": 0.25,
    },
}


def _systematic_cmdp(name: str, seed: int) -> FiniteCMDP:
    spec = SYSTEMATIC_SPECS[name]
    n_states = int(spec["states"])
    n_actions = int(spec["actions"])
    rng_seed = seed + sum(ord(ch) for ch in name)
    rng = np.random.default_rng(rng_seed)
    transitions = np.zeros((n_states, n_actions, n_states), dtype=float)
    terminal = n_states - 1
    if spec["kind"] == "chain":
        for state in range(n_states):
            if state == terminal:
                transitions[state, :, terminal] = 1.0
                continue
            for action in range(n_actions):
                target = min(terminal, state + 1 + min(action, 1))
                transitions[state, action, target] = 1.0 - float(spec["stochasticity"])
                transitions[state, action, state] += float(spec["stochasticity"])
    else:
        for state in range(n_states):
            if state == terminal:
                transitions[state, :, terminal] = 1.0
                continue
            for action in range(n_actions):
                target = min(terminal, state + 1 + (action % 2))
                transitions[state, action, target] += 0.55
                transitions[state, action] += 0.45 * rng.dirichlet(np.ones(n_states))

    rewards = np.zeros((n_states, n_actions), dtype=float)
    costs = np.zeros((n_states, n_actions), dtype=float)
    for state in range(n_states):
        progress = state / max(1, n_states - 1)
        for action in range(n_actions):
            risk = action / max(1, n_actions - 1)
            rewards[state, action] = 0.18 + 0.35 * progress + 0.85 * risk
            costs[state, action] = 0.02 + 0.10 * progress + 0.42 * risk
    rewards[terminal, :] = 0.08
    costs[terminal, :] = 0.01

    start = np.zeros(n_states, dtype=float)
    start[0] = 1.0
    return FiniteCMDP(
        name=name,
        seed=seed,
        transitions=_normalize(transitions),
        rewards=_jitter(rewards, rng_seed + 3, scale=0.010),
        costs=np.maximum(0.0, _jitter(costs, rng_seed + 7, scale=0.003)),
        start=start,
        gamma=float(spec["gamma"]),
        cost_limit=float(spec["cost_limit"]),
    )


def make_cmdp(name: str, seed: int) -> FiniteCMDP:
    """Create a deterministic finite CMDP with seed-level reward/cost jitter."""
    if name == "risky_shortcut":
        return _risky_shortcut(seed)
    if name == "delayed_safety":
        return _delayed_safety(seed)
    if name == "resource_gathering":
        return _resource_gathering(seed)
    if name == "randomized_bandit":
        return _randomized_bandit(seed)
    if name == "stochastic_bridge":
        return _stochastic_bridge(seed)
    if name == "tight_budget_chain":
        return _tight_budget_chain(seed)
    if name in SYSTEMATIC_SPECS:
        return _systematic_cmdp(name, seed)
    raise ValueError(f"unknown CMDP dataset: {name}")
