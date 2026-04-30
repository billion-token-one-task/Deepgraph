"""In-memory registry for real experiment execution capabilities."""
from __future__ import annotations

import copy
import importlib.util


CAPABILITIES = [
    {
        "id": "fairness_classification",
        "implemented": True,
        "domains": ["fairness"],
        "task_types": ["classification"],
        "required_packages": ["numpy", "sklearn", "fairlearn"],
        "runner": "benchmarks.fairness_classification.harness",
        "default_config": {
            "datasets": [
                "openml_adult_sex",
                "openml_german_credit_sex",
                "fairlearn_credit_card_sex",
                "fairlearn_bank_marketing_age",
            ],
            "methods": [
                "logistic_regression",
                "exponentiated_gradient",
                "threshold_optimizer",
                "validation_selected_fairlearn_baseline",
                "validation_selected_preference_cone",
            ],
            "seeds": list(range(10)),
            "primary_metric": "fairness_score",
            "metric_direction": "higher",
            "fairness_penalty": 0.70,
            "paper_title": "Validation-Selected Preference-Cone Thresholding for Public Fairness Benchmarks",
            "scoped_claim": (
                "Validation-selected preference-cone thresholding improves a fairness-weighted objective "
                "over logistic regression and exposes trade-offs relative to fairness-aware "
                "post-processing baselines across public protected-attribute classification datasets."
            ),
            "timeout_seconds": 600,
            "ablations": [
                {
                    "name": "preference_cone_penalty_sweep",
                    "methods": [
                        "preference_cone_threshold",
                        "preference_cone_threshold_penalty_0.00",
                        "preference_cone_threshold_penalty_0.45",
                        "preference_cone_threshold_penalty_0.70",
                        "preference_cone_threshold_penalty_0.90",
                    ],
                },
            ],
        },
    },
    {
        "id": "generic_python_benchmark",
        "implemented": True,
        "domains": ["unknown", "fairness", "safe_rl"],
        "task_types": ["benchmark", "classification", "rl"],
        "required_packages": [],
        "runner": "",
        "default_config": {},
    },
    {
        "id": "safe_rl_cmdp",
        "implemented": True,
        "domains": ["safe_rl"],
        "task_types": ["rl"],
        "required_packages": ["numpy", "scipy"],
        "runner": "benchmarks.safe_rl_cmdp.harness",
        "default_config": {
            "datasets": [
                "risky_shortcut",
                "delayed_safety",
                "resource_gathering",
                "randomized_bandit",
                "stochastic_bridge",
                "tight_budget_chain",
                "systematic_chain_s3_g70_c30",
                "systematic_chain_s4_g82_c35",
                "systematic_random_s4_a3_g78_c40",
                "systematic_random_s5_a2_g90_c45",
            ],
            "methods": [
                "reward_only",
                "lagrangian_penalty_1.00",
                "lagrangian_penalty_3.00",
                "lagrangian_penalty_4.00",
                "lagrangian_grid_best",
                "preference_cone_policy",
                "deterministic_feasible_best",
                "occupancy_lp_optimal",
            ],
            "seeds": list(range(10)),
            "primary_metric": "safe_return",
            "metric_direction": "higher",
            "baseline_method": "reward_only",
            "candidate_method": "lagrangian_penalty_4.00",
            "reference_method": "occupancy_lp_optimal",
            "paper_title": "Auditable Finite CMDP Safe-Return Benchmarks With Fixed-Penalty Baselines",
            "scoped_claim": (
                "An auditable finite-CMDP benchmark harness can report reward-cost trade-offs, "
                "constraint violations, a fixed-penalty Lagrangian diagnostic baseline, oracle "
                "grid-selected diagnostic baselines, and exact occupancy-measure LP references on "
                "hand-designed and systematically generated small safe-RL tasks."
            ),
            "timeout_seconds": 600,
            "safety_penalty": 6.0,
            "preference_penalties": [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0],
            "ablations": [
                {
                    "name": "safe_rl_penalty_sweep",
                    "methods": [
                        "lagrangian_penalty_0.50",
                        "lagrangian_penalty_2.00",
                        "lagrangian_penalty_4.00",
                        "lagrangian_penalty_8.00",
                    ],
                },
                {
                    "name": "safe_return_safety_penalty_sensitivity",
                    "methods": [
                        "reward_only",
                        "lagrangian_grid_best",
                        "occupancy_lp_optimal",
                    ],
                    "safety_penalty": 3.0,
                },
                {
                    "name": "safe_return_safety_penalty_sensitivity",
                    "methods": [
                        "reward_only",
                        "lagrangian_grid_best",
                        "occupancy_lp_optimal",
                    ],
                    "safety_penalty": 9.0,
                },
            ],
        },
    },
]


def list_capabilities() -> list[dict]:
    """Return a copy of registered capabilities."""
    return copy.deepcopy(CAPABILITIES)


def get_capability(capability_id: str) -> dict | None:
    """Return a registered capability by id."""
    for capability in CAPABILITIES:
        if capability["id"] == capability_id:
            return copy.deepcopy(capability)
    return None


def _matches_spec(capability: dict, research_spec: dict) -> bool:
    domain = research_spec.get("domain")
    task_type = research_spec.get("task_type")
    return (
        domain in capability.get("domains", [])
        and task_type in capability.get("task_types", [])
    )


def select_capability(research_spec: dict) -> dict:
    """Select the first implemented candidate matching the research spec."""
    candidates = research_spec.get("candidate_capabilities") or []
    for capability_id in candidates:
        capability = get_capability(capability_id)
        if capability and capability.get("implemented") and _matches_spec(capability, research_spec):
            return capability

    generic = get_capability("generic_python_benchmark")
    if generic:
        return generic
    raise ValueError("generic_python_benchmark capability is not registered")


def missing_dependencies(capability: dict | None) -> list[str]:
    """Return import packages required by a capability that are not installed."""
    if not capability:
        return []
    missing = []
    for package in capability.get("required_packages", []):
        if importlib.util.find_spec(package) is None:
            missing.append(package)
    return missing
