"""Compile deep insights into reusable research execution specs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.artifact_manager import record_artifact


DEFAULT_REQUIRED_EVIDENCE = [
    "baseline_comparison",
    "multi_seed",
    "multi_dataset",
    "statistical_test",
    "ablation",
]


def _load_json(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _combined_text(insight: dict) -> str:
    parts = []
    for key in (
        "title",
        "hypothesis",
        "problem_statement",
        "formal_structure",
        "transformation",
        "evidence_summary",
    ):
        value = insight.get(key)
        if value:
            parts.append(str(value))
    plan = _load_json(insight.get("experimental_plan"), {})
    if isinstance(plan, dict):
        parts.append(json.dumps(plan))
    return " ".join(parts).lower()


def _classify_domain_and_task(insight: dict) -> tuple[str, str, list[str]]:
    text = _combined_text(insight)
    fairness_terms = (
        "fairness",
        "fair",
        "demographic",
        "equalized",
        "group",
        "protected",
    )
    safe_rl_terms = (
        "safe rl",
        "safe reinforcement",
        "cmdp",
        "constrained mdp",
        "markov decision",
        "policy",
    )

    if any(term in text for term in safe_rl_terms):
        return "safe_rl", "rl", [
            "safe_rl_cmdp",
            "generic_python_benchmark",
        ]
    if any(term in text for term in fairness_terms):
        return "fairness", "classification", [
            "fairness_classification",
            "generic_python_benchmark",
        ]
    return "unknown", "benchmark", ["generic_python_benchmark"]


def _metrics_from_plan(insight: dict, domain: str) -> tuple[list[str], list[str]]:
    plan = _load_json(insight.get("experimental_plan"), {})
    metrics = plan.get("metrics") if isinstance(plan, dict) else {}
    primary = None
    if isinstance(metrics, dict):
        primary = metrics.get("primary") or metrics.get("metric_name")
    elif isinstance(metrics, list) and metrics:
        primary = str(metrics[0])

    if not primary:
        if domain == "fairness":
            primary = "fairness_score"
        elif domain == "safe_rl":
            primary = "safe_return"
        else:
            primary = "metric"

    secondary = []
    if domain == "fairness":
        secondary = [
            "accuracy",
            "demographic_parity_gap",
            "equalized_odds_gap",
        ]
    elif domain == "safe_rl":
        secondary = [
            "reward",
            "cost",
            "constraint_violation",
        ]
    return [str(primary)], [metric for metric in secondary if metric != primary]


def compile_research_spec(insight: dict, run_id: int | None = None) -> dict:
    """Compile a deep insight row into a generic research execution spec."""
    domain, task_type, candidate_capabilities = _classify_domain_and_task(insight)
    primary_metrics, secondary_metrics = _metrics_from_plan(insight, domain)
    title = str(insight.get("title") or "Untitled research claim")

    return {
        "schema_version": 1,
        "run_id": run_id,
        "insight_id": insight.get("id"),
        "claim": str(
            insight.get("hypothesis")
            or insight.get("problem_statement")
            or title
        ),
        "domain": domain,
        "task_type": task_type,
        "evidence_level": "proxy",
        "required_evidence": list(DEFAULT_REQUIRED_EVIDENCE),
        "candidate_capabilities": candidate_capabilities,
        "primary_metrics": primary_metrics,
        "secondary_metrics": secondary_metrics,
        "constraints": {
            "offline_first": True,
            "max_runtime_seconds": 600,
        },
        "source": {
            "title": title,
            "tier": insight.get("tier"),
        },
    }


def compile_and_write_research_spec(insight: dict, workdir: Path, run_id: int) -> dict:
    """Compile, persist, and record a research spec artifact."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    spec = compile_research_spec(insight, run_id=run_id)
    path = workdir / "research_spec.json"
    path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
    record_artifact(
        workdir,
        run_id,
        "research_spec",
        path,
        {
            "domain": spec.get("domain"),
            "task_type": spec.get("task_type"),
        },
    )
    return spec
