"""Compile deep_insight experiment data into a structured execution plan."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _names(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    names = []
    for item in items:
        if isinstance(item, dict):
            name = item.get("name") or item.get("dataset") or item.get("method")
        else:
            name = str(item)
        if name:
            names.append(str(name))
    return names


def _success_criteria(scaffold: dict) -> dict:
    success = scaffold.get("success_criteria") if isinstance(scaffold, dict) else {}
    return dict(success) if isinstance(success, dict) else {}


def _metric_from(plan: dict, success: dict) -> tuple[str, str]:
    metrics = plan.get("metrics") if isinstance(plan, dict) else {}
    primary = None
    direction = None
    if isinstance(metrics, dict):
        primary = metrics.get("primary") or metrics.get("metric_name")
        direction = metrics.get("direction") or metrics.get("metric_direction")
    elif isinstance(metrics, list) and metrics:
        primary = str(metrics[0])

    primary = primary or success.get("metric_name") or "metric"
    direction = direction or success.get("metric_direction") or "higher"
    if direction not in ("higher", "lower"):
        direction = "higher"
    return str(primary), str(direction)


def _hypothesis_from(insight: dict) -> str:
    for key in ("hypothesis", "problem_statement", "title"):
        value = insight.get(key)
        if value:
            return str(value)
    return "No hypothesis provided."


def compile_execution_plan(insight: dict, workdir: Path, codebase: dict, scaffold: dict) -> dict:
    """Create and persist execution_plan.json for a forged experiment."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    raw_plan = _load_json(insight.get("experimental_plan"), {})
    plan = raw_plan if isinstance(raw_plan, dict) else {}
    success = _success_criteria(scaffold)
    primary_metric, metric_direction = _metric_from(plan, success)

    execution_plan = {
        "schema_version": 1,
        "hypothesis": _hypothesis_from(insight),
        "primary_metric": primary_metric,
        "metric_direction": metric_direction,
        "stages": [
            {"name": "reproduction", "required": True},
            {"name": "hypothesis_test", "required": True},
            {"name": "ablation", "required": False},
        ],
        "success_criteria": success,
        "datasets": _names(plan.get("datasets")),
        "baselines": _names(plan.get("baselines")),
        "codebase": {
            "name": str(codebase.get("name", "")) if isinstance(codebase, dict) else "",
            "url": str(codebase.get("url", "")) if isinstance(codebase, dict) else "",
        },
    }

    path = workdir / "execution_plan.json"
    path.write_text(json.dumps(execution_plan, indent=2, sort_keys=True), encoding="utf-8")
    return execution_plan
