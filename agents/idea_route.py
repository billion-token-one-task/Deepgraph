"""Route a discovered idea to the right claim/evidence path.

Discovery can be good enough to seed an idea while still being too weak for a
top-tier paper claim. This module makes that distinction explicit and stable.
"""

from __future__ import annotations

import json
from typing import Any


SYNTHETIC_DATASET_MARKERS = (
    "synthetic",
    "simulated",
    "simulation",
    "toy",
    "smoke",
    "probe",
    "dummy",
    "random",
    "minimal",
)

FULL_PAPER_ROUTE = "full_paper"
WORKSHOP_ROUTE = "workshop"
RESEARCH_NOTE_ROUTE = "research_note"
PROBE_ROUTE = "probe"
BLOCKED_ROUTE = "blocked"


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    if value in (None, "", "unknown"):
        return []
    return [value]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = _text(item)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _named_values(rows: Any, *, keys: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for row in _as_list(rows):
        if isinstance(row, dict):
            for key in keys:
                value = _text(row.get(key))
                if value:
                    values.append(value)
                    break
        else:
            value = _text(row)
            if value:
                values.append(value)
    return _unique(values)


def _looks_synthetic_dataset(name: str) -> bool:
    lowered = _text(name).lower()
    return not lowered or any(marker in lowered for marker in SYNTHETIC_DATASET_MARKERS)


def _metric_name(plan: dict[str, Any]) -> str:
    metrics = plan.get("metrics")
    if isinstance(metrics, dict):
        return _text(metrics.get("primary") or metrics.get("name"))
    names = _named_values(metrics, keys=("name", "metric"))
    return names[0] if names else _text(metrics)


def _minimum_seeds(plan: dict[str, Any]) -> int:
    raw = plan.get("minimum_seeds") or plan.get("seeds")
    if isinstance(raw, list):
        return len(raw)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _novelty_band(insight: dict[str, Any]) -> str:
    related = _as_dict(insight.get("related_work_positioning"))
    raw = (
        insight.get("novelty_status")
        or insight.get("novelty")
        or related.get("novelty_status")
        or related.get("position")
        or ""
    )
    lowered = _text(raw).lower()
    if any(token in lowered for token in ("duplicate", "already exists", "not novel", "rejected")):
        return "rejected"
    if any(token in lowered for token in ("partial", "partially", "similar", "incremental", "overlap")):
        return "partial"
    if any(token in lowered for token in ("confirmed", "novel", "new", "gap")):
        return "novel"
    return "unknown"


def _experimentability_band(insight: dict[str, Any], plan: dict[str, Any]) -> str:
    raw = insight.get("experimentability") or plan.get("experimentability") or ""
    lowered = _text(raw).lower()
    if any(token in lowered for token in ("impossible", "infeasible", "blocked", "unsafe")):
        return "blocked"
    if any(token in lowered for token in ("low", "weak", "unclear")):
        return "weak"
    if any(token in lowered for token in ("high", "ready", "feasible")):
        return "ready"
    return "unknown"


def classify_idea_route(
    insight: dict[str, Any],
    plan: dict[str, Any] | None = None,
    method: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify how aggressively the pipeline may claim and benchmark an idea.

    The classifier is deliberately deterministic. It is a contract guardrail, not
    another open-ended reviewer prompt.
    """

    insight = _as_dict(insight)
    plan = _as_dict(plan) or _as_dict(insight.get("experimental_plan"))
    method = _as_dict(method) or _as_dict(insight.get("proposed_method"))

    datasets = _unique(
        _named_values(plan.get("benchmark_targets"), keys=("name", "hf_dataset", "dataset"))
        + _named_values(plan.get("datasets"), keys=("name", "hf_dataset", "dataset"))
    )
    real_datasets = [name for name in datasets if not _looks_synthetic_dataset(name)]
    baselines = _named_values(plan.get("baselines"), keys=("name", "model", "method"))
    models = _unique(
        _named_values(plan.get("model_targets"), keys=("name", "hf_model", "model"))
        + _named_values(plan.get("models"), keys=("name", "hf_model", "model"))
    )
    ablations = _unique(
        _named_values(plan.get("ablations"), keys=("name", "component", "factor"))
        + _named_values(plan.get("components"), keys=("name", "component"))
    )

    definition = _text(method.get("definition") or method.get("pseudocode") or method.get("one_line"))
    method_ok = len(definition) >= 24
    problem_ok = bool(
        _text(insight.get("problem_statement"))
        or _text(insight.get("existing_weakness"))
        or _text(insight.get("title"))
    )
    metric = _metric_name(plan)
    seeds = _minimum_seeds(plan)
    resource_class = _text(insight.get("resource_class") or plan.get("resource_class") or "cpu").lower()
    corpus = " ".join(
        [
            _text(insight.get("title")),
            _text(insight.get("problem_statement")),
            _text(method.get("definition")),
            json.dumps(plan, ensure_ascii=False)[:4000],
        ]
    ).lower()
    requires_model = bool(
        plan.get("requires_real_model")
        or plan.get("real_benchmark_required")
        or resource_class.startswith("gpu")
        or any(token in corpus for token in ("llm", "language model", "transformer", "reasoning", "qa"))
    )

    missing: list[str] = []
    if not method_ok:
        missing.append("method_definition")
    if not problem_ok:
        missing.append("problem_statement")
    if len(real_datasets) < 1:
        missing.append("real_benchmark_dataset")
    if len(baselines) < 2:
        missing.append("two_or_more_baselines")
    if requires_model and not models:
        missing.append("real_model_target")
    if len(ablations) < 2:
        missing.append("mechanism_ablations")
    if seeds < 3:
        missing.append("three_or_more_seeds")
    if not metric:
        missing.append("primary_metric")
    if plan.get("generated_runner_supported") is False:
        missing.append("executable_benchmark_recipe")

    novelty = _novelty_band(insight)
    experimentability = _experimentability_band(insight, plan)
    if novelty == "partial":
        missing.append("sharpen_novelty_boundary")

    if experimentability == "blocked" or novelty == "rejected":
        route = BLOCKED_ROUTE
    elif plan.get("generated_runner_supported") is False:
        route = BLOCKED_ROUTE
    elif not missing and novelty != "partial":
        route = FULL_PAPER_ROUTE
    elif method_ok and problem_ok and real_datasets and len(baselines) >= 1 and len(missing) <= 4:
        route = WORKSHOP_ROUTE
    elif method_ok or problem_ok:
        route = RESEARCH_NOTE_ROUTE
    else:
        route = PROBE_ROUTE

    claim_strength_by_route = {
        FULL_PAPER_ROUTE: "strong",
        WORKSHOP_ROUTE: "moderate",
        RESEARCH_NOTE_ROUTE: "preliminary",
        PROBE_ROUTE: "sanity_only",
        BLOCKED_ROUTE: "blocked",
    }
    required_evidence_by_route = {
        FULL_PAPER_ROUTE: "full_benchmark",
        WORKSHOP_ROUTE: "focused_benchmark",
        RESEARCH_NOTE_ROUTE: "sanity_plus_literature",
        PROBE_ROUTE: "sanity_probe",
        BLOCKED_ROUTE: "contract_revision",
    }

    paper_allowed = route == FULL_PAPER_ROUTE
    experiment_allowed = route != BLOCKED_ROUTE
    reason = (
        "Ready for a full benchmark-backed paper claim."
        if route == FULL_PAPER_ROUTE
        else "Idea is usable, but the current contract cannot support a top-tier paper claim."
        if route in {WORKSHOP_ROUTE, RESEARCH_NOTE_ROUTE, PROBE_ROUTE}
        else "Idea is blocked until the contract is revised."
    )

    return {
        "route": route,
        "claim_strength": claim_strength_by_route[route],
        "required_evidence_level": required_evidence_by_route[route],
        "paper_allowed": paper_allowed,
        "experiment_allowed": experiment_allowed,
        "reason": reason,
        "missing": _unique(missing),
        "benchmark_policy": {
            "requires_full_benchmark_package": paper_allowed,
            "allows_sanity_only": route in {RESEARCH_NOTE_ROUTE, PROBE_ROUTE},
            "minimum_baselines": 2,
            "recommended_baselines": 3,
            "minimum_real_datasets": 1,
            "minimum_models": 1 if requires_model else 0,
            "minimum_ablations": 2,
            "minimum_seeds": 3,
        },
        "decision_inputs": {
            "novelty": novelty,
            "experimentability": experimentability,
            "baseline_count": len(baselines),
            "real_dataset_count": len(real_datasets),
            "model_count": len(models),
            "ablation_count": len(ablations),
            "minimum_seeds": seeds,
            "requires_model": requires_model,
            "primary_metric": metric,
        },
    }
