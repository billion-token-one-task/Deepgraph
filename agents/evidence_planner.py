"""Adaptive evidence planning for deep insights."""

from __future__ import annotations

import json
from typing import Any


EFFICIENCY_KEYWORDS = {
    "latency", "throughput", "memory", "flops", "compute", "cost", "gpu-hour",
    "gpu hours", "wall-clock", "runtime", "inference speed", "training speed",
    "parameter count", "vram", "tokens/sec",
}

BENCHMARK_AUDIT_KEYWORDS = {
    "judge", "annotation", "protocol", "metric choice",
    "leaderboard", "audit", "bias", "variance", "reproducibility", "stress test",
    "data contamination", "benchmark leak",
}

MECHANISTIC_VIS_KEYWORDS = {
    "mechanism", "attention", "representation", "trajectory", "phase transition",
    "cluster", "saliency", "activation", "failure mode", "causal",
}

TREND_VIS_KEYWORDS = {
    "curve", "trend", "trajectory", "scaling", "plateau", "spread", "sweep",
    "tradeoff", "pareto",
}

META_ANALYSIS_KEYWORDS = {
    "meta-analysis", "survey", "synthesis", "trend", "plateau", "aggregate",
    "cross-paper", "cross benchmark", "audit",
}


def _load_json(value: str | dict | list | None, default):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _text_corpus(insight: dict, plan: dict, method: dict) -> str:
    return " ".join(
        str(part or "")
        for part in [
            insight.get("title"),
            insight.get("problem_statement"),
            insight.get("existing_weakness"),
            insight.get("evidence_summary"),
            insight.get("formal_structure"),
            insight.get("transformation"),
            insight.get("mechanism_type"),
            method.get("name"),
            method.get("type"),
            method.get("one_line"),
            method.get("definition"),
            json.dumps(plan.get("metrics", {}), ensure_ascii=False),
            json.dumps(plan.get("expected_results", {}), ensure_ascii=False),
            json.dumps(plan.get("compute_budget", {}), ensure_ascii=False),
        ]
    ).lower()


def _metric_labels(plan: dict) -> list[str]:
    metrics = plan.get("metrics", {}) if isinstance(plan, dict) else {}
    labels: list[str] = []
    if isinstance(metrics, dict):
        primary = metrics.get("primary")
        if isinstance(primary, str) and primary.strip():
            labels.append(primary.strip().lower())
        for item in metrics.get("secondary", []) or []:
            if isinstance(item, str) and item.strip():
                labels.append(item.strip().lower())
    return labels


def _has_any(corpus: str, keywords: set[str]) -> bool:
    return any(keyword in corpus for keyword in keywords)


def _has_benchmark_surface(plan: dict) -> bool:
    return bool((plan.get("baselines") or []) or (plan.get("datasets") or []))


def _has_multiple_method_components(method: dict, plan: dict) -> bool:
    explicit_ablations = plan.get("ablations") or []
    if len(explicit_ablations) >= 2:
        return True

    complexity_score = 0
    if method.get("type") in {"hybrid", "architecture", "training_procedure"}:
        complexity_score += 2
    if len(method.get("key_properties") or []) >= 2:
        complexity_score += 1
    if len(method.get("hyperparameters") or []) >= 3:
        complexity_score += 1
    if method.get("pseudocode"):
        complexity_score += 1
    return complexity_score >= 3


def infer_claim_type(insight: dict) -> str:
    plan = _load_json(insight.get("experimental_plan"), {})
    method = _load_json(insight.get("proposed_method"), {})
    corpus = _text_corpus(insight, plan, method)
    metrics = " ".join(_metric_labels(plan))

    if _has_any(corpus + " " + metrics, EFFICIENCY_KEYWORDS):
        return "efficiency"
    if _has_any(corpus, BENCHMARK_AUDIT_KEYWORDS) or insight.get("mechanism_type") in {"protocol_artifact", "claim_method_gap"}:
        return "benchmark_audit"
    if _has_any(corpus, META_ANALYSIS_KEYWORDS) and not method:
        return "meta_analysis"
    if int(insight.get("tier") or 0) == 1:
        return "mechanistic"
    if not method and not _has_benchmark_surface(plan):
        return "mechanistic"
    return "performance"


def build_evidence_plan(insight: dict) -> dict[str, Any]:
    plan = _load_json(insight.get("experimental_plan"), {})
    method = _load_json(insight.get("proposed_method"), {})
    claim_type = infer_claim_type(insight)
    corpus = _text_corpus(insight, plan, method)
    benchmark_surface = _has_benchmark_surface(plan)
    explicit_ablations = plan.get("ablations") or []
    multi_component = _has_multiple_method_components(method, plan)

    result: dict[str, Any] = {
        "claim_type": claim_type,
        "main_table": {
            "enabled": False,
            "priority": "skip",
            "reason": "",
            "baselines": plan.get("baselines", []),
            "datasets": plan.get("datasets", []),
            "metrics": plan.get("metrics", {}),
        },
        "ablation": {
            "enabled": False,
            "priority": "skip",
            "reason": "",
            "items": explicit_ablations,
        },
        "visualization": {
            "enabled": False,
            "priority": "skip",
            "reason": "",
            "items": [],
        },
        "required_evidence": [],
        "optional_evidence": [],
        "skip_reasons": {},
        "narrative_focus": [],
    }

    if claim_type == "performance":
        if benchmark_surface:
            result["main_table"].update(
                enabled=True,
                priority="required",
                reason="Core claim is comparative performance against baselines on named datasets.",
            )
        else:
            result["skip_reasons"]["main_table"] = "No explicit benchmark surface was specified."

        if explicit_ablations or multi_component:
            priority = "required" if explicit_ablations else "optional"
            reason = (
                "Experimental plan already names ablations for distinct method components."
                if explicit_ablations
                else "Method appears multi-component, so ablations are useful but not mandatory."
            )
            result["ablation"].update(enabled=True, priority=priority, reason=reason)
        else:
            result["skip_reasons"]["ablation"] = "Method does not look multi-component enough to justify a dedicated ablation section."

        if _has_any(corpus, TREND_VIS_KEYWORDS):
            result["visualization"].update(
                enabled=True,
                priority="optional",
                reason="Claim mentions trends or sweeps that are easier to read as figures than tables.",
                items=["metric_trajectory"],
            )
        else:
            result["skip_reasons"]["visualization"] = "Main comparative claim is table-first; no strong trend or sweep cue was detected."

        result["narrative_focus"] = ["comparative performance", "baseline gap", "decision-useful improvement"]

    elif claim_type == "efficiency":
        if benchmark_surface:
            result["main_table"].update(
                enabled=True,
                priority="required",
                reason="Efficiency claims still need baseline comparisons on concrete tasks.",
            )
        else:
            result["skip_reasons"]["main_table"] = "No benchmark surface was specified for task quality comparisons."

        if explicit_ablations or multi_component:
            result["ablation"].update(
                enabled=True,
                priority="optional",
                reason="Ablations can isolate which design choice buys the efficiency gain.",
            )
        else:
            result["skip_reasons"]["ablation"] = "Efficiency claim appears single-intervention; component ablations are optional."

        result["visualization"].update(
            enabled=True,
            priority="required",
            reason="Efficiency ideas need tradeoff or resource-profile figures, not just narrative description.",
            items=["resource_profile", "tradeoff_curve"],
        )
        result["narrative_focus"] = ["quality-efficiency tradeoff", "resource profile", "deployment relevance"]

    elif claim_type == "benchmark_audit":
        if benchmark_surface:
            result["main_table"].update(
                enabled=True,
                priority="optional",
                reason="Comparative tables may help summarize audit findings, but they are not always the primary evidence.",
            )
        else:
            result["skip_reasons"]["main_table"] = "Audit-style claim can be made without a traditional baseline table."

        result["skip_reasons"]["ablation"] = "Benchmark and protocol audits usually do not need module ablations."
        if _has_any(corpus, {"distribution", "breakdown", "variance", "error", "shift", "bias", "trend"}):
            result["visualization"].update(
                enabled=True,
                priority="optional",
                reason="Audit claims often benefit from breakdown or distribution figures.",
                items=["error_breakdown", "distribution_shift_summary"],
            )
        else:
            result["skip_reasons"]["visualization"] = "No explicit distributional or breakdown cue was detected."
        result["narrative_focus"] = ["evaluation validity", "protocol failure mode", "audit evidence"]

    elif claim_type == "mechanistic":
        if benchmark_surface:
            result["main_table"].update(
                enabled=True,
                priority="optional",
                reason="A small comparison table may support the mechanistic claim, but it is not always the center of the paper.",
            )
        else:
            result["skip_reasons"]["main_table"] = "Mechanistic claim does not require a conventional benchmark table by default."

        if explicit_ablations or multi_component:
            result["ablation"].update(
                enabled=True,
                priority="optional",
                reason="Ablations can help isolate the proposed mechanism if the implementation has multiple moving parts.",
            )
        else:
            result["skip_reasons"]["ablation"] = "No clear multi-part intervention detected for mechanism isolation."

        if _has_any(corpus, MECHANISTIC_VIS_KEYWORDS | TREND_VIS_KEYWORDS):
            result["visualization"].update(
                enabled=True,
                priority="optional",
                reason="Mechanistic claims are often easier to communicate with diagnostic or qualitative figures.",
                items=["mechanistic_diagnostic", "qualitative_cases"],
            )
        else:
            result["skip_reasons"]["visualization"] = "Mechanistic claim currently looks text-and-analysis heavy, not figure dependent."
        result["narrative_focus"] = ["mechanism explanation", "failure mode", "diagnostic evidence"]

    else:
        if benchmark_surface and len(plan.get("datasets", []) or []) >= 2:
            result["main_table"].update(
                enabled=True,
                priority="optional",
                reason="A summary table can help organize aggregated evidence across datasets.",
            )
        else:
            result["skip_reasons"]["main_table"] = "Meta-analysis style claim does not need a benchmark main table by default."

        result["skip_reasons"]["ablation"] = "Meta-analysis style claims do not usually require component ablations."
        if _has_any(corpus, TREND_VIS_KEYWORDS | {"aggregate", "spread", "synthesis"}):
            result["visualization"].update(
                enabled=True,
                priority="optional",
                reason="Trend or synthesis claims are often clearer with aggregate figures.",
                items=["trend_summary"],
            )
        else:
            result["skip_reasons"]["visualization"] = "No clear figure-first trend signal was detected."
        result["narrative_focus"] = ["synthesis", "trend summary", "evidence aggregation"]

    for name in ("main_table", "ablation", "visualization"):
        block = result[name]
        if block["enabled"]:
            target = "required_evidence" if block["priority"] == "required" else "optional_evidence"
            result[target].append(name)
        else:
            result["skip_reasons"].setdefault(name, block["reason"] or "Not selected for this idea.")

    return result


def summarize_evidence_plan(plan: dict[str, Any] | None) -> str:
    if not isinstance(plan, dict) or not plan:
        return "Adaptive evidence planning unavailable."

    lines = [
        f"Claim type: {plan.get('claim_type', 'unknown')}",
        f"Required evidence: {', '.join(plan.get('required_evidence') or []) or 'none'}",
        f"Optional evidence: {', '.join(plan.get('optional_evidence') or []) or 'none'}",
    ]
    for key in ("main_table", "ablation", "visualization"):
        block = plan.get(key) or {}
        status = "enabled" if block.get("enabled") else "disabled"
        reason = block.get("reason") or (plan.get("skip_reasons") or {}).get(key, "")
        if reason:
            lines.append(f"{key}: {status} — {reason}")
    return "\n".join(lines)


def wants_visualization(plan: dict[str, Any] | None) -> bool:
    if not isinstance(plan, dict):
        return True
    block = plan.get("visualization") or {}
    return bool(block.get("enabled"))
