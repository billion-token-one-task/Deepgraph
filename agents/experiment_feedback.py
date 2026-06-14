"""Benchmark-driven method feedback for experiment validation loops."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


_METHOD_KEYS = ("method", "name", "label", "variant", "system")
_VALUE_KEYS = (
    "metric",
    "accuracy",
    "score",
    "utility",
    "f1",
    "exact_match",
    "value",
    "mean",
    "candidate_metric",
)
_ORACLE_MARKERS = ("oracle", "upper_bound", "upper-bound", "upper bound")
_CANDIDATE_MARKERS = ("candidate", "target", "proposed", "ours", "method_under_test")
_BASELINE_MARKERS = ("baseline", "vanilla", "cot", "always", "direct", "reason")


def _read_json(path: Path) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _as_float(value: Any) -> float | None:
    if value in (None, "", []):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _metric_direction(criteria: dict[str, Any] | None) -> str:
    raw = str((criteria or {}).get("metric_direction") or (criteria or {}).get("direction") or "higher").lower()
    return "lower" if raw.startswith("low") or raw in {"min", "minimize"} else "higher"


def _effect(value: float | None, reference: float | None, direction: str) -> float | None:
    if value is None or reference is None:
        return None
    return reference - value if direction == "lower" else value - reference


def _beats(value: float | None, reference: float | None, direction: str, eps: float = 1e-12) -> bool | None:
    gap = _effect(value, reference, direction)
    if gap is None:
        return None
    return gap > eps


def _method_name(row: dict[str, Any]) -> str:
    for key in _METHOD_KEYS:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return "unknown"


def _method_value(row: dict[str, Any], metric_name: str | None = None) -> float | None:
    keys: list[str] = []
    if metric_name:
        keys.extend([metric_name, f"mean_{metric_name}", f"{metric_name}_mean"])
    keys.extend(_VALUE_KEYS)
    for key in keys:
        value = _as_float(row.get(key))
        if value is not None:
            return value
    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        if metric_name:
            value = _as_float(metrics.get(metric_name))
            if value is not None:
                return value
        for key in _VALUE_KEYS:
            value = _as_float(metrics.get(key))
            if value is not None:
                return value
    return None


def _rows_from_mapping(mapping: dict[str, Any], metric_name: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, value in mapping.items():
        if isinstance(value, dict):
            row = dict(value)
            row.setdefault("method", name)
            rows.append(row)
        else:
            metric = _as_float(value)
            if metric is not None:
                rows.append({"method": name, metric_name or "metric": metric})
    return rows


def _extract_method_rows(payload: Any, metric_name: str | None = None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    rows: list[dict[str, Any]] = []
    for key in ("per_method", "methods", "main_results", "results", "rows", "table", "leaderboard"):
        value = payload.get(key)
        if isinstance(value, list):
            rows.extend([dict(row) for row in value if isinstance(row, dict)])
        elif isinstance(value, dict):
            rows.extend(_rows_from_mapping(value, metric_name))
    for key in ("candidate", "candidate_method", "target", "target_method"):
        value = payload.get(key)
        if isinstance(value, dict):
            row = dict(value)
            row.setdefault("method", key)
            rows.append(row)
        elif isinstance(value, str):
            metric = _as_float(payload.get("candidate_metric") or payload.get("target_metric"))
            rows.append({"method": value, metric_name or "metric": metric})
    return rows


def _load_benchmark_payloads(workdir: Path) -> dict[str, Any]:
    results_dir = workdir / "results"
    return {
        "benchmark_summary": _read_json(results_dir / "benchmark_summary.json") or {},
        "main_results_table": _read_json(results_dir / "main_results_table.json") or {},
        "routing_analysis": _read_json(results_dir / "routing_analysis.json") or {},
        "ablation_table": _read_json(results_dir / "ablation_table.json") or {},
    }


def _best_row(rows: list[dict[str, Any]], direction: str, metric_name: str | None, *, include_oracle: bool) -> dict[str, Any] | None:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        name = _method_name(row).lower()
        if not include_oracle and any(marker in name for marker in _ORACLE_MARKERS):
            continue
        value = _method_value(row, metric_name)
        if value is None:
            continue
        candidates.append((value, row))
    if not candidates:
        return None
    reverse = direction != "lower"
    return sorted(candidates, key=lambda item: item[0], reverse=reverse)[0][1]


def _candidate_row(rows: list[dict[str, Any]], summary: dict[str, Any], metric_name: str | None) -> dict[str, Any] | None:
    declared = str(summary.get("candidate_method") or summary.get("target_method") or "").strip().lower()
    for row in rows:
        name = _method_name(row).lower()
        role = str(row.get("role") or "").lower()
        if declared and name == declared:
            return row
        if role in {"candidate", "target", "proposed"}:
            return row
        if any(marker in name for marker in _CANDIDATE_MARKERS):
            return row
    metric = _as_float(summary.get("candidate_metric") or summary.get("target_metric"))
    if metric is not None:
        return {"method": summary.get("candidate_method") or "candidate", metric_name or "metric": metric}
    return None


def _oracle_row(rows: list[dict[str, Any]], direction: str, metric_name: str | None) -> dict[str, Any] | None:
    oracle_rows = [row for row in rows if any(marker in _method_name(row).lower() for marker in _ORACLE_MARKERS)]
    return _best_row(oracle_rows, direction, metric_name, include_oracle=True)


def _method_diagnosis(workdir: Path, result: dict[str, Any] | None, criteria: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    payloads = _load_benchmark_payloads(workdir)
    summary = dict(payloads.get("benchmark_summary") or {})
    if isinstance((result or {}).get("benchmark_summary"), dict):
        summary.update((result or {}).get("benchmark_summary") or {})
    metric_name = str((criteria or {}).get("metric_name") or summary.get("metric_name") or "metric")
    direction = _metric_direction(criteria)
    rows: list[dict[str, Any]] = []
    rows.extend(_extract_method_rows(summary, metric_name))
    rows.extend(_extract_method_rows(payloads.get("main_results_table"), metric_name))

    candidate = _candidate_row(rows, summary, metric_name)
    best_non_oracle = _best_row(rows, direction, metric_name, include_oracle=False)
    if candidate is not None and best_non_oracle is not None and _method_name(candidate) == _method_name(best_non_oracle):
        non_candidate = [row for row in rows if _method_name(row) != _method_name(candidate)]
        best_non_oracle = _best_row(non_candidate, direction, metric_name, include_oracle=False) or best_non_oracle
    oracle = _oracle_row(rows, direction, metric_name)

    candidate_value = _method_value(candidate, metric_name) if candidate else None
    best_value = _method_value(best_non_oracle, metric_name) if best_non_oracle else None
    oracle_value = _method_value(oracle, metric_name) if oracle else None
    return (
        {
            "metric_name": metric_name,
            "direction": direction,
            "candidate_method": _method_name(candidate) if candidate else "",
            "candidate_value": candidate_value,
            "best_non_oracle_method": _method_name(best_non_oracle) if best_non_oracle else "",
            "best_non_oracle_value": best_value,
            "oracle_method": _method_name(oracle) if oracle else "",
            "oracle_value": oracle_value,
            "candidate_minus_best_non_oracle": _effect(candidate_value, best_value, direction),
            "beats_best_non_oracle": _beats(candidate_value, best_value, direction),
            "method_rows_seen": len(rows),
        },
        payloads,
    )


def _routing_findings(routing: Any) -> tuple[list[str], list[str]]:
    findings: list[str] = []
    actions: list[str] = []
    if not isinstance(routing, dict):
        return findings, actions
    rate = _as_float(
        routing.get("route_rate")
        or routing.get("candidate_route_rate")
        or routing.get("reasoning_route_rate")
        or routing.get("selected_fraction")
    )
    if rate is None:
        counts = routing.get("counts")
        if isinstance(counts, dict):
            total = sum(int(v or 0) for v in counts.values())
            selected = sum(int(v or 0) for k, v in counts.items() if any(marker in str(k).lower() for marker in _CANDIDATE_MARKERS))
            if total > 0:
                rate = selected / total
    if rate is not None:
        if rate < 0.05:
            findings.append(f"Candidate routing is almost never used (route_rate={rate:.3f}).")
            actions.append("Recalibrate or replace the gate so the candidate method is exercised on enough eligible examples before judging quality.")
        elif rate > 0.95:
            findings.append(f"Candidate routing is effectively always-on (route_rate={rate:.3f}).")
            actions.append("Add a cheap confidence/easy-case gate or fallback path so the method does not degrade simple examples.")
    return findings, actions


def _statistical_findings(summary: dict[str, Any]) -> tuple[list[str], list[str]]:
    findings: list[str] = []
    actions: list[str] = []
    p_value = _as_float(summary.get("paired_bootstrap_p") or summary.get("bootstrap_p") or summary.get("p_value"))
    if p_value is not None and p_value > 0.05:
        findings.append(f"Candidate effect is not statistically reliable yet (p={p_value:.4g}).")
        actions.append("Make a mechanism-level method change and rerun the same seeds; do not paper over this with reporting-only edits.")
    ci_low = _as_float(summary.get("ci_low") or summary.get("bootstrap_ci_low"))
    ci_high = _as_float(summary.get("ci_high") or summary.get("bootstrap_ci_high"))
    if ci_low is not None and ci_high is not None and ci_low <= 0 <= ci_high:
        findings.append(f"Bootstrap confidence interval crosses zero ([{ci_low:.4g}, {ci_high:.4g}]).")
        actions.append("Increase method effect size before treating the benchmark as supportive evidence.")
    return findings, actions


def _artifact_sources(workdir: Path) -> dict[str, str]:
    results_dir = workdir / "results"
    names = ["benchmark_summary.json", "main_results_table.json", "routing_analysis.json", "failure_cases.jsonl", "ablation_table.json"]
    return {name: str(results_dir / name) for name in names if (results_dir / name).exists()}


def build_method_feedback(
    *,
    workdir: Path,
    run_id: int,
    iteration: int | None,
    result: dict[str, Any] | None,
    result_judgement: dict[str, Any] | None,
    history: list[dict[str, Any]],
    criteria: dict[str, Any] | None,
    baseline: float | None,
    best_value: float | None,
) -> dict[str, Any]:
    """Create actionable method feedback from benchmark artifacts and loop status."""
    judgement = result_judgement or {}
    status = str(judgement.get("status") or (result or {}).get("status") or "unknown")
    anomaly_type = str(judgement.get("anomaly_type") or "")
    diagnosis, payloads = _method_diagnosis(workdir, result, criteria)
    summary = payloads.get("benchmark_summary") if isinstance(payloads.get("benchmark_summary"), dict) else {}

    findings: list[str] = []
    actions: list[str] = []
    guardrails: list[str] = [
        "Do not change scoring, answer normalization, dataset splits, seeds, baselines, or oracle labels to manufacture an improvement.",
        "Do not use an oracle/upper-bound method as candidate evidence; compare against the best non-oracle baseline.",
        "Sanity/probe/smoke runs may validate plumbing only and must not be promoted to manuscript evidence.",
    ]

    if anomaly_type in {"no_candidate_diff", "pre_benchmark_guard"}:
        findings.append("This iteration did not produce a benchmarked candidate method change; it is an automation failure, not a scientific refutation.")
        actions.append("Locate the missing method hook and make a real tracked source/config change before another benchmark run.")
        actions.append("If the benchmark contract cannot execute the method, write an explicit harness redesign artifact instead of returning a no-op.")

    candidate_value = diagnosis.get("candidate_value")
    best_non_oracle_value = diagnosis.get("best_non_oracle_value")
    gap = diagnosis.get("candidate_minus_best_non_oracle")
    beats_best = diagnosis.get("beats_best_non_oracle")
    if beats_best is False and candidate_value is not None and best_non_oracle_value is not None:
        findings.append(
            "Candidate trails the best non-oracle method "
            f"({diagnosis.get('candidate_method') or 'candidate'}={candidate_value:.6g}, "
            f"{diagnosis.get('best_non_oracle_method') or 'best_non_oracle'}={best_non_oracle_value:.6g}, "
            f"effect={gap:.6g})."
        )
        actions.append("Change the candidate method path, routing threshold, fallback policy, or prompt/mechanism that caused the measured deficit.")
        actions.append("Run the same benchmark contract after the method change so the next result is comparable.")
    elif beats_best is True:
        findings.append("Candidate currently beats the best non-oracle comparator; next work should preserve the effect and expand evidence quality.")
        actions.append("Freeze the winning mechanism and complete required seeds, ablations, latency/token tables, and failure-case analysis.")
    elif candidate_value is None:
        findings.append("No candidate metric was found in benchmark artifacts.")
        actions.append("Repair the runner output contract so candidate_method, per_method metrics, and FINAL_RESULTS are emitted every run.")

    routing_findings, routing_actions = _routing_findings(payloads.get("routing_analysis") or summary.get("routing_analysis"))
    findings.extend(routing_findings)
    actions.extend(routing_actions)
    stat_findings, stat_actions = _statistical_findings(summary)
    findings.extend(stat_findings)
    actions.extend(stat_actions)

    if not actions:
        actions.append("Inspect benchmark artifacts and propose one concrete method-side change before the next iteration.")
    if best_value is not None and baseline is not None:
        direction = diagnosis.get("direction") or _metric_direction(criteria)
        best_effect = _effect(best_value, baseline, direction)
        if best_effect is not None and best_effect <= 0:
            findings.append(f"Best value still does not beat baseline (baseline={baseline:.6g}, best={best_value:.6g}, effect={best_effect:.6g}).")
            actions.append("Prioritize closing the baseline gap before any manuscript-writing or evidence packaging step.")

    return {
        "schema_version": "deepgraph_method_feedback_v1",
        "run_id": run_id,
        "iteration": iteration,
        "status": status,
        "anomaly_type": anomaly_type,
        "baseline": baseline,
        "best_value": best_value,
        "method_diagnosis": diagnosis,
        "findings": list(dict.fromkeys(findings))[:12],
        "next_actions": list(dict.fromkeys(actions))[:12],
        "guardrails": list(dict.fromkeys(guardrails))[:12],
        "artifact_sources": _artifact_sources(workdir),
        "history_tail": history[-5:],
    }


def write_method_feedback(workdir: Path, payload: dict[str, Any]) -> Path:
    """Persist method feedback and update a latest pointer for the next supervisor turn."""
    iteration = payload.get("iteration")
    if isinstance(iteration, int) and iteration > 0:
        name = f"iter_{iteration:03d}_method_feedback.json"
    else:
        name = "final_method_feedback.json"
    path = workdir / "results" / "method_feedback" / name
    _write_json(path, payload)
    latest = workdir / "results" / "method_feedback" / "latest_method_feedback.json"
    _write_json(latest, payload)
    return path


def load_latest_method_feedback(workdir: Path) -> dict[str, Any] | None:
    payload = _read_json(workdir / "results" / "method_feedback" / "latest_method_feedback.json")
    return payload if isinstance(payload, dict) else None
