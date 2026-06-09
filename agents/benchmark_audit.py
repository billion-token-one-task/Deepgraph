"""Benchmark semantic checks shared by validation and manuscript handoff."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _method_metric(row: dict[str, Any], metric_name: str | None) -> float | None:
    if metric_name:
        value = _as_float(row.get(metric_name))
        if value is not None:
            return value
    return _as_float(row.get("metric_value"))


def _is_upper_bound_method(name: str, row: dict[str, Any]) -> bool:
    label = name.replace("-", "_").replace(" ", "_").lower()
    return bool(row.get("upper_bound")) or "upper_bound" in label or "oracle_router" in label


def benchmark_semantic_warnings(
    summary: dict[str, Any] | None,
    *,
    metric_name: str | None = None,
    candidate_method: str | None = None,
    direction: str = "higher",
) -> list[str]:
    """Return warnings for internally inconsistent benchmark semantics.

    This does not decide whether an experiment improved. It catches cases that
    should not be promoted into paper claims without explanation, such as a
    candidate beating a method recorded as an upper bound.
    """
    if not isinstance(summary, dict):
        return []
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    if not per_method:
        return []
    metric = metric_name or str(summary.get("primary_metric") or summary.get("metric_name") or "metric_value")
    candidate = str(candidate_method or summary.get("candidate_method") or "").strip()
    if not candidate:
        return []
    candidate_row = per_method.get(candidate)
    if not isinstance(candidate_row, dict):
        return []
    candidate_value = _method_metric(candidate_row, metric)
    if candidate_value is None:
        return []

    higher = str(direction or "higher").lower() != "lower"
    warnings: list[str] = []
    for method_name, row in per_method.items():
        if method_name == candidate or not isinstance(row, dict):
            continue
        if not _is_upper_bound_method(str(method_name), row):
            continue
        reference_value = _method_metric(row, metric)
        if reference_value is None:
            continue
        violates = candidate_value > reference_value + 1e-12 if higher else candidate_value < reference_value - 1e-12
        if not violates:
            continue
        delta = candidate_value - reference_value if higher else reference_value - candidate_value
        pct = (delta / abs(reference_value) * 100.0) if abs(reference_value) > 1e-12 else None
        pct_text = f", {pct:+.2f}%" if pct is not None else ""
        warnings.append(
            "Candidate method "
            f"{candidate}={candidate_value:.6f} exceeds benchmark method marked as upper_bound "
            f"{method_name}={reference_value:.6f} (delta {delta:+.6f}{pct_text}). "
            "Treat that comparator as a scoped oracle diagnostic or fix the benchmark before paper claims."
        )
    return warnings


def benchmark_diagnostic_notes(
    summary: dict[str, Any] | None,
    *,
    metric_name: str | None = None,
    candidate_method: str | None = None,
    direction: str = "higher",
) -> list[str]:
    """Return non-blocking notes about benchmark diagnostics.

    These are not semantic failures. They preserve context that a manuscript or
    evidence audit should describe explicitly, such as a candidate tying an
    oracle diagnostic on a small benchmark slice.
    """
    if not isinstance(summary, dict):
        return []
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    if not per_method:
        return []
    metric = metric_name or str(summary.get("primary_metric") or summary.get("metric_name") or "metric_value")
    candidate = str(candidate_method or summary.get("candidate_method") or "").strip()
    if not candidate:
        return []
    candidate_row = per_method.get(candidate)
    if not isinstance(candidate_row, dict):
        return []
    candidate_value = _method_metric(candidate_row, metric)
    if candidate_value is None:
        return []

    higher = str(direction or "higher").lower() != "lower"
    notes: list[str] = []
    for method_name, row in per_method.items():
        if method_name == candidate or not isinstance(row, dict):
            continue
        if not _is_upper_bound_method(str(method_name), row):
            continue
        reference_value = _method_metric(row, metric)
        if reference_value is None:
            continue
        ties = abs(candidate_value - reference_value) <= 1e-12
        if not ties:
            continue
        better_word = "higher" if higher else "lower"
        notes.append(
            "Candidate method "
            f"{candidate}={candidate_value:.6f} ties benchmark method marked as upper_bound "
            f"{method_name}={reference_value:.6f}. Treat this as a diagnostic ceiling tie, "
            f"not evidence that the candidate is {better_word} than the oracle comparator."
        )
    return notes


def benchmark_fairness_warnings_from_diff(diff_text: str | None) -> list[str]:
    """Flag code diffs that may make benchmark scoring unfair."""
    text = str(diff_text or "")
    if not text:
        return []
    added = "\n".join(line[1:] for line in text.splitlines() if line.startswith("+") and not line.startswith("+++"))
    lower = added.lower()
    warnings: list[str] = []
    touches_candidate = "cggr" in lower or "candidate" in lower
    touches_scoring_surface = any(
        marker in lower
        for marker in (
            "canonicaliz",
            "normaliz",
            "_score_answer",
            "_extract_final_answer",
            "before scoring",
            "score_answer",
        )
    )
    candidate_only_signal = any(
        marker in lower
        for marker in (
            "candidate-only",
            "_cggr_canonical",
            "if method_name",
            "method_name ==",
            "method_name.startswith",
        )
    )
    if touches_candidate and touches_scoring_surface and candidate_only_signal:
        warnings.append(
            "Code diff appears to add candidate-specific scoring, answer normalization, or postprocessing. "
            "Benchmark evidence is not paper-ready unless the same evaluator-side normalization is applied to all methods "
            "or the change is justified as a pre-registered method component."
        )
    return warnings



def _summary_total_examples(summary: dict[str, Any]) -> int:
    total = 0
    datasets = summary.get("datasets") if isinstance(summary.get("datasets"), list) else []
    dataset = summary.get("dataset") if isinstance(summary.get("dataset"), dict) else {}
    rows = datasets or ([dataset] if dataset else [])
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in ("num_test", "num_materialized_examples", "num_examples", "count", "n"):
            try:
                value = int(row.get(key) or 0)
            except (TypeError, ValueError):
                value = 0
            if value:
                total += value
                break
    if total:
        return total
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    counts = []
    for row in per_method.values():
        if not isinstance(row, dict):
            continue
        for key in ("count", "n", "num_examples"):
            try:
                value = int(row.get(key) or 0)
            except (TypeError, ValueError):
                value = 0
            if value:
                counts.append(value)
                break
    return max(counts) if counts else 0


def _summary_model_count(summary: dict[str, Any]) -> int:
    models = summary.get("models") if isinstance(summary.get("models"), list) else []
    if models:
        names = set()
        for row in models:
            if isinstance(row, dict):
                text = str(row.get("name") or row.get("hf_model") or row.get("model") or row).strip().lower()
            else:
                text = str(row or "").strip().lower()
            if text:
                names.add(text)
        return len(names)
    model = summary.get("model") if isinstance(summary.get("model"), dict) else {}
    return 1 if model else 0


def _summary_dataset_count(summary: dict[str, Any]) -> int:
    datasets = summary.get("datasets") if isinstance(summary.get("datasets"), list) else []
    if datasets:
        names = set()
        for row in datasets:
            if isinstance(row, dict):
                text = str(row.get("name") or row.get("hf_dataset") or row.get("dataset") or row).strip().lower()
            else:
                text = str(row or "").strip().lower()
            if text:
                names.add(text)
        return len(names)
    dataset = summary.get("dataset") if isinstance(summary.get("dataset"), dict) else {}
    return 1 if dataset else 0


def _extract_p_value(summary: dict[str, Any]) -> float | None:
    sources = []
    for key in ("bootstrap_ci", "statistical_tests", "significance", "pairwise_tests"):
        value = summary.get(key)
        if isinstance(value, dict):
            sources.append(value)
    sources.append(summary)
    for source in sources:
        for key in ("p_value", "paired_permutation_p", "p", "p_vs_strongest"):
            parsed = _as_float(source.get(key))
            if parsed is not None:
                return parsed
    return None


def _candidate_and_strongest(summary: dict[str, Any], metric_name: str | None, direction: str) -> tuple[str, float | None, str, float | None, float | None]:
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    candidate = str(summary.get("candidate_method") or "").strip()
    if not candidate or candidate not in per_method:
        candidate = next((name for name in per_method if any(t in name.lower() for t in ("ours", "candidate", "proposed", "crpp", "cggr"))), "")
    candidate_row = per_method.get(candidate) if candidate in per_method else {}
    candidate_value = _method_metric(candidate_row if isinstance(candidate_row, dict) else {}, metric_name)
    higher = str(direction or "higher").lower() != "lower"
    best_name = ""
    best_value: float | None = None
    for name, row in per_method.items():
        if name == candidate or not isinstance(row, dict) or _is_upper_bound_method(str(name), row):
            continue
        value = _method_metric(row, metric_name)
        if value is None:
            continue
        if best_value is None or (value > best_value if higher else value < best_value):
            best_name, best_value = str(name), value
    gap = None
    if candidate_value is not None and best_value is not None:
        gap = candidate_value - best_value if higher else best_value - candidate_value
    return candidate, candidate_value, best_name, best_value, gap


def full_benchmark_evidence_blockers(summary: dict[str, Any] | None, criteria: dict[str, Any] | None = None) -> list[str]:
    """Hard policy for paper-eligible benchmark evidence.

    A small sanity slice can help debug code, but cannot set
    full_benchmark_completed=true or support a complete manuscript.
    """
    if not isinstance(summary, dict) or not summary:
        return ["full benchmark summary is missing"]
    criteria = criteria or {}
    contract = criteria.get("publication_evidence_contract") if isinstance(criteria.get("publication_evidence_contract"), dict) else {}
    quality_gates = criteria.get("quality_gates") if isinstance(criteria.get("quality_gates"), dict) else {}
    try:
        from config import (
            EXPERIMENT_FULL_BENCHMARK_MIN_BASELINES,
            EXPERIMENT_FULL_BENCHMARK_MIN_DATASETS,
            EXPERIMENT_FULL_BENCHMARK_MIN_EXAMPLES,
            EXPERIMENT_FULL_BENCHMARK_MIN_MODELS,
            EXPERIMENT_FULL_BENCHMARK_REQUIRE_SIGNIFICANCE,
            EXPERIMENT_FULL_BENCHMARK_REQUIRE_STRONGEST_WIN,
        )
    except Exception:
        EXPERIMENT_FULL_BENCHMARK_MIN_BASELINES = 6
        EXPERIMENT_FULL_BENCHMARK_MIN_DATASETS = 2
        EXPERIMENT_FULL_BENCHMARK_MIN_EXAMPLES = 1000
        EXPERIMENT_FULL_BENCHMARK_MIN_MODELS = 2
        EXPERIMENT_FULL_BENCHMARK_REQUIRE_SIGNIFICANCE = True
        EXPERIMENT_FULL_BENCHMARK_REQUIRE_STRONGEST_WIN = True

    min_examples = int(quality_gates.get("full_benchmark_min_examples") or contract.get("full_benchmark_min_examples") or EXPERIMENT_FULL_BENCHMARK_MIN_EXAMPLES)
    min_datasets = int(quality_gates.get("full_benchmark_min_datasets") or contract.get("full_benchmark_min_datasets") or EXPERIMENT_FULL_BENCHMARK_MIN_DATASETS)
    min_models = int(quality_gates.get("full_benchmark_min_models") or contract.get("full_benchmark_min_models") or EXPERIMENT_FULL_BENCHMARK_MIN_MODELS)
    min_baselines = int(quality_gates.get("full_benchmark_min_baselines") or contract.get("full_benchmark_min_baselines") or EXPERIMENT_FULL_BENCHMARK_MIN_BASELINES)
    require_significance = bool(quality_gates.get("require_statistical_significance", EXPERIMENT_FULL_BENCHMARK_REQUIRE_SIGNIFICANCE))
    require_strongest_win = bool(quality_gates.get("require_strongest_baseline_win", EXPERIMENT_FULL_BENCHMARK_REQUIRE_STRONGEST_WIN))

    blockers: list[str] = []
    total_examples = _summary_total_examples(summary)
    dataset_count = _summary_dataset_count(summary)
    model_count = _summary_model_count(summary)
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    deployable_method_count = len([name for name, row in per_method.items() if isinstance(row, dict) and not _is_upper_bound_method(str(name), row)])
    if total_examples < min_examples:
        blockers.append(f"full benchmark has only {total_examples} examples; minimum paper-eligible total is {min_examples}")
    if dataset_count < min_datasets:
        blockers.append(f"full benchmark covers only {dataset_count} dataset(s); minimum is {min_datasets}")
    if model_count < min_models:
        blockers.append(f"full benchmark covers only {model_count} model(s); minimum is {min_models}")
    if max(0, deployable_method_count - 1) < min_baselines:
        blockers.append(f"full benchmark has only {max(0, deployable_method_count - 1)} deployable baseline(s); minimum is {min_baselines}")

    required_analysis = {
        "ablation_table": bool(summary.get("ablation_table") or summary.get("ablations") or summary.get("ablation_results")),
        "cost_quality_frontier": bool(summary.get("quality_cost_frontier") or summary.get("frontier_analysis") or summary.get("cost_utility_tradeoff_table")),
        "per_dataset_breakdown": bool(summary.get("per_dataset_results") or summary.get("per_dataset_table") or (isinstance(summary.get("datasets"), list) and len(summary.get("datasets") or []) >= 2)),
        "difficulty_breakdown": bool(summary.get("difficulty_breakdown_table") or summary.get("easy_medium_hard_breakdown") or summary.get("subset_analysis")),
    }
    for name, present in required_analysis.items():
        if not present:
            blockers.append(f"required benchmark analysis missing: {name}")

    metric_name = str(summary.get("primary_metric") or summary.get("metric_name") or criteria.get("metric_name") or "metric_value")
    direction = str(criteria.get("metric_direction") or contract.get("metric_direction") or "higher")
    candidate, candidate_value, strongest_name, strongest_value, strongest_gap = _candidate_and_strongest(summary, metric_name, direction)
    candidate_text = str(summary.get("candidate_method") or candidate or "").lower()
    if any(token in candidate_text for token in ("route", "routing", "gate", "packet", "selector", "residual")):
        if not (summary.get("route_rate_sweep") or summary.get("routing_sweep") or summary.get("budget_sweep")):
            blockers.append("routing/selective method is missing route-rate or budget sweep")
        if not summary.get("routing_analysis"):
            blockers.append("routing/selective method is missing routing_analysis")

    if require_strongest_win:
        if strongest_name and strongest_gap is not None:
            if strongest_gap <= 0:
                blockers.append(f"candidate {candidate or 'method'} does not beat strongest deployable baseline {strongest_name}: gap {strongest_gap:+.6g}")
        else:
            blockers.append("strongest deployable baseline comparison is unavailable")
    if require_significance:
        p_value = _extract_p_value(summary)
        if p_value is None:
            blockers.append("statistical significance test p-value is missing")
        elif p_value >= 0.05:
            blockers.append(f"statistical significance failed: p={p_value:.4g} >= 0.05")
    return blockers

def best_iteration_benchmark_summary(
    workdir: str | Path | None,
    *,
    best_metric: float | None = None,
    direction: str = "higher",
) -> dict[str, Any]:
    """Load the benchmark summary from the best kept hypothesis iteration."""
    if not workdir:
        return {}
    packet_dir = Path(workdir) / "results" / "iteration_packets"
    if not packet_dir.is_dir():
        return {}
    higher = str(direction or "higher").lower() != "lower"
    selected_metric: float | None = None
    selected_summary: dict[str, Any] = {}
    for path in sorted(packet_dir.glob("hypothesis_testing_*.json")):
        try:
            packet = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(packet, dict) or packet.get("status") != "keep":
            continue
        metric = _as_float(packet.get("metric_value"))
        execution = packet.get("execution_report") if isinstance(packet.get("execution_report"), dict) else {}
        summary = execution.get("benchmark_summary") if isinstance(execution.get("benchmark_summary"), dict) else {}
        if metric is None or not summary:
            continue
        if best_metric is not None and abs(metric - best_metric) <= 1e-12:
            return summary
        if selected_metric is None or (metric > selected_metric if higher else metric < selected_metric):
            selected_metric = metric
            selected_summary = summary
    return selected_summary
