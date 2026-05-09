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
