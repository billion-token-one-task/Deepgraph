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


def _label_key(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _labels_from_values(value: Any, *, keys: tuple[str, ...]) -> list[str]:
    rows = value if isinstance(value, list) else ([value] if value not in (None, "", "unknown") else [])
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if isinstance(row, dict):
            text = ""
            for key in keys:
                text = str(row.get(key) or "").strip()
                if text:
                    break
        else:
            text = str(row or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _label_matches(required: str, observed: list[str]) -> bool:
    req = _label_key(required)
    if not req:
        return True
    for item in observed:
        obs = _label_key(item)
        if obs and (req == obs or req in obs or obs in req):
            return True
    return False


def _summary_dataset_labels(summary: dict[str, Any]) -> list[str]:
    labels = _labels_from_values(summary.get("datasets"), keys=("name", "id", "dataset", "hf_dataset"))
    labels.extend(_labels_from_values(summary.get("dataset"), keys=("name", "id", "dataset", "hf_dataset")))
    per_dataset = summary.get("per_dataset_results") or summary.get("per_dataset_table")
    if isinstance(per_dataset, dict):
        labels.extend(str(key) for key in per_dataset.keys())
    elif isinstance(per_dataset, list):
        labels.extend(_labels_from_values(per_dataset, keys=("name", "id", "dataset", "hf_dataset")))
    out: list[str] = []
    seen: set[str] = set()
    for label in labels:
        key = label.lower()
        if label and key not in seen:
            seen.add(key)
            out.append(label)
    return out


def _summary_model_labels(summary: dict[str, Any]) -> list[str]:
    labels = _labels_from_values(summary.get("models"), keys=("name", "id", "model", "hf_model", "model_id"))
    labels.extend(_labels_from_values(summary.get("model"), keys=("name", "id", "model", "hf_model", "model_id")))
    env = summary.get("environment_report") if isinstance(summary.get("environment_report"), dict) else {}
    labels.extend(_labels_from_values(env.get("model_id"), keys=("name", "id", "model", "hf_model", "model_id")))
    out: list[str] = []
    seen: set[str] = set()
    for label in labels:
        key = label.lower()
        if label and key not in seen:
            seen.add(key)
            out.append(label)
    return out


def _summary_method_labels(summary: dict[str, Any], per_method: dict[str, Any]) -> list[str]:
    labels = [str(name) for name in per_method.keys()]
    labels.extend(_labels_from_values(summary.get("methods"), keys=("name", "method", "id")))
    out: list[str] = []
    seen: set[str] = set()
    for label in labels:
        key = label.lower()
        if label and key not in seen:
            seen.add(key)
            out.append(label)
    return out


def _summary_seed_count(summary: dict[str, Any]) -> int:
    seed_results = summary.get("seed_results") if isinstance(summary.get("seed_results"), list) else []
    for key in ("num_seeds", "seeds"):
        value = summary.get(key)
        if isinstance(value, list):
            return len(value)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = 0
        if parsed:
            return parsed
    return len(seed_results)


def _row_count(row: Any) -> int | None:
    if not isinstance(row, dict):
        return None
    for key in ("num_test", "num_materialized_examples", "num_examples", "count", "n", "size"):
        try:
            value = int(row.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return None


def _summary_dataset_counts(summary: dict[str, Any]) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    datasets = summary.get("datasets") if isinstance(summary.get("datasets"), list) else []
    dataset = summary.get("dataset") if isinstance(summary.get("dataset"), dict) else None
    rows = list(datasets)
    if dataset:
        rows.append(dataset)
    for row in rows:
        if not isinstance(row, dict):
            continue
        labels = _labels_from_values(row, keys=("name", "id", "dataset", "hf_dataset"))
        label = labels[0] if labels else "dataset"
        counts[label] = _row_count(row)
    per_dataset = summary.get("per_dataset_results") or summary.get("per_dataset_table")
    if isinstance(per_dataset, dict):
        for key, row in per_dataset.items():
            if isinstance(row, dict):
                counts.setdefault(str(key), _row_count(row))
    elif isinstance(per_dataset, list):
        for row in per_dataset:
            if not isinstance(row, dict):
                continue
            labels = _labels_from_values(row, keys=("name", "id", "dataset", "hf_dataset"))
            if labels:
                counts.setdefault(labels[0], _row_count(row))
    return counts


def _criteria_protocol(criteria: dict[str, Any], contract: dict[str, Any], quality_gates: dict[str, Any]) -> dict[str, Any]:
    for value in (
        criteria.get("benchmark_protocol"),
        contract.get("benchmark_protocol"),
        quality_gates.get("benchmark_protocol"),
    ):
        if isinstance(value, dict) and value:
            return value
    for source in (criteria, contract, quality_gates):
        manifest = source.get("benchmark_manifest") if isinstance(source.get("benchmark_manifest"), dict) else {}
        protocol = manifest.get("benchmark_protocol") if isinstance(manifest.get("benchmark_protocol"), dict) else {}
        if protocol:
            return protocol
    return {}


def _protocol_artifact_present(summary: dict[str, Any], artifact: str) -> bool:
    name = str(artifact or "").strip().lower()
    if not name:
        return True
    stem = name.removesuffix(".jsonl").removesuffix(".json")
    artifact_paths = summary.get("artifact_paths") if isinstance(summary.get("artifact_paths"), dict) else {}
    if artifact_paths:
        for key, value in artifact_paths.items():
            item = f"{key} {value}".lower()
            if stem in item or name in item:
                return True
    manifest = summary.get("artifact_manifest") or summary.get("benchmark_artifact_manifest")
    if isinstance(manifest, dict):
        raw_items = manifest.get("artifacts") or manifest.get("files") or manifest.get("required_artifacts") or []
        items = []
        if isinstance(raw_items, dict):
            items = list(raw_items.keys())
        elif isinstance(raw_items, list):
            for item in raw_items:
                if isinstance(item, dict):
                    items.append(str(item.get("name") or item.get("path") or item.get("artifact") or ""))
                else:
                    items.append(str(item or ""))
        if items and any(stem in item.lower() or name in item.lower() for item in items):
            return True
    checks = {
        "per_seed_results": bool(summary.get("seed_results") or summary.get("per_seed_results")),
        "per_dataset_results": bool(summary.get("per_dataset_results") or summary.get("per_dataset_table") or summary.get("datasets")),
        "main_results_table": bool(summary.get("main_results_table") or summary.get("per_method")),
        "ablation_table": bool(summary.get("ablation_table") or summary.get("ablations") or summary.get("ablation_results")),
        "latency_tokens_table": bool(summary.get("latency_tokens_table") or summary.get("latency_table") or summary.get("token_cost_table") or summary.get("cost_utility_tradeoff_table")),
        "route_rate_sweep_table": bool(summary.get("route_rate_sweep") or summary.get("route_rate_sweep_table") or summary.get("routing_sweep") or summary.get("budget_sweep")),
        "routing_analysis": bool(summary.get("routing_analysis")),
        "quality_cost_frontier": bool(summary.get("quality_cost_frontier") or summary.get("frontier_analysis")),
        "cost_utility_tradeoff_table": bool(summary.get("cost_utility_tradeoff_table") or summary.get("quality_cost_frontier")),
        "difficulty_breakdown_table": bool(summary.get("difficulty_breakdown_table") or summary.get("easy_medium_hard_breakdown") or summary.get("subset_analysis")),
        "simple_case_degradation": bool(summary.get("simple_case_degradation")),
        "calibration_reliability": bool(summary.get("calibration_reliability") or summary.get("reliability_table")),
        "bootstrap_ci": bool(summary.get("bootstrap_ci") or summary.get("statistical_tests") or summary.get("significance")),
        "run_config": bool(summary.get("run_config") or summary.get("config") or summary.get("budget")),
        "benchmark_summary": True,
        "raw_predictions": bool(summary.get("raw_predictions") or summary.get("raw_prediction_count") or summary.get("raw_predictions_path")),
        "routing_decisions": bool(summary.get("routing_decisions") or summary.get("routing_decision_count") or summary.get("routing_decisions_path") or summary.get("routing_analysis")),
        "failure_cases": bool(summary.get("failure_cases") or summary.get("failure_case_count") is not None),
        "artifact_manifest": isinstance(manifest, dict) and bool(manifest),
    }
    return bool(checks.get(stem, True))


def _protocol_evidence_blockers(
    summary: dict[str, Any],
    *,
    protocol: dict[str, Any],
    contract: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if summary.get("full_benchmark_completed") is False:
        blockers.append("benchmark_summary.full_benchmark_completed is false")
    if summary.get("load_failures"):
        blockers.append("benchmark_summary.load_failures is non-empty")

    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    if not per_method or len(per_method) < 2:
        blockers.append("benchmark_summary.per_method must contain at least two methods")

    requirements = protocol.get("full_benchmark_requirements") if isinstance(protocol.get("full_benchmark_requirements"), dict) else {}
    dataset_protocols = protocol.get("dataset_protocols") if isinstance(protocol.get("dataset_protocols"), list) else []
    required_datasets = requirements.get("required_dataset_names") if isinstance(requirements.get("required_dataset_names"), list) else []
    if not required_datasets:
        required_datasets = contract.get("required_real_benchmarks") or contract.get("required_datasets") or []
    required_datasets = [str(item.get("name") if isinstance(item, dict) else item).strip() for item in required_datasets if str(item.get("name") if isinstance(item, dict) else item).strip()]
    observed_datasets = _summary_dataset_labels(summary)
    if required_datasets and not observed_datasets:
        blockers.append("benchmark dataset coverage metadata is missing")
    missing_datasets = [name for name in required_datasets if not _label_matches(name, observed_datasets)]
    if missing_datasets:
        blockers.append("required benchmark coverage missing: " + ", ".join(missing_datasets))

    counts = _summary_dataset_counts(summary)
    total_examples = _summary_total_examples(summary)
    if dataset_protocols:
        for row in dataset_protocols:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or row.get("canonical_name") or row.get("hf_dataset") or "dataset").strip()
            matched_label = next((label for label in counts if _label_matches(name, [label])), "")
            count = counts.get(matched_label) if matched_label else None
            sample_policy = row.get("sample_policy") if isinstance(row.get("sample_policy"), dict) else {}
            try:
                expected = int(sample_policy.get("expected_examples") or 0)
            except (TypeError, ValueError):
                expected = 0
            if count is None:
                blockers.append(f"materialized example count missing for benchmark dataset {name}")
            elif count <= 0:
                blockers.append(f"benchmark dataset {name} has zero materialized examples")
            elif expected and count < expected:
                blockers.append(f"benchmark dataset {name} has {count} materialized examples; expected official/materialized count is {expected}")
    elif total_examples <= 0:
        blockers.append("materialized benchmark example count is missing")

    required_models = requirements.get("required_model_names") if isinstance(requirements.get("required_model_names"), list) else []
    if not required_models:
        required_models = contract.get("required_models") or []
    required_models = [str(item.get("name") if isinstance(item, dict) else item).strip() for item in required_models if str(item.get("name") if isinstance(item, dict) else item).strip()]
    observed_models = _summary_model_labels(summary)
    if required_models and not observed_models:
        blockers.append("benchmark model coverage metadata is missing")
    missing_models = [name for name in required_models if not _label_matches(name, observed_models)]
    if missing_models:
        blockers.append("required model coverage missing: " + ", ".join(missing_models))

    required_baselines = requirements.get("required_baseline_names") if isinstance(requirements.get("required_baseline_names"), list) else []
    if not required_baselines:
        required_baselines = contract.get("required_baselines") or []
    required_baselines = [str(item.get("name") if isinstance(item, dict) else item).strip() for item in required_baselines if str(item.get("name") if isinstance(item, dict) else item).strip()]
    observed_methods = _summary_method_labels(summary, per_method)
    missing_baselines = [name for name in required_baselines if not _label_matches(name, observed_methods)]
    if missing_baselines:
        blockers.append("required baselines missing: " + ", ".join(missing_baselines))

    seed_policy = protocol.get("seed_policy") if isinstance(protocol.get("seed_policy"), dict) else {}
    try:
        minimum_seeds = max(1, int(seed_policy.get("minimum_repeats") or contract.get("minimum_seeds") or 1))
    except (TypeError, ValueError):
        minimum_seeds = 1
    num_seeds = _summary_seed_count(summary)
    if num_seeds < minimum_seeds:
        blockers.append(f"num_seeds={num_seeds} is below benchmark protocol minimum_seeds={minimum_seeds}")

    required_ablations = requirements.get("required_ablations") if isinstance(requirements.get("required_ablations"), list) else []
    if not required_ablations:
        required_ablations = contract.get("required_ablations") or []
    if required_ablations and not _protocol_artifact_present(summary, "ablation_table.json"):
        blockers.append("required ablation table is missing")

    required_artifacts = requirements.get("required_artifacts") if isinstance(requirements.get("required_artifacts"), list) else []
    for artifact in required_artifacts:
        if not _protocol_artifact_present(summary, str(artifact)):
            blockers.append(f"required benchmark artifact/analysis missing: {artifact}")
    return blockers


def full_benchmark_evidence_blockers(summary: dict[str, Any] | None, criteria: dict[str, Any] | None = None) -> list[str]:
    """Hard policy for paper-eligible benchmark evidence.

    New experiment contracts may carry ``benchmark_protocol``. When present,
    that protocol is authoritative: official/materialized split coverage,
    benchmark-specific seeds, model/baseline lists, and required artifacts are
    checked without applying global example/model/baseline thresholds. Legacy
    summaries without a protocol retain the previous global fallback.
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

    require_significance = bool(quality_gates.get("require_statistical_significance", contract.get("require_statistical_significance", EXPERIMENT_FULL_BENCHMARK_REQUIRE_SIGNIFICANCE)))
    require_strongest_win = bool(quality_gates.get("require_strongest_baseline_win", contract.get("require_strongest_baseline_win", EXPERIMENT_FULL_BENCHMARK_REQUIRE_STRONGEST_WIN)))
    protocol = _criteria_protocol(criteria, contract, quality_gates)

    blockers: list[str] = []
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    if protocol:
        blockers.extend(_protocol_evidence_blockers(summary, protocol=protocol, contract=contract))
    else:
        min_examples = int(quality_gates.get("full_benchmark_min_examples") or contract.get("full_benchmark_min_examples") or EXPERIMENT_FULL_BENCHMARK_MIN_EXAMPLES)
        min_datasets = int(quality_gates.get("full_benchmark_min_datasets") or contract.get("full_benchmark_min_datasets") or EXPERIMENT_FULL_BENCHMARK_MIN_DATASETS)
        min_models = int(quality_gates.get("full_benchmark_min_models") or contract.get("full_benchmark_min_models") or EXPERIMENT_FULL_BENCHMARK_MIN_MODELS)
        min_baselines = int(quality_gates.get("full_benchmark_min_baselines") or contract.get("full_benchmark_min_baselines") or EXPERIMENT_FULL_BENCHMARK_MIN_BASELINES)

        total_examples = _summary_total_examples(summary)
        dataset_count = _summary_dataset_count(summary)
        model_count = _summary_model_count(summary)
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

    metric_name = str(summary.get("primary_metric") or summary.get("metric_name") or criteria.get("metric_name") or contract.get("primary_metric") or "metric_value")
    direction = str(criteria.get("metric_direction") or contract.get("metric_direction") or "higher")
    candidate, candidate_value, strongest_name, strongest_value, strongest_gap = _candidate_and_strongest(summary, metric_name, direction)
    candidate_text = str(summary.get("candidate_method") or candidate or "").lower()
    if any(token in candidate_text for token in ("route", "routing", "gate", "packet", "selector", "residual")):
        if not (summary.get("route_rate_sweep") or summary.get("routing_sweep") or summary.get("budget_sweep") or summary.get("route_rate_sweep_table")):
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
