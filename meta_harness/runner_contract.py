"""Backend-neutral runner lifecycle, FINAL_RESULTS, and metric verification."""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class RunnerContractError(RuntimeError):
    def __init__(self, reason_code: str, detail: str | None = None):
        self.reason_code = reason_code
        super().__init__(reason_code if not detail else f"{reason_code}:{detail}")


class ResearchRunner(ABC):
    @abstractmethod
    def prepare(self) -> None: ...

    @abstractmethod
    def load_dataset(self) -> Any: ...

    @abstractmethod
    def load_model(self) -> Any: ...

    @abstractmethod
    def run_baseline(self) -> Sequence[Mapping[str, Any]]: ...

    @abstractmethod
    def run_candidate(self) -> Sequence[Mapping[str, Any]]: ...

    @abstractmethod
    def compute_metrics(self) -> Mapping[str, Any]: ...

    @abstractmethod
    def emit_final_results(self) -> Mapping[str, Any]: ...


REQUIRED_ARTIFACTS = (
    "final_results",
    "raw_predictions",
    "environment_manifest",
    "dataset_manifest",
    "model_manifest",
)

# Where a p-value may live in FINAL_RESULTS, and under which names. This is the
# single lookup rule: agents/benchmark_audit.py delegates here rather than
# keeping its own copy, because the evidence gate refuses a positive verdict
# without a p-value while the runner contract used to demand only a metric. A
# run could therefore spend its whole budget, produce a clean metric, and only
# discover at the last gate that `supported` was never reachable.
P_VALUE_CONTAINERS = ("bootstrap_ci", "statistical_tests", "significance", "pairwise_tests")
P_VALUE_KEYS = ("p_value", "paired_permutation_p", "p", "p_vs_strongest")

DEFAULT_PERMUTATIONS = 1000


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def extract_p_value(payload: Mapping[str, Any]) -> float | None:
    """Find the significance test result a runner reported, if any."""
    sources: list[Mapping[str, Any]] = []
    for key in P_VALUE_CONTAINERS:
        value = payload.get(key)
        if isinstance(value, Mapping):
            sources.append(value)
    sources.append(payload)
    for source in sources:
        for key in P_VALUE_KEYS:
            parsed = _as_float(source.get(key))
            if parsed is not None:
                return parsed
    return None


def _pair_key(row: Mapping[str, Any]) -> tuple:
    return (row.get("seed"), row.get("sample_index"))


def paired_permutation_test(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    metric_name: str,
    *,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> dict[str, Any]:
    """Two-sided paired permutation test on the metric the run reports.

    The arms are paired on (seed, sample_index), which the runner already
    stamps on every prediction, so the same example under both methods lines up
    exactly. Each permutation swaps a random subset of pairs between arms and
    recomputes the metric from scratch, which keeps the test correct for
    corpus-level metrics such as macro-F1 that do not decompose per example.

    p is computed with the add-one correction, so an observed difference that no
    permutation matches reports 1/(permutations+1) rather than an impossible 0.
    """
    import random

    if permutations < 1:
        raise RunnerContractError("permutation_contract_violation", "permutations")
    baseline_by_key = {_pair_key(row): row for row in baseline_rows}
    candidate_by_key = {_pair_key(row): row for row in candidate_rows}
    keys = sorted(
        baseline_by_key.keys() & candidate_by_key.keys(),
        key=lambda item: tuple("" if part is None else str(part) for part in item),
    )
    if not keys:
        raise RunnerContractError("permutation_contract_violation", "no_paired_examples")

    paired_baseline = [baseline_by_key[key] for key in keys]
    paired_candidate = [candidate_by_key[key] for key in keys]
    observed = recompute_metric(paired_candidate, metric_name) - recompute_metric(
        paired_baseline, metric_name
    )

    rng = random.Random(seed)
    at_least_as_extreme = 0
    for _ in range(permutations):
        left: list[Mapping[str, Any]] = []
        right: list[Mapping[str, Any]] = []
        for base_row, cand_row in zip(paired_baseline, paired_candidate):
            if rng.random() < 0.5:
                left.append(cand_row)
                right.append(base_row)
            else:
                left.append(base_row)
                right.append(cand_row)
        difference = recompute_metric(right, metric_name) - recompute_metric(left, metric_name)
        if abs(difference) >= abs(observed) - 1e-12:
            at_least_as_extreme += 1

    return {
        "paired_permutation_p": (at_least_as_extreme + 1) / (permutations + 1),
        "observed_difference": observed,
        "n_pairs": len(keys),
        "permutations": permutations,
        "seed": seed,
        "test": "two_sided_paired_permutation",
        "metric_name": metric_name,
    }


@dataclass(frozen=True)
class MetricVerification:
    metric_name: str
    direction: str
    baseline_value: float
    candidate_value: float
    recomputed_baseline: float
    recomputed_candidate: float


def _finite(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RunnerContractError("metric_missing", label) from exc
    if not math.isfinite(number):
        raise RunnerContractError("metric_non_finite", label)
    return number


def validate_final_results(
    payload: Mapping[str, Any], *, require_p_value: bool = True
) -> dict[str, Any]:
    """Refuse a FINAL_RESULTS payload the evidence gate could never accept.

    ``require_p_value`` defaults to on because the downstream gate is not
    optional: contracts.scientific_evidence.decide_evidence marks a run
    ``not_significant`` without a p-value, so ``confirmation_allowed`` is False
    and a ``supported`` verdict is unreachable no matter how good the metric is.
    Failing here costs one validation call; failing at the gate costs the whole
    run's budget first.
    """
    required_scalars = (
        "task_protocol",
        "dataset_id",
        "dataset_revision",
        "model_id",
        "model_revision",
        "candidate_method",
        "baseline_method",
        "metric_name",
        "metric_direction",
    )
    missing = [name for name in required_scalars if not str(payload.get(name) or "").strip()]
    if missing:
        raise RunnerContractError("runner_contract_violation", ",".join(missing))
    if payload.get("label_fallback_used") is not False:
        raise RunnerContractError("label_fallback_forbidden")
    if str(payload.get("metric_direction")) not in {"higher", "lower"}:
        raise RunnerContractError("metric_direction_invalid")
    seeds = payload.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise RunnerContractError("seed_contract_violation")
    if int(payload.get("num_examples") or 0) <= 0:
        raise RunnerContractError("sample_count_missing")
    per_method = payload.get("per_method")
    if not isinstance(per_method, Mapping):
        raise RunnerContractError("metric_missing", "per_method")
    metric_name = str(payload["metric_name"])
    baseline_method = str(payload["baseline_method"])
    candidate_method = str(payload["candidate_method"])
    if baseline_method == candidate_method:
        raise RunnerContractError("runner_contract_violation", "candidate_equals_baseline")
    for method in (baseline_method, candidate_method):
        row = per_method.get(method)
        if not isinstance(row, Mapping):
            raise RunnerContractError("metric_missing", method)
        _finite(row.get(metric_name, row.get("metric_value")), label=method)
    if require_p_value:
        p_value = extract_p_value(payload)
        if p_value is None:
            raise RunnerContractError(
                "p_value_missing",
                "expected one of {} at top level or under {}".format(
                    "/".join(P_VALUE_KEYS), "/".join(P_VALUE_CONTAINERS)
                ),
            )
        if not 0.0 <= p_value <= 1.0:
            raise RunnerContractError("p_value_invalid", repr(p_value))
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RunnerContractError("artifact_contract_violation")
    artifact_hashes = payload.get("artifact_hashes")
    if not isinstance(artifact_hashes, Mapping):
        raise RunnerContractError("artifact_hashes_missing")
    missing_artifacts = [name for name in REQUIRED_ARTIFACTS if name not in artifacts]
    if missing_artifacts:
        raise RunnerContractError(
            "artifact_contract_violation", ",".join(missing_artifacts)
        )
    gpu = payload.get("gpu_environment")
    if not isinstance(gpu, Mapping):
        raise RunnerContractError("gpu_environment_missing")
    return dict(payload)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _numeric(value: Any) -> float | None:
    import re

    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", str(value or "").replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def recompute_metric(rows: Sequence[Mapping[str, Any]], metric_name: str) -> float:
    if not rows:
        raise RunnerContractError("metric_missing", "raw_predictions")
    metric_name = str(metric_name).lower()
    if metric_name in {"exact_match", "accuracy"}:
        matches = [
            _normalize_text(row.get("prediction")) == _normalize_text(row.get("target"))
            for row in rows
        ]
        return sum(matches) / len(matches)
    if metric_name == "numeric_accuracy":
        matches = [_numeric(row.get("prediction")) == _numeric(row.get("target")) for row in rows]
        return sum(matches) / len(matches)
    if metric_name in {"f1", "macro_f1"}:
        labels = sorted(
            {_normalize_text(row.get("target")) for row in rows}
            | {_normalize_text(row.get("prediction")) for row in rows}
        )
        scores = []
        for label in labels:
            tp = sum(
                _normalize_text(row.get("target")) == label
                and _normalize_text(row.get("prediction")) == label
                for row in rows
            )
            fp = sum(
                _normalize_text(row.get("target")) != label
                and _normalize_text(row.get("prediction")) == label
                for row in rows
            )
            fn = sum(
                _normalize_text(row.get("target")) == label
                and _normalize_text(row.get("prediction")) != label
                for row in rows
            )
            denominator = 2 * tp + fp + fn
            scores.append((2 * tp / denominator) if denominator else 0.0)
        return sum(scores) / len(scores) if scores else 0.0
    raise RunnerContractError("metric_contract_unsupported", metric_name)


def _rows_for_method(path: Path, method: str) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict) and str(row.get("method")) == method:
                rows.append(row)
    return rows


def verify_metric_from_artifacts(
    final_results_path: str | Path,
    *,
    tolerance: float = 1e-12,
) -> MetricVerification:
    final_path = Path(final_results_path)
    payload = validate_final_results(json.loads(final_path.read_text(encoding="utf-8")))
    raw_ref = Path(str(payload["artifacts"]["raw_predictions"]["path"]))
    if not raw_ref.is_absolute():
        raw_ref = final_path.parent / raw_ref
    expected_hash = str(payload["artifact_hashes"].get("raw_predictions") or "")
    actual_hash = hashlib.sha256(raw_ref.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise RunnerContractError("artifact_hash_mismatch", "raw_predictions")
    baseline_method = str(payload["baseline_method"])
    candidate_method = str(payload["candidate_method"])
    metric_name = str(payload["metric_name"])
    baseline_rows = _rows_for_method(raw_ref, baseline_method)
    candidate_rows = _rows_for_method(raw_ref, candidate_method)
    recomputed_baseline = recompute_metric(baseline_rows, metric_name)
    recomputed_candidate = recompute_metric(candidate_rows, metric_name)
    per_method = payload["per_method"]
    baseline = _finite(
        per_method[baseline_method].get(metric_name, per_method[baseline_method].get("metric_value")),
        label=baseline_method,
    )
    candidate = _finite(
        per_method[candidate_method].get(metric_name, per_method[candidate_method].get("metric_value")),
        label=candidate_method,
    )
    if abs(baseline - recomputed_baseline) > tolerance or abs(candidate - recomputed_candidate) > tolerance:
        raise RunnerContractError("metric_recomputation_mismatch")
    return MetricVerification(
        metric_name=metric_name,
        direction=str(payload["metric_direction"]),
        baseline_value=baseline,
        candidate_value=candidate,
        recomputed_baseline=recomputed_baseline,
        recomputed_candidate=recomputed_candidate,
    )
