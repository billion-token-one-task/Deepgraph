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


def validate_final_results(payload: Mapping[str, Any]) -> dict[str, Any]:
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
