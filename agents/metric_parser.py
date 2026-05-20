"""Metric extraction from experiment run logs."""
from __future__ import annotations

import json
import re
from pathlib import Path

_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_TELEMETRY_RESULT_KEYS = {
    "peak_vram_mb",
    "peak_memory_mb",
    "reserved_vram_gb",
    "target_vram_gb",
    "cuda_device",
    "device",
    "method",
}


def parse_metric_from_log(log_path: Path, metric_name: str) -> float | None:
    """Extract metric value from a run log or evaluate.py output."""
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    for raw in reversed(text.splitlines()):
        line = raw.strip()
        if not line:
            continue
        payload = None
        if line.startswith("FINAL_RESULTS:"):
            _, _, json_text = line.partition(":")
            try:
                payload = json.loads(json_text.strip())
            except (json.JSONDecodeError, TypeError):
                payload = None
        elif line.startswith("{"):
            try:
                payload = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                payload = None
        if isinstance(payload, dict):
            for key in (metric_name, "metric_value"):
                if not key:
                    continue
                raw_value = payload.get(key)
                try:
                    return float(raw_value)
                except (TypeError, ValueError):
                    pass
            numeric_items = []
            for key, raw_value in payload.items():
                if str(key).lower() in _TELEMETRY_RESULT_KEYS:
                    continue
                try:
                    numeric_items.append(float(raw_value))
                except (TypeError, ValueError):
                    continue
            if len(numeric_items) == 1:
                return numeric_items[0]

    patterns = [
        rf'"?{re.escape(metric_name)}"?\s*[:=]\s*({_FLOAT_RE})' if metric_name else None,
        rf'"metric_value"\s*:\s*({_FLOAT_RE})',
        rf'metric_value[:\s]+({_FLOAT_RE})',
        rf'val_bpb[:\s]+({_FLOAT_RE})',
        rf'accuracy[:\s]+({_FLOAT_RE})',
        rf'mAP[:\s]+({_FLOAT_RE})',
    ]
    for pat in patterns:
        if not pat:
            continue
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue
    return None


def parse_benchmark_summary_from_log(log_path: Path) -> dict:
    """Parse structured benchmark output from a run log.

    Preferred format is a single line prefixed with ``FINAL_RESULTS:`` followed
    by JSON.  As a fallback, accept a plain JSON line containing ``per_method``.
    """
    if not log_path.exists():
        return {}
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    for raw in reversed(lines):
        line = raw.strip()
        if not line:
            continue
        payload = None
        if line.startswith("FINAL_RESULTS:"):
            _, _, text = line.partition(":")
            text = text.strip()
            try:
                payload = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                payload = None
        elif line.startswith("{"):
            try:
                payload = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                payload = None
        if isinstance(payload, dict) and (
            isinstance(payload.get("per_method"), dict)
            or isinstance(payload.get("seed_results"), list)
            or payload.get("best_method")
        ):
            return payload
    return {}


def benchmark_scores(summary: dict) -> tuple[str, str | None, float | None, float | None, int]:
    """Return (metric_name, candidate_method, candidate_value, best_other_value, num_seeds)."""
    metric_name = str(summary.get("primary_metric") or summary.get("metric_name") or "metric")
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    candidate_method = str(
        summary.get("candidate_method")
        or ("cggr" if "cggr" in per_method else summary.get("best_method") or "")
    ).strip() or None

    def _metric_for(method_name: str) -> float | None:
        row = per_method.get(method_name)
        if not isinstance(row, dict):
            return None
        raw = row.get(metric_name)
        if raw is None:
            raw = row.get("metric_value")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    candidate_value = _metric_for(candidate_method) if candidate_method else None
    best_other = None
    for method_name, row in per_method.items():
        if method_name == candidate_method or not isinstance(row, dict):
            continue
        try:
            value = float(row.get(metric_name, row.get("metric_value")))
        except (TypeError, ValueError):
            continue
        if best_other is None or value > best_other:
            best_other = value

    seed_results = summary.get("seed_results") if isinstance(summary.get("seed_results"), list) else []
    num_seeds = int(summary.get("num_seeds") or len(seed_results) or 0)
    return metric_name, candidate_method, candidate_value, best_other, num_seeds


def build_benchmark_summary_from_predictions(
    results_dir: Path,
    *,
    candidate_method: str | None = None,
    metric_name: str = "primary_score",
    min_lines: int = 50,
) -> dict:
    """Aggregate partial benchmark metrics from raw_predictions.jsonl."""
    pred_path = results_dir / "raw_predictions.jsonl"
    if not pred_path.exists():
        return {}
    per_method: dict[str, dict[str, float]] = {}
    datasets: set[str] = set()
    seeds: set[int] = set()
    line_count = 0
    try:
        with pred_path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                line_count += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                method = str(row.get("method") or "unknown")
                score = row.get("primary_score", row.get("exact"))
                try:
                    value = float(score)
                except (TypeError, ValueError):
                    continue
                bucket = per_method.setdefault(method, {"_n": 0, "_sum": 0.0})
                bucket["_n"] = int(bucket.get("_n", 0)) + 1
                bucket["_sum"] = float(bucket.get("_sum", 0.0)) + value
                if row.get("dataset"):
                    datasets.add(str(row["dataset"]))
                if row.get("seed") is not None:
                    try:
                        seeds.add(int(row["seed"]))
                    except (TypeError, ValueError):
                        pass
    except OSError:
        return {}

    if line_count < min_lines:
        return {}

    normalized: dict[str, dict[str, float]] = {}
    for method_name, bucket in per_method.items():
        count = int(bucket.pop("_n", 0))
        total = float(bucket.pop("_sum", 0.0))
        if count > 0:
            normalized[method_name] = {metric_name: total / count}
    per_method = normalized

    if not per_method:
        return {}

    resolved_candidate = candidate_method
    if not resolved_candidate:
        for name in per_method:
            if "cggr" in name.lower() or "cpg" in name.lower() or "candidate" in name.lower():
                resolved_candidate = name
                break
        if not resolved_candidate:
            resolved_candidate = next(iter(per_method))

    candidate_value = None
    if resolved_candidate in per_method:
        candidate_value = per_method[resolved_candidate].get(metric_name)

    return {
        "primary_metric": metric_name,
        "metric_name": metric_name,
        "candidate_method": resolved_candidate,
        "per_method": per_method,
        "datasets": sorted(datasets),
        "num_seeds": len(seeds) or 1,
        "full_benchmark_completed": False,
        "partial_from_predictions": True,
        "prediction_lines": line_count,
    }
