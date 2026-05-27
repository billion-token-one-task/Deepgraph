"""Detect duplicate deep insights before experiment forge / requeue."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from db import database as db


def _normalize_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return text


def _text_similar(a: str, b: str, *, threshold: float = 0.82) -> bool:
    left = _normalize_text(a)
    right = _normalize_text(b)
    if not left or not right:
        return False
    if left == right:
        return True
    if len(left) > 20 and len(right) > 20 and (left in right or right in left):
        return True
    return SequenceMatcher(None, left, right).ratio() >= threshold


def _load_json(value: Any, default: Any):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def benchmark_fingerprint(run: dict | None) -> str | None:
    if not run:
        return None
    baseline = run.get("baseline_metric_value")
    best = run.get("best_metric_value")
    if baseline is None or best is None:
        return None
    try:
        baseline_f = float(baseline)
        best_f = float(best)
    except (TypeError, ValueError):
        return None
    metric = str(run.get("baseline_metric_name") or "metric").strip().lower()
    return f"{metric}:{baseline_f:.8f}:{best_f:.8f}"


def _benchmark_summary_flags(workdir: str | None) -> dict[str, Any]:
    if not workdir:
        return {}
    summary_path = Path(workdir) / "results" / "benchmark_summary.json"
    if not summary_path.exists():
        return {}
    summary = _load_json(summary_path.read_text(encoding="utf-8"), {})
    return summary if isinstance(summary, dict) else {}


def _full_benchmark_completed(run: dict | None) -> bool:
    if not run:
        return False
    workdir = str(run.get("workdir") or "").strip()
    manifest_path = Path(workdir) / "results" / "benchmark_artifact_manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path.read_text(encoding="utf-8"), {})
        if isinstance(manifest, dict) and manifest.get("full_benchmark_completed") is True:
            return True
    summary = _benchmark_summary_flags(workdir)
    return bool(summary.get("full_benchmark_completed"))


def find_duplicate_benchmark(insight_id: int, run: dict | None) -> dict | None:
    """Return duplicate metadata when another insight already produced the same score fingerprint."""
    fingerprint = benchmark_fingerprint(run)
    if not fingerprint:
        return None
    rows = db.fetchall(
        """
        SELECT er.deep_insight_id, er.id AS run_id, er.baseline_metric_value, er.best_metric_value
        FROM experiment_runs er
        WHERE er.deep_insight_id != ?
          AND er.status = 'completed'
          AND er.baseline_metric_value IS NOT NULL
          AND er.best_metric_value IS NOT NULL
        ORDER BY er.id ASC
        """,
        (insight_id,),
    )
    for row in rows:
        other_id = int(row["deep_insight_id"])
        if other_id >= insight_id:
            continue
        other_run = dict(row)
        if benchmark_fingerprint(other_run) != fingerprint:
            continue
        return {
            "kind": "benchmark_fingerprint",
            "duplicate_of_insight_id": other_id,
            "duplicate_run_id": int(row["run_id"]),
            "fingerprint": fingerprint,
            "reason": (
                f"Benchmark fingerprint matches insight {other_id} run {row['run_id']} "
                f"({fingerprint}); skipping clone experiment."
            ),
        }
    return None


def find_duplicate_text(insight_id: int, insight: dict | None = None) -> dict | None:
    row = insight or db.fetchone("SELECT * FROM deep_insights WHERE id=?", (insight_id,))
    if not row:
        return None
    title = str(row.get("title") or "")
    hypothesis = str(row.get("hypothesis") or "")
    if not title and not hypothesis:
        return None
    others = db.fetchall(
        """
        SELECT id, title, hypothesis
        FROM deep_insights
        WHERE id < ?
          AND COALESCE(status, 'candidate') NOT IN ('exists')
        ORDER BY id ASC
        """,
        (insight_id,),
    )
    for other in others:
        if _text_similar(title, other.get("title") or ""):
            return {
                "kind": "title",
                "duplicate_of_insight_id": int(other["id"]),
                "reason": f"Title is too similar to earlier insight {other['id']}.",
            }
        if _text_similar(hypothesis, other.get("hypothesis") or "", threshold=0.80):
            return {
                "kind": "hypothesis",
                "duplicate_of_insight_id": int(other["id"]),
                "reason": f"Hypothesis is too similar to earlier insight {other['id']}.",
            }
    return None


def check_insight_duplicate(
    insight_id: int,
    *,
    insight: dict | None = None,
    run: dict | None = None,
    text_dedup: bool = True,
    benchmark_dedup: bool = True,
) -> dict | None:
    if text_dedup:
        dup = find_duplicate_text(insight_id, insight=insight)
        if dup:
            return dup
    if benchmark_dedup:
        if run is None:
            run = db.fetchone(
                """
                SELECT * FROM experiment_runs
                WHERE deep_insight_id=?
                  AND status IN ('completed', 'failed')
                ORDER BY id DESC LIMIT 1
                """,
                (insight_id,),
            )
            run = dict(run) if run else None
        dup = find_duplicate_benchmark(insight_id, run)
        if dup:
            return dup
    return None
