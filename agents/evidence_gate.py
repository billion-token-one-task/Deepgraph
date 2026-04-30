"""Evidence gating for manuscript status decisions."""
from __future__ import annotations

import json
from pathlib import Path

from agents.artifact_manager import artifact_path, ensure_artifact_dirs, record_artifact
from db import database as db


def _unique_ok_seeds(benchmark_results: dict | None) -> set:
    rows = (benchmark_results or {}).get("rows") or []
    return {
        row.get("seed")
        for row in rows
        if row.get("status") == "ok" and row.get("seed") is not None
    }


def _unique_ok_datasets(benchmark_results: dict | None) -> set:
    rows = (benchmark_results or {}).get("rows") or []
    return {
        row.get("dataset")
        for row in rows
        if row.get("status") == "ok" and row.get("dataset")
    }


def _has_ok_rows(benchmark_results: dict | None) -> bool:
    return any(row.get("status") == "ok" for row in (benchmark_results or {}).get("rows") or [])


def _has_ablation_rows(benchmark_results: dict | None) -> bool:
    return any(
        row.get("status") == "ok" and row.get("analysis_type") == "ablation"
        for row in (benchmark_results or {}).get("rows") or []
    )


def _has_error_rows(benchmark_results: dict | None) -> bool:
    return any(row.get("status") == "error" for row in (benchmark_results or {}).get("rows") or [])


def _has_comparisons(statistical_report: dict | None) -> bool:
    return bool((statistical_report or {}).get("comparisons"))


def _comparisons_support_claim(statistical_report: dict | None) -> bool:
    comparisons = (statistical_report or {}).get("comparisons") or []
    if not comparisons:
        return False
    supported = 0
    unsupported = 0
    significant_negative = False
    for item in comparisons:
        try:
            delta = float(item.get("mean_delta", 0.0))
        except (TypeError, ValueError):
            delta = 0.0
        wins = int(item.get("wins") or 0)
        losses = int(item.get("losses") or 0)
        try:
            p_value = float(item.get("paired_sign_test_p", 1.0))
        except (TypeError, ValueError):
            p_value = 1.0
        if delta > 0 and wins >= losses:
            supported += 1
        else:
            unsupported += 1
        if delta < 0 and losses > wins and p_value < 0.05:
            significant_negative = True
    return supported > unsupported and not significant_negative


def _review_recommendation(review: dict | None) -> str:
    return str((review or {}).get("recommendation") or "").lower()


def evaluate_evidence(inputs: dict) -> dict:
    """Evaluate artifact evidence and decide manuscript status."""
    research_spec = inputs.get("research_spec") or {}
    benchmark_results = inputs.get("benchmark_results")
    statistical_report = inputs.get("statistical_report")
    review = inputs.get("review")
    required = set(research_spec.get("required_evidence") or [])
    blocking = []
    satisfied = []
    next_required = []

    if benchmark_results is None or not _has_ok_rows(benchmark_results):
        blocking.append("missing_benchmark_results")
        next_required.append("Run a benchmark suite that produces at least one successful row.")
    else:
        satisfied.append("has_benchmark_results")
        if _has_error_rows(benchmark_results):
            blocking.append("benchmark_suite_has_error_rows")
            next_required.append("Fix benchmark rows with status=error before treating the run as paper-ready.")

    if "baseline_comparison" in required:
        if _has_comparisons(statistical_report):
            satisfied.append("has_baseline_comparison")
        else:
            blocking.append("missing_baseline_comparison")
            next_required.append("Run at least one baseline and one candidate method with paired comparison rows.")

    if "statistical_test" in required:
        if statistical_report is None:
            blocking.append("missing_statistical_report")
            next_required.append("Generate statistical_report.json from benchmark_results.json.")
        elif _has_comparisons(statistical_report):
            satisfied.append("has_statistical_report")
            if not _comparisons_support_claim(statistical_report):
                blocking.append("statistical_comparisons_do_not_support_claim")
                next_required.append("Revise the hypothesis or method because paired dataset comparisons do not support the claimed improvement.")
        else:
            blocking.append("statistical_report_has_no_comparisons")
            next_required.append("Add paired baseline/candidate runs so statistical comparisons can be computed.")

    if "multi_seed" in required:
        seed_count = len(_unique_ok_seeds(benchmark_results))
        if seed_count >= 10:
            satisfied.append("has_multi_seed")
        else:
            blocking.append("benchmark_suite_has_fewer_than_10_seeds")
            next_required.append("Run at least 10 seeds for every configured dataset and method.")

    if "multi_dataset" in required:
        dataset_count = len(_unique_ok_datasets(benchmark_results))
        if dataset_count >= 2:
            satisfied.append("has_multi_dataset")
        else:
            blocking.append("benchmark_suite_has_fewer_than_2_datasets")
            next_required.append("Run at least two offline or bundled datasets for every configured method.")

    if "ablation" in required:
        if _has_ablation_rows(benchmark_results):
            satisfied.append("has_ablation")
        else:
            blocking.append("missing_ablation_rows")
            next_required.append("Run at least one ablation or sensitivity sweep recorded as benchmark result rows.")

    recommendation = _review_recommendation(review)
    if recommendation == "reject":
        blocking.append("review_rejected")
        next_required.extend(review.get("required_experiments") or [])
        manuscript_status = "not_publishable"
    elif recommendation in {"weak_reject", "major_revision", "borderline"}:
        blocking.append("review_requires_revision")
        next_required.extend(review.get("required_experiments") or [])
        manuscript_status = "needs_more_experiments"
    elif blocking:
        if "missing_statistical_report" in blocking or "missing_benchmark_results" in blocking:
            manuscript_status = "preliminary"
        else:
            manuscript_status = "needs_more_experiments"
    else:
        manuscript_status = "paper_ready_candidate"

    return {
        "schema_version": 1,
        "manuscript_status": manuscript_status,
        "blocking_reasons": blocking,
        "satisfied_requirements": satisfied,
        "next_required_experiments": list(dict.fromkeys(next_required)),
    }


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def write_evidence_gate(run_id: int) -> dict:
    """Read run artifacts, evaluate evidence, and write evidence_gate.json."""
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not run:
        return {"status": "error", "reason": "run_not_found", "run_id": run_id}

    workdir = Path(run.get("workdir") or "")
    results_dir = workdir / "artifacts" / "results"
    reviews_dir = workdir / "artifacts" / "reviews"
    review = _load_json(reviews_dir / "review.json", None)
    if isinstance(review, dict) and "review" in review and isinstance(review["review"], dict):
        review = review["review"]

    gate = evaluate_evidence({
        "research_spec": _load_json(workdir / "research_spec.json", {}),
        "benchmark_results": _load_json(results_dir / "benchmark_results.json", None),
        "statistical_report": _load_json(results_dir / "statistical_report.json", None),
        "review": review,
    })
    gate["run_id"] = run_id

    ensure_artifact_dirs(workdir)
    path = artifact_path(workdir, "artifacts/results/evidence_gate.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gate, indent=2, default=str), encoding="utf-8")
    record_artifact(workdir, run_id, "evidence_gate", path, {
        "manuscript_status": gate.get("manuscript_status"),
    })
    return gate
