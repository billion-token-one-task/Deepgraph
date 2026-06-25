"""Helpers for turning manuscript benchmark blockers into follow-up jobs."""

from __future__ import annotations

import json
from pathlib import Path

from agents.loop_router import compact_loop_note, route_blockers
from db import database as db

BENCHMARK_COMPLETION_STAGE = "benchmark_completion_required"

_BENCHMARK_BLOCKER_MARKERS = (
    "benchmark artifact",
    "benchmark evidence",
    "evidence_plan.json",
    "benchmark_artifact_manifest.json is missing",
    "missing or not linked",
    "benchmark summary is missing",
    "per_method must contain at least two",
    "must include at least two methods",
    "at least two methods/baselines",
    "no metric",
    "metric missing",
)

_BENCHMARK_COMPLETION_NEGATIVE_MARKERS = (
    "benchmark evidence passed",
    "benchmark evidence has passed",
    "benchmark evidence already passed",
    "only manuscript polish blockers remain",
    "benchmark evidence passed; only manuscript",
)

_SOFT_BENCHMARK_GAP_MARKERS = (
    "full_benchmark_completed=false",
    "full_benchmark_completed is false",
    "full_benchmark_completed",
    "full benchmark policy",
    "full benchmark evidence",
    "quality gate requires full benchmark",
    "benchmark comparison does not cover",
    "run or present all required baselines",
    "benchmark coverage",
    "required benchmark coverage missing",
    "required baselines missing",
    "required baseline missing",
    "required model coverage missing",
    "model coverage",
    "mip-nerf",
    "required ablation",
    "ablation table",
    "seed",
    "num_seeds",
    "seed(s) found",
    "minimum_seeds",
    "baseline",
    "load_failures",
    "code repair",
    "proof repair",
)


def _is_benchmark_completion_blocker(blocker: str) -> bool:
    text = str(blocker or "").strip().lower()
    if not text:
        return False
    if any(marker in text for marker in _BENCHMARK_COMPLETION_NEGATIVE_MARKERS):
        return False
    return any(marker in text for marker in (_BENCHMARK_BLOCKER_MARKERS + _SOFT_BENCHMARK_GAP_MARKERS))



def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _issue_to_blocker(issue: dict) -> str:
    standard = str(issue.get("standard") or issue.get("severity") or "").strip()
    text = str(issue.get("issue") or issue.get("summary") or issue.get("evidence") or "").strip()
    evidence = str(issue.get("evidence") or "").strip()
    if standard and text:
        blocker = f"{standard}: {text}"
    else:
        blocker = text or standard
    if evidence and evidence not in blocker:
        blocker = f"{blocker} ({evidence})"
    return blocker.strip()


def _collect_quality_report_issues(report: dict) -> list[str]:
    if not isinstance(report, dict):
        return []
    candidates: list[str] = []
    list_keys = (
        "benchmark_completion_blockers",
        "benchmark_evidence_blockers",
        "experiment_scientific_advisories",
        "plain_experiment_advisories",
        "issues",
    )
    for key in list_keys:
        values = report.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, dict):
                candidates.append(_issue_to_blocker(item))
            else:
                candidates.append(str(item))
    nested_keys = (
        ("writing_guideline_audit", "experiment_scope_advisories"),
        ("writing_guideline_audit", "issues"),
        ("scientific_review_gate", "issues"),
        ("benchmark_artifact_manifest", "readiness_blockers"),
    )
    for parent_key, child_key in nested_keys:
        parent = report.get(parent_key)
        if not isinstance(parent, dict):
            continue
        values = parent.get(child_key)
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, dict):
                candidates.append(_issue_to_blocker(item))
            else:
                candidates.append(str(item))
    return _dedupe(candidates)


def _quality_report_from_bundle(bundle: dict) -> dict:
    report = bundle.get("quality_report")
    if isinstance(report, dict):
        return report
    for key in ("quality_report", "quality_report_path", "paper_quality_report"):
        raw = bundle.get(key)
        if not raw or isinstance(raw, dict):
            continue
        try:
            path = Path(str(raw))
            if path.exists() and path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception:
            continue
    workdir = bundle.get("workdir")
    if workdir:
        try:
            path = Path(str(workdir)) / "paper_quality_report.json"
            if path.exists() and path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}

def benchmark_completion_blockers(bundle: dict | None) -> list[str]:
    """Return blockers that need a benchmark/harness completion run.

    Incomplete full-benchmark coverage, insufficient seeds, and missing
    baselines/model families should go back to benchmark completion before
    another manuscript attempt. Manuscript quality reports keep these gaps as
    experiment-scope advisories, so inspect the report in addition to the
    top-level submission blockers.
    """
    if not isinstance(bundle, dict):
        return []
    blockers: list[str] = []
    for key in (
        "submission_blockers",
        "benchmark_completion_blockers",
        "benchmark_evidence_blockers",
        "stage_blockers",
    ):
        raw_blockers = bundle.get(key)
        if not isinstance(raw_blockers, list):
            continue
        for item in raw_blockers:
            if isinstance(item, dict):
                blockers.append(_issue_to_blocker(item))
            else:
                blockers.append(str(item))
    error = str(bundle.get("error") or "").strip()
    if error:
        blockers.append(error)
    quality_report = _quality_report_from_bundle(bundle)
    blockers.extend(_collect_quality_report_issues(quality_report))
    return [blocker for blocker in _dedupe(blockers) if _is_benchmark_completion_blocker(blocker)]



def _read_json_file(path: Path) -> dict:
    try:
        if path.exists() and path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        return {}
    return {}


def benchmark_completion_bundle_from_run(run_id: int, result: dict | None = None) -> dict:
    """Build a benchmark-completion scheduling bundle before manuscript writing.

    The experiment stage owns full-benchmark completeness. If a run's benchmark
    artifact manifest says the package is incomplete, writing should not start;
    the orchestrator should stay in benchmark completion instead.
    """
    try:
        run = db.fetchone("SELECT workdir FROM experiment_runs WHERE id=?", (int(run_id),)) or {}
    except Exception:
        return {}
    workdir_raw = str(run.get("workdir") or "").strip()
    workdir = Path(workdir_raw) if workdir_raw else None
    manifest: dict = {}
    validation_summary: dict = {}
    spec_evidence_plan: dict = {}
    spec_benchmark_manifest: dict = {}
    spec_evidence_path: Path | None = None
    spec_manifest_path: Path | None = None
    if workdir is not None:
        results_dir = workdir / "results"
        spec_dir = workdir / "spec"
        spec_evidence_path = spec_dir / "evidence_plan.json"
        spec_manifest_path = spec_dir / "benchmark_manifest.json"
        manifest = _read_json_file(results_dir / "benchmark_artifact_manifest.json")
        validation_summary = _read_json_file(results_dir / "validation_summary.json")
        spec_evidence_plan = _read_json_file(spec_evidence_path)
        spec_benchmark_manifest = _read_json_file(spec_manifest_path)
    result = result if isinstance(result, dict) else {}
    blockers: list[str] = []

    spec_evidence_file_present = bool(spec_evidence_path and spec_evidence_path.exists())
    spec_manifest_file_present = bool(spec_manifest_path and spec_manifest_path.exists())
    if spec_evidence_file_present and not spec_evidence_plan:
        blockers.append("spec/evidence_plan.json is empty; benchmark evidence plan must be materialized before manuscript writing.")
    if spec_benchmark_manifest:
        protocol = spec_benchmark_manifest.get("benchmark_protocol") if isinstance(spec_benchmark_manifest.get("benchmark_protocol"), dict) else {}
        dataset_protocols = protocol.get("dataset_protocols") if isinstance(protocol.get("dataset_protocols"), list) else []
        if dataset_protocols and not (
            spec_benchmark_manifest.get("benchmark_evidence")
            or spec_evidence_plan.get("benchmark_evidence")
            or spec_evidence_plan.get("claim_evidence_matrix")
        ):
            blockers.append("benchmark evidence sources are missing from spec/evidence_plan.json or benchmark_manifest.json.")

    if manifest:
        if manifest.get("full_benchmark_completed") is not True:
            blockers.append("benchmark_artifact_manifest.full_benchmark_completed is false")
        for item in manifest.get("readiness_blockers") or []:
            blockers.append(str(item))
    elif result.get("full_benchmark_completed") is False or validation_summary.get("full_benchmark_completed") is False:
        blockers.append("benchmark_artifact_manifest.json is missing or not linked.")

    if result.get("full_benchmark_completed") is False:
        blockers.append("run result full_benchmark_completed is false")
    benchmark_summary = result.get("benchmark_summary") if isinstance(result.get("benchmark_summary"), dict) else {}
    if benchmark_summary.get("full_benchmark_completed") is False:
        blockers.append("benchmark_summary.full_benchmark_completed is false")
    if validation_summary.get("full_benchmark_completed") is False:
        blockers.append("validation_summary.full_benchmark_completed is false")

    blockers = [item for item in _dedupe(blockers) if _is_benchmark_completion_blocker(item)]
    if not blockers:
        return {}
    return {
        "error": "Full benchmark evidence is incomplete; manuscript writing is blocked before submission generation.",
        "benchmark_completion_blockers": blockers,
        "benchmark_artifact_manifest": manifest,
        "validation_summary": validation_summary,
        "workdir": workdir_raw,
    }

def schedule_benchmark_completion(
    insight_id: int,
    run_id: int,
    bundle: dict | None,
    *,
    source: str,
    resource_class: str | None = None,
) -> bool:
    """Queue the insight for an automatic benchmark-completion refit if needed."""
    blockers = benchmark_completion_blockers(bundle)
    if not blockers:
        return False
    loop_route = route_blockers(
        blockers,
        context={"source": source, "stage": BENCHMARK_COMPLETION_STAGE, "run_id": run_id},
    )
    short_error = "; ".join(blockers[:6])
    loop_note = compact_loop_note(loop_route)
    note = (
        "Submission bundle is blocked by incomplete benchmark evidence; "
        "queued automatic real-benchmark completion."
    )
    if loop_note:
        note = f"{note} {loop_note}"
    existing = db.fetchone(
        "SELECT id, resource_class FROM auto_research_jobs WHERE deep_insight_id=?",
        (insight_id,),
    )
    chosen_resource = resource_class or (existing.get("resource_class") if existing else None) or "gpu_large"
    if existing:
        db.execute(
            """
            UPDATE auto_research_jobs
            SET status='queued',
                stage=?,
                experiment_run_id=?,
                resource_class=?,
                scheduler_priority=CASE
                    WHEN COALESCE(scheduler_priority, 0) < 2 THEN 2
                    ELSE scheduler_priority
                END,
                assigned_worker=NULL,
                last_note=?,
                last_error=?,
                last_checked_at=CURRENT_TIMESTAMP,
                updated_at=CURRENT_TIMESTAMP
            WHERE deep_insight_id=?
            """,
            (
                BENCHMARK_COMPLETION_STAGE,
                run_id,
                chosen_resource,
                note,
                short_error,
                insight_id,
            ),
        )
    else:
        db.execute(
            """
            INSERT INTO auto_research_jobs
              (deep_insight_id, status, stage, experiment_run_id, resource_class,
               scheduler_priority, last_note, last_error)
            VALUES (?, 'queued', ?, ?, ?, 2, ?, ?)
            """,
            (insight_id, BENCHMARK_COMPLETION_STAGE, run_id, chosen_resource, note, short_error),
        )
    db.commit()
    db.emit_pipeline_event(
        "benchmark_completion_required",
        {
            "deep_insight_id": insight_id,
            "experiment_run_id": run_id,
            "source": source,
            "blockers": blockers,
            "loop_router": loop_route,
            "resource_class": chosen_resource,
        },
        entity_type="deep_insight",
        entity_id=str(insight_id),
        dedupe_key=f"benchmark_completion_required:{insight_id}:{run_id}",
    )
    return True
