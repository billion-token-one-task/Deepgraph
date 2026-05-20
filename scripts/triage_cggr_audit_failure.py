#!/usr/bin/env python3
"""Triage CGGR audit/materialization failures without rewriting evidence.

The intent is to keep the confirmatory artifact immutable while producing a
clear next-run decision: engineering retry, missing evidence/baseline expansion,
or a new preregistered method iteration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_paper_benchmark_artifacts import audit  # noqa: E402


ENGINEERING_MARKERS = (
    "workdir does not exist",
    "results dir does not exist",
    "missing",
    "empty",
    "invalid",
    "bad json",
    "duplicate",
    "overlapping",
    "below paper gate",
    "above paper gate",
    "full_benchmark_completed is not true",
    "sharded_run artifact",
    "zero-token",
    "empty prediction",
    "generation_or_scoring",
    "cell below paper gate",
    "cell above paper gate",
)

EVIDENCE_SCOPE_MARKERS = (
    "required method missing: CAR-Style Certainty Adaptive Routing",
    "required method missing: Self-Route-Style Mode Routing",
    "required method missing: Rational-Metareasoning VOC Routing",
    "strict top-venue",
    "top-venue",
)


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


def classify_audit_blockers(blockers: list[str]) -> dict[str, Any]:
    engineering: list[str] = []
    evidence_scope: list[str] = []
    unknown: list[str] = []
    for blocker in blockers:
        text = str(blocker)
        if _contains_any(text, EVIDENCE_SCOPE_MARKERS):
            evidence_scope.append(text)
        elif _contains_any(text, ENGINEERING_MARKERS):
            engineering.append(text)
        else:
            unknown.append(text)
    return {
        "engineering_retry_blockers": engineering,
        "evidence_scope_blockers": evidence_scope,
        "unclassified_blockers": unknown,
    }


def classify_claim_values(claim_values: dict[str, Any]) -> dict[str, Any]:
    if not claim_values:
        return {
            "claim_values_present": False,
            "scientific_iteration_required": False,
            "reason": "claim_values.json is not available yet",
        }

    decision = str(claim_values.get("claim_support_decision") or "").strip().lower()
    top_venue = str(claim_values.get("top_venue_general_superiority_decision") or "").strip().lower()
    try:
        delta = float(claim_values.get("cggr_vs_baseline_delta"))
    except (TypeError, ValueError):
        delta = None
    try:
        p_value = float(claim_values.get("paired_permutation_p"))
    except (TypeError, ValueError):
        p_value = None

    science_reasons: list[str] = []
    if decision == "rejected":
        science_reasons.append("claim_support_decision=rejected")
    elif decision == "downgraded":
        science_reasons.append("positive point estimate is not confirmatory under the support gate")
    if delta is not None and delta <= 0.0:
        science_reasons.append("CGGR utility delta is non-positive")
    if p_value is not None and p_value >= 0.05:
        science_reasons.append("paired permutation p-value does not clear 0.05")

    evidence_scope_reasons = []
    if "blocked" in top_venue:
        evidence_scope_reasons.append("top-venue general-superiority claim remains blocked")

    return {
        "claim_values_present": True,
        "claim_support_decision": decision or None,
        "top_venue_general_superiority_decision": top_venue or None,
        "cggr_vs_baseline_delta": delta,
        "paired_permutation_p": p_value,
        "scientific_iteration_required": bool(science_reasons),
        "scientific_iteration_reasons": science_reasons,
        "evidence_scope_reasons": evidence_scope_reasons,
    }


def next_actions(classified_blockers: dict[str, Any], claim_triage: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if classified_blockers.get("engineering_retry_blockers"):
        actions.append(
            {
                "type": "engineering_retry",
                "allowed_automation": True,
                "description": "Retry sync/relaunch the same registered benchmark contract without changing method, thresholds, datasets, seeds, or baselines.",
            }
        )
    if classified_blockers.get("evidence_scope_blockers") or claim_triage.get("evidence_scope_reasons"):
        actions.append(
            {
                "type": "evidence_scope_expansion",
                "allowed_automation": True,
                "description": "Register additional evidence shards such as CAR-style, Self-Route-style, and rational-metareasoning baselines; merge into a new strict artifact.",
            }
        )
    if claim_triage.get("scientific_iteration_required"):
        actions.append(
            {
                "type": "method_iteration_v2",
                "allowed_automation": True,
                "requires_new_run_id": True,
                "description": "Design a preregistered CGGR v2 instead of editing the completed artifact or tuning on the final benchmark result.",
                "candidate_changes_to_preregister": [
                    "Fit or calibrate a counterfactual-gain proxy on a development split only, then freeze it before final evaluation.",
                    "Calibrate the LCB threshold and uncertainty penalty on held-out dev cells instead of final test cells.",
                    "Add failure-mode-aware routing features for multihop, boolean, and stress-split cases while preserving the locked final benchmark.",
                    "Pre-register branch token budgets and simple-case non-degradation constraints before launching the next final run.",
                    "Keep all old runs as negative or exploratory evidence; never overwrite raw predictions or claim maps.",
                ],
            }
        )
    if classified_blockers.get("unclassified_blockers"):
        actions.append(
            {
                "type": "manual_review",
                "allowed_automation": False,
                "description": "Unclassified blocker needs human/agent review before relaunching or modifying the method.",
            }
        )
    if not actions:
        actions.append(
            {
                "type": "no_failure_detected",
                "allowed_automation": False,
                "description": "No audit blocker or claim failure was detected by this triage report.",
            }
        )
    return actions


def triage(
    workdir: Path,
    *,
    audit_result: dict[str, Any] | None = None,
    claim_values_path: Path | None = None,
    require_full: bool = True,
    require_top_venue_baselines: bool = False,
) -> dict[str, Any]:
    if audit_result is None:
        audit_result = audit(
            workdir,
            require_full=require_full,
            require_top_venue_baselines=require_top_venue_baselines,
        )
    blockers = [str(item) for item in audit_result.get("blockers") or []]
    classified = classify_audit_blockers(blockers)
    if claim_values_path is None:
        claim_values_path = workdir / "results" / "claim_values.json"
        if not claim_values_path.exists():
            claim_values_path = workdir / "audited_results" / "claim_values.json"
    claim_triage = classify_claim_values(_load_json(claim_values_path))
    actions = next_actions(classified, claim_triage)
    return {
        "schema_version": "cggr_failure_triage_v1",
        "workdir": str(workdir),
        "audit_ok": bool(audit_result.get("ok")),
        "require_full": require_full,
        "require_top_venue_baselines": require_top_venue_baselines,
        "immutable_evidence_policy": {
            "do_not_overwrite_current_artifact": True,
            "method_changes_require_new_preregistered_run": True,
            "final_benchmark_must_not_be_used_as_tuning_feedback": True,
        },
        "audit_blocker_triage": classified,
        "claim_value_triage": claim_triage,
        "recommended_next_actions": actions,
    }


def write_triage_report(
    workdir: Path,
    *,
    audit_result: dict[str, Any] | None = None,
    out_path: Path | None = None,
    require_full: bool = True,
    require_top_venue_baselines: bool = False,
) -> dict[str, Any]:
    report = triage(
        workdir,
        audit_result=audit_result,
        require_full=require_full,
        require_top_venue_baselines=require_top_venue_baselines,
    )
    target = out_path or (workdir / "results" / "cggr_failure_triage.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["written"] = str(target)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-full", action="store_true", default=True)
    parser.add_argument("--no-require-full", action="store_false", dest="require_full")
    parser.add_argument("--require-top-venue-baselines", action="store_true")
    args = parser.parse_args()
    report = write_triage_report(
        args.workdir,
        out_path=args.out,
        require_full=args.require_full,
        require_top_venue_baselines=args.require_top_venue_baselines,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
