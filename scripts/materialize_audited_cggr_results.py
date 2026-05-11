#!/usr/bin/env python3
"""Render manuscript-ready CGGR tables from audited benchmark artifacts.

The script refuses to write output unless the full paper benchmark audit passes.
This keeps manuscript tables downstream of the artifact gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from audit_paper_benchmark_artifacts import audit


METHOD_ORDER = [
    "Vanilla Direct Answering",
    "Always-Reason Chain-of-Thought",
    "Self-Consistency Reasoning",
    "Least-to-Most Prompting",
    "Confidence Gate",
    "Disagreement Routing",
    "Random Budget-Matched Routing",
    "CGGR",
]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _tex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _num(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def _ci_text(method: str, summary: dict) -> str:
    bootstrap = summary.get("bootstrap_ci") if isinstance(summary.get("bootstrap_ci"), dict) else {}
    if method == bootstrap.get("candidate_method"):
        lo, hi = (bootstrap.get("candidate_ci95") or [None, None])[:2]
        return f"[{_num(lo)}, {_num(hi)}]"
    if method == bootstrap.get("baseline_method"):
        lo, hi = (bootstrap.get("baseline_ci95") or [None, None])[:2]
        return f"[{_num(lo)}, {_num(hi)}]"
    return "--"


def _ordered_methods(per_method: dict) -> list[str]:
    ordered = [method for method in METHOD_ORDER if method in per_method]
    ordered.extend(method for method in per_method if method not in ordered and "oracle" not in method.lower())
    ordered.extend(method for method in per_method if method not in ordered)
    return ordered


def _main_table(summary: dict) -> str:
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    per_method_std = summary.get("per_method_std") if isinstance(summary.get("per_method_std"), dict) else {}
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & Quality & Utility & Seed Std. & Tokens/Ex. & Latency/Ex. & 95\% CI \\",
        r"\midrule",
    ]
    for method in _ordered_methods(per_method):
        row = per_method.get(method) or {}
        lines.append(
            " & ".join(
                [
                    _tex_escape(method),
                    _num(row.get("score")),
                    _num(row.get("metric_value")),
                    _num(per_method_std.get(method)),
                    _num(row.get("avg_new_tokens"), 1),
                    _num(row.get("avg_latency_seconds"), 2),
                    _ci_text(method, summary),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _ablation_table(summary: dict) -> str:
    rows = summary.get("ablation_table") or summary.get("ablation_results") or []
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Ablation & Utility & Delta vs. CGGR \\",
        r"\midrule",
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            " & ".join(
                [
                    _tex_escape(row.get("ablation") or row.get("method") or ""),
                    _num(row.get("metric_value")),
                    _num(row.get("delta_vs_cggr")),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _cost_table(summary: dict) -> str:
    rows = summary.get("latency_tokens_table") or summary.get("cost_utility_tradeoff_table") or []
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Utility & Tokens/Ex. & Latency/Ex. & Route Rate \\",
        r"\midrule",
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            " & ".join(
                [
                    _tex_escape(row.get("method") or ""),
                    _num(row.get("metric_value")),
                    _num(row.get("avg_new_tokens"), 1),
                    _num(row.get("avg_latency_seconds"), 2),
                    _num(row.get("route_rate")),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _method_short_label(method: str) -> str:
    labels = {
        "Vanilla Direct Answering": "Direct",
        "Always-Reason Chain-of-Thought": "CoT",
        "Self-Consistency Reasoning": "Self-Cons.",
        "Least-to-Most Prompting": "Least-to-Most",
        "Confidence Gate": "Conf. Gate",
        "Disagreement Routing": "Disagree Route",
        "Random Budget-Matched Routing": "Random Route",
        "CGGR": "CGGR",
    }
    return labels.get(method, method.replace("CGGR/", ""))


def _utility_comparison_figure(summary: dict) -> str:
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    rows: list[tuple[str, float]] = []
    for method in METHOD_ORDER:
        row = per_method.get(method)
        if not isinstance(row, dict):
            continue
        try:
            rows.append((method, float(row.get("metric_value"))))
        except (TypeError, ValueError):
            continue
    if not rows:
        return "% No utility values were available for the audited comparison figure.\n"

    max_value = max(0.001, max(value for _, value in rows))
    scale = 220.0 / max_value
    row_gap = 15
    top = 20 + row_gap * len(rows)
    lines = [
        r"\begin{figure}[t]",
        r"    \centering",
        r"    \resizebox{\linewidth}{!}{%",
        r"    \setlength{\unitlength}{1pt}",
        f"    \\begin{{picture}}(410,{top + 12})",
        r"        \put(105," + str(top) + r"){\makebox(220,0)[l]{\scriptsize Audited utility}}",
    ]
    for idx, (method, value) in enumerate(rows):
        y = top - 18 - idx * row_gap
        width = max(1.0, value * scale)
        lines.extend(
            [
                f"        \\put(0,{y}){{\\makebox(98,0)[l]{{\\scriptsize {_tex_escape(_method_short_label(method))}}}}}",
                f"        \\put(105,{y - 2}){{\\rule{{{width:.1f}pt}}{{5pt}}}}",
                f"        \\put({110 + width:.1f},{y}){{\\makebox(45,0)[l]{{\\scriptsize {_num(value)}}}}}",
            ]
        )
    lines.extend(
        [
            r"    \end{picture}",
            r"    }",
            r"    \caption{Audited utility comparison for deployable methods in the locked benchmark contract.}",
            r"    \label{fig:utility_comparison}",
            r"\end{figure}",
            "",
        ]
    )
    return "\n".join(lines)


def _significance_report(summary: dict, audit_result: dict) -> str:
    bootstrap = summary.get("bootstrap_ci") if isinstance(summary.get("bootstrap_ci"), dict) else {}
    candidate = bootstrap.get("candidate_method") or summary.get("candidate_method") or "CGGR"
    baseline = bootstrap.get("baseline_method") or "Always-Reason Chain-of-Thought"
    lines = [
        "# Significance Report",
        "",
        f"- Candidate method: `{candidate}`",
        f"- Baseline method: `{baseline}`",
        f"- Candidate 95% CI: `{bootstrap.get('candidate_ci95')}`",
        f"- Baseline 95% CI: `{bootstrap.get('baseline_ci95')}`",
        f"- Delta 95% CI: `{bootstrap.get('delta_ci95')}`",
        f"- Observed utility delta: `{_num(bootstrap.get('observed_delta'))}`",
        f"- Paired permutation p-value: `{_num(bootstrap.get('paired_permutation_p'))}`",
        f"- Seeds: `{summary.get('num_seeds')}`",
        f"- Raw prediction rows audited: `{audit_result.get('raw_predictions_lines')}`",
        "",
        "The confidence intervals are computed from paired seed-level aggregates. The permutation test is a paired sign-flip test over the same seed-level candidate-baseline utility deltas. These statistics support manuscript claims only together with the full artifact audit.",
        "",
    ]
    return "\n".join(lines)


def _reproducibility_statement(summary: dict, audit_result: dict) -> str:
    model = summary.get("model") if isinstance(summary.get("model"), dict) else {}
    budget = summary.get("budget") if isinstance(summary.get("budget"), dict) else {}
    datasets = summary.get("datasets") if isinstance(summary.get("datasets"), list) else []
    dataset_names = [str(row.get("name") or row.get("id") or "") for row in datasets if isinstance(row, dict)]
    implementations = summary.get("method_implementation") if isinstance(summary.get("method_implementation"), dict) else {}
    cggr_impl = implementations.get("CGGR") if isinstance(implementations.get("CGGR"), dict) else {}
    claim_scope = summary.get("claim_scope_override") if isinstance(summary.get("claim_scope_override"), dict) else {}
    token_budgets = cggr_impl.get("token_budgets") if isinstance(cggr_impl.get("token_budgets"), dict) else {}
    thresholds = cggr_impl.get("thresholds") if isinstance(cggr_impl.get("thresholds"), dict) else {}
    lines = [
        "# Reproducibility Statement",
        "",
        f"- Source workdir: `{audit_result.get('workdir')}`",
        f"- Model: `{model.get('id') or model.get('name')}`",
        f"- Backend: `{model.get('backend')}`",
        f"- Hardware: `{summary.get('hardware') or model.get('hardware')}`",
        f"- Seeds: `{summary.get('num_seeds') or budget.get('seeds')}`",
        f"- Examples per dataset/seed: `{budget.get('max_examples_per_dataset_seed')}`",
        f"- Datasets: `{', '.join(dataset_names)}`",
        f"- Audit mode: `require_full={audit_result.get('require_full')}`",
        f"- Top-venue baseline audit: `require_top_venue_baselines={audit_result.get('require_top_venue_baselines')}`",
        f"- Full benchmark completed: `{audit_result.get('full_benchmark_completed')}`",
        "",
        "All result tables are downstream of `scripts/audit_paper_benchmark_artifacts.py --require-full`. Smoke, sanity, invalidated, and unmerged shard runs are not admissible for empirical claims.",
        "",
    ]
    if cggr_impl:
        lines.extend(
            [
                "## CGGR Implementation",
                "",
                f"- Estimator type: `{cggr_impl.get('estimator_type')}`",
                f"- Trained estimator: `{cggr_impl.get('trained_estimator')}`",
                f"- Label usage: `{cggr_impl.get('label_usage')}`",
                f"- Routing rule: `{cggr_impl.get('routing_rule')}`",
                f"- Token budgets: `{json.dumps(token_budgets, sort_keys=True)}`",
                f"- Thresholds/ablations: `{json.dumps(thresholds, sort_keys=True)}`",
                "",
                "The active run is therefore a fixed proxy-gated instantiation of the counterfactual-gain formulation, not evidence that a learned routing estimator has been trained.",
                "",
            ]
        )
    if claim_scope:
        lines.extend(
            [
                "## Claim Scope Override",
                "",
                f"- Active method claim: `{claim_scope.get('active_method_claim')}`",
                f"- Learned-router claim allowed: `{claim_scope.get('learned_router_claim_allowed')}`",
                f"- Trained-estimator claim allowed: `{claim_scope.get('trained_estimator_claim_allowed')}`",
                f"- Oracle router in active contract: `{claim_scope.get('oracle_router_in_active_contract')}`",
                f"- Broad top-venue/SOTA superiority allowed: `{claim_scope.get('broad_top_venue_or_sota_superiority_claim_allowed')}`",
                "",
            ]
        )
    return "\n".join(lines)


def _claim_evidence_map(summary: dict, audit_result: dict) -> str:
    bootstrap = summary.get("bootstrap_ci") if isinstance(summary.get("bootstrap_ci"), dict) else {}
    claim_scope = summary.get("claim_scope_override") if isinstance(summary.get("claim_scope_override"), dict) else {}
    top_gate = (
        "strict top-venue audit passed"
        if audit_result.get("require_top_venue_baselines")
        else "blocked unless `--require-top-venue-baselines` passes"
    )
    scope_gate = (
        "fixed proxy-gated scope recorded in `active_claim_scope_override.json`"
        if claim_scope
        else "no explicit claim-scope override found"
    )
    return "\n".join(
        [
            "# Claim Evidence Map",
            "",
            "| Claim type | Evidence artifact | Required statistic |",
            "| --- | --- | --- |",
            f"| Main utility comparison | `main_results_table.tex` | `{bootstrap.get('candidate_method', 'CGGR')}` vs `{bootstrap.get('baseline_method', 'Always-Reason Chain-of-Thought')}`, 95% CI, seed std. |",
            "| Cost/latency trade-off | `cost_latency_table.tex` | Tokens/example, latency/example, route rate. |",
            "| Ablation mechanism | `ablation_table.tex` | Delta vs. CGGR for each registered ablation. |",
            "| Method implementation | `reproducibility_statement.md` and `routing_decisions.jsonl` | Fixed proxy-gated CGGR instantiation, token-budget thresholds, and ablation mapping. |",
            "| Significance | `significance_report.md` | Delta CI and paired permutation p-value. |",
            "| Failure analysis | `failure_analysis.md` | Failure rows by method/dataset/stage with examples retained in the denominator. |",
            "| Reproducibility | `reproducibility_statement.md` | Model, datasets, seeds, hardware, and audit status. |",
            f"| Claim scope | `active_claim_scope_override.json` and `claim_values.json` | {scope_gate}. |",
            f"| Broad adaptive-reasoning superiority | Top-venue baseline shard and strict audit | {top_gate}; requires CAR-style, Self-Route-style, and VOC-style baseline coverage. |",
            "",
        ]
    )


def _completion_audit(summary: dict, audit_result: dict) -> str:
    bootstrap = summary.get("bootstrap_ci") if isinstance(summary.get("bootstrap_ci"), dict) else {}
    datasets = summary.get("datasets") if isinstance(summary.get("datasets"), list) else []
    dataset_names = [str(row.get("name") or row.get("id") or "") for row in datasets if isinstance(row, dict)]
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    decision = _claim_support_decision(summary)
    claim_scope = summary.get("claim_scope_override") if isinstance(summary.get("claim_scope_override"), dict) else {}
    warnings = audit_result.get("warnings") if isinstance(audit_result.get("warnings"), list) else []
    lines = [
        "# Completion Audit",
        "",
        "This audit is generated only after `scripts/audit_paper_benchmark_artifacts.py --require-full` accepts the run artifact. It is an artifact-level paper gate, not a proxy for uninspected claims.",
        "",
        "## Objective As Artifact Criteria",
        "",
        "- Use a real full benchmark rather than smoke, sanity, or reproduction-only evidence.",
        "- Require raw per-example coverage for the locked method, dataset, seed, and example matrix.",
        "- Materialize manuscript tables, snippets, figure code, significance, reproducibility, cost, routing, ablation, and claim-evidence files only after audit passes.",
        "- Admit empirical claims only when claim support decisions are tied to audited files and statistics.",
        "",
        "## Prompt-To-Artifact Checklist",
        "",
        "| Requirement | Evidence | Status |",
        "| --- | --- | --- |",
        f"| Full artifact audit | `audit.ok={audit_result.get('ok')}`, `require_full={audit_result.get('require_full')}` | passed |",
        f"| Full benchmark marker | `full_benchmark_completed={audit_result.get('full_benchmark_completed')}` | passed |",
        f"| Raw prediction coverage | `{audit_result.get('raw_predictions_lines')}` raw rows; duplicate/empty/zero-token gates passed by audit | passed |",
        f"| Dataset coverage | `{', '.join(dataset_names)}` | passed |",
        f"| Method coverage | `{', '.join(sorted(str(method) for method in per_method))}` | passed |",
        f"| Seed coverage | `num_seeds={summary.get('num_seeds')}` | passed |",
        "| Main table | `main_results_table.tex` generated from audited summary | passed |",
        "| Cost and routing table | `cost_latency_table.tex` generated from audited cost/routing diagnostics | passed |",
        "| Ablation table | `ablation_table.tex` generated from registered CGGR ablations | passed |",
        f"| Significance | `significance_report.md`; delta CI `{bootstrap.get('delta_ci95')}`; paired p `{bootstrap.get('paired_permutation_p')}` | passed |",
        "| Reproducibility | `reproducibility_statement.md` generated from audited run metadata | passed |",
        "| Failure analysis | `failure_analysis.md` generated from audited failure rows | passed |",
        f"| Claim scope override | `active_claim_scope_override.json`; learned-router allowed `{claim_scope.get('learned_router_claim_allowed') if claim_scope else None}`; broad SOTA allowed `{claim_scope.get('broad_top_venue_or_sota_superiority_claim_allowed') if claim_scope else None}` | {'passed' if claim_scope else 'missing'} |",
        f"| Claim decision | `claim_values.json`; `claim_support_decision={decision}` | {decision} |",
        f"| Top-venue general-superiority gate | `require_top_venue_baselines={audit_result.get('require_top_venue_baselines')}` | {'passed' if audit_result.get('require_top_venue_baselines') else 'blocked'} |",
        "| Manuscript snippets | `results_section_snippet.tex`, `limitations_snippet.tex`, `utility_comparison_figure.tex` | passed |",
        "",
        "## Residual Warnings",
        "",
    ]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None reported by the artifact audit.")
    lines.extend(["", "Empirical prose must still quote the support decision and cite these materialized artifacts rather than hand-copying numbers from raw JSON.", ""])
    return "\n".join(lines)


def _submission_contract_artifacts(summary: dict, audit_result: dict, claim_values: dict) -> dict[str, Any]:
    datasets = summary.get("datasets") if isinstance(summary.get("datasets"), list) else []
    dataset_names = [str(row.get("name") or row.get("id") or "") for row in datasets if isinstance(row, dict)]
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    methods = sorted(str(method) for method in per_method)
    bootstrap = summary.get("bootstrap_ci") if isinstance(summary.get("bootstrap_ci"), dict) else {}
    claim_scope = summary.get("claim_scope_override") if isinstance(summary.get("claim_scope_override"), dict) else {}
    top_venue_decision = claim_values.get("top_venue_general_superiority_decision")
    claim_decision = claim_values.get("claim_support_decision")
    base_artifacts = [
        "main_results_table.tex",
        "ablation_table.tex",
        "cost_latency_table.tex",
        "significance_report.md",
        "failure_analysis.md",
        "reproducibility_statement.md",
        "claim_evidence_map.md",
        "completion_audit.md",
        "claim_values.json",
    ]
    problem_awareness = {
        "central_question": "When should a QA system answer immediately versus spend extra reasoning budget?",
        "method": "Counterfactual Gain Gated Reasoning",
        "active_executable_scope": claim_scope.get("active_method_claim")
        or "fixed proxy-gated selective-deliberation runner",
        "benchmark_scope": {
            "model": (summary.get("model") or {}).get("id") if isinstance(summary.get("model"), dict) else None,
            "datasets": dataset_names,
            "num_seeds": summary.get("num_seeds"),
            "methods": methods,
        },
        "blocked_overclaims": {
            "learned_router_claim_allowed": claim_values.get("learned_router_claim_allowed"),
            "trained_estimator_claim_allowed": claim_values.get("trained_estimator_claim_allowed"),
            "oracle_router_in_active_contract": claim_values.get("oracle_router_in_active_contract"),
            "broad_top_venue_or_sota_superiority_claim_allowed": claim_values.get(
                "broad_top_venue_or_sota_superiority_claim_allowed"
            ),
        },
    }
    publication_contract = {
        "evidence_tier": "audited_full_benchmark",
        "blocks_manuscript": False,
        "manuscript_allowed": True,
        "claim_scope": "locked-baseline scoped claims only unless strict top-venue audit passes",
        "minimum_seeds": summary.get("num_seeds"),
        "required_real_benchmarks": dataset_names,
        "required_methods": methods,
        "primary_metric": summary.get("primary_metric") or "utility",
        "statistical_test": "paired bootstrap confidence interval plus paired permutation test",
        "quality_gates": {
            "audit_ok": audit_result.get("ok"),
            "requires_full_benchmark_package": audit_result.get("require_full"),
            "full_benchmark_completed": audit_result.get("full_benchmark_completed"),
            "raw_predictions_lines": audit_result.get("raw_predictions_lines"),
            "claim_support_decision": claim_decision,
            "top_venue_general_superiority_decision": top_venue_decision,
        },
        "claim_route": {
            "route": "scoped_full_paper",
            "paper_allowed": True,
            "broad_top_venue_claim_allowed": top_venue_decision == "eligible_under_strict_top_venue_audit",
        },
    }
    evidence_manifest = {
        "source_workdir": str(claim_values.get("source_workdir") or ""),
        "audit": audit_result,
        "summary": {
            "datasets": dataset_names,
            "methods": methods,
            "num_seeds": summary.get("num_seeds"),
            "raw_predictions_lines": audit_result.get("raw_predictions_lines"),
            "claim_support_decision": claim_decision,
            "top_venue_general_superiority_decision": top_venue_decision,
        },
        "materialized_artifacts": base_artifacts,
    }
    claim_evidence_matrix = [
        {
            "claim_id": "C1_scoped_main_utility",
            "claim": "CGGR utility versus the locked baseline set under the audited run contract.",
            "decision": claim_decision,
            "evidence": ["main_results_table.tex", "significance_report.md", "claim_values.json"],
            "statistics": {
                "observed_delta": bootstrap.get("observed_delta"),
                "delta_ci95": bootstrap.get("delta_ci95"),
                "paired_permutation_p": bootstrap.get("paired_permutation_p"),
            },
        },
        {
            "claim_id": "C2_cost_routing",
            "claim": "Cost, latency, and routing behavior for deployable methods.",
            "decision": "materialized",
            "evidence": ["cost_latency_table.tex", "reproducibility_statement.md"],
        },
        {
            "claim_id": "C3_ablations",
            "claim": "Registered CGGR ablation comparisons.",
            "decision": "materialized",
            "evidence": ["ablation_table.tex", "claim_evidence_map.md"],
        },
        {
            "claim_id": "C4_top_venue_general_superiority",
            "claim": "Broad superiority over current adaptive-reasoning/routing methods.",
            "decision": top_venue_decision,
            "evidence": ["claim_values.json", "completion_audit.md"],
        },
    ]
    reviewer_report = {
        "status": "ready_for_scoped_claim_review",
        "high_risk_overclaims": [
            "Do not claim first adaptive-reasoning method.",
            "Do not claim broad SOTA/general superiority unless the strict top-venue baseline audit passes.",
            "Do not describe the active runner as a learned router or trained estimator.",
        ],
        "audit_warnings": claim_values.get("audit_warnings") or [],
        "required_manual_checks_before_submission": [
            "Compile main.tex after audited snippets are included.",
            "Run manuscript_watchdog.py on the final bundle.",
            "Confirm every empirical sentence cites materialized audited artifacts.",
        ],
    }
    paper_quality_report = {
        "status": "artifact_gate_passed_scoped_claims",
        "paper_grade_benchmark": True,
        "smoke_test_evidence_used": False,
        "watchdog_contract_files_materialized": True,
        "claim_support_decision": claim_decision,
        "top_venue_general_superiority_decision": top_venue_decision,
        "remaining_submission_blockers": [
            "Final manuscript bundle must include compiled PDF, ICLR template files, figures, citations, and no evidence-pending placeholders.",
        ],
    }
    return {
        "problem_awareness.json": problem_awareness,
        "publication_evidence_contract.json": publication_contract,
        "evidence_manifest.json": evidence_manifest,
        "claim_evidence_matrix.json": claim_evidence_matrix,
        "reviewer_report.json": reviewer_report,
        "paper_quality_report.json": paper_quality_report,
    }


def _claim_support_decision(summary: dict) -> str:
    bootstrap = summary.get("bootstrap_ci") if isinstance(summary.get("bootstrap_ci"), dict) else {}
    try:
        delta = float(bootstrap.get("observed_delta"))
    except (TypeError, ValueError):
        delta = 0.0
    try:
        p_value = float(bootstrap.get("paired_permutation_p"))
    except (TypeError, ValueError):
        p_value = 1.0
    delta_ci = bootstrap.get("delta_ci95") if isinstance(bootstrap.get("delta_ci95"), list) else []
    try:
        delta_lo = float(delta_ci[0])
    except (TypeError, ValueError, IndexError):
        delta_lo = 0.0
    if delta > 0.0 and delta_lo > 0.0 and p_value < 0.05:
        return "supported"
    if delta > 0.0:
        return "downgraded"
    return "rejected"


def _results_section_snippet(summary: dict, audit_result: dict) -> str:
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    bootstrap = summary.get("bootstrap_ci") if isinstance(summary.get("bootstrap_ci"), dict) else {}
    candidate = bootstrap.get("candidate_method") or summary.get("candidate_method") or "CGGR"
    baseline = bootstrap.get("baseline_method") or "Always-Reason Chain-of-Thought"
    candidate_row = per_method.get(candidate) if isinstance(per_method.get(candidate), dict) else {}
    baseline_row = per_method.get(baseline) if isinstance(per_method.get(baseline), dict) else {}
    decision = _claim_support_decision(summary)
    if decision == "supported":
        decision_text = "The audited evidence supports the CGGR-over-baseline utility claim under this locked benchmark contract."
    elif decision == "downgraded":
        decision_text = "The audited evidence shows a positive point estimate, but the manuscript should phrase it as limited or suggestive because the confidence interval or permutation test does not clear the support gate."
    else:
        decision_text = "The audited evidence does not support a CGGR superiority claim; the manuscript should report the result as negative or inconclusive."
    delta_ci = bootstrap.get("delta_ci95")
    if isinstance(delta_ci, list) and len(delta_ci) >= 2:
        delta_ci_text = f"[{_num(delta_ci[0])}, {_num(delta_ci[1])}]"
    else:
        delta_ci_text = "--"
    lines = [
        r"\paragraph{Audited main result.}",
        (
            f"The audited full benchmark contains {audit_result.get('raw_predictions_lines')} raw predictions "
            f"over {summary.get('num_seeds')} seeds. "
            f"{_tex_escape(candidate)} obtains utility {_num(candidate_row.get('metric_value'))} "
            f"with quality {_num(candidate_row.get('score'))}, compared with "
            f"{_tex_escape(baseline)} utility {_num(baseline_row.get('metric_value'))} "
            f"and quality {_num(baseline_row.get('score'))}."
        ),
        (
            f"The paired seed-level utility delta is {_num(bootstrap.get('observed_delta'))} "
            f"with 95\\% bootstrap CI {_tex_escape(delta_ci_text)} "
            f"and paired permutation p-value {_num(bootstrap.get('paired_permutation_p'))}."
        ),
        _tex_escape(decision_text),
        "",
    ]
    return "\n".join(lines)


def _limitations_snippet(summary: dict, audit_result: dict) -> str:
    model = summary.get("model") if isinstance(summary.get("model"), dict) else {}
    datasets = summary.get("datasets") if isinstance(summary.get("datasets"), list) else []
    dataset_count = len(datasets)
    lines = [
        r"\paragraph{Evidence scope.}",
        (
            f"The empirical claim is scoped to the audited contract: model {_tex_escape(model.get('id') or model.get('name'))}, "
            f"{dataset_count} benchmark targets, {summary.get('num_seeds')} seeds, and the registered CGGR baselines/ablations. "
            "The result should not be described as state of the art or generalized to other model families unless additional audited contracts are added."
        ),
        "",
    ]
    if not audit_result.get("require_top_venue_baselines"):
        lines.extend(
            [
                r"\paragraph{Adaptive-reasoning baseline scope.}",
                (
                    "This materialized package did not require the stricter top-venue baseline audit. "
                    "Therefore, broad superiority over current adaptive-reasoning or routing methods remains blocked until "
                    "CAR-style certainty adaptive routing, Self-Route-style mode routing, and rational-metareasoning "
                    "value-of-computation routing are included in a merged artifact that passes the stricter audit."
                ),
                "",
            ]
        )
    warnings = audit_result.get("warnings") if isinstance(audit_result.get("warnings"), list) else []
    if warnings:
        routing_warnings = sum(1 for item in warnings if str(item).startswith("routing rate is"))
        token_cap_warnings = sum(1 for item in warnings if str(item).startswith("token cap hit rate is"))
        other_warnings = len(warnings) - routing_warnings - token_cap_warnings
        lines.extend(
            [
                r"\paragraph{Non-blocking audit warnings.}",
                (
                    "The full artifact audit passed but reported diagnostics that should be interpreted in the "
                    "failure-analysis and limitations discussion. "
                    f"There were {routing_warnings} per-slice route-rate saturation warnings, "
                    f"{token_cap_warnings} token-cap warnings, and {other_warnings} other warnings. "
                    "These warnings do not invalidate the artifact, but they restrict the claim to the audited "
                    "contract and motivate follow-up work on smoother routing and longer decoding budgets."
                ),
                "",
            ]
        )
    return "\n".join(lines)


def _failure_analysis(workdir: Path) -> str:
    rows = _iter_jsonl(workdir / "results" / "failure_cases.jsonl")
    by_method: dict[str, int] = {}
    by_dataset: dict[str, int] = {}
    by_stage: dict[str, int] = {}
    by_error: dict[str, int] = {}
    for row in rows:
        method = str(row.get("method") or "<no_method>")
        dataset = str(row.get("dataset") or "<no_dataset>")
        stage = str(row.get("stage") or "<no_stage>")
        error = str(row.get("error_type") or row.get("error") or row.get("error_repr") or "<no_error>")
        by_method[method] = by_method.get(method, 0) + 1
        by_dataset[dataset] = by_dataset.get(dataset, 0) + 1
        by_stage[stage] = by_stage.get(stage, 0) + 1
        by_error[error] = by_error.get(error, 0) + 1

    def _table(title: str, values: dict[str, int]) -> list[str]:
        lines = [f"## {title}", "", "| Key | Count |", "| --- | ---: |"]
        if values:
            for key, count in sorted(values.items(), key=lambda item: (-item[1], item[0])):
                lines.append(f"| `{key}` | {count} |")
        else:
            lines.append("| `<none>` | 0 |")
        lines.append("")
        return lines

    lines = [
        "# Failure Analysis",
        "",
        f"- Failure rows: `{len(rows)}`",
        "- These rows are materialized only after the full artifact audit passes.",
        "- Generation/scoring failures are audit blockers; remaining rows should be interpreted as model failure-analysis examples retained in the denominator.",
        "",
    ]
    lines.extend(_table("By Method", by_method))
    lines.extend(_table("By Dataset", by_dataset))
    lines.extend(_table("By Stage", by_stage))
    lines.extend(_table("By Error", by_error))
    lines.extend(["## Example Rows", ""])
    if rows:
        lines.append("| Method | Dataset | Seed | Example | Score | Prediction | Gold |")
        lines.append("| --- | --- | --- | --- | ---: | --- | --- |")
        for row in rows[:5]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{str(row.get('method') or '')}`",
                        f"`{str(row.get('dataset') or '')}`",
                        f"`{str(row.get('seed') or '')}`",
                        f"`{str(row.get('example_id') or '')}`",
                        _num(row.get("primary_score")),
                        _tex_escape(str(row.get("prediction_answer") or row.get("prediction") or ""))[:160],
                        _tex_escape(str(row.get("gold") or ""))[:160],
                    ]
                )
                + " |"
            )
    else:
        lines.append("No failure rows were recorded.")
    lines.append("")
    return "\n".join(lines)


def materialize(workdir: Path, out_dir: Path, *, require_top_venue_baselines: bool = False) -> dict:
    audit_result = audit(
        workdir,
        require_full=True,
        require_top_venue_baselines=require_top_venue_baselines,
    )
    if not audit_result["ok"]:
        return {"ok": False, "audit": audit_result, "written": []}

    results = workdir / "results"
    summary = _load_json(results / "benchmark_summary.json")
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "main_results_table.tex": _main_table(summary),
        "ablation_table.tex": _ablation_table(summary),
        "cost_latency_table.tex": _cost_table(summary),
        "significance_report.md": _significance_report(summary, audit_result),
        "failure_analysis.md": _failure_analysis(workdir),
        "reproducibility_statement.md": _reproducibility_statement(summary, audit_result),
        "claim_evidence_map.md": _claim_evidence_map(summary, audit_result),
        "completion_audit.md": _completion_audit(summary, audit_result),
        "results_section_snippet.tex": _results_section_snippet(summary, audit_result),
        "limitations_snippet.tex": _limitations_snippet(summary, audit_result),
        "utility_comparison_figure.tex": _utility_comparison_figure(summary),
    }
    written = []
    for name, content in outputs.items():
        path = out_dir / name
        path.write_text(content, encoding="utf-8")
        written.append(str(path))

    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    cggr = per_method.get("CGGR") if isinstance(per_method.get("CGGR"), dict) else {}
    best_method = str(summary.get("best_method") or "")
    implementations = summary.get("method_implementation") if isinstance(summary.get("method_implementation"), dict) else {}
    cggr_impl = implementations.get("CGGR") if isinstance(implementations.get("CGGR"), dict) else {}
    claim_scope = summary.get("claim_scope_override") if isinstance(summary.get("claim_scope_override"), dict) else {}
    claim_values = {
        "source_workdir": str(workdir),
        "audit_ok": True,
        "full_benchmark_completed": bool(summary.get("full_benchmark_completed")),
        "best_method": best_method,
        "candidate_method": (summary.get("bootstrap_ci") or {}).get("candidate_method"),
        "baseline_method": (summary.get("bootstrap_ci") or {}).get("baseline_method"),
        "cggr_utility": cggr.get("metric_value"),
        "cggr_quality": cggr.get("score"),
        "cggr_tokens_per_example": cggr.get("avg_new_tokens"),
        "cggr_latency_per_example": cggr.get("avg_latency_seconds"),
        "cggr_seed_std": (summary.get("per_method_std") or {}).get("CGGR"),
        "cggr_implementation_type": cggr_impl.get("estimator_type"),
        "cggr_trained_estimator": cggr_impl.get("trained_estimator"),
        "claim_scope_override": claim_scope,
        "learned_router_claim_allowed": claim_scope.get("learned_router_claim_allowed"),
        "trained_estimator_claim_allowed": claim_scope.get("trained_estimator_claim_allowed"),
        "oracle_router_in_active_contract": claim_scope.get("oracle_router_in_active_contract"),
        "broad_top_venue_or_sota_superiority_claim_allowed": claim_scope.get(
            "broad_top_venue_or_sota_superiority_claim_allowed"
        ),
        "cggr_vs_baseline_delta": (summary.get("bootstrap_ci") or {}).get("observed_delta"),
        "cggr_vs_baseline_delta_ci95": (summary.get("bootstrap_ci") or {}).get("delta_ci95"),
        "paired_permutation_p": (summary.get("bootstrap_ci") or {}).get("paired_permutation_p"),
        "claim_support_decision": _claim_support_decision(summary),
        "top_venue_general_superiority_decision": (
            "eligible_under_strict_top_venue_audit"
            if audit_result.get("require_top_venue_baselines")
            else "blocked_missing_strict_top_venue_baseline_audit"
        ),
        "bootstrap_ci": summary.get("bootstrap_ci"),
        "audit_warnings": audit_result.get("warnings", []),
    }
    claim_values_path = out_dir / "claim_values.json"
    claim_values_path.write_text(json.dumps(claim_values, indent=2, ensure_ascii=False), encoding="utf-8")
    written.append(str(claim_values_path))
    for name, payload in _submission_contract_artifacts(summary, audit_result, claim_values).items():
        path = out_dir / name
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        written.append(str(path))
    return {"ok": True, "audit": audit_result, "written": written}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--require-top-venue-baselines", action="store_true")
    args = parser.parse_args()
    result = materialize(
        args.workdir,
        args.out_dir,
        require_top_venue_baselines=args.require_top_venue_baselines,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
