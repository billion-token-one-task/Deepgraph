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
            r"    \caption{Audited utility comparison for deployable methods in the locked run47 benchmark contract.}",
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
        f"- Full benchmark completed: `{audit_result.get('full_benchmark_completed')}`",
        "",
        "All result tables are downstream of `scripts/audit_paper_benchmark_artifacts.py --require-full`. Smoke, sanity, invalidated, and unmerged shard runs are not admissible for empirical claims.",
        "",
    ]
    return "\n".join(lines)


def _claim_evidence_map(summary: dict) -> str:
    bootstrap = summary.get("bootstrap_ci") if isinstance(summary.get("bootstrap_ci"), dict) else {}
    return "\n".join(
        [
            "# Claim Evidence Map",
            "",
            "| Claim type | Evidence artifact | Required statistic |",
            "| --- | --- | --- |",
            f"| Main utility comparison | `main_results_table.tex` | `{bootstrap.get('candidate_method', 'CGGR')}` vs `{bootstrap.get('baseline_method', 'Always-Reason Chain-of-Thought')}`, 95% CI, seed std. |",
            "| Cost/latency trade-off | `cost_latency_table.tex` | Tokens/example, latency/example, route rate. |",
            "| Ablation mechanism | `ablation_table.tex` | Delta vs. CGGR for each registered ablation. |",
            "| Significance | `significance_report.md` | Delta CI and paired permutation p-value. |",
            "| Reproducibility | `reproducibility_statement.md` | Model, datasets, seeds, hardware, and audit status. |",
            "",
        ]
    )


def _completion_audit(summary: dict, audit_result: dict) -> str:
    bootstrap = summary.get("bootstrap_ci") if isinstance(summary.get("bootstrap_ci"), dict) else {}
    datasets = summary.get("datasets") if isinstance(summary.get("datasets"), list) else []
    dataset_names = [str(row.get("name") or row.get("id") or "") for row in datasets if isinstance(row, dict)]
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    decision = _claim_support_decision(summary)
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
        f"| Claim decision | `claim_values.json`; `claim_support_decision={decision}` | {decision} |",
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
            f"with 95\\% bootstrap CI {_tex_escape(bootstrap.get('delta_ci95'))} "
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
    warnings = audit_result.get("warnings") if isinstance(audit_result.get("warnings"), list) else []
    if warnings:
        shown = "; ".join(_tex_escape(item) for item in warnings[:5])
        if len(warnings) > 5:
            shown += f"; and {len(warnings) - 5} additional audit warnings"
        lines.extend(
            [
                r"\paragraph{Non-blocking audit warnings.}",
                (
                    "The full artifact audit passed but reported diagnostics that should be interpreted in the "
                    f"failure-analysis and limitations discussion: {shown}."
                ),
                "",
            ]
        )
    return "\n".join(lines)


def materialize(workdir: Path, out_dir: Path) -> dict:
    audit_result = audit(workdir, require_full=True)
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
        "reproducibility_statement.md": _reproducibility_statement(summary, audit_result),
        "claim_evidence_map.md": _claim_evidence_map(summary),
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
        "cggr_vs_baseline_delta": (summary.get("bootstrap_ci") or {}).get("observed_delta"),
        "cggr_vs_baseline_delta_ci95": (summary.get("bootstrap_ci") or {}).get("delta_ci95"),
        "paired_permutation_p": (summary.get("bootstrap_ci") or {}).get("paired_permutation_p"),
        "claim_support_decision": _claim_support_decision(summary),
        "bootstrap_ci": summary.get("bootstrap_ci"),
        "audit_warnings": audit_result.get("warnings", []),
    }
    claim_values_path = out_dir / "claim_values.json"
    claim_values_path.write_text(json.dumps(claim_values, indent=2, ensure_ascii=False), encoding="utf-8")
    written.append(str(claim_values_path))
    return {"ok": True, "audit": audit_result, "written": written}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(args.workdir, args.out_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
