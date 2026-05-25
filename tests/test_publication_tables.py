"""Publication table builders."""

from __future__ import annotations

from agents.manuscript_publication_tables import (
    build_publication_ablation_table_tex,
    build_publication_main_results_table_tex,
    replace_tables_in_tex,
)


def test_publication_main_table_uses_booktabs_and_bold_best() -> None:
    state = {
        "method_name": "Contrastive Perceptual Grounding (CPG)",
        "baseline_metric_name": "primary_score",
        "main_results_table": {
            "Vanilla": {"primary_score": 0.30},
            "CPG": {"primary_score": 0.27},
        },
    }
    tex = build_publication_main_results_table_tex(state)
    assert r"\toprule" in tex
    assert r"\label{tab:main_results}" in tex
    assert r"\begin{tabularx}{\linewidth}" in tex
    assert r"\textbf{" in tex
    assert "0.3000" in tex or "0.30" in tex


def test_ablation_table_from_contract() -> None:
    state = {
        "baseline_metric_value": 0.27,
        "baseline_metric_name": "primary_score",
        "publication_evidence_contract": {
            "required_ablations": ["no_discrepancy", "no_classifier", "single_stream"],
        },
    }
    tex = build_publication_ablation_table_tex(state)
    assert r"\label{tab:ablations}" in tex
    assert "discrepancy" in tex
    assert r"\textsc{Pending}" in tex


def test_ablation_disable_property_labels_not_truncated() -> None:
    state = {
        "baseline_metric_name": "primary_score",
        "benchmark_summary": {
            "ablations": {
                "disable_property_1__grounding_discrepancy___delt": {
                    "executed": False,
                    "primary_score": None,
                },
            }
        },
    }
    tex = build_publication_ablation_table_tex(state)
    assert "grounding discrepancy" in tex
    assert "delt &" not in tex


def test_replace_tables_injects_ablation_subsection() -> None:
    state = {
        "baseline_metric_value": 0.27,
        "publication_evidence_contract": {"required_ablations": ["ablation_a"]},
        "main_results_table": {"M1": {"primary_score": 0.1}},
    }
    body = r"\documentclass{article}\begin{document}\section{Experiments}\subsection{Main Results}Text.\end{document}"
    out, meta = replace_tables_in_tex(body, state)
    assert meta.get("injected_ablation_table") or r"\label{tab:ablations}" in out
