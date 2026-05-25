"""Tests for post-pick_main_tex submission enrichment."""

from __future__ import annotations

from pathlib import Path

from agents.manuscript_submission_enrichment import (
    audit_venue_section_lengths,
    build_main_results_table_tex_from_state,
    enrich_submission_main_tex,
)


def _minimal_tex() -> str:
    return r"""
\documentclass{article}
\begin{document}
\begin{abstract}
Short abstract with only a few words here for testing purposes only.
\end{abstract}
\section{Introduction}
Intro paragraph one. Intro paragraph two.
\paragraph{Contributions.}
\begin{itemize}
\item First contribution item.
\item Second contribution item.
\item Third contribution item.
\end{itemize}
\section{Related Work}
Related work paragraph with some citations and prior art discussion.
\section{Method}
Method details and algorithm description for the proposed approach.
\section{Experiments}
Experiment setup and results discussion without figures yet.
\end{document}
"""


def test_enrich_injects_figures_and_table(tmp_path: Path) -> None:
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()
    mot = fig_dir / "fig_motivation_gemini.png"
    ov = fig_dir / "fig_overview_gemini.png"
    main = fig_dir / "fig_main_results.png"
    for p in (mot, ov, main):
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 64)

    orchestrated = {
        "plotting": {
            "assets": [
                {
                    "figure_id": "fig_motivation",
                    "path": str(mot),
                    "objective": "Motivation diagram",
                },
                {
                    "figure_id": "fig_overview",
                    "path": str(ov),
                    "objective": "Overview diagram",
                },
                {
                    "figure_id": "fig_main_results",
                    "path": str(main),
                    "objective": "Main benchmark plot",
                },
            ],
            "plotting_plan": [
                {"figure_id": "fig_main_results", "role": "main_result"},
            ],
        }
    }
    state = {
        "main_results_table": {
            "baseline": {"primary_score": 0.5},
            "ours": {"primary_score": 0.7},
            "oracle": {"primary_score": 0.9},
        },
        "baseline_metric_name": "primary_score",
    }
    tex, meta = enrich_submission_main_tex(_minimal_tex(), orchestrated, state)
    assert "fig_motivation_gemini.png" in tex or "motivation_gemini.png" in tex
    assert "fig_overview" in tex or "overview_gemini" in tex
    assert r"\begin{table}" in tex
    assert meta["injected_table"]
    assert len(meta["injected_figures"]) >= 2
    assert tex.count(r"\includegraphics") >= 2


def test_build_main_results_table_from_json_state() -> None:
    state = {
        "main_results_table": {
            "method_a": {"acc": 0.91},
            "method_b": {"acc": 0.88},
        }
    }
    tex = build_main_results_table_tex_from_state(state)
    assert r"\begin{table}" in tex
    assert "method\\_a" in tex or "method_a" in tex


def test_audit_venue_section_lengths_flags_short_sections() -> None:
    audit = audit_venue_section_lengths(_minimal_tex(), template_id="iclr2026")
    assert audit["pass"] is False
    assert any(f.get("section") == "abstract" for f in audit["failures"])
