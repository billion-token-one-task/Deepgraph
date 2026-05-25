"""Figure reference policy: Gemini PNG vs stale PDF, section title cleanup."""

from __future__ import annotations

import time
from pathlib import Path

from agents.paper_orchestra_pipeline import (
    _prefer_vector_figure_references,
    _purge_stale_gemini_vector_companions,
    strip_manual_section_numbers,
)


def test_strip_manual_section_numbers_removes_duplicate_prefix() -> None:
    tex = r"""
\section{Introduction}
\subsection{2.1 Self-Verification}
\subsection{2.2 Contrastive Learning}
"""
    out = strip_manual_section_numbers(tex)
    assert r"\subsection{Self-Verification}" in out
    assert r"\subsection{Contrastive Learning}" in out
    assert "2.2" not in out.split("Contrastive")[0][-10:]


def test_prefer_vector_keeps_gemini_png_over_stale_pdf(tmp_path: Path) -> None:
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()
    png = fig_dir / "fig_motivation_gemini.png"
    pdf = fig_dir / "fig_motivation_gemini.pdf"
    pdf.write_bytes(b"%PDF-1.4 tiny")
    time.sleep(0.02)
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 32)
    _purge_stale_gemini_vector_companions(fig_dir)
    assert not pdf.exists()
    tex = r"\includegraphics{figures/fig_motivation_gemini.pdf}"
    out = _prefer_vector_figure_references(tmp_path, tex)
    assert "fig_motivation_gemini.png" in out
    assert ".pdf" not in out or "gemini.pdf" not in out


def test_prefer_vector_uses_newer_png_for_non_gemini(tmp_path: Path) -> None:
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()
    pdf = fig_dir / "fig_main_results.pdf"
    png = fig_dir / "fig_main_results.png"
    pdf.write_bytes(b"%PDF-1.4")
    time.sleep(0.02)
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 32)
    tex = r"\includegraphics{figures/fig_main_results.pdf}"
    out = _prefer_vector_figure_references(tmp_path, tex)
    assert "fig_main_results.png" in out
