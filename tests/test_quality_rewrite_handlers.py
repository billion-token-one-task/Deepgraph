"""Tests for the quality-rewrite surgical handlers (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.manuscript_quality_rewrite import (
    _drop_internal_audit_wording,
    _drop_missing_figure_includes,
    _drop_offtopic_bib_entries,
    _match_handler,
    _strip_missing_cite_keys,
)


def test_match_handler_matches_known_prefixes() -> None:
    assert _match_handler("PDF compile did not pass.") == "compile"
    assert _match_handler(
        "The manuscript has citations that are absent from references.bib."
    ) == "missing_cites"
    assert _match_handler("Venue section gate: Discussion") == "venue_section"
    assert _match_handler("nonsense") is None


def test_strip_missing_cite_keys_keeps_others() -> None:
    tex = r"See \cite{foo, bar} and \cite{bar} and \citet{missing}."
    bib = "@article{foo, ...}\n@article{bar, ...}\n"
    out, notes = _strip_missing_cite_keys(tex, bib)
    assert "missing" not in out
    assert r"\cite{foo, bar}" in out or r"\cite{foo,bar}" in out
    assert any("missing" in n for n in notes)


def test_drop_internal_audit_wording_replaces_phrases() -> None:
    tex = "We rely on supplied artifacts and provided materials in the run."
    out, notes = _drop_internal_audit_wording(tex)
    assert "supplied artifact" not in out.lower()
    assert "provided material" not in out.lower()
    assert notes


def test_drop_missing_figure_includes(tmp_path: Path) -> None:
    bundle = tmp_path
    (bundle / "figures").mkdir()
    (bundle / "figures" / "real.png").write_bytes(b"\x89PNG\r\n")
    tex = (
        r"\begin{figure}[t]"
        "\n"
        r"\includegraphics{figures/real.png}"
        "\n"
        r"\label{fig:r}"
        "\n"
        r"\end{figure}"
        "\n"
        r"\begin{figure}[t]"
        "\n"
        r"\includegraphics{figures/missing.png}"
        "\n"
        r"\label{fig:m}"
        "\n"
        r"\end{figure}"
        "\n"
    )
    out, notes = _drop_missing_figure_includes(tex, bundle)
    assert "missing.png" in "\n".join(notes)
    assert r"\includegraphics{figures/missing.png}" not in out
    assert r"\begin{figure}" in out  # the real figure environment remains
    assert "real.png" in out
    # The removed figure's label should not remain inside a live environment;
    # the comment-only stub keeps the file diff minimal.
    assert out.count(r"\end{figure}") == 1


def test_drop_offtopic_bib_entries() -> None:
    bib = (
        "@article{good, title={Vision-language reasoning}, journal={NeurIPS}}\n"
        "@article{bad, title={Pediatric tumor segmentation in CT}, journal={MedIA}}\n"
    )
    out, notes = _drop_offtopic_bib_entries(bib, ("tumor",))
    assert "@article{good" in out
    assert "@article{bad" not in out
    assert notes


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
