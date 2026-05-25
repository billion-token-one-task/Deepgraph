"""Tests for visual layout + figure dedup audit (no PDF required)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from agents.manuscript_visual_audit import (
    _bbox_overlap_fraction,
    _dhash,
    _hamming,
    apply_visual_patches,
    audit_bundle_visuals,
    collect_figures,
    detect_unreferenced_floats,
    find_duplicate_figures,
)


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    img = Image.new("RGB", (64, 64), color)
    img.save(path)


def _write_textured_png(path: Path, seed: int) -> None:
    """Solid-colour PNGs all hash to zero; tests need real texture."""
    import random

    rng = random.Random(seed)
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    pixels = img.load()
    for y in range(64):
        for x in range(64):
            pixels[x, y] = (
                rng.randint(0, 255),
                rng.randint(0, 255),
                rng.randint(0, 255),
            )
    img.save(path)


def test_collect_figures_finds_label_caption_and_ref(tmp_path: Path) -> None:
    (tmp_path / "figures").mkdir()
    _write_png(tmp_path / "figures" / "a.png", (10, 10, 10))
    tex = (
        r"\begin{figure}[t]" "\n"
        r"  \includegraphics[width=0.5\linewidth]{figures/a.png}" "\n"
        r"  \caption{The first figure.}" "\n"
        r"  \label{fig:a}" "\n"
        r"\end{figure}" "\n"
        r"Later we discuss \ref{fig:a}." "\n"
    )
    records = collect_figures(tex, tmp_path)
    assert len(records) == 1
    rec = records[0]
    assert rec.label == "fig:a"
    assert rec.referenced is True
    assert rec.resolved_path is not None and rec.resolved_path.name == "a.png"


def test_unreferenced_float_is_flagged(tmp_path: Path) -> None:
    (tmp_path / "figures").mkdir()
    _write_png(tmp_path / "figures" / "a.png", (10, 10, 10))
    tex = (
        r"\begin{figure}[t]" "\n"
        r"  \includegraphics{figures/a.png}" "\n"
        r"  \label{fig:lonely}" "\n"
        r"\end{figure}"
    )
    records = collect_figures(tex, tmp_path)
    issues = detect_unreferenced_floats(records)
    assert len(issues) == 1
    assert issues[0].kind == "float_unreferenced"


def test_dhash_detects_near_duplicates(tmp_path: Path) -> None:
    img1 = tmp_path / "a.png"
    img2 = tmp_path / "b.png"
    _write_png(img1, (100, 100, 100))
    _write_png(img2, (101, 101, 101))  # essentially identical
    h1 = _dhash(img1)
    h2 = _dhash(img2)
    assert h1 is not None and h2 is not None
    assert _hamming(h1, h2) <= 2


def test_find_duplicate_figures(tmp_path: Path) -> None:
    (tmp_path / "figures").mkdir()
    _write_png(tmp_path / "figures" / "x.png", (50, 50, 50))
    _write_png(tmp_path / "figures" / "y.png", (50, 50, 50))
    _write_textured_png(tmp_path / "figures" / "z.png", seed=7)
    tex = (
        r"\begin{figure}\includegraphics{figures/x.png}\label{a}\end{figure}"
        "\n"
        r"\begin{figure}\includegraphics{figures/y.png}\label{b}\end{figure}"
        "\n"
        r"\begin{figure}\includegraphics{figures/z.png}\label{c}\end{figure}"
        "\n"
        r"see \ref{a},\ref{b},\ref{c}"
    )
    records = collect_figures(tex, tmp_path)
    issues = find_duplicate_figures(records)
    assert len(issues) == 1
    assert issues[0].asset.endswith("y.png")
    assert issues[0].extra["duplicate_of"].endswith("x.png")


def test_bbox_overlap_fraction() -> None:
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 5.0, 15.0, 15.0)
    frac = _bbox_overlap_fraction(a, b)
    # intersection = 25, smaller area = 100 → 0.25
    assert abs(frac - 0.25) < 1e-6
    c = (0.0, 0.0, 5.0, 5.0)
    inside = _bbox_overlap_fraction(a, c)
    assert inside == 1.0  # c entirely inside a


def test_apply_visual_patches_drops_duplicates(tmp_path: Path) -> None:
    (tmp_path / "figures").mkdir()
    _write_png(tmp_path / "figures" / "x.png", (50, 50, 50))
    _write_png(tmp_path / "figures" / "y.png", (50, 50, 50))
    tex = (
        r"\begin{figure}\includegraphics{figures/x.png}\label{a}\end{figure}"
        "\n"
        r"\begin{figure}\includegraphics{figures/y.png}\label{b}\end{figure}"
        "\n"
        r"\ref{a},\ref{b}"
    )
    (tmp_path / "main.tex").write_text(tex, encoding="utf-8")
    audit = audit_bundle_visuals(tmp_path)
    result = apply_visual_patches(tmp_path, audit=audit)
    assert result["ok"]
    after = (tmp_path / "main.tex").read_text(encoding="utf-8")
    # one of x.png / y.png should be removed
    assert ("figures/x.png" in after) ^ ("figures/y.png" in after) or (
        "duplicate figure" in after
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
