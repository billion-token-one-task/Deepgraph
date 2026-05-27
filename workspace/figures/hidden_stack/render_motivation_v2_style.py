#!/usr/bin/env python3
"""Render v2-style three-column motivation figure with paper-aligned content."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent / "fig_motivation_gemini.png"

BLUE = "#2563eb"
BLUE_BG = "#eff6ff"
BLUE_BORDER = "#93c5fd"
ORANGE = "#ea580c"
ORANGE_BG = "#fff7ed"
ORANGE_BORDER = "#fdba74"
GREEN = "#16a34a"
GREEN_BG = "#f0fdf4"
GREEN_BORDER = "#86efac"
TEXT = "#1e293b"
MUTED = "#64748b"
RED = "#dc2626"


def rounded_panel(ax, xy, w, h, face, edge, lw=1.6, ls="-", radius=0.02, zorder=1):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        linestyle=ls,
        transform=ax.transAxes,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def inner_card(ax, xy, w, h, face, edge, title, lines, title_color, zorder=3):
    rounded_panel(ax, xy, w, h, face, edge, lw=1.4, radius=0.015, zorder=zorder)
    x, y = xy
    ax.text(
        x + w / 2,
        y + h - 0.035,
        title,
        ha="center",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=title_color,
        transform=ax.transAxes,
        zorder=zorder + 1,
    )
    yy = y + h - 0.075
    for line in lines:
        ax.text(
            x + 0.025,
            yy,
            f"• {line}",
            ha="left",
            va="top",
            fontsize=8.2,
            color=TEXT,
            transform=ax.transAxes,
            zorder=zorder + 1,
        )
        yy -= 0.038


def arrow(ax, p0, p1, color=MUTED, lw=1.6):
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(arr)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "figure.dpi": 150,
        }
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rounded_panel(ax, (0.03, 0.08), 0.24, 0.84, BLUE_BG, BLUE_BORDER)
    ax.text(0.15, 0.885, "Fixed Inputs", ha="center", va="top", fontsize=10.5, fontweight="bold", color=BLUE)

    rounded_panel(ax, (0.055, 0.58), 0.19, 0.24, "white", BLUE_BORDER, lw=1.2)
    ax.text(0.15, 0.785, "Fixed Model Block", ha="center", fontsize=9, fontweight="bold", color=BLUE)
    ax.text(0.15, 0.735, "Qwen2.5-7B-Instruct", ha="center", fontsize=8.8, color=TEXT)
    ax.text(0.15, 0.695, "(frozen weights)", ha="center", fontsize=8, color=MUTED, style="italic")

    rounded_panel(ax, (0.055, 0.18), 0.19, 0.34, "white", BLUE_BORDER, lw=1.2)
    ax.text(0.15, 0.495, "Fixed Prompt Box", ha="center", fontsize=9, fontweight="bold", color=BLUE)
    ax.text(0.15, 0.44, "Q: What is the capital", ha="center", fontsize=8.2, color=TEXT)
    ax.text(0.15, 0.405, "of Australia?", ha="center", fontsize=8.2, color=TEXT)
    ax.text(0.15, 0.355, "A: (same template)", ha="center", fontsize=8, color=MUTED, style="italic")
    ax.text(0.15, 0.22, "Standardized Input", ha="center", fontsize=7.8, color=MUTED, style="italic")

    arrow(ax, (0.15, 0.56), (0.15, 0.54), color=BLUE)

    rounded_panel(ax, (0.30, 0.08), 0.38, 0.84, ORANGE_BG, ORANGE_BORDER, lw=2.0, ls="--")
    ax.text(0.49, 0.885, "Inference Backends", ha="center", va="top", fontsize=10.5, fontweight="bold", color=ORANGE)

    inner_card(
        ax,
        (0.325, 0.62),
        0.33,
        0.21,
        "#eef2ff",
        BLUE,
        "HuggingFace Transformers",
        ["FP16 precision", "Dynamic batching", "Default stop rules"],
        BLUE,
    )
    inner_card(
        ax,
        (0.325, 0.395),
        0.33,
        0.21,
        "#fff7ed",
        ORANGE,
        "vLLM",
        ["BF16 precision", "Continuous batching", "Optimized CUDA kernels"],
        ORANGE,
    )
    inner_card(
        ax,
        (0.325, 0.17),
        0.33,
        0.21,
        "#f0fdf4",
        GREEN,
        "TGI",
        ["FP16 precision", "Tensor parallelism", "Custom decoding"],
        GREEN,
    )

    rounded_panel(ax, (0.71, 0.08), 0.26, 0.84, GREEN_BG, GREEN_BORDER)
    ax.text(0.84, 0.885, "Benchmark Outcome", ha="center", va="top", fontsize=10.5, fontweight="bold", color=GREEN)

    rounded_panel(ax, (0.735, 0.58), 0.22, 0.27, "white", GREEN_BORDER, lw=1.2)
    ax.text(0.845, 0.835, "Exact Match Accuracy", ha="center", fontsize=8.8, fontweight="bold", color=TEXT)
    bars = [("HF", 0.921, BLUE), ("vLLM", 0.924, ORANGE), ("TGI", 0.918, GREEN)]
    x0, y0, bw, max_h = 0.765, 0.62, 0.045, 0.17
    for i, (label, val, color) in enumerate(bars):
        x = x0 + i * 0.055
        h = (val - 0.90) / 0.05 * max_h
        ax.add_patch(
            patches.Rectangle((x, y0), bw, h, facecolor=color, edgecolor="white", linewidth=0.8, transform=ax.transAxes)
        )
        ax.text(x + bw / 2, y0 - 0.025, label, ha="center", fontsize=7.5, color=TEXT, transform=ax.transAxes)
        ax.text(x + bw / 2, y0 + h + 0.012, f"{val*100:.1f}%", ha="center", fontsize=7.5, color=TEXT, transform=ax.transAxes)

    snippets = [
        (BLUE, [
            ("A: The capital is Canberra, a ", TEXT),
            ("major", RED),
            (" city.", TEXT),
        ]),
        (ORANGE, [
            ("A: Canberra is the capital ", TEXT),
            ("city", RED),
            (".", TEXT),
        ]),
        (GREEN, [
            ("A: The capital is Canberra.", TEXT),
        ]),
    ]
    y_snip = 0.47
    box_h = 0.10
    for color, parts in snippets:
        rounded_panel(ax, (0.735, y_snip - box_h), 0.22, box_h, "white", color, lw=1.1)
        x_text = 0.745
        y_text = y_snip - 0.045
        for text, col in parts:
            ax.text(x_text, y_text, text, ha="left", va="center", fontsize=7.4, color=col,
                    fontweight="bold" if col == RED else "normal", transform=ax.transAxes)
            x_text += len(text) * 0.0048
        y_snip -= (box_h + 0.025)

    arrow(ax, (0.27, 0.5), (0.30, 0.5), color=MUTED, lw=2.0)
    arrow(ax, (0.68, 0.5), (0.71, 0.5), color=MUTED, lw=2.0)

    ax.text(0.915, 0.105, "Blue: HF", fontsize=7, color=BLUE, ha="right", transform=ax.transAxes)
    ax.text(0.915, 0.085, "Orange: vLLM", fontsize=7, color=ORANGE, ha="right", transform=ax.transAxes)
    ax.text(0.915, 0.065, "Green: TGI", fontsize=7, color=GREEN, ha="right", transform=ax.transAxes)

    fig.tight_layout(pad=0.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
