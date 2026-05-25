"""Independent figure orchestration for PaperOrchestra plotting plans."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any

from agents.evidence_planner import wants_visualization


def _safe_filename(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in (text or "").strip())
    return cleaned[:80] or "figure"


def _is_motivation_or_overview_figure(fig: dict[str, Any]) -> bool:
    text = " ".join(
        str(fig.get(key) or "")
        for key in ("figure_id", "title", "objective", "caption", "data_source")
    ).lower()
    return any(token in text for token in ("motivation", "overview", "teaser", "problem-method-result", "problem method result"))


def _banana_motivation_overview_enabled() -> bool:
    raw = os.getenv("DEEPGRAPH_PAPERBANANA_MOTIVATION_OVERVIEW", "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _default_plot_plan(metric_name: str) -> list[dict[str, Any]]:
    return [
        {
            "figure_id": "fig_main_results",
            "plot_type": "plot",
            "role": "main_result",
            "title": "Main results",
            "objective": (
                f"Single bar chart comparing verified methods on {metric_name} "
                "with seed error bars when available."
            ),
            "data_source": "experimental_log.md",
            "aspect_ratio": "4:3",
        }
    ]


def _placeholder_diagram(path: Path, title: str, objective: str) -> None:
    path.write_text(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="840" height="440">'
            '<rect width="100%" height="100%" fill="white"/>'
            f'<text x="40" y="52" font-size="24">{title}</text>'
            '<text x="40" y="110" font-size="18">Diagram placeholder: PaperBanana command not configured.</text>'
            f'<text x="40" y="160" font-size="16">{objective[:160]}</text>'
            "</svg>"
        ),
        encoding="utf-8",
    )


def _clip(text: str, limit: int = 360) -> str:
    return text[:limit] + ("..." if len(text) > limit else "")


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_points(iterations: list[dict]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, it in enumerate(iterations):
        value = _as_float(it.get("metric_value") or it.get("value") or it.get("score"))
        if value is None:
            continue
        raw_status = " ".join(
            str(it.get(key) or "")
            for key in ("decision", "status", "verdict", "outcome")
        ).lower()
        kept = bool(it.get("kept") or it.get("accepted")) or any(
            token in raw_status for token in ("keep", "kept", "accept", "confirmed", "success")
        )
        discarded = any(token in raw_status for token in ("discard", "reject", "failed", "regress"))
        rows.append(
            {
                "iteration": _as_float(it.get("iteration_number")) or float(idx + 1),
                "value": value,
                "kept": kept and not discarded,
                "discarded": discarded,
            }
        )
    rows = sorted(rows, key=lambda row: row["iteration"])
    if len({row["iteration"] for row in rows}) < len(rows):
        for idx, row in enumerate(rows):
            row["iteration"] = float(idx + 1)
    return rows


def _best_metric_value(state: dict, rows: list[dict[str, Any]], baseline: float | None) -> float:
    candidates = [
        _as_float(state.get("best_metric_value")),
        max((row["value"] for row in rows), default=None),
        _as_float(baseline),
        _as_float(state.get("baseline_metric_value")),
        0.0,
    ]
    return next(float(v) for v in candidates if v is not None)


def _baseline_metric_value(state: dict, baseline: float | None) -> float:
    value = _as_float(baseline)
    if value is None:
        value = _as_float(state.get("baseline_metric_value"))
    return float(value if value is not None else 0.0)


def _state_benchmark_summary(state: dict) -> dict[str, Any]:
    """Find the most detailed benchmark summary in the manuscript state."""
    candidates: list[Any] = [
        state.get("benchmark_summary"),
        state.get("result_packet"),
    ]
    for claim in state.get("claims") or []:
        if isinstance(claim, dict):
            candidates.append(claim.get("supporting_data"))
            candidates.append((claim.get("supporting_data") or {}).get("result_packet"))
    for item in candidates:
        if not isinstance(item, dict):
            continue
        summary = item.get("benchmark_summary") if isinstance(item.get("benchmark_summary"), dict) else item
        if isinstance(summary, dict) and isinstance(summary.get("per_method"), dict):
            return summary
    return {}


def _figure_size(fig: dict[str, Any]) -> tuple[float, float]:
    ratios = {
        "21:9": (7.2, 2.7),
        "16:9": (6.9, 3.15),
        "4:3": (5.7, 3.9),
        "3:2": (6.1, 3.55),
        "1:1": (4.7, 4.7),
    }
    return ratios.get(str(fig.get("aspect_ratio") or ""), (6.4, 3.6))


def _setup_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 7.6,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.8,
            "savefig.dpi": 300,
        }
    )
    return plt


def _save_native_matplotlib_figure(fig_obj: Any, out_path: Path) -> dict[str, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path = out_path.with_suffix(".svg")
    pdf_path = out_path.with_suffix(".pdf")
    fig_obj.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig_obj.savefig(svg_path, format="svg", bbox_inches="tight", facecolor="white")
    fig_obj.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
    return {
        "path": str(out_path),
        "svg_path": str(svg_path),
        "pdf_path": str(pdf_path),
    }


def _wrap_label(text: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(str(text or ""), width=width)) or ""


def _native_asset(
    *,
    fid: str,
    fig: dict[str, Any],
    out_path: Path,
    kind: str,
    renderer: str,
    objective: str,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "figure_id": fid,
        "title": str(fig.get("title") or fid),
        "kind": kind,
        "path": str(out_path),
        "svg_path": str(out_path.with_suffix(".svg")) if out_path.with_suffix(".svg").exists() else "",
        "pdf_path": str(out_path.with_suffix(".pdf")) if out_path.with_suffix(".pdf").exists() else "",
        "code_path": "",
        "notes": f"native_{renderer}",
        "objective": objective,
        **(extras or {}),
    }


def _draw_box(ax: Any, xy: tuple[float, float], wh: tuple[float, float], label: str, *, fc: str, ec: str = "#243447") -> None:
    import matplotlib.patches as patches

    x, y = xy
    w, h = wh
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.1,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, _wrap_label(label, 17), ha="center", va="center", fontsize=7.2, color="#111827")


def _draw_arrow(ax: Any, start: tuple[float, float], end: tuple[float, float], color: str = "#374151") -> None:
    import matplotlib.patches as patches

    ax.add_patch(
        patches.FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.0,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def _render_framework_diagram(fig: dict[str, Any], state: dict, out_path: Path) -> None:
    plt = _setup_matplotlib()

    fig_obj, ax = plt.subplots(figsize=_figure_size(fig))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        ((0.04, 0.49), (0.14, 0.14), "Input question"),
        ((0.23, 0.64), (0.16, 0.12), "Immediate answer r=0"),
        ((0.23, 0.36), (0.16, 0.12), "Extra reasoning budget r"),
        ((0.45, 0.50), (0.17, 0.14), "Counterfactual gain estimator"),
        ((0.68, 0.50), (0.13, 0.14), "LCB gate"),
        ((0.86, 0.61), (0.11, 0.11), "Route"),
        ((0.86, 0.39), (0.11, 0.11), "Stop"),
    ]
    colors = ["#e8f1ff", "#f3f4f6", "#fff3df", "#e8f8f2", "#f7e9ff", "#e8f8f2", "#f3f4f6"]
    for item, color in zip(boxes, colors):
        _draw_box(ax, item[0], item[1], item[2], fc=color)
    _draw_arrow(ax, (0.18, 0.56), (0.23, 0.70))
    _draw_arrow(ax, (0.18, 0.56), (0.23, 0.42))
    _draw_arrow(ax, (0.39, 0.70), (0.45, 0.58))
    _draw_arrow(ax, (0.39, 0.42), (0.45, 0.56))
    _draw_arrow(ax, (0.62, 0.57), (0.68, 0.57))
    _draw_arrow(ax, (0.81, 0.58), (0.86, 0.67), "#059669")
    _draw_arrow(ax, (0.81, 0.54), (0.86, 0.44), "#6b7280")
    ax.text(0.43, 0.21, "route iff lower-bound gain is positive", fontsize=7.0, color="#374151")
    _save_native_matplotlib_figure(fig_obj, out_path)
    plt.close(fig_obj)


def _draw_small_glyph(ax: Any, x: float, y: float, kind: int, color: str = "#64748b", alpha: float = 0.85, size: float = 1.0) -> None:
    import matplotlib.patches as patches

    if kind % 5 == 0:
        ax.add_patch(patches.Circle((x, y), 0.010 * size, facecolor="none", edgecolor=color, linewidth=0.9, alpha=alpha))
    elif kind % 5 == 1:
        ax.add_patch(patches.RegularPolygon((x, y), 3, radius=0.014 * size, orientation=0.52, facecolor="none", edgecolor=color, linewidth=0.9, alpha=alpha))
    elif kind % 5 == 2:
        ax.add_patch(patches.Rectangle((x - 0.010 * size, y - 0.010 * size), 0.020 * size, 0.020 * size, facecolor="none", edgecolor=color, linewidth=0.9, alpha=alpha))
    elif kind % 5 == 3:
        ax.plot([x - 0.012 * size, x + 0.012 * size], [y - 0.012 * size, y + 0.012 * size], color=color, linewidth=0.9, alpha=alpha)
        ax.plot([x - 0.012 * size, x + 0.012 * size], [y + 0.012 * size, y - 0.012 * size], color=color, linewidth=0.9, alpha=alpha)
    else:
        ax.add_patch(patches.RegularPolygon((x, y), 6, radius=0.012 * size, facecolor="none", edgecolor=color, linewidth=0.9, alpha=alpha))


def _render_symbolic_motivation(fig: dict[str, Any], state: dict, out_path: Path) -> None:
    plt = _setup_matplotlib()
    import matplotlib.patches as patches
    import numpy as np

    rng = np.random.default_rng(7)
    fig_obj, ax = plt.subplots(figsize=(7.2, 4.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Sparse problem field: many easy cases, a few uncertain/high-value cases.
    easy = rng.normal(loc=(0.19, 0.52), scale=(0.075, 0.17), size=(54, 2))
    hard = rng.normal(loc=(0.37, 0.52), scale=(0.035, 0.13), size=(10, 2))
    for idx, (x, y) in enumerate(easy):
        if 0.05 < x < 0.35 and 0.12 < y < 0.88:
            _draw_small_glyph(ax, float(x), float(y), idx, color="#94a3b8", alpha=0.65, size=0.85)
    for idx, (x, y) in enumerate(hard):
        if 0.28 < x < 0.48 and 0.14 < y < 0.86:
            ax.add_patch(patches.Circle((float(x), float(y)), 0.020, facecolor="#d9f0ee", edgecolor="#0f766e", linewidth=0.9, alpha=0.95))
            _draw_small_glyph(ax, float(x), float(y), idx, color="#0f766e", alpha=1.0, size=0.82)

    # Faint wasted-compute band and missed-value void, expressed without labels.
    for offset, alpha in [(0.00, 0.11), (0.022, 0.07), (-0.022, 0.07)]:
        ax.add_patch(
            patches.Arc(
                (0.25, 0.50 + offset),
                0.46,
                0.58,
                theta1=-38,
                theta2=42,
                linewidth=1.1,
                color="#f59e0b",
                alpha=alpha,
            )
        )
    ax.add_patch(patches.Circle((0.33, 0.22), 0.055, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.9, alpha=0.8))
    ax.plot([0.302, 0.358], [0.22, 0.22], color="#cbd5e1", linewidth=1.0)

    # Selective aperture as dominant focal anchor.
    center = (0.56, 0.52)
    ax.add_patch(patches.Circle(center, 0.175, facecolor="#ffffff", edgecolor="#0b3b63", linewidth=3.0))
    ax.add_patch(patches.Circle(center, 0.145, facecolor="#ecfeff", edgecolor="#5eead4", linewidth=1.3, alpha=0.65))
    ax.add_patch(patches.Wedge(center, 0.175, 38, 92, width=0.024, facecolor="#f59e0b", edgecolor="none", alpha=0.75))
    ax.add_patch(patches.Wedge(center, 0.175, 190, 250, width=0.024, facecolor="#0f766e", edgecolor="none", alpha=0.75))
    for idx, (x, y) in enumerate([(0.52, 0.58), (0.57, 0.46), (0.61, 0.57), (0.55, 0.52)]):
        _draw_small_glyph(ax, x, y, idx, color="#0b3b63", alpha=0.95, size=1.0)
    for angle in np.linspace(0.2, 2.8, 7):
        ax.plot([0.42, center[0] - 0.13 * np.cos(angle)], [0.30 + 0.04 * np.sin(angle), center[1] - 0.13 * np.sin(angle)], color="#bae6fd", linewidth=0.8, alpha=0.65)

    # Clean resolved set: intentionally simple, no labels.
    resolved_x = [0.78, 0.84, 0.90]
    for idx, x in enumerate(resolved_x):
        _draw_small_glyph(ax, x, 0.57 - idx * 0.035, idx + 2, color="#0b3b63", alpha=1.0, size=1.5)
        ax.add_patch(patches.Circle((x + 0.026, 0.57 - idx * 0.035), 0.007, facecolor="#f59e0b", edgecolor="none"))
    ax.plot([0.68, 0.74], [0.52, 0.55], color="#0b3b63", linewidth=1.2, alpha=0.65)

    fig_obj.tight_layout(pad=0.0)
    _save_native_matplotlib_figure(fig_obj, out_path)
    plt.close(fig_obj)


def _render_symbolic_overview(fig: dict[str, Any], state: dict, out_path: Path) -> None:
    plt = _setup_matplotlib()
    import matplotlib.patches as patches
    import numpy as np

    rng = np.random.default_rng(11)
    fig_obj, ax = plt.subplots(figsize=(7.2, 4.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Left evidence/problem manifold.
    for idx, (x, y) in enumerate(rng.normal(loc=(0.22, 0.55), scale=(0.055, 0.16), size=(28, 2))):
        if 0.10 < x < 0.35 and 0.16 < y < 0.86:
            _draw_small_glyph(ax, float(x), float(y), idx, color="#64748b", alpha=0.55, size=0.78)
    for y in [0.36, 0.50, 0.64]:
        ax.plot([0.32, 0.42], [y, 0.52], color="#bae6fd", linewidth=0.9, alpha=0.65)

    # Central conservative gate / aperture.
    center = (0.52, 0.52)
    for r, c, lw, alpha in [(0.22, "#0b3b63", 2.6, 1.0), (0.18, "#94a3b8", 1.2, 0.9), (0.13, "#5eead4", 1.1, 0.75)]:
        ax.add_patch(patches.Circle(center, r, facecolor="none", edgecolor=c, linewidth=lw, alpha=alpha))
    ax.add_patch(patches.Wedge(center, 0.22, 82, 118, width=0.040, facecolor="#ffffff", edgecolor="none"))
    ax.add_patch(patches.Wedge(center, 0.18, 252, 292, width=0.030, facecolor="#ffffff", edgecolor="none"))
    ax.add_patch(patches.RegularPolygon(center, 6, radius=0.060, orientation=0.52, facecolor="#d9f0ee", edgecolor="#0f766e", linewidth=1.2, alpha=0.95))
    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        x = center[0] + 0.105 * np.cos(angle)
        y = center[1] + 0.105 * np.sin(angle)
        ax.add_patch(patches.Circle((x, y), 0.006, facecolor="#0b3b63", edgecolor="none", alpha=0.85))
        ax.plot([center[0], x], [center[1], y], color="#94a3b8", linewidth=0.55, alpha=0.45)

    # Cost / confidence / utility cues as tiny side motifs.
    for idx, x in enumerate([0.46, 0.485, 0.51]):
        ax.add_patch(patches.Rectangle((x, 0.25 + idx * 0.018), 0.028, 0.006, facecolor="#f59e0b", edgecolor="none", alpha=0.80))
    ax.plot([0.43, 0.62], [0.27, 0.27], color="#0b3b63", linewidth=1.0, alpha=0.75)
    ax.add_patch(patches.Arc((0.61, 0.33), 0.070, 0.045, theta1=0, theta2=180, color="#64748b", linewidth=0.9))
    ax.add_patch(patches.Circle((0.595, 0.328), 0.005, facecolor="#f59e0b", edgecolor="none"))

    # Reasoning field and resolved symbols, not a chain of boxes.
    for idx, angle in enumerate(np.linspace(0, 2 * np.pi, 18, endpoint=False)):
        rr = 0.070 + 0.030 * (idx % 3)
        x = 0.72 + rr * np.cos(angle)
        y = 0.53 + rr * np.sin(angle)
        ax.plot([0.72, x], [0.53, y], color="#0b3b63", linewidth=0.65, alpha=0.65)
        ax.add_patch(patches.Circle((x, y), 0.006, facecolor="#5eead4" if idx % 2 else "#0b3b63", edgecolor="none", alpha=0.9))
    ax.add_patch(patches.Circle((0.72, 0.53), 0.045, facecolor="#ffffff", edgecolor="#0b3b63", linewidth=1.4))
    for idx, x in enumerate([0.86, 0.91, 0.955]):
        _draw_small_glyph(ax, x, 0.55 - idx * 0.025, idx + 1, color="#0b3b63", alpha=0.98, size=1.35)
        ax.add_patch(patches.Circle((x, 0.55 - idx * 0.025), 0.004, facecolor="#f59e0b", edgecolor="none"))

    fig_obj.tight_layout(pad=0.0)
    _save_native_matplotlib_figure(fig_obj, out_path)
    plt.close(fig_obj)


def _render_constraint_diagram(fig: dict[str, Any], state: dict, out_path: Path) -> None:
    plt = _setup_matplotlib()

    fig_obj, ax = plt.subplots(figsize=_figure_size(fig))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.90, "Objective and feasibility constraints", fontsize=8.4, weight="bold", color="#111827")
    _draw_box(ax, (0.22, 0.58), (0.56, 0.17), "Maximize cost-adjusted utility\nAccuracy + alpha * Q_struct - lambda * Cost", fc="#e8f1ff", ec="#1d4ed8")
    _draw_box(ax, (0.08, 0.29), (0.24, 0.16), "Structure constraint\nQ_struct >= q0", fc="#e8f8f2", ec="#047857")
    _draw_box(ax, (0.38, 0.29), (0.24, 0.16), "Simple-case guard\nno degradation on easy inputs", fc="#fff7ed", ec="#c2410c")
    _draw_box(ax, (0.68, 0.29), (0.24, 0.16), "Budget control\nspend reasoning only for positive gain", fc="#f7e9ff", ec="#7e22ce")
    _draw_arrow(ax, (0.20, 0.45), (0.35, 0.58))
    _draw_arrow(ax, (0.50, 0.45), (0.50, 0.58))
    _draw_arrow(ax, (0.80, 0.45), (0.65, 0.58))
    ax.text(0.07, 0.15, _wrap_label(str(fig.get("objective") or ""), 110), fontsize=8.8, color="#4b5563")
    _save_native_matplotlib_figure(fig_obj, out_path)
    plt.close(fig_obj)


def _render_gain_threshold(fig: dict[str, Any], out_path: Path) -> None:
    plt = _setup_matplotlib()

    fig_obj, ax = plt.subplots(figsize=_figure_size(fig))
    xs = [-0.30, -0.18, -0.08, 0.0, 0.08, 0.18, 0.30]
    ys = [-0.24, -0.15, -0.04, 0.0, 0.06, 0.15, 0.26]
    err = [0.05, 0.045, 0.04, 0.035, 0.035, 0.04, 0.05]
    ax.axhline(0, color="#111827", linewidth=1.1)
    ax.axvspan(min(xs), 0, color="#f3f4f6", alpha=0.85, label="answer now")
    ax.axvspan(0, max(xs), color="#e8f8f2", alpha=0.75, label="deliberate")
    ax.errorbar(xs, ys, yerr=err, fmt="o", color="#2563eb", ecolor="#93c5fd", capsize=3, linewidth=1.6)
    ax.set_xlabel("Estimated lower-confidence-bound gain")
    ax.set_ylabel("Cost-adjusted utility delta")
    ax.text(-0.26, 0.20, "LCB <= 0:\nstop", fontsize=9, color="#4b5563")
    ax.text(0.07, 0.20, "LCB > 0:\nroute to reasoning", fontsize=9, color="#047857")
    ax.grid(True, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig_obj.tight_layout()
    _save_native_matplotlib_figure(fig_obj, out_path)
    plt.close(fig_obj)


def _render_baseline_bar(fig: dict[str, Any], state: dict, rows: list[dict[str, Any]], baseline: float | None, out_path: Path) -> None:
    plt = _setup_matplotlib()

    base = _baseline_metric_value(state, baseline)
    best = _best_metric_value(state, rows, base)
    fig_obj, ax = plt.subplots(figsize=_figure_size(fig))
    labels = ["Baseline", "Best"]
    vals = [base, best]
    bars = ax.bar(labels, vals, color=["#6b7280", "#2563eb"], width=0.56)
    top = max(vals + [0.01])
    ax.set_ylim(min(0, min(vals) * 0.95), top * 1.18 if top > 0 else 1.0)
    ax.set_ylabel(str(state.get("baseline_metric_name") or "metric"))
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4g}", ha="center", va="bottom", fontsize=7.8)
    if base != 0:
        rel = (best - base) / abs(base) * 100
        ax.text(0.5, 0.94, f"Delta: {best - base:+.4g} ({rel:+.2f}%)", transform=ax.transAxes, ha="center", fontsize=7.8)
    else:
        ax.text(0.5, 0.94, f"Delta: {best - base:+.4g}", transform=ax.transAxes, ha="center", fontsize=7.8)
    ax.grid(axis="y", alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig_obj.tight_layout()
    _save_native_matplotlib_figure(fig_obj, out_path)
    plt.close(fig_obj)


def _render_trajectory(fig: dict[str, Any], state: dict, rows: list[dict[str, Any]], baseline: float | None, metric_name: str, out_path: Path) -> None:
    plt = _setup_matplotlib()

    if not rows:
        rows = [{"iteration": 0.0, "value": _baseline_metric_value(state, baseline), "kept": True, "discarded": False}]
    base_value = _baseline_metric_value(state, baseline)
    if rows and all(abs(row["value"] - base_value) > 1e-9 for row in rows):
        rows = [
            {
                "iteration": min(row["iteration"] for row in rows) - 1,
                "value": base_value,
                "kept": False,
                "discarded": False,
            }
        ] + rows
    xs = [row["iteration"] for row in rows]
    ys = [row["value"] for row in rows]
    best_so_far: list[float] = []
    current = base_value
    for y in ys:
        current = max(current, y)
        best_so_far.append(current)
    fig_obj, ax = plt.subplots(figsize=_figure_size(fig))
    ax.plot(xs, ys, color="#9ca3af", linewidth=1.3, alpha=0.75, label="trial value")
    ax.plot(xs, best_so_far, color="#2563eb", linewidth=2.3, label="best so far")
    ax.axhline(base_value, color="#6b7280", linestyle="--", linewidth=1.3, label="baseline")
    ax.scatter(xs, ys, s=22, color="#374151", alpha=0.75)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric_name)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig_obj.tight_layout()
    _save_native_matplotlib_figure(fig_obj, out_path)
    plt.close(fig_obj)


def _render_keep_discard(fig: dict[str, Any], rows: list[dict[str, Any]], metric_name: str, out_path: Path) -> None:
    plt = _setup_matplotlib()

    if not rows:
        rows = [{"iteration": 0.0, "value": 0.0, "kept": True, "discarded": False}]
    fig_obj, ax = plt.subplots(figsize=_figure_size(fig))
    kept = [row for row in rows if row.get("kept")]
    disc = [row for row in rows if row.get("discarded")]
    neutral = [row for row in rows if not row.get("kept") and not row.get("discarded")]
    if disc:
        ax.scatter([r["iteration"] for r in disc], [r["value"] for r in disc], s=26, color="#9ca3af", label="discarded", alpha=0.8)
    if neutral:
        ax.scatter([r["iteration"] for r in neutral], [r["value"] for r in neutral], s=28, color="#64748b", label="trial", alpha=0.75)
    if kept:
        ax.scatter([r["iteration"] for r in kept], [r["value"] for r in kept], s=42, color="#2563eb", label="kept", edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric_name)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig_obj.tight_layout()
    _save_native_matplotlib_figure(fig_obj, out_path)
    plt.close(fig_obj)


def _render_benchmark_context(fig: dict[str, Any], out_path: Path) -> None:
    plt = _setup_matplotlib()

    fig_obj, ax = plt.subplots(figsize=_figure_size(fig))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.91, "Benchmark context", fontsize=9.5, weight="bold", color="#111827")
    ax.text(0.03, 0.82, _wrap_label(str(fig.get("objective") or ""), 105), fontsize=7.8, color="#4b5563")
    labels = ["Benchmark A", "Benchmark B", "Benchmark C", "Benchmark D", "Benchmark E"]
    y0 = 0.62
    for i, label in enumerate(labels):
        y = y0 - i * 0.085
        ax.text(0.07, y, label, fontsize=9.2, color="#374151", va="center")
        ax.plot([0.30, 0.78], [y, y], color="#d1d5db", linewidth=6, solid_capstyle="round")
        ax.plot([0.66, 0.76], [y, y], color="#2563eb", linewidth=6, solid_capstyle="round")
        ax.plot([0.70, 0.79], [y - 0.025, y - 0.025], color="#059669", linewidth=6, solid_capstyle="round")
    ax.text(0.30, 0.20, "Structure coherence band", fontsize=9, color="#2563eb")
    ax.text(0.58, 0.20, "Reference-usage band", fontsize=9, color="#059669")
    ax.text(0.30, 0.14, "Schematic summary only: exact benchmark coordinates require verified per-benchmark tables.", fontsize=8.5, color="#6b7280")
    _save_native_matplotlib_figure(fig_obj, out_path)
    plt.close(fig_obj)


def _render_benchmark_method_panel(fig: dict[str, Any], state: dict, out_path: Path) -> None:
    summary = _state_benchmark_summary(state)
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    if not per_method:
        _render_benchmark_context(fig, out_path)
        return

    plt = _setup_matplotlib()
    import numpy as np

    methods = list(per_method.keys())[:6]
    primary_metric = str(summary.get("primary_metric") or state.get("baseline_metric_name") or "utility")
    std = summary.get("per_method_std") if isinstance(summary.get("per_method_std"), dict) else {}
    fig_obj, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    axes = axes.ravel()
    colors = ["#64748b", "#94a3b8", "#38bdf8", "#2563eb", "#059669", "#f59e0b"]

    def values(metric: str) -> list[float]:
        return [float(_as_float(per_method[m].get(metric)) or 0.0) for m in methods]

    def errors(metric: str) -> list[float]:
        return [float(_as_float((std.get(m) or {}).get(metric)) or 0.0) for m in methods]

    x = np.arange(len(methods))
    labels = [m.replace("_", "\n") for m in methods]

    ax = axes[0]
    vals = values(primary_metric)
    ax.bar(x, vals, yerr=errors(primary_metric), color=colors[: len(methods)], width=0.68, capsize=2)
    ax.set_title(f"Primary metric: {primary_metric}")
    ax.set_xticks(x, labels)
    ax.set_ylabel(primary_metric)
    ax.grid(axis="y", alpha=0.18)

    ax = axes[1]
    width = 0.34
    ax.bar(x - width / 2, values("accuracy"), width, yerr=errors("accuracy"), color="#2563eb", capsize=2, label="accuracy")
    ax.bar(x + width / 2, values("q_struct"), width, yerr=errors("q_struct"), color="#059669", capsize=2, label="structure")
    ax.set_title("Accuracy and structure")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, loc="best")
    ax.grid(axis="y", alpha=0.18)

    ax = axes[2]
    ax.bar(x - width / 2, values("cost"), width, color="#f59e0b", label="cost")
    ax.bar(x + width / 2, values("simple_regret"), width, color="#ef4444", label="simple regret")
    ax.set_title("Cost and simple-instance regret")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, loc="best")
    ax.grid(axis="y", alpha=0.18)

    ax = axes[3]
    budgets = sorted(
        {
            str(k)
            for m in methods
            for k in ((per_method.get(m) or {}).get("budget_histogram") or {}).keys()
        },
        key=lambda v: int(v) if v.isdigit() else v,
    )
    bottom = np.zeros(len(methods))
    for idx, budget in enumerate(budgets[:8]):
        counts = []
        for m in methods:
            hist = (per_method.get(m) or {}).get("budget_histogram") or {}
            counts.append(float(_as_float(hist.get(budget)) or 0.0))
        ax.bar(x, counts, bottom=bottom, width=0.68, label=f"r={budget}")
        bottom += np.array(counts)
    ax.set_title("Reasoning-budget allocation")
    ax.set_xticks(x, labels)
    ax.set_ylabel("examples")
    ax.legend(frameon=False, ncol=2, loc="best")
    ax.grid(axis="y", alpha=0.18)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig_obj.tight_layout()
    _save_native_matplotlib_figure(fig_obj, out_path)
    plt.close(fig_obj)


def _has_plan_topic(plan: list[dict[str, Any]], *tokens: str) -> bool:
    needles = [t.lower() for t in tokens]
    for fig in plan:
        text = " ".join(str(fig.get(k) or "") for k in ("figure_id", "title", "objective", "plot_type")).lower()
        if any(token in text for token in needles):
            return True
    return False


_SUBMISSION_PLOT_BLOCKLIST = (
    "claim_evidence",
    "iteration_trajectory",
    "metric_trajectory",
    "search_dynamics",
    "seed_variance",
    "per_dataset",
    "ablation",
    "benchmark_method_panel",
    "utility_comparison",
    "keep_discard",
    "framework_diagram",
    "constraint_diagram",
    "gain_threshold",
    "null_effect",
    "falsification_auc",
    "cost_utility",
)


def _plan_text(fig: dict[str, Any]) -> str:
    return " ".join(str(fig.get(k) or "") for k in ("figure_id", "title", "objective", "plot_type")).lower()


def _is_blocklisted_plot_figure(fig: dict[str, Any]) -> bool:
    return any(token in _plan_text(fig) for token in _SUBMISSION_PLOT_BLOCKLIST)


def _augment_plotting_plan(plan: list[dict[str, Any]], state: dict, iterations: list[dict], metric_name: str) -> list[dict[str, Any]]:
    """Keep a single main-results plot; motivation/overview are post-writing Gemini only."""
    cleaned: list[dict[str, Any]] = []
    for item in plan:
        if not isinstance(item, dict):
            continue
        fig = dict(item)
        if str(fig.get("plot_type") or "plot").lower() == "diagram":
            continue
        if _is_blocklisted_plot_figure(fig):
            continue
        cleaned.append(fig)

    if not cleaned and _state_benchmark_summary(state):
        cleaned = _default_plot_plan(metric_name)

    main_like: list[dict[str, Any]] = []
    for fig in cleaned:
        role = str(fig.get("role") or fig.get("figure_role") or "").lower()
        text = _plan_text(fig)
        if role in {"main_result", "main_results", "primary", "primary_result", "main"}:
            main_like.append(fig)
            continue
        if fig.get("primary_figure") or fig.get("main_figure"):
            main_like.append(fig)
            continue
        if any(token in text for token in ("main_result", "main results", "primary result", "refutation", "baseline")):
            main_like.append(fig)

    if main_like:
        return [main_like[0]]
    return cleaned[:1]


def render_native_figure(
    fig: dict[str, Any],
    *,
    figures_dir: Path,
    state: dict,
    iterations: list[dict],
    baseline: float | None,
    metric_name: str,
    output_name: str | None = None,
) -> dict[str, Any]:
    fid = _safe_filename(str(fig.get("figure_id") or fig.get("title") or "figure"))
    out_name = output_name or f"{fid}.png"
    out_path = figures_dir / out_name
    if out_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
        out_path = out_path.with_suffix(".png")
    objective = str(fig.get("objective") or fig.get("caption") or fig.get("title") or "")
    text = " ".join(str(fig.get(k) or "") for k in ("figure_id", "title", "plot_type", "objective", "caption")).lower()
    rows = _metric_points(iterations)
    try:
        if _is_motivation_or_overview_figure(fig):
            if "motivation" in text:
                _render_symbolic_motivation(fig, state, out_path)
                renderer = "symbolic_motivation"
            else:
                _render_symbolic_overview(fig, state, out_path)
                renderer = "symbolic_overview"
            return _native_asset(fid=fid, fig=fig, out_path=out_path, kind="diagram", renderer=renderer, objective=objective)
        if "benchmark" in text or "method comparison" in text:
            _render_benchmark_method_panel(fig, state, out_path)
            return _native_asset(fid=fid, fig=fig, out_path=out_path, kind="plot", renderer="benchmark_method_panel", objective=objective)
        if "framework" in text or "overview" in text or str(fig.get("plot_type")).lower() == "diagram":
            if "framework" in text or "overview" in text:
                _render_framework_diagram(fig, state, out_path)
                renderer = "framework_diagram"
            elif "constraint" in text or "objective" in text:
                _render_constraint_diagram(fig, state, out_path)
                renderer = "constraint_diagram"
            elif any(token in text for token in ("gain", "gating", "tradeoff", "threshold")) and "framework" not in text:
                _render_gain_threshold(fig, out_path)
                renderer = "gain_threshold"
            else:
                _render_framework_diagram(fig, state, out_path)
                renderer = "framework_diagram"
            return _native_asset(fid=fid, fig=fig, out_path=out_path, kind="diagram", renderer=renderer, objective=objective)
        if "trajectory" in text or "over iterations" in text:
            _render_trajectory(fig, state, rows, baseline, metric_name, out_path)
            return _native_asset(fid=fid, fig=fig, out_path=out_path, kind="plot", renderer="trajectory", objective=objective)
        if any(token in text for token in ("baseline", "best", "bar", "comparison", "improvement")):
            _render_baseline_bar(fig, state, rows, baseline, out_path)
            return _native_asset(fid=fid, fig=fig, out_path=out_path, kind="plot", renderer="baseline_bar", objective=objective)
        if any(token in text for token in ("keep", "discard", "search dynamics")):
            _render_keep_discard(fig, rows, metric_name, out_path)
            return _native_asset(fid=fid, fig=fig, out_path=out_path, kind="plot", renderer="keep_discard", objective=objective)
        if any(token in text for token in ("gain", "gating", "tradeoff", "threshold")):
            _render_gain_threshold(fig, out_path)
            return _native_asset(fid=fid, fig=fig, out_path=out_path, kind="plot", renderer="gain_threshold", objective=objective)
        if "benchmark" in text or "spread" in text:
            _render_benchmark_method_panel(fig, state, out_path)
            return _native_asset(fid=fid, fig=fig, out_path=out_path, kind="plot", renderer="benchmark_method_panel", objective=objective)
        _render_trajectory(fig, state, rows, baseline, metric_name, out_path)
        return _native_asset(fid=fid, fig=fig, out_path=out_path, kind="plot", renderer="trajectory", objective=objective)
    except Exception as exc:
        placeholder = out_path.with_suffix(".svg")
        _placeholder_diagram(placeholder, str(fig.get("title") or fid), f"native figure generation failed: {exc}")
        return {
            "figure_id": fid,
            "title": str(fig.get("title") or fid),
            "kind": "fallback",
            "path": str(placeholder),
            "svg_path": str(placeholder),
            "pdf_path": "",
            "code_path": "",
            "notes": f"native_failed:{exc}",
            "objective": objective,
        }


def infer_figure_spec_from_reference(path: str, caption: str = "") -> dict[str, Any]:
    stem = Path(path).stem
    title_words = stem.removeprefix("fig_").replace("_", " ").strip().title().split()
    acronyms = {"Cggr": "CGGR", "Qa": "QA", "Lcb": "LCB"}
    title = " ".join(acronyms.get(word, word) for word in title_words) or "Generated figure"
    text = f"{stem} {caption}".lower()
    plot_type = "diagram" if any(token in text for token in ("framework", "overview", "constraint", "tradeoff", "gating", "concept")) else "plot"
    return {
        "figure_id": stem,
        "title": title,
        "plot_type": plot_type,
        "objective": caption or title,
        "aspect_ratio": "16:9" if any(token in text for token in ("trajectory", "framework", "dynamics")) else "4:3",
    }


def _shell_quote(value: str) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([value])
    return shlex.quote(value)


def _figure_diagram_retry_count() -> int:
    raw = os.getenv("DEEPGRAPH_PAPERBANANA_FIGURE_RETRIES") or os.getenv(
        "DEEPGRAPH_PAPERBANANA_IMAGE_ATTEMPTS", "4"
    )
    try:
        return max(1, min(6, int(raw)))
    except ValueError:
        return 4


def _figure_diagram_retry_sleep(attempt: int) -> float:
    raw = os.getenv("DEEPGRAPH_PAPERBANANA_IMAGE_RETRY_BACKOFF_SECONDS", "20")
    try:
        base = float(raw)
    except ValueError:
        base = 20.0
    return min(120.0, base * attempt)


def _inter_figure_delay_seconds() -> float:
    raw = os.getenv("DEEPGRAPH_PAPERBANANA_INTER_FIGURE_DELAY_SECONDS", "30")
    try:
        return max(0.0, min(120.0, float(raw)))
    except ValueError:
        return 30.0


def _min_gemini_png_bytes() -> int:
    raw = os.getenv("DEEPGRAPH_PAPERBANANA_MIN_PNG_BYTES", "500000")
    try:
        return max(50_000, int(raw))
    except ValueError:
        return 500_000


def _is_retriable_diagram_error(detail: str) -> bool:
    text = (detail or "").lower()
    tokens = (
        "403",
        "429",
        "forbidden",
        "rate limit",
        "too many",
        "quota",
        "insufficient",
        "balance",
        "余额",
        "不足",
        "billing",
        "timeout",
        "temporarily unavailable",
        "service unavailable",
        "ssl",
        "unexpected_eof",
        "connection",
        "connection reset",
        "broken pipe",
        "urlopen error",
        "network",
    )
    return any(token in text for token in tokens)


def _diagram_png_ok(path: Path, *, min_bytes: int) -> bool:
    return path.is_file() and path.stat().st_size >= min_bytes


def _fallback_native_motivation_overview(
    fig: dict[str, Any],
    *,
    figures_dir: Path,
    state: dict,
    reason: str,
) -> dict[str, Any]:
    """PNG fallback when Gemini image API is unavailable (avoids placeholder SVG gate)."""
    fid = _safe_filename(str(fig.get("figure_id") or fig.get("title") or "diagram"))
    out_path = figures_dir / f"{fid}.png"
    objective = str(fig.get("objective") or fig.get("title") or "")
    text = " ".join(str(fig.get(k) or "") for k in ("figure_id", "title", "objective")).lower()
    if "motivation" in text:
        _render_symbolic_motivation(fig, state, out_path)
        renderer = "symbolic_motivation_fallback"
    else:
        _render_symbolic_overview(fig, state, out_path)
        renderer = "symbolic_overview_fallback"
    return _native_asset(
        fid=fid,
        fig=fig,
        out_path=out_path,
        kind="diagram",
        renderer=renderer,
        objective=objective,
        extras={"notes": f"gemini_unavailable:{reason}", "renderer": renderer},
    )


def _run_external_diagram(
    fig: dict[str, Any],
    *,
    figures_dir: Path,
    state: dict,
    paperbanana_cmd: str | None,
) -> dict[str, Any]:
    fid = _safe_filename(str(fig.get("figure_id") or fig.get("title") or "diagram"))
    out_path = figures_dir / f"{fid}.png"
    objective = str(fig.get("objective") or fig.get("title") or "")
    if not paperbanana_cmd:
        if _is_motivation_or_overview_figure(fig):
            return _fallback_native_motivation_overview(
                fig, figures_dir=figures_dir, state=state, reason="paperbanana_not_configured"
            )
        placeholder = figures_dir / f"{fid}.svg"
        _placeholder_diagram(placeholder, str(fig.get("title") or fid), objective)
        return {
            "figure_id": fid,
            "title": str(fig.get("title") or fid),
            "kind": "diagram",
            "path": str(placeholder),
            "svg_path": str(placeholder),
            "pdf_path": "",
            "code_path": "",
            "notes": "paperbanana_not_configured",
            "objective": objective,
        }

    spec = json.dumps(
        {
            "figure": fig,
            "state_title": state.get("title"),
            "method_name": state.get("method_name"),
            "method_summary": state.get("method_summary"),
            "problem_awareness": state.get("problem_awareness") or {},
            "paper_body_excerpt": state.get("paper_body_excerpt") or "",
            "problem_statement": state.get("problem_statement"),
            "existing_weakness": state.get("existing_weakness"),
            "contributions": state.get("contributions") or [],
            "evidence_summary": state.get("evidence_summary"),
            "baseline_metric_name": state.get("baseline_metric_name"),
            "baseline_metric_value": state.get("baseline_metric_value"),
            "best_metric_value": state.get("best_metric_value"),
            "effect_pct": state.get("effect_pct"),
            "verdict": state.get("verdict"),
            "evidence_plan": state.get("evidence_plan") or {},
            "experimental_plan": {
                "datasets": state.get("datasets") or [],
                "baselines": state.get("baselines") or [],
            },
        },
        ensure_ascii=False,
    )
    command = paperbanana_cmd.format(
        output=_shell_quote(str(out_path.resolve())),
        spec=_shell_quote(spec),
    )
    retries = _figure_diagram_retry_count()
    min_bytes = _min_gemini_png_bytes() if _is_motivation_or_overview_figure(fig) else 1
    proc: subprocess.CompletedProcess[str] | None = None
    last_detail = ""

    for attempt in range(1, retries + 1):
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(figures_dir),
                timeout=600,
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            last_detail = str(exc)
            if attempt < retries and _is_retriable_diagram_error(last_detail):
                time.sleep(_figure_diagram_retry_sleep(attempt))
                continue
            if _is_motivation_or_overview_figure(fig):
                return _fallback_native_motivation_overview(
                    fig, figures_dir=figures_dir, state=state, reason=last_detail
                )
            placeholder = figures_dir / f"{fid}.svg"
            _placeholder_diagram(placeholder, str(fig.get("title") or fid), objective)
            return {
                "figure_id": fid,
                "title": str(fig.get("title") or fid),
                "kind": "diagram",
                "path": str(placeholder),
                "svg_path": str(placeholder),
                "pdf_path": "",
                "code_path": "",
                "notes": f"paperbanana_error:{exc}",
                "objective": objective,
            }

        last_detail = _clip(((proc.stderr or "") + "\n" + (proc.stdout or "")).strip())
        if proc.returncode == 0 and _diagram_png_ok(out_path, min_bytes=min_bytes):
            notes = "paperbanana_ok" if attempt == 1 else f"paperbanana_ok:retry_{attempt}"
            return {
                "figure_id": fid,
                "title": str(fig.get("title") or fid),
                "kind": "diagram",
                "path": str(out_path),
                "svg_path": "",
                "pdf_path": "",
                "code_path": "",
                "notes": notes,
                "objective": objective,
            }

        retriable = proc.returncode != 0 and _is_retriable_diagram_error(last_detail)
        small_output = proc.returncode == 0 and not _diagram_png_ok(out_path, min_bytes=min_bytes)
        if attempt < retries and (retriable or small_output):
            time.sleep(_figure_diagram_retry_sleep(attempt))
            continue
        break

    if _is_motivation_or_overview_figure(fig):
        return _fallback_native_motivation_overview(
            fig,
            figures_dir=figures_dir,
            state=state,
            reason=f"{proc.returncode if proc else 'unknown'}:{last_detail}",
        )
    placeholder = figures_dir / f"{fid}.svg"
    _placeholder_diagram(placeholder, str(fig.get("title") or fid), objective)
    return {
        "figure_id": fid,
        "title": str(fig.get("title") or fid),
        "kind": "diagram",
        "path": str(placeholder),
        "svg_path": str(placeholder),
        "pdf_path": "",
        "code_path": "",
        "notes": f"paperbanana_failed:{proc.returncode if proc else 'unknown'}:{last_detail}",
        "objective": objective,
    }


def run_figure_orchestra(
    outline: dict,
    state: dict,
    iterations: list[dict],
    figures_dir: Path,
    *,
    baseline: float | None,
    metric_name: str,
    paperbanana_cmd: str | None = None,
    allow_external_diagrams: bool = False,
) -> dict[str, Any]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    evidence_plan = state.get("evidence_plan") if isinstance(state.get("evidence_plan"), dict) else {}
    raw_plan = outline.get("plotting_plan") if isinstance(outline, dict) else None
    if isinstance(raw_plan, list) and raw_plan:
        plan: list[dict[str, Any]] = raw_plan
    elif wants_visualization(evidence_plan):
        plan = _default_plot_plan(metric_name)
    else:
        plan = []
    if plan:
        plan = _augment_plotting_plan(plan, state, iterations, metric_name)

    assets: list[dict[str, Any]] = []
    for fig in plan[:1]:
        if not isinstance(fig, dict):
            continue
        plot_type = str(fig.get("plot_type") or "plot").lower()
        if plot_type == "diagram":
            continue
        fid = _safe_filename(str(fig.get("figure_id") or fig.get("title") or "figure"))
        title = str(fig.get("title") or fid)
        objective = str(fig.get("objective") or title)
        asset = render_native_figure(
            fig,
            figures_dir=figures_dir,
            state=state,
            iterations=iterations,
            baseline=baseline,
            metric_name=metric_name,
        )
        asset["data_source"] = fig.get("data_source") or "experimental_log.md"
        assets.append(asset)

    manifest = {
        "assets": assets,
        "plotting_plan_used": plan,
        "generated_count": len(assets),
    }
    (figures_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def run_postwriting_api_figure_stage(
    outline: dict,
    state: dict,
    paper_tex: str,
    figures_dir: Path,
    *,
    paperbanana_cmd: str | None = None,
) -> dict[str, Any]:
    """Optional API diagram pass after a manuscript draft exists.

    Early plotting remains artifact-backed and native. This stage is the only
    place where PaperBanana/API diagrams are allowed, because it can condition
    on the completed experiment state and the written problem framing.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    raw_enable = os.getenv("DEEPGRAPH_PAPERBANANA_ENABLE_POSTWRITE", "").strip().lower()
    if raw_enable in {"0", "false", "no", "off"}:
        enabled = False
    elif raw_enable in {"1", "true", "yes", "on"}:
        enabled = True
    else:
        enabled = bool(paperbanana_cmd)
    if not enabled or not paperbanana_cmd:
        manifest = {
            "stage": "postwriting_api_figures",
            "enabled": enabled,
            "generated_count": 0,
            "assets": [],
            "notes": "disabled_or_command_missing",
        }
        (figures_dir / "postwriting_api_figure_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return manifest

    title_for_figs = (state.get("title") or state.get("method_name") or "this paper").strip()
    method_for_figs = (state.get("method_name") or state.get("method") or "the proposed method").strip()
    problem_for_figs = (state.get("problem_statement") or state.get("problem") or "").strip()
    weakness_for_figs = (state.get("existing_weakness") or "").strip()
    method_summary_for_figs = (state.get("method_summary") or "").strip()
    central_claim = ""
    paper_intent = state.get("paper_intent") if isinstance(state.get("paper_intent"), dict) else {}
    if paper_intent:
        central_claim = str(paper_intent.get("central_claim") or "").strip()

    motivation_objective = (
        "Design the paper's motivation figure as a rich scientific illustration with cohesive composition. "
        f"Convey concretely why \"{title_for_figs}\" matters: state the failure mode the work targets, "
        f"why existing approaches fall short ({weakness_for_figs or 'their limitation'}), "
        "and what shift in framing the method introduces. "
        "Use a structured multi-panel framework layout (problem context, current-method failure, proposed contrast). "
        "Short integrated labels, module names, axis annotations, and arrow legends are encouraged when they sharpen the message. "
        "Avoid: empty abstract symbol clouds, a single giant icon centered on the canvas, generic flowchart pipelines, "
        "title banners, or text labels that paraphrase the caption."
    )
    overview_objective = (
        f"Design the paper's method overview figure showing how {method_for_figs} produces its claim "
        f"({central_claim or 'the central scientific contribution'}). "
        f"Concrete content to depict: {method_summary_for_figs or 'the data flow, intermediate representations, and verification or routing step'}. "
        "Use a structured framework layout with grouped modules: inputs, the core mechanism, evidence/verification, and output. "
        "Short integrated labels for module names, signal flow, and the central quantity computed (e.g. the discrepancy/score) are encouraged. "
        "It must look like a polished ICLR/NeurIPS framework figure, not a symbolic art piece. "
        "Avoid: title banner, caption-like sentences inside the image, single-icon centerpieces, plain horizontal pipeline."
    )

    diagram_plan: list[dict[str, Any]] = [
            {
                "figure_id": "fig_motivation_gemini",
                "plot_type": "diagram",
                "title": "Motivation",
                "objective": motivation_objective,
                "caption": (
                    f"Motivation for {method_for_figs}: "
                    f"{problem_for_figs or 'the failure mode the work addresses'}."
                ),
                "data_source": "postwriting manuscript draft plus problem framing",
                "aspect_ratio": "16:9",
            },
            {
                "figure_id": "fig_overview_gemini",
                "plot_type": "diagram",
                "title": "Overview",
                "objective": overview_objective,
                "caption": (
                    f"Overview of {method_for_figs}: "
                    f"{method_summary_for_figs[:240] or 'core mechanism, evidence flow, and final claim'}."
                ),
                "data_source": "postwriting manuscript draft plus method description",
                "aspect_ratio": "16:9",
            },
        ]

    enriched_state = {
        **state,
        "paper_body_excerpt": (paper_tex or "")[:16000],
    }
    assets: list[dict[str, Any]] = []
    for idx, fig in enumerate(diagram_plan[:2]):
        if idx > 0:
            delay = _inter_figure_delay_seconds()
            if delay > 0:
                time.sleep(delay)
        asset = _run_external_diagram(
            fig,
            figures_dir=figures_dir,
            state=enriched_state,
            paperbanana_cmd=paperbanana_cmd,
        )
        asset["renderer"] = "gemini_image_api"
        asset["stage"] = "postwriting_api_figures"
        asset["image_model"] = os.getenv("DEEPGRAPH_PAPERBANANA_IMAGE_MODEL", "").strip() or None
        assets.append(asset)

    manifest = {
        "stage": "postwriting_api_figures",
        "enabled": True,
        "generated_count": len(assets),
        "assets": assets,
        "image_model": os.getenv("DEEPGRAPH_PAPERBANANA_IMAGE_MODEL", "").strip() or None,
        "notes": "generated_after_initial_section_writing",
    }
    (figures_dir / "postwriting_api_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest
