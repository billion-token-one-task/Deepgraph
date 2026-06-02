"""Independent figure orchestration for PaperOrchestra plotting plans."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import textwrap
from pathlib import Path
from typing import Any

from agents.evidence_planner import wants_visualization
from agents.paperorchestra.figure_standard import (
    CONCEPT_FIGURE_RULES,
    FIGURE_STANDARD_VERSION,
    backend_plot_pack,
    default_plot_plan,
    experiment_figure_policy_manifest,
    is_blocklisted_internal_figure,
)


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
    return bool(CONCEPT_FIGURE_RULES["required"])


CONCEPT_REFERENCE_STYLE_NOTE = (
    "Before drawing, follow the PaperOrchestra concept-figure style reference at "
    "{project_root}/动机图和框架图例子: learn the tidy high-information schematic language, "
    "not just the surface cuteness. Use aligned regions, dashed/rounded containers, local zoom-ins, "
    "worked mini examples, compact formula or score callouts, small matrices/tables where useful, "
    "task-specific flat icons, rounded hand-written or marker-like sans labels, flat PowerPoint-like geometry, "
    "and concrete method objects. "
    "Do not copy the examples; adapt only the visual grammar to this paper. "
    "The main composition must be a structured technical schematic with dense semantics, as if carefully assembled in PPT. "
    "Use a pure white canvas by default; only tiny local modules may use very pale tints. Never use a warm yellow/cream full-canvas wash, vignette, gradient, grid paper, graph paper, notebook paper, worksheet lines, or ruled backgrounds. "
    "Choose icons adaptively from the paper's actual domain entities. For multi-agent papers, agent/avatar/robot icons must be visible and semantically meaningful as trace sources, not just tiny decoration; for other domains, use the corresponding entities such as graphs, queries, databases, documents, patients, cells, samples, sensors, or models. "
    "Avoid Times New Roman or formal serif text inside the image, text-only card stacks, isolated icon collages, "
    "large title banners, visible step numbers, numbered circle badges, dashboard layouts, decision boards, full-scene posters, furniture-heavy scenes, "
    "mascot-dominated illustrations, rendered cartoon illustration style, glossy/3D objects, cast shadows, scenic lighting, graph-paper backgrounds, unrelated envelope/tray metaphors, and generic input-output flowcharts."
)


def _default_plot_plan(metric_name: str) -> list[dict[str, Any]]:
    return default_plot_plan(metric_name)


def _backend_plot_pack(metric_name: str) -> list[dict[str, Any]]:
    return backend_plot_pack(metric_name)


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


def _diagram_safe_text(value: Any) -> Any:
    if isinstance(value, str):
        replacements = {
            "CPU-only": "local",
            "cpu-only": "local",
            "CPU only": "local",
            "cpu only": "local",
            "no GPU": "local",
            "without GPU": "without additional training",
        }
        out = value
        for old, new in replacements.items():
            out = out.replace(old, new)
        return out
    if isinstance(value, list):
        return [_diagram_safe_text(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_diagram_safe_text(item) for item in value)
    if isinstance(value, dict):
        return {
            str(key): _diagram_safe_text(item)
            for key, item in value.items()
            if str(key).lower() not in {"artifact_root", "workdir", "path", "paths"}
        }
    return value


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
        if isinstance(summary, dict) and (
            isinstance(summary.get("per_method"), dict)
            or _has_backend_matrix(summary)
        ):
            return summary
    return {}


def _figure_size(fig: dict[str, Any]) -> tuple[float, float]:
    ratios = {
        "21:9": (7.2, 2.7),
        "16:9": (6.9, 3.15),
        "4:1": (7.2, 2.75),
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
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.6,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.8,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
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


def _has_backend_matrix(summary: dict[str, Any]) -> bool:
    if not isinstance(summary, dict):
        return False
    return any(
        isinstance(summary.get(key), dict) and summary.get(key)
        for key in (
            "per_method_backend",
            "per_dataset_backend",
            "by_backend",
            "backend_matrix",
            "method_backend_scores",
        )
    )


def _metric_label(metric_name: str | None) -> str:
    name = str(metric_name or "accuracy").strip()
    aliases = {
        "accuracy": "Accuracy",
        "exact_match": "Exact Match",
        "em": "Exact Match",
        "utility": "Utility",
    }
    return aliases.get(name.lower(), name.replace("_", " ").title())


def _short_method_label(method: str) -> str:
    lookup = {
        "vanilla_direct": "Direct",
        "vanilla direct answering": "Direct",
        "direct": "Direct",
        "naive": "Naive",
        "confidence routing": "Conf.",
        "confidence_weighted_majority": "Conf.",
        "disagreement routing": "Disagree",
        "disagreement_gated_consensus": "Disagree",
        "random budget-matched routing": "Random",
        "random_two_agent": "Random",
        "always multi-agent majority": "Majority",
        "always_five_agents": "Majority",
        "diversity-preserving consensus (ours)": "DPC",
        "diversity-preserving consensus": "DPC",
        "diversity_preserving_consensus": "DPC",
        "oracle routing": "Oracle",
        "oracle_selector": "Oracle",
    }
    key = str(method or "").strip()
    if key in lookup:
        return lookup[key]
    lower_key = key.lower()
    if lower_key in lookup:
        return lookup[lower_key]
    words = key.replace("_", " ").replace("-", " ").split()
    if not words:
        return ""
    return " ".join(word[:1].upper() + word[1:] for word in words[:3])


def _method_palette(methods: list[str]) -> dict[str, str]:
    palette = [
        "#f6c28b",
        "#f4a3a3",
        "#d9a6d8",
        "#8fbce6",
        "#b7d7a8",
        "#c5b4e3",
        "#9fd3c7",
        "#f0d084",
        "#b8c0cc",
        "#e7b7c8",
    ]
    return {method: palette[idx % len(palette)] for idx, method in enumerate(methods)}


def _method_is_oracle(method: str) -> bool:
    return "oracle" in str(method or "").lower()


def _method_is_reference(method: str) -> bool:
    text = str(method or "").lower()
    return any(token in text for token in ("naive", "direct", "baseline", "vanilla"))


def _main_results_figure_spec(metric_name: str, summary: dict[str, Any]) -> dict[str, Any]:
    """Choose main-results layout from evidence instead of forcing panels."""
    fig = _default_plot_plan(metric_name)[0]
    if isinstance(summary.get("per_method"), dict) and len(summary.get("per_method") or {}) >= 4:
        fig = {**fig, "aspect_ratio": "4:1", "layout": "1x4", "placement": "double_column"}
    return fig


def _row_metric(row: Any, metric_name: str | None) -> float | None:
    if not isinstance(row, dict):
        return _as_float(row)
    candidates = [
        metric_name,
        "metric_value",
        "accuracy",
        "exact_match",
        "em",
        "score",
        "utility",
    ]
    for key in candidates:
        if key and _as_float(row.get(str(key))) is not None:
            return _as_float(row.get(str(key)))
    for value in row.values():
        parsed = _as_float(value)
        if parsed is not None:
            return parsed
    return None


def _row_std(row: Any, metric_name: str | None) -> float:
    if not isinstance(row, dict):
        return 0.0
    candidates = [
        "std",
        "seed_std",
        "stddev",
        "standard_deviation",
        f"{metric_name}_std" if metric_name else "",
        "accuracy_std",
        "exact_match_std",
        "em_std",
    ]
    for key in candidates:
        if key and _as_float(row.get(key)) is not None:
            return float(_as_float(row.get(key)) or 0.0)
    return 0.0


def _backend_source(summary: dict[str, Any], dataset: str | None = None) -> dict[str, Any]:
    if dataset and isinstance(summary.get("per_dataset_backend"), dict):
        item = summary["per_dataset_backend"].get(dataset)
        if isinstance(item, dict):
            return item
    for key in ("per_method_backend", "method_backend_scores", "backend_matrix"):
        item = summary.get(key)
        if isinstance(item, dict) and item:
            return item
    item = summary.get("by_backend")
    return item if isinstance(item, dict) else {}


def _backend_score_matrix(
    summary: dict[str, Any],
    *,
    dataset: str | None = None,
    metric_name: str | None = None,
) -> tuple[list[str], list[str], list[list[float]], list[list[float]], str]:
    source = _backend_source(summary, dataset=dataset)
    metric = str(metric_name or summary.get("primary_metric") or summary.get("metric_name") or "accuracy")
    if not source:
        return [], [], [], [], metric

    declared_backends = [str(x) for x in (summary.get("backends") or [])]
    declared_methods = [str(x) for x in (summary.get("methods") or [])]
    first_keys = [str(x) for x in source.keys()]
    looks_backend_first = False
    if declared_backends and set(first_keys).issubset(set(declared_backends)):
        looks_backend_first = True
    elif str(summary.get("matrix_orientation") or "").lower() in {"backend_method", "backend-first"}:
        looks_backend_first = True
    elif "by_backend" in summary and source is summary.get("by_backend"):
        looks_backend_first = True

    normalized: dict[str, dict[str, Any]] = {}
    if looks_backend_first:
        for backend, method_rows in source.items():
            if not isinstance(method_rows, dict):
                continue
            for method, cell in method_rows.items():
                normalized.setdefault(str(method), {})[str(backend)] = cell
    else:
        for method, backend_rows in source.items():
            if isinstance(backend_rows, dict):
                normalized[str(method)] = {str(k): v for k, v in backend_rows.items()}

    methods = [m for m in declared_methods if m in normalized] or list(normalized.keys())
    backends = declared_backends or sorted({b for row in normalized.values() for b in row.keys()})
    values: list[list[float]] = []
    stds: list[list[float]] = []
    for method in methods:
        row_values: list[float] = []
        row_stds: list[float] = []
        for backend in backends:
            cell = normalized.get(method, {}).get(backend, {})
            row_values.append(float(_row_metric(cell, metric) or 0.0))
            row_stds.append(float(_row_std(cell, metric)))
        values.append(row_values)
        stds.append(row_stds)
    return methods, backends, values, stds, metric


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
        "aspect_ratio": fig.get("aspect_ratio"),
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


def _render_main_results_bar(fig: dict[str, Any], state: dict, baseline: float | None, metric_name: str, out_path: Path) -> None:
    summary = _state_benchmark_summary(state)
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    if not per_method:
        _render_baseline_bar(fig, state, _metric_points([]), baseline, out_path)
        return

    plt = _setup_matplotlib()
    import numpy as np

    metric = str(summary.get("primary_metric") or summary.get("metric_name") or metric_name or "accuracy")
    std_table = summary.get("per_method_std") if isinstance(summary.get("per_method_std"), dict) else {}
    methods = list(per_method.keys())[:9]
    values = [float(_row_metric(per_method.get(method), metric) or 0.0) for method in methods]
    errors: list[float] = []
    for method in methods:
        row_std = _row_std(per_method.get(method), metric)
        if not row_std and method in std_table:
            row_std = _row_std(std_table.get(method), metric)
        errors.append(float(row_std or 0.0))

    colors = _method_palette(methods)
    bar_colors = [colors[method] for method in methods]
    hatch_cycle = ["//", "--", "xx", "\\\\", "..", "++", "oo", "///", "\\\\\\"]
    hatches = [
        "///" if _method_is_oracle(method) else hatch_cycle[idx % len(hatch_cycle)]
        for idx, method in enumerate(methods)
    ]
    x = np.arange(len(methods))
    labels = [_short_method_label(method) for method in methods]
    wide = str(fig.get("aspect_ratio") or "") == "4:1" or str(fig.get("chart_type") or "").endswith("1x4")
    proposed_idx = next((idx for idx, method in enumerate(methods) if "diversity" in method.lower() or method.upper() == "DPC"), None)
    reference_idx = next((idx for idx, method in enumerate(methods) if _method_is_reference(method)), None)
    if wide:
        metric_specs = [
            (metric, _metric_label(metric), values, errors, "{:.3f}"),
            ("avg_new_tokens", "Tokens", [float(_as_float((per_method.get(m) or {}).get("avg_new_tokens")) or 0.0) for m in methods], [0.0] * len(methods), "{:.0f}"),
            ("avg_latency_seconds", "Latency", [float(_as_float((per_method.get(m) or {}).get("avg_latency_seconds")) or 0.0) for m in methods], [0.0] * len(methods), "{:.2f}"),
            ("route_rate", "Route", [float(_as_float((per_method.get(m) or {}).get("route_rate")) or 0.0) for m in methods], [0.0] * len(methods), "{:.2f}"),
        ]
        fig_obj, axes_raw = plt.subplots(1, 4, figsize=(9.8, 2.45), sharex=False)
        axes = list(axes_raw)
    else:
        metric_specs = [(metric, _metric_label(metric), values, errors, "{:.3f}")]
        fig_obj, ax = plt.subplots(figsize=_figure_size(fig))
        axes = [ax]
    for ax, (_key, ylabel, vals, errs, fmt) in zip(axes, metric_specs):
        vals_arr = np.array(vals, dtype=float)
        annotate_indices: set[int] = set()
        if len(vals_arr):
            best_idx = int(np.argmax(vals_arr))
            annotate_indices.add(best_idx)
        if proposed_idx is not None:
            annotate_indices.add(proposed_idx)
        if reference_idx is not None and not wide:
            annotate_indices.add(reference_idx)
        bars = ax.bar(
            x,
            vals,
            yerr=errs,
            capsize=2.4,
            width=0.64,
            color=bar_colors,
            edgecolor="#1f2937",
            linewidth=0.72,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        if wide:
            ax.set_title(ylabel, fontweight="bold", pad=4, fontsize=9.2)
        else:
            ax.set_ylabel(ylabel)
        if wide:
            ax.set_xticks([])
            ax.tick_params(axis="x", length=0, pad=1)
        else:
            ax.set_xticks(x, labels, rotation=0, ha="center")
        ax.grid(axis="both", color="#d1d5db", linewidth=0.55, alpha=0.52)
        ax.set_axisbelow(True)
        top = max([v + e for v, e in zip(vals, errs)] + [0.01])
        if ylabel in {"Accuracy", "Route"}:
            ax.set_ylim(0, min(1.08, top + 0.12))
        else:
            ax.set_ylim(0, top * 1.18 if top > 0 else 1.0)
        for idx, (bar, value) in enumerate(zip(bars, vals)):
            show_label = idx in annotate_indices
            if wide and len(vals) <= 7:
                show_label = True
            if not show_label:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(0.006, top * 0.018),
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=7.2 if wide else 7.2,
                rotation=0,
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    if wide:
        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor=bar_colors[idx], edgecolor="#1f2937", hatch=hatches[idx], label=labels[idx])
            for idx in range(len(methods))
        ]
        fig_obj.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(methods), 7),
            frameon=True,
            edgecolor="#d1d5db",
            fontsize=7.8,
            handlelength=1.35,
            columnspacing=0.95,
            bbox_to_anchor=(0.5, 0.015),
        )
        fig_obj.subplots_adjust(left=0.055, right=0.995, top=0.84, bottom=0.18, wspace=0.22)
    else:
        fig_obj.tight_layout(pad=0.45)
    _save_native_matplotlib_figure(fig_obj, out_path)
    plt.close(fig_obj)


def _render_backend_3d_bars(fig: dict[str, Any], state: dict, metric_name: str, out_path: Path) -> None:
    summary = _state_benchmark_summary(state)
    methods, backends, values, stds, metric = _backend_score_matrix(summary, metric_name=metric_name)
    if not methods or not backends:
        _render_main_results_bar(fig, state, None, metric_name, out_path)
        return

    plt = _setup_matplotlib()
    import numpy as np

    fig_obj = plt.figure(figsize=_figure_size(fig))
    ax = fig_obj.add_subplot(111, projection="3d")
    palette = _method_palette(methods)
    dx = 0.56
    dy = 0.56
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    dzs: list[float] = []
    colors: list[str] = []
    errs: list[float] = []
    for mi, method in enumerate(methods):
        for bi, _backend in enumerate(backends):
            xs.append(float(bi))
            ys.append(float(mi))
            zs.append(0.0)
            dzs.append(values[mi][bi])
            colors.append(palette[method])
            errs.append(stds[mi][bi])
    ax.bar3d(xs, ys, zs, dx, dy, dzs, color=colors, alpha=0.86, edgecolor="#1f2937", linewidth=0.22, shade=True)
    for x, y, value, err in zip(xs, ys, dzs, errs):
        if err <= 0:
            continue
        xpos = x + dx / 2
        ypos = y + dy / 2
        ax.plot([xpos, xpos], [ypos, ypos], [max(0.0, value - err), value + err], color="#111827", linewidth=0.65)
        ax.plot([xpos - 0.08, xpos + 0.08], [ypos, ypos], [value + err, value + err], color="#111827", linewidth=0.65)
    ref_values = [
        values[idx][bidx]
        for idx, method in enumerate(methods)
        for bidx in range(len(backends))
        if _method_is_reference(method)
    ]
    if ref_values:
        ref = float(np.mean(ref_values))
        xx, yy = np.meshgrid(np.linspace(0, max(1, len(backends) - 1) + dx, 2), np.linspace(0, max(1, len(methods) - 1) + dy, 2))
        zz = np.full_like(xx, ref)
        ax.plot_surface(xx, yy, zz, color="#111827", alpha=0.08, linewidth=0)
    ax.set_xticks(np.arange(len(backends)) + dx / 2)
    ax.set_xticklabels(backends, rotation=18, ha="right")
    ax.set_yticks(np.arange(len(methods)) + dy / 2)
    ax.set_yticklabels([_short_method_label(m) for m in methods])
    ax.set_zlabel(_metric_label(metric))
    ax.view_init(elev=24, azim=-55)
    fig_obj.tight_layout(pad=0.4)
    _save_native_matplotlib_figure(fig_obj, out_path)
    plt.close(fig_obj)


def _render_quality_cost_tradeoff(fig: dict[str, Any], state: dict, metric_name: str, out_path: Path) -> None:
    summary = _state_benchmark_summary(state)
    per_method = summary.get("per_method") if isinstance(summary.get("per_method"), dict) else {}
    if not per_method:
        _render_main_results_bar(fig, state, None, metric_name, out_path)
        return

    plt = _setup_matplotlib()
    methods = list(per_method.keys())[:10]
    colors = _method_palette(methods)
    fig_obj, ax = plt.subplots(figsize=_figure_size({"aspect_ratio": "4:3"}))
    label_overrides = {
        "vanilla direct answering": "Direct",
        "direct": "Direct",
        "confidence routing": "Conf.",
        "disagreement routing": "Disagree",
        "random budget-matched routing": "Random",
        "always multi-agent majority": "Majority",
        "diversity-preserving consensus": "DPC",
        "diversity preserving consensus": "DPC",
        "oracle routing": "Oracle",
    }
    points: dict[str, tuple[float, float]] = {}
    for method in methods:
        row = per_method.get(method) if isinstance(per_method.get(method), dict) else {}
        tokens = float(_as_float(row.get("avg_new_tokens") or row.get("tokens")) or 0.0)
        score = float(_row_metric(row, metric_name) or 0.0)
        std = float(_row_std(row, metric_name))
        is_oracle = _method_is_oracle(method)
        marker = "*" if is_oracle else "o"
        size = 80 if is_oracle else 52
        ax.errorbar(
            tokens,
            score,
            yerr=std if std else None,
            fmt="none",
            ecolor="#9ca3af",
            elinewidth=0.72,
            capsize=2,
            zorder=1,
        )
        ax.scatter(
            tokens,
            score,
            s=size,
            marker=marker,
            color=colors[method],
            edgecolor="#111827",
            linewidth=0.55,
            zorder=3,
            alpha=0.95,
        )
        method_key = method.lower()
        if "diversity" in method_key and "consensus" in method_key:
            label = "DPC"
        elif "confidence" in method_key and "routing" in method_key:
            label = "Conf."
        elif "disagreement" in method_key and "routing" in method_key:
            label = "Disagree"
        elif "random" in method_key:
            label = "Random"
        elif "majority" in method_key:
            label = "Majority"
        else:
            label = label_overrides.get(method_key, _short_method_label(method))
        dx = 5
        ha = "left"
        if label in {"Majority", "DPC"}:
            dx = -8 if label == "DPC" else -18
            ha = "right"
        dy = -2 if label == "Oracle" else 2
        ax.annotate(label, (tokens, score), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center")
        points[label] = (tokens, score)
    if "Conf." in points and "DPC" in points:
        ax.plot(
            [points["Conf."][0], points["DPC"][0]],
            [points["Conf."][1], points["DPC"][1]],
            color="#64748b",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
            zorder=0,
        )
    max_tokens = max((pt[0] for pt in points.values()), default=1.0)
    ax.set_xlabel("Average new tokens")
    ax.set_ylabel(_metric_label(metric_name))
    ax.set_xlim(0, max_tokens * 1.12)
    ax.set_ylim(0.55, 1.03)
    ax.grid(color="#e5e7eb", linewidth=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.02, 0.97, "Higher is better; left is cheaper", transform=ax.transAxes, ha="left", va="top", fontsize=7.2, color="#374151")
    fig_obj.tight_layout(pad=0.5)
    _save_native_matplotlib_figure(fig_obj, out_path)
    plt.close(fig_obj)


def _render_backend_grouped_bars(fig: dict[str, Any], state: dict, metric_name: str, out_path: Path) -> None:
    """Method-by-backend comparison as a clean 2D grouped bar chart.

    The pipeline keeps 3D bars for true two-hyperparameter sensitivity surfaces;
    ordinary method x backend comparisons are easier to read as grouped bars or
    heatmaps.
    """
    plt = _setup_matplotlib()
    import numpy as np

    summary = _state_benchmark_summary(state)
    methods, backends, values, stds, metric = _backend_score_matrix(summary, metric_name=metric_name)
    if not methods or not backends:
        _render_main_results_bar(fig, state, None, metric_name, out_path)
        return

    fig_obj, ax = plt.subplots(figsize=_figure_size({"aspect_ratio": "4:3"}))
    x = np.arange(len(backends), dtype=float)
    width = min(0.78 / max(1, len(methods)), 0.18)
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
    palette = _method_palette(methods)
    for idx, method in enumerate(methods):
        heights = np.array([row[idx] for row in np.array(values).T], dtype=float)
        errs = np.array([row[idx] for row in np.array(stds).T], dtype=float)
        ax.bar(
            x + offsets[idx],
            heights,
            width=width * 0.92,
            label=_short_method_label(method),
            color=palette[method],
            edgecolor="#1f2937",
            linewidth=0.35,
            yerr=errs if np.any(errs > 0) else None,
            capsize=2.0,
            error_kw={"elinewidth": 0.7, "capthick": 0.7},
        )
    reference_values = [
        values[m_idx][b_idx]
        for m_idx, method in enumerate(methods)
        for b_idx in range(len(backends))
        if _method_is_reference(method)
    ]
    if reference_values:
        ax.axhline(float(np.mean(reference_values)), color="#374151", linestyle="--", linewidth=0.85, alpha=0.75)
    ax.set_ylabel(_metric_label(metric))
    ax.set_xticks(x, [_wrap_label(backend, 9) for backend in backends])
    upper = max(max(row) for row in values)
    lower = min(min(row) for row in values)
    pad = max(0.02, (upper - lower) * 0.24)
    ax.set_ylim(max(0.0, lower - pad), min(1.0, upper + pad) if upper <= 1.0 else upper + pad)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=min(3, len(methods)), loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig_obj.tight_layout(pad=0.45)
    _save_native_matplotlib_figure(fig_obj, out_path)
    plt.close(fig_obj)


def _render_backend_heatmap_single(fig: dict[str, Any], state: dict, metric_name: str, out_path: Path) -> None:
    summary = _state_benchmark_summary(state)
    methods, backends, values, stds, metric = _backend_score_matrix(summary, metric_name=metric_name)
    if not methods or not backends:
        _render_main_results_bar(fig, state, None, metric_name, out_path)
        return

    plt = _setup_matplotlib()
    import matplotlib.patches as patches
    import numpy as np

    matrix = np.array(values, dtype=float)
    fig_obj, ax = plt.subplots(figsize=_figure_size(fig))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=max(0.0, float(np.nanmin(matrix)) - 0.03), vmax=min(1.0, float(np.nanmax(matrix)) + 0.03))
    ax.set_xticks(np.arange(len(backends)), backends, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(methods)), [_short_method_label(method) for method in methods])
    for i, method in enumerate(methods):
        for j, _backend in enumerate(backends):
            value = values[i][j]
            std = stds[i][j]
            ax.text(j, i, f"{value:.3f}\n±{std:.3f}", ha="center", va="center", fontsize=6.8, color="#111827")
    for j in range(len(backends)):
        eligible = [(i, values[i][j]) for i, method in enumerate(methods) if not _method_is_oracle(method)]
        if eligible:
            best_i, _ = max(eligible, key=lambda item: item[1])
            ax.add_patch(patches.Rectangle((j - 0.5, best_i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.4))
    cbar = fig_obj.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label(_metric_label(metric))
    fig_obj.tight_layout(pad=0.45)
    _save_native_matplotlib_figure(fig_obj, out_path)
    plt.close(fig_obj)


def _render_backend_rank_lines_1x4(fig: dict[str, Any], state: dict, metric_name: str, out_path: Path) -> None:
    summary = _state_benchmark_summary(state)
    dataset_map = summary.get("per_dataset_backend") if isinstance(summary.get("per_dataset_backend"), dict) else {}
    datasets = list(dataset_map.keys())[:4] if dataset_map else ["Aggregate"]
    while len(datasets) < 4:
        datasets.append(f"Aggregate {len(datasets) + 1}")

    plt = _setup_matplotlib()
    import numpy as np

    fig_obj, axes = plt.subplots(1, 4, figsize=_figure_size({"aspect_ratio": "4:1"}), sharey=True)
    all_methods: list[str] = []
    panel_data: list[tuple[str, list[str], list[str], list[list[float]], list[list[float]], str]] = []
    for dataset in datasets[:4]:
        use_dataset = dataset if dataset in dataset_map else None
        methods, backends, values, stds, metric = _backend_score_matrix(summary, dataset=use_dataset, metric_name=metric_name)
        panel_data.append((dataset, methods, backends, values, stds, metric))
        for method in methods:
            if method not in all_methods:
                all_methods.append(method)
    palette = _method_palette(all_methods)
    max_rank = max(1, len(all_methods))
    for ax, (dataset, methods, backends, values, stds, _metric) in zip(axes, panel_data):
        if not methods or not backends:
            ax.axis("off")
            continue
        values_arr = np.array(values, dtype=float)
        ranks = np.zeros_like(values_arr)
        for j in range(values_arr.shape[1]):
            order = list(np.argsort(-values_arr[:, j]))
            for rank_idx, method_idx in enumerate(order, start=1):
                ranks[method_idx, j] = rank_idx
        xs = np.arange(len(backends))
        for i, method in enumerate(methods):
            y = ranks[i]
            score_std = np.array(stds[i], dtype=float)
            band = np.clip(score_std * max(1.0, len(methods)) * 3.0, 0.18, 0.75)
            unstable = (float(np.max(y)) - float(np.min(y))) >= 2.0
            ax.fill_between(xs, y - band, y + band, color=palette[method], alpha=0.20, linewidth=0)
            ax.plot(xs, y, marker="o", markersize=3.0, linewidth=1.15, color=palette[method], label=_short_method_label(method))
            if unstable:
                ax.scatter(xs, y, s=18, facecolor=palette[method], edgecolor="#111827", linewidth=0.85, zorder=3)
        ax.set_title(str(dataset)[:18])
        ax.set_xticks(xs, backends, rotation=18, ha="right")
        ax.set_ylim(max_rank + 0.5, 0.5)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Rank")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig_obj.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig_obj.tight_layout(pad=0.35, w_pad=0.5)
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

    def row_for(method: str) -> dict[str, Any]:
        row = per_method.get(method)
        return row if isinstance(row, dict) else {}

    def values(metric: str) -> list[float]:
        return [float(_as_float(row_for(m).get(metric)) or 0.0) for m in methods]

    def errors(metric: str) -> list[float]:
        out = []
        for m in methods:
            row = std.get(m)
            if isinstance(row, dict):
                out.append(float(_as_float(row.get(metric)) or 0.0))
            else:
                out.append(float(_as_float(row) or 0.0) if metric == primary_metric else 0.0)
        return out

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
            for k in (row_for(m).get("budget_histogram") or {}).keys()
        },
        key=lambda v: int(v) if v.isdigit() else v,
    )
    if not budgets and any(row_for(m).get("avg_retained_agents") for m in methods):
        budgets = ["avg_retained_agents"]
    bottom = np.zeros(len(methods))
    for idx, budget in enumerate(budgets[:8]):
        counts = []
        for m in methods:
            hist = row_for(m).get("budget_histogram") or {}
            value = row_for(m).get("avg_retained_agents") if budget == "avg_retained_agents" else hist.get(budget)
            counts.append(float(_as_float(value) or 0.0))
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


def _allowed_optional_experiment_figure(fig: dict[str, Any], summary: dict[str, Any]) -> bool:
    text = " ".join(
        str(fig.get(key) or "")
        for key in ("figure_id", "title", "objective", "chart_type", "data_source")
    ).lower()
    if is_blocklisted_internal_figure(fig):
        return False
    if "ablation" in text:
        return bool(summary.get("ablation_table"))
    if any(token in text for token in ("trend", "sensitivity")):
        return bool(summary.get("trend_table") or summary.get("sensitivity_table"))
    return False


def _augment_plotting_plan(plan: list[dict[str, Any]], state: dict, iterations: list[dict], metric_name: str) -> list[dict[str, Any]]:
    """Enforce PaperOrchestra experiment-figure packs from verified evidence."""
    summary = _state_benchmark_summary(state)
    if _has_backend_matrix(summary):
        return _backend_plot_pack(metric_name)

    cleaned: list[dict[str, Any]] = []
    diagrams: list[dict[str, Any]] = []
    for item in plan:
        if not isinstance(item, dict):
            continue
        fig = dict(item)
        plot_type = str(fig.get("plot_type") or "plot").lower()
        if plot_type == "diagram":
            if _is_motivation_or_overview_figure(fig) and len(diagrams) < 2:
                diagrams.append(fig)
            continue
        if _allowed_optional_experiment_figure(fig, summary):
            cleaned.append(fig)

    main = _main_results_figure_spec(metric_name, summary)
    if not _has_plan_topic(cleaned, "main results", "fig_main_results"):
        cleaned.insert(0, main)
    return cleaned[:3] + diagrams[:2]


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
    chart_type = str(fig.get("chart_type") or "").lower()
    rows = _metric_points(iterations)
    try:
        if fid == "fig_quality_cost_tradeoff" or chart_type == "quality_cost_tradeoff":
            return {
                "figure_id": fid,
                "title": str(fig.get("title") or fid),
                "kind": "blocked",
                "path": "",
                "svg_path": "",
                "pdf_path": "",
                "code_path": "",
                "notes": "blocked_nonstandard_experiment_figure",
                "objective": objective,
                "blocker": "Quality-cost scatter plots are not part of the default experiment figure pack; use standard bar, heatmap, line, sensitivity, or ablation figures backed by artifacts.",
            }
        if fid == "fig_backend_grouped_bars" or chart_type == "backend_grouped_bars":
            _render_backend_grouped_bars(fig, state, metric_name, out_path)
            return _native_asset(
                fid=fid,
                fig=fig,
                out_path=out_path,
                kind="plot",
                renderer="backend_grouped_bars",
                objective=objective,
                extras={"data_source": fig.get("data_source") or "benchmark_summary.json"},
            )
        if fid == "fig_backend_3d_bars" or chart_type == "backend_3d_bars":
            _render_backend_3d_bars(fig, state, metric_name, out_path)
            return _native_asset(fid=fid, fig=fig, out_path=out_path, kind="plot", renderer="backend_3d_bars", objective=objective)
        if fid == "fig_backend_heatmap_single" or chart_type == "backend_heatmap_single":
            _render_backend_heatmap_single(fig, state, metric_name, out_path)
            return _native_asset(fid=fid, fig=fig, out_path=out_path, kind="plot", renderer="backend_heatmap_single", objective=objective)
        if fid == "fig_backend_rank_lines_1x4" or chart_type == "backend_rank_lines_1x4":
            _render_backend_rank_lines_1x4(fig, state, metric_name, out_path)
            return _native_asset(fid=fid, fig=fig, out_path=out_path, kind="plot", renderer="backend_rank_lines_1x4", objective=objective)
        if fid == "fig_main_results" or chart_type == "main_results_bar":
            _render_main_results_bar(fig, state, baseline, metric_name, out_path)
            return _native_asset(fid=fid, fig=fig, out_path=out_path, kind="plot", renderer="main_results_bar", objective=objective)
        if chart_type == "main_results_bar_1x2":
            _render_main_results_bar(fig, state, baseline, metric_name, out_path)
            return _native_asset(fid=fid, fig=fig, out_path=out_path, kind="plot", renderer="main_results_bar", objective=objective)
        if _is_motivation_or_overview_figure(fig):
            return {
                "figure_id": fid,
                "title": str(fig.get("title") or fid),
                "kind": "blocked",
                "path": "",
                "svg_path": "",
                "pdf_path": "",
                "code_path": "",
                "notes": "postwriting_api_required",
                "objective": objective,
                "blocker": "Motivation/overview figures must be generated by Gemini/PaperBanana post-writing stage.",
            }
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
        _render_main_results_bar(fig, state, baseline, metric_name, out_path)
        return _native_asset(fid=fid, fig=fig, out_path=out_path, kind="plot", renderer="main_results_bar", objective=objective)
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

    safe_state = _diagram_safe_text(state)
    spec = json.dumps(
        {
            "figure": fig,
            "state_title": safe_state.get("title"),
            "method_name": safe_state.get("method_name"),
            "method_summary": safe_state.get("method_summary"),
            "problem_awareness": safe_state.get("problem_awareness") or {},
            "paper_body_excerpt": safe_state.get("paper_body_excerpt") or "",
            "problem_statement": safe_state.get("problem_statement"),
            "existing_weakness": safe_state.get("existing_weakness"),
            "contributions": safe_state.get("contributions") or [],
            "evidence_summary": safe_state.get("evidence_summary"),
            "baseline_metric_name": safe_state.get("baseline_metric_name"),
            "baseline_metric_value": safe_state.get("baseline_metric_value"),
            "best_metric_value": safe_state.get("best_metric_value"),
            "effect_pct": safe_state.get("effect_pct"),
            "verdict": safe_state.get("verdict"),
            "evidence_plan": safe_state.get("evidence_plan") or {},
            "experimental_plan": {
                "datasets": safe_state.get("datasets") or [],
                "baselines": safe_state.get("baselines") or [],
            },
            "concept_style_reference": {
                "local_dir": "{project_root}/动机图和框架图例子",
                "learn": [
                    "tidy high-information academic schematic composition",
                    "domain-adaptive task-specific icons attached to concrete method objects",
                    "structured regions with small character accents only as annotations",
                    "local zoom-ins and method-object relationships",
                    "worked miniature examples, score tags, and formula/rule callouts",
                    "rounded hand-written or marker-like sans labels, not manuscript serif text",
                    "compact labels with high visual explanation",
                    "mechanism shown through slips, badges, gauges, matrices, tables, and spatial grouping",
                ],
                "do_not_copy": True,
                "avoid": [
                    "text-only card stacks",
                    "plain input-output pipeline",
                    "generic module boxes",
                    "empty flowchart",
                    "isolated icon collage",
                    "decision board or dashboard layout",
                    "large all-caps headings",
                    "visible step numbers or numbered circle badges",
                    "full-scene poster",
                    "mascot-dominated illustration",
                    "furniture-heavy scene illustration",
                    "Times New Roman or formal serif labels inside the figure",
                    "cute illustration without enough technical content",
                ],
            },
        },
        ensure_ascii=False,
    )
    command = paperbanana_cmd.format(
        output=_shell_quote(str(out_path.resolve())),
        spec=_shell_quote(spec),
    )
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
    if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size <= 0:
        placeholder = figures_dir / f"{fid}.svg"
        _placeholder_diagram(placeholder, str(fig.get("title") or fid), objective)
        detail = _clip(((proc.stderr or "") + "\n" + (proc.stdout or "")).strip())
        return {
            "figure_id": fid,
            "title": str(fig.get("title") or fid),
            "kind": "diagram",
            "path": str(placeholder),
            "svg_path": str(placeholder),
            "pdf_path": "",
            "code_path": "",
            "notes": f"paperbanana_failed:{proc.returncode}:{detail}",
            "objective": objective,
        }
    return {
        "figure_id": fid,
        "title": str(fig.get("title") or fid),
        "kind": "diagram",
        "path": str(out_path),
        "svg_path": "",
        "pdf_path": "",
        "code_path": "",
        "notes": "paperbanana_ok",
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
    if _has_backend_matrix(_state_benchmark_summary(state)):
        plan = _backend_plot_pack(metric_name)
    elif isinstance(raw_plan, list) and raw_plan:
        plan: list[dict[str, Any]] = raw_plan
    elif wants_visualization(evidence_plan):
        plan = _default_plot_plan(metric_name)
    else:
        plan = []
    if plan:
        plan = _augment_plotting_plan(plan, state, iterations, metric_name)

    assets: list[dict[str, Any]] = []
    blockers: list[str] = []
    for fig in plan[:12]:
        if not isinstance(fig, dict):
            continue
        fid = _safe_filename(str(fig.get("figure_id") or fig.get("title") or "figure"))
        title = str(fig.get("title") or fid)
        objective = str(fig.get("objective") or title)
        plot_type = str(fig.get("plot_type") or "plot").lower()
        if plot_type == "diagram":
            is_required_api_diagram = _banana_motivation_overview_enabled() and _is_motivation_or_overview_figure(fig)
            force_banana = is_required_api_diagram and paperbanana_cmd
            prefer_ai = os.getenv("DEEPGRAPH_PAPERBANANA_PREFER_AI", "").strip().lower() in {"1", "true", "yes"}
            if force_banana or (allow_external_diagrams and prefer_ai and paperbanana_cmd):
                asset = _run_external_diagram(
                    fig,
                    figures_dir=figures_dir,
                    state=state,
                    paperbanana_cmd=paperbanana_cmd,
                )
            elif is_required_api_diagram:
                asset = {
                    "figure_id": fid,
                    "title": title,
                    "kind": "blocked",
                    "path": "",
                    "svg_path": "",
                    "pdf_path": "",
                    "code_path": "",
                    "notes": "paperbanana_required_missing",
                    "objective": objective,
                    "blocker": "Motivation/overview figures must be generated by Gemini/PaperBanana post-writing stage.",
                }
            else:
                asset = render_native_figure(
                    fig,
                    figures_dir=figures_dir,
                    state=state,
                    iterations=iterations,
                    baseline=baseline,
                    metric_name=metric_name,
                )
        else:
            asset = render_native_figure(
                fig,
                figures_dir=figures_dir,
                state=state,
                iterations=iterations,
                baseline=baseline,
                metric_name=metric_name,
            )
            asset["data_source"] = fig.get("data_source") or "experimental_log.md"
        if asset.get("blocker"):
            blockers.append(str(asset.get("blocker")))
        assets.append(asset)

    manifest = {
        "standard_version": FIGURE_STANDARD_VERSION,
        "policy": experiment_figure_policy_manifest(),
        "assets": assets,
        "plotting_plan_used": plan,
        "generated_count": len(assets),
        "blockers": blockers,
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
    """Required API diagram pass for motivation/overview figures.

    Early plotting remains artifact-backed and native. This stage is the only
    place where PaperBanana/API diagrams are allowed, because it can condition
    on the completed experiment state and the written problem framing.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not paperbanana_cmd:
        manifest = {
            "stage": "postwriting_api_figures",
            "standard_version": FIGURE_STANDARD_VERSION,
            "policy": experiment_figure_policy_manifest(),
            "required": True,
            "enabled": False,
            "generated_count": 0,
            "assets": [],
            "notes": "missing_paperbanana_command",
            "blockers": ["Motivation/overview figures are required and must be generated by Gemini/PaperBanana post-writing stage."],
        }
        (figures_dir / "postwriting_api_figure_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return manifest

    raw_plan = outline.get("plotting_plan") if isinstance(outline, dict) else None
    candidate_plan = [dict(row) for row in raw_plan or [] if isinstance(row, dict)]
    diagram_plan = [
        row
        for row in candidate_plan
        if str(row.get("plot_type") or "").lower() == "diagram"
        or any(
            token in " ".join(str(row.get(k) or "") for k in ("figure_id", "title", "objective")).lower()
            for token in ("framework", "overview", "method", "problem", "gating", "architecture")
        )
    ]
    plan_text = " ".join(
        " ".join(str(row.get(key) or "") for key in ("figure_id", "title", "objective")).lower()
        for row in diagram_plan
    )
    if "motivation" not in plan_text:
        diagram_plan.insert(
            0,
            {
                "figure_id": "fig_motivation_symbolic",
                "plot_type": "diagram",
                "title": "Motivation",
                "objective": (
                    CONCEPT_REFERENCE_STYLE_NOTE
                    + " Create a high-information flat PPT-built motivation schematic, not a rendered illustration, poster scene, or plain process flow. "
                    "Use three tidy comparison regions with a shared worked example. Use the paper's own domain entities as icons; for this DPC multi-agent setting, show five small agent/avatar icons A1-A5 producing answer bubbles A/A/A/B*/C, support counts, confidence chips, cost chips, token budget, and retained/lost marks. "
                    "Majority voting should visibly erase a high-confidence B* dissent; keep-all should visibly preserve B* but overrun token/latency budget; conditional retention should show a margin cue and a small rule card that keeps B* only when disagreement is meaningful. "
                    "Include concrete numeric cues such as conf=.95, cost=20, m=(3-1)/5, and a budget bar; include small icons only as annotations. "
                    "Use rounded hand-written or marker-like sans labels, not Times New Roman. "
                    "Use a pure white canvas. Only local modules may have very pale tints; no warm yellow/cream full-canvas wash, no vignette, no gradient, no grid, graph-paper, notebook, worksheet, or ruled lines. Use thick clean outlines, flat fills, dashed containers, crisp alignment, and minimal/no shading. "
                    "Avoid current-practice -> limitation -> motivation boxes, generic labels, large blank areas, text-only cards, all-caps titles, visible step numbers, isolated icon boards, full-scene cartoon posters, rendered illustration style, glossy objects, cast shadows, furniture, unrelated envelope/tray metaphors, and arrows carrying the whole story."
                ),
                "caption": (
                    "Motivation figure showing why fixed aggregation policies are insufficient: majority voting may discard "
                    "useful dissent, whereas keep-all reasoning spends unnecessary tokens, motivating conditional selection."
                ),
                "data_source": "postwriting manuscript draft plus figure caption intent",
                "aspect_ratio": "16:9",
            },
        )
    if "overview" not in plan_text and "framework" not in plan_text:
        diagram_plan.insert(
            1 if diagram_plan else 0,
            {
                "figure_id": "fig_overview_symbolic",
                "plot_type": "diagram",
                "title": "Overview",
                "objective": (
                    CONCEPT_REFERENCE_STYLE_NOTE
                    + " Create a mechanism-rich overview as a flat PPT-built structured academic schematic, not an input-output pipeline, decision-board dashboard, rendered illustration, or poster scene. "
                    "Show a worked multi-agent trace: five small agent/avatar icons A1-A5 each emitting an answer bubble with answer, confidence, and cost; a compact trace table; a grouping zoom-in with A x3, B* x1 high-conf, C x1; a central DPC rule card with m=(3-1)/5 and s_g=sum c_i + lambda dissent bonus; a budget bar; retained agent traces {A,B*}; discarded agent trace C; selected answer. "
                    "Make the core mechanism visually central and content-rich with local zoom-ins, score tags, and small matrix/table elements. "
                    "Agent/avatar icons must be visible and relevant in this multi-agent paper: they are the sources of the candidate traces, with answer bubbles connected to them. They should be flat schematic avatars, not glossy mascots. "
                    "Use rounded hand-written or marker-like sans labels, not Times New Roman. "
                    "Use a pure white canvas. Only local modules may have very pale tints; no warm yellow/cream full-canvas wash, no vignette, no gradient, no grid, graph-paper, notebook, worksheet, or ruled lines. Use thick clean outlines, flat fills, dashed containers, crisp alignment, and minimal/no shading. "
                    "Use very few arrows and no stage-chain layout. Do not use generic Module/Decision/Output labels, text-only card stacks, visible step numbers, numbered circle badges, large all-caps headings, isolated widgets, dashboard panels, rendered cartoon style, glossy objects, cast shadows, furniture, unrelated envelope/tray metaphors, or full-scene lab illustrations."
                ),
                "caption": (
                    "Overview figure of Diversity-Preserving Consensus as a non-flow mechanism: candidate evidence, "
                    "consensus support, dissent evidence, and the selected answer are organized around the central selector."
                ),
                "data_source": "postwriting manuscript draft plus figure caption intent",
                "aspect_ratio": "16:9",
            },
        )
    if not diagram_plan:
        pa = state.get("problem_awareness") if isinstance(state.get("problem_awareness"), dict) else {}
        diagram_plan = [
            {
                "figure_id": "fig_problem_method_result_spine",
                "plot_type": "diagram",
                "title": "Problem-method-result spine",
                "objective": (
                    CONCEPT_REFERENCE_STYLE_NOTE
                    + " Show the paper's central question, motivation, proposed mechanism, "
                    "and benchmark result in one ICLR-style method figure. "
                    + str(pa.get("central_question") or state.get("problem_statement") or "")[:220]
                ),
                "data_source": "postwriting manuscript draft plus experiment result packet",
                "aspect_ratio": "16:9",
            }
        ]

    enriched_state = {
        **state,
        "paper_body_excerpt": (paper_tex or "")[:16000],
    }
    assets: list[dict[str, Any]] = []
    for fig in diagram_plan[:2]:
        asset = _run_external_diagram(
            fig,
            figures_dir=figures_dir,
            state=enriched_state,
            paperbanana_cmd=paperbanana_cmd,
        )
        asset["stage"] = "postwriting_api_figures"
        assets.append(asset)

    blockers: list[str] = []
    if len(assets) < 2:
        blockers.append("Expected both motivation and overview post-writing figures.")
    for asset in assets:
        notes = str(asset.get("notes") or "")
        if asset.get("kind") == "fallback" or not asset.get("path") or "paperbanana_failed" in notes or "paperbanana_error" in notes:
            blockers.append(f"Post-writing figure generation failed for {asset.get('figure_id')}.")
    manifest = {
        "stage": "postwriting_api_figures",
        "standard_version": FIGURE_STANDARD_VERSION,
        "policy": experiment_figure_policy_manifest(),
        "required": True,
        "enabled": True,
        "generated_count": len(assets),
        "assets": assets,
        "notes": "generated_after_initial_section_writing",
        "blockers": blockers,
    }
    (figures_dir / "postwriting_api_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest
