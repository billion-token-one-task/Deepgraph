"""Publication-quality figure generation with retryable critic feedback."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from config import RUNTIME_PYTHON


STYLE_PRESETS = [
    {
        "name": "camera_ready_blue",
        "line_color": "#1f77b4",
        "marker": "o",
        "baseline_color": "#6c757d",
        "grid_alpha": 0.20,
        "facecolor": "#ffffff",
    },
    {
        "name": "contrast_orange",
        "line_color": "#e76f51",
        "marker": "s",
        "baseline_color": "#264653",
        "grid_alpha": 0.24,
        "facecolor": "#fcfcfc",
    },
    {
        "name": "reviewer_green",
        "line_color": "#2a9d8f",
        "marker": "D",
        "baseline_color": "#4b5563",
        "grid_alpha": 0.26,
        "facecolor": "#ffffff",
    },
]


def _metric_points(iterations: list[dict]) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for it in iterations:
        if it.get("metric_value") is None:
            continue
        pts.append((float(it.get("iteration_number") or 0), float(it.get("metric_value") or 0)))
    return sorted(pts, key=lambda x: x[0])


def _build_figure_script(
    *,
    points: list[tuple[float, float]],
    baseline: float | None,
    title: str,
    metric_name: str,
    out_svg: Path,
    out_pdf: Path,
    style: dict[str, Any],
) -> str:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return f"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xs = {xs!r}
ys = {ys!r}
fig, ax = plt.subplots(figsize=(7.2, 4.2))
fig.patch.set_facecolor({style["facecolor"]!r})
ax.plot(
    xs,
    ys,
    marker={style["marker"]!r},
    linewidth=2.4,
    markersize=6.5,
    color={style["line_color"]!r},
    label={metric_name!r},
)
if {baseline is not None!r}:
    ax.axhline(
        {baseline!r},
        color={style["baseline_color"]!r},
        linestyle="--",
        linewidth=1.6,
        label="baseline",
    )
ax.set_title({title!r}, fontsize=14, pad=10)
ax.set_xlabel("Iteration", fontsize=11)
ax.set_ylabel({metric_name!r}, fontsize=11)
ax.grid(True, alpha={style["grid_alpha"]!r})
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
handles, labels = ax.get_legend_handles_labels()
if handles:
    ax.legend(frameon=False, loc="best")
fig.tight_layout()
fig.savefig({str(out_svg)!r}, format="svg")
fig.savefig({str(out_pdf)!r}, format="pdf")
plt.close(fig)
"""


def render_metric_figure_artifacts(
    points: list[tuple[float, float]],
    baseline: float | None,
    out_svg: Path,
    *,
    metric_name: str,
    title: str = "Metric trajectory",
    style: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit matplotlib code, then save SVG/PDF artifacts plus the script."""
    style = style or STYLE_PRESETS[0]
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = out_svg.with_suffix(".pdf")
    script = out_svg.with_suffix(".py")
    script.write_text(
        _build_figure_script(
            points=points,
            baseline=baseline,
            title=title,
            metric_name=metric_name,
            out_svg=out_svg,
            out_pdf=out_pdf,
            style=style,
        ),
        encoding="utf-8",
    )
    used_fallback = False
    try:
        subprocess.run([RUNTIME_PYTHON, str(script)], check=True, cwd=str(out_svg.parent), timeout=120)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        used_fallback = True
        out_svg.write_text(
            (
                '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420">'
                '<rect width="100%" height="100%" fill="white"/>'
                f'<text x="30" y="42" font-size="22">{title}</text>'
                '<text x="30" y="86" font-size="16">matplotlib unavailable; install matplotlib for camera-ready figures</text>'
                "</svg>"
            ),
            encoding="utf-8",
        )
        if out_pdf.exists():
            out_pdf.unlink()
    return {
        "svg_path": str(out_svg),
        "pdf_path": str(out_pdf) if out_pdf.exists() else "",
        "code_path": str(script),
        "style": style["name"],
        "used_fallback": used_fallback,
    }


def write_matplotlib_figure_svg(
    points: list[tuple[float, float]],
    baseline: float | None,
    out_svg: Path,
    title: str = "Metric trajectory",
) -> None:
    render_metric_figure_artifacts(
        points,
        baseline,
        out_svg,
        metric_name="metric",
        title=title,
        style=STYLE_PRESETS[0],
    )


def _heuristic_critic(svg_path: Path, metric_name: str, title: str) -> tuple[float, list[str]]:
    if not svg_path.exists():
        return 0.15, ["missing_svg"]
    raw = svg_path.read_text(encoding="utf-8", errors="replace")
    size = svg_path.stat().st_size
    score = 0.25
    notes: list[str] = []
    if size > 500:
        score += 0.20
    else:
        notes.append("svg_too_small")
    if "Iteration" in raw:
        score += 0.10
    else:
        notes.append("missing_x_label")
    if metric_name in raw:
        score += 0.10
    else:
        notes.append("missing_metric_label")
    if title and title in raw:
        score += 0.10
    else:
        notes.append("missing_title")
    if raw.count("<path") + raw.count("<polyline") >= 1:
        score += 0.10
    else:
        notes.append("missing_plot_path")
    if raw.count("<text") >= 4:
        score += 0.08
    else:
        notes.append("sparse_text_annotations")
    return min(score, 0.83), notes


def critic_score_figure(svg_path: Path, metric_name: str, title: str) -> tuple[float, str]:
    """Return (0-1 score, short rationale). Uses heuristics plus optional LLM text critic."""
    score, notes = _heuristic_critic(svg_path, metric_name, title)
    llm_bonus = 0.0
    llm_note = "llm_critic_skipped"
    try:
        from agents.llm_client import call_llm_json

        payload = {
            "file": svg_path.name,
            "bytes": svg_path.stat().st_size if svg_path.exists() else 0,
            "metric": metric_name,
            "title": title,
            "heuristic_notes": notes,
            "svg_excerpt": svg_path.read_text(encoding="utf-8", errors="replace")[:3000] if svg_path.exists() else "",
        }
        out, _ = call_llm_json(
            "You are a strict NeurIPS-style figure critic. Return JSON only with keys score_delta and notes.",
            "Judge whether this scientific figure is camera-ready. Positive score_delta should be <= 0.2.\n"
            + json.dumps(payload, ensure_ascii=False),
        )
        llm_bonus = max(-0.15, min(0.20, float(out.get("score_delta", 0.0))))
        llm_note = str(out.get("notes", "llm_ok"))
    except Exception:
        pass
    final_score = max(0.0, min(1.0, score + llm_bonus))
    merged_notes = notes + ([llm_note] if llm_note else [])
    return final_score, ",".join(merged_notes) if merged_notes else "critic_pass"


def generate_metric_figure_with_retry(
    iterations: list[dict],
    baseline_metric_value: float | None,
    metric_name: str,
    out_svg: Path,
    *,
    title: str | None = None,
    objective: str | None = None,
    min_score: float = 0.60,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Generate SVG/PDF/code artifacts and retry with new styles when critic score is low."""
    points = _metric_points(iterations)
    if not points:
        points = [(0.0, float(baseline_metric_value or 0.0))]

    last_notes = ""
    score = 0.0
    render_meta: dict[str, Any] = {}
    attempts: list[dict[str, Any]] = []
    chosen_title = title or f"{metric_name} trajectory"

    for attempt in range(max_retries + 1):
        style = STYLE_PRESETS[min(attempt, len(STYLE_PRESETS) - 1)]
        render_meta = render_metric_figure_artifacts(
            points,
            baseline_metric_value,
            out_svg,
            metric_name=metric_name,
            title=chosen_title,
            style=style,
        )
        score, last_notes = critic_score_figure(out_svg, metric_name, chosen_title)
        attempts.append(
            {
                "attempt": attempt + 1,
                "style": style["name"],
                "score": score,
                "notes": last_notes,
            }
        )
        if score >= min_score:
            break

    return {
        "ok": score >= min_score,
        "score": score,
        "notes": last_notes,
        "attempts": len(attempts),
        "attempt_log": attempts,
        "objective": objective or "",
        **render_meta,
    }
