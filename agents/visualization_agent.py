"""Automated research visualization pipeline for SciForge runs.

This module turns a completed validation run into manuscript-ready figure
artifacts. It deliberately uses deterministic renderers first: matplotlib for
numeric plots and simple diagrams-as-code for conceptual figures. That keeps
the validation loop stable in offline and CI environments while still leaving
DOT sidecars that can be refined by downstream manuscript agents.
"""

from __future__ import annotations

import html
import json
import math
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Any

from agents import figure_agent
from config import RUNTIME_PYTHON
from db import database as db


MAX_LABEL_CHARS = 52
MAX_KG_EDGES = 14
MAX_RESULT_ROWS = 8


def _json_load(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _as_list(value: Any) -> list:
    loaded = _json_load(value, [])
    if isinstance(loaded, list):
        return loaded
    if loaded in (None, ""):
        return []
    return [loaded]


def _as_dict(value: Any) -> dict[str, Any]:
    loaded = _json_load(value, {})
    return loaded if isinstance(loaded, dict) else {}


def _shorten(value: Any, limit: int = MAX_LABEL_CHARS) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def _escape(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def _wrap_svg_lines(text: Any, width: int = 34, max_lines: int = 3) -> list[str]:
    raw = _shorten(text, width * max_lines)
    lines = textwrap.wrap(raw, width=width) or [""]
    return lines[:max_lines]


def _asset(
    *,
    figure_id: str,
    path: Path,
    asset_kind: str,
    figure_kind: str,
    caption: str,
    source: str,
    metric_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "figure_kind": figure_kind,
        "asset_kind": asset_kind,
        "path": str(path),
        "caption": caption,
        "source": source,
        "metric_name": metric_name or "",
        "metadata": metadata or {},
    }


def _write_text_block_svg(
    path: Path,
    *,
    title: str,
    boxes: list[dict[str, Any]],
    arrows: list[tuple[int, int]] | None = None,
    width: int = 980,
    height: int = 520,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrows = arrows or []
    box_svg: list[str] = []
    for idx, box in enumerate(boxes):
        x = int(box.get("x", 40))
        y = int(box.get("y", 90))
        w = int(box.get("w", 250))
        h = int(box.get("h", 110))
        fill = str(box.get("fill") or "#f8fafc")
        stroke = str(box.get("stroke") or "#334155")
        label = str(box.get("label") or f"Box {idx + 1}")
        body = str(box.get("body") or "")
        lines = _wrap_svg_lines(body, width=max(18, int(w / 9)), max_lines=4)
        text_parts = [
            f'<text x="{x + 18}" y="{y + 32}" font-size="18" font-weight="700" fill="#0f172a">{_escape(label)}</text>'
        ]
        for line_idx, line in enumerate(lines):
            text_parts.append(
                f'<text x="{x + 18}" y="{y + 60 + line_idx * 20}" font-size="14" fill="#334155">{_escape(line)}</text>'
            )
        box_svg.append(
            "\n".join(
                [
                    f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
                    *text_parts,
                ]
            )
        )
    arrow_svg: list[str] = []
    for src, dst in arrows:
        if src >= len(boxes) or dst >= len(boxes):
            continue
        a = boxes[src]
        b = boxes[dst]
        x1 = int(a.get("x", 0)) + int(a.get("w", 0))
        y1 = int(a.get("y", 0)) + int(a.get("h", 0)) // 2
        x2 = int(b.get("x", 0))
        y2 = int(b.get("y", 0)) + int(b.get("h", 0)) // 2
        arrow_svg.append(
            f'<line x1="{x1 + 8}" y1="{y1}" x2="{x2 - 12}" y2="{y2}" stroke="#475569" stroke-width="2" marker-end="url(#arrow)"/>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#475569"/>
  </marker>
</defs>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="40" y="46" font-size="25" font-weight="700" fill="#0f172a">{_escape(title)}</text>
{chr(10).join(arrow_svg)}
{chr(10).join(box_svg)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _write_dot(path: Path, *, title: str, nodes: list[tuple[str, str]], edges: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "digraph G {",
        "  graph [rankdir=LR, bgcolor=white, labelloc=t, fontsize=20, fontname=Helvetica];",
        f"  label={json.dumps(title)};",
        "  node [shape=box, style=\"rounded,filled\", fillcolor=\"#f8fafc\", color=\"#334155\", fontname=Helvetica];",
        "  edge [color=\"#475569\", fontname=Helvetica];",
    ]
    for node_id, label in nodes:
        lines.append(f"  {node_id} [label={json.dumps(_shorten(label, 80))}];")
    for src, dst, label in edges:
        edge_label = f" [label={json.dumps(_shorten(label, 40))}]" if label else ""
        lines.append(f"  {src} -> {dst}{edge_label};")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_bar_script(
    script_path: Path,
    *,
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    out_svg: Path,
    out_pdf: Path,
) -> None:
    script_path.write_text(
        f"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

labels = {labels!r}
values = {values!r}
colors = ["#64748b", "#2563eb", "#0f766e", "#b45309", "#7c3aed", "#be123c", "#0369a1", "#4d7c0f"]
fig, ax = plt.subplots(figsize=(7.6, 4.4))
bars = ax.bar(range(len(values)), values, color=colors[:len(values)], width=0.62)
ax.set_title({title!r}, fontsize=14, pad=10)
ax.set_ylabel({ylabel!r}, fontsize=11)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
ax.grid(axis="y", alpha=0.22)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.annotate(f"{{value:.4g}}", xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig({str(out_svg)!r}, format="svg")
fig.savefig({str(out_pdf)!r}, format="pdf")
plt.close(fig)
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_fallback_bar_svg(path: Path, *, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 820
    height = 460
    plot_x = 90
    plot_y = 90
    plot_w = 650
    plot_h = 260
    max_value = max([abs(v) for v in values] or [1.0]) or 1.0
    bar_w = max(28, int(plot_w / max(1, len(values)) * 0.55))
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="42" y="46" font-size="24" font-weight="700" fill="#0f172a">{_escape(title)}</text>',
        f'<text x="42" y="76" font-size="13" fill="#475569">{_escape(ylabel)}</text>',
        f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#94a3b8"/>',
    ]
    for idx, (label, value) in enumerate(zip(labels, values)):
        cx = plot_x + int((idx + 0.5) * plot_w / max(1, len(values)))
        h = int(abs(value) / max_value * plot_h)
        x = cx - bar_w // 2
        y = plot_y + plot_h - h
        color = "#2563eb" if idx else "#64748b"
        chunks.extend(
            [
                f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{color}"/>',
                f'<text x="{cx}" y="{y - 7}" font-size="12" text-anchor="middle" fill="#0f172a">{value:.4g}</text>',
                f'<text x="{cx}" y="{plot_y + plot_h + 24}" font-size="11" text-anchor="middle" fill="#334155">{_escape(_shorten(label, 18))}</text>',
            ]
        )
    chunks.append("</svg>")
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def _render_bar_chart(
    *,
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    out_svg: Path,
) -> dict[str, str | bool]:
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = out_svg.with_suffix(".pdf")
    script_path = out_svg.with_suffix(".py")
    _write_bar_script(
        script_path,
        labels=labels,
        values=values,
        title=title,
        ylabel=ylabel,
        out_svg=out_svg,
        out_pdf=out_pdf,
    )
    used_fallback = False
    try:
        subprocess.run(
            [RUNTIME_PYTHON, str(script_path)],
            cwd=str(out_svg.parent),
            timeout=120,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        used_fallback = True
        _write_fallback_bar_svg(out_svg, labels=labels, values=values, title=title, ylabel=ylabel)
        if out_pdf.exists():
            out_pdf.unlink()
    return {
        "svg_path": str(out_svg),
        "pdf_path": str(out_pdf) if out_pdf.exists() else "",
        "code_path": str(script_path),
        "used_fallback": used_fallback,
    }


def _fetch_iterations(run_id: int) -> list[dict[str, Any]]:
    rows = db.fetchall(
        """
        SELECT iteration_number, phase, metric_value, metric_name, status, description
        FROM experiment_iterations
        WHERE run_id=?
        ORDER BY iteration_number
        """,
        (run_id,),
    )
    return [dict(row) for row in rows]


def _fetch_kg_relations(insight: dict[str, Any]) -> list[dict[str, Any]]:
    source_node_ids = [str(x) for x in _as_list(insight.get("source_node_ids")) if str(x).strip()]
    source_paper_ids = [str(x) for x in _as_list(insight.get("source_paper_ids")) if str(x).strip()]
    filters: list[str] = []
    params: list[Any] = []
    if source_node_ids:
        filters.append("gr.node_id IN ({})".format(",".join("?" for _ in source_node_ids)))
        params.extend(source_node_ids)
    if source_paper_ids:
        filters.append("gr.paper_id IN ({})".format(",".join("?" for _ in source_paper_ids)))
        params.extend(source_paper_ids)
    where = "WHERE " + " OR ".join(filters) if filters else ""
    sql = f"""
        SELECT
            gr.node_id,
            gr.predicate,
            gr.confidence,
            subj.canonical_name AS subject_name,
            obj.canonical_name AS object_name,
            subj.entity_type AS subject_type,
            obj.entity_type AS object_type
        FROM graph_relations gr
        JOIN graph_entities subj ON subj.id = gr.subject_entity_id
        JOIN graph_entities obj ON obj.id = gr.object_entity_id
        {where}
        ORDER BY gr.confidence DESC, gr.id DESC
        LIMIT {MAX_KG_EDGES}
    """
    try:
        return [dict(row) for row in db.fetchall(sql, tuple(params))]
    except Exception:
        return []


def _fetch_literature_results(insight: dict[str, Any]) -> list[dict[str, Any]]:
    source_node_ids = [str(x) for x in _as_list(insight.get("source_node_ids")) if str(x).strip()]
    source_paper_ids = [str(x) for x in _as_list(insight.get("source_paper_ids")) if str(x).strip()]
    filters: list[str] = []
    params: list[Any] = []
    if source_node_ids:
        filters.append("node_id IN ({})".format(",".join("?" for _ in source_node_ids)))
        params.extend(source_node_ids)
    if source_paper_ids:
        filters.append("paper_id IN ({})".format(",".join("?" for _ in source_paper_ids)))
        params.extend(source_paper_ids)
    where = "WHERE metric_value IS NOT NULL"
    if filters:
        where += " AND (" + " OR ".join(filters) + ")"
    sql = f"""
        SELECT method_name, dataset_name, metric_name, metric_value
        FROM results
        {where}
        ORDER BY metric_value DESC
        LIMIT {MAX_RESULT_ROWS}
    """
    try:
        return [dict(row) for row in db.fetchall(sql, tuple(params))]
    except Exception:
        return []


def _method_name(insight: dict[str, Any]) -> str:
    method = _as_dict(insight.get("proposed_method"))
    return str(method.get("name") or insight.get("title") or "Proposed method")


def _experimental_metric_name(insight: dict[str, Any], fallback: str) -> str:
    plan = _as_dict(insight.get("experimental_plan"))
    metrics = plan.get("metrics")
    if isinstance(metrics, dict):
        return str(metrics.get("primary") or metrics.get("name") or fallback)
    if isinstance(metrics, list) and metrics:
        first = metrics[0]
        if isinstance(first, dict):
            return str(first.get("name") or fallback)
        return str(first or fallback)
    return fallback


def _generate_metric_trajectory(
    *,
    iterations: list[dict[str, Any]],
    baseline_metric_value: float | None,
    metric_name: str,
    figures_dir: Path,
) -> list[dict[str, Any]]:
    out_svg = figures_dir / "fig_metric_trajectory.svg"
    meta = figure_agent.generate_metric_figure_with_retry(
        iterations,
        baseline_metric_value,
        metric_name,
        out_svg,
        title=f"{metric_name} trajectory",
        objective="Show validation-loop metric trajectory against the reproduced baseline.",
    )
    caption = f"Validation trajectory for {metric_name}, with the reproduced baseline shown as reference."
    assets: list[dict[str, Any]] = []
    for key, asset_kind in (("svg_path", "svg"), ("pdf_path", "pdf"), ("code_path", "source")):
        raw = str(meta.get(key) or "").strip()
        if raw and Path(raw).exists():
            assets.append(
                _asset(
                    figure_id="fig_metric_trajectory",
                    figure_kind="metric_trajectory",
                    asset_kind=asset_kind,
                    path=Path(raw),
                    caption=caption,
                    source="experiment_iterations",
                    metric_name=metric_name,
                    metadata={
                        "score": meta.get("score"),
                        "notes": meta.get("notes"),
                        "attempts": meta.get("attempts"),
                        "used_fallback": meta.get("used_fallback"),
                    },
                )
            )
    return assets


def _generate_baseline_comparison(
    *,
    baseline_metric_value: float | None,
    best_metric_value: float | None,
    metric_name: str,
    figures_dir: Path,
) -> list[dict[str, Any]]:
    if baseline_metric_value is None or best_metric_value is None:
        return []
    labels = ["baseline", "best proposed"]
    values = [float(baseline_metric_value), float(best_metric_value)]
    meta = _render_bar_chart(
        labels=labels,
        values=values,
        title=f"Baseline vs proposed ({metric_name})",
        ylabel=metric_name,
        out_svg=figures_dir / "fig_baseline_vs_proposed.svg",
    )
    caption = f"Baseline versus best proposed validation result for {metric_name}."
    assets: list[dict[str, Any]] = []
    for key, asset_kind in (("svg_path", "svg"), ("pdf_path", "pdf"), ("code_path", "source")):
        raw = str(meta.get(key) or "").strip()
        if raw and Path(raw).exists():
            assets.append(
                _asset(
                    figure_id="fig_baseline_vs_proposed",
                    figure_kind="experiment_comparison",
                    asset_kind=asset_kind,
                    path=Path(raw),
                    caption=caption,
                    source="experiment_runs",
                    metric_name=metric_name,
                    metadata={"used_fallback": meta.get("used_fallback")},
                )
            )
    return assets


def _generate_literature_results_chart(
    *,
    rows: list[dict[str, Any]],
    figures_dir: Path,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    metric_name = str(rows[0].get("metric_name") or "metric")
    labels = [
        _shorten(f"{row.get('method_name') or 'method'} / {row.get('dataset_name') or 'dataset'}", 32)
        for row in rows
    ]
    values = [float(row.get("metric_value") or 0.0) for row in rows]
    meta = _render_bar_chart(
        labels=labels,
        values=values,
        title=f"Knowledge graph result snapshot ({metric_name})",
        ylabel=metric_name,
        out_svg=figures_dir / "fig_literature_results.svg",
    )
    caption = "Top extracted literature results from the DeepGraph results table for this insight context."
    assets: list[dict[str, Any]] = []
    for key, asset_kind in (("svg_path", "svg"), ("pdf_path", "pdf"), ("code_path", "source")):
        raw = str(meta.get(key) or "").strip()
        if raw and Path(raw).exists():
            assets.append(
                _asset(
                    figure_id="fig_literature_results",
                    figure_kind="literature_result_chart",
                    asset_kind=asset_kind,
                    path=Path(raw),
                    caption=caption,
                    source="results",
                    metric_name=metric_name,
                    metadata={"row_count": len(rows), "used_fallback": meta.get("used_fallback")},
                )
            )
    return assets


def _generate_overview_diagram(
    *,
    insight: dict[str, Any],
    verdict: str | None,
    figures_dir: Path,
) -> list[dict[str, Any]]:
    method = _as_dict(insight.get("proposed_method"))
    plan = _as_dict(insight.get("experimental_plan"))
    metrics = plan.get("metrics") if isinstance(plan.get("metrics"), dict) else {}
    out_svg = figures_dir / "fig_approach_overview.svg"
    boxes = [
        {
            "x": 44,
            "y": 112,
            "w": 245,
            "h": 145,
            "label": "Current limitation",
            "body": insight.get("existing_weakness") or insight.get("problem_statement") or "Existing method limitations",
            "fill": "#fef2f2",
            "stroke": "#b91c1c",
        },
        {
            "x": 365,
            "y": 112,
            "w": 245,
            "h": 145,
            "label": "Proposed improvement",
            "body": method.get("one_line") or method.get("definition") or _method_name(insight),
            "fill": "#eff6ff",
            "stroke": "#1d4ed8",
        },
        {
            "x": 686,
            "y": 112,
            "w": 245,
            "h": 145,
            "label": "Validation evidence",
            "body": f"Metric: {metrics.get('primary') or 'primary metric'}; verdict: {verdict or 'pending'}",
            "fill": "#ecfdf5",
            "stroke": "#047857",
        },
    ]
    _write_text_block_svg(out_svg, title=_shorten(insight.get("title") or "Approach overview", 80), boxes=boxes, arrows=[(0, 1), (1, 2)])
    out_dot = out_svg.with_suffix(".dot")
    _write_dot(
        out_dot,
        title="Approach overview",
        nodes=[("limitation", "Current limitation"), ("method", _method_name(insight)), ("evidence", "Validation evidence")],
        edges=[("limitation", "method", "addresses"), ("method", "evidence", "validated by")],
    )
    caption = "Overview of the limitation, proposed method, and validation evidence produced by SciForge."
    return [
        _asset(
            figure_id="fig_approach_overview",
            figure_kind="overview_diagram",
            asset_kind="svg",
            path=out_svg,
            caption=caption,
            source="deep_insights",
        ),
        _asset(
            figure_id="fig_approach_overview",
            figure_kind="overview_diagram",
            asset_kind="dot",
            path=out_dot,
            caption=caption,
            source="deep_insights",
        ),
    ]


def _generate_method_architecture_diagram(
    *,
    insight: dict[str, Any],
    figures_dir: Path,
) -> list[dict[str, Any]]:
    method = _as_dict(insight.get("proposed_method"))
    plan = _as_dict(insight.get("experimental_plan"))
    baselines = _as_list(plan.get("baselines"))
    datasets = _as_list(plan.get("datasets"))
    method_label = method.get("name") or insight.get("title") or "Proposed method"
    out_svg = figures_dir / "fig_method_architecture.svg"
    boxes = [
        {
            "x": 44,
            "y": 112,
            "w": 220,
            "h": 130,
            "label": "Inputs",
            "body": ", ".join(_shorten(item.get("name") if isinstance(item, dict) else item, 24) for item in datasets[:3]) or "Experiment data",
            "fill": "#f8fafc",
            "stroke": "#334155",
        },
        {
            "x": 332,
            "y": 92,
            "w": 300,
            "h": 170,
            "label": _shorten(method_label, 28),
            "body": method.get("definition") or method.get("one_line") or "Structured proposed method",
            "fill": "#eff6ff",
            "stroke": "#2563eb",
        },
        {
            "x": 704,
            "y": 112,
            "w": 220,
            "h": 130,
            "label": "Evaluation",
            "body": f"Baselines: {len(baselines)}; metric: {_experimental_metric_name(insight, 'metric')}",
            "fill": "#f0fdf4",
            "stroke": "#15803d",
        },
    ]
    _write_text_block_svg(out_svg, title="Method architecture", boxes=boxes, arrows=[(0, 1), (1, 2)])
    out_dot = out_svg.with_suffix(".dot")
    _write_dot(
        out_dot,
        title="Method architecture",
        nodes=[("inputs", "Inputs"), ("method", method_label), ("eval", "Evaluation")],
        edges=[("inputs", "method", "feeds"), ("method", "eval", "measured by")],
    )
    caption = "Architecture-style diagram derived from the structured proposed method and experimental plan."
    return [
        _asset(
            figure_id="fig_method_architecture",
            figure_kind="method_architecture",
            asset_kind="svg",
            path=out_svg,
            caption=caption,
            source="deep_insights.proposed_method",
        ),
        _asset(
            figure_id="fig_method_architecture",
            figure_kind="method_architecture",
            asset_kind="dot",
            path=out_dot,
            caption=caption,
            source="deep_insights.proposed_method",
        ),
    ]


def _generate_kg_subgraph_diagram(
    *,
    relations: list[dict[str, Any]],
    figures_dir: Path,
) -> list[dict[str, Any]]:
    if not relations:
        return []
    nodes: dict[str, tuple[float, float]] = {}
    labels: dict[str, str] = {}
    edges: list[tuple[str, str, str]] = []
    for row in relations[:MAX_KG_EDGES]:
        subject = _shorten(row.get("subject_name") or "subject", 30)
        obj = _shorten(row.get("object_name") or "object", 30)
        predicate = _shorten(row.get("predicate") or "relates_to", 28)
        labels.setdefault(subject, subject)
        labels.setdefault(obj, obj)
        edges.append((subject, obj, predicate))
    node_names = sorted(labels)
    width = 980
    height = 620
    cx, cy = width / 2, height / 2 + 24
    radius = min(260, max(150, 34 * len(node_names)))
    for idx, name in enumerate(node_names):
        angle = 2 * math.pi * idx / max(1, len(node_names)) - math.pi / 2
        nodes[name] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
    out_svg = figures_dir / "fig_knowledge_subgraph.svg"
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '    <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/>',
        "  </marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="40" y="48" font-size="25" font-weight="700" fill="#0f172a">Knowledge graph subgraph</text>',
    ]
    for src, dst, pred in edges:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        chunks.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#94a3b8" stroke-width="1.6" marker-end="url(#arrow)"/>')
        chunks.append(f'<text x="{mx:.1f}" y="{my:.1f}" font-size="11" text-anchor="middle" fill="#475569">{_escape(pred)}</text>')
    for name, (x, y) in nodes.items():
        chunks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="42" fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.4"/>')
        for idx, line in enumerate(_wrap_svg_lines(name, width=13, max_lines=3)):
            chunks.append(f'<text x="{x:.1f}" y="{y - 8 + idx * 15:.1f}" font-size="11" text-anchor="middle" fill="#0f172a">{_escape(line)}</text>')
    chunks.append("</svg>")
    out_svg.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    out_dot = out_svg.with_suffix(".dot")
    dot_ids = {name: f"n{idx}" for idx, name in enumerate(node_names)}
    _write_dot(
        out_dot,
        title="Knowledge graph subgraph",
        nodes=[(dot_ids[name], name) for name in node_names],
        edges=[(dot_ids[src], dot_ids[dst], pred) for src, dst, pred in edges],
    )
    caption = "Entity-relation subgraph extracted from DeepGraph knowledge graph context for this insight."
    return [
        _asset(
            figure_id="fig_knowledge_subgraph",
            figure_kind="knowledge_graph_subgraph",
            asset_kind="svg",
            path=out_svg,
            caption=caption,
            source="graph_relations",
            metadata={"edge_count": len(edges), "node_count": len(node_names)},
        ),
        _asset(
            figure_id="fig_knowledge_subgraph",
            figure_kind="knowledge_graph_subgraph",
            asset_kind="dot",
            path=out_dot,
            caption=caption,
            source="graph_relations",
            metadata={"edge_count": len(edges), "node_count": len(node_names)},
        ),
    ]


def write_figure_references(workdir: Path, assets: list[dict[str, Any]]) -> str:
    """Reference generated figures in final_report.md when it exists.

    If no final report exists in the experiment workspace, write a standalone
    figure reference file so downstream manuscript agents can still consume the
    visual inventory.
    """
    logical: dict[str, dict[str, Any]] = {}
    for asset in assets:
        if asset.get("asset_kind") != "svg":
            continue
        logical.setdefault(str(asset["figure_id"]), asset)
    lines = ["## Generated Figures", ""]
    for asset in logical.values():
        rel_path = Path(str(asset["path"]))
        try:
            rel = rel_path.relative_to(workdir)
        except ValueError:
            rel = rel_path
        lines.append(f"- `{asset['figure_id']}`: {asset.get('caption') or ''} (`{rel}`)")
    block = "\n".join(lines).strip() + "\n"
    marker_start = "<!-- deepgraph-visualization-agent:start -->"
    marker_end = "<!-- deepgraph-visualization-agent:end -->"
    wrapped = f"\n{marker_start}\n{block}{marker_end}\n"
    for candidate in (workdir / "final_report.md", workdir / "results" / "final_report.md"):
        if not candidate.exists():
            continue
        text = candidate.read_text(encoding="utf-8", errors="replace")
        if marker_start in text and marker_end in text:
            before, rest = text.split(marker_start, 1)
            _, after = rest.split(marker_end, 1)
            candidate.write_text(before.rstrip() + wrapped + after.lstrip(), encoding="utf-8")
        else:
            candidate.write_text(text.rstrip() + "\n" + wrapped, encoding="utf-8")
        return str(candidate)
    ref_path = workdir / "figures" / "figure_references.md"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(block, encoding="utf-8")
    return str(ref_path)


def generate_visualization_bundle(
    *,
    run_id: int,
    workdir: Path,
    insight: dict[str, Any],
    metric_name: str,
    baseline_metric_value: float | None,
    best_metric_value: float | None = None,
    verdict: str | None = None,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    """Generate all SciForge visualization artifacts for a completed run."""
    workdir = Path(workdir)
    figures_dir = workdir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    iterations = _fetch_iterations(run_id)
    assets: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    generators = (
        ("overview_diagram", lambda: _generate_overview_diagram(insight=insight, verdict=verdict, figures_dir=figures_dir)),
        ("method_architecture", lambda: _generate_method_architecture_diagram(insight=insight, figures_dir=figures_dir)),
        (
            "metric_trajectory",
            lambda: _generate_metric_trajectory(
                iterations=iterations,
                baseline_metric_value=baseline_metric_value,
                metric_name=metric_name,
                figures_dir=figures_dir,
            ),
        ),
        (
            "baseline_comparison",
            lambda: _generate_baseline_comparison(
                baseline_metric_value=baseline_metric_value,
                best_metric_value=best_metric_value,
                metric_name=metric_name,
                figures_dir=figures_dir,
            ),
        ),
        ("literature_results", lambda: _generate_literature_results_chart(rows=_fetch_literature_results(insight), figures_dir=figures_dir)),
        ("knowledge_graph_subgraph", lambda: _generate_kg_subgraph_diagram(relations=_fetch_kg_relations(insight), figures_dir=figures_dir)),
    )

    for name, build in generators:
        try:
            produced = build()
            if produced:
                assets.extend(produced)
            else:
                skipped.append({"figure_kind": name, "reason": "no_source_data"})
        except Exception as exc:
            skipped.append({"figure_kind": name, "reason": str(exc)[:240]})

    references_path = write_figure_references(workdir, assets) if assets else ""
    manifest = {
        "run_id": run_id,
        "deep_insight_id": insight.get("id"),
        "metric_name": metric_name,
        "baseline_metric_value": baseline_metric_value,
        "best_metric_value": best_metric_value,
        "verdict": verdict,
        "validation_summary": str(summary_path) if summary_path else "",
        "assets": assets,
        "skipped": skipped,
        "references_path": references_path,
    }
    manifest_path = figures_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "figures_dir": str(figures_dir),
    }
