"""Independent figure orchestration for PaperOrchestra plotting plans."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from agents.evidence_planner import wants_visualization
from agents.figure_agent import generate_metric_figure_with_retry


def _safe_filename(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in (text or "").strip())
    return cleaned[:80] or "figure"


def _default_plot_plan(metric_name: str) -> list[dict[str, Any]]:
    return [
        {
            "figure_id": "fig_metric_trajectory",
            "plot_type": "plot",
            "title": f"{metric_name} trajectory",
            "objective": "Show the optimization trajectory across experiment iterations.",
            "data_source": "experimental_log.md",
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

    spec = json.dumps(
        {
            "figure": fig,
            "state_title": state.get("title"),
            "method_name": state.get("method_name"),
            "method_summary": state.get("method_summary"),
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
    try:
        subprocess.run(
            paperbanana_cmd.format(
                output=shlex.quote(str(out_path)),
                spec=shlex.quote(spec),
            ),
            shell=True,
            cwd=str(figures_dir),
            timeout=600,
            check=False,
        )
    except Exception as exc:
        return {
            "figure_id": fid,
            "title": str(fig.get("title") or fid),
            "kind": "diagram",
            "path": "",
            "svg_path": "",
            "pdf_path": "",
            "code_path": "",
            "notes": f"paperbanana_error:{exc}",
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
) -> dict[str, Any]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    evidence_plan = state.get("evidence_plan") if isinstance(state.get("evidence_plan"), dict) else {}
    raw_plan = outline.get("plotting_plan") if isinstance(outline, dict) else None
    if not wants_visualization(evidence_plan):
        plan: list[dict[str, Any]] = []
    else:
        plan = raw_plan if isinstance(raw_plan, list) and raw_plan else _default_plot_plan(metric_name)

    assets: list[dict[str, Any]] = []
    for fig in plan[:12]:
        if not isinstance(fig, dict):
            continue
        fid = _safe_filename(str(fig.get("figure_id") or fig.get("title") or "figure"))
        title = str(fig.get("title") or fid)
        objective = str(fig.get("objective") or title)
        plot_type = str(fig.get("plot_type") or "plot").lower()
        if plot_type == "diagram":
            asset = _run_external_diagram(
                fig,
                figures_dir=figures_dir,
                state=state,
                paperbanana_cmd=paperbanana_cmd,
            )
        else:
            out_svg = figures_dir / f"{fid}.svg"
            meta = generate_metric_figure_with_retry(
                iterations,
                baseline,
                metric_name,
                out_svg,
                title=title,
                objective=objective,
            )
            asset = {
                "figure_id": fid,
                "title": title,
                "kind": "plot",
                "path": meta.get("svg_path") or str(out_svg),
                "svg_path": meta.get("svg_path") or str(out_svg),
                "pdf_path": meta.get("pdf_path") or "",
                "code_path": meta.get("code_path") or "",
                "critic": {
                    "ok": meta.get("ok"),
                    "score": meta.get("score"),
                    "notes": meta.get("notes"),
                    "attempts": meta.get("attempts"),
                    "attempt_log": meta.get("attempt_log"),
                    "style": meta.get("style"),
                },
                "objective": objective,
                "data_source": fig.get("data_source") or "experimental_log.md",
            }
        assets.append(asset)

    manifest = {
        "assets": assets,
        "plotting_plan_used": plan,
        "generated_count": len(assets),
    }
    (figures_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
