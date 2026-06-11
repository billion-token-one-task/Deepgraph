"""Step 2 (PaperOrchestra §4): execute ``plotting_plan`` via independent figure orchestration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agents.paperorchestra.figure_orchestra import run_figure_orchestra


def run_plotting_stage(
    outline: dict,
    state: dict,
    iterations: list[dict],
    figures_dir: Path,
    *,
    baseline: float | None,
    metric_name: str,
    paperbanana_cmd: str | None = None,
    experiment_plot_plan: list[dict[str, Any]] | None = None,
    experiment_plot_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Delegate artifact-backed experiment figure rendering to Figure Orchestra."""
    plotting_outline = dict(outline or {})
    if experiment_plot_plan:
        plotting_outline["plotting_plan"] = [dict(row) for row in experiment_plot_plan if isinstance(row, dict)]
    return run_figure_orchestra(
        plotting_outline,
        state,
        [dict(x) for x in iterations],
        figures_dir,
        baseline=baseline,
        metric_name=metric_name,
        paperbanana_cmd=paperbanana_cmd,
        experiment_plot_reference=experiment_plot_reference,
    )


def default_paperbanana_cmd() -> str | None:
    v = (os.getenv("DEEPGRAPH_PAPERBANANA_CMD") or "").strip()
    return v or None
