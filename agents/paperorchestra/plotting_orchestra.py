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
) -> dict[str, Any]:
    """Delegate figure rendering to the independent Figure Orchestra."""
    return run_figure_orchestra(
        outline,
        state,
        [dict(x) for x in iterations],
        figures_dir,
        baseline=baseline,
        metric_name=metric_name,
        paperbanana_cmd=paperbanana_cmd,
    )


def default_paperbanana_cmd() -> str | None:
    v = (os.getenv("DEEPGRAPH_PAPERBANANA_CMD") or "").strip()
    return v or None
