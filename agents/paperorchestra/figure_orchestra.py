"""Disabled generic figure boundary for meta-harness-v1.

Image generation and PaperBanana are outside the first closed loop. The former
topic-coupled implementation is preserved in the non-production CGGR example.
These functions report a blocked state and never manufacture a completed
artifact.
"""

from __future__ import annotations

from typing import Any


class FigureGenerationBlocked(RuntimeError):
    """Raised when a caller requests an out-of-scope figure operation."""


def _blocked(stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "blocked",
        "generated_count": 0,
        "assets": [],
        "blockers": [
            "figure generation is outside the meta-harness-v1 first closed loop"
        ],
    }


def run_figure_orchestra(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _blocked("figure_orchestra")


def run_postwriting_api_figure_stage(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _blocked("postwriting_api_figures")


def render_native_figure(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise FigureGenerationBlocked(
        "native figure rendering is outside the meta-harness-v1 first closed loop"
    )


def infer_figure_spec_from_reference(path: str, caption: str = "") -> dict[str, Any]:
    return {
        "status": "blocked",
        "source_path": str(path),
        "caption": str(caption),
        "blocker": "reference-image inference is outside the first closed loop",
    }


def _augment_plotting_plan(
    plan: list[dict[str, Any]],
    state: dict[str, Any],
    iterations: list[dict[str, Any]],
    metric_name: str,
) -> list[dict[str, Any]]:
    return list(plan or [])


def _run_external_diagram(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise FigureGenerationBlocked(
        "external diagram generation is outside the meta-harness-v1 first closed loop"
    )
