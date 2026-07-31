"""Fail-closed entry point for manuscript rendering.

The former implementation contained a CRPP-specific deterministic fallback.
That historical implementation now lives in the disabled
``plugins.examples.cggr`` boundary. The generic runtime does not generate
manuscript prose until a production-eligible renderer is explicitly selected
and the unified evidence state plus reviewer approval permit manuscript work.
"""

from __future__ import annotations

import os
from typing import Any


class ManuscriptGenerationBlocked(RuntimeError):
    """Raised when no approved, evidence-safe manuscript renderer is available."""


def _has_manuscript_authority(state: dict[str, Any]) -> bool:
    return (
        str(state.get("scientific_evidence_state") or "") == "manuscript_allowed"
        and state.get("reviewer_approved") is True
    )


def run_paperorchestra_full(
    state: dict,
    literature_block: str,
    paper_ids: list[str],
    iterations: list,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run an explicitly selected renderer after evidence and review gates.

    The historical example is available only for reproducibility in an
    isolated, non-production test environment. It is never a silent fallback.
    """
    if not _has_manuscript_authority(state):
        raise ManuscriptGenerationBlocked(
            "manuscript rendering requires scientific_evidence_state="
            "'manuscript_allowed' and reviewer_approved=true"
        )

    renderer = str(state.get("manuscript_renderer") or "").strip()
    if renderer == "example.cggr":
        enabled = os.getenv("DEEPGRAPH_ENABLE_NONPROD_EXAMPLE_PLUGINS", "").strip().lower()
        if enabled not in {"1", "true", "yes"}:
            raise ManuscriptGenerationBlocked(
                "example.cggr is non-production and disabled; set the explicit "
                "isolated-test opt-in only outside production"
            )
        from plugins.examples.cggr.full_pipeline import run_paperorchestra_full as run_example

        return run_example(
            state,
            literature_block,
            paper_ids,
            iterations,
            **kwargs,
        )

    raise ManuscriptGenerationBlocked(
        "no production-eligible generic manuscript renderer is registered; "
        "route to manual review"
    )
