"""Fail-closed entry point for manuscript rendering.

The former implementation contained a topic-specific deterministic fallback;
it has been removed (see git history). The generic runtime does not generate
manuscript prose until a production-eligible renderer is registered and the
unified evidence state plus reviewer approval permit manuscript work.
"""

from __future__ import annotations

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
    """Check the evidence and review gates, then refuse: no renderer is registered."""
    if not _has_manuscript_authority(state):
        raise ManuscriptGenerationBlocked(
            "manuscript rendering requires scientific_evidence_state="
            "'manuscript_allowed' and reviewer_approved=true"
        )

    raise ManuscriptGenerationBlocked(
        "no production-eligible generic manuscript renderer is registered; "
        "route to manual review"
    )
