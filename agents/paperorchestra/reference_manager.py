"""Reference expansion gate for PaperOrchestra manuscripts.

The manager treats 50 references as a target, not a hard dependency on one
provider. Manuscript writing is allowed once a verified topic-relevant minimum is
met, while the search loop keeps trying multiple sources to approach the target.
"""

from __future__ import annotations

from typing import Any
DEFAULT_REFERENCE_TARGET = 50
DEFAULT_REFERENCE_MINIMUM = 30


class ReferenceExpansionError(RuntimeError):
    """Raised when verified literature cannot reach the required minimum."""

    def __init__(self, report: dict[str, Any], expanded_literature: dict[str, Any] | None = None):
        self.report = report
        self.expanded_literature = expanded_literature or {}
        super().__init__(
            "Reference manager collected "
            f"{report.get('final_count', 0)}/{report.get('minimum_count', DEFAULT_REFERENCE_MINIMUM)} "
            f"verified references (target {report.get('target_count', DEFAULT_REFERENCE_TARGET)}); "
            "manuscript writing is blocked."
        )


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())
