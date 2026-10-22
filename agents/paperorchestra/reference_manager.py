"""Reference expansion gate for PaperOrchestra manuscripts.

The manager treats 50 references as a target, not a hard dependency on one
provider. Manuscript writing is allowed once a verified topic-relevant minimum is
met, while the search loop keeps trying multiple sources to approach the target.
"""

from __future__ import annotations

from typing import Any




REFERENCE_MANAGER_VERSION = "deepgraph_reference_manager_v1_2026_06_12"
DEFAULT_REFERENCE_TARGET = 50
DEFAULT_REFERENCE_MINIMUM = 30
STOPWORDS = {
    "about",
    "across",
    "after",
    "against",
    "allocation",
    "answer",
    "benchmark",
    "benchmarks",
    "completed",
    "could",
    "large",
    "language",
    "models",
    "paper",
    "reasoning",
    "strongest",
    "system",
    "systems",
    "using",
    "with",
}

NON_PAPER_TITLES = {
    "abstract",
    "acknowledgment",
    "acknowledgments",
    "author index",
    "back matter",
    "bibliography",
    "contents",
    "copyright",
    "dedication",
    "editorial",
    "erratum",
    "foreword",
    "front matter",
    "index",
    "introduction",
    "preface",
    "references",
    "table of contents",
}


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
