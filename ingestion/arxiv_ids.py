"""arXiv ID normalization for deduplication across versions (v1, v2, ...)."""

from __future__ import annotations

import re

_VERSION_SUFFIX = re.compile(r"v\d+$", re.IGNORECASE)


def arxiv_base_id(paper_id: str) -> str:
    """
    Strip trailing version suffix from arXiv id, e.g. 2401.12345v2 -> 2401.12345.
    IDs without 'v' suffix are returned unchanged.
    """
    if not paper_id:
        return ""
    pid = paper_id.strip()
    if not pid:
        return ""
    # arXiv new-style: YYMM.NNNNN; old-style: subject/YYMMNNN
    base = pid
    if "v" in pid:
        parts = pid.rsplit("v", 1)
        if len(parts) == 2 and parts[1].isdigit():
            base = parts[0]
    return base


def is_same_paper(a: str, b: str) -> bool:
    return arxiv_base_id(a) == arxiv_base_id(b) and bool(arxiv_base_id(a))
