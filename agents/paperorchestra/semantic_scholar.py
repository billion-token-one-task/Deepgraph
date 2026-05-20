"""Semantic Scholar Graph API — paper search & metadata (PaperOrchestra App. D-style verification)."""

from __future__ import annotations

import re
import time
from typing import Any

S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_PAPER = "https://api.semanticscholar.org/graph/v1/paper"


def _headers(api_key: str | None) -> dict[str, str]:
    h = {"Accept": "application/json"}
    if api_key:
        h["x-api-key"] = api_key
    return h


def search_papers(
    query: str,
    *,
    limit: int = 8,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """Return paper dicts from S2 search (fields aligned with citation registry needs)."""
    import httpx

    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": "paperId,title,year,authors,abstract,publicationDate,externalIds,venue,citationCount",
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.get(S2_SEARCH, params=params, headers=_headers(api_key))
        r.raise_for_status()
        data = r.json()
    time.sleep(0.35)  # polite default rate
    return list(data.get("data") or [])


def paper_year(p: dict[str, Any]) -> int | None:
    y = p.get("year")
    if isinstance(y, int):
        return y
    pd = p.get("publicationDate") or ""
    if len(pd) >= 4 and pd[:4].isdigit():
        return int(pd[:4])
    return None


def arxiv_id_from_paper(p: dict[str, Any]) -> str | None:
    ext = p.get("externalIds") or {}
    ax = ext.get("ArXiv") or ext.get("DOI")
    if not ax:
        return None
    if isinstance(ax, str) and re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", ax):
        return ax.split("v")[0]
    return None


def paper_to_bibtex_key(p: dict[str, Any]) -> str:
    """Stable cite key: ArXiv id or S2 paperId."""
    aid = arxiv_id_from_paper(p)
    if aid:
        return aid.replace(".", "_").replace("/", "_")
    pid = p.get("paperId") or "unknown"
    return f"S2_{pid.replace(':', '_')}"


def paper_to_bibtex_entry(p: dict[str, Any], cite_key: str) -> str:
    """Single BibTeX record (@article or @misc for arXiv)."""
    title = (p.get("title") or "Untitled").replace("{", "\\{").replace("}", "\\}")
    authors = p.get("authors") or []
    names: list[str] = []
    for a in authors[:40]:
        if isinstance(a, dict) and a.get("name"):
            names.append(a["name"])
    au = " and ".join(names) if names else "Unknown"
    y = paper_year(p) or 2024
    aid = arxiv_id_from_paper(p)
    if aid:
        return (
            f"@misc{{{cite_key},\n"
            f"  title = {{{title}}},\n"
            f"  author = {{{au}}},\n"
            f"  year = {{{y}}},\n"
            f"  note = {{arXiv:{aid}}}\n}}\n"
        )
    pid = p.get("paperId") or ""
    ven = (p.get("venue") or "").replace("{", "\\{")
    return (
        f"@misc{{{cite_key},\n"
        f"  title = {{{title}}},\n"
        f"  author = {{{au}}},\n"
        f"  year = {{{y}}},\n"
        f"  howpublished = {{Semantic Scholar {pid}}},\n"
        f"  note = {{{ven}}}\n}}\n"
    )
