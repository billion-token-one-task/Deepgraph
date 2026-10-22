"""Semantic Scholar Graph API — paper search & metadata (PaperOrchestra App. D-style verification)."""

from __future__ import annotations

import time
from typing import Any

S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"


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
