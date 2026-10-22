"""Compact evidence briefs for PaperOrchestra manuscript generation.

The full DeepGraph manuscript state can be very large and highly repetitive.
This module builds a small, loss-aware brief that keeps the numbers, claims,
constraints, and citation seeds needed by writing agents without replaying the
entire database-shaped state into every LLM call.
"""

from __future__ import annotations

import json
from typing import Any



def _clip(value: Any, limit: int = 800) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value.strip() else []
        return loaded if isinstance(loaded, list) else [loaded]
    return [value]
