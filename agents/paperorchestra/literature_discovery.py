"""Step 3 (PaperOrchestra §4): verified citation registry from graph evidence + S2 discovery."""

from __future__ import annotations

import json
from typing import Any




def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        loaded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    return loaded if isinstance(loaded, list) else []


def _dedupe(items: list[Any]) -> list[Any]:
    seen = set()
    out: list[Any] = []
    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, (dict, list)) else str(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
