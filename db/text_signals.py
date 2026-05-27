"""Normalize JSON list fields from paper_insights (may contain nested lists)."""

from __future__ import annotations


def flatten_text_items(values: list | None) -> list[str]:
    """Flatten limitations/open_questions-style lists into plain strings."""
    if not values:
        return []
    out: list[str] = []
    stack: list = list(values)
    while stack:
        item = stack.pop(0)
        if item is None:
            continue
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append(text)
            continue
        if isinstance(item, (list, tuple)):
            stack[:0] = list(item)
            continue
        if isinstance(item, dict):
            for key in ("text", "content", "question", "limitation", "value"):
                nested = item.get(key)
                if isinstance(nested, str) and nested.strip():
                    out.append(nested.strip())
                    break
            else:
                text = str(item).strip()
                if text:
                    out.append(text)
            continue
        text = str(item).strip()
        if text:
            out.append(text)
    return out
