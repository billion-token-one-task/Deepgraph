"""Shared validation helpers for deep insight completeness."""

from __future__ import annotations

import json

INSIGHT_INPUT_MISSING_ERROR_CODE = "insight_input_missing"
_PLACEHOLDER_TEXT = frozenset(
    {"", "?", "unknown", "n/a", "na", "none", "null", "tbd", "todo"}
)


def _clean_text(value) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in _PLACEHOLDER_TEXT else text


def _json_object(value) -> dict:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _has_field_context(field: dict) -> bool:
    return any(
        _clean_text(field.get(key)) for key in ("node_id", "phenomenon", "framework")
    )


def _has_method_context(method: dict) -> bool:
    return any(
        _clean_text(method.get(key))
        for key in ("name", "one_line", "definition", "type")
    )


def get_evosci_input_issue(insight: dict, *, mode: str = "verification") -> dict | None:
    """Return a structured error when an insight is too underspecified for EvoScientist."""
    missing: list[str] = []
    try:
        tier = int(insight.get("tier") or 0)
    except (TypeError, ValueError):
        tier = 0

    if not _clean_text(insight.get("title")):
        missing.append("title")

    if tier == 1:
        field_a = _json_object(insight.get("field_a"))
        field_b = _json_object(insight.get("field_b"))
        if not _has_field_context(field_a):
            missing.append("Field A")
        if not _has_field_context(field_b):
            missing.append("Field B")
        if not _clean_text(insight.get("formal_structure")):
            missing.append("formal structure")
        if not _clean_text(insight.get("transformation")):
            missing.append("transformation")
    else:
        if not _clean_text(insight.get("problem_statement")):
            missing.append("problem statement")
        method = _json_object(insight.get("proposed_method"))
        if not _has_method_context(method):
            missing.append("proposed method")

    if not missing:
        return None

    purpose = "novelty verification" if mode == "verification" else "deep research"
    insight_id = insight.get("id", "?")
    return {
        "error": (
            f"Deep insight {insight_id} is missing required fields for {purpose}: "
            f"{', '.join(missing)}."
        ),
        "error_code": INSIGHT_INPUT_MISSING_ERROR_CODE,
        "missing_fields": missing,
    }
