"""Tests for R2: fill-order priority informed by R5 corpus audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.manuscript_deterministic_fill import (
    _content_audit_priority,
    _order_fillers_by_priority,
)


def test_order_fillers_promotes_priority_families() -> None:
    fillers = [
        ("related work", "rw1", "rw1 block"),
        ("discussion", "d1", "d1 block"),
        ("method", "m1", "m1 block"),
        ("experiments", "e1", "e1 block"),
        ("introduction", "i1", "i1 block"),
        ("conclusion", "c1", "c1 block"),
    ]
    ordered = _order_fillers_by_priority(fillers, ["method", "intro"])
    section_order = [s for s, _, _ in ordered]
    assert section_order.index("method") < section_order.index("related work")
    assert section_order.index("method") < section_order.index("discussion")
    assert section_order.index("introduction") < section_order.index("related work")


def test_order_fillers_no_priority_keeps_original() -> None:
    fillers = [
        ("related work", "rw1", "rw1 block"),
        ("method", "m1", "m1 block"),
        ("introduction", "i1", "i1 block"),
    ]
    ordered = _order_fillers_by_priority(fillers, [])
    assert ordered == fillers


def test_content_audit_priority_reads_report(tmp_path: Path) -> None:
    bundle = tmp_path
    (bundle / "content_audit_report.json").write_text(
        json.dumps(
            {
                "issues": [
                    {"section_family": "method", "severity": "high"},
                    {"section_family": "intro", "severity": "medium"},
                    {"section_family": None, "severity": "low"},
                    {"section_family": "method", "severity": "low"},
                ]
            }
        ),
        encoding="utf-8",
    )
    families = _content_audit_priority(bundle)
    assert families == ["method", "intro"]


def test_content_audit_priority_missing_report(tmp_path: Path) -> None:
    assert _content_audit_priority(tmp_path) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
