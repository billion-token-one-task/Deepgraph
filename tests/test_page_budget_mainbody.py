"""Main-body page count excludes References."""

from __future__ import annotations

from agents.manuscript_page_budget import ensure_mainbody_end_label, page_budget_blockers


def test_mainbody_label_inserted_before_bibliography() -> None:
    tex = r"\documentclass{article}\begin{document}Body\end{document}"
    tex = tex.replace(
        r"\end{document}",
        r"\bibliographystyle{plain}" + "\n" + r"\bibliography{refs}" + "\n" + r"\end{document}",
    )
    out = ensure_mainbody_end_label(tex)
    assert r"\label{mainbody:end}" in out
    assert out.index(r"\label{mainbody:end}") < out.index(r"\bibliographystyle")


def test_page_budget_blockers_on_fail() -> None:
    blockers = page_budget_blockers(
        {"pass": False, "final": {"target_pages": 9, "page_count": 4, "issue": "too_short"}}
    )
    assert blockers
    assert "HARD FAIL" in blockers[0]
    assert "References excluded" in blockers[0]


def test_page_budget_blockers_empty_on_pass() -> None:
    assert page_budget_blockers({"pass": True, "final": {"page_count": 9}}) == []
