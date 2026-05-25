"""Tests for duplicate-paragraph detection in manuscript_page_budget."""

from __future__ import annotations

import pytest

from agents.manuscript_page_budget import (
    deduplicate_paragraphs,
    find_duplicate_paragraphs,
)


_NEAR_DUPLICATE_TEX = r"""\documentclass{article}
\begin{document}

% deepgraph-fill-rw-extra-11
\paragraph{Related Theme 11.} Prior work on vision-language reasoning emphasizes chain-of-thought prompting, retrieval augmentation, and self-consistency decoding. These techniques improve fluency but do not guarantee grounded visual use.

Some other distinct paragraph about contrastive grounding methodology that should not be flagged as duplicate. It introduces specific notation that does not appear elsewhere.

% deepgraph-fill-rw-extra-10
\paragraph{Related Theme 10.} Prior work on vision-language reasoning emphasizes chain-of-thought prompting, retrieval augmentation, and self-consistency decoding. These techniques improve fluency but do not guarantee grounded visual use.

\end{document}
"""


def test_find_duplicate_paragraphs_detects_identical_relworks() -> None:
    findings = find_duplicate_paragraphs(_NEAR_DUPLICATE_TEX)
    assert len(findings) == 1
    assert findings[0]["jaccard"] >= 0.8
    assert "Related Theme" in findings[0]["paragraph_a_excerpt"]
    assert "Related Theme" in findings[0]["paragraph_b_excerpt"]


def test_deduplicate_paragraphs_keeps_first_occurrence_only() -> None:
    new_tex, removed = deduplicate_paragraphs(_NEAR_DUPLICATE_TEX)
    assert removed == 1
    assert new_tex.count("Related Theme") == 1
    assert "Some other distinct paragraph" in new_tex


def test_find_duplicate_paragraphs_ignores_leading_comment_lines() -> None:
    tex = r"""\documentclass{article}
\begin{document}

% deepgraph-fill-method-extra-3
\paragraph{Design Rationale 3.} The dual-stream architecture keeps visual and textual representations separate until the verification head, preserving the signal that early fusion would erase.

% deepgraph-fill-method-extra-4
\paragraph{Design Rationale 4.} The dual-stream architecture keeps visual and textual representations separate until the verification head, preserving the signal that early fusion would erase.

\end{document}
"""
    findings = find_duplicate_paragraphs(tex)
    assert findings, "Comment-prefixed paragraphs must still be compared"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
